"""
Гибридный поиск: BM25 (FTS5) + плотные векторы, слияние через RRF,
приоритет свежести, буст выверенных ответов, фильтр по ролям.

Почему гибрид: на технических артикулах («SPL WRP-A 2ECO6-38», «Арт. 500095.F»)
эмбеддинги промахиваются, а BM25 попадает точно; на формулировках «чем
отличается насос для скважины от колодезного» — наоборот. RRF объединяет
оба списка без подбора весов и является отраслевым стандартом.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime

import config
import db
import embeddings
import logging_setup
import normtext
import rerank as reranker

log = logging_setup.get("search")


@dataclass
class Hit:
    chunk_id: int
    doc_id: int
    text: str
    heading: str | None
    context: str
    rel_path: str
    file_name: str
    brand: str | None
    doc_type: str | None
    section: str | None
    effective_date: str | None
    is_current: int
    page_from: int | None
    score: float = 0.0
    # Интерпретируемая релевантность 0…1: покрытие значимых слов запроса
    # (см. rerank.relevance). Именно с ней сравнивается MIN_CONFIDENCE.
    # Ранговый score из RRF для порога не годится: он зависит только от
    # места в списке, и «лучший из мусора» получает те же ~0.014, что и
    # точное попадание, — порог отказа не срабатывал никогда.
    relevance: float = 0.0
    channels: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        bits = [self.file_name]
        if self.page_from:
            bits.append(f"с. {self.page_from}")
        if self.effective_date:
            bits.append(f"от {self.effective_date}")
        return ", ".join(bits)


# ------------------------------------------------------------- утилиты -----
_STOP = {"как", "что", "где", "для", "или", "это", "the", "and", "при", "под", "над",
         "чем", "кто", "быть", "если", "так", "уже", "его", "нам", "мне", "она", "они"}

# Синонимы предметной области. В вопросе пишут «производительность», в
# паспорте — «подача»: LSA связывает слова, только если они совместно
# встречаются в базе, а эти пары в одном документе почти не встречаются.
# Разворот делается на стороне ЗАПРОСА: индекс не трогается, а значит
# словарь можно править без пересборки. Сюда же — транслит брендов:
# «grundfos» в вопросе и «Грундфос» на русскоязычной странице паспорта.
SYNONYMS: dict[str, list[str]] = {
    "производительность": ["подача", "расход"],
    "подача": ["производительность", "расход"],
    "расход": ["подача", "производительность"],
    "давление": ["напор"],
    "напор": ["давление"],
    "глубина": ["погружение"],
    "высота": ["напор"],
    "мощность": ["ватт", "квт"],
    "вес": ["масса"],
    "масса": ["вес"],
    "срок": ["гарантия", "ресурс"],
    "ресурс": ["срок", "наработка"],
    "шум": ["вибрация"],
    "цена": ["стоимость"],
    "стоимость": ["цена"],
    "скважина": ["скважинный"],
    "колодец": ["колодезный"],
    "grundfos": ["грундфос"],
    "грундфос": ["grundfos"],
    "wilo": ["вило"],
    "вило": ["wilo"],
    "unipump": ["юнипамп"],
    "юнипамп": ["unipump"],
    "belamos": ["беламос"],
    "беламос": ["belamos"],
    "aquastrong": ["аквастронг"],
    "аквастронг": ["aquastrong"],
    "джилекс": ["jeelex", "vodomet"],
    "jeelex": ["джилекс"],
    "водомет": ["vodomet"],
    "vodomet": ["водомет"],
}


def expand_synonyms(tokens: list[str]) -> list[str]:
    """Синонимы к токенам запроса — без дублей, в порядке появления."""
    out: list[str] = []
    for t in tokens:
        for syn in SYNONYMS.get(t, []):
            if syn not in tokens and syn not in out:
                out.append(syn)
    return out


def _fts_query(text: str) -> str:
    """Готовит запрос для FTS5: OR по значимым токенам + префиксный поиск."""
    text = normtext.canon(text)
    raw = re.findall(r"[\wА-Яа-яЁё\-./]{2,}", text.lower())
    # Числа из одной-двух цифр и дроби — значимая часть вопроса («напор
    # 45 м», «подача 3.6»), их нельзя выбрасывать вместе с предлогами.
    tokens = [t.strip("-./") for t in raw
              if t not in _STOP
              and (len(t) > 2 or re.fullmatch(r"\d+(?:\.\d+)?", t))]
    tokens = [t for t in tokens if t]
    if not tokens:
        return ""
    parts = []
    for t in tokens[:16] + expand_synonyms(tokens[:16])[:8]:
        safe = t.replace('"', "")
        parts.append(f'"{safe}"')
        if len(safe) > 4 and safe.isalpha():
            parts.append(f'"{safe}"*')
            # Префикс от ОСНОВЫ слова, а не от словоформы: «насосов» не
            # находится по «насосов»*, а «насос»* находит и «насосов»,
            # и «насосы», и «насосом». BM25 сам морфологии не знает —
            # это единственное место, где она у него появляется.
            try:
                import lsa
                stem = lsa.normalize_token(safe)
                if stem != safe and len(stem) > 3:
                    parts.append(f'"{stem}"*')
            except Exception:  # noqa: BLE001
                pass
    return " OR ".join(parts)


def _recency_factor(effective_date: str | None) -> float:
    """0.5^(возраст / период полураспада). Документ без даты — нейтрален."""
    if not effective_date:
        return 0.5
    try:
        d = datetime.fromisoformat(effective_date).date()
    except ValueError:
        return 0.5
    age = max((date.today() - d).days, 0)
    return 0.5 ** (age / max(config.RECENCY_HALF_LIFE_DAYS, 1.0))


def allowed_sections(role: str | None) -> set[str] | None:
    """
    Какие разделы базы видит эта роль. None означает «все».

    Ключевой момент — что делать с ролью, которой нет в списке. Раньше
    она получала доступ ко всему: `ROLE_SECTIONS.get(role)` возвращал
    None, и фильтр просто не применялся. Ошибка в опасную сторону, и
    хватало для неё опечатки: «Sales» вместо «sales», незнакомое слово в
    `DEFAULT_ROLE`, роль, выданная в админке с другой раскладкой.
    Сотрудник молча получал дилерский раздел, и заметить это было нечем:
    в журнале есть запись о том, что роль что-то отсекла, и нет записи о
    том, что она не отсекла ничего.

    Теперь наоборот: если разграничение настроено, а роль в нём не
    описана — не показываем ничего и пишем в журнал. Пустую выдачу
    заметят в тот же день, утечку не заметят никогда.
    """
    if not role:
        role = config.DEFAULT_ROLE
    if not config.ROLE_SECTIONS:          # разграничение не настроено вовсе
        return None
    sections = config.ROLE_SECTIONS.get(role)
    if sections is None:
        log.error("роль «%s» не описана в ROLE_SECTIONS — выдача закрыта целиком. "
                  "Проверьте написание роли и настройку разграничения", role)
        return set()
    if "*" in sections:
        return None
    return set(sections)


# ------------------------------------------------------------- каналы ------
def bm25_search(query: str, limit: int) -> list[tuple[int, float]]:
    fts = _fts_query(query)
    if not fts:
        return []
    try:
        rows = db.q("""SELECT rowid, bm25(chunks_fts, 6.0, 3.0, 1.5, 2.0, 1.0, 2.0) AS rank
                       FROM chunks_fts WHERE chunks_fts MATCH ?
                       ORDER BY rank LIMIT ?""", (fts, limit))
    except Exception:  # noqa: BLE001 — синтаксическая ошибка FTS-запроса
        return []
    return [(int(r["rowid"]), -float(r["rank"])) for r in rows]


_dense_warned = False


def dense_search(query: str, limit: int) -> list[tuple[int, float]]:
    global _dense_warned
    store = db.vectors()
    if len(store) == 0:
        if not _dense_warned:
            _dense_warned = True
            log.warning("векторный индекс пуст — смысловой канал не работает. "
                        "Выполните: python index.py train-lsa && python index.py reembed")
        return []
    try:
        vec = embeddings.embed_query(query)
    except Exception as exc:  # noqa: BLE001 — провайдер недоступен: работаем на BM25
        if not _dense_warned:
            _dense_warned = True
            log.warning("смысловой канал недоступен (%s) — поиск идёт "
                        "только по точным словам", exc)
        return []
    # Разная размерность означает, что векторы считались другой моделью.
    # Молча вернуть пустой список нельзя: снаружи это выглядит как «ничего
    # не нашлось», и причину не найти. Сообщаем прямо и один раз.
    if len(vec) != store.dim:
        if not _dense_warned:
            _dense_warned = True
            log.error("векторы в индексе посчитаны моделью с %d измерениями, "
                      "а сейчас выбрана модель с %d. Смысловой канал отключён. "
                      "Пересчитайте: python index.py reembed", store.dim, len(vec))
        return []
    return store.search(vec, limit)


def dense_ready() -> tuple[bool, str]:
    """Готов ли смысловой канал — для диагностики и админки."""
    store = db.vectors()
    if len(store) == 0:
        return False, "векторный индекс пуст"
    try:
        vec = embeddings.embed_query("проверка")
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    if len(vec) != store.dim:
        return False, (f"в индексе векторы по {store.dim} измерений, "
                       f"а модель даёт {len(vec)} — нужен пересчёт (reembed)")
    return True, f"{len(store)} векторов по {store.dim} измерений"


def golden_search(query: str, limit: int = 3, role: str | None = None) -> list[dict]:
    """
    Выверенные экспертом ответы — отдельный канал с наивысшим приоритетом.

    Роль учитывается и здесь. У выверенного ответа есть список разделов,
    к которым он относится: пусто — ответ общий и виден всем, заполнено —
    виден только тем ролям, которым открыты все перечисленные разделы.
    Заполняется он сам, из источников, на которые эксперт сослался: если
    ответ собран из дилерского прайса, он и останется дилерским.
    """
    fts = _fts_query(query)
    if not fts:
        return []
    try:
        rows = db.q("""SELECT g.id, g.question, g.answer, g.source_refs, g.sections,
                              bm25(golden_fts) AS rank
                       FROM golden_fts JOIN golden_qa g ON g.id = golden_fts.rowid
                       WHERE golden_fts MATCH ? AND g.active=1
                       ORDER BY rank LIMIT ?""", (fts, limit * 3))
    except Exception:  # noqa: BLE001
        return []
    allowed = allowed_sections(role)
    out = []
    for r in rows:
        item = dict(r)
        need = {x for x in (item.get("sections") or "").split("|") if x}
        if need and allowed is not None and not need <= allowed:
            log.info("выверенный ответ %s скрыт от роли «%s»: разделы %s",
                     item["id"], role, sorted(need))
            continue
        out.append(item)
        if len(out) >= limit:
            break
    return out


# ----------------------------------------------------------------- RRF -----
def _rrf(ranked: list[tuple[int, float]], k: int) -> dict[int, float]:
    return {cid: 1.0 / (k + rank) for rank, (cid, _s) in enumerate(ranked, start=1)}


def search(query: str, top_k: int | None = None, role: str | None = None,
           include_outdated: bool = False) -> list[Hit]:
    top_k = top_k or config.SEARCH_TOP_K
    n = config.SEARCH_CANDIDATES

    # Смешанная раскладка в вопросе — «джилeкс» с латинской «e» —
    # чинится тем же ремонтом гомоглифов, что и распознанные сканы.
    # Документы этим обрабатываются при индексации, вопросы — здесь.
    try:
        import ocr
        repaired, report = ocr.repair_homoglyphs(query)
        if report.get("repaired"):
            log.debug("вопрос содержал смешанную раскладку, починено: %s → %s",
                      query, repaired)
            query = repaired
    except Exception:  # noqa: BLE001 — ремонт не должен ронять поиск
        pass

    lexical = bm25_search(query, n)
    # Смысловой канал получает вопрос, расширенный синонимами: «какая
    # производительность» без слова «подача» может не найти паспорт,
    # где написано только «подача, м3/ч».
    dense_query = query
    q_tokens = re.findall(r"[\wа-яё]+", normtext.canon(query).lower())
    extra = expand_synonyms(q_tokens)
    if extra:
        dense_query = query + " " + " ".join(extra)
    dense = dense_search(dense_query, n)

    fused: dict[int, float] = {}
    for cid, s in _rrf(lexical, config.RRF_K).items():
        fused[cid] = fused.get(cid, 0.0) + s
    for cid, s in _rrf(dense, config.RRF_K).items():
        fused[cid] = fused.get(cid, 0.0) + s
    if not fused:
        return []

    lex_map = dict(lexical)
    dense_map = dict(dense)

    ids = list(fused.keys())
    rows = db.q(f"""
        SELECT c.id chunk_id, c.doc_id, c.text, c.heading, c.context, c.page_from,
               d.rel_path, d.file_name, d.brand, d.doc_type, d.section,
               d.effective_date, d.is_current, d.status
        FROM chunks c JOIN documents d ON d.id = c.doc_id
        WHERE c.id IN ({','.join('?' * len(ids))})""", ids)

    allowed = allowed_sections(role)
    blocked_by_role = 0
    hits: list[Hit] = []
    for r in rows:
        if r["status"] != "ok":
            continue
        if allowed is not None and r["section"] not in allowed:
            blocked_by_role += 1
            continue
        if not include_outdated and not r["is_current"]:
            continue
        base = fused[r["chunk_id"]]
        recency = _recency_factor(r["effective_date"])
        score = config.RECENCY_ALPHA * base + (1 - config.RECENCY_ALPHA) * base * recency * 2
        hits.append(Hit(
            chunk_id=r["chunk_id"], doc_id=r["doc_id"], text=r["text"],
            heading=r["heading"], context=r["context"] or "", rel_path=r["rel_path"],
            file_name=r["file_name"], brand=r["brand"], doc_type=r["doc_type"],
            section=r["section"], effective_date=r["effective_date"],
            is_current=r["is_current"], page_from=r["page_from"], score=score,
            channels={"bm25": lex_map.get(r["chunk_id"]), "dense": dense_map.get(r["chunk_id"]),
                      "rrf": base, "recency": round(recency, 3)}))

    if not hits and blocked_by_role:
        # Частая причина пустой выдачи при настройке: ROLE_SECTIONS в config.py
        # перечисляет разделы, которых нет среди папок верхнего уровня KB_ROOT.
        print(f"[поиск] роль «{role}» отсекла все {blocked_by_role} найденных фрагментов. "
              f"Сверьте ROLE_SECTIONS в config.py с папками внутри KB_ROOT "
              f"(команда: python index.py stats).")

    hits.sort(key=lambda h: -h.score)
    hits = rerank(query, hits)
    hits = _dedupe_by_document(hits, top_k)

    # Релевантность для порога отказа — по итоговой выдаче, той самой,
    # которая уйдёт в модель. Считается лексически и всегда, независимо
    # от провайдера переранжирования: шкала MIN_CONFIDENCE стабильна.
    if hits:
        texts = ["\n".join(p for p in (h.context, h.heading, h.text) if p)
                 for h in hits]
        for h, rel in zip(hits, reranker.relevance(query, texts)):
            h.relevance = float(rel)
            h.channels["relevance"] = round(float(rel), 4)
    return hits


def _dedupe_by_document(hits: list[Hit], top_k: int, per_doc: int = 2) -> list[Hit]:
    """Не даём одному документу занять всю выдачу."""
    seen: dict[int, int] = {}
    out: list[Hit] = []
    for h in hits:
        c = seen.get(h.doc_id, 0)
        if c >= per_doc:
            continue
        seen[h.doc_id] = c + 1
        out.append(h)
        if len(out) >= top_k:
            break
    return out


def rerank(query: str, hits: list[Hit]) -> list[Hit]:
    """
    Пересортировка первых RERANKER_TOP_N кандидатов (подробности в rerank.py).

    Что здесь важно и чего не было в заглушке:

      • оценка реранкера не подменяет собой итоговый скор, а смешивается
        с ним. Иначе кросс-энкодер полностью затирал бы приоритет свежести,
        и на вопрос про цену всплывал бы прайс позапрошлого года — просто
        потому, что текст в нём сформулирован ближе к вопросу;
      • шкала итоговой оценки сохраняется, поэтому порог MIN_CONFIDENCE,
        по которому ассистент честно отказывается отвечать, продолжает
        значить ровно то же, что и раньше;
      • любой сбой реранкера означает выдачу в исходном порядке,
        а не потерю ответа.
    """
    if not hits or config.RERANKER_PROVIDER == "none":
        return hits
    top = hits[:config.RERANKER_TOP_N]
    rest = hits[config.RERANKER_TOP_N:]

    texts = ["\n".join(p for p in (h.context, h.heading, h.text) if p) for h in top]
    scores = reranker.score(query, texts)
    if scores is None:                      # реранкер недоступен или не успел
        return hits

    before = [h.chunk_id for h in top]
    blended = reranker.blend([h.score for h in top], scores)
    for h, raw, mixed in zip(top, scores, blended):
        h.channels["rerank"] = round(float(raw), 4)
        h.score = float(mixed)
    top.sort(key=lambda h: -h.score)
    if [h.chunk_id for h in top] != before:
        log.debug("переранжирование изменило порядок первых %d кандидатов", len(top))
    return top + rest


def confidence(hits: list[Hit]) -> float:
    """
    Уверенность 0…1, с которой сравнивается MIN_CONFIDENCE.

    Берётся лучшая релевантность по всей выдаче, а не только по первому
    месту: первый элемент мог подняться за счёт свежести, а точное
    совпадение слов стоять вторым. Раньше здесь возвращался ранговый
    RRF-скор — и порог отказа был недостижим математически: минимум
    для первого места (0.0139) выше порога по умолчанию (0.012). Ветка
    «в базе знаний нет данных» не срабатывала никогда.
    """
    return max((h.relevance for h in hits), default=0.0)
