"""
Сборка ответа: роутинг «цена / документ», промпт с цитированием,
честный отказ при недостатке данных, логирование для обучения.

Ключевые правила, зашитые в промпт:
  * отвечать только по переданным фрагментам;
  * обязательно ставить номер источника [1], [2] — сотрудник должен
    иметь возможность проверить;
  * если данных нет — прямо сказать «в базе нет», а не догадываться.
    Ретривер всегда что-то возвращает, поэтому запрет на догадки
    критичен: иначе бот уверенно пересказывает соседний документ.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import config
import db
import llm as llm_mod
import logging_setup
import metrics
import prices
import search as search_mod

logger = logging_setup.get("answer")

SYSTEM_PROMPT = """Ты — внутренний справочный ассистент компании по инженерному оборудованию \
(насосы, отопление, водоснабжение, трубопроводные системы). Отвечаешь сотрудникам компании.

Правила:
1. Отвечай ТОЛЬКО на основании приведённых фрагментов документов. Никаких знаний «вообще».
2. После каждого утверждения ставь номер источника в квадратных скобках: [1], [2].
3. Если во фрагментах нет ответа — так и напиши: «В базе знаний нет данных по этому вопросу» \
и предложи, к какому разделу или ответственному обратиться. Не придумывай и не догадывайся.
4. Технические характеристики (напор, расход, давление, мощность, диаметр, артикул) \
переноси дословно, без округлений и пересчётов.
5. Если фрагменты противоречат друг другу, укажи это и отдай приоритет документу с более \
поздней датой, явно назвав обе версии.
6. Цены называй только если они есть во фрагментах, обязательно с датой прайс-листа.
7. Отвечай кратко и по делу, на русском языке. Списком — только если перечисляешь позиции."""

ANSWER_TEMPLATE = """Ниже фрагменты из корпоративной базы знаний.

{context}

ВОПРОС: {question}"""

NO_ANSWER = ("В базе знаний не нашлось данных по этому вопросу.\n\n"
             "Возможные причины: документа нет в папке BD, он отсканирован без текстового слоя "
             "(нужен OCR) или вопрос стоит переформулировать — например, указать бренд или "
             "артикул. Вопрос записан: эксперт может добавить ответ, и в следующий раз бот "
             "ответит сам.")


@dataclass
class Answer:
    text: str
    hits: list = field(default_factory=list)
    products: list = field(default_factory=list)
    answered: bool = True
    # Нумерованные источники в том порядке, в котором они пронумерованы
    # в контексте модели: сначала прайсы, затем документы. Именно этот
    # список показывается пользователю под ответом.
    sources: list = field(default_factory=list)
    query_id: int | None = None
    confidence: float = 0.0
    latency_ms: int = 0
    used_golden: bool = False
    llm_model: str = ""
    # Куда ушёл вопрос и где остановился — из этого строится воронка ответа.
    route: str = "documents"        # golden | price | documents | none
    stage: str = "answered"         # nothing_found | low_confidence | answered
    channels: str = ""              # каналы, нашедшие лучший фрагмент
    n_candidates: int = 0
    rerank_used: bool = False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def price_context(product_rows: list[dict], start: int = 1,
                  ) -> tuple[str, list[dict], int]:
    """
    Блок прайса как нумерованный источник.

    Раньше цены вставлялись в контекст безымянным блоком, а промпт
    требовал ставить [n] после каждого утверждения — модели приходилось
    выбирать номер, и цена уходила со ссылкой на первый документ. Сотрудник
    открывал паспорт насоса и цены там не находил. Теперь каждый файл
    прайса получает свой номер, и ссылка ведёт туда, откуда цифра.
    """
    groups: dict[str, list[dict]] = {}
    for row in product_rows:
        groups.setdefault(row.get("rel_path") or "", []).append(row)
    blocks, sources = [], []
    n = start
    for rel, rows in groups.items():
        first = rows[0]
        fname = first.get("file_name") or rel or "прайс-лист"
        pdate = first.get("price_date") or ""
        head = f"[{n}] прайс-лист «{fname}»"
        if pdate:
            head += f", от {pdate}"
        head += " — точные данные из таблицы, приоритет над остальными фрагментами"
        blocks.append(head + "\n" + prices.format_products(rows))
        sources.append({"n": n, "kind": "price", "file_name": fname,
                        "rel_path": rel, "date": pdate, "is_current": 1,
                        "page": None})
        n += 1
    return "\n\n".join(blocks), sources, n


def build_context(hits: list[search_mod.Hit], start: int = 1) -> str:
    """
    Контекст для модели — с бюджетом и без дублей.

    Бюджет: маленькие модели (8k окна) получали до 17 тысяч символов и
    отвечали кодом 400, который снаружи выглядел как «модель недоступна».
    Дубли: перекрывающиеся чанки и один документ в двух разделах читались
    моделью как независимые подтверждения одного и того же.
    """
    blocks = []
    budget = config.CONTEXT_MAX_CHARS
    seen_texts: list[str] = []
    for i, h in enumerate(hits, start=start):
        norm = " ".join(h.text.split())[:400]
        if any(norm and (norm in prev or prev in norm) for prev in seen_texts):
            continue
        seen_texts.append(norm)
        header = f"[{i}] {h.file_name}"
        extras = []
        if h.brand:
            extras.append(h.brand)
        if h.doc_type:
            extras.append(h.doc_type)
        if h.page_from:
            extras.append(f"с. {h.page_from}")
        if h.effective_date:
            extras.append(f"дата документа: {h.effective_date}")
        if extras:
            header += " (" + ", ".join(extras) + ")"
        if h.heading:
            header += f" — раздел «{h.heading}»"
        block = f"{header}\n{h.text}"
        if blocks and sum(len(b) for b in blocks) + len(block) > budget:
            break
        blocks.append(block)
    return "\n\n".join(blocks)


def verify_answer(text: str, n_sources: int, context: str) -> tuple[str, dict]:
    """
    Проверка ответа модели по результату: ссылки и числа.

    Ссылки: номер [7] при трёх источниках — реальный случай, модель
    охотно продолжает нумерацию. Битые ссылки убираются из текста.

    Числа: правило «переноси дословно» ничем не подкреплено, а ошибка в
    цифре на технической базе дороже любой стилистики. Числа ответа,
    которых нет в контексте, не удаляются (модель могла законно сложить
    или пересчитать), но помечаются — и для читателя, и в журнале.
    """
    import normtext
    report = {"bad_refs": [], "unsupported_numbers": []}
    if not text.strip():
        return text, report

    def _fix_ref(m):
        n = int(m.group(1))
        if 1 <= n <= n_sources:
            return m.group(0)
        report["bad_refs"].append(n)
        return ""

    text = re.sub(r"\[(\d{1,3})\]", _fix_ref, text)

    def _numbers(raw: str) -> set[str]:
        cleaned = re.sub(r"(?<=\d)[\s  ](?=\d)", "", normtext.canon(raw))
        found = set(re.findall(r"\d+(?:\.\d+)?", cleaned))
        # «18400.00» подтверждает и «18400»: целая часть — то же число.
        for n in list(found):
            if "." in n:
                found.add(n.split(".", 1)[0])
                found.add(n.rstrip("0").rstrip("."))
        return found

    ctx_numbers = _numbers(context)
    body = re.sub(r"\[\d{1,3}\]", "", text)
    for num in _numbers(body):
        if len(num) < 2 and "." not in num:
            continue                      # одиночная цифра — обычно нумерация
        if num not in ctx_numbers:
            report["unsupported_numbers"].append(num)
    if report["unsupported_numbers"]:
        nums = ", ".join(sorted(report["unsupported_numbers"])[:5])
        text += ("\n\n⚠ Числа " + nums + " не найдены в приведённых "
                 "фрагментах — сверьтесь с первоисточником.")
    return text, report


def ask(question: str, user_id: int | None = None, user_name: str | None = None,
        role: str | None = None, chat_id: int | None = None,
        log: bool = True, source: str = "вопрос") -> Answer:
    # source влияет только на очередь к модели: у звонка и голосового
    # сообщения человек ждёт на линии, у пакетной проверки — никто.
    started = time.time()
    db.init()
    role = role or config.DEFAULT_ROLE

    # Попытка переопределить правила не отклоняется, а обезвреживается:
    # формулировка «забудь про Grundfos, меня интересует Wilo» совершенно
    # нормальна, и отказывать на неё было бы хуже, чем пропустить.
    import security
    guard = security.inspect_question(question)
    prompt_question = security.neutralize(question) if guard["suspicious"] else question
    if guard["suspicious"]:
        logger.warning("вопрос содержит попытку переопределить правила "
                       "(пользователь %s): %s", user_id, guard["matched"][:2])

    # Уточняющий вопрос («а цена?», «какая глубина погружения?») наследует
    # модель и бренд из предыдущей реплики этого же разговора. Поиск и
    # маршрутизация работают с дополненным вопросом; сам текст сотрудника
    # не переписывается, а модель получает явную пометку о контексте.
    import dialog
    chat_key = chat_id if chat_id is not None else user_id
    search_question, inherited = dialog.augment(chat_key, question)
    if inherited:
        prompt_question += ("\n(Уточняющий вопрос в разговоре: речь идёт о "
                            + ", ".join(inherited) + ")")
    dialog.remember(chat_key, question)

    # 1. Выверенный экспертом ответ — если есть близкий, отдаём его.
    #
    # Роль учитывается и здесь. Выверенный ответ пишет человек, и он
    # запросто может содержать дилерские условия: эксперт отвечал на
    # вопрос дилера и о разграничении не думал. Канал этот идёт первым и
    # раньше проверок не проходил вовсе, то есть был самым коротким путём
    # к утечке.
    # Кандидатов берём несколько: топ-1 по BM25 может не пройти порог
    # сходства, а дословно совпадающий эталон — стоять вторым.
    golden = search_mod.golden_search(search_question, limit=5, role=role)
    best = max(golden,
               key=lambda g: _golden_similarity(search_question, g["question"]),
               default=None)
    if best and _golden_is_close(search_question, best["question"]):
        g = best
        db.run("UPDATE golden_qa SET hits=hits+1 WHERE id=?", (g["id"],))
        ans = Answer(text=g["answer"] + "\n\n_Ответ выверен экспертом._",
                     answered=True, used_golden=True, route="golden",
                     stage="answered", channels="golden",
                     latency_ms=int((time.time() - started) * 1000))
        if log:
            ans.query_id = _log_query(question, ans, user_id, user_name, role, chat_id)
            _trace(question, ans, "", user_id, user_name, role, started)
        return ans

    # 2. Ценовые вопросы — в структурированную таблицу, не в векторный поиск.
    product_rows: list[dict] = []
    if prices.looks_like_price_question(search_question) \
            or prices.ARTICLE_RX.search(search_question):
        product_rows = prices.search_products(search_question, limit=8, role=role)

    # 3. Документный поиск.
    _t = time.time()
    hits = search_mod.search(search_question, role=role)
    metrics.record_stage("поиск", int((time.time() - _t) * 1000))
    conf = search_mod.confidence(hits)
    logger.debug("найдено фрагментов: %d, лучшая оценка %.5f", len(hits), conf)

    if not hits and not product_rows:
        ans = Answer(text=NO_ANSWER, answered=False, route="none",
                     stage="nothing_found",
                     latency_ms=int((time.time() - started) * 1000))
        if log:
            ans.query_id = _log_query(question, ans, user_id, user_name, role, chat_id)
            _trace(question, ans, "", user_id, user_name, role, started)
        return ans

    if conf < config.MIN_CONFIDENCE and not product_rows:
        ans = Answer(text=NO_ANSWER, hits=hits, answered=False, confidence=conf,
                     route="documents", stage="low_confidence",
                     channels=_channels_of(hits), n_candidates=len(hits),
                     rerank_used=any(h.channels.get("rerank") is not None for h in hits),
                     latency_ms=int((time.time() - started) * 1000))
        if log:
            ans.query_id = _log_query(question, ans, user_id, user_name, role, chat_id)
            _trace(question, ans, "", user_id, user_name, role, started)
        return ans

    # Ценовой вопрос раньше отключал порог целиком: при найденной позиции
    # прайса в контекст шли и документные фрагменты любой степени
    # случайности — и модель дописывала к цене «напор 60 м [3]» из чужого
    # паспорта. Теперь порог применяется к документам всегда: цена без
    # подтверждённых фрагментов идёт одна, без случайного сопровождения.
    if product_rows and conf < config.MIN_CONFIDENCE:
        hits = []

    # Прайс — нумерованный источник наравне с документами: цена в ответе
    # ссылается на файл прайса, а не на случайно выбранный документ.
    price_block, price_sources, next_n = ("", [], 1)
    if product_rows:
        price_block, price_sources, next_n = price_context(product_rows, start=1)
    context = build_context(hits, start=next_n)
    if price_block:
        context = price_block + "\n\n" + context if context else price_block
    doc_sources = [{"n": next_n + i, "kind": "document", "file_name": h.file_name,
                    "rel_path": h.rel_path, "date": h.effective_date,
                    "is_current": h.is_current, "page": h.page_from}
                   for i, h in enumerate(hits)]
    sources = price_sources + doc_sources

    engine = llm_mod.get_llm()
    _t = time.time()
    try:
        # Вопрос человека — самая высокая важность в очереди к модели:
        # он обгоняет фоновую обработку базы, которая иначе заняла бы
        # модель на часы.
        with llm_mod.queue_context(source):
            resp = engine.complete(SYSTEM_PROMPT,
                                   ANSWER_TEMPLATE.format(context=context,
                                                          question=prompt_question))
        text = resp.text.strip()
        tokens_in, tokens_out, model = resp.tokens_in, resp.tokens_out, resp.model
        metrics.record_stage("генерация", int((time.time() - _t) * 1000))
        stage = "answered"
        text, verify_report = verify_answer(text, len(sources), context)
        if verify_report["bad_refs"] or verify_report["unsupported_numbers"]:
            logger.warning("проверка ответа: битые ссылки %s, "
                           "неподтверждённые числа %s",
                           verify_report["bad_refs"],
                           verify_report["unsupported_numbers"])
    except llm_mod.LLMBusy as exc:
        # Очередь переполнена. Показать фрагменты честнее, чем молчать:
        # человек хотя бы увидит, где ответ, пока модель занята.
        logger.warning("отказ по очереди к модели: %s", exc)
        text = (f"{exc}\n\nПока модель занята, вот наиболее подходящие фрагменты базы:\n\n"
                + "\n\n".join(f"[{i}] {h.text[:400]}"
                              for i, h in enumerate(hits[:3], next_n)))
        tokens_in = tokens_out = 0
        model = "queue-busy"
        stage = "llm_busy"
    except Exception as exc:  # noqa: BLE001
        # Текст ошибки маскируется: в нём бывает base_url с паролем АТС
        # или ключом провайдера — журналы это маскируют, и исходящие
        # сообщения обязаны не хуже.
        exc = logging_setup.mask(str(exc))
        text = (f"Модель недоступна ({exc}). Вот наиболее подходящие фрагменты базы:\n\n"
                + "\n\n".join(f"[{i}] {h.text[:400]}"
                              for i, h in enumerate(hits[:3], next_n)))
        tokens_in = tokens_out = 0
        model = "fallback"
        stage = "llm_failed"

    # Последняя проверка — по результату, а не по формулировке вопроса:
    # она ловит утечку из недоступного раздела независимо от того, каким
    # способом её добились.
    leak = security.check_answer_leak(text, hits, search_mod.allowed_sections(role),
                                      products=product_rows)
    if leak["leak"]:
        logger.error("ответ содержал фрагменты из недоступных роли «%s» разделов: "
                     "%s — выдача заменена отказом", role, leak["sections"])
        text = security.SAFE_REFUSAL
        # Чистится всё, а не только текст: артикулы и цены из закрытого
        # раздела иначе оставались в products и печатались потребителями
        # объекта Answer прямо под текстом отказа.
        hits = []
        product_rows = []
        sources = []

    # Фрагменты вместо ответа — не ответ: воронка и метрика answered
    # раньше считали эти случаи успехом, и сбой модели неделями выглядел
    # как нормальная работа.
    ans = Answer(text=text, hits=hits, products=product_rows,
                 answered=(stage == "answered"),
                 sources=sources,
                 confidence=conf, latency_ms=int((time.time() - started) * 1000),
                 llm_model=model,
                 route="price" if product_rows else "documents", stage=stage,
                 channels=_channels_of(hits), n_candidates=len(hits),
                 rerank_used=any(h.channels.get("rerank") is not None for h in hits))
    if log:
        ans.query_id = _log_query(question, ans, user_id, user_name, role, chat_id,
                                  tokens_in, tokens_out)
        # В трассировку идёт вопрос в том виде, в каком его видела модель
        # (после обезвреживания) — иначе разбор жалобы смотрит не на то.
        _trace(prompt_question, ans, context, user_id, user_name, role, started)
    return ans


# Подпись модели: числа с разделителями («55/75», «0,5-40», «500036.F»).
# Именно этими токенами различаются соседние модели, и именно их
# выбрасывало старое сравнение по словам от трёх символов — из-за чего
# «напор Водомет 60/92» получал выверенный ответ про 55/75 со стопроцентным
# сходством. Ответ, помеченный «выверен экспертом», доверием пользуется
# безоговорочным, поэтому ошибка здесь дороже любого промаха поиска.
_SIGNATURE_RX = re.compile(r"\d+(?:[.,/\-хx×]\d+)*(?:\.[a-zа-я]{1,3})?", re.IGNORECASE)

_brand_cache: set[str] | None = None


def _signatures(text: str) -> set[str]:
    """Числовые подписи моделей в едином написании."""
    out = set()
    for m in _SIGNATURE_RX.findall(text.lower()):
        out.add(m.replace(",", ".").replace("х", "x").replace("×", "x"))
    return out


def _known_brands() -> set[str]:
    """Бренды из проиндексированной базы; кэш на процесс."""
    global _brand_cache
    if _brand_cache is None:
        try:
            rows = db.q("SELECT DISTINCT brand FROM documents "
                        "WHERE brand IS NOT NULL AND brand != ''")
            _brand_cache = {r["brand"].lower() for r in rows}
        except Exception:  # noqa: BLE001
            _brand_cache = set()
    return _brand_cache


def _brands_in(text: str) -> set[str]:
    words = set(re.findall(r"[\wа-яё]+", text.lower()))
    return words & _known_brands()


def _golden_similarity(question: str, golden_question: str) -> float:
    """
    Сходство вопроса с эталонным: 0 — точно не тот вопрос.

    Три ступени, от жёсткой к мягкой. Подписи моделей обязаны совпасть
    как множества: вопрос про 60/92 не имеет права получить ответ про
    55/75, каким бы похожим ни был остальной текст. Бренды — то же
    самое: «гарантия на Джилекс» и «гарантия на Вило» различаются одним
    словом, но ответы у них разные. И только после этого сравниваются
    слова — с порогом, который проверяет уже сам вызывающий код.
    """
    sig_a, sig_b = _signatures(question), _signatures(golden_question)
    if (sig_a or sig_b) and sig_a != sig_b:
        return 0.0
    br_a, br_b = _brands_in(question), _brands_in(golden_question)
    if (br_a or br_b) and br_a != br_b:
        return 0.0
    a = set(re.findall(r"[\wа-яё]{3,}", question.lower()))
    b = set(re.findall(r"[\wа-яё]{3,}", golden_question.lower()))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _golden_is_close(question: str, golden_question: str,
                     threshold: float = 0.75) -> bool:
    """Не выдаём выверенный ответ на другой вопрос."""
    return _golden_similarity(question, golden_question) >= threshold


def _trace(question, ans, context, user_id, user_name, role, started) -> None:
    """Сохраняет полную цепочку ответа — для разбора конкретных жалоб."""
    try:
        import tracing
        tracing.record(question, ans,
                       prompt=(ANSWER_TEMPLATE.format(context=context,
                                                      question=question)
                               if context else ""),
                       timings={"total_ms": int((time.time() - started) * 1000),
                                "latency_ms": ans.latency_ms},
                       user_id=user_id, user_name=user_name, role=role)
    except Exception:  # noqa: BLE001 — трассировка не влияет на ответ
        pass


def _channels_of(hits: list) -> str:
    """Какие каналы нашли лучший фрагмент — «bm25», «dense», «bm25+dense»."""
    if not hits:
        return ""
    found = [name for name in ("bm25", "dense")
             if hits[0].channels.get(name) is not None]
    return "+".join(found) or "прочее"


def _log_query(question: str, ans: Answer, user_id, user_name, role, chat_id,
               tokens_in: int = 0, tokens_out: int = 0) -> int:
    sources = [{"chunk_id": h.chunk_id, "doc_id": h.doc_id, "path": h.rel_path,
                "score": round(h.score, 5), "channels": h.channels} for h in ans.hits]
    cur = db.run("""INSERT INTO queries(user_id, user_name, role, chat_id, question, answer,
                    sources_json, top_score, answered, latency_ms, tokens_in, tokens_out,
                    created_at, route, stage, channels, n_candidates, rerank_used)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                 (user_id, user_name, role, chat_id, question, ans.text,
                  json.dumps(sources, ensure_ascii=False), ans.confidence,
                  int(ans.answered), ans.latency_ms, tokens_in, tokens_out, _now(),
                  ans.route, ans.stage, ans.channels or _channels_of(ans.hits),
                  ans.n_candidates or len(ans.hits), int(ans.rerank_used)))
    return int(cur.lastrowid)


# ------------------------------------------------------- обратная связь ----
def record_feedback(query_id: int, user_id: int, verdict: str, comment: str = "") -> None:
    db.run("INSERT INTO feedback(query_id, user_id, verdict, comment, created_at) "
           "VALUES (?,?,?,?,?)", (query_id, user_id, verdict, comment, _now()))
    row = db.q1("SELECT question, sources_json FROM queries WHERE id=?", (query_id,))
    if not row:
        return
    sources = json.loads(row["sources_json"] or "[]")
    # Пары «вопрос → чанк» копятся для будущего дообучения эмбеддера/реранкера.
    label = 1 if verdict == "up" else 0
    db.runmany("INSERT INTO training_pairs(question, chunk_id, doc_id, label, source, "
               "created_at) VALUES (?,?,?,?,?,?)",
               [(row["question"], s["chunk_id"], s["doc_id"], label, "feedback", _now())
                for s in sources[:3]])


def add_golden(question: str, answer_text: str, author_id: int | None = None,
               source_refs: list | None = None,
               sections: list[str] | None = None) -> int:
    """
    Добавляет выверенный ответ.

    `sections` — разделы, к которым ответ относится. Пусто означает
    «ответ общий, виден всем ролям». Если не указано явно, разделы
    берутся из источников: ответ, собранный из дилерского прайса, должен
    остаться дилерским, даже если эксперт об этом не подумал.
    """
    if sections is None and source_refs:
        found: set[str] = set()
        for ref in source_refs:
            chunk_id = ref.get("chunk_id") if isinstance(ref, dict) else None
            if not chunk_id:
                continue
            row = db.q1("SELECT d.section FROM chunks c JOIN documents d "
                        "ON d.id=c.doc_id WHERE c.id=?", (chunk_id,))
            if row and row["section"]:
                found.add(row["section"])
        sections = sorted(found)
    cur = db.run("""INSERT INTO golden_qa(question, answer, source_refs, author_id,
                    created_at, updated_at, sections) VALUES (?,?,?,?,?,?,?)""",
                 (question, answer_text, json.dumps(source_refs or [], ensure_ascii=False),
                  author_id, _now(), _now(), "|".join(sections or [])))
    gid = int(cur.lastrowid)
    db.run("INSERT INTO golden_fts(rowid, question, answer) VALUES (?,?,?)",
           (gid, question, answer_text))
    return gid


def unanswered_report(limit: int = 50) -> list[dict]:
    """
    Пробелы в базе знаний: вопросы без ответа и с дизлайком.
    Раз в неделю выгружается экспертам как список задач на пополнение базы.
    """
    rows = db.q("""
        SELECT q.id, q.question, q.created_at, q.user_name,
               (SELECT COUNT(*) FROM feedback f WHERE f.query_id=q.id AND f.verdict='down') AS dislikes
        FROM queries q
        WHERE q.answered=0
           OR EXISTS (SELECT 1 FROM feedback f WHERE f.query_id=q.id AND f.verdict='down')
        ORDER BY q.created_at DESC LIMIT ?""", (limit,))
    return [dict(r) for r in rows]
