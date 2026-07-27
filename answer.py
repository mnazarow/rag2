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


def build_context(hits: list[search_mod.Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
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
        blocks.append(f"{header}\n{h.text}")
    return "\n\n".join(blocks)


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

    # 1. Выверенный экспертом ответ — если есть близкий, отдаём его.
    golden = search_mod.golden_search(question, limit=1)
    if golden and _golden_is_close(question, golden[0]["question"]):
        g = golden[0]
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
    if prices.looks_like_price_question(question) or prices.ARTICLE_RX.search(question):
        product_rows = prices.search_products(question, limit=8)

    # 3. Документный поиск.
    _t = time.time()
    hits = search_mod.search(question, role=role)
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

    context = build_context(hits)
    if product_rows:
        context = ("[ПРАЙС-ЛИСТ — точные данные из таблицы, приоритет над остальным]\n"
                   + prices.format_products(product_rows) + "\n\n" + context)

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
    except llm_mod.LLMBusy as exc:
        # Очередь переполнена. Показать фрагменты честнее, чем молчать:
        # человек хотя бы увидит, где ответ, пока модель занята.
        logger.warning("отказ по очереди к модели: %s", exc)
        text = (f"{exc}\n\nПока модель занята, вот наиболее подходящие фрагменты базы:\n\n"
                + "\n\n".join(f"[{i}] {h.text[:400]}" for i, h in enumerate(hits[:3], 1)))
        tokens_in = tokens_out = 0
        model = "queue-busy"
    except Exception as exc:  # noqa: BLE001
        text = (f"Модель недоступна ({exc}). Вот наиболее подходящие фрагменты базы:\n\n"
                + "\n\n".join(f"[{i}] {h.text[:400]}" for i, h in enumerate(hits[:3], 1)))
        tokens_in = tokens_out = 0
        model = "fallback"

    # Последняя проверка — по результату, а не по формулировке вопроса:
    # она ловит утечку из недоступного раздела независимо от того, каким
    # способом её добились.
    leak = security.check_answer_leak(text, hits, config.ROLE_SECTIONS.get(role))
    if leak["leak"]:
        logger.error("ответ содержал фрагменты из недоступных роли «%s» разделов: "
                     "%s — выдача заменена отказом", role, leak["sections"])
        text = security.SAFE_REFUSAL
        hits = []

    ans = Answer(text=text, hits=hits, products=product_rows, answered=True,
                 confidence=conf, latency_ms=int((time.time() - started) * 1000),
                 llm_model=model,
                 route="price" if product_rows else "documents", stage="answered",
                 channels=_channels_of(hits), n_candidates=len(hits),
                 rerank_used=any(h.channels.get("rerank") is not None for h in hits))
    if log:
        ans.query_id = _log_query(question, ans, user_id, user_name, role, chat_id,
                                  tokens_in, tokens_out)
        _trace(question, ans, context, user_id, user_name, role, started)
    return ans


def _golden_is_close(question: str, golden_question: str, threshold: float = 0.6) -> bool:
    """Простое пересечение токенов: не выдаём выверенный ответ на другой вопрос."""
    a = set(re.findall(r"[\wа-яё]{3,}", question.lower()))
    b = set(re.findall(r"[\wа-яё]{3,}", golden_question.lower()))
    if not a or not b:
        return False
    return len(a & b) / len(a | b) >= threshold


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
               source_refs: list | None = None) -> int:
    cur = db.run("""INSERT INTO golden_qa(question, answer, source_refs, author_id,
                    created_at, updated_at) VALUES (?,?,?,?,?,?)""",
                 (question, answer_text, json.dumps(source_refs or [], ensure_ascii=False),
                  author_id, _now(), _now()))
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
