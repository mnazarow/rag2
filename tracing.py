"""
Трассировка ответов: полная цепочка от вопроса до ответа.

Зачем. Метрики говорят, что средняя задержка выросла, а доля отказов —
двенадцать процентов. Но когда приходит конкретная жалоба — «спросил
про гарантию на Джилекс, а он ответил про Grundfos», — нужны не средние
значения, а именно этот случай целиком: что нашёл каждый канал, с какими
оценками, что переранжирование сделало с порядком, какой текст ушёл в
модель и что она вернула.

Без такой записи разбор превращается в попытки воспроизвести проблему
на изменившемся индексе, и обычно она не воспроизводится.

Хранится в базе телеметрии, отдельно от индекса: это оперативные данные
с ограниченным сроком жизни, а не то, что нужно возить в резервных копиях.

  python tracing.py list                — последние цепочки
  python tracing.py show 42             — одна целиком
  python tracing.py find "гарантия"     — по тексту вопроса
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import config
import db


def ensure_tables() -> None:
    db.telemetry().executescript("""
    CREATE TABLE IF NOT EXISTS traces (
        id           INTEGER PRIMARY KEY,
        ts           TEXT,
        request_id   TEXT,
        query_id     INTEGER,
        user_id      INTEGER,
        user_name    TEXT,
        role         TEXT,
        question     TEXT,
        route        TEXT,
        stage        TEXT,
        confidence   REAL,
        answered     INTEGER,
        hits_json    TEXT,
        prompt       TEXT,
        answer       TEXT,
        model        TEXT,
        tokens_in    INTEGER,
        tokens_out   INTEGER,
        timings_json TEXT,
        settings_json TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_traces_ts ON traces(ts);
    CREATE INDEX IF NOT EXISTS idx_traces_query ON traces(query_id);
    """)
    db.telemetry().commit()


def record(question: str, ans, prompt: str = "", timings: dict | None = None,
           user_id=None, user_name=None, role=None) -> int | None:
    """Сохраняет цепочку. Никогда не мешает ответу: любая беда гасится."""
    if not config.TRACE_ENABLED:
        return None
    try:
        ensure_tables()
        hits = [{
            "path": h.rel_path, "chunk_id": h.chunk_id, "doc_id": h.doc_id,
            "score": round(h.score, 6), "channels": h.channels,
            "page": h.page_from, "is_current": bool(h.is_current),
            "text": (h.text or "")[:400],
        } for h in (ans.hits or [])]
        settings = {
            "embeddings": config.EMBEDDINGS_PROVIDER,
            "reranker": config.RERANKER_PROVIDER,
            "reranker_weight": config.RERANKER_WEIGHT,
            "llm": config.LLM_PROVIDER,
            "min_confidence": config.MIN_CONFIDENCE,
            "recency_alpha": config.RECENCY_ALPHA,
            "top_k": config.SEARCH_TOP_K,
        }
        import logging_setup
        cur = db.trun("""INSERT INTO traces(ts, request_id, query_id, user_id, user_name,
                         role, question, route, stage, confidence, answered, hits_json,
                         prompt, answer, model, tokens_in, tokens_out, timings_json,
                         settings_json)
                         VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                      (datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       logging_setup.current_request(), ans.query_id, user_id, user_name,
                       role, question, ans.route, ans.stage, ans.confidence,
                       int(ans.answered), json.dumps(hits, ensure_ascii=False),
                       prompt if config.TRACE_PROMPT else "",
                       (ans.text or "")[:6000], ans.llm_model, 0, 0,
                       json.dumps(timings or {}, ensure_ascii=False),
                       json.dumps(settings, ensure_ascii=False)))
        # Срок хранения: цепочки содержат тексты вопросов, держать их вечно
        # неправильно ни с точки зрения места, ни с точки зрения данных.
        db.trun("DELETE FROM traces WHERE id < (SELECT MAX(id) - ? FROM traces)",
                (config.TRACE_KEEP,))
        return int(cur.lastrowid)
    except Exception:  # noqa: BLE001 — трассировка не имеет права ломать ответ
        return None


def recent(limit: int = 50, only_bad: bool = False) -> list[dict]:
    ensure_tables()
    sql = """SELECT id, ts, question, route, stage, confidence, answered, model,
                    user_name, query_id FROM traces"""
    if only_bad:
        sql += " WHERE answered=0 OR stage <> 'answered'"
    sql += " ORDER BY id DESC LIMIT ?"
    return [dict(r) for r in db.tq(sql, (limit,))]


def get(trace_id: int) -> dict:
    ensure_tables()
    row = db.tq1("SELECT * FROM traces WHERE id=?", (trace_id,))
    if row is None:
        return {}
    item = dict(row)
    for key in ("hits_json", "timings_json", "settings_json"):
        item[key.replace("_json", "")] = json.loads(item.pop(key) or "null")
    return item


def by_query(query_id: int) -> dict:
    ensure_tables()
    row = db.tq1("SELECT id FROM traces WHERE query_id=? ORDER BY id DESC LIMIT 1",
                 (query_id,))
    return get(row["id"]) if row else {}


def find(text: str, limit: int = 30) -> list[dict]:
    ensure_tables()
    return [dict(r) for r in db.tq(
        """SELECT id, ts, question, route, stage, confidence, answered
           FROM traces WHERE question LIKE ? ORDER BY id DESC LIMIT ?""",
        (f"%{text}%", limit))]


def main() -> int:
    p = argparse.ArgumentParser(description="Трассировка ответов")
    p.add_argument("command", choices=["list", "show", "find", "bad"])
    p.add_argument("arg", nargs="?")
    args = p.parse_args()
    db.init()
    if args.command in ("list", "bad"):
        rows = recent(40, only_bad=args.command == "bad")
        if not rows:
            print("Записей нет.")
            return 0
        print(f"{'№':>5} {'когда':17} {'ответ':>6} {'уверен.':>8}  вопрос")
        for r in rows:
            print(f"{r['id']:>5} {r['ts'][:16]:17} {'да' if r['answered'] else 'нет':>6} "
                  f"{(r['confidence'] or 0):>8.4f}  {r['question'][:70]}")
    elif args.command == "show":
        t = get(int(args.arg or 0))
        if not t:
            print("Не найдено.")
            return 1
        print(f"Вопрос: {t['question']}")
        print(f"Когда: {t['ts']}   маршрут: {t['route']}   этап: {t['stage']}")
        print(f"Уверенность: {t['confidence']}   модель: {t['model']}")
        print(f"Настройки на тот момент: {json.dumps(t['settings'], ensure_ascii=False)}")
        print(f"Тайминги: {json.dumps(t['timings'], ensure_ascii=False)}")
        print("\nЧто нашлось:")
        for i, h in enumerate(t["hits"] or [], 1):
            channels = ", ".join(f"{k}={v}" for k, v in (h["channels"] or {}).items()
                                 if v is not None)
            print(f"  [{i}] {h['score']:.5f}  {h['path']}")
            print(f"      каналы: {channels}")
            print(f"      {h['text'][:150]}")
        if t.get("prompt"):
            print("\nЧто ушло в модель:")
            print(t["prompt"][:2000])
        print("\nОтвет:")
        print(t["answer"])
    elif args.command == "find":
        for r in find(args.arg or ""):
            print(f"{r['id']:>5} {r['ts'][:16]}  {r['question'][:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
