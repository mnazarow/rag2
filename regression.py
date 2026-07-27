"""
Автоматическая проверка качества поиска.

Зачем. Любое изменение — новая порция документов, другая модель, правка
весов, переобучение — меняет выдачу. Улучшение одного сценария при этом
незаметно ломает другой, и узнают об этом через месяц по жалобам.

Здесь контрольные вопросы прогоняются автоматически: после полной
переиндексации, после пересчёта векторов, после изменения настроек
поиска. Результат сохраняется вместе со слепком настроек, поэтому на
вопрос «после чего стало хуже» есть точный ответ, а не догадки.

  python regression.py run            — прогнать сейчас
  python regression.py history        — что менялось со временем
  python regression.py diff           — сравнить два последних прогона

Набор вопросов берётся из eval/golden.jsonl. Без него всё остальное
в этом модуле бесполезно: измерять нечем.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import config
import db
import logging_setup

log = logging_setup.get("search")

# Настройки, изменение которых меняет выдачу. Именно их слепок
# сохраняется вместе с результатом.
WATCHED = (
    "EMBEDDINGS_PROVIDER", "EMBEDDINGS_MODEL", "LSA_DIM", "LSA_MAX_FEATURES",
    "RERANKER_PROVIDER", "RERANKER_WEIGHT", "RERANKER_TOP_N",
    "SEARCH_CANDIDATES", "SEARCH_TOP_K", "RRF_K",
    "RECENCY_ALPHA", "RECENCY_HALF_LIFE_DAYS", "MIN_CONFIDENCE",
    "CHUNK_TARGET_CHARS", "CHUNK_OVERLAP_CHARS", "CONTEXTUAL_CHUNKS",
)


def ensure_tables() -> None:
    db.connect().executescript("""
    CREATE TABLE IF NOT EXISTS eval_runs (
        id           INTEGER PRIMARY KEY,
        created_at   TEXT,
        reason       TEXT,          -- что вызвало прогон
        questions    INTEGER,
        hit          REAL,
        mrr          REAL,
        answered     REAL,
        settings_json TEXT,         -- слепок настроек на момент прогона
        details_json TEXT,          -- по каждому вопросу: ранг и что нашлось
        index_docs   INTEGER,
        index_chunks INTEGER
    );
    """)
    db.connect().commit()


def snapshot_settings() -> dict:
    return {key: getattr(config, key, None) for key in WATCHED}


def dataset_path() -> Path:
    return Path(config.BASE_DIR) / "eval" / "golden.jsonl"


def run(reason: str = "вручную", dataset: Path | None = None,
        progress=None) -> dict:
    """Прогоняет контрольные вопросы и сохраняет результат в историю."""
    import evaluate as eval_mod
    say = progress or (lambda t: print(t, flush=True))
    db.init()
    ensure_tables()

    path = dataset or dataset_path()
    if not path.exists():
        raise FileNotFoundError(
            f"нет набора контрольных вопросов ({path}). Соберите 50–150 реальных "
            f"вопросов сотрудников с указанием, где лежит ответ — без него "
            f"качество поиска нечем измерять.")
    data = eval_mod.load(path)
    say(f"Прогоняю {len(data)} контрольных вопросов ({reason})")

    result = eval_mod.evaluate(data, config.SEARCH_TOP_K, run_llm=False)
    hit = result[f"hit@{config.SEARCH_TOP_K}"]
    counts = db.q1("SELECT (SELECT COUNT(*) FROM documents WHERE status='ok') d, "
                   "(SELECT COUNT(*) FROM chunks) c")

    cur = db.run("""INSERT INTO eval_runs(created_at, reason, questions, hit, mrr,
                    answered, settings_json, details_json, index_docs, index_chunks)
                    VALUES (?,?,?,?,?,?,?,?,?,?)""",
                 (datetime.now(timezone.utc).isoformat(timespec="seconds"), reason,
                  len(data), hit, result["mrr"], result.get("answered"),
                  json.dumps(snapshot_settings(), ensure_ascii=False, default=str),
                  json.dumps(result["details"], ensure_ascii=False),
                  counts["d"], counts["c"]))

    previous = db.q1("SELECT hit, mrr FROM eval_runs WHERE id<? ORDER BY id DESC LIMIT 1",
                     (cur.lastrowid,))
    delta = None
    if previous:
        delta = {"hit": round(hit - previous["hit"], 3),
                 "mrr": round(result["mrr"] - previous["mrr"], 3)}
        arrow = "лучше" if delta["mrr"] > 0 else "хуже" if delta["mrr"] < 0 else "без изменений"
        say(f"hit@{config.SEARCH_TOP_K}: {hit} (было {previous['hit']}), "
            f"MRR: {result['mrr']} (было {previous['mrr']}) — {arrow}")
        if delta["mrr"] <= -0.03:
            log.warning("качество поиска упало: MRR %.3f → %.3f после «%s»",
                        previous["mrr"], result["mrr"], reason)
    else:
        say(f"hit@{config.SEARCH_TOP_K}: {hit}, MRR: {result['mrr']} — первый прогон")

    misses = [d["question"] for d in result["details"] if d["rank"] is None]
    if misses:
        say(f"Не нашлось ответа на {len(misses)} вопросов")
    return {"id": int(cur.lastrowid), "questions": len(data), "hit": hit,
            "mrr": result["mrr"], "delta": delta, "misses": misses[:20],
            "reason": reason}


def history(limit: int = 40) -> list[dict]:
    ensure_tables()
    rows = db.q("""SELECT id, created_at, reason, questions, hit, mrr,
                          index_docs, index_chunks, settings_json
                   FROM eval_runs ORDER BY id DESC LIMIT ?""", (limit,))
    out = []
    for r in rows:
        item = dict(r)
        item["settings"] = json.loads(item.pop("settings_json") or "{}")
        out.append(item)
    return out


def diff(older: int | None = None, newer: int | None = None) -> dict:
    """
    Чем отличаются два прогона: и числами, и настройками, и по вопросам.

    Самое полезное здесь — список вопросов, которые перестали находиться.
    Средние цифры могут почти не измениться, а конкретный сценарий при
    этом сломаться.
    """
    ensure_tables()
    runs = db.q("SELECT * FROM eval_runs ORDER BY id DESC LIMIT 2") \
        if older is None or newer is None else \
        db.q("SELECT * FROM eval_runs WHERE id IN (?,?) ORDER BY id DESC", (newer, older))
    if len(runs) < 2:
        return {"error": "нужно хотя бы два прогона"}
    new, old = dict(runs[0]), dict(runs[1])
    s_new = json.loads(new["settings_json"] or "{}")
    s_old = json.loads(old["settings_json"] or "{}")
    changed = {k: [s_old.get(k), s_new.get(k)] for k in WATCHED
               if str(s_old.get(k)) != str(s_new.get(k))}

    d_new = {d["question"]: d["rank"] for d in json.loads(new["details_json"] or "[]")}
    d_old = {d["question"]: d["rank"] for d in json.loads(old["details_json"] or "[]")}
    broken = [q for q, r in d_new.items() if r is None and d_old.get(q) is not None]
    fixed = [q for q, r in d_new.items() if r is not None and d_old.get(q) is None]
    worse = [(q, d_old[q], r) for q, r in d_new.items()
             if r and d_old.get(q) and r > d_old[q]]
    better = [(q, d_old[q], r) for q, r in d_new.items()
              if r and d_old.get(q) and r < d_old[q]]
    return {
        "new": {"id": new["id"], "at": new["created_at"], "reason": new["reason"],
                "hit": new["hit"], "mrr": new["mrr"]},
        "old": {"id": old["id"], "at": old["created_at"], "reason": old["reason"],
                "hit": old["hit"], "mrr": old["mrr"]},
        "delta": {"hit": round((new["hit"] or 0) - (old["hit"] or 0), 3),
                  "mrr": round((new["mrr"] or 0) - (old["mrr"] or 0), 3)},
        "settings_changed": changed,
        "broken": broken, "fixed": fixed,
        "worse": worse[:20], "better": better[:20],
    }


def settings_changed_since_last_run() -> dict:
    """Настройки, изменившиеся после последнего прогона — повод прогнать заново."""
    ensure_tables()
    row = db.q1("SELECT settings_json FROM eval_runs ORDER BY id DESC LIMIT 1")
    if not row:
        return {}
    was = json.loads(row["settings_json"] or "{}")
    now = snapshot_settings()
    return {k: [was.get(k), now.get(k)] for k in WATCHED
            if str(was.get(k)) != str(now.get(k))}


def main() -> int:
    p = argparse.ArgumentParser(description="Автопроверка качества поиска")
    p.add_argument("command", choices=["run", "history", "diff"])
    p.add_argument("--reason", default="вручную")
    p.add_argument("--dataset")
    args = p.parse_args()
    db.init()
    if args.command == "run":
        run(reason=args.reason,
            dataset=Path(args.dataset) if args.dataset else None)
    elif args.command == "history":
        rows = history()
        if not rows:
            print("Прогонов ещё не было.")
            return 0
        print(f"{'№':>4} {'когда':17} {'hit':>6} {'MRR':>6} {'вопросов':>9}  причина")
        for r in rows:
            print(f"{r['id']:>4} {r['created_at'][:16]:17} {r['hit']:>6} "
                  f"{r['mrr']:>6} {r['questions']:>9}  {r['reason']}")
    elif args.command == "diff":
        d = diff()
        if "error" in d:
            print(d["error"])
            return 1
        print(f"Было: прогон №{d['old']['id']} ({d['old']['reason']}), "
              f"MRR {d['old']['mrr']}")
        print(f"Стало: прогон №{d['new']['id']} ({d['new']['reason']}), "
              f"MRR {d['new']['mrr']}  ({d['delta']['mrr']:+})")
        if d["settings_changed"]:
            print("\nИзменились настройки:")
            for k, (a, b) in d["settings_changed"].items():
                print(f"  {k}: {a} → {b}")
        if d["broken"]:
            print(f"\nПерестали находиться ({len(d['broken'])}):")
            for q in d["broken"][:15]:
                print(f"  ✗ {q}")
        if d["fixed"]:
            print(f"\nСтали находиться ({len(d['fixed'])}):")
            for q in d["fixed"][:15]:
                print(f"  ✓ {q}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
