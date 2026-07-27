"""
Запуск задачи из очереди одной командой — для расписания и для рук.

  python runjob.py restore_drill
  python runjob.py alerts
  python runjob.py reindex --limit 100

Отличие от прямого вызова модуля: задача проходит через очередь, поэтому
подчиняется тем же блокировкам. Учебное восстановление, запущенное по
расписанию посреди индексации, не испортит индекс — оно просто получит
отказ и напишет об этом в журнал.
"""
from __future__ import annotations

import argparse
import json
import sys

import db
import handlers  # noqa: F401 — регистрация обработчиков
import jobs


def main() -> int:
    p = argparse.ArgumentParser(description="Запуск задачи из очереди")
    p.add_argument("kind", choices=sorted(jobs.HANDLERS))
    p.add_argument("--limit", type=int)
    p.add_argument("--provider")
    p.add_argument("--reason", default="расписание")
    p.add_argument("--wait", action="store_true",
                   help="дождаться выполнения (по умолчанию так и делается)")
    args = p.parse_args()

    db.init()
    jobs.ensure_tables()
    payload = {k: v for k, v in
               {"limit": args.limit, "provider": args.provider,
                "reason": args.reason}.items() if v is not None}
    try:
        job = jobs.enqueue(args.kind, args.kind, payload, created_by="командная строка")
    except jobs.Busy as exc:
        print(f"Отложено: {exc}")
        return 0
    claimed = jobs._claim()
    if claimed is None or claimed["id"] != job["id"]:
        print("Задача поставлена в очередь; её выполнит работающий обработчик.")
        return 0
    jobs._run_one(claimed)
    done = jobs.get(job["id"])
    print(json.dumps({"id": done["id"], "status": done["status"],
                      "result": done.get("result"), "error": done.get("error")},
                     ensure_ascii=False, default=str)[:2000])
    return 0 if done["status"] == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
