"""
Срок хранения данных и удаление по запросу сотрудника.

Тексты вопросов — персональные данные. Хранить их бессрочно нельзя ни
по закону, ни по здравому смыслу: через год записи о том, кто что
спрашивал, не помогают никому, а риск создают.

Здесь три вещи. Плановая очистка по сроку. Удаление всего, что связано
с конкретным сотрудником, — по его запросу или при увольнении.
И обезличивание: если сами вопросы нужны для анализа пробелов в базе,
но связь с человеком не нужна, идентификаторы стираются, а тексты
остаются.

Выверенные ответы и обучающие пары не трогаются никогда: это результат
работы экспертов, а не переписка. Если в выверенном ответе оказались
персональные данные, убирать их нужно осознанно, а не сроком хранения.

  python retention.py status         — что и сколько хранится
  python retention.py clean          — удалить просроченное
  python retention.py forget 12345   — удалить всё об этом сотруднике
  python retention.py anonymize      — стереть связь вопросов с людьми
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone

from pathlib import Path

import config
import db
import logging_setup

log = logging_setup.get("db")


def _cutoff(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")


def status() -> dict:
    """Что сейчас хранится и что попадёт под очистку."""
    db.init()
    out: dict = {"queries_days": config.RETENTION_QUERIES_DAYS,
                 "traces_days": config.RETENTION_TRACES_DAYS}
    total = db.q1("SELECT COUNT(*) n, MIN(created_at) oldest FROM queries")
    out["queries"] = total["n"]
    out["queries_oldest"] = total["oldest"]
    if config.RETENTION_QUERIES_DAYS:
        out["queries_expired"] = db.q1(
            "SELECT COUNT(*) n FROM queries WHERE created_at < ?",
            (_cutoff(config.RETENTION_QUERIES_DAYS),))["n"]
    else:
        out["queries_expired"] = 0
    try:
        import tracing
        tracing.ensure_tables()
        out["traces"] = db.tq1("SELECT COUNT(*) n FROM traces")["n"]
        out["traces_expired"] = db.tq1(
            "SELECT COUNT(*) n FROM traces WHERE ts < ?",
            (_cutoff(config.RETENTION_TRACES_DAYS),))["n"] \
            if config.RETENTION_TRACES_DAYS else 0
    except Exception:  # noqa: BLE001
        out["traces"] = out["traces_expired"] = 0
    out["golden"] = db.q1("SELECT COUNT(*) n FROM golden_qa")["n"]
    out["training_pairs"] = db.q1("SELECT COUNT(*) n FROM training_pairs")["n"]
    out["users"] = db.q1("SELECT COUNT(*) n FROM users")["n"]
    out["identified"] = db.q1("SELECT COUNT(*) n FROM queries "
                              "WHERE user_id IS NOT NULL")["n"]
    return out


def clean(dry: bool = False) -> dict:
    """Удаляет просроченное. Выверенные ответы и обучающие пары не трогает."""
    db.init()
    removed = {"queries": 0, "feedback": 0, "traces": 0}
    if config.RETENTION_QUERIES_DAYS:
        cutoff = _cutoff(config.RETENTION_QUERIES_DAYS)
        removed["queries"] = db.q1("SELECT COUNT(*) n FROM queries WHERE created_at < ?",
                                   (cutoff,))["n"]
        if not dry and removed["queries"]:
            # Оценки привязаны к вопросам — уходят вместе с ними.
            db.run("DELETE FROM feedback WHERE query_id IN "
                   "(SELECT id FROM queries WHERE created_at < ?)", (cutoff,))
            db.run("DELETE FROM queries WHERE created_at < ?", (cutoff,))
    if config.RETENTION_TRACES_DAYS:
        try:
            import tracing
            tracing.ensure_tables()
            cutoff = _cutoff(config.RETENTION_TRACES_DAYS)
            removed["traces"] = db.tq1("SELECT COUNT(*) n FROM traces WHERE ts < ?",
                                       (cutoff,))["n"]
            if not dry and removed["traces"]:
                db.trun("DELETE FROM traces WHERE ts < ?", (cutoff,))
        except Exception:  # noqa: BLE001
            pass
    # Записи очереди к модели персональных данных не содержат, но копятся
    # по строке на каждый запрос. Держим неделю: этого хватает и статистике
    # в админке, и разбору «почему вчера всё тормозило».
    if not dry:
        try:
            import llm_queue
            removed["llm_queue"] = llm_queue.prune(7)
        except Exception:  # noqa: BLE001
            removed["llm_queue"] = 0
    # Учёт обращений к моделям и тайминги стадий растут по нескольку строк
    # на каждый вопрос — без срока хранения телеметрия пухнет вечно.
    if not dry and config.METRICS_KEEP_DAYS:
        try:
            cutoff = _cutoff(config.METRICS_KEEP_DAYS)
            for table in ("model_usage", "stage_timings"):
                db.trun(f"DELETE FROM {table} WHERE ts < ?", (cutoff,))
        except Exception:  # noqa: BLE001
            pass

    # Распакованные архивы: ключ папки — хеш содержимого, поэтому при
    # замене архива новой версией старая распаковка остаётся навсегда.
    removed["unpacked"] = 0
    if not dry:
        try:
            import shutil
            import time as _time
            work = Path(config.ARCHIVE_WORK_DIR)
            if work.exists():
                horizon = _time.time() - 30 * 86400
                for d in work.iterdir():
                    if d.is_dir() and d.stat().st_mtime < horizon:
                        shutil.rmtree(d, ignore_errors=True)
                        removed["unpacked"] += 1
        except Exception:  # noqa: BLE001
            pass

    # Журналы, которые пишутся мимо ротации (редиректы systemd/launchd и
    # cron): усечь до хвоста, когда переросли разумный размер.
    if not dry:
        try:
            for name in ("service.log", "bot-service.log", "schedule.log"):
                f = Path(config.LOG_DIR) / name
                if f.exists() and f.stat().st_size > 50 * 2**20:
                    tail = f.read_bytes()[-5 * 2**20:]
                    f.write_bytes(b"[...ranee usecheno...]\n" + tail)
        except Exception:  # noqa: BLE001
            pass

    # Базы после удалений сами не худеют: раз в месяц уплотняем. Дёшево
    # и возвращает место после больших чисток.
    if not dry:
        try:
            import datetime as _dt
            if _dt.date.today().day == 1:
                db.connect().execute("VACUUM")
                db.telemetry().execute("VACUUM")
        except Exception:  # noqa: BLE001
            pass

    if not dry and any(removed.values()):
        log.warning("очистка по сроку хранения: вопросов %d, цепочек %d, "
                    "записей очереди %d, распаковок %d", removed["queries"],
                    removed["traces"], removed.get("llm_queue", 0),
                    removed["unpacked"])
    return removed


def forget(user_id: int, keep_questions: bool = False) -> dict:
    """
    Удаляет всё, что связано с сотрудником.

    keep_questions оставляет сами тексты вопросов без связи с человеком:
    иногда нужно сохранить статистику пробелов в базе, но убрать
    персональные данные. Это разные вещи, и решать должен человек.
    """
    db.init()
    result = {"user_id": user_id}
    result["queries"] = db.q1("SELECT COUNT(*) n FROM queries WHERE user_id=?",
                              (user_id,))["n"]
    db.run("DELETE FROM feedback WHERE user_id=?", (user_id,))
    if keep_questions:
        db.run("UPDATE queries SET user_id=NULL, user_name=NULL, chat_id=NULL "
               "WHERE user_id=?", (user_id,))
        result["mode"] = "обезличено"
    else:
        db.run("DELETE FROM feedback WHERE query_id IN "
               "(SELECT id FROM queries WHERE user_id=?)", (user_id,))
        db.run("DELETE FROM queries WHERE user_id=?", (user_id,))
        result["mode"] = "удалено"
    try:
        import tracing
        tracing.ensure_tables()
        result["traces"] = db.tq1("SELECT COUNT(*) n FROM traces WHERE user_id=?",
                                  (user_id,))["n"]
        db.trun("DELETE FROM traces WHERE user_id=?", (user_id,))
    except Exception:  # noqa: BLE001
        result["traces"] = 0
    db.run("DELETE FROM users WHERE user_id=?", (user_id,))
    log.warning("удалены данные сотрудника %s: вопросов %d, цепочек %d (%s)",
                user_id, result["queries"], result.get("traces", 0), result["mode"])
    return result


def anonymize(older_than_days: int = 90) -> dict:
    """Стирает связь старых вопросов с людьми, оставляя тексты."""
    db.init()
    cutoff = _cutoff(older_than_days)
    n = db.q1("SELECT COUNT(*) n FROM queries WHERE created_at < ? "
              "AND user_id IS NOT NULL", (cutoff,))["n"]
    db.run("UPDATE queries SET user_id=NULL, user_name=NULL, chat_id=NULL "
           "WHERE created_at < ?", (cutoff,))
    try:
        db.trun("UPDATE traces SET user_id=NULL, user_name=NULL WHERE ts < ?", (cutoff,))
    except Exception:  # noqa: BLE001
        pass
    log.warning("обезличено вопросов: %d (старше %d дней)", n, older_than_days)
    return {"anonymized": n, "older_than_days": older_than_days}


def main() -> int:
    p = argparse.ArgumentParser(description="Срок хранения данных")
    p.add_argument("command", choices=["status", "clean", "forget", "anonymize"])
    p.add_argument("user_id", nargs="?", type=int)
    p.add_argument("--dry", action="store_true", help="показать, но не удалять")
    p.add_argument("--keep-questions", action="store_true",
                   help="оставить тексты вопросов, убрав связь с человеком")
    p.add_argument("--days", type=int, default=90)
    args = p.parse_args()
    db.init()

    if args.command == "status":
        st = status()
        print(f"Вопросов: {st['queries']}, самый старый: {st['queries_oldest'] or '—'}")
        print(f"Срок хранения вопросов: "
              f"{st['queries_days'] or 'бессрочно'} дней, "
              f"просрочено: {st['queries_expired']}")
        print(f"Цепочек трассировки: {st['traces']}, "
              f"срок {st['traces_days']} дней, просрочено: {st['traces_expired']}")
        print(f"Связано с конкретными людьми: {st['identified']} вопросов")
        print(f"Не удаляются никогда: выверенных ответов {st['golden']}, "
              f"обучающих пар {st['training_pairs']}")
        if not st["queries_days"]:
            print("\nСрок хранения не задан — вопросы копятся бессрочно. "
                  "Это персональные данные; задайте RETENTION_QUERIES_DAYS.")
    elif args.command == "clean":
        removed = clean(dry=args.dry)
        print(("Было бы удалено: " if args.dry else "Удалено: ")
              + f"вопросов {removed['queries']}, цепочек {removed['traces']}")
    elif args.command == "forget":
        if not args.user_id:
            print("Укажите идентификатор: python retention.py forget 12345")
            return 2
        print(forget(args.user_id, keep_questions=args.keep_questions))
    elif args.command == "anonymize":
        print(anonymize(args.days))
    return 0


if __name__ == "__main__":
    sys.exit(main())
