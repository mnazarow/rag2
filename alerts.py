"""
Оповещения о том, что иначе замечают поздно.

Метрики и диагностика показывают состояние — но только тому, кто зашёл
в админку. А проблемы, которые здесь проверяются, обнаруживают себя
по-разному: кончившееся место останавливает индексацию сразу, а
отсутствие резервной копии — только в день аварии.

Проверки намеренно немногочисленны. Каждая соответствует событию, после
которого нужно что-то сделать, и молчит, пока делать нечего: система,
которая шлёт сообщения постоянно, перестаёт читаться через неделю.

Повторные напоминания об одном и том же приходят не чаще, чем раз в
ALERT_REPEAT_HOURS, — иначе одна незакрытая проблема заглушает все
остальные.

  python alerts.py check      — проверить и сообщить о проблемах
  python alerts.py list       — что сейчас не так, без отправки
  python alerts.py test       — проверить, что оповещения доходят
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys
import time
from datetime import datetime, timedelta, timezone

import config
import db
import logging_setup

log = logging_setup.get("web")


def ensure_tables() -> None:
    db.telemetry().executescript("""
    CREATE TABLE IF NOT EXISTS alerts (
        id         INTEGER PRIMARY KEY,
        key        TEXT,
        level      TEXT,          -- warning | error
        title      TEXT,
        detail     TEXT,
        action     TEXT,
        first_seen TEXT,
        last_seen  TEXT,
        notified_at TEXT,
        resolved_at TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_alerts_key ON alerts(key);
    """)
    db.telemetry().commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ═══════════════════════════════════════════════════════ сами проверки ═════
def collect() -> list[dict]:
    """Что сейчас не так. Каждая запись — повод что-то сделать."""
    found: list[dict] = []

    def add(key, level, title, detail, action):
        found.append({"key": key, "level": level, "title": title,
                      "detail": detail, "action": action})

    # Резервные копии.
    try:
        import backup
        state = backup.status()
        if not state["count"]:
            add("backup_none", "error", "Резервных копий индекса нет",
                "Выверенные ответы и обучающие пары пересборкой не восстанавливаются.",
                "Раздел «Копии» → «Сделать копию сейчас», затем включить расписание.")
        elif state.get("stale"):
            add("backup_stale", "error", "Свежей резервной копии нет",
                f"Последняя копия сделана {state.get('age_hours')} ч назад "
                f"при пороге {config.BACKUP_ALERT_HOURS} ч.",
                "Проверьте, работает ли расписание: python backup.py schedule")
        if state["count"] and not state.get("installed"):
            add("backup_schedule", "warning", "Копии делаются только вручную",
                "Регулярное расписание не настроено.",
                "python backup.py schedule")
    except Exception as exc:  # noqa: BLE001
        add("backup_error", "error", "Не удалось проверить резервные копии",
            str(exc), "Смотрите журнал подсистемы backup.")

    # Место на диске — на всех разделах, где живут данные, журналы,
    # копии и распаковки. Раньше проверялся только раздел с данными:
    # журналы на системном диске могли заполнить его до отказа, и ни
    # одна проверка этого не видела.
    try:
        seen_devices: set[int] = set()
        for label, path in (("данные", config.DATA_DIR),
                            ("журналы", config.LOG_DIR),
                            ("копии", config.BACKUP_DIR),
                            ("распаковки", config.ARCHIVE_WORK_DIR)):
            try:
                dev = Path(path).stat().st_dev
            except OSError:
                continue
            if dev in seen_devices:
                continue
            seen_devices.add(dev)
            usage = shutil.disk_usage(path)
            free_gb = usage.free / 1e9
            if free_gb < config.ALERT_DISK_FREE_GB:
                add(f"disk_{label}", "error", f"Кончается место на диске ({label})",
                    f"Свободно {free_gb:.1f} ГБ при пороге "
                    f"{config.ALERT_DISK_FREE_GB} ГБ: {path}",
                    "Удалите старые копии и распакованные архивы: "
                    "python backup.py prune, очистите ARCHIVE_WORK_DIR.")
    except Exception:  # noqa: BLE001
        pass

    # Живость бота. Процесс, который умер ночью от нехватки памяти,
    # для сотрудников выглядит как «ассистент молчит», а для админа —
    # никак: в админке всё зелёное. Пульс пишет сам бот раз в минуту.
    try:
        if config.TELEGRAM_BOT_TOKEN:
            import metrics
            beat = metrics.last_beat("бот")
            if beat is None:
                add("bot_down", "error", "Бот не запущен",
                    "Токен Telegram настроен, но бот ни разу не отметился.",
                    "Запустите: python bot.py (или службу kb-assistant-bot).")
            else:
                age_min = (datetime.now(timezone.utc)
                           - datetime.fromisoformat(beat)).total_seconds() / 60
                if age_min > 10:
                    add("bot_silent", "error", "Бот перестал отвечать",
                        f"Последняя отметка о жизни — {age_min:.0f} мин назад.",
                        "Проверьте процесс bot.py и журнал logs/bot.log; "
                        "перезапустите службу kb-assistant-bot.")
    except Exception:  # noqa: BLE001
        pass

    # Копии на том же диске, что и данные, не переживут его отказ.
    try:
        if not config.BACKUP_MIRROR_DIR:
            add("backup_mirror", "warning", "Копии хранятся на том же диске",
                "BACKUP_MIRROR_DIR не задан: отказ диска унесёт и данные, "
                "и все резервные копии разом.",
                "Укажите BACKUP_MIRROR_DIR — сетевую папку или внешний диск.")
        else:
            mirror = Path(config.BACKUP_MIRROR_DIR).expanduser()
            copies = sorted(mirror.glob("*.tar.gz"),
                            key=lambda f: f.stat().st_mtime, reverse=True)                 if mirror.exists() else []
            if not copies:
                add("mirror_empty", "error", "Зеркало копий пусто",
                    f"В {mirror} нет ни одной копии.",
                    "Проверьте доступность папки и запустите python backup.py run.")
            else:
                age_h = (datetime.now(timezone.utc).timestamp()
                         - copies[0].stat().st_mtime) / 3600
                if age_h > config.BACKUP_ALERT_HOURS:
                    add("mirror_stale", "error", "Зеркало копий устарело",
                        f"Свежая копия в зеркале — {age_h:.0f} ч назад "
                        f"при пороге {config.BACKUP_ALERT_HOURS} ч.",
                        "Проверьте доступность зеркала: python backup.py run")
    except Exception:  # noqa: BLE001
        pass

    # Смысловой канал поиска.
    try:
        import search
        ok, note = search.dense_ready()
        if not ok:
            add("dense", "error", "Смысловой поиск не работает", note,
                "Раздел «Качество поиска»: обучить модель и пересчитать векторы.")
    except Exception as exc:  # noqa: BLE001
        add("dense_error", "warning", "Не удалось проверить смысловой поиск",
            str(exc), "")

    # Модель генерации.
    try:
        import llm
        info = llm.describe()
        if info.get("is_stub"):
            add("llm_stub", "error", "Ответы собираются без модели",
                "Основной провайдер — заглушка echo: ответ склеивается из "
                "найденных предложений.",
                "Раздел «Модели»: запустить локальную модель или выбрать облако.")
        elif info.get("failed") and not info.get("ready"):
            add("llm_down", "error", "Модель генерации недоступна",
                "; ".join(f"{f['provider']}: {f['error']}" for f in info["failed"]),
                "Проверьте сервер модели и ключи. Запасной провайдер задаётся "
                "настройкой LLM_FALLBACK.")
        elif info.get("switches"):
            last = info["switches"][-1]
            add("llm_switched", "warning", "Ассистент отвечает запасным провайдером",
                f"{last['at']}: {last['from']} → {last['to']} ({last['why']})",
                "Разберитесь с основным: обычно это упавший сервер модели.")
    except Exception:  # noqa: BLE001
        pass

    # Очередь запросов к модели. Смысл проверки: отказы означают, что модель
    # не успевает за вопросами, и сотрудник вместо ответа получает «попробуйте
    # позже». Пользователям это заметно сразу, администратору — не видно
    # никогда, пока он не посмотрит специально.
    try:
        import llm_queue
        qs = llm_queue.stats(24)
        refused = qs.get("refused", 0) + qs.get("timeout", 0)
        total = qs.get("total", 0)
        if refused and refused > max(2, total * 0.05):
            add("llm_queue_refused", "warning", "Модель не успевает за вопросами",
                f"за сутки отказов по очереди: {refused} при {total} успешных "
                f"запросах, ожидание в очереди в среднем "
                f"{qs.get('wait_avg_ms', 0)} мс",
                "Раздел «Модели», панель очереди: поднять «Одновременных "
                "запросов к модели» на единицу при запасе видеопамяти либо "
                "перенести фоновую обработку базы на ночь.")
        now = llm_queue.status()
        if config.LLM_QUEUE_MAX and \
                now.get("waiting", 0) >= max(5, config.LLM_QUEUE_MAX * 0.8):
            add("llm_queue_full", "warning", "Очередь к модели почти заполнена",
                f"ждут {now['waiting']} при пределе {config.LLM_QUEUE_MAX}",
                "Следующие вопросы получат отказ сразу, не дожидаясь очереди.")
    except Exception:  # noqa: BLE001
        pass

    # Очередь распознавания.
    try:
        pending = db.q1("SELECT COUNT(*) n FROM documents "
                        "WHERE needs_ocr=1 AND status='ok'")["n"]
        if pending > config.ALERT_OCR_QUEUE and config.OCR_PROVIDER == "none":
            add("ocr", "warning", "Сканы не распознаются",
                f"Ждут распознавания {pending} документов, распознаватель не выбран. "
                f"Это в основном сертификаты и декларации.",
                "Раздел «Сканы»: выбрать провайдера и запустить очередь.")
    except Exception:  # noqa: BLE001
        pass

    # Доля отказов за последние сутки.
    try:
        since = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(timespec="seconds")
        row = db.q1("""SELECT COUNT(*) total,
                       SUM(CASE WHEN answered=0 THEN 1 ELSE 0 END) refused
                       FROM queries WHERE created_at > ?""", (since,))
        total = row["total"] or 0
        refused = row["refused"] or 0
        if total >= 20 and refused / total > config.ALERT_REFUSAL_RATE:
            add("refusals", "warning", "Много отказов",
                f"За сутки {refused} из {total} вопросов остались без ответа "
                f"({refused / total:.0%}).",
                "Раздел «Аналитика»: воронка ответа покажет, где теряются "
                "ответы, а группы вопросов — чего не хватает в базе.")
    except Exception:  # noqa: BLE001
        pass

    # Оборванные и упавшие задачи.
    try:
        import jobs
        jobs.ensure_tables()
        broken = db.q("""SELECT id, title, status, error FROM jobs
                         WHERE status IN ('error','stale')
                           AND finished_at > datetime('now','-1 day')""")
        if broken:
            add("jobs", "warning", "Задачи завершились с ошибкой",
                "; ".join(f"№{r['id']} {r['title']}" for r in broken[:5]),
                "Раздел «Конвейер»: посмотреть причину и повторить.")
    except Exception:  # noqa: BLE001
        pass

    # Заявки на доступ.
    try:
        import access
        waiting = access.summary().get("pending", 0)
        if waiting:
            add("access", "warning", "Заявки на доступ ждут решения",
                f"Сотрудников в ожидании: {waiting}.",
                "Раздел «Сотрудники»: подтвердить или отклонить.")
    except Exception:  # noqa: BLE001
        pass

    # Устаревшая смысловая модель.
    try:
        import embeddings
        if config.EMBEDDINGS_PROVIDER == "lsa":
            emb = embeddings.get_embedder()
            trained = int(emb.model.meta.get("documents", 0))
            now = db.q1("SELECT COUNT(*) n FROM chunks")["n"]
            if trained and now > trained * (1 + config.LSA_STALE_RATIO):
                add("lsa_stale", "warning", "Смысловая модель устарела",
                    f"Обучена на {trained} фрагментах, сейчас в базе {now}.",
                    "Раздел «Качество поиска»: обучить заново и пересчитать векторы.")
    except Exception:  # noqa: BLE001
        pass

    # Ключи в открытом виде.
    try:
        import security
        health = security.secrets_health()
        if not health["ok"]:
            add("secrets", "warning", "Ключи лежат небезопасно",
                "; ".join(health["problems"]),
                "Раздел «Настройки» → «Вынести ключи в защищённый файл».")
    except Exception:  # noqa: BLE001
        pass

    return found


# ═══════════════════════════════════════════════════════════ отправка ══════
def _send_telegram(text: str) -> bool:
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_ADMIN_IDS:
        return False
    try:
        import httpx
        client = httpx.Client(timeout=20, proxy=config.TELEGRAM_PROXY or None)
        for admin in config.TELEGRAM_ADMIN_IDS:
            client.post(
                f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage",
                json={"chat_id": admin, "text": text, "parse_mode": "HTML",
                      "disable_web_page_preview": True})
        return True
    except Exception as exc:  # noqa: BLE001
        log.error("не удалось отправить оповещение в Telegram: %s", exc)
        return False


def _send_webhook(payload: dict) -> bool:
    if not config.ALERT_WEBHOOK_URL:
        return False
    try:
        import httpx
        httpx.post(config.ALERT_WEBHOOK_URL, json=payload, timeout=20)
        return True
    except Exception as exc:  # noqa: BLE001
        log.error("не удалось отправить оповещение на webhook: %s", exc)
        return False


def notify(items: list[dict]) -> dict:
    """Отправляет то, о чём ещё не сообщали или сообщали давно."""
    ensure_tables()
    channels = {c.strip() for c in config.ALERT_CHANNELS.split(",") if c.strip()}
    cutoff = (datetime.now(timezone.utc)
              - timedelta(hours=config.ALERT_REPEAT_HOURS)).isoformat(timespec="seconds")
    to_send = []
    for item in items:
        row = db.tq1("SELECT * FROM alerts WHERE key=? AND resolved_at IS NULL",
                     (item["key"],))
        if row is None:
            db.trun("""INSERT INTO alerts(key, level, title, detail, action,
                       first_seen, last_seen) VALUES (?,?,?,?,?,?,?)""",
                    (item["key"], item["level"], item["title"], item["detail"],
                     item["action"], _now(), _now()))
            to_send.append(item)
        else:
            db.trun("UPDATE alerts SET last_seen=?, detail=?, level=? WHERE id=?",
                    (_now(), item["detail"], item["level"], row["id"]))
            if not row["notified_at"] or row["notified_at"] < cutoff:
                to_send.append(item)

    # Закрываем то, что починилось.
    active = {i["key"] for i in items}
    for row in db.tq("SELECT id, key, title FROM alerts WHERE resolved_at IS NULL"):
        if row["key"] not in active:
            db.trun("UPDATE alerts SET resolved_at=? WHERE id=?", (_now(), row["id"]))
            log.info("проблема закрылась: %s", row["title"])

    if not to_send:
        return {"sent": 0, "active": len(items)}

    errors = [i for i in to_send if i["level"] == "error"]
    lines = ["<b>Ассистент базы знаний: требуется внимание</b>", ""]
    for item in to_send:
        mark = "🔴" if item["level"] == "error" else "🟡"
        lines.append(f"{mark} <b>{item['title']}</b>")
        lines.append(item["detail"])
        if item["action"]:
            lines.append(f"<i>{item['action']}</i>")
        lines.append("")
    text = "\n".join(lines)[:4000]

    delivered = False
    if "telegram" in channels:
        delivered = _send_telegram(text) or delivered
    if "webhook" in channels:
        delivered = _send_webhook({"alerts": to_send}) or delivered
    if "log" in channels or not delivered:
        for item in to_send:
            (log.error if item["level"] == "error" else log.warning)(
                "%s — %s", item["title"], item["detail"])
        # Запись в журнал — тоже доставка. Иначе, когда других каналов нет,
        # одно и то же сообщение писалось бы каждый час без конца.
        delivered = True

    if delivered:
        for item in to_send:
            db.trun("UPDATE alerts SET notified_at=? WHERE key=? AND resolved_at IS NULL",
                    (_now(), item["key"]))
    return {"sent": len(to_send), "errors": len(errors), "delivered": delivered,
            "active": len(items)}


def check() -> dict:
    if not config.ALERTS_ENABLED:
        return {"disabled": True}
    db.init()
    items = collect()
    result = notify(items)
    result["items"] = items
    # Отметка о том, что детектор отказов сам жив: если её возраст
    # больше пары часов — не выполняются сами проверки, и тишина в
    # Telegram ничего не значит.
    try:
        import metrics
        metrics.beat("оповещения")
    except Exception:  # noqa: BLE001
        pass
    return result


def active() -> list[dict]:
    """Незакрытые проблемы — для показа в админке."""
    ensure_tables()
    return [dict(r) for r in db.tq(
        "SELECT * FROM alerts WHERE resolved_at IS NULL ORDER BY "
        "CASE level WHEN 'error' THEN 0 ELSE 1 END, id DESC")]


def history(limit: int = 50) -> list[dict]:
    ensure_tables()
    return [dict(r) for r in db.tq(
        "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,))]


def main() -> int:
    p = argparse.ArgumentParser(description="Оповещения о проблемах")
    p.add_argument("command", choices=["check", "list", "test", "history"])
    args = p.parse_args()
    db.init()
    if args.command == "check":
        result = check()
        print(json.dumps({k: v for k, v in result.items() if k != "items"},
                         ensure_ascii=False))
        for item in result.get("items", []):
            print(f"  [{item['level']}] {item['title']}: {item['detail']}")
    elif args.command == "list":
        items = collect()
        if not items:
            print("Всё в порядке.")
            return 0
        for item in items:
            mark = "ОШИБКА " if item["level"] == "error" else "внимание"
            print(f"{mark} {item['title']}")
            print(f"         {item['detail']}")
            if item["action"]:
                print(f"         что делать: {item['action']}")
    elif args.command == "test":
        ok = _send_telegram("Проверка оповещений: если вы это видите, "
                            "сообщения о проблемах будут доходить.")
        print("отправлено" if ok else
              "не отправлено: проверьте TELEGRAM_BOT_TOKEN и TELEGRAM_ADMIN_IDS")
    elif args.command == "history":
        for row in history():
            state = "закрыта" if row["resolved_at"] else "открыта"
            print(f"{row['first_seen'][:16]}  {state:8} {row['title']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
