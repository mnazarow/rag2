"""
Слежение за папкой базы знаний.

  python watcher.py run              — постоянное слежение
  python watcher.py once             — один проход
  python watcher.py structure        — что изменилось в структуре каталогов
  python watcher.py snapshot         — зафиксировать текущую структуру

Отличие от простой переиндексации: помимо содержимого файлов
отслеживается сама структура каталогов. Когда сотрудник заводит новую
папку бренда или переименовывает категорию, это событие видно отдельно —
и это важно, потому что структура папок у вас несёт смысл: первый
уровень определяет раздел и права доступа, третий — тип документа.
Появление папки с неожиданным именем означает, что документы в ней
получат неправильные метаданные, и лучше узнать об этом сразу.

Если установлен watchdog, изменения ловятся событиями файловой системы
и реакция мгновенная. Без него работает опрос — надёжнее на сетевых
дисках и в облачных папках, где события теряются.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import config
import db
import extract
import logging_setup

log = logging_setup.get("watch")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_tables() -> None:
    db.connect().executescript("""
    CREATE TABLE IF NOT EXISTS folder_snapshot (
        rel_path   TEXT PRIMARY KEY,
        depth      INTEGER,
        files      INTEGER,
        subfolders INTEGER,
        seen_at    TEXT
    );
    CREATE TABLE IF NOT EXISTS structure_events (
        id       INTEGER PRIMARY KEY,
        ts       TEXT,
        kind     TEXT,      -- added | removed | renamed | grew | shrank | unknown_category
        path     TEXT,
        detail   TEXT,
        seen     INTEGER DEFAULT 0
    );
    CREATE INDEX IF NOT EXISTS idx_structure_ts ON structure_events(ts);
    """)


def scan_structure() -> dict[str, dict]:
    """Текущая структура: папка → сколько файлов и подпапок."""
    out: dict[str, dict] = {}
    root = config.KB_ROOT
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        try:
            rel = str(path.relative_to(root))
        except ValueError:
            continue
        if any(part.endswith(tuple(config.IGNORE_DIR_PARTS)) for part in path.parts):
            continue
        try:
            entries = list(path.iterdir())
        except OSError:
            continue
        out[extract.nfc(rel)] = {
            "depth": len(path.relative_to(root).parts),
            "files": sum(1 for e in entries if e.is_file()),
            "subfolders": sum(1 for e in entries if e.is_dir()),
        }
    return out


def known_categories() -> set[str]:
    return set(extract.DOC_TYPE_MAP.keys())


def diff_structure(save: bool = True) -> list[dict]:
    """Сравнивает текущую структуру с сохранённой и записывает события."""
    ensure_tables()
    current = scan_structure()
    previous = {r["rel_path"]: dict(r) for r in db.q("SELECT * FROM folder_snapshot")}
    events: list[dict] = []

    added = set(current) - set(previous)
    removed = set(previous) - set(current)

    # Переименование: папка исчезла и появилась другая с тем же составом.
    renamed: list[tuple[str, str]] = []
    for gone in list(removed):
        old = previous[gone]
        for fresh in list(added):
            new = current[fresh]
            if old["files"] == new["files"] and old["subfolders"] == new["subfolders"] \
                    and old["depth"] == new["depth"] and Path(gone).parent == Path(fresh).parent:
                renamed.append((gone, fresh))
                removed.discard(gone)
                added.discard(fresh)
                break

    for old, new in renamed:
        events.append({"kind": "renamed", "path": new,
                       "detail": f"переименована из «{Path(old).name}»"})
    for path in sorted(added):
        info = current[path]
        events.append({"kind": "added", "path": path,
                       "detail": f"новая папка, файлов внутри: {info['files']}"})
        # Новая папка на уровне категорий с незнакомым именем — повод сказать.
        name = Path(path).name
        if info["depth"] == 3 and name not in known_categories():
            events.append({"kind": "unknown_category", "path": path,
                           "detail": f"имя «{name}» не совпадает ни с одной известной "
                                     "категорией — документы получат пустой тип"})
    for path in sorted(removed):
        events.append({"kind": "removed", "path": path,
                       "detail": f"папка исчезла, было файлов: {previous[path]['files']}"})
    for path, info in current.items():
        old = previous.get(path)
        if not old:
            continue
        delta = info["files"] - (old["files"] or 0)
        if delta >= 5:
            events.append({"kind": "grew", "path": path,
                           "detail": f"добавилось файлов: {delta}"})
        elif delta <= -5:
            events.append({"kind": "shrank", "path": path,
                           "detail": f"убыло файлов: {-delta}"})

    if save:
        conn = db.connect()
        conn.execute("DELETE FROM folder_snapshot")
        conn.executemany(
            "INSERT INTO folder_snapshot(rel_path, depth, files, subfolders, seen_at) "
            "VALUES (?,?,?,?,?)",
            [(p, i["depth"], i["files"], i["subfolders"], _now()) for p, i in current.items()])
        conn.executemany(
            "INSERT INTO structure_events(ts, kind, path, detail) VALUES (?,?,?,?)",
            [(_now(), e["kind"], e["path"], e["detail"]) for e in events])
        conn.commit()

    for e in events:
        level = log.warning if e["kind"] in ("removed", "unknown_category") else log.info
        level("структура: %s — %s (%s)", e["kind"], e["path"], e["detail"])
    return events


def pending_structure_events(limit: int = 50) -> list[dict]:
    ensure_tables()
    return [dict(r) for r in db.q(
        "SELECT * FROM structure_events WHERE seen=0 ORDER BY id DESC LIMIT ?", (limit,))]


def mark_seen() -> None:
    db.run("UPDATE structure_events SET seen=1 WHERE seen=0")


def notify_admins(events: list[dict]) -> None:
    """Отправляет администраторам сводку об изменениях структуры."""
    if not events or not config.WATCH_NOTIFY_STRUCTURE:
        return
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_ADMIN_IDS:
        return
    try:
        import httpx
    except ImportError:
        return
    lines = ["<b>Изменения в структуре базы знаний</b>"]
    titles = {"added": "новая папка", "removed": "папка удалена",
              "renamed": "переименование", "grew": "пополнение",
              "shrank": "удаление файлов", "unknown_category": "незнакомая категория"}
    for e in events[:20]:
        lines.append(f"· {titles.get(e['kind'], e['kind'])}: <code>{e['path']}</code>\n"
                     f"  {e['detail']}")
    text = "\n".join(lines)[:3800]
    client_kwargs = {"timeout": 30}
    if config.TELEGRAM_PROXY:
        client_kwargs["proxy"] = config.TELEGRAM_PROXY
    with httpx.Client(**client_kwargs) as client:
        for admin in config.TELEGRAM_ADMIN_IDS:
            try:
                client.post(
                    f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage",
                    json={"chat_id": admin, "text": text, "parse_mode": "HTML"})
            except Exception as exc:  # noqa: BLE001
                log.warning("не удалось уведомить администратора %s: %s", admin, exc)


def run_once(reindex: bool = True) -> dict:
    """Один проход: структура, затем содержимое."""
    logging_setup.new_request(channel="watcher")
    ensure_tables()
    with log.timed("проверка структуры каталогов"):
        events = diff_structure()
    if events:
        notify_admins(events)

    counts = {}
    if reindex:
        import index as index_mod
        with log.timed("инкрементальная переиндексация"):
            counts = index_mod.build(force=False, verbose=False)
        if config.WATCH_REBUILD_GRAPH and counts.get("indexed"):
            try:
                import graph
                g = graph.build_graph()
                graph.render_html(g, config.DATA_DIR / "graph.html")
                log.info("граф пересобран: узлов %d", g["stats"]["documents"])
            except Exception as exc:  # noqa: BLE001
                log.warning("граф не пересобрался: %s", exc)
    return {"structure_events": len(events), "index": counts}


def run_forever(interval: int | None = None) -> None:
    interval = interval or config.WATCH_INTERVAL_SECONDS
    log.info("слежение запущено: %s, интервал %d с", config.KB_ROOT, interval)
    import shutdown
    shutdown.install("слежение")
    observer = _try_watchdog()
    try:
        while not shutdown.stopping():
            if config.reload_if_changed():
                log.warning("настройки изменились — перечитал")
                interval = config.WATCH_INTERVAL_SECONDS
            try:
                result = run_once()
                log.info("проход завершён: %s", result)
            except Exception as exc:  # noqa: BLE001
                log.exception("сбой прохода: %s", exc)
            # Пауза, прерываемая сигналом: с обычным sleep остановка ждала
            # бы целый интервал, и docker успевал перейти к принудительному
            # завершению.
            shutdown.wait(interval)
    except KeyboardInterrupt:
        pass
    finally:
        if observer:
            observer.stop()
    log.info("слежение остановлено")


_pending: set[str] = set()


def _try_watchdog():
    """Если библиотека установлена — реагируем на события сразу."""
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError:
        log.info("watchdog не установлен — работаем опросом папки")
        return None

    class Handler(FileSystemEventHandler):
        def on_any_event(self, event) -> None:
            if event.is_directory:
                log.debug("событие каталога: %s %s", event.event_type, event.src_path)
            _pending.add(event.src_path)

    observer = Observer()
    observer.schedule(Handler(), str(config.KB_ROOT), recursive=True)
    observer.daemon = True
    observer.start()
    log.info("watchdog активен: реакция на изменения мгновенная")
    return observer


def main() -> int:
    p = argparse.ArgumentParser(description="Слежение за базой знаний")
    p.add_argument("command", choices=["run", "once", "structure", "snapshot"],
                   nargs="?", default="once")
    p.add_argument("--interval", type=int)
    p.add_argument("--no-reindex", action="store_true")
    a = p.parse_args()
    db.init()
    logging_setup.setup()

    if a.command == "run":
        run_forever(a.interval)
    elif a.command == "once":
        print(run_once(reindex=not a.no_reindex))
    elif a.command == "structure":
        events = diff_structure(save=False)
        if not events:
            print("Структура каталогов не менялась.")
        for e in events:
            print(f"  {e['kind']:<18} {e['path']}\n      {e['detail']}")
    elif a.command == "snapshot":
        structure = scan_structure()
        diff_structure(save=True)
        print(f"Зафиксировано папок: {len(structure)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
