"""
Резервное копирование индекса.

Что здесь ценно и почему это не то же самое, что копия базы знаний.
Сами файлы в папке BD никуда не денутся — их можно разобрать заново.
Невосстановимо другое: выверенные ответы, накопленные вопросы
сотрудников, обучающие пары и оценки. Пересборка индекса вернёт тексты,
но не вернёт ни одного выверенного ответа. Плюс сама пересборка — это
часы машинного времени, в течение которых ассистент не работает.

  python backup.py create              — сделать снимок
  python backup.py list                — что уже есть
  python backup.py verify [ФАЙЛ]       — проверить, что снимок разворачивается
  python backup.py restore ФАЙЛ        — восстановить (с копией текущего состояния)
  python backup.py prune               — удалить старые по правилу хранения
  python backup.py schedule            — поставить регулярный запуск
  python backup.py status              — когда была последняя копия

Главное отличие от «просто копии папки»: снимок делается средствами
SQLite (backup API), поэтому копия консистентна даже если в этот момент
идёт индексация. Обычное копирование файла базы под нагрузкой даёт
битый файл, который выглядит целым — и это выясняется в тот день,
когда он понадобится.

И каждый снимок сразу же проверяется: разворачивается во временную
папку, база открывается, считаются документы, фрагменты и выверенные
ответы. Резервная копия, которую ни разу не пробовали восстановить, —
это не резервная копия.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import config
import logging_setup

log = logging_setup.get("backup")

MANIFEST = "manifest.json"
# Что попадает в снимок. Пути считаются от DATA_DIR.
PAYLOAD = ("kb.sqlite3", "vectors.npy", "vector_ids.json", "lsa_model.npz")


class BackupError(RuntimeError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _stamp() -> str:
    return _now().strftime("%Y%m%d-%H%M%S")


def workdir(prefix: str) -> Path:
    """
    Временная папка для распаковки — рядом с данными, а не в /tmp.

    Две причины. Первая: убитый посреди работы процесс не успевает
    выполнить очистку, и в /tmp остаётся полная распакованная копия
    индекса — несколько таких, и на диске кончается место, причём
    незаметно. Держа их в своей папке, мы можем прибрать за собой при
    следующем запуске. Вторая: /tmp часто отдельный и маленький раздел,
    а копия индекса — гигабайты.
    """
    base = Path(config.DATA_DIR) / "work"
    base.mkdir(parents=True, exist_ok=True)
    cleanup_workdirs(base)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=base))


def cleanup_workdirs(base: Path | None = None, older_than_hours: float = 6.0) -> int:
    """Прибирает то, что осталось от прерванных операций."""
    base = base or (Path(config.DATA_DIR) / "work")
    if not base.exists():
        return 0
    removed = 0
    cutoff = time.time() - older_than_hours * 3600
    for item in base.iterdir():
        try:
            if item.stat().st_mtime < cutoff:
                shutil.rmtree(item, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    if removed:
        log.warning("убрано временных папок от прерванных операций: %d", removed)
    return removed


def _human(size: float) -> str:
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if size < 1024 or unit == "ГБ":
            return f"{size:.0f} {unit}" if unit == "Б" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} ГБ"


# ------------------------------------------------------ снимок базы SQLite --
def snapshot_sqlite(source: Path, dest: Path) -> dict:
    """
    Консистентная копия базы через backup API SQLite.

    Работает на живой базе: SQLite сам следит за тем, чтобы копия
    соответствовала одному состоянию, даже если в неё сейчас пишут.
    """
    if not source.exists():
        raise BackupError(f"база индекса не найдена: {source}")
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True, timeout=60)
    dst = sqlite3.connect(dest)
    try:
        src.backup(dst)
        dst.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        counts = {}
        for table in ("documents", "chunks", "products", "golden_qa",
                      "training_pairs", "queries", "feedback"):
            try:
                counts[table] = dst.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except sqlite3.Error:
                counts[table] = None
        integrity = dst.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        src.close()
        dst.close()
    if integrity != "ok":
        raise BackupError(f"копия базы не прошла проверку целостности: {integrity}")
    return counts


# --------------------------------------------------------------- создание ---
def create(verify: bool | None = None, note: str = "",
           quiet: bool = False, progress=None) -> Path:
    """Делает снимок индекса и сразу проверяет, что он разворачивается."""
    verify = config.BACKUP_VERIFY if verify is None else verify
    if progress is not None:
        say = lambda *a: progress(" ".join(str(x) for x in a))   # noqa: E731
    elif quiet:
        say = lambda *_a: None                                   # noqa: E731
    else:
        say = lambda *a: print(*a, flush=True)                    # noqa: E731
    config.BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    stamp = _stamp()
    work = workdir("kb_backup_")
    started = time.time()
    try:
        say(f"Снимаю копию индекса ({config.DATA_DIR})…")
        counts = snapshot_sqlite(config.DB_PATH, work / "kb.sqlite3")
        say(f"  база: документов {counts.get('documents')}, "
            f"фрагментов {counts.get('chunks')}, позиций прайсов "
            f"{counts.get('products')}, выверенных ответов {counts.get('golden_qa')}")

        included = ["kb.sqlite3"]
        for name in PAYLOAD[1:]:
            src = Path(config.DATA_DIR) / name
            if name == "lsa_model.npz":
                src = Path(config.LSA_MODEL_PATH)
            if src.exists():
                shutil.copy2(src, work / name)
                included.append(name)
                say(f"  {name}: {_human(src.stat().st_size)}")

        manifest = {
            "created": _now().isoformat(timespec="seconds"),
            "host": platform.node(),
            "kb_root": str(config.KB_ROOT),
            "data_dir": str(config.DATA_DIR),
            "embeddings_provider": config.EMBEDDINGS_PROVIDER,
            "files": included,
            "counts": counts,
            "note": note,
            "format": 1,
        }
        (work / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                     encoding="utf-8")

        suffix = ".tar.gz" if config.BACKUP_COMPRESS else ".tar"
        archive = config.BACKUP_DIR / f"index-{stamp}{suffix}"
        say(f"Упаковываю в {archive.name}…")
        mode = "w:gz" if config.BACKUP_COMPRESS else "w"
        # Пишем во временное имя и переименовываем в конце. Иначе
        # остановка посреди упаковки многогигабайтного архива оставляет
        # обрезанный файл с правильным именем и свежей датой: список
        # копий показывает его как «последняя копия, возраст два часа»,
        # мониторинг зелёный, правило хранения удаляет ради него
        # предыдущую нормальную. Обнаруживается в день, когда копия
        # понадобится.
        partial = archive.with_suffix(archive.suffix + ".partial")
        try:
            with tarfile.open(partial, mode) as tar:
                for item in sorted(work.iterdir()):
                    tar.add(item, arcname=item.name)
            partial.replace(archive)
        finally:
            partial.unlink(missing_ok=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    size = archive.stat().st_size
    say(f"Готово: {_human(size)} за {time.time() - started:.0f} с")

    if verify:
        say("Проверяю, что копия разворачивается…")
        report = verify_archive(archive)
        if not report["ok"]:
            log.error("резервная копия не прошла проверку: %s", report["error"])
            raise BackupError(f"копия создана, но не проходит проверку: {report['error']}")
        say(f"  проверка пройдена: документов {report['counts'].get('documents')}, "
            f"фрагментов {report['counts'].get('chunks')}, "
            f"векторов {report.get('vectors', '—')}")

    if config.BACKUP_MIRROR_DIR:
        try:
            mirror = Path(config.BACKUP_MIRROR_DIR).expanduser()
            mirror.mkdir(parents=True, exist_ok=True)
            shutil.copy2(archive, mirror / archive.name)
            say(f"Вторая копия: {mirror / archive.name}")
        except Exception as exc:  # noqa: BLE001 — основная копия уже сделана
            log.warning("не удалось положить копию в %s: %s",
                        config.BACKUP_MIRROR_DIR, exc)
            say(f"  вторую копию сделать не удалось: {exc}")

    log.info("резервная копия создана: %s (%s)", archive.name, _human(size))
    removed = prune(quiet=True)
    if removed and not quiet:
        say(f"По правилу хранения удалено старых копий: {len(removed)}")
    return archive


# --------------------------------------------------------------- проверка ---
def verify_archive(archive: Path) -> dict:
    """
    Разворачивает копию во временную папку и убеждается, что она живая.

    Проверяется именно то, что понадобится в день восстановления: база
    открывается, целостность в порядке, документы и фрагменты на месте,
    число векторов совпадает с числом записей в списке идентификаторов.
    """
    report = {"archive": str(archive), "ok": False, "error": None, "counts": {}}
    if not archive.exists():
        report["error"] = "файл копии не найден"
        return report
    work = workdir("kb_verify_")
    try:
        with tarfile.open(archive) as tar:
            _safe_extract(tar, work)
        manifest_path = work / MANIFEST
        if manifest_path.exists():
            report["manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
        dbf = work / "kb.sqlite3"
        if not dbf.exists():
            report["error"] = "в копии нет базы индекса"
            return report
        conn = sqlite3.connect(f"file:{dbf}?mode=ro", uri=True)
        try:
            integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                report["error"] = f"проверка целостности: {integrity}"
                return report
            for table in ("documents", "chunks", "products", "golden_qa",
                          "training_pairs"):
                try:
                    report["counts"][table] = conn.execute(
                        f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                except sqlite3.Error:
                    report["counts"][table] = None
            # Не просто «таблица есть», а «данные читаются».
            conn.execute("SELECT rel_path, text_chars FROM documents LIMIT 5").fetchall()
            conn.execute("SELECT text FROM chunks LIMIT 5").fetchall()
        finally:
            conn.close()

        if report["counts"].get("documents") == 0:
            report["error"] = "в копии нет ни одного документа"
            return report

        ids_file, vec_file = work / "vector_ids.json", work / "vectors.npy"
        if ids_file.exists() and vec_file.exists():
            import numpy as np
            ids = json.loads(ids_file.read_text())
            matrix = np.load(vec_file, mmap_mode="r")
            report["vectors"] = int(matrix.shape[0])
            if len(ids) != matrix.shape[0]:
                report["error"] = (f"векторов {matrix.shape[0]}, "
                                   f"а идентификаторов {len(ids)} — копия рассогласована")
                return report
        report["ok"] = True
    except Exception as exc:  # noqa: BLE001
        report["error"] = str(exc)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return report


def _safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    """Распаковка без выхода за пределы папки (защита от путей вида ../)."""
    dest = dest.resolve()
    for member in tar.getmembers():
        target = (dest / member.name).resolve()
        if not str(target).startswith(str(dest)):
            raise BackupError(f"подозрительный путь в архиве: {member.name}")
        if member.issym() or member.islnk():
            raise BackupError(f"ссылки в архиве не допускаются: {member.name}")
    tar.extractall(dest)


# ---------------------------------------------------------- восстановление --
def restore(archive: Path, force: bool = False) -> dict:
    """
    Восстанавливает индекс из копии.

    Текущее состояние не удаляется, а откладывается рядом с пометкой
    «before-restore»: если копия окажется не той, откат займёт минуту.
    """
    report = verify_archive(archive)
    if not report["ok"] and not force:
        raise BackupError(f"копия не прошла проверку ({report['error']}). "
                          f"Восстановление отменено. Если всё же нужно — "
                          f"добавьте --force")

    data = Path(config.DATA_DIR)
    data.mkdir(parents=True, exist_ok=True)
    aside = data / f"before-restore-{_stamp()}"
    aside.mkdir()
    moved = []
    for name in PAYLOAD:
        src = data / name
        if name == "lsa_model.npz":
            src = Path(config.LSA_MODEL_PATH)
        if src.exists():
            shutil.move(str(src), str(aside / name))
            moved.append(name)

    work = workdir("kb_restore_")
    try:
        with tarfile.open(archive) as tar:
            _safe_extract(tar, work)
        restored = []
        for name in PAYLOAD:
            src = work / name
            if not src.exists():
                continue
            dest = Path(config.LSA_MODEL_PATH) if name == "lsa_model.npz" else data / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            restored.append(name)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    log.warning("индекс восстановлен из %s; прежнее состояние в %s",
                archive.name, aside)
    return {"restored": restored, "previous": str(aside), "verify": report,
            "moved": moved}


# ------------------------------------------------------------- хранение ----
def archives() -> list[Path]:
    if not config.BACKUP_DIR.exists():
        return []
    files = [p for p in config.BACKUP_DIR.iterdir()
             if p.name.startswith("index-") and p.suffix in (".gz", ".tar")]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def _parse_stamp(path: Path) -> datetime:
    try:
        raw = path.name.split("index-")[1].split(".")[0]
        return datetime.strptime(raw, "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)
    except Exception:  # noqa: BLE001
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def prune(quiet: bool = False) -> list[str]:
    """
    Правило хранения «дед — отец — сын»: свежие копии за каждый день,
    затем по одной за неделю, затем по одной за месяц. Так и глубина
    истории есть, и диск не заполняется.
    """
    files = archives()
    if not files:
        return []
    keep: set[Path] = set()
    seen_days: set[str] = set()
    seen_weeks: set[str] = set()
    seen_months: set[str] = set()
    now = _now()
    for path in files:                      # уже отсортированы от новых к старым
        when = _parse_stamp(path)
        age = (now - when).days
        day = when.strftime("%Y-%m-%d")
        week = when.strftime("%G-W%V")
        month = when.strftime("%Y-%m")
        if age <= config.BACKUP_KEEP_DAILY and day not in seen_days:
            keep.add(path); seen_days.add(day); continue
        if age <= config.BACKUP_KEEP_WEEKLY * 7 and week not in seen_weeks:
            keep.add(path); seen_weeks.add(week); continue
        if age <= config.BACKUP_KEEP_MONTHLY * 31 and month not in seen_months:
            keep.add(path); seen_months.add(month); continue
    keep.add(files[0])                      # самую свежую не трогаем никогда
    removed = []
    for path in files:
        if path not in keep:
            try:
                path.unlink()
                removed.append(path.name)
            except OSError as exc:
                log.warning("не удалось удалить %s: %s", path, exc)
    if removed and not quiet:
        print("Удалены по правилу хранения:", ", ".join(removed))
    return removed


def status() -> dict:
    """Состояние резервного копирования — для веб-интерфейса и диагностики."""
    files = archives()
    info = {"count": len(files), "dir": str(config.BACKUP_DIR),
            "schedule": config.BACKUP_SCHEDULE, "installed": schedule_installed(),
            "total_bytes": sum(p.stat().st_size for p in files)}
    if files:
        latest = files[0]
        when = _parse_stamp(latest)
        hours = (_now() - when).total_seconds() / 3600
        info.update(latest=latest.name, latest_at=when.isoformat(timespec="seconds"),
                    latest_bytes=latest.stat().st_size,
                    age_hours=round(hours, 1),
                    stale=hours > config.BACKUP_ALERT_HOURS)
    else:
        info.update(latest=None, stale=True, age_hours=None)
    return info


# -------------------------------------------------------------- расписание --
def _cron_line() -> str:
    python = sys.executable or "python3"
    script = Path(__file__).resolve()
    return (f"{config.BACKUP_SCHEDULE} cd {script.parent} && "
            f"{python} {script} create --quiet "
            f">> {config.BACKUP_DIR / 'backup.log'} 2>&1")


MARKER = "# резервная копия индекса базы знаний"


def schedule_installed() -> bool:
    system = platform.system()
    try:
        if system == "Windows":
            r = subprocess.run(["schtasks", "/query", "/tn", "KBIndexBackup"],
                               capture_output=True, text=True)
            return r.returncode == 0
        r = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
        return MARKER in r.stdout
    except Exception:  # noqa: BLE001
        return False


def schedule_install(remove: bool = False) -> str:
    """Ставит (или снимает) регулярный запуск средствами самой системы."""
    system = platform.system()
    config.BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    if system == "Windows":
        if remove:
            subprocess.run(["schtasks", "/delete", "/tn", "KBIndexBackup", "/f"],
                           check=False)
            return "Задача KBIndexBackup удалена"
        parts = config.BACKUP_SCHEDULE.split()
        minute, hour = (parts + ["0", "3"])[:2]
        time_of_day = f"{int(hour):02d}:{int(minute):02d}"
        cmd = f'"{sys.executable}" "{Path(__file__).resolve()}" create --quiet'
        r = subprocess.run(["schtasks", "/create", "/tn", "KBIndexBackup",
                            "/tr", cmd, "/sc", "daily", "/st", time_of_day, "/f"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise BackupError(r.stderr.strip() or r.stdout.strip())
        return f"Задача KBIndexBackup создана, запуск ежедневно в {time_of_day}"

    if not shutil.which("crontab"):
        raise BackupError(
            "в системе нет crontab. Поставьте запуск вручную: "
            f"{_cron_line()}")
    current = subprocess.run(["crontab", "-l"], capture_output=True, text=True).stdout
    lines = [l for l in current.splitlines()
             if MARKER not in l and "backup.py create" not in l]
    if not remove:
        lines += [MARKER, _cron_line()]
    payload = "\n".join(lines).strip() + "\n"
    proc = subprocess.run(["crontab", "-"], input=payload, text=True,
                          capture_output=True)
    if proc.returncode != 0:
        raise BackupError(proc.stderr.strip())
    return ("Регулярный запуск снят" if remove else
            f"Регулярный запуск поставлен: {config.BACKUP_SCHEDULE} "
            f"(проверить: crontab -l)")


# -------------------------------------------------------------------- CLI ---
def show_list() -> None:
    files = archives()
    if not files:
        print(f"Копий пока нет. Папка: {config.BACKUP_DIR}")
        print("Сделать первую: python backup.py create")
        return
    print(f"Копии в {config.BACKUP_DIR}:")
    for path in files:
        when = _parse_stamp(path).astimezone()
        print(f"  {when:%d.%m.%Y %H:%M}  {_human(path.stat().st_size):>9}  {path.name}")
    info = status()
    print(f"\nВсего: {info['count']}, занято {_human(info['total_bytes'])}")
    if info.get("stale"):
        print(f"ВНИМАНИЕ: последней копии больше {config.BACKUP_ALERT_HOURS} ч.")
    print("Регулярный запуск: " + ("настроен" if info["installed"] else
                                   "не настроен (python backup.py schedule)"))


def main() -> int:
    p = argparse.ArgumentParser(description="Резервное копирование индекса")
    p.add_argument("command", choices=["create", "list", "verify", "restore",
                                       "prune", "schedule", "unschedule", "status"])
    p.add_argument("path", nargs="?", help="файл копии для verify/restore")
    p.add_argument("--note", default="", help="пометка к копии")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no-verify", action="store_true",
                   help="не проверять копию сразу после создания")
    p.add_argument("--force", action="store_true",
                   help="восстановить даже если копия не прошла проверку")
    args = p.parse_args()

    try:
        if args.command == "create":
            create(verify=not args.no_verify, note=args.note, quiet=args.quiet)
        elif args.command == "list":
            show_list()
        elif args.command == "status":
            print(json.dumps(status(), ensure_ascii=False, indent=2))
        elif args.command == "verify":
            target = Path(args.path) if args.path else (archives() or [None])[0]
            if target is None:
                print("Нечего проверять — копий нет.")
                return 1
            report = verify_archive(Path(target))
            print(f"{Path(target).name}: "
                  f"{'в порядке' if report['ok'] else 'НЕ В ПОРЯДКЕ — ' + str(report['error'])}")
            for table, n in report["counts"].items():
                print(f"  {table:16} {n}")
            if "vectors" in report:
                print(f"  {'векторов':16} {report['vectors']}")
            return 0 if report["ok"] else 1
        elif args.command == "restore":
            if not args.path:
                print("Укажите файл копии: python backup.py restore "
                      "data/backups/index-….tar.gz")
                return 2
            result = restore(Path(args.path), force=args.force)
            print(f"Восстановлено: {', '.join(result['restored'])}")
            print(f"Прежнее состояние сохранено в {result['previous']}")
            print("Проверьте поиск, и если всё в порядке — эту папку можно удалить.")
        elif args.command == "prune":
            removed = prune()
            print(f"Удалено копий: {len(removed)}" if removed else "Удалять нечего.")
        elif args.command == "schedule":
            print(schedule_install())
        elif args.command == "unschedule":
            print(schedule_install(remove=True))
    except BackupError as exc:
        print(f"Ошибка: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
