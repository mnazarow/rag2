"""
Регулярные задания: копии, оповещения, очистка, учебное восстановление.

Раньше расписание было только у резервного копирования. Но регулярности
требуют ещё три вещи, и каждая по своей причине.

Оповещения нужно проверять часто: смысл в том, чтобы узнать о проблеме
раньше, чем о ней сообщит сотрудник. Очистку по сроку хранения — редко,
но обязательно, иначе персональные данные копятся бессрочно просто
потому, что никто не вспомнил. Учебное восстановление — раз в месяц:
проверка при создании копии смотрит, что архив цел, а это проверяет,
что развёрнутый из него индекс действительно ищет.

Ставится средствами самой системы: crontab на Linux и macOS, планировщик
задач на Windows. Своего демона тут нет намеренно — он был бы ещё одним
процессом, который может тихо умереть.

  python schedule.py install     — поставить всё
  python schedule.py status      — что настроено
  python schedule.py remove      — снять
"""
from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import config
import logging_setup

log = logging_setup.get("web")

MARKER = "# ассистент базы знаний"

# Что, когда и почему именно так.
TASKS = {
    "backup": {
        "title": "резервная копия индекса",
        "cron": config.BACKUP_SCHEDULE,
        "command": "backup.py create --quiet",
        "why": "ночью, когда индексация уже закончилась",
    },
    "alerts": {
        "title": "проверка проблем",
        "cron": "17 * * * *",
        "command": "alerts.py check",
        "why": "раз в час: смысл в том, чтобы узнать раньше сотрудников",
    },
    "retention": {
        "title": "очистка по сроку хранения",
        "cron": "40 4 * * 0",
        "command": "retention.py clean",
        "why": "раз в неделю ночью: данных немного, спешить некуда",
    },
    "drill": {
        "title": "учебное восстановление из копии",
        "cron": "0 5 1 * *",
        "command": "runjob.py restore_drill",
        "why": "раз в месяц: проверяет не архив, а работоспособность индекса из него",
    },
    "update": {
        "title": "инкрементальная индексация",
        "cron": "0 2 * * *",
        "command": "index.py update",
        "why": "ночью: подхватывает всё, что положили за день",
    },
}


def _python() -> str:
    return sys.executable or "python3"


def _line(name: str, task: dict) -> str:
    script = Path(__file__).resolve().parent
    logs = Path(config.LOG_DIR) / "schedule.log"
    parts = task["command"].split()
    command = f"{_python()} {script / parts[0]} " + " ".join(parts[1:])
    return (f"{task['cron']} cd {script} && {command.strip()} "
            f">> {logs} 2>&1  {MARKER} {name}")


def install(only: list[str] | None = None) -> str:
    system = platform.system()
    chosen = {k: v for k, v in TASKS.items() if not only or k in only}
    Path(config.LOG_DIR).mkdir(parents=True, exist_ok=True)

    if system == "Windows":
        done = []
        for name, task in chosen.items():
            parts = task["cron"].split()
            minute, hour = (parts + ["0", "3"])[:2]
            when = f"{int(hour):02d}:{int(minute):02d}" if minute.isdigit() \
                and hour.isdigit() else "03:30"
            freq = "HOURLY" if name == "alerts" else \
                   "WEEKLY" if name == "retention" else \
                   "MONTHLY" if name == "drill" else "DAILY"
            script = Path(__file__).resolve().parent / task["command"].split()[0]
            cmd = f'"{_python()}" "{script}" ' + " ".join(task["command"].split()[1:])
            r = subprocess.run(["schtasks", "/create", "/tn", f"KB_{name}",
                                "/tr", cmd, "/sc", freq, "/st", when, "/f"],
                               capture_output=True, text=True)
            done.append(f"{name}: " + ("создано" if r.returncode == 0
                                       else r.stderr.strip()[:80]))
        return "\n".join(done)

    if not shutil.which("crontab"):
        return ("В системе нет crontab. Поставьте задания вручную:\n"
                + "\n".join(_line(n, t) for n, t in chosen.items()))
    current = subprocess.run(["crontab", "-l"], capture_output=True, text=True).stdout
    lines = [l for l in current.splitlines() if MARKER not in l]
    lines += [_line(n, t) for n, t in chosen.items()]
    payload = "\n".join(lines).strip() + "\n"
    proc = subprocess.run(["crontab", "-"], input=payload, text=True,
                          capture_output=True)
    if proc.returncode != 0:
        return f"Не удалось поставить расписание: {proc.stderr.strip()}"
    log.warning("расписание установлено: %s", ", ".join(chosen))
    return ("Поставлено заданий: " + str(len(chosen)) + "\n"
            + "\n".join(f"  {t['cron']:14} {t['title']} — {t['why']}"
                        for t in chosen.values())
            + "\nПроверить: crontab -l")


def remove() -> str:
    system = platform.system()
    if system == "Windows":
        for name in TASKS:
            subprocess.run(["schtasks", "/delete", "/tn", f"KB_{name}", "/f"],
                           capture_output=True)
        return "Задания удалены"
    if not shutil.which("crontab"):
        return "В системе нет crontab"
    current = subprocess.run(["crontab", "-l"], capture_output=True, text=True).stdout
    lines = [l for l in current.splitlines() if MARKER not in l]
    subprocess.run(["crontab", "-"], input="\n".join(lines).strip() + "\n",
                   text=True, capture_output=True)
    log.warning("расписание снято")
    return "Расписание снято"


def status() -> dict:
    system = platform.system()
    installed: dict[str, bool] = {}
    if system == "Windows":
        for name in TASKS:
            r = subprocess.run(["schtasks", "/query", "/tn", f"KB_{name}"],
                               capture_output=True)
            installed[name] = r.returncode == 0
    elif shutil.which("crontab"):
        current = subprocess.run(["crontab", "-l"], capture_output=True,
                                 text=True).stdout
        for name in TASKS:
            installed[name] = f"{MARKER} {name}" in current
    else:
        installed = {name: False for name in TASKS}
    return {"system": system,
            "tasks": [{"name": n, **TASKS[n], "installed": installed.get(n, False)}
                      for n in TASKS],
            "installed": sum(1 for v in installed.values() if v),
            "total": len(TASKS)}


def main() -> int:
    p = argparse.ArgumentParser(description="Регулярные задания")
    p.add_argument("command", choices=["install", "remove", "status"])
    p.add_argument("--only", help="через запятую: backup,alerts,retention,drill,update")
    args = p.parse_args()
    if args.command == "install":
        print(install([x.strip() for x in args.only.split(",")] if args.only else None))
    elif args.command == "remove":
        print(remove())
    else:
        st = status()
        print(f"Система: {st['system']}, настроено {st['installed']} из {st['total']}")
        for task in st["tasks"]:
            mark = "✓" if task["installed"] else " "
            print(f" {mark} {task['cron']:14} {task['title']}")
            print(f"                  {task['why']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
