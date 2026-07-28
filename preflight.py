"""
Проверка перед стартом: можно ли вообще работать с такой настройкой.

Зачем. До сих пор любой процесс поднимался при какой угодно
конфигурации. Папки базы знаний не существует — админка запускается,
systemd показывает `active`, всё выглядит исправным. Первым о поломке
узнаёт сотрудник, задавший вопрос, а между запуском и этим вопросом
могут пройти сутки. Хуже того, часть таких поломок вообще не даёт
ошибок: смысловой поиск молча отключается, и система продолжает
отвечать — просто заметно хуже.

Проверки разделены на два вида, и разделение здесь важнее самих
проверок.

**Не запускаемся** — то, при чём работа невозможна или опасна: некуда
писать данные, база повреждена, админка открыта наружу без всякой
защиты. Молча продолжать в таких случаях хуже, чем не подняться:
упавшая служба видна сразу, а тихо открытая наружу админка не видна
никому.

**Предупреждаем и работаем** — то, что ухудшает качество, но работать
не мешает: не обучена модель смыслового поиска, не распознаны сканы,
недоступна модель генерации. Останавливать из-за этого нельзя: бот,
отвечающий хуже, всё равно лучше бота, который не отвечает.

  python preflight.py            — проверить и показать
  python preflight.py --strict   — вернуть код 2, если есть предупреждения
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import config
import logging_setup

log = logging_setup.get("web")


def _writable(path: Path) -> tuple[bool, str]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".preflight"
        probe.write_text("x", encoding="utf-8")
        probe.unlink()
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def check(role: str = "процесс") -> dict:
    """
    role — что именно запускается: «админка», «бот», «индексация».
    От этого зависит, что считать критичным: боту нужен токен, а
    индексации он безразличен.
    """
    fatal: list[str] = []
    warn: list[str] = []
    ok: list[str] = []

    # --- куда пишем ---
    writable, why = _writable(Path(config.DATA_DIR))
    if writable:
        ok.append(f"папка данных доступна на запись: {config.DATA_DIR}")
    else:
        fatal.append(f"нет доступа на запись в папку данных {config.DATA_DIR}: {why}")

    free_gb = shutil.disk_usage(config.DATA_DIR).free / 2**30 if writable else 0
    if writable and free_gb < 1:
        fatal.append(f"на диске с данными свободно {free_gb:.1f} ГБ — "
                     "индекс и резервные копии писать некуда")
    elif writable and free_gb < 10:
        warn.append(f"на диске с данными свободно всего {free_gb:.1f} ГБ")

    # --- база знаний ---
    if role in ("админка", "индексация", "бот"):
        if not Path(config.KB_ROOT).exists():
            (fatal if role == "индексация" else warn).append(
                f"папка базы знаний не найдена: {config.KB_ROOT}. "
                "Проверьте настройку KB_ROOT")
        else:
            ok.append(f"база знаний на месте: {config.KB_ROOT}")

    # --- хранилище ---
    try:
        import db
        db.init()
        docs = db.q1("SELECT COUNT(*) n FROM documents WHERE status='ok'")["n"]
        ok.append(f"хранилище открывается, документов в индексе: {docs}")
        if docs == 0 and role != "индексация":
            warn.append("индекс пуст — выполните: python index.py build")
        store = db.vectors()
        if getattr(store, "broken", ""):
            fatal.append(f"векторный индекс повреждён: {store.broken}. "
                         "Восстановите из копии или выполните reembed")
        elif len(store) == 0 and docs:
            warn.append("смысловой поиск не работает: векторов нет. "
                        "Выполните: python index.py train-lsa && python index.py reembed")
    except Exception as exc:  # noqa: BLE001
        fatal.append(f"не удалось открыть хранилище: {exc}")

    # --- доступ к админке ---
    if role == "админка":
        import security
        listening_outside = config.ADMIN_HOST not in ("127.0.0.1", "localhost", "::1")
        protected = security.accounts_enabled() or bool(config.ADMIN_TOKEN)
        if listening_outside and not protected:
            fatal.append(
                f"админка слушает {config.ADMIN_HOST}, а защиты нет: ни учётных "
                "записей, ни ADMIN_TOKEN. Любой, кто дотянется до порта, сможет "
                "восстановить индекс из копии и прочитать всю базу")
        elif config.ADMIN_TRUST_PROXY and not security.accounts_enabled():
            fatal.append(
                "включён ADMIN_TRUST_PROXY, но учётных записей нет. За обратным "
                "прокси адрес клиента всегда локальный, поэтому правило «с "
                "локального адреса можно всё» открыло бы админку наружу. "
                "Заведите учётную запись: python security.py adduser")
        else:
            ok.append("доступ к админке ограничен")
        health = security.secrets_health()
        for problem in health.get("problems", []):
            warn.append(problem)

    # --- бот ---
    if role == "бот":
        if not config.TELEGRAM_BOT_TOKEN:
            fatal.append("не задан TELEGRAM_BOT_TOKEN — боту нечем подключиться")
        else:
            ok.append("токен бота задан")
        if not config.TELEGRAM_ADMIN_IDS:
            warn.append("не задан TELEGRAM_ADMIN_IDS — заявки на доступ "
                        "никому не придут")

    # --- разграничение доступа ---
    if config.ROLE_SECTIONS:
        known = set(config.ROLE_SECTIONS)
        if config.DEFAULT_ROLE not in known:
            fatal.append(
                f"роль по умолчанию «{config.DEFAULT_ROLE}» не описана в "
                f"ROLE_SECTIONS (есть: {', '.join(sorted(known))}). "
                "Выдача будет закрыта целиком, а при опечатке в другую сторону "
                "была бы открыта целиком")
        if Path(config.KB_ROOT).exists():
            real = {p.name for p in Path(config.KB_ROOT).iterdir() if p.is_dir()}
            listed = {s for v in config.ROLE_SECTIONS.values() for s in v} - {"*"}
            unknown = listed - real
            if unknown and real:
                warn.append("в ROLE_SECTIONS перечислены разделы, которых нет в "
                            f"базе: {', '.join(sorted(unknown))}. Обычно это "
                            "опечатка, и роль не увидит ничего")
    else:
        warn.append("разграничение доступа не настроено (ROLE_SECTIONS пуст) — "
                    "все роли видят всю базу, включая дилерские цены")

    # --- модель генерации ---
    if role in ("админка", "бот"):
        if config.LLM_PROVIDER == "echo":
            warn.append("модель генерации — заглушка echo: ответ склеивается из "
                        "найденных предложений")
        else:
            ok.append(f"модель генерации: {config.LLM_PROVIDER}")

    return {"role": role, "fatal": fatal, "warn": warn, "ok": ok}


def render(report: dict) -> str:
    lines = []
    for item in report["fatal"]:
        lines.append(f"  ✗ {item}")
    for item in report["warn"]:
        lines.append(f"  ! {item}")
    if not report["fatal"] and not report["warn"]:
        lines.append(f"  ✓ проверка перед запуском пройдена ({len(report['ok'])} пунктов)")
    elif not report["fatal"]:
        lines.append(f"  ✓ запускаемся, замечаний: {len(report['warn'])}")
    else:
        lines.append("  Запуск прерван. Исправьте отмеченное знаком ✗ и запустите снова.")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Проверка перед запуском")
    p.add_argument("--role", default="админка",
                   choices=["админка", "бот", "индексация", "процесс"])
    p.add_argument("--strict", action="store_true",
                   help="считать предупреждения ошибками")
    args = p.parse_args()
    report = check(args.role)
    print(f"Проверка перед запуском ({args.role}):")
    print(render(report))
    for item in report["ok"]:
        print(f"  ✓ {item}")
    if report["fatal"]:
        return 2
    return 2 if (args.strict and report["warn"]) else 0


if __name__ == "__main__":
    sys.exit(main())
