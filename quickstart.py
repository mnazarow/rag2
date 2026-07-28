"""
Быстрый старт: те же десять шагов, что печатает установщик, — но с
состоянием и кнопками.

Зачем это в интерфейсе. Установщик печатает список команд и на этом
прощается. Дальше человек остаётся один на один с терминалом и десятью
длинными путями, а главное — не имеет никакого способа понять, где он
сейчас находится. Проиндексировал ли он базу? Обучил ли смысловую
модель? Шаги три и четыре внешне ничем не отличаются: обе команды
отработали без ошибок, а поиск всё равно находит только точные слова.
Пропущенный шаг обнаруживается через неделю по жалобе сотрудника.

Поэтому здесь у каждого шага есть три вещи: состояние (сделано, не
сделано, не требуется), настройки, которые ему нужны, и кнопка. Команда
для терминала показывается рядом — она никуда не делась и остаётся
единственным способом сделать то же самое по SSH.

Порядок шагов повторяет вывод установщика намеренно, вплоть до
нумерации: человек, который начал по бумажке, должен видеть здесь ровно
то же самое.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import config
import db

# Шаг: ключ, заголовок, зачем, команда в терминале, что нужно настроить,
# какое действие выполняет кнопка.
#
# Действие задаётся как {"kind": вид задачи} — тогда кнопка ставит задачу
# в очередь, или {"post": путь, "body": тело} для отдельных ручек.


def _docs() -> dict:
    try:
        row = db.q1("SELECT COUNT(*) n FROM documents WHERE status='ok'")
        chunks = db.q1("SELECT COUNT(*) n FROM chunks")
        return {"documents": row["n"] if row else 0,
                "chunks": chunks["n"] if chunks else 0}
    except Exception:  # noqa: BLE001
        return {"documents": 0, "chunks": 0}


def _step_kb_root() -> dict:
    root = Path(config.KB_ROOT)
    exists = root.exists()
    files = 0
    if exists:
        try:
            # Считаем не всё дерево, а до первой тысячи: на 116 ГБ полный
            # обход занял бы минуты, а ответ «файлы есть» нужен сразу.
            for i, _ in enumerate(root.rglob("*")):
                files = i + 1
                if files >= 1000:
                    break
        except OSError:
            pass
    return {
        "done": exists and files > 0,
        "detail": (f"{root} — файлов не меньше {files}" if exists and files
                   else f"{root} — папки нет или она пуста"),
        "settings": ["KB_ROOT"],
        "command": "nano .env      # параметр KB_ROOT",
        "hint": "Укажите именно корень базы, а не подпапку: имена разделов "
                "в разграничении доступа берутся из папок первого уровня.",
    }


def _step_index() -> dict:
    counts = _docs()
    return {
        "done": counts["documents"] > 0,
        "detail": (f"документов {counts['documents']}, фрагментов {counts['chunks']}"
                   if counts["documents"] else "индекс пуст"),
        "settings": ["MAX_FILE_MB", "CHUNK_TARGET_CHARS", "EXTRACT_ARCHIVES"],
        "command": "python index.py build",
        "action": {"kind": "reindex_full", "title": "полная индексация базы"},
        "action_label": "Проиндексировать базу",
        "hint": "На двенадцати тысячах файлов проход занимает от семи минут "
                "до получаса. Прерывать безопасно: повторный запуск "
                "продолжит с того же места.",
    }


def _step_semantic() -> dict:
    import search
    ready, note = (False, "")
    try:
        ready, note = search.dense_ready()
    except Exception as exc:  # noqa: BLE001
        note = str(exc)
    return {
        "done": ready,
        "detail": note,
        "settings": ["EMBEDDINGS_PROVIDER", "LSA_DIM", "LSA_MIN_DF"],
        "command": "python index.py train-lsa && python index.py reembed",
        "action": {"kind": "train_lsa", "title": "обучение смысловой модели",
                   "then": "reembed"},
        "action_label": "Обучить модель и пересчитать векторы",
        "hint": "Самый пропускаемый шаг, и самый заметный по последствиям: "
                "без него находятся только точные слова из документа, а "
                "вопрос, заданный своими словами, остаётся без ответа. "
                "Внешне при этом всё работает.",
    }


def _step_check_search() -> dict:
    counts = _docs()
    return {
        "done": None,                      # это проверка, а не состояние
        "detail": ("можно проверять" if counts["documents"]
                   else "сначала нужен индекс"),
        "settings": ["SEARCH_TOP_K", "MIN_CONFIDENCE", "RERANKER_PROVIDER"],
        "command": 'python ask.py "какой напор у Водомет 55/50"',
        "action": {"goto": "search"},
        "action_label": "Открыть проверку поиска",
        "hint": "Задайте вопрос словами сотрудника, а не словами из "
                "документа — проверять надо именно это.",
    }


def _step_backup() -> dict:
    try:
        import backup
        archives = backup.archives()
        last = archives[0].name if archives else ""
    except Exception:  # noqa: BLE001
        archives, last = [], ""
    scheduled = False
    try:
        import schedule as schedule_mod
        state = schedule_mod.status()
        scheduled = any(t["name"] == "backup" and t["installed"]
                        for t in state["tasks"])
    except Exception:  # noqa: BLE001
        pass
    return {
        "done": bool(archives) and scheduled,
        "detail": (f"копий: {len(archives)}, последняя {last}, "
                   f"расписание: {'стоит' if scheduled else 'не стоит'}"
                   if archives else "копий нет"),
        "settings": ["BACKUP_DIR", "BACKUP_SCHEDULE", "BACKUP_KEEP_DAILY"],
        "command": "python backup.py create && python schedule.py install",
        "action": {"kind": "backup", "title": "резервная копия индекса"},
        "action_label": "Сделать копию сейчас",
        "extra": {"label": "Поставить расписание",
                  "post": "/api/schedule", "body": {"action": "install"}},
        "hint": "Индекс — это часы машинного времени и накопленные выверенные "
                "ответы, которые пересборкой не восстанавливаются.",
    }


def _step_ocr() -> dict:
    try:
        pending = db.q1("SELECT COUNT(*) n FROM documents "
                        "WHERE needs_ocr=1 AND status='ok'")["n"]
    except Exception:  # noqa: BLE001
        pending = 0
    provider = config.OCR_PROVIDER
    return {
        "done": pending == 0,
        "skip": pending == 0,
        "detail": (f"ждут распознавания: {pending}, распознаватель: {provider}"
                   if pending else "сканов, требующих распознавания, нет"),
        "settings": ["OCR_PROVIDER", "OCR_LANGUAGES", "OCR_MAX_LATIN_RATIO"],
        "command": "python ocr.py providers && python ocr.py run",
        "action": {"kind": "ocr", "title": "распознавание сканов"},
        "action_label": "Распознать сканы",
        "hint": "Сертификаты и декларации почти всегда сканы. Проверьте, что "
                "выбранный распознаватель не подменяет кириллицу латиницей: "
                "«МОСКВА» не должна превращаться в «MOCKBA».",
    }


def _step_queue() -> dict:
    import llm_queue
    state = llm_queue.status()
    local = config.LLM_PROVIDER == "local"
    return {
        "done": True,
        "skip": not local,
        "detail": (f"одновременно {state['limit'] or 'без ограничения'}, "
                   f"сейчас выполняется {state.get('running', 0)}, "
                   f"ждут {state.get('waiting', 0)}"),
        "settings": ["LLM_MAX_CONCURRENT", "LLM_QUEUE_MAX", "LLM_QUEUE_TIMEOUT"],
        "command": "python llm_queue.py status",
        "action": {"goto": "models"},
        "action_label": "Открыть очередь",
        "hint": "Для одной видеокарты один одновременный запрос — правильное "
                "значение: иначе десять вопросов делят её память и быстрый "
                "ответ не получает никто. Для облака очередь не нужна.",
    }


def _step_webui() -> dict:
    return {
        "done": True,
        "detail": f"вы сейчас здесь: http://{config.ADMIN_HOST}:{config.ADMIN_PORT}/",
        "settings": ["ADMIN_HOST", "ADMIN_PORT"],
        "command": "python webui.py",
        "hint": "",
    }


def _step_preflight() -> dict:
    import preflight
    report = preflight.check("админка")
    return {
        "done": not report["fatal"],
        "detail": ("проверка пройдена" if not report["fatal"]
                   else "; ".join(report["fatal"])[:300]),
        "warnings": report["warn"],
        "settings": ["ADMIN_TRUST_PROXY", "ADMIN_TOKEN", "DEFAULT_ROLE",
                     "ROLE_SECTIONS"],
        "command": "python preflight.py",
        "action": {"goto": "diag"},
        "action_label": "Открыть диагностику",
        "hint": "Проверка не даст запуститься с админкой, открытой наружу без "
                "пароля, и с ролью по умолчанию, которой нет в разграничении "
                "доступа.",
    }


def _step_telegram() -> dict:
    token = bool(config.TELEGRAM_BOT_TOKEN)
    admins = bool(config.TELEGRAM_ADMIN_IDS)
    try:
        users = db.q1("SELECT COUNT(*) n FROM users WHERE approved=1")["n"]
    except Exception:  # noqa: BLE001
        users = 0
    return {
        "done": token and admins,
        "detail": ("токен задан, " if token else "токен не задан, ")
                  + ("администраторы указаны, " if admins
                     else "администраторы не указаны, ")
                  + f"сотрудников с доступом: {users}",
        "settings": ["TELEGRAM_BOT_TOKEN", "TELEGRAM_ADMIN_IDS",
                     "TELEGRAM_ALLOWED_IDS"],
        "command": "python bot.py",
        "action": {"goto": "users"},
        "action_label": "Открыть список сотрудников",
        "hint": "Бот — отдельный процесс, из интерфейса он не запускается: "
                "иначе при перезапуске админки он бы останавливался вместе с "
                "ней. Запускайте его службой или командой выше.",
    }


STEPS = [
    ("kb_root", "Указать путь к базе знаний", _step_kb_root),
    ("index", "Проиндексировать базу", _step_index),
    ("semantic", "Включить смысловой поиск", _step_semantic),
    ("check", "Проверить поиск", _step_check_search),
    ("backup", "Настроить резервное копирование", _step_backup),
    ("ocr", "Распознать сканы", _step_ocr),
    ("queue", "Проверить очередь к модели", _step_queue),
    ("webui", "Веб-интерфейс", _step_webui),
    ("preflight", "Проверить настройку перед выходом наружу", _step_preflight),
    ("telegram", "Подключить Telegram", _step_telegram),
]


def state() -> dict:
    """Состояние всех шагов. Ошибка в одном не должна ронять остальные."""
    steps = []
    for key, title, fn in STEPS:
        try:
            item = fn()
        except Exception as exc:  # noqa: BLE001
            item = {"done": False, "detail": f"не удалось проверить: {exc}",
                    "settings": [], "command": "", "hint": ""}
        item.update({"key": key, "title": title, "number": len(steps) + 1})
        steps.append(item)

    required = [s for s in steps if not s.get("skip")]
    done = [s for s in required if s.get("done")]
    # Первый невыполненный обязательный шаг — с него и продолжать.
    nxt = next((s["key"] for s in required if s.get("done") is False), "")
    return {
        "steps": steps,
        "done": len(done),
        "total": len(required),
        "next": nxt,
        "python": _python_hint(),
        "base_dir": str(config.BASE_DIR),
    }


def _python_hint() -> str:
    """Каким интерпретатором запускать команды на этой установке."""
    venv = Path(config.BASE_DIR) / "venv" / "bin" / "python"
    if venv.exists():
        return str(venv)
    return shutil.which("python3") or "python"
