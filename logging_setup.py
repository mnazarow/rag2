"""
Сквозное логирование всех подсистем.

Что даёт:
  · единый формат для всех модулей, человекочитаемый или JSON;
  · отдельный уровень для каждой подсистемы — можно включить подробности
    только у поиска, не заливая журнал разбором документов;
  · сквозной идентификатор запроса: одна цепочка «вопрос в Telegram →
    поиск → обращение к модели → ответ» видна целиком по одному номеру;
  · ротация файлов по размеру, отдельный файл для ошибок;
  · маскирование чувствительного: токены, ключи, телефоны и адреса
    не должны попадать в журнал даже случайно;
  · запись в журнал событий базы, чтобы админка показывала их без
    доступа к файлам на диске.

Уровни: TRACE (5) — всё до последнего вызова, DEBUG, INFO, WARNING,
ERROR, CRITICAL. TRACE добавлен специально: при разборе проблем нужно
видеть каждый найденный фрагмент с его оценкой, а в DEBUG это утонет.
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
import re
import sys
import threading
import time
import uuid
from contextvars import ContextVar
from pathlib import Path

import config

TRACE = 5
logging.addLevelName(TRACE, "TRACE")

# Подсистемы, у каждой свой уровень в настройках.
SUBSYSTEMS = {
    "index": "Индексация: обход папки, извлечение текста, чанкинг",
    "extract": "Разбор файлов по форматам",
    "embed": "Векторизация и кэш эмбеддингов",
    "search": "Поиск: каналы, слияние, переранжирование, оценки",
    "answer": "Формирование ответа, промпт, отказы",
    "llm": "Обращения к языковой модели и к моделям зрения",
    "prices": "Разбор прайс-листов и поиск по артикулам",
    "media": "Видео, изображения, чертежи",
    "ocr": "Распознавание сканов и проверка на подмену кириллицы латиницей",
    "backup": "Резервные копии индекса и проверка восстановления",
    "voice": "Распознавание и синтез речи",
    "sip": "Телефония",
    "crawl": "Обход сайтов и поиск в интернете",
    "bot": "Telegram: сообщения, кнопки, доступ",
    "web": "Веб-интерфейс администратора",
    "db": "Хранилище",
    "watch": "Слежение за папкой и изменениями структуры",
}

_request_id: ContextVar[str] = ContextVar("request_id", default="-")
_user_id: ContextVar[str] = ContextVar("user_id", default="-")
_channel: ContextVar[str] = ContextVar("channel", default="-")

_configured = False
_lock = threading.Lock()

# Что вырезаем из сообщений до записи.
_MASKS = [
    (re.compile(r"(?i)(bot)?\d{8,10}:[A-Za-z0-9_-]{30,}"), "<токен-телеграм>"),
    (re.compile(r"(?i)(api[-_ ]?key|authorization|token|secret|password|пароль)"
                r"[\"'\s:=]+([^\s\"',}]{6,})"), r"\1=<скрыто>"),
    (re.compile(r"(?<![\d\w])(?:\+7|8)[\s(-]?\d{3}[\s)-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}(?!\d)"), "<телефон>"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b"), "<почта>"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "<адрес>"),
]


def mask(text: str) -> str:
    if not config.LOG_MASK_SENSITIVE:
        return text
    for rx, repl in _MASKS:
        text = rx.sub(repl, text)
    return text


class ContextFilter(logging.Filter):
    """Добавляет к каждой записи номер запроса, пользователя и канал."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        record.user_id = _user_id.get()
        record.channel = _channel.get()
        if isinstance(record.msg, str):
            record.msg = mask(record.msg)
        return True


class HumanFormatter(logging.Formatter):
    COLORS = {"TRACE": "\033[90m", "DEBUG": "\033[36m", "INFO": "\033[32m",
              "WARNING": "\033[33m", "ERROR": "\033[31m", "CRITICAL": "\033[1;31m"}
    RESET = "\033[0m"

    def __init__(self, color: bool = False) -> None:
        super().__init__(datefmt="%H:%M:%S")
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, self.datefmt)
        name = record.name.replace("kb.", "")
        rid = getattr(record, "request_id", "-")
        rid_part = f" [{rid}]" if rid and rid != "-" else ""
        level = record.levelname
        if self.color and level in self.COLORS:
            level = f"{self.COLORS[level]}{level:<8}{self.RESET}"
        else:
            level = f"{level:<8}"
        text = f"{ts} {level} {name:<8}{rid_part} {record.getMessage()}"
        if record.exc_info:
            text += "\n" + self.formatException(record.exc_info)
        return text


class JsonFormatter(logging.Formatter):
    """Для сбора во внешнюю систему: Loki, ELK, что угодно."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
            "user": getattr(record, "user_id", "-"),
            "channel": getattr(record, "channel", "-"),
        }
        for key, value in getattr(record, "extra_fields", {}).items():
            payload[key] = value
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class DatabaseHandler(logging.Handler):
    """
    Дублирует важные записи в базу, чтобы админка показывала журнал
    без доступа к файлам. Пишет только от WARNING и выше плюс события,
    помеченные как значимые.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self._ready = False

    def _ensure(self) -> None:
        if self._ready:
            return
        import db
        db.telemetry().executescript("""
        CREATE TABLE IF NOT EXISTS log_records (
            id         INTEGER PRIMARY KEY,
            ts         TEXT,
            level      TEXT,
            subsystem  TEXT,
            request_id TEXT,
            message    TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_log_ts ON log_records(ts);
        """)
        self._ready = True

    def emit(self, record: logging.LogRecord) -> None:
        try:
            import db
            self._ensure()
            db.trun("INSERT INTO log_records(ts, level, subsystem, request_id, message) "
                   "VALUES (?,?,?,?,?)",
                   (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created)),
                    record.levelname, record.name.replace("kb.", ""),
                    getattr(record, "request_id", "-"), record.getMessage()[:2000]))
            db.trun("DELETE FROM log_records WHERE id < "
                   "(SELECT MAX(id) - ? FROM log_records)", (config.LOG_DB_KEEP,))
        except Exception:  # noqa: BLE001 — журнал не имеет права ломать работу
            pass


def setup(force: bool = False) -> None:
    """Вызывается один раз при старте любого процесса."""
    global _configured
    with _lock:
        if _configured and not force:
            return
        root = logging.getLogger("kb")
        root.handlers.clear()
        root.setLevel(TRACE)
        root.propagate = False

        ctx = ContextFilter()
        json_mode = config.LOG_FORMAT == "json"

        if config.LOG_TO_CONSOLE:
            console = logging.StreamHandler(sys.stderr)
            console.setLevel(getattr(logging, config.LOG_LEVEL_CONSOLE, logging.INFO)
                             if config.LOG_LEVEL_CONSOLE != "TRACE" else TRACE)
            console.setFormatter(JsonFormatter() if json_mode
                                 else HumanFormatter(color=sys.stderr.isatty()))
            console.addFilter(ctx)
            root.addHandler(console)

        if config.LOG_TO_FILE:
            log_dir = Path(config.LOG_DIR)
            log_dir.mkdir(parents=True, exist_ok=True)
            main = logging.handlers.RotatingFileHandler(
                log_dir / "assistant.log", maxBytes=config.LOG_MAX_MB * 1024 * 1024,
                backupCount=config.LOG_BACKUPS, encoding="utf-8")
            main.setLevel(getattr(logging, config.LOG_LEVEL_FILE, logging.DEBUG)
                          if config.LOG_LEVEL_FILE != "TRACE" else TRACE)
            main.setFormatter(JsonFormatter() if json_mode else HumanFormatter())
            main.addFilter(ctx)
            root.addHandler(main)

            errors = logging.handlers.RotatingFileHandler(
                log_dir / "errors.log", maxBytes=config.LOG_MAX_MB * 1024 * 1024,
                backupCount=config.LOG_BACKUPS, encoding="utf-8")
            errors.setLevel(logging.WARNING)
            errors.setFormatter(JsonFormatter() if json_mode else HumanFormatter())
            errors.addFilter(ctx)
            root.addHandler(errors)

        if config.LOG_TO_DB:
            handler = DatabaseHandler()
            handler.addFilter(ctx)
            root.addHandler(handler)

        # Индивидуальные уровни подсистем: LOG_LEVEL_SEARCH=TRACE и т.п.
        for name in SUBSYSTEMS:
            value = os.environ.get(f"LOG_LEVEL_{name.upper()}")
            if value:
                level = TRACE if value.upper() == "TRACE" else \
                    getattr(logging, value.upper(), logging.INFO)
                logging.getLogger(f"kb.{name}").setLevel(level)

        # Шумные чужие библиотеки приглушаем.
        for noisy in ("httpx", "httpcore", "urllib3", "asyncio", "aiogram",
                      "PIL", "matplotlib"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        _configured = True


class SubsystemLogger(logging.LoggerAdapter):
    def trace(self, msg, *args, **kwargs) -> None:
        if self.logger.isEnabledFor(TRACE):
            self.logger.log(TRACE, msg, *args, **kwargs)

    def timed(self, message: str):
        """Контекстный менеджер: пишет длительность операции."""
        return _Timer(self, message)


class _Timer:
    def __init__(self, log: SubsystemLogger, message: str) -> None:
        self.log, self.message, self.started = log, message, 0.0

    def __enter__(self):
        self.started = time.time()
        self.log.debug("%s — начало", self.message)
        return self

    def __exit__(self, exc_type, exc, tb):
        ms = int((time.time() - self.started) * 1000)
        if exc_type is None:
            self.log.info("%s — готово за %d мс", self.message, ms)
        else:
            self.log.error("%s — сбой за %d мс: %s", self.message, ms, exc)
        return False


def get(subsystem: str) -> SubsystemLogger:
    setup()
    return SubsystemLogger(logging.getLogger(f"kb.{subsystem}"), {})


# ------------------------------------------------- контекст запроса ---------
def new_request(user_id: str | int = "-", channel: str = "-") -> str:
    """Начало новой цепочки. Возвращает короткий номер для показа в ответе."""
    rid = uuid.uuid4().hex[:8]
    _request_id.set(rid)
    _user_id.set(str(user_id))
    _channel.set(channel)
    return rid


def current_request() -> str:
    return _request_id.get()


def set_request(rid: str) -> None:
    _request_id.set(rid)


def levels() -> dict:
    """Текущие уровни подсистем — для админки."""
    setup()
    out = {}
    for name, description in SUBSYSTEMS.items():
        logger = logging.getLogger(f"kb.{name}")
        level = logger.level or logging.getLogger("kb").level
        out[name] = {"description": description,
                     "level": logging.getLevelName(level)}
    return out


def set_level(subsystem: str, level: str) -> None:
    setup()
    value = TRACE if level.upper() == "TRACE" else getattr(logging, level.upper(), None)
    if value is None:
        raise ValueError(f"неизвестный уровень: {level}")
    logging.getLogger(f"kb.{subsystem}").setLevel(value)


def tail(lines: int = 200, level: str | None = None, subsystem: str | None = None) -> list[str]:
    """Последние строки журнала — для показа в админке."""
    path = Path(config.LOG_DIR) / "assistant.log"
    if not path.exists():
        return []
    try:
        with path.open("rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            block = min(size, lines * 400)
            fh.seek(size - block)
            data = fh.read().decode("utf-8", "ignore")
    except OSError:
        return []
    out = data.splitlines()[-lines * 3:]
    if level:
        out = [ln for ln in out if level.upper() in ln]
    if subsystem:
        out = [ln for ln in out if subsystem in ln]
    return out[-lines:]
