"""
Конфигурация RAG-ассистента корпоративной базы знаний.

Все настройки читаются из переменных окружения (файл .env рядом с проектом).
Провайдеры моделей подключаются по имени — см. embeddings.py и llm.py.
"""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


# Какие переменные пришли из .env, а какие заданы снаружи — в systemd,
# docker-compose или прямо в командной строке. Различать необходимо:
# заданное снаружи главнее файла, иначе контейнер, которому передали свой
# DATA_DIR, получил бы значение из .env и полез не туда.
#
# Список хранится в самом окружении, а не в переменной модуля, потому что
# при сохранении настроек модуль перечитывается целиком: обычная
# переменная обнулилась бы, и различать стало бы нечем.
# Службы macOS (launchd) получают PATH без каталогов Homebrew, и всё,
# что стоит через brew — ollama, ffmpeg, tesseract, poppler, — для них
# «не установлено», хотя из терминала прекрасно видно. Одна и та же
# админка из терминала и из автозапуска вела себя по-разному, и
# предупреждение «нет ollama» при стоящем ollama — ровно отсюда.
# Дополняем PATH здесь, потому что config импортируется первым и все
# проверки через shutil.which после этого честны в обоих режимах.
import platform as _platform
if _platform.system() == "Darwin":
    for _extra in ("/opt/homebrew/bin", "/usr/local/bin"):
        if os.path.isdir(_extra) and _extra not in os.environ.get("PATH", "").split(":"):
            os.environ["PATH"] = os.environ.get("PATH", "") + ":" + _extra

_MARKER = "_KB_KEYS_FROM_DOTENV"


def _load_dotenv() -> None:
    """
    Минимальный загрузчик .env без внешних зависимостей.

    Раньше здесь стоял `setdefault`, и это была тихая, но заметная в
    работе ошибка: после первой загрузки значение оставалось в окружении
    навсегда, и повторное чтение файла ничего не меняло. Правка настройки
    в админке записывалась в .env, модули послушно перечитывались — и
    продолжали работать со старым значением. Со стороны это выглядит так,
    будто настройка ни на что не влияет, и её меняют ещё раз, и ещё.
    """
    env_file = BASE_DIR / ".env"
    if not env_file.exists():
        return
    from_file = set(x for x in os.environ.get(_MARKER, "").split(",") if x)
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        # Значение, заданное снаружи, файл не перебивает — но своё
        # собственное, прочитанное отсюда же в прошлый раз, обновляет.
        if key in os.environ and key not in from_file:
            continue
        os.environ[key] = value
        from_file.add(key)
    os.environ[_MARKER] = ",".join(sorted(from_file))


_load_dotenv()


def env_mtime() -> float:
    """Когда .env менялся в последний раз. 0, если файла нет."""
    env_file = BASE_DIR / ".env"
    try:
        return env_file.stat().st_mtime
    except OSError:
        return 0.0


def reload_if_changed() -> bool:
    """
    Перечитать настройки, если файл изменился.

    Нужно соседним процессам. Настройки правят в админке, а бот, слежение
    за папкой и телефония — отдельные процессы (в контейнерах вообще
    отдельные контейнеры). Раньше они работали со старыми значениями до
    ручного перезапуска, и это никак не показывалось: администратор
    менял порог отказа, проверял в админке — работает, а бот продолжал
    отвечать по-старому неделями.

    Проверка дешёвая: один stat файла. Вызывается из основного цикла.
    """
    now = env_mtime()
    if not now:
        return False
    # Отметку держим в окружении, а не в переменной модуля: перечитывание
    # выполняет тело модуля заново, и обычная переменная обнулилась бы —
    # получился бы бесконечный цикл перечитываний.
    seen = os.environ.get("_KB_ENV_MTIME", "")
    if seen == f"{now:.6f}":
        return False
    os.environ["_KB_ENV_MTIME"] = f"{now:.6f}"
    if not seen:                      # первый вызов — просто запомнили
        return False
    import importlib
    import sys as _sys
    importlib.reload(_sys.modules[__name__])
    return True


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------- источники --
# Корень корпоративной базы знаний (папка BD).
KB_ROOT = Path(_env("KB_ROOT", str(Path.home() / "Claude/Projects/RAG2/BD"))).expanduser()

# Куда складывать индекс, логи, кэш извлечённого текста.
DATA_DIR = Path(_env("DATA_DIR", str(BASE_DIR / "data"))).expanduser()
DB_PATH = DATA_DIR / "kb.sqlite3"
# Телеметрия (метрики сервера и журнал) пишется постоянно, а с документами
# не соединяется никогда. В отдельном файле она не мешает индексации и
# ответам бота конкурировать за блокировку.
TELEMETRY_SEPARATE = _env("TELEMETRY_SEPARATE", "1") == "1"
TELEMETRY_PATH = DATA_DIR / "telemetry.sqlite3"
# Сколько ждать освобождения блокировки, прежде чем признать неудачу.
DB_BUSY_TIMEOUT_MS = _env_int("DB_BUSY_TIMEOUT_MS", 15000)
VECTORS_PATH = DATA_DIR / "vectors.npy"
# Где лежат векторы: numpy — файл рядом с индексом (просто и быстро до
# сотен тысяч фрагментов), qdrant — отдельный сервис (переживает
# параллельную запись и фильтрует по разделу внутри поиска).
VECTOR_BACKEND = _env("VECTOR_BACKEND", "numpy")            # numpy | qdrant
QDRANT_URL = _env("QDRANT_URL", "http://127.0.0.1:6333")
QDRANT_COLLECTION = _env("QDRANT_COLLECTION", "kb_chunks")
QDRANT_API_KEY = _env("QDRANT_API_KEY", "")
QDRANT_BATCH = _env_int("QDRANT_BATCH", 512)
QDRANT_TIMEOUT = _env_float("QDRANT_TIMEOUT", 30.0)
VECTOR_IDS_PATH = DATA_DIR / "vector_ids.json"

# Расширения, из которых извлекаем текст.
TEXT_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md", ".html", ".htm", ".msg", ".pptx"}
TABLE_EXTENSIONS = {".xlsx", ".xlsm", ".xls", ".csv"}
# Чертежи и 3D: текста почти не несут, но нужны как вложения к ответу.
DRAWING_EXTENSIONS = {".dwg", ".dxf", ".rfa", ".rvt", ".rte", ".igs", ".iges",
                      ".stp", ".step", ".ipt", ".sat", ".obj", ".stl", ".m3d"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff",
                    ".svg", ".jfif", ".cr2", ".nef", ".arw"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".wmv", ".m4v", ".webm"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}
ARCHIVE_EXTENSIONS = {".zip", ".rar", ".7z", ".tar", ".gz", ".tgz"}

# Мусор, который никогда не индексируем.
IGNORE_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}
IGNORE_PREFIXES = ("~$", ".~", "._")
# Сохранённые веб-страницы и их ресурсы (в базе их сотни — чистый шум).
IGNORE_DIR_PARTS = {"_files", "__MACOSX"}
IGNORE_EXTENSIONS = {".js", ".css", ".map", ".woff", ".woff2", ".ttf", ".eot", ".ico",
                     ".db", ".dwl", ".dwl2", ".bak", ".tmp", ".temp", ".swp", ".part",
                     ".crdownload", ".err", ".idx", ".ldb", ".lock", ".pyc"}

MAX_FILE_MB = _env_int("MAX_FILE_MB", 120)
# Большие книги Excel — это, как правило, программы подбора с макросами и
# формулами, а не прайсы. Разбор такой книги занимает минуты и почти ничего
# не даёт: для них создаётся карточка, а содержимое не вычитывается.
TABLE_MAX_MB = _env_int("TABLE_MAX_MB", 8)

# ------------------------------------------------- нетекстовые материалы ----
# Карточка объекта: даже для DWG, фото и видео создаётся текстовый фрагмент из
# пути, соседних файлов и описания. Так чертёж «привязывается по смыслу»
# к папке бренда и находится по названию модели.
ASSET_CARDS = _env("ASSET_CARDS", "1") == "1"
# Распаковывать архивы во временную папку и индексировать содержимое.
EXTRACT_ARCHIVES = _env("EXTRACT_ARCHIVES", "1") == "1"
ARCHIVE_MAX_DEPTH = _env_int("ARCHIVE_MAX_DEPTH", 3)
ARCHIVE_MAX_RATIO = _env_int("ARCHIVE_MAX_RATIO", 200)      # защита от zip-бомб
ARCHIVE_WORK_DIR = Path(_env("ARCHIVE_WORK_DIR", str(DATA_DIR / "unpacked")))

# Транскрипция видео и аудио: none | whisper | faster-whisper | yandex | sber
ASR_PROVIDER = _env("ASR_PROVIDER", "none")
ASR_MODEL = _env("ASR_MODEL", "large-v3-turbo")
ASR_LANGUAGE = _env("ASR_LANGUAGE", "ru")
ASR_SEGMENT_SECONDS = _env_int("ASR_SEGMENT_SECONDS", 60)   # длина смыслового куска
ASR_DEVICE = _env("ASR_DEVICE", "auto")

# Описание изображений и превью чертежей: none | openai | gigachat | yandex | local
# «local» по умолчанию: как только скачана зрительная модель из каталога
# (или в ollama есть qwen3-vl), описания начинают делаться сами — локально,
# бесплатно и без отправки изображений наружу. Пока модели нет, описание
# честно пропускается с подсказкой, что скачать.
VISION_PROVIDER = _env("VISION_PROVIDER", "local")
VISION_MODEL = _env("VISION_MODEL", "qwen/qwen3-vl-32b-instruct")
VISION_MAX_SIDE = _env_int("VISION_MAX_SIDE", 1280)

# Конвертация CAD: путь к ODA File Converter (DWG -> DXF) и к FreeCAD.
ODA_CONVERTER = _env("ODA_CONVERTER", "")
FREECAD_CMD = _env("FREECAD_CMD", "freecadcmd")

# ------------------------------------------------ распознавание сканов (OCR) --
# Сертификаты и декларации почти всегда лежат сканами: текстового слоя в них
# нет, и без распознавания они для поиска не существуют.
#
#   none      — выключено
#   tesseract — локально, бесплатно, нужен языковой пакет rus (см. документацию)
#   vlm       — зрительная модель через OpenAI-совместимый endpoint
#               (Qwen3-VL и подобные). Лучшее качество на печатях и таблицах.
#   yandex    — Yandex Vision OCR
#   gigachat  — GigaChat Vision
OCR_PROVIDER = _env("OCR_PROVIDER", "none")
OCR_LANGUAGES = _env("OCR_LANGUAGES", "rus+eng")
OCR_MODEL = _env("OCR_MODEL", "qwen/qwen3-vl-32b-instruct")
OCR_DPI = _env_int("OCR_DPI", 300)              # 300 — компромисс качества и времени
OCR_MAX_PAGES = _env_int("OCR_MAX_PAGES", 30)   # страниц на документ
OCR_MIN_CHARS = _env_int("OCR_MIN_CHARS", 80)   # меньше — считаем, что не распозналось
OCR_WORKERS = _env_int("OCR_WORKERS", 2)
OCR_TIMEOUT = _env_float("OCR_TIMEOUT", 180.0)
OCR_TESSERACT_CMD = _env("OCR_TESSERACT_CMD", "tesseract")
OCR_TESSERACT_PSM = _env_int("OCR_TESSERACT_PSM", 6)
OCR_BASE_URL = _env("OCR_BASE_URL", "")         # пусто = взять OPENAI_BASE_URL
OCR_API_KEY = _env("OCR_API_KEY", "")
# Главная защита: часть распознавателей подменяет кириллицу похожей латиницей —
# «МОСКВА» превращается в «MOCKBA». Для базы с русскими артикулами это яд:
# документ навсегда перестаёт находиться. Страница с такой подменой
# отбраковывается и переотправляется другому провайдеру.
OCR_CYRILLIC_GUARD = _env("OCR_CYRILLIC_GUARD", "1") == "1"
# Допустимая доля «латинских вкраплений» среди русских слов.
OCR_MAX_LATIN_RATIO = _env_float("OCR_MAX_LATIN_RATIO", 0.08)
# Документ с оценкой ниже этой остаётся в очереди: текст в индекс попадёт
# (лучше, чем ничего), но при появлении более точного распознавателя
# документ будет обработан заново.
OCR_MIN_QUALITY = _env_float("OCR_MIN_QUALITY", 0.6)
# Куда откатываться, если основной провайдер провалил проверку (через запятую).
OCR_FALLBACK = _env("OCR_FALLBACK", "")

# ------------------------------------------------------------ безопасность ---
# Ключи и токены хранятся отдельно от остальных настроек: файл с правами
# 600, который не попадает ни в архив обновления, ни в резервные копии.
SECRETS_FILE = Path(_env("SECRETS_FILE", str(BASE_DIR / "secrets.env"))).expanduser()
# Команда, печатающая секреты в формате KEY=VALUE. Через неё подключается
# внешнее хранилище: vault, менеджер паролей, systemd-credentials.
SECRETS_CMD = _env("SECRETS_CMD", "")

# Вход в админку. Пока список пуст, действует прежнее поведение: доступ по
# общему токену ADMIN_TOKEN, а без него — только с локального адреса.
ADMIN_USERS_FILE = Path(_env("ADMIN_USERS_FILE", str(DATA_DIR / "admin_users.json")))
ADMIN_SESSION_HOURS = _env_int("ADMIN_SESSION_HOURS", 12)
# Роли админки: viewer — только смотреть, operator — запускать задачи,
# admin — менять настройки, восстанавливать индекс, выдавать доступ.
ADMIN_DEFAULT_ROLE = _env("ADMIN_DEFAULT_ROLE", "viewer")

# Подбор пароля. Задержка после неудачной попытки от перебора не спасает:
# сервер многопоточный, и сотня одновременных соединений даёт сотню
# попыток в секунду, сколько ни спи внутри каждой. Спасает счётчик.
ADMIN_LOGIN_MAX_FAILS = _env_int("ADMIN_LOGIN_MAX_FAILS", 10)
ADMIN_LOGIN_BLOCK_MINUTES = _env_int("ADMIN_LOGIN_BLOCK_MINUTES", 15)

# Стоит ли перед админкой свой обратный прокси. Признак нельзя выводить
# автоматически: заголовки X-Forwarded-* ставит кто угодно. А цена
# ошибки высокая — при прокси на той же машине адресом клиента для всех
# внешних запросов становится 127.0.0.1, и правило «с локального адреса
# можно всё» открывает админку всему интернету. Поэтому: включили —
# верим заголовкам прокси и обязательно заводим учётные записи.
ADMIN_TRUST_PROXY = _env("ADMIN_TRUST_PROXY", "0") == "1"

# Ограничение частоты обращений к боту: защита бюджета на модель и от
# случайного прогона всей базы вопросами.
RATE_LIMIT_PER_USER_HOUR = _env_int("RATE_LIMIT_PER_USER_HOUR", 60)
RATE_LIMIT_PER_USER_DAY = _env_int("RATE_LIMIT_PER_USER_DAY", 300)
RATE_LIMIT_TOTAL_DAY = _env_int("RATE_LIMIT_TOTAL_DAY", 3000)

# Защита от попыток заставить бота нарушить правила: сотрудник может
# прислать сообщение, составленное так, чтобы бот выдал содержимое
# недоступного ему раздела.
PROMPT_GUARD = _env("PROMPT_GUARD", "1") == "1"

# Срок хранения персональных данных: тексты вопросов, оценки, цепочки.
# 0 = хранить бессрочно (так делать не стоит).
RETENTION_QUERIES_DAYS = _env_int("RETENTION_QUERIES_DAYS", 365)
RETENTION_TRACES_DAYS = _env_int("RETENTION_TRACES_DAYS", 90)

# ------------------------------------------------------------ оповещения -----
# Куда сообщать о проблемах, которые иначе замечают поздно.
ALERTS_ENABLED = _env("ALERTS_ENABLED", "1") == "1"
ALERT_CHANNELS = _env("ALERT_CHANNELS", "telegram")     # telegram, log, webhook
ALERT_WEBHOOK_URL = _env("ALERT_WEBHOOK_URL", "")
ALERT_REPEAT_HOURS = _env_int("ALERT_REPEAT_HOURS", 12)  # не напоминать чаще
ALERT_DISK_FREE_GB = _env_int("ALERT_DISK_FREE_GB", 10)
# Раз в столько минут админка сама гоняет проверки оповещений — дублируя
# cron: снесённый crontab не должен превращаться в молчащий мониторинг.
# 0 — выключить самопроверку (только cron).
ALERTS_SELF_CHECK_MINUTES = _env_int("ALERTS_SELF_CHECK_MINUTES", 60)
ALERT_OCR_QUEUE = _env_int("ALERT_OCR_QUEUE", 50)
ALERT_REFUSAL_RATE = _env_float("ALERT_REFUSAL_RATE", 0.35)

# ------------------------------------------------------------ трассировка ----
# Полная цепочка одного ответа: вопрос → что нашлось с какими оценками →
# какой промпт ушёл в модель → что она ответила. Без этого разбор жалобы
# «бот ответил неправильно» превращается в гадание.
#
# Важно про персональные данные: в записи попадает текст вопроса и куски
# документов. Поэтому срок хранения ограничен, а сама трассировка живёт
# в отдельной базе телеметрии, которая не попадает в резервные копии индекса.
TRACE_ENABLED = _env("TRACE_ENABLED", "1") == "1"
TRACE_KEEP = _env_int("TRACE_KEEP", 3000)          # сколько последних цепочек хранить
TRACE_PROMPT = _env("TRACE_PROMPT", "1") == "1"    # сохранять ли текст промпта

# --------------------------------------------------- резервное копирование ----
# Индекс — это часы машинного времени плюс выверенные ответы и обучающие
# пары, которые пересборкой не восстанавливаются.
BACKUP_DIR = Path(_env("BACKUP_DIR", str(DATA_DIR / "backups"))).expanduser()
BACKUP_KEEP_DAILY = _env_int("BACKUP_KEEP_DAILY", 7)
BACKUP_KEEP_WEEKLY = _env_int("BACKUP_KEEP_WEEKLY", 4)
BACKUP_KEEP_MONTHLY = _env_int("BACKUP_KEEP_MONTHLY", 6)
BACKUP_COMPRESS = _env("BACKUP_COMPRESS", "1") == "1"
# Проверять каждый снимок сразу после создания: развернуть во временную папку
# и убедиться, что база открывается и данные на месте.
BACKUP_VERIFY = _env("BACKUP_VERIFY", "1") == "1"
# Расписание для установки службы: cron-выражение (по умолчанию 3:30 ночи).
BACKUP_SCHEDULE = _env("BACKUP_SCHEDULE", "30 3 * * *")
# Дополнительная копия готового снимка (второй диск, сетевая папка, NAS).
BACKUP_MIRROR_DIR = _env("BACKUP_MIRROR_DIR", "")
# Предупреждать в веб-интерфейсе, если снимка не было столько часов.
BACKUP_ALERT_HOURS = _env_int("BACKUP_ALERT_HOURS", 48)

# ------------------------------------------------------------------ чанкинг --
CHUNK_TARGET_CHARS = _env_int("CHUNK_TARGET_CHARS", 1400)
CHUNK_OVERLAP_CHARS = _env_int("CHUNK_OVERLAP_CHARS", 200)
CHUNK_MIN_CHARS = _env_int("CHUNK_MIN_CHARS", 120)
# Contextual Retrieval: дописывать к чанку сгенерированный LLM контекст.
# Заметно повышает качество, но требует LLM-вызова на каждый чанк при индексации.
CONTEXTUAL_CHUNKS = _env("CONTEXTUAL_CHUNKS", "0") == "1"

# --------------------------------------------------------------- эмбеддинги --
# gigachat | yandex | openai | local | onnx | lsa | hashing
EMBEDDINGS_PROVIDER = _env("EMBEDDINGS_PROVIDER", "lsa")
EMBEDDINGS_MODEL = _env("EMBEDDINGS_MODEL", "")
EMBEDDINGS_DIM = _env_int("EMBEDDINGS_DIM", 1024)
EMBEDDINGS_BATCH = _env_int("EMBEDDINGS_BATCH", 32)

# --- LSA: смысловая модель, обучаемая на вашей же базе -----------------------
# Не требует интернета, весов и видеокарты: раскладывает матрицу
# «фрагмент × слово», построенную прямо по вашему индексу. Даёт настоящую
# смысловую близость в пределах лексики базы. Обучение — минуты,
# файл модели — десятки мегабайт.
LSA_MODEL_PATH = Path(_env("LSA_MODEL_PATH", str(DATA_DIR / "lsa_model.npz")))
LSA_DIM = _env_int("LSA_DIM", 256)
LSA_MAX_FEATURES = _env_int("LSA_MAX_FEATURES", 60_000)
LSA_MIN_DF = _env_int("LSA_MIN_DF", 2)
# Насколько база может вырасти, прежде чем модель считается устаревшей
# и в журнал попадает напоминание переобучить (python index.py train-lsa).
LSA_STALE_RATIO = _env_float("LSA_STALE_RATIO", 0.35)

# --- ONNX: готовая модель без torch ------------------------------------------
# Путь к экспортированной модели (model.onnx) и к папке с токенизатором.
# Так запускаются BGE-M3, USER-bge-m3, ru-en-RoSBERTa без установки torch
# и без полутора гигабайт зависимостей.
ONNX_MODEL_PATH = _env("ONNX_MODEL_PATH", "")
ONNX_TOKENIZER_DIR = _env("ONNX_TOKENIZER_DIR", "")
ONNX_POOLING = _env("ONNX_POOLING", "mean")        # mean | cls
ONNX_MAX_TOKENS = _env_int("ONNX_MAX_TOKENS", 512)
ONNX_THREADS = _env_int("ONNX_THREADS", 0)         # 0 = по числу ядер

# ---------------------------------------------------------------------- LLM --
# local | gigachat | yandex | openai | echo
#
# local — модель на своём сервере: данные не покидают периметр и за токены
# платить не нужно. Адрес подхватывается автоматически, если сервер запущен
# из раздела «Модели».
LLM_PROVIDER = _env("LLM_PROVIDER", "local")
# Запасные провайдеры через запятую. Если основной не ответил — ассистент
# не замолкает, а отвечает следующим, и это видно в журнале и в админке.
# Пустое значение означает «работать только основным».
LLM_FALLBACK = _env("LLM_FALLBACK", "")

# Локальная модель. Пусто = взять адрес у запущенного сервера моделей.
LOCAL_LLM_BASE_URL = _env("LOCAL_LLM_BASE_URL", "")
LOCAL_LLM_MODEL = _env("LOCAL_LLM_MODEL", "")
LOCAL_LLM_API_KEY = _env("LOCAL_LLM_API_KEY", "")
LOCAL_LLM_TIMEOUT = _env_float("LOCAL_LLM_TIMEOUT", 180.0)
# Куда обращаться, если сервер моделей не запущен, но модель поднята
# отдельно — например, LM Studio или Ollama на этой же машине.
LOCAL_LLM_FALLBACK_URL = _env("LOCAL_LLM_FALLBACK_URL", "")
LLM_MODEL = _env("LLM_MODEL", "")
LLM_TEMPERATURE = _env_float("LLM_TEMPERATURE", 0.1)
LLM_MAX_TOKENS = _env_int("LLM_MAX_TOKENS", 1200)

# ------------------------------------------------- очередь запросов к модели --
# Все обращения к модели проходят через очередь. Причина простая: одна
# видеокарта выполняет запросы по очереди в любом случае, но без очереди
# это происходит внутри модели — и тогда десять одновременных вопросов
# превращаются в десять медленных ответов вместо одного быстрого и
# девяти ожидающих. Плюс при нехватке видеопамяти сервер модели просто
# падает, вместо того чтобы подождать.
#
# 1 — по одному запросу за раз. Значение по умолчанию: для одной
# видеокарты правильное почти всегда. Увеличивать имеет смысл, когда
# карт несколько или когда модель облачная и очередь не нужна вовсе
# (тогда ставьте 0 — без ограничения, но учёт очереди сохраняется).
LLM_MAX_CONCURRENT = _env_int("LLM_MAX_CONCURRENT", 1)
# Сколько запросов может стоять в очереди. Одиннадцатый получает честный
# отказ сразу, а не через две минуты ожидания.
LLM_QUEUE_MAX = _env_int("LLM_QUEUE_MAX", 20)
# Сколько ждать своей очереди, секунды. Дольше двух минут ждать ответа
# в чате никто не станет, поэтому отказ лучше молчания.
LLM_QUEUE_TIMEOUT = _env_float("LLM_QUEUE_TIMEOUT", 120.0)
# Через сколько секунд занятое место считается брошенным. Нужно на
# случай, когда процесс убит посреди запроса: иначе место осталось бы
# занятым навсегда. Должно быть заметно больше LOCAL_LLM_TIMEOUT.
LLM_QUEUE_SLOT_TTL = _env_float("LLM_QUEUE_SLOT_TTL", 300.0)
# Очередь общая для всех процессов (админка, бот, фоновые задания) —
# через служебную базу. Выключение оставляет ограничение только внутри
# процесса: тогда фоновой пересчёт приставок может конкурировать с
# живыми вопросами.
LLM_QUEUE_SHARED = _env("LLM_QUEUE_SHARED", "1") == "1"

# Ключи провайдеров
GIGACHAT_AUTH_KEY = _env("GIGACHAT_AUTH_KEY")          # base64(client_id:client_secret)
GIGACHAT_SCOPE = _env("GIGACHAT_SCOPE", "GIGACHAT_API_CORP")
YANDEX_API_KEY = _env("YANDEX_API_KEY")
YANDEX_FOLDER_ID = _env("YANDEX_FOLDER_ID")
OPENAI_API_KEY = _env("OPENAI_API_KEY")
OPENAI_BASE_URL = _env("OPENAI_BASE_URL", "https://api.openai.com/v1")

# ------------------------------------------------------------------- поиск ---
SEARCH_CANDIDATES = _env_int("SEARCH_CANDIDATES", 40)   # сколько берём из каждого канала
SEARCH_TOP_K = _env_int("SEARCH_TOP_K", 6)              # сколько отдаём в LLM
RRF_K = _env_int("RRF_K", 60)                           # константа Reciprocal Rank Fusion
# Приоритет свежести: score = ALPHA*relevance + (1-ALPHA)*0.5^(age/HALF_LIFE)
RECENCY_ALPHA = _env_float("RECENCY_ALPHA", 0.85)
RECENCY_HALF_LIFE_DAYS = _env_float("RECENCY_HALF_LIFE_DAYS", 540.0)
# Матрица векторов больше этого размера (МБ) читается через mmap:
# файл отображается в память, и система подгружает только читаемые
# страницы. Экономит гигабайты RAM у бота и админки, которые векторы
# только читают. 0 — всегда загружать целиком.
VECTORS_MMAP_MB = _env_int("VECTORS_MMAP_MB", 512)
# Сколько минут уточняющий вопрос («а цена?») помнит, о какой модели
# шёл разговор. 0 — выключить наследование темы.
DIALOG_MEMORY_MINUTES = _env_int("DIALOG_MEMORY_MINUTES", 15)
# Бюджет контекста для модели, символов. Шесть фрагментов по 2800
# символов плюс прайс — это больше окна маленькой модели; обрезаем
# честно здесь, а не ошибкой 400 у провайдера.
CONTEXT_MAX_CHARS = _env_int("CONTEXT_MAX_CHARS", 12000)
# Ниже этой релевантности (0…1, покрытие значимых слов вопроса лучшим
# фрагментом) считаем, что ответа в базе нет, и честно об этом говорим.
# Раньше порог сравнивался с ранговым RRF-скором и был недостижим:
# минимум для первого места (0.0139) выше прежнего порога (0.012).
MIN_CONFIDENCE = _env_float("MIN_CONFIDENCE", 0.08)

# ----------------------------------------------------------- переранжирование --
# Кросс-энкодер читает пару «вопрос + фрагмент» целиком и оценивает,
# отвечает ли фрагмент на вопрос. Это точнее любого поиска, но дороже,
# поэтому применяется только к первым RERANKER_TOP_N кандидатам.
#
#   none    — выключено
#   lexical — встроенный лексический реранкер: без моделей и интернета,
#             считает покрытие терминов запроса, близость слов друг к другу
#             и точные совпадения артикулов. Даёт заметную часть эффекта.
#   onnx    — bge-reranker-v2-m3 в формате ONNX, на процессоре, без torch
#   local   — sentence-transformers CrossEncoder (нужен torch)
#   openai  — /v1/rerank совместимый endpoint (vLLM, TEI, Infinity, Jina)
RERANKER_PROVIDER = _env("RERANKER_PROVIDER", "lexical")
RERANKER_MODEL = _env("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_TOP_N = _env_int("RERANKER_TOP_N", 20)     # сколько кандидатов пересортировать
RERANKER_BATCH = _env_int("RERANKER_BATCH", 16)
RERANKER_TIMEOUT = _env_float("RERANKER_TIMEOUT", 20.0)
# Доля веса кросс-энкодера при смешивании с исходным гибридным скором.
# 1.0 — доверять только реранкеру, 0.0 — не учитывать его вовсе.
# 0.8 оставляет реранкеру решающий голос, но не даёт ему полностью
# затереть приоритет свежести и точные попадания BM25 по артикулу.
RERANKER_WEIGHT = _env_float("RERANKER_WEIGHT", 0.8)
RERANKER_CACHE = _env("RERANKER_CACHE", "1") == "1"
RERANKER_ONNX_PATH = _env("RERANKER_ONNX_PATH", "")
RERANKER_TOKENIZER_DIR = _env("RERANKER_TOKENIZER_DIR", "")
RERANKER_MAX_TOKENS = _env_int("RERANKER_MAX_TOKENS", 512)
# Совместимый endpoint для RERANKER_PROVIDER=openai
RERANKER_BASE_URL = _env("RERANKER_BASE_URL", "")
RERANKER_API_KEY = _env("RERANKER_API_KEY", "")

# --------------------------------------------------------------- Telegram ----
TELEGRAM_BOT_TOKEN = _env("TELEGRAM_BOT_TOKEN")
# Пустой список = пускать всех (только для теста!). Иначе — id через запятую.
TELEGRAM_ALLOWED_IDS = [
    int(x) for x in _env("TELEGRAM_ALLOWED_IDS", "").replace(" ", "").split(",") if x.strip().isdigit()
]
TELEGRAM_ADMIN_IDS = [
    int(x) for x in _env("TELEGRAM_ADMIN_IDS", "").replace(" ", "").split(",") if x.strip().isdigit()
]
# Максимальный размер файла-первоисточника для отправки в чат (Telegram лимит 50 МБ).
TELEGRAM_MAX_DOC_MB = _env_int("TELEGRAM_MAX_DOC_MB", 45)
# Прокси для Telegram: socks5://user:pass@host:1080 или http://host:3128
TELEGRAM_PROXY = _env("TELEGRAM_PROXY")
# Отдельный прокси для обращений к моделям (пусто = напрямую)
LLM_PROXY = _env("LLM_PROXY")
# Прокси для краулера сайтов
CRAWLER_PROXY = _env("CRAWLER_PROXY")

# ------------------------------------------------------------------ роли -----
# Роль -> какие разделы базы доступны (по первому уровню папки внутри KB_ROOT).
# Значения по умолчанию — под структуру папок заказчика. Разделяются
# точкой с запятой, потому что в названиях разделов встречается запятая
# («ПРОДУКЦИЯ SPL, AQUASTRONG, МЕГАТРОН»).
_ROLE_SECTIONS_DEFAULT = (
    "admin:*;"
    "sales:ДИЛЕРСКАЯ ПРОДУКЦИЯ|РОЗНИЧНАЯ ПРОДУКЦИЯ|"
    "ПРОДУКЦИЯ SPL, AQUASTRONG, МЕГАТРОН|11УТ|Битрикс24|ПОСТАВЩИКИ;"
    "engineer:ДИЛЕРСКАЯ ПРОДУКЦИЯ|РОЗНИЧНАЯ ПРОДУКЦИЯ|"
    "ПРОДУКЦИЯ SPL, AQUASTRONG, МЕГАТРОН;"
    "dealer:ДИЛЕРСКАЯ ПРОДУКЦИЯ"
)


def _parse_roles(raw: str) -> dict[str, set[str]]:
    """
    Разбирает «роль:раздел|раздел; роль:раздел» в словарь.

    Раньше это был словарь прямо в исходнике, и получалось, что самая
    ответственная настройка системы — кто какие разделы видит — менялась
    только правкой кода на сервере. То есть в панели её не было, в
    справочнике тоже, а ошибка в ней означает либо пустую выдачу, либо
    доступ к дилерским ценам у всех.
    """
    out: dict[str, set[str]] = {}
    for entry in raw.split(";"):
        if ":" not in entry:
            continue
        role, sections = entry.split(":", 1)
        names = {s.strip() for s in sections.split("|") if s.strip()}
        if role.strip() and names:
            out[role.strip()] = names
    return out


ROLE_SECTIONS = _parse_roles(_env("ROLE_SECTIONS", _ROLE_SECTIONS_DEFAULT))
DEFAULT_ROLE = _env("DEFAULT_ROLE", "sales")

# ----------------------------------------------------- сайты и веб-поиск ----
# Домены, которые обходит краулер (по одному в строке в crawl_sources.txt).
CRAWL_SOURCES_FILE = BASE_DIR / "crawl_sources.txt"
CRAWL_MAX_PAGES = _env_int("CRAWL_MAX_PAGES", 500)
CRAWL_DELAY_SECONDS = _env_float("CRAWL_DELAY_SECONDS", 1.5)
CRAWL_RENDER_JS = _env("CRAWL_RENDER_JS", "0") == "1"       # нужен playwright
CRAWL_RESPECT_ROBOTS = _env("CRAWL_RESPECT_ROBOTS", "1") == "1"
# Разрешить обходчику ходить на внутренние адреса. По умолчанию нельзя:
# цель для обхода выбираем не мы, а чужие ссылки и чужие карты сайта, и
# взломанному сайту достаточно поставить ссылку на служебный адрес
# облака, чтобы наш сервер сходил туда сам и положил ответ в базу.
# Включать только для обхода своего внутреннего портала.
CRAWL_ALLOW_PRIVATE = _env("CRAWL_ALLOW_PRIVATE", "0") == "1"
CRAWL_USER_AGENT = _env("CRAWL_USER_AGENT",
                        "KB-Assistant/1.0 (внутренний ассистент; контакт: it@company.ru)")

# Поиск в интернете: none | yandex | searxng | tavily
WEB_SEARCH_PROVIDER = _env("WEB_SEARCH_PROVIDER", "none")
YANDEX_SEARCH_API_KEY = _env("YANDEX_SEARCH_API_KEY")
YANDEX_SEARCH_FOLDER = _env("YANDEX_SEARCH_FOLDER", YANDEX_FOLDER_ID)
SEARXNG_URL = _env("SEARXNG_URL", "http://localhost:8080")
TAVILY_API_KEY = _env("TAVILY_API_KEY")
# Ограничить веб-поиск списком доменов партнёров (пусто = без ограничения)
WEB_SEARCH_DOMAINS = [d for d in _env("WEB_SEARCH_DOMAINS", "").replace(" ", "").split(",") if d]

# ------------------------------------------------------ локальные модели ----
# Куда складывать скачанные веса.
MODELS_DIR = Path(_env("MODELS_DIR", str(DATA_DIR / "models"))).expanduser()
# Зеркало площадки с моделями: из России она недоступна напрямую.
HF_MIRROR = _env("HF_MIRROR", "")
# Порт локального сервера модели (OpenAI-совместимый интерфейс).
LOCAL_MODEL_PORT = _env_int("LOCAL_MODEL_PORT", 8000)
# Какую долю видеопамяти отдавать модели. 0.90 — почти всю; снижайте,
# если на тех же картах работают эмбеддер и распознавание речи.
LOCAL_MODEL_GPU_FRACTION = _env_float("LOCAL_MODEL_GPU_FRACTION", 0.90)
# Максимальная длина контекста при запуске: больше контекст — больше памяти.
LOCAL_MODEL_CONTEXT = _env_int("LOCAL_MODEL_CONTEXT", 32768)
# Поднимать сервер модели автоматически при старте ассистента.
LOCAL_MODEL_AUTOSTART = _env("LOCAL_MODEL_AUTOSTART", "0") == "1"
LOCAL_MODEL_ID = _env("LOCAL_MODEL_ID", "")

# --------------------------------------------------- метрики и статистика ---
# Как часто снимать показания сервера, секунд.
METRICS_INTERVAL_SECONDS = _env_int("METRICS_INTERVAL_SECONDS", 30)
# Сколько суток хранить историю показаний.
METRICS_KEEP_DAYS = _env_int("METRICS_KEEP_DAYS", 30)
METRICS_ENABLED = _env("METRICS_ENABLED", "1") == "1"
# Стоимость токенов для подсчёта расходов, рублей за миллион.
COST_INPUT_PER_MTOK = _env_float("COST_INPUT_PER_MTOK", 650.0)
COST_OUTPUT_PER_MTOK = _env_float("COST_OUTPUT_PER_MTOK", 650.0)

# ---------------------------------------------------------- логирование -----
# Куда и насколько подробно писать. Уровни: TRACE, DEBUG, INFO, WARNING,
# ERROR, CRITICAL. Отдельный уровень для подсистемы задаётся переменной
# LOG_LEVEL_<ИМЯ>, например LOG_LEVEL_SEARCH=TRACE.
LOG_DIR = Path(_env("LOG_DIR", str(BASE_DIR / "logs"))).expanduser()
LOG_LEVEL_CONSOLE = _env("LOG_LEVEL_CONSOLE", "INFO")
LOG_LEVEL_FILE = _env("LOG_LEVEL_FILE", "DEBUG")
LOG_FORMAT = _env("LOG_FORMAT", "human")            # human | json
LOG_TO_CONSOLE = _env("LOG_TO_CONSOLE", "1") == "1"
LOG_TO_FILE = _env("LOG_TO_FILE", "1") == "1"
LOG_TO_DB = _env("LOG_TO_DB", "1") == "1"
LOG_MAX_MB = _env_int("LOG_MAX_MB", 50)
LOG_BACKUPS = _env_int("LOG_BACKUPS", 5)
LOG_DB_KEEP = _env_int("LOG_DB_KEEP", 5000)
# Вырезать из журнала токены, ключи, телефоны, почту и адреса.
LOG_MASK_SENSITIVE = _env("LOG_MASK_SENSITIVE", "1") == "1"
# Записывать полный текст вопросов и расшифровок. Осторожно: это
# персональные данные, храните ограниченно и с доступом по праву.
LOG_PAYLOADS = _env("LOG_PAYLOADS", "0") == "1"

# ----------------------------------------------------------------- голос ----
# Распознавание голосовых сообщений: none | faster-whisper | whisper | gigaam
#                                     | yandex | sber
VOICE_STT_PROVIDER = _env("VOICE_STT_PROVIDER", "none")
VOICE_STT_MODEL = _env("VOICE_STT_MODEL", "large-v3-turbo")
# Синтез речи: none | silero | piper | yandex | sber | openai
VOICE_TTS_PROVIDER = _env("VOICE_TTS_PROVIDER", "none")
VOICE_TTS_MODEL = _env("VOICE_TTS_MODEL", "v4_ru")
VOICE_TTS_SPEAKER = _env("VOICE_TTS_SPEAKER", "baya")
VOICE_TTS_SAMPLE_RATE = _env_int("VOICE_TTS_SAMPLE_RATE", 24000)
# Отвечать голосом: never | on_voice | always
VOICE_REPLY_MODE = _env("VOICE_REPLY_MODE", "on_voice")
# Максимальная длина озвучиваемого текста, символов.
VOICE_MAX_CHARS = _env_int("VOICE_MAX_CHARS", 900)
# Папка с образцами голосов для клонирования.
VOICE_PROFILES_DIR = Path(_env("VOICE_PROFILES_DIR", str(DATA_DIR / "voices"))).expanduser()
# Клонирование голоса по образцу: none | xtts | f5 | openvoice
VOICE_CLONE_PROVIDER = _env("VOICE_CLONE_PROVIDER", "none")
# Подтверждение, что на клонирование получено письменное согласие.
VOICE_CLONE_CONSENT = _env("VOICE_CLONE_CONSENT", "0") == "1"

# ------------------------------------------------------------ телефония -----
# Подключение к АТС: none | ari (Asterisk REST Interface) | sip (регистрация
# как обычный SIP-аппарат через pjsua2)
SIP_MODE = _env("SIP_MODE", "none")
SIP_SERVER = _env("SIP_SERVER", "")
SIP_PORT = _env_int("SIP_PORT", 5060)
SIP_USER = _env("SIP_USER", "")
SIP_PASSWORD = _env("SIP_PASSWORD", "")
SIP_TRANSPORT = _env("SIP_TRANSPORT", "udp")        # udp | tcp | tls
SIP_EXTENSION = _env("SIP_EXTENSION", "")
ARI_URL = _env("ARI_URL", "http://127.0.0.1:8088")
ARI_USER = _env("ARI_USER", "")
ARI_PASSWORD = _env("ARI_PASSWORD", "")
ARI_APP = _env("ARI_APP", "kb-assistant")
# Адрес приёма звука от станции. По умолчанию только локальный: у этого
# порта нет никакой авторизации, и любой, кто до него дотянулся, получал
# ответы по базе знаний голосом. Если станция на другой машине —
# поставьте её адрес, а не 0.0.0.0.
AUDIOSOCKET_HOST = _env("AUDIOSOCKET_HOST", "127.0.0.1")

# Кто звонит и что ему можно. Формат: номер:роль через запятую.
# У телефона нет входа по паролю, поэтому опознание только по номеру.
SIP_KNOWN_CALLERS = _env("SIP_KNOWN_CALLERS", "")
# Роль для всех остальных. Намеренно отдельная от DEFAULT_ROLE: то, что
# можно сотруднику в Telegram, не обязано быть можно любому дозвонившемуся.
SIP_GUEST_ROLE = _env("SIP_GUEST_ROLE", "guest")
# Принимать звонки только с известных номеров.
SIP_ONLY_KNOWN_CALLERS = _env("SIP_ONLY_KNOWN_CALLERS", "0") == "1"
# Ограничения на канал: иначе один скрипт выжигает бюджет на модель.
SIP_MAX_CALLS_PER_HOUR = _env_int("SIP_MAX_CALLS_PER_HOUR", 20)
SIP_MAX_TURNS = _env_int("SIP_MAX_TURNS", 15)
AUDIOSOCKET_PORT = _env_int("AUDIOSOCKET_PORT", 8090)
SIP_GREETING = _env("SIP_GREETING", "Здравствуйте! Задайте вопрос по базе знаний "
                                    "после сигнала.")
SIP_MAX_CALL_SECONDS = _env_int("SIP_MAX_CALL_SECONDS", 600)
SIP_BARGE_IN = _env("SIP_BARGE_IN", "1") == "1"

# --------------------------------------------------- слежение за папкой -----
WATCH_INTERVAL_SECONDS = _env_int("WATCH_INTERVAL_SECONDS", 300)
# Уведомлять администраторов об изменениях структуры каталогов.
WATCH_NOTIFY_STRUCTURE = _env("WATCH_NOTIFY_STRUCTURE", "1") == "1"
# Пересобирать граф связности после переиндексации.
WATCH_REBUILD_GRAPH = _env("WATCH_REBUILD_GRAPH", "0") == "1"

# ------------------------------------------------------------- админка ------
ADMIN_HOST = _env("ADMIN_HOST", "127.0.0.1")
ADMIN_PORT = _env_int("ADMIN_PORT", 8800)
ADMIN_TOKEN = _env("ADMIN_TOKEN", "")     # пусто = без пароля (только localhost!)

DATA_DIR.mkdir(parents=True, exist_ok=True)

# Права: в data/ лежат вопросы сотрудников (персональные данные), роли и
# журнал доступа, в .env — токен бота. На сервере с несколькими учётками
# режим 0755/0644 означает «читает кто угодно», включая копии в backups/.
# Чужие права не трогаем (общая папка команды — законный случай), а свои
# умолчания ужимаем до владельца.
import stat as _stat
for _p, _mode in ((DATA_DIR, 0o700), (BASE_DIR / ".env", 0o600)):
    try:
        if _p.exists() and _p.stat().st_uid == os.getuid()                 and _p.stat().st_mode & 0o077:
            os.chmod(_p, _mode)
    except (OSError, AttributeError):      # windows или чужой файл
        pass
