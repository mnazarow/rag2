"""
Веб-интерфейс администратора.

  python webui.py            — http://127.0.0.1:8800

Разделы:
  Обзор        — что происходит прямо сейчас: нагрузка сервера и видеокарт,
                 вопросы, задержки, расход на модели, состояние базы;
  Конвейер     — живая схема обработки: какой этап работает, где ошибки;
  Модели       — каталог локальных моделей, что поместится в имеющуюся
                 видеопамять, установка и запуск в один щелчок, аналитика
                 использования;
  База знаний  — аудит, граф связности, очередь распознавания, изменения
                 структуры каталогов, запуск переиндексации;
  Голос и АТС  — распознавание, синтез, голоса по образцу, проверка SIP;
  Настройки    — все параметры с пояснением, рекомендацией и примером;
  Журналы      — просмотр с фильтрами, уровни подробности по подсистемам;
  Запросы      — что спрашивали, оценки, пробелы, выверенные ответы;
  Диагностика  — самопроверка всех компонентов.

Написано на стандартной библиотеке: прототип должен запускаться без
установки чего-либо. Графики рисуются на месте, без внешних библиотек.
Для продакшена маршруты один в один переносятся на FastAPI.
"""
from __future__ import annotations

import hmac
import json
import os
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import config
import db
import logging_setup
import settings_schema

log = logging_setup.get("web")

PIPELINE_STAGES = [
    ("scan", "Обход папки", "поиск файлов и проверка изменений по хэшу"),
    ("extract", "Извлечение", "PDF, Word, Excel, презентации, письма, страницы"),
    ("assets", "Нетекстовые", "карточки чертежей, фото и видео, распаковка архивов"),
    ("enrich", "Обогащение", "речь, описания изображений, надписи с чертежей"),
    ("chunk", "Фрагменты", "структурная нарезка и контекстные приставки"),
    ("embed", "Векторизация", "эмбеддинги и запись в векторный индекс"),
    ("prices", "Прайсы", "разбор в таблицу товаров"),
    ("search", "Поиск", "гибридный поиск, переранжирование, свежесть"),
    ("answer", "Ответ", "формулировка моделью и цитирование источников"),
]

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------ телеметрия ----
def ensure_events_table() -> None:
    db.connect().executescript("""
    CREATE TABLE IF NOT EXISTS pipeline_events (
        id INTEGER PRIMARY KEY, ts TEXT, stage TEXT, status TEXT,
        detail TEXT, processed INTEGER, total INTEGER, ms INTEGER);
    CREATE INDEX IF NOT EXISTS idx_events_ts ON pipeline_events(ts);
    """)


def emit(stage: str, status: str, detail: str = "", processed: int = 0,
         total: int = 0, ms: int = 0) -> None:
    try:
        ensure_events_table()
        db.run("INSERT INTO pipeline_events(ts, stage, status, detail, processed, total, ms) "
               "VALUES (?,?,?,?,?,?,?)", (_now(), stage, status, detail, processed, total, ms))
        db.run("DELETE FROM pipeline_events WHERE id < (SELECT MAX(id)-3000 FROM pipeline_events)")
    except Exception:  # noqa: BLE001
        pass


def pipeline_state() -> dict:
    ensure_events_table()
    stages = []
    for key, title, note in PIPELINE_STAGES:
        row = db.q1("SELECT * FROM pipeline_events WHERE stage=? ORDER BY id DESC LIMIT 1", (key,))
        stages.append({"key": key, "title": title, "note": note,
                       "status": row["status"] if row else "idle",
                       "detail": row["detail"] if row else "",
                       "processed": row["processed"] if row else 0,
                       "total": row["total"] if row else 0,
                       "ts": row["ts"] if row else ""})
    recent = [dict(r) for r in db.q(
        "SELECT ts, stage, status, detail FROM pipeline_events ORDER BY id DESC LIMIT 25")]
    return {"stages": stages, "recent": recent, "jobs": running_jobs()}


def extract_errors(limit: int = 300) -> dict:
    """Файлы, из которых не удалось достать текст, — с причинами.

    Раньше эти документы были видны только числом в сводке: чтобы узнать,
    ЧТО именно не проиндексировалось и почему, приходилось лезть в базу
    руками. Список с причинами превращает «где-то 14 ошибок» в план
    работ: битые файлы заменить, запароленные — разблокировать, редкие
    форматы — конвертировать.
    """
    def reason(text: str) -> str:
        first = (text or "неизвестная ошибка").strip().splitlines()[0]
        return first[:160]

    rows = [dict(r) for r in db.q(
        "SELECT rel_path, ext, error, indexed_at FROM documents "
        "WHERE status='error' ORDER BY indexed_at DESC, id DESC LIMIT ?",
        (limit,))]
    total = db.q1("SELECT COUNT(*) n FROM documents WHERE status='error'")["n"]
    by_reason: dict[str, int] = {}
    for r in rows:
        r["reason"] = reason(r.get("error"))
        by_reason[r["reason"]] = by_reason.get(r["reason"], 0) + 1
    top = sorted(by_reason.items(), key=lambda kv: -kv[1])
    return {"total": total, "shown": len(rows), "errors": rows,
            "by_reason": [{"reason": k, "count": v} for k, v in top[:12]]}


# ------------------------------------------------------------ фоновые задачи --
def start_job(name: str, fn, *args, **kwargs) -> str:
    """Запускает длительную операцию в фоне и показывает её ход в интерфейсе."""
    job_id = f"{name}-{int(time.time())}"
    entry = {"id": job_id, "name": name, "status": "running", "started": time.time(),
             "lines": [], "result": None, "error": None}
    with _jobs_lock:
        _jobs[job_id] = entry

    def progress(text: str) -> None:
        with _jobs_lock:
            entry["lines"].append(str(text)[:300])
            entry["lines"] = entry["lines"][-200:]

    def run() -> None:
        try:
            entry["result"] = fn(*args, progress=progress, **kwargs) \
                if "progress" in fn.__code__.co_varnames else fn(*args, **kwargs)
            entry["status"] = "done"
        except Exception as exc:  # noqa: BLE001
            entry["status"] = "error"
            entry["error"] = str(exc)
            log.exception("задача «%s» завершилась ошибкой", name)
        finally:
            entry["finished"] = time.time()

    threading.Thread(target=run, daemon=True).start()
    return job_id


def running_jobs() -> list[dict]:
    with _jobs_lock:
        return [{"id": j["id"], "name": j["name"], "status": j["status"],
                 "seconds": int(time.time() - j["started"]),
                 "tail": j["lines"][-3:], "error": j["error"]}
                for j in sorted(_jobs.values(), key=lambda x: -x["started"])[:8]]


def job_detail(job_id: str) -> dict:
    with _jobs_lock:
        j = _jobs.get(job_id)
        return dict(j) if j else {"error": "задача не найдена"}


# ---------------------------------------------------------------- .env ------
def read_env() -> dict:
    path = config.BASE_DIR / ".env"
    out: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def env_with_secrets() -> dict:
    """Полная картина настроек для проверки связей между ними.

    Ключи хранятся не в .env, а в защищённом файле. Проверка «для
    GigaChat нужен ключ» по одному .env их не видела и запрещала
    переключить провайдера даже с заполненным ключом — сохранение
    отклонялось, и выбор «возвращался» к прежнему.
    """
    full = read_env()
    try:
        import security
        for key, value in security.load_secrets().items():
            if str(value).strip() and not str(full.get(key, "")).strip():
                full[key] = str(value)
    except Exception:  # noqa: BLE001 — нет файла ключей — нет и ключей
        pass
    return full


# Ошибки, которые можно показать целиком: они говорят о действии
# пользователя, а не об устройстве сервера.
# RuntimeError здесь не случайно: наши собственные проверки («веса ещё
# не загружены — нажмите Скачать», «vllm на macOS не работает») бросают
# именно его, и прятать такой текст за номером журнала — значит отнимать
# у человека готовый план действий. Чужие RuntimeError из библиотек
# наружу не попадают: провайдеры и подпроцессы заворачивают их в свои
# типы, а неожиданное падение — это почти всегда не RuntimeError.
_SAFE_ERRORS = ("Busy", "ValueError", "LLMBusy", "Blocked", "OcrError",
                "RuntimeError", "JobError")


def safe_error(exc: BaseException, context: str = "") -> str:
    """
    Что показать в интерфейсе вместо текста исключения.

    Текст исключения сплошь и рядом содержит абсолютные пути, имена
    внутренних хостов, куски ответов провайдеров и версии библиотек. Всё
    это — карта сервера для того, кто до админки дотянулся, и она же
    ничего не говорит человеку, который просто нажал кнопку. Поэтому
    наружу идёт понятная фраза и номер, по которому в журнале лежат
    подробности; в журнал пишется всё как есть.

    Исключения, которые мы бросаем сами и которые описывают действие
    пользователя («уже выполняется», «модель занята»), показываются
    целиком: в них нет ничего внутреннего, а польза прямая.
    """
    name = type(exc).__name__
    if name in _SAFE_ERRORS:
        return str(exc)
    ticket = uuid.uuid4().hex[:8]
    log.error("[%s] %s%s: %s", ticket, context, " — " if context else "",
              exc, exc_info=True)
    return (f"Не удалось выполнить. Подробности в журнале по номеру {ticket} "
            f"(раздел «Журналы», поиск по номеру).")


def _audit_safe_changes(changed: dict, before: dict) -> dict:
    """
    Что записать в журнал действий об изменении настроек.

    Для обычных настроек — было и стало: без этого разбор «когда у нас
    поехал порог отказа» невозможен. Для ключей — только факт замены:
    само значение в журнал попадать не должно ни при каких условиях.
    """
    secret_keys = {s["key"] for s in settings_schema.SETTINGS
                   if s["type"] in SECRET_TYPES}
    out = {}
    for key, value in list(changed.items())[:30]:
        if key in secret_keys:
            out[key] = ["<был задан>" if before.get(key) else "<не был задан>",
                        "<заменён>" if str(value).strip() else "<стёрт>"]
        else:
            out[key] = [before.get(key), value]
    return out


def _behind_https(handler) -> bool:
    """
    Пришёл ли запрос по https.

    Прямо это не видно: сервер слушает голый http, TLS завершает
    обратный прокси. Прокси сообщает об этом заголовком, но заголовок
    клиентский — верить ему можно, только если мы точно знаем, что перед
    нами свой прокси. Отсюда настройка ADMIN_TRUST_PROXY: без неё
    заголовки игнорируются, потому что иначе любой клиент объявляет свой
    запрос защищённым и локальным.
    """
    if not config.ADMIN_TRUST_PROXY:
        return False
    proto = (handler.headers.get("X-Forwarded-Proto") or "").split(",")[0].strip()
    return proto.lower() == "https"


SECRET_TYPES = {"secret"}


def _split_secrets(updates: dict) -> tuple[dict, dict]:
    """
    Разделяет сохраняемое на обычные настройки и ключи.

    Ключи не должны попадать в `.env` — этот файл входит в архив
    обновления и в резервную копию, а файл ключей нет. Раньше admin-форма
    писала всё подряд в одно место, и вынесение ключей в защищённый файл
    отменялось первым же сохранением настроек из браузера.

    Пустое значение секрета означает «не трогать», а не «стереть»: в
    браузер значение не отдаётся вовсе, поэтому пустое поле — это
    нормальное состояние заполненного ключа, а не намерение его удалить.
    Чтобы стереть ключ, есть отдельная кнопка.
    """
    by_key = {s["key"]: s for s in settings_schema.SETTINGS}
    plain, secrets = {}, {}
    for key, value in updates.items():
        spec = by_key.get(key)
        if spec and spec["type"] in SECRET_TYPES:
            if str(value).strip():
                secrets[key] = str(value)
        else:
            plain[key] = value
    return plain, secrets


def write_env(updates: dict) -> None:
    path = config.BASE_DIR / ".env"
    plain, secrets = _split_secrets(updates)
    if secrets:
        import security
        security.save_secrets(secrets)
        for key, value in secrets.items():          # чтобы подействовало сразу
            os.environ[key] = value
        log.warning("ключи сохранены в защищённый файл: %s",
                    ", ".join(sorted(secrets)))
    current = read_env()
    current.update({k: str(v) for k, v in plain.items()})
    # Если ключ когда-то попал в .env, при первом же сохранении убираем
    # его оттуда: он уже лежит в защищённом файле.
    for key in list(current):
        spec = next((s for s in settings_schema.SETTINGS if s["key"] == key), None)
        if spec and spec["type"] in SECRET_TYPES and key in secrets:
            current.pop(key, None)
    known = {s["key"] for s in settings_schema.SETTINGS}
    lines = [f"# Изменено через веб-интерфейс {_now()}",
             "# Ключи и пароли сюда не пишутся — они в отдельном файле "
             f"({config.SECRETS_FILE}).", ""]
    group = None
    for s in settings_schema.SETTINGS:
        if s["group"] != group:
            group = s["group"]
            lines.append(f"\n# ---- {group} ----")
        if s["key"] in current:
            lines.append(f"{s['key']}={current[s['key']]}")
    extra = [k for k in current if k not in known]
    if extra:
        lines.append("\n# ---- прочее ----")
        lines += [f"{k}={current[k]}" for k in extra]
    # Через временный файл: обрезанный .env система читает молча и
    # поднимается на значениях по умолчанию — с другой папкой базы и
    # другим провайдером модели.
    db.atomic_write(path, lambda fh: fh.write(
        ("\n".join(lines) + "\n").encode("utf-8")))
    log.info("настройки сохранены: изменено ключей %d", len(plain))


def audit(action: str, detail: str = "", payload: dict | None = None,
          who: str = "админка") -> None:
    """
    Журнал действий администратора.

    Восстановление из копии, полная переиндексация, смена порога отказа —
    всё это меняет поведение системы для всех сразу. Без записи о том,
    кто и когда это сделал, ни один разбор инцидента не доводится до конца.
    """
    db.connect().execute("""CREATE TABLE IF NOT EXISTS admin_log (
        id INTEGER PRIMARY KEY, ts TEXT, who TEXT, action TEXT,
        detail TEXT, payload_json TEXT)""")
    db.run("INSERT INTO admin_log(ts, who, action, detail, payload_json) VALUES (?,?,?,?,?)",
           (_now(), who, action, detail,
            json.dumps(payload or {}, ensure_ascii=False, default=str)))
    log.warning("действие администратора: %s — %s", action, detail)


def audit_log(limit: int = 200) -> list[dict]:
    db.connect().execute("""CREATE TABLE IF NOT EXISTS admin_log (
        id INTEGER PRIMARY KEY, ts TEXT, who TEXT, action TEXT,
        detail TEXT, payload_json TEXT)""")
    return [dict(r) for r in db.q(
        "SELECT id, ts, who, action, detail FROM admin_log ORDER BY id DESC LIMIT ?",
        (limit,))]


# Настройки, после смены которых модуль надо перечитать прямо сейчас,
# иначе интерфейс покажет старое состояние и введёт в заблуждение.
RELOAD_MAP = {
    "embeddings": ("EMBEDDINGS_PROVIDER", "EMBEDDINGS_MODEL", "LSA_DIM",
                   "LSA_MAX_FEATURES", "LSA_MIN_DF", "LSA_MODEL_PATH",
                   "ONNX_MODEL_PATH", "ONNX_TOKENIZER_DIR", "ONNX_POOLING"),
    "rerank": ("RERANKER_PROVIDER", "RERANKER_MODEL", "RERANKER_ONNX_PATH",
               "RERANKER_TOKENIZER_DIR", "RERANKER_BASE_URL", "RERANKER_API_KEY"),
    "llm": ("LLM_PROVIDER", "LLM_MODEL", "LLM_FALLBACK", "OPENAI_BASE_URL",
            "OPENAI_API_KEY", "GIGACHAT_AUTH_KEY", "GIGACHAT_SCOPE",
            "YANDEX_API_KEY", "YANDEX_FOLDER_ID", "LOCAL_LLM_BASE_URL",
            "LOCAL_LLM_MODEL", "LOCAL_LLM_API_KEY", "LOCAL_LLM_TIMEOUT"),
    # Ограничение одновременных запросов должно действовать сразу. Пока оно
    # вступало бы в силу «при следующем запуске», администратор, снявший
    # нагрузку с карты, видел бы, что ничего не изменилось, и снимал бы её
    # дальше.
    "llm_queue": ("LLM_MAX_CONCURRENT", "LLM_QUEUE_MAX", "LLM_QUEUE_TIMEOUT",
                  "LLM_QUEUE_SLOT_TTL", "LLM_QUEUE_SHARED"),
    "security": ("ADMIN_SESSION_HOURS", "SECRETS_FILE", "SECRETS_CMD",
                 "RATE_LIMIT_PER_USER_HOUR", "RATE_LIMIT_PER_USER_DAY",
                 "RATE_LIMIT_TOTAL_DAY"),
}


# Настройки, которые нельзя перечитывать на ходу во время длительной
# операции: смена провайдера или размерности векторов посреди
# переиндексации даёт следующую пачку другой размерности, задача рвётся
# на середине, а уже записанная часть векторов остаётся в файле.
_UNSAFE_WHILE_BUSY = {
    "EMBEDDINGS_PROVIDER", "EMBEDDINGS_MODEL", "EMBEDDINGS_DIM", "LSA_DIM",
    "LSA_MAX_FEATURES", "LSA_MIN_DF", "ONNX_MODEL_PATH", "VECTOR_BACKEND",
    "QDRANT_URL", "QDRANT_COLLECTION", "DATA_DIR", "KB_ROOT",
}


def busy_with_indexing() -> str:
    """Идёт ли сейчас работа, которой помешает смена настроек."""
    try:
        import jobs
        for job in jobs.recent(20):
            if job["status"] == "running" and \
                    set(jobs.RESOURCES.get(job["kind"], ())) & {"index", "vectors", "model"}:
                return job.get("title") or job["kind"]
    except Exception:  # noqa: BLE001
        pass
    return ""


def _reload_after_settings(changed: dict) -> list[str]:
    """Перечитывает модули, которых коснулась правка, без перезапуска процесса."""
    done = []
    import importlib
    importlib.reload(config)
    for module_name, keys in RELOAD_MAP.items():
        if not any(k in changed for k in keys):
            continue
        try:
            module = importlib.import_module(module_name)
            importlib.reload(module)
            if hasattr(module, "reset"):
                module.reset()
            done.append(module_name)
        except Exception as exc:  # noqa: BLE001 — сообщим, но не уроним сохранение
            log.warning("не удалось перечитать модуль %s: %s", module_name, exc)
    if done:
        log.info("перечитаны модули: %s", ", ".join(done))
    return done


# ------------------------------------------------------------ диагностика ---
def diagnostics() -> list[dict]:
    """Самопроверка: что настроено, что работает, чего не хватает."""
    import shutil
    checks: list[dict] = []

    def add(name: str, ok: bool | None, detail: str, hint: str = "") -> None:
        checks.append({"name": name, "ok": ok, "detail": detail, "hint": hint})

    add("Папка базы знаний", config.KB_ROOT.exists(), str(config.KB_ROOT),
        "Укажите правильный путь в настройке KB_ROOT")
    try:
        docs = db.q1("SELECT COUNT(*) n FROM documents WHERE status='ok'")["n"]
        add("Индекс", docs > 0, f"документов: {docs}",
            "Запустите индексацию: python index.py build")
    except Exception as exc:  # noqa: BLE001
        add("Индекс", False, str(exc), "Проверьте DATA_DIR — он должен быть на локальном диске")
    vectors = len(db.vectors())
    chunks = db.q1("SELECT COUNT(*) n FROM chunks")["n"]
    add("Векторный индекс", vectors >= chunks * 0.95 if chunks else None,
        f"векторов {vectors} при {chunks} фрагментах",
        "Досчитайте: python index.py repair")

    import embeddings as emb_mod
    import search as search_mod
    emb_info = emb_mod.describe()
    add("Смысловой канал поиска", emb_info["ready"] and not emb_info["is_stub"],
        f"{emb_info['provider']}: {emb_info.get('error') or emb_info.get('detail') or '—'}",
        "hashing — заглушка без смысловой близости. Обучите свою модель: "
        "python index.py train-lsa, затем python index.py reembed")
    dense_ok, dense_note = search_mod.dense_ready()
    add("Векторы соответствуют модели", dense_ok, dense_note,
        "Векторы посчитаны другой моделью — смысловой канал отключён. "
        "Пересчитайте: python index.py reembed")
    if config.EMBEDDINGS_PROVIDER == "lsa":
        trained = 0
        try:
            trained = int(emb_mod.get_embedder().model.meta.get("documents", 0))
        except Exception:  # noqa: BLE001 — модель ещё не обучена
            pass
        add("Свежесть смысловой модели",
            None if not trained else chunks <= trained * (1 + config.LSA_STALE_RATIO),
            f"обучена на {trained} фрагментах, сейчас {chunks}" if trained
            else "модель не обучена",
            "База заметно выросла — переобучите: python index.py train-lsa")
    add("Языковая модель", config.LLM_PROVIDER != "echo", config.LLM_PROVIDER,
        "echo — заглушка без модели; выберите провайдера и укажите ключ")

    import rerank as rerank_mod
    rr = rerank_mod.describe()
    add("Переранжирование", rr["ready"] if rr["enabled"] else None,
        f"{rr['provider']}: {rr.get('error') or rr.get('detail') or '—'}",
        "Пересортировка первых двадцати кандидатов повышает шанс, что нужный "
        "фрагмент попадёт в ответ, а не останется на седьмом месте")

    import ocr as ocr_mod
    pending = db.q1("SELECT COUNT(*) n FROM documents "
                    "WHERE needs_ocr=1 AND status='ok'")["n"]
    add("Распознавание сканов",
        True if pending == 0 else config.OCR_PROVIDER != "none",
        f"ждут распознавания: {pending}, провайдер: {config.OCR_PROVIDER}",
        "Сертификаты и декларации почти всегда сканы. Пока они не распознаны, "
        "поиск их не видит. Что доступно: python ocr.py providers")

    import llm as llm_mod
    llm_info = llm_mod.describe()
    add("Модель генерации",
        bool(llm_info.get("ready")) and not llm_info.get("is_stub"),
        f"{llm_info.get('primary')}"
        + (f" → {', '.join(llm_info.get('chain', [])[1:])}"
           if len(llm_info.get("chain", [])) > 1 else "")
        + (f": {llm_info['failed'][0]['error'][:60]}"
           if llm_info.get("failed") and not llm_info.get("ready") else ""),
        "echo — заглушка: ответ склеивается из найденных предложений. "
        "Раздел «Модели»: запустить локальную модель или выбрать облако.")

    import llm_queue
    qstat = llm_queue.stats(24)
    qnow = llm_queue.status()
    refused = qstat.get("refused", 0) + qstat.get("timeout", 0)
    q_total = qstat.get("total", 0)
    add("Очередь к модели",
        not qstat.get("error") and (q_total == 0 or refused <= max(1, q_total * 0.05)),
        (f"одновременно {qnow['limit'] or 'без ограничения'}, "
         f"сейчас выполняется {qnow.get('running', 0)}, ждут {qnow.get('waiting', 0)}; "
         f"за сутки запросов {q_total}, отказов {refused}, "
         f"ожидание в среднем {qstat.get('wait_avg_ms', 0)} мс")
        + (f"; ошибка: {qstat['error'][:60]}" if qstat.get("error") else ""),
        "Отказы означают, что модель не успевает за вопросами. Либо поднимите "
        "LLM_MAX_CONCURRENT, если есть запас видеопамяти, либо перенесите "
        "фоновую обработку базы на ночь: python llm_queue.py stats")

    import security
    add("Вход в админку", security.accounts_enabled() or bool(config.ADMIN_TOKEN),
        f"учётных записей: {len(security.list_users())}"
        if security.accounts_enabled()
        else ("общий токен" if config.ADMIN_TOKEN else "только локальный адрес"),
        "Заведите учётные записи в разделе «Безопасность»: роли различают "
        "«посмотреть» и «восстановить индекс из копии».")
    secrets = security.secrets_health()
    add("Хранение ключей", secrets["ok"],
        "; ".join(secrets["problems"])[:90] if secrets["problems"]
        else "ключей в открытом виде нет",
        "Раздел «Безопасность» → «Вынести ключи в защищённый файл».")

    import schedule as schedule_mod
    sched = schedule_mod.status()
    add("Регулярные задания", sched["installed"] == sched["total"],
        f"настроено {sched['installed']} из {sched['total']}",
        "python schedule.py install — копии, проверки, очистка, "
        "учебное восстановление")

    import alerts as alerts_mod
    active_alerts = alerts_mod.active()
    add("Открытые проблемы", not active_alerts,
        f"{len(active_alerts)}: " + "; ".join(a["title"] for a in active_alerts[:3])
        if active_alerts else "нет",
        "Раздел «Безопасность»: там же написано, что делать с каждой.")

    add("Срок хранения вопросов", bool(config.RETENTION_QUERIES_DAYS),
        f"{config.RETENTION_QUERIES_DAYS} дней" if config.RETENTION_QUERIES_DAYS
        else "не задан — вопросы копятся бессрочно",
        "Тексты вопросов — персональные данные. Задайте RETENTION_QUERIES_DAYS.")

    import backup as backup_mod
    binfo = backup_mod.status()
    add("Резервные копии индекса", (not binfo["stale"]) if binfo["count"] else False,
        f"копий {binfo['count']}" + (f", последняя {binfo['age_hours']} ч назад"
                                     if binfo.get("age_hours") is not None else ""),
        "Выверенные ответы и обучающие пары пересборкой не восстанавливаются: "
        "python backup.py create, затем python backup.py schedule")
    add("Расписание копий", binfo["installed"],
        binfo["schedule"] if binfo["installed"] else "не настроено",
        "python backup.py schedule")

    add("Telegram", bool(config.TELEGRAM_BOT_TOKEN), "токен задан"
        if config.TELEGRAM_BOT_TOKEN else "токен не задан", "Получите токен у @BotFather")
    add("Белый список сотрудников", bool(config.TELEGRAM_ALLOWED_IDS),
        f"{len(config.TELEGRAM_ALLOWED_IDS)} человек" if config.TELEGRAM_ALLOWED_IDS
        else "пусто — бот пустит любого", "Заполните TELEGRAM_ALLOWED_IDS")

    for tool, why in (("ffmpeg", "видео и голос"), ("pdftotext", "разбор PDF"),
                      ("bsdtar", "распаковка архивов"), ("nvidia-smi", "видеокарты")):
        found = shutil.which(tool)
        add(f"Утилита {tool}", bool(found), found or "не найдена",
            f"Нужна для: {why}")

    for module, why in (("numpy", "векторы"), ("openpyxl", "прайсы"),
                        ("httpx", "обращения к моделям"), ("docx", "документы Word"),
                        ("fitz", "быстрый разбор PDF"), ("trafilatura", "веб-страницы"),
                        ("aiogram", "Telegram"), ("sentence_transformers", "локальные эмбеддинги"),
                        ("faster_whisper", "распознавание речи"), ("torch", "локальные модели"),
                        ("vllm", "запуск локальной языковой модели"),
                        ("psutil", "точные метрики сервера")):
        try:
            __import__(module)
            add(f"Библиотека {module}", True, "установлена", "")
        except ImportError:
            add(f"Библиотека {module}", None, "не установлена", f"Нужна для: {why}")

    import models as models_mod
    hw = models_mod.hardware()
    add("Видеопамять", bool(hw["vram_total_gb"]),
        f"{hw['vram_total_gb']} ГБ на {len(hw['gpus'])} картах" if hw["gpus"]
        else "карт не найдено", "Без видеокарты локальные модели будут очень медленными")
    add("Свободно на диске", hw["disk_free_gb"] > 20, f"{hw['disk_free_gb']} ГБ",
        "Для весов моделей и индекса нужен запас")

    state = models_mod.status()
    add("Сервер локальной модели", state.get("running") if state.get("pid") else None,
        f"{state.get('model','—')} на {state.get('base_url','—')}" if state.get("running")
        else "не запущен", "Раздел «Модели» → запустить")

    if config.SIP_MODE != "none":
        import sip
        result = sip.check()
        add("Телефония", result["ok"], config.SIP_MODE,
            "; ".join(result["problems"]) if result["problems"] else "")
    return checks


# ------------------------------------------------------------------ HTML ----
PAGE = r"""<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ассистент базы знаний · администрирование</title><style>
*{box-sizing:border-box}
body{margin:0;background:#0f1116;color:#e6e8ec;font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
header{display:flex;align-items:center;gap:14px;padding:12px 20px;background:#171a21;
 border-bottom:1px solid #262a33;position:sticky;top:0;z-index:20;flex-wrap:wrap}
header h1{font-size:14px;margin:0;font-weight:600;white-space:nowrap}
nav{display:flex;gap:3px;flex-wrap:wrap}
nav button{background:transparent;border:1px solid transparent;color:#8b93a3;padding:6px 12px;
 border-radius:8px;cursor:pointer;font-size:13px}
nav button.on{background:#232833;color:#fff;border-color:#2f3542}
#health{margin-left:auto;font-size:12px;color:#8b93a3}
main{padding:20px;max-width:1400px;margin:0 auto}
section{display:none} section.on{display:block}
h2{font-size:15px;margin:24px 0 12px;color:#fff} h2:first-child{margin-top:0}
h3{font-size:13px;margin:16px 0 8px;color:#c8cede;text-transform:uppercase;letter-spacing:.05em}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(158px,1fr));gap:10px}
.card{background:#171a21;border:1px solid #262a33;border-radius:10px;padding:11px 13px}
.card .v{font-size:21px;font-weight:600} .card .k{font-size:12px;color:#8b93a3;margin-top:2px}
.card.warn{border-color:#5a4520} .card.bad{border-color:#5a2626}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:900px){.grid2{grid-template-columns:1fr}}
.panel{background:#171a21;border:1px solid #262a33;border-radius:10px;padding:14px 16px}
.flow{display:flex;flex-wrap:wrap;gap:9px;margin:6px 0}
.stage{position:relative;flex:1 1 165px;min-width:158px;background:#171a21;
 border:1px solid #262a33;border-radius:10px;padding:10px 12px}
.stage.running{border-color:#4E79A7;box-shadow:0 0 0 1px #4E79A7 inset}
.stage.ok{border-color:#2e5b3f} .stage.error{border-color:#E15759}
.stage .t{font-weight:600;font-size:13px} .stage .n{font-size:11px;color:#767e8d;margin-top:2px}
.stage .d{font-size:12px;color:#9aa3b2;margin-top:6px;min-height:16px}
.bar{height:3px;background:#232833;border-radius:2px;margin-top:7px;overflow:hidden}
.bar i{display:block;height:100%;background:#4E79A7;width:0;transition:width .4s}
.dotst{position:absolute;top:10px;right:11px;width:8px;height:8px;border-radius:50%;background:#39404d}
.stage.running .dotst{background:#4E79A7;animation:pulse 1.1s infinite}
.stage.ok .dotst{background:#59A14F} .stage.error .dotst{background:#E15759}
@keyframes pulse{50%{opacity:.25}}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:6px 9px;border-bottom:1px solid #232833;vertical-align:top}
th{color:#8b93a3;font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.05em}
tr:hover td{background:#1a1e26}
input,select,textarea{background:#0f1116;border:1px solid #2c313c;color:#e6e8ec;
 border-radius:7px;padding:7px 10px;font-size:13px;font-family:inherit}
input[type=text],input[type=password],select,textarea{width:290px}
input[type=checkbox]{width:auto}
button.act{background:#2f6fb0;border:0;color:#fff;padding:8px 16px;border-radius:8px;
 cursor:pointer;font-size:13px}
button.act.sec{background:#232833;color:#c8cede}
button.act.warn{background:#8a5a1a} button.act.bad{background:#8a2e2e}
button.act:disabled{opacity:.45;cursor:default}
.muted{color:#8b93a3} .warn{color:#F28E2B} .bad{color:#E15759} .good{color:#59A14F}
pre{background:#0c0e13;border:1px solid #232833;border-radius:8px;padding:11px;overflow:auto;
 font-size:12px;max-height:520px;white-space:pre-wrap;line-height:1.45}
.toolbar{display:flex;gap:9px;align-items:center;margin:12px 0;flex-wrap:wrap}
.grp{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:#8b93a3;
 margin:20px 0 8px;border-bottom:1px solid #232833;padding-bottom:5px}
.setting{background:#171a21;border:1px solid #262a33;border-radius:10px;padding:12px 14px;margin-bottom:9px}
.setting .row{display:flex;gap:14px;align-items:flex-start;flex-wrap:wrap}
.setting .lbl{flex:1;min-width:320px}
.setting .key{font-family:ui-monospace,Menlo,monospace;font-size:11px;color:#6f7889}
.setting h4{margin:0 0 2px;font-size:14px;font-weight:600}
.setting p{margin:4px 0;color:#9aa3b2;font-size:12.5px}
.setting .rec{color:#c9b57b} .setting .ex{font-family:ui-monospace,Menlo,monospace;color:#6f7889;font-size:12px}
.setting.changed{border-color:#3d5a80}
.setting .mark{font-size:11px;padding:1px 6px;border-radius:20px;margin-left:6px;
  background:#22304a;color:#8fb0dd;vertical-align:middle}
.setting .mark.def{background:#20242c;color:#6f7889}
.setting .mark.key{background:#3a2a20;color:#d9a06a}
.setting .mark.ok{background:#1e3326;color:#7fc48f}
.setting .mark.warn{background:#3a3220;color:#d9c46a}
.setting .side{display:flex;flex-direction:column;gap:5px;min-width:230px}
.setting .side .tiny{font-size:11px;color:#6f7889}
.setting .side button{font-size:11px;padding:3px 8px}
.grpnav{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0 14px}
.grpnav a{font-size:12px;padding:4px 10px;border-radius:20px;background:#171a21;
  border:1px solid #262a33;color:#9aa3b2;text-decoration:none;cursor:pointer}
.grpnav a:hover{border-color:#3d5a80;color:#cfd6e2}
.grpnav a b{color:#8fb0dd;font-weight:600}
.grpnav a .n{color:#6f7889}
.chip{display:inline-block;background:#252a34;border-radius:4px;padding:1px 7px;margin:2px 3px 0 0;font-size:11px}
.chip.ok{background:#1e3a29;color:#8fd6a8} .chip.no{background:#3a2020;color:#e59a9a}
.chip.star{background:#3a3320;color:#e2cd8a}
svg{display:block;width:100%}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:#8b93a3;margin-top:6px}
.legend i{display:inline-block;width:10px;height:3px;margin-right:5px;vertical-align:middle}
.modelrow{display:flex;gap:12px;align-items:flex-start;padding:11px;border-bottom:1px solid #232833}
.modelrow:hover{background:#1a1e26}
.modelrow .info{flex:1}
.filterbar input{width:220px}
</style></head><body>
<header><h1>Ассистент базы знаний</h1>
<nav>
 <button data-t="quick" class="on">Быстрый старт</button>
 <button data-t="overview">Обзор</button>
 <button data-t="flow">Конвейер</button>
 <button data-t="models">Модели</button>
 <button data-t="kb">База знаний</button>
 <button data-t="organize">Порядок в базе</button>
 <button data-t="search">Качество поиска</button>
 <button data-t="eval">Контрольные вопросы</button>
 <button data-t="ocr">Сканы</button>
 <button data-t="backup">Копии</button>
 <button data-t="graph">Граф</button>
 <button data-t="voice">Голос и АТС</button>
 <button data-t="settings">Настройки</button>
 <button data-t="logs">Журналы</button>
 <button data-t="queries">Запросы</button>
 <button data-t="analytics">Аналитика</button>
 <button data-t="users">Сотрудники</button>
 <button data-t="telegram">Телеграм</button>
 <button data-t="safety">Безопасность</button>
 <button data-t="diag">Диагностика</button>
</nav><div id="health"></div></header>
<main>

<section id="overview">
  <h2>Состояние системы</h2>
  <div class="cards" id="ovCards"></div>
  <div class="grid2" style="margin-top:16px">
    <div class="panel"><h3>Нагрузка сервера, последние сутки</h3>
      <div id="chartServer"></div>
      <div class="legend"><span><i style="background:#4E79A7"></i>процессор</span>
        <span><i style="background:#F28E2B"></i>память</span>
        <span><i style="background:#59A14F"></i>видеокарта</span></div></div>
    <div class="panel"><h3>Видеопамять и температура</h3>
      <div id="chartGpu"></div>
      <div class="legend" id="gpuLegend"></div></div>
  </div>
  <div class="grid2" style="margin-top:14px">
    <div class="panel"><h3>Вопросы по дням</h3><div id="chartQueries"></div></div>
    <div class="panel"><h3>Задержка ответа по этапам</h3><div id="chartStages"></div></div>
  </div>
  <div class="grid2" style="margin-top:14px">
    <div class="panel"><h3>Откуда берутся ответы</h3><div id="chartSections"></div></div>
    <div class="panel"><h3>Расход на модели</h3><table id="tblModels"></table></div>
  </div>
</section>

<section id="flow">
  <h2>Конвейер обработки</h2>
  <div class="flow" id="stages"></div>
  <h2>Фоновые задачи</h2>
  <table id="jobs"><thead><tr><th>Задача</th><th style="width:120px">Состояние</th>
    <th style="width:80px">Время</th><th>Ход работы</th>
    <th style="width:110px"></th></tr></thead><tbody></tbody></table>
  <p class="muted">Очередь переживает перезапуск: задача, прерванная посередине,
    помечается как оборванная и ставится заново одной кнопкой. Две задачи,
    которые пишут в одно и то же — например индексация и пересчёт векторов, —
    одновременно не выполняются: вторая получает отказ, а не портит индекс.</p>
  <h2>Последние события</h2>
  <table id="events"><thead><tr><th>Время</th><th>Этап</th><th>Статус</th>
    <th>Подробности</th></tr></thead><tbody></tbody></table>

  <h2>Ошибки извлечения</h2>
  <p class="muted">Файлы, из которых не удалось достать текст: они не попали в
    индекс, и ассистент про них не знает. Причина подсказывает, что делать:
    битый файл — заменить у поставщика, запароленный — снять пароль,
    редкий формат — конвертировать. После исправления файла индексация
    подхватит его сама.</p>
  <div class="panel" id="exErrSummary">Загружаю…</div>
  <table id="exErrs"><thead><tr><th>Файл</th>
    <th style="width:45%">Причина</th></tr></thead><tbody></tbody></table>
</section>

<section id="quick" class="on">
  <h2>Быстрый старт</h2>
  <p class="muted">Те же шаги, что напечатал установщик, — но видно, какие из
    них уже сделаны. Это и есть главное: третий и четвёртый шаги внешне
    неразличимы, обе команды отрабатывают без ошибок, а поиск при
    пропущенном третьем находит только точные слова из документа.
    Обнаруживается это обычно через неделю, по жалобе сотрудника.</p>
  <div class="panel" id="qsSummary">Загружаю…</div>
  <div id="qsSteps"></div>
</section>

<section id="models">
  <h2>Кто отвечает на вопросы</h2>
  <p class="muted">Локальная модель работает на ваших картах: вопросы сотрудников
    и содержимое документов не покидают сервер, за токены платить не нужно.
    Облако запускается сразу и не требует видеокарт, но каждый вопрос уходит
    наружу и стоит денег. Можно выбрать основным одно, а запасным другое:
    если основной не ответил, ассистент не замолкает, а отвечает следующим —
    и это видно в журнале.</p>
  <div class="panel" id="llmPanel">Загружаю…</div>
  <div class="toolbar">
    <label class="muted">основной
      <select id="llmPrimary">
        <option value="local">local — своя модель</option>
        <option value="gigachat">gigachat — облако Сбера</option>
        <option value="yandex">yandex — облако Яндекса</option>
        <option value="openai">openai — совместимый сервис</option>
        <option value="echo">echo — заглушка без модели</option>
      </select></label>
    <label class="muted">запасной
      <select id="llmFallback">
        <option value="">нет</option>
        <option value="local">local</option>
        <option value="gigachat">gigachat</option>
        <option value="yandex">yandex</option>
        <option value="openai">openai</option>
      </select></label>
    <button class="act" onclick="switchLlm()">Применить</button>
    <button class="act sec" onclick="probeLlm()">Проверить живым запросом</button>
  </div>
  <div class="toolbar">
    <label class="muted">загруженная модель
      <select id="llmServe" style="min-width:260px"></select></label>
    <button class="act" onclick="serveAndUse()">Запустить и использовать</button>
    <span class="muted">поднимет сервер модели и сделает её основной
      (провайдер local); загрузка весов в память — несколько минут</span>
  </div>
  <div id="llmProbe"></div>

  <h2>Очередь запросов к модели</h2>
  <p class="muted">Видеокарта выполняет запросы по одному в любом случае —
    вопрос лишь в том, где образуется очередь. Если отправить модели десять
    вопросов разом, она разделит память на десять частей и ответит на каждый
    в десять раз медленнее: быстрого ответа не получит никто, а при нехватке
    видеопамяти сервер модели упадёт и не ответит вообще. С очередью первый
    вопрос обрабатывается на полной скорости, остальные ждут — суммарное
    время то же, но первый ответ приходит сразу. Поэтому по умолчанию стоит
    один одновременный запрос. Через очередь проходят все обращения: вопросы
    из чата, звонки, распознавание сканов и фоновая обработка базы.</p>
  <p class="muted">Живой вопрос всегда обгоняет фоновую работу. Это важнее,
    чем кажется: пересчёт смысловых приставок ставит в очередь десятки тысяч
    заданий, и при честном порядке «кто встал, тот и первый» сотрудник ждал
    бы окончания всего прогона.</p>
  <div class="panel" id="queuePanel">Загружаю…</div>
  <div class="toolbar">
    <button class="act sec" onclick="loadQueue()">Обновить</button>
    <button class="act sec" onclick="clearQueue()">Снять зависшие места</button>
    <span class="muted">Снимать нужно только после аварийной остановки:
      обычно место освобождается само по сроку.</span>
  </div>
  <div id="setQueue"></div>

  <h2>Железо</h2>
  <div class="cards" id="hwCards"></div>
  <h2>Сервер модели</h2>
  <div class="toolbar" id="serverBar"></div>
  <div id="mProgress"></div>

  <h2>Журнал действий с моделями</h2>
  <p class="muted">Загрузки, запуски, остановки и их ошибки — только про
    модели, без шума остального журнала. Обновляется сам, пока открыт раздел.</p>
  <div id="mActions" class="panel">—</div>
  <h2>Каталог моделей</h2>
  <div class="toolbar filterbar">
    <input id="mFilter" placeholder="фильтр по названию">
    <select id="mKind">
      <option value="">все назначения</option><option value="llm">языковые</option>
      <option value="embedding">эмбеддинги</option><option value="reranker">переранжирование</option>
      <option value="vision">зрение</option><option value="asr">распознавание речи</option>
      <option value="tts">синтез речи</option></select>
    <label class="muted"><input type="checkbox" id="mFits" checked> только те, что поместятся</label>
  </div>
  <div class="panel" id="mList" style="padding:0"></div>
  <h2>Использование моделей</h2>
  <table id="mUsage"></table>
</section>

<section id="kb">
  <h2>Управление индексом</h2>
  <div class="toolbar">
    <button class="act" onclick="job('reindex')">Переиндексировать изменённое</button>
    <button class="act sec" onclick="job('reindex_full')">Полная переиндексация</button>
    <button class="act sec" onclick="job('repair')">Досчитать векторы</button>
    <button class="act sec" onclick="job('graph')">Построить граф</button>
    <button class="act sec" onclick="runAudit()">Проверить базу</button>
    <button class="act sec" onclick="job('structure')">Сверить структуру папок</button>
  </div>
  <div class="cards" id="kbCards"></div>
  <div class="panel" id="kbHint" style="margin-top:12px"></div>

  <h2>Контекст фрагментов через модель</h2>
  <p class="muted">К каждому фрагменту дописывается фраза о том, из какого
    документа он и о чём речь. Дешёвый вариант из пути к файлу работает
    всегда; здесь — полный, где фразу пишет модель, читая фрагмент вместе
    с началом документа. Стоит денег: одно обращение на фрагмент. Поэтому
    сначала оценка, потом проба на сотне, и только потом вся база.</p>
  <div class="panel" id="ctxPanel">Загружаю…</div>
  <div class="toolbar">
    <button class="act sec" onclick="job('contextual',{limit:parseInt($('ctxLimit').value)||100})">
      Обработать</button>
    <label class="muted">фрагментов <input id="ctxLimit" value="100" style="width:80px"></label>
  </div>

  <h2>Сайты производителей</h2>
  <p class="muted">По нескольким брендам локальной документации в базе почти
    нет — только ссылки на порталы. Прежде чем запускать обход, стоит
    запросить у поставщиков официальные выгрузки: это надёжнее и снимает
    правовые вопросы. Обход — запасной путь.</p>
  <div class="panel" id="crawlPanel">Загружаю…</div>
  <div class="toolbar">
    <textarea id="crawlList" rows="4" style="width:520px"
      placeholder="https://wilo.ru/&#10;https://imp-pump.ru/documents/"></textarea>
    <div>
      <button class="act sec" onclick="saveSources()">Сохранить список</button>
      <button class="act" onclick="job('crawl',{limit:parseInt($('crawlLimit').value)||null})">
        Обойти сайты</button>
      <div class="muted" style="margin-top:6px">страниц с сайта, максимум
        <input id="crawlLimit" value="200" style="width:70px"></div>
    </div>
  </div>

  <h2>Расшифровка видео и аудио</h2>
  <p class="muted">Расшифровка режется на куски с таймкодами, поэтому бот
    отвечает ссылкой на конкретную минуту записи, а не на файл целиком.
    Прогон однократный, прерывать безопасно.</p>
  <div class="panel" id="mediaPanel">Загружаю…</div>
  <div class="toolbar">
    <button class="act" onclick="job('media',{limit:parseInt($('mediaLimit').value)||null})">
      Расшифровать</button>
    <label class="muted">файлов <input id="mediaLimit" value="" placeholder="все" style="width:70px"></label>
  </div>

  <h2>Изменения структуры каталогов</h2>
  <table id="structure"><thead><tr><th>Когда</th><th>Что</th><th>Папка</th>
    <th>Подробности</th></tr></thead><tbody></tbody></table>
  <h2>Аудит</h2>
  <pre id="auditOut">Нажмите «Проверить базу».</pre>
</section>

<!-- ═══════════════════════════════════ качество поиска ═══════════════════ -->
<section id="organize">
  <h2>Порядок в базе знаний</h2>
  <p class="muted">Структура папок — это метаданные: раздел определяет права
    доступа, бренд попадает в карточку каждого фрагмента, папка типа задаёт
    маршрут документа, дата в имени управляет вытеснением версий. Здесь видно,
    где база отступает от этих правил, и собирается план уборки. Сама папка
    базы знаний подключена только на чтение — план выполняет человек.
    Пошаговый алгоритм — в документации, раздел 25.</p>
  <div class="panel" id="orgProgress">Загружаю…</div>

  <div class="toolbar">
    <button class="act" onclick="loadOrganize()">Проверить снова</button>
    <a class="act sec" href="/api/organize/plan" download>Скачать план уборки (.sh)</a>
    <a class="act sec" href="/api/organize/csv" download>Все находки таблицей (.csv)</a>
    <button class="act sec" onclick="job('reindex')">Переиндексировать изменённое</button>
    <button class="act sec" onclick="job('structure')">Сверить структуру папок</button>
  </div>
  <p class="muted">План уборки — черновик команд переименования, каждая
    закомментирована: раскомментируйте те, с которыми согласны, и выполните
    на машине с базой. После уборки нажмите «Переиндексировать изменённое».</p>

  <h2>С чего начать: о ком спрашивают</h2>
  <div id="orgAsked" class="panel">—</div>

  <h2>Бренды-двойники</h2>
  <p class="muted">Одно имя в двух написаниях — для поиска это два разных
    бренда, и вопрос находит только половину документов. Слейте папки в одну.</p>
  <div id="orgTwins">—</div>

  <h2>Файлы без типа</h2>
  <p class="muted">Тип не определился из пути — файл лежит вне папок типов.
    Подсказка выведена из имени файла.</p>
  <div id="orgUntyped">—</div>

  <h2>Прайсы и сертификаты без даты</h2>
  <p class="muted">Без даты в имени не работает вытеснение версий: старый и
    новый прайс равноправны. Подсказка — из времени изменения файла,
    сверьте её с содержимым.</p>
  <div id="orgUndated">—</div>

  <h2>Имена, которые врут со временем</h2>
  <div id="orgBadNames">—</div>

  <h2>Точные дубли</h2>
  <div id="orgDups">—</div>

  <h2>Пробелы покрытия</h2>
  <p class="muted">У бренда нет документов целого типа — готовый список задач
    на сбор: одно письмо поставщику за строку.</p>
  <div id="orgGaps">—</div>
</section>

<section id="search">
  <h2>Из чего складывается ответ</h2>
  <p class="muted">Поиск идёт двумя каналами. Текстовый находит точные слова и
    артикулы — он незаменим для обозначений вроде «SPL WRP-A 2ECO6-38».
    Смысловой находит ответ, когда вопрос задан не теми словами, что в
    документе: «какая производительность» → «подача, м³/ч». Дальше оба
    списка объединяются, к оценке добавляется приоритет свежести, и первые
    двадцать кандидатов пересортировывает переранжирование.</p>
  <div class="flow" id="searchFlow"></div>

  <h2>Смысловой канал</h2>
  <div class="panel" id="semPanel">Загружаю…</div>
  <div class="toolbar">
    <select id="reProv" title="провайдер смыслового поиска">
      <option value="">— выберите провайдера —</option>
      <option value="lsa">lsa — своя модель, обучается на вашей базе</option>
      <option value="local">local — готовая модель (USER-bge-m3), скачается сама</option>
      <option value="onnx">onnx — готовая модель в формате onnx</option>
      <option value="gigachat">gigachat — облако Сбера</option>
      <option value="yandex">yandex — облако Яндекса</option>
      <option value="openai">openai-совместимый сервер</option>
      <option value="hashing">hashing — заглушка</option>
    </select>
    <button class="act" onclick="switchEmb()">Переключить провайдера</button>
    <button class="act sec" onclick="job('train_lsa')">Обучить модель на базе</button>
    <button class="act sec" onclick="job('reembed',{})">Пересчитать векторы</button>
  </div>
  <p class="muted">«Переключить провайдера» делает всё одной задачей:
    ставит недостающие пакеты, скачивает веса (для local — модель
    USER-bge-m3, дообученную под русский), при необходимости обучает свою
    модель, проверяет провайдера пробным текстом, сохраняет настройки и
    пересчитывает векторы. Если какой-то шаг не удался — настройки не
    меняются, поиск продолжает работать как раньше, а причина видна в
    разделе «Конвейер». Порядок при первом запуске остаётся прежним:
    индексация → обучить модель → пересчитать векторы.</p>

  <h2>Переранжирование</h2>
  <div class="panel" id="rrPanel">Загружаю…</div>
  <div class="cards" id="rrCards" style="margin-top:10px"></div>

  <h2>Проверить на живом вопросе</h2>
  <div class="toolbar">
    <input id="tstQ" placeholder="например: что будет если насос поработает без воды"
           style="width:440px">
    <button class="act" onclick="testSearch()">Найти</button>
    <label class="muted"><input type="checkbox" id="tstRR" checked> с переранжированием</label>
  </div>
  <p class="muted">Показывает, какой канал нашёл каждый фрагмент, с какой оценкой
    и как переранжирование изменило порядок. Это главный инструмент разбора
    «почему бот ответил не то».</p>
  <div id="tstOut"></div>

  <h2>Сравнить настройки на контрольных вопросах</h2>
  <div class="toolbar">
    <input id="cmpFile" value="eval/golden.jsonl" style="width:280px">
    <button class="act" onclick="job('compare',{dataset:$('cmpFile').value})">Прогнать сравнение</button>
  </div>
  <p class="muted">Прогоняет ваш набор контрольных вопросов без переранжирования
    и с ним при разных весах. Ориентируйтесь на MRR: он учитывает не только
    «нашлось ли», но и на каком месте. Это единственный честный способ
    выбрать настройку — на глаз такая разница не видна. Результат появится
    в разделе «Конвейер» и в журнале.</p>

  <h2>Настройки поиска</h2>
  <div id="setSearch"></div>
</section>

<!-- ═══════════════════════════════════════ сканы ════════════════════════ -->
<section id="eval">
  <h2>Контрольные вопросы</h2>
  <p class="muted">Единственный честный способ узнать, стал ли ассистент точнее, —
    прогнать один и тот же набор вопросов до и после изменения. Здесь этот набор
    собирается. Рабочий объём — 150–200 вопросов; лучший источник — реальные
    вопросы сотрудников из журнала ниже. К каждому вопросу укажите, где лежит
    ответ (часть пути с брендом и моделью) и какая цифра должна прозвучать.
    Особо ценны «пары-двойники»: вопрос про модель X с запретом цифр соседней
    модели Y — только они ловят уверенный ответ про другой товар.</p>
  <div class="panel" id="evPanel">Загружаю…</div>
  <div class="panel" id="evProblems" style="display:none"></div>

  <h2 id="evFormTitle">Добавить вопрос</h2>
  <div class="panel">
    <div class="toolbar">
      <input id="evQ" placeholder="вопрос — как его задал бы сотрудник" style="width:640px">
    </div>
    <div class="toolbar">
      <input id="evEF" placeholder="где ответ: часть пути, напр. ДЖИЛЕКС/4ПАСПОРТ/Водомет 55" style="width:315px"
             title="expect_files — документ, который должен найтись (через запятую можно несколько)">
      <input id="evET" placeholder="ожидаемые цифры в ответе: 75, 3.6" style="width:315px"
             title="expect_text — подстроки, которые обязаны быть в ответе">
    </div>
    <div class="toolbar">
      <input id="evRF" placeholder="подмена: путь двойника, напр. Водомет 60" style="width:315px"
             title="reject_files — документ, который считается ПОДМЕНОЙ, если стоит выше нужного">
      <input id="evRT" placeholder="запрещено в ответе: 92, 60/92" style="width:315px"
             title="reject_text — подстроки, которых в ответе быть не должно">
    </div>
    <div class="toolbar">
      <button class="act" onclick="evSave()">Сохранить вопрос</button>
      <button class="act sec" onclick="evClear()">Очистить форму</button>
      <span class="muted" id="evFormNote"></span>
    </div>
  </div>

  <h2>Кандидаты из журнала</h2>
  <p class="muted">Вопросы, которые сотрудники уже задавали — в первую очередь те,
    на которые бот не ответил. Нажмите «в набор», укажите, где лежит ответ,
    и сохраните.</p>
  <div id="evCandidates" class="panel">—</div>

  <h2>Набор <span id="evCount"></span></h2>
  <div id="evList">Загружаю…</div>

  <h2>Замер</h2>
  <div class="toolbar">
    <button class="act" onclick="job('regression',{reason:'вручную из раздела'})">Прогнать поиск</button>
    <button class="act sec" onclick="job('eval_llm')">Прогнать с генерацией (долго)</button>
  </div>
  <p class="muted">«Прогнать поиск» — быстрый замер: находятся ли нужные документы
    (hit, MRR) и нет ли подмен. «С генерацией» дополнительно проверяет сами
    ответы: есть ли ожидаемые цифры и нет ли запрещённых. Результат — в разделе
    «Конвейер»; история поисковых замеров со слепком настроек — ниже.</p>
  <div id="evRuns">—</div>
</section>

<section id="ocr">
  <h2>Распознавание сканов</h2>
  <p class="muted">Сертификаты и декларации почти всегда лежат сканами — картинками
    без текстового слоя. Для поиска таких документов не существует, а спрашивают
    их чаще всего. Прогон однократный: распознанное сразу нарезается на
    фрагменты и попадает в поиск.</p>
  <div class="cards" id="ocrCards"></div>
  <div class="toolbar">
    <button class="act" onclick="job('ocr',{limit:parseInt($('ocrLimit').value)||null})">Распознать очередь</button>
    <button class="act sec" onclick="job('ocr_retry')">Повторить неудавшиеся</button>
    <label class="muted">только первых <input id="ocrLimit" value="" placeholder="все" style="width:70px"></label>
  </div>

  <h2>Чем можно распознавать на этой машине</h2>
  <table id="ocrProv"><thead><tr><th style="width:130px">Режим</th><th style="width:90px">Состояние</th>
    <th>Подробности и что делать</th></tr></thead><tbody></tbody></table>

  <h2>Проверка на подмену кириллицы латиницей</h2>
  <p class="muted">Главная опасность распознавания русских документов — не пропущенная
    буква, а подмена: «МОСКВА» превращается в «MOCKBA», где все буквы латинские.
    Текст выглядит верным, а документ теряется навсегда. Вставьте сюда кусок
    распознанного текста, чтобы увидеть, что сделает проверка.</p>
  <div class="toolbar">
    <textarea id="guardIn" rows="3" style="width:560px"
      placeholder="CEPTИФИKAT COOTBETCTBИЯ, гopoд MOCKBA"></textarea>
    <button class="act" onclick="testGuard()">Проверить</button>
  </div>
  <div id="guardOut"></div>

  <h2>Очередь и результаты</h2>
  <div class="toolbar">
    <button class="act sec" onclick="loadOcr()">Обновить</button>
    <label class="muted"><input type="checkbox" id="ocrOnlyBad"> только проблемные</label>
  </div>
  <table id="ocrTable"><thead><tr><th>Документ</th><th style="width:70px">Стр.</th>
    <th style="width:100px">Состояние</th><th style="width:100px">Провайдер</th>
    <th style="width:90px">Качество</th><th style="width:130px">Когда</th></tr></thead><tbody></tbody></table>

  <h2>Настройки распознавания</h2>
  <div id="setOcr"></div>
</section>

<!-- ══════════════════════════════════════ копии ═════════════════════════ -->
<section id="backup">
  <h2>Резервные копии индекса</h2>
  <p class="muted">Файлы базы знаний никуда не денутся — их можно разобрать заново.
    Невосстановимо другое: выверенные ответы, накопленные вопросы сотрудников и
    обучающие пары. Пересборка вернёт тексты, но не вернёт ни одного выверенного
    ответа. Плюс сама пересборка — это часы, в течение которых ассистент не
    работает.</p>
  <div class="cards" id="bkCards"></div>
  <div class="toolbar">
    <button class="act" onclick="job('backup')">Сделать копию сейчас</button>
    <button class="act sec" onclick="job('backup_verify')">Проверить последнюю</button>
    <button class="act sec" onclick="job('backup_prune')">Удалить старые по правилу</button>
    <button class="act sec" onclick="job('backup_schedule')">Включить расписание</button>
    <button class="act sec" onclick="job('backup_unschedule')">Выключить расписание</button>
  </div>
  <div class="panel" id="bkPanel">Загружаю…</div>

  <h2>Что лежит в копиях</h2>
  <table id="bkTable"><thead><tr><th>Когда</th><th style="width:90px">Размер</th>
    <th>Содержимое</th><th style="width:110px">Пометка</th>
    <th style="width:190px"></th></tr></thead><tbody></tbody></table>
  <p class="muted">Снимок базы делается средствами самой SQLite, а не копированием
    файла: копия остаётся целостной, даже если в этот момент идёт индексация.
    Обычное копирование под нагрузкой даёт битый файл, который выглядит целым —
    и выясняется это в тот день, когда он понадобился.</p>

  <h2>Восстановление</h2>
  <div class="toolbar">
    <select id="bkPick" style="width:340px"></select>
    <button class="act warn" onclick="restoreBackup()">Восстановить из копии</button>
  </div>
  <p class="muted">Текущее состояние не удаляется, а откладывается рядом в папку
    с пометкой «перед восстановлением»: если развернули не ту копию, откат
    займёт минуту. Копия, не прошедшая проверку, не восстанавливается.</p>
  <div id="bkRestoreOut"></div>

  <h2>Настройки копирования</h2>
  <div id="setBackup"></div>
</section>

<section id="graph">
  <div class="toolbar">
    <button class="act" onclick="buildGraph()">Построить заново</button>
    <label class="muted">порог близости
      <input id="gSim" type="text" value="0.50" style="width:64px"></label>
    <label class="muted">связей у документа
      <input id="gDeg" type="text" value="6" style="width:52px"></label>
    <label class="muted">ограничить документами
      <input id="gLimit" type="text" placeholder="все" style="width:70px"></label>
    <button class="act sec" onclick="window.open('/graph.html','_blank')">Открыть в новой вкладке</button>
    <span id="graphMsg" class="muted"></span>
  </div>
  <p class="muted" style="max-width:900px;margin:0 0 10px">
    По умолчанию показываются не отдельные документы, а группы — разделы базы.
    Двойной щелчок раскрывает группу на уровень ниже: раздел → бренд → категория →
    документ. Слева — фильтры, раскладки и готовые виды: «Проблемы» показывает
    документы без смысловых связей, «Покрытие» — что у какого бренда есть, «Сканы» —
    что ждёт распознавания. Порог близости и число связей меняют плотность картины:
    чем выше порог, тем меньше случайных линий.
  </p>
  <iframe id="graphFrame" src="/graph.html"
   style="width:100%;height:78vh;border:1px solid #262a33;border-radius:10px;background:#0f1116"></iframe>
</section>

<section id="voice">
  <h2>Проверка распознавания и синтеза</h2>
  <div class="toolbar">
    <input id="ttsText" type="text" style="width:420px"
      value="Насос SPL WRP-A 2ECO6-38, напор 45 м, расход 3,6 м³/ч">
    <select id="ttsVoice"></select>
    <button class="act" onclick="testTts()">Озвучить</button>
    <button class="act sec" onclick="showNormalized()">Показать подготовку текста</button>
  </div>
  <pre id="voiceOut" style="max-height:200px">Здесь появится результат.</pre>
  <audio id="player" controls style="width:100%;margin-top:8px;display:none"></audio>
  <h2>Голоса</h2>
  <table id="voices"><thead><tr><th>Идентификатор</th><th>Движок</th><th>Тип</th>
    <th>Образец</th></tr></thead><tbody></tbody></table>
  <p class="muted" style="max-width:760px">Голос человека относится к биометрическим
   персональным данным. Прежде чем добавлять голос по образцу, получите письменное
   согласие сотрудника, где прямо сказано об использовании записи для обучения модели
   синтеза речи — общей формулировки о согласии на обработку персональных данных
   недостаточно.</p>
  <h2>Телефония</h2>
  <div class="toolbar"><button class="act sec" onclick="checkSip()">Проверить связь с АТС</button></div>
  <pre id="sipOut">Нажмите «Проверить связь с АТС».</pre>
</section>

<section id="settings">
  <h2>Все настройки системы</h2>
  <p class="muted">Здесь собрано всё, чем можно управлять, — с описанием, что
    настройка делает, рекомендацией, как её выбирать, и примером значения.
    Ничего не спрятано в файлах: если параметр есть в системе, он есть и на
    этом экране. Значения проверяются при сохранении, поэтому опечатка не
    всплывёт через час на следующем запуске модуля.</p>
  <div class="panel" id="setSummary">Загружаю…</div>
  <div class="toolbar">
    <input id="sFilter" placeholder="поиск по названию, ключу или описанию"
           style="width:300px">
    <label class="muted"><input type="checkbox" id="sChanged"> только изменённые</label>
    <button class="act" onclick="saveSettings()">Сохранить всё</button>
    <span class="muted">Вступают в силу при следующем запуске модулей; поиск,
      переранжирование и очередь подхватывают изменения сразу.</span>
    <span id="saved" class="good"></span>
  </div>
  <div id="setGroups" class="grpnav"></div>
  <div id="setIssues"></div>
  <div id="settingsList"></div>
</section>

<section id="logs">
  <h2>Подробность по подсистемам</h2>
  <div class="panel" id="logLevels"></div>
  <h2>Журнал</h2>
  <div class="toolbar">
    <select id="lgLevel"><option value="">все уровни</option><option>TRACE</option>
      <option>DEBUG</option><option>INFO</option><option>WARNING</option><option>ERROR</option></select>
    <input id="lgSub" placeholder="подсистема" style="width:160px">
    <input id="lgQ" placeholder="поиск: номер ошибки или текст" style="width:230px"
           onkeydown="if(event.key==='Enter')loadLogs()">
    <input id="lgLines" type="text" value="300" style="width:80px">
    <button class="act sec" onclick="loadLogs()">Обновить</button>
    <label class="muted"><input type="checkbox" id="lgAuto"> автообновление</label>
  </div>
  <pre id="logOut">Нажмите «Обновить».</pre>
</section>

<section id="queries">
  <h2>Последние вопросы</h2>
  <table id="qtable"><thead><tr><th>Время</th><th>Кто</th><th>Вопрос</th>
   <th>Уверенность</th><th>мс</th><th>Оценка</th></tr></thead><tbody></tbody></table>
  <h2>Вопросы без ответа</h2>
  <table id="gaps"><thead><tr><th>Время</th><th>Вопрос</th><th></th></tr></thead><tbody></tbody></table>
  <h2>Разбор конкретного ответа</h2>
  <p class="muted">Полная цепочка: что нашёл каждый канал с какими оценками,
    что переранжирование сделало с порядком, какой текст ушёл в модель и что
    она вернула. Это то, чем разбирают жалобу «бот ответил неправильно»:
    средние показатели на такой вопрос не отвечают.</p>
  <div class="toolbar">
    <input id="trFind" placeholder="поиск по тексту вопроса" style="width:280px">
    <button class="act sec" onclick="loadTraces()">Показать</button>
    <label class="muted"><input type="checkbox" id="trBad" onchange="loadTraces()">
      только отказы</label>
  </div>
  <table id="traces"><thead><tr><th style="width:130px">Когда</th><th>Вопрос</th>
    <th style="width:110px">Маршрут</th><th style="width:90px">Уверен.</th>
    <th style="width:80px">Ответ</th><th style="width:100px"></th></tr></thead><tbody></tbody></table>
  <div id="traceOut"></div>

  <h2>Выверенные ответы</h2>
  <div class="toolbar">
    <input id="gQ" placeholder="вопрос" style="width:300px">
    <input id="gA" placeholder="эталонный ответ" style="width:420px">
    <button class="act" onclick="addGolden()">Добавить</button>
  </div>
  <table id="golden"><thead><tr><th>Вопрос</th><th>Ответ</th><th>Использован</th></tr></thead><tbody></tbody></table>
</section>

<!-- ════════════════════════════════════ аналитика ═══════════════════════ -->
<section id="analytics">
  <div class="toolbar">
    <label class="muted">период
      <select id="anHours" onchange="loadAnalytics()">
        <option value="24">сутки</option><option value="168">неделя</option>
        <option value="720" selected>месяц</option><option value="8760">год</option>
      </select></label>
    <button class="act sec" onclick="loadAnalytics()">Обновить</button>
  </div>

  <h2>Воронка ответа</h2>
  <p class="muted">Путь вопроса от «задан» до «оценён». Смотреть надо на потери
    между ступенями: они показывают, где именно теряются ответы — в поиске,
    на пороге уверенности или в генерации. Без этой картины улучшают наугад.</p>
  <div class="panel" id="funnel">Загружаю…</div>
  <div id="funnelLoss"></div>
  <h3>По маршрутам</h3>
  <p class="muted">Каким путём получен ответ: выверенным ответом эксперта,
    точной выборкой из прайса или поиском по документам.</p>
  <table id="routes"><thead><tr><th>Маршрут</th><th style="width:90px">Вопросов</th>
    <th style="width:80px">Доля</th><th style="width:110px">Ответов</th>
    <th style="width:110px">Среднее, мс</th></tr></thead><tbody></tbody></table>

  <h2>Уверенность и порог отказа</h2>
  <p class="muted">Слева обычно скопление случайных совпадений, справа — настоящие
    находки. Порог ставят во впадину между ними. Красная отметка — текущее
    значение MIN_CONFIDENCE. Самая дорогая ошибка порога — отсечь ответ,
    который сотрудник счёл полезным: заметить её невозможно.</p>
  <div class="panel" id="confHist">Загружаю…</div>
  <div id="confAdvice"></div>

  <h2>Вклад каналов</h2>
  <p class="muted">Кто на самом деле находит ответы. Если почти всё приносит
    текстовый канал, переходить на более тяжёлую смысловую модель
    бессмысленно. Если заметная доля приходит только из смыслового —
    наоборот, более сильная модель даст ещё прирост.</p>
  <div class="cards" id="chCards"></div>
  <div class="panel" id="chVerdict" style="margin-top:10px"></div>

  <h2>Чего не хватает в базе</h2>
  <p class="muted">Вопросы, оставшиеся без ответа или оценённые как неверные,
    сгруппированы по темам. Это готовый список задач владельцам контента,
    отсортированный по частоте: список из двухсот отдельных вопросов
    бесполезен, а «двадцать три вопроса про подбор частотника» — уже
    понятно, какой документ нужен.</p>
  <div class="toolbar">
    <button class="act" onclick="loadGaps()">Собрать группы</button>
    <label class="muted">минимум в группе
      <input id="gapMin" value="2" style="width:56px"></label>
    <button class="act sec" onclick="exportGaps()">Выгрузить списком</button>
  </div>
  <div id="gapsOut"><span class="muted">Нажмите «Собрать группы».</span></div>

  <h2>Проверки качества</h2>
  <p class="muted">Контрольные вопросы прогоняются после переиндексации и смены
    настроек, результат сохраняется вместе со слепком настроек. На вопрос
    «после чего стало хуже» появляется точный ответ.</p>
  <div class="toolbar">
    <button class="act" onclick="job('regression',{reason:'вручную из админки'})">Прогнать сейчас</button>
  </div>
  <div class="panel" id="regPanel">Загружаю…</div>
  <table id="regHistory"><thead><tr><th style="width:130px">Когда</th>
    <th style="width:70px">hit</th><th style="width:70px">MRR</th>
    <th style="width:90px">Вопросов</th><th>Повод</th></tr></thead><tbody></tbody></table>

  <h2>Журнал действий администратора</h2>
  <p class="muted">Кто и когда менял настройки, запускал переиндексацию,
    восстанавливал индекс из копии и выдавал доступ.</p>
  <table id="adminLog"><thead><tr><th style="width:130px">Когда</th>
    <th style="width:170px">Кто</th><th style="width:130px">Действие</th>
    <th>Подробности</th></tr></thead><tbody></tbody></table>
</section>

<!-- ═══════════════════════════════════ сотрудники ═══════════════════════ -->
<section id="users">
  <h2>Доступ к боту</h2>
  <p class="muted">Незнакомый человек, написавший боту, получает отказ и
    предложение оставить заявку командой <code>/request</code>. Заявка
    появляется здесь: видно имя, ник в Telegram, когда обратился и что
    написал. Доступ выдаётся вместе с ролью — она определяет, какие разделы
    базы человек увидит.</p>
  <div class="cards" id="uCards"></div>
  <div class="panel" id="uWarn"></div>

  <h2>Заявки, ждущие решения</h2>
  <table id="uPending"><thead><tr><th>Сотрудник</th><th style="width:120px">ID</th>
    <th>Пояснение</th><th style="width:150px">Роль</th>
    <th style="width:210px"></th></tr></thead><tbody></tbody></table>

  <h2>Все, кто обращался</h2>
  <div class="toolbar">
    <input id="uFilter" placeholder="поиск по имени или ID" style="width:240px"
           oninput="renderUsers()">
    <select id="uStatus" onchange="renderUsers()">
      <option value="">все состояния</option>
      <option value="approved">есть доступ</option>
      <option value="pending">ждут решения</option>
      <option value="denied">отказано</option>
      <option value="blocked">заблокированы</option>
      <option value="new">не обращались</option>
    </select>
    <button class="act sec" onclick="loadUsers()">Обновить</button>
  </div>
  <table id="uAll"><thead><tr><th>Сотрудник</th><th style="width:110px">ID</th>
    <th style="width:130px">Состояние</th><th style="width:150px">Роль</th>
    <th style="width:90px">Вопросов</th><th style="width:130px">Последний раз</th>
    <th style="width:200px"></th></tr></thead><tbody></tbody></table>

  <h2>Роли и разделы</h2>
  <p class="muted">Роль определяет, какие разделы базы попадают в поиск для
    этого сотрудника. Настраивается в ROLE_SECTIONS; имена разделов должны
    совпадать с папками первого уровня внутри папки базы знаний.</p>
  <table id="uRoles"><thead><tr><th style="width:160px">Роль</th>
    <th>Доступные разделы</th></tr></thead><tbody></tbody></table>
</section>

<!-- ═══════════════════════════════════ телеграм ═════════════════════════ -->
<section id="telegram">
  <h2>Телеграм</h2>
  <p class="muted">Всё про бота в одном месте: настройки, заявки сотрудников
    на доступ и право дообучения. Сотрудник пишет боту → отправляет
    <code>/request</code> → заявка появляется ниже → вы подтверждаете её
    одним нажатием, сразу выбрав роль.</p>
  <div class="cards" id="tgCards"></div>

  <h2>Заявки на доступ</h2>
  <table id="tgPending"><thead><tr><th>Сотрудник</th><th style="width:120px">ID</th>
    <th>Пояснение</th><th style="width:150px">Роль</th>
    <th style="width:210px"></th></tr></thead><tbody></tbody></table>

  <h2>Дообучение</h2>
  <p class="muted">Сотрудник с признаком «дообучение» может закреплять
    эталонные ответы прямо из Telegram командой
    <code>/учить вопрос | ответ</code> — на похожий вопрос бот дальше
    отвечает именно так. Признак не зависит от роли: снабженец может
    отлично знать насосы и учить бота, не получая доступа к дилерским
    ценам. Каждый добавленный ответ виден в разделе
    «Контрольные вопросы» → «Выверенные ответы».</p>
  <table id="tgTrainers"><thead><tr><th>Сотрудник</th><th style="width:110px">ID</th>
    <th style="width:130px">Роль</th><th style="width:150px">Дообучение</th>
    <th style="width:120px">Ответов добавил</th></tr></thead><tbody></tbody></table>

  <h2>Настройки Telegram</h2>
  <div id="setTelegram"></div>
</section>

<!-- ═══════════════════════════════════ безопасность ═════════════════════ -->
<section id="safety">
  <h2>Что требует внимания прямо сейчас</h2>
  <p class="muted">Проверяется автоматически раз в час. Каждая запись —
    повод что-то сделать; когда делать нечего, здесь пусто. Повторные
    напоминания об одном и том же приходят не чаще раза в
    несколько часов, иначе одна незакрытая проблема заглушает остальные.</p>
  <div id="alertsOut">Загружаю…</div>
  <div class="toolbar">
    <button class="act" onclick="job('alerts',{})">Проверить сейчас</button>
  </div>

  <h2>Вход в администрирование</h2>
  <p class="muted">Пока учётных записей нет, действует прежнее правило: общий
    токен, а без него — доступ только с локального адреса. Как только заведена
    первая запись, включается вход по логину и паролю. Роли различают
    «посмотреть статистику» и «восстановить индекс из копии» — до сих пор это
    мог сделать один и тот же человек с одним и тем же паролем.</p>
  <div class="panel" id="accountsPanel">Загружаю…</div>
  <div class="toolbar">
    <input id="accLogin" placeholder="логин" style="width:150px">
    <input id="accName" placeholder="имя и фамилия" style="width:200px">
    <input id="accPass" type="password" placeholder="пароль, от 8 символов" style="width:190px">
    <select id="accRole">
      <option value="admin">admin — полный доступ</option>
      <option value="operator">operator — запускать задачи</option>
      <option value="viewer">viewer — только смотреть</option>
    </select>
    <button class="act" onclick="addAccount()">Создать</button>
  </div>
  <table id="accounts"><thead><tr><th>Логин</th><th>Имя</th><th style="width:110px">Роль</th>
    <th style="width:150px">Создана</th><th style="width:110px"></th></tr></thead><tbody></tbody></table>

  <h2>Сменить свой пароль</h2>
  <div class="toolbar">
    <input id="pwCur" type="password" placeholder="текущий пароль" style="width:190px">
    <input id="pwNew" type="password" placeholder="новый, от 8 символов" style="width:190px">
    <button class="act" onclick="changePassword()">Сменить</button>
    <span class="muted" id="pwMsg"></span>
  </div>

  <h2>Ключи и токены</h2>
  <p class="muted">Ключи выносятся из <code>.env</code> в отдельный файл с правами
    600, который не попадает ни в архив обновления, ни в резервную копию.
    Шифровать их своими силами было бы самообманом: ключ шифрования пришлось бы
    держать рядом. Правильный путь — внешнее хранилище, для него есть настройка
    SECRETS_CMD: любая команда, печатающая KEY=VALUE.</p>
  <div class="panel" id="secretsPanel">Загружаю…</div>
  <div class="toolbar">
    <button class="act" onclick="moveSecrets()">Вынести ключи в защищённый файл</button>
  </div>

  <h2>Регулярные задания</h2>
  <p class="muted">Ставятся средствами самой системы: crontab на Linux и macOS,
    планировщик задач на Windows. Своего демона здесь нет намеренно — он был бы
    ещё одним процессом, который может тихо умереть.</p>
  <table id="schedule"><thead><tr><th style="width:34px"></th><th style="width:120px">Когда</th>
    <th style="width:230px">Что</th><th>Почему так</th></tr></thead><tbody></tbody></table>
  <div class="toolbar">
    <button class="act" onclick="scheduleAction('install')">Поставить всё</button>
    <button class="act sec" onclick="scheduleAction('remove')">Снять</button>
  </div>

  <h2>Срок хранения данных</h2>
  <p class="muted">Тексты вопросов — персональные данные. Выверенные ответы и
    обучающие пары не удаляются никогда: это результат работы экспертов, а не
    переписка.</p>
  <div class="panel" id="retentionPanel">Загружаю…</div>
  <div class="toolbar">
    <button class="act sec" onclick="cleanRetention(true)">Показать, что удалится</button>
    <button class="act warn" onclick="cleanRetention(false)">Удалить просроченное</button>
    <input id="forgetId" placeholder="ID сотрудника" style="width:140px">
    <button class="act bad" onclick="forgetUser()">Удалить все его данные</button>
  </div>
  <div id="retentionOut"></div>
</section>

<section id="diag">
  <h2>Самопроверка</h2>
  <div class="toolbar"><button class="act" onclick="loadDiag()">Проверить всё</button></div>
  <table id="diagTable"><thead><tr><th>Что проверяем</th><th>Состояние</th>
   <th>Подробности</th><th>Что делать</th></tr></thead><tbody></tbody></table>
</section>
</main>
<script>
const RU = n => (n ?? 0).toLocaleString('ru');
const $ = id => document.getElementById(id);
// Если доступ пропал — например, администратор только что включил вход по
// учётным записям, — страница уходит на форму входа, а не сыплет ошибками.
const _fetch = window.fetch.bind(window);
window.fetch = async (...args) => {
  const r = await _fetch(...args);
  if (r.status === 401 && !location.pathname.startsWith('/login')) {
    location.href = '/login';
    throw new Error('нужен вход');
  }
  return r;
};
document.querySelectorAll('nav button').forEach(b => b.onclick = () => {
  document.querySelectorAll('nav button').forEach(x => x.classList.remove('on'));
  document.querySelectorAll('section').forEach(x => x.classList.remove('on'));
  b.classList.add('on'); $(b.dataset.t).classList.add('on');
  if (b.dataset.t === 'quick') loadQuick();
  if (b.dataset.t === 'models') { loadModels(); loadLlm(); loadModelProgress(); }
  if (b.dataset.t === 'diag') loadDiag();
  if (b.dataset.t === 'voice') loadVoices();
  if (b.dataset.t === 'logs') { loadLogLevels(); loadLogs(); }
  if (b.dataset.t === 'queries') { loadQueries(); loadTraces(); }
  if (b.dataset.t === 'kb') loadKb();
  if (b.dataset.t === 'organize') loadOrganize();
  if (b.dataset.t === 'search') loadSearch();
  if (b.dataset.t === 'eval') loadEval();
  if (b.dataset.t === 'ocr') loadOcr();
  if (b.dataset.t === 'backup') loadBackup();
  if (b.dataset.t === 'analytics') loadAnalytics();
  if (b.dataset.t === 'users') loadUsers();
  if (b.dataset.t === 'telegram') loadTelegram();
  if (b.dataset.t === 'safety') loadSafety();
  if (b.dataset.t === 'graph') $('graphFrame').src='/graph.html?t='+Date.now();
});
const STATUS_RU={running:'выполняется',ok:'готово',error:'ошибка',idle:'ожидание',done:'готово'};
const JOB_STATUS_RU={queued:'в очереди',running:'выполняется',done:'готово',
  error:'ошибка',cancelled:'снята',stale:'прервана'};
async function jobAction(what,id){
  const r=await (await fetch('/api/job/'+what,{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({id})})).json();
  alert(r.message||''); refreshFlow();
}

/* ------------------------------- простые графики без библиотек ------------ */
function lineChart(el, series, opts={}) {
  const W=560,H=170,P={l:34,r:10,t:10,b:20};
  const all=series.flatMap(s=>s.data).filter(v=>v!=null);
  const max=opts.max ?? Math.max(1,...all), min=0;
  const n=Math.max(...series.map(s=>s.data.length),1);
  const x=i=>P.l+(W-P.l-P.r)*(n>1?i/(n-1):0.5);
  const y=v=>H-P.b-(H-P.t-P.b)*((v-min)/(max-min||1));
  let g=`<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" style="height:170px">`;
  for(let k=0;k<=4;k++){const v=min+(max-min)*k/4;
    g+=`<line x1="${P.l}" y1="${y(v)}" x2="${W-P.r}" y2="${y(v)}" stroke="#232833"/>`;
    g+=`<text x="4" y="${y(v)+3}" fill="#6f7889" font-size="9">${Math.round(v)}</text>`;}
  series.forEach(s=>{
    const pts=s.data.map((v,i)=>`${x(i)},${y(v??0)}`).join(' ');
    g+=`<polyline points="${pts}" fill="none" stroke="${s.color}" stroke-width="1.6"/>`;
    if(opts.fill) g+=`<polygon points="${x(0)},${H-P.b} ${pts} ${x(s.data.length-1)},${H-P.b}"
      fill="${s.color}" opacity=".08"/>`;});
  if(opts.labels){const step=Math.max(1,Math.floor(opts.labels.length/6));
    opts.labels.forEach((t,i)=>{if(i%step===0)
      g+=`<text x="${x(i)}" y="${H-6}" fill="#6f7889" font-size="9" text-anchor="middle">${t}</text>`;});}
  el.innerHTML=g+'</svg>';
}
function barChart(el, items, opts={}) {
  const W=560,H=Math.max(120,items.length*26+20),P={l:opts.labelWidth??170,r:52};
  const max=Math.max(1,...items.map(i=>i.value));
  let g=`<svg viewBox="0 0 ${W} ${H}" style="height:${H}px">`;
  items.forEach((it,i)=>{const y=i*26+12,w=(W-P.l-P.r)*(it.value/max);
    g+=`<text x="0" y="${y+11}" fill="#c8cede" font-size="11">${esc(it.label).slice(0,34)}</text>`;
    g+=`<rect x="${P.l}" y="${y+2}" width="${Math.max(w,1)}" height="13" rx="3" fill="${it.color||'#4E79A7'}"/>`;
    g+=`<text x="${P.l+w+6}" y="${y+13}" fill="#8b93a3" font-size="10">${RU(it.value)}${opts.suffix||''}</text>`;});
  el.innerHTML=g+'</svg>';
}
const esc = s => String(s??'').replace(/[<>&"]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;'}[c]));

/* ------------------------------------ обзор ------------------------------- */
async function refreshOverview(){
  const d = await (await fetch('/api/overview')).json();
  const u=d.usage,i=d.index,hw=d.hardware;
  const cards=[
    ['Документов',RU(i.documents)],['Фрагментов',RU(i.chunks)],
    ['Позиций прайса',RU(i.products)],['Вопросов за неделю',RU(u.queries)],
    ['Отвечено',(u.answer_rate*100).toFixed(0)+'%'],
    ['Задержка, медиана',u.p50_ms+' мс'],['95-й перцентиль',u.p95_ms+' мс'],
    ['Оценки',u.up+' / '+u.down],
    ['Расход на модели',RU(d.models.total.cost_rub)+' ₽'],
    ['Токенов',RU(d.models.total.tokens_in+d.models.total.tokens_out)],
    ['Сканов без OCR',RU(i.needs_ocr)],['Ошибок разбора',RU(i.errors)],
    ['Процессор',(d.now.cpu??0)+'%'],['Память',(d.now.ram_gb??0)+' / '+(d.now.ram_total_gb??0)+' ГБ'],
    ['Видеопамять',hw.vram_total_gb?hw.vram_total_gb+' ГБ':'нет карт'],
    ['Свободно на диске',hw.disk_free_gb+' ГБ'],
  ];
  $('ovCards').innerHTML=cards.map(([k,v],idx)=>{
    const bad=(k==='Ошибок разбора'&&i.errors>50)||(k==='Отвечено'&&u.answer_rate<0.6);
    return `<div class="card${bad?' warn':''}"><div class="v">${v}</div><div class="k">${k}</div></div>`;}).join('');

  const p=d.server.points;
  lineChart($('chartServer'),[
    {data:p.map(x=>x.cpu),color:'#4E79A7'},
    {data:p.map(x=>x.ram),color:'#F28E2B'},
    {data:p.map(x=>(x.gpu_util||[])[0]??0),color:'#59A14F'},
  ],{max:100,labels:p.map(x=>x.ts),fill:true});
  const nGpu=(p[0]?.gpu_mem||[]).length;
  lineChart($('chartGpu'), nGpu?[
    ...Array.from({length:nGpu},(_,k)=>({data:p.map(x=>x.gpu_mem[k]??0),color:['#4E79A7','#E15759'][k%2]})),
    {data:p.map(x=>(x.gpu_temp||[])[0]??0),color:'#EDC948'},
  ]:[{data:[0],color:'#333'}],{max:100,labels:p.map(x=>x.ts)});
  $('gpuLegend').innerHTML=(d.server.gpu_names||[]).map((n,k)=>
    `<span><i style="background:${['#4E79A7','#E15759'][k%2]}"></i>${esc(n)} — память</span>`).join('')
    +'<span><i style="background:#EDC948"></i>температура</span>';

  lineChart($('chartQueries'),[{data:u.by_day.map(x=>x.n),color:'#4E79A7'},
    {data:u.by_day.map(x=>x.misses),color:'#E15759'}],
    {labels:u.by_day.map(x=>x.day.slice(5)),fill:true});
  barChart($('chartStages'),u.stages.map(s=>({label:s.stage,value:Math.round(s.avg_ms)})),
    {suffix:' мс',labelWidth:130});
  barChart($('chartSections'),u.sections.slice(0,8).map(([n,v])=>({label:n,value:v})));
  $('tblModels').innerHTML='<thead><tr><th>Модель</th><th>Вызовов</th><th>Токенов</th>'+
    '<th>Среднее, мс</th><th>Расход, ₽</th></tr></thead><tbody>'+
    d.models.by_model.map(m=>`<tr><td>${esc(m.model)}<div class="muted" style="font-size:11px">${esc(m.provider)} · ${esc(m.kind)}</div></td>
      <td>${RU(m.calls)}</td><td>${RU((m.tin||0)+(m.tout||0))}</td><td>${m.avg_ms}</td>
      <td>${RU(m.cost_rub)}</td></tr>`).join('')+'</tbody>';
  $('health').textContent=`док. ${RU(i.documents)} · вопросов ${RU(u.queries)} · ЦП ${d.now.cpu??0}%`;
}

/* ----------------------------------- конвейер ----------------------------- */
async function refreshFlow(){
  const s=await (await fetch('/api/state')).json();
  $('stages').innerHTML=s.stages.map(st=>{
    const pct=st.total?Math.round(st.processed/st.total*100):(st.status==='ok'?100:0);
    return `<div class="stage ${st.status}"><span class="dotst"></span>
      <div class="t">${st.title}</div><div class="n">${st.note}</div>
      <div class="d">${esc(st.detail)||'<span class="muted">'+STATUS_RU[st.status]+'</span>'}</div>
      <div class="bar"><i style="width:${pct}%"></i></div></div>`;}).join('');
  document.querySelector('#events tbody').innerHTML=s.recent.map(e=>
    `<tr><td class="muted">${(e.ts||'').replace('T',' ').slice(5,19)}</td><td>${esc(e.stage)}</td>
     <td class="${e.status==='error'?'bad':e.status==='running'?'warn':'good'}">${STATUS_RU[e.status]||e.status}</td>
     <td>${esc(e.detail)}</td></tr>`).join('');
  const jd=await (await fetch('/api/jobs')).json();
  document.querySelector('#jobs tbody').innerHTML=(jd.jobs||[]).map(j=>{
    const cls={running:'warn',queued:'muted',done:'good',error:'bad',
               stale:'bad',cancelled:'muted'}[j.status]||'muted';
    const act = j.status==='queued'
      ? `<button class="act sec" style="padding:3px 8px;font-size:12px"
           onclick="jobAction('cancel',${j.id})">снять</button>`
      : (j.status==='error'||j.status==='stale'
         ? `<button class="act sec" style="padding:3px 8px;font-size:12px"
              onclick="jobAction('retry',${j.id})">повторить</button>` : '');
    return `<tr><td>№${j.id} ${esc(j.title)}
       <div class="muted" style="font-size:11px">${esc(j.created_by||'')}
         ${j.attempt>1?`· попытка ${j.attempt}`:''}</div></td>
     <td class="${cls}">${JOB_STATUS_RU[j.status]||j.status}</td>
     <td>${j.seconds} с</td>
     <td class="muted">${esc(j.progress||'')}
       ${j.error?'<div class="bad">'+esc(String(j.error).slice(0,200))+'</div>':''}</td>
     <td>${act}</td></tr>`;}).join('')
     || '<tr><td colspan="5" class="muted">Задач пока не было.</td></tr>';
  loadExtractErrors();
}
async function loadExtractErrors(){
  const d=await (await fetch('/api/extract/errors')).json();
  if(!d.total){
    $('exErrSummary').innerHTML='<span class="good">Ошибок извлечения нет — все файлы разобраны.</span>';
    document.querySelector('#exErrs tbody').innerHTML='';
    return;
  }
  $('exErrSummary').innerHTML=
    `<b class="bad">Файлов с ошибками: ${d.total}</b>`+
    (d.shown<d.total?` <span class="muted">(показаны последние ${d.shown})</span>`:'')+
    '<div style="margin-top:6px">'+d.by_reason.map(r=>
      `<span class="muted" style="margin-right:12px">${esc(r.reason)} — <b>${r.count}</b></span>`).join('')+'</div>';
  document.querySelector('#exErrs tbody').innerHTML=d.errors.map(e=>
    `<tr><td title="${esc(e.rel_path)}">${esc(e.rel_path)}
       <div class="muted" style="font-size:11px">${esc(e.ext||'')} · ${(e.indexed_at||'').replace('T',' ').slice(0,16)}</div></td>
     <td class="bad" title="${esc(e.error||'')}">${esc(e.reason)}</td></tr>`).join('');
}
setInterval(()=>{ if($('flow').classList.contains('on')) refreshFlow(); },2500);
setInterval(()=>{ if($('overview').classList.contains('on')) refreshOverview(); },15000);
refreshOverview(); refreshFlow();

/* ------------------------------------ модели ------------------------------ */
let MODELS=[];
async function loadModels(){
  const d=await (await fetch('/api/models')).json();
  MODELS=d.catalog;
  const hw=d.hardware;
  $('hwCards').innerHTML=[
    ['Процессор',hw.cpu_cores+' ядер'],['Память',hw.ram_gb+' ГБ'],
    ['Видеопамять',hw.vram_total_gb?hw.vram_total_gb+' ГБ':'нет'],
    ['Карт',hw.gpus.length],['Свободно на диске',hw.disk_free_gb+' ГБ'],
    ...Object.entries(hw.engines).map(([k,v])=>[k,v?'есть':'нет']),
  ].map(([k,v])=>`<div class="card"><div class="v">${v}</div><div class="k">${k}</div></div>`).join('');
  const st=d.server;
  $('serverBar').innerHTML = st.running
    ? `<span class="good">Работает: <b>${esc(st.model)}</b> (${esc(st.engine)}) на ${esc(st.base_url)}, ${Math.round((st.uptime_seconds||0)/60)} мин</span>
       <button class="act bad" onclick="modelAction('stop')">Остановить</button>`
    : `<span class="muted">Сервер модели не запущен. Выберите модель ниже и нажмите «Запустить».</span>`;
  renderModels();
  $('mUsage').innerHTML='<thead><tr><th>Модель</th><th>Назначение</th><th>Вызовов</th>'+
    '<th>Вход</th><th>Выход</th><th>Среднее, мс</th><th>Максимум</th><th>Ошибок</th><th>₽</th></tr></thead><tbody>'+
    (d.usage.by_model.map(m=>`<tr><td>${esc(m.model)}</td><td>${esc(m.kind)}</td><td>${RU(m.calls)}</td>
      <td>${RU(m.tin)}</td><td>${RU(m.tout)}</td><td>${m.avg_ms}</td><td>${m.max_ms}</td>
      <td class="${m.errors?'bad':''}">${m.errors}</td><td>${RU(m.cost_rub)}</td></tr>`).join('')
      || '<tr><td colspan="9" class="muted">Обращений к моделям пока не было.</td></tr>')+'</tbody>';
}
function renderModels(){
  const f=$('mFilter').value.toLowerCase(), kind=$('mKind').value, onlyFits=$('mFits').checked;
  $('mList').innerHTML=MODELS.filter(m=>
    (!kind||m.kind===kind) && (!onlyFits||m.fits!==false) &&
    (!f||(m.title+m.id+m.notes).toLowerCase().includes(f))
  ).map(m=>`<div class="modelrow"><div class="info">
    <div><b>${esc(m.title)}</b> ${m.recommended?'<span class="chip star">рекомендуется</span>':''}
      ${m.installed?'<span class="chip ok">установлена</span>':''}
      ${m.fits===false?'<span class="chip no">не поместится</span>':''}</div>
    <div class="muted" style="font-size:12px;margin:3px 0">${esc(m.id)} · ${esc(m.params)} ·
      ${m.vram_gb} ГБ · ${esc(m.quant)} · контекст ${RU(m.context)} · ${esc(m.license)}</div>
    <div style="font-size:12.5px">${esc(m.notes)}</div>
    <div class="muted" style="font-size:12px;margin-top:3px">Русский: ${esc(m.russian)}</div></div>
    <div style="display:flex;flex-direction:column;gap:6px">
      <button class="act sec" onclick="checkModel('${m.id}')">Проверить</button>
      <button class="act sec" onclick="modelAction('install','${m.id}')">Скачать</button>
      ${['llm','vision'].includes(m.kind)?`<button class="act" onclick="modelAction('serve','${m.id}')">Запустить</button>`:''}
    </div></div>`).join('') || '<div style="padding:14px" class="muted">Ничего не найдено.</div>';
}
['mFilter','mKind','mFits'].forEach(id=>$(id).addEventListener('input',renderModels));
async function modelAction(action,id){
  const r=await (await fetch('/api/models/'+action,{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({id})})).json();
  alert(r.message||JSON.stringify(r));
  loadModels();
}

/* ---------------------------------- база знаний --------------------------- */
async function loadKb(){
  const d=await (await fetch('/api/kb')).json();
  $('kbCards').innerHTML=Object.entries({
    'Документов':d.index.documents,'Фрагментов':d.index.chunks,'Векторов':d.index.vectors,
    'Позиций прайса':d.index.products,'Дублей':d.index.duplicates,'Устаревших':d.index.outdated,
    'Сканов без OCR':d.index.needs_ocr,'Ошибок':d.index.errors,
    'Выверенных ответов':d.index.golden,'Обучающих пар':d.index.training_pairs,
  }).map(([k,v])=>`<div class="card"><div class="v">${RU(v)}</div><div class="k">${k}</div></div>`).join('');
  document.querySelector('#structure tbody').innerHTML=(d.structure||[]).map(e=>
    `<tr><td class="muted">${(e.ts||'').replace('T',' ').slice(5,16)}</td><td>${esc(e.kind)}</td>
     <td><code>${esc(e.path)}</code></td><td>${esc(e.detail)}</td></tr>`).join('')
     || '<tr><td colspan="4" class="muted">Изменений не зафиксировано.</td></tr>';
  loadKbHint(); loadExtras();
}
async function loadExtras(){
  const c=await (await fetch('/api/contextual')).json();
  const st=c.status, e=c.estimate;
  $('ctxPanel').innerHTML=
    `<div>${dot(st.done>0?true:null)} С приставкой от модели: <b>${RU(st.done)}</b>
       из ${RU(st.total)} фрагментов · в кэше ${RU(st.cached)}</div>
     <div class="muted" style="margin-top:6px">Оценка на весь остаток:
       ~${RU(e.tokens_in)} токенов на вход, ~${RU(e.tokens_out)} на выход,
       примерно <b>${e.cost_rub} ₽</b> и ${e.hours_sequential} ч
       (модель ${esc(String(e.model))}).</div>`
    + (String(e.model)==='echo'?`<div class="warn" style="margin-top:8px">Сейчас
       выбрана заглушка вместо модели — приставки получились бы бессмысленными.
       Сначала выберите провайдера генерации.</div>`:'')
    + `<div class="muted" style="margin-top:6px">После обработки обязательно
       пересчитайте векторы, иначе поиск не увидит новых приставок.</div>`;

  const s=await (await fetch('/api/sources')).json();
  $('crawlList').value=(s.sources||[]).join('\n');
  $('crawlPanel').innerHTML=
    `<div>${dot(s.sources.length>0)} Источников в списке: <b>${s.sources.length}</b>
      · страниц с сайтов в индексе: <b>${RU(s.pages)}</b></div>
     <div class="muted" style="margin-top:6px">Файл списка:
       <code>${esc(s.file)}</code> · поиск в интернете: ${esc(s.provider||'выключен')}</div>`;

  const m=await (await fetch('/api/media/state')).json();
  $('mediaPanel').innerHTML=
    `<div>${dot(m.total===0?true:(m.provider!=='none'))} Записей в базе:
       <b>${RU(m.total)}</b> · расшифровано: <b>${RU(m.done)}</b>
       · отрезков с таймкодами: ${RU(m.segments)}</div>
     <div class="muted" style="margin-top:6px">Распознавание речи:
       ${esc(m.provider)}</div>`
    + (m.provider==='none'&&m.total?`<div class="warn" style="margin-top:8px">
       Распознавание речи выключено — записи в поиске не участвуют.
       Выберите ASR_PROVIDER в настройках.</div>`:'');
}
async function saveSources(){
  const r=await (await fetch('/api/sources',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({sources:$('crawlList').value})})).json();
  alert('Сохранено адресов: '+r.count); loadExtras();
}
function dot(ok){return ok===true?'<span style="color:#59A14F">●</span>'
  :ok===false?'<span style="color:#E15759">●</span>':'<span style="color:#8b93a3">●</span>';}
function human(b){const u=['Б','КБ','МБ','ГБ'];let i=0;b=b||0;
  while(b>=1024&&i<3){b/=1024;i++;} return (i?b.toFixed(1):b)+' '+u[i];}

async function loadKbHint(){
  const m=await (await fetch('/api/maintenance')).json();
  const parts=[];
  if(m.embeddings.is_stub||!m.embeddings.ready)
    parts.push(`<div class="bad">Смысловой поиск не включён — работает только поиск
      по точным словам. Раздел «Качество поиска».</div>`);
  else if(m.lsa_stale)
    parts.push(`<div class="warn">База выросла с ${RU(m.lsa_trained_on)} до ${RU(m.chunks)}
      фрагментов — смысловую модель пора переобучить.</div>`);
  if(m.pending_vectors)
    parts.push(`<div class="warn">Без векторов осталось фрагментов: ${RU(m.pending_vectors)}.
      Нужен пересчёт в разделе «Качество поиска».</div>`);
  if(m.ocr.queue&&m.ocr.provider==='none')
    parts.push(`<div class="warn">Сканов ждёт распознавания: ${RU(m.ocr.queue)},
      распознаватель не выбран. Раздел «Сканы».</div>`);
  if(m.backup.stale)
    parts.push(`<div class="bad">Свежей резервной копии индекса нет. Раздел «Копии».</div>`);
  $('kbHint').innerHTML=parts.join('')||'<span class="good">Всё настроено: смысловой поиск включён, сканы распознаны, копии свежие.</span>';
}

/* ------------------------------- качество поиска -------------------------- */
/* ------------------------------- порядок в базе --------------------------- */
function orgBar(pct){
  const col = pct>=90?'#3fb950':pct>=60?'#d29922':'#f85149';
  return `<div style="background:#2a2f3a;border-radius:5px;height:9px;width:180px;display:inline-block;vertical-align:middle;margin:0 8px">
    <div style="background:${col};height:9px;border-radius:5px;width:${pct}%"></div></div><b>${pct}%</b>`;
}
async function loadOrganize(){
  const st=await (await fetch('/api/organize')).json();
  const pr=st.progress||{};
  $('orgProgress').innerHTML = pr.total
    ? `Файлов в индексе: <b>${pr.total}</b><br>
       раздел определился: ${orgBar(pr.section)}<br>
       бренд определился: ${orgBar(pr.brand)}<br>
       тип определился: ${orgBar(pr.type)}
       <span class="muted">(цель — больше 90%)</span><br>
       прайсы и сертификаты с датой: ${orgBar(pr.dated_total?Math.round(100*pr.dated_ok/pr.dated_total):100)}
       <span class="muted">(${pr.dated_ok} из ${pr.dated_total})</span>`
    : '<span class="muted">Индекс пуст — сначала проиндексируйте базу.</span>';
  $('orgAsked').innerHTML = (st.top_asked||[]).length
    ? st.top_asked.map(b=>`<span style="margin-right:14px"><b>${esc(b.brand)}</b>
        <span class="muted">${b.asked} вопр.</span></span>`).join('')
    : '<span class="muted">Журнал вопросов пока пуст — приоритет появится с первыми вопросами сотрудников.</span>';
  $('orgTwins').innerHTML = (st.brand_twins||[]).length
    ? '<table><tr><th>написания</th><th>файлов</th></tr>'
      + st.brand_twins.map(t=>`<tr><td>${t.variants.map(v=>esc(v.brand)).join(' ↔ ')}</td>
          <td class="muted">${t.variants.map(v=>v.files).join(' / ')}</td></tr>`).join('')+'</table>'
    : '<div class="panel good">Двойников не найдено.</div>';
  $('orgUntyped').innerHTML = (st.untyped||[]).length
    ? '<table><tr><th>файл</th><th>подсказка</th></tr>'
      + st.untyped.map(u=>`<tr><td><code>${esc(u.path)}</code></td>
          <td>${u.hint?('→ '+esc(u.hint)):'<span class="muted">?</span>'}</td></tr>`).join('')+'</table>'
    : '<div class="panel good">У всех файлов определился тип.</div>';
  $('orgUndated').innerHTML = (st.undated||[]).length
    ? '<table><tr><th>файл</th><th>тип</th><th>дата файла</th></tr>'
      + st.undated.map(u=>`<tr><td><code>${esc(u.path)}</code></td><td>${esc(u.doc_type)}</td>
          <td class="muted">${esc(u.mtime_hint||'')}</td></tr>`).join('')+'</table>'
    : '<div class="panel good">Все сменяемые документы датированы.</div>';
  $('orgBadNames').innerHTML = (st.bad_names||[]).length
    ? '<table><tr><th>файл</th><th>слово</th></tr>'
      + st.bad_names.map(b=>`<tr><td><code>${esc(b.path)}</code></td>
          <td>«${esc(b.word)}»</td></tr>`).join('')+'</table>'
    : '<div class="panel good">Имён со словами «новый/финал/копия» нет.</div>';
  $('orgDups').innerHTML = (st.duplicates||[]).length
    ? st.duplicates.map(d=>`<div class="panel" style="margin-bottom:6px">
        <b>${d.count} копии:</b><br>${d.paths.map(p=>`<code>${esc(p)}</code>`).join('<br>')}</div>`).join('')
    : '<div class="panel good">Точных дублей нет.</div>';
  $('orgGaps').innerHTML = (st.gaps||[]).length
    ? '<table><tr><th>бренд</th><th>чего не хватает</th></tr>'
      + st.gaps.map(g=>`<tr><td>${esc(g.brand)}</td>
          <td class="muted">${g.missing.map(esc).join(', ')}</td></tr>`).join('')+'</table>'
    : '<div class="panel good">У каждого бренда есть документы всех основных типов.</div>';
}

/* ------------------------------- контрольные вопросы --------------------- */
let EV_ITEMS = [];
async function loadEval(){
  const st=await (await fetch('/api/eval')).json();
  EV_ITEMS = st.items||[];
  const pct=Math.min(100, Math.round(100*st.count/st.target));
  const bar=`<div style="background:#2a2f3a;border-radius:6px;height:10px;width:320px;display:inline-block;vertical-align:middle">
    <div style="background:${st.count>=st.target?'#3fb950':'#d29922'};height:10px;border-radius:6px;width:${pct}%"></div></div>`;
  $('evPanel').innerHTML =
    `<b>${st.count}</b> из ${st.target} вопросов ${bar}<br>
     <span class="muted">с проверкой цифр ответа: ${st.with_text} · пар-двойников: ${st.with_twins}
     · файл: <code>${esc(st.path)}</code></span>`;
  const pr=$('evProblems');
  if ((st.problems||[]).length){
    pr.style.display='';
    pr.innerHTML='<b>Слабости набора</b> <span class="muted">— метрика не лучше своих эталонов</span><ul style="margin:6px 0 0 18px">'
      + st.problems.map(x=>`<li>${esc(x)}</li>`).join('')+'</ul>';
  } else { pr.style.display='none'; }
  $('evCount').textContent = st.count ? `(${st.count})` : '';
  $('evCandidates').innerHTML = (st.candidates||[]).length
    ? '<table><tr><th>вопрос</th><th>задавали</th><th>ответ был</th><th></th></tr>'
      + st.candidates.map(c=>`<tr><td>${esc(c.question)}</td><td>${c.asked}</td>
        <td>${c.answered?'да':'<b style="color:#f85149">нет</b>'}</td>
        <td><button class="act sec" onclick="evUse(this.dataset.q)" data-q="${esc(c.question)}">в набор</button></td></tr>`).join('')
      + '</table>'
    : '<span class="muted">Журнал пока пуст — кандидаты появятся после первых вопросов сотрудников.</span>';
  $('evList').innerHTML = EV_ITEMS.length
    ? '<table><tr><th>№</th><th>вопрос</th><th>где ответ</th><th>ожидается</th><th>запрещено</th><th></th></tr>'
      + EV_ITEMS.map((it,i)=>`<tr><td>${i+1}</td><td>${esc(it.question)}</td>
        <td class="muted">${esc((it.expect_files||[]).join(', '))}</td>
        <td class="muted">${esc((it.expect_text||[]).join(', '))}</td>
        <td class="muted">${esc([...(it.reject_files||[]),...(it.reject_text||[])].join(', '))}</td>
        <td style="white-space:nowrap">
          <button class="act sec" onclick="evEdit(${i})" title="править">✎</button>
          <button class="act sec" onclick="evTwin(${i})" title="создать вопрос-двойник">⧉</button>
          <button class="act sec" onclick="evDel(${i})" title="удалить">✕</button></td></tr>`).join('')
      + '</table>'
    : '<span class="muted">Набор пуст.</span>';
  $('evRuns').innerHTML = (st.runs||[]).length
    ? '<table><tr><th>когда</th><th>повод</th><th>вопросов</th><th>hit</th><th>MRR</th></tr>'
      + st.runs.map(r=>`<tr><td>${esc((r.created_at||'').replace('T',' ').slice(0,16))}</td>
        <td>${esc(r.reason)}</td><td>${r.questions}</td><td>${r.hit}</td><td>${r.mrr}</td></tr>`).join('')
      + '</table>'
    : '<span class="muted">Замеров ещё не было.</span>';
}
function evFill(it, idx){
  $('evQ').value = it.question||'';
  $('evEF').value = (it.expect_files||[]).join(', ');
  $('evET').value = (it.expect_text||[]).join(', ');
  $('evRF').value = (it.reject_files||[]).join(', ');
  $('evRT').value = (it.reject_text||[]).join(', ');
  $('evQ').dataset.index = idx===undefined?'':idx;
  $('evFormTitle').textContent = idx===undefined?'Добавить вопрос':`Править вопрос №${idx+1}`;
  $('evQ').scrollIntoView({behavior:'smooth',block:'center'});
}
function evClear(){ evFill({}, undefined); $('evFormTitle').textContent='Добавить вопрос'; $('evFormNote').textContent=''; }
function evEdit(i){ evFill(EV_ITEMS[i], i); }
function evUse(q){ evFill({question:q}, undefined);
  $('evFormNote').textContent='укажите, где лежит ответ, и сохраните'; }
function evTwin(i){
  const it=EV_ITEMS[i], rx=/\d+(?:[.,]\d+)?(?:[\/\-]\d+(?:[.,]\d+)?)+/g;
  const sigs=(it.question.match(rx))||[];
  evFill({question: it.question.replace(rx,'«МОДЕЛЬ-СОСЕД»'),
          reject_files: it.expect_files||[],
          reject_text: [...(it.expect_text||[]), ...sigs]}, undefined);
  $('evFormNote').textContent='замените «МОДЕЛЬ-СОСЕД» на соседнюю модель и укажите её ожидания';
}
async function evSave(){
  const idx=$('evQ').dataset.index;
  const body={item:{question:$('evQ').value, expect_files:$('evEF').value,
    expect_text:$('evET').value, reject_files:$('evRF').value, reject_text:$('evRT').value}};
  if (idx!=='' && idx!==undefined) body.index=parseInt(idx);
  const r=await (await fetch('/api/eval/save',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();
  if (r.error){ $('evFormNote').textContent=r.error; return; }
  evClear(); loadEval();
}
async function evDel(i){
  if (!confirm('Удалить вопрос №'+(i+1)+'?')) return;
  const r=await (await fetch('/api/eval/delete',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({index:i})})).json();
  if (r.error) alert(r.error);
  loadEval();
}

async function loadSearch(){
  const m=await (await fetch('/api/maintenance')).json();
  const e=m.embeddings, r=m.rerank, st=r.stats||{};

  $('searchFlow').innerHTML=[
    ['Текстовый канал','BM25 по словам и артикулам',true,'работает всегда'],
    ['Смысловой канал','векторы: '+e.provider,e.ready&&!e.is_stub,
      e.error||e.detail||''],
    ['Объединение','RRF + приоритет свежести',true,'константа RRF_K='+(m.rrf_k??60)],
    ['Переранжирование',r.provider,r.enabled?r.ready:null,r.error||r.detail||''],
  ].map(([t,n,ok,d])=>`<div class="stage ${ok===true?'ok':ok===false?'error':''}">
    <span class="dotst"></span><div class="t">${esc(t)}</div><div class="n">${esc(n)}</div>
    <div class="d">${esc(String(d).slice(0,110))}</div></div>`).join('');

  const meta=m.lsa_meta||{};
  $('semPanel').innerHTML=
   `<div>${dot(e.ready&&!e.is_stub)} <b>Провайдер:</b> ${esc(e.provider)}
      ${e.dim?'· '+e.dim+' измерений':''} ${e.error?'<span class="bad">— '+esc(e.error)+'</span>':''}</div>
    <div class="muted" style="margin-top:6px">${esc(e.detail||'')}</div>`
   + (meta.vocab?`<div style="margin-top:8px">Словарь: <b>${RU(meta.vocab)}</b> слов ·
      обучена на <b>${RU(meta.documents)}</b> фрагментах ·
      обучение заняло ${meta.trained_seconds} с ·
      файл ${human(m.lsa_bytes)}</div>`:'')
   + `<div style="margin-top:6px">Векторов в индексе: <b>${RU(m.vectors)}</b> из
      <b>${RU(m.chunks)}</b> фрагментов${m.pending_vectors?
      ' · <span class="warn">не посчитано: '+RU(m.pending_vectors)+'</span>':''}</div>
    <div style="margin-top:6px">${dot(m.dense_ok)} ${esc(m.dense_note||'')}</div>`
   + (m.dense_ok?'':`<div class="bad" style="margin-top:8px">Смысловой канал сейчас
      не участвует в поиске. Пока это так, вопрос, заданный не словами документа,
      не найдётся вовсе.</div>`)
   + (e.is_stub?`<div class="bad" style="margin-top:8px">Это заглушка. Смысловой
      близости она не даёт: «производительность» и «подача» для неё разные слова.
      Нажмите «Обучить модель на базе», затем «Пересчитать векторы».</div>`:'')
   + (m.lsa_stale?`<div class="warn" style="margin-top:8px">База выросла с
      ${RU(m.lsa_trained_on)} до ${RU(m.chunks)} фрагментов. Слова, появившиеся
      после обучения, модели неизвестны — переобучите.</div>`:'');

  $('rrPanel').innerHTML=
   `<div>${dot(r.enabled?r.ready:null)} <b>Режим:</b> ${esc(r.provider)} —
      ${esc(r.error||r.detail||'')}</div>
    <div class="muted" style="margin-top:6px">Вес ${m.rerank_weight}: доля голоса
      реранкера против оценки поиска. Единица означает «доверять только ему» —
      тогда пропадает приоритет свежести и на вопрос про цену всплывёт прайс
      позапрошлого года. Пересортировываются первые ${m.rerank_top_n} кандидатов.</div>`;
  $('rrCards').innerHTML=[
    ['Вызовов',st.calls||0],['Пар оценено',st.pairs||0],['Взято из кэша',st.cached||0],
    ['Среднее, мс',r.avg_ms||0],['Сбоев',st.failures||0],
  ].map(([k,v])=>`<div class="card${k==='Сбоев'&&v?' bad':''}">
    <div class="v">${RU(v)}</div><div class="k">${k}</div></div>`).join('');
}
function switchEmb(){
  const p=$('reProv').value;
  if(!p){ alert('Сначала выберите провайдера в списке слева.'); return; }
  job('embed_switch',{provider:p});
}

async function testSearch(){
  const q=$('tstQ').value.trim(); if(!q) return;
  $('tstOut').innerHTML='<span class="muted">Ищу…</span>';
  const d=await (await fetch('/api/search/test',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({question:q,rerank:$('tstRR').checked})})).json();
  if(d.error){$('tstOut').innerHTML='<div class="bad">'+esc(d.error)+'</div>';return;}
  $('tstOut').innerHTML=
   `<div class="panel" style="margin-bottom:10px">Найдено за <b>${d.ms}</b> мс ·
      текстовый канал дал <b>${d.bm25}</b>, смысловой <b>${d.dense}</b> ·
      уверенность <b>${d.confidence}</b> ${d.confidence<d.min_confidence?
      '<span class="warn">(ниже порога '+d.min_confidence+' — бот ответит «в базе нет данных»)</span>':''}
      ${d.moved?'· переранжирование изменило порядок':''}</div>`
   + '<table><thead><tr><th style="width:34px">#</th><th>Фрагмент</th>'
   + '<th style="width:80px">Итог</th><th style="width:220px">По каналам</th></tr></thead><tbody>'
   + d.hits.map((h,i)=>`<tr><td>${i+1}${h.was!=null&&h.was!==i?
       ` <span class="muted" style="font-size:11px">было ${h.was+1}</span>`:''}</td>
     <td><code>${esc(h.path)}</code><div class="muted" style="font-size:12px;margin-top:3px">${esc(h.text)}</div></td>
     <td>${h.score}</td>
     <td class="muted" style="font-size:12px">${Object.entries(h.channels).map(([k,v])=>
        k+' '+(v==null?'—':(+v).toFixed(3))).join('<br>')}</td></tr>`).join('')
   + '</tbody></table>';
}

/* ------------------------------------- сканы ------------------------------ */
async function loadOcr(){
  const d=await (await fetch('/api/ocr/state')).json();
  $('ocrCards').innerHTML=[
    ['Ждут распознавания',d.queue],['Распознано',d.done],['С ошибкой',d.failed],
    ['Слабое качество',d.weak],['Страниц распознано',d.pages],
    ['Среднее качество',d.avg_quality??'—'],
  ].map(([k,v])=>`<div class="card${(k==='С ошибкой'&&v)?' bad':(k==='Слабое качество'&&v)?' warn':''}">
    <div class="v">${typeof v==='number'?RU(v):esc(String(v))}</div><div class="k">${k}</div></div>`).join('');
  document.querySelector('#ocrProv tbody').innerHTML=Object.entries(d.available||{}).map(([k,v])=>
    `<tr><td><b>${esc(k)}</b>${k===d.provider?' <span class="muted">(выбран)</span>':''}</td>
     <td>${v==='готов'?'<span class="good">готов</span>':'<span class="muted">нет</span>'}</td>
     <td class="muted">${esc(v==='готов'?(OCR_NOTES[k]||''):v)}</td></tr>`).join('');
  const only=$('ocrOnlyBad').checked;
  const rows=(d.documents||[]).filter(r=>!only||r.error||(r.quality!=null&&r.quality<d.min_quality));
  document.querySelector('#ocrTable tbody').innerHTML=rows.map(r=>
    `<tr><td><code>${esc(r.path)}</code>${r.error?'<div class="bad" style="font-size:12px">'+esc(r.error)+'</div>':''}</td>
     <td>${r.pages??'—'}</td>
     <td>${r.needs_ocr?(r.error?'<span class="bad">ошибка</span>':'<span class="warn">в очереди</span>')
        :'<span class="good">распознан</span>'}</td>
     <td class="muted">${esc(r.provider||'—')}</td>
     <td class="${r.quality!=null&&r.quality<d.min_quality?'warn':''}">${r.quality??'—'}</td>
     <td class="muted">${(r.at||'').replace('T',' ').slice(0,16)}</td></tr>`).join('')
     || '<tr><td colspan="6" class="muted">Сканов, требующих распознавания, нет.</td></tr>';
}
const OCR_NOTES={
  tesseract:'Бесплатно и офлайн. Обязателен языковой пакет rus — без него русский текст выйдет латиницей.',
  vlm:'Зрительная модель через совместимый адрес. Лучшее качество на печатях и таблицах, самая надёжная кириллица. Можно поднять на своих картах.',
  yandex:'Yandex Vision OCR: российский контур, хорошо держит кириллицу.'};
async function testGuard(){
  const t=$('guardIn').value; if(!t.trim())return;
  const d=await (await fetch('/api/ocr/guard',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({text:t})})).json();
  $('guardOut').innerHTML=`<div class="panel">
    <div class="muted">Было:</div><div><code>${esc(d.before)}</code></div>
    <div class="muted" style="margin-top:8px">Стало:</div><div><code>${esc(d.after)}</code></div>
    <div style="margin-top:10px">Исправлено слов: <b>${d.fixed}</b>
      (смешанных ${d.fixed_mixed}, целиком латинских ${d.fixed_latin}) ·
      осталось подозрительных: <b>${(d.ratio*100).toFixed(1)}%</b> при пороге
      ${(d.threshold*100).toFixed(0)}% · оценка качества <b>${d.quality}</b></div>
    <div style="margin-top:6px" class="${d.accepted?'good':'warn'}">${d.accepted?
      'Страница с таким текстом была бы принята.':
      'Страница была бы отправлена запасному распознавателю (OCR_FALLBACK).'}</div>
    ${d.vocab?'':'<div class="warn" style="margin-top:6px">Словарь базы недоступен — обучите смысловую модель, тогда проверка станет точнее.</div>'}
  </div>`;
}

/* ------------------------------------- копии ------------------------------ */
async function loadBackup(){
  const d=await (await fetch('/api/backup/state')).json();
  const s=d.status;
  $('bkCards').innerHTML=[
    ['Копий',s.count],['Занято',human(s.total_bytes)],
    ['Последняя',s.age_hours!=null?Math.round(s.age_hours)+' ч назад':'нет'],
    ['Расписание',s.installed?'есть':'нет'],
    ['Вторая копия',d.mirror?'есть':'нет'],
  ].map(([k,v])=>`<div class="card${(k==='Последняя'&&s.stale)?' bad':(k==='Расписание'&&!s.installed)?' warn':''}">
    <div class="v" style="font-size:${typeof v==='number'?'21':'15'}px">${typeof v==='number'?RU(v):esc(String(v))}</div>
    <div class="k">${k}</div></div>`).join('');
  $('bkPanel').innerHTML=
   `<div>${dot(s.count?!s.stale:false)} <b>Папка копий:</b> <code>${esc(s.dir)}</code></div>
    <div style="margin-top:6px">${dot(s.installed)} <b>Расписание:</b>
      ${s.installed?'настроено, '+esc(s.schedule):'не настроено — копии делаются только вручную'}</div>
    <div style="margin-top:6px">${dot(!!d.mirror)} <b>Вторая копия:</b>
      ${d.mirror?'<code>'+esc(d.mirror)+'</code>':'не задана — копия лежит на том же диске, что и индекс'}</div>
    <div class="muted" style="margin-top:8px">Хранение: ежедневные за ${d.keep_daily} дн.,
      затем по одной за неделю (${d.keep_weekly}) и за месяц (${d.keep_monthly}).
      Самая свежая копия не удаляется никогда.</div>`
   + (s.stale?`<div class="bad" style="margin-top:8px">Свежей копии нет больше
      ${d.alert_hours} ч. Выверенные ответы и обучающие пары пересборкой не
      восстанавливаются.</div>`:'')
   + (d.mirror?'':`<div class="warn" style="margin-top:8px">Заполните BACKUP_MIRROR_DIR:
      копия на том же диске не спасает от отказа этого диска.</div>`);
  document.querySelector('#bkTable tbody').innerHTML=(d.archives||[]).map(a=>
    `<tr><td>${(a.created||'').replace('T',' ').slice(0,16)}</td><td>${human(a.bytes)}</td>
     <td class="muted">${a.counts?Object.entries(a.counts).filter(([,v])=>v)
        .map(([k,v])=>NAMES_RU[k]||k).slice(0,4).join(', '):'—'}
        ${a.counts?`<div style="font-size:12px">документов ${RU(a.counts.documents)},
        фрагментов ${RU(a.counts.chunks)}, выверенных ${RU(a.counts.golden_qa)}</div>`:''}</td>
     <td class="muted">${esc(a.note||'')}</td>
     <td><button class="act sec" onclick="verifyOne('${esc(a.name)}')">Проверить</button></td></tr>`).join('')
     || '<tr><td colspan="5" class="muted">Копий пока нет. Нажмите «Сделать копию сейчас».</td></tr>';
  $('bkPick').innerHTML=(d.archives||[]).map(a=>
    `<option value="${esc(a.name)}">${(a.created||'').replace('T',' ').slice(0,16)} · ${human(a.bytes)} · ${esc(a.name)}</option>`).join('');
}
const NAMES_RU={documents:'документы',chunks:'фрагменты',products:'прайсы',
  golden_qa:'выверенные ответы',training_pairs:'обучающие пары',queries:'запросы',feedback:'оценки'};
async function verifyOne(name){
  const r=await (await fetch('/api/backup/verify',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({name})})).json();
  alert(r.ok?`Копия в порядке: документов ${r.counts.documents}, фрагментов ${r.counts.chunks}, векторов ${r.vectors??'—'}`
           :`Копия НЕ в порядке: ${r.error}`);
}
async function restoreBackup(){
  const name=$('bkPick').value; if(!name) return;
  if(!confirm(`Восстановить индекс из ${name}?\n\nТекущее состояние будет отложено рядом — откат займёт минуту.`)) return;
  $('bkRestoreOut').innerHTML='<span class="muted">Восстанавливаю…</span>';
  const r=await (await fetch('/api/backup/restore',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({name})})).json();
  $('bkRestoreOut').innerHTML=r.error
    ? `<div class="panel bad">${esc(r.error)}</div>`
    : `<div class="panel"><div class="good">${esc(r.message||'Поставлено в очередь')}</div>
       <div class="muted" style="margin-top:6px">Прежнее состояние система
       отложит рядом, в папке <code>before-restore-*</code>: если после
       восстановления поиск ведёт себя не так, откат займёт минуту.</div></div>`;
  loadBackup();
}

async function job(kind,extra={}){
  const r=await (await fetch('/api/job',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({kind,...extra})})).json();
  alert(r.message); document.querySelector('nav button[data-t=flow]').click();
}
async function buildGraph(){
  $('graphMsg').textContent='Строю…';
  const r=await (await fetch('/api/graph',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({similarity:parseFloat($('gSim').value)||0.5,
      edges:parseInt($('gDeg').value)||6,
      limit:parseInt($('gLimit').value)||null})})).json();
  $('graphMsg').textContent=r.message;
  $('graphFrame').src='/graph.html?t='+Date.now();
}
async function runAudit(){
  $('auditOut').textContent='Считаю…';
  const r=await (await fetch('/api/audit')).json();
  $('auditOut').textContent=r.text;
}

/* -------------------------------------- голос ----------------------------- */
async function loadVoices(){
  const d=await (await fetch('/api/voices')).json();
  document.querySelector('#voices tbody').innerHTML=d.voices.map(v=>
    `<tr><td>${esc(v.id)}</td><td>${esc(v.provider)}</td><td>${esc(v.kind)}</td>
     <td>${v.sample_seconds?v.sample_seconds+' с':'—'}</td></tr>`).join('')
     || '<tr><td colspan="4" class="muted">Синтез выключен или голоса не настроены.</td></tr>';
  $('ttsVoice').innerHTML=d.voices.map(v=>`<option>${esc(v.id)}</option>`).join('');
}
async function testTts(){
  $('voiceOut').textContent='Синтезирую…';
  const r=await (await fetch('/api/tts',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({text:$('ttsText').value,voice:$('ttsVoice').value})})).json();
  $('voiceOut').textContent=r.message||'';
  if(r.url){$('player').src=r.url+'?t='+Date.now();$('player').style.display='block';}
}
async function showNormalized(){
  const r=await (await fetch('/api/normalize',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({text:$('ttsText').value})})).json();
  $('voiceOut').textContent='Как это будет произнесено:\n\n'+r.text;
}
async function checkSip(){
  const r=await (await fetch('/api/sip')).json();
  $('sipOut').textContent=r.text;
}

/* --------------------------------- быстрый старт -------------------------- */
let QUICK = null;
async function loadQuick(){
  QUICK = await (await fetch('/api/quickstart')).json();
  const pct = QUICK.total ? Math.round(QUICK.done / QUICK.total * 100) : 0;
  const left = QUICK.total - QUICK.done;
  $('qsSummary').innerHTML =
    `<div><b>Сделано ${QUICK.done} из ${QUICK.total}</b>
       ${left ? `· осталось ${left}` : '· всё готово'}</div>
     <div class="bar" style="margin-top:8px"><i style="width:${pct}%"></i></div>
     <div class="muted" style="margin-top:8px">Команды для терминала показаны у
       каждого шага: интерфейс и инструкция установщика делают одно и то же.
       Интерпретатор этой установки: <code>${esc(QUICK.python)}</code></div>`;
  renderQuickSteps();
}
function renderQuickSteps(){
  $('qsSteps').innerHTML = (QUICK.steps || []).map(step => {
    const isNext = step.key === QUICK.next;
    const mark = step.skip
      ? '<span class="mark def">не требуется</span>'
      : (step.done === true ? '<span class="mark ok">сделано</span>'
        : step.done === null ? '<span class="mark">проверка</span>'
        : '<span class="mark warn">не сделано</span>');
    // Настройки шага — те же карточки, что в разделе «Настройки»:
    // описание, рекомендация и пример живут в одном месте и не расходятся.
    const ids = [];
    const fields = (SETTINGS.length ? (step.settings || []) : []).map(key => {
      const idx = SETTINGS.findIndex(x => x.key === key);
      if (idx < 0) return '';
      ids.push(idx);
      return settingCard(SETTINGS[idx], 'qs_' + idx);
    }).join('');
    const buttons = [];
    if (step.action && step.action.kind)
      buttons.push(`<button class="act" onclick="runQuick('${step.key}')">
        ${esc(step.action_label || 'Выполнить')}</button>`);
    if (step.action && step.action.goto)
      buttons.push(`<button class="act sec"
        onclick="document.querySelector('nav button[data-t=${step.action.goto}]').click()">
        ${esc(step.action_label || 'Открыть')}</button>`);
    if (step.extra)
      buttons.push(`<button class="act sec" onclick="quickExtra('${step.key}')">
        ${esc(step.extra.label)}</button>`);
    if (fields)
      buttons.push(`<button class="act sec" onclick="saveQuick(${step.number})">
        Сохранить настройки шага</button>`);
    return `<div class="setting${isNext ? ' changed' : ''}" id="qs_step_${step.key}">
      <div class="row"><div class="lbl">
        <h4>${step.number}. ${esc(step.title)} ${mark}</h4>
        <p>${esc(step.detail || '')}</p>
        ${step.hint ? `<p class="rec">${esc(step.hint)}</p>` : ''}
        ${(step.warnings || []).map(w =>
            `<p class="rec">! ${esc(w)}</p>`).join('')}
        ${step.command ? `<p class="ex">${esc(quickCommand(step.command))}</p>` : ''}
      </div></div>
      ${fields ? `<div style="margin-top:10px" data-idx="${ids.join(',')}"
                       id="qs_fields_${step.number}">${fields}</div>` : ''}
      ${buttons.length ? `<div class="toolbar" style="margin-top:8px">
         ${buttons.join('')}<span class="good" id="qs_saved_${step.number}"></span>
       </div>` : ''}
    </div>`;
  }).join('');
}
// Путь к интерпретатору подставляем только там, где команда его требует.
// Иначе получается «/usr/bin/python3 nano .env», и человек это копирует.
function quickCommand(cmd){
  return cmd.split(' && ')
            .map(part => part.startsWith('python ')
                 ? QUICK.python + part.slice(6) : part)
            .join(' && ');
}
async function runQuick(key){
  const r = await (await fetch('/api/quickstart/run', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({step:key})})).json();
  alert(r.message || r.error || 'Готово');
  document.querySelector('nav button[data-t=flow]').click();
}
async function quickExtra(key){
  const step = QUICK.steps.find(x => x.key === key);
  if (!step || !step.extra) return;
  const r = await (await fetch(step.extra.post, {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify(step.extra.body || {})})).json();
  alert(r.message || r.error || 'Готово');
  loadQuick();
}
async function saveQuick(number){
  const box = $('qs_fields_' + number);
  if (!box) return;
  const payload = {};
  (box.dataset.idx || '').split(',').filter(Boolean).forEach(i => {
    const setting = SETTINGS[+i], el = $('qs_' + i);
    if (!el) return;
    if (setting.type === 'secret' && !el.value) return;
    payload[setting.key] = setting.type === 'bool' ? (el.checked ? '1' : '0') : el.value;
  });
  if (await submitSettings(payload, 'qs_saved_' + number)) { await loadSettings(); loadQuick(); }
}
setInterval(() => { if ($('quick').classList.contains('on')) loadQuick(); }, 20000);

/* ----------------------------------- настройки ---------------------------- */
let SETTINGS=[], SETGROUPS=[], SETMETA={};
async function loadSettings(){
  const d=await (await fetch('/api/settings')).json();
  SETTINGS=d.settings||d;            // терпим старый формат ответа
  SETGROUPS=d.groups||[];
  SETMETA=d;
  renderSetSummary();
  renderGroupNav();
  renderSettings();
  renderGroup('setSearch',['Поиск'],
    ['EMBEDDINGS_PROVIDER','EMBEDDINGS_MODEL','LSA_DIM','LSA_MAX_FEATURES',
     'LSA_MIN_DF','LSA_STALE_RATIO','ONNX_MODEL_PATH','ONNX_TOKENIZER_DIR',
     'ONNX_POOLING','ONNX_MAX_TOKENS'],'loadSearch');
  renderGroup('setOcr',['Распознавание сканов'],null,'loadOcr');
  renderGroup('setTelegram',['Telegram'],null,'loadTelegram');
  renderGroup('setBackup',['Резервные копии'],null,'loadBackup');
  renderGroup('setQueue',['Очередь к модели'],null,'loadQueue');
}
// Сводка сверху отвечает на два вопроса, которые задают первыми: сколько
// всего можно настроить и что у нас уже изменено относительно умолчаний.
// Второе важнее: в реальной установке меняют десяток настроек из двух
// сотен, и именно они объясняют, почему система ведёт себя не как у всех.
function renderSetSummary(){
  const total=SETTINGS.length;
  const changed=SETTINGS.filter(s=>!s.is_default&&s.type!=='secret').length;
  const keys=SETTINGS.filter(s=>s.type==='secret');
  const filled=keys.filter(s=>s.filled).length;
  const sec=SETMETA.secrets||{};
  $('setSummary').innerHTML=
    `<div>Всего настроек: <b>${total}</b> в ${SETGROUPS.length} разделах ·
      изменено относительно умолчаний: <b>${changed}</b> ·
      ключей задано: <b>${filled}</b> из ${keys.length}</div>
     <div class="muted" style="margin-top:6px">Обычные настройки хранятся в
       ${esc(SETMETA.env_file||'.env')}, ключи и пароли — отдельно в
       ${esc(SETMETA.secrets_file||'secrets.env')} с правами 600: этот файл
       не попадает ни в архив обновления, ни в резервную копию.</div>`
    + ((sec.problems||[]).length?`<div class="warn" style="margin-top:6px">
        ${sec.problems.map(esc).join('<br>')}</div>`:'');
}
function renderGroupNav(){
  $('setGroups').innerHTML=SETGROUPS.map(g=>
    `<a onclick="jumpToGroup('${esc(g.name).replace(/'/g,"")}')">${esc(g.name)}
      <span class="n">${g.count}</span>${g.changed?` <b>${g.changed}</b>`:''}</a>`
    ).join('');
}
function jumpToGroup(name){
  $('sFilter').value=''; $('sChanged').checked=false; renderSettings();
  const el=document.getElementById('grp_'+cssId(name));
  if(el) el.scrollIntoView({behavior:'smooth',block:'start'});
}
function cssId(s){ return s.replace(/[^a-zA-Zа-яА-Я0-9]/g,'_'); }
// Одна и та же разметка настройки используется и в общем разделе, и внутри
// тематических: описание, рекомендация и пример живут в settings_schema.py,
// то есть в одном месте, и не расходятся между экранами.
function settingCard(s,id){
  let field;
  if(s.type==='bool') field=`<input type="checkbox" id="${id}" ${['True','1','true'].includes(String(s.value))?'checked':''}>`;
  else if(s.type==='enum') field=`<select id="${id}">${(s.options||[]).map(o=>
    `<option ${String(s.value)===o?'selected':''}>${esc(o)}</option>`).join('')}</select>`;
  else if(s.type==='suggest') field=`<input id="${id}" type="text" list="${id}_dl"
    value="${esc(s.value)}" placeholder="выберите из списка или впишите своё">
    <datalist id="${id}_dl">${(s.options||[]).map(o=>
      `<option value="${esc(o)}">`).join('')}</datalist>`;
  else field=`<input id="${id}" type="${s.type==='secret'?'password':'text'}"
    value="${s.type==='secret'?'':esc(s.value)}"
    placeholder="${s.type==='secret'?(s.filled?'ключ задан — впишите новый, чтобы заменить':'не задан'):''}">`;
  // Отметки: что изменено относительно умолчания, а что нет. Из двухсот
  // настроек человека интересуют именно изменённые.
  const mark = s.type==='secret'
    ? `<span class="mark key">${s.filled?'ключ задан':'ключ не задан'}</span>`
    : (s.is_default?'<span class="mark def">по умолчанию</span>'
                   :'<span class="mark">изменено</span>');
  const def = s.type==='secret' ? ''
    : `<div class="tiny">По умолчанию: <code>${esc(String(s.default)===''?'пусто':s.default)}</code></div>`;
  const rng = s.range
    ? `<div class="tiny">Допустимо ${s.range.min}…${s.range.max},
        обычно ${s.range.usual_min}…${s.range.usual_max}</div>` : '';
  const act = s.type==='secret'
    ? (s.filled?`<button class="act sec" onclick="forgetSecret('${s.key}')">Стереть ключ</button>`:'')
    : (s.is_default?'':`<button class="act sec" onclick="resetSetting('${s.key}')">Вернуть по умолчанию</button>`);
  // У ключа примера нет намеренно: образец ключа показывать незачем, а
  // придуманный образец люди пробуют вставлять как есть.
  const ex = s.example
    ? `<p class="ex">Пример: ${esc(s.key)}=${esc(s.example)}</p>` : '';
  return `<div class="setting${s.is_default||s.type==='secret'?'':' changed'}"
      id="set_${s.key}"><div class="row"><div class="lbl">
    <h4>${esc(s.title)} ${mark}</h4><div class="key">${esc(s.key)}</div>
    <p>${esc(s.help)}</p><p class="rec">Рекомендация: ${esc(s.rec)}</p>
    ${ex}</div>
    <div class="side">${field}${def}${rng}${act}</div></div></div>`;
}
function renderSettings(){
  const f=$('sFilter').value.toLowerCase();
  const onlyChanged=$('sChanged').checked;
  let html='',group=null,shown=0;
  SETTINGS.forEach((s,i)=>{
    if(f && !(s.title+s.key+s.help+s.rec+s.group).toLowerCase().includes(f)) return;
    if(onlyChanged && (s.is_default||s.type==='secret')) return;
    if(s.group!==group){group=s.group;
      html+=`<div class="grp" id="grp_${cssId(group)}">${esc(group)}</div>`;}
    html+=settingCard(s,'f'+i); shown++;
  });
  $('settingsList').innerHTML=html||
    `<div class="muted" style="padding:14px">${onlyChanged
      ? 'Всё стоит на значениях по умолчанию.' : 'Ничего не найдено.'}</div>`;
  $('settingsList').querySelectorAll('input,select').forEach(el=>{
    el.addEventListener('change',scheduleCheck);
    el.addEventListener('input',scheduleCheck);
  });
}
async function resetSetting(key){
  const r=await (await fetch('/api/settings/reset',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({keys:[key]})})).json();
  if(r.error){ alert(r.error); return; }
  loadSettings();
}
async function forgetSecret(key){
  if(!confirm('Стереть ключ '+key+'? Модуль, которому он нужен, перестанет работать.'))
    return;
  const r=await (await fetch('/api/settings/forget-secret',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({key})})).json();
  alert(r.message||r.error||'Готово');
  loadSettings();
}
// Тематический блок настроек внутри раздела: те же карточки с пояснениями,
// но только нужные группы или ключи, и своя кнопка сохранения.
function renderGroup(container,groups,keys,reload){
  const picked=SETTINGS.map((s,i)=>[s,i]).filter(([s])=>
    (groups&&groups.includes(s.group))||(keys&&keys.includes(s.key)));
  if(!picked.length){$(container).innerHTML='<div class="muted">Нет настроек.</div>';return;}
  $(container).innerHTML=
    `<div class="toolbar"><button class="act" onclick="saveGroup('${container}')">Сохранить настройки</button>
      <span class="muted">Вступают в силу при следующем запуске модулей; поиск и
      переранжирование подхватывают изменения сразу.</span>
      <span id="${container}Saved" class="good"></span></div>`
    + picked.map(([s,i])=>settingCard(s,container+'_'+i)).join('');
  $(container).dataset.idx=picked.map(([,i])=>i).join(',');
  $(container).dataset.reload=reload||'';
}
async function saveGroup(container){
  const payload={};
  const picked=($(container).dataset.idx||'').split(',').filter(Boolean);
  picked.forEach(i=>{
    const s=SETTINGS[+i], el=$(container+'_'+i); if(!el)return;
    if(s.type==='secret' && !el.value) return;
    payload[s.key]=s.type==='bool'?(el.checked?'1':'0'):el.value;
  });
  const ok=await submitSettings(payload, container+'Saved');
  if(ok){
    picked.forEach(i=>{const s=SETTINGS[+i]; s.value=payload[s.key];});
    const fn=$(container).dataset.reload; if(fn&&window[fn]) window[fn]();
  }
}

// Проверка значений живёт на сервере (settings_schema.validate): одни и те же
// правила и для админки, и для командной строки. Ошибку сохранить нельзя,
// предупреждение — можно, но только осознанно.
function issuesHtml(issues){
  return issues.map(i=>`<div style="margin-bottom:7px">
    <b class="${i.level==='error'?'bad':'warn'}">${i.level==='error'?'Ошибка':'Внимание'}:</b>
    ${esc(i.title)} <code>${esc(i.key)}</code> — ${esc(i.message)}
    ${i.hint?`<div class="muted">${esc(i.hint)}</div>`:''}</div>`).join('');
}
async function submitSettings(payload, statusId, force){
  const box=$('setIssues');
  const r=await (await fetch('/api/settings',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify(force?{...payload,force:true}:payload)})).json();
  if(r.ok){
    if(box) box.innerHTML = (r.issues&&r.issues.length)
      ? `<div class="panel">${issuesHtml(r.issues)}</div>` : '';
    if(statusId&&$(statusId)){ $(statusId).textContent='Сохранено';
      setTimeout(()=>$(statusId).textContent='',3000); }
    if(r.note) alert(r.note);
    if(r.changed&&r.changed.length) loadSettings();
    return true;
  }
  if(r.note) alert(r.note);
  const html=`<div class="panel">${issuesHtml(r.issues)}</div>`;
  if(box) box.innerHTML=html;
  if(r.errors){
    alert('Сохранить нельзя: '+r.issues.filter(i=>i.level==='error')
      .map(i=>i.key+' — '+i.message).join('\n'));
    return false;
  }
  const list=r.issues.map(i=>'• '+i.key+': '+i.message).join('\n');
  if(confirm('Значения необычные, но допустимые:\n\n'+list+'\n\nСохранить всё равно?'))
    return submitSettings(payload, statusId, true);
  return false;
}
$('sFilter').addEventListener('input',renderSettings);
$('sChanged').addEventListener('change',renderSettings);
async function saveSettings(){
  const payload={};
  SETTINGS.forEach((s,i)=>{const el=$('f'+i); if(!el)return;
    // Пустое поле ключа означает «не трогать»: значение в браузер не
    // приходит вовсе, поэтому пустым оно выглядит и когда ключ задан.
    if(s.type==='secret' && !el.value) return;
    payload[s.key]=s.type==='bool'?(el.checked?'1':'0'):el.value;});
  await submitSettings(payload,'saved');
}
// Проверка «на лету»: не дожидаясь сохранения, показываем, что не так.
let checkTimer=null;
function scheduleCheck(){
  clearTimeout(checkTimer);
  checkTimer=setTimeout(async()=>{
    const payload={};
    SETTINGS.forEach((s,i)=>{const el=$('f'+i); if(!el)return;
      if(s.type==='secret' && !el.value) return;
      payload[s.key]=s.type==='bool'?(el.checked?'1':'0'):el.value;});
    const r=await (await fetch('/api/settings/check',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)})).json();
    $('setIssues').innerHTML=(r.issues&&r.issues.length)
      ? `<div class="panel">${issuesHtml(r.issues)}</div>` : '';
  },500);
}
// Быстрый старт открыт по умолчанию, а карточки настроек в нём — те же,
// что в разделе «Настройки». Поэтому сначала настройки, потом шаги.
loadSettings().then(loadQuick);

/* ------------------------------- безопасность ----------------------------- */
async function loadSafety(){
  const a=await (await fetch('/api/alerts')).json();
  $('alertsOut').innerHTML=(a.active||[]).length
    ? (a.active||[]).map(x=>`<div class="panel" style="margin-bottom:8px">
        <div class="${x.level==='error'?'bad':'warn'}"><b>${esc(x.title)}</b></div>
        <div style="margin-top:4px">${esc(x.detail)}</div>
        ${x.action?`<div class="muted" style="margin-top:4px">${esc(x.action)}</div>`:''}
        <div class="muted" style="font-size:11px;margin-top:5px">замечено
          ${(x.first_seen||'').replace('T',' ').slice(0,16)}</div></div>`).join('')
    : '<div class="panel good">Всё в порядке: проблем, требующих внимания, нет.</div>';

  const w=await (await fetch('/api/whoami')).json();
  $('accountsPanel').innerHTML = w.accounts
    ? (w.default_password
        ? `<div class="bad"><b>Действует пароль по умолчанию admin/admin.</b>
           Он известен всем, а интерфейс может быть открыт из сети —
           смените пароль в форме ниже.</div>` : '')
      + `<div>${dot(true)} Вход по учётным записям включён.
        ${w.account?`Вы вошли как <b>${esc(w.account.full_name||w.account.login)}</b>
        (${esc(w.account.role)}).`:''}</div>
       <div class="muted" style="margin-top:6px">${Object.entries(w.roles||{})
         .map(([r,d])=>`<b>${esc(r)}</b> — ${esc(d)}`).join('<br>')}</div>`
    : `<div class="warn">${dot(false)} Учётных записей нет. Сейчас доступ даёт
        общий токен, а без него — только локальный адрес. Как только админку
        откроют наружу, этого мало: любой, кто дотянется до порта, сможет
        восстановить индекс из копии. Заведите хотя бы одну запись.</div>`;
  document.querySelector('#accounts tbody').innerHTML=(w.users||[]).map(u=>
    `<tr><td><b>${esc(u.login)}</b></td><td>${esc(u.full_name||'')}</td>
     <td>${esc(u.role)}</td>
     <td class="muted">${(u.created_at||'').replace('T',' ').slice(0,16)}</td>
     <td><button class="act sec" onclick="delAccount('${esc(u.login)}')">Удалить</button></td></tr>`).join('')
     || '<tr><td colspan="5" class="muted">Записей нет.</td></tr>';

  const sec=w.secrets||{};
  $('secretsPanel').innerHTML = sec.ok===undefined
    ? '<span class="muted">Недоступно.</span>'
    : `<div>${dot(sec.ok)} Файл ключей: <code>${esc(sec.file||'')}</code>
        ${sec.exists?`· права ${esc(sec.mode||'')}`:'· пока не создан'}
        ${sec.external?'· внешнее хранилище подключено':''}</div>`
      + (sec.problems||[]).map(x=>`<div class="warn" style="margin-top:6px">${esc(x)}</div>`).join('')
      + (sec.ok?'<div class="good" style="margin-top:6px">Ключей в открытом виде не найдено.</div>':'');

  const sch=await (await fetch('/api/schedule')).json();
  document.querySelector('#schedule tbody').innerHTML=(sch.tasks||[]).map(t=>
    `<tr><td>${t.installed?'<span class="good">✓</span>':'<span class="muted">—</span>'}</td>
     <td><code>${esc(t.cron)}</code></td><td>${esc(t.title)}</td>
     <td class="muted">${esc(t.why)}</td></tr>`).join('');

  const r=await (await fetch('/api/retention')).json();
  $('retentionPanel').innerHTML=
    `<div>Вопросов: <b>${RU(r.queries)}</b>, срок хранения
       ${r.queries_days?r.queries_days+' дней':'<span class="bad">не задан</span>'},
       просрочено: <b>${RU(r.queries_expired)}</b></div>
     <div style="margin-top:6px">Цепочек разбора: ${RU(r.traces)},
       срок ${r.traces_days} дней, просрочено: ${RU(r.traces_expired)}</div>
     <div class="muted" style="margin-top:6px">Связано с конкретными людьми:
       ${RU(r.identified)} вопросов · не удаляются никогда: выверенных ответов
       ${RU(r.golden)}, обучающих пар ${RU(r.training_pairs)}</div>`
    + (!r.queries_days?`<div class="bad" style="margin-top:8px">Срок хранения не
       задан — вопросы копятся бессрочно. Это персональные данные; задайте
       RETENTION_QUERIES_DAYS.</div>`:'');
}
async function addAccount(){
  const r=await (await fetch('/api/users/admin',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({action:'add',login:$('accLogin').value,
      password:$('accPass').value,role:$('accRole').value,
      full_name:$('accName').value})})).json();
  if(r.error){ alert(r.error); return; }
  $('accPass').value=''; loadSafety();
}
async function changePassword(){
  const r=await (await fetch('/api/password',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({current:$('pwCur').value,new:$('pwNew').value})})).json();
  $('pwMsg').textContent=r.message||r.error||'';
  if(r.message){ $('pwCur').value='';$('pwNew').value=''; loadSafety(); }
}
async function delAccount(login){
  if(!confirm('Удалить учётную запись «'+login+'»?')) return;
  await fetch('/api/users/admin',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({action:'delete',login})});
  loadSafety();
}
async function moveSecrets(){
  const r=await (await fetch('/api/secrets/move',{method:'POST',
    headers:{'Content-Type':'application/json'},body:'{}'})).json();
  alert(r.message + (r.moved&&r.moved.length?': '+r.moved.join(', '):''));
  loadSafety();
}
async function scheduleAction(action){
  const r=await (await fetch('/api/schedule',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({action})})).json();
  alert(r.message); loadSafety();
}
async function cleanRetention(dry){
  if(!dry && !confirm('Удалить данные, у которых вышел срок хранения?')) return;
  const r=await (await fetch('/api/retention',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({action:'clean',dry})})).json();
  $('retentionOut').innerHTML=`<div class="panel">${dry?'Было бы удалено':'Удалено'}:
    вопросов ${RU(r.queries)}, цепочек ${RU(r.traces)}</div>`;
  if(!dry) loadSafety();
}
async function forgetUser(){
  const id=parseInt($('forgetId').value); if(!id) return;
  if(!confirm('Удалить все данные сотрудника '+id+'? Это необратимо.')) return;
  const r=await (await fetch('/api/retention',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({action:'forget',user_id:id})})).json();
  $('retentionOut').innerHTML=`<div class="panel">Сотрудник ${id}: ${esc(r.mode)},
    вопросов ${RU(r.queries)}, цепочек ${RU(r.traces)}</div>`;
}

/* ------------------------------- модель ответа ---------------------------- */
/* ---------------------- прогресс и журнал действий с моделями ------------ */
function mBar(pct, note, color){
  const width = pct==null ? 100 : Math.min(pct,100);
  const anim = pct==null ? 'opacity:.55;' : '';
  return `<div style="background:#2a2f3a;border-radius:6px;height:14px;max-width:460px;position:relative;margin:6px 0">
    <div style="background:${color||'#2f6fb0'};height:14px;border-radius:6px;width:${width}%;${anim}"></div>
    <span style="position:absolute;top:-1px;left:8px;font-size:11px;color:#e6e8ec">${esc(note||'')}</span></div>`;
}
async function loadModelProgress(){
  let d;
  try{ d=await (await fetch('/api/models/progress')).json(); }catch(e){ return; }
  let html='';
  const dl=d.download;
  if(dl && !dl.error && !dl.done){
    html += `<div class="panel" style="margin-top:8px"><b>Загрузка весов: ${esc(dl.model)}</b>
      ${mBar(dl.percent, dl.percent!=null?dl.percent+'%':'подготовка…')}
      <span class="muted">${esc(dl.note||'')}${dl.stale?' · давно нет новостей — смотрите «Конвейер»':''}</span></div>`;
  } else if(dl && dl.error){
    html += `<div class="panel"><span class="bad"><b>Загрузка ${esc(dl.model)} не удалась:</b>
      ${esc(dl.error)}</span></div>`;
  } else if(dl && dl.done){
    html += `<div class="panel good">Загрузка ${esc(dl.model)} завершена.</div>`;
  }
  const sv=d.server||{};
  if(sv.running){
    html += `<div class="panel" style="margin-top:8px"><b>Запуск: ${esc(sv.model||'')}</b>
      (${esc(sv.engine||'')})
      ${sv.ready
        ? mBar(100,'отвечает на '+(sv.base_url||''),'#3fb950')
        : mBar(null,'загружаются веса… '+Math.round((sv.elapsed||0))+' с')}
      ${sv.ready?'' :'<span class="muted">Большая модель поднимается несколько минут; строка позеленеет, когда сервер начнёт отвечать.</span>'}</div>`;
  }
  $('mProgress').innerHTML=html;
  const ACT_RU={'загрузка начата':'⬇','загрузка завершена':'✓','загрузка не удалась':'✗',
    'запуск':'▶','запуск не удался':'✗','остановка':'■'};
  $('mActions').innerHTML=(d.log||[]).length
    ? '<table><tr><th style="width:130px">когда</th><th style="width:170px">действие</th><th>модель</th><th>подробности</th></tr>'
      + d.log.map(a=>`<tr><td class="muted">${esc((a.ts||'').replace('T',' ').slice(5,16))}</td>
        <td>${ACT_RU[a.action]||''} ${esc(a.action)}</td><td><b>${esc(a.model)}</b></td>
        <td class="muted">${esc(a.detail||'')}</td></tr>`).join('')+'</table>'
    : '<span class="muted">Действий с моделями ещё не было.</span>';
}
setInterval(()=>{if($('models').classList.contains('on'))loadModelProgress();},3000);

async function loadLlm(){
  const d=await (await fetch('/api/llm')).json();
  const x=d.describe||{};
  $('llmPrimary').value=d.primary||'local';
  $('llmFallback').value=(d.fallback||'').split(',')[0]||'';
  const chain=(x.chain||[]).map(p=>p===x.active?`<b>${esc(p)}</b>`:esc(p)).join(' → ');
  $('llmPanel').innerHTML=
   `<div>${dot(!!(x.ready&&x.ready.length)&&!x.is_stub)} <b>Цепочка:</b> ${chain}
      ${x.model?`· модель ${esc(x.model)}`:''}</div>`
   + (x.ready||[]).map(r=>`<div class="muted" style="margin-top:5px">
       ✓ ${esc(r.provider)} готов${r.model?', модель '+esc(r.model):''}
       ${r.base_url?' · '+esc(r.base_url):''}</div>`).join('')
   + (x.failed||[]).map(r=>`<div class="bad" style="margin-top:5px">
       ✗ ${esc(r.provider)}: ${esc(r.error)}</div>`).join('')
   + (x.is_stub?`<div class="bad" style="margin-top:8px">Сейчас основной —
      заглушка echo: ответ собирается из найденных предложений, а не пишется
      моделью. Это половина смысла ассистента. Запустите локальную модель ниже
      или выберите облако и укажите ключ.</div>`:'')
   + (d.server&&d.server.running?`<div class="good" style="margin-top:8px">
      Сервер модели работает: ${esc(d.server.model)} (${esc(d.server.engine)})
      на ${esc(d.server.base_url)}, ${Math.round((d.server.uptime_seconds||0)/60)} мин.
      Провайдер local подхватывает этот адрес сам.</div>`
      :`<div class="muted" style="margin-top:8px">Свой сервер модели не запущен.
        Выберите модель в каталоге ниже и нажмите «Запустить», либо укажите адрес
        уже поднятой модели в LOCAL_LLM_BASE_URL.</div>`)
   + ((x.switches||[]).length?`<div class="warn" style="margin-top:8px">
      Переключения на запасного: ${x.switches.map(sw=>
        `${esc(sw.at)} ${esc(sw.from)}→${esc(sw.to)} (${esc(sw.why)})`).join('; ')}</div>`:'');
  renderQueue(d.queue||{}, d.queue_stats||{});
  const sel=$('llmServe');
  const inst=d.installed||[];
  sel.innerHTML = inst.length
    ? inst.map(m=>`<option value="${esc(m.id)}" ${m.serving?'selected':''}>
        ${esc(m.title)} · ${esc(m.params)} · ${m.vram_gb} ГБ · ${esc(m.engine)}${m.serving?' — работает':''}</option>`).join('')
    : '<option value="">нет загруженных — скачайте в каталоге ниже</option>';
  sel.disabled = !inst.length;
}
async function serveAndUse(){
  const id=$('llmServe').value;
  if(!id){ alert('Сначала скачайте модель в каталоге ниже.'); return; }
  if(!confirm('Запустить «'+id+'» и сделать её основной? Текущий сервер модели будет перезапущен.')) return;
  const r=await (await fetch('/api/models/serve',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({id})})).json();
  alert(r.message||'');
  loadLlm();
}
const SRC_RU={'вопрос':'вопрос из чата','голос':'голос и телефон',
  'проверка':'проверка связи','приставки':'смысловые приставки',
  'сканы':'распознавание сканов','фон':'фоновая обработка'};
function renderQueue(q,s){
  const cap=q.limit?`${q.limit} одновременно`:'без ограничения';
  const busy=(q.running||0)+(q.waiting||0);
  const refused=(s.refused||0)+(s.timeout||0);
  const total=s.total||0;
  const share=total+refused?refused/(total+refused)*100:0;
  $('queuePanel').innerHTML=
   `<div>${dot(!q.error&&(total===0||share<5))} <b>${cap}</b>,
      очередь ${q.shared?'общая для всех процессов':'только внутри процесса'}
      · выполняется ${q.running||0}, ждут ${q.waiting||0}</div>`
   + (q.error?`<div class="bad" style="margin-top:6px">${esc(q.error)}</div>`:'')
   + (busy?`<table style="margin-top:8px"><thead><tr><th>Состояние</th>
       <th>Кто спрашивает</th><th>Важность</th><th>Процесс</th><th>Ждёт</th></tr></thead>
       <tbody>${(q.items||[]).map(it=>`<tr>
         <td class="${it.state==='running'?'good':'warn'}">
           ${it.state==='running'?'выполняется':'ждёт'}</td>
         <td>${esc(SRC_RU[it.source]||it.source||'—')}</td>
         <td>${it.priority===0?'срочно':'фоновая'}</td>
         <td class="muted">${it.pid}</td>
         <td>${it.age_s!==null?it.age_s+' с':'—'}</td></tr>`).join('')}</tbody></table>`
      :'<div class="muted" style="margin-top:6px">Сейчас очереди нет.</div>')
   + `<div class="muted" style="margin-top:10px">За сутки: запросов ${RU(total)},
      отказов ${refused}${total+refused?` (${share.toFixed(1)} %)`:''},
      ожидание в очереди в среднем ${RU(s.wait_avg_ms||0)} мс,
      у худших пяти процентов ${RU(s.wait_p95_ms||0)} мс,
      сам запрос к модели в среднем ${RU(s.run_avg_ms||0)} мс.</div>`
   + ((s.by_source||[]).length?`<table style="margin-top:8px"><thead><tr>
       <th>Кто спрашивает</th><th>Запросов</th><th>Ожидание, мс</th>
       <th>Запрос, мс</th></tr></thead><tbody>${s.by_source.map(r=>`<tr>
         <td>${esc(SRC_RU[r.source]||r.source||'—')}</td><td>${RU(r.n)}</td>
         <td>${RU(Math.round(r.wait||0))}</td>
         <td>${RU(Math.round(r.run||0))}</td></tr>`).join('')}</tbody></table>`:'')
   + (share>=5?`<div class="warn" style="margin-top:8px">Отказов заметно много.
      Это значит, что модель не успевает за вопросами. Два решения: поднять
      «Одновременных запросов к модели» на единицу, если есть запас
      видеопамяти — и проверить, что среднее время самого запроса не выросло;
      либо перенести фоновую обработку базы на ночь.</div>`:'')
   + (q.limit===0?`<div class="warn" style="margin-top:8px">Ограничение снято.
      Для облака это нормально. Для локальной модели означает, что при
      одновременных вопросах быстрый ответ не получит никто.</div>`:'');
}
async function loadQueue(){
  const d=await (await fetch('/api/llm/queue')).json();
  renderQueue(d.status||{}, d.stats||{});
}
async function clearQueue(){
  const r=await (await fetch('/api/llm/queue/clear',{method:'POST',
    headers:{'Content-Type':'application/json'},body:'{}'})).json();
  alert(r.message||JSON.stringify(r));
  loadQueue();
}
setInterval(()=>{ if($('models').classList.contains('on')) loadQueue(); },5000);
async function switchLlm(){
  const r=await (await fetch('/api/llm/switch',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({primary:$('llmPrimary').value,
                         fallback:[$('llmFallback').value].filter(Boolean)})})).json();
  if(r.error){ alert(r.error); return; }
  loadLlm();
}
async function probeLlm(){
  $('llmProbe').innerHTML='<span class="muted">Спрашиваю модель…</span>';
  const r=await (await fetch('/api/llm/probe',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({provider:$('llmPrimary').value})})).json();
  $('llmProbe').innerHTML=r.ok
    ? `<div class="panel"><span class="good">Ответила за ${r.ms} мс</span>
       · модель ${esc(r.model||'')} ${r.base_url?'· '+esc(r.base_url):''}
       <div class="muted" style="margin-top:6px">Ответ: ${esc(r.answer)}</div>
       <div class="muted">Токенов: ${r.tokens_in} на вход, ${r.tokens_out} на выход</div></div>`
    : `<div class="panel bad">Не ответила: ${esc(r.error)}</div>`;
}
async function checkModel(id){
  const r=await (await fetch('/api/models/check?id='+encodeURIComponent(id))).json();
  alert((r.ok?'Запустится.':'Не запустится:\n• '+(r.problems||[]).join('\n• '))
    + (r.notes&&r.notes.length?'\n\nУчтите:\n• '+r.notes.join('\n• '):''));
}

/* ------------------------------ разбор ответа ----------------------------- */
async function loadTraces(){
  const find=($('trFind').value||'').trim();
  const q = find ? 'find='+encodeURIComponent(find)
                 : ($('trBad').checked ? 'bad=1' : '');
  const d=await (await fetch('/api/trace?'+q)).json();
  document.querySelector('#traces tbody').innerHTML=(d.traces||[]).map(t=>
    `<tr><td class="muted">${(t.ts||'').replace('T',' ').slice(0,16)}</td>
     <td>${esc(t.question)}</td>
     <td class="muted">${esc(t.route||'')}</td>
     <td>${(t.confidence??0).toFixed(4)}</td>
     <td class="${t.answered?'good':'warn'}">${t.answered?'да':'отказ'}</td>
     <td><button class="act sec" style="padding:3px 9px;font-size:12px"
          onclick="showTrace(${t.id})">разобрать</button></td></tr>`).join('')
     || '<tr><td colspan="6" class="muted">Записей нет. Трассировка включается настройкой TRACE_ENABLED.</td></tr>';
}
async function showTrace(id){
  const t=await (await fetch('/api/trace?id='+id)).json();
  if(!t.id){ $('traceOut').innerHTML='<div class="bad">Цепочка не найдена.</div>'; return; }
  $('traceOut').innerHTML=`<div class="panel" style="margin-top:10px">
    <div><b>${esc(t.question)}</b></div>
    <div class="muted" style="margin-top:4px">${(t.ts||'').replace('T',' ').slice(0,16)}
      · маршрут ${esc(t.route||'')} · этап ${esc(STAGE_RU[t.stage]||t.stage||'')}
      · уверенность ${(t.confidence??0).toFixed(5)} · модель ${esc(t.model||'—')}
      ${t.user_name?'· спросил '+esc(t.user_name):''}</div>
    <div class="muted" style="margin-top:4px">Настройки на тот момент:
      ${Object.entries(t.settings||{}).map(([k,v])=>k+'='+v).join(' · ')}</div>
    <h3 style="margin-top:14px">Что нашлось</h3>
    <table><thead><tr><th style="width:34px">#</th><th>Фрагмент</th>
      <th style="width:80px">Оценка</th><th style="width:230px">По каналам</th></tr></thead><tbody>`
    + (t.hits||[]).map((h,i)=>`<tr><td>${i+1}</td>
        <td><code>${esc(h.path)}</code>
          <div class="muted" style="font-size:12px">${esc(h.text)}</div></td>
        <td>${h.score}</td>
        <td class="muted" style="font-size:12px">${Object.entries(h.channels||{})
          .map(([k,v])=>k+' '+(v==null?'—':(+v).toFixed(3))).join('<br>')}</td></tr>`).join('')
    + `</tbody></table>`
    + (t.prompt?`<h3 style="margin-top:14px">Что ушло в модель</h3>
        <pre style="max-height:280px">${esc(t.prompt)}</pre>`:'')
    + `<h3 style="margin-top:14px">Ответ</h3><pre>${esc(t.answer||'')}</pre>
    <div class="toolbar"><button class="act" onclick="goldenFromTrace(${t.id})">
      Сделать выверенным ответом</button></div></div>`;
  window.__trace=t;
}
function goldenFromTrace(id){
  const t=window.__trace; if(!t) return;
  document.querySelector('nav button[data-t=queries]').click();
  $('gQ').value=t.question; $('gA').value=t.answer||'';
  $('gQ').scrollIntoView({behavior:'smooth'});
}

/* ----------------------------------- аналитика ---------------------------- */
let GAPS=null;
async function loadAnalytics(){
  const h=$('anHours').value;
  const d=await (await fetch('/api/analytics?hours='+h)).json();
  const f=d.funnel;
  if(!f.total){ $('funnel').innerHTML='<span class="muted">Вопросов за период не было.</span>';
    $('funnelLoss').innerHTML=''; }
  else {
    $('funnel').innerHTML=f.steps.map(st=>{
      const w=Math.max(st.share*100,0.6);
      const drop=st.lost?` <span class="warn">−${RU(st.lost)}</span>`:'';
      return `<div style="margin-bottom:9px">
        <div style="display:flex;justify-content:space-between;font-size:13px">
          <span><b>${esc(st.title)}</b> <span class="muted">${esc(st.note)}</span></span>
          <span>${RU(st.n)} · ${(st.share*100).toFixed(0)}%${drop}</span></div>
        <div style="height:9px;background:#232833;border-radius:5px;margin-top:4px;overflow:hidden">
          <i style="display:block;height:100%;width:${w}%;background:#4E79A7"></i></div></div>`;
    }).join('');
    $('funnelLoss').innerHTML=f.losses.length
      ? '<div class="panel">'+f.losses.map(l=>
          `<div style="margin-bottom:8px"><b class="warn">${esc(l.where)}: ${RU(l.n)}</b>
           <div class="muted">${esc(l.what_to_do)}</div></div>`).join('')+'</div>'
      : '<div class="panel good">Заметных потерь между ступенями нет.</div>';
  }
  document.querySelector('#routes tbody').innerHTML=(f.routes||[]).map(r=>
    `<tr><td>${esc(d.route_ru[r.route]||r.route)}</td><td>${RU(r.n)}</td>
     <td>${(r.share*100).toFixed(0)}%</td><td>${RU(r.ok)}</td><td>${r.ms}</td></tr>`).join('')
    || '<tr><td colspan="5" class="muted">Нет данных.</td></tr>';

  const c=d.confidence;
  if(!c.total){ $('confHist').innerHTML='<span class="muted">Данных пока нет.</span>';
                $('confAdvice').innerHTML=''; }
  else {
    const peak=Math.max(...c.bins.map(b=>b.n))||1;
    $('confHist').innerHTML=
      `<div style="display:flex;align-items:flex-end;gap:2px;height:150px">`
      + c.bins.map(b=>{
          const isT=b.from<=c.threshold&&c.threshold<b.to;
          const up=b.up, down=b.down, other=b.n-up-down;
          const px=v=>Math.round(v/peak*140);
          return `<div title="${b.from}–${b.to}: ${b.n} (👍${up} 👎${down})"
            style="flex:1;display:flex;flex-direction:column;justify-content:flex-end;
            ${isT?'outline:2px solid #E15759;outline-offset:1px':''}">
            <div style="height:${px(down)}px;background:#E15759"></div>
            <div style="height:${px(up)}px;background:#59A14F"></div>
            <div style="height:${px(other)}px;background:#3d4756"></div></div>`;
        }).join('') + `</div>
      <div class="muted" style="display:flex;justify-content:space-between;margin-top:5px">
        <span>${c.min}</span><span>порог ${c.threshold} · медиана ${c.median}</span>
        <span>${c.max}</span></div>
      <div class="muted" style="margin-top:8px">
        <span style="color:#59A14F">■</span> оценено как полезное ·
        <span style="color:#E15759">■</span> оценено как неверное ·
        <span style="color:#8b93a3">■</span> без оценки</div>`;
    let advice=`<div class="panel"><div>Ниже порога: <b>${RU(c.below_threshold)}</b> из ${RU(c.total)}`;
    if(c.good_below) advice+=` · <span class="bad">из них ${c.good_below} были оценены
      положительно — эти ответы порог отсёк напрасно</span>`;
    if(c.bad_above) advice+=` · <span class="warn">выше порога прошло ${c.bad_above}
      ответов, оценённых как неверные</span>`;
    advice+=`</div><table style="margin-top:10px"><thead><tr><th>Порог</th>
      <th>Отсекает</th><th>Из них полезных</th><th>Из них неверных</th></tr></thead><tbody>`
      + c.suggestions.map(sg=>`<tr${c.recommended&&sg.threshold===c.recommended.threshold
          ?' style="background:#1c2530"':''}><td>${sg.threshold}</td>
        <td>${RU(sg.cuts)} (${(sg.cuts_share*100).toFixed(0)}%)</td>
        <td class="${sg.cuts_good?'bad':''}">${sg.cuts_good}</td>
        <td class="good">${sg.cuts_bad}</td></tr>`).join('')
      + `</tbody></table>`;
    if(c.recommended) advice+=`<div style="margin-top:8px">По накопленным оценкам
      лучше всего разделяет полезные и неверные ответы порог
      <b>${c.recommended.threshold}</b>. Ставить его стоит, только если оценок
      набралось достаточно: сейчас их ${RU(c.rated)}.</div>`;
    else advice+=`<div class="warn" style="margin-top:8px">Оценок пока нет — подобрать
      порог по данным не на чем. Напомните сотрудникам про кнопки под ответом.</div>`;
    $('confAdvice').innerHTML=advice+'</div>';
  }

  const ch=d.channels;
  $('chCards').innerHTML=(ch.channels||[]).map(x=>
    `<div class="card"><div class="v">${(x.share*100).toFixed(0)}%</div>
     <div class="k">${esc(x.title)} · ${RU(x.n)}</div></div>`).join('')
    || '<span class="muted">Нет данных.</span>';
  $('chVerdict').innerHTML=`<div>${esc(ch.verdict)}</div>`
    + `<div class="muted" style="margin-top:6px">Переранжирование участвовало
       в ${RU(ch.rerank_used)} из ${RU(ch.rerank_of)} ответов.</div>`;

  loadRegression(); loadAdminLog();
}

async function loadGaps(){
  $('gapsOut').innerHTML='<span class="muted">Группирую…</span>';
  const d=await (await fetch('/api/analytics/gaps?hours='+$('anHours').value
    +'&min='+($('gapMin').value||2))).json();
  GAPS=d;
  if(!d.total){ $('gapsOut').innerHTML='<div class="panel good">Вопросов без ответа за период нет.</div>'; return; }
  $('gapsOut').innerHTML=
    `<div class="panel" style="margin-bottom:10px">Без ответа: <b>${RU(d.total)}</b> ·
      сгруппировано в темы: <b>${RU(d.grouped)}</b> ·
      одиночных: ${RU(d.singles_total)}</div>`
    + (d.groups||[]).map(g=>`<div class="panel" style="margin-bottom:8px">
        <div style="display:flex;justify-content:space-between">
          <b>${esc(g.title)}</b><span class="warn">${g.size} вопросов</span></div>
        <div class="muted" style="font-size:12px;margin-top:3px">
          ${Object.entries(g.stages).map(([k,v])=>esc(STAGE_RU[k]||k)+': '+v).join(' · ')}
          · последний ${(g.last_at||'').replace('T',' ').slice(0,16)}</div>
        <ul style="margin:8px 0 0;padding-left:18px">
          ${g.questions.map(q=>`<li style="margin-bottom:3px">${esc(q.text)}
            <span class="muted" style="font-size:11px">${esc(q.who||'')}</span></li>`).join('')}
        </ul></div>`).join('')
    + (d.singles.length?`<div class="panel"><b>Одиночные вопросы</b>
        <ul style="margin:8px 0 0;padding-left:18px">${d.singles.map(q=>
          `<li>${esc(q.text)}</li>`).join('')}</ul></div>`:'');
}
const STAGE_RU={nothing_found:'ничего не нашлось',low_confidence:'низкая уверенность',
  bad_feedback:'оценён как неверный',answered:'отвечено'};
function exportGaps(){
  if(!GAPS){ alert('Сначала соберите группы.'); return; }
  const lines=['Чего не хватает в базе знаний',''];
  (GAPS.groups||[]).forEach((g,i)=>{
    lines.push(`${i+1}. ${g.title} — ${g.size} вопросов`);
    g.questions.forEach(q=>lines.push('   • '+q.text));
    lines.push('');
  });
  if(GAPS.singles.length){ lines.push('Одиночные вопросы:');
    GAPS.singles.forEach(q=>lines.push('   • '+q.text)); }
  const blob=new Blob([lines.join('\n')],{type:'text/plain;charset=utf-8'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob); a.download='пробелы-базы-знаний.txt'; a.click();
}

async function loadRegression(){
  const d=await (await fetch('/api/regression')).json();
  const changed=Object.entries(d.pending_changes||{});
  $('regPanel').innerHTML = !d.dataset_exists
    ? `<div class="warn">Нет набора контрольных вопросов (<code>${esc(d.dataset)}</code>).
       Соберите 50–150 реальных вопросов сотрудников с указанием, где лежит ответ:
       без него качество поиска нечем измерять, и все остальные настройки
       подбираются вслепую.</div>`
    : (d.history.length
        ? `<div>Последний прогон: <b>${(d.history[0].created_at||'').replace('T',' ').slice(0,16)}</b>
           · hit ${d.history[0].hit} · MRR ${d.history[0].mrr}
           · повод: ${esc(d.history[0].reason)}</div>`
          + (d.diff && !d.diff.error ? `<div style="margin-top:6px">Относительно предыдущего:
             MRR ${d.diff.delta.mrr>=0?'+':''}${d.diff.delta.mrr}
             ${d.diff.broken.length?`· <span class="bad">перестали находиться:
             ${d.diff.broken.length}</span>`:''}
             ${d.diff.fixed.length?`· <span class="good">стали находиться:
             ${d.diff.fixed.length}</span>`:''}</div>`
             + (d.diff.broken.length?`<ul class="muted" style="margin:6px 0 0;padding-left:18px">
                ${d.diff.broken.slice(0,6).map(q=>`<li>${esc(q)}</li>`).join('')}</ul>`:'')
            : '')
        : '<span class="muted">Прогонов ещё не было.</span>')
    + (changed.length?`<div class="warn" style="margin-top:8px">После последнего прогона
        изменились настройки: ${changed.map(([k,v])=>`${esc(k)} ${esc(String(v[0]))}→${esc(String(v[1]))}`).join(', ')}.
        Стоит прогнать заново.</div>`:'');
  document.querySelector('#regHistory tbody').innerHTML=(d.history||[]).map(r=>
    `<tr><td class="muted">${(r.created_at||'').replace('T',' ').slice(0,16)}</td>
     <td>${r.hit}</td><td>${r.mrr}</td><td>${r.questions}</td>
     <td class="muted">${esc(r.reason)}</td></tr>`).join('')
     || '<tr><td colspan="5" class="muted">Прогонов не было.</td></tr>';
}

async function loadAdminLog(){
  const d=await (await fetch('/api/adminlog')).json();
  document.querySelector('#adminLog tbody').innerHTML=(d.entries||[]).map(e=>
    `<tr><td class="muted">${(e.ts||'').replace('T',' ').slice(0,16)}</td>
     <td class="muted">${esc(e.who)}</td><td>${esc(e.action)}</td>
     <td>${esc(e.detail)}</td></tr>`).join('')
     || '<tr><td colspan="4" class="muted">Действий пока не было.</td></tr>';
}

/* ---------------------------------- сотрудники ---------------------------- */
let USERS={users:[],roles:[]};
async function loadUsers(){
  USERS=await (await fetch('/api/users')).json();
  const s=USERS.summary;
  $('uCards').innerHTML=[
    ['Ждут решения',s.counts.pending],['Есть доступ',s.counts.approved],
    ['Отказано',s.counts.denied],['Заблокированы',s.counts.blocked],
    ['Всего обращались',s.total],
  ].map(([k,v])=>`<div class="card${k==='Ждут решения'&&v?' warn':''}">
    <div class="v">${RU(v)}</div><div class="k">${k}</div></div>`).join('');
  $('uWarn').innerHTML = USERS.from_env
    ? `<span class="muted">Список TELEGRAM_ALLOWED_IDS заполнен и имеет приоритет:
       перечисленные в нём проходят без подтверждения.</span>`
    : (s.counts.approved
        ? '<span class="good">Доступ выдаётся поимённо — так и должно быть.</span>'
        : `<span class="bad">Ни один сотрудник не подтверждён и список
           TELEGRAM_ALLOWED_IDS пуст. Проверьте, что бот не отвечает
           посторонним.</span>`);
  document.querySelector('#uRoles tbody').innerHTML=
    Object.entries(USERS.role_sections||{}).map(([role,secs])=>
      `<tr><td><b>${esc(role)}</b></td><td class="muted">${secs.map(esc).join(', ')||'—'}</td></tr>`).join('')
      || '<tr><td colspan="2" class="muted">Роли не настроены.</td></tr>';
  renderUsers();
}
function roleSelect(u,id){
  return `<select id="${id}">${(USERS.roles||[]).map(r=>
    `<option ${u.role===r?'selected':''}>${esc(r)}</option>`).join('')}</select>`;
}
function renderUsers(){
  const pending=USERS.users.filter(u=>u.status==='pending');
  document.querySelector('#uPending tbody').innerHTML=pending.map(u=>
    `<tr><td><b>${esc(u.full_name||'без имени')}</b>
       ${u.user_name?`<div class="muted">@${esc(u.user_name)}</div>`:''}
       <div class="muted" style="font-size:11px">обратился
         ${(u.requested_at||'').replace('T',' ').slice(0,16)}</div></td>
     <td><code>${u.user_id}</code></td>
     <td class="muted">${esc(u.note||'—')}</td>
     <td>${roleSelect(u,'role_'+u.user_id)}</td>
     <td><button class="act" onclick="decideUser(${u.user_id},true)">Выдать доступ</button>
         <button class="act sec" onclick="decideUser(${u.user_id},false)">Отклонить</button></td></tr>`).join('')
    || '<tr><td colspan="5" class="muted">Заявок нет.</td></tr>';

  const f=($('uFilter').value||'').toLowerCase(), st=$('uStatus').value;
  const rows=USERS.users.filter(u=>(!st||u.status===st) &&
    (!f||((u.full_name||'')+(u.user_name||'')+u.user_id).toLowerCase().includes(f)));
  document.querySelector('#uAll tbody').innerHTML=rows.map(u=>{
    const cls=u.status==='approved'?'good':u.status==='pending'?'warn':
              u.status==='blocked'?'bad':'muted';
    return `<tr><td>${esc(u.full_name||'без имени')}
       ${u.user_name?`<span class="muted">@${esc(u.user_name)}</span>`:''}
       ${u.is_admin?'<span class="muted"> · администратор</span>':''}
       ${u.from_env?'<span class="muted"> · из настроек</span>':''}</td>
     <td><code>${u.user_id}</code></td>
     <td class="${cls}">${esc(u.status_ru)}</td>
     <td>${roleSelect(u,'arole_'+u.user_id)}
         <button class="act sec" style="padding:4px 8px;font-size:12px"
           onclick="saveRole(${u.user_id},'arole_${u.user_id}')">ok</button></td>
     <td>${RU(u.asked)}</td>
     <td class="muted">${(u.last_q||u.last_seen||'').replace('T',' ').slice(0,16)}</td>
     <td>${u.status==='approved'
        ? `<button class="act bad" onclick="blockUser(${u.user_id})">Закрыть доступ</button>`
        : `<button class="act" onclick="decideUser(${u.user_id},true,'arole_${u.user_id}')">Выдать доступ</button>`}
     </td></tr>`;}).join('')
    || '<tr><td colspan="7" class="muted">Никто ещё не обращался.</td></tr>';
}
async function decideUser(id,approve,roleField){
  const sel=$(roleField||('role_'+id));
  const r=await (await fetch('/api/users/decide',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({user_id:id,approve,role:sel?sel.value:null})})).json();
  loadUsers();
}
async function blockUser(id){
  if(!confirm('Закрыть доступ этому сотруднику?')) return;
  await fetch('/api/users/block',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({user_id:id})});
  loadUsers();
}
async function saveRole(id,field){
  await fetch('/api/users/role',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({user_id:id,role:$(field).value})});
  loadUsers();
}

/* ------------------------------------ телеграм ---------------------------- */
async function loadTelegram(){
  USERS=await (await fetch('/api/users')).json();
  const s=USERS.summary;
  const trainers=USERS.users.filter(u=>u.trainer).length;
  $('tgCards').innerHTML=[
    ['Ждут решения',s.counts.pending],['Есть доступ',s.counts.approved],
    ['Дообучают бота',trainers],['Всего обращались',s.total],
  ].map(([k,v])=>`<div class="card${k==='Ждут решения'&&v?' warn':''}">
    <div class="v">${RU(v)}</div><div class="k">${k}</div></div>`).join('');

  const pending=USERS.users.filter(u=>u.status==='pending');
  document.querySelector('#tgPending tbody').innerHTML=pending.map(u=>
    `<tr><td><b>${esc(u.full_name||'без имени')}</b>
       ${u.user_name?`<div class="muted">@${esc(u.user_name)}</div>`:''}
       <div class="muted" style="font-size:11px">обратился
         ${(u.requested_at||'').replace('T',' ').slice(0,16)}</div></td>
     <td><code>${u.user_id}</code></td>
     <td class="muted">${esc(u.note||'—')}</td>
     <td>${roleSelect(u,'tgrole_'+u.user_id)}</td>
     <td><button class="act" onclick="tgDecide(${u.user_id},true)">Разрешить</button>
         <button class="act sec" onclick="tgDecide(${u.user_id},false)">Отклонить</button></td></tr>`).join('')
    || '<tr><td colspan="5" class="muted">Заявок нет. Сотрудник оставляет заявку командой /request в боте.</td></tr>';

  const known=USERS.users.filter(u=>u.status==='approved'||u.trainer);
  document.querySelector('#tgTrainers tbody').innerHTML=known.map(u=>
    `<tr><td>${esc(u.full_name||'без имени')}
       ${u.user_name?`<span class="muted">@${esc(u.user_name)}</span>`:''}</td>
     <td><code>${u.user_id}</code></td>
     <td class="muted">${esc(u.role||'—')}</td>
     <td><label><input type="checkbox" ${u.trainer?'checked':''}
          onchange="setTrainer(${u.user_id},this.checked)">
          ${u.trainer?'<span class="good">учит бота</span>':'выключено'}</label></td>
     <td>${RU(u.taught||0)}</td></tr>`).join('')
    || '<tr><td colspan="5" class="muted">Пока некому включать: нет сотрудников с доступом.</td></tr>';
}
async function tgDecide(id,approve){
  const sel=$('tgrole_'+id);
  await fetch('/api/users/decide',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({user_id:id,approve,role:sel?sel.value:null})});
  loadTelegram();
}
async function setTrainer(id,on){
  await fetch('/api/users/trainer',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({user_id:id,on})});
  loadTelegram();
}

/* ------------------------------------ журналы ----------------------------- */
async function loadLogLevels(){
  const d=await (await fetch('/api/log/levels')).json();
  $('logLevels').innerHTML=Object.entries(d).map(([name,info])=>
    `<div style="display:flex;gap:12px;align-items:center;padding:5px 0;border-bottom:1px solid #232833">
      <div style="width:110px"><b>${esc(name)}</b></div>
      <div class="muted" style="flex:1;font-size:12.5px">${esc(info.description)}</div>
      <select onchange="setLevel('${name}',this.value)" style="width:130px">
        ${['TRACE','DEBUG','INFO','WARNING','ERROR'].map(l=>
          `<option ${info.level===l?'selected':''}>${l}</option>`).join('')}</select></div>`).join('');
}
async function setLevel(sub,level){
  await fetch('/api/log/levels',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({subsystem:sub,level})});
}
async function loadLogs(){
  const params={lines:$('lgLines').value||'300',
    level:$('lgLevel').value,subsystem:$('lgSub').value};
  if($('lgQ').value.trim()) params.q=$('lgQ').value.trim();
  const d=await (await fetch('/api/log?'+new URLSearchParams(params))).json();
  $('logOut').textContent=d.lines.join('\n')
    ||(params.q?'Ничего не найдено по «'+params.q+'» — поиск шёл по всей глубине журналов.':'Пусто.');
  $('logOut').scrollTop=$('logOut').scrollHeight;
}
setInterval(()=>{if($('lgAuto').checked&&$('logs').classList.contains('on'))loadLogs();},4000);

/* ------------------------------------ запросы ----------------------------- */
async function loadQueries(){
  const r=await (await fetch('/api/queries')).json();
  document.querySelector('#qtable tbody').innerHTML=r.recent.map(q=>
    `<tr><td class="muted">${(q.created_at||'').replace('T',' ').slice(5,19)}</td>
     <td>${esc(q.user_name)||'—'}</td><td>${esc(q.question)}</td>
     <td>${(q.top_score||0).toFixed(4)}</td><td>${q.latency_ms||0}</td>
     <td>${q.verdict==='up'?'👍':q.verdict==='down'?'👎':''}</td></tr>`).join('');
  document.querySelector('#gaps tbody').innerHTML=r.gaps.map(q=>
    `<tr><td class="muted">${(q.created_at||'').replace('T',' ').slice(5,19)}</td>
     <td>${esc(q.question)}</td>
     <td><button class="act sec" onclick="fillGolden('${esc(q.question).replace(/'/g,'')}')">Ответить</button></td></tr>`).join('')
     || '<tr><td colspan="3" class="muted">Все вопросы получили ответ.</td></tr>';
  document.querySelector('#golden tbody').innerHTML=r.golden.map(g=>
    `<tr><td>${esc(g.question)}</td><td>${esc(g.answer).slice(0,160)}</td><td>${g.hits}</td></tr>`).join('')
    || '<tr><td colspan="3" class="muted">Пока ни одного.</td></tr>';
}
function fillGolden(q){$('gQ').value=q;document.querySelector('nav button[data-t=queries]').click();
  $('gA').focus();}
async function addGolden(){
  const r=await (await fetch('/api/golden',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({question:$('gQ').value,answer:$('gA').value})})).json();
  alert(r.message); $('gQ').value='';$('gA').value='';loadQueries();
}

/* --------------------------------- диагностика ---------------------------- */
async function loadDiag(){
  const d=await (await fetch('/api/diagnostics')).json();
  document.querySelector('#diagTable tbody').innerHTML=d.checks.map(c=>{
    const mark=c.ok===true?'<span class="good">в порядке</span>':
      c.ok===false?'<span class="bad">проблема</span>':'<span class="warn">не настроено</span>';
    return `<tr><td>${esc(c.name)}</td><td>${mark}</td><td class="muted">${esc(c.detail)}</td>
      <td class="muted">${c.ok===true?'':esc(c.hint)}</td></tr>`;}).join('');
}
</script></body></html>
"""


LOGIN_PAGE = r"""<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Вход · Ассистент базы знаний</title><style>
body{margin:0;height:100vh;display:flex;align-items:center;justify-content:center;
 background:#0f1116;color:#e6e8ec;font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
form{background:#171a21;border:1px solid #262a33;border-radius:12px;padding:28px 30px;width:340px}
h1{font-size:16px;margin:0 0 4px}
p.sub{color:#8b93a3;font-size:13px;margin:0 0 20px}
label{display:block;font-size:12px;color:#8b93a3;margin:14px 0 5px}
input{width:100%;background:#0f1116;border:1px solid #2c313c;color:#e6e8ec;
 border-radius:8px;padding:9px 11px;font-size:14px;box-sizing:border-box}
button{width:100%;margin-top:20px;background:#2f6fb0;border:0;color:#fff;
 padding:10px;border-radius:8px;cursor:pointer;font-size:14px}
.err{color:#E15759;font-size:13px;margin-top:12px;min-height:18px}
</style></head><body>
<form onsubmit="enter(event)">
  <h1>Ассистент базы знаний</h1>
  <p class="sub">Вход в администрирование</p>
  <label>Логин</label><input id="login" autofocus autocomplete="username">
  <label>Пароль</label><input id="password" type="password" autocomplete="current-password">
  <button type="submit">Войти</button>
  <div class="err" id="err"></div>
</form>
<script>
async function enter(e){
  e.preventDefault();
  const r = await (await fetch('/api/login',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({login:document.getElementById('login').value,
                         password:document.getElementById('password').value})})).json();
  if(r.ok){ location.href='/'; }
  else { document.getElementById('err').textContent = r.error || 'не вышло'; }
}
</script></body></html>"""


TOKEN_PAGE = r"""<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Вход · Ассистент базы знаний</title><style>
body{margin:0;height:100vh;display:flex;align-items:center;justify-content:center;
 background:#0f1116;color:#e6e8ec;font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
form{background:#171a21;border:1px solid #262a33;border-radius:12px;padding:28px 30px;width:340px}
h1{font-size:16px;margin:0 0 4px}
p.sub{color:#8b93a3;font-size:13px;margin:0 0 20px}
label{display:block;font-size:12px;color:#8b93a3;margin:14px 0 5px}
input{width:100%;background:#0f1116;border:1px solid #2c313c;color:#e6e8ec;
 border-radius:8px;padding:9px 11px;font-size:14px;box-sizing:border-box}
button{width:100%;margin-top:20px;background:#2f6fb0;border:0;color:#fff;
 border-radius:8px;padding:10px;font-size:14px;cursor:pointer}
.err{color:#f85149;font-size:13px;margin-top:12px;min-height:18px}
p.hint{color:#8b93a3;font-size:12px;margin:16px 0 0}
</style></head><body>
<form onsubmit="enter(event)">
  <h1>Ассистент базы знаний</h1>
  <p class="sub">Вход в администрирование</p>
  <label>Пароль администратора</label>
  <input id="token" type="password" autofocus autocomplete="current-password">
  <button type="submit">Войти</button>
  <div class="err" id="err"></div>
  <p class="hint">Пароль создан при установке (ADMIN_TOKEN в файле .env
  папки установки) — его печатал установщик.</p>
</form>
<script>
async function enter(e){
  e.preventDefault();
  const r = await (await fetch('/api/token-login',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({token:document.getElementById('token').value})})).json();
  if(r.ok){ location.href='/'; }
  else { document.getElementById('err').textContent = r.error || 'не вышло'; }
}
</script></body></html>"""


# --------------------------------------------------------------- сервер -----
class Handler(BaseHTTPRequestHandler):
    server_version = "KBAdmin/2.0"

    def log_message(self, *args) -> None:
        pass

    # ------------------------------------------------------------ доступ ----
    def _cookie(self, name: str) -> str:
        raw = self.headers.get("Cookie") or ""
        for part in raw.split(";"):
            if "=" in part:
                key, value = part.split("=", 1)
                if key.strip() == name:
                    return value.strip()
        return ""

    def _account(self) -> dict | None:
        import security
        return security.session(self._cookie("kb_session"))

    def _authorized(self) -> bool:
        """
        Пускать ли этот запрос.

        Порядок такой. Если заведены учётные записи — только они, и роль
        решает, что можно делать. Если нет, но задан общий токен — по
        токену. Если нет ни того ни другого — только с локального адреса:
        открытая наружу админка без всякой проверки означает, что любой,
        кто дотянулся до порта, может восстановить индекс из копии.

        Отдельно про «локальный адрес». Когда перед админкой стоит
        обратный прокси на той же машине — обычная и рекомендуемая
        схема, — адресом клиента для всех внешних запросов становится
        127.0.0.1. То есть правило «с локального адреса можно всё»
        открывало админку без пароля всему интернету, причём именно в той
        конфигурации, которую сами же и советуем. Теперь при
        ADMIN_TRUST_PROXY признаком «свой» служит не адрес соединения, а
        адрес из X-Forwarded-For, и пускать он не будет никого, кроме
        локальных; а вход по паролю в такой схеме нужен обязательно.
        """
        import security
        path = urllib.parse.urlparse(self.path).path
        if path in ("/login", "/api/login", "/api/token-login", "/healthz"):
            return True
        if security.accounts_enabled():
            account = self._account()
            if account is None:
                return False
            # Тело запроса нужно правилам: один и тот же путь /api/job
            # запускает и переиндексацию, и восстановление из копии.
            payload = self._peek_body() if self.command == "POST" else None
            return security.may(account["role"], self.command, path, payload)
        if config.ADMIN_TOKEN:
            # Токен принимается заголовком или куком. В query-строке он
            # оседает в журналах прокси, в истории браузера и в Referer
            # при первом же переходе по внешней ссылке, поэтому из ссылки
            # он принимается ровно один раз — при открытии страницы, где
            # тут же перекладывается в кук, а адрес очищается.
            given = self.headers.get("X-Admin-Token") or self._cookie("kb_token")
            return hmac.compare_digest(given, config.ADMIN_TOKEN)
        if not self._client_is_local():
            log.error("отказано в доступе к админке с адреса %s: ни учётных записей, "
                      "ни токена не настроено", self._client_ip())
            return False
        return True

    def _client_ip(self) -> str:
        """Настоящий адрес клиента с учётом обратного прокси."""
        direct = self.client_address[0] if self.client_address else ""
        if config.ADMIN_TRUST_PROXY:
            forwarded = (self.headers.get("X-Forwarded-For") or "").split(",")
            if forwarded and forwarded[0].strip():
                return forwarded[0].strip()
        return direct

    def _client_is_local(self) -> bool:
        return self._client_ip() in ("127.0.0.1", "::1", "localhost")

    def _safe_job(self, job: dict | None) -> dict | None:
        """
        Карточка задачи без внутренних подробностей.

        В `error` у упавшей задачи лежит полная трассировка: пути в
        файловой системе, имена модулей, куски окружения. Смотреть на неё
        может любая роль — все GET открыты, — а чинить всё равно будет
        администратор, у которого есть журнал. Поэтому ролям ниже admin
        отдаём первую строку ошибки, без стека.
        """
        if not job:
            return job
        import security
        account = self._account()
        role = (account or {}).get("role") if security.accounts_enabled() else "admin"
        if role == "admin" or not job.get("error"):
            return job
        first = str(job["error"]).strip().splitlines()[-1][:200]
        return {**job, "error": first,
                "error_note": "полный текст ошибки виден администратору"}

    def _who(self) -> str:  # noqa: D401
        """
        Кто выполнил действие — для журнала.

        Пишем только то, что действительно установлено. Имя из учётной
        записи — установлено: человек ввёл пароль. Имя из заголовка
        X-Admin-User — установлено лишь тогда, когда его проставил свой
        обратный прокси; сам по себе это обычный клиентский заголовок,
        который подставляется одной строкой curl. Раньше он принимался
        всегда, то есть запись в журнале о восстановлении индекса
        указывала на того, на кого захотел указать отправитель, — журнал
        аудита, ради которого всё и заводилось, подделывался тривиально.
        Теперь заголовку верим только при ADMIN_TRUST_PROXY.
        """
        account = self._account()
        addr = self._client_ip() or "?"
        if account:
            return f"{account.get('full_name') or account['login']} ({addr})"
        if config.ADMIN_TRUST_PROXY:
            name = (self.headers.get("X-Admin-User")
                    or self.headers.get("X-Forwarded-User"))
            if name:
                return f"{name[:60]} ({addr}, по данным прокси)"
        return f"админка, {addr}"

    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def _json(self, data, code: int = 200) -> None:
        self._send(code, json.dumps(data, ensure_ascii=False, default=str).encode())

    def _body(self) -> dict:
        return self._peek_body()

    def _peek_body(self) -> dict:
        """
        Тело запроса, прочитанное один раз.

        Читать `rfile` дважды нельзя, а тело нужно и проверке прав (там
        решает вид задачи), и самому обработчику. Поэтому читаем один раз
        и запоминаем.
        """
        cached = getattr(self, "_body_cache", None)
        if cached is not None:
            return cached
        length = int(self.headers.get("Content-Length") or 0)
        body: dict = {}
        if length:
            try:
                body = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, ValueError):
                body = {}
        if not isinstance(body, dict):
            body = {}
        self._body_cache = body
        return body

    def _csrf_ok(self) -> bool:
        """
        Защита от запроса, отправленного чужой страницей.

        Требуем `Content-Type: application/json`. Обычная HTML-форма
        такой заголовок поставить не может, а запрос из скрипта на чужом
        сайте с ним требует разрешения CORS, которого мы не даём. Проверка
        нужна во всех режимах: `SameSite=Strict` у куки закрывает только
        вход по учётной записи, а в режимах «общий токен» и «только с
        локального адреса» куки нет вовсе — там авторизует сам факт
        запроса с этой машины. То есть страница, открытая администратором
        в соседней вкладке, могла отправить `POST /api/job` с
        восстановлением индекса, и это сработало бы.
        """
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        return ctype == "application/json"

    # ------------------------------------------------------------- GET ------
    def do_GET(self) -> None:  # noqa: N802, C901
        parsed = urllib.parse.urlparse(self.path)
        path, query = parsed.path, urllib.parse.parse_qs(parsed.query)
        import security

        # Проверка живости — без авторизации и без обращения к базе.
        # Иначе получается ловушка: как только админку закрывают токеном,
        # проверка живости контейнера начинает получать 401, докер считает
        # контейнер больным и перезапускает его по кругу.
        if path == "/healthz":
            return self._send(200, b'{"ok": true}')

        # Метрики Prometheus — с локального адреса без авторизации:
        # скрейпер стоит рядом и токенов не умеет, а секретов в метриках
        # нет. Снаружи — только с токеном, как и всё остальное.
        if path == "/metrics" and self._client_is_local():
            import metrics
            return self._send(200, metrics.prometheus().encode(), "text/plain")

        # Общий токен из ссылки принимается один раз: перекладываем его в
        # кук и очищаем адрес, чтобы дальше он не светился ни в истории
        # браузера, ни в Referer, ни в журнале обратного прокси.
        if config.ADMIN_TOKEN and not security.accounts_enabled() \
                and query.get("token") and path in ("/", "/index.html"):
            if hmac.compare_digest(query["token"][0], config.ADMIN_TOKEN):
                secure = "; Secure" if _behind_https(self) else ""
                self.send_response(303)
                self.send_header("Location", "/")
                self.send_header("Set-Cookie",
                                 f"kb_token={config.ADMIN_TOKEN}; Path=/; HttpOnly; "
                                 f"SameSite=Strict; Max-Age=43200{secure}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return None

        if not self._authorized():
            if path in ("/", "/index.html", "/login"):
                # Человеку — форма входа, а не голое «unauthorized»:
                # с учётными записями своя, в токен-режиме — поле пароля.
                if security.accounts_enabled():
                    return self._send(200, LOGIN_PAGE.encode(), "text/html")
                if config.ADMIN_TOKEN:
                    return self._send(200, TOKEN_PAGE.encode(), "text/html")
            return self._send(401, b"unauthorized", "text/plain")

        if path in ("/", "/index.html"):
            return self._send(200, PAGE.encode(), "text/html")
        if path == "/login":
            return self._send(200, LOGIN_PAGE.encode(), "text/html")
        if path == "/api/whoami":
            account = self._account()
            return self._json({
                "accounts": security.accounts_enabled(),
                "default_password": security.default_password_active(),
                "account": account,
                "role": (account or {}).get("role"),
                "roles": security.ROLES,
                "users": security.list_users() if (account or {}).get("role") == "admin"
                else [],
                "secrets": security.secrets_health()
                if (account or {}).get("role") == "admin"
                or not security.accounts_enabled() else {},
            })
        if path == "/api/quickstart":
            import quickstart
            return self._json(quickstart.state())
        if path == "/api/eval":
            import evalpanel
            return self._json(evalpanel.state())
        if path == "/api/organize":
            import organize
            return self._json(organize.state())
        if path == "/api/organize/plan":
            import organize
            body = organize.cleanup_plan().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/x-shellscript; charset=utf-8")
            self.send_header("Content-Disposition",
                             'attachment; filename="plan_uborki.sh"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return None
        if path == "/api/organize/csv":
            import organize
            body = organize.problems_csv().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition",
                             'attachment; filename="problemy_bazy.csv"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return None

        if path == "/api/schedule":
            import schedule as schedule_mod
            return self._json(schedule_mod.status())
        if path == "/api/alerts":
            import alerts
            return self._json({"active": alerts.active(),
                               "history": alerts.history(30),
                               "enabled": config.ALERTS_ENABLED,
                               "channels": config.ALERT_CHANNELS})
        if path == "/api/retention":
            import retention
            return self._json(retention.status())
        if path == "/api/state":
            return self._json(pipeline_state())

        if path == "/api/overview":
            import metrics
            import models as models_mod
            # Метрики сервера живут в отдельной базе телеметрии, а не в
            # основной. Обращение сюда через db.q1 на свежей установке
            # роняло весь обзорный экран: таблицы в основной базе нет и
            # не будет. Проверено на чистой установке — раньше первый
            # экран после установки просто не открывался.
            metrics.ensure_tables()
            last = db.tq1("SELECT * FROM server_metrics ORDER BY id DESC LIMIT 1")
            return self._json({
                "index": metrics.index_report(),
                "usage": metrics.usage_report(168),
                "models": metrics.model_report(168),
                "server": metrics.server_series(24),
                "hardware": models_mod.hardware(),
                "now": {"cpu": last["cpu_percent"] if last else 0,
                        "ram_gb": last["ram_used_gb"] if last else 0,
                        "ram_total_gb": last["ram_total_gb"] if last else 0},
            })

        if path == "/api/models":
            import metrics
            import models as models_mod
            return self._json({"catalog": models_mod.catalog(),
                               "hardware": models_mod.hardware(),
                               "server": models_mod.status(),
                               "usage": metrics.model_report(168)})

        if path == "/api/kb":
            import metrics
            import watcher
            watcher.ensure_tables()
            return self._json({"index": metrics.index_report(),
                               "structure": [dict(r) for r in db.q(
                                   "SELECT * FROM structure_events ORDER BY id DESC LIMIT 40")]})

        if path == "/api/llm":
            import llm as llm_mod
            import llm_queue
            import models as models_mod
            return self._json({
                "describe": llm_mod.describe(),
                "server": models_mod.status(),
                "providers": list(llm_mod.PROVIDERS),
                "primary": config.LLM_PROVIDER,
                "fallback": config.LLM_FALLBACK,
                "model": config.LLM_MODEL,
                "local_model": config.LOCAL_LLM_MODEL,
                "local_url": config.LOCAL_LLM_BASE_URL,
                "queue": llm_queue.status(),
                "queue_stats": llm_queue.stats(24),
                "installed": models_mod.installed_llms(),
            })

        if path == "/api/llm/queue":
            import llm_queue
            return self._json({"status": llm_queue.status(),
                               "stats": llm_queue.stats(
                                   int(query.get("hours", ["24"])[0] or 24))})

        if path == "/api/models/progress":
            import models as models_mod
            state = models_mod.progress_state()
            state["log"] = models_mod.action_log(40)
            return self._json(state)

        if path == "/api/models/check":
            import models as models_mod
            return self._json(models_mod.check(query.get("id", [""])[0]))

        if path == "/api/trace":
            import tracing
            if query.get("id"):
                return self._json(tracing.get(int(query["id"][0])))
            if query.get("query_id"):
                return self._json(tracing.by_query(int(query["query_id"][0])))
            if query.get("find"):
                return self._json({"traces": tracing.find(query["find"][0])})
            return self._json({"traces": tracing.recent(
                60, only_bad=query.get("bad", ["0"])[0] == "1")})

        if path == "/api/sources":
            import crawl
            crawl.ensure_tables()
            return self._json({
                "sources": crawl.read_sources(),
                "file": str(config.CRAWL_SOURCES_FILE),
                "pages": db.q1("SELECT COUNT(*) n FROM documents "
                               "WHERE source_type='manufacturer_site'")["n"],
                "provider": config.WEB_SEARCH_PROVIDER,
            })

        if path == "/api/media/state":
            counts = db.q1("""SELECT
                SUM(CASE WHEN asset_kind IN ('video','audio') THEN 1 ELSE 0 END) total,
                SUM(CASE WHEN asset_kind IN ('video','audio')
                     AND enriched LIKE '%asr%' THEN 1 ELSE 0 END) done
                FROM documents WHERE status='ok'""")
            return self._json({
                "total": counts["total"] or 0, "done": counts["done"] or 0,
                "provider": config.ASR_PROVIDER,
                "segments": db.q1("SELECT COUNT(*) n FROM media_segments")["n"],
            })

        if path == "/api/contextual":
            import contextual
            return self._json({"status": contextual.status(),
                               "estimate": contextual.estimate()})

        if path == "/api/analytics":
            import analytics
            hours = int(query.get("hours", ["720"])[0])
            return self._json({
                "funnel": analytics.funnel(hours),
                "confidence": analytics.confidence_histogram(hours),
                "channels": analytics.channel_report(hours),
                "route_ru": analytics.ROUTE_RU,
                "hours": hours,
            })

        if path == "/api/analytics/gaps":
            import analytics
            return self._json(analytics.gaps(
                hours=int(query.get("hours", ["720"])[0]),
                min_group=int(query.get("min", ["2"])[0])))

        if path == "/api/users":
            import access
            return self._json({"users": access.listing(), "summary": access.summary(),
                               "roles": sorted(config.ROLE_SECTIONS),
                               "role_sections": {k: sorted(v) for k, v
                                                 in config.ROLE_SECTIONS.items()},
                               "from_env": bool(config.TELEGRAM_ALLOWED_IDS),
                               "admins": config.TELEGRAM_ADMIN_IDS})

        if path == "/api/jobs":
            import jobs
            return self._json({"jobs": [self._safe_job(j) for j in jobs.recent(40)]})

        if path == "/api/jobs/one":
            import jobs
            return self._json(self._safe_job(
                jobs.get(int(query.get("id", ["0"])[0]))))

        if path == "/api/extract/errors":
            return self._json(extract_errors())

        if path == "/api/adminlog":
            return self._json({"entries": audit_log(
                int(query.get("limit", ["200"])[0]))})

        if path == "/api/regression":
            import regression
            return self._json({"history": regression.history(),
                               "diff": regression.diff(),
                               "pending_changes": regression.settings_changed_since_last_run(),
                               "dataset": str(regression.dataset_path()),
                               "dataset_exists": regression.dataset_path().exists()})

        if path == "/api/ocr/state":
            import ocr as ocr_mod
            rows = db.q("""SELECT rel_path, pages, needs_ocr, ocr_provider, ocr_quality,
                           ocr_at, ocr_error FROM documents
                           WHERE (needs_ocr=1 OR ocr_provider IS NOT NULL) AND status='ok'
                           ORDER BY needs_ocr DESC, ocr_quality ASC, pages DESC LIMIT 300""")
            agg = db.q1("""SELECT COUNT(*) done, SUM(ocr_pages) pages, AVG(ocr_quality) q
                           FROM documents WHERE ocr_provider IS NOT NULL""")
            return self._json({
                "provider": config.OCR_PROVIDER,
                "fallback": config.OCR_FALLBACK,
                "min_quality": config.OCR_MIN_QUALITY,
                "available": ocr_mod.available(),
                "queue": db.q1("SELECT COUNT(*) n FROM documents "
                               "WHERE needs_ocr=1 AND status='ok'")["n"],
                "done": agg["done"] or 0,
                "pages": agg["pages"] or 0,
                "avg_quality": round(agg["q"], 2) if agg["q"] is not None else None,
                "failed": db.q1("SELECT COUNT(*) n FROM documents "
                                "WHERE ocr_error IS NOT NULL")["n"],
                "weak": db.q1("SELECT COUNT(*) n FROM documents WHERE ocr_quality "
                              "IS NOT NULL AND ocr_quality < ?",
                              (config.OCR_MIN_QUALITY,))["n"],
                "documents": [{"path": r["rel_path"], "pages": r["pages"],
                               "needs_ocr": bool(r["needs_ocr"]),
                               "provider": r["ocr_provider"],
                               "quality": r["ocr_quality"], "at": r["ocr_at"],
                               "error": r["ocr_error"]} for r in rows],
            })

        if path == "/api/backup/state":
            import backup as backup_mod
            archives = []
            for item in backup_mod.archives()[:40]:
                info = {"name": item.name, "bytes": item.stat().st_size,
                        "created": backup_mod._parse_stamp(item).isoformat(timespec="seconds")}
                # Манифест лежит внутри архива и читается без полной распаковки.
                try:
                    import tarfile
                    with tarfile.open(item) as tar:
                        fh = tar.extractfile(backup_mod.MANIFEST)
                        if fh:
                            manifest = json.loads(fh.read().decode("utf-8"))
                            info["counts"] = manifest.get("counts")
                            info["note"] = manifest.get("note")
                            info["provider"] = manifest.get("embeddings_provider")
                except Exception as exc:  # noqa: BLE001 — покажем хотя бы размер
                    info["note"] = f"манифест не прочитан: {exc}"
                archives.append(info)
            return self._json({
                "status": backup_mod.status(), "archives": archives,
                "mirror": config.BACKUP_MIRROR_DIR,
                "keep_daily": config.BACKUP_KEEP_DAILY,
                "keep_weekly": config.BACKUP_KEEP_WEEKLY,
                "keep_monthly": config.BACKUP_KEEP_MONTHLY,
                "alert_hours": config.BACKUP_ALERT_HOURS,
            })

        if path == "/api/maintenance":
            import backup as backup_mod
            import embeddings as emb_mod
            import ocr as ocr_mod
            import rerank as rerank_mod
            chunks = db.q1("SELECT COUNT(*) n FROM chunks")["n"]
            trained = 0
            try:
                emb = emb_mod.get_embedder()
                trained = int(getattr(emb, "model", None).meta.get("documents", 0)) \
                    if isinstance(emb, emb_mod.LSAEmbedder) else 0
            except Exception:  # noqa: BLE001 — провайдер не готов, покажем ошибку ниже
                pass
            meta, model_bytes = {}, 0
            try:
                emb = emb_mod.get_embedder()
                if isinstance(emb, emb_mod.LSAEmbedder):
                    meta = emb.model.meta
                    model_bytes = Path(config.LSA_MODEL_PATH).stat().st_size \
                        if Path(config.LSA_MODEL_PATH).exists() else 0
            except Exception:  # noqa: BLE001
                pass
            import search as search_mod
            dense_ok, dense_note = search_mod.dense_ready()
            return self._json({
                "embeddings": emb_mod.describe(),
                "dense_ok": dense_ok, "dense_note": dense_note,
                "rerank": rerank_mod.describe(),
                "backup": backup_mod.status(),
                "chunks": chunks,
                "vectors": len(db.vectors()),
                "pending_vectors": db.q1("SELECT COUNT(*) n FROM chunks "
                                         "WHERE embedded=0")["n"],
                "lsa_meta": meta,
                "lsa_bytes": model_bytes,
                "rerank_weight": config.RERANKER_WEIGHT,
                "rerank_top_n": config.RERANKER_TOP_N,
                "rrf_k": config.RRF_K,
                "lsa_trained_on": trained,
                "lsa_stale": bool(trained and chunks > trained * (1 + config.LSA_STALE_RATIO)),
                "ocr": {
                    "provider": config.OCR_PROVIDER,
                    "queue": db.q1("SELECT COUNT(*) n FROM documents "
                                   "WHERE needs_ocr=1 AND status='ok'")["n"],
                    "done": db.q1("SELECT COUNT(*) n FROM documents "
                                  "WHERE ocr_provider IS NOT NULL")["n"],
                    "available": ocr_mod.available(),
                },
            })

        if path == "/api/settings":
            import security
            return self._json({"settings": settings_schema.as_json(),
                               "groups": settings_schema.groups(),
                               "secrets": security.secrets_health(),
                               "env_file": str(config.BASE_DIR / ".env"),
                               "secrets_file": str(config.SECRETS_FILE)})

        if path == "/api/audit":
            import contextlib
            import io
            import audit
            rep = audit.build_report()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                audit.print_report(rep)
            return self._json({"text": buf.getvalue(), "report": rep})

        if path == "/api/voices":
            try:
                import voice
                return self._json({"voices": voice.voices()})
            except Exception as exc:  # noqa: BLE001
                return self._json({"voices": [], "error": safe_error(exc, "список голосов")})

        if path == "/api/sip":
            import sip
            result = sip.check()
            lines = [f"Режим: {result['mode']}"]
            if result.get("asterisk"):
                lines.append(f"Asterisk: {result['asterisk']}")
            if result.get("audiosocket"):
                lines.append(f"AudioSocket слушает: {result['audiosocket']}")
            if result.get("dialplan"):
                lines.append("\nДобавьте в extensions.conf:\n" + result["dialplan"])
            if result["problems"]:
                lines.append("\nНужно поправить:")
                lines += [f"  · {p}" for p in result["problems"]]
            else:
                lines.append("\nВсё готово к приёму звонков.")
            return self._json({"text": "\n".join(lines), "result": result})

        if path == "/api/log":
            return self._json({"lines": logging_setup.tail(
                int(query.get("lines", ["300"])[0]),
                query.get("level", [""])[0] or None,
                query.get("subsystem", [""])[0] or None,
                search=query.get("q", [""])[0] or None)})
        if path == "/api/log/levels":
            return self._json(logging_setup.levels())

        if path == "/api/queries":
            recent = [dict(r) for r in db.q("""
                SELECT q.id, q.created_at, q.user_name, q.question, q.top_score, q.latency_ms,
                  (SELECT verdict FROM feedback f WHERE f.query_id=q.id ORDER BY f.id DESC LIMIT 1) verdict
                FROM queries q ORDER BY q.id DESC LIMIT 50""")]
            gaps = [dict(r) for r in db.q(
                "SELECT created_at, question FROM queries WHERE answered=0 "
                "ORDER BY id DESC LIMIT 30")]
            golden = [dict(r) for r in db.q(
                "SELECT question, answer, hits FROM golden_qa WHERE active=1 "
                "ORDER BY id DESC LIMIT 50")]
            return self._json({"recent": recent, "gaps": gaps, "golden": golden})

        if path == "/api/diagnostics":
            return self._json({"checks": diagnostics()})
        if path == "/api/job":
            return self._json(job_detail(query.get("id", [""])[0]))
        if path == "/metrics":
            import metrics
            return self._send(200, metrics.prometheus().encode(), "text/plain")

        if path == "/graph.html":
            f = config.DATA_DIR / "graph.html"
            if f.exists():
                return self._send(200, f.read_bytes(), "text/html")
            return self._send(200, "<body style='background:#0f1116;color:#8b93a3;"
                                   "font-family:sans-serif;padding:24px'>Граф ещё не построен. "
                                   "Нажмите «Построить граф».</body>".encode(), "text/html")
        if path == "/tts.ogg":
            f = config.DATA_DIR / "tts_preview.ogg"
            if f.exists():
                return self._send(200, f.read_bytes(), "audio/ogg")
            return self._send(404, b"", "text/plain")
        return self._send(404, b"not found", "text/plain")

    # ------------------------------------------------------------ POST ------
    def do_POST(self) -> None:  # noqa: N802, C901
        path = urllib.parse.urlparse(self.path).path
        if not self._csrf_ok():
            return self._send(415, b'{"error":"nuzhen Content-Type: application/json"}')
        if not self._authorized():
            return self._send(401, b"unauthorized", "text/plain")
        payload = self._body()

        # ------------------------------------------- вход по общему паролю --
        if path == "/api/token-login":
            import security
            addr = self.client_address[0] if self.client_address else "?"
            if not config.ADMIN_TOKEN or security.accounts_enabled():
                return self._json({"error": "этот способ входа отключён"}, 400)
            gate = security.login_attempt_allowed("__token__", addr)
            if not gate["ok"]:
                log.error("вход по паролю заблокирован после серии неудачных "
                          "попыток с адреса %s", addr)
                return self._json({"error": gate["message"]}, 429)
            given = str(payload.get("token", ""))
            if not hmac.compare_digest(given, config.ADMIN_TOKEN):
                left = security.login_failed("__token__", addr)
                log.warning("неверный пароль администратора с адреса %s "
                            "(осталось попыток: %d)", addr, left)
                time.sleep(1.0)
                return self._json({"error": "неверный пароль"}, 401)
            security.login_succeeded("__token__", addr)
            body = json.dumps({"ok": True}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            secure = "; Secure" if _behind_https(self) else ""
            self.send_header("Set-Cookie",
                             f"kb_token={config.ADMIN_TOKEN}; Path=/; HttpOnly; "
                             f"SameSite=Strict; Max-Age=43200{secure}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return None

        # --------------------------------------------------------- вход ----
        if path == "/api/login":
            import security
            addr = self.client_address[0] if self.client_address else "?"
            login = str(payload.get("login", ""))
            # Подбор пароля должен упираться в счётчик, а не в задержку.
            # Задержка здесь не мешала вовсе: сервер многопоточный, и
            # двести одновременных соединений давали двести попыток в
            # секунду, сколько ни спи в каждой из них.
            gate = security.login_attempt_allowed(login, addr)
            if not gate["ok"]:
                log.error("вход заблокирован после серии неудачных попыток: "
                          "%s с адреса %s", login, addr)
                return self._json({"error": gate["message"]}, 429)
            account = security.check_password(login, str(payload.get("password", "")))
            if account is None:
                left = security.login_failed(login, addr)
                log.warning("неудачная попытка входа в админку: %s с адреса %s "
                            "(осталось попыток: %d)", login, addr, left)
                time.sleep(1.0)
                return self._json({"error": "неверный логин или пароль"}, 401)
            security.login_succeeded(login, addr)
            token = security.open_session(account)
            body = json.dumps({"ok": True, "account": account}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            # Secure ставим, когда снаружи https: за обратным прокси об
            # этом говорит X-Forwarded-Proto, но верить заголовку можно
            # только если прокси свой — отсюда отдельная настройка.
            secure = "; Secure" if _behind_https(self) else ""
            self.send_header("Set-Cookie",
                             f"kb_session={token}; Path=/; HttpOnly; "
                             f"SameSite=Strict; Max-Age="
                             f"{int(config.ADMIN_SESSION_HOURS * 3600)}{secure}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            audit("вход", f"{account['login']} ({account['role']})")
            return None

        if path == "/api/logout":
            import security
            security.close_session(self._cookie("kb_session"))
            return self._json({"ok": True})

        if path == "/api/users/admin":
            import security
            action = payload.get("action")
            try:
                if action == "add":
                    result = security.add_user(
                        str(payload.get("login", "")), str(payload.get("password", "")),
                        str(payload.get("role", config.ADMIN_DEFAULT_ROLE)),
                        str(payload.get("full_name", "")))
                    audit("учётная запись", f"создана: {result['login']} "
                          f"({result['role']})")
                    return self._json(result)
                if action == "delete":
                    ok = security.delete_user(str(payload.get("login", "")))
                    audit("учётная запись", f"удалена: {payload.get('login')}")
                    return self._json({"ok": ok})
            except Exception as exc:  # noqa: BLE001
                return self._json({"error": safe_error(exc)}, 400)
            return self._json({"error": "неизвестное действие"}, 400)

        if path == "/api/password":
            import security
            if not security.accounts_enabled():
                return self._json({"error": "учётных записей нет"}, 400)
            account = self._account()
            if account is None:
                return self._json({"error": "нужно войти"}, 401)
            if security.check_password(account["login"],
                                       str(payload.get("current", ""))) is None:
                return self._json({"error": "текущий пароль неверен"}, 400)
            try:
                security.set_password(account["login"], str(payload.get("new", "")))
            except ValueError as exc:
                return self._json({"error": str(exc)}, 400)
            audit("учётная запись", f"пароль изменён: {account['login']}")
            return self._json({"message": "Пароль изменён."})

        if path == "/api/secrets/move":
            import security
            result = security.move_secrets_from_env()
            audit("ключи", result["message"])
            return self._json(result)

        if path == "/api/quickstart/run":
            # Шаг быстрого старта выполняется тем же способом, что и всё
            # остальное: через очередь заданий. Отдельного пути тут нет
            # намеренно — иначе действие из «Быстрого старта» обошло бы
            # блокировки и могло запуститься параллельно с тем же самым,
            # запущенным из своего раздела.
            import jobs
            import quickstart
            key = str(payload.get("step") or "")
            step = next((fn() for k, _t, fn in quickstart.STEPS if k == key), None)
            action = (step or {}).get("action") or {}
            kind = action.get("kind")
            if not kind:
                return self._json({"error": "у этого шага нет действия"}, 400)
            try:
                job = jobs.enqueue(kind, action.get("title") or kind,
                                   {"reason": "быстрый старт"},
                                   created_by=self._who())
            except jobs.Busy as exc:
                return self._json({"message": str(exc)}, 409)
            except Exception as exc:  # noqa: BLE001
                return self._json({"message": safe_error(exc)}, 400)
            jobs.start_worker()
            audit("быстрый старт", f"{key}: {kind}")
            message = f"«{action.get('title') or kind}» поставлено в очередь (№{job['id']})."
            # У шага может быть продолжение: обучить модель и следом
            # пересчитать векторы. Ставим сразу оба — по отдельности их
            # запускать бессмысленно, а забыть второй легко.
            if action.get("then"):
                try:
                    second = jobs.enqueue(action["then"], action["then"],
                                          {"reason": "быстрый старт"},
                                          created_by=self._who())
                    message += (f" Следом выполнится «{action['then']}» "
                                f"(№{second['id']}).")
                except Exception as exc:  # noqa: BLE001
                    message += f" Второй шаг поставить не вышло: {exc}"
            return self._json({"ok": True, "job": job["id"], "message": message})

        if path == "/api/schedule":
            import schedule as schedule_mod
            action = payload.get("action")
            text = (schedule_mod.install() if action == "install"
                    else schedule_mod.remove())
            audit("расписание", action or "")
            return self._json({"message": text})

        if path == "/api/retention":
            import retention
            action = payload.get("action")
            if action == "clean":
                result = retention.clean(dry=bool(payload.get("dry")))
                audit("хранение", f"очистка: вопросов {result['queries']}, "
                                  f"цепочек {result['traces']}")
                return self._json(result)
            if action == "forget":
                result = retention.forget(int(payload.get("user_id", 0)),
                                          keep_questions=bool(payload.get("keep")))
                audit("хранение", f"удалены данные сотрудника "
                                  f"{payload.get('user_id')}")
                return self._json(result)
            return self._json({"error": "неизвестное действие"}, 400)

        if path == "/api/settings":
            values = {k: v for k, v in payload.items() if k != "force"}
            # Смена провайдера смыслового поиска — не «записать строчку»,
            # а целый сценарий: пакеты, веса, конвертация, проверка,
            # пересчёт векторов. Прямое сохранение здесь честно падало
            # («не заполнен ONNX_MODEL_PATH»), но требовало от человека
            # ручной подготовки. Делегируем задаче embed_switch — она
            # готовит всё сама и сохраняет настройку только после успеха.
            notes: list[str] = []

            def _delegate(kind: str, title: str, payload_job: dict,
                          what: str) -> None:
                import handlers  # noqa: F401 — регистрация обработчиков
                import jobs
                jobs.start_worker()
                try:
                    job = jobs.enqueue(kind, title, payload_job, wait=True)
                    audit("задача", title, payload_job)
                    busy = busy_with_indexing()
                    notes.append(
                        f"{what} — отдельной задачей №{job['id']}: она сама "
                        f"скачает и подготовит всё нужное, проверит модель "
                        f"пробным запросом и только после успеха сохранит "
                        f"настройку. Ход — в разделе «Конвейер»."
                        + (f" Сейчас выполняется «{busy}» — задача начнётся "
                           f"сразу после неё." if busy else ""))
                except jobs.Busy as exc:
                    notes.append(f"{what}: не поставлено — {exc}")

            # Смысловой поиск: смена провайдера ИЛИ модели.
            new_prov = str(values.get("EMBEDDINGS_PROVIDER", "")).strip()
            new_emb_model = str(values.get("EMBEDDINGS_MODEL", "")).strip()
            prov_changed = new_prov and new_prov != config.EMBEDDINGS_PROVIDER
            model_changed = ("EMBEDDINGS_MODEL" in values
                             and new_emb_model != str(config.EMBEDDINGS_MODEL or ""))
            if prov_changed or model_changed:
                values.pop("EMBEDDINGS_PROVIDER", None)
                values.pop("EMBEDDINGS_MODEL", None)
                target = new_prov or config.EMBEDDINGS_PROVIDER
                _delegate("embed_switch", "смена провайдера смыслового поиска",
                          {"provider": target, "model": new_emb_model or None},
                          f"Смысловой поиск переключается на «{target}»"
                          + (f" с моделью «{new_emb_model}»"
                             if new_emb_model else ""))

            # Генерация: смена провайдера, облачной или локальной модели.
            new_llm = str(values.get("LLM_PROVIDER", "")).strip()
            new_cloud = str(values.get("LLM_MODEL", "")).strip()
            new_local = str(values.get("LOCAL_LLM_MODEL", "")).strip()
            llm_prov_changed = new_llm and new_llm != config.LLM_PROVIDER
            cloud_changed = ("LLM_MODEL" in values
                             and new_cloud != str(config.LLM_MODEL or ""))
            local_changed = ("LOCAL_LLM_MODEL" in values and new_local
                             and new_local != str(config.LOCAL_LLM_MODEL or ""))
            if llm_prov_changed or cloud_changed or local_changed:
                values.pop("LLM_PROVIDER", None)
                target = new_llm or config.LLM_PROVIDER
                model_arg = None
                if target == "local":
                    if local_changed:
                        values.pop("LOCAL_LLM_MODEL", None)
                        model_arg = new_local
                else:
                    values.pop("LLM_MODEL", None)
                    if cloud_changed:
                        model_arg = new_cloud
                _delegate("llm_switch", "смена модели генерации",
                          {"provider": target, "model": model_arg},
                          f"Генерация переключается на «{target}»"
                          + (f" с моделью «{model_arg}»" if model_arg else ""))

            # Остальные «провайдерные» настройки: реранкер, сканы, зрение,
            # речь, хранилище векторов. Для каждой есть автонастройка —
            # пакеты, программы и модели ставятся сами, настройка
            # сохраняется только после успешной подготовки.
            import components as components_mod
            for comp_key in sorted(components_mod.KEYS):
                if comp_key not in values:
                    continue
                comp_val = str(values[comp_key]).strip()
                if comp_val == str(getattr(config, comp_key, "") or "").strip():
                    continue
                values.pop(comp_key)
                _delegate("component_setup",
                          f"настройка: {components_mod.title(comp_key)}",
                          {"key": comp_key, "value": comp_val},
                          f"{components_mod.title(comp_key)}: переключение "
                          f"на «{comp_val}»")
            note = " ".join(notes)
            issues = settings_schema.validate(values, full=env_with_secrets())
            errors = [i for i in issues if i["level"] == "error"]
            warnings = [i for i in issues if i["level"] == "warning"]
            # Ошибки не сохраняем никогда: с таким значением модуль просто
            # не запустится, а связать поломку с давней правкой уже трудно.
            # Предупреждения — сохраняем, но только после подтверждения.
            if errors or (warnings and not payload.get("force")):
                return self._json({"ok": False, "issues": issues, "note": note,
                                   "errors": len(errors), "warnings": len(warnings)},
                                  200)
            before = read_env()
            changed = {k: v for k, v in values.items()
                       if str(before.get(k, "")) != str(v)}
            # Часть настроек нельзя менять, пока идёт индексация: следующая
            # пачка векторов пойдёт по другой модели, задача оборвётся, а
            # записанная часть останется в файле — индекс, собранный из
            # двух разных моделей, выглядит целым и молча плохо ищет.
            risky = sorted(set(changed) & _UNSAFE_WHILE_BUSY)
            busy = busy_with_indexing() if risky else ""
            if busy:
                return self._json({
                    "ok": False, "errors": 1, "warnings": 0,
                    "issues": [{"key": risky[0], "level": "error",
                                "title": "Сейчас идёт длительная операция",
                                "message": f"выполняется «{busy}»",
                                "hint": "Эти настройки задают, как считаются "
                                        "векторы. Смена на ходу оборвёт задачу "
                                        "и оставит индекс, собранный по двум "
                                        "разным моделям. Провайдера смыслового "
                                        "поиска можно сменить и не дожидаясь: "
                                        "кнопка «Переключить провайдера» в "
                                        "разделе «Поиск» встанет в очередь и "
                                        "выполнится сразу после текущей "
                                        "задачи. Остальное — дождитесь "
                                        "окончания или снимите задачу в "
                                        "разделе «Конвейер»: "
                                        + ", ".join(risky)}]}, 200)
            write_env(values)
            # В журнал пишем, что менялось, но не значения ключей. Иначе
            # весь смысл выноса ключей в защищённый файл пропадает:
            # журнал лежит в основной базе, а база — первым файлом в
            # резервной копии. Один архив у подрядчика — и там открытым
            # текстом все ключи компании.
            audit("настройки", f"изменено ключей: {len(changed)}",
                  _audit_safe_changes(changed, before))
            _reload_after_settings(changed)
            return self._json({"ok": True, "changed": sorted(changed),
                               "issues": issues, "note": note})

        if path == "/api/settings/reset":
            # Вернуть настройку к значению по умолчанию. Без этой кнопки
            # «как было» приходится искать в документации, а на глаз
            # умолчание не отличить от значения, которое кто-то выставил
            # таким же намеренно.
            keys = [k for k in (payload.get("keys") or []) if isinstance(k, str)]
            by_key = {s["key"]: s for s in settings_schema.SETTINGS}
            values = {}
            for key in keys:
                spec = by_key.get(key)
                if spec is None or spec["type"] in SECRET_TYPES:
                    continue
                default = spec.get("default", "")
                values[key] = ("1" if default else "0") \
                    if isinstance(default, bool) else str(default)
            if not values:
                return self._json({"error": "нечего возвращать"}, 400)
            write_env(values)
            audit("настройки", "возврат к умолчаниям: " + ", ".join(sorted(values)))
            _reload_after_settings(values)
            return self._json({"ok": True, "reset": sorted(values)})

        if path == "/api/settings/forget-secret":
            # Стереть ключ. Отдельным действием, потому что пустое поле в
            # форме означает «не трогать»: значение ключа в браузер не
            # отдаётся, и пустым оно выглядит всегда.
            import security
            key = str(payload.get("key") or "")
            if key not in {s["key"] for s in settings_schema.SETTINGS}:
                return self._json({"error": "неизвестная настройка"}, 400)
            removed = security.forget_secret(key)
            audit("настройки", f"ключ стёрт: {key}")
            os.environ.pop(key, None)
            _reload_after_settings({key: ""})
            return self._json({"ok": True, "removed": removed,
                               "message": f"Ключ {key} стёрт."})

        if path == "/api/settings/check":
            issues = settings_schema.validate(
                {k: v for k, v in payload.items() if k != "force"},
                full=env_with_secrets())
            return self._json({"issues": issues,
                               "errors": sum(1 for i in issues if i["level"] == "error"),
                               "warnings": sum(1 for i in issues if i["level"] == "warning")})

        if path == "/api/log/levels":
            try:
                logging_setup.set_level(payload["subsystem"], payload["level"])
                return self._json({"ok": True})
            except Exception as exc:  # noqa: BLE001
                return self._json({"error": safe_error(exc)}, 400)

        # Все длительные операции идут через очередь (jobs.py): она
        # переживает перезапуск процесса и не даёт запустить две задачи,
        # которые пишут в одно и то же.
        if path == "/api/job":
            kind = payload.get("kind")
            titles = {
                "reindex": "переиндексация изменённого",
                "reindex_full": "полная переиндексация",
                "repair": "досчёт векторов",
                "train_lsa": "обучение смысловой модели",
                "reembed": "пересчёт векторов",
                "embed_switch": "смена провайдера смыслового поиска",
                "llm_switch": "смена модели генерации",
                "component_setup": "настройка компонента",
                "ocr": "распознавание сканов",
                "ocr_retry": "повтор распознавания",
                "backup": "резервная копия",
                "backup_verify": "проверка копии",
                "backup_prune": "удаление старых копий",
                "restore": "восстановление из копии",
                "graph": "построение графа",
                "structure": "сверка структуры папок",
                "compare": "сравнение настроек поиска",
                "regression": "проверка качества на контрольных вопросах",
                "eval_llm": "полный замер с генерацией ответов",
                "crawl": "обход сайтов производителей",
                "media": "расшифровка видео и аудио",
                "contextual": "контекстные приставки через модель",
            }
            if kind not in titles:
                return self._json({"message": f"неизвестная задача: {kind}"}, 400)
            import handlers  # noqa: F401 — регистрация обработчиков
            import jobs
            jobs.start_worker()
            payload_clean = {k: v for k, v in payload.items() if k != "kind"}
            # Смена провайдера поиска умеет ждать: во время многочасовой
            # переиндексации отказ «дождитесь окончания» означал бы, что
            # провайдера не сменить практически никогда.
            waits = kind in ("embed_switch", "llm_switch", "component_setup")
            try:
                job = jobs.enqueue(kind, titles[kind], payload_clean, wait=waits)
            except jobs.Busy as exc:
                return self._json({"message": str(exc)}, 409)
            except Exception as exc:  # noqa: BLE001
                return self._json({"message": safe_error(exc, "постановка задачи")}, 500)
            audit("задача", titles[kind], payload_clean)
            note = ""
            if waits:
                busy = busy_with_indexing()
                if busy:
                    note = (f" Сейчас выполняется «{busy}» — переключение "
                            f"начнётся сразу после неё, ничего делать не нужно.")
            return self._json({"message": f"Задача поставлена (№{job['id']}). "
                                          f"Ход виден в разделе «Конвейер».{note}"})

        if path == "/api/job/cancel":
            import jobs
            ok = jobs.cancel(int(payload.get("id", 0)))
            return self._json({"message": "Задача снята." if ok else
                               "Снять нельзя: она уже выполняется или её нет."})

        if path == "/api/job/retry":
            import handlers  # noqa: F401
            import jobs
            try:
                job = jobs.retry(int(payload.get("id", 0)))
            except jobs.Busy as exc:
                return self._json({"message": str(exc)}, 409)
            except Exception as exc:  # noqa: BLE001
                return self._json({"message": str(exc)}, 400)
            jobs.start_worker()
            return self._json({"message": f"Поставлена заново (№{job['id']})."})

        if path == "/api/llm/probe":
            import llm as llm_mod
            return self._json(llm_mod.probe(payload.get("provider") or None))

        if path == "/api/llm/queue/clear":
            import llm_queue
            freed = llm_queue.clear()
            audit("очередь к модели", f"снято мест: {freed}")
            return self._json({"message": f"Снято мест: {freed}.",
                               "status": llm_queue.status()})

        if path == "/api/llm/switch":
            import llm as llm_mod
            primary = str(payload.get("primary") or "")
            if primary not in llm_mod.PROVIDERS:
                return self._json({"error": f"неизвестный провайдер: {primary}"}, 400)
            fallback = ",".join(x for x in (payload.get("fallback") or [])
                                if x in llm_mod.PROVIDERS and x != primary)
            write_env({"LLM_PROVIDER": primary, "LLM_FALLBACK": fallback})
            _reload_after_settings({"LLM_PROVIDER": primary, "LLM_FALLBACK": fallback})
            llm_mod.reset()
            audit("модель генерации", f"{primary}"
                  + (f", запасные: {fallback}" if fallback else ""))
            return self._json({"ok": True, "describe": llm_mod.describe()})

        if path == "/api/sources":
            import crawl
            lines = [ln.strip() for ln in str(payload.get("sources", "")).splitlines()
                     if ln.strip()]
            crawl.write_sources(lines)
            audit("источники", f"список сайтов: {len(lines)} адресов")
            return self._json({"ok": True, "count": len(lines)})

        # ------------------------------------------------------ сотрудники --
        if path == "/api/users/decide":
            import access
            user_id = int(payload.get("user_id", 0))
            approve = bool(payload.get("approve"))
            user = access.decide(user_id, approve, role=payload.get("role") or None,
                                 by=self._who(), note=payload.get("note", ""))
            audit("доступ", ("выдан" if approve else "отклонён") +
                  f": {user.get('full_name') or user_id}", {"user_id": user_id,
                                                            "role": user.get("role")})
            return self._json(user)

        if path == "/api/users/block":
            import access
            user_id = int(payload.get("user_id", 0))
            user = access.block(user_id, by=self._who(), note=payload.get("note", ""))
            audit("доступ", f"заблокирован: {user.get('full_name') or user_id}",
                  {"user_id": user_id})
            return self._json(user)

        if path == "/api/users/role":
            import access
            user_id = int(payload.get("user_id", 0))
            user = access.set_role(user_id, str(payload.get("role")), by=self._who())
            audit("роль", f"{user.get('full_name') or user_id} → {user.get('role')}",
                  {"user_id": user_id})
            return self._json(user)

        if path == "/api/users/trainer":
            import access
            user_id = int(payload.get("user_id", 0))
            on = bool(payload.get("on"))
            user = access.set_trainer(user_id, on, by=self._who())
            audit("дообучение", ("включено" if on else "выключено")
                  + f": {user.get('full_name') or user_id}", {"user_id": user_id})
            return self._json(user)

        if path.startswith("/api/models/"):
            action = path.rsplit("/", 1)[-1]
            import models as models_mod
            model_id = payload.get("id", "")
            try:
                if action == "install":
                    import jobs
                    try:
                        job = jobs.enqueue("model_install", f"загрузка {model_id}",
                                           {"id": model_id},
                                           created_by=self._who())
                    except jobs.Busy as exc:
                        return self._json({"message": str(exc)}, 409)
                    jobs.start_worker()
                    audit("загрузка модели", model_id)
                    return self._json({"message": f"Загрузка поставлена в очередь "
                                                  f"(№{job['id']}). Ход виден в "
                                                  f"разделе «Конвейер»; она "
                                                  f"переживёт перезапуск."})
                if action == "serve":
                    state = models_mod.serve(model_id)
                    return self._json({"message": f"Модель {model_id} запускается на "
                                                  f"{state['base_url']}. Загрузка весов "
                                                  "занимает несколько минут."})
                if action == "stop":
                    return self._json({"message": "Остановлено." if models_mod.stop()
                                                  else "Ничего не запущено."})
            except Exception as exc:  # noqa: BLE001
                return self._json({"message": safe_error(exc)}, 500)

        if path == "/api/tts":
            try:
                import voice
                out = config.DATA_DIR / "tts_preview.ogg"
                voice.synthesize(payload.get("text", ""), out, payload.get("voice") or None)
                return self._json({"message": "Готово.", "url": "/tts.ogg"})
            except Exception as exc:  # noqa: BLE001
                return self._json({"message": safe_error(exc)})

        if path == "/api/normalize":
            import voice
            return self._json({"text": voice.normalize_for_speech(payload.get("text", ""))})

        if path == "/api/graph":
            try:
                import graph
                g = graph.build_graph(float(payload.get("similarity") or 0.5),
                                      int(payload.get("edges") or 6),
                                      payload.get("limit") or None)
                graph.render_html(g, config.DATA_DIR / "graph.html")
                s = g["stats"]
                return self._json({"message": f"документов {s['documents']}, "
                                              f"смысловых связей {s['edges_semantic']}, "
                                              f"без связей {s['orphans']}, "
                                              f"дубликатов {s['dupes']}"})
            except Exception as exc:  # noqa: BLE001
                return self._json({"message": safe_error(exc)}, 500)

        if path == "/api/eval/save":
            import evalpanel
            index = payload.get("index")
            try:
                result = evalpanel.save_item(payload.get("item") or {},
                                             int(index) if index is not None else None)
            except ValueError as exc:
                return self._json({"error": str(exc)}, 400)
            log.warning("контрольный вопрос сохранён (%s): %s",
                        "правка" if index is not None else "новый",
                        result["item"]["question"][:80])
            return self._json({"message": f"Сохранено, в наборе {result['count']}."})

        if path == "/api/eval/delete":
            import evalpanel
            try:
                result = evalpanel.delete_item(int(payload.get("index", -1)))
            except (ValueError, TypeError) as exc:
                return self._json({"error": str(exc)}, 400)
            log.warning("контрольный вопрос удалён: %s", result["removed"][:80])
            return self._json({"message": f"Удалено, осталось {result['count']}."})

        if path == "/api/golden":
            import answer as answer_mod
            q, a = payload.get("question", "").strip(), payload.get("answer", "").strip()
            if len(q) < 5 or len(a) < 5:
                return self._json({"message": "Слишком короткий вопрос или ответ."}, 400)
            gid = answer_mod.add_golden(q, a)
            return self._json({"message": f"Добавлено под номером {gid}."})

        # --- разбор одного вопроса: какой канал что нашёл и что сделал реранкер --
        if path == "/api/search/test":
            import rerank as rerank_mod
            import search as search_mod
            question = (payload.get("question") or "").strip()
            if len(question) < 3:
                return self._json({"error": "слишком короткий вопрос"}, 400)
            saved = config.RERANKER_PROVIDER
            if not payload.get("rerank", True):
                config.RERANKER_PROVIDER = "none"
                rerank_mod.reset()
            try:
                started = time.time()
                bm25 = len(search_mod.bm25_search(question, config.SEARCH_CANDIDATES))
                dense = len(search_mod.dense_search(question, config.SEARCH_CANDIDATES))
                hits = search_mod.search(question, top_k=10)
                elapsed = round((time.time() - started) * 1000)
            except Exception as exc:  # noqa: BLE001
                return self._json({"error": safe_error(exc)}, 500)
            finally:
                if config.RERANKER_PROVIDER != saved:
                    config.RERANKER_PROVIDER = saved
                    rerank_mod.reset()
            return self._json({
                "ms": elapsed, "bm25": bm25, "dense": dense,
                "confidence": round(search_mod.confidence(hits), 5),
                "min_confidence": config.MIN_CONFIDENCE,
                "moved": any(h.channels.get("rerank") is not None for h in hits),
                "hits": [{"path": h.rel_path, "score": round(h.score, 5),
                          "text": (h.text or "")[:220].replace("\n", " "),
                          "channels": {k: v for k, v in h.channels.items()}}
                         for h in hits],
            })

        # --- проверка текста на подмену кириллицы латиницей ----------------------
        if path == "/api/ocr/guard":
            import ocr as ocr_mod
            text = payload.get("text") or ""
            fixed, stats = ocr_mod.repair_homoglyphs(text)
            quality = ocr_mod.quality(fixed, stats)
            return self._json({
                "before": text[:2000], "after": fixed[:2000],
                "fixed": stats["fixed"], "fixed_mixed": stats["fixed_mixed"],
                "fixed_latin": stats["fixed_latin"], "ratio": stats["ratio"],
                "threshold": config.OCR_MAX_LATIN_RATIO, "quality": quality,
                "accepted": stats["ratio"] <= config.OCR_MAX_LATIN_RATIO,
                "vocab": len(ocr_mod.corpus_vocab()),
            })

        if path in ("/api/backup/verify", "/api/backup/restore"):
            import backup as backup_mod
            # Имя приходит из браузера, поэтому берём из него только имя файла
            # и сверяем со списком реально существующих копий: подставить
            # посторонний путь через эту ручку нельзя.
            wanted = Path(str(payload.get("name", ""))).name
            target = next((a for a in backup_mod.archives() if a.name == wanted), None)
            if target is None:
                return self._json({"error": f"копия «{wanted}» не найдена"}, 400)
            if path.endswith("verify"):
                return self._json(backup_mod.verify_archive(target))
            # Восстановление идёт через очередь, а не прямо в потоке
            # HTTP-запроса. Причина: восстановление сначала отодвигает
            # текущие данные в сторону и только потом раскладывает новые.
            # Обрыв между этими шагами (перезапуск по деплою, закрытая
            # вкладка, разорванное соединение) оставляет систему вообще
            # без индекса — при следующем старте она поднимется как
            # чистая установка, а настоящий индекс будет лежать рядом в
            # папке before-restore-*, и в интерфейсе об этом ни слова.
            # Очередь переживает перезапуск и не даёт запуститься второму
            # восстановлению поверх первого.
            import jobs
            try:
                job = jobs.enqueue("restore", f"восстановление из {target.name}",
                                   {"name": target.name,
                                    "force": bool(payload.get("force"))},
                                   created_by=self._who())
            except jobs.Busy as exc:
                return self._json({"error": str(exc)}, 409)
            except Exception as exc:  # noqa: BLE001
                return self._json({"error": safe_error(exc)}, 400)
            jobs.start_worker()
            audit("восстановление индекса", target.name)
            return self._json({
                "ok": True, "job": job["id"],
                "message": f"Восстановление из «{target.name}» поставлено в "
                           f"очередь (№{job['id']}). Ход виден в разделе "
                           f"«Конвейер»; бот и поиск на это время лучше не "
                           f"трогать."})

        return self._send(404, b"not found", "text/plain")


def _build_graph() -> dict:
    import graph
    g = graph.build_graph()
    graph.render_html(g, config.DATA_DIR / "graph.html")
    return g["stats"]


def _metrics_thread() -> None:
    """
    Вечно живой поток админки: метрики, пульс и — раз в час — проверки
    оповещений. Проверки продублированы здесь не случайно: расписание
    cron можно снести при миграции сервера, и тогда тишина в Telegram
    неотличима от «всё хорошо». Отказавший детектор отказов — самый
    опасный класс отказов, поэтому у него два независимых носителя:
    cron и этот поток.
    """
    import alerts
    import metrics
    last_alerts_check = 0.0
    while True:
        try:
            metrics.collect()
            metrics.beat("админка")
        except Exception:  # noqa: BLE001
            pass
        if (config.ALERTS_ENABLED
                and time.time() - last_alerts_check > config.ALERTS_SELF_CHECK_MINUTES * 60
                and config.ALERTS_SELF_CHECK_MINUTES > 0):
            last_alerts_check = time.time()
            try:
                alerts.check()
            except Exception:  # noqa: BLE001
                log.warning("проверка оповещений из админки не отработала",
                            exc_info=True)
        time.sleep(config.METRICS_INTERVAL_SECONDS)


def main() -> None:
    import preflight
    import shutdown

    logging_setup.setup()
    db.init()
    ensure_events_table()
    import metrics
    metrics.ensure_tables()

    # Проверка перед стартом. Смысл в том, чтобы нерабочая конфигурация
    # обнаружилась здесь, а не первым вопросом сотрудника через неделю:
    # процесс, который поднялся и показывает systemd «active», выглядит
    # исправным, даже если папки базы знаний не существует.
    report = preflight.check("админка")
    print(preflight.render(report))
    if report["fatal"]:
        log.error("запуск прерван: %s", "; ".join(report["fatal"]))
        raise SystemExit(2)

    shutdown.install("админка")

    if config.METRICS_ENABLED:
        threading.Thread(target=_metrics_thread, daemon=True).start()

    # Обработчик очереди поднимается вместе с процессом, а не по нажатию
    # кнопки. Иначе задача, поставленная на повтор и пережившая
    # перезапуск, не выполнится никогда, а занятый ею ресурс останется
    # занятым: следующая индексация будет получать отказ «уже выполняется».
    try:
        import handlers  # noqa: F401 — регистрация обработчиков задач
        import jobs
        jobs.ensure_tables()
        jobs.reap_stale()
        jobs.start_worker()
    except Exception as exc:  # noqa: BLE001 — очередь не должна мешать старту
        log.error("не удалось поднять обработчик очереди: %s", exc)

    srv = ThreadingHTTPServer((config.ADMIN_HOST, config.ADMIN_PORT), Handler)
    srv.daemon_threads = True
    url = f"http://{config.ADMIN_HOST}:{config.ADMIN_PORT}/"
    log.info("админка запущена: %s", url)
    print(f"Админка запущена: {url}")
    if not config.ADMIN_TOKEN and config.ADMIN_HOST not in ("127.0.0.1", "localhost"):
        print("ВНИМАНИЕ: ADMIN_TOKEN не задан, а интерфейс слушает не только localhost.")

    shutdown.on_stop("закрыть приём запросов", srv.shutdown)
    shutdown.on_stop("дописать векторы", lambda: db.vectors().save())
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    shutdown.wait_actions()
    srv.server_close()
    log.warning("админка остановлена")
    print("\nОстановлено.")


if __name__ == "__main__":
    sys.exit(main())
