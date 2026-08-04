"""
Хранилище: SQLite + FTS5 (лексический поиск) + numpy-матрица векторов.

Почему так для прототипа:
  * ноль внешних сервисов — запускается на ноутбуке за минуту;
  * FTS5 даёт полноценный BM25, который на технических артикулах
    («WRP-A 2ECO6-38», «Арт. 500095.F») работает лучше эмбеддингов;
  * при переходе в прод векторный канал меняется на Qdrant без изменения
    остального кода — интерфейс поиска один (см. search.py).
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

import config

_local = threading.local()

SCHEMA = """
-- ------------------------------------------------------------- документы --
CREATE TABLE IF NOT EXISTS documents (
    id              INTEGER PRIMARY KEY,
    rel_path        TEXT UNIQUE NOT NULL,      -- путь относительно KB_ROOT
    abs_path        TEXT NOT NULL,
    file_name       TEXT NOT NULL,
    ext             TEXT NOT NULL,
    section         TEXT,                      -- 1-й уровень: РОЗНИЧНАЯ ПРОДУКЦИЯ и т.п.
    brand           TEXT,                      -- 2-й уровень: BAXI, Джилекс, SPL...
    doc_type        TEXT,                      -- КАТАЛОГ / ПАСПОРТ / СЕРТИФИКАТ / ПРАЙС...
    content_hash    TEXT NOT NULL,             -- sha256 содержимого
    size_bytes      INTEGER,
    mtime           REAL,
    effective_date  TEXT,                      -- дата документа (из имени файла), ISO
    is_current      INTEGER DEFAULT 1,         -- 0 = вытеснен более свежей версией
    superseded_by   INTEGER,                   -- id более свежей версии
    version_key     TEXT,                      -- ключ «одной и той же сущности» документа
    pages           INTEGER,
    text_chars      INTEGER,
    needs_ocr       INTEGER DEFAULT 0,         -- текстовый слой пуст → нужен OCR
    kind            TEXT DEFAULT 'text',       -- text | table | asset | archive | web
    asset_kind      TEXT,                      -- image | video | audio | drawing | archive
    source_type     TEXT DEFAULT 'internal_kb',-- internal_kb | manufacturer_site | live_web
    source_url      TEXT,                      -- откуда взято, если из интернета
    enriched        TEXT,                      -- какие обогащения выполнены: asr,vision,ocr,cad
    indexed_at      TEXT,
    status          TEXT DEFAULT 'ok',         -- ok | skipped | error | deleted
    error           TEXT
);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash);
CREATE INDEX IF NOT EXISTS idx_documents_vkey ON documents(version_key);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);

-- ------------------------------------------------------------------ чанки --
CREATE TABLE IF NOT EXISTS chunks (
    id            INTEGER PRIMARY KEY,
    doc_id        INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord           INTEGER NOT NULL,
    page_from     INTEGER,
    page_to       INTEGER,
    heading       TEXT,        -- ближайший заголовок раздела
    context       TEXT,        -- контекстная приставка (Contextual Retrieval)
    text          TEXT NOT NULL,
    n_chars       INTEGER,
    embedded      INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

-- Полнотекстовый индекс. content='' — внешнее хранение, экономит место.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, heading, context, brand, doc_type, file_name,
    content='', tokenize='unicode61 remove_diacritics 2'
);

-- ------------------------------------------- отрезки видео и аудио --------
-- Расшифровка речи режется на смысловые куски с таймкодами: ответ бота
-- ссылается не на «видео целиком», а на конкретную минуту.
CREATE TABLE IF NOT EXISTS media_segments (
    id          INTEGER PRIMARY KEY,
    doc_id      INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    chunk_id    INTEGER,
    start_sec   REAL,
    end_sec     REAL,
    text        TEXT,
    speaker     TEXT
);
CREATE INDEX IF NOT EXISTS idx_media_doc ON media_segments(doc_id);

-- --------------------------------------------------- прайс-листы (товары) --
-- Цены НЕ идут через векторный поиск: галлюцинация цены недопустима.
CREATE TABLE IF NOT EXISTS products (
    id            INTEGER PRIMARY KEY,
    doc_id        INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    brand         TEXT,
    article       TEXT,
    name          TEXT,
    price         REAL,
    currency      TEXT DEFAULT 'RUB',
    unit          TEXT,
    attrs_json    TEXT,          -- прочие колонки прайса как есть
    price_date    TEXT,          -- дата прайс-листа
    is_current    INTEGER DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_products_article ON products(article);
CREATE INDEX IF NOT EXISTS idx_products_brand ON products(brand);

CREATE VIRTUAL TABLE IF NOT EXISTS products_fts USING fts5(
    article, name, brand, content='', tokenize='unicode61 remove_diacritics 2'
);

-- ------------------------------------------------ выверенные ответы (QA) --
-- Главный механизм «дообучения»: ответ, подтверждённый экспертом,
-- становится документом с наивысшим приоритетом.
CREATE TABLE IF NOT EXISTS golden_qa (
    id           INTEGER PRIMARY KEY,
    question     TEXT NOT NULL,
    answer       TEXT NOT NULL,
    source_refs  TEXT,           -- JSON: список chunk_id / путей
    author_id    INTEGER,
    created_at   TEXT,
    updated_at   TEXT,
    hits         INTEGER DEFAULT 0,
    active       INTEGER DEFAULT 1
);
CREATE VIRTUAL TABLE IF NOT EXISTS golden_fts USING fts5(
    question, answer, content='', tokenize='unicode61 remove_diacritics 2'
);

-- --------------------------------------------------- диалоги и обратная связь --
CREATE TABLE IF NOT EXISTS queries (
    id            INTEGER PRIMARY KEY,
    user_id       INTEGER,
    user_name     TEXT,
    role          TEXT,
    chat_id       INTEGER,
    question      TEXT NOT NULL,
    answer        TEXT,
    sources_json  TEXT,
    top_score     REAL,
    answered      INTEGER DEFAULT 1,   -- 0 = бот отказался (нет данных)
    latency_ms    INTEGER,
    tokens_in     INTEGER,
    tokens_out    INTEGER,
    created_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_queries_answered ON queries(answered);
CREATE INDEX IF NOT EXISTS idx_queries_created ON queries(created_at);

CREATE TABLE IF NOT EXISTS feedback (
    id           INTEGER PRIMARY KEY,
    query_id     INTEGER REFERENCES queries(id) ON DELETE CASCADE,
    user_id      INTEGER,
    verdict      TEXT,          -- up | down | fixed
    comment      TEXT,
    created_at   TEXT
);

-- Пары «вопрос → правильный чанк» для последующего дообучения
-- эмбеддера/реранкера (см. документ, раздел «самообучение»).
CREATE TABLE IF NOT EXISTS training_pairs (
    id           INTEGER PRIMARY KEY,
    question     TEXT NOT NULL,
    chunk_id     INTEGER,
    doc_id       INTEGER,
    label        INTEGER,       -- 1 = релевантен, 0 = нерелевантен
    source       TEXT,          -- feedback | expert | auto
    created_at   TEXT
);

-- Кэш эмбеддингов по хэшу текста: переиндексация не платит дважды.
CREATE TABLE IF NOT EXISTS embedding_cache (
    text_hash   TEXT PRIMARY KEY,
    provider    TEXT,
    dim         INTEGER,
    vector      BLOB
);

-- Пользователи бота и их роли.
CREATE TABLE IF NOT EXISTS users (
    user_id     INTEGER PRIMARY KEY,
    user_name   TEXT,
    full_name   TEXT,
    role        TEXT,
    approved    INTEGER DEFAULT 0,
    created_at  TEXT
);
"""


def _prepare(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Общие настройки соединения — одинаковые для обеих баз."""
    conn.row_factory = sqlite3.Row
    # WAL быстрее и, главное, позволяет читать во время записи: без него
    # бот встаёт на время индексации. На сетевых и смонтированных дисках
    # (FUSE, SMB) WAL не работает — там откатываемся на обычный журнал.
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    except sqlite3.OperationalError:
        conn.execute("PRAGMA journal_mode=DELETE")
    # Ждать освобождения блокировки, а не падать сразу: при параллельной
    # работе бота, слежения за папкой и админки короткие пересечения
    # неизбежны, и правильная реакция на них — подождать.
    conn.execute(f"PRAGMA busy_timeout={config.DB_BUSY_TIMEOUT_MS}")
    return conn


def connect() -> sqlite3.Connection:
    """Соединение на поток (SQLite не любит шаринг между потоками)."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        conn = _prepare(sqlite3.connect(config.DB_PATH, timeout=30,
                                        check_same_thread=False))
        conn.executescript(SCHEMA)
        _migrate(conn)
        _local.conn = conn
    return conn


TELEMETRY_SCHEMA = """
CREATE TABLE IF NOT EXISTS telemetry_marker (created_at TEXT);
"""


def telemetry() -> sqlite3.Connection:
    """
    Отдельная база для телеметрии: метрик сервера и записей журнала.

    Почему врозь. Эти две вещи пишутся постоянно — метрики каждые
    несколько секунд, журнал на каждое событие, — а читаются редко и
    никогда не соединяются с документами. Держать их в одной базе
    с индексом означает, что фоновая запись метрик конкурирует за
    блокировку с индексацией и с ответами бота.

    Разнесение по файлам снимает это полностью и стоит одной строки в
    настройках. Полный переход на PostgreSQL остаётся следующим шагом,
    но именно эта, самая частая, конкуренция закрывается уже здесь.
    """
    if not config.TELEMETRY_SEPARATE:
        return connect()
    conn = getattr(_local, "telemetry", None)
    if conn is None:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        conn = _prepare(sqlite3.connect(config.TELEMETRY_PATH, timeout=30,
                                        check_same_thread=False))
        conn.executescript(TELEMETRY_SCHEMA)
        _local.telemetry = conn
        _move_telemetry_tables(conn)
    return conn


TELEMETRY_TABLES = ("server_metrics", "model_usage", "stage_timings", "log_records")


def _move_telemetry_tables(tel: sqlite3.Connection) -> None:
    """
    Переносит накопленную телеметрию из основной базы в отдельную.

    Выполняется один раз при обновлении. Данные не теряются: сначала
    копируем, убеждаемся, что скопировалось, и только потом убираем
    из основной базы.
    """
    main = connect()
    for table in TELEMETRY_TABLES:
        exists = main.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                              (table,)).fetchone()
        if not exists:
            continue
        rows = main.execute(f"SELECT * FROM {table}").fetchall()
        if rows:
            already = tel.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table,)).fetchone()
            if not already:
                ddl = main.execute("SELECT sql FROM sqlite_master WHERE name=?",
                                   (table,)).fetchone()[0]
                tel.executescript(ddl + ";")
            columns = [d[0] for d in main.execute(f"SELECT * FROM {table} LIMIT 1").description]
            placeholders = ",".join("?" * len(columns))
            tel.executemany(
                f"INSERT OR IGNORE INTO {table}({','.join(columns)}) VALUES ({placeholders})",
                [tuple(r) for r in rows])
            tel.commit()
            moved = tel.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            if moved < len(rows):
                continue                 # что-то пошло не так — исходное не трогаем
        main.execute(f"DROP TABLE IF EXISTS {table}")
        main.commit()


def tq(sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
    return telemetry().execute(sql, params).fetchall()


def tq1(sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
    return telemetry().execute(sql, params).fetchone()


def trun(sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
    conn = telemetry()
    cur = conn.execute(sql, params)
    conn.commit()
    return cur


# Версия схемы. Растёт при каждом изменении, которое нельзя выразить
# простым добавлением колонки. Пока все изменения — добавления, поэтому
# миграции идемпотентны; версия нужна, чтобы при первом же необратимом
# изменении было от чего отталкиваться, а не гадать по набору колонок.
SCHEMA_VERSION = 3


def schema_version(conn: sqlite3.Connection) -> int:
    conn.execute("CREATE TABLE IF NOT EXISTS schema_meta ("
                 "key TEXT PRIMARY KEY, value TEXT)")
    row = conn.execute("SELECT value FROM schema_meta WHERE key='version'").fetchone()
    return int(row[0]) if row else 0


def _set_schema_version(conn: sqlite3.Connection, version: int) -> None:
    conn.execute("INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('version', ?)",
                 (str(version),))
    conn.execute("INSERT OR REPLACE INTO schema_meta(key, value) "
                 "VALUES ('migrated_at', datetime('now'))")
    conn.commit()


def _add_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    """
    Добавляет колонку, если её нет.

    Проверка «нет колонки — добавить» не атомарна, а соединений у нас
    много: своё у каждого потока, плюс отдельные процессы бота и админки.
    На пустой базе они стартуют одновременно, все видят, что колонки нет,
    и вторая попытка падает. Раньше это выглядело как случайная ошибка при
    первом запуске после обновления — воспроизводилась она только под
    нагрузкой, поэтому и жила долго. Ошибка «колонка уже есть» здесь
    означает ровно то, чего мы и добивались.
    """
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
    except sqlite3.OperationalError as exc:
        if "duplicate column name" not in str(exc).lower():
            raise


def _migrate(conn: sqlite3.Connection) -> None:
    """Добавляет колонки, появившиеся позже, не ломая существующий индекс."""
    was = schema_version(conn)
    have = {r[1] for r in conn.execute("PRAGMA table_info(documents)")}
    for column, ddl in (("kind", "TEXT DEFAULT 'text'"),
                        ("asset_kind", "TEXT"),
                        ("source_type", "TEXT DEFAULT 'internal_kb'"),
                        ("source_url", "TEXT"),
                        ("enriched", "TEXT"),
                        # результат распознавания сканов
                        ("ocr_provider", "TEXT"),
                        ("ocr_at", "TEXT"),
                        ("ocr_pages", "INTEGER"),
                        ("ocr_quality", "REAL"),
                        ("ocr_error", "TEXT")):
        if column not in have:
            _add_column(conn, "documents", column, ddl)

    # Колонки, по которым строится воронка ответа и вклад каналов.
    # Без них видно только «ответил или нет», а не где именно теряются
    # ответы: в поиске, на пороге уверенности или в генерации.
    # Заявки на доступ: сотрудник пишет боту, администратор подтверждает
    # в веб-интерфейсе. Без этих полей нельзя ни отличить «ещё не просил»
    # от «отказано», ни понять, кто и когда выдал доступ.
    have_u = {r[1] for r in conn.execute("PRAGMA table_info(users)")}
    for column, ddl in (("status", "TEXT DEFAULT 'new'"),   # new|pending|approved|denied|blocked
                        ("requested_at", "TEXT"),
                        ("decided_at", "TEXT"),
                        ("decided_by", "TEXT"),
                        ("note", "TEXT"),
                        ("last_seen", "TEXT"),
                        ("questions", "INTEGER DEFAULT 0"),
                        # Признак «дообучение»: сотрудник может добавлять
                        # выверенные ответы из Telegram, не будучи админом.
                        ("trainer", "INTEGER DEFAULT 0")):
        if column not in have_u:
            _add_column(conn, "users", column, ddl)
    # Разовое согласование старых записей с новой колонкой. Раньше этот
    # UPDATE выполнялся при каждом открытии соединения, а соединение
    # заводится на каждый поток, то есть на каждый запрос к админке. Это
    # означало транзакцию записи на каждый просмотр страницы —
    # конкурирующую за блокировку базы с идущей индексацией. Условие
    # ниже делает его настоящим no-op, когда согласовывать нечего.
    if conn.execute("SELECT 1 FROM users WHERE approved=1 AND "
                    "(status IS NULL OR status='new') LIMIT 1").fetchone():
        conn.execute("UPDATE users SET status='approved' WHERE approved=1 AND "
                     "(status IS NULL OR status='new')")

    # Выверенные ответы пишет человек, и они запросто содержат дилерские
    # условия: эксперт отвечал на вопрос дилера и о разграничении не думал.
    # Этот канал идёт первым и раньше проверок не проходил вовсе, то есть
    # был самым коротким путём к утечке. Пустое значение означает «ответ
    # общий», список разделов — «виден только тем, кому открыты все они».
    have_g = {r[1] for r in conn.execute("PRAGMA table_info(golden_qa)")}
    for column, ddl in (("sections", "TEXT"),):
        if column not in have_g:
            _add_column(conn, "golden_qa", column, ddl)

    have_q = {r[1] for r in conn.execute("PRAGMA table_info(queries)")}
    for column, ddl in (("route", "TEXT"),            # golden|price|documents|none
                        ("stage", "TEXT"),            # где остановились
                        ("channels", "TEXT"),         # каналы лучшего фрагмента
                        ("n_candidates", "INTEGER"),  # сколько нашёл поиск
                        ("rerank_used", "INTEGER DEFAULT 0"),
                        ("trace_id", "INTEGER")):
        if column not in have_q:
            _add_column(conn, "queries", column, ddl)
    conn.commit()

    if was != SCHEMA_VERSION:
        _set_schema_version(conn, SCHEMA_VERSION)
        if was:
            import logging_setup
            logging_setup.get("db").info(
                "схема базы обновлена: версия %d → %d", was, SCHEMA_VERSION)


def init() -> None:
    connect()


def q(sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
    return connect().execute(sql, params).fetchall()


def q1(sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
    return connect().execute(sql, params).fetchone()


def run(sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
    conn = connect()
    cur = conn.execute(sql, params)
    conn.commit()
    return cur


def runmany(sql: str, seq: Iterable[Sequence[Any]]) -> None:
    conn = connect()
    conn.executemany(sql, seq)
    conn.commit()


# --------------------------------------------------------------- векторы ----
def atomic_write(path: Path, write: "Callable[[Any], Any]") -> None:
    """
    Записать файл так, чтобы он никогда не оказался наполовину записанным.

    Пишем во временный файл рядом, сбрасываем на диск и переименовываем.
    Переименование внутри одной файловой системы атомарно: читатель видит
    либо старое содержимое целиком, либо новое целиком, и никогда —
    обрезанное. Для индекса, модели поиска и файла настроек это
    принципиально: обрезанный файл система читает молча и продолжает
    работать «как будто ничего нет», а обнаруживается это по жалобам
    через недели.

    Временный файл кладём рядом, а не в /tmp: переименование между
    разными файловыми системами не атомарно и вырождается в копирование.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    try:
        with open(tmp, "wb") as fh:
            write(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


class VectorStore:
    """
    Плоский индекс в numpy: cosine = dot(нормированные векторы).

    До ~1 млн чанков это работает за десятки миллисекунд и не требует
    отдельного сервиса. При росте базы меняется на Qdrant — интерфейс
    (add / search / drop_doc) остаётся тем же.
    """

    def __init__(self) -> None:
        self.dim = config.EMBEDDINGS_DIM
        self.ids: list[int] = []
        self.matrix: np.ndarray = np.zeros((0, self.dim), dtype=np.float32)
        self._index: dict[int, int] = {}
        # Свежедобавленные векторы копятся здесь и вливаются в матрицу
        # одним vstack перед чтением. Раньше vstack делался на каждый
        # add — на большой базе это квадратичное копирование гигабайтной
        # матрицы, и ночная индексация «зависала» именно тут.
        self._pending: list[np.ndarray] = []
        self.load()

    # --- персистентность ---
    def load(self) -> None:
        """
        Читает векторы с диска.

        Ошибку чтения раньше гасили молча, обнуляя индекс. Это худшее из
        возможных поведений: система поднимается как ни в чём не бывало,
        смысловой канал поиска исчезает, а узнают об этом по жалобам
        сотрудников через недели. Теперь повреждение — громкая запись в
        журнал и пометка, которую видно в диагностике.
        """
        self.broken = ""
        self._pending = []
        if not (Path(config.VECTORS_PATH).exists()
                and Path(config.VECTOR_IDS_PATH).exists()):
            return
        try:
            # Большая матрица читается через mmap: процессам, которые
            # векторы только читают (бот, админка), незачем держать в
            # памяти гигабайты целиком. Дозапись всё равно материализует
            # матрицу — это плата индексации, а не каждого вопроса.
            size_mb = Path(config.VECTORS_PATH).stat().st_size / 2**20
            mmap_mode = ("r" if config.VECTORS_MMAP_MB
                         and size_mb > config.VECTORS_MMAP_MB else None)
            matrix = np.load(config.VECTORS_PATH, mmap_mode=mmap_mode)
            ids = json.loads(Path(config.VECTOR_IDS_PATH).read_text())
            # Рассогласование — тоже повреждение: обычно это остановка
            # процесса между записью двух файлов.
            if len(ids) != matrix.shape[0]:
                raise ValueError(f"векторов {matrix.shape[0]}, идентификаторов "
                                 f"{len(ids)} — файлы рассогласованы")
            self.matrix, self.ids = matrix, ids
            self.dim = self.matrix.shape[1] if self.matrix.size else self.dim
            self._index = {cid: i for i, cid in enumerate(self.ids)}
        except Exception as exc:  # noqa: BLE001
            self.broken = str(exc)
            self.ids, self.matrix, self._index = [], np.zeros((0, self.dim), np.float32), {}
            import logging_setup
            logging_setup.get("db").error(
                "векторный индекс повреждён (%s). Смысловой поиск отключён, "
                "работает только поиск по точным словам. Восстановите из копии "
                "или пересчитайте: python index.py reembed", exc)

    def save(self) -> None:
        """
        Сохраняет векторы.

        Пишем во временные файлы и переименовываем. Переименование внутри
        одной файловой системы атомарно, поэтому на диске всегда лежит
        либо старая пара файлов целиком, либо новая целиком. Раньше
        запись шла поверх и двумя отдельными шагами, а остановка процесса
        между ними (при `docker stop` это происходило каждый раз)
        оставляла обрезанный файл векторов или рассогласованную пару.
        """
        self._flush_pending()
        # Матрица, отображённая из этого же файла и не изменившаяся,
        # в перезаписи не нуждается — а перезапись через самоё себя
        # ещё и опасна.
        if isinstance(self.matrix, np.memmap):
            return
        atomic_write(Path(config.VECTORS_PATH), lambda fh: np.save(fh, self.matrix))
        atomic_write(Path(config.VECTOR_IDS_PATH),
                     lambda fh: fh.write(json.dumps(self.ids).encode()))

    def _flush_pending(self) -> None:
        """Вливает накопленные пачки в матрицу одним vstack."""
        if not self._pending:
            return
        parts = ([np.asarray(self.matrix)] if self.matrix.size else []) + self._pending
        self.matrix = np.vstack(parts)
        self._pending = []

    # --- запись ---
    def add(self, chunk_ids: Sequence[int], vectors: np.ndarray) -> None:
        if len(chunk_ids) == 0:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.clip(norms, 1e-9, None)
        start = len(self.ids)
        self._pending.append(vectors)
        self.ids = list(self.ids) if not isinstance(self.ids, list) else self.ids
        self.ids.extend(int(c) for c in chunk_ids)
        # Индекс дополняется, а не перестраивается: перестройка словаря
        # на каждую пачку — это O(n²) на всю индексацию.
        for i, cid in enumerate(chunk_ids):
            self._index[int(cid)] = start + i
        self.dim = vectors.shape[1]

    def drop_chunks(self, chunk_ids: Iterable[int]) -> None:
        drop = set(chunk_ids)
        if not drop or not self.ids:
            return
        self._flush_pending()
        keep = [i for i, cid in enumerate(self.ids) if cid not in drop]
        self.matrix = self.matrix[keep] if keep else np.zeros((0, self.dim), np.float32)
        self.ids = [self.ids[i] for i in keep]
        self._index = {cid: i for i, cid in enumerate(self.ids)}

    # --- чтение ---
    def search(self, vector: np.ndarray, top_k: int,
               allowed: set[int] | None = None) -> list[tuple[int, float]]:
        self._flush_pending()
        if self.matrix.size == 0:
            return []
        v = np.asarray(vector, dtype=np.float32).reshape(-1)
        v = v / max(float(np.linalg.norm(v)), 1e-9)
        if v.shape[0] != self.matrix.shape[1]:
            return []
        scores = self.matrix @ v
        if allowed is not None:
            mask = np.array([cid in allowed for cid in self.ids])
            scores = np.where(mask, scores, -1.0)
        n = min(top_k, len(scores))
        idx = np.argpartition(-scores, n - 1)[:n] if n < len(scores) else np.arange(len(scores))
        idx = idx[np.argsort(-scores[idx])]
        return [(self.ids[i], float(scores[i])) for i in idx if scores[i] > -1.0]

    def __len__(self) -> int:
        return len(self.ids)


_store: VectorStore | None = None


def vectors():
    """
    Хранилище векторов: файл рядом с индексом или Qdrant.

    Оба варианта отвечают на одни и те же вызовы (add, drop_chunks,
    search, save, len), поэтому поиск и индексация не знают, какой из
    них используется, и переключение не требует изменений в коде.
    """
    global _store
    if _store is None:
        if config.VECTOR_BACKEND == "qdrant":
            try:
                import vectors_qdrant
                _store = vectors_qdrant.QdrantStore()
            except Exception as exc:  # noqa: BLE001
                import logging_setup
                logging_setup.get("db").error(
                    "Qdrant недоступен (%s) — работаю на файловом хранилище. "
                    "Проверьте QDRANT_URL или верните VECTOR_BACKEND=numpy", exc)
                _store = VectorStore()
        else:
            _store = VectorStore()
    return _store


def reset_vectors() -> None:
    """Забыть открытое хранилище — после смены настроек или восстановления."""
    global _store
    _store = None
