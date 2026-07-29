"""
Индексация базы знаний.

  python index.py build            — полный проход (первый запуск)
  python index.py update           — инкрементально: только изменённое
  python index.py watch            — следить за папкой и доиндексировать
  python index.py stats            — что в индексе
  python index.py train-lsa        — обучить смысловую модель на своей базе
  python index.py reembed          — пересчитать векторы (смена модели поиска)
  python index.py ocr-queue        — список сканов, требующих OCR

Инкрементальность: файл переиндексируется, только если изменился его
sha256. Переименование папки, копирование, синхронизация облака —
всё это не вызывает лишней работы. Удалённые файлы вычищаются сверкой
(reconciliation) на каждом проходе.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import chunk as chunker
import config
import db
import embeddings
import extract
import logging_setup
import prices
import shutdown

log = logging_setup.get("index")

# Первая причина, по которой не считаются векторы. Хранится, чтобы не
# засорять журнал одним и тем же сообщением на каждом файле и чтобы
# сказать о ней в конце индексации — там её точно увидят.
_embedding_problem: str | None = None

try:
    from webui import emit                    # телеметрия для схемы в админке
except Exception:                             # noqa: BLE001
    def emit(*_a, **_kw): pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        kind, _reason = extract.classify(path)
        if kind != "skip":
            yield path, kind


def _fts_insert(chunk_ids: list[int]) -> None:
    conn = db.connect()
    rows = conn.execute(f"""
        SELECT c.id, c.text, c.heading, c.context, d.brand, d.doc_type, d.file_name
        FROM chunks c JOIN documents d ON d.id=c.doc_id
        WHERE c.id IN ({','.join('?' * len(chunk_ids))})""", chunk_ids).fetchall()
    conn.executemany(
        "INSERT INTO chunks_fts(rowid, text, heading, context, brand, doc_type, file_name) "
        "VALUES (?,?,?,?,?,?,?)",
        [(r["id"], r["text"] or "", r["heading"] or "", r["context"] or "",
          r["brand"] or "", r["doc_type"] or "", r["file_name"] or "") for r in rows])
    conn.commit()


def _drop_document(doc_id: int) -> None:
    """Полностью убирает документ из всех индексов."""
    conn = db.connect()
    chunk_ids = [r["id"] for r in conn.execute("SELECT id FROM chunks WHERE doc_id=?", (doc_id,))]
    if chunk_ids:
        conn.executemany("INSERT INTO chunks_fts(chunks_fts, rowid, text, heading, context, "
                         "brand, doc_type, file_name) VALUES ('delete',?,?,?,?,?,?,?)",
                         [(cid, "", "", "", "", "", "") for cid in chunk_ids])
        db.vectors().drop_chunks(chunk_ids)
    conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
    conn.execute("DELETE FROM products WHERE doc_id=?", (doc_id,))
    conn.commit()


_last_vec_save = 0.0


def index_file(path: Path, force: bool = False, verbose: bool = False) -> str:
    """Возвращает: indexed | unchanged | skipped | error."""
    try:
        rel = str(path.relative_to(config.KB_ROOT))
    except ValueError:
        rel = "unpacked/" + str(path.relative_to(config.ARCHIVE_WORK_DIR))
    stat = path.stat()
    content_hash = extract.file_hash(path)

    existing = db.q1("SELECT * FROM documents WHERE rel_path=?", (rel,))
    if existing and existing["content_hash"] == content_hash and not force \
            and existing["status"] == "ok":
        db.run("UPDATE documents SET status='ok' WHERE id=?", (existing["id"],))
        return "unchanged"

    # Дедупликация по содержимому: один и тот же прайс лежит и в 1КАТАЛОГ,
    # и в 2ПРАЙС_ЛИСТ, сертификаты дублируются по нескольким папкам.
    # Индексируем один раз, второй экземпляр помечаем дублем.
    twin = db.q1("SELECT id, rel_path FROM documents WHERE content_hash=? AND rel_path<>? "
                 "AND status='ok' LIMIT 1", (content_hash, rel))
    if twin and not force:
        db.run("""INSERT OR REPLACE INTO documents(id, rel_path, abs_path, file_name, ext,
                  content_hash, size_bytes, mtime, indexed_at, status, error, is_current)
                  VALUES ((SELECT id FROM documents WHERE rel_path=?),?,?,?,?,?,?,?,?,?,?,0)""",
               (rel, rel, str(path), path.name, path.suffix.lower(), content_hash,
                stat.st_size, stat.st_mtime, _now(), "duplicate",
                f"дубль: {twin['rel_path']}"))
        return "skipped"

    meta = extract.path_meta(path)
    kind, _ = extract.classify(path)

    if kind == "archive":
        result, kind = _handle_archive(path, meta, verbose)
    elif kind == "asset":
        result = extract.asset_card(path, meta)
    else:
        result = extract.extract(path)
        # Чертёж, экспортированный в PDF, несёт только основную надпись —
        # десятки символов. Это мало для «документа», но достаточно, чтобы
        # найти модель. Поэтому дополняем текст карточкой, а не выбрасываем.
        if config.ASSET_CARDS and not result.error and 0 < result.n_chars < 400:
            card = extract.asset_card(path, meta, description=result.text.strip()[:400])
            result = extract.Extracted(card.pages, 1, needs_ocr=result.needs_ocr)
        elif config.ASSET_CARDS and not result.error and result.n_chars == 0:
            result = extract.asset_card(path, meta)
            result.needs_ocr = True

    if existing:
        _drop_document(existing["id"])
        doc_id = existing["id"]
        db.run("""UPDATE documents SET abs_path=?, file_name=?, ext=?, section=?, brand=?,
                  doc_type=?, content_hash=?, size_bytes=?, mtime=?, effective_date=?,
                  version_key=?, pages=?, text_chars=?, needs_ocr=?, indexed_at=?,
                  status=?, error=?, is_current=1, superseded_by=NULL WHERE id=?""",
               (str(path), path.name, path.suffix.lower(), meta.section, meta.brand,
                meta.doc_type, content_hash, stat.st_size, stat.st_mtime, meta.effective_date,
                meta.version_key, result.n_pages, result.n_chars, int(result.needs_ocr),
                _now(), "error" if result.error else "ok", result.error, doc_id))
    else:
        cur = db.run("""INSERT INTO documents(rel_path, abs_path, file_name, ext, section,
                        brand, doc_type, content_hash, size_bytes, mtime, effective_date,
                        version_key, pages, text_chars, needs_ocr, indexed_at, status, error)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                     (rel, str(path), path.name, path.suffix.lower(), meta.section, meta.brand,
                      meta.doc_type, content_hash, stat.st_size, stat.st_mtime,
                      meta.effective_date, meta.version_key, result.n_pages, result.n_chars,
                      int(result.needs_ocr), _now(),
                      "error" if result.error else "ok", result.error))
        doc_id = int(cur.lastrowid)

    # Содержимое папок «Архив», «Старые» и т.п. в выдачу не попадает,
    # но остаётся в индексе — его можно найти явным запросом истории.
    if meta.is_archive:
        db.run("UPDATE documents SET is_current=0 WHERE id=?", (doc_id,))
    db.run("UPDATE documents SET kind=?, asset_kind=? WHERE id=?",
           (kind, extract.asset_kind(path) if kind == "asset" else None, doc_id))

    if result.error:
        if verbose:
            print(f"   ошибка: {rel}: {result.error}")
        return "error"

    # Прайс-листы дополнительно уходят в структурированную таблицу.
    if kind == "table" and path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        n = prices.index_price_file(doc_id, path, meta.brand, meta.effective_date)
        if verbose and n:
            print(f"   прайс: {n} позиций")

    if result.n_chars < 40:
        return "skipped"

    doc_meta = {"brand": meta.brand, "doc_type": meta.doc_type, "section": meta.section,
                "file_name": path.name, "effective_date": meta.effective_date}
    chunks = chunker.chunk_document(result.pages, doc_meta)
    if not chunks:
        return "skipped"

    conn = db.connect()
    chunk_ids: list[int] = []
    for c in chunks:
        cur = conn.execute(
            "INSERT INTO chunks(doc_id, ord, page_from, page_to, heading, context, text, n_chars) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (doc_id, c.ord, c.page_from, c.page_to, c.heading, c.context, c.text, len(c.text)))
        chunk_ids.append(int(cur.lastrowid))
    conn.commit()
    _fts_insert(chunk_ids)

    # Векторы считаются отдельно от разбора. Если смысловая модель ещё не
    # обучена или провайдер недоступен, документ всё равно остаётся в
    # индексе и находится текстовым поиском — терять часы разбора из-за
    # ненастроенной модели неправильно. Векторы досчитываются потом
    # командой reembed, о чём build сообщит в конце.
    try:
        vectors = embeddings.embed_texts([c.indexed_text for c in chunks])
        db.vectors().add(chunk_ids, vectors)
        conn.executemany("UPDATE chunks SET embedded=1 WHERE id=?",
                         [(cid,) for cid in chunk_ids])
        conn.commit()
    except Exception as exc:  # noqa: BLE001 — текстовый канал уже работает
        global _embedding_problem
        if _embedding_problem is None:
            _embedding_problem = str(exc)
            log.warning("векторы не считаются (%s). Документы индексируются "
                        "и находятся текстовым поиском; смысловой канал "
                        "включится после reembed.", exc)
        if verbose:
            print(f"   векторы не посчитаны: {exc}")
    return "indexed"


def _handle_archive(path: Path, meta: "extract.Meta", verbose: bool):
    """Распаковывает архив и создаёт карточку с перечнем содержимого."""
    dest = config.ARCHIVE_WORK_DIR / extract.file_hash(path)[:16]
    if dest.exists() and any(dest.iterdir()):
        files = [f for f in dest.rglob("*") if f.is_file()]
        n, err = len(files), None
    else:
        n, err = extract.unpack_archive(path, dest)
    if err:
        if verbose:
            print(f"   архив {path.name}: {err}")
        return extract.asset_card(path, meta, description=f"не удалось распаковать: {err}"), "asset"
    names = [f.name for f in dest.rglob("*") if f.is_file()][:40]
    card = extract.asset_card(
        path, meta,
        description=f"распакован, файлов внутри: {n}. Содержимое: " + "; ".join(names))
    if verbose:
        print(f"   архив {path.name}: распакован, файлов {n}")
    return card, "archive"


def reconcile(seen_paths: set[str]) -> int:
    """Удалённые с диска файлы убираем из индекса."""
    removed = 0
    for row in db.q("SELECT id, rel_path FROM documents WHERE status<>'deleted'"):
        if row["rel_path"] not in seen_paths:
            _drop_document(row["id"])
            db.run("UPDATE documents SET status='deleted', is_current=0 WHERE id=?", (row["id"],))
            removed += 1
    return removed


def build(force: bool = False, limit: int | None = None, verbose: bool = False,
          progress=None) -> dict:
    # say — единая точка вывода: в консоль при запуске из терминала,
    # в карточку задачи при запуске из админки.
    say = progress or (lambda t: print(t, flush=True))
    db.init()
    started = time.time()
    counts = {"indexed": 0, "unchanged": 0, "skipped": 0, "error": 0}
    seen: set[str] = set()
    files = [p for p, _k in iter_files(config.KB_ROOT)]
    # Содержимое распакованных архивов индексируется вторым проходом.
    if config.EXTRACT_ARCHIVES and config.ARCHIVE_WORK_DIR.exists():
        files += [p for p, _k in iter_files(config.ARCHIVE_WORK_DIR)]
    if limit:
        files = files[:limit]
    total = len(files)
    emit("scan", "ok", f"найдено файлов: {total}", total, total)
    say(f"Файлов к обработке: {total}")
    for i, path in enumerate(files, 1):
        try:
            seen.add(str(path.relative_to(config.KB_ROOT)))
        except ValueError:                      # файл из распакованного архива
            seen.add("unpacked/" + str(path.relative_to(config.ARCHIVE_WORK_DIR)))
        try:
            status = index_file(path, force=force, verbose=verbose)
        except Exception as exc:  # noqa: BLE001
            status = "error"
            if verbose:
                print(f"   сбой на {path.name}: {exc}")
        counts[status] = counts.get(status, 0) + 1
        # Останавливаться нужно в точке, где данные согласованы, а не там,
        # где застал сигнал. Дописываем векторы и выходим: обработанное
        # сохранено, следующий запуск продолжит с этого места.
        if shutdown.stopping():
            db.vectors().save()
            say(f"Остановка по сигналу: обработано {i} из {total}. "
                "Запустите индексацию снова — продолжится с этого места.")
            emit("extract", "ok", f"остановлено на {i} из {total}", i, total)
            return counts
        if i % 25 == 0 or i == total:
            emit("extract", "running" if i < total else "ok",
                 f"обработано {i} из {total}", i, total)
            emit("embed", "running" if i < total else "ok",
                 f"векторов: {len(db.vectors())}", len(db.vectors()), 0)
            # Сохранение — по времени, а не по числу файлов. Матрица
            # пишется на диск целиком: на большой базе сохранение каждые
            # 25 файлов означало терабайты записи и «зависшую» индексацию.
            global _last_vec_save
            if time.time() - _last_vec_save > 120:
                db.vectors().save()      # чтобы обрыв индексации не стоил всей работы
                _last_vec_save = time.time()
            done = time.time() - started
            rate = i / max(done, 0.01)
            eta = (total - i) / max(rate, 0.01)
            say(f"[{i}/{total}] {counts} | {rate:.1f} файл/с | "
                f"осталось ~{eta/60:.1f} мин")
    removed = reconcile(seen) if not limit else 0
    deprecated = prices.deprecate_older_prices()
    db.vectors().save()
    emit("extract", "ok", f"готово: {counts}", total, total)
    emit("prices", "ok", f"устаревших прайсов помечено: {deprecated}")
    say(f"Готово за {(time.time()-started)/60:.1f} мин. {counts}, "
        f"удалено из индекса: {removed}, устаревших прайсов помечено: {deprecated}")

    pending = db.q1("SELECT COUNT(*) n FROM chunks WHERE embedded=0")["n"]
    if pending:
        if _embedding_problem:
            say(f"Смысловой канал поиска не включён: {_embedding_problem}")
        say(f"Без векторов осталось фрагментов: {pending}. Сейчас работает "
            f"только поиск по точным словам — вопрос, заданный не теми "
            f"словами, что в документе, не найдётся.")
        say("Включить смысловой поиск: python index.py train-lsa, "
            "затем python index.py reembed")
    return counts


def watch(interval: int = 60) -> None:
    """
    Простое слежение опросом. Для продакшена лучше watchdog/inotify,
    но опрос по mtime+hash надёжнее работает на сетевых дисках и в облачных
    папках, где события файловой системы теряются.
    """
    print(f"Слежу за {config.KB_ROOT} (интервал {interval} c). Ctrl+C для выхода.")
    while True:
        try:
            build(force=False, verbose=False)
        except KeyboardInterrupt:
            print("Остановлено.")
            return
        except Exception as exc:  # noqa: BLE001
            print(f"Ошибка обхода: {exc}")
        time.sleep(interval)


def stats() -> None:
    db.init()
    total = db.q1("SELECT COUNT(*) n FROM documents WHERE status='ok'")["n"]
    chunks = db.q1("SELECT COUNT(*) n FROM chunks")["n"]
    products = db.q1("SELECT COUNT(*) n FROM products WHERE is_current=1")["n"]
    ocr = db.q1("SELECT COUNT(*) n FROM documents WHERE needs_ocr=1")["n"]
    errors = db.q1("SELECT COUNT(*) n FROM documents WHERE status='error'")["n"]
    chars = db.q1("SELECT COALESCE(SUM(text_chars),0) n FROM documents")["n"]
    print(f"Документов в индексе : {total}")
    print(f"Чанков               : {chunks}")
    print(f"Векторов             : {len(db.vectors())}")
    print(f"Позиций в прайсах    : {products}")
    print(f"Сканов (нужен OCR)   : {ocr}")
    print(f"Ошибок разбора       : {errors}")
    print(f"Символов текста      : {chars:,}".replace(",", " "))
    print("\nПо типам документов:")
    for r in db.q("""SELECT doc_type, COUNT(*) n, SUM(text_chars) chars FROM documents
                     WHERE status='ok' GROUP BY doc_type ORDER BY n DESC"""):
        print(f"  {str(r['doc_type'] or '—'):<22} {r['n']:>5}  {(r['chars'] or 0)//1000:>8} тыс. симв.")
    print("\nПо разделам:")
    sections = set()
    for r in db.q("""SELECT section, COUNT(*) n FROM documents WHERE status='ok'
                     GROUP BY section ORDER BY n DESC"""):
        if r["section"]:
            sections.add(r["section"])
        print(f"  {str(r['section'] or '—'):<40} {r['n']:>5}")

    # Проверка настройки ролей: раздел, не указанный ни в одной роли,
    # будет невидим для всех, кроме admin, — и это легко не заметить.
    covered = set()
    for role, allowed in config.ROLE_SECTIONS.items():
        if "*" not in allowed:
            covered |= set(allowed)
    orphans = sections - covered
    if orphans:
        print("\n⚠ Разделы, не указанные ни в одной роли (видны только роли admin):")
        for s in sorted(orphans):
            print(f"  {s}")
        print("  Поправьте ROLE_SECTIONS в config.py.")


def repair() -> int:
    """
    Досчитывает векторы для чанков, которых нет в векторном индексе.
    Нужно, если индексация была прервана: SQLite фиксирует чанки сразу,
    а матрица векторов сохраняется пачками.
    """
    db.init()
    store = db.vectors()
    have = set(store.ids)
    rows = db.q("SELECT id, text, heading, context FROM chunks")
    missing = [r for r in rows if r["id"] not in have]
    if not missing:
        print("Векторный индекс согласован.")
        return 0
    print(f"Досчитываю векторы для {len(missing)} чанков…")
    batch = 200
    for start in range(0, len(missing), batch):
        part = missing[start:start + batch]
        texts = ["\n".join(x for x in (r["context"], r["heading"], r["text"]) if x)
                 for r in part]
        store.add([r["id"] for r in part], embeddings.embed_texts(texts))
        store.save()
        print(f"  {min(start + batch, len(missing))}/{len(missing)}", flush=True)
    return len(missing)


def train_lsa(dim: int | None = None) -> None:
    """Обучает смысловую модель на текущем содержимом индекса."""
    db.init()
    print("Обучаю смысловую модель на вашей базе.")
    print("Это единственный шаг, который нужен для включения смыслового поиска;")
    print("интернет и видеокарта не требуются.\n")
    model = embeddings.LSAEmbedder.train(dim=dim)
    embeddings.reset()
    print(f"\nМодель сохранена: {config.LSA_MODEL_PATH}")
    print(f"Обучена на {model.meta.get('documents')} фрагментах, "
          f"словарь {model.meta.get('vocab')} слов, {model.dim} измерений.")
    print("\nТеперь пересчитайте векторы, чтобы поиск начал ими пользоваться:")
    print("    python index.py reembed")


def reembed(provider: str | None = None, batch: int = 256,
            only_missing: bool = False, progress=None) -> int:
    """
    Пересчитывает векторы по уже разобранным фрагментам.

    Смысл: разбор 116 ГБ файлов занимает часы, а смена модели поиска —
    минуты. Эта команда меняет только векторы, не трогая ни файлы, ни
    тексты фрагментов, ни выверенные ответы.
    """
    say = progress or (lambda t: print(t, flush=True))
    db.init()
    if provider:
        if provider not in embeddings.PROVIDERS:
            say(f"Неизвестный провайдер: {provider}. "
                f"Доступны: {', '.join(embeddings.PROVIDERS)}")
            return 0
        config.EMBEDDINGS_PROVIDER = provider
        embeddings.reset()

    info = embeddings.describe()
    if info.get("error"):
        say(f"Провайдер «{info['provider']}» не готов: {info['error']}")
        return 0
    say(f"Провайдер: {info['provider']} — {info['detail']}")
    if info["is_stub"]:
        say("ВНИМАНИЕ: это заглушка без смысловой близости. "
            "Для рабочего поиска нужен lsa или onnx.")

    store = db.vectors()
    if only_missing:
        have = set(store.ids)
        rows = [r for r in db.q("SELECT id FROM chunks") if r["id"] not in have]
        ids = [r["id"] for r in rows]
    else:
        # Полная пересборка: у другой модели другая размерность,
        # старые векторы несовместимы и должны уйти целиком.
        store.ids = []
        store.matrix = np.zeros((0, info["dim"] or config.EMBEDDINGS_DIM), dtype=np.float32)
        store._index = {}
        store._pending = []
        ids = [r["id"] for r in db.q(
            "SELECT c.id FROM chunks c JOIN documents d ON d.id=c.doc_id "
            "WHERE d.status='ok' ORDER BY c.id")]
    if not ids:
        say("Нечего пересчитывать.")
        return 0

    say(f"Пересчитываю {len(ids)} фрагментов пачками по {batch}…")
    started = time.time()
    done = 0
    for start in range(0, len(ids), batch):
        part = ids[start:start + batch]
        rows = db.q(f"SELECT id, context, heading, text FROM chunks "
                    f"WHERE id IN ({','.join('?' * len(part))})", part)
        rows = sorted(rows, key=lambda r: part.index(r["id"]))
        texts = ["\n".join(x for x in (r["context"], r["heading"], r["text"]) if x)
                 for r in rows]
        store.add([r["id"] for r in rows], embeddings.embed_texts(texts))
        done += len(rows)
        if start % (batch * 8) == 0 or done == len(ids):
            global _last_vec_save
            if time.time() - _last_vec_save > 120 or done == len(ids):
                store.save()
                _last_vec_save = time.time()
            speed = done / max(time.time() - started, 1e-6)
            left = (len(ids) - done) / max(speed, 1e-6)
            say(f"{done}/{len(ids)}  ~{speed:.0f} фрагм./с, "
                f"осталось ~{left / 60:.1f} мин")
    store.save()
    db.run("UPDATE chunks SET embedded=1")
    emit("reembed", {"provider": info["provider"], "chunks": done,
                     "seconds": round(time.time() - started, 1)})
    say(f"Готово: {done} векторов за {(time.time() - started) / 60:.1f} мин.")
    say("Проверьте поиск: python ask.py \"как подобрать насос для скважины\"")
    return done


def ocr_queue() -> None:
    db.init()
    rows = db.q("SELECT rel_path, pages FROM documents WHERE needs_ocr=1 ORDER BY pages DESC")
    print(f"Документов без текстового слоя: {len(rows)}")
    for r in rows[:200]:
        print(f"  {r['pages'] or '?':>4} стр.  {r['rel_path']}")


def main() -> None:
    import signal
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)   # тихий выход при | head
    shutdown.install("индексация")

    parser = argparse.ArgumentParser(description="Индексация корпоративной базы знаний")
    parser.add_argument("command", choices=["build", "update", "watch", "stats", "repair",
                                            "train-lsa", "reembed", "ocr-queue"])
    parser.add_argument("--force", action="store_true", help="переиндексировать всё заново")
    parser.add_argument("--limit", type=int, help="обработать только N файлов (для теста)")
    parser.add_argument("--interval", type=int, default=60, help="интервал watch, секунд")
    parser.add_argument("--provider", help="провайдер эмбеддингов для reembed")
    parser.add_argument("--dim", type=int, help="число измерений для train-lsa")
    parser.add_argument("--batch", type=int, default=256, help="размер пачки для reembed")
    parser.add_argument("--only-missing", action="store_true",
                        help="reembed: досчитать недостающие, не пересобирая всё")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.command in ("build", "update"):
        build(force=args.force or args.command == "build" and False,
              limit=args.limit, verbose=args.verbose)
    elif args.command == "watch":
        watch(args.interval)
    elif args.command == "stats":
        stats()
    elif args.command == "repair":
        repair()
    elif args.command == "train-lsa":
        train_lsa(dim=args.dim)
    elif args.command == "reembed":
        reembed(provider=args.provider, batch=args.batch, only_missing=args.only_missing)
    elif args.command == "ocr-queue":
        ocr_queue()


if __name__ == "__main__":
    sys.exit(main())
