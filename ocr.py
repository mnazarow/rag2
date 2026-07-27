"""
Распознавание сканов.

Зачем. Сертификаты и декларации о соответствии в базе лежат сканами:
текстового слоя в них нет, и для поиска этих документов попросту не
существует. При этом именно за ними обращаются чаще всего — их просят
контрагенты и запрашивает тендерная площадка. Один прогон распознавания
превращает несколько сотен «невидимых» файлов в находимые.

  python ocr.py queue                 — что ждёт распознавания
  python ocr.py run                   — распознать всё, что ждёт
  python ocr.py run --limit 20        — только первые 20 (проба)
  python ocr.py check ФАЙЛ            — распознать один файл и показать результат
  python ocr.py providers             — что доступно на этой машине

О подмене кириллицы латиницей
-----------------------------
Главная опасность распознавания русских документов — не пропущенная
буква, а подмена: «МОСКВА» превращается в «MOCKBA», где все буквы
латинские. Внешне текст выглядит правильным, а для поиска документ
потерян навсегда: запрос «Москва» его больше не найдёт. То же
происходит с артикулами вида «СП-45» → «CП-45».

Здесь это лечится в три приёма:

  1. Смешанные слова, где к русским буквам примешаны латинские
     двойники, чинятся автоматически — направление подмены очевидно.
  2. Целиком «латинские» слова сверяются со словарём вашей же базы:
     если такого слова в базе нет, а его русское прочтение есть —
     слово исправляется. Словарь берётся из обученной смысловой модели.
  3. Если после этого доля подозрительных слов на странице всё ещё
     выше порога, страница признаётся испорченной и переотправляется
     запасному провайдеру (OCR_FALLBACK). Плохой текст в индекс
     не попадает: пустое место честнее подделки.
"""
from __future__ import annotations

import argparse
import base64
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import config
import db
import logging_setup

log = logging_setup.get("ocr")


class OcrError(RuntimeError):
    pass


# ------------------------------------------------- защита от подмены букв ---
# Латинские буквы, неотличимые на вид от русских. Только они и участвуют
# в проверке: «Grundfos» рядом с русским текстом — это нормально и трогать
# его нельзя.
HOMOGLYPHS = {
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М",
    "O": "О", "P": "Р", "T": "Т", "X": "Х", "Y": "У",
    "a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у",
}
_LATIN = set(HOMOGLYPHS)
_WORD_RX = re.compile(r"[A-Za-zА-Яа-яЁё]+")

_vocab_cache: set[str] | None = None


def corpus_vocab() -> set[str]:
    """
    Словарь слов вашей базы — из обученной смысловой модели.

    Он нужен, чтобы отличить настоящее латинское слово от подменённого
    русского: «MOCKBA» в базе не встречается ни разу, а «москва» —
    сплошь и рядом.
    """
    global _vocab_cache
    if _vocab_cache is not None:
        return _vocab_cache
    _vocab_cache = set()
    try:
        import lsa
        path = Path(config.LSA_MODEL_PATH)
        if path.exists():
            _vocab_cache = set(lsa.LSAModel.load(path).vocab)
    except Exception:  # noqa: BLE001 — без словаря просто меньше исправлений
        pass
    return _vocab_cache


def _to_cyrillic(word: str) -> str:
    return "".join(HOMOGLYPHS.get(ch, ch) for ch in word)


def _classify(word: str) -> str:
    """latin | cyrillic | mixed — какими буквами написано слово."""
    has_lat = any("a" <= c.lower() <= "z" for c in word)
    has_cyr = any(c.lower() in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя" for c in word)
    if has_lat and has_cyr:
        return "mixed"
    return "latin" if has_lat else "cyrillic"


def repair_homoglyphs(text: str) -> tuple[str, dict]:
    """
    Чинит подменённые буквы и оценивает, насколько тексту можно верить.

    Возвращает исправленный текст и отчёт: сколько слов починено, сколько
    осталось подозрительных и какая доля от всех слов это составляет.
    """
    import lsa
    vocab = corpus_vocab()
    stats = {"words": 0, "fixed_mixed": 0, "fixed_latin": 0, "suspicious": 0}

    def replace(match: re.Match) -> str:
        word = match.group(0)
        stats["words"] += 1
        kind = _classify(word)

        if kind == "mixed":
            # Русское слово с вкраплениями латиницы. Если все латинские
            # буквы — двойники, направление подмены однозначно.
            latin = [c for c in word if "a" <= c.lower() <= "z"]
            if all(c in _LATIN for c in latin):
                stats["fixed_mixed"] += 1
                return _to_cyrillic(word)
            stats["suspicious"] += 1
            return word

        if kind == "latin" and len(word) >= 2 and all(c in _LATIN for c in word):
            # Слово целиком из двойников: либо настоящая латинская
            # аббревиатура, либо подменённое русское. Решает словарь базы.
            cyr = _to_cyrillic(word)
            if vocab:
                lat_known = lsa.normalize_token(word.lower()) in vocab
                cyr_known = lsa.normalize_token(cyr.lower()) in vocab
                if cyr_known and not lat_known:
                    stats["fixed_latin"] += 1
                    return cyr
                if lat_known:
                    return word
            stats["suspicious"] += 1
            return word
        return word

    fixed = _WORD_RX.sub(replace, text)
    total = max(stats["words"], 1)
    stats["ratio"] = stats["suspicious"] / total
    stats["fixed"] = stats["fixed_mixed"] + stats["fixed_latin"]
    return fixed, stats


def quality(text: str, stats: dict) -> float:
    """Грубая оценка «на сколько этому тексту можно верить», 0…1."""
    if not text.strip():
        return 0.0
    letters = sum(1 for c in text if c.isalpha())
    if letters < 20:
        return 0.0
    # Мусорные символы — верный признак неудачного распознавания.
    junk = sum(1 for c in text if c in "|~^`¦°¤§") / max(len(text), 1)
    score = 1.0 - min(stats.get("ratio", 0.0) * 3, 1.0) - min(junk * 5, 0.5)
    return float(max(0.0, min(1.0, score)))


# ----------------------------------------------------------- растеризация ---
def rasterize(path: Path, dpi: int | None = None,
              max_pages: int | None = None) -> list[Path]:
    """Переводит страницы PDF в картинки. Для изображений — возвращает как есть."""
    dpi = dpi or config.OCR_DPI
    max_pages = max_pages or config.OCR_MAX_PAGES
    if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"):
        return [path]
    if path.suffix.lower() != ".pdf":
        raise OcrError(f"не умею растеризовать {path.suffix}")
    if not shutil.which("pdftoppm"):
        raise OcrError("нужен pdftoppm из пакета poppler-utils "
                       "(Linux: apt install poppler-utils, macOS: brew install poppler)")
    out = Path(tempfile.mkdtemp(prefix="ocr_"))
    subprocess.run(
        ["pdftoppm", "-r", str(dpi), "-f", "1", "-l", str(max_pages),
         "-png", str(path), str(out / "p")],
        check=True, capture_output=True, timeout=config.OCR_TIMEOUT * 2)
    return sorted(out.glob("p*.png"))


# --------------------------------------------------------------- провайдеры --
class TesseractOCR:
    """
    Локально, бесплатно, без интернета. Требует языкового пакета rus:
    без него распознаётся латиница, и как раз получается «MOCKBA».
    Поэтому наличие пакета проверяется до запуска, а не по факту.
    """
    name = "tesseract"

    def __init__(self) -> None:
        cmd = config.OCR_TESSERACT_CMD
        if not shutil.which(cmd):
            raise OcrError(
                "tesseract не установлен. Linux: apt install tesseract-ocr "
                "tesseract-ocr-rus; macOS: brew install tesseract tesseract-lang")
        self.cmd = cmd
        # Первая строка вывода — заголовок «List of available languages…»,
        # сами языки идут дальше по одному в строке.
        out = subprocess.run([cmd, "--list-langs"], capture_output=True,
                             text=True).stdout.splitlines()
        langs = {line.strip() for line in out[1:] if line.strip()}
        need = [l for l in config.OCR_LANGUAGES.split("+") if l]
        missing = [l for l in need if l not in langs]
        if missing:
            raise OcrError(
                f"в tesseract нет языковых пакетов: {', '.join(missing)}. "
                f"Установлены: {', '.join(sorted(langs - {'osd'})) or 'нет'}. "
                f"Без пакета rus русский текст будет распознан латиницей — "
                f"это ровно та подмена, которой нельзя допускать. "
                f"Linux: apt install tesseract-ocr-rus; "
                f"macOS: brew install tesseract-lang")

    def read(self, image: Path) -> str:
        r = subprocess.run(
            [self.cmd, str(image), "stdout", "-l", config.OCR_LANGUAGES,
             "--psm", str(config.OCR_TESSERACT_PSM)],
            capture_output=True, text=True, timeout=config.OCR_TIMEOUT)
        if r.returncode != 0:
            raise OcrError(r.stderr.strip()[:300])
        return r.stdout


PROMPT = (
    "Перед тобой скан документа на русском языке — сертификат, декларация "
    "соответствия или паспорт изделия.\n"
    "Перепиши весь видимый текст дословно, сверху вниз, сохраняя строки, "
    "номера, даты и обозначения.\n"
    "Требования, обязательные к соблюдению:\n"
    "1. Русские буквы записывай русскими буквами. Никогда не заменяй их "
    "похожими латинскими: «МОСКВА», а не «MOCKBA».\n"
    "2. Не переводи, не пересказывай и не исправляй текст — только переписывай.\n"
    "3. Номера, артикулы и коды переписывай посимвольно.\n"
    "4. Таблицы передавай строками, разделяя ячейки знаком |.\n"
    "5. Если участок нечитаем, поставь [нрзб] вместо догадки.\n"
    "В ответе — только текст документа, без пояснений."
)


class VlmOCR:
    """
    Зрительная модель через OpenAI-совместимый endpoint.

    Лучший вариант по качеству на печатях, подписях и таблицах, и он же
    самый устойчивый к подмене букв: модель читает текст как язык,
    а не как набор форм. Подходит Qwen3-VL, в том числе поднятый локально
    на ваших двух картах — тогда сканы никуда не уходят из периметра.
    """
    name = "vlm"

    def __init__(self) -> None:
        import httpx
        base = config.OCR_BASE_URL or config.OPENAI_BASE_URL
        if not base:
            raise OcrError("не задан OCR_BASE_URL (или OPENAI_BASE_URL)")
        self.url = base.rstrip("/") + "/chat/completions"
        key = config.OCR_API_KEY or config.OPENAI_API_KEY
        self.client = httpx.Client(timeout=config.OCR_TIMEOUT,
                                   proxy=config.LLM_PROXY or None,
                                   headers={"Authorization": f"Bearer {key}"} if key else {})

    def read(self, image: Path) -> str:
        import llm_queue
        data = base64.b64encode(image.read_bytes()).decode()
        payload = {
            "model": config.OCR_MODEL,
            "temperature": 0,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{data}"}},
            ]}],
        }
        # Зрительная модель обычно стоит на тех же картах, что и модель
        # ответов, поэтому идёт в ту же очередь и с фоновой важностью:
        # распознавание трёхсот сканов не должно задерживать ответ в чате.
        with llm_queue.slot(source="сканы", priority=5, provider="vlm"):
            r = self.client.post(self.url, json=payload)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]


class YandexVisionOCR:
    """Yandex Vision OCR — российский контур, хорошо держит кириллицу."""
    name = "yandex"
    URL = "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText"

    def __init__(self) -> None:
        import httpx
        if not config.YANDEX_API_KEY:
            raise OcrError("не задан YANDEX_API_KEY")
        self.client = httpx.Client(timeout=config.OCR_TIMEOUT,
                                   proxy=config.LLM_PROXY or None)

    def read(self, image: Path) -> str:
        data = base64.b64encode(image.read_bytes()).decode()
        r = self.client.post(self.URL, headers={
            "Authorization": f"Api-Key {config.YANDEX_API_KEY}",
            "x-folder-id": config.YANDEX_FOLDER_ID,
            "x-data-logging-enabled": "false",
        }, json={"mimeType": "PNG", "languageCodes": ["ru", "en"],
                 "model": "page", "content": data})
        r.raise_for_status()
        result = r.json().get("result", {}).get("textAnnotation", {})
        return result.get("fullText", "")


PROVIDERS = {"tesseract": TesseractOCR, "vlm": VlmOCR, "yandex": YandexVisionOCR}


def make(name: str):
    if name not in PROVIDERS:
        raise OcrError(f"неизвестный провайдер OCR: {name}. "
                       f"Доступны: {', '.join(PROVIDERS)}")
    return PROVIDERS[name]()


def available() -> dict[str, str]:
    """Что реально можно запустить на этой машине — и почему нельзя остальное."""
    out = {}
    for name in PROVIDERS:
        try:
            make(name)
            out[name] = "готов"
        except Exception as exc:  # noqa: BLE001
            out[name] = str(exc)
    return out


# ------------------------------------------------------------- распознавание --
def read_document(path: Path, provider: str | None = None,
                  verbose: bool = False) -> dict:
    """
    Распознаёт один файл: страница за страницей, с проверкой каждой.

    Страница, не прошедшая проверку на подмену букв, переотправляется
    провайдерам из OCR_FALLBACK. Если не помогло — страница не попадает
    в индекс, а причина остаётся в отчёте.
    """
    chain = [provider or config.OCR_PROVIDER]
    chain += [p.strip() for p in config.OCR_FALLBACK.split(",") if p.strip()]
    chain = [p for i, p in enumerate(chain) if p and p not in chain[:i]]
    if not chain or chain == ["none"]:
        raise OcrError("OCR_PROVIDER не задан")

    engines = {}
    for name in chain:
        try:
            engines[name] = make(name)
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"   {name}: {exc}")
    if not engines:
        raise OcrError(f"ни один из провайдеров не запустился: {', '.join(chain)}")

    started = time.time()
    images = rasterize(path)
    pages: list[str] = []
    used: list[str] = []
    rejected = 0
    qualities: list[float] = []
    try:
        for i, image in enumerate(images, 1):
            best_text, best_q, best_name = "", -1.0, None
            for name in chain:
                engine = engines.get(name)
                if engine is None:
                    continue
                try:
                    raw = engine.read(image)
                except Exception as exc:  # noqa: BLE001
                    if verbose:
                        print(f"   стр. {i}: {name} — ошибка: {exc}")
                    continue
                text, stats = (repair_homoglyphs(raw) if config.OCR_CYRILLIC_GUARD
                               else (raw, {"ratio": 0.0, "fixed": 0}))
                q = quality(text, stats)
                if verbose:
                    print(f"   стр. {i}: {name} — {len(text)} симв., "
                          f"исправлено {stats.get('fixed', 0)}, "
                          f"подозрительных {stats.get('ratio', 0):.1%}, "
                          f"оценка {q:.2f}")
                if q > best_q:
                    best_text, best_q, best_name = text, q, name
                # Порог пройден — дальше по цепочке не идём.
                if (stats.get("ratio", 0.0) <= config.OCR_MAX_LATIN_RATIO
                        and len(text.strip()) >= config.OCR_MIN_CHARS):
                    break
            if best_name and best_q > 0 and len(best_text.strip()) >= config.OCR_MIN_CHARS:
                pages.append(best_text)
                used.append(best_name)
                qualities.append(best_q)
            else:
                rejected += 1
                pages.append("")
    finally:
        # Временные картинки убираем всегда, даже если распознавание упало.
        if images and images[0].parent != path.parent:
            shutil.rmtree(images[0].parent, ignore_errors=True)

    text = "\n\n".join(p for p in pages if p)
    return {
        "text": text, "pages": pages, "n_pages": len(images),
        "recognized": len(qualities), "rejected": rejected,
        "provider": used[0] if used else None,
        "providers": sorted(set(used)),
        "quality": round(sum(qualities) / len(qualities), 3) if qualities else 0.0,
        "chars": len(text), "seconds": round(time.time() - started, 1),
    }


# ------------------------------------------------------------- очередь ------
def queue(limit: int | None = None) -> list:
    db.init()
    sql = ("SELECT id, rel_path, abs_path, pages, ocr_provider, ocr_error "
           "FROM documents WHERE needs_ocr=1 AND status='ok' "
           "ORDER BY (ocr_error IS NOT NULL), pages")
    if limit:
        sql += f" LIMIT {int(limit)}"
    return db.q(sql)


def show_queue() -> None:
    rows = queue()
    done = db.q1("SELECT COUNT(*) n FROM documents WHERE ocr_provider IS NOT NULL")["n"]
    print(f"Ждут распознавания: {len(rows)}   уже распознано: {done}")
    failed = [r for r in rows if r["ocr_error"]]
    if failed:
        print(f"Из них с прошлой ошибкой: {len(failed)}")
    for r in rows[:40]:
        mark = "!" if r["ocr_error"] else " "
        print(f" {mark} {r['pages'] or '?':>4} стр.  {r['rel_path']}")
    if len(rows) > 40:
        print(f"   … и ещё {len(rows) - 40}")


def apply_to_index(doc_id: int, result: dict, verbose: bool = False) -> int:
    """Кладёт распознанный текст в индекс: фрагменты, поиск, векторы."""
    import chunk as chunker
    import embeddings
    import index as index_mod

    if not result["text"].strip():
        return 0
    row = db.q1("SELECT rel_path, file_name, brand, doc_type, section, "
                "effective_date FROM documents WHERE id=?", (doc_id,))
    index_mod._drop_document(doc_id)
    doc_meta = {"brand": row["brand"], "doc_type": row["doc_type"],
                "section": row["section"], "file_name": row["file_name"],
                "effective_date": row["effective_date"]}
    chunks = chunker.chunk_document([p for p in result["pages"] if p], doc_meta)
    if not chunks:
        return 0
    conn = db.connect()
    ids = []
    for c in chunks:
        cur = conn.execute(
            "INSERT INTO chunks(doc_id, ord, page_from, page_to, heading, context, "
            "text, n_chars) VALUES (?,?,?,?,?,?,?,?)",
            (doc_id, c.ord, c.page_from, c.page_to, c.heading, c.context,
             c.text, len(c.text)))
        ids.append(int(cur.lastrowid))
    conn.commit()
    index_mod._fts_insert(ids)
    try:
        db.vectors().add(ids, embeddings.embed_texts([c.indexed_text for c in chunks]))
        db.vectors().save()
        conn.executemany("UPDATE chunks SET embedded=1 WHERE id=?", [(i,) for i in ids])
        conn.commit()
    except Exception as exc:  # noqa: BLE001 — текстовый поиск уже работает
        if verbose:
            print(f"   векторы не посчитаны: {exc}")
    return len(ids)


def run(limit: int | None = None, provider: str | None = None,
        retry_failed: bool = False, verbose: bool = False, progress=None) -> dict:
    say = progress or (lambda t: print(t))
    db.init()
    rows = queue(limit)
    if not retry_failed:
        rows = [r for r in rows if not r["ocr_error"]]
    if not rows:
        say("Очередь пуста — все сканы распознаны.")
        return {}

    state = available()
    say("Провайдеры распознавания на этой машине:")
    for name, note in state.items():
        say(f"  {name:10} {note if note == 'готов' else 'недоступен: ' + note[:110]}")
    say()

    total = {"documents": 0, "chars": 0, "pages": 0, "failed": 0, "rejected": 0}
    started = time.time()
    for n, r in enumerate(rows, 1):
        path = Path(r["abs_path"])
        say(f"[{n}/{len(rows)}] {r['rel_path']}")
        if not path.exists():
            db.run("UPDATE documents SET ocr_error=? WHERE id=?",
                   ("файл не найден", r["id"]))
            total["failed"] += 1
            continue
        try:
            result = read_document(path, provider=provider, verbose=verbose)
        except Exception as exc:  # noqa: BLE001
            say(f"   не удалось: {exc}")
            db.run("UPDATE documents SET ocr_error=? WHERE id=?",
                   (str(exc)[:400], r["id"]))
            total["failed"] += 1
            continue

        n_chunks = apply_to_index(r["id"], result, verbose=verbose)
        # Слабый результат текст в индекс всё-таки отдаёт — что-то лучше,
        # чем ничего, — но документ остаётся в очереди: когда появится
        # более точный распознаватель, он будет обработан заново.
        weak = result["chars"] and result["quality"] < config.OCR_MIN_QUALITY
        note = (f"качество {result['quality']:.2f} ниже порога "
                f"{config.OCR_MIN_QUALITY}: возможна подмена букв") if weak else None
        db.run("""UPDATE documents SET needs_ocr=?, text_chars=?, ocr_provider=?,
                  ocr_at=?, ocr_pages=?, ocr_quality=?, ocr_error=? WHERE id=?""",
               (0 if (result["chars"] and not weak) else 1, result["chars"],
                result["provider"],
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                result["recognized"], result["quality"], note, r["id"]))
        total["documents"] += 1
        total["chars"] += result["chars"]
        total["pages"] += result["recognized"]
        total["rejected"] += result["rejected"]
        total["weak"] = total.get("weak", 0) + bool(weak)
        say(f"   {result['chars']} симв., страниц {result['recognized']}"
              f"/{result['n_pages']}, качество {result['quality']:.2f}"
              f"{' — слабо, оставлен в очереди' if weak else ''}, "
              f"фрагментов {n_chunks}, {result['seconds']} с "
              f"({', '.join(result['providers']) or 'нет'})")

    elapsed = time.time() - started
    say(f"\nГотово за {elapsed / 60:.1f} мин. Документов: {total['documents']}, "
          f"страниц: {total['pages']}, символов: {total['chars']:,}".replace(",", " "))
    if total["rejected"]:
        say(f"Страниц отбраковано проверкой на подмену букв: {total['rejected']}")
    if total.get("weak"):
        say(f"Документов с низким качеством распознавания: {total['weak']} — "
              f"текст в индексе есть, но они остались в очереди на повтор "
              f"более точным распознавателем.")
    if total["failed"]:
        say(f"Не удалось распознать: {total['failed']} — "
              f"список: python ocr.py queue")
    return total


def main() -> int:
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)   # тихий выход при | head
    p = argparse.ArgumentParser(description="Распознавание сканов базы знаний")
    p.add_argument("command", choices=["queue", "run", "check", "providers"])
    p.add_argument("path", nargs="?", help="файл для команды check")
    p.add_argument("--limit", type=int)
    p.add_argument("--provider")
    p.add_argument("--retry-failed", action="store_true",
                   help="повторить документы, на которых была ошибка")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    if args.command == "providers":
        for name, note in available().items():
            print(f"{name:10} {note}")
    elif args.command == "queue":
        show_queue()
    elif args.command == "run":
        run(limit=args.limit, provider=args.provider,
            retry_failed=args.retry_failed, verbose=args.verbose)
    elif args.command == "check":
        if not args.path:
            print("укажите файл: python ocr.py check путь/к/скану.pdf")
            return 2
        result = read_document(Path(args.path), provider=args.provider, verbose=True)
        print(f"\nСтраниц: {result['recognized']}/{result['n_pages']}, "
              f"символов: {result['chars']}, качество: {result['quality']}, "
              f"провайдеры: {', '.join(result['providers']) or 'нет'}")
        print("-" * 70)
        print(result["text"][:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
