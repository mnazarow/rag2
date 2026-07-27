"""
Обогащение нетекстовых материалов: речь из видео, описания изображений,
данные из чертежей.

  python media.py transcribe            — расшифровать видео и аудио
  python media.py describe              — описать изображения
  python media.py cad                   — вытащить надписи из чертежей
  python media.py clip <doc_id> <sec>   — вырезать фрагмент видео вокруг секунды

Смысл: каждый нетекстовый файл уже попал в индекс карточкой (путь, бренд,
соседние файлы). Здесь карточка дополняется содержимым — расшифровкой речи
с таймкодами, описанием кадра, надписями с чертежа. Фрагменты расшифровки
хранятся отдельно в media_segments, чтобы ответ вёл не на «видео целиком»,
а на конкретную минуту.

Провайдеры подключаются переменными окружения, по умолчанию всё выключено:
обогащение — тяжёлая операция, её запускают отдельно от обычной индексации.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import chunk as chunker
import config
import db
import embeddings
import extract


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def hhmmss(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}" if s >= 3600 \
        else f"{s // 60:02d}:{s % 60:02d}"


# ------------------------------------------------------- распознавание речи --
class ASRError(RuntimeError):
    pass


def extract_audio(video: Path, out_wav: Path) -> None:
    """16 кГц моно — то, что ждут все модели распознавания."""
    subprocess.run(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1",
                    "-ar", "16000", "-c:a", "pcm_s16le", str(out_wav)],
                   capture_output=True, timeout=3600, check=True)


def transcribe_file(path: Path) -> list[dict]:
    """
    Возвращает список отрезков: [{start, end, text}].

    faster-whisper заметно быстрее и экономнее оригинального whisper;
    внешний VAD (Silero) даёт лучший результат, чем встроенный в whisper.
    """
    provider = config.ASR_PROVIDER
    if provider == "none":
        raise ASRError("ASR_PROVIDER=none — распознавание выключено")

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        if path.suffix.lower() in config.VIDEO_EXTENSIONS:
            extract_audio(path, wav)
        else:
            wav = path

        if provider == "faster-whisper":
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise ASRError("нужен faster-whisper: pip install faster-whisper") from exc
            device = config.ASR_DEVICE if config.ASR_DEVICE != "auto" else "auto"
            model = WhisperModel(config.ASR_MODEL, device=device,
                                 compute_type="int8" if device == "cpu" else "float16")
            segments, _info = model.transcribe(
                str(wav), language=config.ASR_LANGUAGE, vad_filter=True,
                word_timestamps=False)
            return [{"start": s.start, "end": s.end, "text": s.text.strip()}
                    for s in segments if s.text.strip()]

        if provider == "whisper":
            try:
                import whisper
            except ImportError as exc:
                raise ASRError("нужен openai-whisper: pip install openai-whisper") from exc
            model = whisper.load_model(config.ASR_MODEL)
            res = model.transcribe(str(wav), language=config.ASR_LANGUAGE)
            return [{"start": s["start"], "end": s["end"], "text": s["text"].strip()}
                    for s in res.get("segments", []) if s["text"].strip()]

        if provider in ("yandex", "sber"):
            raise ASRError(
                f"провайдер {provider} требует ключей и загрузки файла в облако; "
                "реализуется по образцу llm.py — см. документ, раздел «видео»")

        raise ASRError(f"неизвестный ASR_PROVIDER: {provider}")


def group_segments(segments: list[dict], window: int | None = None) -> list[dict]:
    """
    Склеивает короткие реплики в смысловые куски нужной длины.
    Слишком мелкие отрезки плохо ищутся: во фразе «а теперь нажимаем сюда»
    нет ни одного искомого слова.
    """
    window = window or config.ASR_SEGMENT_SECONDS
    out: list[dict] = []
    cur = {"start": None, "end": None, "text": []}
    for seg in segments:
        if cur["start"] is None:
            cur["start"] = seg["start"]
        cur["end"] = seg["end"]
        cur["text"].append(seg["text"])
        if cur["end"] - cur["start"] >= window:
            out.append({"start": cur["start"], "end": cur["end"],
                        "text": " ".join(cur["text"])})
            cur = {"start": None, "end": None, "text": []}
    if cur["text"]:
        out.append({"start": cur["start"], "end": cur["end"], "text": " ".join(cur["text"])})
    return out


def index_transcript(doc_id: int, path: Path, groups: list[dict]) -> int:
    """Кладёт расшифровку в индекс: каждый кусок — отдельный фрагмент с таймкодом."""
    conn = db.connect()
    meta = extract.path_meta(path)
    doc_meta = {"brand": meta.brand, "doc_type": meta.doc_type, "section": meta.section,
                "file_name": path.name, "effective_date": meta.effective_date}
    conn.execute("DELETE FROM media_segments WHERE doc_id=?", (doc_id,))
    chunk_ids, texts = [], []
    for i, g in enumerate(groups):
        stamp = f"[{hhmmss(g['start'])}–{hhmmss(g['end'])}]"
        heading = f"Видео, фрагмент {stamp}"
        context = chunker.build_context(doc_meta, heading)
        text = f"{stamp} {g['text']}"
        cur = conn.execute(
            "INSERT INTO chunks(doc_id, ord, page_from, page_to, heading, context, text, n_chars) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (doc_id, 1000 + i, None, None, heading, context, text, len(text)))
        cid = int(cur.lastrowid)
        chunk_ids.append(cid)
        texts.append(f"{context}\n{heading}\n{text}")
        conn.execute("INSERT INTO media_segments(doc_id, chunk_id, start_sec, end_sec, text) "
                     "VALUES (?,?,?,?,?)", (doc_id, cid, g["start"], g["end"], g["text"]))
    conn.commit()
    if chunk_ids:
        import index as index_mod
        index_mod._fts_insert(chunk_ids)
        db.vectors().add(chunk_ids, embeddings.embed_texts(texts))
        db.vectors().save()
    return len(chunk_ids)


def transcribe_queue(limit: int | None = None, force: bool = False,
                     progress=None) -> dict:
    """
    Расшифровывает очередь видео и аудио.

    Расшифровка режется на куски с таймкодами, поэтому бот отвечает
    ссылкой на конкретную минуту записи, а не на файл целиком.
    Прогон однократный; прерывать безопасно — уже расшифрованное
    помечено и повторно не обрабатывается.
    """
    say = progress or (lambda t: print(t, flush=True))
    db.init()
    where = "" if force else "AND (enriched IS NULL OR enriched NOT LIKE '%asr%')"
    rows = db.q(f"""SELECT id, abs_path, file_name FROM documents
                    WHERE asset_kind IN ('video','audio') AND status='ok' {where}
                    ORDER BY size_bytes""")
    if limit:
        rows = rows[:limit]
    say(f"К расшифровке: {len(rows)} файлов")
    done = failed = chunks = 0
    for i, r in enumerate(rows, 1):
        path = Path(r["abs_path"])
        if not path.exists():
            failed += 1
            continue
        say(f"[{i}/{len(rows)}] {r['file_name']}")
        try:
            segments = transcribe_file(path)
        except Exception as exc:  # noqa: BLE001 — одна запись не должна ронять прогон
            say(f"    не вышло: {exc}")
            failed += 1
            continue
        groups = group_segments(segments)
        n = index_transcript(r["id"], path, groups)
        db.run("UPDATE documents SET enriched=COALESCE(enriched,'')||'asr,' WHERE id=?",
               (r["id"],))
        done += 1
        chunks += n
        say(f"    фрагментов: {n}, реплик: {len(segments)}")
    return {"files": len(rows), "transcribed": done, "failed": failed,
            "chunks": chunks}


def cmd_transcribe(limit: int | None, force: bool) -> None:
    db.init()
    where = "" if force else "AND (enriched IS NULL OR enriched NOT LIKE '%asr%')"
    rows = db.q(f"""SELECT id, abs_path, file_name FROM documents
                    WHERE asset_kind IN ('video','audio') AND status='ok' {where}
                    ORDER BY size_bytes""")
    if limit:
        rows = rows[:limit]
    print(f"К расшифровке: {len(rows)} файлов")
    for i, r in enumerate(rows, 1):
        path = Path(r["abs_path"])
        if not path.exists():
            continue
        print(f"  [{i}/{len(rows)}] {r['file_name']}", flush=True)
        try:
            segments = transcribe_file(path)
        except (ASRError, subprocess.SubprocessError) as exc:
            print(f"      не вышло: {exc}")
            continue
        groups = group_segments(segments)
        n = index_transcript(r["id"], path, groups)
        db.run("UPDATE documents SET enriched=COALESCE(enriched,'')||'asr,' WHERE id=?", (r["id"],))
        print(f"      фрагментов: {n}, реплик: {len(segments)}")


def cmd_clip(doc_id: int, at_sec: float, before: float = 5, after: float = 40) -> str:
    """Вырезает фрагмент видео вокруг момента — то, что бот пришлёт в чат."""
    row = db.q1("SELECT abs_path, file_name FROM documents WHERE id=?", (doc_id,))
    if not row:
        raise SystemExit("документ не найден")
    src = Path(row["abs_path"])
    out = config.DATA_DIR / "clips"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{doc_id}_{int(at_sec)}.mp4"
    start = max(at_sec - before, 0)
    subprocess.run(["ffmpeg", "-y", "-ss", str(start), "-i", str(src),
                    "-t", str(before + after), "-c:v", "libx264", "-preset", "veryfast",
                    "-c:a", "aac", "-movflags", "+faststart", str(dst)],
                   capture_output=True, timeout=900, check=True)
    return str(dst)


# ------------------------------------------------- описание изображений -----
VISION_PROMPT = (
    "Опиши это изображение для поиска по корпоративной базе инженерного "
    "оборудования (насосы, отопление, водоснабжение). Укажи: что изображено "
    "(тип изделия), ракурс, цвет и материал, контекст (студийная съёмка, "
    "монтаж на объекте, производство, 3D-визуализация). Отдельной строкой "
    "«Надписи:» приведи дословно весь читаемый текст с шильдиков, табличек "
    "и маркировки. Только факты, без домыслов. По-русски, до 80 слов."
)


def describe_image(path: Path) -> str:
    """Описание изображения через VLM. Возвращает пустую строку, если выключено."""
    if config.VISION_PROVIDER == "none":
        return ""
    import base64
    import httpx

    data = path.read_bytes()
    if len(data) > 8 * 1024 * 1024:
        return ""
    b64 = base64.b64encode(data).decode()
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"

    if config.VISION_PROVIDER == "openai":
        client = httpx.Client(timeout=180, proxy=config.LLM_PROXY or None)
        r = client.post(f"{config.OPENAI_BASE_URL}/chat/completions",
                        headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"},
                        json={"model": config.VISION_MODEL, "messages": [{
                            "role": "user", "content": [
                                {"type": "text", "text": VISION_PROMPT},
                                {"type": "image_url",
                                 "image_url": {"url": f"data:{mime};base64,{b64}"}}]}],
                            "max_tokens": 400})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    raise RuntimeError(f"провайдер зрения {config.VISION_PROVIDER} не реализован в прототипе")


def cmd_describe(limit: int | None, force: bool) -> None:
    db.init()
    where = "" if force else "AND (enriched IS NULL OR enriched NOT LIKE '%vision%')"
    rows = db.q(f"""SELECT id, abs_path, file_name FROM documents
                    WHERE asset_kind='image' AND status='ok' {where}""")
    if limit:
        rows = rows[:limit]
    print(f"К описанию: {len(rows)} изображений")
    done = 0
    for i, r in enumerate(rows, 1):
        path = Path(r["abs_path"])
        if not path.exists():
            continue
        try:
            text = describe_image(path)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}] {r['file_name']}: {exc}")
            continue
        if not text:
            continue
        _append_description(r["id"], path, text, tag="vision")
        done += 1
        if i % 20 == 0:
            print(f"  обработано {i}/{len(rows)}", flush=True)
    print(f"Описано изображений: {done}")


def _append_description(doc_id: int, path: Path, description: str, tag: str) -> None:
    """Пересобирает карточку объекта, дописав в неё полученное описание."""
    meta = extract.path_meta(path)
    card = extract.asset_card(path, meta, description=description)
    conn = db.connect()
    import index as index_mod
    index_mod._drop_document(doc_id)
    doc_meta = {"brand": meta.brand, "doc_type": meta.doc_type, "section": meta.section,
                "file_name": path.name, "effective_date": meta.effective_date}
    chunks = chunker.chunk_document(card.pages, doc_meta)
    ids, texts = [], []
    for c in chunks:
        cur = conn.execute(
            "INSERT INTO chunks(doc_id, ord, heading, context, text, n_chars) "
            "VALUES (?,?,?,?,?,?)",
            (doc_id, c.ord, c.heading, c.context, c.text, len(c.text)))
        ids.append(int(cur.lastrowid))
        texts.append(c.indexed_text)
    conn.commit()
    if ids:
        index_mod._fts_insert(ids)
        db.vectors().add(ids, embeddings.embed_texts(texts))
        db.vectors().save()
    db.run(f"UPDATE documents SET enriched=COALESCE(enriched,'')||'{tag},', "
           f"text_chars=? WHERE id=?", (card.n_chars, doc_id))


# ------------------------------------------------------------- чертежи ------
def dwg_to_dxf(path: Path, out_dir: Path) -> Path | None:
    """
    Конвертация DWG в DXF. Нужен ODA File Converter (бесплатный) —
    путь задаётся переменной ODA_CONVERTER. Без него DWG остаётся
    только карточкой по имени файла и папке.
    """
    if not config.ODA_CONVERTER or not Path(config.ODA_CONVERTER).exists():
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([config.ODA_CONVERTER, str(path.parent), str(out_dir),
                    "ACAD2018", "DXF", "0", "1", path.name],
                   capture_output=True, timeout=300)
    candidate = out_dir / (path.stem + ".dxf")
    return candidate if candidate.exists() else None


def cmd_cad(limit: int | None, force: bool) -> None:
    """
    Достаёт надписи из чертежей. Порядок действий:
      1. если рядом лежит PDF с тем же именем — берём текст оттуда (бесплатно
         и надёжно: в основной надписи есть модель, тип изделия и масса);
      2. иначе конвертируем DWG в DXF через ODA и вытаскиваем тексты и
         атрибуты блоков.
    """
    db.init()
    where = "" if force else "AND (enriched IS NULL OR enriched NOT LIKE '%cad%')"
    rows = db.q(f"""SELECT id, abs_path, file_name FROM documents
                    WHERE asset_kind='drawing' AND status='ok' {where}""")
    if limit:
        rows = rows[:limit]
    print(f"Чертежей к обработке: {len(rows)}")
    from_pdf = from_dxf = 0
    work = config.DATA_DIR / "dxf"
    for i, r in enumerate(rows, 1):
        path = Path(r["abs_path"])
        if not path.exists():
            continue
        text = ""
        twin = path.with_suffix(".pdf")
        if not twin.exists():
            for cand in path.parent.glob(f"{path.stem}*.pdf"):
                twin = cand
                break
        if twin.exists():
            res = extract.extract(twin)
            if not res.error and res.n_chars > 10:
                text = res.text.strip()[:800]
                from_pdf += 1
        if not text and path.suffix.lower() == ".dwg":
            dxf = dwg_to_dxf(path, work)
            if dxf:
                res = extract.extract_dxf(dxf)
                if not res.error and res.n_chars > 10:
                    text = res.text.strip()[:800]
                    from_dxf += 1
        if text:
            _append_description(r["id"], path, "надписи на чертеже — " + text, tag="cad")
        if i % 50 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
    print(f"Из PDF-двойников: {from_pdf}, из DXF: {from_dxf}")


def main() -> None:
    p = argparse.ArgumentParser(description="Обогащение нетекстовых материалов")
    p.add_argument("command", choices=["transcribe", "describe", "cad", "clip"])
    p.add_argument("args", nargs="*")
    p.add_argument("--limit", type=int)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    if a.command == "transcribe":
        cmd_transcribe(a.limit, a.force)
    elif a.command == "describe":
        cmd_describe(a.limit, a.force)
    elif a.command == "cad":
        cmd_cad(a.limit, a.force)
    elif a.command == "clip":
        db.init()
        print(cmd_clip(int(a.args[0]), float(a.args[1])))


if __name__ == "__main__":
    sys.exit(main())
