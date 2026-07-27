"""
Разбиение документов на чанки.

Стратегия (по итогам обзора практик 2025-2026):
  * structure-aware: режем по заголовкам разделов, затем по абзацам,
    и только в последнюю очередь — «жёстко» по символам;
  * таблицы не рвём построчно — блок строк с одинаковой структурой
    держим вместе, иначе теряется связь «параметр → значение»;
  * к каждому чанку добавляется контекстная приставка
    (Contextual Retrieval): бренд, тип документа, название файла, раздел.
    Дешёвый детерминированный вариант работает всегда; при
    CONTEXTUAL_CHUNKS=1 приставка дополняется LLM-описанием.

Исследование arXiv 2606.00881 показывает, что тяжёлые методы чанкинга
не окупают вычислений: простые структурные правила дают сопоставимое
качество на порядки дешевле. Поэтому здесь именно они.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import config

# Заголовки: markdown-подобные, нумерованные разделы, КАПС-строки.
_HEADING_RX = re.compile(
    r"^(?:#{1,6}\s+.+"
    r"|\d+(?:\.\d+)*\.?\s+[А-ЯЁA-Z][^\n]{2,80}"
    r"|[А-ЯЁA-Z][А-ЯЁA-Z0-9 \-«»\"'()/,.]{4,80})$"
)
_TABLE_ROW_RX = re.compile(r"^[^|\n]{0,80}(\|[^|\n]{0,80}){2,}$")


@dataclass
class Chunk:
    ord: int
    text: str
    heading: str | None = None
    context: str = ""
    page_from: int | None = None
    page_to: int | None = None

    @property
    def indexed_text(self) -> str:
        """То, что реально уходит в эмбеддер и в BM25."""
        parts = [p for p in (self.context, self.heading, self.text) if p]
        return "\n".join(parts)


def _normalize(text: str) -> str:
    text = text.replace("­", "").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Убираем «растянутые» точки оглавления: «Раздел ......... 15»
    text = re.sub(r"[.·]{6,}", " ... ", text)
    return text.strip()


def _is_heading(line: str) -> bool:
    line = line.strip()
    if not (3 < len(line) < 100):
        return False
    if line.endswith((".", ",", ";")) and not line.startswith("#"):
        return False
    return bool(_HEADING_RX.match(line))


def _split_blocks(text: str) -> list[tuple[str | None, str]]:
    """Делит текст на (заголовок, тело). Таблицы склеиваются в один блок."""
    blocks: list[tuple[str | None, str]] = []
    heading: str | None = None
    buf: list[str] = []
    table_buf: list[str] = []

    def flush() -> None:
        nonlocal buf, table_buf
        if table_buf:
            buf.append("\n".join(table_buf))
            table_buf = []
        body = "\n".join(buf).strip()
        if body:
            blocks.append((heading, body))
        buf = []

    for line in text.splitlines():
        if _TABLE_ROW_RX.match(line):
            table_buf.append(line)
            continue
        if table_buf:
            buf.append("\n".join(table_buf))
            table_buf = []
        if _is_heading(line):
            flush()
            heading = line.strip().lstrip("#").strip()
            continue
        buf.append(line)
    flush()
    return blocks


def _pack(body: str, target: int, overlap: int) -> list[str]:
    """Собирает абзацы в куски ~target символов с перекрытием."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    out: list[str] = []
    cur = ""
    for para in paragraphs:
        if len(para) > target * 2:                    # очень длинный абзац — режем
            for i in range(0, len(para), target - overlap):
                piece = para[i:i + target]
                if piece.strip():
                    out.append(piece.strip())
            continue
        if len(cur) + len(para) + 2 <= target:
            cur = f"{cur}\n\n{para}" if cur else para
        else:
            if cur:
                out.append(cur)
            tail = cur[-overlap:] if cur and overlap else ""
            cur = f"{tail}\n{para}".strip() if tail else para
    if cur:
        out.append(cur)
    return out


def build_context(meta: dict, heading: str | None) -> str:
    """
    Детерминированная контекстная приставка — работает без LLM и без затрат.
    Именно она чаще всего спасает поиск: в чанке «Напор 45 м, расход 3,6 м³/ч»
    иначе нет ни бренда, ни модели, ни типа документа.
    """
    bits = []
    if meta.get("brand"):
        bits.append(str(meta["brand"]))
    if meta.get("doc_type"):
        bits.append(str(meta["doc_type"]).lower())
    title = str(meta.get("file_name", ""))
    title = re.sub(r"\.[a-zA-Z0-9]+$", "", title)
    if title:
        bits.append(f"«{title}»")
    if meta.get("effective_date"):
        bits.append(f"от {meta['effective_date']}")
    line = "Фрагмент документа: " + ", ".join(bits) if bits else ""
    if heading:
        line += f". Раздел: {heading}"
    if meta.get("section"):
        line += f". Каталог: {meta['section']}"
    return line.strip()


def chunk_document(pages: list[str], meta: dict,
                   target: int | None = None,
                   overlap: int | None = None) -> list[Chunk]:
    target = target or config.CHUNK_TARGET_CHARS
    overlap = overlap or config.CHUNK_OVERLAP_CHARS
    chunks: list[Chunk] = []
    ordinal = 0

    for page_no, page in enumerate(pages, start=1):
        page = _normalize(page)
        if len(page) < config.CHUNK_MIN_CHARS:
            continue
        for heading, body in _split_blocks(page):
            for piece in _pack(body, target, overlap):
                if len(piece) < config.CHUNK_MIN_CHARS:
                    continue
                chunks.append(Chunk(
                    ord=ordinal,
                    text=piece,
                    heading=heading,
                    context=build_context(meta, heading),
                    page_from=page_no,
                    page_to=page_no,
                ))
                ordinal += 1

    # Документ без внятной структуры (один сплошной поток) — режем целиком.
    if not chunks and pages:
        whole = _normalize("\n\n".join(pages))
        for piece in _pack(whole, target, overlap):
            if len(piece) >= config.CHUNK_MIN_CHARS:
                chunks.append(Chunk(ord=ordinal, text=piece,
                                    context=build_context(meta, None)))
                ordinal += 1
    return chunks
