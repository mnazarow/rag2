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
import normtext

# Заголовки: markdown-подобные, нумерованные разделы, КАПС-строки.
_HEADING_RX = re.compile(
    r"^(?:#{1,6}\s+.+"
    r"|\d+(?:\.\d+)*\.?\s+[А-ЯЁA-Z][^\n]{2,80}"
    r"|[А-ЯЁA-Z][А-ЯЁA-Z0-9 \-«»\"'()/,.]{4,80})$"
)
# Таблица — строка с разделителем «|». Одного разделителя достаточно:
# основная форма паспорта — «параметр | значение», и при требовании двух
# разделителей она не считалась таблицей, а значит могла быть разорвана
# между параметром и значением. Ложные срабатывания отсекаются ниже:
# таблицей считаются только две и более подряд идущие такие строки.
_TABLE_ROW_RX = re.compile(r"^[^|\n]{0,80}(\|[^|\n]{0,80}){1,}$")
# «Карточка модели»: строка, начинающая описание конкретного изделия —
# «Водомёт 55/75 …», «БЦПЭ 0,5-40У …». По таким строкам ставится
# граница блока, иначе характеристики сорока соседних моделей склеиваются
# в один фрагмент, и модель в ответе выбирается наугад.
_MODEL_LINE_RX = re.compile(
    r"^[A-ZА-ЯЁ][\w\-«».]{0,30}\s?\d+(?:[.,]\d+)?(?:[/\-x×]\d+(?:[.,]\d+)?)+")
# Сколько уровней заголовков накапливать в цепочку «КАТАЛОГ › НАСОСЫ › МОДЕЛЬ».
_CHAIN_DEPTH = 3


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
    # Единое написание («ё»→«е», «м³/ч»→«м3/ч», «3,6»→«3.6») — то же
    # самое делает поиск с запросом, иначе индекс и вопрос не совпадут.
    text = normtext.canon(text)
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


def _split_blocks(text: str, chain: list[str] | None = None,
                  ) -> tuple[list[tuple[str | None, str, bool]], list[str]]:
    """
    Делит текст на блоки (заголовок, тело, это_таблица).

    Заголовки накапливаются в цепочку: «КАТАЛОГ › НАСОСЫ СКВАЖИННЫЕ ›
    ВОДОМЕТ 55/75», а не затирают друг друга — при обратном порядке
    вёрстки терялось как раз имя модели. Цепочка возвращается наружу
    и передаётся следующей странице: раздел, продолжающийся за разрыв
    страницы, сохраняет своё название.

    Страница, свёрстанная капсом целиком (каждая строка выглядит
    заголовком), раньше исчезала из индекса молча — тела блоков были
    пусты. Теперь такая страница отдаётся одним блоком с текстом как
    есть: данные в базе важнее красоты разбиения.
    """
    blocks: list[tuple[str | None, str, bool]] = []
    chain = list(chain or [])
    chain_at_start = list(chain)
    body_seen_after_heading = False
    buf: list[str] = []
    table_buf: list[str] = []

    def heading_str() -> str | None:
        return " › ".join(chain[-_CHAIN_DEPTH:]) if chain else None

    def flush_table() -> None:
        nonlocal table_buf
        if not table_buf:
            return
        if len(table_buf) >= 2:
            # Настоящая таблица — отдельным блоком, чтобы резать по
            # строкам и повторять шапку (см. _pack).
            blocks.append((heading_str(), "\n".join(table_buf), True))
        else:
            # Одинокая строка с «|» — обычный текст.
            buf.extend(table_buf)
        table_buf = []

    def flush() -> None:
        nonlocal buf
        flush_table()
        body = "\n".join(buf).strip()
        if body:
            blocks.append((heading_str(), body, False))
        buf = []

    for line in text.splitlines():
        if _TABLE_ROW_RX.match(line) and line.count("|"):
            table_buf.append(line)
            continue
        flush_table()
        if _is_heading(line):
            had_body = bool(buf)
            flush()
            title = line.strip().lstrip("#").strip()
            if had_body or body_seen_after_heading:
                chain[:] = [title]        # новый раздел после тела
                body_seen_after_heading = False
            else:
                chain.append(title)       # вложенный заголовок без тела
            continue
        if _MODEL_LINE_RX.match(line) and buf:
            # Граница между карточками моделей: прошлую — в отдельный блок.
            flush()
        if line.strip():
            body_seen_after_heading = True
        buf.append(line)
    flush()

    if not blocks and text.strip():
        # Вся страница — «заголовки»: капс-вёрстка каталога. Отдаём её
        # одним блоком под заголовком, действовавшим до этой страницы
        # (или под собственной первой строкой — чтобы у блока был
        # заголовок и его не выбросил фильтр коротких обрывков: это
        # данные, а не мусор вёрстки).
        if chain_at_start:
            head = " › ".join(chain_at_start[-_CHAIN_DEPTH:])
        else:
            head = text.strip().splitlines()[0].strip()
        blocks.append((head, text.strip(), False))
        chain = chain[-2:]                # последние строки — заголовок для следующей
    return blocks, chain


def _pack_table(body: str, target: int) -> list[str]:
    """
    Режет таблицу по строкам и повторяет шапку в каждом куске.

    Раньше длинная таблица резалась по символам: второй кусок начинался
    с середины строки «| 96 | 3,0 | 1160» — без имени модели и без
    названий колонок. Модель, читая такой фрагмент, не может знать,
    какое из чисел напор, а какое подача, — и уверенно выбирает не то.
    """
    rows = [r for r in body.splitlines() if r.strip()]
    if not rows:
        return []
    header, data = rows[0], rows[1:]
    if not data:
        return [header]
    out: list[str] = []
    cur: list[str] = [header]
    size = len(header)
    for row in data:
        if size + len(row) + 1 > target and len(cur) > 1:
            out.append("\n".join(cur))
            cur, size = [header], len(header)
        cur.append(row)
        size += len(row) + 1
    if len(cur) > 1:
        out.append("\n".join(cur))
    return out


def _pack(body: str, target: int, overlap: int, is_table: bool = False) -> list[str]:
    """Собирает абзацы в куски ~target символов с перекрытием."""
    if is_table:
        return _pack_table(body, target)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    out: list[str] = []
    cur = ""
    for para in paragraphs:
        if len(para) > target * 2:                    # очень длинный абзац — режем
            # По возможности — по границам строк, а не по символам.
            if para.count("\n") >= 3:
                out.extend(_pack("\n\n".join(para.split("\n")), target, overlap))
                continue
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

    # Цепочка заголовков живёт сквозь страницы: таблица характеристик,
    # продолжающаяся на следующей странице, сохраняет своё название.
    chain: list[str] = []
    for page_no, page in enumerate(pages, start=1):
        page = _normalize(page)
        if len(page) < 20:
            # Пустая или служебная страница. Порог сознательно ниже
            # CHUNK_MIN_CHARS: страница сертификата с 90 символами —
            # это данные, а не мусор.
            continue
        blocks, chain = _split_blocks(page, chain)
        for heading, body, is_table in blocks:
            for piece in _pack(body, target, overlap, is_table=is_table):
                # Короткий кусок с заголовком — осмысленная строка
                # («Максимальное содержание песка — 180 г/м³» под своим
                # разделом); без заголовка — скорее обрывок вёрстки.
                floor = 30 if heading else config.CHUNK_MIN_CHARS
                if len(piece) < floor:
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
