"""
Разбор прайс-листов в структурированную таблицу products.

Зачем отдельно от RAG: векторный поиск возвращает *похожие* строки, а не
гарантированно нужную. Для цены и артикула это недопустимо — ошибка в цене
стоит денег. Поэтому прайсы живут в обычной таблице с точным поиском по
артикулу и полнотекстовым по наименованию, а бот-роутер отправляет
«ценовые» вопросы именно туда.

Формат прайсов в базе (проверено на реальных файлах): шапка не в первой
строке, подзаголовки групп товаров повторяются по ходу листа, колонки
характеристик у каждой группы свои. Парсер это переживает.
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import config
import db
import extract

ARTICLE_HINTS = ("артикул", "код", "арт.", "арт", "sku", "код товара", "номенклатура")
NAME_HINTS = ("наименование", "название", "товар", "описание", "продукция", "модель")
PRICE_HINTS = ("цена", "прайс", "стоимость", "руб", "ррц", "розничная", "price")
UNIT_HINTS = ("ед.", "ед", "единица", "упак", "шт")

_NUM_RX = re.compile(r"^\s*-?\d[\d\s ]*([.,]\d+)?\s*$")


def _norm(v) -> str:
    return re.sub(r"\s+", " ", str(v)).strip() if v is not None else ""


def _to_price(v) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v) if float(v) > 0 else None
    s = _norm(v).replace(" ", "").replace(" ", "").replace(",", ".")
    s = re.sub(r"[^\d.]", "", s)
    if not s or s.count(".") > 1:
        return None
    try:
        val = float(s)
    except ValueError:
        return None
    return val if val > 0 else None


def _find_header(rows: list[list], scan: int = 30) -> tuple[int, dict[str, int]] | None:
    """Ищет строку-шапку и сопоставляет колонки."""
    best: tuple[int, dict[str, int], int] | None = None
    for i, row in enumerate(rows[:scan]):
        cells = [_norm(c).lower() for c in row]
        mapping: dict[str, int] = {}
        for j, cell in enumerate(cells):
            if not cell:
                continue
            if "article" not in mapping and any(h == cell or cell.startswith(h) for h in ARTICLE_HINTS):
                mapping["article"] = j
            elif "name" not in mapping and any(h in cell for h in NAME_HINTS):
                mapping["name"] = j
            elif "price" not in mapping and any(h in cell for h in PRICE_HINTS):
                mapping["price"] = j
            elif "unit" not in mapping and any(cell == h for h in UNIT_HINTS):
                mapping["unit"] = j
        score = len(mapping) + (1 if "price" in mapping else 0)
        if "name" in mapping and score >= 2 and (best is None or score > best[2]):
            best = (i, mapping, score)
    return (best[0], best[1]) if best else None


def parse_workbook(path: Path) -> list[dict]:
    """Возвращает список товаров из всех листов книги."""
    try:
        import openpyxl
    except ImportError:
        return []
    items: list[dict] = []
    try:
        if path.suffix.lower() in (".xlsx", ".xlsm"):
            wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
            sheets = [(ws.title, [list(r) for r in ws.iter_rows(values_only=True)])
                      for ws in wb.worksheets]
            wb.close()
        else:
            import xlrd
            book = xlrd.open_workbook(str(path))
            sheets = [(sh.name, [sh.row_values(r) for r in range(sh.nrows)])
                      for sh in book.sheets()]
    except Exception:  # noqa: BLE001
        return []

    for sheet_name, rows in sheets:
        found = _find_header(rows)
        if not found:
            continue
        header_row, cols = found
        extra_names = {j: _norm(rows[header_row][j])
                       for j in range(len(rows[header_row]))
                       if j not in cols.values() and _norm(rows[header_row][j])}
        for row in rows[header_row + 1:]:
            if not row:
                continue
            name = _norm(row[cols["name"]]) if cols.get("name", -1) < len(row) else ""
            article = _norm(row[cols["article"]]) if "article" in cols and cols["article"] < len(row) else ""
            price = _to_price(row[cols["price"]]) if "price" in cols and cols["price"] < len(row) else None
            # Артикул — короткий токен без пробелов. Иначе это название группы,
            # случайно попавшее в колонку «Артикул» (так устроены реальные прайсы).
            if article and not re.fullmatch(r"[\w\-./№]{1,24}", article):
                article = ""
            if not name and not article:
                continue
            # Строка-подзаголовок группы или длинное описание: нет ни цены, ни артикула.
            if price is None and not article:
                continue
            if len(name) > 250:                       # абзац описания, а не позиция
                continue
            attrs = {}
            for j, title in extra_names.items():
                if j < len(row):
                    val = _norm(row[j])
                    if val:
                        attrs[title] = val
            items.append({
                "sheet": sheet_name,
                "article": article,
                "name": name,
                "price": price,
                "unit": _norm(row[cols["unit"]]) if "unit" in cols and cols["unit"] < len(row) else "",
                "attrs": attrs,
            })
    return items


def index_price_file(doc_id: int, abs_path: Path, brand: str | None,
                     price_date: str | None) -> int:
    """Загружает прайс в таблицу products, вытесняя прошлую версию этого файла."""
    items = parse_workbook(abs_path)
    if not items:
        return 0
    db.run("DELETE FROM products WHERE doc_id=?", (doc_id,))
    rows = [(doc_id, brand, it["article"], it["name"], it["price"], "RUB",
             it["unit"], json.dumps(it["attrs"], ensure_ascii=False), price_date, 1)
            for it in items]
    db.runmany(
        "INSERT INTO products(doc_id, brand, article, name, price, currency, unit, "
        "attrs_json, price_date, is_current) VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    # Обновляем FTS
    conn = db.connect()
    ids = conn.execute("SELECT id, article, name, brand FROM products WHERE doc_id=?",
                       (doc_id,)).fetchall()
    conn.executemany(
        "INSERT INTO products_fts(rowid, article, name, brand) VALUES (?,?,?,?)",
        [(r["id"], r["article"] or "", r["name"] or "", r["brand"] or "") for r in ids])
    conn.commit()
    return len(items)


def deprecate_older_prices() -> int:
    """
    Помечает устаревшие прайсы: если по одному version_key есть несколько
    документов — актуальным остаётся самый свежий по effective_date.
    Решает проблему «в базе лежит и старый, и новый прайс».
    """
    rows = db.q("""
        SELECT id, version_key, effective_date, mtime FROM documents
        WHERE status='ok' AND version_key<>'' AND effective_date IS NOT NULL
          AND doc_type IN ('ПРАЙС-ЛИСТ', 'КАТАЛОГ', 'ПАСПОРТ', 'СЕРТИФИКАТ')
    """)
    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(r["version_key"], []).append(r)
    changed = 0
    for _, group in groups.items():
        if len(group) < 2:
            continue
        group.sort(key=lambda r: (r["effective_date"] or "0000", r["mtime"] or 0), reverse=True)
        newest = group[0]
        for old in group[1:]:
            db.run("UPDATE documents SET is_current=0, superseded_by=? WHERE id=?",
                   (newest["id"], old["id"]))
            db.run("UPDATE products SET is_current=0 WHERE doc_id=?", (old["id"],))
            changed += 1
    return changed


# ------------------------------------------------------------------ поиск ---
PRICE_QUESTION_RX = re.compile(
    r"\b(цен[аыуе]|стоимост|прайс|почём|почем|сколько стоит|ррц|скидк|прайс-лист)\b", re.I)
ARTICLE_RX = re.compile(r"\b(?:арт\.?|артикул|код)\s*[:№]?\s*([A-Za-zА-Яа-я0-9][\w\-./]{2,20})\b", re.I)
BARE_ARTICLE_RX = re.compile(r"\b(\d{4,8}(?:\.[A-ZА-Я0-9]{1,3})?)\b")


def looks_like_price_question(text: str) -> bool:
    return bool(PRICE_QUESTION_RX.search(text))


def search_products(query: str, limit: int = 10, only_current: bool = True) -> list[dict]:
    """Точный поиск по артикулу + полнотекстовый по наименованию."""
    results: list[dict] = []
    seen: set[int] = set()
    where_current = "AND p.is_current=1" if only_current else ""

    articles = ARTICLE_RX.findall(query) + BARE_ARTICLE_RX.findall(query)
    for art in articles:
        for row in db.q(
            f"""SELECT p.*, d.rel_path, d.file_name FROM products p
                JOIN documents d ON d.id=p.doc_id
                WHERE p.article = ? COLLATE NOCASE {where_current} LIMIT ?""",
                (art, limit)):
            if row["id"] not in seen:
                seen.add(row["id"])
                results.append(dict(row))
    if len(results) >= limit:
        return results[:limit]

    terms = [t for t in re.findall(r"[\wА-Яа-яё\-]{3,}", query) if not PRICE_QUESTION_RX.match(t)]
    if terms:
        fts_query = " OR ".join(f'"{t}"' for t in terms[:8])
        try:
            for row in db.q(
                f"""SELECT p.*, d.rel_path, d.file_name,
                           bm25(products_fts) AS rank
                    FROM products_fts
                    JOIN products p ON p.id = products_fts.rowid
                    JOIN documents d ON d.id = p.doc_id
                    WHERE products_fts MATCH ? {where_current}
                    ORDER BY rank LIMIT ?""", (fts_query, limit * 2)):
                if row["id"] not in seen:
                    seen.add(row["id"])
                    results.append(dict(row))
        except Exception:  # noqa: BLE001 — некорректный FTS-запрос
            pass
    return results[:limit]


def format_products(items: list[dict]) -> str:
    if not items:
        return ""
    lines = []
    for it in items:
        price = f"{it['price']:,.2f} ₽".replace(",", " ") if it.get("price") else "цена не указана"
        art = f"арт. {it['article']}, " if it.get("article") else ""
        date_note = f" (прайс от {it['price_date']})" if it.get("price_date") else ""
        lines.append(f"• {it['name']} — {art}{price}{date_note}")
        attrs = json.loads(it["attrs_json"]) if it.get("attrs_json") else {}
        useful = {k: v for k, v in attrs.items() if len(str(v)) < 40}
        if useful:
            lines.append("   " + "; ".join(f"{k}: {v}" for k, v in list(useful.items())[:6]))
    return "\n".join(lines)
