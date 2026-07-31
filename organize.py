"""
Порядок в базе знаний: раздел админки для владельца папки BD.

Инструмент к алгоритму уборки из документации (раздел 25). Система
папку BD не изменяет никогда — она подключена только на чтение, и это
принцип, а не ограничение. Поэтому раздел не «наводит порядок сам», а
делает три вещи, которые честно можно сделать за владельца: находит
беспорядок и раскладывает его по разобранным категориям; расставляет
приоритеты по реальной спрашиваемости брендов; и превращает находки в
готовый план — команды переименования, которые человек проверяет и
выполняет сам.

Все проверки читают уже собранный индекс (таблицу documents), а не
ходят по диску: на базе в сотню гигабайт обход диска — минуты, запрос
к индексу — миллисекунды.
"""
from __future__ import annotations

import re
import shlex
from collections import Counter, defaultdict

import config
import db
import logging_setup

log = logging_setup.get("web")

# Типы, у которых версии сменяются, — дата в имени для них обязательна.
DATED_TYPES = ("ПРАЙС-ЛИСТ", "СЕРТИФИКАТ")

# Слова в именах, смысл которых меняется со временем на противоположный.
BAD_NAME_RX = re.compile(
    r"\b(нов(ый|ая|ое|ые)|финал\w*|последн\w*|актуальн\w*|итог\w*|"
    r"копия|copy|final|new|last)\b|\(\d+\)", re.IGNORECASE)

# Подсказки типа по имени файла — для плана уборки нетипизированных.
TYPE_HINTS = [
    (re.compile(r"прайс|price|ррц|прейскурант", re.I), "2ПРАЙС_ЛИСТ"),
    (re.compile(r"паспорт|passport", re.I), "4ПАСПОРТ"),
    (re.compile(r"сертификат|декларац|соответств|cert", re.I), "5СЕРТИФИКАТ"),
    (re.compile(r"руководство|инструкц|монтаж|эксплуатац|manual", re.I),
     "3РУКОВОДСТВО"),
    (re.compile(r"каталог|catalog|брошюр|букле", re.I), "1КАТАЛОГ"),
    (re.compile(r"опросн", re.I), "6ОПРОСНЫЙ ЛИСТ"),
    (re.compile(r"чертеж|чертёж|\bdwg\b|\brvt\b|\bdxf\b", re.I),
     "9ЧЕРТЕЖИ_DWG_REVIT_3D"),
]

# Транслитерация для поиска брендов-двойников: Jeelex ↔ Джилекс не
# поймать, а GRUNDFOS ↔ ГРУНДФОС — вполне.
_TRANSLIT = {"а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e",
             "ё": "e", "ж": "zh", "з": "z", "и": "i", "й": "i", "к": "k",
             "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r",
             "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "c",
             "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "",
             "э": "e", "ю": "yu", "я": "ya"}


def _translit(word: str) -> str:
    return "".join(_TRANSLIT.get(ch, ch) for ch in word.lower())


def metadata_progress() -> dict:
    """Доля файлов, у которых структура папок дала метаданные."""
    row = db.q1("""SELECT COUNT(*) total,
                          SUM(CASE WHEN section  IS NOT NULL THEN 1 ELSE 0 END) with_section,
                          SUM(CASE WHEN brand    IS NOT NULL THEN 1 ELSE 0 END) with_brand,
                          SUM(CASE WHEN doc_type IS NOT NULL THEN 1 ELSE 0 END) with_type,
                          SUM(CASE WHEN effective_date IS NOT NULL THEN 1 ELSE 0 END) with_date
                   FROM documents WHERE status='ok' AND is_current=1""")
    total = row["total"] or 0
    dated_row = db.q1(
        f"""SELECT COUNT(*) total,
                   SUM(CASE WHEN effective_date IS NOT NULL THEN 1 ELSE 0 END) dated
            FROM documents WHERE status='ok' AND is_current=1
              AND doc_type IN ({','.join('?' * len(DATED_TYPES))})""",
        DATED_TYPES)

    def pct(x):
        return round(100 * (x or 0) / total) if total else 0

    return {"total": total,
            "section": pct(row["with_section"]), "brand": pct(row["with_brand"]),
            "type": pct(row["with_type"]), "date": pct(row["with_date"]),
            "dated_total": dated_row["total"] or 0,
            "dated_ok": dated_row["dated"] or 0}


def untyped(limit: int = 50) -> list[dict]:
    """Файлы без типа — с подсказкой, куда их положить."""
    rows = db.q("""SELECT rel_path, file_name FROM documents
                   WHERE status='ok' AND is_current=1 AND doc_type IS NULL
                   ORDER BY rel_path LIMIT ?""", (limit,))
    out = []
    for r in rows:
        hint = next((folder for rx, folder in TYPE_HINTS
                     if rx.search(r["file_name"])), None)
        out.append({"path": r["rel_path"], "hint": hint})
    return out


def undated(limit: int = 50) -> list[dict]:
    """Сменяемые документы без даты в имени. Дата — из mtime, как подсказка."""
    rows = db.q(
        f"""SELECT rel_path, file_name, doc_type, mtime FROM documents
            WHERE status='ok' AND is_current=1 AND effective_date IS NULL
              AND doc_type IN ({','.join('?' * len(DATED_TYPES))})
            ORDER BY rel_path LIMIT ?""", (*DATED_TYPES, limit))
    out = []
    for r in rows:
        hint = None
        if r["mtime"]:
            from datetime import date
            hint = date.fromtimestamp(r["mtime"]).isoformat()
        out.append({"path": r["rel_path"], "doc_type": r["doc_type"],
                    "mtime_hint": hint})
    return out


def bad_names(limit: int = 50) -> list[dict]:
    """Имена со словами, которые врут со временем: «новый», «финал», «(1)»."""
    rows = db.q("""SELECT rel_path, file_name FROM documents
                   WHERE status='ok' AND is_current=1 ORDER BY rel_path""")
    out = []
    for r in rows:
        # Подчёркивания и дефисы в именах — обычное дело («прайс_новый»),
        # а \b между «_» и буквой границы не видит: нормализуем.
        normalized = re.sub(r"[_\-.]+", " ", r["file_name"])
        m = BAD_NAME_RX.search(normalized)
        if m:
            out.append({"path": r["rel_path"], "word": m.group(0)})
            if len(out) >= limit:
                break
    return out


def brand_twins() -> list[dict]:
    """
    Бренды-двойники: одно и то же имя в двух написаниях.

    Ловятся транслит (GRUNDFOS ↔ ГРУНДФОС), регистр и дефисы. Для поиска
    это два разных бренда: карточки фрагментов получают разные приставки,
    и вопрос находит только половину документов.
    """
    rows = db.q("""SELECT brand, COUNT(*) n FROM documents
                   WHERE status='ok' AND is_current=1 AND brand IS NOT NULL
                   GROUP BY brand""")
    groups: dict[str, list] = defaultdict(list)
    for r in rows:
        base = re.split(r"\s*/\s*", r["brand"])[0]     # без товарной линейки
        key = re.sub(r"[\s\-_.]+", "", _translit(base))
        groups[key].append({"brand": r["brand"], "files": r["n"]})
    twins = []
    for variants in groups.values():
        names = {v["brand"] for v in variants}
        if len(names) > 1:
            twins.append({"variants": sorted(variants,
                                             key=lambda v: -v["files"])})
    return twins


def top_asked_brands(limit: int = 10) -> list[dict]:
    """С каких брендов начинать уборку: о ком реально спрашивают."""
    try:
        rows = db.q("""SELECT LOWER(q.question) question FROM queries q
                       ORDER BY q.id DESC LIMIT 2000""")
    except Exception:  # noqa: BLE001
        return []
    brands = [r["brand"] for r in db.q(
        """SELECT DISTINCT brand FROM documents
           WHERE brand IS NOT NULL AND status='ok'""")]
    counts: Counter = Counter()
    for r in rows:
        q = r["question"]
        for b in brands:
            base = re.split(r"\s*/\s*", b)[0].lower()
            if base and base in q:
                counts[b] += 1
    return [{"brand": b, "asked": n} for b, n in counts.most_common(limit)]


def exact_duplicates(limit: int = 15) -> list[dict]:
    """Один файл в нескольких местах — по хэшу содержимого."""
    import audit
    return audit.duplicates()["top"][:limit]


def coverage_gaps() -> list[dict]:
    import audit
    return audit.coverage()["gaps"][:30]


def cleanup_plan() -> str:
    """
    План уборки: команды mv, которые владелец проверяет и выполняет сам.

    Система папку BD не трогает — принципиально. План собирает четыре
    категории: нетипизированные файлы с уверенной подсказкой типа,
    недатированные прайсы и сертификаты (дата — из времени изменения
    файла, ПРОВЕРЬТЕ её перед запуском), дубли (перенос лишних копий в
    _dubli/ для ручного разбора) и переименования «новый/финал».
    Каждая команда закомментирована: план — это черновик решений
    человека, а не скрипт для слепого запуска.
    """
    lines = [
        "#!/bin/bash",
        "# План уборки базы знаний — сформирован ассистентом.",
        "# ЭТО ЧЕРНОВИК: каждая команда закомментирована намеренно.",
        "# Раскомментируйте те, с которыми согласны, и запустите:",
        "#   bash план_уборки.sh",
        "# Пути отсчитываются от корня базы знаний:",
        f"cd {shlex.quote(str(config.KB_ROOT))} || exit 1",
        "set -e", "",
    ]

    items = untyped(limit=200)
    if items:
        lines += ["# ---- 1. Файлы без типа: разложить по папкам типов ----",
                  "# Подсказка выведена из имени файла — проверьте каждую."]
        for it in items:
            if not it["hint"]:
                continue
            src = it["path"]
            parts = src.split("/")
            dest = "/".join(parts[:-1] + [it["hint"], parts[-1]])
            lines.append(f"# mkdir -p {shlex.quote('/'.join(parts[:-1] + [it['hint']]))}")
            lines.append(f"# mv {shlex.quote(src)} {shlex.quote(dest)}")
        lines.append("")

    items = undated(limit=200)
    if items:
        lines += ["# ---- 2. Прайсы и сертификаты без даты в имени ----",
                  "# Дата взята из времени изменения файла — она бывает",
                  "# датой копирования, а не документа. СВЕРЬТЕ с содержимым."]
        for it in items:
            if not it["mtime_hint"]:
                continue
            src = it["path"]
            stem, dot, ext = src.rpartition(".")
            dest = f"{stem}_{it['mtime_hint']}.{ext}" if dot else f"{src}_{it['mtime_hint']}"
            lines.append(f"# mv {shlex.quote(src)} {shlex.quote(dest)}")
        lines.append("")

    dups = exact_duplicates(limit=100)
    if dups:
        lines += ["# ---- 3. Точные дубли: лишние копии в _dubli/ ----",
                  "# Первый путь в группе остаётся, остальные — на разбор.",
                  "# mkdir -p _dubli"]
        for g in dups:
            for extra in g["paths"][1:]:
                lines.append(f"# mv {shlex.quote(extra)} _dubli/")
        lines.append("")

    bads = bad_names(limit=200)
    if bads:
        lines += ["# ---- 4. Имена со словами «новый/финал/копия» ----",
                  "# Замените слово на дату документа. Автозамены нет",
                  "# намеренно: правильную дату знает только человек."]
        for it in bads:
            lines.append(f"#   {it['path']}   ← «{it['word']}»")
        lines.append("")

    lines += ["# После уборки: python index.py update && python audit.py report"]
    return "\n".join(lines) + "\n"


def problems_csv() -> str:
    """Все находки одной таблицей — для владельца контента."""
    rows = [("категория", "путь", "деталь")]
    for it in untyped(limit=500):
        rows.append(("без типа", it["path"], it["hint"] or ""))
    for it in undated(limit=500):
        rows.append(("без даты", it["path"], it["mtime_hint"] or ""))
    for it in bad_names(limit=500):
        rows.append(("плохое имя", it["path"], it["word"]))
    for g in exact_duplicates(limit=200):
        for p in g["paths"][1:]:
            rows.append(("дубль", p, f"копий: {g['count']}"))
    for t in brand_twins():
        names = " | ".join(v["brand"] for v in t["variants"])
        rows.append(("бренды-двойники", names, ""))
    for g in coverage_gaps():
        rows.append(("пробел покрытия", g["brand"], ", ".join(g["missing"])))
    out = []
    for r in rows:
        out.append(";".join('"' + str(c).replace('"', '""') + '"' for c in r))
    return "﻿" + "\n".join(out) + "\n"      # BOM — чтобы Excel понял UTF-8


def state() -> dict:
    """Всё, что нужно разделу, одним вызовом."""
    return {
        "progress": metadata_progress(),
        "untyped": untyped(),
        "undated": undated(),
        "bad_names": bad_names(),
        "brand_twins": brand_twins(),
        "duplicates": exact_duplicates(),
        "gaps": coverage_gaps(),
        "top_asked": top_asked_brands(),
    }
