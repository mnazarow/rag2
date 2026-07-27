"""
Аудит проиндексированной базы знаний.

  python audit.py report                — все проверки в консоль
  python audit.py report --json out.json

Что проверяется (по убыванию практической пользы):
  1. покрытие: матрица «бренд × тип документа» — где дыры в базе;
  2. дубликаты и почти-дубликаты по содержимому;
  3. документы-сироты: фрагменты, у которых нет близких соседей —
     обычно это либо уникальное знание, либо мусор;
  4. мёртвые документы: те, что ни разу не попали в выдачу;
  5. подозрение на противоречие: два актуальных документа одной сущности;
  6. распределение длин фрагментов и выбросы;
  7. полнота обработки: сканы без OCR, чертежи без надписей,
     видео без расшифровки, изображения без описания.

Это важнее любой картинки: картинка показывает, что «что-то не так»,
а отчёт говорит, с каким конкретно файлом идти к ответственному.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict

import numpy as np

import config
import db


def coverage() -> dict:
    """Матрица «бренд × тип документа». Пустая клетка — пробел в базе."""
    rows = db.q("""SELECT COALESCE(section,'—') section, COALESCE(brand,'—') brand,
                          COALESCE(doc_type,'—') doc_type, COUNT(*) n
                   FROM documents WHERE status='ok' AND is_current=1
                   GROUP BY section, brand, doc_type""")
    matrix: dict[str, dict[str, int]] = defaultdict(dict)
    types: Counter = Counter()
    for r in rows:
        key = f"{r['section']} / {r['brand']}"
        matrix[key][r["doc_type"]] = r["n"]
        types[r["doc_type"]] += r["n"]
    key_types = [t for t, _ in types.most_common(8)]
    gaps = []
    for brand, present in matrix.items():
        missing = [t for t in ("КАТАЛОГ", "ПРАЙС-ЛИСТ", "ПАСПОРТ", "СЕРТИФИКАТ", "РУКОВОДСТВО")
                   if t not in present]
        if missing:
            gaps.append({"brand": brand, "missing": missing})
    return {"matrix": dict(matrix), "types": key_types, "gaps": gaps}


def duplicates() -> dict:
    """Точные дубли по хэшу содержимого."""
    groups = db.q("""SELECT content_hash, COUNT(*) n, GROUP_CONCAT(rel_path, '||') paths
                     FROM documents WHERE status IN ('ok','duplicate')
                     GROUP BY content_hash HAVING COUNT(*) > 1 ORDER BY n DESC""")
    items = [{"count": g["n"], "paths": (g["paths"] or "").split("||")[:6]} for g in groups]
    return {"groups": len(items), "extra_copies": sum(i["count"] - 1 for i in items),
            "top": items[:15]}


def near_duplicates(sample: int = 4000, threshold: float = 0.97) -> dict:
    """
    Почти-дубли: фрагменты с косинусной близостью выше порога.
    Считаем на выборке — полная матрица на 70 тысячах векторов не нужна,
    для оценки масштаба проблемы хватает случайной подвыборки.
    """
    store = db.vectors()
    if len(store) < 50:
        return {"pairs": 0, "note": "векторов слишком мало"}
    idx = np.arange(len(store))
    if len(idx) > sample:
        rng = np.random.default_rng(0)
        idx = rng.choice(idx, sample, replace=False)
    m = store.matrix[idx]
    pairs = 0
    step = 500
    for start in range(0, len(m), step):
        block = m[start:start + step] @ m.T
        block[np.arange(len(block)), np.arange(start, start + len(block))] = 0
        pairs += int((block > threshold).sum() // 2)
    return {"checked": int(len(m)), "pairs": pairs,
            "share": round(pairs / max(len(m), 1), 4)}


def orphans(sample: int = 4000, quantile: float = 0.02) -> dict:
    """
    Фрагменты, у которых ближайший сосед очень далеко.
    Такие либо содержат уникальное знание (хорошо), либо это мусор,
    обрывки колонтитулов и служебных страниц (плохо).
    """
    store = db.vectors()
    if len(store) < 100:
        return {"items": [], "note": "векторов слишком мало"}
    idx = np.arange(len(store))
    if len(idx) > sample:
        rng = np.random.default_rng(1)
        idx = rng.choice(idx, sample, replace=False)
    m = store.matrix[idx]
    best = np.full(len(m), -1.0)
    step = 500
    for start in range(0, len(m), step):
        block = m[start:start + step] @ m.T
        block[np.arange(len(block)), np.arange(start, start + len(block))] = -1
        best[start:start + len(block)] = block.max(axis=1)
    cut = float(np.quantile(best, quantile))
    lonely = [int(store.ids[idx[i]]) for i in np.argsort(best)[:40]]
    items = []
    if lonely:
        rows = db.q(f"""SELECT c.id, d.rel_path, substr(c.text,1,110) t
                        FROM chunks c JOIN documents d ON d.id=c.doc_id
                        WHERE c.id IN ({','.join('?' * len(lonely))})""", lonely)
        items = [{"chunk_id": r["id"], "path": r["rel_path"], "text": r["t"]} for r in rows]
    return {"threshold": round(cut, 4), "items": items[:20]}


def dead_documents(limit: int = 30) -> dict:
    """Документы, ни разу не попавшие в выдачу за всю историю запросов."""
    total_q = db.q1("SELECT COUNT(*) n FROM queries")["n"]
    if total_q < 20:
        return {"note": f"запросов пока мало ({total_q}) — статистика не показательна",
                "items": []}
    used: set[int] = set()
    for r in db.q("SELECT sources_json FROM queries WHERE sources_json IS NOT NULL"):
        for s in json.loads(r["sources_json"] or "[]"):
            used.add(s.get("doc_id"))
    rows = db.q("""SELECT id, rel_path, text_chars FROM documents
                   WHERE status='ok' AND is_current=1 ORDER BY text_chars DESC""")
    dead = [{"id": r["id"], "path": r["rel_path"], "chars": r["text_chars"]}
            for r in rows if r["id"] not in used]
    return {"total_documents": len(rows), "never_used": len(dead), "top": dead[:limit]}


def conflicts(limit: int = 25) -> dict:
    """
    Кандидаты на противоречие: несколько актуальных документов с одним
    ключом версии. Обычно это две редакции прайса или паспорта,
    из которых ни одна не помечена устаревшей.
    """
    rows = db.q("""SELECT version_key, COUNT(*) n,
                          GROUP_CONCAT(rel_path || ' [' || COALESCE(effective_date,'без даты') || ']', '||') paths
                   FROM documents
                   WHERE status='ok' AND is_current=1 AND version_key <> ''
                     AND doc_type IN ('ПРАЙС-ЛИСТ','КАТАЛОГ','ПАСПОРТ','СЕРТИФИКАТ')
                   GROUP BY version_key HAVING COUNT(*) > 1 ORDER BY n DESC""")
    return {"groups": len(rows),
            "top": [{"count": r["n"], "docs": (r["paths"] or "").split("||")[:4]}
                    for r in rows[:limit]]}


def chunk_stats() -> dict:
    rows = db.q("SELECT n_chars FROM chunks")
    if not rows:
        return {}
    lengths = np.array([r["n_chars"] for r in rows])
    return {
        "count": int(len(lengths)),
        "median": int(np.median(lengths)),
        "p05": int(np.quantile(lengths, 0.05)),
        "p95": int(np.quantile(lengths, 0.95)),
        "too_short": int((lengths < 150).sum()),
        "too_long": int((lengths > 4000).sum()),
    }


def enrichment() -> dict:
    """Полнота обработки нетекстовых материалов."""
    def count(where: str, params=()) -> int:
        return db.q1(f"SELECT COUNT(*) n FROM documents WHERE {where}", params)["n"]
    return {
        "scans_without_ocr": count("needs_ocr=1 AND status='ok'"),
        "videos": count("asset_kind='video'"),
        "videos_transcribed": count("asset_kind='video' AND enriched LIKE '%asr%'"),
        "images": count("asset_kind='image'"),
        "images_described": count("asset_kind='image' AND enriched LIKE '%vision%'"),
        "drawings": count("asset_kind='drawing'"),
        "drawings_with_text": count("asset_kind='drawing' AND enriched LIKE '%cad%'"),
        "archives": count("kind='archive'"),
        "from_web": count("source_type<>'internal_kb'"),
    }


def build_report() -> dict:
    db.init()
    return {
        "coverage": coverage(),
        "duplicates": duplicates(),
        "near_duplicates": near_duplicates(),
        "orphans": orphans(),
        "dead_documents": dead_documents(),
        "conflicts": conflicts(),
        "chunks": chunk_stats(),
        "enrichment": enrichment(),
    }


def print_report(rep: dict) -> None:
    print("=" * 74)
    print("АУДИТ БАЗЫ ЗНАНИЙ")
    print("=" * 74)

    ch = rep["chunks"]
    if ch:
        print(f"\nФрагментов: {ch['count']}, медиана {ch['median']} симв. "
              f"(5-й перцентиль {ch['p05']}, 95-й {ch['p95']})")
        print(f"  слишком коротких (<150): {ch['too_short']}, "
              f"слишком длинных (>4000): {ch['too_long']}")

    d = rep["duplicates"]
    print(f"\nДУБЛИКАТЫ: групп {d['groups']}, лишних копий {d['extra_copies']}")
    for item in d["top"][:5]:
        print(f"  ×{item['count']}: {item['paths'][0][:90]}")

    nd = rep["near_duplicates"]
    if "pairs" in nd:
        print(f"\nПОЧТИ-ДУБЛИ: на выборке {nd.get('checked','?')} фрагментов "
              f"найдено {nd['pairs']} очень близких пар")

    c = rep["conflicts"]
    print(f"\nКАНДИДАТЫ НА ПРОТИВОРЕЧИЕ: {c['groups']} групп")
    for item in c["top"][:5]:
        print(f"  ×{item['count']}:")
        for doc in item["docs"][:3]:
            print(f"      {doc[:95]}")

    g = rep["coverage"]["gaps"]
    print(f"\nПРОБЕЛЫ В ПОКРЫТИИ: у {len(g)} брендов нет обязательных типов документов")
    for item in g[:10]:
        print(f"  {item['brand'][:55]:<57} нет: {', '.join(item['missing'])}")

    o = rep["orphans"]
    if o.get("items"):
        print(f"\nФРАГМЕНТЫ-СИРОТЫ (порог близости {o['threshold']}):")
        for item in o["items"][:6]:
            print(f"  {item['path'][:70]}")
            print(f"      {item['text'][:90]}")

    dd = rep["dead_documents"]
    if dd.get("items") is not None and dd.get("never_used") is not None:
        print(f"\nНИ РАЗУ НЕ ПОПАЛИ В ВЫДАЧУ: {dd['never_used']} из {dd['total_documents']}")
        for item in dd.get("top", [])[:6]:
            print(f"  {item['path'][:85]}")
    elif dd.get("note"):
        print(f"\nМЁРТВЫЕ ДОКУМЕНТЫ: {dd['note']}")

    e = rep["enrichment"]
    print("\nПОЛНОТА ОБРАБОТКИ НЕТЕКСТОВЫХ МАТЕРИАЛОВ")
    print(f"  сканы без распознавания : {e['scans_without_ocr']}")
    print(f"  видео расшифровано      : {e['videos_transcribed']} из {e['videos']}")
    print(f"  изображений описано     : {e['images_described']} из {e['images']}")
    print(f"  чертежей с надписями    : {e['drawings_with_text']} из {e['drawings']}")
    print(f"  архивов распаковано     : {e['archives']}")
    print(f"  материалов из интернета : {e['from_web']}")


def main() -> int:
    p = argparse.ArgumentParser(description="Аудит базы знаний")
    p.add_argument("command", choices=["report"], nargs="?", default="report")
    p.add_argument("--json", help="сохранить отчёт в файл")
    a = p.parse_args()
    rep = build_report()
    print_report(rep)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, ensure_ascii=False, indent=2)
        print(f"\nОтчёт сохранён: {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
