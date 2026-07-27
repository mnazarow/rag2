"""
Граф связности базы знаний.

  python graph.py build --out graph.html

Главная проблема таких графов — нечитаемость. Четыре тысячи документов,
соединённых линиями, превращаются в клубок, из которого ничего не видно.
Поэтому здесь сделано иначе.

По умолчанию показывается не документ, а группа: раздел или бренд.
Сорок узлов вместо четырёх тысяч — это уже карта, а не клубок. Двойной
щелчок по группе раскрывает её на уровень ниже: раздел → бренд →
категория → документ. Так работает исследование от общего к частному,
и на каждом уровне на экране остаётся обозримое число узлов.

Связи между группами — это сумма связей между их документами: толщина
линии показывает, насколько сильно связаны, скажем, раздел розницы
и раздел дилерской продукции.

Дальше — фильтры, раскладки и режимы подсветки, чтобы отвечать на
конкретные вопросы: где документы без связей, где устаревшие версии,
где дубликаты, чего не хватает у бренда, что вообще ни разу не
находилось в ответах.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import config
import db
import logging_setup

log = logging_setup.get("web")

PALETTE = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1",
           "#76B7B2", "#EDC948", "#FF9DA7", "#9C755F", "#BAB0AC",
           "#86BCB6", "#D37295", "#A0CBE8", "#FFBE7D", "#8CD17D"]


def document_vectors() -> tuple[list[int], np.ndarray]:
    """Средний вектор документа — центр масс его фрагментов."""
    store = db.vectors()
    if len(store) == 0:
        return [], np.zeros((0, 1), np.float32)
    chunk_to_doc = {r["id"]: r["doc_id"] for r in db.q("SELECT id, doc_id FROM chunks")}
    acc: dict[int, list[np.ndarray]] = defaultdict(list)
    for i, cid in enumerate(store.ids):
        doc = chunk_to_doc.get(cid)
        if doc is not None:
            acc[doc].append(store.matrix[i])
    ids, rows = [], []
    for doc, vecs in acc.items():
        v = np.mean(vecs, axis=0)
        n = np.linalg.norm(v)
        if n > 0:
            ids.append(doc)
            rows.append(v / n)
    return ids, (np.vstack(rows) if rows else np.zeros((0, 1), np.float32))


def _category(rel_path: str, doc_type: str | None) -> str:
    if doc_type:
        return doc_type
    parts = Path(rel_path).parts
    return parts[2] if len(parts) > 3 else "—"


def build_graph(similarity: float = 0.5, max_edges_per_node: int = 6,
                limit: int | None = None) -> dict:
    """Собирает данные графа. Агрегация и фильтры делаются уже в браузере."""
    db.init()
    rows = db.q("""SELECT id, rel_path, file_name, section, brand, doc_type, kind,
                          asset_kind, text_chars, effective_date, is_current,
                          needs_ocr, enriched, status, source_type
                   FROM documents WHERE status IN ('ok','duplicate')""")
    if limit:
        rows = rows[:limit]

    # Какие документы вообще попадали в ответы.
    used: set[int] = set()
    for r in db.q("SELECT sources_json FROM queries WHERE sources_json IS NOT NULL"):
        for s in json.loads(r["sources_json"] or "[]"):
            if s.get("doc_id"):
                used.add(s["doc_id"])

    # Дубликаты по содержимому.
    dupe_groups: dict[str, list[int]] = defaultdict(list)
    for r in db.q("SELECT id, content_hash FROM documents WHERE status IN ('ok','duplicate')"):
        dupe_groups[r["content_hash"]].append(r["id"])
    dupes = {i for group in dupe_groups.values() if len(group) > 1 for i in group}

    sections = sorted({(r["section"] or "—") for r in rows})
    color_of = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(sections)}

    nodes = []
    for r in rows:
        chars = r["text_chars"] or 0
        nodes.append({
            "id": r["id"],
            "label": (r["file_name"] or "")[:70],
            "path": r["rel_path"],
            "section": r["section"] or "—",
            "brand": r["brand"] or "—",
            "cat": _category(r["rel_path"], r["doc_type"]),
            "type": r["doc_type"] or "—",
            "kind": r["asset_kind"] or r["kind"] or "text",
            "chars": chars,
            "date": r["effective_date"] or "",
            "year": (r["effective_date"] or "")[:4],
            "current": 1 if r["is_current"] else 0,
            "ocr": 1 if r["needs_ocr"] else 0,
            "dupe": 1 if r["id"] in dupes else 0,
            "used": 1 if r["id"] in used else 0,
            "enriched": r["enriched"] or "",
            "web": 1 if (r["source_type"] or "internal_kb") != "internal_kb" else 0,
            "color": color_of[r["section"] or "—"],
        })

    keep = {n["id"] for n in nodes}
    edges: list[dict] = []

    # Структурные связи: документы из одной папки. Соединяем цепочкой,
    # а не каждый с каждым — иначе большая папка сама по себе даёт клубок.
    by_folder: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        by_folder[str(Path(r["rel_path"]).parent)].append(r["id"])
    for members in by_folder.values():
        for i in range(min(len(members) - 1, 60)):
            edges.append({"s": members[i], "t": members[i + 1], "k": "folder", "w": 0.2})

    # Смысловые связи.
    ids, mat = document_vectors()
    orphan_ids: set[int] = set()
    if len(ids) > 1:
        mask = [i for i, d in enumerate(ids) if d in keep]
        ids = [ids[i] for i in mask]
        mat = mat[mask]
        connected: set[int] = set()
        step = 400
        seen_pairs: set[tuple[int, int]] = set()
        for start in range(0, len(mat), step):
            block = mat[start:start + step] @ mat.T
            for local, row in enumerate(block):
                i = start + local
                row[i] = -1
                order = np.argsort(-row)[:max_edges_per_node]
                for j in order:
                    if row[j] < similarity:
                        break
                    a, b = ids[i], ids[int(j)]
                    pair = (a, b) if a < b else (b, a)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    edges.append({"s": pair[0], "t": pair[1], "k": "sem",
                                  "w": round(float(row[j]), 3)})
                    connected.add(a)
                    connected.add(b)
        orphan_ids = set(ids) - connected

    for n in nodes:
        n["orphan"] = 1 if n["id"] in orphan_ids else 0

    stats = {
        "documents": len(nodes),
        "edges_semantic": sum(1 for e in edges if e["k"] == "sem"),
        "edges_folder": sum(1 for e in edges if e["k"] == "folder"),
        "orphans": len(orphan_ids),
        "outdated": sum(1 for n in nodes if not n["current"]),
        "scans": sum(1 for n in nodes if n["ocr"]),
        "dupes": sum(1 for n in nodes if n["dupe"]),
        "never_used": sum(1 for n in nodes if not n["used"]),
        "sections": {s: sum(1 for n in nodes if n["section"] == s) for s in sections},
        "similarity": similarity,
    }
    log.info("граф: узлов %d, смысловых связей %d, сирот %d",
             stats["documents"], stats["edges_semantic"], stats["orphans"])
    return {"nodes": nodes, "edges": edges, "stats": stats, "colors": color_of}


HTML_TEMPLATE = r"""<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8">
<title>Граф связности базы знаний</title><style>
*{box-sizing:border-box}
body{margin:0;font:13px/1.45 -apple-system,Segoe UI,Roboto,sans-serif;
 background:#0f1116;color:#e6e8ec;overflow:hidden}
#wrap{display:flex;height:100vh}
#side{width:310px;flex:none;background:#171a21;border-right:1px solid #262a33;
 overflow-y:auto;padding:14px}
#side::-webkit-scrollbar{width:8px}#side::-webkit-scrollbar-thumb{background:#2b3140;border-radius:4px}
#stage{flex:1;position:relative;min-width:0}
canvas{display:block;cursor:grab}
h1{font-size:15px;margin:0 0 3px}
h2{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:#8b93a3;
 margin:16px 0 7px;font-weight:600;border-bottom:1px solid #232833;padding-bottom:4px}
label{font-size:12px;color:#98a1b0;display:block;margin:8px 0 3px}
select,input[type=text],input[type=search]{width:100%;background:#0f1116;border:1px solid #2c313c;
 color:#e6e8ec;border-radius:6px;padding:6px 8px;font-size:12.5px}
input[type=range]{width:100%;margin:2px 0}
.row{display:flex;gap:6px}.row>*{flex:1}
.chk{display:flex;align-items:center;gap:7px;padding:3px 0;font-size:12.5px;cursor:pointer}
.chk input{margin:0}
.stat{display:flex;justify-content:space-between;padding:2.5px 0;font-size:12.5px;
 border-bottom:1px solid #21252e;cursor:pointer}
.stat:hover{color:#fff}.stat b{font-weight:600}
.legend{display:flex;align-items:center;gap:7px;padding:2.5px 0;cursor:pointer;font-size:12.5px}
.legend.off{opacity:.32}.dot{width:10px;height:10px;border-radius:3px;flex:none}
.btn{background:#232833;border:1px solid #2f3542;color:#c8cede;border-radius:6px;
 padding:5px 9px;font-size:12px;cursor:pointer}
.btn:hover{background:#2b313d}.btn.on{background:#2f6fb0;color:#fff;border-color:#2f6fb0}
.btns{display:flex;flex-wrap:wrap;gap:5px;margin-top:5px}
#tip{position:absolute;pointer-events:none;background:#1d2129;border:1px solid #333a46;
 border-radius:8px;padding:9px 11px;max-width:430px;font-size:12px;display:none;
 box-shadow:0 10px 34px rgba(0,0,0,.6);z-index:6}
#tip b{color:#fff}.muted{color:#8b93a3}
.chip{display:inline-block;background:#252a34;border-radius:4px;padding:1px 6px;
 margin:2px 3px 0 0;font-size:11px}
.warn{color:#F28E2B}.bad{color:#E15759}.good{color:#59A14F}
#crumbs{position:absolute;top:10px;left:14px;font-size:12.5px;color:#8b93a3;z-index:5}
#crumbs a{color:#7fb0e0;cursor:pointer;text-decoration:none}
#hint{position:absolute;bottom:10px;left:14px;font-size:11px;color:#5f6875;z-index:5}
#counter{position:absolute;top:10px;right:14px;font-size:12px;color:#8b93a3;
 background:#171a21cc;padding:4px 9px;border-radius:6px;z-index:5}
#detail{position:absolute;right:14px;top:44px;width:330px;max-height:70vh;overflow:auto;
 background:#171a21;border:1px solid #2b3140;border-radius:10px;padding:12px;display:none;z-index:6}
#detail h3{margin:0 0 6px;font-size:13px}
#detail .close{float:right;cursor:pointer;color:#8b93a3}
table{width:100%;border-collapse:collapse;font-size:12px}
td{padding:2px 0;border-bottom:1px solid #232833;vertical-align:top}
td:first-child{color:#8b93a3;width:38%}
</style></head><body>
<div id="wrap">
<div id="side">
 <h1>Граф связности</h1>
 <div class="muted" style="font-size:11.5px">Начните с общего вида, двойным щелчком
  раскрывайте группы до отдельных документов.</div>

 <h2>Уровень</h2>
 <div class="btns" id="levelBtns"></div>

 <h2>Раскладка</h2>
 <select id="layout">
   <option value="cluster">кластеры по группам</option>
   <option value="force">силовая</option>
   <option value="radial">радиальное дерево</option>
   <option value="circle">по кругу</option>
   <option value="grid">сеткой</option>
 </select>
 <label>Разрежённость <span id="spreadV" class="muted"></span></label>
 <input type="range" id="spread" min="40" max="260" value="110">
 <div class="btns"><button class="btn" id="relayout">Пересчитать</button>
  <button class="btn" id="fit">Вписать</button></div>

 <h2>Поиск</h2>
 <input type="search" id="q" placeholder="имя, путь, бренд, артикул">
 <div class="btns"><button class="btn" id="isolate">Показать только найденное</button></div>

 <h2>Готовые виды</h2>
 <div class="btns" id="presets"></div>

 <h2>Что показывать</h2>
 <label class="chk"><input type="checkbox" id="fOrphan"> только без смысловых связей</label>
 <label class="chk"><input type="checkbox" id="fOutdated"> только устаревшие версии</label>
 <label class="chk"><input type="checkbox" id="fOcr"> только сканы без распознавания</label>
 <label class="chk"><input type="checkbox" id="fDupe"> только дубликаты</label>
 <label class="chk"><input type="checkbox" id="fUnused"> только не попадавшие в ответы</label>
 <label class="chk"><input type="checkbox" id="fAsset"> только нетекстовые материалы</label>
 <label class="chk"><input type="checkbox" id="fWeb"> только загруженные с сайтов</label>
 <label>Тип документа</label><select id="fType"></select>
 <label>Вид материала</label><select id="fKind"></select>
 <label>Год документа</label><select id="fYear"></select>

 <h2>Связи</h2>
 <label class="chk"><input type="checkbox" id="eSem" checked> смысловые</label>
 <label class="chk"><input type="checkbox" id="eFolder"> структурные (общая папка)</label>
 <label>Порог близости <span id="simV" class="muted"></span></label>
 <input type="range" id="sim" min="0" max="95" value="0">
 <label>Максимум связей у узла <span id="degV" class="muted"></span></label>
 <input type="range" id="deg" min="1" max="12" value="6">

 <h2>Вид узлов</h2>
 <label>Цвет</label>
 <select id="colorBy">
   <option value="section">по разделу базы</option>
   <option value="type">по типу документа</option>
   <option value="kind">по виду материала</option>
   <option value="state">по состоянию (проблемы)</option>
   <option value="year">по году документа</option>
 </select>
 <label>Размер</label>
 <select id="sizeBy">
   <option value="chars">по объёму текста</option>
   <option value="count">по числу документов в группе</option>
   <option value="degree">по числу связей</option>
   <option value="uniform">одинаковый</option>
 </select>
 <label class="chk"><input type="checkbox" id="showLabels" checked> подписи</label>
 <label class="chk"><input type="checkbox" id="dimOthers" checked> приглушать несвязанные при наведении</label>

 <h2>Сводка</h2>
 <div id="stats"></div>

 <h2 id="legendTitle">Разделы базы</h2>
 <div id="legend"></div>

 <h2>Экспорт</h2>
 <div class="btns"><button class="btn" id="png">Картинкой</button>
  <button class="btn" id="csv">Список в CSV</button></div>
</div>
<div id="stage">
  <canvas id="c"></canvas>
  <div id="tip"></div>
  <div id="crumbs"></div>
  <div id="counter"></div>
  <div id="detail"></div>
  <div id="hint">колесо — масштаб · перетаскивание — сдвиг · щелчок — карточка ·
   двойной щелчок — раскрыть группу</div>
</div>
</div>
<script>
const DATA = __DATA__;
const RAW = DATA.nodes, EDGES = DATA.edges;
const byId = new Map(RAW.map(n => [n.id, n]));
const esc = s => String(s ?? '').replace(/[<>&"]/g, c => (
  {'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;'}[c]));
const RU = n => (n ?? 0).toLocaleString('ru');

const LEVELS = [
  {key:'section', title:'Разделы'}, {key:'brand', title:'Бренды'},
  {key:'cat', title:'Категории'}, {key:'doc', title:'Документы'},
];
const STATE_COLORS = {
  'в порядке':'#59A14F', 'без связей':'#E15759', 'устаревший':'#7a828f',
  'скан без OCR':'#F28E2B', 'дубликат':'#B07AA1', 'не находился':'#EDC948',
};
let level = 0;           // текущий уровень агрегации
let path = [];           // выбранный путь: [раздел, бренд, категория]
let offSections = new Set();
let nodes = [], links = [], hover = null, selected = null;
let scale = 1, ox = 0, oy = 0, isolateMode = false;

const el = id => document.getElementById(id);
const cv = el('c'), ctx = cv.getContext('2d');

/* ------------------------------------------- фильтрация исходных данных --- */
function passesFilters(n) {
  if (offSections.has(n.section)) return false;
  if (el('fOrphan').checked && !n.orphan) return false;
  if (el('fOutdated').checked && n.current) return false;
  if (el('fOcr').checked && !n.ocr) return false;
  if (el('fDupe').checked && !n.dupe) return false;
  if (el('fUnused').checked && n.used) return false;
  if (el('fAsset').checked && ['text','table'].includes(n.kind)) return false;
  if (el('fWeb').checked && !n.web) return false;
  const t = el('fType').value, k = el('fKind').value, y = el('fYear').value;
  if (t && n.type !== t) return false;
  if (k && n.kind !== k) return false;
  if (y && n.year !== y) return false;
  if (path[0] && n.section !== path[0]) return false;
  if (path[1] && n.brand !== path[1]) return false;
  if (path[2] && n.cat !== path[2]) return false;
  const q = el('q').value.trim().toLowerCase();
  if (q && isolateMode) {
    if (!((n.label + ' ' + n.path + ' ' + n.brand).toLowerCase().includes(q))) return false;
  }
  return true;
}
function nodeState(n) {
  if (n.orphan) return 'без связей';
  if (!n.current) return 'устаревший';
  if (n.ocr) return 'скан без OCR';
  if (n.dupe) return 'дубликат';
  if (!n.used) return 'не находился';
  return 'в порядке';
}
function groupKeyOf(n) {
  return level === 0 ? n.section : level === 1 ? n.section + ' / ' + n.brand
       : level === 2 ? n.cat : null;
}

/* ------------------------------------------------------ сборка вершин ----- */
function rebuild() {
  const kept = RAW.filter(passesFilters);
  const keptIds = new Set(kept.map(n => n.id));
  const minSim = +el('sim').value / 100;
  const useSem = el('eSem').checked, useFolder = el('eFolder').checked;

  if (level === 3) {
    nodes = kept.map(n => ({...n, gid: n.id, count: 1, isGroup: false}));
  } else {
    const groups = new Map();
    for (const n of kept) {
      const key = groupKeyOf(n);
      let g = groups.get(key);
      if (!g) {
        g = {gid: key, label: (level === 1 ? n.brand : key), isGroup: true, count: 0,
             chars: 0, section: n.section, brand: n.brand, cat: n.cat,
             type: '', kind: '', orphan: 0, current: 0, ocr: 0, dupe: 0, used: 0,
             web: 0, members: []};
        groups.set(key, g);
      }
      g.count++; g.chars += n.chars; g.members.push(n.id);
      g.orphan += n.orphan; g.ocr += n.ocr; g.dupe += n.dupe;
      g.current += n.current ? 0 : 1; g.used += n.used ? 0 : 1; g.web += n.web;
    }
    nodes = [...groups.values()];
  }

  const index = new Map(nodes.map((n, i) => [n.gid, i]));
  const memberGroup = new Map();
  if (level === 3) { for (const n of nodes) memberGroup.set(n.id, n.gid); }
  else { for (const g of nodes) for (const m of g.members) memberGroup.set(m, g.gid); }

  const agg = new Map();
  for (const e of EDGES) {
    if (e.k === 'sem' && (!useSem || e.w < minSim)) continue;
    if (e.k === 'folder' && !useFolder) continue;
    if (!keptIds.has(e.s) || !keptIds.has(e.t)) continue;
    const a = memberGroup.get(e.s), b = memberGroup.get(e.t);
    if (a === undefined || b === undefined || a === b) continue;
    const key = a < b ? a + '|' + b : b + '|' + a;
    const cur = agg.get(key);
    if (cur) { cur.n++; cur.w = Math.max(cur.w, e.w); }
    else agg.set(key, {a, b, n: 1, w: e.w, k: e.k});
  }
  // Ограничение степени: у каждого узла оставляем самые сильные связи.
  const maxDeg = +el('deg').value;
  const perNode = new Map();
  const all = [...agg.values()].sort((x, y) => y.n - x.n || y.w - x.w);
  links = [];
  for (const e of all) {
    const da = perNode.get(e.a) || 0, dbn = perNode.get(e.b) || 0;
    if (da >= maxDeg && dbn >= maxDeg) continue;
    perNode.set(e.a, da + 1); perNode.set(e.b, dbn + 1);
    links.push({s: index.get(e.a), t: index.get(e.b), n: e.n, w: e.w, k: e.k});
  }
  links = links.filter(l => l.s !== undefined && l.t !== undefined);
  nodes.forEach((n, i) => n.degree = 0);
  links.forEach(l => { nodes[l.s].degree++; nodes[l.t].degree++; });

  layout();
  applyStyle();
  renderPanel();
  fitView();
}

/* --------------------------------------------------------- раскладки ----- */
function layout() {
  const mode = el('layout').value, N = nodes.length, spread = +el('spread').value;
  const R = spread * Math.sqrt(Math.max(N, 1)) * 0.9;
  if (!N) return;

  if (mode === 'circle') {
    nodes.forEach((n, i) => { const a = i / N * 6.2832;
      n.x = Math.cos(a) * R; n.y = Math.sin(a) * R; });
    return;
  }
  if (mode === 'grid') {
    const cols = Math.ceil(Math.sqrt(N)), gap = spread * 0.8;
    nodes.forEach((n, i) => { n.x = (i % cols - cols / 2) * gap;
      n.y = (Math.floor(i / cols) - cols / 2) * gap; });
    return;
  }
  if (mode === 'radial') {
    // Дерево по структуре: центр — текущая область, лучи — группы.
    const groups = new Map();
    nodes.forEach(n => {
      const k = level === 3 ? n.cat : n.section;
      if (!groups.has(k)) groups.set(k, []);
      groups.get(k).push(n);
    });
    let gi = 0;
    const G = groups.size;
    for (const [, members] of groups) {
      const base = gi / G * 6.2832; gi++;
      members.forEach((n, j) => {
        const ring = 1 + Math.floor(j / 14);
        const off = (j % 14 - 6.5) * 0.055;
        n.x = Math.cos(base + off) * R * 0.35 * ring;
        n.y = Math.sin(base + off) * R * 0.35 * ring;
      });
    }
    return;
  }

  // Кластеры и силовая: сначала расставляем группы, потом узлы внутри.
  const clusterMode = mode === 'cluster';
  const anchors = new Map();
  if (clusterMode) {
    const keys = [...new Set(nodes.map(n => level === 3 ? n.cat : n.section))];
    keys.forEach((k, i) => {
      const a = i / keys.length * 6.2832;
      anchors.set(k, {x: Math.cos(a) * R * 0.85, y: Math.sin(a) * R * 0.85});
    });
  }
  nodes.forEach((n, i) => {
    const a = i * 2.399963, r = R * Math.sqrt(i / N);
    const base = clusterMode ? anchors.get(level === 3 ? n.cat : n.section) : {x: 0, y: 0};
    n.x = base.x + Math.cos(a) * r * (clusterMode ? 0.28 : 1);
    n.y = base.y + Math.sin(a) * r * (clusterMode ? 0.28 : 1);
    n.vx = n.vy = 0;
  });

  const k = spread * 1.15;
  const steps = N > 900 ? 90 : N > 300 ? 160 : 260;
  for (let step = 0; step < steps; step++) {
    const temp = 1 - step / steps;
    const cell = k * 2, grid = new Map();
    for (const n of nodes) {
      const key = ((n.x / cell) | 0) + ':' + ((n.y / cell) | 0);
      if (!grid.has(key)) grid.set(key, []);
      grid.get(key).push(n);
    }
    for (const n of nodes) {
      let fx = 0, fy = 0;
      const gx = (n.x / cell) | 0, gy = (n.y / cell) | 0;
      for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) {
        const bucket = grid.get((gx + dx) + ':' + (gy + dy)); if (!bucket) continue;
        for (const m of bucket) {
          if (m === n) continue;
          let ddx = n.x - m.x, ddy = n.y - m.y, d2 = ddx * ddx + ddy * ddy;
          if (d2 < 1e-4) { ddx = Math.random() - .5; ddy = Math.random() - .5; d2 = 1e-4; }
          if (d2 > 9 * k * k) continue;
          const f = k * k / d2;
          fx += ddx * f; fy += ddy * f;
        }
      }
      n.vx = fx; n.vy = fy;
    }
    for (const l of links) {
      const a = nodes[l.s], b = nodes[l.t];
      const ddx = a.x - b.x, ddy = a.y - b.y;
      const d = Math.max(Math.hypot(ddx, ddy), .01);
      const f = (d * d / k) * (l.k === 'sem' ? 1.5 : 0.5) * Math.min(1 + Math.log1p(l.n) / 3, 3);
      const ux = ddx / d * f, uy = ddy / d * f;
      a.vx -= ux; a.vy -= uy; b.vx += ux; b.vy += uy;
    }
    if (clusterMode) {
      for (const n of nodes) {
        const anchor = anchors.get(level === 3 ? n.cat : n.section);
        if (anchor) { n.vx += (anchor.x - n.x) * 0.06 * k; n.vy += (anchor.y - n.y) * 0.06 * k; }
      }
    }
    for (const n of nodes) {
      const d = Math.max(Math.hypot(n.vx, n.vy), .01);
      const lim = Math.min(d, k * temp * 2);
      n.x += n.vx / d * lim; n.y += n.vy / d * lim;
      if (!clusterMode) { n.x -= n.x * 0.0012; n.y -= n.y * 0.0012; }
    }
  }
}

/* ------------------------------------------------------ цвета и размеры --- */
let legendItems = [];
function applyStyle() {
  const colorBy = el('colorBy').value, sizeBy = el('sizeBy').value;
  const palette = ["#4E79A7","#F28E2B","#59A14F","#E15759","#B07AA1","#76B7B2",
                   "#EDC948","#FF9DA7","#9C755F","#BAB0AC","#86BCB6","#D37295"];
  const keyOf = n => colorBy === 'section' ? n.section
    : colorBy === 'type' ? (n.type || '—')
    : colorBy === 'kind' ? (n.kind || '—')
    : colorBy === 'year' ? (n.year || 'без даты')
    : nodeState(n);
  const counts = new Map();
  nodes.forEach(n => { const k = keyOf(n); counts.set(k, (counts.get(k) || 0) + (n.count || 1)); });
  const keys = [...counts.keys()].sort((a, b) => counts.get(b) - counts.get(a));
  const map = new Map();
  keys.forEach((k, i) => map.set(k, colorBy === 'state' ? (STATE_COLORS[k] || '#888')
                                                       : palette[i % palette.length]));
  legendItems = keys.map(k => ({key: k, color: map.get(k), count: counts.get(k)}));

  const maxCount = Math.max(...nodes.map(n => n.count || 1), 1);
  const maxDeg = Math.max(...nodes.map(n => n.degree || 0), 1);
  nodes.forEach(n => {
    n.fill = map.get(keyOf(n));
    if (sizeBy === 'uniform') n.r = 5;
    else if (sizeBy === 'count') n.r = 3.5 + 12 * Math.sqrt((n.count || 1) / maxCount);
    else if (sizeBy === 'degree') n.r = 3.5 + 10 * Math.sqrt((n.degree || 0) / maxDeg);
    else n.r = 3 + Math.min(Math.log10((n.chars || 0) + 10) * 2.6, 13);
    if (level < 3) n.r = Math.max(n.r, 6);
  });
}

/* ------------------------------------------------------------ отрисовка --- */
function resize() {
  const stage = el('stage');
  cv.width = stage.clientWidth * devicePixelRatio;
  cv.height = stage.clientHeight * devicePixelRatio;
  cv.style.width = stage.clientWidth + 'px';
  cv.style.height = stage.clientHeight + 'px';
  draw();
}
function fitView() {
  if (!nodes.length) { draw(); return; }
  const xs = nodes.map(n => n.x), ys = nodes.map(n => n.y);
  const w = Math.max(...xs) - Math.min(...xs) || 1, h = Math.max(...ys) - Math.min(...ys) || 1;
  const cw = cv.width / devicePixelRatio, ch = cv.height / devicePixelRatio;
  scale = Math.min(cw / (w * 1.25), ch / (h * 1.25), 4);
  ox = -(Math.min(...xs) + w / 2) * scale;
  oy = -(Math.min(...ys) + h / 2) * scale;
  draw();
}
function neighbours(node) {
  const set = new Set([nodes.indexOf(node)]);
  links.forEach(l => { if (nodes[l.s] === node) set.add(l.t); if (nodes[l.t] === node) set.add(l.s); });
  return set;
}
function draw() {
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  ctx.clearRect(0, 0, cv.width, cv.height);
  const cx = cv.width / devicePixelRatio / 2 + ox, cy = cv.height / devicePixelRatio / 2 + oy;
  ctx.save(); ctx.translate(cx, cy); ctx.scale(scale, scale);
  const focus = (hover || selected) && el('dimOthers').checked
    ? neighbours(hover || selected) : null;

  for (const l of links) {
    const a = nodes[l.s], b = nodes[l.t];
    const dim = focus && !(focus.has(l.s) && focus.has(l.t));
    const strength = l.k === 'sem'
      ? Math.min(0.12 + l.w * 0.4 + Math.log1p(l.n) / 12, 0.75) : 0.07;
    ctx.strokeStyle = l.k === 'sem'
      ? `rgba(130,175,255,${dim ? strength * 0.15 : strength})`
      : `rgba(255,255,255,${dim ? 0.015 : 0.05})`;
    ctx.lineWidth = (l.k === 'sem' ? Math.min(0.5 + Math.log1p(l.n) / 2.4, 4) : 0.5) / scale;
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
  }
  const q = el('q').value.trim().toLowerCase();
  nodes.forEach((n, i) => {
    const dim = focus && !focus.has(i);
    const match = q && ((n.label || '') + ' ' + (n.path || '') + ' ' + (n.brand || ''))
      .toLowerCase().includes(q);
    ctx.globalAlpha = dim ? 0.16 : 1;
    ctx.beginPath();
    const r = n.r / Math.max(Math.sqrt(scale), .55);
    if (n.isGroup) ctx.arc(n.x, n.y, r, 0, 6.2832);
    else if (n.kind === 'drawing') ctx.rect(n.x - r, n.y - r, r * 2, r * 2);
    else if (n.kind === 'image') { ctx.moveTo(n.x, n.y - r); ctx.lineTo(n.x + r, n.y);
      ctx.lineTo(n.x, n.y + r); ctx.lineTo(n.x - r, n.y); ctx.closePath(); }
    else if (n.kind === 'video') { ctx.moveTo(n.x, n.y - r); ctx.lineTo(n.x + r, n.y + r);
      ctx.lineTo(n.x - r, n.y + r); ctx.closePath(); }
    else ctx.arc(n.x, n.y, r, 0, 6.2832);
    ctx.fillStyle = n.fill; ctx.fill();
    if (match || n === hover || n === selected) {
      ctx.lineWidth = 2.2 / scale;
      ctx.strokeStyle = match ? '#EDC948' : '#fff';
      ctx.stroke();
    } else if (n.isGroup) {
      ctx.lineWidth = 1 / scale; ctx.strokeStyle = 'rgba(255,255,255,.22)'; ctx.stroke();
    }
    ctx.globalAlpha = 1;
  });
  if (el('showLabels').checked) {
    const limit = level < 3 ? 260 : 90;
    const show = nodes.length <= limit || scale > 1.7;
    if (show) {
      ctx.fillStyle = 'rgba(230,232,236,.9)';
      const fs = Math.min(Math.max(10 / scale, 3), 14 / scale);
      ctx.font = fs + 'px sans-serif';
      ctx.textAlign = 'center';
      nodes.forEach(n => {
        const r = n.r / Math.max(Math.sqrt(scale), .55);
        const text = (n.label || n.gid || '').slice(0, level < 3 ? 26 : 20);
        ctx.fillText(text, n.x, n.y + r + fs * 1.1);
      });
      ctx.textAlign = 'start';
    }
  }
  ctx.restore();
  el('counter').textContent = `${RU(nodes.length)} узлов · ${RU(links.length)} связей`;
}

/* ------------------------------------------------------------ панель ------ */
function renderPanel() {
  el('levelBtns').innerHTML = LEVELS.map((l, i) =>
    `<button class="btn ${i === level ? 'on' : ''}" onclick="setLevel(${i})">${l.title}</button>`).join('');
  el('crumbs').innerHTML = '<a onclick="setPath([])">вся база</a>' +
    path.map((p, i) => ` / <a onclick="setPath(${i + 1})">${esc(p)}</a>`).join('');
  const s = DATA.stats;
  const rows = [
    ['Документов всего', s.documents, () => {}],
    ['Без смысловых связей', s.orphans, () => toggleFilter('fOrphan')],
    ['Устаревших версий', s.outdated, () => toggleFilter('fOutdated')],
    ['Сканов без OCR', s.scans, () => toggleFilter('fOcr')],
    ['Дубликатов', s.dupes, () => toggleFilter('fDupe')],
    ['Не попадали в ответы', s.never_used, () => toggleFilter('fUnused')],
  ];
  el('stats').innerHTML = rows.map(([k, v], i) =>
    `<div class="stat" onclick="statClick(${i})"><span>${k}</span><b>${RU(v)}</b></div>`).join('');
  window.__statActions = rows.map(r => r[2]);
  el('legendTitle').textContent = {section:'Разделы базы', type:'Типы документов',
    kind:'Виды материалов', state:'Состояние', year:'Годы'}[el('colorBy').value];
  el('legend').innerHTML = legendItems.slice(0, 22).map(item =>
    `<div class="legend" data-k="${esc(item.key)}"><span class="dot" style="background:${item.color}"></span>
     <span style="flex:1">${esc(item.key)}</span><b>${RU(item.count)}</b></div>`).join('');
  if (el('colorBy').value === 'section') {
    document.querySelectorAll('#legend .legend').forEach(node => {
      if (offSections.has(node.dataset.k)) node.classList.add('off');
      node.onclick = () => {
        const k = node.dataset.k;
        if (offSections.has(k)) offSections.delete(k); else offSections.add(k);
        rebuild();
      };
    });
  }
}
function statClick(i) { window.__statActions[i](); }
function toggleFilter(id) { el(id).checked = !el(id).checked; rebuild(); }
function setLevel(i) { level = i; rebuild(); }
function setPath(n) { path = Array.isArray(n) ? n : path.slice(0, n); rebuild(); }

/* --------------------------------------------------------- карточка ------- */
function showDetail(n) {
  const d = el('detail');
  if (!n) { d.style.display = 'none'; return; }
  let rows;
  if (n.isGroup) {
    rows = [['Документов', RU(n.count)], ['Объём текста', RU(Math.round(n.chars / 1000)) + ' тыс. симв.'],
      ['Без связей', RU(n.orphan)], ['Устаревших', RU(n.current)],
      ['Сканов без OCR', RU(n.ocr)], ['Дубликатов', RU(n.dupe)],
      ['Не находились', RU(n.used)], ['Связей с другими', RU(n.degree)]];
  } else {
    rows = [['Путь', esc(n.path)], ['Раздел', esc(n.section)], ['Бренд', esc(n.brand)],
      ['Тип', esc(n.type)], ['Вид', esc(n.kind)], ['Дата', esc(n.date) || '—'],
      ['Символов', RU(n.chars)], ['Состояние', nodeState(n)],
      ['Обогащение', esc(n.enriched) || '—'], ['Связей', RU(n.degree)]];
  }
  d.innerHTML = `<span class="close" onclick="showDetail(null)">×</span>
    <h3>${esc(n.label || n.gid)}</h3><table>` +
    rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('') + '</table>' +
    (n.isGroup ? `<div class="btns" style="margin-top:8px">
      <button class="btn" onclick="drillInto('${esc(n.gid).replace(/'/g,'')}')">Раскрыть</button></div>` : '');
  d.style.display = 'block';
}
function drillInto(gid) {
  const n = nodes.find(x => String(x.gid) === gid);
  if (!n) return;
  if (level === 0) path = [n.section];
  else if (level === 1) path = [n.section, n.brand];
  else if (level === 2) path = [n.section, n.brand, n.cat];
  level = Math.min(level + 1, 3);
  showDetail(null); rebuild();
}

/* ----------------------------------------------------- взаимодействие ----- */
let drag = null;
cv.addEventListener('mousedown', e => { drag = {x: e.clientX, y: e.clientY, ox, oy, moved: false};
  cv.style.cursor = 'grabbing'; });
window.addEventListener('mouseup', () => { drag = null; cv.style.cursor = 'grab'; });
function pick(e) {
  const rect = cv.getBoundingClientRect();
  const cx = cv.width / devicePixelRatio / 2 + ox, cy = cv.height / devicePixelRatio / 2 + oy;
  const mx = (e.clientX - rect.left - cx) / scale, my = (e.clientY - rect.top - cy) / scale;
  let best = null, bestD = 16 / scale;
  for (const n of nodes) {
    const d = Math.hypot(n.x - mx, n.y - my);
    if (d < Math.max(bestD, n.r / Math.max(Math.sqrt(scale), .55) + 3 / scale)) { bestD = d; best = n; }
  }
  return best;
}
cv.addEventListener('mousemove', e => {
  if (drag) { drag.moved = true; ox = drag.ox + (e.clientX - drag.x);
    oy = drag.oy + (e.clientY - drag.y); draw(); return; }
  const best = pick(e);
  const tip = el('tip');
  if (best !== hover) { hover = best; draw(); }
  if (best) {
    const rect = cv.getBoundingClientRect();
    const flags = [];
    if (best.isGroup) {
      if (best.orphan) flags.push(`<span class="bad">без связей: ${best.orphan}</span>`);
      if (best.ocr) flags.push(`<span class="warn">сканов: ${best.ocr}</span>`);
      if (best.current) flags.push(`<span class="muted">устаревших: ${best.current}</span>`);
      tip.innerHTML = `<b>${esc(best.label)}</b><br><span class="muted">${RU(best.count)} документов,
        ${RU(Math.round(best.chars / 1000))} тыс. символов</span>` +
        (flags.length ? '<br>' + flags.join(' · ') : '') +
        '<br><span class="muted">двойной щелчок — раскрыть</span>';
    } else {
      if (best.orphan) flags.push('<span class="bad">нет смысловых связей</span>');
      if (!best.current) flags.push('<span class="muted">устаревшая версия</span>');
      if (best.ocr) flags.push('<span class="warn">скан без распознавания</span>');
      if (best.dupe) flags.push('<span class="warn">дубликат</span>');
      if (!best.used) flags.push('<span class="muted">не попадал в ответы</span>');
      tip.innerHTML = `<b>${esc(best.label)}</b><br><span class="muted">${esc(best.path)}</span><br>
        <span class="chip">${esc(best.section)}</span><span class="chip">${esc(best.brand)}</span>
        <span class="chip">${esc(best.type)}</span><span class="chip">${esc(best.kind)}</span>
        ${best.date ? '<span class="chip">' + esc(best.date) + '</span>' : ''}
        <br>символов: ${RU(best.chars)}` + (flags.length ? '<br>' + flags.join(' · ') : '');
    }
    tip.style.display = 'block';
    tip.style.left = Math.min(e.clientX - rect.left + 16, rect.width - 450) + 'px';
    tip.style.top = (e.clientY - rect.top + 14) + 'px';
  } else tip.style.display = 'none';
});
cv.addEventListener('click', e => { if (drag && drag.moved) return;
  selected = pick(e); showDetail(selected); draw(); });
cv.addEventListener('dblclick', e => {
  const n = pick(e);
  if (n && n.isGroup) drillInto(String(n.gid));
});
cv.addEventListener('wheel', e => { e.preventDefault();
  const f = e.deltaY < 0 ? 1.13 : 1 / 1.13;
  scale = Math.max(0.05, Math.min(20, scale * f)); draw(); }, {passive: false});

/* ------------------------------------------------------ элементы панели --- */
function fillSelect(id, values, label) {
  el(id).innerHTML = `<option value="">${label}</option>` +
    values.map(v => `<option>${esc(v)}</option>`).join('');
}
fillSelect('fType', [...new Set(RAW.map(n => n.type))].filter(Boolean).sort(), 'любой');
fillSelect('fKind', [...new Set(RAW.map(n => n.kind))].filter(Boolean).sort(), 'любой');
fillSelect('fYear', [...new Set(RAW.map(n => n.year))].filter(Boolean).sort().reverse(), 'любой');

const PRESETS = [
  ['Обзор', () => { level = 0; path = []; resetFilters(); el('layout').value = 'cluster'; }],
  ['Проблемы', () => { level = 3; resetFilters(); el('fOrphan').checked = true;
                       el('colorBy').value = 'state'; el('layout').value = 'force'; }],
  ['Устаревшее', () => { level = 3; resetFilters(); el('fOutdated').checked = true;
                         el('colorBy').value = 'section'; }],
  ['Дубликаты', () => { level = 3; resetFilters(); el('fDupe').checked = true; }],
  ['Сканы', () => { level = 3; resetFilters(); el('fOcr').checked = true;
                    el('colorBy').value = 'section'; }],
  ['Не находились', () => { level = 3; resetFilters(); el('fUnused').checked = true; }],
  ['Медиа и чертежи', () => { level = 2; resetFilters(); el('fAsset').checked = true;
                              el('colorBy').value = 'kind'; }],
  ['Покрытие', () => { level = 2; path = []; resetFilters();
                       el('colorBy').value = 'type'; el('sizeBy').value = 'count';
                       el('layout').value = 'cluster'; }],
];
el('presets').innerHTML = PRESETS.map((p, i) =>
  `<button class="btn" onclick="applyPreset(${i})">${p[0]}</button>`).join('');
function applyPreset(i) { PRESETS[i][1](); rebuild(); }
function resetFilters() {
  ['fOrphan','fOutdated','fOcr','fDupe','fUnused','fAsset','fWeb']
    .forEach(id => el(id).checked = false);
  ['fType','fKind','fYear'].forEach(id => el(id).value = '');
  el('q').value = ''; isolateMode = false;
}

['fOrphan','fOutdated','fOcr','fDupe','fUnused','fAsset','fWeb','fType','fKind','fYear',
 'eSem','eFolder','colorBy','sizeBy','layout']
  .forEach(id => el(id).addEventListener('change', rebuild));
el('showLabels').addEventListener('change', draw);
el('dimOthers').addEventListener('change', draw);
el('q').addEventListener('input', () => { isolateMode ? rebuild() : draw(); });
el('isolate').addEventListener('click', () => { isolateMode = !isolateMode;
  el('isolate').classList.toggle('on', isolateMode); rebuild(); });
el('sim').addEventListener('input', () => { el('simV').textContent =
  (+el('sim').value / 100).toFixed(2); rebuild(); });
el('deg').addEventListener('input', () => { el('degV').textContent = el('deg').value; rebuild(); });
el('spread').addEventListener('input', () => { el('spreadV').textContent = el('spread').value; });
el('spread').addEventListener('change', () => { layout(); fitView(); });
el('relayout').addEventListener('click', () => { layout(); fitView(); });
el('fit').addEventListener('click', fitView);
el('png').addEventListener('click', () => {
  const a = document.createElement('a');
  a.download = 'граф-базы-знаний.png'; a.href = cv.toDataURL('image/png'); a.click();
});
el('csv').addEventListener('click', () => {
  const head = level === 3
    ? 'путь;раздел;бренд;тип;вид;дата;символов;состояние\n'
    : 'группа;документов;символов;без связей;устаревших;сканов;дубликатов\n';
  const body = nodes.map(n => level === 3
    ? [n.path, n.section, n.brand, n.type, n.kind, n.date, n.chars, nodeState(n)].join(';')
    : [n.label, n.count, n.chars, n.orphan, n.current, n.ocr, n.dupe].join(';')).join('\n');
  const blob = new Blob(['﻿' + head + body], {type: 'text/csv;charset=utf-8'});
  const a = document.createElement('a');
  a.download = 'граф-базы-знаний.csv'; a.href = URL.createObjectURL(blob); a.click();
});
window.addEventListener('resize', resize);
el('simV').textContent = '0.00'; el('degV').textContent = '6'; el('spreadV').textContent = '110';
resize(); rebuild();
</script></body></html>
"""


def render_html(graph: dict, out_path: Path) -> None:
    payload = json.dumps(graph, ensure_ascii=False, separators=(",", ":"))
    out_path.write_text(HTML_TEMPLATE.replace("__DATA__", payload), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Граф связности базы знаний")
    p.add_argument("command", choices=["build"], nargs="?", default="build")
    p.add_argument("--out", default="graph.html")
    p.add_argument("--similarity", type=float, default=0.5,
                   help="порог смысловой близости для связи")
    p.add_argument("--edges", type=int, default=6, help="сколько связей строить у документа")
    p.add_argument("--limit", type=int)
    p.add_argument("--json")
    a = p.parse_args()
    logging_setup.setup()

    graph = build_graph(a.similarity, a.edges, a.limit)
    render_html(graph, Path(a.out))
    if a.json:
        Path(a.json).write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")
    s = graph["stats"]
    print(f"Документов: {s['documents']}, смысловых связей: {s['edges_semantic']}, "
          f"структурных: {s['edges_folder']}")
    print(f"Без связей: {s['orphans']}, устаревших: {s['outdated']}, "
          f"сканов: {s['scans']}, дубликатов: {s['dupes']}, "
          f"не попадали в ответы: {s['never_used']}")
    print(f"Файл: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
