"""
Аналитика работы ассистента.

Отличие от метрик: метрики отвечают на вопрос «что происходит», аналитика —
«что с этим делать». Здесь четыре инструмента, каждый отвечает на свой
управленческий вопрос.

  Воронка ответа            где именно теряются ответы: в поиске, на пороге
                            уверенности или в генерации
  Гистограмма уверенности   какой порог отказа поставить, чтобы бот не
                            выдумывал, но и не молчал на реальных вопросах
  Вклад каналов             нужен ли смысловой поиск и не пора ли перейти
                            на модель посильнее
  Группы вопросов без       чего не хватает в базе знаний — готовый список
  ответа                    задач владельцам контента

  python analytics.py funnel
  python analytics.py confidence
  python analytics.py channels
  python analytics.py gaps
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone

import config
import db


def _since(hours: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat(timespec="seconds")


# ─────────────────────────────────────────────────────── воронка ответа ────
def funnel(hours: int = 168) -> dict:
    """
    Путь вопроса от «задан» до «оценён».

    Читать её нужно сверху вниз, глядя на потери между ступенями.
    Потеря на «что-то нашлось» означает пробел в базе или в поиске.
    Потеря на «уверенность выше порога» — либо порог слишком высок,
    либо поиск приносит не то. Потеря на «модель ответила» — сбои
    провайдера. Отсутствие оценок означает, что качество никто не
    подтверждает, и все выводы о нём — догадки.
    """
    since = _since(hours)
    total = db.q1("SELECT COUNT(*) n FROM queries WHERE created_at > ?", (since,))["n"]
    if not total:
        return {"hours": hours, "total": 0, "steps": [], "routes": [], "losses": []}

    found = db.q1("""SELECT COUNT(*) n FROM queries WHERE created_at > ?
                     AND stage IS NOT NULL AND stage <> 'nothing_found'""", (since,))["n"]
    confident = db.q1("""SELECT COUNT(*) n FROM queries WHERE created_at > ?
                         AND stage = 'answered'""", (since,))["n"]
    answered = db.q1("SELECT COUNT(*) n FROM queries WHERE created_at > ? AND answered=1",
                     (since,))["n"]
    rated = db.q1("""SELECT COUNT(DISTINCT q.id) n FROM queries q
                     JOIN feedback f ON f.query_id = q.id
                     WHERE q.created_at > ?""", (since,))["n"]
    good = db.q1("""SELECT COUNT(DISTINCT q.id) n FROM queries q
                    JOIN feedback f ON f.query_id = q.id
                    WHERE q.created_at > ? AND f.verdict='up'""", (since,))["n"]

    # Старые записи не знают про stage — не приписываем им того, чего не знаем.
    legacy = db.q1("SELECT COUNT(*) n FROM queries WHERE created_at > ? AND stage IS NULL",
                   (since,))["n"]

    steps = [
        {"key": "asked", "title": "Задан вопрос", "n": total,
         "note": "все обращения к ассистенту"},
        {"key": "found", "title": "Что-то нашлось", "n": found,
         "note": "поиск вернул хотя бы один фрагмент или позицию прайса"},
        {"key": "confident", "title": "Уверенность выше порога", "n": confident,
         "note": f"оценка лучшего фрагмента больше MIN_CONFIDENCE = {config.MIN_CONFIDENCE}"},
        {"key": "answered", "title": "Ассистент ответил", "n": answered,
         "note": "модель сформулировала ответ по найденному"},
        {"key": "rated", "title": "Ответ оценён", "n": rated,
         "note": "сотрудник нажал «полезно» или «неверно»"},
        {"key": "good", "title": "Оценён положительно", "n": good,
         "note": "из оценённых"},
    ]
    for i, step in enumerate(steps):
        base = steps[i - 1]["n"] if i else total
        step["share"] = round(step["n"] / total, 3) if total else 0.0
        step["kept"] = round(step["n"] / base, 3) if base else 0.0
        step["lost"] = base - step["n"] if i else 0

    losses = []
    if total and found < total:
        losses.append({"where": "поиск ничего не нашёл", "n": total - found,
                       "what_to_do": "смотрите раздел «Пробелы»: скорее всего, "
                                     "этих документов нет в базе либо они не "
                                     "распознаны"})
    if found and confident < found:
        losses.append({"where": "не хватило уверенности", "n": found - confident,
                       "what_to_do": "проверьте порог по гистограмме уверенности: "
                                     "возможно, MIN_CONFIDENCE завышен"})
    if confident and answered < confident:
        losses.append({"where": "модель не ответила", "n": confident - answered,
                       "what_to_do": "сбои провайдера генерации — смотрите журнал llm"})
    if answered and rated < answered * 0.05:
        losses.append({"where": "почти никто не оценивает ответы", "n": answered - rated,
                       "what_to_do": "без оценок качество нечем подтвердить; "
                                     "напомните сотрудникам про кнопки под ответом"})

    routes = [dict(r) for r in db.q("""
        SELECT COALESCE(route,'—') route, COUNT(*) n,
               SUM(CASE WHEN answered=1 THEN 1 ELSE 0 END) ok,
               AVG(latency_ms) ms
        FROM queries WHERE created_at > ? GROUP BY route ORDER BY n DESC""", (since,))]
    for r in routes:
        r["ms"] = int(r["ms"] or 0)
        r["share"] = round(r["n"] / total, 3)

    return {"hours": hours, "total": total, "steps": steps, "losses": losses,
            "routes": routes, "legacy": legacy}


ROUTE_RU = {"golden": "выверенный ответ", "price": "прайс-лист",
            "documents": "документы базы", "none": "ничего не найдено", "—": "не указан"}


# ──────────────────────────────────────────── гистограмма уверенности ────
def confidence_histogram(hours: int = 720, bins: int = 24) -> dict:
    """
    Распределение оценок лучшего фрагмента, с отметкой текущего порога.

    Это прямой инструмент подбора MIN_CONFIDENCE. На гистограмме почти
    всегда видны два скопления: слева — случайные совпадения, справа —
    настоящие находки. Порог ставится во впадину между ними. Сейчас его
    подбирают вслепую, и любое значение выглядит одинаково правдоподобно.

    Отдельно считается, что было бы при других порогах: сколько вопросов
    отсеклось бы и какая доля из них на самом деле получила хорошую оценку.
    Отсечь ответ, который сотрудник счёл полезным, — самая дорогая ошибка
    порога, потому что заметить её невозможно.
    """
    since = _since(hours)
    rows = db.q("""SELECT q.top_score s, q.answered,
                          (SELECT verdict FROM feedback f WHERE f.query_id=q.id
                           ORDER BY f.id DESC LIMIT 1) verdict
                   FROM queries q
                   WHERE q.created_at > ? AND q.top_score IS NOT NULL AND q.top_score > 0""",
                (since,))
    scores = [(r["s"], r["verdict"]) for r in rows]
    if not scores:
        return {"hours": hours, "total": 0, "bins": [], "threshold": config.MIN_CONFIDENCE,
                "suggestions": []}

    values = sorted(s for s, _ in scores)
    lo, hi = values[0], values[-1]
    if hi <= lo:
        hi = lo + 1e-6
    width = (hi - lo) / bins
    buckets = []
    for i in range(bins):
        a, b = lo + i * width, lo + (i + 1) * width
        inside = [(s, v) for s, v in scores if (a <= s < b or (i == bins - 1 and s == b))]
        buckets.append({
            "from": round(a, 5), "to": round(b, 5), "n": len(inside),
            "up": sum(1 for _, v in inside if v == "up"),
            "down": sum(1 for _, v in inside if v == "down"),
        })

    # Что было бы при другом пороге.
    rated = [(s, v) for s, v in scores if v in ("up", "down")]
    suggestions = []
    seen: set[float] = set()
    for q in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
        threshold = round(values[min(int(len(values) * q), len(values) - 1)], 5)
        if threshold in seen:
            continue          # на плотных распределениях соседние квантили совпадают
        seen.add(threshold)
        cuts = sum(1 for s, _ in scores if s < threshold)
        suggestions.append({
            "threshold": threshold,
            "cuts": cuts,
            "cuts_share": round(cuts / len(scores), 3),
            "cuts_good": sum(1 for s, v in rated if s < threshold and v == "up"),
            "cuts_bad": sum(1 for s, v in rated if s < threshold and v == "down"),
        })
    # Значение, которое отсекает больше всего плохих ответов и меньше всего хороших.
    scored = [s for s in suggestions if s["cuts_bad"] or s["cuts_good"]]
    best = max(scored, key=lambda s: s["cuts_bad"] - 3 * s["cuts_good"], default=None)

    current = config.MIN_CONFIDENCE
    return {
        "hours": hours, "total": len(scores), "bins": buckets,
        "threshold": current,
        "below_threshold": sum(1 for s, _ in scores if s < current),
        "rated": len(rated),
        "good_below": sum(1 for s, v in rated if s < current and v == "up"),
        "bad_above": sum(1 for s, v in rated if s >= current and v == "down"),
        "suggestions": suggestions,
        "recommended": best,
        "min": round(lo, 5), "max": round(hi, 5),
        "median": round(values[len(values) // 2], 5),
    }


# ───────────────────────────────────────────────────── вклад каналов ────
def channel_report(hours: int = 720) -> dict:
    """
    Кто на самом деле находит ответы.

    Если почти всё приносит текстовый канал, смысловая модель не окупает
    себя и переходить на более тяжёлую бессмысленно. Если заметная доля
    ответов приходит только из смыслового канала — наоборот, более сильная
    модель даст ещё прирост. Эти два вывода противоположны, и без цифр
    выбирают наугад.
    """
    since = _since(hours)
    rows = db.q("""SELECT COALESCE(channels,'—') ch, COUNT(*) n,
                          AVG(top_score) score,
                          SUM(CASE WHEN answered=1 THEN 1 ELSE 0 END) ok
                   FROM queries WHERE created_at > ? AND stage IS NOT NULL
                     AND stage <> 'nothing_found'
                   GROUP BY channels ORDER BY n DESC""", (since,))
    total = sum(r["n"] for r in rows) or 1
    names = {"bm25": "только текстовый", "dense": "только смысловой",
             "bm25+dense": "оба канала", "golden": "выверенный ответ",
             "прочее": "прочее", "—": "не указан", "": "не указан"}
    channels = [{"key": r["ch"], "title": names.get(r["ch"], r["ch"]),
                 "n": r["n"], "share": round(r["n"] / total, 3),
                 "avg_score": round(r["score"] or 0, 5),
                 "answered": r["ok"]} for r in rows]

    only_dense = sum(r["n"] for r in rows if r["ch"] == "dense")
    only_bm25 = sum(r["n"] for r in rows if r["ch"] == "bm25")
    both = sum(r["n"] for r in rows if r["ch"] == "bm25+dense")

    if only_dense / total > 0.15:
        verdict = ("Смысловой канал приносит заметную долю ответов в одиночку. "
                   "Более сильная модель (USER-bge-m3 через ONNX) даст ещё прирост.")
    elif only_dense / total < 0.03 and total > 50:
        verdict = ("Смысловой канал почти ничего не добавляет сверх текстового. "
                   "Либо вопросы задают словами документов, либо модель не обучена "
                   "на актуальном содержимом — проверьте раздел «Качество поиска».")
    else:
        verdict = ("Каналы дополняют друг друга — обычная и здоровая картина "
                   "для гибридного поиска.")

    rerank = db.q1("""SELECT SUM(rerank_used) used, COUNT(*) n FROM queries
                      WHERE created_at > ? AND stage IS NOT NULL
                        AND stage <> 'nothing_found'""", (since,))
    return {"hours": hours, "total": total, "channels": channels,
            "only_dense": only_dense, "only_bm25": only_bm25, "both": both,
            "rerank_used": rerank["used"] or 0, "rerank_of": rerank["n"] or 0,
            "verdict": verdict}


# ──────────────────────────────────── группировка вопросов без ответа ────
_STOP = {
    "как", "что", "где", "для", "или", "это", "при", "под", "над", "чем", "кто",
    "если", "так", "уже", "его", "нам", "мне", "она", "они", "какой", "какая",
    "какие", "нужно", "можно", "есть", "ли", "не", "на", "в", "из", "по", "до",
    "от", "с", "со", "к", "у", "о", "об", "за", "же", "бы", "то", "мы", "вы",
    "быть", "и", "а", "но", "да", "нет", "все", "вот", "там", "тут",
    "почему", "зачем", "когда", "сколько", "какому", "каком", "мочь", "хотеть",
}


def _keywords(text: str) -> list[str]:
    import lsa
    out = []
    for raw in re.findall(r"[а-яёa-z0-9][а-яёa-z0-9\-./]{2,}", text.lower()):
        raw = raw.strip("-./")
        if len(raw) < 3 or raw in _STOP:
            continue
        out.append(lsa.normalize_token(raw))
    return out


def gaps(hours: int = 720, min_group: int = 2, limit: int = 25) -> dict:
    """
    Группирует вопросы, оставшиеся без ответа, по темам.

    Список отдельных вопросов почти бесполезен: в нём двести строк, и
    непонятно, за что браться. Сгруппированный — это план работ:
    «двадцать три вопроса про подбор частотного преобразователя» уже
    говорит, какой документ нужно добавить в базу.

    Группировка простая и объяснимая: вопросы связываются по общим
    значимым словам, вес слова тем выше, чем оно реже. Никакой модели
    здесь не нужно — тем в базе десятки, а не тысячи, и прозрачность
    в этом месте важнее точности.
    """
    since = _since(hours)
    rows = db.q("""SELECT id, question, created_at, user_name, top_score, stage
                   FROM queries
                   WHERE created_at > ? AND (answered=0 OR stage IN
                         ('nothing_found','low_confidence'))
                   ORDER BY id DESC LIMIT 2000""", (since,))
    # Плохо оценённые ответы — тоже пробел, просто другого рода.
    bad = db.q("""SELECT q.id, q.question, q.created_at, q.user_name, q.top_score,
                         'bad_feedback' stage
                  FROM queries q JOIN feedback f ON f.query_id=q.id
                  WHERE q.created_at > ? AND f.verdict='down' LIMIT 500""", (since,))
    items = [dict(r) for r in rows] + [dict(r) for r in bad]
    if not items:
        return {"hours": hours, "total": 0, "groups": [], "singles": []}

    docs = [(it, set(_keywords(it["question"]))) for it in items]
    docs = [(it, kw) for it, kw in docs if kw]

    # Редкость слова: чем в меньшем числе вопросов встретилось, тем весомее.
    df: Counter = Counter()
    for _it, kw in docs:
        df.update(kw)
    n_docs = len(docs) or 1

    def weight(term: str) -> float:
        return math.log(n_docs / (1 + df[term])) + 1.0

    def similarity(a: set[str], b: set[str]) -> float:
        """
        Мера близости двух вопросов: доля общего веса от меньшего из них.

        Не Жаккар. Вопросы сотрудников бывают и в три слова, и в
        пятнадцать; Жаккар наказывает за длину и разводит по разным
        группам «частотник для насоса» и «как подобрать частотный
        преобразователь по мощности двигателя», хотя это одна тема.
        Здесь считается, насколько короткий вопрос покрыт длинным.
        """
        shared = a & b
        if not shared:
            return 0.0
        num = sum(weight(t) for t in shared)
        den = min(sum(weight(t) for t in a), sum(weight(t) for t in b))
        return num / den if den else 0.0

    # Жадная кластеризация: берём самый «тяжёлый» ещё не разобранный вопрос
    # как центр группы и притягиваем к нему похожие.
    NEAR, ATTACH = 0.45, 0.34
    used: set[int] = set()
    clusters: list[list[int]] = []
    order = sorted(range(len(docs)), key=lambda i: -sum(weight(t) for t in docs[i][1]))
    for i in order:
        if i in used:
            continue
        members = [i]
        for j in range(len(docs)):
            if j != i and j not in used and similarity(docs[i][1], docs[j][1]) >= NEAR:
                members.append(j)
        if len(members) >= min_group:
            used.update(members)
            clusters.append(members)

    # Слияние: две группы про одно и то же (жадный проход мог развести
    # «частотник для насоса» и «подбор частотного преобразователя»
    # по разным центрам) объединяются, если хоть одна пара вопросов
    # из разных групп достаточно похожа.
    merged = True
    while merged:
        merged = False
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                if any(similarity(docs[x][1], docs[y][1]) >= NEAR
                       for x in clusters[a] for y in clusters[b]):
                    clusters[a].extend(clusters[b])
                    del clusters[b]
                    merged = True
                    break
            if merged:
                break

    # Третий проход: одиночные вопросы, похожие на уже собранную группу,
    # присоединяются к ней. Без него тема рассыпается на «ядро» и хвост
    # из формулировок, которые к центру не подошли, а к соседям подошли бы.
    for i in range(len(docs)):
        if i in used:
            continue
        best, best_score = None, ATTACH
        for cluster in clusters:
            score = max(similarity(docs[i][1], docs[m][1]) for m in cluster)
            if score >= best_score:
                best, best_score = cluster, score
        if best is not None:
            best.append(i)
            used.add(i)

    groups = []
    for members in clusters:
        member_items = [docs[m][0] for m in members]
        terms: Counter = Counter()
        for m in members:
            for t in docs[m][1]:
                terms[t] += weight(t)
        groups.append({
            "size": len(members),
            "terms": [t for t, _w in terms.most_common(6)],
            "title": ", ".join(t for t, _w in terms.most_common(4)),
            "questions": [{"id": it["id"], "text": it["question"][:200],
                           "at": it["created_at"], "who": it["user_name"],
                           "stage": it["stage"], "score": it["top_score"]}
                          for it in member_items[:12]],
            "stages": dict(Counter(it["stage"] for it in member_items)),
            "last_at": max((it["created_at"] or "") for it in member_items),
        })
    groups.sort(key=lambda g: -g["size"])

    singles = [docs[i][0] for i in range(len(docs)) if i not in used]
    return {
        "hours": hours, "total": len(items),
        "groups": groups[:limit],
        "grouped": sum(g["size"] for g in groups),
        "singles": [{"id": it["id"], "text": it["question"][:200],
                     "at": it["created_at"], "stage": it["stage"]}
                    for it in singles[:40]],
        "singles_total": len(singles),
    }


STAGE_RU = {"nothing_found": "ничего не нашлось", "low_confidence": "низкая уверенность",
            "bad_feedback": "оценён как неверный", "answered": "отвечено"}


# ────────────────────────────────────────────────────────────── CLI ────
def main() -> int:
    p = argparse.ArgumentParser(description="Аналитика работы ассистента")
    p.add_argument("command", choices=["funnel", "confidence", "channels", "gaps"])
    p.add_argument("--hours", type=int, default=720)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    db.init()

    if args.command == "funnel":
        data = funnel(args.hours)
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        if not data["total"]:
            print("Вопросов за период не было.")
            return 0
        print(f"Воронка ответа за {args.hours} ч\n")
        for step in data["steps"]:
            bar = "█" * int(step["share"] * 40)
            print(f"  {step['title']:26} {step['n']:>6}  {step['share']:>6.0%} {bar}")
        if data["losses"]:
            print("\nГде теряются ответы:")
            for loss in data["losses"]:
                print(f"  • {loss['where']}: {loss['n']}")
                print(f"    {loss['what_to_do']}")
        print("\nПо маршрутам:")
        for r in data["routes"]:
            print(f"  {ROUTE_RU.get(r['route'], r['route']):22} {r['n']:>6} "
                  f"({r['share']:.0%}), в среднем {r['ms']} мс")

    elif args.command == "confidence":
        data = confidence_histogram(args.hours)
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        if not data["total"]:
            print("Оценок уверенности за период нет.")
            return 0
        peak = max(b["n"] for b in data["bins"]) or 1
        print(f"Уверенность по {data['total']} вопросам, порог {data['threshold']}\n")
        for b in data["bins"]:
            mark = " ← порог" if b["from"] <= data["threshold"] < b["to"] else ""
            print(f"  {b['from']:.4f}–{b['to']:.4f} {b['n']:>5} "
                  f"{'█' * int(b['n'] / peak * 34)}{mark}")
        print(f"\nНиже порога: {data['below_threshold']} вопросов")
        if data["rated"]:
            print(f"Из них были оценены положительно: {data['good_below']} — "
                  f"это ответы, которые порог отсёк напрасно")
        print("\nЧто было бы при другом пороге:")
        for s in data["suggestions"]:
            print(f"  {s['threshold']:.5f}: отсекает {s['cuts']} ({s['cuts_share']:.0%}), "
                  f"из них хороших {s['cuts_good']}, плохих {s['cuts_bad']}")

    elif args.command == "channels":
        data = channel_report(args.hours)
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        print(f"Вклад каналов за {args.hours} ч (всего {data['total']} вопросов)\n")
        for c in data["channels"]:
            print(f"  {c['title']:22} {c['n']:>6} ({c['share']:.0%})  "
                  f"средняя оценка {c['avg_score']}")
        print(f"\n{data['verdict']}")

    elif args.command == "gaps":
        data = gaps(args.hours)
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        if not data["total"]:
            print("Вопросов без ответа за период нет.")
            return 0
        print(f"Вопросов без ответа: {data['total']}, "
              f"сгруппировано: {data['grouped']}\n")
        for g in data["groups"]:
            print(f"  ● {g['size']:>3} вопросов — {g['title']}")
            for q in g["questions"][:3]:
                print(f"        {q['text'][:90]}")
        if data["singles_total"]:
            print(f"\nОдиночных вопросов, не вошедших в группы: {data['singles_total']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
