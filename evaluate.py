"""
Оценка качества на «золотом» наборе вопросов.

  python evaluate.py --dataset eval/golden.jsonl
  python evaluate.py --audit          # проверить сам набор, а не систему

Формат строки датасета:
  {"question": "...",
   "expect_files": ["часть пути или имени файла"],
   "expect_text":  ["подстрока, которая обязана быть в ответе"],
   "reject_files": ["путь, который считается ПОДМЕНОЙ, если стоит выше нужного"],
   "reject_text":  ["подстрока, которой в ответе быть НЕ должно"]}

Метрики:
  hit@k       — доля вопросов, где нужный документ попал в топ-k;
  mrr         — средний обратный ранг первого правильного документа;
  substituted — доля вопросов, где выше нужного документа стоит подмена;
  answered    — доля вопросов, на которые бот вообще ответил;
  contains    — доля ответов, содержащих ожидаемую подстроку;
  clean       — доля ответов без запрещённой подстроки.

Поля reject_* — это «вопросы-двойники»: спрашиваем про модель 60/92 и
запрещаем в ответе цифры соседней 55/75. Именно так ловятся самые
дорогие ошибки — когда система уверенно отвечает про другой товар.
Метрика, в которой таких пар нет, подмену не увидит вовсе.

Без такого набора улучшать RAG вслепую невозможно: любое изменение
чанкинга или весов надо мерить, а не «чувствовать».
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import answer as answer_mod
import config
import db
import search as search_mod


def load(path: Path) -> list[dict]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("//"):
            items.append(json.loads(line))
    return items


def _first_match(paths: list[str], patterns: list[str]) -> int | None:
    """Ранг (с единицы) первого пути, содержащего любой из шаблонов."""
    for i, p in enumerate(paths, 1):
        if any(pat.lower() in p.lower() for pat in patterns):
            return i
    return None


def evaluate(dataset: list[dict], top_k: int = 6, run_llm: bool = False) -> dict:
    hits_at_k = 0
    reciprocal = 0.0
    substituted = 0
    checked_subst = 0
    answered = 0
    contains = 0
    checked_contains = 0
    clean = 0
    checked_clean = 0
    details = []

    for item in dataset:
        question = item["question"]
        found = search_mod.search(question, top_k=top_k)
        paths = [h.rel_path for h in found]
        rank = _first_match(paths, item.get("expect_files", []))
        if rank:
            hits_at_k += 1
            reciprocal += 1 / rank
        row = {"question": question, "rank": rank, "top": paths[:3]}

        # Подмена: документ-двойник стоит выше нужного (или нужного нет
        # вовсе, а двойник есть). Это хуже промаха: промах честный, а
        # подмена выглядит как ответ.
        rejects = item.get("reject_files", [])
        if rejects:
            checked_subst += 1
            wrong = _first_match(paths, rejects)
            if wrong is not None and (rank is None or wrong < rank):
                substituted += 1
                row["substituted_by"] = paths[wrong - 1]

        if run_llm:
            # Замер идёт пакетом: пропускаем вперёд живые вопросы.
            res = answer_mod.ask(question, log=False, source="фон")
            answered += int(res.answered)
            expects = item.get("expect_text", [])
            if expects:
                checked_contains += 1
                if any(e.lower() in res.text.lower() for e in expects):
                    contains += 1
            forbidden = item.get("reject_text", [])
            if forbidden:
                checked_clean += 1
                hit_bad = [e for e in forbidden if e.lower() in res.text.lower()]
                if hit_bad:
                    row["forbidden_in_answer"] = hit_bad
                else:
                    clean += 1
            row["answer"] = res.text[:200]
        details.append(row)

    n = max(len(dataset), 1)
    return {
        "n": len(dataset),
        f"hit@{top_k}": round(hits_at_k / n, 3),
        "mrr": round(reciprocal / n, 3),
        "substituted": round(substituted / checked_subst, 3) if checked_subst else None,
        "answered": round(answered / n, 3) if run_llm else None,
        "contains": round(contains / checked_contains, 3) if checked_contains else None,
        "clean": round(clean / checked_clean, 3) if checked_clean else None,
        "details": details,
    }


def audit(dataset: list[dict]) -> list[str]:
    """
    Проверка самого набора: метрика не лучше своих эталонов.

    Три конструктивных слабости, из-за которых набор может показывать
    зелёные цифры при реальных ошибках:
      — шаблон вроде «4ПАСПОРТ» засчитывает паспорт любого бренда и
        любой модели, то есть не отличает правильный документ от подмены;
      — вопрос без expect_text проверяет только поиск, но не ответ:
        модель может процитировать нужный файл и переврать цифру;
      — наборы без пар-двойников (reject_*) не видят подмену соседней
        моделью — самую дорогую ошибку в прайсовой тематике.
    """
    problems: list[str] = []
    all_paths = [r["rel_path"] for r in
                 db.q("SELECT rel_path FROM documents WHERE status='ok'")]
    total = max(len(all_paths), 1)

    seen: dict[str, int] = {}
    without_text = 0
    with_reject = 0
    for i, item in enumerate(dataset, 1):
        q = item.get("question", "").strip().lower()
        if q in seen:
            problems.append(f"строка {i}: вопрос дублирует строку {seen[q]}")
        seen[q] = i
        if not item.get("expect_files"):
            problems.append(f"строка {i}: нет expect_files — вопрос ничего не проверяет")
        if item.get("expect_text") or item.get("reject_text"):
            pass
        else:
            without_text += 1
        if item.get("reject_files") or item.get("reject_text"):
            with_reject += 1
        for pat in item.get("expect_files", []):
            matched = sum(1 for p in all_paths if pat.lower() in p.lower())
            if matched == 0 and all_paths:
                problems.append(f"строка {i}: «{pat}» не совпадает ни с одним "
                                f"проиндексированным файлом — вопрос всегда промах")
            elif matched > max(20, total // 10):
                problems.append(
                    f"строка {i}: «{pat}» совпадает с {matched} файлами из {total} — "
                    f"шаблон слишком общий, засчитает и подмену; укажите бренд "
                    f"или модель в пути")
    if without_text:
        problems.append(
            f"{without_text} из {len(dataset)} вопросов без expect_text/reject_text — "
            f"для них проверяется только поиск, но не правильность ответа")
    if not with_reject:
        problems.append(
            "в наборе нет ни одной пары-двойника (reject_files/reject_text) — "
            "подмена соседней моделью останется незамеченной; добавьте вопросы "
            "вида «характеристика модели X» с запретом цифр соседней модели Y")
    return problems


def compare(dataset: list[dict], top_k: int, progress=None) -> dict:
    """
    Сравнивает настройки поиска на одном и том же наборе вопросов.

    Это единственный честный способ решить, включать ли переранжирование
    и с каким весом: на глаз такая разница не видна, а на контрольном
    наборе видна сразу.

    Векторы здесь не пересчитываются — сравниваются только настройки
    выдачи. Чтобы сравнить модели эмбеддингов между собой, выполните
    между прогонами `python index.py reembed --provider ...`.
    """
    import rerank as rerank_mod
    say = progress or (lambda t: print(t, flush=True))
    rows: list[dict] = []
    variants = [("без переранжирования", "none", 0.0)]
    providers = ["lexical"]
    if config.RERANKER_PROVIDER not in ("none", "lexical"):
        providers.append(config.RERANKER_PROVIDER)
    for provider in providers:
        for weight in (0.5, 0.8, 1.0):
            variants.append((f"{provider}, вес {weight}", provider, weight))

    saved = (config.RERANKER_PROVIDER, config.RERANKER_WEIGHT)
    say(f"Вопросов в наборе: {len(dataset)}   "
        f"смысловой канал: {config.EMBEDDINGS_PROVIDER}")
    say(f"{'настройка':30} {('hit@' + str(top_k)):>8} {'MRR':>8}")
    try:
        for label, provider, weight in variants:
            config.RERANKER_PROVIDER, config.RERANKER_WEIGHT = provider, weight
            rerank_mod.reset()
            if provider != "none" and rerank_mod.get() is None:
                say(f"{label:30} {'—':>8} {'недоступен':>8}")
                rows.append({"label": label, "provider": provider, "weight": weight,
                             "available": False})
                continue
            result = evaluate(dataset, top_k, run_llm=False)
            hit = result["hit@" + str(top_k)]
            say(f"{label:30} {hit:>8} {result['mrr']:>8}")
            rows.append({"label": label, "provider": provider, "weight": weight,
                         "available": True, "hit": hit, "mrr": result["mrr"]})
    finally:
        config.RERANKER_PROVIDER, config.RERANKER_WEIGHT = saved
        rerank_mod.reset()
    best = max((r for r in rows if r.get("available")),
               key=lambda r: r["mrr"], default=None)
    if best:
        say(f"Лучшая настройка по MRR: {best['label']} ({best['mrr']})")
    say("MRR учитывает не только «нашлось ли», но и на каком месте оказалось.")
    return {"questions": len(dataset), "top_k": top_k,
            "embeddings": config.EMBEDDINGS_PROVIDER, "variants": rows,
            "best": best}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="eval/golden.jsonl")
    p.add_argument("--top-k", type=int, default=config.SEARCH_TOP_K)
    p.add_argument("--llm", action="store_true", help="прогнать и генерацию тоже")
    p.add_argument("--compare", action="store_true",
                   help="сравнить настройки переранжирования между собой")
    p.add_argument("--audit", action="store_true",
                   help="проверить качество самого набора вопросов")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    path = Path(args.dataset)
    if not path.exists():
        print(f"Нет файла {path}. Создайте набор из реальных вопросов сотрудников.")
        return 1
    db.init()
    data = load(path)
    if args.audit:
        problems = audit(data)
        if not problems:
            print(f"Набор из {len(data)} вопросов пригоден для замера.")
            return 0
        print(f"Слабости набора ({len(problems)}) — метрика не лучше своих эталонов:")
        for msg in problems:
            print(f"  · {msg}")
        return 1
    if args.compare:
        compare(data, args.top_k)
        return 0
    result = evaluate(data, args.top_k, args.llm)

    print(f"Вопросов: {result['n']}")
    print(f"hit@{args.top_k}: {result[f'hit@{args.top_k}']}")
    print(f"MRR      : {result['mrr']}")
    if result["substituted"] is not None:
        print(f"Подменено документом-двойником: {result['substituted']}")
    if result["answered"] is not None:
        print(f"Отвечено : {result['answered']}")
    if result["contains"] is not None:
        print(f"Содержит ожидаемое: {result['contains']}")
    if result["clean"] is not None:
        print(f"Без запрещённого: {result['clean']}")

    misses = [d for d in result["details"] if d["rank"] is None]
    if misses:
        print(f"\nПромахи ({len(misses)}):")
        for d in misses[:20]:
            print(f"  ✗ {d['question']}")
            for t in d["top"]:
                print(f"      нашлось: {t}")
    swaps = [d for d in result["details"]
             if d.get("substituted_by") or d.get("forbidden_in_answer")]
    if swaps:
        print(f"\nПодмены ({len(swaps)}) — уверенные ответы не о том:")
        for d in swaps[:20]:
            print(f"  ✗ {d['question']}")
            if d.get("substituted_by"):
                print(f"      выше нужного стоит: {d['substituted_by']}")
            if d.get("forbidden_in_answer"):
                print(f"      в ответе запрещённое: {', '.join(d['forbidden_in_answer'])}")
    if args.verbose:
        print(json.dumps(result["details"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
