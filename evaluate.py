"""
Оценка качества на «золотом» наборе вопросов.

  python evaluate.py --dataset eval/golden.jsonl

Формат строки датасета:
  {"question": "...", "expect_files": ["часть пути или имени файла"],
   "expect_text": ["подстрока, которая обязана быть в ответе"]}

Метрики:
  hit@k      — доля вопросов, где нужный документ попал в топ-k;
  mrr        — средний обратный ранг первого правильного документа;
  answered   — доля вопросов, на которые бот вообще ответил;
  contains   — доля ответов, содержащих ожидаемую подстроку.

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


def evaluate(dataset: list[dict], top_k: int = 6, run_llm: bool = False) -> dict:
    hits_at_k = 0
    reciprocal = 0.0
    answered = 0
    contains = 0
    checked_contains = 0
    details = []

    for item in dataset:
        question = item["question"]
        found = search_mod.search(question, top_k=top_k)
        paths = [h.rel_path for h in found]
        rank = None
        for i, p in enumerate(paths, 1):
            if any(exp.lower() in p.lower() for exp in item.get("expect_files", [])):
                rank = i
                break
        if rank:
            hits_at_k += 1
            reciprocal += 1 / rank
        row = {"question": question, "rank": rank, "top": paths[:3]}

        if run_llm:
            # Замер идёт пакетом: пропускаем вперёд живые вопросы.
            res = answer_mod.ask(question, log=False, source="фон")
            answered += int(res.answered)
            expects = item.get("expect_text", [])
            if expects:
                checked_contains += 1
                if any(e.lower() in res.text.lower() for e in expects):
                    contains += 1
            row["answer"] = res.text[:200]
        details.append(row)

    n = max(len(dataset), 1)
    return {
        "n": len(dataset),
        f"hit@{top_k}": round(hits_at_k / n, 3),
        "mrr": round(reciprocal / n, 3),
        "answered": round(answered / n, 3) if run_llm else None,
        "contains": round(contains / checked_contains, 3) if checked_contains else None,
        "details": details,
    }


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
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    path = Path(args.dataset)
    if not path.exists():
        print(f"Нет файла {path}. Создайте набор из реальных вопросов сотрудников.")
        return 1
    db.init()
    data = load(path)
    if args.compare:
        compare(data, args.top_k)
        return 0
    result = evaluate(data, args.top_k, args.llm)

    print(f"Вопросов: {result['n']}")
    print(f"hit@{args.top_k}: {result[f'hit@{args.top_k}']}")
    print(f"MRR      : {result['mrr']}")
    if result["answered"] is not None:
        print(f"Отвечено : {result['answered']}")
    if result["contains"] is not None:
        print(f"Содержит ожидаемое: {result['contains']}")

    misses = [d for d in result["details"] if d["rank"] is None]
    if misses:
        print(f"\nПромахи ({len(misses)}):")
        for d in misses[:20]:
            print(f"  ✗ {d['question']}")
            for t in d["top"]:
                print(f"      нашлось: {t}")
    if args.verbose:
        print(json.dumps(result["details"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
