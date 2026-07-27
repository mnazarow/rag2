"""
CLI для проверки качества поиска и ответов без Telegram.

  python ask.py "какой напор у Водомет 55/75"
  python ask.py --debug "цена насоса ВИНТОВИК 3"
  python ask.py --role engineer "как оформить заказ клиента в 1С"
"""
from __future__ import annotations

import argparse
import sys

import answer as answer_mod
import config
import db


def main() -> int:
    p = argparse.ArgumentParser(description="Задать вопрос базе знаний")
    p.add_argument("question", nargs="+")
    p.add_argument("--role", default=config.DEFAULT_ROLE)
    p.add_argument("--debug", action="store_true", help="показать каналы поиска и скоры")
    p.add_argument("--no-log", action="store_true")
    args = p.parse_args()

    question = " ".join(args.question)
    db.init()
    res = answer_mod.ask(question, role=args.role, log=not args.no_log)

    print("=" * 78)
    print(f"ВОПРОС: {question}")
    print("=" * 78)
    print(res.text)
    print()

    if res.products:
        print("— Позиции прайса —")
        for prod in res.products[:8]:
            price = f"{prod['price']:.2f} ₽" if prod.get("price") else "—"
            print(f"  {prod.get('article',''):<12} {str(prod.get('name',''))[:52]:<54} {price}")
        print()

    if res.hits:
        print("— Источники —")
        for i, h in enumerate(res.hits, 1):
            flag = "" if h.is_current else "  ⚠ УСТАРЕВШАЯ ВЕРСИЯ"
            print(f"  [{i}] {h.rel_path}{flag}")
            if args.debug:
                print(f"      score={h.score:.5f} {h.channels}")
                print(f"      {h.text[:180].replace(chr(10),' ')}…")
    print(f"\n(уверенность {res.confidence:.5f}, {res.latency_ms} мс, "
          f"модель: {res.llm_model or config.LLM_PROVIDER}"
          f"{', выверенный ответ' if res.used_golden else ''})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
