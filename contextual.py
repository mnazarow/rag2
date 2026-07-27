"""
Контекстная приставка, сгенерированная моделью.

Что это. К каждому фрагменту дописывается одна-две фразы о том, из
какого документа он и о чём речь, — чтобы фрагмент можно было понять
в отрыве от документа. Без этого кусок вида «допускается не более
180 г/м³» не находится по запросу «сколько песка выдержит насос»:
в самом фрагменте нет ни слова «песок», ни «насос».

Дешёвый детерминированный вариант работает всегда: приставка
собирается из пути к файлу, бренда, типа документа и заголовка раздела.
Он бесплатен и закрывает большую часть эффекта. Здесь — полный вариант,
где приставку пишет модель, читая фрагмент вместе с началом документа.
По опубликованным измерениям это снижает долю неудачных поисков
примерно на треть само по себе.

Цена вопроса честно: одно обращение к модели на каждый фрагмент. На
базе в сорок тысяч фрагментов это заметные разовые расходы и часы
машинного времени. Поэтому здесь есть три вещи, без которых так делать
нельзя: оценка стоимости до запуска, кэш по содержимому (повторная
индексация не платит дважды) и возможность обработать только часть базы.

  python contextual.py estimate          — сколько это будет стоить
  python contextual.py run --limit 500   — проба на пятистах фрагментах
  python contextual.py run               — весь остаток
  python contextual.py clear             — убрать сгенерированные приставки
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from datetime import datetime, timezone

import config
import db
import logging_setup

log = logging_setup.get("index")

PROMPT = (
    "Ниже — начало документа и один его фрагмент.\n"
    "Напиши одну короткую фразу (до 25 слов), которая поможет найти этот "
    "фрагмент поиском: о каком изделии, документе и параметре идёт речь.\n"
    "Требования: только по-русски; называй вещи так, как их называет "
    "сотрудник, а не так, как написано в документе (например «расход воды», "
    "если в тексте «подача, м³/ч»); не пересказывай сам фрагмент; "
    "не выдумывай того, чего нет.\n"
    "В ответе — только эта фраза, без пояснений.\n\n"
    "=== НАЧАЛО ДОКУМЕНТА ===\n{head}\n\n"
    "=== ФРАГМЕНТ ===\n{chunk}\n"
)

MARK = "[смысл] "        # чем помечена сгенерированная часть приставки


def ensure_tables() -> None:
    db.connect().executescript("""
    CREATE TABLE IF NOT EXISTS context_cache (
        text_hash  TEXT PRIMARY KEY,
        model      TEXT,
        context    TEXT,
        created_at TEXT
    );
    """)
    db.connect().commit()


def _hash(text: str, head: str) -> str:
    return hashlib.sha256((head[:600] + "||" + text).encode("utf-8")).hexdigest()


def pending(limit: int | None = None) -> list:
    """Фрагменты, у которых ещё нет сгенерированной приставки."""
    sql = """SELECT c.id, c.text, c.context, c.doc_id, d.file_name
             FROM chunks c JOIN documents d ON d.id = c.doc_id
             WHERE d.status='ok' AND (c.context IS NULL OR c.context NOT LIKE ?)
             ORDER BY c.id"""
    params: tuple = (f"%{MARK}%",)
    if limit:
        sql += " LIMIT ?"
        params = (f"%{MARK}%", limit)
    return db.q(sql, params)


def estimate() -> dict:
    """
    Сколько это будет стоить и сколько займёт.

    Считается по фактическим длинам фрагментов вашей базы, а не по
    среднему по больнице: разброс между каталогом и паспортом большой.
    """
    db.init()
    ensure_tables()
    rows = db.q("""SELECT COUNT(*) n, SUM(LENGTH(text)) chars FROM chunks c
                   JOIN documents d ON d.id=c.doc_id WHERE d.status='ok'""")[0]
    todo = db.q1("""SELECT COUNT(*) n FROM chunks c JOIN documents d ON d.id=c.doc_id
                    WHERE d.status='ok' AND (c.context IS NULL OR c.context NOT LIKE ?)""",
                 (f"%{MARK}%",))["n"]
    cached = db.q1("SELECT COUNT(*) n FROM context_cache")["n"]
    chars = rows["chars"] or 0
    # Начало документа добавляет к каждому запросу примерно 600 символов.
    tokens_in = int((chars + 600 * (rows["n"] or 0)) / 3.2)
    tokens_out = int((rows["n"] or 0) * 40 / 3.2)
    cost = (tokens_in / 1_000_000 * config.COST_INPUT_PER_MTOK
            + tokens_out / 1_000_000 * config.COST_OUTPUT_PER_MTOK)
    return {
        "chunks": rows["n"] or 0, "todo": todo, "cached": cached,
        "tokens_in": tokens_in, "tokens_out": tokens_out,
        "cost_rub": round(cost, 2),
        # Оценка времени: одно обращение около секунды при последовательной
        # работе; облачные провайдеры обычно позволяют несколько параллельно.
        "hours_sequential": round(todo / 3600, 1),
        "model": config.LLM_MODEL or config.LLM_PROVIDER,
    }


def _document_head(doc_id: int, limit: int = 600) -> str:
    row = db.q1("""SELECT text FROM chunks WHERE doc_id=? ORDER BY ord LIMIT 1""",
                (doc_id,))
    return (row["text"] or "")[:limit] if row else ""


def run(limit: int | None = None, progress=None, dry: bool = False) -> dict:
    """Генерирует приставки для фрагментов, у которых их ещё нет."""
    import llm as llm_mod
    say = progress or (lambda t: print(t, flush=True))
    db.init()
    ensure_tables()

    if config.LLM_PROVIDER == "echo" and not dry:
        raise RuntimeError(
            "LLM_PROVIDER=echo — это заглушка без модели. Контекстные приставки "
            "будут бессмысленными. Выберите провайдера и укажите ключ.")

    rows = pending(limit)
    if not rows:
        say("Все фрагменты уже обработаны.")
        return {"processed": 0, "cached": 0, "failed": 0}

    est = estimate()
    say(f"К обработке: {len(rows)} фрагментов. Ориентировочно "
        f"{est['cost_rub']} ₽ и {est['hours_sequential']} ч на всю базу.")
    if dry:
        return {"planned": len(rows), **est}

    engine = llm_mod.get_llm()
    heads: dict[int, str] = {}
    processed = from_cache = failed = busy = 0
    started = time.time()
    conn = db.connect()

    # Обход по индексу, а не по enumerate: когда модель занята живыми
    # вопросами, фрагмент нужно повторить, а не пропустить.
    i = 0
    while i < len(rows):
        row = rows[i]
        i += 1
        head = heads.get(row["doc_id"])
        if head is None:
            head = heads[row["doc_id"]] = _document_head(row["doc_id"])
        key = _hash(row["text"], head)
        hit = db.q1("SELECT context FROM context_cache WHERE text_hash=?", (key,))
        if hit:
            generated = hit["context"]
            from_cache += 1
        else:
            try:
                # Фоновая важность в очереди к модели. Это принципиально:
                # прогон идёт по десяткам тысяч фрагментов и без уступки
                # занял бы модель на всё время, пока сотрудник ждёт ответа
                # в чате.
                with llm_mod.queue_context("приставки"):
                    resp = engine.complete(
                        "Ты помогаешь искать в технической документации.",
                        PROMPT.format(head=head, chunk=row["text"][:3000]))
                generated = " ".join(resp.text.strip().split())[:300]
                busy = 0
            except llm_mod.LLMBusy as exc:
                # Очередь переполнена живыми вопросами. Это не сбой:
                # работа фоновая, спешить некуда. Ждём и пробуем снова, но
                # если модель занята подряд слишком долго — останавливаемся
                # с понятным сообщением, а не топчемся сутки.
                busy += 1
                if busy > 10:
                    say("Модель занята живыми вопросами уже долго — "
                        "останавливаюсь. Запустите обработку ночью: "
                        "обработанное сохранено, продолжится с этого места.")
                    break
                say(f"    модель занята ({exc}) — жду 30 с, попытка {busy} из 10")
                time.sleep(30)
                i -= 1                      # повторить тот же фрагмент
                continue
            except Exception as exc:  # noqa: BLE001 — один сбой не рушит прогон
                failed += 1
                if failed <= 3:
                    say(f"    не вышло на фрагменте {row['id']}: {exc}")
                if failed > 20 and failed > i * 0.5:
                    say("Слишком много ошибок подряд — останавливаюсь.")
                    break
                continue
            db.run("INSERT OR REPLACE INTO context_cache(text_hash, model, context, "
                   "created_at) VALUES (?,?,?,?)",
                   (key, resp.model, generated,
                    datetime.now(timezone.utc).isoformat(timespec="seconds")))
        base = (row["context"] or "").split(MARK)[0].strip()
        conn.execute("UPDATE chunks SET context=? WHERE id=?",
                     (f"{base}\n{MARK}{generated}".strip(), row["id"]))
        processed += 1
        if i % 50 == 0 or i == len(rows):
            conn.commit()
            speed = i / max(time.time() - started, 1e-6)
            say(f"{i}/{len(rows)} · {speed:.1f} фрагм./с · из кэша {from_cache} · "
                f"сбоев {failed}")
    conn.commit()

    say("Готово. Чтобы поиск начал пользоваться новыми приставками, "
        "пересчитайте векторы: python index.py reembed")
    return {"processed": processed, "cached": from_cache, "failed": failed,
            "seconds": round(time.time() - started, 1)}


def clear() -> int:
    """Убирает сгенерированные приставки, оставляя детерминированные."""
    db.init()
    rows = db.q("SELECT id, context FROM chunks WHERE context LIKE ?", (f"%{MARK}%",))
    conn = db.connect()
    for r in rows:
        conn.execute("UPDATE chunks SET context=? WHERE id=?",
                     ((r["context"] or "").split(MARK)[0].strip(), r["id"]))
    conn.commit()
    return len(rows)


def status() -> dict:
    db.init()
    ensure_tables()
    total = db.q1("""SELECT COUNT(*) n FROM chunks c JOIN documents d ON d.id=c.doc_id
                     WHERE d.status='ok'""")["n"]
    done = db.q1("SELECT COUNT(*) n FROM chunks WHERE context LIKE ?",
                 (f"%{MARK}%",))["n"]
    return {"total": total, "done": done, "todo": total - done,
            "enabled": config.CONTEXTUAL_CHUNKS,
            "cached": db.q1("SELECT COUNT(*) n FROM context_cache")["n"]}


def main() -> int:
    p = argparse.ArgumentParser(description="Контекстная приставка через модель")
    p.add_argument("command", choices=["estimate", "run", "clear", "status"])
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    db.init()
    if args.command == "estimate":
        est = estimate()
        print(f"Фрагментов в базе: {est['chunks']}, без приставки: {est['todo']}")
        print(f"В кэше уже есть: {est['cached']}")
        print(f"Токенов на вход: ~{est['tokens_in']:,}".replace(",", " "))
        print(f"Токенов на выход: ~{est['tokens_out']:,}".replace(",", " "))
        print(f"Ориентировочная стоимость: {est['cost_rub']} ₽ "
              f"(модель {est['model']})")
        print(f"Время при последовательной работе: ~{est['hours_sequential']} ч")
        print("\nЭто разовые расходы: результат кэшируется, повторная "
              "индексация тех же файлов ничего не стоит.")
    elif args.command == "run":
        run(limit=args.limit)
    elif args.command == "clear":
        print(f"Приставок убрано: {clear()}")
    elif args.command == "status":
        st = status()
        print(f"Фрагментов: {st['total']}, с приставкой от модели: {st['done']}, "
              f"осталось: {st['todo']}, в кэше: {st['cached']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
