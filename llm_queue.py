"""
Очередь запросов к модели генерации.

Зачем она нужна. Видеокарта выполняет запросы по одному в любом случае.
Разница в том, где именно образуется очередь. Без неё десять
одновременных вопросов уходят на сервер модели разом: он берёт их все,
режет память на десять частей и отвечает на каждый в десять раз
медленнее. В результате никто не получает ответ быстро, а при нехватке
видеопамяти сервер просто падает — и не отвечает вообще никому.

С очередью первый вопрос обрабатывается на полной скорости, остальные
ждут. Суммарное время то же, но первый отвечает сразу, и никто не
падает. Поэтому по умолчанию стоит один одновременный запрос
(`LLM_MAX_CONCURRENT`): для одной видеокарты это правильное значение
почти всегда.

Два свойства, ради которых очередь сделана именно так.

**Очередь общая для всех процессов.** Админка, Telegram-бот и фоновые
задания — это разные процессы, и семафор внутри одного из них не
остановил бы остальные. Самый опасный случай как раз межпроцессный:
пересчёт контекстных приставок на сорок тысяч фрагментов идёт часами и
без общей очереди занимал бы модель полностью, пока человек в чате ждёт
ответа. Согласование идёт через служебную базу — тот же приём, что в
очереди длительных операций, и по той же причине: переживает
перезапуск и не требует отдельного демона.

**Живой вопрос идёт раньше фоновой работы.** Честная очередь «кто встал,
тот и первый» здесь была бы вредной: пакетное задание встаёт в неё сорок
тысяч раз подряд. Поэтому у запроса есть важность: вопрос человека — 0,
фоновая обработка — 5, и вопрос обгоняет пакет, не дожидаясь его конца.
Внутри одной важности порядок обычный — по времени.

Отказ вместо молчания. Если очередь длиннее `LLM_QUEUE_MAX`, ответ
приходит сразу: «модель занята, попробуйте через минуту». Это лучше, чем
две минуты ожидания и та же ошибка в конце. Место, занятое погибшим
процессом, освобождается по `LLM_QUEUE_SLOT_TTL`.

  python llm_queue.py status    — кто выполняется и кто ждёт
  python llm_queue.py stats     — ожидание и отказы за сутки
  python llm_queue.py clear     — снять зависшие места (после аварии)
"""
from __future__ import annotations

import argparse
import contextlib
import contextvars
import os
import socket
import sys
import threading
import time
from datetime import datetime, timezone

import config
import logging_setup

log = logging_setup.get("llm")

# Важность: меньше — раньше. Значения разнесены, чтобы между ними можно
# было вставить новое, не переписывая существующие.
PRIORITY = {
    "вопрос": 0,        # человек ждёт ответ прямо сейчас
    "голос": 0,         # то же самое, но ещё чувствительнее к задержке
    "проверка": 1,      # проверка связи из админки — короткая, пропускаем рано
    "приставки": 5,     # пакетная обработка базы
    "сканы": 5,         # распознавание зрительной моделью на тех же картах
    "фон": 5,
}
DEFAULT_PRIORITY = 0

# Как часто опрашивать очередь. Пауза между попытками растёт, но
# ограничена сверху: это время, которое модель простаивает между двумя
# запросами. При ответе в десять секунд сотая доля секунды — ничто, а
# опрос дешёвый: сначала идёт чтение и только при свободном месте —
# попытка его занять.
POLL_MIN, POLL_MAX = 0.01, 0.1


class LLMBusy(RuntimeError):
    """Очередь переполнена или ожидание не уложилось в срок."""


# Кто именно спрашивает — задаётся вызывающим кодом на время запроса.
# contextvars, а не глобальная переменная: админка обрабатывает запросы
# в потоках, и подпись не должна протекать из одного в другой.
_source = contextvars.ContextVar("llm_source", default="вопрос")
_priority = contextvars.ContextVar("llm_priority", default=None)

# Запасной вариант, когда общая очередь выключена или база недоступна:
# ограничение хотя бы внутри процесса. Это меньше, чем общая очередь, но
# намного лучше, чем ничего.
_local_lock = threading.Condition()
_local_running = 0
_local_waiting = 0

_TABLE = """
CREATE TABLE IF NOT EXISTS llm_queue (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    state       TEXT NOT NULL,          -- waiting | running | done | timeout | refused
    priority    INTEGER NOT NULL,
    source      TEXT,
    provider    TEXT,
    pid         INTEGER,
    host        TEXT,
    created_at  TEXT NOT NULL,
    started_at  TEXT,
    finished_at TEXT,
    expires_at  REAL,
    waited_ms   INTEGER,
    ran_ms      INTEGER,
    ok          INTEGER,
    error       TEXT
);
CREATE INDEX IF NOT EXISTS idx_llm_queue_state ON llm_queue(state, priority, id);
CREATE INDEX IF NOT EXISTS idx_llm_queue_created ON llm_queue(created_at);
"""

# Проверяется по самому соединению, а не флагом: соединение своё у каждого
# потока, а при смене папки данных (тесты, переезд) оно меняется целиком.
_ready_conn = None


def ensure_tables() -> None:
    global _ready_conn
    import db
    conn = db.telemetry()
    if _ready_conn is conn:
        return
    conn.executescript(_TABLE)
    conn.commit()
    _ready_conn = conn


def reset() -> None:
    """Перечитать состояние после смены настроек или базы (нужно тестам)."""
    global _ready_conn, _local_running, _local_waiting, _last_reap
    _ready_conn = None
    _last_reap = 0.0
    with _local_lock:
        _local_running = 0
        _local_waiting = 0
        _local_lock.notify_all()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def priority_of(source: str) -> int:
    return PRIORITY.get(source, DEFAULT_PRIORITY)


@contextlib.contextmanager
def context(source: str, priority: int | None = None):
    """
    Подпись запросов внутри блока: кто спрашивает и насколько это срочно.

        with llm_queue.context("приставки"):
            engine.complete(...)      # уйдёт в очередь с фоновой важностью
    """
    t1 = _source.set(source)
    t2 = _priority.set(priority)
    try:
        yield
    finally:
        _source.reset(t1)
        _priority.reset(t2)


def limit() -> int:
    """0 означает «без ограничения», но учёт очереди всё равно ведётся."""
    return max(0, int(config.LLM_MAX_CONCURRENT))


# ------------------------------------------------------- общая очередь --

_last_reap = 0.0
REAP_EVERY = 1.0        # чаще смысла нет, а запись в базу на каждый опрос — вредна


def _reap(force: bool = False) -> int:
    """
    Освобождает места, занятые погибшими процессами.

    Срок жизни места — не сердцебиение, а прямой срок: запрос к модели
    в принципе не может идти дольше своего сетевого таймаута. Это проще
    отдельного потока с сердцебиением и не может «залипнуть» само.
    """
    global _last_reap
    now = time.time()
    if not force and now - _last_reap < REAP_EVERY:
        return 0
    _last_reap = now
    import db
    freed = db.trun(
        "UPDATE llm_queue SET state='timeout', finished_at=?, expires_at=NULL, "
        "error='место освобождено по сроку: процесс не завершил запрос' "
        "WHERE state IN ('running','waiting') AND expires_at IS NOT NULL "
        "AND expires_at < ?", (_now(), now)).rowcount
    if freed:
        log.warning("освобождено мест в очереди к модели по сроку: %d", freed)
    return freed


def _enter_shared(source: str, priority: int, provider: str) -> int:
    """Встать в очередь. Возвращает номер места."""
    import db
    ensure_tables()
    _reap(force=True)
    deadline = time.time() + max(1.0, config.LLM_QUEUE_TIMEOUT)
    waiting = db.tq1("SELECT COUNT(*) n FROM llm_queue WHERE state='waiting'")["n"]
    if config.LLM_QUEUE_MAX and waiting >= config.LLM_QUEUE_MAX:
        db.trun("INSERT INTO llm_queue(state, priority, source, provider, pid, host, "
                "created_at, finished_at, error) VALUES ('refused',?,?,?,?,?,?,?,?)",
                (priority, source, provider, os.getpid(), socket.gethostname(),
                 _now(), _now(), f"очередь заполнена: ждут {waiting}"))
        raise LLMBusy(f"модель занята: в очереди уже {waiting} запросов. "
                      "Попробуйте через минуту.")
    # Срок ожидающей записи — её собственный таймаут плюс небольшой запас,
    # а не срок занятого места. Разница важна: если процесс умер, пока
    # стоял в очереди, его запись продолжает держать место в порядке
    # очереди, и все пришедшие следом ждут покойника. Пять минут такого
    # ожидания выглядят как «ассистент завис».
    cur = db.trun(
        "INSERT INTO llm_queue(state, priority, source, provider, pid, host, "
        "created_at, expires_at) VALUES ('waiting',?,?,?,?,?,?,?)",
        (priority, source, provider, os.getpid(), socket.gethostname(),
         _now(), deadline + 5.0))
    return int(cur.lastrowid)


def _try_start(ticket: int, priority: int) -> bool:
    """
    Занять место, если оно свободно и раньше нас в очереди никого нет.

    Проверка и захват — одним оператором UPDATE с условиями-подзапросами.
    Это важнее, чем кажется: если сначала посчитать занятые места, а
    потом отдельно занять своё, два процесса увидят «свободно»
    одновременно и оба пройдут — ровно тот случай, ради которого очередь
    и заводилась. Один UPDATE SQLite выполняет целиком под блокировкой
    записи, поэтому такого разрыва здесь нет.
    """
    import db
    _reap()
    cap = limit()
    expires = time.time() + max(30.0, config.LLM_QUEUE_SLOT_TTL)
    if not cap:                                   # без ограничения — просто отметиться
        cur = db.trun("UPDATE llm_queue SET state='running', started_at=?, "
                      "expires_at=? WHERE id=? AND state='waiting'",
                      (_now(), expires, ticket))
        return cur.rowcount == 1
    # Сначала дешёвое чтение. Без него каждый опрос каждого ждущего брал бы
    # блокировку записи, и двадцать человек в очереди нагружали бы базу
    # сильнее, чем сама модель. Читать и потом писать безопасно ровно
    # потому, что решает всё равно условие внутри UPDATE.
    free = db.tq1(
        "SELECT (SELECT COUNT(*) FROM llm_queue WHERE state='running') "
        "     + (SELECT COUNT(*) FROM llm_queue w WHERE w.state='waiting' "
        "        AND (w.priority < ? OR (w.priority = ? AND w.id < ?))) AS busy",
        (priority, priority, ticket))
    if free is not None and free["busy"] >= cap:
        return False
    cur = db.trun(
        "UPDATE llm_queue SET state='running', started_at=?, expires_at=? "
        "WHERE id=? AND state='waiting' AND "
        "  (SELECT COUNT(*) FROM llm_queue WHERE state='running') "
        "  + (SELECT COUNT(*) FROM llm_queue w WHERE w.state='waiting' "
        "     AND (w.priority < ? OR (w.priority = ? AND w.id < ?))) < ?",
        (_now(), expires, ticket, priority, priority, ticket, cap))
    return cur.rowcount == 1


def _finish_shared(ticket: int, waited_ms: int, ran_ms: int,
                   ok: bool, error: str = "") -> None:
    import db
    try:
        db.trun("UPDATE llm_queue SET state='done', finished_at=?, waited_ms=?, "
                "ran_ms=?, ok=?, error=?, expires_at=NULL WHERE id=?",
                (_now(), waited_ms, ran_ms, 1 if ok else 0, error[:300], ticket))
    except Exception as exc:  # noqa: BLE001 — освобождение места важнее отчёта
        log.error("не удалось закрыть место в очереди %s: %s", ticket, exc)


def _drop_shared(ticket: int, reason: str) -> None:
    import db
    try:
        db.trun("UPDATE llm_queue SET state='timeout', finished_at=?, error=?, "
                "expires_at=NULL WHERE id=?", (_now(), reason[:300], ticket))
    except Exception:  # noqa: BLE001
        pass


# ------------------------------------------------- очередь внутри процесса --

def _enter_local() -> None:
    global _local_running, _local_waiting
    cap = limit()
    deadline = time.time() + max(1.0, config.LLM_QUEUE_TIMEOUT)
    with _local_lock:
        if config.LLM_QUEUE_MAX and _local_waiting >= config.LLM_QUEUE_MAX:
            raise LLMBusy(f"модель занята: в очереди уже {_local_waiting} запросов.")
        _local_waiting += 1
        try:
            while cap and _local_running >= cap:
                if not _local_lock.wait(timeout=max(0.0, deadline - time.time())):
                    if _local_running >= cap:
                        raise LLMBusy(
                            f"модель не освободилась за {config.LLM_QUEUE_TIMEOUT:.0f} с. "
                            "Попробуйте позже.")
            _local_running += 1
        finally:
            _local_waiting -= 1


def _leave_local() -> None:
    global _local_running
    with _local_lock:
        _local_running = max(0, _local_running - 1)
        _local_lock.notify()


# ------------------------------------------------------------------- вход --

@contextlib.contextmanager
def slot(source: str | None = None, priority: int | None = None,
         provider: str = ""):
    """
    Место в очереди на время одного обращения к модели.

    Через него проходят все запросы без исключения: и вопросы из чата, и
    фоновая обработка, и проверка связи из админки. Единая точка нужна не
    ради красоты, а чтобы ограничение нельзя было обойти, забыв про него
    в новом месте кода.
    """
    src = source or _source.get()
    prio = priority if priority is not None else _priority.get()
    if prio is None:
        prio = priority_of(src)

    started_wait = time.time()
    shared = bool(config.LLM_QUEUE_SHARED)
    ticket = None
    if shared:
        try:
            ticket = _enter_shared(src, prio, provider)
        except LLMBusy:
            raise
        except Exception as exc:  # noqa: BLE001 — база недоступна, не теряем запрос
            log.warning("общая очередь недоступна (%s), ограничиваю в пределах "
                        "процесса", exc)
            shared = False

    if shared and ticket is not None:
        deadline = started_wait + max(1.0, config.LLM_QUEUE_TIMEOUT)
        pause = POLL_MIN
        while True:
            try:
                if _try_start(ticket, prio):
                    break
            except Exception as exc:  # noqa: BLE001
                log.warning("сбой очереди при захвате места (%s), продолжаю "
                            "без общей очереди", exc)
                _drop_shared(ticket, f"сбой очереди: {exc}")
                ticket, shared = None, False
                break
            if time.time() > deadline:
                _drop_shared(ticket, "не дождался очереди")
                raise LLMBusy(
                    f"модель занята: не дождались очереди за "
                    f"{config.LLM_QUEUE_TIMEOUT:.0f} с. Попробуйте позже.")
            time.sleep(pause)
            pause = min(POLL_MAX, pause * 1.6)

    if not shared:
        _enter_local()

    waited_ms = int((time.time() - started_wait) * 1000)
    if waited_ms > 1000:
        log.info("ожидание очереди к модели: %d мс (%s, важность %d)",
                 waited_ms, src, prio)
    ran = time.time()
    try:
        yield {"ticket": ticket, "waited_ms": waited_ms, "source": src,
               "priority": prio}
    except Exception as exc:
        if ticket is not None:
            _finish_shared(ticket, waited_ms, int((time.time() - ran) * 1000),
                           False, str(exc))
        else:
            _leave_local()
        raise
    else:
        if ticket is not None:
            _finish_shared(ticket, waited_ms, int((time.time() - ran) * 1000), True)
        else:
            _leave_local()


# ------------------------------------------------------------ наблюдение --

def status() -> dict:
    """Что происходит с очередью прямо сейчас."""
    out = {"limit": limit(), "shared": bool(config.LLM_QUEUE_SHARED),
           "queue_max": config.LLM_QUEUE_MAX,
           "timeout": config.LLM_QUEUE_TIMEOUT,
           "running": 0, "waiting": 0, "items": []}
    if not config.LLM_QUEUE_SHARED:
        with _local_lock:
            out["running"], out["waiting"] = _local_running, _local_waiting
        return out
    try:
        import db
        ensure_tables()
        _reap(force=True)
        rows = db.tq("SELECT id, state, priority, source, pid, created_at, started_at "
                     "FROM llm_queue WHERE state IN ('running','waiting') "
                     "ORDER BY state DESC, priority, id LIMIT 50")
        now = time.time()
        for r in rows:
            item = dict(r)
            try:
                born = datetime.fromisoformat(r["created_at"]).timestamp()
                item["age_s"] = round(now - born, 1)
            except Exception:  # noqa: BLE001
                item["age_s"] = None
            out["items"].append(item)
            out[r["state"]] = out.get(r["state"], 0) + 1
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
    return out


def stats(hours: int = 24) -> dict:
    """Ожидание, отказы и загрузка за период — для админки и оповещений."""
    out = {"hours": hours, "total": 0, "refused": 0, "timeout": 0,
           "wait_avg_ms": 0, "wait_p95_ms": 0, "run_avg_ms": 0, "by_source": []}
    if not config.LLM_QUEUE_SHARED:
        return out
    try:
        import db
        ensure_tables()
        since = datetime.fromtimestamp(time.time() - hours * 3600,
                                       timezone.utc).isoformat(timespec="seconds")
        row = db.tq1("SELECT COUNT(*) n, AVG(waited_ms) w, AVG(ran_ms) r "
                     "FROM llm_queue WHERE state='done' AND created_at >= ?", (since,))
        out["total"] = row["n"] or 0
        out["wait_avg_ms"] = int(row["w"] or 0)
        out["run_avg_ms"] = int(row["r"] or 0)
        waits = [r["waited_ms"] for r in db.tq(
            "SELECT waited_ms FROM llm_queue WHERE state='done' AND created_at >= ? "
            "AND waited_ms IS NOT NULL ORDER BY waited_ms", (since,))]
        if waits:
            out["wait_p95_ms"] = int(waits[min(len(waits) - 1,
                                               int(len(waits) * 0.95))])
        out["refused"] = db.tq1("SELECT COUNT(*) n FROM llm_queue "
                                "WHERE state='refused' AND created_at >= ?",
                                (since,))["n"]
        out["timeout"] = db.tq1("SELECT COUNT(*) n FROM llm_queue "
                                "WHERE state='timeout' AND created_at >= ?",
                                (since,))["n"]
        out["by_source"] = [dict(r) for r in db.tq(
            "SELECT source, COUNT(*) n, AVG(waited_ms) wait, AVG(ran_ms) run "
            "FROM llm_queue WHERE created_at >= ? GROUP BY source "
            "ORDER BY n DESC LIMIT 10", (since,))]
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
    return out


def clear() -> int:
    """Снять все занятые места. Нужно после аварийной остановки."""
    import db
    ensure_tables()
    cur = db.trun("UPDATE llm_queue SET state='timeout', finished_at=?, "
                  "error='снято вручную', expires_at=NULL "
                  "WHERE state IN ('running','waiting')", (_now(),))
    log.warning("очередь к модели очищена вручную: снято %d", cur.rowcount)
    return cur.rowcount


def prune(days: int = 7) -> int:
    """Убрать старые записи очереди — они нужны только для статистики."""
    import db
    ensure_tables()
    since = datetime.fromtimestamp(time.time() - days * 86400,
                                   timezone.utc).isoformat(timespec="seconds")
    cur = db.trun("DELETE FROM llm_queue WHERE created_at < ? "
                  "AND state NOT IN ('running','waiting')", (since,))
    return cur.rowcount


def main() -> int:
    p = argparse.ArgumentParser(description="Очередь запросов к модели")
    p.add_argument("command", choices=["status", "stats", "clear", "prune"])
    p.add_argument("--hours", type=int, default=24)
    p.add_argument("--days", type=int, default=7)
    args = p.parse_args()

    if args.command == "status":
        st = status()
        cap = "без ограничения" if not st["limit"] else f"{st['limit']} одновременно"
        print(f"Ограничение: {cap}, очередь "
              f"{'общая для всех процессов' if st['shared'] else 'только в процессе'}")
        print(f"Выполняется: {st.get('running', 0)}, ждут: {st.get('waiting', 0)}")
        for item in st["items"]:
            print(f"  {item['state']:8} важность {item['priority']} "
                  f"{item['source'] or '—':12} процесс {item['pid']} "
                  f"{item.get('age_s') or 0:.1f} с")
        if st.get("error"):
            print(f"Ошибка: {st['error']}")
    elif args.command == "stats":
        st = stats(args.hours)
        print(f"За {st['hours']} ч: запросов {st['total']}, "
              f"отказов по переполнению {st['refused']}, "
              f"не дождались {st['timeout']}")
        print(f"Ожидание в очереди: в среднем {st['wait_avg_ms']} мс, "
              f"95-й процентиль {st['wait_p95_ms']} мс")
        print(f"Сам запрос к модели: в среднем {st['run_avg_ms']} мс")
        for row in st["by_source"]:
            print(f"  {row['source'] or '—':14} {row['n']:6} шт  "
                  f"ожидание {int(row['wait'] or 0):6} мс  "
                  f"запрос {int(row['run'] or 0):6} мс")
        if st["refused"] and st["total"]:
            share = st["refused"] / (st["total"] + st["refused"]) * 100
            print(f"\nОтказов {share:.1f} %. Если это заметно — "
                  "увеличьте LLM_MAX_CONCURRENT (при запасе видеопамяти) "
                  "или LLM_QUEUE_MAX.")
    elif args.command == "clear":
        print(f"Снято мест: {clear()}")
    else:
        print(f"Удалено записей: {prune(args.days)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
