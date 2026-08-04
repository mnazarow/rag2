"""
Очередь длительных операций.

Зачем отдельный модуль. Индексация, распознавание сканов, обучение модели
и пересчёт векторов идут минутами и часами. Раньше они запускались просто
потоками внутри процесса админки, и это давало две неприятности.

Первая: перезапуск процесса терял задачу без следа. Человек нажал
«переиндексировать», ушёл, сервер обновился — и никто не узнает, что
работа не сделана.

Вторая, более опасная: ничто не мешало запустить две одинаковые операции
сразу. Два пересчёта векторов пишут в один и тот же файл, и результат —
испорченный векторный индекс, который выглядит целым. Здесь это закрыто
блокировками: у каждой задачи есть имя ресурса, который она занимает, и
вторая попытка получает понятный отказ, а не начинает работу.

Очередь хранится в той же базе, что и всё остальное, поэтому переживает
перезапуск: задача, прерванная на середине, помечается как оборванная и
может быть перезапущена одной кнопкой.

  python jobs.py list           — что в очереди и что выполнялось
  python jobs.py worker         — обработчик очереди (обычно запускает админка)
  python jobs.py cancel ID      — снять задачу
  python jobs.py retry ID       — поставить заново
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
import traceback
from datetime import datetime, timezone

import db
import logging_setup

log = logging_setup.get("index")

# Ресурсы, за которые задачи конкурируют. Две задачи, которым нужен хотя
# бы один общий ресурс, одновременно не выполняются никогда.
#
# Список составлен по тому, что задача реально изменяет. Например,
# индексация пишет и в базу, и в файл векторов, поэтому она конфликтует
# и с пересчётом векторов, и с распознаванием сканов: результат
# одновременной записи в один файл векторов выглядит целым, а на деле
# испорчен, и обнаруживается это только по странным ответам бота.
RESOURCES: dict[str, tuple[str, ...]] = {
    "reindex":       ("index", "vectors"),
    "reindex_full":  ("index", "vectors"),
    "repair":        ("vectors",),
    "reembed":       ("vectors",),       # перезаписывает файл векторов целиком
    "train_lsa":     ("model",),
    # Смена провайдера: качает веса, может обучать модель и целиком
    # перезаписывает векторы — держит все три ресурса.
    "embed_switch":  ("vectors", "model", "models"),
    # Смена генерации: качает веса и перезапускает сервер модели.
    "llm_switch":    ("model", "models"),
    "ocr":           ("index", "vectors"),
    "ocr_retry":     ("index", "vectors"),
    "media":         ("index", "vectors"),
    "contextual":    ("index",),
    "crawl":         ("crawl", "index", "vectors"),
    "structure":     ("index",),
    "backup":        ("backup",),
    "backup_verify": ("backup",),
    "backup_prune":  ("backup",),
    "restore":       ("index", "vectors", "model", "backup"),
    "graph":         ("graph",),
    "compare":       ("eval",),
    "regression":    ("eval",),
    "alerts":        ("alerts",),
    "retention":     ("retention",),
    "restore_drill": ("eval", "backup"),
    # Загрузка весов: десятки гигабайт и часы. Держим за отдельным
    # ресурсом, чтобы две загрузки не писали в одну папку, и в очереди —
    # чтобы перезапуск процесса не терял ни саму работу, ни память о ней.
    "model_install": ("models",),
}

# Задача считается брошенной, если обработчик не подавал признаков жизни
# дольше этого срока (процесс убит, сервер перезагрузился).
HEARTBEAT_TIMEOUT = 180


class JobError(RuntimeError):
    pass


class Busy(JobError):
    """Ресурс уже занят другой задачей."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_tables() -> None:
    db.connect().executescript("""
    CREATE TABLE IF NOT EXISTS jobs (
        id           INTEGER PRIMARY KEY,
        kind         TEXT NOT NULL,
        title        TEXT,
        resource     TEXT,
        payload_json TEXT,
        priority     INTEGER DEFAULT 5,     -- меньше = важнее
        status       TEXT DEFAULT 'queued', -- queued|running|done|error|cancelled|stale
        attempt      INTEGER DEFAULT 0,
        max_attempts INTEGER DEFAULT 1,
        worker       TEXT,
        heartbeat    TEXT,
        progress     TEXT,
        log_json     TEXT,
        result_json  TEXT,
        error        TEXT,
        created_by   TEXT,
        created_at   TEXT,
        started_at   TEXT,
        finished_at  TEXT
    );
    CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status, priority, id);
    """)
    db.connect().commit()


# ------------------------------------------------------------ постановка ---
def enqueue(kind: str, title: str = "", payload: dict | None = None,
            priority: int = 5, max_attempts: int = 1,
            created_by: str = "админка", wait: bool = False) -> dict:
    """
    Ставит задачу в очередь. Если ресурс уже занят — отказывает.

    Отказ здесь лучше очереди: человек, нажавший кнопку второй раз, должен
    узнать, что работа уже идёт, а не получить вторую такую же через час.

    wait=True меняет это для задач, которые пользователь ХОЧЕТ выполнить
    после текущей работы: смена провайдера поиска во время многочасовой
    переиндексации иначе была невозможна вовсе — отказ «дождитесь
    окончания» приходил и через час, и через пять, и выбор «сам
    возвращался» к прежнему. Повторная постановка ТАКОЙ ЖЕ задачи
    отклоняется в любом случае: вторая копия не нужна никогда.
    """
    ensure_tables()
    needed = set(RESOURCES.get(kind, (kind,)))
    reap_stale()
    # Активных задач всегда единицы, поэтому пересечение считаем в Python:
    # это понятнее любого условия в SQL и не зависит от формата хранения.
    for row in db.q("SELECT id, kind, title, resource FROM jobs "
                    "WHERE status IN ('queued','running')"):
        if row["kind"] == kind:
            raise Busy(
                f"такая задача уже есть: {row['title'] or row['kind']} "
                f"(№{row['id']}). Второй раз ставить не нужно — ход виден "
                f"в разделе «Конвейер».")
        busy_res = set((row["resource"] or "").split(","))
        clash = needed & busy_res
        if clash and not wait:
            raise Busy(
                f"уже выполняется: {row['title'] or row['kind']} (задача №{row['id']}). "
                f"Обе задачи меняют одно и то же ({', '.join(sorted(clash))}), "
                f"а одновременная запись испортила бы индекс. Дождитесь окончания.")
    cur = db.run("""INSERT INTO jobs(kind, title, resource, payload_json, priority,
                    max_attempts, created_by, created_at)
                    VALUES (?,?,?,?,?,?,?,?)""",
                 (kind, title or kind, ",".join(sorted(needed)),
                  json.dumps(payload or {}, ensure_ascii=False),
                  priority, max_attempts, created_by, _now()))
    job_id = int(cur.lastrowid)
    log.info("задача поставлена в очередь: %s (№%d)", title or kind, job_id)
    _wake()
    return get(job_id)


def get(job_id: int) -> dict:
    row = db.q1("SELECT * FROM jobs WHERE id=?", (job_id,))
    if row is None:
        return {}
    item = dict(row)
    item["log"] = json.loads(item.pop("log_json", None) or "[]")
    item["payload"] = json.loads(item.pop("payload_json", None) or "{}")
    item["result"] = json.loads(item.pop("result_json", None) or "null")
    return item


def recent(limit: int = 30) -> list[dict]:
    ensure_tables()
    reap_stale()
    rows = db.q("""SELECT id, kind, title, status, attempt, max_attempts, progress,
                          error, created_by, created_at, started_at, finished_at
                   FROM jobs ORDER BY
                     CASE status WHEN 'running' THEN 0 WHEN 'queued' THEN 1 ELSE 2 END,
                     id DESC LIMIT ?""", (limit,))
    out = []
    for r in rows:
        item = dict(r)
        started = item.get("started_at") or item.get("created_at")
        item["seconds"] = _elapsed(started, item.get("finished_at"))
        out.append(item)
    return out


def _elapsed(started: str | None, finished: str | None) -> int:
    if not started:
        return 0
    try:
        a = datetime.fromisoformat(started)
        b = datetime.fromisoformat(finished) if finished else datetime.now(timezone.utc)
        return max(int((b - a).total_seconds()), 0)
    except Exception:  # noqa: BLE001
        return 0


def cancel(job_id: int) -> bool:
    """Снимает задачу из очереди. Уже выполняющуюся не прерывает."""
    row = db.q1("SELECT status FROM jobs WHERE id=?", (job_id,))
    if row is None or row["status"] != "queued":
        return False
    db.run("UPDATE jobs SET status='cancelled', finished_at=? WHERE id=?",
           (_now(), job_id))
    return True


def retry(job_id: int) -> dict:
    """Ставит задачу заново — с теми же параметрами."""
    old = get(job_id)
    if not old:
        raise JobError("задача не найдена")
    return enqueue(old["kind"], old["title"], old["payload"],
                   created_by=f"повтор №{job_id}")


def reap_stale() -> int:
    """
    Помечает брошенные задачи.

    Задача считается брошенной, если она в состоянии «выполняется», а
    обработчик давно не подавал признаков жизни: процесс убит или сервер
    перезагрузился. Без этой уборки ресурс остался бы занят навсегда,
    и кнопка «переиндексировать» перестала бы работать до правки базы.
    """
    cutoff = time.time() - HEARTBEAT_TIMEOUT
    stale = []
    for row in db.q("SELECT id, heartbeat, title FROM jobs WHERE status='running'"):
        beat = row["heartbeat"]
        try:
            ts = datetime.fromisoformat(beat).timestamp() if beat else 0
        except Exception:  # noqa: BLE001
            ts = 0
        if ts < cutoff:
            stale.append(row["id"])
            log.warning("задача «%s» (№%d) прервана: обработчик молчит",
                        row["title"], row["id"])
    for job_id in stale:
        db.run("""UPDATE jobs SET status='stale', finished_at=?,
                  error='прервана: процесс завершился во время выполнения'
                  WHERE id=?""", (_now(), job_id))
    return len(stale)


# ------------------------------------------------------------- выполнение --
HANDLERS: dict[str, callable] = {}


def handler(kind: str):
    """Регистрирует обработчик задачи данного вида."""
    def wrap(fn):
        HANDLERS[kind] = fn
        return fn
    return wrap


_wakeup = threading.Event()
_worker_started = False


def _wake() -> None:
    _wakeup.set()


def _claim() -> dict | None:
    """Берёт следующую задачу, не позволяя двум обработчикам взять одну.

    Задачи, чьи ресурсы заняты выполняющейся сейчас работой,
    пропускаются: с появлением wait=True в очереди может стоять смена
    провайдера, ждущая конца переиндексации, — второй обработчик
    (например, запущенный из командной строки) не должен взять её
    раньше времени.
    """
    running = db.q("SELECT resource FROM jobs WHERE status='running'")
    busy: set[str] = set()
    for r in running:
        busy |= set(x for x in (r["resource"] or "").split(",") if x)
    row = None
    for cand in db.q("""SELECT id, kind, resource FROM jobs
                        WHERE status='queued' ORDER BY priority, id"""):
        need = set(x for x in (cand["resource"] or "").split(",") if x)
        if need & busy:
            continue
        row = cand
        break
    if row is None:
        return None
    # Условие в UPDATE — та самая защита от гонки: если другой обработчик
    # успел первым, изменённых строк будет ноль и мы просто попробуем снова.
    cur = db.run("""UPDATE jobs SET status='running', worker=?, started_at=?,
                    heartbeat=?, attempt=attempt+1
                    WHERE id=? AND status='queued'""",
                 (f"{socket.gethostname()}:{os.getpid()}", _now(), _now(), row["id"]))
    if cur.rowcount != 1:
        return None
    return get(row["id"])


def _run_one(job: dict) -> None:
    fn = HANDLERS.get(job["kind"])
    lines: list[str] = []
    last_beat = [0.0]

    def progress(text: str) -> None:
        lines.append(str(text)[:300])
        del lines[:-200]
        now = time.time()
        # Сердцебиение и ход пишем не чаще раза в секунду: задача может
        # печатать сотни строк, и запись каждой в базу сама станет узким местом.
        if now - last_beat[0] > 1.0:
            last_beat[0] = now
            db.run("UPDATE jobs SET heartbeat=?, progress=?, log_json=? WHERE id=?",
                   (_now(), str(text)[:200],
                    json.dumps(lines[-60:], ensure_ascii=False), job["id"]))

    if fn is None:
        db.run("""UPDATE jobs SET status='error', error=?, finished_at=? WHERE id=?""",
               (f"нет обработчика для «{job['kind']}»", _now(), job["id"]))
        return
    try:
        result = fn(job["payload"], progress)
        db.run("""UPDATE jobs SET status='done', result_json=?, finished_at=?,
                  progress=?, log_json=? WHERE id=?""",
               (json.dumps(result, ensure_ascii=False, default=str), _now(),
                (lines[-1] if lines else "готово"),
                json.dumps(lines[-60:], ensure_ascii=False), job["id"]))
        log.info("задача «%s» (№%d) выполнена", job["title"], job["id"])
    except Exception as exc:  # noqa: BLE001 — задача не должна ронять обработчик
        again = job["attempt"] < job["max_attempts"]
        db.run("""UPDATE jobs SET status=?, error=?, finished_at=?, log_json=? WHERE id=?""",
               ("queued" if again else "error",
                f"{exc}\n{traceback.format_exc()[-1200:]}",
                None if again else _now(),
                json.dumps(lines[-60:], ensure_ascii=False), job["id"]))
        log.error("задача «%s» (№%d) упала: %s%s", job["title"], job["id"], exc,
                  " — будет повторена" if again else "")


def worker_loop(stop: threading.Event | None = None) -> None:
    ensure_tables()
    log.info("обработчик очереди запущен")
    while not (stop and stop.is_set()):
        reap_stale()
        job = _claim()
        if job is None:
            _wakeup.wait(timeout=3.0)
            _wakeup.clear()
            continue
        _run_one(job)


def start_worker() -> None:
    """Запускает обработчик в фоне — вызывается админкой при старте."""
    global _worker_started
    if _worker_started:
        return
    _worker_started = True
    threading.Thread(target=worker_loop, daemon=True).start()


# ------------------------------------------------------------------- CLI ---
def main() -> int:
    p = argparse.ArgumentParser(description="Очередь длительных операций")
    p.add_argument("command", choices=["list", "worker", "cancel", "retry", "reap"])
    p.add_argument("job_id", nargs="?", type=int)
    args = p.parse_args()
    db.init()
    ensure_tables()
    if args.command == "list":
        rows = recent(40)
        if not rows:
            print("Очередь пуста.")
            return 0
        print(f"{'№':>5} {'состояние':12} {'сек':>6}  задача")
        for r in rows:
            print(f"{r['id']:>5} {r['status']:12} {r['seconds']:>6}  {r['title']}"
                  + (f"  — {r['error'][:70]}" if r["error"] else ""))
    elif args.command == "worker":
        import handlers  # noqa: F401 — регистрация обработчиков
        worker_loop()
    elif args.command == "cancel":
        print("снята" if cancel(args.job_id) else "нельзя снять: уже выполняется или нет такой")
    elif args.command == "retry":
        print("поставлена заново:", retry(args.job_id)["id"])
    elif args.command == "reap":
        print("помечено прерванных:", reap_stale())
    return 0


if __name__ == "__main__":
    sys.exit(main())
