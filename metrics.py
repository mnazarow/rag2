"""
Метрики сервера и аналитика работы системы.

  python metrics.py collect        — снять показания один раз
  python metrics.py daemon         — снимать по расписанию
  python metrics.py report         — сводка за период
  python metrics.py series cpu     — ряд значений для графика

Собирается две группы данных.

Нагрузка сервера: процессор, память, диск, сеть, видеокарты — их загрузка,
занятая видеопамять, температура и потребление. Это нужно, чтобы понимать,
упирается ли система в железо и когда пора добавлять ресурсы.

Работа ассистента: сколько вопросов, какая доля осталась без ответа, как
распределяются задержки по этапам, сколько потрачено токенов и денег,
какие модели используются и с какой скоростью отвечают, какие разделы
базы попадают в ответы чаще других, что спрашивают.

Всё пишется в ту же базу, без внешних сервисов. При желании те же
показатели отдаются в формате Prometheus — тогда графики можно строить
в Grafana, а историю хранить дольше.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import config
import db
import logging_setup

log = logging_setup.get("web")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_tables() -> None:
    # Метрики живут в отдельной базе (см. db.telemetry): пишутся постоянно,
    # с документами не соединяются, конкурировать за блокировку с
    # индексацией им незачем.
    db.telemetry().executescript("""
    CREATE TABLE IF NOT EXISTS server_metrics (
        id            INTEGER PRIMARY KEY,
        ts            TEXT,
        cpu_percent   REAL,
        load_1        REAL,
        ram_used_gb   REAL,
        ram_total_gb  REAL,
        disk_used_gb  REAL,
        disk_total_gb REAL,
        net_rx_mb     REAL,
        net_tx_mb     REAL,
        gpu_json      TEXT,
        procs         INTEGER
    );
    CREATE INDEX IF NOT EXISTS idx_metrics_ts ON server_metrics(ts);

    CREATE TABLE IF NOT EXISTS model_usage (
        id           INTEGER PRIMARY KEY,
        ts           TEXT,
        model        TEXT,
        provider     TEXT,
        kind         TEXT,        -- llm | embedding | vision | asr | tts | rerank
        tokens_in    INTEGER,
        tokens_out   INTEGER,
        latency_ms   INTEGER,
        ok           INTEGER,
        error        TEXT,
        request_id   TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_usage_ts ON model_usage(ts);
    CREATE INDEX IF NOT EXISTS idx_usage_model ON model_usage(model);

    CREATE TABLE IF NOT EXISTS stage_timings (
        id         INTEGER PRIMARY KEY,
        ts         TEXT,
        request_id TEXT,
        stage      TEXT,
        ms         INTEGER
    );
    CREATE INDEX IF NOT EXISTS idx_stage_ts ON stage_timings(ts);
    """)


# ------------------------------------------------------------ сбор данных ---
_last_net: dict = {}


def _cpu_percent() -> float:
    """Загрузка процессора без внешних библиотек."""
    try:
        import psutil
        return psutil.cpu_percent(interval=0.4)
    except ImportError:
        pass
    if platform.system() != "Linux":
        return 0.0
    try:
        def snapshot():
            fields = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
            values = [int(x) for x in fields]
            return sum(values), values[3]
        total1, idle1 = snapshot()
        time.sleep(0.3)
        total2, idle2 = snapshot()
        dt, di = total2 - total1, idle2 - idle1
        return round(100 * (dt - di) / dt, 1) if dt else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def _ram() -> tuple[float, float]:
    try:
        import psutil
        m = psutil.virtual_memory()
        return round(m.used / 1024 ** 3, 2), round(m.total / 1024 ** 3, 2)
    except ImportError:
        pass
    if platform.system() == "Linux":
        try:
            info = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, value = line.split(":", 1)
                info[key] = int(value.strip().split()[0]) * 1024
            total = info.get("MemTotal", 0)
            available = info.get("MemAvailable", 0)
            return round((total - available) / 1024 ** 3, 2), round(total / 1024 ** 3, 2)
        except Exception:  # noqa: BLE001
            pass
    return 0.0, 0.0


def _net() -> tuple[float, float]:
    """Накопительный трафик в мегабайтах."""
    try:
        import psutil
        io = psutil.net_io_counters()
        return round(io.bytes_recv / 1024 ** 2, 1), round(io.bytes_sent / 1024 ** 2, 1)
    except ImportError:
        pass
    if platform.system() == "Linux":
        try:
            rx = tx = 0
            for line in Path("/proc/net/dev").read_text().splitlines()[2:]:
                name, data = line.split(":", 1)
                if name.strip() == "lo":
                    continue
                parts = data.split()
                rx += int(parts[0])
                tx += int(parts[8])
            return round(rx / 1024 ** 2, 1), round(tx / 1024 ** 2, 1)
        except Exception:  # noqa: BLE001
            pass
    return 0.0, 0.0


def _load1() -> float:
    try:
        return round(os.getloadavg()[0], 2)
    except (AttributeError, OSError):
        return 0.0


def _procs() -> int:
    try:
        import psutil
        return len(psutil.pids())
    except ImportError:
        pass
    try:
        return len([p for p in Path("/proc").iterdir() if p.name.isdigit()])
    except Exception:  # noqa: BLE001
        return 0


def collect() -> dict:
    """Снимает показания и записывает в базу."""
    ensure_tables()
    import models as models_mod
    ram_used, ram_total = _ram()
    disk = shutil.disk_usage(str(config.DATA_DIR))
    rx, tx = _net()
    cards = models_mod.gpus()
    row = {
        "ts": _now(),
        "cpu_percent": _cpu_percent(),
        "load_1": _load1(),
        "ram_used_gb": ram_used,
        "ram_total_gb": ram_total,
        "disk_used_gb": round(disk.used / 1024 ** 3, 1),
        "disk_total_gb": round(disk.total / 1024 ** 3, 1),
        "net_rx_mb": rx,
        "net_tx_mb": tx,
        "gpu_json": json.dumps(cards, ensure_ascii=False),
        "procs": _procs(),
    }
    db.trun("""INSERT INTO server_metrics(ts, cpu_percent, load_1, ram_used_gb,
              ram_total_gb, disk_used_gb, disk_total_gb, net_rx_mb, net_tx_mb,
              gpu_json, procs) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
           tuple(row.values()))
    cutoff = (datetime.now(timezone.utc) - timedelta(days=config.METRICS_KEEP_DAYS)).isoformat()
    db.trun("DELETE FROM server_metrics WHERE ts < ?", (cutoff,))
    row["gpus"] = cards
    return row


def record_model_call(model: str, provider: str, kind: str, tokens_in: int = 0,
                      tokens_out: int = 0, latency_ms: int = 0, ok: bool = True,
                      error: str = "") -> None:
    """Вызывается после каждого обращения к модели."""
    if not config.METRICS_ENABLED:
        return
    try:
        ensure_tables()
        db.trun("""INSERT INTO model_usage(ts, model, provider, kind, tokens_in,
                  tokens_out, latency_ms, ok, error, request_id)
                  VALUES (?,?,?,?,?,?,?,?,?,?)""",
               (_now(), model, provider, kind, tokens_in, tokens_out, latency_ms,
                int(ok), error[:300], logging_setup.current_request()))
    except Exception:  # noqa: BLE001
        pass


def record_stage(stage: str, ms: int) -> None:
    if not config.METRICS_ENABLED:
        return
    try:
        ensure_tables()
        db.trun("INSERT INTO stage_timings(ts, request_id, stage, ms) VALUES (?,?,?,?)",
               (_now(), logging_setup.current_request(), stage, ms))
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------- отчёты ----
def _since(hours: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


def server_series(hours: int = 24, points: int = 120) -> dict:
    """Ряды для графиков нагрузки."""
    ensure_tables()
    rows = db.tq("SELECT * FROM server_metrics WHERE ts > ? ORDER BY ts", (_since(hours),))
    if not rows:
        return {"points": [], "gpu_names": []}
    step = max(1, len(rows) // points)
    sampled = rows[::step]
    gpu_names: list[str] = []
    out = []
    for r in sampled:
        cards = json.loads(r["gpu_json"] or "[]")
        if cards and not gpu_names:
            gpu_names = [c["name"] for c in cards]
        out.append({
            "ts": r["ts"][11:16],
            "cpu": r["cpu_percent"],
            "load": r["load_1"],
            "ram": round(r["ram_used_gb"] / max(r["ram_total_gb"], 0.01) * 100, 1),
            "ram_gb": r["ram_used_gb"],
            "disk": round(r["disk_used_gb"] / max(r["disk_total_gb"], 0.01) * 100, 1),
            "gpu_util": [c.get("utilization", 0) for c in cards],
            "gpu_mem": [round(c.get("memory_used_gb", 0) / max(c.get("memory_total_gb", 1), 0.01)
                              * 100, 1) for c in cards],
            "gpu_temp": [c.get("temperature", 0) for c in cards],
            "gpu_power": [c.get("power_w", 0) for c in cards],
        })
    return {"points": out, "gpu_names": gpu_names}


def model_report(hours: int = 168) -> dict:
    """Аналитика по использованию моделей."""
    ensure_tables()
    since = _since(hours)
    by_model = [dict(r) for r in db.tq("""
        SELECT model, provider, kind, COUNT(*) calls,
               SUM(tokens_in) tin, SUM(tokens_out) tout,
               AVG(latency_ms) avg_ms, MAX(latency_ms) max_ms,
               SUM(CASE WHEN ok=0 THEN 1 ELSE 0 END) errors
        FROM model_usage WHERE ts > ? GROUP BY model, provider, kind
        ORDER BY calls DESC""", (since,))]
    for m in by_model:
        m["avg_ms"] = int(m["avg_ms"] or 0)
        m["cost_rub"] = round(
            (m["tin"] or 0) / 1_000_000 * config.COST_INPUT_PER_MTOK +
            (m["tout"] or 0) / 1_000_000 * config.COST_OUTPUT_PER_MTOK, 2)
        m["tokens_per_second"] = round(
            (m["tout"] or 0) / max((m["avg_ms"] or 1) * (m["calls"] or 1) / 1000, 0.001), 1)
    total = {
        "calls": sum(m["calls"] for m in by_model),
        "tokens_in": sum(m["tin"] or 0 for m in by_model),
        "tokens_out": sum(m["tout"] or 0 for m in by_model),
        "cost_rub": round(sum(m["cost_rub"] for m in by_model), 2),
        "errors": sum(m["errors"] for m in by_model),
    }
    hourly = [dict(r) for r in db.tq("""
        SELECT substr(ts, 1, 13) hour, COUNT(*) calls,
               SUM(tokens_in + tokens_out) tokens, AVG(latency_ms) avg_ms
        FROM model_usage WHERE ts > ? GROUP BY hour ORDER BY hour""", (since,))]
    return {"by_model": by_model, "total": total, "hourly": hourly}


def usage_report(hours: int = 168) -> dict:
    """Как пользуются ассистентом."""
    since = _since(hours)
    total = db.q1("SELECT COUNT(*) n, AVG(latency_ms) ms FROM queries WHERE created_at > ?",
                  (since,))
    unanswered = db.q1("SELECT COUNT(*) n FROM queries WHERE created_at > ? AND answered=0",
                       (since,))["n"]
    # feedback и queries живут в ОСНОВНОЙ базе, stage_timings — в базе
    # телеметрии. Перепутанный помощник здесь не даёт неверных чисел: он
    # роняет весь обзорный экран сообщением «нет такой таблицы».
    feedback = db.q1("""SELECT SUM(verdict='up') up, SUM(verdict='down') down
                        FROM feedback WHERE created_at > ?""", (since,))
    by_user = [dict(r) for r in db.q("""
        SELECT COALESCE(user_name, 'без имени') who, COUNT(*) n,
               AVG(top_score) score, SUM(CASE WHEN answered=0 THEN 1 ELSE 0 END) misses
        FROM queries WHERE created_at > ? GROUP BY user_id ORDER BY n DESC LIMIT 20""",
        (since,))]
    by_hour = [dict(r) for r in db.q("""
        SELECT substr(created_at, 12, 2) hour, COUNT(*) n
        FROM queries WHERE created_at > ? GROUP BY hour ORDER BY hour""", (since,))]
    by_day = [dict(r) for r in db.q("""
        SELECT substr(created_at, 1, 10) day, COUNT(*) n,
               SUM(CASE WHEN answered=0 THEN 1 ELSE 0 END) misses,
               AVG(latency_ms) ms
        FROM queries WHERE created_at > ? GROUP BY day ORDER BY day""", (since,))]
    latencies = [r["latency_ms"] for r in
                 db.q("SELECT latency_ms FROM queries WHERE created_at > ? "
                      "AND latency_ms IS NOT NULL ORDER BY latency_ms", (since,))]

    def pct(p: float) -> int:
        if not latencies:
            return 0
        return int(latencies[min(int(len(latencies) * p), len(latencies) - 1)])

    # Какие разделы базы реально попадают в ответы.
    sections: dict[str, int] = {}
    for r in db.q("SELECT sources_json FROM queries WHERE created_at > ? "
                  "AND sources_json IS NOT NULL", (since,)):
        for src in json.loads(r["sources_json"] or "[]"):
            path = src.get("path", "")
            section = path.split("/")[0] if "/" in path else "—"
            sections[section] = sections.get(section, 0) + 1

    stages = [dict(r) for r in db.tq("""
        SELECT stage, COUNT(*) n, AVG(ms) avg_ms, MAX(ms) max_ms
        FROM stage_timings WHERE ts > ? GROUP BY stage ORDER BY avg_ms DESC""", (since,))]

    return {
        "queries": total["n"] or 0,
        "avg_latency_ms": int(total["ms"] or 0),
        "p50_ms": pct(0.5), "p95_ms": pct(0.95), "p99_ms": pct(0.99),
        "unanswered": unanswered,
        "answer_rate": round(1 - unanswered / max(total["n"] or 1, 1), 3),
        "up": int(feedback["up"] or 0), "down": int(feedback["down"] or 0),
        "by_user": by_user, "by_hour": by_hour, "by_day": by_day,
        "sections": sorted(sections.items(), key=lambda x: -x[1]),
        "stages": stages,
    }


def index_report() -> dict:
    """Состояние базы знаний в цифрах."""
    def one(sql: str, params=()) -> int:
        row = db.q1(sql, params)
        return int(list(row)[0]) if row else 0
    by_kind = [dict(r) for r in db.q("""
        SELECT COALESCE(asset_kind, kind) k, COUNT(*) n, SUM(text_chars) chars
        FROM documents WHERE status='ok' GROUP BY k ORDER BY n DESC""")]
    by_section = [dict(r) for r in db.q("""
        SELECT COALESCE(section,'—') s, COUNT(*) n FROM documents
        WHERE status='ok' GROUP BY s ORDER BY n DESC""")]
    return {
        "documents": one("SELECT COUNT(*) FROM documents WHERE status='ok'"),
        "chunks": one("SELECT COUNT(*) FROM chunks"),
        "vectors": len(db.vectors()),
        "products": one("SELECT COUNT(*) FROM products WHERE is_current=1"),
        "needs_ocr": one("SELECT COUNT(*) FROM documents WHERE needs_ocr=1"),
        "errors": one("SELECT COUNT(*) FROM documents WHERE status='error'"),
        "duplicates": one("SELECT COUNT(*) FROM documents WHERE status='duplicate'"),
        "outdated": one("SELECT COUNT(*) FROM documents WHERE is_current=0"),
        "golden": one("SELECT COUNT(*) FROM golden_qa WHERE active=1"),
        "training_pairs": one("SELECT COUNT(*) FROM training_pairs"),
        "chars": one("SELECT COALESCE(SUM(text_chars),0) FROM documents WHERE status='ok'"),
        "by_kind": by_kind, "by_section": by_section,
    }


def full_report(hours: int = 168) -> dict:
    ensure_tables()
    import models as models_mod
    return {
        "generated": _now(),
        "hardware": models_mod.hardware(),
        "model_server": models_mod.status(),
        "server": server_series(min(hours, 72)),
        "models": model_report(hours),
        "usage": usage_report(hours),
        "index": index_report(),
    }


def prometheus() -> str:
    """Те же показатели в формате, который понимает Prometheus."""
    ensure_tables()
    idx = index_report()
    use = usage_report(24)
    lines = [
        "# HELP kb_documents Документов в индексе",
        "# TYPE kb_documents gauge",
        f"kb_documents {idx['documents']}",
        "# HELP kb_chunks Фрагментов в индексе",
        "# TYPE kb_chunks gauge",
        f"kb_chunks {idx['chunks']}",
        "# HELP kb_queries_24h Вопросов за сутки",
        "# TYPE kb_queries_24h gauge",
        f"kb_queries_24h {use['queries']}",
        "# HELP kb_answer_rate Доля отвеченных вопросов",
        "# TYPE kb_answer_rate gauge",
        f"kb_answer_rate {use['answer_rate']}",
        "# HELP kb_latency_p95_ms Задержка ответа, 95-й перцентиль",
        "# TYPE kb_latency_p95_ms gauge",
        f"kb_latency_p95_ms {use['p95_ms']}",
    ]
    row = db.tq1("SELECT * FROM server_metrics ORDER BY id DESC LIMIT 1")
    if row:
        lines += [
            "# TYPE kb_cpu_percent gauge", f"kb_cpu_percent {row['cpu_percent']}",
            "# TYPE kb_ram_used_gb gauge", f"kb_ram_used_gb {row['ram_used_gb']}",
        ]
        for card in json.loads(row["gpu_json"] or "[]"):
            i = card["index"]
            lines += [
                f'kb_gpu_utilization{{gpu="{i}"}} {card.get("utilization", 0)}',
                f'kb_gpu_memory_used_gb{{gpu="{i}"}} {card.get("memory_used_gb", 0)}',
                f'kb_gpu_temperature{{gpu="{i}"}} {card.get("temperature", 0)}',
            ]
    for m in model_report(24)["by_model"]:
        safe = m["model"].replace('"', "")
        lines.append(f'kb_model_calls_24h{{model="{safe}"}} {m["calls"]}')
        lines.append(f'kb_model_latency_ms{{model="{safe}"}} {m["avg_ms"]}')
    return "\n".join(lines) + "\n"


def daemon(interval: int | None = None) -> None:
    interval = interval or config.METRICS_INTERVAL_SECONDS
    log.info("сбор метрик запущен, интервал %d с", interval)
    while True:
        try:
            collect()
        except Exception as exc:  # noqa: BLE001
            log.warning("сбой сбора метрик: %s", exc)
        time.sleep(interval)


def main() -> int:
    p = argparse.ArgumentParser(description="Метрики и аналитика")
    p.add_argument("command", choices=["collect", "daemon", "report", "series",
                                       "prometheus"], nargs="?", default="report")
    p.add_argument("what", nargs="?", default="cpu")
    p.add_argument("--hours", type=int, default=168)
    p.add_argument("--interval", type=int)
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    db.init()
    logging_setup.setup()

    if a.command == "collect":
        row = collect()
        print(f"Процессор {row['cpu_percent']}%, память {row['ram_used_gb']}/"
              f"{row['ram_total_gb']} ГБ, диск {row['disk_used_gb']}/{row['disk_total_gb']} ГБ")
        for card in row["gpus"]:
            print(f"  Видеокарта {card['index']}: загрузка {card['utilization']}%, "
                  f"память {card['memory_used_gb']}/{card['memory_total_gb']} ГБ, "
                  f"{card['temperature']}°C, {card['power_w']} Вт")
    elif a.command == "daemon":
        daemon(a.interval)
    elif a.command == "prometheus":
        print(prometheus())
    elif a.command == "series":
        print(json.dumps(server_series(a.hours), ensure_ascii=False, indent=2))
    else:
        rep = full_report(a.hours)
        if a.json:
            print(json.dumps(rep, ensure_ascii=False, indent=2))
            return 0
        u, m, i = rep["usage"], rep["models"], rep["index"]
        print("=" * 68)
        print(f"СВОДКА ЗА {a.hours} ЧАСОВ")
        print("=" * 68)
        print(f"\nБаза: документов {i['documents']}, фрагментов {i['chunks']}, "
              f"позиций прайса {i['products']}")
        print(f"      сканов без OCR {i['needs_ocr']}, ошибок {i['errors']}, "
              f"дублей {i['duplicates']}, устаревших {i['outdated']}")
        print(f"\nВопросов: {u['queries']}, отвечено {u['answer_rate'] * 100:.0f}%, "
              f"👍 {u['up']} 👎 {u['down']}")
        print(f"Задержка: медиана {u['p50_ms']} мс, 95-й перцентиль {u['p95_ms']} мс, "
              f"99-й {u['p99_ms']} мс")
        if u["stages"]:
            print("\nПо этапам:")
            for s in u["stages"]:
                print(f"  {s['stage']:<16} среднее {int(s['avg_ms'])} мс, "
                      f"максимум {s['max_ms']} мс, вызовов {s['n']}")
        if m["by_model"]:
            print("\nМодели:")
            for x in m["by_model"]:
                print(f"  {x['model']:<28} вызовов {x['calls']:>6}, "
                      f"токенов {(x['tin'] or 0) + (x['tout'] or 0):>9}, "
                      f"среднее {x['avg_ms']:>5} мс, расход {x['cost_rub']:>8} ₽")
            print(f"  ИТОГО расход: {m['total']['cost_rub']} ₽")
        if u["sections"]:
            print("\nОткуда берутся ответы:")
            for name, n in u["sections"][:8]:
                print(f"  {name[:44]:<46} {n}")
        hw = rep["hardware"]
        print(f"\nСервер: {hw['cpu_model']} ({hw['cpu_cores']} ядер), "
              f"память {hw['ram_gb']} ГБ, свободно на диске {hw['disk_free_gb']} ГБ")
        for g in hw["gpus"]:
            print(f"  Видеокарта {g['index']}: {g['name']}, "
                  f"{g['memory_used_gb']}/{g['memory_total_gb']} ГБ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
