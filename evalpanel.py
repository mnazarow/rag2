"""
Ведение набора контрольных вопросов из веб-интерфейса.

Набор — это eval/golden.jsonl, и он остаётся обычным файлом: его можно
править руками, хранить в git и переносить между установками. Панель
только читает и переписывает его — атомарно, сохраняя строки-комментарии
в шапке.

Зачем отдельный раздел. Следующий шаг качества — 150–200 контрольных
вопросов от эксперта, а эксперт не работает в терминале. Здесь он
добавляет вопросы, сразу видит слабости набора (аудит), берёт кандидатов
прямо из журнала неотвеченных вопросов и запускает замер кнопкой.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import config
import db
import logging_setup

log = logging_setup.get("web")

# Рабочий объём набора: ниже — измерение шумит, больше — можно.
TARGET_QUESTIONS = 150

FIELDS = ("question", "expect_files", "expect_text", "reject_files", "reject_text")


def dataset_path() -> Path:
    return Path(config.BASE_DIR) / "eval" / "golden.jsonl"


def load() -> dict:
    """Читает набор: комментарии шапки отдельно, вопросы отдельно."""
    path = dataset_path()
    comments: list[str] = []
    items: list[dict] = []
    broken: list[str] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("//"):
                comments.append(line)
                continue
            try:
                row = json.loads(stripped)
                items.append({f: row.get(f) for f in FIELDS if row.get(f)})
            except json.JSONDecodeError:
                broken.append(stripped[:80])
    return {"comments": comments, "items": items, "broken": broken}


def _write(comments: list[str], items: list[dict]) -> None:
    lines = list(comments)
    for it in items:
        row = {f: it[f] for f in FIELDS if it.get(f)}
        lines.append(json.dumps(row, ensure_ascii=False))
    body = ("\n".join(lines) + "\n").encode("utf-8")
    path = dataset_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    db.atomic_write(path, lambda fh: fh.write(body))


def _clean_item(raw: dict) -> dict:
    """Проверяет и приводит вопрос к каноническому виду."""
    question = str(raw.get("question") or "").strip()
    if len(question) < 5:
        raise ValueError("вопрос слишком короткий")
    item: dict = {"question": question}
    for f in FIELDS[1:]:
        value = raw.get(f)
        if isinstance(value, str):
            value = [x.strip() for x in value.split(",")]
        if not isinstance(value, list):
            continue
        value = [str(x).strip() for x in value if str(x).strip()]
        if value:
            item[f] = value
    if not item.get("expect_files"):
        raise ValueError("нужен expect_files: часть пути документа, "
                         "в котором лежит ответ (лучше с брендом и моделью)")
    return item


def save_item(raw: dict, index: int | None = None) -> dict:
    """Добавляет вопрос или заменяет существующий (index с нуля)."""
    data = load()
    item = _clean_item(raw)
    dup = next((i for i, it in enumerate(data["items"])
                if it["question"].lower() == item["question"].lower()
                and i != index), None)
    if dup is not None:
        raise ValueError(f"такой вопрос уже есть (№{dup + 1})")
    if index is None:
        data["items"].append(item)
    else:
        if not 0 <= index < len(data["items"]):
            raise ValueError("нет такого вопроса")
        data["items"][index] = item
    _write(data["comments"], data["items"])
    return {"count": len(data["items"]), "item": item}


def delete_item(index: int) -> dict:
    data = load()
    if not 0 <= index < len(data["items"]):
        raise ValueError("нет такого вопроса")
    removed = data["items"].pop(index)
    _write(data["comments"], data["items"])
    return {"count": len(data["items"]), "removed": removed["question"]}


def candidates(limit: int = 20) -> list[dict]:
    """
    Кандидаты в контрольные вопросы — из журнала.

    Лучший источник эталонов — реальные вопросы сотрудников, на которые
    бот не ответил или ответил с низкой уверенностью: именно их эксперт
    и должен превратить в контрольные, указав, где лежит ответ.
    """
    existing = {it["question"].lower() for it in load()["items"]}
    out: list[dict] = []
    try:
        rows = db.q("""SELECT question, COUNT(*) n, MAX(created_at) last_at,
                              MIN(answered) answered
                       FROM queries
                       WHERE LENGTH(question) BETWEEN 10 AND 200
                       GROUP BY LOWER(question)
                       ORDER BY MIN(answered) ASC, n DESC, last_at DESC
                       LIMIT ?""", (limit * 3,))
    except Exception:  # noqa: BLE001
        return []
    for r in rows:
        q = r["question"].strip()
        if q.lower() in existing or q.startswith("/"):
            continue
        out.append({"question": q, "asked": r["n"],
                    "answered": bool(r["answered"]), "last_at": r["last_at"]})
        if len(out) >= limit:
            break
    return out


def make_twin(item: dict) -> dict:
    """
    Заготовка вопроса-двойника: тот же вопрос про соседнюю модель.

    Числовые подписи модели из вопроса переезжают в запреты, а поля
    ожиданий эксперт заполняет цифрами соседней модели. Смысл пары: она
    ловит самую дорогую ошибку — уверенный ответ про другой товар.
    """
    signature_rx = re.compile(r"\d+(?:[.,]\d+)?(?:[/\-]\d+(?:[.,]\d+)?)+")
    signatures = signature_rx.findall(item.get("question") or "")
    return {
        "question": signature_rx.sub("«МОДЕЛЬ-СОСЕД»", item.get("question") or ""),
        "expect_files": [],
        "expect_text": [],
        "reject_files": item.get("expect_files") or [],
        "reject_text": (item.get("expect_text") or []) + signatures,
    }


def state() -> dict:
    """Всё, что нужно разделу: набор, аудит, кандидаты, история замеров."""
    import evaluate
    import regression
    data = load()
    problems: list[str] = []
    try:
        problems = evaluate.audit(data["items"])
    except Exception as exc:  # noqa: BLE001
        problems = [f"аудит не отработал: {exc}"]
    try:
        runs = regression.history(10)
    except Exception:  # noqa: BLE001
        runs = []
    count = len(data["items"])
    with_text = sum(1 for it in data["items"]
                    if it.get("expect_text") or it.get("reject_text"))
    with_twins = sum(1 for it in data["items"]
                     if it.get("reject_files") or it.get("reject_text"))
    return {
        "path": str(dataset_path()),
        "items": data["items"],
        "broken": data["broken"],
        "count": count,
        "target": TARGET_QUESTIONS,
        "with_text": with_text,
        "with_twins": with_twins,
        "problems": problems,
        "candidates": candidates(),
        "runs": runs,
    }
