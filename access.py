"""
Доступ сотрудников к боту.

Как это работает. Незнакомый человек, написавший боту, получает вежливый
отказ и предложение оставить заявку. Заявка появляется в админке, где
администратор видит имя, ник в Telegram и когда обратились, — и одним
нажатием выдаёт доступ, сразу назначая роль. Роль определяет, какие
разделы базы человек увидит: снабженец не должен читать дилерские цены.

Почему не просто список идентификаторов в настройках. Список работает,
пока людей пятеро. Дальше начинается: кто этот номер, когда добавили,
почему у него доступ к дилерскому разделу, кто уволился. Здесь на каждый
такой вопрос есть ответ в базе, а выдача и отзыв доступа — действие с
автором и датой, а не правка текстового файла на сервере.

Список TELEGRAM_ALLOWED_IDS продолжает работать и имеет приоритет: он
удобен для первичной настройки, когда админки ещё нет под рукой.
"""
from __future__ import annotations

from datetime import datetime, timezone

import config
import db
import logging_setup

log = logging_setup.get("bot")

STATUSES = {
    "new": "не обращался",
    "pending": "ждёт решения",
    "approved": "есть доступ",
    "denied": "отказано",
    "blocked": "заблокирован",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get(user_id: int) -> dict | None:
    row = db.q1("SELECT * FROM users WHERE user_id=?", (user_id,))
    return dict(row) if row else None


def ensure(user_id: int, user_name: str | None = None,
           full_name: str | None = None) -> dict:
    """Заводит карточку сотрудника при первом обращении."""
    row = get(user_id)
    if row is None:
        preapproved = user_id in config.TELEGRAM_ALLOWED_IDS \
            or user_id in config.TELEGRAM_ADMIN_IDS
        role = "admin" if user_id in config.TELEGRAM_ADMIN_IDS else config.DEFAULT_ROLE
        db.run("""INSERT INTO users(user_id, user_name, full_name, role, approved,
                  status, created_at, last_seen)
                  VALUES (?,?,?,?,?,?,?,?)""",
               (user_id, user_name, full_name, role, int(preapproved),
                "approved" if preapproved else "new", _now(), _now()))
        row = get(user_id)
    else:
        # Имя в Telegram человек меняет; держим карточку в актуальном виде.
        db.run("UPDATE users SET user_name=COALESCE(?, user_name), "
               "full_name=COALESCE(?, full_name), last_seen=? WHERE user_id=?",
               (user_name, full_name, _now(), user_id))
        row = get(user_id)
    return row


def is_allowed(user: dict) -> bool:
    """Пускать ли этого человека к базе."""
    if user["user_id"] in config.TELEGRAM_ADMIN_IDS:
        return True
    if user.get("status") == "blocked":
        return False
    # Список из настроек — быстрый способ выдать доступ до настройки админки.
    if config.TELEGRAM_ALLOWED_IDS and user["user_id"] in config.TELEGRAM_ALLOWED_IDS:
        return True
    return bool(user.get("approved"))


def request_access(user_id: int, comment: str = "") -> dict:
    """Сотрудник оставляет заявку. Повторная заявка не создаёт дубля."""
    user = ensure(user_id)
    if user.get("status") in ("approved",):
        return {"ok": True, "status": "approved",
                "message": "Доступ у вас уже есть."}
    if user.get("status") == "blocked":
        return {"ok": False, "status": "blocked",
                "message": "Доступ закрыт администратором."}
    if user.get("status") == "pending":
        return {"ok": True, "status": "pending",
                "message": "Заявка уже отправлена, ждёт решения администратора."}
    db.run("UPDATE users SET status='pending', requested_at=?, note=? WHERE user_id=?",
           (_now(), (comment or "").strip()[:300], user_id))
    log.info("заявка на доступ: %s (%s)", user_id, user.get("full_name") or "")
    return {"ok": True, "status": "pending",
            "message": "Заявка отправлена администратору. Как только её "
                       "подтвердят, я напишу вам сам."}


def decide(user_id: int, approve: bool, role: str | None = None,
           by: str = "админка", note: str = "") -> dict:
    """Решение администратора: выдать или отклонить доступ."""
    user = ensure(user_id)
    status = "approved" if approve else "denied"
    db.run("""UPDATE users SET approved=?, status=?, role=COALESCE(?, role),
              decided_at=?, decided_by=?, note=COALESCE(NULLIF(?,''), note)
              WHERE user_id=?""",
           (int(approve), status, role, _now(), by, note.strip()[:300], user_id))
    log.warning("доступ %s: %s (%s), роль %s, решил %s",
                "выдан" if approve else "отклонён", user_id,
                user.get("full_name") or "", role or user.get("role"), by)
    return get(user_id)


def block(user_id: int, by: str = "админка", note: str = "") -> dict:
    """
    Закрывает доступ. Отдельно от «отказано»: заблокированный не может
    подать заявку заново, отклонённый — может.
    """
    ensure(user_id)
    db.run("""UPDATE users SET approved=0, status='blocked', decided_at=?,
              decided_by=?, note=COALESCE(NULLIF(?,''), note) WHERE user_id=?""",
           (_now(), by, note.strip()[:300], user_id))
    log.warning("доступ заблокирован: %s, решил %s", user_id, by)
    return get(user_id)


def set_role(user_id: int, role: str, by: str = "админка") -> dict:
    ensure(user_id)
    db.run("UPDATE users SET role=?, decided_at=?, decided_by=? WHERE user_id=?",
           (role, _now(), by, user_id))
    log.info("роль изменена: %s → %s (%s)", user_id, role, by)
    return get(user_id)


def listing(status: str | None = None) -> list[dict]:
    """Все, кто когда-либо обращался, с их активностью."""
    sql = """SELECT u.*,
                    (SELECT COUNT(*) FROM queries q WHERE q.user_id=u.user_id) asked,
                    (SELECT MAX(created_at) FROM queries q WHERE q.user_id=u.user_id) last_q
             FROM users u"""
    params: tuple = ()
    if status:
        sql += " WHERE u.status=?"
        params = (status,)
    sql += " ORDER BY CASE u.status WHEN 'pending' THEN 0 WHEN 'approved' THEN 1 " \
           "ELSE 2 END, u.last_seen DESC"
    out = []
    for r in db.q(sql, params):
        item = dict(r)
        item["status_ru"] = STATUSES.get(item.get("status") or "new", item.get("status"))
        item["sections"] = sorted(config.ROLE_SECTIONS.get(item.get("role") or "", []))
        item["from_env"] = item["user_id"] in config.TELEGRAM_ALLOWED_IDS
        item["is_admin"] = item["user_id"] in config.TELEGRAM_ADMIN_IDS
        out.append(item)
    return out


def summary() -> dict:
    rows = listing()
    counts = {key: 0 for key in STATUSES}
    for r in rows:
        counts[r.get("status") or "new"] = counts.get(r.get("status") or "new", 0) + 1
    return {"total": len(rows), "counts": counts,
            "pending": counts.get("pending", 0),
            "open_to_all": not config.TELEGRAM_ALLOWED_IDS
            and not any(r["approved"] for r in rows)}


def pending_notice() -> str:
    """Текст для уведомления администраторам в Telegram."""
    rows = listing("pending")
    if not rows:
        return ""
    lines = ["Заявки на доступ к базе знаний:"]
    for r in rows[:10]:
        who = r.get("full_name") or r.get("user_name") or "без имени"
        nick = f" (@{r['user_name']})" if r.get("user_name") else ""
        lines.append(f"• {who}{nick}, ID {r['user_id']}")
        if r.get("note"):
            lines.append(f"   {r['note']}")
    lines.append("\nПодтвердить: раздел «Сотрудники» в веб-интерфейсе.")
    return "\n".join(lines)
