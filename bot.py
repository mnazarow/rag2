"""
Telegram-бот.

Логика ответов вынесена в чистые функции (handle_message / handle_callback),
поэтому она одинакова для двух режимов запуска:

  * aiogram 3 — если библиотека установлена (рекомендуется для продакшена:
    FSM, middleware, вебхуки, Mini Apps);
  * встроенный long-polling на httpx — если aiogram нет. Нужен, чтобы
    прототип запускался «из коробки» без установки зависимостей.

Функциональность:
  · белый список сотрудников и роли (роль ограничивает разделы базы);
  · ответ с нумерованными источниками;
  · кнопки 👍 / 👎 под каждым ответом — это и есть контур обучения;
  · кнопка «Файл» — прислать документ-первоисточник;
  · /expert <вопрос> | <ответ> — эксперт добавляет выверенный ответ;
  · /gaps — список вопросов без ответа (задачи на пополнение базы);
  · /stats — состояние индекса.
"""
from __future__ import annotations

import asyncio
import html
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import access
import answer as answer_mod
import config
import db
import logging_setup
import metrics

log = logging_setup.get("bot")

API = "https://api.telegram.org/bot{token}/{method}"

WELCOME = (
    "Здравствуйте! Я справочный бот по корпоративной базе знаний.\n\n"
    "Спрашивайте про оборудование, характеристики, паспорта, сертификаты, "
    "прайсы, регламенты 1С и Битрикс24. Например:\n"
    "• <i>какой напор у Водомет 55/75</i>\n"
    "• <i>цена на насос ВИНТОВИК 3</i>\n"
    "• <i>как оформить заказ клиента в 1С УТ</i>\n\n"
    "Под каждым ответом есть кнопки 👍/👎 — они помогают боту становиться точнее."
)

HELP = (
    "<b>Команды</b>\n"
    "/start — начало работы\n"
    "/voice — включить или выключить голосовые ответы\n"
    "/stats — что сейчас в индексе\n"
    "/gaps — вопросы, на которые бот не смог ответить\n"
    "/request — запросить доступ к базе знаний\n"
    "/учить вопрос | ответ — добавить выверенный ответ (для экспертов и "
    "сотрудников с признаком «дообучение»; /expert работает так же)\n"
    "/whoami — ваш id и роль"
)


def _reload_dependents() -> None:
    """Перечитать модули, зависящие от настроек, после правки .env."""
    import importlib
    for name in ("embeddings", "rerank", "llm", "llm_queue", "search", "security"):
        try:
            module = importlib.import_module(name)
            importlib.reload(module)
            if hasattr(module, "reset"):
                module.reset()
        except Exception as exc:  # noqa: BLE001
            log.warning("не удалось перечитать %s: %s", name, exc)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ----------------------------------------------------------- пользователи --
# Вся работа с доступом вынесена в access.py: там же она доступна админке.
def ensure_user(user_id: int, user_name: str | None = None,
                full_name: str | None = None) -> dict:
    return access.ensure(user_id, user_name, full_name)


def is_allowed(user: dict) -> bool:
    return access.is_allowed(user)


def access_rate_check(user_id: int) -> dict:
    """Проверка частоты обращений. Администраторов не ограничиваем."""
    if user_id in config.TELEGRAM_ADMIN_IDS:
        return {"ok": True}
    import security
    return security.rate_check(user_id)


def access_denied_text(user: dict) -> str:
    """
    Что видит человек, у которого доступа нет.

    Важно не бросать его в тупик: сотрудник, впервые написавший боту,
    должен понимать, что делать дальше, а не гадать, сломался бот или
    его не пустили.
    """
    status = user.get("status") or "new"
    if status == "blocked":
        return ("Доступ к базе знаний закрыт. Если это ошибка, обратитесь "
                "к администратору.")
    if status == "pending":
        return ("Заявка на доступ уже отправлена и ждёт решения администратора. "
                "Как только её подтвердят, я напишу вам сам.")
    if status == "denied":
        return ("По вашей заявке принято отрицательное решение. "
                "Если обстоятельства изменились, отправьте /request ещё раз "
                "с пояснением, для чего нужен доступ.")
    return ("База знаний доступна сотрудникам компании.\n"
            "Чтобы получить доступ, отправьте команду /request — "
            "я передам заявку администратору.\n"
            f"Ваш ID: <code>{user['user_id']}</code>")


# ---------------------------------------------------------- форматирование --
def format_answer(res: answer_mod.Answer) -> str:
    text = html.escape(res.text)

    # Источники — из общего нумерованного списка (прайсы + документы),
    # той же нумерации, что видела модель. Старый путь по res.hits
    # оставлен для ответов, собранных без этого списка.
    sources = res.sources or [
        {"n": i, "kind": "document", "file_name": h.file_name,
         "rel_path": h.rel_path, "page": h.page_from,
         "is_current": h.is_current}
        for i, h in enumerate(res.hits, 1)]
    src_block = ""
    if sources:
        lines = ["", "<b>Источники:</b>"]
        for s in sources:
            mark = "" if s.get("is_current", 1) else " ⚠️ устаревшая версия"
            if s.get("kind") == "price":
                mark = " (прайс-лист)" + mark
            page = f", с. {s['page']}" if s.get("page") else ""
            lines.append(f"{s['n']}. {html.escape(s['file_name'])}{page}{mark}")
            if s.get("rel_path"):
                lines.append(f"    <code>{html.escape(s['rel_path'])}</code>")
        src_block = "\n" + "\n".join(lines)

    # Лимит Telegram — 4096. Резать нужно текст ответа, а не хвост
    # склейки: слепой срез откусывал именно список источников, а если
    # попадал внутрь HTML-сущности или тега — Telegram отвечал 400
    # «can't parse entities», и сообщение молча пропадало.
    budget = 3900 - len(src_block)
    if len(text) > budget:
        cut = text[:max(budget, 0)]
        # Рвём по границе слова: внутри «&quot;» пробелов не бывает,
        # значит сущность не будет разрезана.
        space = cut.rfind(" ", max(0, len(cut) - 120))
        if space > 0:
            cut = cut[:space]
        text = cut + "…"
    return text + src_block


def answer_keyboard(query_id: int | None, res: answer_mod.Answer) -> dict:
    rows = []
    if query_id:
        rows.append([
            {"text": "👍 Помогло", "callback_data": f"fb:up:{query_id}"},
            {"text": "👎 Не то", "callback_data": f"fb:down:{query_id}"},
        ])
    if res.hits:
        rows.append([{"text": f"📎 Файл: {res.hits[0].file_name[:28]}",
                      "callback_data": f"doc:{res.hits[0].doc_id}"}])
        if len(res.hits) > 1:
            rows.append([{"text": f"📎 Файл: {res.hits[1].file_name[:28]}",
                          "callback_data": f"doc:{res.hits[1].doc_id}"}])
    return {"inline_keyboard": rows} if rows else {}


# ------------------------------------------------------------- обработчики --
def handle_message(user_id: int, chat_id: int, user_name: str | None,
                   full_name: str | None, text: str) -> dict:
    """Возвращает описание того, что отправить: {text, keyboard}."""
    user = ensure_user(user_id, user_name, full_name)
    text_raw = (text or "").strip()
    if not is_allowed(user):
        if text_raw.startswith("/request"):
            comment = text_raw[len("/request"):].strip()
            result = access.request_access(user_id, comment)
            out = {"text": result["message"]}
            if result["status"] == "pending":
                who = user.get("full_name") or user.get("user_name") or "без имени"
                nick = f" (@{user['user_name']})" if user.get("user_name") else ""
                out["notify_admins"] = (
                    "🔑 Заявка на доступ к базе знаний\n"
                    f"{html.escape(who)}{html.escape(nick)}, ID <code>{user_id}</code>"
                    + (f"\n<i>{html.escape(comment[:300])}</i>" if comment else "")
                    + "\n\nПодтвердить: раздел «Сотрудники» в веб-интерфейсе.")
            return out
        return {"text": access_denied_text(user)}

    text = (text or "").strip()
    if not text:
        return {"text": "Отправьте вопрос текстом."}

    # Ограничение частоты — только для вопросов, не для команд: посмотреть
    # свою роль или статистику можно всегда.
    if not text.startswith("/"):
        limit = access_rate_check(user_id)
        if not limit["ok"]:
            return {"text": limit["message"]}

    if text.startswith("/start"):
        return {"text": WELCOME}
    if text.startswith("/help"):
        return {"text": HELP}
    if text.startswith("/request"):
        return {"text": "Доступ у вас уже есть — можно просто задавать вопросы."}
    if text.startswith("/whoami"):
        return {"text": f"ID: <code>{user_id}</code>\nРоль: <b>{user['role']}</b>\n"
                        f"Доступные разделы: {', '.join(sorted(config.ROLE_SECTIONS.get(user['role'], {'—'})))}"}
    if text.startswith("/voice"):
        return {"text": toggle_voice(user_id)}
    if text.startswith("/stats"):
        return {"text": stats_text()}
    if text.startswith("/gaps"):
        return {"text": gaps_text()}
    if text.startswith(("/expert", "/учить", "/teach")):
        return {"text": handle_expert(user, text)}

    logging_setup.new_request(user_id, "telegram")
    res = answer_mod.ask(text, user_id=user_id, user_name=user_name,
                         role=user["role"], chat_id=chat_id)
    out = {"text": format_answer(res), "keyboard": answer_keyboard(res.query_id, res)}
    if wants_voice(user_id, False):
        try:
            import voice
            reply = config.DATA_DIR / f"reply_{user_id}_{int(time.time()*1000)}.ogg"
            voice.synthesize(res.text, reply)
            out["voice"] = str(reply)
        except Exception as exc:  # noqa: BLE001
            log.warning("не удалось озвучить ответ: %s", exc)
    return out


_voice_pref: dict[int, bool] = {}


def toggle_voice(user_id: int) -> str:
    """Сотрудник сам решает, хочет ли слышать ответы голосом."""
    if config.VOICE_TTS_PROVIDER == "none":
        return ("Синтез речи не настроен. Администратор включает его в разделе "
                "«Голос и АТС» веб-интерфейса.")
    current = _voice_pref.get(user_id, config.VOICE_REPLY_MODE == "always")
    _voice_pref[user_id] = not current
    return ("Буду отвечать и голосом тоже." if not current
            else "Перехожу на текстовые ответы.")


def wants_voice(user_id: int, incoming_was_voice: bool) -> bool:
    if config.VOICE_TTS_PROVIDER == "none":
        return False
    if config.VOICE_REPLY_MODE == "never":
        return False
    if user_id in _voice_pref:
        return _voice_pref[user_id]
    if config.VOICE_REPLY_MODE == "always":
        return True
    return incoming_was_voice


def handle_voice(user_id: int, chat_id: int, user_name: str | None,
                 full_name: str | None, audio_path: Path) -> dict:
    """
    Голосовое сообщение: распознать, ответить, при необходимости озвучить.
    Расшифровка всегда показывается текстом — сотрудник должен видеть,
    что именно услышал бот, иначе непонятно, почему ответ странный.
    """
    import voice
    logging_setup.new_request(user_id, "telegram-voice")
    user = ensure_user(user_id, user_name, full_name)
    if not is_allowed(user):
        return {"text": "Доступ к базе знаний выдаёт администратор."}
    try:
        question = voice.transcribe(audio_path)
    except Exception as exc:  # noqa: BLE001
        log.warning("не удалось распознать голосовое: %s", exc)
        return {"text": "Не получилось распознать голосовое сообщение: "
                        f"{logging_setup.mask(str(exc))}"}
    if len(question.strip()) < 3:
        return {"text": "Я не разобрал вопрос. Попробуйте ещё раз или напишите текстом."}

    res = answer_mod.ask(question, user_id=user_id, user_name=user_name,
                         role=user["role"], chat_id=chat_id, source="голос")
    prefix = f"<i>Услышал: {html.escape(question)}</i>\n\n"
    out = {"text": prefix + format_answer(res),
           "keyboard": answer_keyboard(res.query_id, res)}
    if wants_voice(user_id, True):
        try:
            reply = config.DATA_DIR / f"reply_{user_id}_{int(time.time()*1000)}.ogg"
            voice.synthesize(res.text, reply)
            out["voice"] = str(reply)
        except Exception as exc:  # noqa: BLE001
            log.warning("не удалось озвучить ответ: %s", exc)
    return out


def handle_expert(user: dict, text: str) -> str:
    """/expert (или /учить) вопрос | ответ — выверенный ответ в golden-базу.

    Право дают роль admin или признак «дообучение» — его администратор
    включает сотруднику в разделе «Телеграм» веб-интерфейса.
    """
    if not access.can_train(user):
        return ("Добавлять выверенные ответы могут эксперты и сотрудники "
                "с признаком «дообучение». Попросите администратора включить "
                "его вам в разделе «Телеграм» веб-интерфейса.")
    for prefix in ("/expert", "/учить", "/teach"):
        if text.startswith(prefix):
            payload = text[len(prefix):].strip()
            break
    if "|" not in payload:
        return ("Формат: <code>/учить вопрос | ответ</code>\n"
                "Пример: <code>/учить Какой напор у Водомет 55/75? | "
                "Напор 75 м, подача 55 л/мин.</code>")
    question, expert_answer = (p.strip() for p in payload.split("|", 1))
    if len(question) < 5 or len(expert_answer) < 5:
        return "Слишком короткий вопрос или ответ."
    gid = answer_mod.add_golden(question, expert_answer, author_id=user["user_id"])
    return (f"Добавлено в базу выверенных ответов (#{gid}).\n"
            "Теперь на похожий вопрос бот ответит именно так.")


def handle_callback(user_id: int, data: str) -> dict:
    """
    Обработка нажатий кнопок. Возвращает {alert} и опционально
    {send_document}.

    Доступ проверяется здесь заново, а не наследуется от сообщения, к
    которому прикреплена кнопка. Причина — в устройстве Telegram: нажатие
    кнопки это отдельный запрос с произвольным содержимым, и отправить
    его можно любым клиентом, не видя ни кнопки, ни самого сообщения. Без
    проверки достаточно было прислать `doc:1`, `doc:2`, … чтобы выкачать
    из базы любой документ, включая закрытые для этой роли разделы, — и
    это работало даже у заблокированного сотрудника.
    """
    user = ensure_user(user_id)
    if not is_allowed(user):
        return {"alert": "Нет доступа к базе знаний."}

    if data.startswith("fb:"):
        try:
            _, verdict, raw_id = data.split(":", 2)
            query_id = int(raw_id)
        except ValueError:
            return {"alert": ""}
        # Оценку ставит только автор вопроса. Иначе посторонний накручивает
        # обучающие пары и вызывает рассылку админам с чужим текстом.
        owner = db.q1("SELECT user_id FROM queries WHERE id=?", (query_id,))
        if owner is None or (owner["user_id"] is not None
                             and owner["user_id"] != user_id):
            return {"alert": "Эта оценка не ваша."}
        answer_mod.record_feedback(query_id, user_id, verdict)
        if verdict == "up":
            return {"alert": "Спасибо! Учтено."}
        return {"alert": "Записал. Вопрос уйдёт эксперту на разбор.",
                "notify_experts": query_id}

    if data.startswith("doc:"):
        try:
            doc_id = int(data.split(":", 1)[1])
        except ValueError:
            return {"alert": ""}
        row = db.q1("SELECT abs_path, file_name, size_bytes, section "
                    "FROM documents WHERE id=?", (doc_id,))
        if not row:
            return {"alert": "Документ не найден."}
        # Тот же фильтр разделов, что и в поиске: роль решает, какие папки
        # человек вообще видит. Ответ одинаковый и для «нет такого
        # документа», и для «он вам не положен» — иначе перебор номеров
        # показывает, что лежит в закрытом разделе.
        allowed = config.ROLE_SECTIONS.get(user.get("role") or config.DEFAULT_ROLE)
        if allowed is not None and "*" not in allowed \
                and (row["section"] or "") not in allowed:
            log.warning("сотрудник %s запросил файл из недоступного раздела «%s»",
                        user_id, row["section"])
            return {"alert": "Документ не найден."}
        size_mb = (row["size_bytes"] or 0) / 1024 / 1024
        if size_mb > config.TELEGRAM_MAX_DOC_MB:
            return {"alert": f"Файл {size_mb:.0f} МБ — больше лимита Telegram. "
                             f"Путь: {row['file_name']}"}
        return {"send_document": row["abs_path"], "alert": "Отправляю файл…"}
    return {"alert": ""}


def stats_text() -> str:
    docs = db.q1("SELECT COUNT(*) n FROM documents WHERE status='ok'")["n"]
    chunks = db.q1("SELECT COUNT(*) n FROM chunks")["n"]
    prods = db.q1("SELECT COUNT(*) n FROM products WHERE is_current=1")["n"]
    golden = db.q1("SELECT COUNT(*) n FROM golden_qa WHERE active=1")["n"]
    queries = db.q1("SELECT COUNT(*) n FROM queries")["n"]
    ups = db.q1("SELECT COUNT(*) n FROM feedback WHERE verdict='up'")["n"]
    downs = db.q1("SELECT COUNT(*) n FROM feedback WHERE verdict='down'")["n"]
    unans = db.q1("SELECT COUNT(*) n FROM queries WHERE answered=0")["n"]
    return (f"<b>Индекс</b>\nДокументов: {docs}\nФрагментов: {chunks}\n"
            f"Позиций прайса: {prods}\nВыверенных ответов: {golden}\n\n"
            f"<b>Использование</b>\nВопросов: {queries}\nБез ответа: {unans}\n"
            f"👍 {ups}   👎 {downs}")


def gaps_text(limit: int = 15) -> str:
    rows = answer_mod.unanswered_report(limit)
    if not rows:
        return "Вопросов без ответа нет."
    lines = ["<b>Вопросы, требующие пополнения базы:</b>"]
    for r in rows:
        mark = "👎" if r["dislikes"] else "∅"
        lines.append(f"{mark} {html.escape(r['question'][:120])}")
    lines.append("\nДобавить ответ: <code>/expert вопрос | ответ</code>")
    return "\n".join(lines)


# ------------------------------------------------ рантайм 1: встроенный ----
class SimpleBot:
    """Long-polling без внешних зависимостей (нужен только httpx)."""

    def __init__(self, token: str) -> None:
        import httpx
        self.token = token
        # Прокси задаётся одной строкой: socks5://... или http://...
        # Для socks5 нужен пакет httpx[socks].
        kwargs = {"timeout": 70}
        if config.TELEGRAM_PROXY:
            kwargs["proxy"] = config.TELEGRAM_PROXY
            print(f"Telegram через прокси: {config.TELEGRAM_PROXY.split('@')[-1]}")
        self.client = httpx.AsyncClient(**kwargs)
        self._sem = asyncio.Semaphore(8)
        self.offset = 0

    async def call(self, method: str, **params):
        r = await self.client.post(API.format(token=self.token, method=method), json=params)
        return r.json()

    async def send(self, chat_id: int, text: str, keyboard: dict | None = None,
                   reply_to: int | None = None):
        params = {"chat_id": chat_id, "text": text, "parse_mode": "HTML",
                  "disable_web_page_preview": True}
        if keyboard:
            params["reply_markup"] = keyboard
        if reply_to:
            params["reply_to_message_id"] = reply_to
        return await self.call("sendMessage", **params)

    async def send_document(self, chat_id: int, path: str):
        import httpx
        p = Path(path)
        if not p.exists():
            await self.send(chat_id, "Файл недоступен по пути на диске.")
            return
        with p.open("rb") as fh:
            files = {"document": (p.name, fh)}
            await self.client.post(API.format(token=self.token, method="sendDocument"),
                                   data={"chat_id": chat_id}, files=files)

    async def run(self):
        me = await self.call("getMe")
        if not me.get("ok"):
            raise SystemExit(f"Не удалось подключиться к Telegram: {me}")
        print(f"Бот запущен: @{me['result']['username']}")
        import shutdown
        while not shutdown.stopping():
            # Настройки правят в админке, а бот — отдельный процесс.
            # Дешёвая проверка (один stat файла) на каждом обороте: иначе
            # изменённый порог отказа или сменённый провайдер модели
            # доходят до бота только после ручного перезапуска, и никто
            # об этом не догадывается.
            if config.reload_if_changed():
                log.warning("настройки изменились — перечитал")
                _reload_dependents()
            metrics.beat("бот")
            try:
                upd = await self.call("getUpdates", offset=self.offset, timeout=50)
            except Exception as exc:  # noqa: BLE001
                log.warning("ошибка получения обновлений: %s", exc)
                await asyncio.sleep(3)
                continue
            for u in upd.get("result", []):
                self.offset = u["update_id"] + 1
                # Вопросы обрабатываются параллельно, а не по одному:
                # иначе десятый спрашивающий ждёт десять генераций подряд.
                # Ограничитель держит разумный предел, а настоящий предел
                # нагрузки на модель — общая межпроцессная очередь.
                asyncio.create_task(self._dispatch_safe(u))

    async def _dispatch_safe(self, u: dict) -> None:
        async with self._sem:
            try:
                await self._dispatch(u)
            except Exception as exc:  # noqa: BLE001
                # Молчание для сотрудника хуже любой ошибки: он не знает,
                # сломалось или медленно, и задаёт вопрос снова.
                log.exception("ошибка обработки обновления")
                chat_id = (u.get("message") or {}).get("chat", {}).get("id")
                if chat_id:
                    rid = u.get("update_id", "-")
                    try:
                        await self.send(chat_id,
                                        "Не получилось обработать сообщение. "
                                        "Попробуйте ещё раз; если повторится — "
                                        f"сообщите администратору код <code>{rid}</code>.")
                    except Exception:  # noqa: BLE001
                        pass

    async def download_file(self, file_id: str, dest: Path) -> bool:
        info = await self.call("getFile", file_id=file_id)
        if not info.get("ok"):
            return False
        path = info["result"]["file_path"]
        url = f"https://api.telegram.org/file/bot{self.token}/{path}"
        r = await self.client.get(url)
        if r.status_code != 200:
            return False
        dest.write_bytes(r.content)
        return True

    async def send_voice(self, chat_id: int, path: str) -> None:
        p = Path(path)
        if not p.exists():
            return
        with p.open("rb") as fh:
            await self.client.post(API.format(token=self.token, method="sendVoice"),
                                   data={"chat_id": chat_id}, files={"voice": (p.name, fh)})

    async def _dispatch(self, u: dict):
        message = u.get("message") or {}
        if "voice" in message or "audio" in message or "video_note" in message:
            m = message
            frm = m.get("from", {})
            await self.call("sendChatAction", chat_id=m["chat"]["id"], action="typing")
            media = m.get("voice") or m.get("audio") or m.get("video_note")
            tmp = config.DATA_DIR / f"in_{frm.get('id')}_{m.get('message_id', 0)}.oga"
            if not await self.download_file(media["file_id"], tmp):
                await self.send(m["chat"]["id"], "Не удалось скачать голосовое сообщение.")
                return
            out = await asyncio.to_thread(
                handle_voice, frm.get("id"), m["chat"]["id"], frm.get("username"),
                frm.get("first_name"), tmp)
            await self.send(m["chat"]["id"], out["text"], out.get("keyboard"),
                            reply_to=m["message_id"])
            if out.get("voice"):
                await self.send_voice(m["chat"]["id"], out["voice"])
            for f in (tmp, Path(out.get("voice") or "")):
                if f and f.exists():
                    f.unlink(missing_ok=True)
            return
        if "message" in u and "text" in u["message"]:
            m = u["message"]
            frm = m.get("from", {})
            await self.call("sendChatAction", chat_id=m["chat"]["id"], action="typing")
            out = await asyncio.to_thread(
                handle_message, frm.get("id"), m["chat"]["id"], frm.get("username"),
                frm.get("first_name"), m["text"])
            await self.send(m["chat"]["id"], out["text"], out.get("keyboard"),
                            reply_to=m["message_id"])
            if out.get("voice"):
                await self.send_voice(m["chat"]["id"], out["voice"])
        elif "callback_query" in u:
            cb = u["callback_query"]
            res = await asyncio.to_thread(handle_callback, cb["from"]["id"], cb.get("data", ""))
            await self.call("answerCallbackQuery", callback_query_id=cb["id"],
                            text=res.get("alert", ""))
            if res.get("send_document"):
                await self.send_document(cb["message"]["chat"]["id"], res["send_document"])
            if res.get("notify_admins"):
                for admin in config.TELEGRAM_ADMIN_IDS:
                    await self.send(admin, res["notify_admins"])
            if res.get("notify_experts"):
                for admin in config.TELEGRAM_ADMIN_IDS:
                    row = db.q1("SELECT question FROM queries WHERE id=?",
                                (res["notify_experts"],))
                    if row:
                        await self.send(admin, "👎 Плохой ответ на вопрос:\n"
                                               f"<i>{html.escape(row['question'][:300])}</i>\n\n"
                                               "Добавьте эталон: <code>/expert вопрос | ответ</code>")


# ------------------------------------------------------ рантайм 2: aiogram --
async def run_aiogram(token: str) -> None:
    from aiogram import Bot, Dispatcher, F
    from aiogram.client.default import DefaultBotProperties
    from aiogram.enums import ParseMode
    from aiogram.types import CallbackQuery, FSInputFile, InlineKeyboardMarkup, Message

    session = None
    if config.TELEGRAM_PROXY:
        from aiogram.client.session.aiohttp import AiohttpSession
        session = AiohttpSession(proxy=config.TELEGRAM_PROXY)
        print(f"Telegram через прокси: {config.TELEGRAM_PROXY.split('@')[-1]}")
    bot = Bot(token, session=session,
              default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher()

    @dp.message(F.text)
    async def on_message(message: Message) -> None:
        await bot.send_chat_action(message.chat.id, "typing")
        out = await asyncio.to_thread(
            handle_message, message.from_user.id, message.chat.id,
            message.from_user.username, message.from_user.first_name, message.text)
        kb = out.get("keyboard")
        markup = InlineKeyboardMarkup.model_validate(kb) if kb else None
        await message.reply(out["text"], reply_markup=markup,
                            disable_web_page_preview=True)
        if out.get("notify_admins"):
            for admin in config.TELEGRAM_ADMIN_IDS:
                await bot.send_message(admin, out["notify_admins"])
        if out.get("voice"):
            await bot.send_voice(message.chat.id, FSInputFile(out["voice"]))

    @dp.message(F.voice | F.audio | F.video_note)
    async def on_voice(message: Message) -> None:
        await bot.send_chat_action(message.chat.id, "typing")
        media = message.voice or message.audio or message.video_note
        tmp = config.DATA_DIR / f"in_{message.from_user.id}_{message.message_id}.oga"
        await bot.download(media, destination=tmp)
        out = await asyncio.to_thread(
            handle_voice, message.from_user.id, message.chat.id,
            message.from_user.username, message.from_user.first_name, tmp)
        kb = out.get("keyboard")
        await message.reply(out["text"],
                            reply_markup=InlineKeyboardMarkup.model_validate(kb) if kb else None,
                            disable_web_page_preview=True)
        if out.get("voice"):
            await bot.send_voice(message.chat.id, FSInputFile(out["voice"]))
        for f in (tmp, Path(out.get("voice") or "")):
            if f and f.exists():
                f.unlink(missing_ok=True)

    @dp.callback_query()
    async def on_callback(call: CallbackQuery) -> None:
        res = await asyncio.to_thread(handle_callback, call.from_user.id, call.data or "")
        await call.answer(res.get("alert", ""))
        if res.get("send_document"):
            await bot.send_document(call.message.chat.id, FSInputFile(res["send_document"]))
        if res.get("notify_experts"):
            row = db.q1("SELECT question FROM queries WHERE id=?", (res["notify_experts"],))
            for admin in config.TELEGRAM_ADMIN_IDS:
                if row:
                    await bot.send_message(
                        admin, "👎 Плохой ответ на вопрос:\n"
                               f"<i>{html.escape(row['question'][:300])}</i>\n\n"
                               "Добавьте эталон: <code>/expert вопрос | ответ</code>")

    me = await bot.get_me()
    print(f"Бот запущен (aiogram): @{me.username}")

    async def _pulse() -> None:
        while True:
            metrics.beat("бот")
            await asyncio.sleep(60)

    pulse = asyncio.create_task(_pulse())
    try:
        await dp.start_polling(bot)
    finally:
        pulse.cancel()


def main() -> None:
    import logging_setup as ls
    import preflight
    import shutdown

    ls.setup()
    db.init()
    report = preflight.check("бот")
    print("Проверка перед запуском:")
    print(preflight.render(report))
    if report["fatal"]:
        log.error("бот не запущен: %s", "; ".join(report["fatal"]))
        raise SystemExit(2)
    shutdown.install("бот")

    try:
        import aiogram  # noqa: F401
        asyncio.run(run_aiogram(config.TELEGRAM_BOT_TOKEN))
    except ImportError:
        print("aiogram не установлен — запускаю встроенный поллер "
              "(для продакшена: pip install aiogram)")
        asyncio.run(SimpleBot(config.TELEGRAM_BOT_TOKEN).run())
    except KeyboardInterrupt:
        pass
    log.warning("бот остановлен")


if __name__ == "__main__":
    main()
