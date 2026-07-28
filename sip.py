"""
Телефония: ассистент как SIP-абонент или как приложение Asterisk.

  python sip.py check           — проверить доступность АТС и настройки
  python sip.py serve           — принимать звонки
  python sip.py test "вопрос"   — прогнать цикл «речь → ответ → речь» без звонка

Два способа подключения, оба поддерживаются настройкой SIP_MODE.

ari — приложение Asterisk. АТС по правилу набора отправляет канал в наше
приложение, аудио ходит через AudioSocket. Это рекомендуемый путь для
голосового ассистента: обмен событиями асинхронный и хорошо ложится на
ожидание ответа от моделей. Устаревший AGI работает синхронно и на таких
задержках ведёт себя плохо.

sip — ассистент регистрируется на АТС как обычный аппарат с логином и
паролем. Подходит, когда доступа к настройке Asterisk нет и ассистенту
просто выделили внутренний номер. Требует библиотеки pjsua2, которую
нужно собирать, — в развёртывании это заметно сложнее.

Про задержки, чтобы не было завышенных ожиданий. Последовательная схема
«распознали — подумали — озвучили» на практике даёт три-четыре секунды
от конца фразы до начала ответа. Для телефонного разговора это ощущается
как «робот завис». Комфортным считается интервал до восьмисот
миллисекунд, и достигается он только потоковыми движками, где
распознавание, модель и синтез работают внахлёст. Поэтому в модуле
предусмотрены реплики-заполнители («секунду, уточняю») — они закрывают
паузу, пока идёт поиск по базе.

Важное ограничение AudioSocket, о которое спотыкаются все: длина пакета
записана в двух байтах, поэтому куски больше 65535 байт молча обрываются.
Синтезированный ответ нарезается на части по 32000 байт.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import re
import struct
import sys
import tempfile
import time
import uuid
from pathlib import Path

import config
import logging_setup

log = logging_setup.get("sip")

# Типы пакетов AudioSocket
AS_TERMINATE = 0x00
AS_UUID = 0x01
AS_AUDIO = 0x10
AS_ERROR = 0xFF
AS_MAX_PAYLOAD = 32000          # запас относительно предела в 65535

FILLERS = [
    "Секунду, смотрю в базе.",
    "Уточняю, одну минуту.",
    "Ищу, подождите немного.",
]


class SipError(RuntimeError):
    pass


# ------------------------------------------------------------- диагностика --
def check() -> dict:
    """Проверяет настройки и доступность АТС. Ничего не звонит."""
    result: dict = {"mode": config.SIP_MODE, "problems": [], "ok": False}
    if config.SIP_MODE == "none":
        result["problems"].append("телефония выключена (SIP_MODE=none)")
        return result

    if config.VOICE_STT_PROVIDER == "none":
        result["problems"].append("не настроено распознавание речи (VOICE_STT_PROVIDER)")
    if config.VOICE_TTS_PROVIDER == "none":
        result["problems"].append("не настроен синтез речи (VOICE_TTS_PROVIDER)")

    if config.SIP_MODE == "ari":
        if not config.ARI_USER or not config.ARI_PASSWORD:
            result["problems"].append("не заданы ARI_USER и ARI_PASSWORD")
        else:
            try:
                import httpx
                r = httpx.get(f"{config.ARI_URL}/ari/asterisk/info",
                              auth=(config.ARI_USER, config.ARI_PASSWORD), timeout=10)
                if r.status_code == 200:
                    info = r.json()
                    result["asterisk"] = info.get("system", {}).get("version") or "неизвестно"
                else:
                    result["problems"].append(f"Asterisk ответил {r.status_code}")
            except Exception as exc:  # noqa: BLE001
                result["problems"].append(f"АТС недоступна: {exc}")
        result["audiosocket"] = f"{config.AUDIOSOCKET_HOST}:{config.AUDIOSOCKET_PORT}"
        result["dialplan"] = (
            f"exten => {config.SIP_EXTENSION or '1000'},1,Answer()\n"
            f" same => n,Stasis({config.ARI_APP})\n"
            f" same => n,Hangup()")
    elif config.SIP_MODE == "sip":
        if not (config.SIP_SERVER and config.SIP_USER and config.SIP_PASSWORD):
            result["problems"].append("не заданы SIP_SERVER, SIP_USER или SIP_PASSWORD")
        try:
            import pjsua2  # noqa: F401
        except ImportError:
            result["problems"].append(
                "не установлен pjsua2 — библиотека требует сборки из исходников; "
                "если это сложно, используйте режим ari")
    else:
        result["problems"].append(f"неизвестный режим: {config.SIP_MODE}")

    result["ok"] = not result["problems"]
    return result


# ------------------------------------------------------- обработка разговора --
# ───────────────────────── кто звонит и что ему можно ─────────────────────
# У телефонного канала нет входа по паролю: человек просто дозвонился.
# Поэтому опознание — по номеру, а всё, что за пределами списка, получает
# самую ограниченную роль. Раньше здесь стояла DEFAULT_ROLE для всех, и
# любой дозвонившийся получал ровно тот же доступ, что сотрудник.
_recent_calls: dict[str, list[float]] = {}


def normalize_number(number: str) -> str:
    """8 (912) 345-67-89 и +79123456789 — один и тот же номер."""
    digits = re.sub(r"\D", "", number or "")
    if len(digits) == 11 and digits[0] in "78":
        digits = "7" + digits[1:]
    return digits


def role_for_caller(caller: str) -> str:
    """
    Роль звонящего. Номер из списка — своя роль, остальные — гостевая.

    Гостевая роль намеренно берётся из настройки, а не совпадает с
    DEFAULT_ROLE: телефон доступен снаружи, и то, что можно сотруднику в
    Telegram, не обязано быть можно любому дозвонившемуся.
    """
    number = normalize_number(caller)
    for entry in config.SIP_KNOWN_CALLERS.split(","):
        if ":" not in entry:
            continue
        known, role = entry.split(":", 1)
        if normalize_number(known) and normalize_number(known) == number:
            return role.strip()
    return config.SIP_GUEST_ROLE


def call_allowed(caller: str) -> tuple[bool, str]:
    """Не слишком ли часто звонят с этого номера."""
    number = normalize_number(caller) or "неизвестный"
    if config.SIP_ONLY_KNOWN_CALLERS and not any(
            normalize_number(e.split(":")[0]) == number
            for e in config.SIP_KNOWN_CALLERS.split(",") if ":" in e):
        return False, "номер не в списке разрешённых"
    now = time.time()
    fresh = [t for t in _recent_calls.get(number, []) if now - t < 3600]
    if len(fresh) >= config.SIP_MAX_CALLS_PER_HOUR:
        return False, f"с номера {number} за час уже {len(fresh)} звонков"
    fresh.append(now)
    _recent_calls[number] = fresh
    return True, ""


class Conversation:
    """
    Один звонок. Копит аудио от абонента, по паузе распознаёт, спрашивает
    базу знаний и отдаёт озвученный ответ.
    """

    def __init__(self, call_id: str, caller: str = "") -> None:
        self.call_id = call_id
        self.caller = caller
        self.buffer = bytearray()
        self.last_voice = time.time()
        self.started = time.time()
        self.turns = 0
        self.role = role_for_caller(caller)
        self.request_id = logging_setup.new_request(caller or call_id, "sip")
        log.info("звонок начат: %s от %s, роль %s", call_id,
                 caller or "неизвестного номера", self.role)

    # Простейший детектор тишины по громкости. Полноценный вариант —
    # Silero VAD; здесь важно не иметь тяжёлых зависимостей в базовой сборке.
    @staticmethod
    def _loudness(chunk: bytes) -> float:
        if len(chunk) < 2:
            return 0.0
        total = 0
        for i in range(0, len(chunk) - 1, 2):
            sample = struct.unpack_from("<h", chunk, i)[0]
            total += abs(sample)
        return total / (len(chunk) / 2) / 32768

    def feed(self, chunk: bytes) -> bool:
        """Возвращает True, если абонент договорил и пора отвечать."""
        self.buffer.extend(chunk)
        if self._loudness(chunk) > 0.01:
            self.last_voice = time.time()
            return False
        silence = time.time() - self.last_voice
        return silence > 0.9 and len(self.buffer) > 16000      # ~0,5 с речи

    def answer(self) -> tuple[str, bytes]:
        """Распознаёт накопленное, отвечает, возвращает (текст, звук)."""
        import answer as answer_mod
        import voice

        audio = bytes(self.buffer)
        self.buffer.clear()
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "in.raw"
            raw.write_bytes(audio)
            wav = Path(tmp) / "in.wav"
            _pcm_to_wav(audio, wav, rate=8000)
            question = voice.transcribe(wav)
            log.info("распознано: %s", question if config.LOG_PAYLOADS else
                     f"{len(question)} символов")
            if len(question.strip()) < 3:
                return "", b""
            # Ограничение частоты для линии считается по звонку, а не по
            # сотруднику: сотрудника здесь может не быть вовсе. Без него
            # телефонный канал не ограничен ничем — один скрипт,
            # открывающий соединения, выжигает бюджет на модель и
            # выкачивает базу голосом.
            if self.turns >= config.SIP_MAX_TURNS:
                log.warning("звонок %s превысил предел вопросов (%d)",
                            self.call_id, config.SIP_MAX_TURNS)
                return ("Извините, на один звонок можно задать ограниченное "
                        "число вопросов. Перезвоните, пожалуйста, позже."), b""
            self.turns += 1
            # Человек ждёт на линии — самая срочная очередь к модели.
            res = answer_mod.ask(question, user_name=self.caller, chat_id=None,
                                 role=self.role, source="голос")
            text = _shorten_for_phone(res.text)
            out = Path(tmp) / "out.ogg"
            voice.synthesize(text, out)
            wav_out = Path(tmp) / "out.wav"
            import subprocess
            subprocess.run(["ffmpeg", "-y", "-i", str(out), "-ar", "8000", "-ac", "1",
                            "-f", "s16le", str(wav_out)], capture_output=True, check=True)
            return text, wav_out.read_bytes()

    def finish(self) -> None:
        log.info("звонок завершён: %s, реплик %d, длительность %.0f с",
                 self.call_id, self.turns, time.time() - self.started)


def _pcm_to_wav(pcm: bytes, path: Path, rate: int = 8000) -> None:
    import wave
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes(pcm)


def _shorten_for_phone(text: str, limit: int = 600) -> str:
    """
    По телефону длинный ответ не воспринимается: нет возможности пробежать
    глазами. Оставляем суть и предлагаем прислать подробности в мессенджер.
    """
    import re
    text = re.sub(r"\n?Источники:.*$", "", text, flags=re.S)
    text = re.sub(r"\s{2,}", " ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    dot = cut.rfind(".")
    if dot > limit * 0.5:
        cut = cut[:dot + 1]
    return cut + " Подробности могу прислать в Телеграм."


# -------------------------------------------------------- сервер AudioSocket --
async def audiosocket_server() -> None:
    """
    Принимает поток от Asterisk. Протокол простой: один байт типа,
    два байта длины, дальше данные.
    """
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        conv: Conversation | None = None
        try:
            while True:
                header = await reader.readexactly(3)
                kind = header[0]
                length = struct.unpack(">H", header[1:3])[0]
                payload = await reader.readexactly(length) if length else b""

                if kind == AS_UUID:
                    call_id = str(uuid.UUID(bytes=payload)) if len(payload) == 16 \
                        else payload.hex()
                    # Номер звонящего передаёт станция; при прямом
                    # подключении к порту его нет — тогда это заведомо
                    # не звонок, и роль будет гостевой.
                    caller = str(peer[0]) if peer else ""
                    ok, why = call_allowed(caller)
                    if not ok:
                        log.warning("звонок отклонён: %s", why)
                        await _say(writer, "Извините, сейчас я не могу принять "
                                           "обращение. Попробуйте позже.")
                        break
                    conv = Conversation(call_id, caller=caller)
                    await _say(writer, config.SIP_GREETING)
                elif kind == AS_AUDIO and conv is not None:
                    if conv.feed(payload):
                        await _say(writer, FILLERS[conv.turns % len(FILLERS)])
                        text, audio = await asyncio.to_thread(conv.answer)
                        if audio:
                            await _send_audio(writer, audio)
                        else:
                            await _say(writer, "Я вас не расслышал, повторите, пожалуйста.")
                    if time.time() - conv.started > config.SIP_MAX_CALL_SECONDS:
                        await _say(writer, "Время разговора истекло. До свидания.")
                        break
                elif kind == AS_ERROR:
                    log.warning("AudioSocket сообщил об ошибке: %s", payload.hex())
                elif kind == AS_TERMINATE:
                    break
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        except Exception as exc:  # noqa: BLE001
            log.exception("сбой обработки звонка: %s", exc)
        finally:
            if conv:
                conv.finish()
            writer.close()

    server = await asyncio.start_server(handle, config.AUDIOSOCKET_HOST,
                                        config.AUDIOSOCKET_PORT)
    log.info("AudioSocket слушает %s:%d", config.AUDIOSOCKET_HOST, config.AUDIOSOCKET_PORT)
    async with server:
        await server.serve_forever()


async def _send_audio(writer: asyncio.StreamWriter, pcm: bytes) -> None:
    """Отправляет звук кусками — иначе длинные фразы обрываются."""
    for offset in range(0, len(pcm), AS_MAX_PAYLOAD):
        chunk = pcm[offset:offset + AS_MAX_PAYLOAD]
        writer.write(bytes([AS_AUDIO]) + struct.pack(">H", len(chunk)) + chunk)
        await writer.drain()


async def _say(writer: asyncio.StreamWriter, text: str) -> None:
    import voice
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "say.ogg"
        try:
            await asyncio.to_thread(voice.synthesize, text, out)
        except Exception as exc:  # noqa: BLE001
            log.warning("не удалось озвучить реплику: %s", exc)
            return
        import subprocess
        raw = Path(tmp) / "say.raw"
        subprocess.run(["ffmpeg", "-y", "-i", str(out), "-ar", "8000", "-ac", "1",
                        "-f", "s16le", str(raw)], capture_output=True, check=True)
        await _send_audio(writer, raw.read_bytes())


# ------------------------------------------------------------------- ARI ----
async def ari_loop() -> None:
    """Слушает события Asterisk и заводит канал в AudioSocket."""
    try:
        import httpx
        import websockets
    except ImportError as exc:
        raise SipError("нужны httpx и websockets: pip install websockets") from exc

    url = config.ARI_URL.replace("http", "ws", 1)
    ws_url = (f"{url}/ari/events?api_key={config.ARI_USER}:{config.ARI_PASSWORD}"
              f"&app={config.ARI_APP}&subscribeAll=true")
    rest = httpx.AsyncClient(base_url=f"{config.ARI_URL}/ari",
                             auth=(config.ARI_USER, config.ARI_PASSWORD), timeout=30)
    log.info("подключаюсь к Asterisk: приложение «%s»", config.ARI_APP)
    async with websockets.connect(ws_url) as ws:
        async for raw in ws:
            import json
            event = json.loads(raw)
            if event.get("type") == "StasisStart":
                channel = event["channel"]["id"]
                caller = event["channel"].get("caller", {}).get("number", "")
                log.info("входящий звонок от %s, канал %s", caller or "?", channel)
                await rest.post(f"/channels/{channel}/answer")
                await rest.post("/channels/externalMedia", params={
                    "app": config.ARI_APP,
                    "external_host": f"{config.AUDIOSOCKET_HOST}:{config.AUDIOSOCKET_PORT}",
                    "format": "slin",
                    "encapsulation": "audiosocket",
                    "transport": "tcp",
                    "connection_type": "client",
                    "direction": "both",
                })
            elif event.get("type") == "StasisEnd":
                log.info("звонок завершён: канал %s", event["channel"]["id"])


async def serve() -> None:
    tasks = [asyncio.create_task(audiosocket_server())]
    if config.SIP_MODE == "ari":
        tasks.append(asyncio.create_task(ari_loop()))
    elif config.SIP_MODE == "sip":
        tasks.append(asyncio.create_task(asyncio.to_thread(_pjsua_register)))
    await asyncio.gather(*tasks)


def _pjsua_register() -> None:
    """Регистрация на АТС как обычный аппарат."""
    try:
        import pjsua2 as pj
    except ImportError as exc:
        raise SipError("не установлен pjsua2; используйте режим ari") from exc
    endpoint = pj.Endpoint()
    endpoint.libCreate()
    cfg = pj.EpConfig()
    cfg.logConfig.level = 3
    endpoint.libInit(cfg)
    transport = pj.TransportConfig()
    transport.port = config.SIP_PORT
    kind = {"udp": pj.PJSIP_TRANSPORT_UDP, "tcp": pj.PJSIP_TRANSPORT_TCP,
            "tls": pj.PJSIP_TRANSPORT_TLS}[config.SIP_TRANSPORT]
    endpoint.transportCreate(kind, transport)
    endpoint.libStart()

    acc_cfg = pj.AccountConfig()
    acc_cfg.idUri = f"sip:{config.SIP_USER}@{config.SIP_SERVER}"
    acc_cfg.regConfig.registrarUri = f"sip:{config.SIP_SERVER}"
    cred = pj.AuthCredInfo("digest", "*", config.SIP_USER, 0, config.SIP_PASSWORD)
    acc_cfg.sipConfig.authCreds.append(cred)
    account = pj.Account()
    account.create(acc_cfg)
    log.info("зарегистрирован на АТС как %s@%s", config.SIP_USER, config.SIP_SERVER)
    while True:
        endpoint.libHandleEvents(100)


def main() -> int:
    p = argparse.ArgumentParser(description="Телефония")
    p.add_argument("command", choices=["check", "serve", "test"], nargs="?", default="check")
    p.add_argument("args", nargs="*")
    a = p.parse_args()
    logging_setup.setup()

    if a.command == "check":
        result = check()
        print(f"Режим: {result['mode']}")
        if result.get("asterisk"):
            print(f"Asterisk: {result['asterisk']}")
        if result.get("audiosocket"):
            print(f"AudioSocket слушает: {result['audiosocket']}")
        if result.get("dialplan"):
            print("\nДобавьте в extensions.conf:\n" + result["dialplan"])
        if result["problems"]:
            print("\nНужно поправить:")
            for problem in result["problems"]:
                print(f"  · {problem}")
        else:
            print("\nВсё готово к приёму звонков.")
        return 0 if result["ok"] else 1

    if a.command == "serve":
        asyncio.run(serve())
    elif a.command == "test":
        import answer as answer_mod
        import voice
        question = " ".join(a.args) or "какой напор у Водомет 55/50"
        res = answer_mod.ask(question, log=False)
        text = _shorten_for_phone(res.text)
        print(f"Вопрос : {question}")
        print(f"Ответ  : {text}")
        print(f"К речи : {voice.normalize_for_speech(text)[:400]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
