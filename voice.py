"""
Голос: распознавание входящих сообщений и синтез ответов.

  python voice.py stt <файл>              — распознать
  python voice.py tts "текст" out.ogg     — озвучить
  python voice.py voices                  — список голосов
  python voice.py add-voice <имя> <файл>  — добавить голос по образцу
  python voice.py normalize "текст"       — показать текст после подготовки

Три отдельные задачи, которые часто путают.

Распознавание. Голосовое сообщение Telegram — это OGG с кодеком Opus,
для моделей его нужно привести к 16 кГц моно. На коротких репликах
задержка меньше секунды, чего для чата достаточно; поточное
распознавание нужно только в телефонии.

Синтез. Здесь важнее движка оказывается подготовка текста. Ни одна
модель не прочитает «SPL WRP-A 2ECO6-38» и «Ду50 PN16, 3,6 м³/ч» так,
чтобы это можно было понять на слух. Поэтому перед синтезом текст
проходит нормализацию: единицы измерения раскрываются словами, дробные
числа читаются по-русски, буквенно-цифровые обозначения проговариваются
по частям. Это отдельный слой, и он работает с любым движком.

Клонирование голоса. Технически доступно, но упирается не в технику.
Голос — биометрические персональные данные, а лучшие по качеству модели
(XTTS, Fish Speech) распространяются по некоммерческим лицензиям и для
работы компании непригодны. Поэтому клонирование выключено и требует
явного подтверждения того, что письменное согласие сотрудника получено.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import config
import logging_setup

log = logging_setup.get("voice")


class VoiceError(RuntimeError):
    pass


# ------------------------------------------------------------- конвертация --
def to_wav16k(src: Path, dst: Path) -> None:
    """Приводит любой звук к 16 кГц моно — то, что ждут модели."""
    subprocess.run(["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "1",
                    "-ar", "16000", "-c:a", "pcm_s16le", str(dst)],
                   capture_output=True, timeout=600, check=True)


def to_opus(src: Path, dst: Path) -> None:
    """Голосовое сообщение Telegram: OGG/Opus, 48 кГц."""
    subprocess.run(["ffmpeg", "-y", "-i", str(src), "-c:a", "libopus",
                    "-b:a", "32k", "-ar", "48000", "-ac", "1", str(dst)],
                   capture_output=True, timeout=600, check=True)


def duration(path: Path) -> float:
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                              "format=duration", "-of", "csv=p=0", str(path)],
                             capture_output=True, text=True, timeout=30).stdout
        return float(out.strip() or 0)
    except Exception:  # noqa: BLE001
        return 0.0


# ------------------------------------------------------------ распознавание --
def transcribe(path: Path) -> str:
    """Распознаёт короткое голосовое сообщение. Возвращает текст."""
    provider = config.VOICE_STT_PROVIDER
    if provider == "none":
        raise VoiceError("распознавание выключено (VOICE_STT_PROVIDER=none)")
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "in.wav"
        to_wav16k(path, wav)
        secs = duration(wav)
        log.debug("распознавание: %s, длительность %.1f с, провайдер %s",
                  path.name, secs, provider)

        if provider == "faster-whisper":
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise VoiceError("нужен faster-whisper: pip install faster-whisper") from exc
            model = _cached("fw", lambda: WhisperModel(
                config.VOICE_STT_MODEL,
                device="auto" if config.ASR_DEVICE == "auto" else config.ASR_DEVICE,
                compute_type="int8"))
            segments, _info = model.transcribe(str(wav), language=config.ASR_LANGUAGE,
                                               vad_filter=True)
            return " ".join(s.text.strip() for s in segments).strip()

        if provider == "whisper":
            try:
                import whisper
            except ImportError as exc:
                raise VoiceError("нужен openai-whisper") from exc
            model = _cached("w", lambda: whisper.load_model(config.VOICE_STT_MODEL))
            return model.transcribe(str(wav), language=config.ASR_LANGUAGE)["text"].strip()

        if provider == "gigaam":
            try:
                import gigaam
            except ImportError as exc:
                raise VoiceError("нужен gigaam: pip install gigaam") from exc
            model = _cached("gigaam", lambda: gigaam.load_model("v3_rnnt"))
            return str(model.transcribe(str(wav))).strip()

        if provider == "yandex":
            return _yandex_stt(wav)
        if provider == "sber":
            return _sber_stt(wav)
    raise VoiceError(f"неизвестный провайдер распознавания: {provider}")


_CACHE: dict = {}


def _cached(key: str, factory):
    if key not in _CACHE:
        log.info("загружаю модель «%s» — первый запуск может занять время", key)
        _CACHE[key] = factory()
    return _CACHE[key]


def _yandex_stt(wav: Path) -> str:
    import httpx
    data = wav.read_bytes()
    r = httpx.post(
        "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize",
        params={"lang": "ru-RU", "folderId": config.YANDEX_FOLDER_ID,
                "format": "lpcm", "sampleRateHertz": "16000"},
        headers={"Authorization": f"Api-Key {config.YANDEX_API_KEY}"},
        content=data[44:], timeout=120)
    r.raise_for_status()
    return r.json().get("result", "")


def _sber_stt(wav: Path) -> str:
    import httpx
    import llm as llm_mod
    token = llm_mod.GigaChatLLM()._auth()          # переиспользуем авторизацию
    r = httpx.post("https://smartspeech.sber.ru/rest/v1/speech:recognize",
                   params={"language": "ru-RU", "model": "general"},
                   headers={"Authorization": f"Bearer {token}",
                            "Content-Type": "audio/x-pcm;bit=16;rate=16000"},
                   content=wav.read_bytes()[44:], timeout=120, verify=False)
    r.raise_for_status()
    return " ".join(r.json().get("result", []))


# ------------------------------------------------ подготовка текста к речи --
UNITS = {
    "м³/ч": "кубометров в час", "м3/ч": "кубометров в час",
    "мм": "миллиметров", "см": "сантиметров",
    "м3/ч": "кубометров в час", "л/мин": "литров в минуту", "кВт": "киловатт",
    "Вт": "ватт", "В": "вольт", "А": "ампер", "бар": "бар", "МПа": "мегапаскаль",
    "кПа": "килопаскаль", "°C": "градусов Цельсия", "кг": "килограммов",
    "мбар": "миллибар", "об/мин": "оборотов в минуту", "Гц": "герц",
    "Ду": "диаметр условный", "Ру": "давление условное", "DN": "диаметр условный",
    "PN": "давление номинальное", "м": "метров", "л": "литров", "шт": "штук",
    "мин": "минут", "ч": "часов", "г": "граммов", "т": "тонн", "мА": "миллиампер",
    "кВА": "киловольт-ампер", "дБ": "децибел", "%": "процентов",
}
ABBR = {
    "УПД": "у-пэ-дэ", "АУПД": "а-у-пэ-дэ", "БТП": "бэ-тэ-пэ", "ХВС": "хэ-вэ-эс",
    "ГВС": "гэ-вэ-эс", "ЦО": "цэ-о", "УТ": "у-тэ", "1С": "один-эс",
    "НДС": "эн-дэ-эс", "ГОСТ": "гост", "ТУ": "тэ-у", "ЕАЭС": "е-а-э-эс",
    "PDF": "пэ-дэ-эф", "CAD": "кад", "DWG": "дэ-вэ-жэ",
}
LETTERS = {
    "A": "эй", "B": "би", "C": "си", "D": "ди", "E": "и", "F": "эф", "G": "джи",
    "H": "эйч", "I": "ай", "J": "джей", "K": "кей", "L": "эль", "M": "эм",
    "N": "эн", "O": "оу", "P": "пи", "Q": "кью", "R": "ар", "S": "эс",
    "T": "ти", "U": "ю", "V": "ви", "W": "дабл-ю", "X": "икс", "Y": "уай", "Z": "зет",
}


def _number_ru(text: str) -> str:
    """Дробные числа: «3,6» → «три и шесть десятых» читается плохо, оставляем
    «три запятая шесть» — так понятнее на слух и не искажает значение."""
    return re.sub(r"(\d+),(\d+)", r"\1 запятая \2", text)


def normalize_for_speech(text: str) -> str:
    """
    Готовит текст к озвучиванию. Порядок важен: сначала убираем разметку,
    потом раскрываем единицы, потом разбираем обозначения моделей.
    """
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[*_`#]+", "", text)
    text = re.sub(r"\[(\d+)\]", r" источник \1 ", text)          # ссылки на источники
    text = re.sub(r"https?://\S+", " ссылка ", text)

    for unit, spoken in sorted(UNITS.items(), key=lambda x: -len(x[0])):
        # Односимвольные единицы («В», «А», «м», «ч») раскрываются ТОЛЬКО
        # после числа: иначе заглавная «В» в начале предложения читалась
        # как «вольт» — стандартный отказ начинался со слов «вольт базе
        # знаний…». Многосимвольные — после числа или пробела, как раньше.
        if len(unit) == 1:
            text = re.sub(rf"(?<=\d)\s?{re.escape(unit)}(?=[\s,.;:)!?]|$)",
                          f" {spoken} ", text)
        else:
            text = re.sub(rf"(?<![А-Яа-яЁёA-Za-z]){re.escape(unit)}(?=[\s\d,.;:)!?]|$)",
                          f" {spoken} ", text)
    for abbr, spoken in ABBR.items():
        text = re.sub(rf"\b{re.escape(abbr)}\b", spoken, text)

    def spell(match: re.Match) -> str:
        token = match.group(0)
        parts = re.findall(r"[A-Za-z]+|\d+|[-/]", token)
        out = []
        for part in parts:
            if part.isdigit():
                out.append(part)
            elif part in "-/":
                out.append("тире" if part == "-" else "дробь")
            else:
                out.append(" ".join(LETTERS.get(ch.upper(), ch) for ch in part))
        return " " + " ".join(out) + " "

    # Обозначения вида WRP-A, 2ECO6-38, EVR32-3-2: буквы по алфавиту, цифры числом.
    text = re.sub(r"\b[A-Za-z0-9]*\d[A-Za-z0-9]*(?:[-/][A-Za-z0-9]+)*\b",
                  lambda m: spell(m) if re.search(r"[A-Za-z]", m.group(0)) else m.group(0),
                  text)
    # Латинские аббревиатуры-марки без цифр: SPL, WRP, EVR — тоже по буквам.
    text = re.sub(r"\b[A-Z]{2,6}(?:-[A-Z0-9]{1,4})*\b", spell, text)
    text = _number_ru(text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# ------------------------------------------------------------------ синтез --
def voices() -> list[dict]:
    """Доступные голоса: встроенные в движок плюс добавленные образцами."""
    out: list[dict] = []
    if config.VOICE_TTS_PROVIDER == "silero":
        for name in ("aidar", "baya", "kseniya", "xenia", "eugene"):
            out.append({"id": name, "title": name, "kind": "встроенный",
                        "provider": "silero"})
    if config.VOICE_TTS_PROVIDER == "yandex":
        for name in ("alena", "filipp", "ermil", "jane", "madirus", "omazh", "zahar"):
            out.append({"id": name, "title": name, "kind": "встроенный",
                        "provider": "yandex"})
    profiles = config.VOICE_PROFILES_DIR
    if profiles.exists():
        for meta_file in profiles.glob("*/voice.json"):
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                out.append({"id": meta["id"], "title": meta.get("title", meta["id"]),
                            "kind": "по образцу", "provider": meta.get("provider", "?"),
                            "sample_seconds": meta.get("sample_seconds"),
                            "consent": meta.get("consent", False),
                            "created": meta.get("created")})
            except Exception:  # noqa: BLE001
                continue
    return out


def synthesize(text: str, out_path: Path, voice: str | None = None) -> Path:
    """Озвучивает текст. Возвращает путь к готовому файлу OGG/Opus."""
    provider = config.VOICE_TTS_PROVIDER
    if provider == "none":
        raise VoiceError("синтез выключен (VOICE_TTS_PROVIDER=none)")
    voice = voice or config.VOICE_TTS_SPEAKER
    prepared = normalize_for_speech(text)[:config.VOICE_MAX_CHARS]
    log.debug("синтез: голос %s, символов %d", voice, len(prepared))

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "out.wav"
        if provider == "silero":
            _silero(prepared, wav, voice)
        elif provider == "piper":
            _piper(prepared, wav)
        elif provider == "yandex":
            _yandex_tts(prepared, wav, voice)
        elif provider == "sber":
            _sber_tts(prepared, wav, voice)
        elif provider == "clone":
            _clone_tts(prepared, wav, voice)
        else:
            raise VoiceError(f"неизвестный провайдер синтеза: {provider}")
        to_opus(wav, out_path)
    return out_path


def _silero(text: str, wav: Path, speaker: str) -> None:
    """
    Silero: работает на процессоре, качество русского хорошее,
    лицензия допускает использование в компании.
    """
    try:
        import torch
    except ImportError as exc:
        raise VoiceError("нужен torch для Silero: pip install torch") from exc
    model = _cached("silero", lambda: torch.hub.load(
        repo_or_dir="snakers4/silero-models", model="silero_tts",
        language="ru", speaker=config.VOICE_TTS_MODEL)[0])
    audio = model.apply_tts(text=text, speaker=speaker,
                            sample_rate=config.VOICE_TTS_SAMPLE_RATE)
    import wave
    import numpy as np
    data = (audio.numpy() * 32767).astype("int16")
    with wave.open(str(wav), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(config.VOICE_TTS_SAMPLE_RATE)
        fh.writeframes(data.tobytes())


def _piper(text: str, wav: Path) -> None:
    if not shutil.which("piper"):
        raise VoiceError("не найден piper в PATH")
    subprocess.run(["piper", "--model", config.VOICE_TTS_MODEL, "--output_file", str(wav)],
                   input=text.encode("utf-8"), capture_output=True, timeout=300, check=True)


def _yandex_tts(text: str, wav: Path, voice: str) -> None:
    import httpx
    r = httpx.post("https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize",
                   headers={"Authorization": f"Api-Key {config.YANDEX_API_KEY}"},
                   data={"text": text, "lang": "ru-RU", "voice": voice,
                         "folderId": config.YANDEX_FOLDER_ID,
                         "format": "lpcm", "sampleRateHertz": "48000"},
                   timeout=180)
    r.raise_for_status()
    _write_wav(wav, r.content, 48000)


def _sber_tts(text: str, wav: Path, voice: str) -> None:
    import httpx
    import llm as llm_mod
    token = llm_mod.GigaChatLLM()._auth()
    r = httpx.post("https://smartspeech.sber.ru/rest/v1/text:synthesize",
                   params={"format": "wav16", "voice": voice},
                   headers={"Authorization": f"Bearer {token}",
                            "Content-Type": "application/text"},
                   content=text.encode("utf-8"), timeout=180, verify=False)
    r.raise_for_status()
    wav.write_bytes(r.content)


def _write_wav(path: Path, pcm: bytes, rate: int) -> None:
    import wave
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes(pcm)


# ------------------------------------------------------ голос по образцу ----
CONSENT_NOTICE = (
    "Клонирование голоса выключено. Голос человека относится к биометрическим "
    "персональным данным, и для обучения модели нужно отдельное письменное "
    "согласие сотрудника, где прямо сказано об использовании записи для синтеза "
    "речи. Общей формулировки «согласие на обработку персональных данных» "
    "недостаточно — это подтверждается судебной практикой. Кроме того, "
    "лучшие по качеству открытые модели распространяются по некоммерческим "
    "лицензиям и в работе компании применяться не могут. Получив согласие и "
    "выбрав модель с подходящей лицензией, включите VOICE_CLONE_CONSENT=1."
)


def add_voice(name: str, sample: Path, title: str = "", consent: bool = False) -> dict:
    """
    Добавляет голос по образцу. Образец — 10–30 секунд чистой речи
    без музыки и посторонних шумов, лучше несколько разных фраз.
    """
    if not config.VOICE_CLONE_CONSENT and not consent:
        raise VoiceError(CONSENT_NOTICE)
    if config.VOICE_CLONE_PROVIDER == "none":
        raise VoiceError("не выбран движок клонирования (VOICE_CLONE_PROVIDER)")
    if not sample.exists():
        raise VoiceError(f"не найден образец: {sample}")

    folder = config.VOICE_PROFILES_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    wav = folder / "sample.wav"
    to_wav16k(sample, wav)
    secs = duration(wav)
    if secs < 5:
        raise VoiceError(f"образец слишком короткий: {secs:.1f} с, нужно хотя бы 10")
    if secs > 120:
        log.warning("образец длиннее двух минут — для клонирования достаточно 15–30 секунд")

    meta = {
        "id": name,
        "title": title or name,
        "provider": config.VOICE_CLONE_PROVIDER,
        "sample_seconds": round(secs, 1),
        "consent": True,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "Согласие подтверждено при добавлении. Храните письменный документ.",
    }
    (folder / "voice.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    log.info("добавлен голос «%s», образец %.1f с, движок %s",
             name, secs, config.VOICE_CLONE_PROVIDER)
    return meta


def _clone_tts(text: str, wav: Path, voice: str) -> None:
    """Синтез голосом из профиля-образца."""
    folder = config.VOICE_PROFILES_DIR / voice
    sample = folder / "sample.wav"
    if not sample.exists():
        raise VoiceError(f"нет образца для голоса «{voice}»")
    provider = config.VOICE_CLONE_PROVIDER
    if provider == "f5":
        try:
            from f5_tts.api import F5TTS
        except ImportError as exc:
            raise VoiceError("нужен f5-tts") from exc
        model = _cached("f5", lambda: F5TTS())
        model.infer(ref_file=str(sample), ref_text="", gen_text=text, file_wave=str(wav))
        return
    if provider == "xtts":
        raise VoiceError(
            "XTTS распространяется по некоммерческой лицензии Coqui (CPML) и не "
            "может использоваться в работе компании. Выберите другой движок.")
    if provider == "openvoice":
        raise VoiceError("OpenVoice не поддерживает русский язык как основной; "
                         "используйте его только как перенос тембра поверх Silero")
    raise VoiceError(f"неизвестный движок клонирования: {provider}")


# --------------------------------------------------------------------- CLI --
def main() -> int:
    p = argparse.ArgumentParser(description="Голосовой контур")
    p.add_argument("command", choices=["stt", "tts", "voices", "add-voice", "normalize"])
    p.add_argument("args", nargs="*")
    p.add_argument("--voice")
    p.add_argument("--title", default="")
    p.add_argument("--consent", action="store_true",
                   help="подтверждаю, что письменное согласие получено")
    a = p.parse_args()
    logging_setup.setup()

    if a.command == "stt":
        print(transcribe(Path(a.args[0])))
    elif a.command == "tts":
        out = Path(a.args[1]) if len(a.args) > 1 else Path("out.ogg")
        print(synthesize(a.args[0], out, a.voice))
    elif a.command == "voices":
        for v in voices():
            mark = " (по образцу)" if v["kind"] == "по образцу" else ""
            print(f"  {v['id']:<16} {v['provider']:<10}{mark}")
        if not voices():
            print("  Синтез выключен или голоса не настроены.")
    elif a.command == "add-voice":
        try:
            print(add_voice(a.args[0], Path(a.args[1]), a.title, a.consent))
        except VoiceError as exc:
            print(exc)
            return 1
    elif a.command == "normalize":
        print(normalize_for_speech(" ".join(a.args)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
