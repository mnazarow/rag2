"""
Автонастройка компонентов при смене настройки.

Единый принцип для всех «провайдерных» настроек: выбор значения в форме —
это заказ на готовый к работе компонент, а не запись строчки в файл.
Здесь для каждой такой настройки описано, что нужно компоненту
(python-пакеты, системные программы, файлы моделей, ключи, работающие
серверы) — и всё, что можно поставить автоматически, ставится само.
Настройка сохраняется только после успешной подготовки: если что-то не
вышло, значение не меняется, а ошибка называет следующее действие.

Смысловой поиск и генерация живут в своих модулях (embed_setup,
llm_setup): у них есть ещё пересчёт векторов и пробные запросы. Здесь —
все остальные: переранжирование, сканы, зрение, речь, хранилище векторов.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import config
import logging_setup
from embed_setup import _export_onnx, _has_module, _key_set, _pip_install

log = logging_setup.get("setup")


def _brew_or_apt(pkgs_brew: list[str], pkgs_apt: list[str], say) -> bool:
    """Ставит системную программу менеджером системы. False — не смогли."""
    import platform
    if platform.system() == "Darwin" and shutil.which("brew"):
        cmd = ["brew", "install", *pkgs_brew]
    elif shutil.which("apt-get"):
        # -n: без интерактивного запроса пароля — из веб-интерфейса
        # спросить его не у кого; нет прав — честно скажем команду.
        cmd = ["sudo", "-n", "apt-get", "install", "-y", *pkgs_apt]
    else:
        return False
    say("Устанавливаю: " + " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except Exception as exc:  # noqa: BLE001
        say(f"Не получилось: {exc}")
        return False
    if proc.returncode != 0:
        tail = ((proc.stdout or "") + (proc.stderr or "")).strip().splitlines()[-5:]
        say("Не получилось:\n" + "\n".join(tail))
        return False
    return True


# ------------------------------------------------------- подготовка по ключам
def _prep_reranker(value: str, say) -> dict:
    if value in ("none", "lexical"):
        return {}
    if value == "onnx":
        if not _has_module("onnxruntime"):
            _pip_install(["onnxruntime"], say)
        if config.RERANKER_ONNX_PATH and Path(config.RERANKER_ONNX_PATH).exists():
            return {}
        found = sorted(Path(config.MODELS_DIR).rglob("*rerank*.onnx")) \
            if Path(config.MODELS_DIR).exists() else []
        if found:
            say(f"Нашёл готовый onnx-файл реранкера: {found[0]}")
            return {"RERANKER_ONNX_PATH": str(found[0])}
        repo = config.RERANKER_MODEL or "BAAI/bge-reranker-v2-m3"
        say("Файла модели нет — конвертирую сама.")
        exported = _export_onnx(repo, say)
        if exported:
            return {"RERANKER_ONNX_PATH": exported}
        raise RuntimeError(
            "для onnx-реранкера нужен файл модели, автоматическая конвертация "
            "не удалась (подробности выше). Вариант «lexical» работает сразу "
            "и без моделей. Настройка не изменена.")
    if value == "local":
        if not _has_module("sentence_transformers"):
            _pip_install(["sentence-transformers"], say)
        return {}
    if value == "openai":
        if not (config.RERANKER_BASE_URL or config.OPENAI_BASE_URL):
            raise RuntimeError(
                "для внешнего реранкера нужен адрес — заполните "
                "RERANKER_BASE_URL (или OPENAI_BASE_URL). Настройка не изменена.")
        return {}
    return {}


def _prep_ocr(value: str, say) -> dict:
    if value == "none":
        return {}
    if value == "tesseract":
        if shutil.which("tesseract"):
            say("tesseract уже установлен.")
        elif not _brew_or_apt(["tesseract", "tesseract-lang"],
                              ["tesseract-ocr", "tesseract-ocr-rus"], say):
            raise RuntimeError(
                "tesseract не установился автоматически. Поставьте вручную: "
                "brew install tesseract tesseract-lang (мак) или "
                "apt-get install tesseract-ocr tesseract-ocr-rus (линукс). "
                "Настройка не изменена.")
        try:
            langs = subprocess.run(["tesseract", "--list-langs"],
                                   capture_output=True, text=True,
                                   timeout=15).stdout
            if "rus" not in langs:
                say("Внимание: языкового пакета rus не видно — без него "
                    "русский текст распознается латиницей.")
        except Exception:  # noqa: BLE001
            pass
        return {}
    if value == "vlm":
        import media
        try:
            base, mdl = media.local_vision_endpoint()
            say(f"Зрительная модель найдена: {mdl} ({base}).")
            return {}
        except RuntimeError:
            pass
        if config.OCR_BASE_URL or config.OPENAI_BASE_URL:
            return {}
        import models
        say("Зрительной модели нет — скачиваю Qwen3-VL 8B (разово, "
            "несколько гигабайт)…")
        models.install("qwen3-vl-8b", progress=say)
        media.local_vision_endpoint()   # не нашлась — её ошибка объяснит
        return {}
    if value == "yandex":
        if not _key_set("YANDEX_API_KEY"):
            raise RuntimeError("не задан YANDEX_API_KEY — впишите его в "
                               "настройках. Настройка не изменена.")
        return {}
    return {}


def _prep_vision(value: str, say) -> dict:
    if value == "none":
        return {}
    if value == "local":
        import media
        try:
            _, mdl = media.local_vision_endpoint()
            say(f"Модель зрения найдена: {mdl}.")
        except RuntimeError:
            import models
            say("Модели зрения нет — скачиваю Qwen3-VL 8B (разово)…")
            models.install("qwen3-vl-8b", progress=say)
            media.local_vision_endpoint()
        return {}
    if value == "openai":
        if not str(config.OPENAI_BASE_URL or "").strip():
            raise RuntimeError("для openai-совместимого зрения нужен адрес — "
                               "заполните OPENAI_BASE_URL. Настройка не изменена.")
        return {}
    if value in ("gigachat", "yandex"):
        key = "GIGACHAT_AUTH_KEY" if value == "gigachat" else "YANDEX_API_KEY"
        if not _key_set(key):
            raise RuntimeError(f"не задан {key} — впишите его в настройках. "
                               f"Настройка не изменена.")
        return {}
    return {}


def _prep_asr(value: str, say) -> dict:
    packages = {"faster-whisper": ("faster_whisper", ["faster-whisper"]),
                "whisper": ("whisper", ["openai-whisper"]),
                "sber": ("gigaam", ["gigaam"])}
    if value in packages:
        module, pkgs = packages[value]
        if not _has_module(module):
            _pip_install(pkgs, say)
        if not shutil.which("ffmpeg"):
            if not _brew_or_apt(["ffmpeg"], ["ffmpeg"], say):
                say("Внимание: нет ffmpeg — звук из видео достать не выйдет. "
                    "Поставьте: brew install ffmpeg / apt-get install ffmpeg.")
        return {}
    if value == "yandex":
        if not _key_set("YANDEX_API_KEY"):
            raise RuntimeError("не задан YANDEX_API_KEY — впишите его в "
                               "настройках. Настройка не изменена.")
    return {}


def _prep_tts(value: str, say) -> dict:
    if value == "silero":
        for module, pkg in (("torch", "torch"), ("torchaudio", "torchaudio")):
            if not _has_module(module):
                say(f"Ставлю {pkg} — это надолго и займёт несколько гигабайт.")
                _pip_install([pkg], say)
        return {}
    if value == "piper":
        if not shutil.which("piper") and not _has_module("piper"):
            say("Внимание: piper не найден — поставьте его вручную "
                "(github.com/rhasspy/piper), иначе синтез не заработает.")
        return {}
    if value == "yandex":
        if not _key_set("YANDEX_API_KEY"):
            raise RuntimeError("не задан YANDEX_API_KEY — впишите его в "
                               "настройках. Настройка не изменена.")
    if value == "sber":
        if not _key_set("GIGACHAT_AUTH_KEY"):
            raise RuntimeError("не задан GIGACHAT_AUTH_KEY — впишите его в "
                               "настройках. Настройка не изменена.")
    return {}


def _prep_vectors(value: str, say) -> dict:
    if value != "qdrant":
        return {}
    if not _has_module("qdrant_client"):
        _pip_install(["qdrant-client"], say)
    url = (config.QDRANT_URL or "http://127.0.0.1:6333").rstrip("/")
    try:
        import httpx
        alive = httpx.get(url + "/collections", timeout=5).status_code == 200
    except Exception:  # noqa: BLE001
        alive = False
    if not alive:
        raise RuntimeError(
            f"сервер qdrant по адресу {url} не отвечает. Поднимите его "
            f"(docker run -d -p 6333:6333 qdrant/qdrant) или укажите "
            f"правильный QDRANT_URL. Настройка не изменена.")
    say("Сервер qdrant отвечает. После переключения перенесите векторы: "
        "python vectors_qdrant.py migrate")
    return {}


REGISTRY: dict[str, tuple[str, object]] = {
    "RERANKER_PROVIDER": ("Переранжирование", _prep_reranker),
    "OCR_PROVIDER": ("Распознавание сканов", _prep_ocr),
    "VISION_PROVIDER": ("Описание изображений", _prep_vision),
    "ASR_PROVIDER": ("Распознавание речи", _prep_asr),
    "VOICE_TTS_PROVIDER": ("Синтез речи", _prep_tts),
    "VECTOR_BACKEND": ("Хранилище векторов", _prep_vectors),
}
KEYS = set(REGISTRY)


def title(key: str) -> str:
    return REGISTRY[key][0]


def switch(key: str, value: str, progress=None, persist: bool = True) -> dict:
    """Готовит компонент под новое значение и сохраняет настройку."""
    say = progress or (lambda text: log.info("%s", text))
    if key not in REGISTRY:
        raise RuntimeError(f"для настройки {key} автонастройки нет — "
                           f"сохраните её обычным способом")
    label, prepare = REGISTRY[key]
    value = str(value or "").strip()
    say(f"{label}: переключаю на «{value or 'пусто'}».")
    updates = {key: value}
    updates.update(prepare(value, say) or {})
    if persist:
        import webui
        webui.write_env(updates)
        webui._reload_after_settings(updates)
        say("Настройки сохранены: " + ", ".join(sorted(updates)))
    log.info("%s: переключено на «%s»", label, value)
    return {"key": key, "value": value, "updates": updates}
