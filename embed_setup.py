"""
Переключение провайдера смыслового поиска одним действием.

Раньше смена провайдера была разложена на ручные шаги: поставить пакеты,
скачать веса, вписать настройки, пересчитать векторы — и любая ошибка в
середине оставляла систему в положении «выбрал одно, работает другое».
Хуже того, выбор провайдера при пересчёте векторов жил только в памяти
процесса и после перезапуска молча возвращался к прежнему.

Здесь весь сценарий собран в одну функцию switch() с проверкой каждого
шага: недостающие пакеты ставятся сами, веса из каталога скачиваются
сами, необученная своя модель обучается сама, а перед сохранением
настроек провайдер обязан ответить на пробный текст. Если что-то не
получилось — настройки не меняются вовсе, и поиск продолжает работать
на прежнем провайдере; ошибка при этом объясняет, что сделать.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import config
import logging_setup

log = logging_setup.get("embed")

# Пакеты, без которых провайдер не поднимется. Имя модуля и имя пакета
# в pip различаются — храним оба.
PACKAGES: dict[str, list[tuple[str, str]]] = {
    "onnx": [("onnxruntime", "onnxruntime")],
    "local": [("sentence_transformers", "sentence-transformers")],
}
# Пакеты желательные: без них хуже (медленнее, не все модели), но работает.
NICE_TO_HAVE: dict[str, list[tuple[str, str]]] = {
    "onnx": [("tokenizers", "tokenizers")],
}
# Ключи, без которых облачный провайдер бесполезен.
KEYS: dict[str, tuple[str, str]] = {
    "gigachat": ("GIGACHAT_AUTH_KEY", "ключ авторизации GigaChat"),
    "yandex": ("YANDEX_API_KEY", "ключ API Yandex"),
}
DEFAULT_LOCAL_MODEL = "deepvk/USER-bge-m3"


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:  # noqa: BLE001 — битый пакет считаем отсутствующим
        return False


def _key_set(key: str) -> bool:
    """Задан ли ключ — в настройках, окружении или защищённом файле."""
    if str(getattr(config, key, "") or os.environ.get(key, "")).strip():
        return True
    try:
        import security
        return bool(str(security.load_secrets().get(key, "")).strip())
    except Exception:  # noqa: BLE001
        return False


def _pip_install(packages: list[str], say) -> None:
    """Ставит пакеты тем же питоном, которым работает система.

    Две попытки: обычная и с --break-system-packages (на системах, где
    питон управляется пакетным менеджером, без этого флага pip
    отказывается). Если не вышло — в ошибке последние строки вывода pip,
    а не голое «не удалось».
    """
    base = [sys.executable, "-m", "pip", "install", *packages]
    last_out = ""
    for attempt, cmd in enumerate((base, base + ["--break-system-packages"])):
        say(("Устанавливаю пакеты: " if not attempt else
             "Повторяю установку с --break-system-packages: ")
            + ", ".join(packages))
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=1800)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"установка пакетов не уложилась в 30 минут: "
                f"{', '.join(packages)}. Проверьте связь с интернетом "
                f"или поставьте вручную: pip install "
                f"{' '.join(packages)}") from exc
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"не удалось запустить pip ({sys.executable} -m pip). "
                f"Проверьте установку Python.") from exc
        if proc.returncode == 0:
            say("Пакеты установлены.")
            return
        last_out = (proc.stdout or "") + (proc.stderr or "")
    tail = "\n".join(last_out.strip().splitlines()[-8:])
    raise RuntimeError(
        f"не удалось установить пакеты ({', '.join(packages)}). "
        f"Что написал pip:\n{tail}\n"
        f"Поставьте вручную: pip install {' '.join(packages)}")


def _catalog_spec(name: str):
    """Модель из каталога по идентификатору или репозиторию, если есть."""
    import models
    for m in models.CATALOG:
        if m.kind == "embedding" and name in (m.id, m.repo):
            return m
    return None


def _ensure_local_weights(name: str, say) -> str:
    """Веса для sentence-transformers: скачиваем из каталога, если можем.

    Возвращает имя/путь, который писать в EMBEDDINGS_MODEL: локальную
    папку, если веса скачаны нами (работает без интернета), иначе
    исходное имя — библиотека скачает сама при первом обращении.
    """
    import models
    spec = _catalog_spec(name)
    if spec is None:
        say(f"Модели «{name}» нет в каталоге — библиотека скачает её сама "
            f"при первом обращении"
            + (f" (через зеркало {config.HF_MIRROR})" if config.HF_MIRROR
               else ". Если из вашей сети площадка huggingface недоступна, "
                    "укажите зеркало в настройке HF_MIRROR") + ".")
        return name
    if not models.is_installed(spec):
        say(f"Скачиваю веса модели «{spec.title}» ({spec.repo})…")
        models.install(spec.id, progress=say)
    path = models.local_path(spec)
    if path.exists() and any(path.iterdir()):
        say(f"Использую локальную копию весов: {path}")
        return str(path)
    return spec.repo


def _export_onnx(repo: str, say) -> str | None:
    """Конвертирует модель в onnx сама: optimum-cli export onnx.

    Возвращает путь к model.onnx или None — тогда вызывающая сторона
    объяснит запасные пути. Все шаги — установка optimum, скачивание,
    конвертация — идут в журнал задачи строка за строкой.
    """
    target = Path(config.MODELS_DIR) / (repo.split("/")[-1].lower() + "-onnx")
    found = sorted(target.rglob("*.onnx")) if target.exists() else []
    if found:
        say(f"Конвертированная копия уже есть: {found[0]}")
        return str(found[0])
    if not _has_module("optimum"):
        try:
            _pip_install(["optimum[exporters]"], say)
        except RuntimeError as exc:
            say(f"Не удалось поставить optimum: {exc}")
            return None
    cli = Path(sys.executable).parent / "optimum-cli"
    cmd = ([str(cli)] if cli.exists() else ["optimum-cli"]) + \
        ["export", "onnx", "--model", repo, str(target)]
    env = dict(os.environ)
    if config.HF_MIRROR:
        env["HF_ENDPOINT"] = config.HF_MIRROR
        say(f"Скачивание пойдёт через зеркало {config.HF_MIRROR}")
    say(f"Конвертирую «{repo}» в onnx — разовая операция на несколько минут…")
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, env=env)
        for line in proc.stdout:  # type: ignore[union-attr]
            line = line.rstrip()
            if line:
                say(line[-200:])
        proc.wait(timeout=3600)
    except FileNotFoundError:
        say("Команда optimum-cli не нашлась после установки optimum.")
        return None
    except Exception as exc:  # noqa: BLE001
        say(f"Конвертация не удалась: {exc}")
        return None
    if proc.returncode != 0:
        say(f"Конвертация не удалась (код {proc.returncode}) — "
            f"подробности в строках выше.")
        return None
    found = sorted(target.rglob("*.onnx"))
    if found:
        say(f"Модель сконвертирована: {found[0]}")
        return str(found[0])
    say("Конвертация закончилась без ошибок, но файла .onnx не появилось.")
    return None


def _ensure_onnx_model(model: str | None, say) -> str | None:
    """Файл model.onnx: указанный, скачанный, из каталога — или конвертируем.

    Возвращает путь для ONNX_MODEL_PATH или None, если путь уже задан
    и существует. Если файла нет и добыть не вышло — честная ошибка с
    планом действий.
    """
    import models
    current = Path(config.ONNX_MODEL_PATH or "")
    if config.ONNX_MODEL_PATH and current.exists():
        return None
    if config.ONNX_MODEL_PATH and not current.exists():
        say(f"Внимание: указанный ONNX_MODEL_PATH не существует: {current}")
    # Указана модель — попробуем скачать и поискать в ней onnx-файл.
    if model:
        spec = _catalog_spec(model)
        if spec is not None:
            if not models.is_installed(spec):
                say(f"Скачиваю «{spec.title}» ({spec.repo})…")
                models.install(spec.id, progress=say)
            found = sorted(models.local_path(spec).rglob("*.onnx"))
            if found:
                say(f"Нашёл onnx-файл в скачанной модели: {found[0]}")
                return str(found[0])
            say(f"В «{spec.repo}» файла .onnx нет — этот репозиторий "
                f"содержит только веса для sentence-transformers.")
    # Может, onnx-файл уже лежит среди скачанных моделей.
    found = sorted(Path(config.MODELS_DIR).rglob("*.onnx")) \
        if Path(config.MODELS_DIR).exists() else []
    if found:
        say(f"Нашёл готовый onnx-файл среди скачанных моделей: {found[0]}")
        return str(found[0])
    # Готового нет нигде — конвертируем сами. Это и есть «скачивается и
    # устанавливается само»: optimum скачает веса и выгрузит их в onnx.
    repo = model or config.EMBEDDINGS_MODEL or DEFAULT_LOCAL_MODEL
    spec = _catalog_spec(repo)
    exported = _export_onnx(spec.repo if spec else repo, say)
    if exported:
        return exported
    raise RuntimeError(
        "для провайдера onnx нужен файл model.onnx: автоматическая "
        "конвертация не удалась (подробности — в журнале задачи выше). "
        "Варианты: 1) выбрать провайдер «local» — он использует те же "
        "модели без конвертации и всё скачает сам; 2) на машине с "
        "интернетом выполнить `optimum-cli export onnx --model "
        "deepvk/USER-bge-m3 ./user-bge-m3-onnx`, перенести папку на "
        "сервер и указать путь в ONNX_MODEL_PATH. Настройки не изменены.")


def switch(provider: str, model: str | None = None, progress=None,
           persist: bool = True) -> dict:
    """Полный сценарий смены провайдера смыслового поиска.

    Пакеты, веса, обучение, пробный запрос, сохранение настроек — всё
    здесь, с откатом при любой ошибке. Пересчёт векторов НЕ входит:
    его запускает вызывающая сторона (это отдельная длинная задача).
    """
    say = progress or (lambda text: log.info("%s", text))
    import embeddings
    provider = (provider or "").strip()
    if provider not in embeddings.PROVIDERS:
        raise RuntimeError(
            f"неизвестный провайдер смыслового поиска: «{provider}». "
            f"Доступны: {', '.join(sorted(embeddings.PROVIDERS))}.")
    if provider in embeddings.STUB_PROVIDERS:
        say("Внимание: это заглушка без смысловой близости — качество "
            "поиска будет держаться на одном текстовом канале.")

    updates: dict[str, str] = {"EMBEDDINGS_PROVIDER": provider}

    # 1. Пакеты: обязательные ставим, желательные — по возможности.
    missing = [pip for mod, pip in PACKAGES.get(provider, ())
               if not _has_module(mod)]
    if missing:
        _pip_install(missing, say)
    for mod, pip in NICE_TO_HAVE.get(provider, ()):
        if not _has_module(mod):
            try:
                _pip_install([pip], say)
            except RuntimeError as exc:
                say(f"Не удалось поставить необязательный пакет {pip}: {exc}. "
                    f"Продолжаю без него.")

    # 2. Ключи и адреса облачных провайдеров — без них нет смысла ехать дальше.
    if provider in KEYS:
        key, title = KEYS[provider]
        if not _key_set(key):
            raise RuntimeError(
                f"не задан {key} ({title}). Впишите его: раздел «Настройки» → "
                f"группа «Модели». Настройки не изменены.")
    if provider == "openai" and not str(config.OPENAI_BASE_URL or "").strip():
        raise RuntimeError(
            "для openai-совместимого провайдера нужен адрес сервиса — "
            "заполните OPENAI_BASE_URL в настройках. Настройки не изменены.")

    # 3. Модель: веса, обучение.
    if provider == "lsa" and not Path(config.LSA_MODEL_PATH).exists():
        say("Своя смысловая модель ещё не обучена — обучаю на вашей базе. "
            "Интернет для этого не нужен.")
        try:
            embeddings.LSAEmbedder.train(progress=say)
            embeddings.reset()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"не удалось обучить свою модель: {exc}. Обычно это значит, "
                f"что база ещё не проиндексирована — сначала запустите "
                f"индексацию. Настройки не изменены.") from exc
    if provider == "local":
        updates["EMBEDDINGS_MODEL"] = _ensure_local_weights(
            model or config.EMBEDDINGS_MODEL or DEFAULT_LOCAL_MODEL, say)
    if provider == "onnx":
        onnx_path = _ensure_onnx_model(model, say)
        if onnx_path:
            updates["ONNX_MODEL_PATH"] = onnx_path

    # 4. Пробный запрос. До этой строки настройки в силе прежние; меняем
    #    их в памяти, проверяем и при неудаче откатываем.
    say("Проверяю провайдер на пробном тексте…")
    saved = {k: getattr(config, k) for k in
             ("EMBEDDINGS_PROVIDER", "EMBEDDINGS_MODEL", "ONNX_MODEL_PATH")}
    for key, value in updates.items():
        setattr(config, key, value)
    embeddings.reset()
    try:
        emb = embeddings.get_embedder()
        vec = emb.embed(["проверка связи: подача насоса, м3/ч"])
        if getattr(vec, "ndim", 0) != 2 or vec.shape[0] != 1 or vec.shape[1] < 2:
            raise RuntimeError(f"провайдер вернул вектор странной формы: "
                               f"{getattr(vec, 'shape', '?')}")
        dim = int(vec.shape[1])
    except Exception as exc:  # noqa: BLE001
        for key, value in saved.items():
            setattr(config, key, value)
        embeddings.reset()
        raise RuntimeError(
            f"провайдер «{provider}» не прошёл проверку: {exc}. "
            f"Настройки не изменены — поиск продолжает работать на "
            f"«{saved['EMBEDDINGS_PROVIDER']}».") from exc
    say(f"Провайдер отвечает. Размерность векторов: {dim}.")

    # 5. Сохранение — только после успешной проверки.
    if persist:
        import webui
        webui.write_env(updates)
        say("Настройки сохранены: " + ", ".join(sorted(updates)))
    log.info("смысловой поиск переключён на «%s» (размерность %d)",
             provider, dim)
    return {"provider": provider, "dim": dim, "updates": updates}
