"""
Смена провайдера и модели генерации одним действием.

Тот же принцип, что в embed_setup: выбор в настройках — это не «записать
строчку», а сценарий. Для локальной модели — найти её в каталоге, при
необходимости скачать веса, поднять сервер; для облачной — проверить
ключи и адреса. В конце — обязательный пробный вопрос: провайдер, который
не ответил на «работает ли связь», не сохраняется, и ассистент продолжает
работать на прежнем. Ошибка каждого шага объясняет следующее действие.
"""
from __future__ import annotations

import config
import logging_setup

log = logging_setup.get("llm")

KEYS: dict[str, tuple[str, str]] = {
    "gigachat": ("GIGACHAT_AUTH_KEY", "ключ авторизации GigaChat"),
    "yandex": ("YANDEX_API_KEY", "ключ API Yandex"),
}


def _key_set(key: str) -> bool:
    import os
    if str(getattr(config, key, "") or os.environ.get(key, "")).strip():
        return True
    try:
        import security
        return bool(str(security.load_secrets().get(key, "")).strip())
    except Exception:  # noqa: BLE001
        return False


def _pick_local_model(model: str | None, say):
    """Какую модель каталога запускать. Возвращает ModelSpec.

    Порядок: явно выбранная (по имени, тегу или репозиторию) → уже
    работающая → уже скачанная → рекомендуемая, которая помещается
    в память. Ошибка — только когда не подошло ничто.
    """
    import models
    if model:
        wanted = model.strip()
        for m in models.CATALOG:
            if m.kind == "llm" and wanted in (m.id, m.ollama_tag, m.repo):
                return m
        known = ", ".join(sorted(
            (m.ollama_tag or m.id) for m in models.CATALOG if m.kind == "llm"))
        raise RuntimeError(
            f"модели «{wanted}» нет в каталоге. Доступны: {known}.")
    st = models.status()
    if st.get("running"):
        spec = models.BY_ID.get(st.get("model") or "")
        if spec is not None and spec.kind == "llm":
            say(f"Использую уже работающую модель «{spec.title}».")
            return spec
    installed = models.installed_llms()
    if installed:
        say(f"Использую уже скачанную модель «{installed[0]['title']}».")
        return models.BY_ID[installed[0]["id"]]
    hw = models.hardware()
    vram = hw.get("vram_total_gb") or 0
    for m in models.CATALOG:
        if m.kind == "llm" and m.recommended and (not vram or m.fits(vram)):
            say(f"Скачанных моделей нет — беру рекомендуемую «{m.title}».")
            return m
    raise RuntimeError(
        "не нашлось модели, которая поместится в память этой машины. "
        "Выберите модель вручную в разделе «Модели».")


def switch(provider: str, model: str | None = None, progress=None,
           persist: bool = True) -> dict:
    """Полный сценарий смены провайдера/модели генерации."""
    say = progress or (lambda text: log.info("%s", text))
    import llm as llm_mod
    provider = (provider or "").strip()
    if provider not in llm_mod.PROVIDERS:
        raise RuntimeError(
            f"неизвестный провайдер генерации: «{provider}». "
            f"Доступны: {', '.join(sorted(llm_mod.PROVIDERS))}.")
    if provider == "echo":
        say("Внимание: echo — заглушка без модели, отвечает цитатами из "
            "найденных документов.")

    updates: dict[str, str] = {"LLM_PROVIDER": provider}

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

    if provider == "local":
        import models
        spec = _pick_local_model(model, say)
        if not models.is_installed(spec):
            say(f"Скачиваю веса «{spec.title}» — это может занять "
                f"десятки минут…")
            models.install(spec.id, progress=say)
        st = models.status()
        if not (st.get("running") and st.get("model") == spec.id):
            say(f"Запускаю модель «{spec.title}»…")
            st = models.serve(spec.id, apply_config=False)
        updates["LOCAL_LLM_MODEL"] = st.get("served_name") or spec.id
    elif model:
        updates["LLM_MODEL"] = model.strip()

    # Пробный вопрос — до сохранения. Меняем конфигурацию в памяти,
    # проверяем, при неудаче откатываем.
    say("Проверяю провайдер живым вопросом…")
    saved = {k: getattr(config, k) for k in
             ("LLM_PROVIDER", "LLM_MODEL", "LOCAL_LLM_MODEL")}
    for key, value in updates.items():
        setattr(config, key, value)
    llm_mod.reset()
    result = llm_mod.probe(provider)
    if not result.get("ok"):
        for key, value in saved.items():
            setattr(config, key, value)
        llm_mod.reset()
        raise RuntimeError(
            f"провайдер «{provider}» не ответил на пробный вопрос: "
            f"{result.get('error')}. Настройки не изменены — отвечает "
            f"по-прежнему «{saved['LLM_PROVIDER']}».")
    say(f"Провайдер ответил за {result.get('ms')} мс: "
        f"«{(result.get('answer') or '')[:60]}»")

    if persist:
        import webui
        webui.write_env(updates)
        say("Настройки сохранены: " + ", ".join(sorted(updates)))
    log.info("генерация переключена на «%s» (%s)", provider,
             updates.get("LOCAL_LLM_MODEL") or updates.get("LLM_MODEL") or "—")
    return {"provider": provider, "updates": updates,
            "answer": result.get("answer"), "ms": result.get("ms")}
