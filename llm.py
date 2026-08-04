"""
Провайдеры генерации.

  gigachat — GigaChat 2/3 (Lite/Pro/Max/Ultra)
  yandex   — YandexGPT 5.x через Yandex AI Studio
  openai   — любой OpenAI-совместимый endpoint: Cloud.ru Foundation Models,
             локальный vLLM/Ollama с Qwen3 / T-Pro, корпоративный шлюз
  echo     — офлайн-заглушка: собирает экстрактивный ответ из найденных
             фрагментов без обращения к модели. Нужна, чтобы проверить
             качество поиска отдельно от качества генерации.

Смена провайдера — одна переменная окружения, остальной код не меняется.

Все обращения к модели проходят через очередь (`llm_queue`): и вопросы
из чата, и фоновая обработка базы, и проверка связи из админки. Точка
входа одна намеренно — ограничение, которое можно обойти, забыв про него
в новом месте кода, не ограничивает ничего.
"""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass

import config
import llm_queue
import logging_setup

log = logging_setup.get("llm")

# Чтобы вызывающему коду не приходилось знать про модуль очереди.
LLMBusy = llm_queue.LLMBusy
queue_context = llm_queue.context


@dataclass
class LLMResponse:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    model: str = ""


class LLMError(RuntimeError):
    pass


class BaseLLM:
    name = "base"
    model = ""

    def complete(self, system: str, user: str) -> LLMResponse:
        raise NotImplementedError


class EchoLLM(BaseLLM):
    """
    Без обращения к модели: выдаёт наиболее релевантные предложения
    из переданного контекста. Полезно как baseline и для тестов.
    """
    name = "echo"
    model = "extractive-baseline"

    def complete(self, system: str, user: str) -> LLMResponse:
        question = ""
        m = re.search(r"ВОПРОС:\s*(.+?)\s*$", user, flags=re.S)
        if m:
            question = m.group(1).strip()
        blocks = re.findall(r"\[(\d+)\][^\n]*\n(.*?)(?=\n\[\d+\]|\Z)", user, flags=re.S)
        q_tokens = set(re.findall(r"[а-яёa-z0-9]{3,}", question.lower()))
        scored: list[tuple[float, str, str]] = []
        for num, body in blocks:
            for sentence in re.split(r"(?<=[.!?;])\s+|\n", body):
                s = sentence.strip()
                if len(s) < 25:
                    continue
                tokens = set(re.findall(r"[а-яёa-z0-9]{3,}", s.lower()))
                if not tokens:
                    continue
                overlap = len(q_tokens & tokens) / (len(q_tokens) or 1)
                if overlap > 0:
                    scored.append((overlap, s, num))
        scored.sort(key=lambda x: -x[0])
        if not scored:
            return LLMResponse("В базе знаний не нашлось прямого ответа на этот вопрос.")
        lines = [f"{s} [{num}]" for _, s, num in scored[:5]]
        return LLMResponse("\n".join(lines), model=self.model)


class GigaChatLLM(BaseLLM):
    name = "gigachat"
    OAUTH = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    API = "https://gigachat.devices.sberbank.ru/api/v1"

    def __init__(self, model: str | None = None) -> None:
        import httpx
        self.model = model or config.LLM_MODEL or "GigaChat-2-Max"
        self.client = httpx.Client(timeout=180, verify=False, proxy=config.LLM_PROXY or None)
        self._token: str | None = None
        self._expires = 0.0

    def _auth(self) -> str:
        if self._token and time.time() < self._expires - 60:
            return self._token
        if not config.GIGACHAT_AUTH_KEY:
            raise LLMError("не задан GIGACHAT_AUTH_KEY")
        r = self.client.post(
            self.OAUTH,
            headers={"Authorization": f"Basic {config.GIGACHAT_AUTH_KEY}",
                     "RqUID": hashlib.md5(str(time.time()).encode()).hexdigest(),
                     "Content-Type": "application/x-www-form-urlencoded"},
            data={"scope": config.GIGACHAT_SCOPE})
        r.raise_for_status()
        data = r.json()
        self._token = data["access_token"]
        self._expires = data.get("expires_at", time.time() * 1000 + 1_800_000) / 1000
        return self._token

    def complete(self, system: str, user: str) -> LLMResponse:
        r = self.client.post(
            f"{self.API}/chat/completions",
            headers={"Authorization": f"Bearer {self._auth()}"},
            json={"model": self.model,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}],
                  "temperature": config.LLM_TEMPERATURE,
                  "max_tokens": config.LLM_MAX_TOKENS})
        r.raise_for_status()
        data = r.json()
        usage = data.get("usage", {})
        return LLMResponse(data["choices"][0]["message"]["content"],
                           usage.get("prompt_tokens", 0),
                           usage.get("completion_tokens", 0),
                           self.model)


class YandexLLM(BaseLLM):
    name = "yandex"
    URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

    def __init__(self, model: str | None = None) -> None:
        import httpx
        self.client = httpx.Client(timeout=180, proxy=config.LLM_PROXY or None)
        self.model = model or config.LLM_MODEL or \
            f"gpt://{config.YANDEX_FOLDER_ID}/yandexgpt/latest"

    def complete(self, system: str, user: str) -> LLMResponse:
        r = self.client.post(
            self.URL,
            headers={"Authorization": f"Api-Key {config.YANDEX_API_KEY}",
                     "x-folder-id": config.YANDEX_FOLDER_ID},
            json={"modelUri": self.model,
                  "completionOptions": {"temperature": config.LLM_TEMPERATURE,
                                        "maxTokens": str(config.LLM_MAX_TOKENS)},
                  "messages": [{"role": "system", "text": system},
                               {"role": "user", "text": user}]})
        r.raise_for_status()
        result = r.json()["result"]
        usage = result.get("usage", {})
        return LLMResponse(result["alternatives"][0]["message"]["text"],
                           int(usage.get("inputTextTokens", 0)),
                           int(usage.get("completionTokens", 0)),
                           self.model)


class OpenAICompatibleLLM(BaseLLM):
    name = "openai"

    def __init__(self, model: str | None = None) -> None:
        import httpx
        self.client = httpx.Client(timeout=180, proxy=config.LLM_PROXY or None)
        self.model = model or config.LLM_MODEL or "gpt-4o-mini"

    def complete(self, system: str, user: str) -> LLMResponse:
        r = self.client.post(
            f"{config.OPENAI_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"},
            json={"model": self.model,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}],
                  "temperature": config.LLM_TEMPERATURE,
                  "max_tokens": config.LLM_MAX_TOKENS})
        r.raise_for_status()
        data = r.json()
        usage = data.get("usage", {})
        return LLMResponse(data["choices"][0]["message"]["content"],
                           usage.get("prompt_tokens", 0),
                           usage.get("completion_tokens", 0),
                           self.model)


class LocalLLM(OpenAICompatibleLLM):
    """
    Модель, поднятая на своём сервере (vLLM, llama.cpp, LM Studio, Ollama).

    Адрес не нужно прописывать руками: если сервер запущен из раздела
    «Модели», он сам сообщает, где слушает, и провайдер берёт адрес
    оттуда. Это убирает самую частую ошибку — модель работает, а в
    настройках остался старый порт, и ассистент молча ходит в пустоту.

    Данные при таком варианте не покидают сервер вообще: ни вопросы
    сотрудников, ни содержимое документов.
    """
    name = "local"

    def __init__(self, model: str | None = None) -> None:
        import httpx
        import models as models_mod
        state = models_mod.status()
        self.base_url = (config.LOCAL_LLM_BASE_URL
                         or (state.get("base_url") if state.get("running") else "")
                         or config.LOCAL_LLM_FALLBACK_URL)
        if not self.base_url:
            raise LLMError(
                "локальная модель не запущена. Раздел «Модели» → выбрать модель "
                "и нажать «Запустить», либо указать адрес в LOCAL_LLM_BASE_URL.")
        # Имя модели — то, под которым её знает САМ сервер (served_name,
        # например «qwen3:32b» у ollama), а не наш внутренний идентификатор
        # из каталога: сервер на незнакомое имя отвечает «404 Not Found».
        self.model = (model or config.LOCAL_LLM_MODEL
                      or state.get("served_name") or state.get("model")
                      or config.LLM_MODEL or "local")
        self.api_key = config.LOCAL_LLM_API_KEY
        self.client = httpx.Client(timeout=config.LOCAL_LLM_TIMEOUT)
        self.server = state

    def complete(self, system: str, user: str) -> LLMResponse:
        import httpx
        r = self.client.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
            json={"model": self.model,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}],
                  "temperature": config.LLM_TEMPERATURE,
                  "max_tokens": config.LLM_MAX_TOKENS})
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = (r.text or "")[:300]
            if r.status_code == 404:
                raise LLMError(
                    f"сервер {self.base_url} не знает модель «{self.model}». "
                    "Нажмите «Запустить и использовать» в разделе «Модели» "
                    "или проверьте LOCAL_LLM_MODEL в настройках. "
                    f"Ответ сервера: {body}") from exc
            raise LLMError(
                f"сервер {self.base_url} ответил {r.status_code}: {body}"
            ) from exc
        data = r.json()
        usage = data.get("usage", {})
        return LLMResponse(data["choices"][0]["message"]["content"],
                           usage.get("prompt_tokens", 0),
                           usage.get("completion_tokens", 0),
                           self.model)


PROVIDERS = {
    "local": LocalLLM,
    "echo": EchoLLM,
    "gigachat": GigaChatLLM,
    "yandex": YandexLLM,
    "openai": OpenAICompatibleLLM,
}

class _Instrumented:
    """Обёртка: считает токены, задержку и ошибки каждого обращения."""

    def __init__(self, inner: BaseLLM) -> None:
        self._inner = inner
        self.name = inner.name
        self.model = inner.model

    @staticmethod
    def _transient(exc: Exception) -> bool:
        """Стоит ли повторить: перегрузка и сетевые сбои проходят сами."""
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (429, 500, 502, 503, 504):
            return True
        text = str(exc).lower()
        return any(w in text for w in ("timeout", "timed out", "connection",
                                       "пустой ответ"))

    def complete(self, system: str, user: str) -> LLMResponse:
        import metrics
        for attempt in range(3):
            started = time.time()
            try:
                res = self._inner.complete(system, user)
                # content может быть None: облачные шлюзы возвращают его
                # при срабатывании своих фильтров. Раньше это давало
                # AttributeError и «Модель недоступна» без объяснений.
                if not (res.text or "").strip():
                    raise LLMError("модель вернула пустой ответ")
                metrics.record_model_call(res.model or self.model, self.name, "llm",
                                          res.tokens_in, res.tokens_out,
                                          int((time.time() - started) * 1000), True)
                return res
            except Exception as exc:
                metrics.record_model_call(self.model, self.name, "llm", 0, 0,
                                          int((time.time() - started) * 1000),
                                          False, str(exc))
                # Транзиентное (429, 5xx, сеть, пустой ответ) — повторяем
                # с паузой; остальное сразу наверх, к запасному провайдеру.
                if attempt < 2 and self._transient(exc):
                    log.warning("повтор %d после сбоя «%s» у %s",
                                attempt + 1, exc, self.name)
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise
        raise LLMError("необъяснимый выход из цикла повторов")


class Routed:
    """
    Основной провайдер плюс запасные.

    Зачем. Локальная модель — самый выгодный вариант: данные не покидают
    сервер, за токены платить не нужно. Но сервер может не подняться
    после перезагрузки, видеопамяти может не хватить, модель может
    зависнуть. Тогда правильное поведение — не «ассистент не работает»,
    а «ответ пришёл из облака, и в журнале написано почему».

    Обратный порядок тоже осмыслен: облако основным, локальная модель
    запасной на случай, когда интернет пропал или кончились деньги.

    Переключение записывается в журнал и видно в админке: молча
    переехать на другого провайдера система не должна — это меняет и
    стоимость, и то, куда уходят данные.
    """

    def __init__(self, primary: str, fallbacks: list[str]) -> None:
        self.chain = [primary] + [f for f in fallbacks if f and f != primary]
        self._engines: dict[str, object] = {}
        self._broken: dict[str, str] = {}
        self.name = primary
        self.active = primary
        self.model = ""
        self.switches: list[dict] = []

    def _engine(self, name: str):
        if name in self._engines:
            return self._engines[name]
        if name in self._broken:
            raise LLMError(self._broken[name])
        if name not in PROVIDERS:
            raise LLMError(f"неизвестный провайдер LLM: {name}")
        try:
            engine = _Instrumented(PROVIDERS[name]())
        except Exception as exc:  # noqa: BLE001
            self._broken[name] = str(exc)
            raise LLMError(str(exc)) from exc
        self._engines[name] = engine
        return engine

    def complete(self, system: str, user: str) -> LLMResponse:
        # Место в очереди берётся один раз на весь запрос, включая переход
        # к запасному провайдеру: с точки зрения нагрузки это по-прежнему
        # один вопрос, а не два.
        with llm_queue.slot(provider=self.active):
            return self._complete(system, user)

    def _complete(self, system: str, user: str) -> LLMResponse:
        errors = []
        for i, name in enumerate(self.chain):
            try:
                engine = self._engine(name)
                result = engine.complete(system, user)
                if name != self.active:
                    note = (f"ответ получен от «{name}» вместо «{self.active}»: "
                            + "; ".join(errors))
                    log.warning("%s", note)
                    self.switches.append({"from": self.active, "to": name,
                                          "why": "; ".join(errors)[:300],
                                          "at": time.strftime("%Y-%m-%d %H:%M:%S")})
                    del self.switches[:-20]
                    self.active = name
                self.model = getattr(engine, "model", "")
                return result
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{name}: {exc}")
                if i == len(self.chain) - 1:
                    raise LLMError("ни один провайдер не ответил. "
                                   + "; ".join(errors)) from exc
        raise LLMError("список провайдеров пуст")

    def describe(self) -> dict:
        info = {"primary": self.chain[0], "chain": self.chain,
                "active": self.active, "model": self.model,
                "broken": dict(self._broken), "switches": list(self.switches),
                "is_stub": self.chain[0] == "echo"}
        for name in self.chain:
            try:
                engine = self._engine(name)
                info.setdefault("ready", []).append(
                    {"provider": name, "model": getattr(engine, "model", ""),
                     "base_url": getattr(getattr(engine, "_inner", engine),
                                         "base_url", "")})
            except Exception as exc:  # noqa: BLE001
                info.setdefault("failed", []).append({"provider": name,
                                                      "error": str(exc)})
        return info


_llm = None


def get_llm():
    global _llm
    if _llm is None:
        fallbacks = [x.strip() for x in config.LLM_FALLBACK.split(",") if x.strip()]
        _llm = Routed(config.LLM_PROVIDER, fallbacks)
    return _llm


def describe() -> dict:
    """Состояние провайдеров генерации — для админки и диагностики."""
    try:
        info = get_llm().describe()
    except Exception as exc:  # noqa: BLE001
        info = {"primary": config.LLM_PROVIDER, "error": str(exc)}
    info["queue"] = llm_queue.status()
    return info


def probe(provider: str | None = None) -> dict:
    """
    Проверка живым запросом: отвечает ли модель и за сколько.

    Одно дело «настройки заполнены», другое — «модель действительно
    отвечает». Разница обычно обнаруживается в самый неподходящий момент.
    """
    name = provider or config.LLM_PROVIDER
    started = time.time()
    try:
        engine = PROVIDERS[name]() if name in PROVIDERS else None
        if engine is None:
            return {"ok": False, "provider": name, "error": "неизвестный провайдер"}
        # Проверка связи тоже занимает модель, поэтому идёт через очередь.
        # Важность выше фоновой, но ниже живого вопроса: администратор
        # подождёт, сотрудник в чате — нет.
        with llm_queue.slot(source="проверка", provider=name) as place:
            result = engine.complete(
                "Ты отвечаешь одним словом.",
                "Ответь одним словом: работает ли связь?")
        return {"ok": True, "provider": name, "model": result.model,
                "ms": int((time.time() - started) * 1000),
                "queue_wait_ms": place["waited_ms"],
                "answer": (result.text or "").strip()[:200],
                "tokens_in": result.tokens_in, "tokens_out": result.tokens_out,
                "base_url": getattr(engine, "base_url", "")}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "provider": name, "error": str(exc),
                "ms": int((time.time() - started) * 1000)}


def queue_status() -> dict:
    """Что сейчас в очереди к модели — для админки и диагностики."""
    return llm_queue.status()


def reset() -> None:
    """Сбросить кэш провайдера — после смены настроек."""
    global _llm
    _llm = None
    llm_queue.reset()
