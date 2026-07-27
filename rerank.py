"""
Переранжирование найденных фрагментов.

Зачем это нужно. Поиск устроен дёшево и приблизительно: текстовый канал
считает совпадение слов, смысловой — близость векторов. Оба смотрят на
запрос и на фрагмент по отдельности, а не на их сочетание. Из-за этого
регулярно случается так: нужный документ найден, но стоит седьмым, а в
ответ модели уходят первые пять — и ассистент отвечает «в базе нет
данных», хотя данные есть.

Переранжирование читает каждую пару «вопрос + фрагмент» целиком и
отвечает на один вопрос: отвечает ли этот фрагмент на этот вопрос.
Это дороже, поэтому применяется только к первым RERANKER_TOP_N
кандидатам — двадцати вместо сорока тысяч.

Провайдеры:

  none     — выключено
  lexical  — встроенный, без моделей и интернета. Считает, сколько
             значимых слов запроса покрыто фрагментом, насколько кучно
             они в нём расположены и совпали ли артикулы дословно.
             Это не кросс-энкодер, но именно эти три признака
             отвечают за большую часть его пользы на технической базе.
  onnx     — bge-reranker-v2-m3 в формате ONNX: настоящий кросс-энкодер
             на процессоре, без torch. Около полусекунды на двадцать пар.
  local    — sentence-transformers CrossEncoder (нужен torch)
  openai   — совместимый endpoint /v1/rerank (vLLM, TEI, Infinity, Jina)

Важное свойство: сбой или таймаут реранкера никогда не роняет ответ.
Если модель недоступна, выдача отдаётся в исходном порядке, а причина
пишется в журнал.
"""
from __future__ import annotations

import math
import re
import threading
import time
from pathlib import Path

import numpy as np

import config
import logging_setup

log = logging_setup.get("search")

_TOKEN_RX = re.compile(r"[a-zа-яёA-ZА-ЯЁ0-9][a-zа-яёA-ZА-ЯЁ0-9\-./]*")

# Слова, которые есть почти в каждом вопросе и ничего не различают.
_STOP = {
    "как", "что", "где", "для", "или", "это", "при", "под", "над", "чем", "кто",
    "быть", "если", "так", "уже", "его", "нам", "мне", "она", "они", "какой",
    "какая", "какие", "нужно", "можно", "есть", "ли", "не", "на", "в", "из",
    "по", "до", "от", "с", "со", "к", "у", "о", "об", "за", "же", "бы", "то",
    "the", "and", "for", "with", "of", "is", "a", "in", "to",
}


class RerankError(RuntimeError):
    pass


# ------------------------------------------------------------- служебное ---
def _tokens(text: str) -> list[str]:
    import lsa
    out = []
    for raw in _TOKEN_RX.findall(text.lower()):
        raw = raw.strip("-./")
        if len(raw) < 2 or raw in _STOP:
            continue
        out.append(lsa.normalize_token(raw))
    return out


def _is_code(token: str) -> bool:
    """Артикул или обозначение модели: буквы вперемешку с цифрами."""
    return any(c.isdigit() for c in token) and any(c.isalpha() for c in token)


# --------------------------------------------------- лексический реранкер --
class LexicalReranker:
    """
    Работает без единого скачанного байта и без видеокарты.

    Три признака, в порядке важности:

      покрытие  — какая доля значимых слов запроса вообще встретилась во
                  фрагменте, с весом по редкости слова: «кавитация» весит
                  больше, чем «насос», потому что «насос» есть везде;
      кучность  — насколько близко друг к другу эти слова расположены.
                  Фрагмент, где «давление» и «настройка» стоят в одном
                  предложении, отвечает на вопрос, а фрагмент, где они
                  на разных страницах, — почти наверняка нет;
      артикулы  — дословное совпадение обозначения модели. Это самый
                  надёжный сигнал на технической базе, и он получает
                  отдельную прибавку.

    Веса редкости берутся из уже обученной смысловой модели: там они
    посчитаны по вашей базе. Если модель ещё не обучена, редкость
    оценивается по длине слова и наличию цифр — грубее, но работает.
    """
    name = "lexical"

    def __init__(self) -> None:
        self.idf: dict[str, float] = {}
        self.default_idf = 3.0
        try:
            import lsa
            path = Path(config.LSA_MODEL_PATH)
            if path.exists():
                model = lsa.LSAModel.load(path)
                self.idf = {t: float(model.idf[i]) for t, i in model.vocab.items()}
                self.default_idf = float(np.percentile(model.idf, 85))
        except Exception as exc:  # noqa: BLE001 — работаем на приближении
            log.debug("веса редкости слов недоступны: %s", exc)

    def weight(self, token: str) -> float:
        if token in self.idf:
            return self.idf[token]
        # Приближение: длинные слова и обозначения моделей редки почти всегда.
        base = self.default_idf
        if _is_code(token):
            base *= 1.4
        elif len(token) > 9:
            base *= 1.15
        return base

    def score_pairs(self, query: str, texts: list[str]) -> list[float]:
        q_tokens = _tokens(query)
        if not q_tokens:
            return [0.0] * len(texts)
        q_unique = list(dict.fromkeys(q_tokens))
        weights = {t: self.weight(t) for t in q_unique}
        total_weight = sum(weights.values()) or 1.0
        q_codes = {t for t in q_unique if _is_code(t)}

        out = []
        for text in texts:
            d_tokens = _tokens(text)
            if not d_tokens:
                out.append(0.0)
                continue
            positions: dict[str, list[int]] = {}
            for i, tok in enumerate(d_tokens):
                if tok in weights:
                    positions.setdefault(tok, []).append(i)

            covered = sum(weights[t] for t in positions)
            coverage = covered / total_weight

            proximity = self._proximity(positions, len(d_tokens))

            # Артикул, совпавший дословно, — почти гарантия попадания.
            code_hit = (len(q_codes & set(positions)) / len(q_codes)) if q_codes else 0.0

            # Очень длинный фрагмент покрывает слова случайно — слегка штрафуем.
            length_penalty = 1.0 / (1.0 + math.log1p(len(d_tokens) / 400.0))

            score = (0.60 * coverage + 0.25 * proximity + 0.15 * code_hit) * length_penalty
            out.append(float(score))
        return out

    @staticmethod
    def _proximity(positions: dict[str, list[int]], n_tokens: int) -> float:
        """Насколько кучно слова запроса лежат во фрагменте: 0…1."""
        found = [p for p in positions.values() if p]
        if len(found) < 2:
            return 1.0 if found else 0.0
        flat = sorted((pos, i) for i, plist in enumerate(found) for pos in plist)
        need = len(found)
        best = None
        seen: dict[int, int] = {}
        left = 0
        for right, (pos, term) in enumerate(flat):
            seen[term] = seen.get(term, 0) + 1
            while len(seen) == need:
                span = pos - flat[left][0]
                best = span if best is None else min(best, span)
                lterm = flat[left][1]
                seen[lterm] -= 1
                if not seen[lterm]:
                    del seen[lterm]
                left += 1
        if best is None:
            return 0.0
        # Окно размером с одно предложение считаем идеальным; чем шире
        # разброс слов по фрагменту, тем ниже оценка.
        ideal = max(need * 3, 12)
        return 1.0 if best <= ideal else float(ideal / best)


# --------------------------------------------------------- ONNX-реранкер ---
class OnnxReranker:
    """
    Настоящий кросс-энкодер (BGE-reranker-v2-m3) без torch.

    Файлы готовятся на машине с интернетом:
        optimum-cli export onnx --model BAAI/bge-reranker-v2-m3 ./reranker-onnx
    и переносятся на сервер; путь указывается в RERANKER_ONNX_PATH.
    """
    name = "onnx"

    def __init__(self) -> None:
        import onnxruntime as ort
        import embeddings
        path = Path(config.RERANKER_ONNX_PATH or "")
        if not path.exists():
            raise RerankError(
                "не задан RERANKER_ONNX_PATH — путь к model.onnx кросс-энкодера")
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(str(path), providers=providers)
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.tokenizer = embeddings._load_tokenizer(
            config.RERANKER_TOKENIZER_DIR or path.parent, config.RERANKER_MAX_TOKENS)

    def score_pairs(self, query: str, texts: list[str]) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(texts), config.RERANKER_BATCH):
            part = texts[start:start + config.RERANKER_BATCH]
            encoded = [self.tokenizer.encode_pair(query, t)[: config.RERANKER_MAX_TOKENS]
                       for t in part]
            width = max(len(e) for e in encoded)
            ids = np.zeros((len(encoded), width), dtype=np.int64)
            mask = np.zeros((len(encoded), width), dtype=np.int64)
            for i, e in enumerate(encoded):
                ids[i, : len(e)] = e
                mask[i, : len(e)] = 1
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            feed = {k: v for k, v in feed.items() if k in self.inputs}
            logits = np.asarray(self.session.run(None, feed)[0], dtype=np.float32)
            logits = logits[:, 0] if logits.ndim == 2 else logits.ravel()
            scores.extend(1.0 / (1.0 + np.exp(-logits)))     # логит -> вероятность
        return [float(s) for s in scores]


class LocalReranker:
    """sentence-transformers CrossEncoder — если в системе есть torch."""
    name = "local"

    def __init__(self) -> None:
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(config.RERANKER_MODEL, max_length=config.RERANKER_MAX_TOKENS)

    def score_pairs(self, query: str, texts: list[str]) -> list[float]:
        raw = self.model.predict([(query, t) for t in texts],
                                 batch_size=config.RERANKER_BATCH)
        arr = np.asarray(raw, dtype=np.float32).ravel()
        return [float(1.0 / (1.0 + math.exp(-x))) for x in arr]


class ApiReranker:
    """Внешний endpoint /v1/rerank: vLLM, TEI, Infinity, Jina, Cohere-совместимые."""
    name = "openai"

    def __init__(self) -> None:
        import httpx
        base = config.RERANKER_BASE_URL or config.OPENAI_BASE_URL
        self.url = base.rstrip("/") + "/rerank"
        key = config.RERANKER_API_KEY or config.OPENAI_API_KEY
        self.client = httpx.Client(timeout=config.RERANKER_TIMEOUT,
                                   proxy=config.LLM_PROXY or None,
                                   headers={"Authorization": f"Bearer {key}"} if key else {})

    def score_pairs(self, query: str, texts: list[str]) -> list[float]:
        r = self.client.post(self.url, json={"model": config.RERANKER_MODEL,
                                             "query": query, "documents": texts,
                                             "top_n": len(texts)})
        r.raise_for_status()
        data = r.json()
        results = data.get("results") or data.get("data") or []
        out = [0.0] * len(texts)
        for item in results:
            idx = int(item.get("index", 0))
            if 0 <= idx < len(out):
                out[idx] = float(item.get("relevance_score", item.get("score", 0.0)))
        return out


PROVIDERS = {
    "lexical": LexicalReranker,
    "onnx": OnnxReranker,
    "local": LocalReranker,
    "openai": ApiReranker,
}

_reranker = None
_lock = threading.Lock()
_cache: dict[tuple, float] = {}
_CACHE_LIMIT = 20_000
# Провайдер, который уже пробовали запустить и не смогли: повторять попытку
# на каждом запросе бессмысленно — это только добавит задержки.
_failed: dict[str, str] = {}

STATS = {"calls": 0, "pairs": 0, "cached": 0, "ms": 0.0, "failures": 0}


def get():
    """Возвращает готовый реранкер либо None, если он выключен или не запустился."""
    global _reranker
    name = config.RERANKER_PROVIDER
    if name == "none":
        return None
    with _lock:
        if _reranker is not None and getattr(_reranker, "name", None) == name:
            return _reranker
        if name in _failed:
            return None
        if name not in PROVIDERS:
            _failed[name] = "неизвестный провайдер"
            log.error("неизвестный провайдер переранжирования: %s. Доступны: %s",
                      name, ", ".join(PROVIDERS))
            return None
        try:
            _reranker = PROVIDERS[name]()
            log.info("переранжирование включено: %s", name)
        except Exception as exc:  # noqa: BLE001 — работаем без реранкера
            _failed[name] = str(exc)
            log.warning("переранжирование «%s» не запустилось: %s. "
                        "Поиск продолжает работать, выдача идёт в исходном порядке.",
                        name, exc)
            _reranker = None
    return _reranker


def reset() -> None:
    """Сбросить состояние — после смены настроек в админке."""
    global _reranker
    with _lock:
        _reranker = None
        _failed.clear()
    _cache.clear()


def score(query: str, texts: list[str]) -> list[float] | None:
    """Оценки 0…1 для пар «вопрос + фрагмент». None — реранкер не сработал."""
    engine = get()
    if engine is None or not texts:
        return None
    keys = [(engine.name, query, t[:400]) for t in texts]
    out: list[float | None] = [None] * len(texts)
    todo: list[int] = []
    if config.RERANKER_CACHE:
        for i, k in enumerate(keys):
            if k in _cache:
                out[i] = _cache[k]
                STATS["cached"] += 1
            else:
                todo.append(i)
    else:
        todo = list(range(len(texts)))
    if todo:
        started = time.time()
        try:
            fresh = _with_timeout(engine.score_pairs, query, [texts[i] for i in todo])
        except Exception as exc:  # noqa: BLE001 — любая беда = идём без реранкера
            STATS["failures"] += 1
            log.warning("переранжирование не отработало: %s", exc)
            return None
        if fresh is None:
            STATS["failures"] += 1
            log.warning("переранжирование не уложилось в %.0f с — "
                        "выдача в исходном порядке", config.RERANKER_TIMEOUT)
            return None
        for i, s in zip(todo, fresh):
            out[i] = float(s)
            if config.RERANKER_CACHE:
                if len(_cache) >= _CACHE_LIMIT:
                    _cache.clear()
                _cache[keys[i]] = float(s)
        STATS["calls"] += 1
        STATS["pairs"] += len(todo)
        STATS["ms"] += (time.time() - started) * 1000
    return [float(x or 0.0) for x in out]


def _with_timeout(fn, *args):
    """Считает в отдельном потоке: зависший реранкер не держит ответ вечно."""
    result: list = [None]
    error: list = [None]

    def run():
        try:
            result[0] = fn(*args)
        except Exception as exc:  # noqa: BLE001
            error[0] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(config.RERANKER_TIMEOUT)
    if thread.is_alive():
        return None
    if error[0] is not None:
        raise error[0]
    return result[0]


def blend(base_scores: list[float], rerank_scores: list[float]) -> list[float]:
    """
    Смешивает оценку поиска с оценкой реранкера.

    Итог остаётся в той же шкале, что и исходные оценки: максимум не
    меняется. Это важно, потому что порог MIN_CONFIDENCE, по которому
    ассистент отказывается отвечать, настроен именно на неё. Реранкер
    здесь меняет порядок, а не смысл числа.

    Оценки поиска приводятся к отрезку от нуля до единицы — их абсолютная
    величина ничего не значит сама по себе. А оценки реранкера берутся
    как есть: это уже вероятности «фрагмент отвечает на вопрос».
    Разница принципиальная. Если растянуть и их тоже, то в случае, когда
    реранкер честно говорит «все двадцать одинаково средние», крошечные
    случайные различия раздуются до полной шкалы и перемешают выдачу.
    Пока реранкер не видит разницы, порядок определяет поиск.
    """
    if not base_scores:
        return []
    w = max(0.0, min(1.0, config.RERANKER_WEIGHT))
    base = np.asarray(base_scores, dtype=np.float64)
    rr = np.clip(np.asarray(rerank_scores, dtype=np.float64), 0.0, 1.0)
    top = float(base.max())

    lo, hi = float(base.min()), float(base.max())
    unit_base = (np.full_like(base, 0.5) if hi - lo < 1e-12 else (base - lo) / (hi - lo))

    mixed = (1.0 - w) * unit_base + w * rr
    peak = float(mixed.max()) or 1.0
    return [float(x * top / peak) for x in mixed]


def describe() -> dict:
    """Состояние реранкера для веб-интерфейса и диагностики."""
    name = config.RERANKER_PROVIDER
    info = {"provider": name, "enabled": name != "none", "ready": False,
            "error": None, "stats": dict(STATS)}
    if name == "none":
        info["detail"] = "выключено — выдача идёт в том порядке, что дал поиск"
        return info
    engine = get()
    info["ready"] = engine is not None
    info["error"] = _failed.get(name)
    if STATS["calls"]:
        info["avg_ms"] = round(STATS["ms"] / STATS["calls"], 1)
    if name == "lexical":
        info["detail"] = "встроенный лексический реранкер, моделей не требует"
    elif engine is not None:
        info["detail"] = config.RERANKER_MODEL
    else:
        info["detail"] = "не запустился, поиск работает без него"
    return info
