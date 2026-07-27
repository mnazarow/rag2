"""
Провайдеры эмбеддингов — смысловой канал поиска.

  lsa       — модель, обучаемая на вашей же базе (по умолчанию).
              Ничего не скачивает, работает на процессоре, даёт настоящую
              смысловую близость в пределах лексики базы. Обучается за
              минуты командой `python index.py train-lsa`.
  onnx      — готовая модель (BGE-M3, USER-bge-m3, ru-en-RoSBERTa) в формате
              ONNX: качество внешней модели без установки torch.
  local     — sentence-transformers (нужен torch, лучше всего с видеокартой)
  gigachat  — Embeddings API Сбера (дёшево, РФ-юрисдикция, контекст 512 токенов)
  yandex    — Yandex AI Studio Text Embeddings
  openai    — любой OpenAI-совместимый endpoint (Cloud.ru, vLLM, Ollama, LM Studio)
  hashing   — офлайн-заглушка на хэш-векторах символьных n-грамм.
              Смысловой близости не даёт: «производительность» и «подача»
              для неё разные слова. Оставлена только для проверки пайплайна
              на пустой машине, в работе использовать нельзя.

Все векторы нормируются и кэшируются в SQLite по хэшу текста. Ключ кэша
включает отпечаток модели, поэтому после переобучения или смены модели
старые векторы не подмешиваются к новым.

Смена провайдера не требует переразбора базы: `python index.py reembed`
пересчитывает векторы по уже сохранённым фрагментам.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Sequence

import numpy as np

import config
import db


class EmbeddingError(RuntimeError):
    pass


# ------------------------------------------------------------------ кэш ----
def _cache_get(text_hash: str, provider: str) -> np.ndarray | None:
    row = db.q1("SELECT vector, dim FROM embedding_cache WHERE text_hash=? AND provider=?",
                (text_hash, provider))
    if row is None:
        return None
    return np.frombuffer(row["vector"], dtype=np.float32).reshape(row["dim"])


def _cache_put(text_hash: str, provider: str, vec: np.ndarray) -> None:
    db.run("INSERT OR REPLACE INTO embedding_cache(text_hash, provider, dim, vector) "
           "VALUES (?,?,?,?)",
           (text_hash, provider, int(vec.shape[0]), vec.astype(np.float32).tobytes()))


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ------------------------------------------------------------ провайдеры ---
class BaseEmbedder:
    name = "base"
    dim = config.EMBEDDINGS_DIM

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class HashingEmbedder(BaseEmbedder):
    """
    Детерминированный хэш-вектор символьных 3-5 грамм (feature hashing).
    Даёт лексическую близость, устойчив к опечаткам, работает мгновенно.
    Назначение — smoke-тест пайплайна и офлайн-разработка.
    """
    name = "hashing"

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim or config.EMBEDDINGS_DIM

    def _vec(self, text: str) -> np.ndarray:
        text = re.sub(r"\s+", " ", text.lower())[:8000]
        v = np.zeros(self.dim, dtype=np.float32)
        tokens = re.findall(r"[a-zа-яё0-9]+", text)
        grams: list[str] = list(tokens)
        for tok in tokens:
            padded = f"^{tok}$"
            for n in (3, 4):
                grams.extend(padded[i:i + n] for i in range(len(padded) - n + 1))
        for g in grams:
            h = hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            sign = 1.0 if h[4] & 1 else -1.0
            v[idx] += sign
        norm = float(np.linalg.norm(v))
        return v / norm if norm else v

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)


class LSAEmbedder(BaseEmbedder):
    """
    Смысловая модель, обученная на самой базе знаний (см. lsa.py).

    Почему она стоит по умолчанию. Готовые модели дают качество выше, но
    требуют полутора гигабайт весов и загрузки с площадки, доступ к которой
    из России закрыт. Эта модель обучается за минуты на уже разобранных
    фрагментах, не требует ничего кроме numpy — и, в отличие от заглушки,
    действительно понимает, что «производительность насоса» и «подача,
    м³/ч» — про одно и то же, если оба выражения встречаются в базе.

    Модель обучается один раз и сохраняется в файл. Когда база заметно
    вырастает, в журнал пишется напоминание переобучить: новые слова,
    появившиеся после обучения, модели неизвестны.
    """
    name = "lsa"
    MIN_CHUNKS = 200

    def __init__(self, path: str | Path | None = None, allow_train: bool = False) -> None:
        import lsa
        self._lsa = lsa
        self.path = Path(path or config.LSA_MODEL_PATH)
        if self.path.exists():
            self.model = lsa.LSAModel.load(self.path)
        elif allow_train:
            self.model = self.train(save_to=self.path)
        else:
            # Обучаться «на ходу» посреди индексации нельзя: модель вышла бы
            # обученной на первых попавшихся файлах, а не на всей базе.
            raise EmbeddingError(
                f"модель смыслового поиска ещё не обучена ({self.path}). "
                f"Сначала проиндексируйте базу, затем выполните: "
                f"python index.py train-lsa && python index.py reembed")
        self.dim = self.model.dim
        self.name = f"lsa-{self.fingerprint}"
        self._check_staleness()

    @property
    def fingerprint(self) -> str:
        """Отпечаток модели — чтобы кэш векторов не пережил переобучение."""
        head = self.model.components[:, : min(64, self.model.components.shape[1])]
        return hashlib.blake2b(head.tobytes(), digest_size=4).hexdigest()

    # -------------------------------------------------------------- обучение --
    @classmethod
    def corpus(cls, limit: int | None = None) -> list[str]:
        """Тексты для обучения — ровно то, что уходит в поиск."""
        sql = ("SELECT c.context, c.heading, c.text FROM chunks c "
               "JOIN documents d ON d.id = c.doc_id WHERE d.status='ok'")
        if limit:
            sql += f" LIMIT {int(limit)}"
        return ["\n".join(p for p in (r["context"], r["heading"], r["text"]) if p)
                for r in db.q(sql)]

    @classmethod
    def train(cls, save_to: str | Path | None = None, dim: int | None = None,
              max_features: int | None = None, progress=None):
        import lsa
        db.init()
        texts = cls.corpus()
        if len(texts) < cls.MIN_CHUNKS:
            raise EmbeddingError(
                f"для обучения смысловой модели нужно хотя бы {cls.MIN_CHUNKS} "
                f"фрагментов, сейчас в базе {len(texts)}. Сначала выполните "
                f"индексацию: python index.py build")
        model = lsa.LSAModel.fit(
            texts,
            dim=dim or config.LSA_DIM,
            max_features=max_features or config.LSA_MAX_FEATURES,
            min_df=config.LSA_MIN_DF,
            progress=progress or (lambda t: print(f"  {t}", flush=True)))
        path = Path(save_to or config.LSA_MODEL_PATH)
        model.save(path)
        return model

    def _check_staleness(self) -> None:
        trained_on = int(self.model.meta.get("documents") or 0)
        if not trained_on:
            return
        try:
            now = int(db.q1("SELECT COUNT(*) n FROM chunks")["n"])
        except Exception:  # noqa: BLE001 — база ещё не создана
            return
        if now > trained_on * (1 + config.LSA_STALE_RATIO):
            import logging_setup
            logging_setup.get("embed").warning(
                "смысловая модель обучена на %d фрагментах, сейчас в базе %d. "
                "Новые слова ей неизвестны — переобучите: python index.py train-lsa",
                trained_on, now)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        return self.model.transform(list(texts))


# ------------------------------------------------------- токенизатор ONNX ---
class _WordPieceTokenizer:
    """
    Запасной токенизатор для моделей семейства BERT (vocab.txt), когда
    библиотека tokenizers не установлена. Модели на SentencePiece
    (BGE-M3, XLM-R) требуют именно её — об этом сообщается явно.
    """

    def __init__(self, vocab_path: Path) -> None:
        self.vocab: dict[str, int] = {}
        for i, line in enumerate(vocab_path.read_text(encoding="utf-8").splitlines()):
            self.vocab[line.rstrip("\n")] = i
        self.unk = self.vocab.get("[UNK]", 100)
        self.cls = self.vocab.get("[CLS]", 101)
        self.sep = self.vocab.get("[SEP]", 102)
        self.pad = self.vocab.get("[PAD]", 0)

    def _word(self, word: str) -> list[int]:
        if word in self.vocab:
            return [self.vocab[word]]
        out, start = [], 0
        while start < len(word):
            end = len(word)
            piece_id = None
            while start < end:
                piece = word[start:end]
                if start > 0:
                    piece = "##" + piece
                if piece in self.vocab:
                    piece_id = self.vocab[piece]
                    break
                end -= 1
            if piece_id is None:
                return [self.unk]
            out.append(piece_id)
            start = end
        return out

    def _pieces(self, text: str, budget: int) -> list[int]:
        ids: list[int] = []
        for w in re.findall(r"\w+|[^\w\s]", text.lower(), re.UNICODE):
            ids.extend(self._word(w))
            if len(ids) >= budget:
                break
        return ids[:budget]

    def encode(self, text: str, max_tokens: int) -> list[int]:
        return [self.cls] + self._pieces(text, max_tokens - 2) + [self.sep]

    def encode_pair(self, first: str, second: str, max_tokens: int) -> list[int]:
        # Вопрос коротий и важнее — ему отдаём до четверти окна, остальное тексту.
        head = self._pieces(first, max(8, max_tokens // 4))
        tail = self._pieces(second, max_tokens - len(head) - 3)
        return [self.cls] + head + [self.sep] + tail + [self.sep]


class _Tokenizer:
    """Единая обёртка: и для быстрой библиотеки, и для запасного варианта."""

    def __init__(self, fast=None, wordpiece=None, max_tokens: int = 512) -> None:
        self.fast, self.wp, self.max_tokens = fast, wordpiece, max_tokens

    def encode(self, text: str) -> list[int]:
        if self.fast is not None:
            return self.fast.encode(text).ids
        return self.wp.encode(text, self.max_tokens)

    def encode_pair(self, first: str, second: str) -> list[int]:
        if self.fast is not None:
            return self.fast.encode(first, second).ids
        return self.wp.encode_pair(first, second, self.max_tokens)


def _load_tokenizer(directory: str | Path, max_tokens: int) -> _Tokenizer:
    """Токенизатор модели: быстрый из библиотеки tokenizers либо запасной."""
    d = Path(directory)
    if not d.exists():
        raise EmbeddingError(f"папка токенизатора не найдена: {d}")
    tj = d / "tokenizer.json"
    if tj.exists():
        try:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(str(tj))
            tok.enable_truncation(max_length=max_tokens)
            return _Tokenizer(fast=tok, max_tokens=max_tokens)
        except ImportError:
            pass
    vocab = d / "vocab.txt"
    if vocab.exists():
        return _Tokenizer(wordpiece=_WordPieceTokenizer(vocab), max_tokens=max_tokens)
    raise EmbeddingError(
        f"в {d} нет ни vocab.txt, ни работающего tokenizer.json. "
        f"Для моделей на SentencePiece (BGE-M3, XLM-R) установите библиотеку: "
        f"pip install tokenizers")


class OnnxEmbedder(BaseEmbedder):
    """
    Готовая модель в формате ONNX — качество внешней модели без torch.

    Как получить файлы: на машине с интернетом выполнить
        optimum-cli export onnx --model deepvk/USER-bge-m3 ./user-bge-m3-onnx
    и перенести папку на сервер. Затем указать ONNX_MODEL_PATH и
    ONNX_TOKENIZER_DIR. Работает на процессоре; при установленном
    onnxruntime-gpu автоматически задействует видеокарту.
    """
    name = "onnx"

    def __init__(self, model_path: str | None = None) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise EmbeddingError("нужен onnxruntime: pip install onnxruntime") from exc
        path = Path(model_path or config.ONNX_MODEL_PATH or "")
        if not path.exists():
            raise EmbeddingError(
                "не задан ONNX_MODEL_PATH — путь к файлу model.onnx. "
                "См. раздел «Смысловой поиск» в документации.")
        opts = ort.SessionOptions()
        if config.ONNX_THREADS:
            opts.intra_op_num_threads = config.ONNX_THREADS
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(str(path), opts, providers=providers)
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.tokenizer = _load_tokenizer(
            config.ONNX_TOKENIZER_DIR or path.parent, config.ONNX_MAX_TOKENS)
        self.model_name = path.name
        out_shape = self.session.get_outputs()[0].shape
        self.dim = int(out_shape[-1]) if isinstance(out_shape[-1], int) else config.EMBEDDINGS_DIM

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        batch = [self.tokenizer.encode(t)[: config.ONNX_MAX_TOKENS] for t in texts]
        width = max(len(ids) for ids in batch)
        input_ids = np.zeros((len(batch), width), dtype=np.int64)
        mask = np.zeros((len(batch), width), dtype=np.int64)
        for i, ids in enumerate(batch):
            input_ids[i, : len(ids)] = ids
            mask[i, : len(ids)] = 1
        feed = {"input_ids": input_ids, "attention_mask": mask}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.zeros_like(input_ids)
        feed = {k: v for k, v in feed.items() if k in self.inputs}
        out = self.session.run(None, feed)[0]
        if out.ndim == 3:
            if config.ONNX_POOLING == "cls":
                vecs = out[:, 0, :]
            else:
                m = mask[:, :, None].astype(np.float32)
                vecs = (out * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        else:
            vecs = out
        vecs = np.asarray(vecs, dtype=np.float32)
        self.dim = vecs.shape[1]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-9, None)


class LocalEmbedder(BaseEmbedder):
    """sentence-transformers: BGE-M3, USER-bge-m3 (deepvk), Qwen3-Embedding."""
    name = "local"

    def __init__(self, model_name: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError(
                "нужен sentence-transformers: pip install sentence-transformers") from exc
        self.model_name = model_name or config.EMBEDDINGS_MODEL or "BAAI/bge-m3"
        self.model = SentenceTransformer(self.model_name)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True,
                                            batch_size=config.EMBEDDINGS_BATCH),
                          dtype=np.float32)


class GigaChatEmbedder(BaseEmbedder):
    """
    GigaChat Embeddings. Внимание: контекст модели — 512 токенов,
    длинные чанки обрезаются. Учтено в config.CHUNK_TARGET_CHARS.
    """
    name = "gigachat"
    OAUTH = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    API = "https://gigachat.devices.sberbank.ru/api/v1"

    def __init__(self, model: str | None = None) -> None:
        import httpx
        self._httpx = httpx
        self.model = model or config.EMBEDDINGS_MODEL or "Embeddings"
        self.dim = config.EMBEDDINGS_DIM
        self._token: str | None = None
        self._expires = 0.0
        # verify=False: сертификаты Минцифры часто не установлены в системе.
        self.client = httpx.Client(timeout=60, verify=False, proxy=config.LLM_PROXY or None)

    def _auth(self) -> str:
        if self._token and time.time() < self._expires - 60:
            return self._token
        if not config.GIGACHAT_AUTH_KEY:
            raise EmbeddingError("не задан GIGACHAT_AUTH_KEY")
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

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        r = self.client.post(f"{self.API}/embeddings",
                             headers={"Authorization": f"Bearer {self._auth()}"},
                             json={"model": self.model, "input": list(texts)})
        r.raise_for_status()
        vecs = [np.asarray(d["embedding"], dtype=np.float32) for d in r.json()["data"]]
        self.dim = vecs[0].shape[0]
        return np.vstack(vecs)


class YandexEmbedder(BaseEmbedder):
    """Yandex AI Studio Text Embeddings (doc/query — разные модели)."""
    name = "yandex"
    URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding"

    def __init__(self, model: str | None = None) -> None:
        import httpx
        self.client = httpx.Client(timeout=60, proxy=config.LLM_PROXY or None)
        self.doc_uri = model or config.EMBEDDINGS_MODEL or \
            f"emb://{config.YANDEX_FOLDER_ID}/text-search-doc/latest"
        self.query_uri = f"emb://{config.YANDEX_FOLDER_ID}/text-search-query/latest"
        self.dim = config.EMBEDDINGS_DIM

    def _one(self, text: str, uri: str) -> np.ndarray:
        r = self.client.post(self.URL,
                             headers={"Authorization": f"Api-Key {config.YANDEX_API_KEY}",
                                      "x-folder-id": config.YANDEX_FOLDER_ID},
                             json={"modelUri": uri, "text": text[:8000]})
        r.raise_for_status()
        vec = np.asarray(r.json()["embedding"], dtype=np.float32)
        self.dim = vec.shape[0]
        return vec

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        return np.vstack([self._one(t, self.doc_uri) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._one(text, self.query_uri)


class OpenAICompatibleEmbedder(BaseEmbedder):
    """OpenAI-совместимый /v1/embeddings: Cloud.ru, vLLM, Ollama, TEI."""
    name = "openai"

    def __init__(self, model: str | None = None) -> None:
        import httpx
        self.client = httpx.Client(timeout=120, proxy=config.LLM_PROXY or None)
        self.model = model or config.EMBEDDINGS_MODEL or "text-embedding-3-large"
        self.dim = config.EMBEDDINGS_DIM

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        r = self.client.post(f"{config.OPENAI_BASE_URL}/embeddings",
                             headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"},
                             json={"model": self.model, "input": list(texts)})
        r.raise_for_status()
        data = sorted(r.json()["data"], key=lambda d: d.get("index", 0))
        vecs = [np.asarray(d["embedding"], dtype=np.float32) for d in data]
        self.dim = vecs[0].shape[0]
        return np.vstack(vecs)


PROVIDERS = {
    "lsa": LSAEmbedder,
    "onnx": OnnxEmbedder,
    "hashing": HashingEmbedder,
    "local": LocalEmbedder,
    "gigachat": GigaChatEmbedder,
    "yandex": YandexEmbedder,
    "openai": OpenAICompatibleEmbedder,
}

# Провайдеры, которые не дают смысловой близости. Поиск с ними работает,
# но фактически держится на одном BM25 — об этом честно предупреждаем.
STUB_PROVIDERS = {"hashing"}

_embedder: BaseEmbedder | None = None


def get_embedder() -> BaseEmbedder:
    global _embedder
    if _embedder is None:
        name = config.EMBEDDINGS_PROVIDER
        if name not in PROVIDERS:
            raise EmbeddingError(
                f"неизвестный провайдер эмбеддингов: {name}. "
                f"Доступны: {', '.join(PROVIDERS)}")
        _embedder = PROVIDERS[name]()
        if name in STUB_PROVIDERS:
            import logging_setup
            logging_setup.get("embed").warning(
                "смысловой поиск работает на заглушке (%s): близость по смыслу "
                "не учитывается, качество держится на текстовом канале. "
                "Переключите EMBEDDINGS_PROVIDER на lsa или onnx.", name)
    return _embedder


def reset() -> None:
    """Сбросить провайдер — после смены настроек или переобучения модели."""
    global _embedder
    _embedder = None


def describe() -> dict:
    """Что сейчас реально используется — для веб-интерфейса и диагностики."""
    info = {"provider": config.EMBEDDINGS_PROVIDER,
            "is_stub": config.EMBEDDINGS_PROVIDER in STUB_PROVIDERS,
            "ready": False, "dim": None, "detail": "", "error": None}
    try:
        emb = get_embedder()
        info.update(ready=True, dim=emb.dim, name=emb.name)
        if isinstance(emb, LSAEmbedder):
            m = emb.model.meta
            info["detail"] = (
                f"обучена на {m.get('documents', '?')} фрагментах, "
                f"словарь {m.get('vocab', '?')} слов, {emb.dim} измерений")
        else:
            info["detail"] = getattr(emb, "model_name", None) or config.EMBEDDINGS_MODEL or ""
    except Exception as exc:  # noqa: BLE001
        info["error"] = str(exc)
    return info


def embed_texts(texts: Sequence[str], use_cache: bool = True) -> np.ndarray:
    """Эмбеддинги с кэшем: повторная индексация неизменных чанков бесплатна."""
    emb = get_embedder()
    if not texts:
        return np.zeros((0, emb.dim), np.float32)
    out: list[np.ndarray | None] = [None] * len(texts)
    todo_idx: list[int] = []
    todo_txt: list[str] = []
    for i, t in enumerate(texts):
        h = _hash_text(t)
        cached = _cache_get(h, emb.name) if use_cache else None
        if cached is not None:
            out[i] = cached
        else:
            todo_idx.append(i)
            todo_txt.append(t)
    import time as _time
    import metrics
    for start in range(0, len(todo_txt), config.EMBEDDINGS_BATCH):
        batch = todo_txt[start:start + config.EMBEDDINGS_BATCH]
        _t0 = _time.time()
        try:
            vecs = emb.embed(batch)
            metrics.record_model_call(
                getattr(emb, "model_name", None) or getattr(emb, "model", emb.name),
                emb.name, "embedding", sum(len(t) // 4 for t in batch), 0,
                int((_time.time() - _t0) * 1000), True)
        except Exception as exc:
            metrics.record_model_call(emb.name, emb.name, "embedding", 0, 0,
                                      int((_time.time() - _t0) * 1000), False, str(exc))
            raise
        for j, vec in enumerate(vecs):
            i = todo_idx[start + j]
            out[i] = vec
            if use_cache:
                _cache_put(_hash_text(texts[i]), emb.name, vec)
    return np.vstack([v for v in out if v is not None])


def embed_query(text: str) -> np.ndarray:
    return get_embedder().embed_query(text)
