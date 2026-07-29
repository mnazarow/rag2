"""
Смысловая модель поиска, обучаемая на вашей же базе.

Зачем она нужна. Готовые модели вроде BGE-M3 дают лучшее качество, но
требуют полутора гигабайт весов, библиотеки на несколько гигабайт и
загрузки из интернета, который в России к этой площадке закрыт. Эта
модель обучается прямо на вашей базе за пару минут, работает на
процессоре, не требует ничего кроме numpy и даёт настоящую смысловую
близость — а не хэш-заглушку.

Как устроено. Каждый фрагмент превращается в разреженный вектор весов
слов (частота слова, делённая на его распространённость по базе), затем
эта матрица раскладывается усечённым сингулярным разложением. Полученные
двести пятьдесят шесть измерений и есть «смысл»: слова, встречающиеся в
одних и тех же документах, оказываются рядом, и запрос «производительность
насоса» находит фрагмент, где написано «подача 3,6 м³/ч».

Разложение считается рандомизированным методом в два прохода по
разреженной матрице — это стандартный приём для больших матриц, и он
укладывается в память при любом разумном размере базы.

Ограничение, о котором нужно знать честно: модель понимает только те
слова, которые есть в вашей базе, и не знает синонимов, ни разу в ней не
встретившихся. Готовая модель на этом выигрывает. Поэтому LSA — хороший
рабочий вариант на старте и запасной, когда до внешних моделей не
достучаться, но при первой возможности стоит перейти на BGE-M3 или
USER-bge-m3.
"""
from __future__ import annotations

import json
import pickle
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

import logging_setup

log = logging_setup.get("embed")

_TOKEN_RX = re.compile(r"[a-zа-яё0-9][a-zа-яё0-9\-./]{1,}")

# ---------------------------------------------------------- основа слова ----
# Русский язык изменяет слово на конце: «скважина», «скважины», «скважинный»,
# «в скважине». Пока поиск считает это разными словами, он не находит
# очевидного. Здесь работает стандартный алгоритм Snowball для русского —
# тот же, что применяют поисковые системы. Он не требует словаря, работает
# за микросекунды и сводит все эти формы к одной основе «скважин».
#
# Аккуратность важна в обе стороны: артикулы и обозначения моделей
# («SPL WRP-A 2ECO6-38», «500095.F») не трогаются вовсе — там любое
# «упрощение» означает, что деталь перестанет находиться.
_VOWELS = "аеиоуыэюя"

_PERFECTIVE_1 = ("вшись", "вши", "в")
_PERFECTIVE_2 = ("ившись", "ывшись", "ивши", "ывши", "ив", "ыв")
_ADJECTIVE = ("иями", "ями", "ими", "ыми", "его", "ого", "ему", "ому", "ее", "ие",
              "ые", "ое", "ей", "ий", "ый", "ой", "ем", "им", "ым", "ом", "их",
              "ых", "ую", "юю", "ая", "яя", "ою", "ею")
_PARTICIPLE_1 = ("ющ", "нн", "вш", "ем", "щ")
_PARTICIPLE_2 = ("ивш", "ывш", "ующ")
_REFLEXIVE = ("ся", "сь")
_VERB_1 = ("ешь", "ете", "йте", "нно", "ла", "на", "ли", "ем", "ло", "но", "ет",
           "ют", "ны", "ть", "й", "л", "н")
_VERB_2 = ("ейте", "уйте", "ила", "ыла", "ена", "ите", "или", "ыли", "ило", "ыло",
           "ено", "ует", "уют", "ить", "ыть", "ишь", "ены", "ей", "уй", "ил", "ыл",
           "им", "ым", "ен", "ят", "ит", "ыт", "ую", "ю")
_NOUN = ("иями", "ями", "ами", "иях", "иям", "ием", "ией", "ев", "ов", "ие", "ье",
         "еи", "ии", "ей", "ой", "ий", "ям", "ем", "ам", "ом", "ах", "ях", "ию",
         "ью", "ия", "ья", "а", "е", "и", "й", "о", "у", "ы", "ь", "ю", "я")
_SUPERLATIVE = ("ейше", "ейш")
_DERIVATIONAL = ("ость", "ост")

# Внутри каждой группы длинные окончания должны проверяться первыми,
# иначе «скважины» потеряет только «ы» вместо «ины».
_PERFECTIVE_1, _PERFECTIVE_2, _ADJECTIVE, _PARTICIPLE_1, _PARTICIPLE_2, \
    _REFLEXIVE, _VERB_1, _VERB_2, _NOUN = (
        tuple(sorted(group, key=len, reverse=True))
        for group in (_PERFECTIVE_1, _PERFECTIVE_2, _ADJECTIVE, _PARTICIPLE_1,
                      _PARTICIPLE_2, _REFLEXIVE, _VERB_1, _VERB_2, _NOUN))

STOP = {
    "и", "в", "во", "не", "что", "он", "на", "с", "со", "как", "а", "то", "все",
    "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по",
    "только", "ее", "мне", "было", "вот", "от", "меня", "еще", "нет", "о", "из",
    "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли", "если", "уже", "или",
    "ни", "быть", "был", "него", "до", "вас", "нибудь", "опять", "уж", "вам",
    "ведь", "там", "потом", "себя", "ничего", "ей", "может", "они", "тут", "где",
    "есть", "надо", "ней", "для", "мы", "тебя", "их", "чем", "была", "сам",
    "чтоб", "без", "будто", "чего", "раз", "тоже", "себе", "под", "будет", "ж",
    "тогда", "кто", "этот", "того", "потому", "этого", "какой", "совсем", "ним",
    "здесь", "этом", "один", "почти", "мой", "тем", "чтобы", "нее", "были",
    "куда", "зачем", "всех", "никогда", "можно", "при", "наконец", "два", "об",
    "другой", "хоть", "после", "над", "больше", "тот", "через", "эти", "нас",
    "про", "всего", "них", "какая", "много", "разве", "три", "эту", "моя",
    "впрочем", "хорошо", "свою", "этой", "перед", "иногда", "лучше", "чуть",
    "том", "нельзя", "такой", "им", "более", "всегда", "конечно", "всю", "между",
    "the", "of", "and", "to", "in", "is", "it", "for", "on", "as", "with", "by",
}


def _regions(word: str) -> tuple[int, int]:
    """Границы, за которыми разрешено отрезать: RV и R2 из алгоритма Snowball."""
    rv = len(word)
    for i, ch in enumerate(word):
        if ch in _VOWELS:
            rv = i + 1
            break
    r1 = len(word)
    for i in range(1, len(word)):
        if word[i] not in _VOWELS and word[i - 1] in _VOWELS:
            r1 = i + 1
            break
    r2 = len(word)
    for i in range(r1 + 1, len(word)):
        if word[i] not in _VOWELS and word[i - 1] in _VOWELS:
            r2 = i + 1
            break
    return rv, r2


def _cut(word: str, endings, start: int, prefix: str = "") -> str | None:
    """
    Отрезает первое подходящее окончание, если оно целиком лежит за start.

    prefix — обязательная буква перед окончанием (в правилах алгоритма
    часть форм распознаётся только после «а» или «я»: «работ-а-ет»).
    Сама эта буква тоже отрезается.
    """
    for end in endings:
        full = prefix + end
        if word.endswith(full) and len(word) - len(full) >= start:
            return word[: len(word) - len(full)]
    return None


_stem_cache: dict[str, str] = {}


def normalize_token(token: str) -> str:
    """
    Основа слова: «скважина», «скважины», «скважинный» → «скважин».

    Слова с цифрами (артикулы, обозначения моделей) возвращаются как есть.
    """
    token = token.strip("-./")
    if len(token) < 4 or any(ch.isdigit() for ch in token):
        return token
    cached = _stem_cache.get(token)
    if cached is not None:
        return cached
    stem = _stem(token)
    # Второй проход выравнивает известную неровность алгоритма: «клапанов»
    # после одного прохода становится «клапан», а само «клапан» — «клап»,
    # и одно и то же слово в разных падежах расходится. Повторное усечение
    # приводит обе формы к общей основе.
    if len(stem) >= 5:
        stem = _stem(stem)
    _stem_cache[token] = stem
    return stem


def _stem(token: str) -> str:
    word = token.replace("ё", "е")
    if not any(c in _VOWELS for c in word):
        return token
    rv, r2 = _regions(word)

    # Шаг 1: деепричастие, либо возвратность + прилагательное/глагол/существительное.
    step1 = _cut(word, _PERFECTIVE_1, rv, "а") or _cut(word, _PERFECTIVE_1, rv, "я") \
        or _cut(word, _PERFECTIVE_2, rv)
    if step1 is None:
        word = _cut(word, _REFLEXIVE, rv) or word
        rv, r2 = _regions(word)
        adjectival = _cut(word, _ADJECTIVE, rv)
        if adjectival is not None:
            word = (_cut(adjectival, _PARTICIPLE_1, rv, "а")
                    or _cut(adjectival, _PARTICIPLE_1, rv, "я")
                    or _cut(adjectival, _PARTICIPLE_2, rv) or adjectival)
        else:
            word = (_cut(word, _VERB_1, rv, "а") or _cut(word, _VERB_1, rv, "я")
                    or _cut(word, _VERB_2, rv) or _cut(word, _NOUN, rv) or word)
    else:
        word = step1

    # Шаг 2: висящее «и».
    if word.endswith("и") and len(word) - 1 >= rv:
        word = word[:-1]
    # Шаг 3: словообразовательный суффикс «ость».
    word = _cut(word, _DERIVATIONAL, r2) or word
    # Шаг 4: удвоенное «нн», превосходная степень, мягкий знак.
    if word.endswith("нн"):
        word = word[:-1]
    else:
        superlative = _cut(word, _SUPERLATIVE, rv)
        if superlative is not None:
            word = superlative[:-1] if superlative.endswith("нн") else superlative
    if word.endswith("ь"):
        word = word[:-1]
    # Слишком короткая основа теряет смысл — оставляем исходное слово.
    return word if len(word) >= 3 else token


def tokenize(text: str) -> list[str]:
    import normtext
    out = []
    for raw in _TOKEN_RX.findall(normtext.canon(text).lower()):
        if raw in STOP or len(raw) < 2:
            continue
        out.append(normalize_token(raw))
    return out


try:                                   # ускоряет разложение в несколько раз,
    import scipy.sparse as _sp         # но не обязателен: без scipy работает
except Exception:                      # noqa: BLE001   тот же код на numpy
    _sp = None


# Сколько чисел разрешено держать в одном промежуточном массиве.
# 40 млн float32 — это 160 МБ, безопасно даже на слабой машине.
_BUDGET = 40_000_000


class SparseMatrix:
    """
    Разреженная матрица «фрагмент × слово» в формате CSR.

    Умножение написано через np.add.reduceat, а не через поэлементный
    np.add.at: разница в скорости — примерно в тридцать раз, и именно она
    определяет, обучится модель за минуту или за час. Для умножения на
    транспонированную матрицу один раз строится столбцовый порядок (CSC),
    чтобы и там суммировать непрерывными отрезками.

    Если в системе есть scipy, используется он — результат тот же,
    просто быстрее.
    """

    def __init__(self, indptr: np.ndarray, indices: np.ndarray,
                 data: np.ndarray, n_cols: int) -> None:
        self.indptr = np.asarray(indptr, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int32)
        self.data = np.asarray(data, dtype=np.float32)
        self.n_cols = int(n_cols)
        self._csc: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._sp = self._spT = None
        if _sp is not None:
            self._sp = _sp.csr_matrix((self.data, self.indices, self.indptr),
                                      shape=(self.n_rows, self.n_cols))
            self._spT = self._sp.T.tocsr()

    @property
    def n_rows(self) -> int:
        return len(self.indptr) - 1

    @property
    def nnz(self) -> int:
        return len(self.data)

    # ------------------------------------------------------------ внутреннее --
    def _csc_parts(self):
        """Порядок значений по столбцам — строится один раз и переиспользуется."""
        if self._csc is None:
            order = np.argsort(self.indices, kind="stable")
            counts = np.bincount(self.indices, minlength=self.n_cols).astype(np.int64)
            colptr = np.zeros(self.n_cols + 1, dtype=np.int64)
            np.cumsum(counts, out=colptr[1:])
            rows = np.repeat(np.arange(self.n_rows, dtype=np.int32),
                             np.diff(self.indptr))
            self._csc = (colptr, rows[order], self.data[order])
        return self._csc

    @staticmethod
    def _segment_sum(contrib: np.ndarray, seg: np.ndarray, n_out: int,
                     offset: int, out: np.ndarray) -> None:
        """Суммирует contrib по отрезкам seg — пустые отрезки пропускаются."""
        counts = np.diff(seg)
        nonempty = np.nonzero(counts)[0]
        if nonempty.size == 0:
            return
        res = np.add.reduceat(contrib, seg[nonempty], axis=0)
        out[offset + nonempty] = res

    def _batches(self, ptr: np.ndarray, k: int):
        """Режет матрицу на порции так, чтобы промежуток влезал в бюджет."""
        limit = max(_BUDGET // max(k, 1), 1)
        n = len(ptr) - 1
        start = 0
        while start < n:
            base = ptr[start]
            end = int(np.searchsorted(ptr, base + limit, side="right")) - 1
            end = max(end, start + 1)
            end = min(end, n)
            yield start, end
            start = end

    # ------------------------------------------------------------ умножение --
    def dot(self, dense: np.ndarray) -> np.ndarray:
        """X @ M — координаты фрагментов в пространстве слов."""
        dense = np.ascontiguousarray(dense, dtype=np.float32)
        if self._sp is not None:
            return np.asarray(self._sp @ dense, dtype=np.float32)
        k = dense.shape[1]
        out = np.zeros((self.n_rows, k), dtype=np.float32)
        for start, end in self._batches(self.indptr, k):
            a, b = int(self.indptr[start]), int(self.indptr[end])
            if a == b:
                continue
            contrib = dense[self.indices[a:b]] * self.data[a:b, None]
            self._segment_sum(contrib, self.indptr[start:end + 1] - a,
                              end - start, start, out)
        return out

    def transpose_dot(self, dense: np.ndarray) -> np.ndarray:
        """Xᵀ @ M — нужно для уточняющих проходов разложения."""
        dense = np.ascontiguousarray(dense, dtype=np.float32)
        if self._spT is not None:
            return np.asarray(self._spT @ dense, dtype=np.float32)
        k = dense.shape[1]
        colptr, rows, vals = self._csc_parts()
        out = np.zeros((self.n_cols, k), dtype=np.float32)
        for start, end in self._batches(colptr, k):
            a, b = int(colptr[start]), int(colptr[end])
            if a == b:
                continue
            contrib = dense[rows[a:b]] * vals[a:b, None]
            self._segment_sum(contrib, colptr[start:end + 1] - a,
                              end - start, start, out)
        return out


class LSAModel:
    """Обученная модель: словарь, веса редкости слов и матрица проекции."""

    def __init__(self, vocab: dict[str, int], idf: np.ndarray,
                 components: np.ndarray, meta: dict | None = None) -> None:
        self.vocab = vocab
        self.idf = idf.astype(np.float32)
        self.components = components.astype(np.float32)   # (k, V)
        self.meta = meta or {}

    @property
    def dim(self) -> int:
        return self.components.shape[0]

    # ------------------------------------------------------------ обучение --
    @classmethod
    def fit(cls, texts: list[str], dim: int = 256, max_features: int = 60_000,
            min_df: int = 2, progress=None) -> "LSAModel":
        say = progress or (lambda t: log.info("%s", t))
        started = time.time()
        say(f"Собираю словарь по {len(texts)} фрагментам")

        # Первый проход считает, в скольких фрагментах встретилось каждое
        # слово. Разобранные токены не запоминаются: на большой базе это
        # были бы гигабайты памяти, а повторный разбор стоит секунды.
        df: Counter = Counter()
        for i, text in enumerate(texts):
            df.update(set(tokenize(text)))
            if i and i % 20000 == 0:
                say(f"  разобрано {i} из {len(texts)}")

        n_docs = len(texts)
        # Токены с цифрами (артикулы, обозначения моделей) освобождены от
        # порога min_df: уникальный артикул встречается ровно в одном
        # фрагменте — это его нормальное состояние, а не шум. Без этого
        # исключения смысловой канал слеп именно к самым точным запросам.
        candidates = [(t, c) for t, c in df.items()
                      if (c >= min_df or any(ch.isdigit() for ch in t))
                      and c < n_docs * 0.6]
        candidates.sort(key=lambda x: -x[1])
        vocab = {t: i for i, (t, _c) in enumerate(candidates[:max_features])}
        if len(vocab) < 50:
            raise RuntimeError("слишком мало текста для обучения модели поиска")
        say(f"Словарь: {len(vocab)} слов из {len(df)} встреченных")

        idf = np.zeros(len(vocab), dtype=np.float32)
        for term, index in vocab.items():
            idf[index] = np.log((1 + n_docs) / (1 + df[term])) + 1.0

        say("Строю матрицу «фрагмент × слово»")
        indptr = [0]
        indices_all: list[np.ndarray] = []
        data_all: list[np.ndarray] = []
        for text in texts:
            counts: Counter = Counter(t for t in tokenize(text) if t in vocab)
            if not counts:
                indptr.append(indptr[-1])
                continue
            cols = np.fromiter((vocab[t] for t in counts), dtype=np.int32, count=len(counts))
            vals = np.fromiter(counts.values(), dtype=np.float32, count=len(counts))
            vals = (1.0 + np.log(vals)) * idf[cols]
            norm = float(np.linalg.norm(vals))
            if norm > 0:
                vals /= norm
            indices_all.append(cols)
            data_all.append(vals)
            indptr.append(indptr[-1] + len(cols))
        matrix = SparseMatrix(np.asarray(indptr, dtype=np.int64),
                              np.concatenate(indices_all) if indices_all else np.zeros(0, np.int32),
                              np.concatenate(data_all) if data_all else np.zeros(0, np.float32),
                              len(vocab))
        say(f"Ненулевых значений: {matrix.nnz:,}".replace(",", " "))

        dim = min(dim, min(matrix.n_rows, matrix.n_cols) - 1)
        oversample = 16
        rng = np.random.default_rng(42)
        say(f"Считаю разложение до {dim} измерений — самый долгий шаг")
        omega = rng.standard_normal((matrix.n_cols, dim + oversample)).astype(np.float32)
        sample = matrix.dot(omega)
        # Два уточняющих прохода заметно улучшают качество разложения.
        # Между ними ортогонализация: без неё числа расходятся по величине
        # и в float32 разложение теряет точность.
        for step in range(2):
            sample, _ = np.linalg.qr(sample)
            sample = matrix.dot(matrix.transpose_dot(sample))
            say(f"  уточняющий проход {step + 1} из 2 — {time.time() - started:.0f} с")
        q, _ = np.linalg.qr(sample)
        projected = matrix.transpose_dot(q).T           # (k+p, V)
        _u, singular, vt = np.linalg.svd(projected, full_matrices=False)
        components = vt[:dim]

        elapsed = time.time() - started
        say(f"Готово за {elapsed:.0f} с. Измерений: {dim}, объяснённая доля "
            f"первых десяти: {float(singular[:10].sum() / max(singular.sum(), 1e-9)):.2f}")
        meta = {"documents": n_docs, "vocab": len(vocab), "dim": dim,
                "trained_seconds": round(elapsed, 1),
                "singular_head": [round(float(x), 3) for x in singular[:10]]}
        return cls(vocab, idf, components, meta)

    # ------------------------------------------------------- преобразование --
    def transform(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: Counter = Counter(t for t in tokenize(text) if t in self.vocab)
            if not counts:
                continue
            cols = np.fromiter((self.vocab[t] for t in counts), dtype=np.int32, count=len(counts))
            vals = np.fromiter(counts.values(), dtype=np.float32, count=len(counts))
            vals = (1.0 + np.log(vals)) * self.idf[cols]
            norm = float(np.linalg.norm(vals))
            if norm > 0:
                vals /= norm
            vec = self.components[:, cols] @ vals
            n = float(np.linalg.norm(vec))
            out[row] = vec / n if n > 0 else vec
        return out

    # ---------------------------------------------------------- сохранение --
    def save(self, path: Path) -> None:
        """
        Сохраняет модель поиска.

        Через временный файл и переименование. Модель обучается часами и
        существует в единственном экземпляре: остановка процесса посреди
        записи уничтожала её целиком, а система после этого просто тихо
        переходила на поиск по точным словам.
        """
        import db
        db.atomic_write(Path(path), lambda fh: np.savez_compressed(
            fh, idf=self.idf, components=self.components,
            terms=np.array(list(self.vocab.keys()), dtype=object),
            meta=json.dumps(self.meta, ensure_ascii=False)))

    @classmethod
    def load(cls, path: Path) -> "LSAModel":
        blob = np.load(path, allow_pickle=True)
        terms = list(blob["terms"])
        vocab = {t: i for i, t in enumerate(terms)}
        meta = json.loads(str(blob["meta"])) if "meta" in blob else {}
        return cls(vocab, blob["idf"], blob["components"], meta)
