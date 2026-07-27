"""
Хранилище векторов в Qdrant — за тем же интерфейсом, что и numpy-индекс.

Зачем. Сейчас векторы лежат в одном файле и грузятся в память целиком.
На нынешних объёмах это работает быстро, но упирается в три вещи:
память растёт линейно с базой, две записи одновременно портят файл,
и фильтровать (по разделу, бренду, актуальности) приходится уже после
поиска, а не внутри него.

Qdrant снимает всё три. Разворачивается одним контейнером:

    docker run -d --name qdrant -p 6333:6333 \\
        -v $(pwd)/data/qdrant:/qdrant/storage qdrant/qdrant

Дальше в настройках: VECTOR_BACKEND=qdrant и QDRANT_URL=http://127.0.0.1:6333.
Перенос существующего индекса — одной командой, файлы заново не
разбираются:

    python vectors_qdrant.py migrate

Обращение идёт по обычному HTTP через httpx, без клиентской библиотеки:
одной зависимостью меньше, а API Qdrant стабилен и хорошо задокументирован.
"""
from __future__ import annotations

import argparse
import sys
from typing import Iterable, Sequence

import numpy as np

import config
import logging_setup

log = logging_setup.get("db")


class QdrantError(RuntimeError):
    pass


class QdrantStore:
    """
    Тот же набор методов, что у numpy-хранилища: add, drop_chunks,
    search, save, load, len. Благодаря этому поиск и индексация
    не знают, куда именно легли векторы.
    """

    def __init__(self, url: str | None = None, collection: str | None = None,
                 dim: int | None = None) -> None:
        import httpx
        self.url = (url or config.QDRANT_URL).rstrip("/")
        self.collection = collection or config.QDRANT_COLLECTION
        self.dim = dim or config.EMBEDDINGS_DIM
        headers = {"api-key": config.QDRANT_API_KEY} if config.QDRANT_API_KEY else {}
        self.client = httpx.Client(timeout=config.QDRANT_TIMEOUT, headers=headers)
        self._ensure_collection()

    # ----------------------------------------------------------- служебное --
    def _call(self, method: str, path: str, **kwargs):
        try:
            r = self.client.request(method, self.url + path, **kwargs)
        except Exception as exc:  # noqa: BLE001
            raise QdrantError(f"Qdrant недоступен по адресу {self.url}: {exc}") from exc
        if r.status_code >= 400:
            raise QdrantError(f"Qdrant ответил {r.status_code}: {r.text[:300]}")
        return r.json()

    def _ensure_collection(self) -> None:
        try:
            info = self._call("GET", f"/collections/{self.collection}")
            existing = info["result"]["config"]["params"]["vectors"]["size"]
            if existing != self.dim:
                raise QdrantError(
                    f"в коллекции «{self.collection}» векторы по {existing} измерений, "
                    f"а модель даёт {self.dim}. Пересоздайте коллекцию: "
                    f"python vectors_qdrant.py migrate --recreate")
            self.dim = existing
            return
        except QdrantError as exc:
            if "404" not in str(exc):
                raise
        self._call("PUT", f"/collections/{self.collection}", json={
            "vectors": {"size": self.dim, "distance": "Cosine"},
            # Фильтрация по разделу и бренду выполняется внутри хранилища,
            # поэтому по этим полям нужен индекс.
            "optimizers_config": {"default_segment_number": 2},
        })
        for field in ("doc_id", "section", "brand", "is_current"):
            try:
                self._call("PUT", f"/collections/{self.collection}/index",
                           json={"field_name": field,
                                 "field_schema": "integer" if field in
                                 ("doc_id", "is_current") else "keyword"},
                           params={"wait": "true"})
            except QdrantError:
                pass
        log.info("создана коллекция Qdrant «%s» на %d измерений",
                 self.collection, self.dim)

    # -------------------------------------------------------------- запись --
    def add(self, chunk_ids: Sequence[int], vectors: np.ndarray,
            payloads: list[dict] | None = None) -> None:
        if len(chunk_ids) == 0:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.clip(norms, 1e-9, None)
        self.dim = vectors.shape[1]
        points = []
        for i, cid in enumerate(chunk_ids):
            points.append({"id": int(cid), "vector": vectors[i].tolist(),
                           "payload": (payloads[i] if payloads else {})})
        # Пачками: одним запросом на десятки тысяч точек Qdrant подавится.
        for start in range(0, len(points), config.QDRANT_BATCH):
            self._call("PUT", f"/collections/{self.collection}/points",
                       params={"wait": "true"},
                       json={"points": points[start:start + config.QDRANT_BATCH]})

    def drop_chunks(self, chunk_ids: Iterable[int]) -> None:
        ids = [int(c) for c in chunk_ids]
        if not ids:
            return
        self._call("POST", f"/collections/{self.collection}/points/delete",
                   params={"wait": "true"}, json={"points": ids})

    # -------------------------------------------------------------- чтение --
    def search(self, vector: np.ndarray, top_k: int,
               allowed: set[int] | None = None) -> list[tuple[int, float]]:
        vec = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm:
            vec = vec / norm
        body: dict = {"vector": vec.tolist(), "limit": top_k, "with_payload": False}
        # Список разрешённых фрагментов может быть огромным; передавать его
        # целиком бессмысленно. Роль отсекается по разделу в payload.
        if allowed is not None and len(allowed) <= 1000:
            body["filter"] = {"must": [{"has_id": [int(x) for x in allowed]}]}
        result = self._call("POST", f"/collections/{self.collection}/points/search",
                            json=body)
        return [(int(p["id"]), float(p["score"])) for p in result.get("result", [])]

    def search_filtered(self, vector: np.ndarray, top_k: int,
                        sections: list[str] | None = None,
                        current_only: bool = True) -> list[tuple[int, float]]:
        """
        Поиск с отбором внутри хранилища.

        Ради этого Qdrant и нужен: раньше роль отсекала документы уже
        после поиска, и если весь топ оказывался за пределами доступных
        разделов, выдача становилась пустой. Здесь отбор идёт до того,
        как считается близость.
        """
        vec = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm:
            vec = vec / norm
        must: list[dict] = []
        if sections:
            must.append({"key": "section", "match": {"any": sections}})
        if current_only:
            must.append({"key": "is_current", "match": {"value": 1}})
        body = {"vector": vec.tolist(), "limit": top_k, "with_payload": False}
        if must:
            body["filter"] = {"must": must}
        result = self._call("POST", f"/collections/{self.collection}/points/search",
                            json=body)
        return [(int(p["id"]), float(p["score"])) for p in result.get("result", [])]

    # ------------------------------------------------- совместимость с API --
    def save(self) -> None:
        """Qdrant пишет на диск сам — метод оставлен ради общего интерфейса."""

    def load(self) -> None:
        """То же: коллекция открывается при подключении."""

    def __len__(self) -> int:
        try:
            info = self._call("GET", f"/collections/{self.collection}")
            return int(info["result"].get("points_count") or 0)
        except QdrantError:
            return 0

    @property
    def ids(self) -> list[int]:
        """
        Полный список идентификаторов. Нужен только для сверки индекса;
        на большой коллекции это дорого, поэтому читается порциями.
        """
        out: list[int] = []
        offset = None
        while True:
            body = {"limit": 10000, "with_payload": False, "with_vector": False}
            if offset is not None:
                body["offset"] = offset
            res = self._call("POST", f"/collections/{self.collection}/points/scroll",
                             json=body)["result"]
            out.extend(int(p["id"]) for p in res.get("points", []))
            offset = res.get("next_page_offset")
            if not offset:
                return out

    def health(self) -> dict:
        try:
            info = self._call("GET", f"/collections/{self.collection}")["result"]
            return {"ok": True, "points": info.get("points_count"),
                    "status": info.get("status"),
                    "dim": info["config"]["params"]["vectors"]["size"],
                    "url": self.url, "collection": self.collection}
        except QdrantError as exc:
            return {"ok": False, "error": str(exc), "url": self.url,
                    "collection": self.collection}


# ------------------------------------------------------------- перенос -----
def migrate(recreate: bool = False, progress=None) -> dict:
    """Переносит векторы из файла в Qdrant, ничего не переразбирая."""
    import db
    say = progress or (lambda t: print(t, flush=True))
    db.init()
    source = db.VectorStore()
    if len(source) == 0:
        raise QdrantError("векторный индекс пуст — сначала выполните reembed")
    store = QdrantStore(dim=source.dim)
    if recreate:
        say(f"Пересоздаю коллекцию {store.collection}")
        store._call("DELETE", f"/collections/{store.collection}")
        store._ensure_collection()

    # Payload нужен, чтобы фильтровать по разделу прямо в хранилище.
    meta = {r["id"]: r for r in db.q("""
        SELECT c.id, d.section, d.brand, d.is_current, c.doc_id
        FROM chunks c JOIN documents d ON d.id = c.doc_id""")}
    say(f"Переношу {len(source)} векторов по {source.dim} измерений")
    batch = config.QDRANT_BATCH
    for start in range(0, len(source.ids), batch):
        ids = source.ids[start:start + batch]
        rows = source.matrix[start:start + batch]
        payloads = [{"doc_id": (meta.get(i) or {}).get("doc_id"),
                     "section": (meta.get(i) or {}).get("section") or "",
                     "brand": (meta.get(i) or {}).get("brand") or "",
                     "is_current": int((meta.get(i) or {}).get("is_current") or 1)}
                    for i in ids]
        store.add(ids, rows, payloads)
        say(f"{min(start + batch, len(source.ids))}/{len(source.ids)}")
    health = store.health()
    say(f"Готово: в коллекции {health.get('points')} точек")
    return health


def main() -> int:
    p = argparse.ArgumentParser(description="Векторы в Qdrant")
    p.add_argument("command", choices=["migrate", "health", "drop"])
    p.add_argument("--recreate", action="store_true")
    args = p.parse_args()
    if args.command == "migrate":
        migrate(recreate=args.recreate)
    elif args.command == "health":
        import json
        print(json.dumps(QdrantStore().health(), ensure_ascii=False, indent=2))
    elif args.command == "drop":
        store = QdrantStore()
        store._call("DELETE", f"/collections/{store.collection}")
        print("коллекция удалена")
    return 0


if __name__ == "__main__":
    sys.exit(main())
