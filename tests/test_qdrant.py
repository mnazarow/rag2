"""
Проверка клиента Qdrant на подставном сервере.

Настоящего Qdrant в тестовом окружении нет, поэтому здесь поднимается
маленький сервер, отвечающий на те же адреса, что и Qdrant. Это
проверяет ровно то, что может сломаться в нашем коде: какие запросы
уходят, как читается ответ, нормализуются ли векторы, режется ли
загрузка на пачки, что происходит при ошибке.

Совместимость с настоящим Qdrant этим, разумеется, не доказывается —
проверять её нужно на живом сервере, и в документации это сказано прямо.
"""
from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tests.base import Isolated  # noqa: F401 — путь к проекту

import numpy as np


class FakeQdrant(BaseHTTPRequestHandler):
    """Минимальный Qdrant в памяти: коллекция, точки, поиск по косинусу."""

    points: dict[int, list[float]] = {}
    payloads: dict[int, dict] = {}
    collection: dict | None = None
    batches: list[int] = []

    def log_message(self, *args):
        pass

    @property
    def route(self) -> str:
        """Путь без строки запроса: Qdrant принимает ?wait=true и подобное."""
        return self.path.split("?", 1)[0]

    def _read(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    def _reply(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.route.startswith("/collections/"):
            if FakeQdrant.collection is None:
                return self._reply(404, {"status": {"error": "Not found"}})
            return self._reply(200, {"result": {
                "points_count": len(FakeQdrant.points),
                "status": "green",
                "config": {"params": {"vectors": FakeQdrant.collection}}}})
        return self._reply(404, {})

    def do_PUT(self):
        body = self._read()
        if self.route.endswith("/points"):
            FakeQdrant.batches.append(len(body["points"]))
            for p in body["points"]:
                FakeQdrant.points[int(p["id"])] = p["vector"]
                FakeQdrant.payloads[int(p["id"])] = p.get("payload") or {}
            return self._reply(200, {"result": {"status": "completed"}})
        if "/index" in self.route:
            return self._reply(200, {"result": {}})
        FakeQdrant.collection = body["vectors"]
        return self._reply(200, {"result": True})

    def do_POST(self):
        body = self._read()
        if self.route.endswith("/points/search"):
            query = np.asarray(body["vector"], dtype=np.float32)
            allowed = None
            flt = body.get("filter", {}).get("must", [])
            for cond in flt:
                if "has_id" in cond:
                    allowed = set(cond["has_id"])
                if cond.get("key") == "section":
                    wanted = set(cond["match"]["any"])
                    allowed = {i for i, pl in FakeQdrant.payloads.items()
                               if pl.get("section") in wanted}
            scored = []
            for pid, vec in FakeQdrant.points.items():
                if allowed is not None and pid not in allowed:
                    continue
                scored.append((pid, float(np.dot(query, np.asarray(vec, np.float32)))))
            scored.sort(key=lambda x: -x[1])
            return self._reply(200, {"result": [
                {"id": p, "score": s} for p, s in scored[:body.get("limit", 10)]]})
        if self.route.endswith("/points/delete"):
            for pid in body["points"]:
                FakeQdrant.points.pop(int(pid), None)
            return self._reply(200, {"result": {}})
        if self.route.endswith("/points/scroll"):
            return self._reply(200, {"result": {
                "points": [{"id": p} for p in FakeQdrant.points],
                "next_page_offset": None}})
        return self._reply(404, {})

    def do_DELETE(self):
        FakeQdrant.collection = None
        FakeQdrant.points.clear()
        return self._reply(200, {"result": True})


class TestQdrantStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeQdrant)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        FakeQdrant.points.clear()
        FakeQdrant.payloads.clear()
        FakeQdrant.batches.clear()
        FakeQdrant.collection = None
        import config
        config.QDRANT_BATCH = 4
        import vectors_qdrant
        self.mod = vectors_qdrant
        self.store = vectors_qdrant.QdrantStore(
            url=f"http://127.0.0.1:{self.port}", collection="test", dim=8)

    def test_collection_created_with_right_dim(self):
        self.assertEqual(FakeQdrant.collection["size"], 8)
        self.assertEqual(FakeQdrant.collection["distance"], "Cosine")

    def test_add_normalises_and_batches(self):
        vectors = np.random.default_rng(0).standard_normal((10, 8)).astype(np.float32) * 5
        self.store.add(list(range(10)), vectors)
        self.assertEqual(len(FakeQdrant.points), 10)
        # Загрузка режется на пачки по QDRANT_BATCH.
        self.assertEqual(FakeQdrant.batches, [4, 4, 2])
        for vec in FakeQdrant.points.values():
            self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=5)

    def test_search_returns_nearest(self):
        vectors = np.eye(8, dtype=np.float32)
        self.store.add(list(range(8)), vectors)
        found = self.store.search(np.eye(8, dtype=np.float32)[3], top_k=3)
        self.assertEqual(found[0][0], 3)
        self.assertAlmostEqual(found[0][1], 1.0, places=5)

    def test_search_with_allowed_ids(self):
        self.store.add(list(range(8)), np.eye(8, dtype=np.float32))
        found = self.store.search(np.eye(8, dtype=np.float32)[3], top_k=5,
                                  allowed={5, 6})
        self.assertEqual({p for p, _ in found}, {5, 6})

    def test_filter_by_section_inside_store(self):
        """
        Ради этого Qdrant и нужен: раньше роль отсекала документы уже
        после поиска, и выдача могла остаться пустой.
        """
        self.store.add([1, 2], np.eye(8, dtype=np.float32)[:2],
                       payloads=[{"section": "РОЗНИЦА", "is_current": 1},
                                 {"section": "ДИЛЕРЫ", "is_current": 1}])
        found = self.store.search_filtered(np.eye(8, dtype=np.float32)[1], top_k=5,
                                           sections=["РОЗНИЦА"], current_only=False)
        self.assertEqual([p for p, _ in found], [1])

    def test_drop(self):
        self.store.add([1, 2, 3], np.eye(8, dtype=np.float32)[:3])
        self.store.drop_chunks([2])
        self.assertNotIn(2, FakeQdrant.points)
        self.assertEqual(len(self.store), 2)

    def test_health_and_ids(self):
        self.store.add([7, 8], np.eye(8, dtype=np.float32)[:2])
        health = self.store.health()
        self.assertTrue(health["ok"])
        self.assertEqual(health["points"], 2)
        self.assertEqual(set(self.store.ids), {7, 8})

    def test_unreachable_server_raises_clearly(self):
        with self.assertRaises(self.mod.QdrantError) as ctx:
            self.mod.QdrantStore(url="http://127.0.0.1:1", collection="t", dim=8)
        self.assertIn("недоступен", str(ctx.exception))

    def test_dimension_conflict_is_explained(self):
        with self.assertRaises(self.mod.QdrantError) as ctx:
            self.mod.QdrantStore(url=f"http://127.0.0.1:{self.port}",
                                 collection="test", dim=1024)
        self.assertIn("пересоздайте", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
