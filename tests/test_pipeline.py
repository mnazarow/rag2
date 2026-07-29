"""
Проверки конвейера целиком: индексация, поиск, копии, очередь задач.

Здесь всё работает на временной базе из нескольких документов, поэтому
набор проходит за секунды и его можно запускать после каждой правки.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from tests.base import Isolated


DOCS = {
    "РОЗНИЦА/GRUNDFOS/1КАТАЛОГ/Каталог GRU-45.txt":
        "Каталог оборудования GRUNDFOS, модель GRU-45.\n"
        "Насос скважинный GRU-45. Подача 8.2 кубометра в час, напор 38 метров.\n"
        "Потребляемая мощность 2 киловатта, напряжение 220 вольт.\n"
        "Корпус из нержавеющей стали, рабочее колесо из технополимера.\n"
        "Комплект поставки: насос, кабель, конденсатор, паспорт изделия.\n",
    "РОЗНИЦА/GRUNDFOS/3РУКОВОДСТВО/Руководство GRU-45.txt":
        "Руководство по монтажу и эксплуатации GRUNDFOS GRU-45.\n"
        "Перед первым пуском заполните корпус водой.\n"
        "Работа всухую разрушает уплотнение за несколько минут.\n"
        "Обязательно установите обратный клапан на напорной линии.\n"
        "Кабель нельзя сращивать в скважине, применяйте термоусадочную муфту.\n",
    "РОЗНИЦА/GRUNDFOS/4ПАСПОРТ/Паспорт GRU-45.txt":
        "Паспорт изделия GRUNDFOS GRU-45.\n"
        "Гарантийный срок эксплуатации 24 месяца со дня продажи.\n"
        "Класс защиты оболочки IP68, температура жидкости до 35 градусов.\n"
        "Допустимое содержание песка не более 180 граммов на кубометр.\n",
    "РОЗНИЦА/WILO/1КАТАЛОГ/Каталог WIL-20.txt":
        "Каталог оборудования WILO, модель WIL-20.\n"
        "Насос циркуляционный WIL-20. Подача 2.4 кубометра в час, напор 6 метров.\n"
        "Применение: система отопления частного дома, тёплый пол.\n",
}


class TestIndexAndSearch(Isolated):
    def setUp(self):
        super().setUp()
        for rel, text in DOCS.items():
            path = Path(self.config.KB_ROOT) / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        import index as index_mod
        self.index = index_mod
        self.stats = index_mod.build(progress=lambda *_a: None)

    def test_documents_indexed(self):
        self.assertEqual(self.stats["error"], 0)
        n = self.db.q1("SELECT COUNT(*) n FROM documents WHERE status='ok'")["n"]
        self.assertEqual(n, len(DOCS))

    def test_metadata_from_path(self):
        row = self.db.q1("SELECT brand, doc_type, section FROM documents "
                         "WHERE file_name LIKE 'Паспорт%'")
        self.assertEqual(row["brand"], "GRUNDFOS")
        self.assertEqual(row["section"], "РОЗНИЦА")
        self.assertIn("ПАСПОРТ", (row["doc_type"] or "").upper())

    def test_text_channel_finds_exact_words(self):
        import search
        hits = search.bm25_search("обратный клапан", 10)
        self.assertTrue(hits, "текстовый канал не нашёл точную фразу")

    def test_reindex_is_incremental(self):
        again = self.index.build(progress=lambda *_a: None)
        self.assertEqual(again["indexed"], 0)
        self.assertEqual(again["unchanged"], len(DOCS))

    def test_deleted_file_leaves_index(self):
        target = Path(self.config.KB_ROOT) / "РОЗНИЦА/WILO/1КАТАЛОГ/Каталог WIL-20.txt"
        target.unlink()
        self.index.build(progress=lambda *_a: None)
        left = self.db.q1("SELECT COUNT(*) n FROM documents WHERE status='ok'")["n"]
        self.assertEqual(left, len(DOCS) - 1)


class TestSemanticChannel(Isolated):
    """Смысловая модель обучается на самой базе и включает второй канал."""

    def setUp(self):
        super().setUp()
        # Модели нужен хоть какой-то словарь, поэтому корпус побольше.
        # Каждый документ делаем непохожим на остальные, иначе индексатор
        # честно отбросит их как дубликаты по содержимому.
        for i in range(70):
            for rel, text in DOCS.items():
                path = Path(self.config.KB_ROOT) / f"вариант{i}" / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                body = (text.replace("GRU-45", f"GRU-{45 + i}")
                            .replace("WIL-20", f"WIL-{20 + i}")
                            + f"\nЗаводской номер партии {100000 + i}, "
                              f"дата выпуска {2020 + i % 6} год.\n")
                path.write_text(body, encoding="utf-8")
        import index as index_mod
        self.index = index_mod
        index_mod.build(progress=lambda *_a: None)

    def test_train_and_reembed(self):
        import embeddings
        import search
        model = embeddings.LSAEmbedder.train(progress=lambda *_a: None)
        self.assertGreater(model.dim, 0)
        embeddings.reset()
        n = self.index.reembed(provider="lsa", progress=lambda *_a: None)
        self.assertGreater(n, 0)
        ok, note = search.dense_ready()
        self.assertTrue(ok, note)
        self.assertTrue(search.dense_search("гарантийный срок", 5))

    def test_dimension_mismatch_is_reported(self):
        """
        Смена провайдера без пересчёта раньше молча давала пустую выдачу.
        Теперь это видно и в проверке, и в журнале.
        """
        import embeddings
        import search
        embeddings.LSAEmbedder.train(progress=lambda *_a: None)
        embeddings.reset()
        self.index.reembed(provider="lsa", progress=lambda *_a: None)
        self.config.EMBEDDINGS_PROVIDER = "hashing"
        embeddings.reset()
        search._dense_warned = False
        ok, note = search.dense_ready()
        self.assertFalse(ok)
        self.assertIn("пересчёт", note)


class TestBackup(Isolated):
    def setUp(self):
        super().setUp()
        self.db.run("INSERT INTO documents(rel_path, abs_path, file_name, ext, "
                    "content_hash, status) VALUES ('a.txt','/a.txt','a.txt','.txt','h','ok')")
        self.db.run("INSERT INTO chunks(doc_id, ord, text, n_chars) VALUES (1,0,'текст',5)")
        self.db.run("INSERT INTO golden_qa(question, answer, created_at) "
                    "VALUES ('вопрос','выверенный ответ','2026-01-01')")

    def test_create_verify_restore(self):
        import backup
        archive = backup.create(quiet=True)
        self.assertTrue(archive.exists())

        report = backup.verify_archive(archive)
        self.assertTrue(report["ok"], report.get("error"))
        self.assertEqual(report["counts"]["documents"], 1)
        self.assertEqual(report["counts"]["golden_qa"], 1)

        # Портим базу и восстанавливаем.
        self.db.run("DELETE FROM golden_qa")
        self.assertEqual(self.db.q1("SELECT COUNT(*) n FROM golden_qa")["n"], 0)
        self.db._local.conn.close()
        self.db._local.conn = None
        backup.restore(archive)
        self.assertEqual(self.db.q1("SELECT COUNT(*) n FROM golden_qa")["n"], 1)

    def test_broken_archive_refused(self):
        import backup
        bad = self.tmp / "index-20200101-000000.tar.gz"
        bad.write_text("это не архив", encoding="utf-8")
        report = backup.verify_archive(bad)
        self.assertFalse(report["ok"])
        with self.assertRaises(backup.BackupError):
            backup.restore(bad)

    def test_retention_keeps_latest(self):
        import backup
        backup.create(quiet=True)
        kept = backup.archives()
        backup.prune(quiet=True)
        self.assertIn(kept[0], backup.archives())


class TestJobQueue(Isolated):
    def setUp(self):
        super().setUp()
        import jobs
        self.jobs = jobs
        jobs.ensure_tables()
        self.db.run("DELETE FROM jobs")

    def test_conflicting_jobs_refused(self):
        self.jobs.enqueue("reindex", "переиндексация")
        with self.assertRaises(self.jobs.Busy):
            self.jobs.enqueue("reembed", "пересчёт векторов")
        with self.assertRaises(self.jobs.Busy):
            self.jobs.enqueue("ocr", "распознавание")

    def test_independent_jobs_allowed(self):
        self.jobs.enqueue("reindex", "переиндексация")
        job = self.jobs.enqueue("backup", "копия")
        self.assertEqual(job["status"], "queued")

    def test_runs_and_records_result(self):
        @self.jobs.handler("test_ok")
        def _ok(payload, progress):
            progress("шаг")
            return {"echo": payload.get("x")}
        self.jobs.RESOURCES["test_ok"] = ("test",)
        job = self.jobs.enqueue("test_ok", "проверка", {"x": 42})
        self.jobs._run_one(self.jobs._claim())
        done = self.jobs.get(job["id"])
        self.assertEqual(done["status"], "done")
        self.assertEqual(done["result"]["echo"], 42)

    def test_failure_is_recorded(self):
        @self.jobs.handler("test_fail")
        def _fail(payload, progress):
            raise RuntimeError("так и было задумано")
        self.jobs.RESOURCES["test_fail"] = ("test",)
        job = self.jobs.enqueue("test_fail", "падение")
        self.jobs._run_one(self.jobs._claim())
        done = self.jobs.get(job["id"])
        self.assertEqual(done["status"], "error")
        self.assertIn("так и было задумано", done["error"])

    def test_stale_job_frees_resource(self):
        """Убитый процесс не должен блокировать кнопку навсегда."""
        self.jobs.enqueue("reindex", "переиндексация")
        self.jobs._claim()
        self.db.run("UPDATE jobs SET heartbeat='2000-01-01T00:00:00+00:00'")
        self.assertEqual(self.jobs.reap_stale(), 1)
        job = self.jobs.enqueue("reindex", "ещё раз")
        self.assertEqual(job["status"], "queued")


class TestAccess(Isolated):
    def test_request_and_approve(self):
        import access
        self.config.TELEGRAM_ALLOWED_IDS.clear()
        self.config.TELEGRAM_ADMIN_IDS.clear()
        user = access.ensure(777, "petrov", "Пётр Петров")
        self.assertFalse(access.is_allowed(user))

        result = access.request_access(777, "нужен доступ к паспортам")
        self.assertEqual(result["status"], "pending")
        self.assertEqual(len(access.listing("pending")), 1)

        access.decide(777, True, role="sales", by="тест")
        self.assertTrue(access.is_allowed(access.get(777)))

        access.block(777, by="тест")
        self.assertFalse(access.is_allowed(access.get(777)))

    def test_repeat_request_is_idempotent(self):
        import access
        access.request_access(778)
        access.request_access(778)
        self.assertEqual(len(access.listing("pending")), 1)


class TestSettingsValidation(unittest.TestCase):
    def setUp(self):
        import settings_schema
        self.schema = settings_schema

    def _levels(self, values, full=None):
        return {(i["key"], i["level"]) for i in self.schema.validate(values, full or {})}

    def test_type_errors(self):
        self.assertIn(("SEARCH_TOP_K", "error"), self._levels({"SEARCH_TOP_K": "abc"}))
        self.assertIn(("LSA_DIM", "error"), self._levels({"LSA_DIM": "12.5"}))

    def test_range_errors(self):
        self.assertIn(("MIN_CONFIDENCE", "error"), self._levels({"MIN_CONFIDENCE": "5"}))
        self.assertIn(("RERANKER_WEIGHT", "error"), self._levels({"RERANKER_WEIGHT": "1.5"}))

    def test_dependency_errors(self):
        issues = self._levels({"EMBEDDINGS_PROVIDER": "onnx", "ONNX_MODEL_PATH": ""})
        self.assertIn(("ONNX_MODEL_PATH", "error"), issues)

    def test_cross_field_errors(self):
        issues = self._levels({"CHUNK_OVERLAP_CHARS": "2000", "CHUNK_TARGET_CHARS": "1400"})
        self.assertIn(("CHUNK_OVERLAP_CHARS", "error"), issues)
        issues = self._levels({"SEARCH_TOP_K": "50", "SEARCH_CANDIDATES": "40"})
        self.assertIn(("SEARCH_TOP_K", "error"), issues)

    def test_open_admin_without_token(self):
        issues = self._levels({"ADMIN_HOST": "0.0.0.0", "ADMIN_TOKEN": ""})
        self.assertIn(("ADMIN_TOKEN", "error"), issues)

    def test_good_values_pass(self):
        issues = self.schema.validate({"SEARCH_TOP_K": "6", "RERANKER_WEIGHT": "0.8"},
                                      full={"SEARCH_CANDIDATES": "40"})
        self.assertEqual([i for i in issues if i["level"] == "error"], [])


class TestAnalytics(Isolated):
    def setUp(self):
        super().setUp()
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        rows = [
            ("как подобрать частотный преобразователь", 0, "nothing_found", 0.0),
            ("нужен ли частотный преобразователь насосу", 0, "nothing_found", 0.0),
            ("подбор частотного преобразователя по мощности", 0, "nothing_found", 0.0),
            ("какая гарантия на насос", 1, "answered", 0.05),
            ("степень защиты насоса", 1, "answered", 0.04),
        ]
        for q, answered, stage, score in rows:
            self.db.run("""INSERT INTO queries(question, answered, stage, route,
                           channels, top_score, created_at, latency_ms)
                           VALUES (?,?,?,?,?,?,?,10)""",
                        (q, answered, stage,
                         "documents" if answered else "none",
                         "bm25+dense" if answered else "", score, now))

    def test_funnel_counts(self):
        import analytics
        f = analytics.funnel(24)
        self.assertEqual(f["total"], 5)
        self.assertEqual(f["steps"][1]["n"], 2)      # что-то нашлось
        self.assertTrue(any("поиск ничего не нашёл" in loss["where"]
                            for loss in f["losses"]))

    def test_gaps_group_similar_questions(self):
        import analytics
        g = analytics.gaps(24)
        self.assertTrue(g["groups"])
        biggest = g["groups"][0]
        self.assertEqual(biggest["size"], 3)
        self.assertIn("преобразовател", biggest["terms"])

    def test_channels(self):
        import analytics
        c = analytics.channel_report(24)
        self.assertEqual(c["both"], 2)

    def test_confidence_histogram(self):
        import analytics
        h = analytics.confidence_histogram(24)
        self.assertEqual(h["total"], 2)
        self.assertTrue(h["bins"])


if __name__ == "__main__":
    unittest.main()


class TestPipelineFingerprint(Isolated):
    """Индекс помнит, каким конвейером собран, и замечает расхождение."""

    def _seed(self):
        self.db.run("INSERT INTO documents (rel_path, abs_path, file_name, ext, "
                    "content_hash, status) VALUES ('a','/a','a','.txt','h','ok')")
        self.db.run("INSERT INTO chunks (doc_id, ord, text, n_chars) "
                    "VALUES (1,0,'текст',5)")

    def test_fresh_index_not_stale(self):
        import index
        self._seed()
        index.record_pipeline()
        self.assertFalse(index.pipeline_state()["stale"])

    def test_changed_pipeline_detected_and_alerted(self):
        import alerts
        import index
        self._seed()
        self.db.run("INSERT OR REPLACE INTO schema_meta(key, value) "
                    "VALUES ('pipeline', 'старый')")
        self.assertTrue(index.pipeline_state()["stale"])
        keys = [i["key"] for i in alerts.collect()]
        self.assertIn("pipeline_stale", keys)

    def test_old_install_without_fingerprint(self):
        import index
        self._seed()
        st = index.pipeline_state()
        self.assertTrue(st["never_recorded"])


class TestVectorsMmap(Isolated):
    """Большая матрица читается через mmap и переживает дозапись."""

    def test_mmap_roundtrip(self):
        import importlib
        import os

        import numpy as np
        os.environ["VECTORS_MMAP_MB"] = "1"
        importlib.reload(self.config)
        try:
            store = self.db.VectorStore()
            vecs = np.random.rand(1500, 256).astype(np.float32)
            store.add(range(1, 1501), vecs)
            store.save()
            fresh = self.db.VectorStore()
            self.assertIsInstance(fresh.matrix, np.memmap)
            self.assertEqual(fresh.search(vecs[0], 1)[0][0], 1)
            fresh.add([2000], np.random.rand(1, 256).astype(np.float32))
            fresh.save()
            again = self.db.VectorStore()
            self.assertEqual(len(again), 1501)
        finally:
            os.environ.pop("VECTORS_MMAP_MB", None)
            importlib.reload(self.config)
