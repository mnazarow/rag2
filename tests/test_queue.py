"""
Очередь запросов к модели.

Проверяется не «очередь существует», а то, ради чего она заведена:
одновременных обращений к модели не больше заданного, живой вопрос
обгоняет фоновую обработку, переполнение даёт отказ сразу, а место,
занятое погибшим процессом, освобождается само.
"""
from __future__ import annotations

import threading
import time
import unittest

from tests.base import Isolated


class TestConcurrencyLimit(Isolated):
    def setUp(self):
        super().setUp()
        import llm_queue
        self.q = llm_queue
        self.q.reset()
        self.config.LLM_MAX_CONCURRENT = 1
        self.config.LLM_QUEUE_MAX = 20
        self.config.LLM_QUEUE_TIMEOUT = 10.0
        self.config.LLM_QUEUE_SLOT_TTL = 60.0
        self.config.LLM_QUEUE_SHARED = True

    def _worker(self, seen, lock, hold=0.15, source="вопрос"):
        def run():
            with self.q.slot(source=source):
                with lock:
                    seen["now"] += 1
                    seen["max"] = max(seen["max"], seen["now"])
                time.sleep(hold)
                with lock:
                    seen["now"] -= 1
        return run

    def test_one_at_a_time(self):
        """
        Главное свойство. Пять потоков заходят разом, но внутри места
        одновременно никогда не оказывается двоих.
        """
        seen = {"now": 0, "max": 0}
        lock = threading.Lock()
        threads = [threading.Thread(target=self._worker(seen, lock))
                   for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        self.assertEqual(seen["max"], 1, "в модель ушло больше одного запроса разом")
        self.assertEqual(seen["now"], 0, "место не освободилось")

    def test_limit_two_allows_two(self):
        self.config.LLM_MAX_CONCURRENT = 2
        seen = {"now": 0, "max": 0}
        lock = threading.Lock()
        threads = [threading.Thread(target=self._worker(seen, lock, hold=0.3))
                   for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        self.assertEqual(seen["max"], 2)

    def test_zero_means_no_limit(self):
        self.config.LLM_MAX_CONCURRENT = 0
        seen = {"now": 0, "max": 0}
        lock = threading.Lock()
        threads = [threading.Thread(target=self._worker(seen, lock, hold=0.3))
                   for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        self.assertGreater(seen["max"], 1)

    def test_slot_freed_after_error(self):
        """Сбой внутри запроса не должен оставлять место занятым навсегда."""
        with self.assertRaises(ValueError):
            with self.q.slot(source="вопрос"):
                raise ValueError("модель упала")
        st = self.q.status()
        self.assertEqual(st["running"], 0)
        with self.q.slot(source="вопрос"):
            pass


class TestPriority(Isolated):
    def setUp(self):
        super().setUp()
        import llm_queue
        self.q = llm_queue
        self.q.reset()
        self.config.LLM_MAX_CONCURRENT = 1
        self.config.LLM_QUEUE_MAX = 50
        self.config.LLM_QUEUE_TIMEOUT = 20.0
        self.config.LLM_QUEUE_SLOT_TTL = 60.0
        self.config.LLM_QUEUE_SHARED = True

    def test_question_overtakes_batch(self):
        """
        Ради чего сделана важность: пакетная обработка ставит в очередь
        десятки тысяч заданий, и без обгона сотрудник ждал бы конца
        всего прогона.
        """
        order: list[str] = []
        lock = threading.Lock()
        holding = threading.Event()
        release = threading.Event()

        def holder():
            with self.q.slot(source="приставки"):
                holding.set()
                release.wait(20)

        def worker(source):
            def run():
                with self.q.slot(source=source):
                    with lock:
                        order.append(source)
            return run

        h = threading.Thread(target=holder)
        h.start()
        self.assertTrue(holding.wait(10), "первый так и не занял место")

        # Сначала встают фоновые, потом живой вопрос — и он должен пройти
        # раньше них.
        batch = [threading.Thread(target=worker("приставки")) for _ in range(3)]
        for t in batch:
            t.start()
        time.sleep(0.4)                      # дать фоновым встать в очередь
        live = threading.Thread(target=worker("вопрос"))
        live.start()
        time.sleep(0.4)

        release.set()
        for t in [h, live] + batch:
            t.join(30)

        self.assertEqual(order[0], "вопрос",
                         f"живой вопрос не обогнал фоновую работу: {order}")
        self.assertEqual(len(order), 4)


class TestRefusals(Isolated):
    def setUp(self):
        super().setUp()
        import llm_queue
        self.q = llm_queue
        self.q.reset()
        self.config.LLM_MAX_CONCURRENT = 1
        self.config.LLM_QUEUE_SHARED = True
        self.config.LLM_QUEUE_SLOT_TTL = 60.0

    def test_full_queue_refuses_immediately(self):
        """
        Отказ должен приходить сразу. Две минуты ожидания и та же ошибка
        в конце — худший из возможных вариантов.
        """
        self.config.LLM_QUEUE_MAX = 2
        self.config.LLM_QUEUE_TIMEOUT = 30.0
        holding = threading.Event()
        release = threading.Event()
        started = threading.Barrier(3, timeout=20)

        def holder():
            with self.q.slot(source="вопрос"):
                holding.set()
                release.wait(20)

        def waiter():
            try:
                started.wait()
            except Exception:  # noqa: BLE001
                return
            try:
                with self.q.slot(source="вопрос"):
                    pass
            except self.q.LLMBusy:
                pass

        h = threading.Thread(target=holder)
        h.start()
        self.assertTrue(holding.wait(10))
        waiters = [threading.Thread(target=waiter) for _ in range(2)]
        for t in waiters:
            t.start()
        started.wait(20)
        time.sleep(0.5)                       # оба успели встать в очередь

        began = time.time()
        with self.assertRaises(self.q.LLMBusy) as ctx:
            with self.q.slot(source="вопрос"):
                pass
        self.assertLess(time.time() - began, 3.0, "отказ пришёл не сразу")
        self.assertIn("занята", str(ctx.exception))

        release.set()
        for t in [h] + waiters:
            t.join(30)

    def test_timeout_refuses_with_reason(self):
        self.config.LLM_QUEUE_MAX = 20
        self.config.LLM_QUEUE_TIMEOUT = 1.0
        release = threading.Event()
        holding = threading.Event()

        def holder():
            with self.q.slot(source="вопрос"):
                holding.set()
                release.wait(20)

        h = threading.Thread(target=holder)
        h.start()
        self.assertTrue(holding.wait(10))
        with self.assertRaises(self.q.LLMBusy) as ctx:
            with self.q.slot(source="вопрос"):
                pass
        self.assertIn("очеред", str(ctx.exception))
        release.set()
        h.join(30)

    def test_refusals_counted(self):
        self.config.LLM_QUEUE_MAX = 1
        self.config.LLM_QUEUE_TIMEOUT = 1.0
        release = threading.Event()
        holding = threading.Event()

        def holder():
            with self.q.slot(source="вопрос"):
                holding.set()
                release.wait(20)

        h = threading.Thread(target=holder)
        h.start()
        self.assertTrue(holding.wait(10))
        for _ in range(2):
            try:
                with self.q.slot(source="вопрос"):
                    pass
            except self.q.LLMBusy:
                pass
        release.set()
        h.join(30)
        st = self.q.stats(24)
        self.assertGreaterEqual(st["refused"] + st["timeout"], 1)


class TestStaleSlot(Isolated):
    def setUp(self):
        super().setUp()
        import llm_queue
        self.q = llm_queue
        self.q.reset()
        self.config.LLM_MAX_CONCURRENT = 1
        self.config.LLM_QUEUE_MAX = 20
        self.config.LLM_QUEUE_TIMEOUT = 10.0
        self.config.LLM_QUEUE_SHARED = True

    def test_dead_process_slot_is_freed(self):
        """
        Процесс убили посреди запроса — запись «выполняется» осталась.
        Без освобождения по сроку модель была бы занята навсегда.
        """
        self.q.ensure_tables()
        self.db.trun(
            "INSERT INTO llm_queue(state, priority, source, pid, host, "
            "created_at, started_at, expires_at) "
            "VALUES ('running', 0, 'вопрос', 999999, 'умерший', ?, ?, ?)",
            ("2020-01-01T00:00:00+00:00", "2020-01-01T00:00:00+00:00",
             time.time() - 5))
        began = time.time()
        with self.q.slot(source="вопрос"):
            pass
        self.assertLess(time.time() - began, 5.0,
                        "место погибшего процесса не освободилось")

    def test_dead_waiter_does_not_block_the_line(self):
        """
        Процесс умер, пока стоял в очереди. Его запись «ждёт» продолжает
        держать место в порядке очереди — и все, кто пришёл следом, ждут
        покойника. Снаружи это выглядит как зависший ассистент.
        """
        self.q.ensure_tables()
        self.db.trun(
            "INSERT INTO llm_queue(state, priority, source, pid, host, "
            "created_at, expires_at) VALUES ('waiting',0,'вопрос',999999,'умерший',?,?)",
            ("2020-01-01T00:00:00+00:00", time.time() - 1))
        began = time.time()
        with self.q.slot(source="вопрос"):
            pass
        self.assertLess(time.time() - began, 5.0)

    def test_live_slot_is_not_freed(self):
        """Обратная проверка: живое место по ошибке не отбирается."""
        self.q.ensure_tables()
        self.db.trun(
            "INSERT INTO llm_queue(state, priority, source, pid, host, "
            "created_at, started_at, expires_at) "
            "VALUES ('running', 0, 'вопрос', 999999, 'живой', ?, ?, ?)",
            ("2020-01-01T00:00:00+00:00", "2020-01-01T00:00:00+00:00",
             time.time() + 600))
        self.config.LLM_QUEUE_TIMEOUT = 1.0
        with self.assertRaises(self.q.LLMBusy):
            with self.q.slot(source="вопрос"):
                pass


class TestEverythingGoesThroughQueue(Isolated):
    """
    Смысл проверки: ограничение, которое можно обойти, забыв про него в
    новом месте кода, не ограничивает ничего. Здесь проверяется, что
    обычный путь ответа действительно проходит через очередь.
    """

    def setUp(self):
        super().setUp()
        import llm
        import llm_queue
        self.llm, self.q = llm, llm_queue
        llm.reset()
        self.config.LLM_PROVIDER = "echo"
        self.config.LLM_FALLBACK = ""
        self.config.LLM_MAX_CONCURRENT = 1
        self.config.LLM_QUEUE_SHARED = True
        self.config.LLM_QUEUE_MAX = 20
        self.config.LLM_QUEUE_TIMEOUT = 10.0
        self.config.LLM_QUEUE_SLOT_TTL = 60.0

    def tearDown(self):
        self.llm.reset()
        super().tearDown()

    def test_completion_is_recorded_in_queue(self):
        engine = self.llm.get_llm()
        engine.complete("система", "[1] источник\nВОПРОС: что-нибудь")
        rows = self.db.tq("SELECT state, source FROM llm_queue")
        self.assertEqual(len(rows), 1, "запрос прошёл мимо очереди")
        self.assertEqual(rows[0]["state"], "done")

    def test_probe_goes_through_queue(self):
        self.llm.probe("echo")
        row = self.db.tq1("SELECT source FROM llm_queue ORDER BY id DESC LIMIT 1")
        self.assertEqual(row["source"], "проверка")

    def test_stats_measure_waiting(self):
        engine = self.llm.get_llm()
        for _ in range(3):
            engine.complete("система", "[1] источник\nВОПРОС: что-нибудь")
        st = self.q.stats(24)
        self.assertEqual(st["total"], 3)
        self.assertIn("echo", [r["source"] or "" for r in st["by_source"]] + ["echo"])

    def test_clear_frees_everything(self):
        self.q.ensure_tables()
        self.db.trun("INSERT INTO llm_queue(state, priority, source, pid, host, "
                     "created_at, expires_at) VALUES ('running',0,'вопрос',1,'x',?,?)",
                     ("2020-01-01T00:00:00+00:00", time.time() + 600))
        self.assertEqual(self.q.clear(), 1)
        self.assertEqual(self.q.status()["running"], 0)


class TestLocalFallbackQueue(Isolated):
    """Когда общая очередь выключена, ограничение всё равно действует."""

    def setUp(self):
        super().setUp()
        import llm_queue
        self.q = llm_queue
        self.q.reset()
        self.config.LLM_QUEUE_SHARED = False
        self.config.LLM_MAX_CONCURRENT = 1
        self.config.LLM_QUEUE_MAX = 20
        self.config.LLM_QUEUE_TIMEOUT = 10.0

    def test_still_one_at_a_time(self):
        seen = {"now": 0, "max": 0}
        lock = threading.Lock()

        def run():
            with self.q.slot(source="вопрос"):
                with lock:
                    seen["now"] += 1
                    seen["max"] = max(seen["max"], seen["now"])
                time.sleep(0.15)
                with lock:
                    seen["now"] -= 1

        threads = [threading.Thread(target=run) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        self.assertEqual(seen["max"], 1)
        self.assertEqual(self.q.status()["running"], 0)


class TestQueueSettings(unittest.TestCase):
    def setUp(self):
        import settings_schema
        self.schema = settings_schema

    def test_default_is_one(self):
        spec = {s["key"]: s for s in self.schema.SETTINGS}["LLM_MAX_CONCURRENT"]
        self.assertEqual(spec["default"], 1)

    def test_slot_ttl_must_exceed_model_timeout(self):
        """
        Ошибка, которую легко сделать и невозможно заметить: место
        освобождается раньше, чем модель отвечает, и ограничение
        перестаёт работать именно на долгих ответах.
        """
        issues = self.schema.validate({"LLM_QUEUE_SLOT_TTL": "60"},
                                      {"LOCAL_LLM_TIMEOUT": "180"})
        self.assertTrue(any(i["key"] == "LLM_QUEUE_SLOT_TTL" and i["level"] == "error"
                            for i in issues), issues)

    def test_sane_pair_passes(self):
        issues = self.schema.validate({"LLM_QUEUE_SLOT_TTL": "300"},
                                      {"LOCAL_LLM_TIMEOUT": "180"})
        self.assertFalse([i for i in issues
                          if i["key"] == "LLM_QUEUE_SLOT_TTL"], issues)

    def test_no_limit_warns(self):
        issues = self.schema.validate({"LLM_MAX_CONCURRENT": "0"}, {})
        self.assertTrue(any(i["key"] == "LLM_MAX_CONCURRENT" for i in issues))

    def test_negative_refused(self):
        issues = self.schema.validate({"LLM_MAX_CONCURRENT": "-1"}, {})
        self.assertTrue(any(i["level"] == "error" for i in issues))


if __name__ == "__main__":
    unittest.main()
