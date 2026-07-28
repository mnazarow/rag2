"""
Запуск и остановка: то, что происходит с системой не во время работы.

Эти проверки закрывают целый класс поломок, у которых нет ни одного
сообщения об ошибке. Остановка посреди записи оставляет обрезанный
файл, система при следующем старте читает его молча и продолжает
работать — просто хуже. Узнают об этом по жалобам через недели, и
связать жалобу с давним перезапуском уже невозможно.
"""
from __future__ import annotations

import json
import signal
import subprocess
import sys
import threading
import time
import unittest
import urllib.request
from pathlib import Path

import numpy as np

from tests.base import ROOT, Isolated


class TestAtomicWrites(Isolated):
    """
    Файлы, которые нельзя увидеть наполовину записанными: индекс, модель
    поиска, настройки, учётные записи.
    """

    def test_vectors_survive_interrupted_write(self):
        """
        Пишем поверх существующего индекса и «падаем» посреди записи.
        На диске должен остаться прежний индекс целиком, а не обрубок.
        """
        store = self.db.VectorStore()
        store.add([1, 2, 3], np.random.rand(3, store.dim).astype("float32"))
        store.save()

        def boom(_fh):
            raise OSError("диск кончился")

        with self.assertRaises(OSError):
            self.db.atomic_write(Path(self.config.VECTORS_PATH), boom)

        again = self.db.VectorStore()
        self.assertEqual(len(again), 3, "прежний индекс должен уцелеть")
        self.assertEqual(again.broken, "")

    def test_no_temporary_files_left_behind(self):
        store = self.db.VectorStore()
        store.add([1], np.random.rand(1, store.dim).astype("float32"))
        store.save()
        leftovers = list(Path(self.config.DATA_DIR).glob("*.tmp*"))
        self.assertFalse(leftovers, f"остались временные файлы: {leftovers}")

    def test_mismatched_files_are_reported_not_hidden(self):
        """
        Рассогласованная пара файлов — обычный след остановки посреди
        записи. Раньше она молча превращалась в пустой индекс.
        """
        store = self.db.VectorStore()
        store.add([1, 2, 3], np.random.rand(3, store.dim).astype("float32"))
        store.save()
        Path(self.config.VECTOR_IDS_PATH).write_text("[1]")
        again = self.db.VectorStore()
        self.assertTrue(again.broken, "повреждение должно быть замечено")
        self.assertEqual(len(again), 0)

    def test_settings_file_written_atomically(self):
        import importlib

        import webui
        importlib.reload(webui)
        env = self.config.BASE_DIR / ".env"
        backup = env.read_text(encoding="utf-8") if env.exists() else None
        try:
            webui.write_env({"SEARCH_TOP_K": "7"})
            self.assertIn("SEARCH_TOP_K=7", env.read_text(encoding="utf-8"))
            self.assertFalse(list(self.config.BASE_DIR.glob(".env.tmp*")))
        finally:
            if backup is None:
                env.unlink(missing_ok=True)
            else:
                env.write_text(backup, encoding="utf-8")


class TestShutdownModule(unittest.TestCase):
    def setUp(self):
        import importlib

        import shutdown
        importlib.reload(shutdown)
        self.shutdown = shutdown

    def test_actions_run_in_reverse_order(self):
        order = []
        self.shutdown.on_stop("первое", lambda: order.append(1))
        self.shutdown.on_stop("второе", lambda: order.append(2))
        self.shutdown.run_actions()
        self.assertEqual(order, [2, 1], "закрывать нужно в обратном порядке")

    def test_hanging_action_does_not_block_forever(self):
        """
        Одно подвисшее действие не должно превращать корректную
        остановку в убийство процесса.
        """
        self.shutdown.on_stop("зависшее", lambda: time.sleep(30))
        started = time.time()
        self.shutdown.run_actions(timeout=1.0)
        self.assertLess(time.time() - started, 5.0)

    def test_failing_action_does_not_stop_the_rest(self):
        done = []

        def broken():
            raise RuntimeError("не вышло")

        self.shutdown.on_stop("важное", lambda: done.append("сделано"))
        self.shutdown.on_stop("сломанное", broken)
        self.shutdown.run_actions()
        self.assertEqual(done, ["сделано"])

    def test_wait_is_interrupted_by_the_signal(self):
        """
        Пауза в фоновом цикле должна прерываться, иначе цикл с паузой в
        минуту задерживает выключение на минуту — и docker успевает
        перейти к принудительному завершению.
        """
        threading.Timer(0.2, self.shutdown.request_stop).start()
        started = time.time()
        self.shutdown.wait(10.0)
        self.assertLess(time.time() - started, 3.0)
        self.assertTrue(self.shutdown.stopping())


class TestPreflight(Isolated):
    def setUp(self):
        super().setUp()
        import importlib

        import preflight
        importlib.reload(preflight)
        self.preflight = preflight

    def test_open_admin_without_protection_is_fatal(self):
        self.config.ADMIN_HOST = "0.0.0.0"
        self.config.ADMIN_TOKEN = ""
        self.config.ADMIN_USERS_FILE = self.tmp / "нет.json"
        report = self.preflight.check("админка")
        self.assertTrue(any("защиты нет" in x for x in report["fatal"]), report)

    def test_proxy_without_accounts_is_fatal(self):
        self.config.ADMIN_TRUST_PROXY = True
        self.config.ADMIN_USERS_FILE = self.tmp / "нет.json"
        report = self.preflight.check("админка")
        self.assertTrue(any("ADMIN_TRUST_PROXY" in x for x in report["fatal"]), report)

    def test_default_role_outside_the_list_is_fatal(self):
        self.config.ROLE_SECTIONS = {"sales": ["РОЗНИЦА"]}
        self.config.DEFAULT_ROLE = "Sales"
        report = self.preflight.check("админка")
        self.assertTrue(any("роль по умолчанию" in x for x in report["fatal"]), report)

    def test_empty_index_is_only_a_warning(self):
        """
        Пустой индекс — повод сообщить, а не отказаться запускаться:
        именно так выглядит первая установка.
        """
        report = self.preflight.check("админка")
        self.assertFalse(report["fatal"], report["fatal"])
        self.assertTrue(any("индекс пуст" in x for x in report["warn"]), report)

    def test_bot_without_token_is_fatal(self):
        self.config.TELEGRAM_BOT_TOKEN = ""
        report = self.preflight.check("бот")
        self.assertTrue(any("TELEGRAM_BOT_TOKEN" in x for x in report["fatal"]), report)


class TestScheduleUsesTheQueue(unittest.TestCase):
    """
    Регулярные задания должны идти через очередь. Прямой вызов из cron —
    отдельный процесс, который не знает о блокировках: ночной проход и
    нажатая в админке переиндексация пишут в один файл векторов, и часть
    работы пропадает без единой ошибки.
    """

    def test_index_and_backup_go_through_runjob(self):
        import schedule
        for name in ("update", "backup"):
            command = schedule.TASKS[name]["command"]
            self.assertTrue(command.startswith("runjob.py"),
                            f"задание «{name}» идёт мимо очереди: {command}")


class TestAdminProcessStopsCleanly(unittest.TestCase):
    """
    Сквозная проверка: поднимаем настоящий процесс админки, посылаем
    сигнал остановки и смотрим, что он завершился сам и быстро.

    Проверка именно сквозная, потому что ломалось здесь на стыке:
    обработчик сигнала выполняется в главном потоке, а закрытие сервера
    ждёт завершения цикла, который крутится в том же главном потоке.
    Каждая часть по отдельности работала, вместе — процесс висел до
    принудительного завершения.
    """

    def test_sigterm_stops_the_process(self):
        import tempfile
        data = tempfile.mkdtemp(prefix="kbstop_")
        logs = tempfile.mkdtemp(prefix="kbstoplog_")
        import os
        env = dict(os.environ, DATA_DIR=data, LOG_DIR=logs,
                   KB_ROOT=data, ADMIN_PORT="8893", LLM_PROVIDER="echo",
                   ADMIN_HOST="127.0.0.1")
        proc = subprocess.Popen([sys.executable, "webui.py"], cwd=str(ROOT), env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            for _ in range(80):
                try:
                    urllib.request.urlopen("http://127.0.0.1:8893/healthz",
                                           timeout=1).read()
                    break
                except Exception:  # noqa: BLE001
                    time.sleep(0.25)
            else:
                self.skipTest("админка не поднялась в этой среде")

            started = time.time()
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.fail("процесс не остановился по сигналу — пришлось бы убивать")
            self.assertLess(time.time() - started, 30)
        finally:
            if proc.poll() is None:
                proc.kill()

    def test_healthz_answers_without_authorisation(self):
        """
        Проверка живости не должна требовать токена: иначе, закрыв
        админку, мы получаем вечно «нездоровый» контейнер и бесконечный
        цикл перезапусков.
        """
        import os
        import tempfile
        data = tempfile.mkdtemp(prefix="kbhealth_")
        env = dict(os.environ, DATA_DIR=data, LOG_DIR=data, KB_ROOT=data,
                   ADMIN_PORT="8892", LLM_PROVIDER="echo",
                   ADMIN_HOST="127.0.0.1", ADMIN_TOKEN="ochen-sekretnyj-token")
        proc = subprocess.Popen([sys.executable, "webui.py"], cwd=str(ROOT), env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            body = None
            for _ in range(80):
                try:
                    body = urllib.request.urlopen(
                        "http://127.0.0.1:8892/healthz", timeout=1).read()
                    break
                except Exception:  # noqa: BLE001
                    time.sleep(0.25)
            if body is None:
                self.skipTest("админка не поднялась в этой среде")
            self.assertTrue(json.loads(body)["ok"])
            # А вот всё остальное без токена по-прежнему закрыто.
            with self.assertRaises(Exception):
                urllib.request.urlopen("http://127.0.0.1:8892/api/state", timeout=2)
        finally:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    unittest.main()
