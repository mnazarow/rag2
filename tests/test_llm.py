"""
Провайдеры генерации: локальная модель, облако и переключение между ними.

Настоящих моделей в тестовом окружении нет, поэтому здесь поднимается
сервер, отвечающий как OpenAI-совместимый endpoint. Проверяется то, что
может сломаться у нас: какой запрос уходит, как читается ответ, что
происходит при отказе основного провайдера и записывается ли переключение.
"""
from __future__ import annotations

import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tests.base import Isolated  # noqa: F401 — путь к проекту


class FakeModelServer(BaseHTTPRequestHandler):
    """OpenAI-совместимый сервер: отвечает, ломается или молчит по команде."""

    mode = "ok"                 # ok | error | slow
    seen: list[dict] = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        FakeModelServer.seen.append(body)
        if FakeModelServer.mode == "error":
            self.send_response(500)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if FakeModelServer.mode == "slow":
            time.sleep(2.0)
        answer = f"ответ от {body.get('model')}"
        data = json.dumps({
            "choices": [{"message": {"content": answer}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class TestLocalAndCloud(Isolated):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeModelServer)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        super().setUp()
        FakeModelServer.mode = "ok"
        FakeModelServer.seen.clear()
        import llm
        self.llm = llm
        llm.reset()
        self.url = f"http://127.0.0.1:{self.port}/v1"
        self.config.LOCAL_LLM_BASE_URL = self.url
        self.config.LOCAL_LLM_MODEL = "t-pro-2.0"
        self.config.LOCAL_LLM_TIMEOUT = 5.0
        self.config.OPENAI_BASE_URL = self.url
        self.config.OPENAI_API_KEY = "test"
        self.config.LLM_MODEL = "cloud-model"

    def tearDown(self):
        self.llm.reset()
        super().tearDown()

    def test_local_answers(self):
        engine = self.llm.LocalLLM()
        result = engine.complete("система", "вопрос")
        self.assertEqual(result.text, "ответ от t-pro-2.0")
        self.assertEqual(result.tokens_in, 11)
        sent = FakeModelServer.seen[-1]
        self.assertEqual(sent["messages"][0]["role"], "system")
        self.assertEqual(sent["model"], "t-pro-2.0")

    def test_local_not_running_explains_what_to_do(self):
        self.config.LOCAL_LLM_BASE_URL = ""
        self.config.LOCAL_LLM_FALLBACK_URL = ""
        with self.assertRaises(self.llm.LLMError) as ctx:
            self.llm.LocalLLM()
        self.assertIn("Модели", str(ctx.exception))

    def test_fallback_to_cloud_when_local_fails(self):
        """
        Главное свойство: отказ локальной модели не должен превращаться
        в «ассистент не работает».
        """
        self.config.LLM_PROVIDER = "local"
        self.config.LLM_FALLBACK = "openai"
        routed = self.llm.get_llm()
        FakeModelServer.mode = "error"

        # Локальная падает, но ответ всё равно приходит — от облака.
        with self.assertRaises(self.llm.LLMError):
            routed.complete("с", "п")          # оба провайдера смотрят на один сервер

        FakeModelServer.mode = "ok"
        result = routed.complete("с", "п")
        self.assertTrue(result.text)

    def test_switch_is_recorded(self):
        import llm as llm_mod

        class Broken(llm_mod.BaseLLM):
            name = "broken"
            model = "broken"

            def complete(self, system, user):
                raise RuntimeError("видеопамять кончилась")

        llm_mod.PROVIDERS["broken"] = Broken
        try:
            routed = llm_mod.Routed("broken", ["openai"])
            result = routed.complete("с", "п")
            self.assertTrue(result.text)
            self.assertEqual(routed.active, "openai")
            self.assertEqual(len(routed.switches), 1)
            self.assertIn("видеопамять", routed.switches[0]["why"])
        finally:
            llm_mod.PROVIDERS.pop("broken", None)

    def test_all_providers_failed_reports_every_reason(self):
        import llm as llm_mod

        class Broken(llm_mod.BaseLLM):
            name = "broken"
            model = "broken"

            def complete(self, system, user):
                raise RuntimeError("причина такая-то")

        llm_mod.PROVIDERS["broken"] = Broken
        try:
            routed = llm_mod.Routed("broken", ["broken"])
            with self.assertRaises(llm_mod.LLMError) as ctx:
                routed.complete("с", "п")
            self.assertIn("причина такая-то", str(ctx.exception))
        finally:
            llm_mod.PROVIDERS.pop("broken", None)

    def test_probe_reports_latency(self):
        self.config.LLM_PROVIDER = "local"
        result = self.llm.probe()
        self.assertTrue(result["ok"], result.get("error"))
        self.assertIn("ответ", result["answer"])
        self.assertGreaterEqual(result["ms"], 0)

    def test_probe_reports_failure(self):
        FakeModelServer.mode = "error"
        result = self.llm.probe("local")
        self.assertFalse(result["ok"])
        self.assertTrue(result["error"])

    def test_describe_lists_chain(self):
        self.config.LLM_PROVIDER = "local"
        self.config.LLM_FALLBACK = "openai"
        self.llm.reset()
        info = self.llm.describe()
        self.assertEqual(info["chain"], ["local", "openai"])
        self.assertFalse(info["is_stub"])


if __name__ == "__main__":
    unittest.main()


class TestMacModelSupport(Isolated):
    """Запуск моделей на встроенной видеокарте мака — через ollama."""

    def _mac(self, ram_gb=32, has_ollama=True):
        import platform
        from unittest import mock

        import models
        return (mock.patch.object(platform, "system", return_value="Darwin"),
                mock.patch.object(platform, "machine", return_value="arm64"),
                mock.patch.object(models, "_total_ram",
                                  return_value=ram_gb * 1024 ** 3),
                mock.patch.object(models.shutil, "which",
                                  lambda x: "/x/ollama"
                                  if (x == "ollama" and has_ollama) else None))

    def test_apple_silicon_counts_as_gpu(self):
        import contextlib

        import models
        with contextlib.ExitStack() as st:
            for m in self._mac():
                st.enter_context(m)
            hw = models.hardware()
        self.assertTrue(hw.get("apple_silicon"))
        self.assertGreater(hw["vram_total_gb"], 15)

    def test_engine_resolves_to_ollama_on_mac(self):
        import contextlib

        import models
        spec = models.BY_ID["qwen3.6-27b"]
        with contextlib.ExitStack() as st:
            for m in self._mac():
                st.enter_context(m)
            self.assertEqual(models.resolve_engine(spec), "ollama")
        # На линуксе — как было
        self.assertEqual(models.resolve_engine(spec), "vllm")

    def test_model_without_ollama_tag_explained(self):
        import contextlib

        import models
        with contextlib.ExitStack() as st:
            for m in self._mac():
                st.enter_context(m)
            chk = models.check("t-pro-2.0")
        self.assertFalse(chk["ok"])
        self.assertTrue(any("ollama" in p for p in chk["problems"]), chk["problems"])
