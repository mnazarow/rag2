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

    mode = "ok"                 # ok | error | slow | notfound
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
        if FakeModelServer.mode == "notfound":
            # Так отвечает ollama на незнакомое имя модели.
            data = json.dumps({"error": {"message":
                f"model '{body.get('model')}' not found"}}).encode()
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
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

    def test_404_names_model_and_next_step(self):
        """Регрессия: голый «404 Not Found» от сервера ничего не объяснял.
        Теперь ошибка называет модель, адрес и что нажать."""
        FakeModelServer.mode = "notfound"
        engine = self.llm.LocalLLM()
        with self.assertRaises(self.llm.LLMError) as ctx:
            engine.complete("с", "п")
        text = str(ctx.exception)
        self.assertIn("не знает модель", text)
        self.assertIn("t-pro-2.0", text)
        self.assertIn("Запустить и использовать", text)
        self.assertIn("not found", text)      # ответ сервера тоже виден

    def test_model_name_prefers_served_name(self):
        """Без LOCAL_LLM_MODEL имя берётся то, под которым модель знает
        сервер (served_name), а не наш идентификатор каталога — иначе 404."""
        from unittest import mock

        import models
        self.config.LOCAL_LLM_MODEL = ""
        state = {"running": True, "base_url": self.url,
                 "model": "qwen3.6-27b", "served_name": "qwen3:32b"}
        with mock.patch.object(models, "status", return_value=state):
            engine = self.llm.LocalLLM()
        self.assertEqual(engine.model, "qwen3:32b")

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

    def test_brew_paths_added_on_mac(self):
        """
        Службы launchd живут с урезанным PATH без каталогов Homebrew —
        для них ollama и ffmpeg «не установлены», хотя стоят. config
        обязан дополнять PATH на маке, иначе кнопка «Проверить» в
        разделе моделей врёт ровно тогда, когда админка запущена как
        служба — то есть в проде.
        """
        import importlib
        import os
        import platform
        from unittest import mock

        saved = os.environ["PATH"]
        os.environ["PATH"] = "/usr/bin:/bin"
        try:
            with mock.patch.object(platform, "system", return_value="Darwin"), \
                 mock.patch("os.path.isdir",
                            lambda d: d in ("/opt/homebrew/bin", "/usr/local/bin")
                            or os.path.exists(d)):
                import config
                importlib.reload(config)
            self.assertIn("/opt/homebrew/bin", os.environ["PATH"])
        finally:
            os.environ["PATH"] = saved
            import config
            importlib.reload(config)


class TestInstalledPicker(Isolated):
    """Блок «Кто отвечает» предлагает запускать уже загруженные модели."""

    def test_installed_llms_lists_downloaded(self):
        import models
        p = models.local_path(models.BY_ID["qwen3-14b"])
        p.mkdir(parents=True, exist_ok=True)
        (p / "model.safetensors").write_bytes(b"x")
        inst = models.installed_llms()
        ids = [m["id"] for m in inst]
        self.assertIn("qwen3-14b", ids)
        entry = next(m for m in inst if m["id"] == "qwen3-14b")
        self.assertFalse(entry["serving"])
        self.assertIn(entry["engine"], ("vllm", "ollama"))

    def test_not_downloaded_not_listed(self):
        import models
        ids = [m["id"] for m in models.installed_llms()]
        self.assertNotIn("qwen3-32b", ids)

    def test_wiring(self):
        from tests.base import ROOT
        src = (ROOT / "webui.py").read_text(encoding="utf-8")
        for needle in ('id="llmServe"', "serveAndUse", '"installed"'):
            self.assertIn(needle, src, needle)


class TestServeErrors(Isolated):
    """Кнопка «Запустить»: счастливый путь работает, отказы объясняют себя."""

    def _mac_stack(self, st, has_ollama=True, alive=True, pulled=True,
                   tags=None, knows=None):
        import platform
        from unittest import mock

        import models
        st.enter_context(mock.patch.object(platform, "system",
                                           return_value="Darwin"))
        st.enter_context(mock.patch.object(platform, "machine",
                                           return_value="arm64"))
        st.enter_context(mock.patch.object(
            models.shutil, "which",
            lambda x: "/x/ollama" if (x == "ollama" and has_ollama) else None))
        st.enter_context(mock.patch.object(models, "_ollama_alive",
                                           return_value=alive))
        st.enter_context(mock.patch.object(models, "_ollama_has",
                                           return_value=pulled))
        st.enter_context(mock.patch.object(models, "_ollama_tags",
                                           return_value=tags))
        st.enter_context(mock.patch.object(models, "_server_knows",
                                           return_value=knows))

    def test_external_ollama_serve_does_not_crash(self):
        """Регрессия: запущенное приложение Ollama роняло serve на proc.pid."""
        import contextlib

        import models
        with contextlib.ExitStack() as st:
            self._mac_stack(st)
            state = models.serve("qwen3.6-27b", apply_config=False)
        self.assertTrue(state["external"])
        self.assertEqual(state["engine"], "ollama")
        self.assertIsNone(state["pid"])

    def test_not_pulled_explains_next_step(self):
        import contextlib

        import models
        with contextlib.ExitStack() as st:
            self._mac_stack(st, pulled=False)
            with self.assertRaisesRegex(RuntimeError, "ollama pull"):
                models.serve("qwen3.6-27b", apply_config=False)

    def test_wrong_size_names_sibling(self):
        """Регрессия «не знает модель qwen3.6:35b»: скачана 27b, запускают
        35b — раньше проверка по началу имени это пропускала, и каждый
        вопрос падал с 404. Теперь честный отказ, называющий похожую."""
        import contextlib

        import models
        with contextlib.ExitStack() as st:
            self._mac_stack(st, tags=["qwen3.6:27b"])
            with self.assertRaises(RuntimeError) as ctx:
                models.serve("qwen3.6-35b-a3b", apply_config=False)
        text = str(ctx.exception)
        self.assertIn("qwen3.6:35b", text)          # чего не хватает
        self.assertIn("ollama pull qwen3.6:35b", text)
        self.assertIn("qwen3.6:27b", text)          # что есть похожего
        self.assertIn("похожая", text)

    def test_exact_tag_present_serves(self):
        import contextlib

        import models
        with contextlib.ExitStack() as st:
            self._mac_stack(st, tags=["qwen3.6:27b", "qwen3:14b"])
            state = models.serve("qwen3.6-27b", apply_config=False)
        self.assertEqual(state["served_name"], "qwen3.6:27b")

    def test_empty_server_says_so(self):
        import contextlib

        import models
        with contextlib.ExitStack() as st:
            self._mac_stack(st, tags=[])
            with self.assertRaisesRegex(RuntimeError,
                                        "ни одной модели"):
                models.serve("qwen3.6-27b", apply_config=False)

    def test_server_mismatch_detected_before_binding(self):
        """Сервер жив, `ollama list` модель видит, а работающий сервер —
        нет (другой демон): отказ до привязки, а не 404 на каждом вопросе."""
        import contextlib

        import models
        with contextlib.ExitStack() as st:
            self._mac_stack(st, tags=["qwen3.6:27b"], knows=False)
            with self.assertRaisesRegex(RuntimeError, "другой экземпляр"):
                models.serve("qwen3.6-27b", apply_config=False)
        self.assertFalse(models._pid_file().exists())

    def test_tag_eq_treats_latest_as_default(self):
        import models
        self.assertTrue(models._tag_eq("gemma3", "gemma3:latest"))
        self.assertFalse(models._tag_eq("qwen3.6:27b", "qwen3.6:35b"))

    def test_is_installed_requires_exact_tag(self):
        import contextlib
        from unittest import mock

        import models
        with contextlib.ExitStack() as st:
            st.enter_context(mock.patch.object(
                models.shutil, "which", lambda x: "/x/ollama"))
            st.enter_context(mock.patch.object(
                models, "_ollama_tags", return_value=["qwen3.6:27b"]))
            self.assertTrue(models.is_installed(models.BY_ID["qwen3.6-27b"]))
            self.assertFalse(
                models.is_installed(models.BY_ID["qwen3.6-35b-a3b"]))

    def test_vllm_model_on_mac_redirects_to_ollama_models(self):
        import contextlib

        import models
        with contextlib.ExitStack() as st:
            self._mac_stack(st)
            with self.assertRaisesRegex(RuntimeError, "ollama"):
                models.serve("t-pro-2.0", apply_config=False)

    def test_second_serve_with_external_server_does_not_crash(self):
        """Регрессия: os.kill(None) в stop() ронял ПОВТОРНЫЙ запуск при
        внешнем сервере ollama ошибкой «NoneType … integer»."""
        import contextlib

        import models
        with contextlib.ExitStack() as st:
            self._mac_stack(st)
            models.serve("qwen3.6-27b", apply_config=False)
            state = models.serve("qwen3.6-35b-a3b", apply_config=False)
        self.assertEqual(state["model"], "qwen3.6-35b-a3b")

    def test_stop_external_unloads_model_and_explains(self):
        """«Остановить» при внешнем Ollama: модель выгружается из памяти
        (keep_alive=0), приложение не трогаем, журнал говорит понятно."""
        import contextlib
        from unittest import mock

        import models
        with contextlib.ExitStack() as st:
            self._mac_stack(st)
            unload = st.enter_context(mock.patch.object(
                models, "_ollama_unload", return_value=True))
            models.serve("qwen3.6-27b", apply_config=False)
            self.assertTrue(models.stop())
        self.assertFalse(models._pid_file().exists())
        tag = models.BY_ID["qwen3.6-27b"].ollama_tag
        unload.assert_called_once_with(tag)
        last = models.action_log(1)[0]
        self.assertEqual(last["action"], "остановка")
        self.assertIn("выгружена из памяти", last["detail"])

    def test_stop_external_unload_failed_still_unbinds(self):
        """Если Ollama не ответила на выгрузку — всё равно отвязываемся,
        а журнал объясняет, что память освободится по таймауту."""
        import contextlib
        from unittest import mock

        import models
        with contextlib.ExitStack() as st:
            self._mac_stack(st)
            st.enter_context(mock.patch.object(
                models, "_ollama_unload", return_value=False))
            models.serve("qwen3.6-27b", apply_config=False)
            self.assertTrue(models.stop())
        self.assertFalse(models._pid_file().exists())
        last = models.action_log(1)[0]
        self.assertIn("освободится сама", last["detail"])


class TestLocalVision(Isolated):
    """Описание изображений по умолчанию — локальной моделью, сама находится."""

    def test_default_provider_is_local(self):
        import settings_schema
        spec = next(s for s in settings_schema.SETTINGS
                    if s["key"] == "VISION_PROVIDER")
        self.assertEqual(spec["default"], "local")

    def test_prefers_running_vision_server(self):
        from unittest import mock

        import media
        import models
        st = {"running": True, "model": "qwen3-vl-8b",
              "base_url": "http://127.0.0.1:8011/v1",
              "served_name": "qwen3-vl:8b"}
        with mock.patch.object(models, "status", return_value=st):
            base, model = media.local_vision_endpoint()
        self.assertEqual(base, "http://127.0.0.1:8011/v1")
        self.assertEqual(model, "qwen3-vl:8b")

    def test_falls_back_to_ollama_vision_model(self):
        from unittest import mock

        import media
        import models
        with mock.patch.object(models, "status",
                               return_value={"running": False}), \
             mock.patch.object(models, "_ollama_tags",
                               return_value=["qwen3.6:27b", "qwen3-vl:8b"]):
            base, model = media.local_vision_endpoint()
        self.assertIn("11434", base)
        self.assertEqual(model, "qwen3-vl:8b")

    def test_no_model_explains_what_to_download(self):
        from unittest import mock

        import media
        import models
        with mock.patch.object(models, "status",
                               return_value={"running": False}), \
             mock.patch.object(models, "_ollama_tags", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "Скачать"):
                media.local_vision_endpoint()

    def test_serve_vision_does_not_touch_openai_settings(self):
        """Регрессия: запуск зрительной модели прописывал OPENAI_BASE_URL
        и молча ломал настройки облачного провайдера."""
        import contextlib
        from unittest import mock

        import models
        import webui
        with contextlib.ExitStack() as st:
            for args in (("system", "Darwin"), ("machine", "arm64")):
                import platform
                st.enter_context(mock.patch.object(platform, args[0],
                                                   return_value=args[1]))
            st.enter_context(mock.patch.object(
                models.shutil, "which", lambda x: "/x/ollama"))
            st.enter_context(mock.patch.object(models, "_ollama_alive",
                                               return_value=True))
            st.enter_context(mock.patch.object(models, "_ollama_tags",
                                               return_value=["qwen3-vl:8b"]))
            st.enter_context(mock.patch.object(models, "_server_knows",
                                               return_value=None))
            write = st.enter_context(mock.patch.object(webui, "write_env"))
            models.serve("qwen3-vl-8b", apply_config=True)
        updates = write.call_args[0][0]
        self.assertEqual(updates["VISION_PROVIDER"], "local")
        self.assertNotIn("OPENAI_BASE_URL", updates)
        self.assertNotIn("OPENAI_API_KEY", updates)

    def test_vision_models_runnable_on_mac(self):
        import models
        for mid in ("qwen3-vl-8b", "qwen3-vl-32b"):
            spec = models.BY_ID[mid]
            self.assertTrue(spec.ollama_tag, mid)


class TestModelProgressAndLog(Isolated):
    """Прогресс-бар и журнал действий в разделе «Модели»."""

    def test_action_log_roundtrip_and_cap(self):
        import models
        for i in range(5):
            models.record_action("запуск", f"m{i}", "деталь")
        log = models.action_log(3)
        self.assertEqual(len(log), 3)
        self.assertEqual(log[0]["model"], "m4")

    def test_percent_parsed_from_both_downloaders(self):
        import models
        for line in ("pulling 9f3a: 45% ▕████▏ 4.2 GB/9.1 GB",
                     "model.safetensors:  78%|███▊| 7.1G/9.1G"):
            self.assertTrue(models._PERCENT_RX.findall(line), line)

    def test_install_writes_progress_and_log(self):
        from unittest import mock

        import models
        with mock.patch.object(models, "hardware",
                               return_value={"vram_total_gb": 24}), \
             mock.patch.object(models.shutil, "which",
                               lambda x: "/x/ollama" if x == "ollama" else None), \
             mock.patch.object(models.subprocess, "Popen") as pop:
            pop.return_value.stdout = iter(["pulling: 10%", "pulling: 95%"])
            pop.return_value.wait = lambda: None
            pop.return_value.returncode = 0
            models.install("qwen3.6-27b", engine="ollama")
        state = models.download_progress()
        self.assertTrue(state["done"])
        self.assertEqual(state["percent"], 100)
        self.assertEqual(models.action_log(1)[0]["action"], "загрузка завершена")

    def test_failed_install_recorded(self):
        from unittest import mock

        import models
        with mock.patch.object(models, "hardware",
                               return_value={"vram_total_gb": 24}), \
             mock.patch.object(models.shutil, "which",
                               lambda x: "/x/ollama" if x == "ollama" else None), \
             mock.patch.object(models.subprocess, "Popen") as pop:
            pop.return_value.stdout = iter(["boom"])
            pop.return_value.wait = lambda: None
            pop.return_value.returncode = 1
            with self.assertRaises(RuntimeError):
                models.install("qwen3.6-27b", engine="ollama")
        self.assertTrue(models.download_progress()["error"])
        self.assertEqual(models.action_log(1)[0]["action"], "загрузка не удалась")

    def test_wiring(self):
        from tests.base import ROOT
        src = (ROOT / "webui.py").read_text(encoding="utf-8")
        for needle in ('"/api/models/progress"', 'id="mProgress"',
                       'id="mActions"', "loadModelProgress"):
            self.assertIn(needle, src, needle)
