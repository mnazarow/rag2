"""
Смена провайдера и модели генерации под ключ.

Главные свойства — те же, что у смены смыслового поиска: при любой
неудаче настройки не меняются, провайдер сохраняется только после
успешного пробного вопроса, ошибки называют следующее действие.
"""
from __future__ import annotations

import contextlib
import unittest
from unittest import mock

from tests.base import Isolated

import llm_setup


class TestLlmSwitch(Isolated):
    def _probe_ok(self, st):
        import llm
        st.enter_context(mock.patch.object(
            llm, "probe",
            return_value={"ok": True, "ms": 7, "answer": "работает"}))

    def test_unknown_provider_lists_available(self):
        with self.assertRaises(RuntimeError) as ctx:
            llm_setup.switch("nonexistent", persist=False)
        self.assertIn("local", str(ctx.exception))

    def test_cloud_without_key_refuses(self):
        before = self.config.LLM_PROVIDER
        with self.assertRaises(RuntimeError) as ctx:
            llm_setup.switch("gigachat", persist=False)
        self.assertIn("GIGACHAT_AUTH_KEY", str(ctx.exception))
        self.assertEqual(self.config.LLM_PROVIDER, before)

    def test_openai_needs_base_url(self):
        self.config.OPENAI_BASE_URL = ""
        with self.assertRaisesRegex(RuntimeError, "OPENAI_BASE_URL"):
            llm_setup.switch("openai", persist=False)

    def test_local_downloads_serves_and_saves_served_name(self):
        import models
        spec = models.BY_ID["qwen3.6-27b"]
        with contextlib.ExitStack() as st:
            self._probe_ok(st)
            st.enter_context(mock.patch.object(models, "is_installed",
                                               return_value=False))
            install = st.enter_context(mock.patch.object(models, "install"))
            st.enter_context(mock.patch.object(models, "status",
                                               return_value={"running": False}))
            st.enter_context(mock.patch.object(
                models, "serve",
                return_value={"served_name": "qwen3.6:27b", "model": spec.id}))
            result = llm_setup.switch("local", model="qwen3.6:27b",
                                      persist=False)
        install.assert_called_once()
        self.assertEqual(install.call_args[0][0], "qwen3.6-27b")
        self.assertEqual(result["updates"]["LOCAL_LLM_MODEL"], "qwen3.6:27b")
        self.assertEqual(result["updates"]["LLM_PROVIDER"], "local")

    def test_local_unknown_model_lists_catalog(self):
        with self.assertRaises(RuntimeError) as ctx:
            llm_setup.switch("local", model="нет-такой", persist=False)
        self.assertIn("qwen3.6:27b", str(ctx.exception))

    def test_failed_probe_rolls_back(self):
        import llm
        before = self.config.LLM_PROVIDER
        with contextlib.ExitStack() as st:
            st.enter_context(mock.patch.object(
                llm, "probe",
                return_value={"ok": False, "error": "нет связи"}))
            with self.assertRaises(RuntimeError) as ctx:
                llm_setup.switch("echo", persist=False)
        self.assertIn("нет связи", str(ctx.exception))
        self.assertEqual(self.config.LLM_PROVIDER, before)

    def test_persist_writes_settings(self):
        import webui
        with contextlib.ExitStack() as st:
            self._probe_ok(st)
            write = st.enter_context(mock.patch.object(webui, "write_env"))
            llm_setup.switch("echo", persist=True)
        saved = write.call_args[0][0]
        self.assertEqual(saved["LLM_PROVIDER"], "echo")

    def test_cloud_model_goes_to_llm_model(self):
        with contextlib.ExitStack() as st:
            self._probe_ok(st)
            result = llm_setup.switch("echo", model="GigaChat-2-Max",
                                      persist=False)
        self.assertEqual(result["updates"]["LLM_MODEL"], "GigaChat-2-Max")


class TestWiring(unittest.TestCase):
    def test_job_settings_and_installers_wired(self):
        from tests.base import ROOT
        src = (ROOT / "webui.py").read_text(encoding="utf-8")
        for needle in ('"llm_switch"', "Генерация переключается",
                       "Смысловой поиск переключается",
                       "s.type==='suggest'"):
            self.assertIn(needle, src, needle)
        handlers_src = (ROOT / "handlers.py").read_text(encoding="utf-8")
        self.assertIn('@jobs.handler("llm_switch")', handlers_src)
        import jobs
        self.assertIn("llm_switch", jobs.RESOURCES)

        import settings_schema
        by_key = {s["key"]: s for s in settings_schema.SETTINGS}
        self.assertIn("local", by_key["LLM_PROVIDER"]["options"])
        self.assertEqual(by_key["LLM_MODEL"]["type"], "suggest")
        self.assertEqual(by_key["LOCAL_LLM_MODEL"]["type"], "suggest")
        self.assertIn("qwen3.6:27b", by_key["LOCAL_LLM_MODEL"]["options"])
        self.assertEqual(by_key["EMBEDDINGS_MODEL"]["type"], "enum")
        self.assertIn("deepvk/USER-bge-m3",
                      by_key["EMBEDDINGS_MODEL"]["options"])

    def test_cad_tools_in_installers(self):
        """ODA File Converter и FreeCAD ставятся и настраиваются при
        установке и обновлении на всех трёх системах."""
        from tests.base import ROOT
        for name in ("install.sh", "update.sh"):
            src = (ROOT / "install" / name).read_text(encoding="utf-8")
            for needle in ("oda-file-converter", "freecad",
                           "ODA_CONVERTER", "FREECAD_CMD", "set_env_default"):
                self.assertIn(needle, src, f"{name}: {needle}")
        ps = (ROOT / "install" / "install.ps1").read_text(encoding="utf-8")
        self.assertIn("ODAFileConverter", ps)
        self.assertIn("FreeCAD", ps)

    def test_suggest_type_is_not_strict(self):
        """Список моделей — подсказка: своё значение не считается ошибкой."""
        import settings_schema
        issues = settings_schema.validate({"LLM_MODEL": "моя-модель-x"},
                                          full={})
        self.assertFalse([i for i in issues if i["key"] == "LLM_MODEL"],
                         issues)


if __name__ == "__main__":
    unittest.main()
