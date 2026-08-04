"""
Автонастройка компонентов при смене настройки.

Свойства: настройка сохраняется только после успешной подготовки;
ошибки называют команду или настройку, которой не хватает; смена
параметра из формы уходит в задачу, а не падает на валидации.
"""
from __future__ import annotations

import contextlib
import unittest
from unittest import mock

from tests.base import Isolated

import components


class TestComponents(Isolated):
    def test_registry_covers_provider_settings(self):
        for key in ("RERANKER_PROVIDER", "OCR_PROVIDER", "VISION_PROVIDER",
                    "ASR_PROVIDER", "VOICE_TTS_PROVIDER", "VECTOR_BACKEND"):
            self.assertIn(key, components.KEYS)

    def test_unknown_key_refuses(self):
        with self.assertRaisesRegex(RuntimeError, "автонастройки нет"):
            components.switch("SEARCH_TOP_K", "8", persist=False)

    def test_lexical_reranker_needs_nothing(self):
        out = components.switch("RERANKER_PROVIDER", "lexical", persist=False)
        self.assertEqual(out["updates"], {"RERANKER_PROVIDER": "lexical"})

    def test_onnx_reranker_autoconverts_model(self):
        onnx = self.tmp / "bge-reranker-v2-m3-onnx" / "model.onnx"
        onnx.parent.mkdir(parents=True)
        onnx.write_bytes(b"x")
        with contextlib.ExitStack() as st:
            st.enter_context(mock.patch.object(components, "_has_module",
                                               return_value=True))
            st.enter_context(mock.patch.object(components, "_export_onnx",
                                               return_value=str(onnx)))
            out = components.switch("RERANKER_PROVIDER", "onnx", persist=False)
        self.assertEqual(out["updates"]["RERANKER_ONNX_PATH"], str(onnx))

    def test_onnx_reranker_failure_suggests_lexical(self):
        with contextlib.ExitStack() as st:
            st.enter_context(mock.patch.object(components, "_has_module",
                                               return_value=True))
            st.enter_context(mock.patch.object(components, "_export_onnx",
                                               return_value=None))
            with self.assertRaisesRegex(RuntimeError, "lexical"):
                components.switch("RERANKER_PROVIDER", "onnx", persist=False)

    def test_tesseract_missing_names_install_command(self):
        with contextlib.ExitStack() as st:
            st.enter_context(mock.patch.object(components.shutil, "which",
                                               return_value=None))
            st.enter_context(mock.patch.object(components, "_brew_or_apt",
                                               return_value=False))
            with self.assertRaisesRegex(RuntimeError, "tesseract-ocr-rus"):
                components.switch("OCR_PROVIDER", "tesseract", persist=False)

    def test_vision_local_downloads_model_when_missing(self):
        import media
        import models
        with contextlib.ExitStack() as st:
            st.enter_context(mock.patch.object(
                media, "local_vision_endpoint",
                side_effect=[RuntimeError("нет"),
                             ("http://127.0.0.1:11434/v1", "qwen3-vl:8b")]))
            install = st.enter_context(mock.patch.object(models, "install"))
            components.switch("VISION_PROVIDER", "local", persist=False)
        install.assert_called_once()
        self.assertEqual(install.call_args[0][0], "qwen3-vl-8b")

    def test_asr_installs_python_package(self):
        with contextlib.ExitStack() as st:
            st.enter_context(mock.patch.object(components, "_has_module",
                                               return_value=False))
            pip = st.enter_context(mock.patch.object(components,
                                                     "_pip_install"))
            st.enter_context(mock.patch.object(components.shutil, "which",
                                               return_value="/usr/bin/ffmpeg"))
            components.switch("ASR_PROVIDER", "faster-whisper", persist=False)
        pip.assert_called_once()
        self.assertEqual(pip.call_args[0][0], ["faster-whisper"])

    def test_qdrant_unreachable_names_docker_command(self):
        with contextlib.ExitStack() as st:
            st.enter_context(mock.patch.object(components, "_has_module",
                                               return_value=True))
            with self.assertRaisesRegex(RuntimeError, "docker run"):
                components.switch("VECTOR_BACKEND", "qdrant", persist=False)

    def test_cloud_providers_require_keys(self):
        for key, value, need in (("OCR_PROVIDER", "yandex", "YANDEX_API_KEY"),
                                 ("VOICE_TTS_PROVIDER", "sber",
                                  "GIGACHAT_AUTH_KEY")):
            with self.assertRaisesRegex(RuntimeError, need):
                components.switch(key, value, persist=False)

    def test_persist_saves_and_reloads(self):
        import webui
        with contextlib.ExitStack() as st:
            write = st.enter_context(mock.patch.object(webui, "write_env"))
            st.enter_context(mock.patch.object(webui,
                                               "_reload_after_settings"))
            components.switch("RERANKER_PROVIDER", "lexical", persist=True)
        write.assert_called_once_with({"RERANKER_PROVIDER": "lexical"})


class TestQueueAllowsDifferentComponents(Isolated):
    def test_same_kind_different_payload_queues(self):
        import jobs
        jobs.ensure_tables()
        self.db.run("DELETE FROM jobs")
        jobs.enqueue("component_setup", "настройка: сканы",
                     {"key": "OCR_PROVIDER", "value": "tesseract"})
        job = jobs.enqueue("component_setup", "настройка: речь",
                           {"key": "ASR_PROVIDER", "value": "faster-whisper"},
                           wait=True)
        self.assertEqual(job["status"], "queued")
        with self.assertRaisesRegex(jobs.Busy, "уже есть"):
            jobs.enqueue("component_setup", "настройка: сканы",
                         {"key": "OCR_PROVIDER", "value": "tesseract"},
                         wait=True)


class TestWiring(unittest.TestCase):
    def test_settings_form_delegates_component_changes(self):
        from tests.base import ROOT
        src = (ROOT / "webui.py").read_text(encoding="utf-8")
        for needle in ('"component_setup"', "components_mod.KEYS",
                       "настройка компонента"):
            self.assertIn(needle, src, needle)
        handlers_src = (ROOT / "handlers.py").read_text(encoding="utf-8")
        self.assertIn('@jobs.handler("component_setup")', handlers_src)
        import jobs
        self.assertIn("component_setup", jobs.RESOURCES)


if __name__ == "__main__":
    unittest.main()
