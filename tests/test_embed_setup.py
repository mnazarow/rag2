"""
Смена провайдера смыслового поиска под ключ.

Главные свойства: при любой неудаче настройки не меняются (поиск
продолжает работать на прежнем провайдере), ошибки объясняют следующее
действие, а успешное переключение сохраняется в настройки — а не живёт
в памяти процесса до первого перезапуска.
"""
from __future__ import annotations

import contextlib
import unittest
from unittest import mock

import numpy as np

from tests.base import Isolated

import embed_setup


class FakeEmbedder:
    def __init__(self, dim=64):
        self.dim = dim

    def embed(self, texts):
        return np.ones((len(texts), self.dim), dtype=np.float32)


class TestSwitch(Isolated):
    def _ok_stack(self, st, dim=64):
        import embeddings
        st.enter_context(mock.patch.object(embed_setup, "_has_module",
                                           return_value=True))
        st.enter_context(mock.patch.object(embeddings, "get_embedder",
                                           return_value=FakeEmbedder(dim)))

    def test_unknown_provider_lists_available(self):
        with self.assertRaises(RuntimeError) as ctx:
            embed_setup.switch("nonexistent", persist=False)
        self.assertIn("lsa", str(ctx.exception))

    def test_cloud_without_key_refuses_and_keeps_settings(self):
        before = self.config.EMBEDDINGS_PROVIDER
        with contextlib.ExitStack() as st:
            self._ok_stack(st)
            with self.assertRaises(RuntimeError) as ctx:
                embed_setup.switch("gigachat", persist=False)
        self.assertIn("GIGACHAT_AUTH_KEY", str(ctx.exception))
        self.assertIn("не изменены", str(ctx.exception))
        self.assertEqual(self.config.EMBEDDINGS_PROVIDER, before)

    def test_openai_needs_base_url(self):
        self.config.OPENAI_BASE_URL = ""
        with contextlib.ExitStack() as st:
            self._ok_stack(st)
            with self.assertRaisesRegex(RuntimeError, "OPENAI_BASE_URL"):
                embed_setup.switch("openai", persist=False)

    def test_local_switch_succeeds_and_reports_updates(self):
        with contextlib.ExitStack() as st:
            self._ok_stack(st)
            st.enter_context(mock.patch.object(
                embed_setup, "_ensure_local_weights",
                return_value="deepvk/USER-bge-m3"))
            result = embed_setup.switch("local", persist=False)
        self.assertEqual(result["provider"], "local")
        self.assertEqual(result["dim"], 64)
        self.assertEqual(result["updates"]["EMBEDDINGS_PROVIDER"], "local")
        self.assertEqual(result["updates"]["EMBEDDINGS_MODEL"],
                         "deepvk/USER-bge-m3")

    def test_failed_probe_rolls_config_back(self):
        """Провайдер не ответил — конфигурация в памяти прежняя."""
        import embeddings
        before = self.config.EMBEDDINGS_PROVIDER
        with contextlib.ExitStack() as st:
            st.enter_context(mock.patch.object(embed_setup, "_has_module",
                                               return_value=True))
            st.enter_context(mock.patch.object(
                embed_setup, "_ensure_local_weights", return_value="x"))
            st.enter_context(mock.patch.object(
                embeddings, "get_embedder",
                side_effect=RuntimeError("нет связи с сервером")))
            with self.assertRaises(RuntimeError) as ctx:
                embed_setup.switch("local", persist=False)
        text = str(ctx.exception)
        self.assertIn("нет связи с сервером", text)
        self.assertIn(before, text)          # на чём продолжает работать
        self.assertEqual(self.config.EMBEDDINGS_PROVIDER, before)

    def test_pip_failure_shows_pip_output(self):
        bad = mock.Mock(returncode=1, stdout="",
                        stderr="ERROR: No matching distribution found for x")
        with mock.patch.object(embed_setup.subprocess, "run",
                               return_value=bad) as run:
            with self.assertRaises(RuntimeError) as ctx:
                embed_setup._pip_install(["x"], lambda t: None)
        self.assertIn("No matching distribution", str(ctx.exception))
        self.assertEqual(run.call_count, 2)  # вторая попытка с флагом
        self.assertIn("--break-system-packages", run.call_args[0][0])

    def test_lsa_trains_automatically_when_missing(self):
        import embeddings
        with contextlib.ExitStack() as st:
            self._ok_stack(st)
            train = st.enter_context(mock.patch.object(
                embeddings.LSAEmbedder, "train"))
            embed_setup.switch("lsa", persist=False)
        train.assert_called_once()

    def test_onnx_without_file_explains_alternatives(self):
        with contextlib.ExitStack() as st:
            st.enter_context(mock.patch.object(embed_setup, "_has_module",
                                               return_value=True))
            with self.assertRaises(RuntimeError) as ctx:
                embed_setup.switch("onnx", persist=False)
        text = str(ctx.exception)
        self.assertIn("optimum-cli", text)
        self.assertIn("local", text)          # более простой путь назван
        self.assertIn("не изменены", text)

    def test_onnx_autoconverts_when_possible(self):
        """Выбор onnx без готового файла: модель конвертируется сама,
        путь прописывается в ONNX_MODEL_PATH — ручной подготовки нет."""
        import embeddings
        onnx_file = self.tmp / "user-bge-m3-onnx" / "model.onnx"
        onnx_file.parent.mkdir(parents=True)
        onnx_file.write_bytes(b"onnx")
        with contextlib.ExitStack() as st:
            st.enter_context(mock.patch.object(embed_setup, "_has_module",
                                               return_value=True))
            st.enter_context(mock.patch.object(
                embed_setup, "_export_onnx", return_value=str(onnx_file)))
            st.enter_context(mock.patch.object(embeddings, "get_embedder",
                                               return_value=FakeEmbedder()))
            result = embed_setup.switch("onnx", persist=False)
        self.assertEqual(result["updates"]["ONNX_MODEL_PATH"], str(onnx_file))

    def test_persist_writes_settings(self):
        import webui
        with contextlib.ExitStack() as st:
            self._ok_stack(st)
            st.enter_context(mock.patch.object(
                embed_setup, "_ensure_local_weights", return_value="m"))
            write = st.enter_context(mock.patch.object(webui, "write_env"))
            embed_setup.switch("local", persist=True)
        write.assert_called_once()
        saved = write.call_args[0][0]
        self.assertEqual(saved["EMBEDDINGS_PROVIDER"], "local")


class TestSecretsVisibleToValidation(Isolated):
    def test_required_key_found_in_secrets_file(self):
        """Ключ в защищённом файле — а не в .env. Проверка связей обязана
        его видеть, иначе провайдера с заполненным ключом не выбрать."""
        import security
        import webui
        with mock.patch.object(security, "load_secrets",
                               return_value={"GIGACHAT_AUTH_KEY": "k"}):
            with mock.patch.object(webui, "read_env", return_value={}):
                full = webui.env_with_secrets()
        self.assertEqual(full.get("GIGACHAT_AUTH_KEY"), "k")

        import settings_schema
        issues = settings_schema.validate(
            {"EMBEDDINGS_PROVIDER": "gigachat"}, full=full)
        self.assertFalse([i for i in issues if i["level"] == "error"], issues)


class TestWiring(unittest.TestCase):
    def test_job_and_ui_wired(self):
        from tests.base import ROOT
        src = (ROOT / "webui.py").read_text(encoding="utf-8")
        for needle in ('"embed_switch"', "switchEmb", 'id="reProv"',
                       "Переключить провайдера", "env_with_secrets",
                       # Смена провайдера из формы настроек делегируется
                       # задаче, а не падает на «не заполнен ONNX_MODEL_PATH».
                       "отдельной задачей", "if(r.note) alert(r.note)"):
            self.assertIn(needle, src, needle)
        handlers_src = (ROOT / "handlers.py").read_text(encoding="utf-8")
        self.assertIn('@jobs.handler("embed_switch")', handlers_src)
        import jobs
        self.assertIn("embed_switch", jobs.RESOURCES)


if __name__ == "__main__":
    unittest.main()
