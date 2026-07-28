"""
Проверки самого кода, а не поведения.

Две вещи, которые невозможно поймать обычным тестом, потому что они не
падают там, где написаны, — они падают у пользователя на экране, куда
тесты не заглядывают.

Первая: обращение к таблице не в ту базу. С тех пор как телеметрия
вынесена в отдельный файл, у нас два набора помощников — `db.q` для
основной базы и `db.tq` для служебной. Перепутанный помощник даёт не
неверное число, а «нет такой таблицы»: на свежей установке обзорный
экран просто не открывался, и обнаружилось это только при проверке
чистой установки.

Вторая: запрос к модели мимо очереди. Ограничение, которое можно обойти,
забыв про него в новом месте кода, не ограничивает ничего — а забыть
легко, потому что без очереди всё продолжает работать и даже быстрее,
пока пользователей мало.
"""
from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

from tests.base import ROOT

# Таблицы, живущие в отдельной базе телеметрии.
TELEMETRY_TABLES = {
    "server_metrics", "model_usage", "stage_timings", "log_records",
    "traces", "llm_queue", "alerts", "alert_history", "telemetry_marker",
}
TELEMETRY_FN = {"tq", "tq1", "trun"}
MAIN_FN = {"q", "q1", "run", "runmany"}
NEUTRAL = {"sqlite_master", "schema_meta", "chunks_fts"}
TABLE_RE = re.compile(r"\b(?:FROM|INTO|UPDATE|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.I)


def _modules():
    return sorted(p for p in ROOT.glob("*.py"))


class TestDatabaseRouting(unittest.TestCase):
    def test_every_query_goes_to_its_own_database(self):
        wrong = []
        for path in _modules():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "db"):
                    continue
                fn = node.func.attr
                if fn not in TELEMETRY_FN | MAIN_FN:
                    continue
                if not node.args or not isinstance(node.args[0], ast.Constant):
                    continue
                sql = node.args[0].value
                if not isinstance(sql, str):
                    continue
                for table in {t.lower() for t in TABLE_RE.findall(sql)} - NEUTRAL:
                    telemetry = table in TELEMETRY_TABLES
                    if telemetry != (fn in TELEMETRY_FN):
                        where = "телеметрии" if telemetry else "основной"
                        wrong.append(f"{path.name}:{node.lineno} db.{fn} → "
                                     f"{table} (таблица в базе {where})")
        self.assertFalse(wrong, "запрос уйдёт не в ту базу:\n" + "\n".join(wrong))


class TestEverythingUsesTheQueue(unittest.TestCase):
    """
    Обращаться к модели можно только через `llm.get_llm()` — этот путь
    берёт место в очереди. Прямое создание провайдера в обход очереди
    допустимо ровно в двух местах: в самом llm.py и в проверке связи,
    которая берёт место явно.
    """
    ALLOWED = {"llm.py", "voice.py"}          # voice берёт только авторизацию

    def test_no_direct_provider_calls(self):
        offenders = []
        for path in _modules():
            if path.name in self.ALLOWED:
                continue
            src = path.read_text(encoding="utf-8")
            for i, line in enumerate(src.splitlines(), 1):
                if re.search(r"\b(GigaChatLLM|YandexLLM|OpenAICompatibleLLM|"
                             r"LocalLLM|EchoLLM)\s*\(", line):
                    offenders.append(f"{path.name}:{i}: {line.strip()}")
        self.assertFalse(offenders,
                         "провайдер создаётся напрямую, минуя очередь:\n"
                         + "\n".join(offenders))

    def test_llm_module_takes_a_slot(self):
        src = (ROOT / "llm.py").read_text(encoding="utf-8")
        self.assertIn("llm_queue.slot(", src)
        # Оба пути к модели — обычный ответ и проверка связи — под очередью.
        self.assertGreaterEqual(src.count("llm_queue.slot("), 2)

    def test_vlm_ocr_takes_a_slot(self):
        """Зрительная модель стоит на тех же картах, значит — та же очередь."""
        src = (ROOT / "ocr.py").read_text(encoding="utf-8")
        self.assertIn("llm_queue.slot(", src)


class TestSettingsAreDocumented(unittest.TestCase):
    """
    Каждая настройка должна быть в панели администратора и описана там:
    иначе она существует только в голове того, кто её добавил, и найти её
    можно лишь чтением исходников.
    """

    def test_every_setting_is_in_the_admin_panel(self):
        import settings_schema
        described = {s["key"] for s in settings_schema.SETTINGS}
        src = (ROOT / "config.py").read_text(encoding="utf-8")
        declared = set(re.findall(r"^([A-Z][A-Z0-9_]+)\s*=\s*_env", src, re.M))
        missing = sorted(declared - described)
        self.assertFalse(missing,
                         "настройки нет в панели администратора: "
                         + ", ".join(missing))

    def test_every_setting_has_help_recommendation_and_example(self):
        import settings_schema
        bad = []
        for s in settings_schema.SETTINGS:
            for field in ("title", "help", "rec"):
                if not str(s.get(field) or "").strip():
                    bad.append(f"{s['key']}: нет поля {field}")
            # У ключей пример намеренно пустой: показывать образец ключа
            # незачем, а придуманный образец люди пробуют вставлять как есть.
            if s["type"] != "secret" and "example" not in s:
                bad.append(f"{s['key']}: нет примера")
        self.assertFalse(bad, "\n".join(bad))

    def test_groups_are_contiguous(self):
        """
        Настройки одного раздела должны идти подряд: иначе на экране
        появляется три одинаковых заголовка, ссылка ведёт только на
        первый, а в .env повторяется комментарий-заголовок.
        """
        import settings_schema
        seen, order = set(), []
        for s in settings_schema.SETTINGS:
            if not order or order[-1] != s["group"]:
                order.append(s["group"])
        repeated = [g for g in order if order.count(g) > 1]
        self.assertFalse(set(repeated), f"раздел разорван: {set(repeated)}")
        self.assertEqual(len(order), len(set(order)))
        seen.update(order)

    def test_enum_options_include_the_default(self):
        import settings_schema
        bad = [s["key"] for s in settings_schema.SETTINGS
               if s["type"] == "enum" and s.get("default")
               and s["default"] not in (s.get("options") or [])]
        self.assertFalse(bad, f"умолчание вне списка вариантов: {bad}")

    def test_ranges_contain_the_default(self):
        import settings_schema
        bad = []
        for s in settings_schema.SETTINGS:
            bounds = settings_schema.RANGES.get(s["key"])
            if not bounds:
                continue
            try:
                value = float(s.get("default", 0))
            except (TypeError, ValueError):
                continue
            if not bounds[0] <= value <= bounds[3]:
                bad.append(f"{s['key']}: умолчание {value} вне {bounds[0]}…{bounds[3]}")
        self.assertFalse(bad, "\n".join(bad))


class TestSettingsActuallyApply(unittest.TestCase):
    """
    Правка настройки должна доходить до работающих модулей.

    Проверка выглядит мелкой, а закрывает самую обидную поломку: админка
    сохраняет значение, пишет «перечитаны модули», а модули продолжают
    работать со старым. Виноват был `setdefault` в загрузчике `.env` —
    один раз попав в окружение, значение больше не обновлялось никогда.
    Со стороны это выглядит как «настройка ни на что не влияет».
    """

    def setUp(self):
        self.env = ROOT / ".env"
        self.backup = self.env.read_text(encoding="utf-8") if self.env.exists() else None

    def tearDown(self):
        if self.backup is None:
            self.env.unlink(missing_ok=True)
        else:
            self.env.write_text(self.backup, encoding="utf-8")
        import importlib

        import config
        importlib.reload(config)

    def test_change_in_env_reaches_the_module(self):
        import importlib

        import config
        self.env.write_text("SEARCH_TOP_K=6\n", encoding="utf-8")
        importlib.reload(config)
        self.assertEqual(config.SEARCH_TOP_K, 6)
        self.env.write_text("SEARCH_TOP_K=11\n", encoding="utf-8")
        importlib.reload(config)
        self.assertEqual(config.SEARCH_TOP_K, 11,
                         "правка .env не дошла до модуля — настройка выглядит "
                         "не работающей")

    def test_outside_environment_wins_over_the_file(self):
        """
        Обратная сторона: значение, переданное контейнером или службой,
        файл перебивать не должен — иначе .env из образа уведёт систему
        не туда.
        """
        import importlib
        import os

        import config
        os.environ["SEARCH_TOP_K"] = "7"
        os.environ.pop("_KB_KEYS_FROM_DOTENV", None)
        try:
            self.env.write_text("SEARCH_TOP_K=3\n", encoding="utf-8")
            importlib.reload(config)
            self.assertEqual(config.SEARCH_TOP_K, 7)
        finally:
            os.environ.pop("SEARCH_TOP_K", None)
            os.environ.pop("_KB_KEYS_FROM_DOTENV", None)


class TestSecretsNeverLeak(unittest.TestCase):
    """
    Ключи не должны ни приходить в браузер, ни попадать в `.env`.

    Причина у двух правил одна и та же и вполне земная: `.env` входит в
    архив обновления и в резервную копию, а страница админки — в кэш
    браузера, отладчик и «сохранить страницу». Файл ключей с правами 600
    не входит никуда.
    """

    def test_secret_values_are_not_sent_to_the_browser(self):
        import config
        import security
        import settings_schema
        keys = [s["key"] for s in settings_schema.SETTINGS if s["type"] == "secret"]
        self.assertTrue(keys)
        for key in keys:
            setattr(config, key, "очень-секретное-значение")
        try:
            payload = settings_schema.as_json()
            for item in payload:
                if item["type"] == "secret":
                    self.assertEqual(item["value"], "", item["key"])
                    self.assertTrue(item["filled"], item["key"])
            self.assertNotIn("очень-секретное-значение",
                             str(payload))
        finally:
            for key in keys:
                setattr(config, key, "")
        # И тот же список ключей должен знать модуль безопасности:
        # иначе проверка «ключи лежат в .env открытым текстом» их не увидит.
        self.assertFalse(set(keys) - set(security.SECRET_KEYS),
                         "ключ описан как секрет, но не числится в SECRET_KEYS")

    def test_secret_keys_and_schema_agree(self):
        import security
        import settings_schema
        described = {s["key"] for s in settings_schema.SETTINGS}
        self.assertFalse(set(security.SECRET_KEYS) - described,
                         "ключ есть в SECRET_KEYS, но его нет в панели настроек")


if __name__ == "__main__":
    unittest.main()


class TestQuickStart(unittest.TestCase):
    """
    Быстрый старт повторяет то, что печатает установщик. Если списки
    разойдутся, человек, начавший по бумажке, увидит в интерфейсе другое —
    а это ровно тот случай, когда перестают доверять обоим.
    """

    def test_every_step_has_command_and_explanation(self):
        import quickstart
        bad = []
        for key, title, fn in quickstart.STEPS:
            step = fn()
            if not title:
                bad.append(f"{key}: нет заголовка")
            if not step.get("command"):
                bad.append(f"{key}: нет команды для терминала")
            if step.get("detail") is None:
                bad.append(f"{key}: нечего показать о состоянии")
        self.assertFalse(bad, "\n".join(bad))

    def test_settings_of_steps_exist_in_the_panel(self):
        import quickstart
        import settings_schema
        known = {s["key"] for s in settings_schema.SETTINGS}
        missing = []
        for key, _title, fn in quickstart.STEPS:
            for name in fn().get("settings", []):
                if name not in known:
                    missing.append(f"{key} → {name}")
        self.assertFalse(missing,
                         "шаг ссылается на настройку, которой нет в панели: "
                         + ", ".join(missing))

    def test_actions_are_real_job_kinds(self):
        """Кнопка должна ставить задачу, которая существует."""
        import handlers  # noqa: F401 — регистрация обработчиков
        import jobs
        import quickstart
        bad = []
        for key, _title, fn in quickstart.STEPS:
            action = fn().get("action") or {}
            for field in ("kind", "then"):
                kind = action.get(field)
                if kind and kind not in jobs.HANDLERS:
                    bad.append(f"{key}.{field} → {kind}")
        self.assertFalse(bad, "нет такого вида задачи: " + ", ".join(bad))

    def test_installer_and_panel_agree(self):
        """
        Каждая команда из вывода установщика должна встречаться в шагах, и
        наоборот. Именно эта пара и расходится со временем.
        """
        import re

        import quickstart
        script = (ROOT / "install" / "install.sh").read_text(encoding="utf-8")
        printed = set(re.findall(r"\$TARGET/([a-z_]+\.py)", script))
        printed -= {"webui.py"}          # запускается службой, в шагах отдельно
        in_steps = set()
        for _key, _title, fn in quickstart.STEPS:
            in_steps.update(re.findall(r"([a-z_]+\.py)", fn().get("command", "")))
        forgotten = printed - in_steps
        self.assertFalse(forgotten,
                         "установщик советует, а в быстром старте нет: "
                         + ", ".join(sorted(forgotten)))
