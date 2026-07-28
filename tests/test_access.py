"""
Разграничение доступа: что именно видит сотрудник с каждой ролью.

Каждая проверка здесь соответствует конкретному способу получить чужие
данные, а не абстрактной «безопасности». Все они найдены аудитом кода и
все до исправления работали.
"""
from __future__ import annotations

import unittest

from tests.base import Isolated

DOCS = [
    # раздел, бренд, файл, текст, артикул, цена
    ("РОЗНИЧНАЯ ПРОДУКЦИЯ", "ДЖИЛЕКС", "Прайс розница.txt", "Насос Водомет", "ВД-55", 15000),
    ("ДИЛЕРСКАЯ ПРОДУКЦИЯ", "ДЖИЛЕКС", "Прайс дилер.txt", "Насос Водомет", "ВД-55", 9000),
]


class AccessBase(Isolated):
    def setUp(self):
        super().setUp()
        self.config.ROLE_SECTIONS = {
            "sales": ["РОЗНИЧНАЯ ПРОДУКЦИЯ"],
            "dealer": ["РОЗНИЧНАЯ ПРОДУКЦИЯ", "ДИЛЕРСКАЯ ПРОДУКЦИЯ"],
            "admin": ["*"],
        }
        self.config.DEFAULT_ROLE = "sales"
        self.doc_ids = {}
        for section, brand, name, text, article, price in DOCS:
            cur = self.db.run(
                "INSERT INTO documents(rel_path, abs_path, file_name, ext, section, "
                "brand, content_hash, status, is_current) "
                "VALUES (?,?,?,?,?,?,?,'ok',1)",
                (f"{section}/{brand}/{name}", f"/tmp/{name}", name, ".txt",
                 section, brand, name))
            doc_id = int(cur.lastrowid)
            self.doc_ids[section] = doc_id
            chunk = self.db.run(
                "INSERT INTO chunks(doc_id, ord, text, n_chars) VALUES (?,?,?,?)",
                (doc_id, 0, text, len(text)))
            self.db.run("INSERT INTO chunks_fts(rowid, text, heading, context, "
                        "brand, doc_type, file_name) VALUES (?,?,?,?,?,?,?)",
                        (int(chunk.lastrowid), text, "", "", brand, "", name))
            prod = self.db.run(
                "INSERT INTO products(doc_id, brand, article, name, price, is_current) "
                "VALUES (?,?,?,?,?,1)", (doc_id, brand, article, text, price))
            self.db.run("INSERT INTO products_fts(rowid, article, name, brand) "
                        "VALUES (?,?,?,?)", (int(prod.lastrowid), article, text, brand))


class TestPriceChannel(AccessBase):
    """
    Цены — самое закрытое, что есть в базе, и раньше этот канал роль не
    смотрел вовсе. Достаточно было спросить «сколько стоит ВД-55».
    """

    def test_retail_role_does_not_see_dealer_price(self):
        import prices
        rows = prices.search_products("ВД-55", role="sales")
        self.assertTrue(rows, "розничная цена должна находиться")
        self.assertTrue(all(r["section"] == "РОЗНИЧНАЯ ПРОДУКЦИЯ" for r in rows), rows)
        self.assertTrue(all(r["price"] == 15000 for r in rows), rows)

    def test_dealer_sees_both(self):
        import prices
        rows = prices.search_products("ВД-55", role="dealer")
        self.assertEqual({r["section"] for r in rows},
                         {"РОЗНИЧНАЯ ПРОДУКЦИЯ", "ДИЛЕРСКАЯ ПРОДУКЦИЯ"})

    def test_unknown_role_sees_nothing(self):
        import prices
        self.assertEqual(prices.search_products("ВД-55", role="Sales"), [])


class TestSectionFilterFailsClosed(AccessBase):
    """
    Роль, которой нет в списке, раньше получала доступ ко всему. Хватало
    опечатки в одной букве.
    """

    def test_unknown_role_gets_nothing(self):
        import search
        self.assertEqual(search.allowed_sections("снабженец"), set())

    def test_known_role_gets_its_sections(self):
        import search
        self.assertEqual(search.allowed_sections("sales"), {"РОЗНИЧНАЯ ПРОДУКЦИЯ"})

    def test_star_means_everything(self):
        import search
        self.assertIsNone(search.allowed_sections("admin"))

    def test_no_role_config_means_no_restriction(self):
        """Разграничение не настроено — ограничивать нечем, и это нормально."""
        import search
        self.config.ROLE_SECTIONS = {}
        self.assertIsNone(search.allowed_sections("кто угодно"))

    def test_search_returns_nothing_for_unknown_role(self):
        import search
        self.assertEqual(search.search("Водомет", role="Sales"), [])


class TestLeakCheckSeesPrices(AccessBase):
    def test_price_from_closed_section_counts_as_leak(self):
        import security
        leak = security.check_answer_leak(
            "цена 9000", [], {"РОЗНИЧНАЯ ПРОДУКЦИЯ"},
            products=[{"section": "ДИЛЕРСКАЯ ПРОДУКЦИЯ", "price": 9000}])
        self.assertTrue(leak["leak"])
        self.assertEqual(leak["sections"], ["ДИЛЕРСКАЯ ПРОДУКЦИЯ"])

    def test_allowed_price_is_not_a_leak(self):
        import security
        leak = security.check_answer_leak(
            "цена 15000", [], {"РОЗНИЧНАЯ ПРОДУКЦИЯ"},
            products=[{"section": "РОЗНИЧНАЯ ПРОДУКЦИЯ", "price": 15000}])
        self.assertFalse(leak["leak"])


class TestGoldenAnswersRespectRole(AccessBase):
    def test_golden_scoped_to_closed_section_is_hidden(self):
        import answer as answer_mod
        import search
        chunk = self.db.q1("SELECT c.id FROM chunks c JOIN documents d ON d.id=c.doc_id "
                           "WHERE d.section='ДИЛЕРСКАЯ ПРОДУКЦИЯ'")
        answer_mod.add_golden("какая скидка дилеру на Водомет",
                              "Скидка 40 процентов.",
                              source_refs=[{"chunk_id": chunk["id"]}])
        self.assertEqual(search.golden_search("скидка дилеру Водомет", role="sales"), [])
        self.assertTrue(search.golden_search("скидка дилеру Водомет", role="dealer"))

    def test_general_golden_is_visible_to_everyone(self):
        import answer as answer_mod
        import search
        answer_mod.add_golden("как оформить возврат", "По заявлению в течение 14 дней.")
        self.assertTrue(search.golden_search("как оформить возврат", role="sales"))


class TestBotButtons(AccessBase):
    """
    Нажатие кнопки в Telegram — отдельный запрос с произвольным
    содержимым, который отправляется любым клиентом. Раньше он не
    проверялся вовсе: `doc:1`, `doc:2`, … выгружали всю базу, и это
    работало даже у заблокированного сотрудника.
    """

    def setUp(self):
        super().setUp()
        import bot
        self.bot = bot
        self.config.TELEGRAM_ADMIN_IDS = []
        self.config.TELEGRAM_ALLOWED_IDS = []
        self.db.run("INSERT INTO users(user_id, role, approved, status) "
                    "VALUES (100, 'sales', 1, 'approved')")
        self.db.run("INSERT INTO users(user_id, role, approved, status) "
                    "VALUES (200, 'sales', 0, 'blocked')")

    def test_blocked_employee_gets_nothing(self):
        res = self.bot.handle_callback(200, f"doc:{self.doc_ids['РОЗНИЧНАЯ ПРОДУКЦИЯ']}")
        self.assertNotIn("send_document", res)

    def test_closed_section_file_is_not_sent(self):
        res = self.bot.handle_callback(100, f"doc:{self.doc_ids['ДИЛЕРСКАЯ ПРОДУКЦИЯ']}")
        self.assertNotIn("send_document", res)
        # Ответ такой же, как для несуществующего документа: иначе перебор
        # номеров показывает, что в закрытом разделе что-то есть.
        self.assertEqual(res["alert"],
                         self.bot.handle_callback(100, "doc:999999")["alert"])

    def test_own_section_file_is_sent(self):
        res = self.bot.handle_callback(100, f"doc:{self.doc_ids['РОЗНИЧНАЯ ПРОДУКЦИЯ']}")
        self.assertIn("send_document", res)

    def test_feedback_only_from_the_author(self):
        self.db.run("INSERT INTO queries(id, user_id, question, created_at, answered) "
                    "VALUES (7, 100, 'вопрос', '2026-01-01T00:00:00+00:00', 1)")
        stranger = self.bot.handle_callback(100, "fb:down:7")
        self.assertIn("notify_experts", stranger)
        self.db.run("UPDATE queries SET user_id=999 WHERE id=7")
        res = self.bot.handle_callback(100, "fb:down:7")
        self.assertNotIn("notify_experts", res)

    def test_broken_payload_does_not_crash(self):
        for data in ("doc:abc", "fb:up:xx", "doc:", "", "мусор"):
            self.bot.handle_callback(100, data)


class TestJobPermissions(unittest.TestCase):
    """
    Очередь заданий — второй вход к тем же действиям. Восстановление
    индекса закрыто на своём эндпоинте, но точно так же запускалось
    через POST /api/job.
    """

    def setUp(self):
        import security
        self.security = security

    def test_operator_cannot_restore_through_the_queue(self):
        self.assertFalse(self.security.may("operator", "POST", "/api/job",
                                           {"kind": "restore"}))
        self.assertFalse(self.security.may("operator", "POST", "/api/job",
                                           {"kind": "backup_prune"}))

    def test_operator_can_still_reindex(self):
        self.assertTrue(self.security.may("operator", "POST", "/api/job",
                                          {"kind": "reindex"}))

    def test_admin_can_do_everything(self):
        self.assertTrue(self.security.may("admin", "POST", "/api/job",
                                          {"kind": "restore"}))

    def test_viewer_cannot_run_jobs(self):
        self.assertFalse(self.security.may("viewer", "POST", "/api/job",
                                           {"kind": "reindex"}))


class TestLoginBruteForce(Isolated):
    def setUp(self):
        super().setUp()
        import importlib

        import security
        importlib.reload(security)
        self.security = security
        self.config.ADMIN_LOGIN_MAX_FAILS = 3
        self.config.ADMIN_LOGIN_BLOCK_MINUTES = 15

    def test_blocks_after_the_limit(self):
        self.assertTrue(self.security.login_attempt_allowed("petrov", "10.0.0.1")["ok"])
        for _ in range(3):
            self.security.login_failed("petrov", "10.0.0.1")
        blocked = self.security.login_attempt_allowed("petrov", "10.0.0.1")
        self.assertFalse(blocked["ok"])
        self.assertIn("попыток", blocked["message"])

    def test_changing_login_does_not_help(self):
        """Перебор логинов с одного адреса упирается в тот же счётчик."""
        for i in range(3):
            self.security.login_failed(f"user{i}", "10.0.0.2")
        self.assertFalse(self.security.login_attempt_allowed("user9", "10.0.0.2")["ok"])

    def test_success_clears_the_counter(self):
        self.security.login_failed("petrov", "10.0.0.3")
        self.security.login_succeeded("petrov", "10.0.0.3")
        self.assertTrue(self.security.login_attempt_allowed("petrov", "10.0.0.3")["ok"])

    def test_other_user_unaffected(self):
        for _ in range(3):
            self.security.login_failed("petrov", "10.0.0.4")
        self.assertTrue(self.security.login_attempt_allowed("ivanov", "10.0.0.5")["ok"])


class TestLogMasking(unittest.TestCase):
    """
    Маскировался только текст сообщения, а не подставляемые значения.
    Весь проект пишет `log.error("…: %s", exc)`, поэтому пароль из текста
    исключения попадал в журнал как есть — а журнал читает любая роль.
    """

    def test_secret_in_argument_is_masked(self):
        import logging

        import logging_setup
        record = logging.LogRecord("test", logging.ERROR, __file__, 1,
                                   "не удалось подключиться: %s",
                                   ("ошибка на ws://ari:password=ochen-sekretno@host",),
                                   None)
        logging_setup.ContextFilter().filter(record)
        self.assertNotIn("ochen-sekretno", record.getMessage())

    def test_exception_object_is_masked(self):
        import logging

        import logging_setup
        exc = RuntimeError("api_key=sk-tajnyj-kljuch-12345678")
        record = logging.LogRecord("test", logging.ERROR, __file__, 1,
                                   "сбой: %s", (exc,), None)
        logging_setup.ContextFilter().filter(record)
        self.assertNotIn("sk-tajnyj-kljuch-12345678", record.getMessage())


class TestAuditKeepsSecrets(Isolated):
    def test_key_value_never_reaches_the_audit_log(self):
        """
        Журнал действий лежит в основной базе, а база — первым файлом в
        резервной копии. Ключ, записанный сюда, уезжает в каждый архив.
        """
        import importlib

        import webui
        importlib.reload(webui)
        safe = webui._audit_safe_changes(
            {"OPENAI_API_KEY": "sk-nastojashchij", "SEARCH_TOP_K": "8"},
            {"OPENAI_API_KEY": "sk-staryj", "SEARCH_TOP_K": "6"})
        self.assertNotIn("sk-nastojashchij", str(safe))
        self.assertNotIn("sk-staryj", str(safe))
        self.assertEqual(safe["SEARCH_TOP_K"], ["6", "8"])


if __name__ == "__main__":
    unittest.main()
