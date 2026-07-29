"""
Память диалога: уточняющие вопросы наследуют тему разговора.

Три инварианта. Уточнение без своей модели наследует модель из прошлой
реплики. Вопрос со своей моделью не наследует ничего — иначе «а 60/92?»
получил бы в довесок 55/75 и ценовой канал нашёл бы оба. И вопрос о
новом предмете («сколько стоит кабель») разговор не продолжает: чужая
подпись модели увела бы точный поиск к прошлому товару.
"""
from __future__ import annotations

import unittest
from unittest import mock

from tests.base import Isolated

import dialog


class TestFollowupDetection(unittest.TestCase):
    def setUp(self):
        self._p = mock.patch.object(dialog, "_brands",
                                    return_value={"джилекс", "вило"})
        self._p.start()

    def tearDown(self):
        self._p.stop()

    def test_bare_price_question_is_followup(self):
        for q in ("а цена?", "сколько стоит?", "а почём"):
            self.assertTrue(dialog.is_followup(q), q)

    def test_attribute_question_is_followup(self):
        for q in ("какая глубина погружения?", "а мощность", "какой напор"):
            self.assertTrue(dialog.is_followup(q), q)

    def test_question_with_own_model_is_not(self):
        for q in ("а 60/92?", "цена арт. 500036", "какой напор у водомет 55/75"):
            self.assertFalse(dialog.is_followup(q), q)

    def test_question_with_brand_is_not(self):
        self.assertFalse(dialog.is_followup("а у джилекс какая гарантия"))

    def test_new_subject_is_not(self):
        for q in ("сколько стоит кабель", "как оформить возврат"):
            self.assertFalse(dialog.is_followup(q), q)


class TestInheritance(Isolated):
    def setUp(self):
        super().setUp()
        dialog.reset()
        self._p = mock.patch.object(dialog, "_brands",
                                    return_value={"джилекс"})
        self._p.start()

    def tearDown(self):
        self._p.stop()
        dialog.reset()
        super().tearDown()

    def test_price_followup_inherits_model(self):
        dialog.remember(1, "какой напор у насоса Водомет 55/75 джилекс")
        q, inherited = dialog.augment(1, "а цена?")
        self.assertIn("55/75", q)
        self.assertIn("джилекс", q)
        self.assertIn("55/75", inherited)

    def test_other_chat_not_affected(self):
        dialog.remember(1, "какой напор у Водомет 55/75")
        q, inherited = dialog.augment(2, "а цена?")
        self.assertEqual(q, "а цена?")
        self.assertEqual(inherited, [])

    def test_own_model_wins(self):
        dialog.remember(1, "какой напор у Водомет 55/75")
        q, inherited = dialog.augment(1, "какой напор у Водомет 60/92")
        self.assertNotIn("55/75", q)
        self.assertEqual(inherited, [])

    def test_expired_conversation_forgotten(self):
        import time as _time
        dialog.remember(1, "какой напор у Водомет 55/75")
        with mock.patch.object(dialog.time, "time",
                               return_value=_time.time() + 3600):
            q, inherited = dialog.augment(1, "а цена?")
        self.assertEqual(inherited, [])

    def test_disabled_by_setting(self):
        dialog.remember(1, "какой напор у Водомет 55/75")
        with mock.patch.object(dialog.config, "DIALOG_MEMORY_MINUTES", 0):
            q, inherited = dialog.augment(1, "а цена?")
        self.assertEqual(inherited, [])

    def test_empty_reply_does_not_erase_memory(self):
        dialog.remember(1, "какой напор у Водомет 55/75")
        dialog.remember(1, "спасибо")
        _q, inherited = dialog.augment(1, "а цена?")
        self.assertIn("55/75", inherited)


class TestEndToEnd(Isolated):
    """Сквозной: уточнение «а цена?» находит цену обсуждённой модели."""

    def setUp(self):
        import os
        os.environ["ROLE_SECTIONS"] = ""
        super().setUp()
        dialog.reset()
        import normtext
        self.db.run(
            "INSERT INTO documents (rel_path, abs_path, file_name, ext, "
            "content_hash, section, status) VALUES ('п/прайс.xlsx','/п','прайс.xlsx','.xlsx','h','п','ok')")
        did = self.db.q1("SELECT id FROM documents")["id"]
        for name, art, price in (("Насос Водомёт ПРОФ 55/75", "500036", 18400.0),
                                 ("Насос Водомёт ПРОФ 60/92", "500037", 21900.0)):
            self.db.run("INSERT INTO products (doc_id, name, article, price, is_current) "
                        "VALUES (?,?,?,?,1)", (did, name, art, price))
            pid = self.db.q1("SELECT id FROM products WHERE article=?", (art,))["id"]
            self.db.run("INSERT INTO products_fts (rowid, article, name, brand) "
                        "VALUES (?,?,?,?)", (pid, art, normtext.canon(name), ""))

    def tearDown(self):
        dialog.reset()
        super().tearDown()

    def test_followup_price(self):
        import importlib

        import answer
        import prices
        importlib.reload(prices)
        answer._brand_cache = set()
        dialog.remember(7, "какой напор у насоса водомет 55/75")
        q, inherited = dialog.augment(7, "а цена?")
        self.assertIn("55/75", inherited)
        rows = prices.search_products(q)
        self.assertTrue(rows)
        self.assertTrue(all("55/75" in r["name"] for r in rows),
                        [r["name"] for r in rows])


if __name__ == "__main__":
    unittest.main()
