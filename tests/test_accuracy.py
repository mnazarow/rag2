"""
Проверки против подмен ответа.

Каждый тест здесь — это реальный найденный дефект, при котором система
УВЕРЕННО отвечала не о том: выверенный ответ уходил на вопрос про
соседнюю модель, капс-страницы каталогов молча исчезали из индекса,
порог честного отказа был математически недостижим, а прайсовый канал
подкладывал цену другого товара под грифом «точные данные».
"""
from __future__ import annotations

import unittest

from tests.base import Isolated

import answer as answer_mod
import chunk as chunker
import normtext


class TestGoldenMatching(unittest.TestCase):
    """Выверенный ответ не должен уходить на вопрос про другую модель."""

    def setUp(self):
        answer_mod._brand_cache = {"джилекс", "вило", "grundfos"}

    def tearDown(self):
        answer_mod._brand_cache = None

    def test_neighbor_model_rejected(self):
        self.assertEqual(0.0, answer_mod._golden_similarity(
            "какой напор у насоса Водомет 60/92",
            "какой напор у насоса Водомет 55/75"))

    def test_other_brand_rejected(self):
        self.assertEqual(0.0, answer_mod._golden_similarity(
            "какая гарантия на насосы джилекс",
            "какая гарантия на насосы вило"))

    def test_same_question_passes(self):
        self.assertTrue(answer_mod._golden_is_close(
            "какой максимальный напор у насоса водомет 55/75",
            "какой напор у насоса водомет 55/75"))

    def test_decimal_comma_vs_dot_same_signature(self):
        self.assertGreater(answer_mod._golden_similarity(
            "характеристики насоса БЦПЭ 0,5-40 джилекс",
            "характеристики насоса БЦПЭ 0.5-40 джилекс"), 0.7)

    def test_number_only_in_golden_rejected(self):
        """Эталон конкретнее вопроса — отдавать нельзя."""
        self.assertEqual(0.0, answer_mod._golden_similarity(
            "какая гарантия на водомет",
            "гарантия 3 года на водомет"))


class TestCapsPages(unittest.TestCase):
    """Капс-вёрстка каталога не должна исчезать из индекса."""

    def test_caps_page_indexed(self):
        caps = ("КАТАЛОГ ПРОДУКЦИИ 2025\nНАСОСЫ СКВАЖИННЫЕ\n"
                "ДЖИЛЕКС ВОДОМЕТ 55/75\nНАПОР МАКС 75 М\nПОДАЧА 3.6 М3/Ч")
        chunks = chunker.chunk_document([caps], {"file_name": "к.pdf"})
        self.assertTrue(chunks)
        joined = " ".join(c.text for c in chunks)
        self.assertIn("НАПОР МАКС 75", joined)

    def test_normal_page_does_not_hide_caps_page(self):
        normal = "Руководство по монтажу скважинного насоса. " * 10
        caps = "ДЖИЛЕКС ВОДОМЕТ 55/75\nНАПОР МАКС 75 М\nПОДАЧА 3.6 М3/Ч"
        chunks = chunker.chunk_document([normal, caps], {"file_name": "к.pdf"})
        joined = " ".join(c.text for c in chunks)
        self.assertIn("НАПОР МАКС 75", joined)


class TestTableChunking(unittest.TestCase):
    """Таблицы: шапка в каждом куске, строки не рвутся посередине."""

    def _table(self, n):
        rows = ["Модель | Напор макс., м | Подача, м3/ч"]
        rows += [f"Водомет {50 + i}/{60 + i} | {60 + i} | 3.{i % 9}" for i in range(n)]
        return "\n".join(rows)

    def test_header_repeated(self):
        chunks = chunker.chunk_document(
            ["ХАРАКТЕРИСТИКИ\n" + self._table(120)], {"file_name": "т.pdf"})
        table_chunks = [c for c in chunks if "|" in c.text]
        self.assertGreater(len(table_chunks), 1)
        for c in table_chunks:
            self.assertTrue(c.text.splitlines()[0].startswith("Модель |"),
                            "шапка потеряна: " + c.text[:60])

    def test_rows_not_torn(self):
        chunks = chunker.chunk_document(
            ["ХАРАКТЕРИСТИКИ\n" + self._table(120)], {"file_name": "т.pdf"})
        for c in chunks:
            for line in c.text.splitlines():
                if "|" in line and not line.startswith("Модель"):
                    self.assertRegex(line, r"^Водомет \d+/\d+ \|",
                                     "строка начата с середины: " + line[:40])

    def test_two_column_passport_table_kept_together(self):
        text = ("ХАРАКТЕРИСТИКИ ИЗДЕЛИЯ\nНапор максимальный, м | 75\n"
                "Подача номинальная, м3/ч | 3.6\nМощность, Вт | 1150\n"
                "Масса, кг | 12")
        chunks = chunker.chunk_document([text], {"file_name": "п.pdf"})
        self.assertTrue(chunks)
        self.assertIn("Напор максимальный", chunks[0].text)
        self.assertIn("Масса", chunks[0].text)

    def test_model_cards_not_glued(self):
        text = ("ОПИСАНИЕ МОДЕЛЕЙ\n"
                "Водомет 55/75 обеспечивает напор до 75 метров и подачу 3.6 "
                "кубометра в час, вес 12 кг, кабель 30 метров в комплекте.\n"
                "Водомет 60/92 обеспечивает напор до 92 метров и подачу 3.0 "
                "кубометра в час, вес 14 кг, кабель 40 метров в комплекте.")
        chunks = chunker.chunk_document([text], {"file_name": "к.pdf"})
        for c in chunks:
            self.assertFalse("55/75" in c.text and "60/92" in c.text,
                             "модели склеены в один фрагмент")


class TestConfidence(Isolated):
    """Порог отказа достижим и различает вопросы."""

    def setUp(self):
        import os
        os.environ["ROLE_SECTIONS"] = ""
        super().setUp()

    def _index(self, text):
        self.db.run(
            "INSERT INTO documents (rel_path, abs_path, file_name, ext, "
            "content_hash, section, status) VALUES ('р/п.pdf','/п','п.pdf','.pdf','h','р','ok')")
        did = self.db.q1("SELECT id FROM documents")["id"]
        self.db.run("INSERT INTO chunks (doc_id, ord, text, n_chars) VALUES (?,0,?,?)",
                    (did, text, len(text)))
        cid = self.db.q1("SELECT id FROM chunks")["id"]
        self.db.run("INSERT INTO chunks_fts (rowid, text) VALUES "
                    "(?, (SELECT text FROM chunks WHERE id=?))", (cid, cid))

    def test_relevant_passes_garbage_refused(self):
        import importlib

        import search
        importlib.reload(search)
        self._index("Водомет 55/75: напор максимальный 75 м, подача 3.6 м3/ч")
        good = search.confidence(search.search("какой напор у насоса водомет"))
        bad = search.confidence(search.search("расписание электричек на завтра"))
        self.assertGreater(good, self.config.MIN_CONFIDENCE)
        self.assertLess(bad, self.config.MIN_CONFIDENCE)


class TestNormtext(unittest.TestCase):
    """Индекс и запрос нормализуются одинаково."""

    def test_yo_folded(self):
        self.assertEqual(normtext.canon("Водомёт"), "Водомет")

    def test_units_unified(self):
        for raw in ("3,6 м³/ч", "3.6 м3/час", "3.6 куб.м/час", "3.6 кубометра в час"):
            self.assertIn("м3/ч", normtext.canon(raw), raw)

    def test_decimal_comma(self):
        self.assertEqual(normtext.canon("подача 3,6"), "подача 3.6")

    def test_articles_untouched(self):
        for art in ("500095.F", "2ECO6-38", "WRP-A", "ip68"):
            self.assertEqual(normtext.canon(art), art)


class TestPriceChannel(Isolated):
    """Прайс не подкладывает чужую модель и не считает год артикулом."""

    def setUp(self):
        import os
        os.environ["ROLE_SECTIONS"] = ""
        super().setUp()
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

    def test_model_signature_required(self):
        import importlib

        import prices
        importlib.reload(prices)
        rows = prices.search_products("сколько стоит водомет 55/75")
        self.assertTrue(rows)
        self.assertTrue(all("55/75" in r["name"] for r in rows),
                        [r["name"] for r in rows])

    def test_year_is_not_article(self):
        import prices
        self.assertEqual(prices.BARE_ARTICLE_RX.findall("цена в прайсе 2024 года"), [])
        self.assertEqual(prices.BARE_ARTICLE_RX.findall("цена 500036"), ["500036"])


class TestPriceAsSource(unittest.TestCase):
    """Блок прайса нумеруется как источник, документы идут следом."""

    def test_numbering(self):
        rows = [{"name": "Насос", "article": "1", "price": 100.0,
                 "rel_path": "п/прайс.xlsx", "file_name": "прайс.xlsx",
                 "price_date": "2026-01-01"}]
        block, sources, next_n = answer_mod.price_context(rows, start=1)
        self.assertTrue(block.startswith("[1] прайс-лист «прайс.xlsx», от 2026-01-01"))
        self.assertEqual(next_n, 2)
        self.assertEqual(sources[0]["kind"], "price")


if __name__ == "__main__":
    unittest.main()
