"""
Проверки разбора текста: основа слова, нарезка на фрагменты, цены, даты.

Это те места, где ошибка тихая и дорогая. Слово, разобранное не так,
не ломает ничего заметного — просто документ перестаёт находиться,
и понять это можно только через жалобы сотрудников.
"""
from __future__ import annotations

import unittest

from tests.base import Isolated  # noqa: F401  (важен путь к проекту)

import lsa
import chunk as chunker
import ocr
import rerank


class TestStemmer(unittest.TestCase):
    """Основа русского слова: все формы должны сходиться к одной."""

    GROUPS = [
        ["скважина", "скважины", "скважине", "скважинный", "скважинного", "скважинных"],
        ["насос", "насоса", "насосу", "насосом", "насосы", "насосов", "насосами"],
        ["устанавливать", "устанавливает", "устанавливается"],
        ["уплотнение", "уплотнения", "уплотнений"],
        ["работает", "работать", "работы", "работа", "работой"],
        ["декларация", "декларации", "декларацией"],
        ["давление", "давления", "давлением"],
        ["клапан", "клапана", "клапаны", "клапанов", "клапаном"],
        ["температура", "температуры", "температуре"],
        ["фильтр", "фильтра", "фильтры", "фильтров"],
        ["двигатель", "двигателя", "двигателем", "двигатели"],
        ["кабель", "кабеля", "кабелем", "кабели"],
    ]

    def test_forms_collapse(self):
        for group in self.GROUPS:
            stems = {lsa.normalize_token(w) for w in group}
            self.assertEqual(len(stems), 1,
                             f"формы разошлись: {group} → {stems}")

    def test_articles_untouched(self):
        """Артикулы и обозначения моделей менять нельзя ни при каких условиях."""
        for code in ("500095.f", "2eco6-38", "gru-45", "wrp-a", "м3", "ip68"):
            self.assertEqual(lsa.normalize_token(code), code)

    def test_distinct_words_stay_distinct(self):
        pairs = [("насос", "напор"), ("подача", "давление"),
                 ("фильтр", "фланец"), ("клапан", "колесо")]
        for a, b in pairs:
            self.assertNotEqual(lsa.normalize_token(a), lsa.normalize_token(b),
                                f"разные слова слиплись: {a} и {b}")

    def test_short_and_empty(self):
        for word in ("", "и", "на", "abc"):
            lsa.normalize_token(word)      # не должно падать


class TestChunking(unittest.TestCase):
    def test_pieces_within_bounds(self):
        text = ("Руководство по монтажу. " * 400)
        chunks = chunker.chunk_document([text], {"brand": "TEST", "doc_type": "РУКОВОДСТВО",
                                                 "file_name": "т.txt", "section": "РОЗНИЦА"})
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c.text), chunker.config.CHUNK_TARGET_CHARS * 2)
            self.assertGreaterEqual(len(c.text), chunker.config.CHUNK_MIN_CHARS)

    def test_context_prefix_present(self):
        chunks = chunker.chunk_document(
            ["Гарантийный срок 24 месяца. " * 30],
            {"brand": "ДЖИЛЕКС", "doc_type": "ПАСПОРТ", "file_name": "п.txt",
             "section": "РОЗНИЦА"})
        self.assertTrue(chunks)
        self.assertIn("ДЖИЛЕКС", chunks[0].indexed_text)

    def test_empty_input(self):
        self.assertEqual(chunker.chunk_document([], {}), [])
        self.assertEqual(chunker.chunk_document(["  "], {}), [])


class TestCyrillicGuard(unittest.TestCase):
    """
    Защита от подмены кириллицы латиницей.

    Самая важная проверка во всём наборе: ошибка здесь означает, что
    сертификаты навсегда перестают находиться по русским словам, и
    заметить это по внешнему виду текста невозможно.
    """

    def test_mixed_word_repaired(self):
        fixed, stats = ocr.repair_homoglyphs("Город МОCКВА")   # C — латинская
        self.assertEqual(fixed, "Город МОСКВА")
        self.assertEqual(stats["fixed_mixed"], 1)

    def test_real_latin_untouched(self):
        source = "Насос Grundfos SP-45 произведён в Дании, класс IP68"
        fixed, _ = ocr.repair_homoglyphs(source)
        self.assertEqual(fixed, source)

    def test_full_latin_flagged_without_vocab(self):
        ocr._vocab_cache = set()          # словаря базы нет
        _fixed, stats = ocr.repair_homoglyphs("MOCKBA")
        self.assertEqual(stats["suspicious"], 1)

    def test_full_latin_repaired_with_vocab(self):
        ocr._vocab_cache = {"москв", "насос"}
        fixed, stats = ocr.repair_homoglyphs("MOCKBA")
        self.assertEqual(fixed, "МОСКВА")
        self.assertEqual(stats["fixed_latin"], 1)

    def test_mixed_words_are_fully_repairable(self):
        """
        Слово, где к кириллице примешана латиница, чинится целиком — и
        такой странице можно доверять: направление подмены однозначно.
        """
        ocr._vocab_cache = set()
        dirty = "CEPTИФИKAT COOTBETCTBИЯ BЫДAH OPГAHOM CEPTИФИKAЦИИ"
        fixed, stats = ocr.repair_homoglyphs(dirty)
        self.assertEqual(fixed, "СЕРТИФИКАТ СООТВЕТСТВИЯ ВЫДАН ОРГАНОМ СЕРТИФИКАЦИИ")
        self.assertEqual(stats["suspicious"], 0)

    def test_quality_drops_when_repair_impossible(self):
        """
        А вот слова целиком из латинских двойников починить нельзя без
        словаря: они остаются подозрительными, и доверие к странице падает.
        Именно такую страницу нужно переотправить другому распознавателю.
        """
        ocr._vocab_cache = set()
        clean = ("Сертификат соответствия выдан органом по сертификации "
                 "в городе Москва, срок действия до 2027 года")
        dirty = ("PACXOДOMEP CEKЦИOHHЫЙ MAPKИPOBKA XPOMATOГPAФ PEOCTAT "
                 "KOMMУTATOP TPAHCФOPMATOP")
        _, s1 = ocr.repair_homoglyphs(clean)
        _, s2 = ocr.repair_homoglyphs(dirty)
        self.assertEqual(s1["suspicious"], 0)
        self.assertGreater(s2["ratio"], ocr.config.OCR_MAX_LATIN_RATIO)
        self.assertGreater(ocr.quality(clean, s1), ocr.quality(dirty, s2))

    def tearDown(self):
        ocr._vocab_cache = None


class TestRerankBlend(unittest.TestCase):
    """
    Смешивание оценок поиска и переранжирования.

    Ключевое свойство: когда реранкер не видит разницы между кандидатами,
    порядок должен определять поиск. Нарушение этого правила однажды уже
    ухудшало выдачу — растянутый шум перемешивал результаты.
    """

    def setUp(self):
        import config
        self.saved = config.RERANKER_WEIGHT
        config.RERANKER_WEIGHT = 0.8

    def tearDown(self):
        import config
        config.RERANKER_WEIGHT = self.saved

    def test_keeps_scale(self):
        base = [0.05, 0.04, 0.03]
        out = rerank.blend(base, [0.9, 0.5, 0.1])
        self.assertAlmostEqual(max(out), max(base), places=6)

    def test_flat_reranker_keeps_search_order(self):
        base = [0.05, 0.04, 0.03]
        out = rerank.blend(base, [0.5, 0.5, 0.5])
        self.assertEqual(out, sorted(out, reverse=True))

    def test_reranker_can_reorder(self):
        base = [0.05, 0.04, 0.03]
        out = rerank.blend(base, [0.0, 0.1, 0.99])
        self.assertGreater(out[2], out[0])

    def test_empty(self):
        self.assertEqual(rerank.blend([], []), [])


class TestLexicalReranker(unittest.TestCase):
    def test_article_match_wins(self):
        engine = rerank.LexicalReranker()
        scores = engine.score_pairs("цена артикула 500036", [
            "Артикул 500036 — насос скважинный GRU-45 — 25 900 рублей за штуку.",
            "Артикул 500029 — насос скважинный GRU-40 — 21 600 рублей за штуку.",
            "Цены указаны с учётом налога, отгрузка со склада в Москве.",
        ])
        self.assertEqual(max(range(3), key=lambda i: scores[i]), 0)

    def test_empty_query(self):
        engine = rerank.LexicalReranker()
        self.assertEqual(engine.score_pairs("", ["что-нибудь"]), [0.0])


if __name__ == "__main__":
    unittest.main()
