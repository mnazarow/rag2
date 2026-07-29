"""
Метрика качества не лучше своих эталонов.

Здесь проверяется сама оценка: что подмена документом-двойником
считается подменой, а аудит набора замечает слишком общие шаблоны и
отсутствие пар-двойников. Эти проверки существуют потому, что первый
вариант метрики был конструктивно слеп: «4ПАСПОРТ» засчитывал паспорт
любого бренда, и три реальных дефекта поиска на нём выглядели зелёными.
"""
from __future__ import annotations

import unittest
from unittest import mock

from tests.base import ROOT, Isolated  # noqa: F401  (ROOT нужен для sys.path)

import evaluate


class FakeHit:
    def __init__(self, rel_path: str):
        self.rel_path = rel_path


def _fake_search(paths):
    return lambda q, top_k=6: [FakeHit(p) for p in paths]


class TestSubstitution(unittest.TestCase):
    """Подмена — это когда двойник стоит выше нужного документа."""

    def test_twin_above_expected_counts_as_substituted(self):
        dataset = [{"question": "напор Водомет 60/92",
                    "expect_files": ["60_92"],
                    "reject_files": ["55_75"]}]
        with mock.patch.object(evaluate.search_mod, "search",
                               _fake_search(["ДЖИЛЕКС/Водомет 55_75.pdf",
                                             "ДЖИЛЕКС/Водомет 60_92.pdf"])):
            res = evaluate.evaluate(dataset, top_k=6)
        self.assertEqual(res["substituted"], 1.0)
        self.assertEqual(res["details"][0]["substituted_by"],
                         "ДЖИЛЕКС/Водомет 55_75.pdf")

    def test_expected_above_twin_is_clean(self):
        dataset = [{"question": "напор Водомет 60/92",
                    "expect_files": ["60_92"],
                    "reject_files": ["55_75"]}]
        with mock.patch.object(evaluate.search_mod, "search",
                               _fake_search(["ДЖИЛЕКС/Водомет 60_92.pdf",
                                             "ДЖИЛЕКС/Водомет 55_75.pdf"])):
            res = evaluate.evaluate(dataset, top_k=6)
        self.assertEqual(res["substituted"], 0.0)

    def test_twin_found_when_expected_missing(self):
        """Нужного нет вовсе, двойник есть — это тоже подмена."""
        dataset = [{"question": "напор Водомет 60/92",
                    "expect_files": ["60_92"],
                    "reject_files": ["55_75"]}]
        with mock.patch.object(evaluate.search_mod, "search",
                               _fake_search(["ДЖИЛЕКС/Водомет 55_75.pdf"])):
            res = evaluate.evaluate(dataset, top_k=6)
        self.assertEqual(res["substituted"], 1.0)

    def test_no_rejects_means_metric_absent(self):
        dataset = [{"question": "в", "expect_files": ["x"]}]
        with mock.patch.object(evaluate.search_mod, "search", _fake_search([])):
            res = evaluate.evaluate(dataset, top_k=6)
        self.assertIsNone(res["substituted"])


class TestAudit(Isolated):
    """Аудит набора должен замечать то, из-за чего метрика врёт."""

    def _index_docs(self, paths):
        for i, p in enumerate(paths):
            self.db.run(
                "INSERT INTO documents (rel_path, abs_path, file_name, ext, "
                "content_hash, section, status) VALUES (?, ?, ?, ?, ?, ?, 'indexed')",
                (p, "/kb/" + p, p.rsplit("/", 1)[-1], ".pdf", f"hash{i}",
                 p.split("/", 1)[0]))

    def test_too_generic_pattern_reported(self):
        self._index_docs([f"BRAND{i}/4ПАСПОРТ/doc{i}.pdf" for i in range(30)])
        problems = evaluate.audit(
            [{"question": "вопрос", "expect_files": ["4ПАСПОРТ"],
              "expect_text": ["75"], "reject_text": ["92"]}])
        self.assertTrue(any("слишком общий" in m for m in problems), problems)

    def test_dead_pattern_reported(self):
        self._index_docs(["ДЖИЛЕКС/4ПАСПОРТ/vodomet.pdf"])
        problems = evaluate.audit(
            [{"question": "вопрос", "expect_files": ["НЕСУЩЕСТВУЮЩЕЕ"],
              "expect_text": ["75"], "reject_text": ["92"]}])
        self.assertTrue(any("всегда промах" in m for m in problems), problems)

    def test_missing_twins_and_text_reported(self):
        self._index_docs(["ДЖИЛЕКС/4ПАСПОРТ/vodomet.pdf"])
        problems = evaluate.audit(
            [{"question": "вопрос", "expect_files": ["vodomet"]}])
        self.assertTrue(any("пары-двойника" in m for m in problems), problems)
        self.assertTrue(any("expect_text" in m for m in problems), problems)

    def test_good_dataset_passes(self):
        self._index_docs(["ДЖИЛЕКС/4ПАСПОРТ/vodomet 60_92.pdf",
                          "ДЖИЛЕКС/4ПАСПОРТ/vodomet 55_75.pdf"])
        problems = evaluate.audit(
            [{"question": "напор водомет 60/92",
              "expect_files": ["vodomet 60_92"], "expect_text": ["92"],
              "reject_files": ["vodomet 55_75"], "reject_text": ["75 м"]},
             {"question": "напор водомет 55/75",
              "expect_files": ["vodomet 55_75"], "expect_text": ["75"],
              "reject_files": ["vodomet 60_92"], "reject_text": ["92 м"]}])
        self.assertEqual(problems, [], problems)


if __name__ == "__main__":
    unittest.main()
