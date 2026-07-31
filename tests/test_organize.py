"""
Порядок в базе: находки честные, план уборки безопасный.

Главный инвариант раздела — система папку BD не изменяет. План уборки
обязан состоять только из закомментированных команд: раскомментирование
— осознанное действие человека, а не поведение по умолчанию.
"""
from __future__ import annotations

import time
import unittest

from tests.base import Isolated

import organize


class TestOrganize(Isolated):
    def _doc(self, rel, section=None, brand=None, doc_type=None,
             effective_date=None, content_hash=None, mtime=None):
        fn = rel.rsplit("/", 1)[-1]
        self.db.run(
            "INSERT INTO documents (rel_path, abs_path, file_name, ext, "
            "content_hash, section, brand, doc_type, effective_date, mtime, "
            "status, is_current) VALUES (?,?,?,?,?,?,?,?,?,?,'ok',1)",
            (rel, "/kb/" + rel, fn, ".pdf", content_hash or rel,
             section, brand, doc_type, effective_date, mtime or time.time()))

    def test_progress_counts(self):
        self._doc("Р/ДЖИЛЕКС/1КАТАЛОГ/каталог.pdf", "Р", "джилекс",
                  "КАТАЛОГ", "2026-01-01")
        self._doc("Р/непонятный.pdf", "Р")
        pr = organize.metadata_progress()
        self.assertEqual(pr["total"], 2)
        self.assertEqual(pr["brand"], 50)
        self.assertEqual(pr["type"], 50)

    def test_untyped_with_hint(self):
        self._doc("Р/ДЖИЛЕКС/паспорт_водомет.pdf", "Р", "джилекс")
        items = organize.untyped()
        self.assertEqual(items[0]["hint"], "4ПАСПОРТ")

    def test_undated_only_replaceable_types(self):
        self._doc("Р/Д/2ПРАЙС_ЛИСТ/прайс.xlsx", "Р", "д", "ПРАЙС-ЛИСТ")
        self._doc("Р/Д/3РУКОВОДСТВО/мануал.pdf", "Р", "д", "РУКОВОДСТВО")
        items = organize.undated()
        self.assertEqual(len(items), 1)
        self.assertIn("прайс", items[0]["path"])
        self.assertTrue(items[0]["mtime_hint"])

    def test_bad_names_catch_underscore(self):
        self._doc("Р/Д/1КАТАЛОГ/каталог_новый.pdf", "Р", "д", "КАТАЛОГ")
        self._doc("Р/Д/1КАТАЛОГ/каталог (2).pdf", "Р", "д", "КАТАЛОГ",
                  content_hash="other")
        words = {b["word"] for b in organize.bad_names()}
        self.assertEqual(len(words), 2, words)

    def test_brand_twins_via_translit(self):
        self._doc("Р/GRUNDFOS/1КАТАЛОГ/a.pdf", "Р", "grundfos", "КАТАЛОГ")
        self._doc("Р/ГРУНДФОС/4ПАСПОРТ/b.pdf", "Р", "грундфос", "ПАСПОРТ",
                  content_hash="b")
        twins = organize.brand_twins()
        self.assertEqual(len(twins), 1)
        names = {v["brand"] for v in twins[0]["variants"]}
        self.assertEqual(names, {"grundfos", "грундфос"})

    def test_plan_is_fully_commented(self):
        """Ни одной живой команды: каждая строка действия закомментирована."""
        self._doc("Р/Д/паспорт_х.pdf", "Р", "д")
        self._doc("Р/Д/2ПРАЙС_ЛИСТ/прайс.xlsx", "Р", "д", "ПРАЙС-ЛИСТ")
        self._doc("Р/Д/копия.pdf", "Р", "д", "КАТАЛОГ", content_hash="dup")
        self._doc("Р/Д/копия2.pdf", "Р", "д", "КАТАЛОГ", content_hash="dup")
        plan = organize.cleanup_plan()
        for line in plan.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            self.assertIn(stripped.split()[0], ("cd", "set"),
                          f"живая команда в плане: {line}")
        self.assertIn("mv ", plan)          # но сами команды в плане есть

    def test_csv_has_all_categories(self):
        self._doc("Р/Д/паспорт_х.pdf", "Р", "д")
        self._doc("Р/Д/2ПРАЙС_ЛИСТ/прайс_новый.xlsx", "Р", "д", "ПРАЙС-ЛИСТ")
        csv = organize.problems_csv()
        for cat in ("без типа", "без даты", "плохое имя"):
            self.assertIn(cat, csv, cat)

    def test_top_asked_brands(self):
        self._doc("Р/ДЖИЛЕКС/1КАТАЛОГ/к.pdf", "Р", "джилекс", "КАТАЛОГ")
        self._doc("Р/ВИЛО/1КАТАЛОГ/к2.pdf", "Р", "вило", "КАТАЛОГ",
                  content_hash="k2")
        for q in ("какой напор у джилекс", "цена джилекс", "гарантия вило"):
            self.db.run("INSERT INTO queries (question, answered, created_at) "
                        "VALUES (?, 1, datetime('now'))", (q,))
        top = organize.top_asked_brands()
        self.assertEqual(top[0]["brand"], "джилекс")
        self.assertEqual(top[0]["asked"], 2)

    def test_state_has_everything(self):
        st = organize.state()
        for key in ("progress", "untyped", "undated", "bad_names",
                    "brand_twins", "duplicates", "gaps", "top_asked"):
            self.assertIn(key, st)


class TestWiring(unittest.TestCase):
    def test_endpoints_and_section_wired(self):
        from tests.base import ROOT
        src = (ROOT / "webui.py").read_text(encoding="utf-8")
        for needle in ('"/api/organize"', '"/api/organize/plan"',
                       '"/api/organize/csv"', 'data-t="organize"',
                       "loadOrganize"):
            self.assertIn(needle, src, needle)


if __name__ == "__main__":
    unittest.main()
