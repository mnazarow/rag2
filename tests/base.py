"""
Общая обвязка для тестов.

Каждый тест работает на своей временной базе и своей папке данных,
поэтому запускать их можно на рабочей машине не боясь: настоящий
индекс они не видят и не трогают.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class Isolated(unittest.TestCase):
    """Тест на отдельной временной базе."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="kbtest_"))
        os.environ["DATA_DIR"] = str(self.tmp)
        os.environ["KB_ROOT"] = str(self.tmp / "BD")
        (self.tmp / "BD").mkdir(parents=True, exist_ok=True)
        import config
        import db
        import importlib
        importlib.reload(config)
        importlib.reload(db)
        self.config, self.db = config, db
        db.init()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)
