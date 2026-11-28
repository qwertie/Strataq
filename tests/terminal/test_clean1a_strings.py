"""The four strings the owner approved word for word (11-07), and the rule for the third."""
from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class Wording(unittest.TestCase):
    def test_the_adopted_files_line_names_what_was_not_downloaded(self):
        src = (ROOT / "strata_cli" / "launch" / "runner.py").read_text(encoding="utf-8")
        self.assertIn("({path.name}). No model files were downloaded.\")", src)
        self.assertNotIn("({path.name}). Nothing was downloaded.", src)


if __name__ == "__main__":
    unittest.main()
