"""Checks source and Control launch scripts when their folder differs from the caller's working folder."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import setup  # noqa: E402


class RunScriptPaths(unittest.TestCase):
    def test_run_script_names_its_config_and_server_from_its_own_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            install = Path(tmp) / "Strata"
            install.mkdir()
            for win in (True, False):
                for control in (True, False):
                    with self.subTest(win=win, control=control):
                        root = Path(tmp) / "Control" if control else install
                        python = root / ".venv" / "python.exe"
                        server = root / "sycl" / "serve" / "server_intel.py" if control else None
                        with mock.patch.object(setup, "home", lambda: install), \
                                mock.patch.object(setup, "ROOT", root), mock.patch.object(setup, "WIN", win), \
                                mock.patch.object(sys, "executable", str(python)):
                            script = setup.write_run_script("iq3_s", install / "strata-iq3_s.json", 8080,
                                                            open_browser=False, server=server)
                            text = script.read_text(encoding="utf-8")
                        here = "%~dp0" if win else "$here/"
                        cd = "cd /d" if win else "cd"
                        self.assertIn(f'{cd} "{root if control else here + "."}"', text)
                        self.assertIn(f'"{here}strata-iq3_s.json"', text)
                        self.assertIn(f'"{python if control else here + str(python.relative_to(install))}"', text)
                        expected_server = str(server) if control else here + "strata_server" + os.sep + "server.py"
                        self.assertIn(f'"{expected_server}"', text)
                        self.assertNotIn('"--open"', text)
                        if not win:
                            self.assertIn('here=$(cd "$(dirname "$0")" && pwd)', text)


if __name__ == "__main__":
    unittest.main()
