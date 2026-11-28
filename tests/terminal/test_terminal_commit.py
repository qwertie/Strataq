"""package_terminal.py must work where there is no git (a tree copy without .git): the commit comes from STRATA_COMMIT, falls back to git when it exists, and is simply left out when neither answers."""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packaging"))
import package_terminal  # noqa: E402

SHA = "0123456789abcdef0123456789abcdef01234567"


class SourceCommit(unittest.TestCase):
    def test_env_wins_and_is_lower_cased(self):
        with mock.patch.dict(os.environ, {"STRATA_COMMIT": " " + SHA.upper() + " "}), \
             mock.patch.object(package_terminal.subprocess, "run", side_effect=AssertionError("git must not run")):
            self.assertEqual(package_terminal.source_commit(), SHA)

    def test_no_git_installed_gives_empty_not_a_crash(self):
        env = {k: v for k, v in os.environ.items() if k != "STRATA_COMMIT"}
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch.object(package_terminal.subprocess, "run", side_effect=FileNotFoundError("git")):
            self.assertEqual(package_terminal.source_commit(), "")

    def test_bad_env_value_falls_back_to_git(self):
        done = subprocess.CompletedProcess([], 0, stdout=SHA + "\n", stderr="")
        with mock.patch.dict(os.environ, {"STRATA_COMMIT": "not-a-sha"}), \
             mock.patch.object(package_terminal.subprocess, "run", return_value=done):
            self.assertEqual(package_terminal.source_commit(), SHA)

    def test_not_a_checkout_gives_no_commit_file_value(self):
        done = subprocess.CompletedProcess([], 128, stdout="", stderr="fatal: not a git repository")
        env = {k: v for k, v in os.environ.items() if k != "STRATA_COMMIT"}
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch.object(package_terminal.subprocess, "run", return_value=done):
            self.assertEqual(package_terminal.source_commit(), "")


if __name__ == "__main__":
    unittest.main()
