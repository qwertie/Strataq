"""A process given a settings file of its own (STRATA_SETTINGS_FILE) never adopts an older Strata install (tester report
2026-10-17: files were written into the tester's old T:\\Strata); the shared %APPDATA% / ~/.config file with the old
install's folders is not read, and settings are saved to the own file only."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from strata_control import models  # noqa: E402
import setup as strata_setup  # noqa: E402


class OwnSettingsFile(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        base = Path(self.dir.name)
        self.old = base / "T-Strata"
        self.old.mkdir()
        (self.old / "strata-gyro-gyro-s.json").write_text(json.dumps({"exe": "strata.exe", "tokenizer": "t", "args": []}))
        shared = base / "appdata" / "Strata"
        shared.mkdir(parents=True)
        (shared / "settings.json").write_text(json.dumps({"installs": [str(self.old)], "data_dir": str(base / "T-data")}))
        (base / "xdg" / "strata").mkdir(parents=True)
        (base / "xdg" / "strata" / "settings.json").write_text(json.dumps({"installs": [str(self.old)], "data_dir": str(base / "T-data")}))
        self.own_data = base / "own" / "Strata"
        self.own_data.mkdir(parents=True)
        self.env = {"APPDATA": str(base / "appdata"), "XDG_CONFIG_HOME": str(base / "xdg"),
                    "LOCALAPPDATA": str(base / "local"), "XDG_DATA_HOME": str(base / "xdg-data"), "STRATA_COMPONENTS_DIR": ""}
        self.base = base
        self.terminal = base / "terminal"
        self.terminal.mkdir()

    def test_the_terminal_still_remembers_the_old_install(self):
        with mock.patch.dict(os.environ, self.env), mock.patch.dict(os.environ, {}, clear=False), mock.patch.object(models, "ROOT", self.terminal):
            os.environ.pop("STRATA_SETTINGS_FILE", None)
            os.environ.pop("STRATA_INSTALL_DIR", None)
            self.assertEqual(models.resolve_install(), (self.old.resolve(), "remembered"))

    def test_a_process_with_its_own_settings_file_does_not(self):
        env = {**self.env, "STRATA_SETTINGS_FILE": str(self.own_data / "settings.json")}
        with mock.patch.dict(os.environ, env), mock.patch.object(models, "ROOT", self.terminal):
            os.environ.pop("STRATA_INSTALL_DIR", None)
            self.assertEqual(models.settings_path(), self.own_data / "settings.json")
            self.assertEqual(models.remembered_installs(), [])
            # no older install is adopted: with nothing of its own yet it gets the default install folder (beside the components folder)
            found, how = models.resolve_install()
            self.assertEqual(how, "default")
            self.assertNotEqual(found, self.old.resolve())
            self.assertEqual(strata_setup.settings_path(), self.own_data / "settings.json")
            self.assertEqual(strata_setup.load_settings(), {})
            # the installer's own list of "other installs" has no sibling scan and no shared settings
            self.assertEqual(strata_setup.other_installs(strata_setup.load_settings()), [])

    def test_saving_settings_goes_to_the_own_file_only(self):
        env = {**self.env, "STRATA_SETTINGS_FILE": str(self.own_data / "settings.json")}
        shared = self.base / "appdata" / "Strata" / "settings.json"
        before = shared.read_bytes()
        with mock.patch.dict(os.environ, env):
            strata_setup.save_settings({"data_dir": str(self.own_data / "data")})
        self.assertEqual(shared.read_bytes(), before)
        self.assertIn("data_dir", json.loads((self.own_data / "settings.json").read_text()))


if __name__ == "__main__":
    unittest.main()
