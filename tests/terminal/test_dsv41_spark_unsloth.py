"""Unsloth's DeepSeek sets on a DGX Spark (linux-aarch64): a set never run there is not offered (model list, `--model`, `--model-path`, the
app's "I already have the model" check, setup), and the line says what does work. On every other computer nothing changes."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import settings as S
from strata_cli.launch.catalog import Catalog
from strata_cli.launch.runner import LaunchError, Runner
from strata_control import dsv41_fit, dsv41_sets, models
from tests.terminal.launch_support import HW_12, Home, catalog

PC = {**HW_12, "ram_gb": 64.0, "disk_free_gb": 900.0, "engine_vendor": "nvidia"}
UD = ("UD-IQ1_M",)          # the set that SparkGate's app-check and setup tests try
BOTH = ("UD-IQ1_S", "UD-IQ1_M")   # both ran on a DGX Spark (10-30): AARCH64_NOT_YET is empty; SparkGate patches it back to test the mechanism


def ud_files(model):
    return {**dsv41_sets.tables()["sets"][model]["layouts"][0], **dsv41_sets.tables()["sets"][model]["engram_alt"]}


class Lists(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        self.hw = dict(PC)
        p = mock.patch.object(Catalog, "hardware", lambda s, fresh=False: dict(self.hw))
        p.start()
        self.addCleanup(p.stop)

    def models(self):
        return [r.model for r in catalog(self.hw).rows() if r.family == "deepseek41"]

    def test_a_pc_is_offered_them(self):
        got = self.models()
        for m in BOTH:
            self.assertIn(m, got)

    def test_model_flag_on_a_pc_is_unchanged(self):
        self.assertEqual(catalog(self.hw).resolve("ud-iq1_s"), ("deepseek41", "UD-IQ1_S"))


class OwnFiles(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        self.hw = dict(PC)
        p = mock.patch.object(Catalog, "hardware", lambda s, fresh=False: dict(self.hw))
        p.start()
        self.addCleanup(p.stop)

    def folder(self, model):
        folder = Path(os.environ["STRATA_INSTALL_DIR"]) / model
        folder.mkdir(parents=True, exist_ok=True)
        for name in dsv41_sets.tables()["sets"][model]["layouts"][0]:
            (folder / name).write_bytes(b"GGUF")
        return folder

    def test_the_same_folder_on_another_computer_is_the_model(self):
        folder = self.folder("UD-IQ1_S")
        with mock.patch.object(sys, "platform", "linux"):
            r = Runner(S.LaunchSettings(model_path=str(folder)), catalog=catalog(self.hw))
            r.check_own_files()
        self.assertEqual((r.s.family, r.s.model), ("deepseek41", "UD-IQ1_S"))


class AppCheck(unittest.TestCase):
    def preview(self, model, platform):
        with mock.patch.object(models, "_dsv41_files", return_value=ud_files(model)), mock.patch.object(sys, "platform", platform), \
                __import__("tempfile").TemporaryDirectory() as tmp:
            return models.plan_import(tmp)

    def test_on_a_pc_the_preview_is_the_set(self):
        for model in BOTH:
            plan = self.preview(model, "linux")
            self.assertEqual(plan["file_set_status"], "plausible", plan["issues"])
            self.assertEqual(plan["candidate_selection"]["model"], model)
            self.assertNotIn(SPARK_LINE, plan["messages"])


class Setup(unittest.TestCase):
    """setup.py is the last stop: the line, before an engine, a download or a pack is touched. Used by SparkGate.test_setup_stops_on_a_spark."""

    def runTest(self):
        pass

    def run_setup(self, model, platform, machine=None):
        import contextlib
        import importlib.util
        import io
        import types
        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location("setup_spark_ud", root / "setup.py")
        setup = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(setup)
        a = types.SimpleNamespace(model=model, models_dir="/nonexistent/models", yes=True, gguf_dir=None)
        out = io.StringIO()
        with mock.patch.object(sys, "platform", platform), mock.patch("platform.machine", return_value=machine or "x86_64"),                 contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            try:
                setup.install_dsv41(a, "deepseek41", {}, False, False, 13, True, 8080, 128.0, False)
            except BaseException as exc:                      # noqa: BLE001 - fail() exits; anything after the check would raise something else
                return type(exc).__name__, out.getvalue()
        return None, out.getvalue()

SPARK_LINE = "Unsloth's 1-bit DeepSeek sets were not tested on a DGX Spark yet; the 2.00 and 2.25 sets were."


def spark():
    """A DGX Spark: linux-aarch64 (the two patches the one rule reads)."""
    from contextlib import ExitStack
    stack = ExitStack()
    stack.enter_context(mock.patch.object(sys, "platform", "linux"))
    stack.enter_context(mock.patch("platform.machine", return_value="aarch64"))
    return stack


class SparkGate(unittest.TestCase):
    """The same gate for the DGX Spark (dsv41_fit.AARCH64_NOT_YET): the sets were never run there. Removing the names from the tuple lifts it."""

    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        self.hw = {**PC, "engine_vendor": "nvidia"}
        p = mock.patch.object(Catalog, "hardware", lambda s, fresh=False: dict(self.hw))
        p.start()
        self.addCleanup(p.stop)
        q = mock.patch.object(dsv41_fit, "AARCH64_NOT_YET", BOTH)     # the mechanism, as it was while the sets were kept off
        q.start()
        self.addCleanup(q.stop)

    def test_the_rule(self):
        self.assertEqual(dsv41_fit.AARCH64_NOT_YET, BOTH)
        self.assertEqual(dsv41_fit.AARCH64_NOT_YET_LINE, SPARK_LINE)
        with spark():
            for m in BOTH:
                self.assertEqual(dsv41_fit.not_here(m), SPARK_LINE)
            self.assertIsNone(dsv41_fit.not_here("IST-DAS-2.25bpw"))
        with mock.patch.object(sys, "platform", "linux"), mock.patch("platform.machine", return_value="x86_64"):
            self.assertIsNone(dsv41_fit.not_here("UD-IQ1_S"))

    def test_not_in_the_list_on_a_spark_but_on_a_pc(self):
        with spark():
            got = [r.model for r in catalog(self.hw).rows() if r.family == "deepseek41"]
        for m in BOTH:
            self.assertNotIn(m, got)
        self.assertIn("IST-DAS-2.25bpw", got)
        pc = [r.model for r in catalog(self.hw).rows() if r.family == "deepseek41"]
        for m in BOTH:
            self.assertIn(m, pc)

    def test_model_flag_on_a_spark_says_what_works(self):
        with spark():
            with self.assertRaises(ValueError) as cm:
                catalog(self.hw).resolve("ud-iq1_s", None)
        self.assertIn(SPARK_LINE, str(cm.exception))
        self.assertIn("ist-das-2.25bpw", str(cm.exception))

    def test_model_path_of_a_ud_folder_on_a_spark_stops(self):
        folder = Path(os.environ["STRATA_INSTALL_DIR"]) / "UD-IQ1_M"
        folder.mkdir(parents=True, exist_ok=True)
        for name in dsv41_sets.tables()["sets"]["UD-IQ1_M"]["layouts"][0]:
            (folder / name).write_bytes(b"GGUF")
        with spark():
            with self.assertRaises(LaunchError) as cm:
                Runner(S.LaunchSettings(model_path=str(folder)), catalog=catalog(self.hw)).check_own_files()
        self.assertIn(SPARK_LINE, cm.exception.text)

    def test_app_check_on_a_spark(self):
        import tempfile
        for model in UD:
            with mock.patch.object(models, "_dsv41_files", return_value=ud_files(model)), tempfile.TemporaryDirectory() as tmp, spark():
                plan = models.plan_import(tmp)
            self.assertIn(SPARK_LINE, plan["messages"])
            self.assertIn("not_on_this_spark", plan["issues"])
            self.assertIsNone(plan["candidate_selection"])
            self.assertIn(SPARK_LINE.rstrip("."), plan["plain_issues"])

    def test_setup_stops_on_a_spark(self):
        for model in UD:
            exc, text = Setup().run_setup(model, "linux", machine="aarch64")
            self.assertEqual(exc, "SystemExit", (model, text[-300:]))
            self.assertIn(SPARK_LINE, text)
            self.assertIn("nothing has been downloaded", text)


class SparkOffered(unittest.TestCase):
    """Both Unsloth sets ran on a DGX Spark (2026-10-30): the production rule hides nothing there."""

    def test_nothing_is_hidden_on_a_spark(self):
        self.assertEqual(dsv41_fit.AARCH64_NOT_YET, ())
        with spark():
            for m in BOTH + ("IST-DAS-2.25bpw",):
                self.assertIsNone(dsv41_fit.not_here(m))


if __name__ == "__main__":
    unittest.main()
