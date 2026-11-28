"""Whether a model drafts with its MTP layer: the choice, the Terminal flag, the engine's /metrics facts and the retry
without a drafter. strata_common/drafting.py is the one place."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from strata_cli.launch import settings as S
from strata_cli.launch.runner import derive_config
from strata_common import drafting
from strata_common.engine_client import nodraft_retry_args
from tests.terminal.launch_support import run_cli

GYRO = "qwen3.8-flash-next-gyro-gyro-s"


class Decisions(unittest.TestCase):
    def test_both_sides_know_the_same_modes(self):
        self.assertEqual(drafting.MODES, S.DRAFTING_MODES)
        self.assertEqual(drafting.normalize(None), "auto")
        self.assertEqual(drafting.normalize("MTP"), "mtp")
        self.assertEqual(drafting.normalize("dflash2"), "auto", "a choice this Strata no longer offers is Auto")
        self.assertEqual(drafting.normalize("fastest"), "auto", "an unknown value is Auto, never an error")

    def test_auto_and_mtp_run_the_mtp_when_its_layer_is_here(self):
        for mode in ("auto", "mtp", None):
            self.assertEqual(drafting.decide(mode, True), drafting.Decision(drafting.normalize(mode), "mtp"))
            self.assertEqual(drafting.decide(mode, None).drafter, "mtp", "not known is taken as there")

    def test_no_layer_runs_none_and_says_so(self):
        d = drafting.decide("auto", False)
        self.assertEqual((d.drafter, d.note), ("off", drafting.NONE_HERE))
        self.assertEqual(drafting.summary_line(d), "Drafting: off (no drafter is downloaded yet)")

    def test_off_is_off(self):
        self.assertEqual(drafting.decide("off", True), drafting.Decision("off", "off"))
        self.assertEqual(drafting.summary_line(drafting.decide("mtp", True)), "Drafting: MTP")
        self.assertEqual(drafting.summary_line(drafting.decide("off", True)), "Drafting: Off")

    def test_the_layer_is_looked_for_on_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt = Path(tmp) / "mtp" / "rt"
            self.assertFalse(drafting.mtp_ready(["--mtp", str(rt)]))
            rt.mkdir(parents=True)
            (rt / "experts.bin").write_bytes(b"e")
            self.assertTrue(drafting.mtp_ready(["--mtp", str(rt)]))
        self.assertIsNone(drafting.mtp_ready(["--pack", "/d/packs/x"]), "no layer named")
        self.assertIsNone(drafting.mtp_ready(["--mtp", "4"]), "Qwen3.6's draft depth")



class WhatTheEngineStarted(unittest.TestCase):
    def test_the_metrics_facts(self):
        self.assertEqual(drafting.engine_info(["--mtp", "/d/mtp/rt"]), {"drafting": "mtp", "drafting_note": ""})
        self.assertEqual(drafting.engine_info(["--pack", "/p"], drafting.SHORT_OF_VRAM),
                         {"drafting": "off", "drafting_note": "Drafting: off (the cards were short of memory)"})

    def test_the_no_drafter_retry_exists_for_an_auto_start_with_a_drafter(self):
        base = {"model_name": GYRO, "args": ["--pack", "/d/packs/gyro", "--native", "/d/m.gguf", "--mtp", "/d/mtp/rt", "--spec", "4"]}
        alt = nodraft_retry_args({**base, "drafting": "auto"}, ["--effort", "x"])
        self.assertIsNotNone(alt)
        self.assertNotIn("--mtp", alt)
        self.assertEqual(alt[-2:], ["--effort", "x"], "what the start adds goes along")
        self.assertIn("--native", alt, "everything else is the same start")
        self.assertIsNotNone(nodraft_retry_args(dict(base)), "no choice saved means Auto")
        self.assertIsNone(nodraft_retry_args({**base, "drafting": "mtp"}), "named by the person: it fails with its reason")
        self.assertIsNone(nodraft_retry_args({**base, "drafting": "off"}), "no drafter to drop")
        drafting.take_note()
        nodraft_retry_args({**base, "drafting": "auto"})
        self.assertEqual(drafting.take_note(), "", "working the retry out leaves the note of the real start alone")


class TerminalFlag(unittest.TestCase):
    BASE = {"exe": "component:strata-engine", "args": ["--pack", "/d/packs/x", "--mtp", "/d/mtp/rt"], "port": 8080, "model_name": GYRO}

    def test_the_flag_is_listed_planned_copyable_and_saved_in_the_launch_config(self):
        code, out, _ = run_cli(["launch", "--help"])
        self.assertIn("auto|mtp|off", out)
        cfg = derive_config(self.BASE, S.LaunchSettings(drafting="mtp"), bool_flags=frozenset())
        self.assertEqual(cfg["drafting"], "mtp")
        self.assertNotIn("drafting", derive_config(self.BASE, S.LaunchSettings(), bool_flags=frozenset()), "Auto writes nothing: nobody chose")
        self.assertIn("--drafting off", S.flags_text(S.LaunchSettings(drafting="off")))
        self.assertNotIn("--drafting", S.flags_text(S.LaunchSettings()))
        code, out, err = run_cli(["launch", "--plan", "--drafting", "dflash2"])
        self.assertNotEqual(code, 0)

    def test_off_writes_no_draft_layer(self):
        cfg = derive_config(self.BASE, S.LaunchSettings(drafting="off"), bool_flags=frozenset())
        self.assertIs(cfg["mtp"], False)
        self.assertNotIn("--mtp", cfg["args"])

    def test_a_saved_preset_keeps_the_choice(self):
        s = S.LaunchSettings.from_dict(S.LaunchSettings(drafting="mtp").to_dict())
        self.assertEqual(s.drafting, "mtp")
        self.assertIsNone(S.LaunchSettings.from_dict({}).drafting)
        self.assertIsNone(S.LaunchSettings.from_dict({"split_engine": "on"}).drafting, "a key this Strata no longer has is ignored")


if __name__ == "__main__":
    unittest.main()
