"""An installed DeepSeek engine is updated by the explicit update (`strata update`), never by a start.

* update_dsv41: the signed feed's newer package of the SAME kind is downloaded, checked, put next to the old one, started once, and only then every saved
  run config is pointed at it; older folders go, except one a running engine uses; nothing installed means nothing happens.
* The start (strata_control/runtime.py) only repairs: a run config whose folder an update removed is pointed at the newest installed package, locally.
  It makes no request of any kind.
* The check keeps what is newer (engine_component.cached_offers) for the update notice; `strata launch`'s own upgrade is not touched here.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import sys
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from strata_common import cfg_paths, dsv41_paths  # noqa: E402
from strata_control import runtime, updater
from strata_common import engine_component as ec, signing
from tests.terminal import test_dsv41_upgrade as _up  # noqa: E402

EXE = dsv41_paths.exe_name()


class UpdatePath(_up.Upgrade):
    def setUp(self):
        super().setUp()
        self.install_dir = self.root / "data"
        self.cfg1 = self.install_dir / "strata-a.json"
        self.cfg2 = self.root / "run" / "app-1.json"
        self.cfg2.parent.mkdir()
        for c in (self.cfg1, self.cfg2):
            self.write_cfg(c, self.old)
        self.configs = [self.cfg1, self.cfg2]
        for p in (mock.patch.object(dsv41_paths, "saved_configs", return_value=[self.cfg2]),
                  mock.patch.object(dsv41_paths, "test_start", return_value=(True, "strata-dsv41 0.1.49.1 build x")),
                  mock.patch.object(dsv41_paths, "in_use", return_value=False)):
            p.start()
            self.addCleanup(p.stop)
        self.lines = []

    def write_cfg(self, path, exe):
        Path(path).write_text(json.dumps({"exe": str(exe), "args": ["--pack", "p"], "tokenizer": "t",
                                          "lib_dirs": ["C:/other", str(Path(exe).parent)]}), encoding="utf-8")

    def exe_of(self, path):
        return Path(cfg_paths.read_config(path)["exe"])

    def ctx(self):
        return updater.Context(components=self.root / "c", source=self.url, trusted=signing.load_trusted_keys(), hardware=self.hw,
                               emit=lambda e: self.lines.append(e.get("message")), environ={**os.environ, "STRATA_INSTALL_DIR": str(self.install_dir)})

    def run_step(self):
        ctx = self.ctx()
        state = updater.State(ctx.components)
        feed = updater.open_feed(ctx, state)
        return ctx, state, feed

    # ---- the explicit update ------------------------------------------------------------------------------------------
    def test_the_update_installs_next_to_the_old_points_every_saved_config_at_it_and_keeps_the_old(self):
        self.publish("0.1.49.1")
        ctx, state, feed = self.run_step()
        r = updater.update_dsv41(ctx, feed, state)
        self.assertEqual((r["status"], r["installed_before"], r["installed_after"]), ("updated", "0.1.49", "0.1.49.1"), r)
        new = dsv41_paths.installed(self.dest)[-1]
        self.assertEqual(self.folders(), ["0.1.49", "0.1.49.1"], "the older package stays for Settings > Engine")
        self.assertEqual(self.exe_of(self.cfg1), new)
        self.assertEqual(self.exe_of(self.cfg2), new, "every saved run config, not one")
        libs = cfg_paths.read_config(self.cfg1)["lib_dirs"]
        self.assertEqual(libs[0], "C:/other")
        self.assertIn(str(new.parent), libs)
        self.assertEqual(r["profiles_repointed"], 2)
        self.assertEqual(self.tree(self.eng), self.before, "the Qwen engine's files must not change")

    def test_the_old_folder_is_there_until_the_new_one_has_started(self):
        self.publish("0.1.49.1")
        ctx, state, feed = self.run_step()
        seen = {}

        def start(exe):
            seen["folders"] = self.folders()
            seen["cfg"] = self.exe_of(self.cfg1)
            return True, "strata-dsv41 0.1.49.1"
        with mock.patch.object(dsv41_paths, "test_start", side_effect=start):
            updater.update_dsv41(ctx, feed, state)
        self.assertEqual(seen["folders"], ["0.1.49", "0.1.49.1"])
        self.assertEqual(seen["cfg"], self.old, "no config is touched before the test start passed")

    def test_a_package_that_does_not_start_changes_nothing(self):
        self.publish("0.1.49.1")
        ctx, state, feed = self.run_step()
        with mock.patch.object(dsv41_paths, "test_start", return_value=(False, "it did not start correctly (x)")):
            with self.assertRaises(updater.UpdateError):
                updater.update_dsv41(ctx, feed, state)
        self.assertEqual(self.folders(), ["0.1.49"])
        self.assertEqual(self.exe_of(self.cfg1), self.old)
        self.assertEqual(self.exe_of(self.cfg2), self.old)

    def test_a_folder_a_running_engine_uses_is_not_removed_and_the_configs_still_move(self):
        self.publish("0.1.49.1")
        ctx, state, feed = self.run_step()
        with mock.patch.object(dsv41_paths, "in_use", return_value=True):
            r = updater.update_dsv41(ctx, feed, state)
        self.assertEqual(r["status"], "updated")
        self.assertEqual(self.folders(), ["0.1.49", "0.1.49.1"])
        self.assertTrue(r["kept_in_use"])
        self.assertIn("0.1.49.1", str(self.exe_of(self.cfg2)))
        # a later update (nothing newer) leaves it; the start-side repair never removes anything
        self.assertEqual(dsv41_paths.prune(self.dest, dsv41_paths.installed(self.dest)[-1]), [])
        self.assertEqual(self.folders(), ["0.1.49.1"])

    def test_nothing_newer_or_nothing_installed_changes_nothing(self):
        ctx, state, feed = self.run_step()
        r = updater.update_dsv41(ctx, feed, state)
        self.assertEqual((r["status"], r["changed"]), ("current", False))
        self.assertEqual(self.folders(), ["0.1.49"])
        with mock.patch.object(dsv41_paths, "saved_configs", return_value=[]), mock.patch.dict(os.environ, {}):
            ctx = updater.Context(components=self.root / "c2", source=self.url, trusted=signing.load_trusted_keys(), hardware=self.hw, environ={})
            r = updater.update_dsv41(ctx, feed, state)
        self.assertEqual((r["status"], r["changed"]), ("none-installed", False))

    def test_a_package_of_another_backend_is_not_an_update_of_this_one(self):
        self.publish("0.1.49.1")
        ctx, state, feed = self.run_step()
        with mock.patch.object(dsv41_paths, "kind_of", side_effect=lambda n: "windows-x86_64-hip-rdna3" if n.startswith("0.1.49-") else "x"):
            r = updater.update_dsv41(ctx, feed, state)
        self.assertNotEqual(r["status"], "updated")
        self.assertEqual(self.folders(), ["0.1.49"])

    def test_the_update_step_is_part_of_run_update_with_the_engine(self):
        self.publish("0.1.49.1")
        ctx = self.ctx()
        with mock.patch.object(updater, "update_engine", return_value={"component": "engine", "status": "current", "changed": False}), \
                mock.patch.object(updater, "decide_engine", return_value={"status": "current", "offered": None, "installed": None}):
            out = updater.run_update(ctx, engine=True, control=False)
        names = [r["component"] for r in out["results"]]
        self.assertEqual(names, ["engine", "dsv41"], out)
        self.assertEqual(out["results"][1]["status"], "updated")
        with mock.patch.object(updater, "decide_engine", return_value={"status": "current", "offered": None, "installed": None}):
            out = updater.run_update(ctx, engine=False, control=False)
        self.assertEqual(out["results"], [], "--no-engine leaves the DeepSeek engine alone too")

    # ---- what the check keeps for the update notice ---------------------------------------------------------------------
    def test_the_check_says_a_newer_deepseek_engine_is_there_and_the_notice_can_read_it_without_the_network(self):
        self.publish("0.1.49.1")
        ctx = self.ctx()
        with mock.patch.object(updater, "decide_engine", return_value={"status": "current", "offered": None, "installed": None}), \
                mock.patch.object(updater, "decide_control", return_value={"status": "not-in-feed", "installed": None, "offered": None}):
            report = updater.check(ctx)
        self.assertEqual((report["dsv41"]["status"], report["dsv41"]["installed"], report["dsv41"]["offered"]), ("available", "0.1.49", "0.1.49.1"))
        with mock.patch.object(urllib.request, "urlopen", side_effect=AssertionError("network")), mock.patch.object(socket, "create_connection", side_effect=AssertionError("network")):
            offers = ec.cached_offers(ctx.components)
            self.assertEqual(offers["dsv41"], "0.1.49.1")
            self.assertIn("DeepSeek engine 0.1.49.1", ec.cached_notice(ctx.components))
            self.assertEqual(ec.status(ctx.components)["update_available"]["dsv41"], "0.1.49.1")
        state = updater.State(ctx.components)
        feed = updater.open_feed(ctx, state)
        self.assertEqual(updater.update_dsv41(ctx, feed, state)["status"], "updated")
        self.assertIsNone(ec.cached_offers(ctx.components)["dsv41"], "installed: nothing to offer any more")

    # ---- the start ----------------------------------------------------------------------------------------------------------
    def test_a_start_repairs_a_config_whose_folder_an_update_removed_and_touches_the_network_not_at_all(self):
        self.publish("0.1.49.1")
        ctx, state, feed = self.run_step()
        self.assertEqual(updater.update_dsv41(ctx, feed, state)["status"], "updated")
        new = dsv41_paths.installed(self.dest)[-1]
        self.write_cfg(self.cfg2, self.old)                                           # (a config written again from a model's older file: its folder is gone)
        shutil.rmtree(self.old.parent)                                                # (the person deleted the older engine in Settings > Engine)
        self.assertFalse(self.old.exists())
        log = self.root / "run" / "app-1.host.log"
        profile = {"source_config": str(self.cfg2)}
        with mock.patch.object(urllib.request, "urlopen", side_effect=AssertionError("a start reached the network")), \
                mock.patch.object(socket, "create_connection", side_effect=AssertionError("a start opened a connection")), \
                mock.patch("subprocess.Popen", side_effect=AssertionError("a start ran a program")), \
                mock.patch.object(runtime.profiles, "load", return_value=(profile, "r")), \
                mock.patch.object(runtime, "_paths", return_value=(None, None, log)):
            runtime._repair_dsv41_paths("app-1")
        self.assertEqual(self.exe_of(self.cfg2), new)
        self.assertIn("now uses 0.1.49.1", log.read_text(encoding="utf-8"))

    def test_a_start_leaves_a_config_whose_engine_is_there_and_the_feed_unread(self):
        self.publish("0.1.49.1")
        before = self.cfg2.read_bytes()
        with mock.patch.object(urllib.request, "urlopen", side_effect=AssertionError("network")), \
                mock.patch.object(runtime.profiles, "load", return_value=({"source_config": str(self.cfg2)}, "r")):
            runtime._repair_dsv41_paths("app-1")
        self.assertEqual(self.cfg2.read_bytes(), before)
        self.assertEqual(self.folders(), ["0.1.49"], "a newer package in the feed is not fetched by a start")

    def test_the_start_side_modules_do_not_reach_the_updater_the_feed_or_the_launch_upgrade(self):
        import ast
        for name in ("strata_control/runtime.py", "strata_common/dsv41_paths.py"):
            tree = ast.parse((Path(__file__).resolve().parents[2] / name).read_text(encoding="utf-8"))
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(a.name for a in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module)
                    imported.update(f"{node.module}.{a.name}" for a in node.names)
            for bad in ("updater", "dsv41_engine", "feed", "transfer", "components", "urllib.request", "http.client"):
                self.assertFalse([i for i in imported if i == bad or i.endswith("." + bad)], f"{name} imports {bad}")

    def test_a_config_that_is_not_ours_is_left_alone(self):
        for exe in ("@component:strata-engine", str(self.root / "mine" / EXE)):
            c = self.root / "run" / "own.json"
            c.write_text(json.dumps({"exe": exe, "args": []}), encoding="utf-8")
            self.assertEqual(dsv41_paths.repair_missing([c]), [])
            self.assertEqual(json.loads(c.read_text(encoding="utf-8"))["exe"], exe)
        c = self.root / "run" / "broken.json"
        c.write_text("{not json", encoding="utf-8")
        self.assertEqual(dsv41_paths.repair_missing([c, self.root / "run" / "missing.json"]), [])


class StartCheck(unittest.TestCase):
    """dsv41_paths.test_start: strata-ds41 has no --version; started with no arguments it loads its libraries and
    refuses to go on without --pack, which is the sign it runs."""

    def start(self, returncode, stdout="", stderr=""):
        done = mock.Mock(returncode=returncode, stdout=stdout, stderr=stderr)
        with mock.patch("subprocess.run", return_value=done) as run:
            result = dsv41_paths.test_start(Path("/x/engine-dsv41/0.2.14-a") / EXE)
        self.assertEqual(run.call_args.args[0], [str(Path("/x/engine-dsv41/0.2.14-a") / EXE)])
        return result

    def test_the_refusal_without_a_pack_means_it_starts(self):
        self.assertEqual(self.start(2, stderr="strata-ds41: --pack <model>/maya-ds4 is needed\n"),
                         (True, "strata-ds41: --pack <model>/maya-ds4 is needed"))

    def test_anything_else_is_a_failed_start_with_its_last_line(self):
        ok, why = self.start(-1073741515, stderr="")                     # a Windows DLL that is not found: nothing printed
        self.assertFalse(ok)
        self.assertIn("did not start correctly", why)
        ok, why = self.start(127, stderr="error while loading shared libraries: libhipblas.so.3\n")
        self.assertFalse(ok)
        self.assertIn("libhipblas.so.3", why)

    def test_a_config_saved_before_strata_ds41_is_still_in_its_package_folder(self):
        for name in (EXE, "strata-dsv41.exe" if os.name == "nt" else "strata-dsv41"):
            exe = str(Path("/x") / dsv41_paths.DIR / "0.2.13-a" / name)
            with mock.patch.dict(os.environ, {"STRATA_DSV41_EXE": ""}):
                self.assertEqual(dsv41_paths.engine_root(exe), Path("/x") / dsv41_paths.DIR)


for _name in dir(_up.Upgrade):                       # (the parent's own tests are not repeated here)
    if _name.startswith("test_") and _name not in UpdatePath.__dict__:
        setattr(UpdatePath, _name, None)


if __name__ == "__main__":
    unittest.main()
