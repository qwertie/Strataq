"""strata_server/test_memory_settings.py - POST /v1/memory (docs/MEMORY_MANAGER_PLAN.md, M6): the Memory panel's
controls change the engine's settings live (CHANGE_SETTINGS) and, saved as defaults, write them to the run config's
"args".  The real StrataEngine and Service over HTTP with test_parallel's fake engine behind them, so no GPU is needed.

    python -m pytest strata_server/test_memory_settings.py -q
"""
from __future__ import annotations

import contextlib
import io
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from strata_server.test_parallel import FakeEngineCase

CFG = {"exe": "engine/strata-q38.exe", "args": ["--pack", "data/packs/q2_0", "--rows", "2"],
       "args_written": ["--pack", "data/packs/q2_0", "--rows", "2"], "model_name": "qwen3.8-flash-next-q2_0"}


class MemorySettings(FakeEngineCase):
    def start(self, settings=True, config=False):
        super().start(2, more=["--settings"] if settings else [])
        self.cfg_path = Path(self.tmp.name) / "strata-q2_0.json"
        if config:
            self.cfg_path.write_text(json.dumps(CFG, indent=1), encoding="utf-8")
            self.svc.config_path = str(self.cfg_path)

    def post(self, body, origin="own", headers=None):
        h = {"Content-Type": "application/json", **(headers or {})}
        if origin:
            h["Origin"] = self.base if origin == "own" else origin
        req = urllib.request.Request(self.base + "/v1/memory", method="POST", headers=h,
                                     data=body if isinstance(body, bytes) else json.dumps(body).encode())
        try:
            with contextlib.redirect_stdout(io.StringIO()), urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def sent(self):
        """Gets the CHANGE_SETTINGS lines the fake engine received."""
        return [x for x in self.log.read_text().splitlines() if x.startswith("CHANGE_SETTINGS")] \
            if self.log.exists() else []

    def saved(self):
        return json.loads(self.cfg_path.read_text(encoding="utf-8"))

    def test_a_change_for_this_session(self):
        self.start()
        got = self.get("/v1/memory")
        self.assertEqual((got["can_change"], got["config"]), (True, False))
        self.assertEqual(set(got["defaults"].values()), {None})
        self.assertEqual(got["memories"][0]["setting"], 2000 << 20)
        status, got = self.post({"set": {"memory_mib.gpu0": 3000, "drafter": True, "memory_mib.gpu1": 100}})
        self.assertEqual(status, 200, got)
        self.assertEqual((got["changed"], got["saved"]), (["memory_mib.gpu0", "drafter", "memory_mib.gpu1"], False))
        gpu = got["memories"][0]
        self.assertEqual((gpu["setting"], gpu["smallest"], gpu["largest"]), (3000 << 20, 1000 << 20, 7000 << 20))
        self.assertEqual(got["settings"]["memory_mib.gpu0"], 3000)
        self.assertEqual(self.sent(), ["CHANGE_SETTINGS memory_mib.gpu0=3000 drafter=on memory_mib.gpu1=100"])

    def test_status_follows_live_concurrency(self):
        """Paged engines distinguish their scheduling limit from their protocol capacity."""
        self.start()
        self.engine.info["concurrent"] = 1
        self.assertEqual(self.get("/v1/status")["concurrency"]["serving"], 1)
        status, _ = self.post({"set": {"concurrent": 2}})
        self.assertEqual(status, 200)
        self.assertEqual(self.get("/v1/status")["concurrency"]["serving"], 2)

    def test_a_change_that_got_less_than_asked(self):
        self.start()
        status, got = self.post({"set": {"memory_mib.ram": 9000}})
        self.assertEqual((status, got.get("shortfall")), (200, "the RAM's pool: out of memory"), got)
        self.assertNotIn("shortfall", self.get("/v1/memory"))   # it was this change's

    def test_the_engine_refuses(self):
        self.start(config=True)
        before = self.cfg_path.read_bytes()
        status, got = self.post({"set": {"memory_mib.gpu0": 500, "rows": 8}, "save": "default"})
        self.assertEqual((status, got["error"]["type"]), (409, "refused"), got)
        self.assertEqual(got["error"]["message"], "the biggest active conversation needs 1000 MiB")
        self.assertEqual(self.cfg_path.read_bytes(), before)
        self.assertFalse(self.cfg_path.with_name(self.cfg_path.name + ".bak").exists())
        self.assertEqual(self.get("/v1/memory")["settings"]["rows"], 4)      # nothing of it changed

    def test_an_engine_that_cannot_change_settings(self):
        self.start(settings=False)
        self.assertFalse(self.get("/v1/memory")["can_change"])
        status, got = self.post({"set": {"rows": 4}})
        self.assertEqual((status, got["error"]["type"]), (409, "unsupported"), got)

    def test_bad_requests(self):
        self.start()
        for body in ({"set": {"context": 0}}, {"set": {"rows": 0}}, {"set": {"rows": 17}},
                     {"set": {"tokens_min": 1.5}}, {"set": {"concurrent": True}}, {"set": {"drafter": "yes"}},
                     {"set": {"drafter": 1}}, {"set": {}}, {"set": {"rows": 4}, "save": "forever"},
                     {"set": {"rows": 4}, "save": "default"}, {"nothing": 1}, b"not json"):
            status, got = self.post(body)
            self.assertEqual(status, 400, (body, got))
        status, got = self.post({"set": {"rows": 2, "tokens_maxx": 5}})
        self.assertIn("tokens_maxx", got["error"]["message"])                # the bad one is named
        self.svc.config_path = str(self.cfg_path)
        status, got = self.post({"set": {"memory_mib.gpu1": 100}, "save": "default"})   # no flag to save it to
        self.assertEqual(status, 400, got)
        self.assertEqual(self.sent(), [])

    def test_legacy_engine_cannot_change_its_context_limit(self):
        self.start(config=True)
        before = self.cfg_path.read_bytes()
        status, got = self.post({"set": {"context": 2048}, "save": "default"})
        self.assertEqual((status, got["error"]["type"]), (409, "unsupported"), got)
        self.assertIn("per-conversation limit", got["error"]["message"])
        self.assertEqual(self.sent(), [])
        self.assertEqual(self.cfg_path.read_bytes(), before)

    def test_legacy_engine_cannot_claim_a_vram_conversation_limit(self):
        self.start(config=True)
        before = self.cfg_path.read_bytes()
        status, got = self.post({"set": {"vram_conversations": 1}, "save": "default"})
        self.assertEqual((status, got["error"]["type"]), (409, "unsupported"), got)
        self.assertEqual(self.sent(), [])
        self.assertEqual(self.cfg_path.read_bytes(), before)

    def test_legacy_engine_cannot_claim_a_stale_eviction_policy(self):
        self.start(config=True)
        before = self.cfg_path.read_bytes()
        status, got = self.post({"set": {"conversation_idle_seconds": 66}, "save": "default"})
        self.assertEqual((status, got["error"]["type"]), (409, "unsupported"), got)
        self.assertEqual(self.sent(), [])
        self.assertEqual(self.cfg_path.read_bytes(), before)

    def test_saved_as_default(self):
        self.start(config=True)
        status, got = self.post({"set": {"rows": 4, "tokens_min": 512, "drafter": "off"}, "save": "default"})
        self.assertEqual(status, 200, got)
        self.assertTrue(got["saved"])
        self.assertEqual(got["defaults"], {**dict.fromkeys(got["defaults"]), "rows": 4, "tokens_min": 512,
                                           "drafter": "off"})
        cfg = self.saved()
        self.assertEqual(cfg["args"], ["--pack", "data/packs/q2_0", "--rows", "4", "--tokens-min", "512",
                                       "--drafter", "off"])
        self.assertEqual(cfg["user_args"], ["--rows", "4", "--tokens-min", "512", "--drafter", "off"])
        self.assertEqual(cfg["model_name"], CFG["model_name"])
        self.assertEqual(json.loads(self.cfg_path.with_name(self.cfg_path.name + ".bak").read_text()), CFG)
        status, got = self.post({"set": {"rows": 6}})                       # this session only: the file stays
        self.assertEqual((status, got["saved"]), (200, False))
        self.assertEqual(self.saved(), cfg)
        self.assertEqual(got["settings"]["rows"], 6)

    def test_a_layer_split_saves_every_card(self):
        from strata_server import runconfig
        settings = {"rows": 8, "memory_mib.gpu0": 9216, "memory_mib.gpu0.1": 9216, "memory_mib.ram": 8192}
        values = runconfig.check_live({"memory_mib.gpu0.1": 4096, "rows": 4}, True, list(settings)[1:3])
        new, changed = runconfig.apply_live({"args": ["--gpu-mib", "9216,9216"]}, values,
                                            {**settings, "memory_mib.gpu0.1": 4096})
        self.assertEqual(new["args"], ["--gpu-mib", "9216,4096", "--rows", "4"])
        self.assertEqual(changed, ["memory_mib.gpu0", "memory_mib.gpu0.1", "rows"])
        with self.assertRaises(ValueError):                               # a card the engine does not have
            runconfig.check_live({"memory_mib.gpu1.2": 4096}, True, list(settings)[1:3])

    def test_only_from_strata_s_own_page(self):
        self.start()
        self.assertEqual(self.post({"set": {"rows": 4}}, origin="https://evil.example")[0], 403)
        self.svc.api_key = "secret"                     # (with a key, a foreign page reaches the own-page check)
        key = {"Authorization": "Bearer secret"}
        self.assertEqual(self.post({"set": {"rows": 4}}, origin="https://evil.example", headers=key)[0], 403)
        self.assertEqual(self.post(b'{"set": {"rows": 4}}', headers={**key, "Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.post({"set": {"rows": 4}}, origin=None)[0], 401)
        self.assertEqual(self.sent(), [])
        self.assertEqual(self.post({"set": {"rows": 4}}, origin=None, headers=key)[0], 200)

    def test_it_waits_for_the_engine_s_turn(self):
        self.start()
        self.svc.fifo.acquire()                         # a request has the engine
        threading.Timer(0.3, self.svc.fifo.release).start()
        t0 = time.monotonic()
        self.assertEqual(self.post({"set": {"rows": 3}})[0], 200)
        self.assertGreaterEqual(time.monotonic() - t0, 0.25)
        self.svc.SETTINGS_WAIT_S = 0.2
        with self.svc.fifo:
            status, got = self.post({"set": {"rows": 5}})
        self.assertEqual(status, 503, got)
        self.assertEqual(self.sent(), ["CHANGE_SETTINGS rows=3"])
