"""Request inspection through HTTP; no native model or GPU is required."""
import json
from array import array
import threading
import time
import unittest
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from strata_server.frontend import ChatTemplate
from strata_server.server import ByteTokenizer, Service, serve
from strata_server.test_server import start_http
from strata_common.engine_client import EngineDied, MockEngine


class ReloadableEngine(MockEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.running, self.unloaded = True, False

    def alive(self):
        return self.running

    def unload(self):
        self.running, self.unloaded = False, True

    def restart(self):
        self.running, self.unloaded = True, False


class MonitorOff(unittest.TestCase):
    """#332 is opt-in: by default no prompt or answer is kept, and the monitor's endpoints do not exist."""

    def test_off_by_default(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "Hello.", max_context=4096), tok,
                      ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            req = urllib.request.Request(base + "/v1/chat/completions", headers={"Content-Type": "application/json"},
                                         data=json.dumps({"messages": [{"role": "user", "content": "secret"}]}).encode())
            with urllib.request.urlopen(req, timeout=10) as r:
                self.assertEqual(r.status, 200)
            self.assertEqual(len(svc.api_requests), 0)
            with urllib.request.urlopen(base + "/health", timeout=10) as r:
                self.assertNotIn("api_monitor", json.loads(r.read()))   # the default answer is unchanged
            for path in ("/api/requests", "/api-monitor", "/api/context"):
                with self.assertRaises(urllib.error.HTTPError) as e:
                    urllib.request.urlopen(base + path, timeout=10)
                self.assertEqual(e.exception.code, 404, path)
                e.exception.close()
        finally:
            httpd.shutdown()
            httpd.server_close()


class InitialMemory(unittest.TestCase):
    """The first monitor poll can use a real reading while prompt processing owns the engine."""

    def test_start_and_reload_cache_memory_before_requests(self):
        for lazy in (False, True):
            with self.subTest(lazy=lazy):
                tok = ByteTokenizer()
                engine = ReloadableEngine(tok, "Hello.")
                engine.running = not lazy
                engine.can_report_memory = True
                report = {"memories": [{"id": "gpu0", "name": "GPU", "total": 4096, "free": 1024,
                          "unified": False, "uses": {"weights": 3072}, "paused_count": None, "paused_budget": None}],
                          "settings": {"rows": 4}}
                engine.memory = mock.Mock(return_value=report)
                svc = Service(engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
                if lazy:
                    engine.memory.assert_not_called()
                    svc.load()
                self.assertEqual(engine.memory.call_count, 1)
                svc.CONTEXT_WAIT_S = 0
                with svc.fifo:
                    reading = svc.memory_view()
                self.assertTrue(reading["busy"])
                self.assertEqual(reading["memories"][0]["uses"], {"weights": 3072})
                self.assertEqual(reading["settings"], {"rows": 4})
                self.assertIsNotNone(reading["at"])
                svc.status.update(busy=True, first_token=None)
                reading = svc.memory_view()
                self.assertTrue(reading["busy"])
                self.assertEqual(reading["settings"], {"rows": 4})
                self.assertEqual(engine.memory.call_count, 1)
                svc.live_reqs[1] = ({"busy": True, "first_token": time.time()}, [])
                self.assertFalse(svc.memory_view()["busy"])
                self.assertEqual(engine.memory.call_count, 2)
                svc.live_reqs.clear()
                engine.memory.reset_mock()
                svc.status.update(busy=False)
                engine.unload()
                engine.memory.return_value = {**report, "settings": {"rows": 6}}
                with svc.fifo:
                    svc.ensure_loaded()
                    reading = svc.memory_view()
                self.assertEqual(engine.memory.call_count, 1)
                self.assertEqual(reading["settings"], {"rows": 6})

    def test_monitor_failure_does_not_prevent_loading(self):
        tok = ByteTokenizer()
        engine = ReloadableEngine(tok, "Hello.")
        engine.can_report_memory = True
        engine.memory = mock.Mock(side_effect=EngineDied("report unavailable"))
        svc = Service(engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.assertTrue(svc.loaded())
        self.assertIsNone(svc.memory_last["at"])

    def test_failed_restart_releases_initialization_gate(self):
        tok = ByteTokenizer()
        engine = ReloadableEngine(tok, "Hello.")
        svc = Service(engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        engine.unload()
        engine.restart = mock.Mock(side_effect=EngineDied("cannot start"))
        with self.assertRaises(EngineDied), svc.fifo:
            svc.ensure_loaded()
        self.assertFalse(svc.initializing_engine)


class Monitor(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = ReloadableEngine(self.tok, "Thought.</think>\nHello <script>alert(1)</script>.", max_context=16384)
        self.svc = Service(self.engine, self.tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.svc.api_monitor = True                       # opt-in ("api_monitor": true / --api-monitor)
        self.httpd, self.base = start_http(self, self.svc)

    def request(self, path, body=None, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            raw = response.read().decode()
            return response.status, json.loads(raw) if "application/json" in response.headers.get("Content-Type", "") else raw

    def chat(self, api="openai", **kwargs):
        # Anthropic thinks only when asked (#278): "thinking" there, reasoning_effort for OpenAI
        think = {"thinking": {"type": "enabled", "budget_tokens": 1024}} if api == "anthropic" else             {"reasoning_effort": "low"}
        return self.request("/v1/messages" if api == "anthropic" else "/v1/chat/completions",
                            {"messages": [{"role": "user", "content": "Say hello."}],
                             "max_tokens": 256, **think, **kwargs})

    def latest(self):
        summary = self.request("/api/requests")[1]["requests"][0]
        return self.request("/api/requests?id=" + summary["id"])[1]

    def test_output_and_reasoning_for_both_dialects_and_stream_modes(self):
        for api in ("openai", "anthropic"):
            for stream in (False, True):
                with self.subTest(api=api, stream=stream):
                    code, reply = self.chat(api, stream=stream)
                    self.assertEqual(code, 200)
                    record = self.latest()
                    self.assertEqual(record["state"], "completed")
                    self.assertEqual(record["http_status"], 200)
                    self.assertIn("Say hello.", record["input"])
                    self.assertIn("<script>alert(1)</script>", record["output"])
                    self.assertIn("Thought.", record["reasoning"])
                    self.assertGreater(record["usage"].get("completion_tokens", record["usage"].get("output_tokens", 0)), 0)
                    self.assertGreaterEqual(record["wallclock_s"], record["first_token_s"])
                    self.assertNotIn("_clock", record)
                    if not stream:
                        self.assertEqual(json.loads(record["response"]), reply)
                    else:
                        self.assertNotIn("response", record)

    def test_error_and_engine_death_are_not_successes(self):
        # a request the server refuses is an error record: the strict mode (fit_max_tokens off) refuses a max_tokens
        # that overshoots the context, the default shortens it (strata_server/test_server.py MaxTokens)
        self.svc.fit_max_tokens = False
        self.assertEqual(self.chat(max_tokens=20000)[0], 400)
        self.svc.fit_max_tokens = True
        self.assertEqual(self.latest()["state"], "error")
        for api in ("openai", "anthropic"):
            for stream in (False, True):
                with self.subTest(api=api, stream=stream), mock.patch.object(self.engine, "generate", side_effect=EngineDied("stopped")):
                    self.chat(api, stream=stream)
                    record = self.latest()
                    self.assertEqual(record["state"], "error")
                    self.assertIn("stopped", record["error"]["message"])
                    self.assertEqual(record["output"], "")
                    self.assertNotIn("timings", record)

    def test_history_is_authorized_and_does_not_capture_headers(self):
        self.svc.api_key = "secret"
        auth = {"Authorization": "Bearer secret"}
        self.assertEqual(self.request("/api/requests")[0], 401)
        self.assertEqual(self.request("/api/requests?id=missing")[0], 401)
        self.assertEqual(len(self.svc.api_requests), 0)
        self.assertEqual(self.request("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}, auth)[0], 200)
        summary = self.request("/api/requests", headers=auth)[1]["requests"][0]
        self.assertNotIn("input", summary)
        self.assertNotIn("output", summary)
        self.assertEqual(self.request("/api/requests?id=" + summary["id"])[0], 401)
        detail = self.request("/api/requests?id=" + summary["id"], headers=auth)[1]
        self.assertNotIn("secret", json.dumps(detail))

    def test_capture_is_bounded_and_eviction_returns_404(self):
        huge = "x" * 262200
        first = self.svc.begin_request("/v1/chat/completions", {"messages": [{"content": huge}]})
        self.assertEqual(len(first["input"]), 262144)
        self.assertTrue(first["input_truncated"])
        for _ in range(100):
            self.svc.begin_request("/v1/chat/completions", {"messages": []})
        self.assertEqual(len(self.request("/api/requests")[1]["requests"]), 100)
        self.assertEqual(self.request("/api/requests?id=" + first["id"])[0], 404)
        # Exercise output capture without making a mock engine emit 262K tokens.
        handler = object.__new__(self.httpd.RequestHandlerClass)
        handler.record = self.svc.api_requests[-1]
        chunk = {"choices": [{"delta": {"content": huge, "reasoning_content": huge}}]}
        list(handler._capture((item for item in [chunk]), "openai"))
        for field in ("output", "reasoning"):
            self.assertEqual(len(handler.record[field]), 262144)
            self.assertTrue(handler.record[field + "_truncated"])

    def test_queue_wait_and_loading_are_attributed_separately(self):
        self.engine.unload()
        entered, release = threading.Event(), threading.Event()
        original = self.engine.restart
        def restart():
            entered.set()
            self.assertTrue(release.wait(5))
            original()
        with mock.patch.object(self.engine, "restart", side_effect=restart), ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.chat)
            self.assertTrue(entered.wait(5))
            second = pool.submit(self.chat)
            deadline = time.monotonic() + 5
            while len(self.svc.api_requests) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(len(self.svc.api_requests), 2)
            history = self.request("/api/requests")[1]
            self.assertEqual({r["state"] for r in history["requests"]}, {"queued", "loading"})
            self.assertFalse(history["loaded"])
            self.assertTrue(history["auto_load"])
            time.sleep(0.03)
            release.set()
            self.assertEqual(first.result()[0], 200)
            self.assertEqual(second.result()[0], 200)
        records = list(self.svc.api_requests)
        self.assertGreater(records[0]["load_s"], 0.02)
        self.assertGreater(records[1]["queue_s"], 0.02)
        self.assertLess(records[1]["load_s"], records[0]["load_s"])
        self.assertTrue(all(r["state"] == "completed" for r in records))

    def test_metrics_name_the_request_whose_prompt_is_read(self):
        reading, release = threading.Event(), threading.Event()
        generate = self.engine.generate
        def held(*args, **kwargs):
            reading.set()
            self.assertTrue(release.wait(5))
            yield from generate(*args, **kwargs)
        self.engine.reused = 5
        with mock.patch.object(self.engine, "generate", side_effect=held), ThreadPoolExecutor(max_workers=1) as pool:
            reply = pool.submit(self.chat)
            self.assertTrue(reading.wait(5))
            live = self.request("/metrics")[1]["live"]
            release.set()
            self.assertEqual(reply.result()[0], 200)
        self.assertEqual((live["state"], live["request"], live["prompt_reused"]),
                         ("reading", self.svc.api_requests[-1]["id"], 5))
        # parallel requests: the one in a slot that reads, not the newest
        running = [{"request_id": "a", "monitor_id": "m1"}, {"request_id": "b", "monitor_id": "m2"}]
        slots = [{"state": "reading", "request_id": "a", "reused": 7}, {"state": "generating", "request_id": "b"}]
        self.assertEqual(self.svc._find_reading_request(running[-1], running, slots),
                         {"request": "m1", "prompt_reused": 7})

    def test_metrics_list_the_phase_of_each_generating_request(self):
        self.svc.live_reqs.update({k: ({"busy": True, "generated": 1, "first_token": first, "phase": phase}, [])
                                   for k, first, phase in ((1, 1.0, "thinking"), (2, None, "reading the prompt"),
                                                           (3, 2.0, "writing a tool call: Bash"))})
        self.addCleanup(self.svc.live_reqs.clear)
        self.assertEqual(self.svc.metrics()["live"]["phases"], ["thinking", "writing a tool call: Bash"])

    def test_metrics_keep_the_last_error_until_a_request_ends_well(self):
        def failing(*args, **kwargs):
            raise ValueError("the prompt does not fit")
            yield
        with mock.patch.object(self.engine, "generate", side_effect=failing):
            self.assertGreaterEqual(self.chat()[0], 400)
        self.assertEqual(self.svc.metrics()["live"]["error"]["message"], "the prompt does not fit")
        self.assertEqual(self.chat()[0], 200)
        self.assertIsNone(self.svc.metrics()["live"]["error"])

    def test_requests_the_engine_holds_back_count_as_queued(self):
        slots = [{"slot": 0, "state": "generating"}, {"slot": 1, "state": "queued"}, {"slot": 2, "state": "queued"}]
        with mock.patch.object(self.engine, "batch", 2, create=True), \
             mock.patch.object(self.engine, "slots_view", return_value=slots, create=True):
            self.assertEqual(self.svc.metrics()["live"]["queued"], 2)

    def test_context_of_an_engine_that_cannot_list_it(self):
        code, body = self.request("/api/context")
        self.assertEqual(code, 200)
        self.assertEqual((body["available"], body["reason"]), (False, "engine"))   # an engine before "conversations"
        self.assertEqual(self.request("/api/context?id=main")[0], 404)
        self.assertTrue(self.request("/health")[1]["api_monitor"])

    def holds(self, held):
        """The engine answers LIST_CONVERSATIONS / GET_CONTEXT_WINDOWS from `held` ({id: (place, state, tokens,
        bytes)}; a session holds up to 4096 tokens, 100 bytes each), and remembers what it was asked."""
        self.asked = []
        self.engine.can_list = True

        def listing(marker=(), preview=0):
            self.asked.append(("list", list(marker), preview))
            out = []
            for cid, (place, state, ids, size) in held.items():
                at = next((i + len(marker) for i in range(len(ids)) if marker and ids[i:i + len(marker)] == list(marker)), None)
                cap = 4096 if place == "vram" else len(ids)
                out.append({"id": cid, "place": place, "state": state, "count": len(ids), "capacity": cap, "bytes": size,
                            "used": 100 * len(ids) if place == "vram" else size,
                            "preview": ids[at:at + preview] if at is not None and preview else []})
            return out
        self.engine.list_conversations = listing
        self.engine.context_windows = lambda ids: self.asked.append(("get", list(ids))) or \
            {i: array("i", held[i][2]) if i in held else None for i in ids}

    def test_context_hides_unused_addresses_and_names_solo_like_batch(self):
        self.holds({"main": ("vram", "holding", [1, 2], 100),
                    "slot0": ("vram", "empty", [], 0),
                    "slot1": ("vram", "holding", [3], 100)})
        entries = self.request("/api/context")[1]["entries"]
        self.assertEqual([(e["id"], e["name"]) for e in entries],
                         [("main", "Conversation 1"), ("slot1", "Conversation 2")])

    def test_context_lists_by_name_and_place_then_reads_one(self):
        # "é" is two bytes, one token each: the first token's piece is empty, the second completes it
        ids = [256, *b"system\nbe brief", 257, 256, *"user\nhi é".encode(), 257]
        self.holds({"main": ("vram", "holding", ids, 409600), "parked4": ("ram", "parked", [1, 2], 1536)})
        code, body = self.request("/api/context")
        self.assertEqual(code, 200)
        self.assertEqual((body["available"], body["busy"], body["max_context"]), (True, False, self.svc.reported_ctx()))
        # the user-turn header asks for the start of the first user message
        self.assertEqual(self.asked, [("list", [256, *b"user\n"], self.svc.PREVIEW_TOKENS)])
        session, parked = body["entries"]
        self.assertEqual({k: session[k] for k in ("id", "name", "kind", "where", "state", "preview")},
                         {"id": "main", "name": "Conversation 1", "kind": "session", "where": "vram", "state": "holding",
                          "preview": "hi é"})
        self.assertEqual({k: session[k] for k in ("count", "capacity", "bytes", "used")},
                         {"count": len(ids), "capacity": 4096, "bytes": 409600, "used": 100 * len(ids)})
        self.assertEqual({k: parked[k] for k in ("id", "name", "kind", "where", "count", "capacity", "bytes", "used",
                                                 "preview")},
                         {"id": "parked4", "name": "Parked conversation 1 of 1", "kind": "parked", "where": "ram",
                          "count": 2, "capacity": 2, "bytes": 1536, "used": 1536, "preview": ""})
        self.assertNotIn("tokens", session)              # the list carries no context windows
        code, got = self.request("/api/context?id=main")
        self.assertEqual((code, got["busy"]), (200, False))
        self.assertEqual(self.asked[-1], ("get", ["main"]))
        self.assertEqual(got["tokens"], ids)
        self.assertEqual(got["pieces"][-4:], [" ", "", "é", "<|im_end|>"])
        self.assertEqual(self.request("/api/context?id=parked9")[0], 404)   # no longer held
        self.assertEqual(self.request("/api/context?id=x%20y")[0], 400)

    def test_context_while_a_request_has_the_engine(self):
        self.holds({"main": ("vram", "holding", [5, 6], 0), "slot0": ("vram", "holding", [7], 0)})
        self.svc.CONTEXT_WAIT_S = 0.05
        with self.svc.fifo:                              # a request is running: nothing read yet
            body = self.request("/api/context")[1]
        self.assertEqual((body["busy"], body["at"], body["entries"]), (True, None, []))
        first = self.request("/api/context")[1]
        self.assertEqual(self.request("/api/context?id=main")[1]["tokens"], [5, 6])
        with self.svc.fifo:                              # ... then the list and the context window read last
            body = self.request("/api/context")[1]
            again = self.request("/api/context?id=main")
            other = self.request("/api/context?id=slot0")
        self.assertEqual((body["busy"], body["at"], body["entries"]), (True, first["at"], first["entries"]))
        self.assertEqual((again[0], again[1]["busy"], again[1]["tokens"]), (200, True, [5, 6]))
        self.assertEqual(other[0], 503)                  # not read before: it waits for the engine's turn

    def test_memory(self):
        self.assertEqual(self.request("/v1/memory")[1]["reason"], "engine")   # an engine before "memory"
        self.engine.can_report_memory = True
        self.engine.memory = lambda: {"memories": [
            {"id": "gpu0", "name": "GPU", "total": 1000, "free": 100, "unified": False, "paused_count": None,
             "paused_budget": None, "uses": {"experts": 600, "conversations": 200}},
            {"id": "ram", "name": None, "total": None, "free": None, "unified": False, "paused_count": 1,
             "paused_budget": 50, "uses": {"paused": 20}}], "settings": {"context": 4096}}
        self.enterContext(mock.patch("strata_base.memory.read_ram", return_value=mock.Mock(total=5000, available=3000)))
        code, body = self.request("/v1/memory")
        self.assertEqual((code, body["available"], body["busy"], body["settings"]),
                         (200, True, False, {"context": 4096}))
        self.assertEqual(body["machine"]["ram_total"], 5000)
        gpu, ram = body["memories"]
        self.assertEqual(gpu["other"], 100)              # in use there, not named by the engine
        self.assertEqual((ram["total"], ram["free"], ram["other"]), (5000, 3000, 1980))   # the machine's RAM
        self.svc.CONTEXT_WAIT_S = 0.05
        with self.svc.fifo:                              # a request is running: the last answer, marked busy
            again = self.request("/v1/memory")[1]
        self.assertEqual((again["busy"], again["at"], again["memories"]), (True, body["at"], body["memories"]))
        web = Path(__file__).parent / "web"            # the Memory panel draws it
        html, js = (web / "index.html").read_text(encoding="utf-8"), (web / "app.js").read_text(encoding="utf-8")
        for el in ("mem-card", "mem-panel"):
            self.assertIn(f'id="{el}"', html)
            self.assertIn(f'"{el}"', js)
        self.assertIn('"v1/memory"', js)
        self.assertIn('import("./memory-panel.js")', js)  # shared with the desktop app (apps/host, MemoryPanel.tsx)
        self.assertIn('href="web/memory-panel.css"', html)
        for name, text in (("memory-panel.js", "export function drawMemoryPanel"), ("memory-panel.css", ".mem-bar ")):
            code, body = self.request("/web/" + name)
            self.assertEqual(code, 200)
            self.assertIn(text, body)

    def test_memory_of_a_gpu_whose_memory_is_system_ram(self):
        """Strix Halo: the GPU's bar is its BIOS share, and what it borrows beyond the share counts against the RAM."""
        self.engine.can_report_memory = True
        self.engine.memory = lambda: {"memories": [
            {"id": "gpu0", "name": "Radeon", "total": 9000, "free": 4000, "unified": True, "paused_count": None,
             "paused_budget": None, "uses": {"experts": 600, "conversations": 200}},
            {"id": "ram", "name": None, "total": None, "free": None, "unified": False, "paused_count": None,
             "paused_budget": None, "uses": {}}], "settings": {}}
        split = {"share": 1000, "share_used": 900, "in_share": 700, "borrowed": 250, "borrow_limit": 4000,
                 "borrowed_by": "engine"}
        self.enterContext(mock.patch("strata_base.memory.read_ram", return_value=mock.Mock(total=5000, available=3000)))
        with mock.patch("strata_common.telemetry.read_shared_gpu_memory", return_value=split) as read:
            gpu, ram = self.request("/v1/memory")[1]["memories"]
        self.assertEqual(read.call_args.args[:2], ("Radeon", 0))
        self.assertEqual((gpu["total"], gpu["free"], gpu["in_share"], gpu["borrowed"], gpu["other"]),
                         (1000, 100, 700, 250, 200))            # other: the share's use beyond the engine's 700
        self.assertEqual((gpu["runtime_total"], gpu["runtime_free"]), (9000, 4000))
        self.assertEqual((ram["gpu_borrowed"], ram["other"]), (250, 1750))
        split.update(in_share=None, borrowed_by="device")      # Linux: no per-process split
        with mock.patch("strata_common.telemetry.read_shared_gpu_memory", return_value=split):
            gpu = self.request("/v1/memory")[1]["memories"][0]
        self.assertEqual((gpu["in_share"], gpu["other"]), (800, 100))   # at most what the share holds
        # the engine's page-locked RAM, which the GPU maps, is in its borrowed use but drawn as the engine's RAM
        memory = self.engine.memory()
        memory["memories"][1].update(mapped=200, uses={"experts": 200})
        self.engine.memory = lambda: memory
        with mock.patch("strata_common.telemetry.read_shared_gpu_memory", return_value=split):
            gpu, ram = self.request("/v1/memory")[1]["memories"]
        self.assertEqual((gpu["borrowed"], ram["gpu_borrowed"], ram["other"]), (50, 50, 1750))

    def test_context_needs_the_key(self):
        self.svc.api_key = "k"
        self.assertEqual(self.request("/api/context")[0], 401)
        self.assertEqual(self.request("/api/context", headers={"Authorization": "Bearer k"})[0], 200)

    def test_page_and_relative_assets_are_served(self):
        code, html = self.request("/api-monitor")
        self.assertEqual(code, 200)
        self.assertIn('src="/web/monitor.js"', html)
        code, script = self.request("/web/monitor.js")
        self.assertEqual(code, 200)
        self.assertIn("textContent", script)
        self.assertIn("panel.drawConversations(", script)
        self.assertNotIn("innerHTML = e.preview", script)
        for el in ("ctx-view", "ctx-refresh", "ctx-save", "ctx-binary", "ctx-slot", "ctx-sum", "ctx-note", "ctx-pager",
                   "ctx-prev", "ctx-next", "ctx-range"):
            self.assertIn(f'id="{el}"', html)
            self.assertIn(f'"{el}"', script)

    def test_the_main_page_links_to_the_monitor(self):
        web = Path(__file__).parent / "web"
        html, js = (web / "index.html").read_text(encoding="utf-8"), (web / "app.js").read_text(encoding="utf-8")
        for el in ("api-monitor-link", "req-monitor"):
            self.assertIn(f'id="{el}"', html)
            self.assertIn(f'"{el}"', js)
        self.assertIn('href="api-monitor"', html)
        self.assertEqual(self.request("/web/../server.py")[0], 404)


class EngineConversations(unittest.TestCase):
    """StrataEngine.list_conversations and context_windows: the engine's LIST_CONVERSATIONS and GET_CONTEXT_WINDOWS."""

    @staticmethod
    def bare(lines):
        """A StrataEngine without a process: its output lines are queued up front."""
        import io
        import queue
        from types import SimpleNamespace
        from strata_common.engine_client import StrataEngine
        engine = StrataEngine.__new__(StrataEngine)
        engine.info = {}
        engine.proc = SimpleNamespace(stdin=io.StringIO(), poll=lambda: None)
        engine.lines, engine.wlock, engine.ended = queue.Queue(), threading.Lock(), False
        for line in lines:
            engine.lines.put(line + "\n")
        return engine

    def test_the_list_and_its_previews(self):
        engine = self.bare(["CONVERSATION main vram holding 3 204800 3092376453 1048576 40,41",
                            "NOTE sessions: something else", "CONVERSATION slot0 vram decoding 1 204800 3092376453 65536",
                            "CONVERSATION parked4 disk parked 2 2 1535867580 1535867580", "END_CONVERSATIONS"])
        got = engine.list_conversations([256, 117], 2)
        self.assertEqual(engine.proc.stdin.getvalue(), "LIST_CONVERSATIONS 256,117 2\n")
        self.assertEqual(got, [
            {"id": "main", "place": "vram", "state": "holding", "count": 3, "capacity": 204800, "bytes": 3092376453,
             "used": 1048576, "preview": [40, 41]},
            {"id": "slot0", "place": "vram", "state": "decoding", "count": 1, "capacity": 204800, "bytes": 3092376453,
             "used": 65536, "preview": []},
            {"id": "parked4", "place": "disk", "state": "parked", "count": 2, "capacity": 2, "bytes": 1535867580,
             "used": 1535867580, "preview": []}])
        engine = self.bare(["END_CONVERSATIONS"])
        engine.list_conversations()                       # no marker: no previews asked for
        self.assertEqual(engine.proc.stdin.getvalue(), "LIST_CONVERSATIONS\n")

    def test_context_windows(self):
        engine = self.bare(["CONTEXT_WINDOW parked4 2 1,2", "CONTEXT_WINDOW slot1 0", "CONTEXT_WINDOW parked3 gone",
                            "END_CONTEXT_WINDOWS"])
        got = engine.context_windows(["parked4", "slot1", "parked3"])
        self.assertEqual(engine.proc.stdin.getvalue(), "GET_CONTEXT_WINDOWS parked4 slot1 parked3\n")
        self.assertEqual({k: v if v is None else v.tolist() for k, v in got.items()},
                         {"parked4": [1, 2], "slot1": [], "parked3": None})

    def test_memory_discovery_details(self):
        engine = self.bare(["MEMORY_DETAIL drafter_label Use%20DSpark%20drafter%20%281.2%20GiB%29",
                            "MEMORY_DETAIL kv_quantization FP8%20and%20FP4",
                            "SETTINGS has_drafter=1 suffix_draft=0", "END_MEMORY"])
        report = engine.memory()
        self.assertEqual(report["details"], {"drafter_label": "Use DSpark drafter (1.2 GiB)",
                                             "kv_quantization": "FP8 and FP4"})
        self.assertEqual(report["settings"]["has_drafter"], 1)

    def test_memory(self):
        engine = self.bare(["MEMORY gpu0 total=1000 free=300 unified=1 weights=100 experts=450 name=Radeon  8060S",
                            "NOTE something else", "MEMORY ram paused=300 paused_count=2 paused_budget=800",
                            "MEMORY ssd engram=70", "SETTINGS context=32768 concurrent=1", "END_MEMORY"])
        got = engine.memory()
        self.assertEqual(engine.proc.stdin.getvalue(), "GET_MEMORY_USE\n")
        none = {"allocated_pct": None, "fragmented_pct": None,   # no page pool: the old engines
                "setting": None, "smallest": None, "largest": None, "mapped": None}
        self.assertEqual(got, {"memories": [
            {"id": "gpu0", "name": "Radeon 8060S", "total": 1000, "free": 300, "unified": True, "paused_count": None,
             "paused_budget": None, **none, "uses": {"weights": 100, "experts": 450}},
            {"id": "ram", "name": None, "total": None, "free": None, "unified": False, "paused_count": 2,
             "paused_budget": 800, **none, "uses": {"paused": 300}},
            {"id": "ssd", "name": None, "total": None, "free": None, "unified": False, "paused_count": None,
             "paused_budget": None, **none, "uses": {"engram": 70}}], "settings": {"context": 32768, "concurrent": 1}})
        draft = self.bare(["MEMORY gpu0 experts=9 drafter_experts=7", "END_MEMORY"]).memory()
        self.assertEqual(draft["memories"][0]["uses"], {"experts": 9, "drafter_experts": 7})
        pool = self.bare(["MEMORY gpu0 experts=9 pool_pages=800 allocated_pages=780 fragmented_pages=3 setting=500 "
                          "smallest=200 largest=900 name=card", "SETTINGS", "END_MEMORY"]).memory()["memories"][0]
        self.assertEqual((pool["allocated_pct"], pool["fragmented_pct"], pool["uses"]), (97.5, 0.38, {"experts": 9}))
        self.assertEqual((pool["setting"], pool["smallest"], pool["largest"]), (500, 200, 900))
        self.assertEqual(self.bare(["SETTINGS", "END_MEMORY"]).memory(), {"memories": [], "settings": {}})
        got = self.bare(["CONVERSATION slot0 vram decoding 3 4096 1000 900", "SETTINGS rows=8", "END_MEMORY"]).memory()
        self.assertEqual(got["conversations"], [{"id": "slot0", "place": "vram", "state": "decoding", "count": 3,
                                                 "capacity": 4096, "bytes": 1000, "used": 900, "preview": []}])

    def test_change_settings(self):
        from strata_common.engine_client import SettingsRefused
        engine = self.bare(["MEMORY gpu0 setting=100", "SETTINGS rows=4", "END_MEMORY"])
        got = engine.change_settings({"rows": 4, "drafter": "on"})
        self.assertEqual(engine.proc.stdin.getvalue(), "CHANGE_SETTINGS rows=4 drafter=on\n")
        self.assertEqual((got["memories"][0]["setting"], got["settings"]), (100, {"rows": 4}))
        self.assertNotIn("shortfall", got)
        got = self.bare(["MEMORY ram setting=300 mapped=300", "SETTINGS", "SHORTFALL the RAM's pool: out of memory",
                         "END_MEMORY"]).change_settings({"memory_mib.ram": 1})
        self.assertEqual((got["memories"][0]["mapped"], got["shortfall"]), (300, "the RAM's pool: out of memory"))
        with self.assertRaises(SettingsRefused) as refused:
            self.bare(["ERR the GPU's pool would keep 2.00 GiB"]).change_settings({"memory_mib.gpu0": 2048})
        self.assertEqual(str(refused.exception), "the GPU's pool would keep 2.00 GiB")

    def test_an_engine_that_refuses_or_ends(self):
        from strata_common.engine_client import EngineDied
        with self.assertRaises(EngineDied):
            self.bare(["ERR unknown command"]).list_conversations()
        engine = self.bare(["CONVERSATION main vram empty 0 4096 409600 0"])
        engine.lines.put(None)                            # its output closed
        with self.assertRaises(EngineDied):
            engine.list_conversations()


class ConversationCacheCard(unittest.TestCase):
    """#596: /metrics' "conversation_cache": the parked conversations from the engine's log lines, the budget and
    slots from its INFO line, and how much of the prompts the cache gave back."""

    PARK = ("strata serve: conversation cache: parked 5000 tokens in 12.0 ms; parked=1 bytes=104857600 evictions=0 "
            "snapshot_bytes=104857600 reused_kv_bytes=0\n")
    RESTORE = "strata serve: conversation cache: restored 4000 tokens (exact) in 30.0 ms; parked=2 bytes=209715200\n"

    def test_the_log(self):
        from strata_server.server import ConvCacheLog
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "strata.log"
            log.write_text("an earlier run\n" + self.PARK, encoding="utf-8")
            start = log.stat().st_size                 # this run starts here: the earlier run's park is not counted
            c = ConvCacheLog()
            self.assertEqual(c.poll(str(log), start)["parked"], 0)
            with open(log, "a", encoding="utf-8") as f:
                f.write("strata serve: something else\n" + self.PARK + self.RESTORE + "strata serve: conversation ca")
            st = c.poll(str(log), start)
            self.assertEqual((st["parked"], st["bytes"], st["parks"], st["restores"]), (2, 209715200, 1, 1))
            self.assertEqual((st["last_event"], st["last_tokens"]), ("restored", 4000))
            with open(log, "a", encoding="utf-8") as f:   # the cut line is read once it is whole
                f.write("che: dropped 1 superseded copy of this conversation; parked=1\n")
            self.assertEqual(c.poll(str(log), start)["parked"], 1)
            self.assertEqual(c.poll(str(log), log.stat().st_size)["parked"], 0)   # the engine started again
            self.assertEqual(c.poll(None, None)["parked"], 0)

    def test_metrics(self):
        tok = ByteTokenizer()
        engine = MockEngine(tok, "Thought.</think>\nHello.", max_context=4096)
        engine.info = {"conversation_cache_mib": 8192, "conversation_cache_slots": 4}
        svc = Service(engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
                c = json.loads(r.read())["conversation_cache"]
            self.assertEqual((c["enabled"], c["budget_mib"], c["slots"], c["parked"], c["requests"]),
                             (True, 8192, 4, 0, 0))
            req = urllib.request.Request(base + "/v1/chat/completions", headers={"Content-Type": "application/json"},
                                         data=json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode())
            with urllib.request.urlopen(req, timeout=10) as r:
                r.read()
            with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
                c = json.loads(r.read())["conversation_cache"]
            self.assertEqual(c["requests"], 1)
            self.assertGreater(c["last_prompt"], 0)
            engine.info = {}
            with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
                self.assertFalse(json.loads(r.read())["conversation_cache"]["enabled"])   # off: the reuse lines only
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_the_card_is_on_the_page(self):
        web = Path(__file__).parent / "web"
        html, js = (web / "index.html").read_text(encoding="utf-8"), (web / "app.js").read_text(encoding="utf-8")
        for el in ("cc-card", "cc-slots-text", "cc-mem-text", "cc-facts", "cc-note"):
            self.assertIn(f'id="{el}"', html)
            self.assertIn(f'"{el}"', js)


class SharedGpuMemory(unittest.TestCase):
    """The readers behind the BIOS division of a GPU whose memory is system RAM (telemetry.read_shared_gpu_memory)."""

    def test_windows_counters_are_summed_for_the_engine_and_the_adapter(self):
        from strata_common.telemetry import sum_gpu_memory_counters
        luid = "0x00000000_0x00014277"
        dedicated = [("pid_42_luid_0x00000000_0x00014277_phys_0", 700), ("pid_42_luid_0x00000000_0x00014277_phys_1", 5),
                     ("pid_42_luid_0x00000000_0x000166C2_phys_0", 99),      # another adapter
                     ("pid_420_luid_0x00000000_0x00014277_phys_0", 88)]     # another process whose pid starts alike
        shared = [("pid_42_luid_0x00000000_0x00014277_phys_0", 250), ("pid_7_luid_0x00000000_0x00014277_phys_0", 3)]
        adapter = [("luid_0x00000000_0x00014277_phys_0", 900), ("luid_0x00000000_0x000166F4_phys_0", 4)]
        self.assertEqual(sum_gpu_memory_counters((dedicated, shared, adapter), 42, luid), (705, 250, 900))
        self.assertEqual(sum_gpu_memory_counters((dedicated, shared, adapter), None, luid), (None, None, 900))

    def test_linux_reads_the_vram_and_gtt_files(self):
        import tempfile
        from strata_common import telemetry
        with tempfile.TemporaryDirectory() as d:
            node = Path(d) / "class/kfd/kfd/topology/nodes/1"
            node.mkdir(parents=True)
            (node / "properties").write_text("simd_count 80\ngfx_target_version 110501\ndrm_render_minor 128\n")
            dev = Path(d) / "class/drm/renderD128/device"
            dev.mkdir(parents=True)
            for k, v in (("vram_total", 512 << 20), ("vram_used", 300 << 20), ("gtt_total", 60 << 30),
                         ("gtt_used", 40 << 30)):
                (dev / f"mem_info_{k}").write_text(f"{v}\n")
            with mock.patch.object(telemetry.os, "name", "posix"):
                got = telemetry.read_shared_gpu_memory("AMD Radeon", 0, 123, sysfs=d)
                self.assertIsNone(telemetry.read_shared_gpu_memory("AMD Radeon", 1, 123, sysfs=d))
        self.assertEqual(got, {"share": 512 << 20, "share_used": 300 << 20, "in_share": None, "borrowed": 40 << 30,
                               "borrow_limit": 60 << 30, "borrowed_by": "device"})


if __name__ == "__main__":
    unittest.main()
