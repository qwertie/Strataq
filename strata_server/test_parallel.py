"""#465: "parallel": N - several requests decode together in the engine's batch slots (--batch N).

A fake engine (a Python script speaking the engine's GEN / BGEN / BT / BDONE / BADM / BSTOP lines) runs behind the
real StrataEngine and Service, so the server's side is tested without a GPU: requests up to N run at once, more
wait for a slot, /metrics shows the slots, each request's history row is its own, and a conversation's next turn
goes back to the slot that holds it."""
import json
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from strata_server.frontend import ChatTemplate
from strata_server.server import ByteTokenizer, Service, serve
from strata_common.engine_client import StrataEngine, engine_args, parallel_args
from strata_control import bench
from strata_common.client import Endpoint

# The fake engine: one token every STEP seconds per active slot (a "window" serves every active slot at once);
# GEN (solo) streams T lines; BGEN reads the prompt (one T line, DONE), answers BADM and continues in the slot.
# Every reply is the bytes of "ok, done." then <|im_end|> (ByteTokenizer: 257).
FAKE_BATCH = r'''import queue, sys, threading, time
args = sys.argv[1:]
slots = int(args[args.index("--batch") + 1]) if "--batch" in args else 0
fit = int(args[args.index("--fit") + 1]) if "--fit" in args else slots
fail = "--fail-window" in args
groups = int(args[args.index("--says-groups") + 1]) if "--says-groups" in args else 0   # INFO batch_groups=G (--batch-groups auto)        # #997: the first window over two slots fails
STEP = 0.02
CH = 32
lines, stop = queue.Queue(), threading.Event()
def reader():
    for l in sys.stdin:
        l = l.strip()
        if l == "STOP":
            stop.set()
        else:
            lines.put(l)
    lines.put(None)
threading.Thread(target=reader, daemon=True).start()
print("INFO engine=0.1.39" + (f" batch_slots={fit}" if fit >= 2 else "") +
      (" slot_cache=1" if "--slotcache" in args else "") + (f" batch_groups={groups}" if groups > 1 else ""), flush=True)
if "--fake-stage-plan" in args:
    print("strata serve: layer split: layers 0-11 (CUDA0), 12-23 (CUDA1), 24-35 (CUDA2), "
          "36-47 (CUDA3), one hand-off per window", file=sys.stderr, flush=True)
LISTS = "--lists" in args   # READY ... conversations: it answers LIST_CONVERSATIONS and GET_CONTEXT_WINDOWS
PREEMPT = "--preempt" in args   # READY ... preempt: a BGEN of a higher priority= ends a busy slot's request
# READY ... memory settings: it answers GET_MEMORY_USE and CHANGE_SETTINGS (a GPU pool under 1000 MiB is refused; a
# RAM size gets a SHORTFALL)
concurrent = 1
SETTINGS = "--settings" in args
print("READY 4096 stop" + (" conversations" if LISTS else "") + (" preempt" if PREEMPT else "") +
      (" memory settings" if SETTINGS else ""), flush=True)
pool_mib, rows = 2000, 4     # (--settings) the GPU pool's size and the rows setting
priority = {}        # (--preempt) slot -> the priority of the request it decodes
main_held, slot_held, parked = [], {}, []   # (--lists) what the sessions hold, the conversations parked
LONG = list(b"LONGREPLY")
def continued(ids):        # (the whole reply, how much of it the prompt already ends with: a request continued)
    ids = [int(x) for x in ids]
    long_one = any(ids[i:i + len(LONG)] == LONG for i in range(len(ids))) or \
        b"Cancellation benchmark:" in bytes(int(x) & 255 for x in ids)
    R = list(b"ok, " + b"la " * 15 + b"done." if long_one else b"ok, done.") + [257]
    return R, max(k for k in range(len(R)) if k == 0 or ids[-k:] == R[:k])
def reply(ids):            # the rest of the reply after what the prompt already ends with
    R, k = continued(ids)
    return R[k:]
active = {}          # slot -> [tokens left, max_new, produced]
stopped = set()      # BSTOPped slots: they end "cancel"
# #879: a prompt with NFSOLO / NFADMIT ends its first GEN / BGEN with DONE ... nonfinite (nothing emitted); one with
# NFSLOT ends its slot with BDONE ... nonfinite after its second token
fired, nf_slots = set(), set()
def window():        # one batch window: every active slot one token
    if fail and len(active) >= 2:     # as the engine: the reason on stdout, then exit code 1
        print("ERR verify batch: layer 34 never rang (graph finished)", flush=True)
        sys.exit(1)
    processed = len(active)
    for b in sorted(active):
        left, max_new, produced = active[b]
        if b in nf_slots and produced >= 2:
            nf_slots.discard(b)
            print(f"BDONE {b} {produced} nonfinite 1.0", flush=True)
            del active[b]
            continue
        t = left.pop(0) if left else 257
        produced += 1
        print(f"BT {b} {t}", flush=True)
        if t != 257 and produced < max_new:
            slot_held[b].append(t)            # fed in the next window (the last token is not)
        if t == 257 or produced >= max_new:
            fin = 'stop' if t == 257 else 'cancel' if b in stopped else 'length'
            stopped.discard(b)
            print(f"BDONE {b} {produced} {fin} 1.0", flush=True)
            del active[b]
        else:
            active[b] = [left, max_new, produced]
    if "--fake-batch-sample" in args and processed and not active:
        print(f"BSTAT 1 {processed} {processed} 20", flush=True)
    if "--fake-stage-sample" in args and processed and not active:
        print("PSTAT 4 20000 16000 14000 12000 10000", flush=True)
log = open(args[args.index("--log") + 1], "a") if "--log" in args else None
PPK = int(args[args.index("--pp-every") + 1]) if "--pp-every" in args else 1
READF = "--readfield" in args   # DONE's 15th field: the prompt tokens the engine read (a continued prompt: the 1 it copies)
REUSE = "--reuse" in args     # DONE's reused field as the engine gives it: a continued prompt is all held but its last
def reused(ids):              # token (the request's own output included), a fresh one the first 3 tokens
    return 0 if not REUSE else len(ids) - 1 if continued(ids)[1] else 3
while True:
    try:
        line = lines.get(timeout=STEP if active else None)
    except queue.Empty:
        line = ""
    if line is None or line == "QUIT":
        break
    if line.startswith("BYIELD "):
        continue                              # too late: the read it was for has ended
    if line.startswith("BSTOP "):
        b = int(line.split()[1])
        if b in active:
            active[b][1] = 0
            stopped.add(b)
        continue
    if line.startswith(("GEN ", "BGEN ")):
        f = line.split()
        slot = int(f[1]) if f[0] == "BGEN" else None
        max_new = int(f[2] if f[0] == "BGEN" else f[1])
        ids = f[-1].split(",")
        if bytes(int(x) & 255 for x in ids).find(b"REFUSEME") >= 0:   # #1059: refused up front, then idle (no DONE)
            print("ERR refused: images are not enabled", flush=True)
            continue
        # #1994: the engine's own view of its slots differs: BUSYSLOT is refused once, BUSYALL always
        pbx = bytes(int(x) & 255 for x in ids)
        if slot is not None and (b"BUSYALL" in pbx or (b"BUSYSLOT" in pbx and "busyslot" not in fired)):
            fired.add("busyslot")
            print(f"ERR BGEN: no such free slot (--batch {slots}) or a bad max_new", flush=True)
            if log:
                log.write(f"REFUSED {slot}\n"); log.flush()
            continue
        if PREEMPT and slot is not None:
            p = next((int(k[len("priority="):]) for k in f[3:-1] if k.startswith("priority=")), 0)
            if slot in active and p <= priority[slot]:
                print(f"ERR BGEN: slot {slot} decodes a request of priority {priority[slot]}", flush=True)
                continue
            if slot in active:
                print(f"BDONE {slot} {active.pop(slot)[2]} preempted 1.0", flush=True)
                if log:
                    log.write(f"PREEMPT {slot}\n"); log.flush()
            priority[slot] = p
        if log:
            log.write(f"{f[0]} {slot} {len(ids)} {max(len(active), 0)}\n"); log.flush()
        pb = bytes(int(x) & 255 for x in ids)
        mark = b"NFSOLO" if f[0] == "GEN" else b"NFADMIT"
        if mark in pb and mark not in fired:
            fired.add(mark)
            print(f"DONE 0 {len(ids)} 5.0 0.0 nonfinite 0 0 0", flush=True)
            if slot is not None:
                print(f"BADM {slot} 0", flush=True)
            continue
        if slot is not None and b"PREEMPTADMIT" in pb and "preemptadmit" not in fired:
            fired.add("preemptadmit")
            print(f"DONE 0 {len(ids)} 5.0 0.0 preempted 0 0 {reused(ids)}", flush=True)
            print(f"BADM {slot} 0", flush=True)
            continue
        if slot is not None and b"NFSLOT" in pb:
            nf_slots.add(slot)
        toks = reply(ids)
        prompt = [int(x) for x in ids]
        if main_held and prompt[:len(main_held)] != main_held:   # another conversation takes the session
            parked.append(main_held)
        stop.clear()
        # the prompt read: CH tokens a chunk, a PP line each; a BYIELD <s> gives way at a chunk boundary
        given = None
        for q in range(CH, len(ids) - 1, CH):
            time.sleep(STEP / 2)
            if (q // CH) % PPK:   # --pp-every K: a PP line and a boundary every K chunks (the pipelined groups' pieces)
                continue
            print(f"PP {q} {len(ids)} 1 1", flush=True)
            held = []
            while True:
                try:
                    l2 = lines.get_nowait()
                except queue.Empty:
                    break
                if l2 and l2.startswith("BYIELD "):
                    ys = int(l2.split()[1])
                    if len(ids) - 1 - q > CH and (slot is None or ys == slot) and given is None:
                        given = (ys, q)
                elif l2 == "STOP":
                    stop.set()
                else:
                    held.append(l2)
            for l2 in held:
                lines.put(l2)
            if given:
                break
            if stop.is_set():
                break
            if active and (q // CH) % 4 == 0:     # the slots decode between the prompt's chunks (as the engine)
                window()
        if given:
            if log:
                log.write(f"YIELD {given[0]} {given[1]}\n"); log.flush()
            print(f"YIELDED {given[0]} {given[1]}", flush=True)
            print(f"DONE 0 {len(ids)} 5.0 0.0 cancel 0 0 {reused(ids)}", flush=True)
            if slot is not None:
                print(f"BADM {slot} 0", flush=True)
            continue
        n = (0 if "--zero-admit" in args else 1) if slot is not None else max_new
        out = 0
        for t in toks[:n]:
            if stop.is_set():
                break
            print(f"T {t}", flush=True)
            out += 1
            time.sleep(STEP)
        fin = "stop" if out and toks[out - 1] == 257 else ("cancel" if stop.is_set() else "length")
        main_held = prompt + toks[:max(out - 1, 0)]
        if slot is not None:
            slot_held[slot] = prompt + toks[:out]
        print(f"DONE {out} {len(ids)} 5.0 {out * STEP * 1000:.1f} {fin} 0 0 {reused(ids)}" +
              (f" 0 0 0 0 0.0 {len(ids) - reused(ids)}" if READF else ""), flush=True)
        if slot is not None:
            cont = fin == "length" and max_new > out
            if cont:
                active[slot] = [toks[out:], max_new, out]
            print(f"BADM {slot} {1 if cont else 0}", flush=True)
        continue
    if LISTS and line.startswith(("LIST_CONVERSATIONS", "GET_CONTEXT_WINDOWS")):
        held = [("main", "vram", "holding" if main_held else "empty", main_held, 4096, 409600)]
        held += [(f"slot{b}", "vram", "decoding" if b in active else "holding" if slot_held.get(b) else "empty",
                  slot_held.get(b, []), 4096, 409600) for b in range(fit)]
        held += [(f"parked{k + 1}", "ram", "parked", t, len(t), 100 * len(t)) for k, t in enumerate(parked)]
        f = line.split()
        ids = lambda t: (" " + ",".join(map(str, t))) if t else ""
        if f[0] == "LIST_CONVERSATIONS":
            marker, n = ([int(x) for x in f[1].split(",")], int(f[2])) if len(f) > 2 else ([], 0)
            for cid, place, state, t, cap, size in held:     # (a session's tokens take 100 bytes each here)
                at = next((i + len(marker) for i in range(len(t)) if marker and t[i:i + len(marker)] == marker), None)
                used = size if place != "vram" else 100 * len(t)
                print(f"CONVERSATION {cid} {place} {state} {len(t)} {cap} {size} {used}" +
                      (ids(t[at:at + n]) if at is not None else ""), flush=True)
            print("END_CONVERSATIONS", flush=True)
        else:
            known = {h[0]: h[3] for h in held}
            for cid in f[1:]:
                print(f"CONTEXT_WINDOW {cid} " + (f"{len(known[cid])}{ids(known[cid])}" if cid in known else "gone"), flush=True)
            print("END_CONTEXT_WINDOWS", flush=True)
        continue
    if SETTINGS and line.startswith(("GET_MEMORY_USE", "CHANGE_SETTINGS")):
        new = dict(kv.split("=", 1) for kv in line.split()[1:])
        if int(new.get("memory_mib.gpu0", 1000)) < 1000:
            print("ERR the biggest active conversation needs 1000 MiB", flush=True)
            continue
        if log and new:
            log.write(line + "\n"); log.flush()
        pool_mib, rows = int(new.get("memory_mib.gpu0", pool_mib)), int(new.get("rows", rows))
        print(f"MEMORY gpu0 total={8000 << 20} free={1000 << 20} experts={pool_mib << 20} setting={pool_mib << 20} "
              f"smallest={1000 << 20} largest={7000 << 20} name=card", flush=True)
        if "concurrent" in new:
            concurrent = int(new["concurrent"])
        print(f"SETTINGS rows={rows} concurrent={concurrent} memory_mib.gpu0={pool_mib}", flush=True)
        if "memory_mib.ram" in new:
            print("SHORTFALL the RAM's pool: out of memory", flush=True)
        print("END_MEMORY", flush=True)
        continue
    if line:
        print("ERR unknown " + line[:20], flush=True)
        continue
    window()
'''


class ParallelArgs(unittest.TestCase):
    def test_parallel_config_key(self):
        self.assertEqual(parallel_args({"parallel": 4}, []), ["--batch", "4"])
        self.assertEqual(parallel_args({"parallel": 1}, []), [])
        self.assertEqual(parallel_args({}, []), [])
        self.assertEqual(parallel_args({"parallel": 2}, ["--batch", "3"]), [])       # the args' own value wins
        self.assertEqual(parallel_args({"parallel": "4"}, []), [])                   # said and ignored
        self.assertEqual(parallel_args({"parallel": True}, []), [])
        self.assertEqual(parallel_args({"parallel": 12}, []), ["--batch", "12"])     # the engine warns and caps
    def test_parallel_above_the_window_says_so_unless_batch_mtp(self):
        import contextlib, io, os
        def said(args, env):
            buf = io.StringIO()
            old = os.environ.get("STRATA_BATCH_MTP")
            if env is None:
                os.environ.pop("STRATA_BATCH_MTP", None)
            else:
                os.environ["STRATA_BATCH_MTP"] = env
            try:
                with contextlib.redirect_stdout(buf):
                    r = parallel_args({"parallel": 12}, args)
            finally:
                if old is None:
                    os.environ.pop("STRATA_BATCH_MTP", None)
                else:
                    os.environ["STRATA_BATCH_MTP"] = old
            self.assertEqual(r, ["--batch", "12"])
            return "at most" in buf.getvalue()
        self.assertTrue(said([], None))                  # 0.1.39: the window holds 8 rows
        self.assertTrue(said([], "0"))
        self.assertFalse(said(["--batch-mtp"], None))    # --batch-mtp waves more through the window
        self.assertFalse(said([], "1"))

        args = engine_args({"args": ["--pack", "p"], "parallel": 2})
        self.assertEqual(args[-2:], ["--batch", "2"])


class PickSlot(unittest.TestCase):
    def test_vram_conversation_limit_waits_even_when_token_space_is_available(self):
        engine = self.engine(2)
        engine.info["vram_conversations"] = 1
        engine.slot_busy[0] = True
        engine.slot_live[0] = {"state": "decoding", "priority": 0}
        self.assertIsNone(engine.pick_slot([1, 2, 3]))
        self.assertFalse(engine.can_admit_prompt([1, 2, 3], priority=0))
        self.assertTrue(engine.can_admit_prompt([1, 2, 3], priority=1))
        engine.slot_live[0]["state"] = "ending"
        self.assertEqual(engine.pick_slot([1, 2, 3]), 1)
        engine.slot_live[0] = None
        engine.slot_held[0] = [4, 5, 6]
        self.assertEqual(engine.pick_slot([1, 2, 3]), 1, "idle caches can move to RAM without blocking admission")
        engine.info["vram_conversations"] = 2
        engine.slot_live[0] = {"state": "reading", "prompt_tokens": 20, "generated": 0, "priority": 0}
        self.assertEqual(engine.pick_slot([1, 2, 3]), 1, "a live limit increase opens another resident place")

    def test_shared_gpu_budget_waits_for_space_without_reserving_unused_output(self):
        engine = self.engine(2)
        engine.max_context = 2000
        engine.info["tokens_max"] = 1000
        engine.slot_busy[0] = True
        engine.slot_live[0] = {"state": "decoding", "prompt_tokens": 600, "generated": 101, "priority": 0}
        prompt = [1] * 300
        self.assertEqual(engine.pick_slot(prompt), 1)
        self.assertIsNone(engine.pick_slot(prompt + [1]), "the combined live token count must fit")
        self.assertFalse(engine.can_admit_prompt(prompt + [1], priority=0))
        self.assertTrue(engine.can_admit_prompt(prompt + [1], priority=1), "higher priority can displace lower work")
        engine.slot_live[0]["_generated_start"] = 100
        self.assertEqual(engine.pick_slot([1] * 400), 1, "continued legs must not count earlier output twice")
        engine.slot_live[0] = None
        engine.slot_held[0] = [2] * 900
        self.assertEqual(engine.pick_slot(prompt + [1]), 1, "idle caches can be paused by the engine")

    def test_temporary_effort_turn_does_not_hide_a_cancelling_conversations_checkpoint(self):
        engine = self.engine(2)
        engine._tail_role_token = 8678
        history = list(range(1000, 1600))
        marker = engine.TURN_ID
        held = history + [marker, 8678, 4, 5, marker, 74455, 6]
        prompt = history + [marker, 846, 7, 8, marker, 8678, 9, marker, 74455, 6]
        cancelled = threading.Event()
        cancelled.set()
        engine.slot_busy[0] = True
        engine.slot_live[0] = {"state": "decoding", "_cancel": cancelled, "_prompt": held}
        self.assertEqual(engine.get_cached_prefix(held, prompt), len(history))
        self.assertIsNone(engine.pick_slot(prompt))

    def engine(self, n):
        e = StrataEngine("missing-executable", [], lazy=True)
        e.batch = n
        e.slot_order = list(range(n))
        e.slot_busy = [False] * n
        e.slot_held = [[] for _ in range(n)]
        e.slot_used = [0.0] * n
        e.slot_live = [None] * n
        e.slot_handover = {}
        e.slot_releasing = set()
        e.slot_group = [0] * n
        return e

    def test_the_slot_that_holds_the_conversation(self):
        e = self.engine(3)
        e.slot_held = [[1, 2, 3], [], [1, 2, 3, 4, 5]]
        e.slot_used = [5.0, 0.0, 9.0]
        self.assertEqual(e.pick_slot([1, 2, 3, 4, 5, 6]), 2)       # the longest held start
        self.assertEqual(e.pick_slot([1, 2, 3, 9]), 0)
        self.assertEqual(e.pick_slot([7, 8]), 1)                    # no match: the empty slot first
        e.slot_busy[1] = True
        self.assertEqual(e.pick_slot([7, 8]), 0)                    # then the one used longest ago
        e.slot_busy = [True] * 3
        self.assertIsNone(e.pick_slot([7, 8]))
        e.slot_busy = [False] * 3
        self.assertEqual(e.pick_slot([1, 2, 3]), 1)                 # a held prompt is never the WHOLE prompt

    def test_a_rewritten_next_turn_goes_to_the_slot_of_its_chat(self):
        # #2518: two chats in two slots; each next turn renders the history again but not the reply it decoded or the
        # assistant header, so the held stream (prompt + reply) is no start of it.  The engine's turn checkpoint sits at
        # the last <|im_start|> of the old prompt: the prompt that agrees up to there belongs to that slot.  Before, both
        # went to the least recently used slot in turn, and the first one overwrote the other chat's cache.
        T = StrataEngine.TURN_ID
        hist_a, hist_b = list(range(1000, 1600)), list(range(2000, 2600))
        held_a = hist_a + [T, 7, 8, 9] + [50, 51, 52]                  # prompt (ends with the assistant header) + reply
        held_b = hist_b + [T, 7, 8, 9] + [60, 61]
        e = self.engine(2)
        e.slot_held = [held_a, held_b]
        e.slot_used = [10.0, 20.0]                                      # slot 0 was used longer ago
        next_a = hist_a + [T, 5, 6, 7, 8, 9] + list(range(3000, 3100))  # the history again, then new turns
        next_b = hist_b + [T, 5, 6, 7, 8, 9] + list(range(3000, 3100))
        self.assertEqual(e.pick_slot(next_b), 1)                        # the LRU slot 0 holds the other chat
        self.assertEqual(e.pick_slot(next_a), 0)
        e.slot_busy[0] = True                                           # a busy slot is never picked
        self.assertEqual(e.pick_slot(next_a), 1)
        e.slot_busy[0] = False
        # a chat that agrees with neither (a new one, or one that rewrote its history) keeps the old order
        self.assertEqual(e.pick_slot(list(range(5000, 5200))), 0)
        self.assertEqual(e.pick_slot(hist_a[:300] + [T, 5, 6] + list(range(3000, 3100))), 0)
        e.slot_held[0] = []                                             # an empty slot still comes before evicting one
        self.assertEqual(e.pick_slot(list(range(5000, 5200))), 0)
        # the whole held stream continued is still the longer match
        e.slot_held = [held_a, held_b + [T, 1, 2]]
        self.assertEqual(e.pick_slot(held_b + [T, 1, 2, 3, 4]), 1)

    def test_batch_groups_spread_requests_over_the_groups(self):
        # #1249: with --batch-groups 2 over 4 slots (groups {0,1} and {2,3}) a second request goes to the other group,
        # and a short held prefix does not pull it back into the busy one
        e = self.engine(4)
        e.slot_groups = 2
        e.slot_group = [0, 0, 1, 1]
        e.slot_busy[0] = True
        e.slot_held[1] = [1, 2, 3]
        self.assertEqual(e.pick_slot([1, 2, 3, 4, 5]), 2)
        e.slot_held[1] = list(range(1, 601))                       # a long prefix is worth the unbalanced group
        self.assertEqual(e.pick_slot(list(range(1, 700))), 1)
        e.slot_groups, e.slot_group = 1, [0] * 4                   # without groups: any held prefix, as before
        e.slot_held[1] = [1, 2, 3]
        self.assertEqual(e.pick_slot([1, 2, 3, 4, 5]), 1)

    def test_slots_view(self):
        e = self.engine(2)
        e.slot_held[1] = [1, 2]
        e.slot_live[0] = {"slot": 0, "state": "decoding", "prompt_tokens": 9, "generated": 4,
                          "started": time.time() - 2, "first_token": time.time() - 1}
        v = e.slots_view()
        self.assertEqual(v[0]["state"], "decoding")
        self.assertEqual(v[0]["generated"], 4)
        self.assertEqual(v[1], {"slot": 1, "state": "idle", "held_tokens": 2})


class FakeEngineCase(unittest.TestCase):
    """The real StrataEngine and Service over HTTP, the fake engine behind them."""

    def start(self, slots, fit=None, slot_cache=False, fail=False, reuse=False, says_groups=0,
              more=(), stage_plan=False, batch_sample=False, stage_sample=False):
        import strata_server.server as server
        self.tmp = tempfile.TemporaryDirectory()
        script = Path(self.tmp.name) / "fake_strata.py"
        script.write_text(FAKE_BATCH, encoding="utf-8")
        self.log = Path(self.tmp.name) / "requests.log"
        real = server.subprocess.Popen
        extra = ["--batch", str(slots)] + (["--fit", str(fit)] if fit is not None else []) + ["--log", str(self.log)]
        extra += ["--slotcache"] if slot_cache else []
        extra += ["--fail-window"] if fail else []
        extra += ["--reuse"] if reuse else []
        extra += ["--says-groups", str(says_groups)] if says_groups else []
        extra += ["--fake-stage-plan"] if stage_plan else []
        extra += ["--fake-batch-sample"] if batch_sample else []
        extra += ["--fake-stage-sample"] if stage_sample else []
        extra += list(more)
        with mock.patch.object(server.subprocess, "Popen",
                               lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
            self.engine = StrataEngine("strata", extra, log=str(Path(self.tmp.name) / "engine.log") if stage_plan else None)
        tok = ByteTokenizer()
        self.svc = Service(self.engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        if getattr(self, "httpd", None):
            self.httpd.shutdown()
            self.httpd.server_close()
        if getattr(self, "engine", None):
            self.engine.unload()
        if getattr(self, "tmp", None):
            self.tmp.cleanup()

    def chat(self, text, max_tokens=64, **fields):
        body = {"messages": [{"role": "user", "content": text}], "max_tokens": max_tokens, "reasoning_effort": "none",
                **fields}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())

    def message(self, text, max_tokens=64):
        body = {"model": "m", "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens}
        req = urllib.request.Request(self.base + "/v1/messages", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=10) as r:
            return json.loads(r.read().decode())

    def _two_at_once(self, text_b):
        """A long request first, then one that has to go to a batch slot (the engine is busy with the first)."""
        out = {}
        def go(k, text):
            try:
                out[k] = ("ok", self.chat(text, max_tokens=64))
            except urllib.error.HTTPError as e:
                out[k] = ("http", e.code, e.read().decode(), dict(e.headers))
        ta = threading.Thread(target=go, args=("a", "LONGREPLY please"))
        ta.start()
        time.sleep(0.15)
        tb = threading.Thread(target=go, args=("b", text_b))
        tb.start()
        ta.join(30); tb.join(30)
        return out

    def race(self, first, second, delay=0.15, max_tokens=64):
        """`first` sent, `second` after `delay` seconds; -> {text: (reply, seconds it took)}"""
        res = {}

        def go(t):
            t0 = time.time()
            r = self.chat(t, max_tokens=max_tokens)
            res[t] = (r["choices"][0]["message"]["content"], time.time() - t0)
        a = threading.Thread(target=go, args=(first,))
        a.start()
        time.sleep(delay)
        go(second)
        a.join(30)
        return res


class ContextView(FakeEngineCase):
    """The API monitor's Context view with --batch: the engine's own session (where a request alone runs), each slot
    and each parked conversation, as the engine lists them - also while slots decode."""

    def view(self):
        return {e["name"]: e for e in self.svc.context_view()["entries"]}

    def text(self, cid):
        code, body = self.svc.context_window(cid)
        self.assertEqual(code, 200, body)
        return "".join(body["pieces"])

    def test_conversations_by_name_and_place_then_one_context_window(self):
        self.start(2, more=["--lists"])
        self.assertEqual(self.chat("hello")["choices"][0]["message"]["content"], "ok, done.")
        self.assertEqual(self.chat("another conversation")["choices"][0]["message"]["content"], "ok, done.")
        v = self.view()
        self.assertEqual(list(v), ["Conversation 1", "Parked conversation 1 of 1"])
        solo, parked = v["Conversation 1"], v["Parked conversation 1 of 1"]
        self.assertEqual((solo["id"], solo["where"], solo["state"], solo["kind"]), ("main", "vram", "holding", "session"))
        self.assertEqual((solo["capacity"], solo["bytes"], solo["used"]), (4096, 409600, 100 * solo["count"]))
        self.assertEqual((parked["id"], parked["where"], parked["kind"]), ("parked1", "ram", "parked"))
        self.assertEqual((parked["capacity"], parked["bytes"]), (parked["count"], 100 * parked["count"]))
        self.assertTrue(solo["preview"].startswith("another conversation"), solo["preview"])   # the first user message
        self.assertTrue(parked["preview"].startswith("hello"), parked["preview"])
        self.assertNotIn("tokens", solo)                   # the list carries no context windows
        self.assertTrue(self.text("main").endswith("ok, done."))   # (its end token is never fed)
        self.assertIn("hello", self.text("parked1"))
        self.assertEqual(self.svc.context_window("parked9")[0], 404)   # not held (any more)

    def test_slots_are_listed_while_they_decode(self):
        self.start(2, more=["--lists"])
        out, seen = {}, []

        def watch():
            while not out:
                seen.extend(e["state"] for name, e in self.view().items() if e["kind"] == "session")
                time.sleep(0.02)
        t = threading.Thread(target=watch)
        t.start()
        try:
            out.update(self._two_at_once("second one"))
        finally:
            out.setdefault("done", True)
            t.join()
        self.assertEqual((out["a"][0], out["b"][0]), ("ok", "ok"), out)
        self.assertIn("decoding", seen)
        time.sleep(0.3)                                 # (a slot is freed at its BDONE, in the background)
        held = [self.text(e["id"]) for name, e in self.view().items() if e["kind"] == "session" and e["count"]]
        self.assertTrue(any(h.endswith("ok, done.") for h in held), held)

    def test_an_engine_without_the_commands(self):
        self.start(2)
        self.chat("hello")
        self.assertEqual(self.svc.context_view()["reason"], "engine")


class ParallelService(FakeEngineCase):
    """The real StrataEngine and Service over HTTP, the fake engine behind them: slots, queues, benchmarks, telemetry.
    The yielding, restart and failure tests are in strata_server/test_parallel_yield.py (this file alone ran past 100 s)."""

    def test_followup_waits_for_cancelling_cache_instead_of_prefilling_another_slot(self):
        self.start(2)
        engine, cancel = self.engine, threading.Event()
        history = [7] * 600
        original = history + [engine.TURN_ID, 8, 9]
        followup = history + [engine.TURN_ID, 10, 11]
        with engine.slot_cv:
            engine.slot_busy[0] = True
            engine.slot_claims.add(0)
            engine.slot_live[0] = {"state": "decoding", "_cancel": cancel, "_prompt": original}
            self.assertEqual(engine.pick_slot(followup), 1, "an intentional live fork must not wait")
            cancel.set()
            self.assertIsNone(engine.pick_slot(followup), "cancellation must hand the matching cache back first")
            engine.slot_held[1] = original
            self.assertEqual(engine.pick_slot(followup), 1, "an equally useful free cache need not wait")
            engine.slot_held[1] = []
            self.assertEqual(engine.pick_slot([12] * 600), 1, "an unrelated request need not wait")
            engine.slot_handover[0] = threading.Event()
            self.assertEqual(engine.pick_slot(followup), 1, "preemption must not wait for its own handover")
            engine.slot_handover.clear()
            engine.slot_held[0] = original
            engine.slot_releasing.add(0)
            engine.slot_live[0] = None
            self.assertIsNone(engine.pick_slot(followup), "the BDONE drain keeps the reuse barrier")
        output, failures = [], []

        def run():
            try:
                output.extend(t for t in engine.generate_batched(followup, 3, {}, threading.Event()) if t is not None)
            except Exception as error:
                failures.append(error)

        worker = threading.Thread(target=run)
        worker.start()
        try:
            end = time.monotonic() + 3
            while engine.admitting == 0 and time.monotonic() < end:
                time.sleep(0.01)
            self.assertEqual(engine.admitting, 1)
            self.assertNotIn("BGEN", self.log.read_text())
        finally:
            with engine.slot_cv:
                engine.slot_busy[0] = False
                engine.slot_releasing.discard(0)
                engine.slot_claims.discard(0)
                engine.slot_cv.notify_all()
            worker.join(10)
        self.assertFalse(worker.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(output, list(b"ok,"))
        self.assertIn("BGEN 0 ", self.log.read_text())

    def test_zero_token_batch_admission_preserves_output_and_cached_prefix(self):
        self.start(2, more=["--zero-admit"])
        prompt = list(b"zero-token admission")
        with self.engine.slot_cv:
            self.engine.slot_busy[1] = True
        try:
            for limit in (3, 1):
                with self.subTest(max_new=limit):
                    output = [token for token in self.engine.generate_batched(prompt, limit, {}, threading.Event())
                              if token is not None]
                    self.assertEqual(output, list(b"ok,")[:limit])
                    self.assertEqual(self.engine.slot_held[0], prompt + output[:-1])
        finally:
            with self.engine.slot_cv:
                self.engine.slot_busy[1] = False
                self.engine.slot_cv.notify_all()
        self.assertIn("BGEN", self.log.read_text())

    def test_preempted_zero_token_admission_waits_and_continues(self):
        self.start(2)
        prompt = list(b"PREEMPTADMIT")
        with self.engine.slot_cv:
            self.engine.slot_busy[1] = True
        try:
            output = [token for token in self.engine.generate_batched(prompt, 3, {}, threading.Event())
                      if token is not None]
            self.assertEqual(output, list(b"ok,"))
            self.assertEqual(self.engine.slot_held[0], prompt + output[:-1])
            self.assertEqual(sum(line.startswith("BGEN") for line in self.log.read_text().splitlines()), 2)
        finally:
            with self.engine.slot_cv:
                self.engine.slot_busy[1] = False
                self.engine.slot_cv.notify_all()

    def test_bgen_refused_by_the_engine_is_retried_not_a_400(self):
        """#1994: the engine refuses a BGEN ("no such free slot") although the server saw the slot free: the request
        goes to another slot (or waits) and is answered; the same conversation's next turn works too."""
        self.start(2)
        self.engine.SLOT_PARK_S = 0.2
        out = self._two_at_once("BUSYSLOT hello")
        self.assertEqual(out["b"][0], "ok", out["b"])
        self.assertIn("REFUSED", self.log.read_text())
        again = self.chat("BUSYSLOT hello")                     # the conversation is not poisoned
        self.assertEqual(again["choices"][0]["message"]["content"], "ok, done.")
        time.sleep(0.5)
        self.assertEqual(self.engine.slot_busy, [False, False])

    def test_bgen_refused_for_good_is_a_retryable_503_and_frees_the_slots(self):
        """#1994: when the engine keeps refusing, the answer is a 503 (retryable), never a 400, and no slot stays busy."""
        self.start(2)
        self.engine.SLOT_PARK_S = 0.05
        out = self._two_at_once("BUSYALL hello")
        self.assertEqual(out["b"][0], "http", out["b"])
        self.assertEqual(out["b"][1], 503, out["b"])
        time.sleep(0.5)
        self.assertEqual(self.engine.slot_busy, [False, False])
        self.assertEqual(self.chat("fine again")["choices"][0]["message"]["content"], "ok, done.")

    def test_engine_reports_fewer_slots(self):
        self.start(4, fit=2)
        self.assertEqual(self.engine.batch, 2)
        self.assertEqual(self.get("/v1/status")["concurrency"]["serving"], 2)

    def test_reported_capacity_can_exceed_initial_batch(self):
        """Paged engines expose their step capacity even when started with --batch 1."""
        self.start(1, fit=8)
        self.assertEqual(self.engine.batch, 8)
        self.assertEqual(len(self.engine.slot_q), 8)
        self.svc.max_queue = 1
        self.assertEqual(self.svc.admission_limit(), 9)

    def test_admission_cap_uses_effective_slots_and_rejects_before_stream(self):
        self.start(4, fit=2)
        self.svc.max_queue = 1
        self.assertEqual(self.svc.admission_limit(), 3)
        self.assertTrue(all(self.svc.reserve_inference() for _ in range(3)))
        cases = (
            ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "stream": True}),
            ("/v1/messages", {"model": "m", "max_tokens": 8,
                              "messages": [{"role": "user", "content": "hi"}], "stream": True}),
            ("/v1/responses", {"model": "m", "input": "hi", "stream": True}),
        )
        try:
            for path, body in cases:
                req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(req, timeout=10)
                err = caught.exception
                self.assertEqual(err.code, 429)
                self.assertEqual(err.headers.get("Retry-After"), "1")
                self.assertIn("admission_full", err.read().decode())
            snapshot = self.get("/v1/telemetry")["work"]
            self.assertEqual((snapshot["admission_reserved"], snapshot["admission_limit"],
                              snapshot["max_queue"], snapshot["admission_rejected"]), (3, 3, 1, 3))
        finally:
            for _ in range(3):
                self.svc.release_inference()
        self.assertEqual(self.chat("capacity freed")["choices"][0]["message"]["content"], "ok, done.")
        self.assertEqual(self.get("/v1/telemetry")["work"]["admission_reserved"], 0)

    def test_queue_timeout_serial_fifo_returns_429_and_recovers(self):
        self.start(1)
        self.svc.queue_timeout_s = 0.12
        self.svc.max_queue = 1
        self.svc.fifo.acquire()
        try:
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.chat("waiting for serial turn")
            self.assertEqual(caught.exception.code, 429)
            self.assertEqual(caught.exception.headers.get("Retry-After"), "1")
            self.assertEqual(json.loads(caught.exception.read())["error"]["code"], "queue_wait_timeout")
            work = self.get("/v1/telemetry")["work"]
            self.assertEqual((work["queue_timeout_s"], work["queue_timeouts"], work["queued"]),
                             (0.12, 1, 0))
        finally:
            self.svc.fifo.release()
        self.assertEqual(self.chat("next turn")["choices"][0]["message"]["content"], "ok, done.")
        self.assertEqual(self.get("/v1/telemetry")["work"]["admission_reserved"], 0)

    def test_queue_timeout_parallel_control_and_stream_formats(self):
        self.start(2)
        self.svc.queue_timeout_s = 0.12
        self.svc.max_queue = 1
        self.engine.ctl.acquire()
        cases = (
            ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}, "data: "),
            ("/v1/messages", {"model": "m", "max_tokens": 8,
                              "messages": [{"role": "user", "content": "hi"}]}, "event: error"),
            ("/v1/responses", {"model": "m", "input": "hi"}, "event: response.failed"),
        )
        try:
            for path, body, marker in cases:
                for stream in (False, True):
                    req = urllib.request.Request(self.base + path,
                                                 data=json.dumps({**body, "stream": stream}).encode(),
                                                 headers={"Content-Type": "application/json"})
                    if stream:
                        with urllib.request.urlopen(req, timeout=10) as response:
                            data = response.read().decode()
                        self.assertIn(marker, data)
                        self.assertIn("queue_wait_timeout", data)
                    else:
                        with self.assertRaises(urllib.error.HTTPError) as caught:
                            urllib.request.urlopen(req, timeout=10)
                        self.assertEqual(caught.exception.code, 429)
                        self.assertIn("queue_wait_timeout", caught.exception.read().decode())
            work = self.get("/v1/telemetry")["work"]
            self.assertEqual(work["queue_timeouts"], 6)
            self.assertEqual((work["admission_reserved"], work["waiting"]), (0, 0))
        finally:
            self.engine.ctl.release()
        self.assertEqual(self.chat("healthy")["choices"][0]["message"]["content"], "ok, done.")

    def test_queue_timeout_parallel_slot_wait_releases_control(self):
        self.start(2)
        self.svc.queue_timeout_s = 0.12
        with self.engine.slot_cv:
            self.engine.slot_busy[:] = [True, True]
        try:
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.chat("waiting for a batch slot")
            self.assertEqual(caught.exception.code, 429)
            self.assertIn("queue_wait_timeout", caught.exception.read().decode())
            self.assertFalse(self.engine.ctl.locked())
            self.assertEqual(self.engine.waiting, 0)
        finally:
            with self.engine.slot_cv:
                self.engine.slot_busy[:] = [False, False]
                self.engine.slot_cv.notify_all()
        self.assertEqual(self.chat("slot freed")["choices"][0]["message"]["content"], "ok, done.")

    def test_benchmark_overload_suite_preserves_batch_recovery(self):
        self.start(2)
        self.svc.max_queue = 0
        self.svc.queue_timeout_s = 0.15
        out = Path(self.tmp.name) / "overload"
        report = bench.run(Endpoint(self.base, None), Endpoint(self.base, None), out,
                           suite="parallel-overload-v1", concurrency=6)
        self.assertTrue(report["successful"])
        overload = report["phases"][1]
        self.assertEqual(overload["kind"], "overload")
        self.assertGreater(overload["summary"]["expected_overload"], 0)
        self.assertEqual(overload["summary"]["failed"], 0)
        self.assertTrue(report["phases"][2]["passed"])
        self.assertEqual(self.get("/v1/telemetry")["work"]["admission_reserved"], 0)

    def test_benchmark_cancel_suite_releases_batch_queue_and_slots(self):
        self.start(2)
        self.svc.max_queue = 1
        self.svc.queue_timeout_s = 5
        out = Path(self.tmp.name) / "cancel"
        report = bench.run(Endpoint(self.base, None), Endpoint(self.base, None), out,
                           suite="parallel-cancel-v1", concurrency=2, request_timeout=15)
        self.assertTrue(report["successful"], bench.format_summary(bench.summary(report)))
        self.assertEqual([row["kind"] for row in report["phases"]],
                         ["warmup", "cancel_active", "cancel_queued", "recovery"])
        self.assertEqual(report["phases"][2]["summary"]["companion_successful"], 2)
        self.assertEqual(self.get("/v1/telemetry")["work"]["admission_reserved"], 0)

    def test_benchmark_prefill_cancel_observes_progress_and_companion_completion(self):
        self.start(2)
        out = Path(self.tmp.name) / "prefill-cancel"
        report = bench.run(Endpoint(self.base, None), Endpoint(self.base, None), out,
                           suite="parallel-prefill-cancel-v1", concurrency=2, request_timeout=15)
        self.assertTrue(report["successful"], bench.format_summary(bench.summary(report)))
        self.assertEqual([row["kind"] for row in report["phases"]],
                         ["warmup", "cancel_prefill", "recovery"])
        phase = report["phases"][1]
        self.assertTrue(phase["summary"]["prompt_read_seen"])
        self.assertGreater(phase["requests"][0]["prompt_read_at_close"], 0)
        self.assertLess(phase["requests"][0]["prompt_read_at_close"],
                        phase["requests"][0]["prompt_tokens_at_close"])
        self.assertEqual(phase["summary"]["client_disconnect_delta"], 1)
        self.assertEqual(phase["summary"]["companion_successful"], 1)
        self.assertEqual(self.get("/v1/telemetry")["work"]["admission_reserved"], 0)
        data = (out / "results.json").read_text()
        self.assertNotIn("Explain how a serving system", data)

    def test_telemetry_keeps_engine_waiting_age_visible(self):
        self.start(2)
        self.svc.max_queue = 1
        self.svc.queue_timeout_s = 0.5
        self.engine.ctl.acquire()
        outcome = []
        def request():
            try:
                self.chat("waiting")
            except urllib.error.HTTPError as error:
                outcome.append(error.code)
                error.read()
        worker = threading.Thread(target=request)
        try:
            worker.start()
            cutoff = time.monotonic() + 2
            while time.monotonic() < cutoff:
                work = self.get("/v1/telemetry")["work"]
                if any(row["state"] == "waiting" for row in work["requests"]):
                    break
                time.sleep(0.01)
            else:
                self.fail("engine wait was not visible in telemetry")
            time.sleep(0.04)
            work = self.get("/v1/telemetry")["work"]
            self.assertGreater(work["oldest_queue_s"], 0.03)
            self.assertTrue(any(row["state"] == "waiting" for row in work["requests"]))
        finally:
            self.engine.ctl.release()
            worker.join(timeout=3)
        self.assertEqual(outcome, [])

    def test_batch_groups_auto_follows_the_engine(self):
        """--batch-groups auto: the engine picks the groups and says so (INFO batch_groups=G); the server spreads the
        requests over those groups, and a plain number is read as before."""
        self.start(8, says_groups=4, more=["--batch-groups", "auto"])
        e = self.engine
        self.assertEqual((e.batch, e.slot_groups), (8, 4))
        self.assertEqual(e.slot_group, [0, 0, 1, 1, 2, 2, 3, 3])
        self.assertEqual(e.slot_order[:4], [0, 2, 4, 6])
        telemetry = self.get("/v1/telemetry")
        self.assertEqual([s["group"] for s in telemetry["work"]["slots"]], [0, 0, 1, 1, 2, 2, 3, 3])
        self.tearDown(); self.httpd = self.engine = self.tmp = None
        self.start(8, more=["--batch-groups", "auto"])          # an engine that reports none: one group
        self.assertEqual(self.engine.slot_groups, 1)
        self.tearDown(); self.httpd = self.engine = self.tmp = None
        self.start(8, more=["--batch-groups", "2"])
        self.assertEqual(self.engine.slot_groups, 2)
        self.tearDown(); self.httpd = self.engine = self.tmp = None
        self.start(8, says_groups=4)                           # 0.1.41: no flag on a layer split = the engine's choice
        self.assertEqual(self.engine.slot_groups, 4)
        self.tearDown(); self.httpd = self.engine = self.tmp = None
        self.start(8, says_groups=4, more=["--batch-groups", "1"])   # the opt-out
        self.assertEqual(self.engine.slot_groups, 1)

    def test_engine_log_resolves_four_stage_layer_and_device_mapping(self):
        self.start(8, says_groups=4, stage_plan=True)
        self.svc.terminal_visibility_order = [2, 0, 3, 1]
        self.svc.gpu_indices = [2, 0, 3, 1]
        telemetry = self.get("/v1/telemetry")
        topology = telemetry["topology"]
        self.assertEqual(topology["source"], "engine_start_log")
        self.assertTrue(topology["resolved"])
        self.assertEqual(telemetry["quality"]["placement"], "reported")
        self.assertEqual([(row["stage"], row["device_index"], row["layer_start"], row["layer_end"])
                          for row in topology["stages"]],
                         [(0, 2, 0, 11), (1, 0, 12, 23), (2, 3, 24, 35), (3, 1, 36, 47)])
        clean = Endpoint(self.base).snapshot()
        self.assertEqual(clean["topology"]["stages"], topology["stages"])
        self.svc.terminal_visibility_order = []
        partial = self.get("/v1/telemetry")
        self.assertFalse(partial["topology"]["resolved"])
        self.assertEqual(len(partial["topology"]["stages"]), 4)
        self.assertIsNone(partial["topology"]["stages"][0]["device_index"])
        self.assertEqual(partial["quality"]["placement"], "partial")
        self.engine.unload()
        unloaded = self.get("/v1/telemetry")["topology"]
        self.assertFalse(unloaded["resolved"])
        self.assertEqual(unloaded["stages"], [])

    def test_engine_batch_window_sample_reaches_telemetry_without_entering_reply(self):
        self.start(2, batch_sample=True)
        replies = self.race("first LONGREPLY", "second LONGREPLY", delay=0.05, max_tokens=64)
        self.assertTrue(all("ok, " in reply for reply, _ in replies.values()))
        self.assertTrue(any(line.startswith("BGEN") for line in self.log.read_text().splitlines()))
        deadline = time.monotonic() + 2
        while True:
            telemetry = self.get("/v1/telemetry")
            if telemetry["quality"]["batch_window_rows"] == "measured":
                break
            if time.monotonic() >= deadline:
                self.fail("fake engine's BSTAT sample did not reach telemetry")
            time.sleep(0.02)
        work = telemetry["work"]
        self.assertGreaterEqual(work["effective_rows_window"], 1.0)
        self.assertLessEqual(work["effective_rows_window"], 2.0)
        self.assertEqual(work["batch_active_rows_window"], work["effective_rows_window"])
        self.assertEqual(work["batch_sample_windows"], 1)
        self.assertEqual(work["batch_sample_span_s"], 0.02)
        self.assertGreaterEqual(work["batch_sample_age_s"], 0)
        clean = Endpoint(self.base).snapshot()
        self.assertEqual(clean["quality"]["batch_window_rows"], "measured")
        self.assertGreaterEqual(clean["work"]["batch_sample_rows"], 1)
        self.engine.unload()
        unloaded = self.get("/v1/telemetry")
        self.assertIsNone(unloaded["work"]["effective_rows_window"])
        self.assertEqual(unloaded["quality"]["batch_window_rows"], "unavailable")

    def test_pipeline_stage_host_spans_reach_telemetry_without_entering_reply(self):
        self.start(2, says_groups=2, stage_plan=True, stage_sample=True)
        replies = self.race("first LONGREPLY", "second LONGREPLY", delay=0.05, max_tokens=64)
        self.assertTrue(all("ok, " in reply for reply, _ in replies.values()))
        deadline = time.monotonic() + 2
        while True:
            telemetry = self.get("/v1/telemetry")
            if telemetry["quality"].get("stage_in_flight_source") == "engine_pstat_host_span":
                break
            if time.monotonic() >= deadline:
                self.fail("fake engine's PSTAT sample did not reach telemetry")
            time.sleep(0.02)
        self.assertEqual(telemetry["work"]["stage_in_flight_us"], [16000, 14000, 12000, 10000])
        clean = Endpoint(self.base).snapshot()
        self.assertEqual(clean["quality"]["stage_in_flight"], "measured_host_span")
        self.assertEqual([row["percent"] for row in clean["work"]["stage_in_flight"]],
                         [80.0, 70.0, 60.0, 50.0])
        self.engine.unload()
        unloaded = self.get("/v1/telemetry")
        self.assertNotIn("stage_in_flight_us", unloaded["work"])

    def test_a_burst_of_connections_is_not_reset(self):
        """30-40 clients at once got "connection reset by peer" with the listen backlog of 5 (4 x R9700 burst): the
        server listens with a deep backlog (STRATA_HTTP_BACKLOG, 256), so 60 simultaneous requests all get answers."""
        import strata_server.server as server
        self.assertGreaterEqual(server.Server.request_queue_size, 64)
        self.start(2)
        errs, ok = [], []
        def one(i):
            try:
                self.chat(f"burst {i}", max_tokens=8)
                ok.append(i)
            except Exception as e:                      # noqa: BLE001
                errs.append(repr(e))
        th = [threading.Thread(target=one, args=(i,)) for i in range(60)]
        for t in th: t.start()
        for t in th: t.join()
        self.assertEqual((len(ok), errs[:2]), (60, []))

    def test_stop_strings_in_a_batch_slot(self):
        """#454: a stop string cuts the answer in --batch mode too, and the slot is freed for the next request."""
        self.start(2)
        body = {"messages": [{"role": "user", "content": "hi LONGREPLY"}], "max_tokens": 64, "reasoning_effort": "none",
                "stop": ["la la"]}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            c = json.loads(r.read().decode())["choices"][0]
        self.assertEqual((c["message"]["content"], c["finish_reason"]), ("ok, ", "stop"))
        self.assertEqual(self.chat("again")["choices"][0]["message"]["content"], "ok, done.")

    def test_engine_turns_batching_off(self):
        self.start(4, fit=0)
        self.assertEqual(self.engine.batch, 0)
        self.assertEqual(self.chat("hi")["choices"][0]["message"]["content"], "ok, done.")

    def test_concurrent_requests_share_the_slots(self):
        self.start(2)
        # the first one decodes long enough for the others to arrive while it runs (also on a busy PC)
        texts = ["first question LONGREPLY", "second question, longer", "a third one waits for a slot"]
        reply = {t: "ok, " + "la " * 15 + "done." if "LONGREPLY" in t else "ok, done." for t in texts}
        out, errors, seen, terminal_seen = {}, [], [], []

        def go(t):
            try:
                out[t] = self.chat(t)
            except Exception as e:   # noqa: BLE001 - reported below
                errors.append(e)

        threads = [threading.Thread(target=go, args=(t,)) for t in texts]
        for th in threads:
            th.start()
            time.sleep(0.05)
        deadline = time.time() + 30
        while any(th.is_alive() for th in threads) and time.time() < deadline:
            m = self.get("/metrics")["live"]
            seen.append(m)
            terminal_seen.append(self.get("/v1/telemetry"))
            time.sleep(0.01)
        for th in threads:
            th.join(30)
        self.assertEqual(errors, [])
        for t in texts:
            self.assertEqual(out[t]["choices"][0]["message"]["content"], reply[t])
        # the slots were reported, and a request was admitted while another decoded in a slot (the fake logs how many
        # slots were active at each admission)
        self.assertTrue(any(m.get("parallel") == 2 and len(m.get("slots", [])) == 2 for m in seen))
        self.assertTrue(any(t["work"]["requests"] for t in terminal_seen))
        self.assertTrue(any(s.get("request_id") for t in terminal_seen for s in t["work"]["slots"]))
        self.assertTrue(any(isinstance(r.get("first_engine_token_s"), (int, float)) and
                            r["first_engine_token_s"] > 0
                            for t in terminal_seen for r in t["work"]["requests"]
                            if r["state"] == "decoding"), "live per-request first engine token was not observed")
        for t in terminal_seen:
            self.assertEqual(t["quality"]["request_correlation"], "reported")
            self.assertNotIn("input", json.dumps(t))
        admitted = [int(x.split()[3]) for x in self.log.read_text().splitlines() if x.startswith("BGEN")]
        self.assertIn(1, admitted)
        # every request recorded its own row; nothing is left running
        with self.svc.status_lock:
            self.assertEqual(len(self.svc.history), 3)
            self.assertEqual(self.svc.live_reqs, {})
            self.assertFalse(self.svc.status["busy"])
        self.assertEqual(sorted(r["output_tokens"] for r in self.svc.history),
                         sorted(len(reply[t]) + 1 for t in texts))
        self.assertTrue(all(isinstance(r.get("first_engine_token_s"), (int, float)) and
                            r["first_engine_token_s"] > 0 for r in self.svc.history))
        clean = Endpoint(self.base).snapshot()
        self.assertTrue(all(isinstance(r.get("first_engine_token_s"), (int, float))
                            for r in clean["recent"][:3]))
        # never more than two in the slots at once (the fake logs how many were active at each admission)
        active = [int(x.split()[3]) for x in self.log.read_text().splitlines() if x.startswith("BGEN")]
        self.assertTrue(all(a <= 1 for a in active), active)


class Preemption(FakeEngineCase):
    """A request of a higher priority that finds every slot busy takes the slot of one of lower priority (an engine
    whose READY says "preempt"); the request it ended continues from where it was once a slot is free."""

    def test_the_higher_priority_takes_a_slot_and_every_answer_completes(self):
        self.start(2, more=["--preempt"])
        low = ["first question LONGREPLY", "second question LONGREPLY"]
        out, done = {}, []

        def go(text, prio):
            out[text] = self.chat(text, priority=prio)["choices"][0]["message"]["content"]
            done.append(text)
        threads = [threading.Thread(target=go, args=(t, 0)) for t in low]
        for th in threads:
            th.start()
            time.sleep(0.15)
        time.sleep(0.15)
        go("urgent question", 5)
        for th in threads:
            th.join(30)
        self.assertEqual(out, {**{t: "ok, " + "la " * 15 + "done." for t in low}, "urgent question": "ok, done."})
        self.assertEqual(done[0], "urgent question")
        self.assertEqual(self.log.read_text().count("PREEMPT"), 1)
        self.assertEqual(self.engine.slot_handover, {})

    def test_equal_priorities_wait_for_a_slot(self):
        self.start(2, more=["--preempt"])
        self.race("first question LONGREPLY", "second question")
        self.assertNotIn("PREEMPT", self.log.read_text())


if __name__ == "__main__":
    unittest.main()
