"""Runs one check of the NVIDIA test session (run.ps1) on one engine build and writes what it printed.

    drive.py <scenario> --exe ENGINE --out FILE --data STRATA_DATA [--cards 1,0] [--args "..."] [--env K=V ...]
             [--deepseek GGUF_DIR] [--other SAVED_FILE]

Scenarios (the engine's --serve protocol unless said otherwise):
  q38       Qwen3.8: the protocol's commands and bad lines, two batch slots, slots that continue a conversation
  q38state  Qwen3.8: prefix reuse, parking, SAVE / RESTORE, and a RESTORE of --other (the build before's file)
  q36       Qwen3.6: the same for its engine (strata-q36)
  multi     Qwen3.8: a 2,000-token prompt, a 128-token answer, a second turn; the startup lines appended
  dsgroups  strata-ds41: --conversations 1..4; standalone identity and aggregate decode speed, DSpark as configured
  groups    Qwen3.8: four standalone answers against two rounds of four batch answers and aggregate decode speed
  cutmove   strata-q38: a 96-token story in a batch slot; with --move-cut the first card's memory is halved after 24
            tokens and set back after 48 (the layer cut moves between two steps); the answers go to FILE.settings
  ds        DeepSeek V4.1: greedy and seeded answers, a 128-token one for the speed
  dsmemory  strata-ds41: greedy answers and a continuation, with fixed expert placement and no drafting
  dscutmove strata-ds41: cutmove's 96-token story using the DeepSeek tokenizer and chat format
  server    the HTTP server on port 8091 with this engine (Qwen3.8): two greedy chats and their speed
  config    no engine start: FILE gets the run config (strata-<model>.json) that `server` uses, for
            tools/measure_rows.py
FILE gets the engine's stdout (or the server's answers), FILE.stderr its stderr; FILE ends with "exit <code>".
FILE.gpu.csv and FILE.sys.csv get what the cards and the PC were doing, once a second (Sampler).
--cards are the cards the engine sees, numbered as nvidia-smi numbers them. --env adds the check's settings.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import queue
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
from strata_common.expert_fill import wait_for_experts, wait_for_http_experts  # noqa: E402
from dsv41.smoke import tokenizer  # noqa: E402
from report import GPU_FIELDS, get_cut_move_problem, startup_lines  # noqa: E402

#: Never 8080, where the owner's own server runs
PORT = 8091
DS_PACK = "packs/deepseek-v4.1-flash-ist-das-2.00bpw"
QWEN_USER, QWEN_ANSWER = "<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", choices=["q38", "q38state", "q36", "multi", "cutmove", "ds", "dsmemory",
                                        "dscutmove", "dsgroups", "groups", "server", "config"])
    ap.add_argument("--exe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--cards", default="")
    ap.add_argument("--args", default="")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--deepseek")
    ap.add_argument("--other")
    ap.add_argument("--move-cut", action="store_true")
    ap.add_argument("--conversations", type=int, choices=range(1, 5), default=4)
    a = ap.parse_args()
    env = dict(e.split("=", 1) for e in a.env)
    if a.scenario == "config":
        Path(a.out).write_text(json.dumps(make_run_config(a, env), indent=1), encoding="utf-8")
        return 0
    with Sampler(a.out):
        return serve(a, env) if a.scenario == "server" else run_engine(a, env)


def get_windows_free_memory() -> tuple[int, int]:
    """Gets free commit charge and physical RAM for the Windows kit's group checks."""
    class Status(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in ("total_phys", "avail_phys", "total_page", "avail_page",
                                                   "total_virtual", "avail_virtual", "avail_extended")]
    status = Status()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return status.avail_page, status.avail_phys


def watch_group_memory(process: subprocess.Popen, output: str) -> threading.Event:
    """Kills only this check's engine if free physical RAM falls below 4 GiB."""
    stop = threading.Event()
    def watch():
        while not stop.wait(0.25) and process.poll() is None:
            try:
                commit, physical = get_windows_free_memory()
                if physical >= 4 * 2**30:
                    continue
                reason = f"free commit {commit / 2**30:.2f} GiB, free physical RAM {physical / 2**30:.2f} GiB"
            except OSError as error:
                with open(output + ".watchdog-read-error", "a", encoding="utf-8") as log:
                    log.write(f"PID {process.pid}: {error}\n")
                continue
            with open(output + ".watchdog", "a", encoding="utf-8") as log:
                log.write(f"PID {process.pid}: {reason}\n")
            process.kill()
            return
    threading.Thread(target=watch, daemon=True).start()
    return stop


def run_engine(a, env: dict[str, str]) -> int:
    """Runs a scenario of the engine's --serve protocol. Returns 0 if the engine got through it and exited with 0."""
    if a.cards:
        env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=a.cards)
    if a.scenario == "groups":
        commit, physical = get_windows_free_memory()
        print(f"[groups] before start: free commit {commit / 2**30:.2f} GiB; "
              f"free physical RAM {physical / 2**30:.2f} GiB", flush=True)

    deepseek = a.scenario in ("ds", "dsmemory", "dscutmove", "dsgroups")
    if deepseek and os.name == "nt":
        import psutil
        blockers = {"docker desktop", "com.docker.backend", "vmmem", "vmmemwsl"}
        for process in psutil.process_iter(["name"]):
            if (process.info["name"] or "").lower().removesuffix(".exe") in blockers:
                raise ValueError(f"DeepSeek check blocked by {process.info['name']}")
        commit, physical = get_windows_free_memory()
        print(f"[DeepSeek] before start: free commit {commit / 2**30:.2f} GiB; "
              f"free physical RAM {physical / 2**30:.2f} GiB", flush=True)

    e = Engine([a.exe, "--serve", *model_args(a), *a.args.split()], env, a.out)
    stop_guard = (watch_group_memory(e.p, a.out)
                  if a.scenario == "groups" or (deepseek and os.name == "nt") else None)
    failed = False
    try:
        e.until("READY")
        wait_for_experts(lambda: e.ask_quietly("GET_MEMORY_USE", "END_MEMORY"))
        e.record_memory("ready")
        SCENARIOS[a.scenario](e, a)
        e.record_memory("after scenario")
        e.send("QUIT")
    except (SystemExit, queue.Empty, OSError, ValueError, TypeError) as x:
        failed = True
        e.note(f"STOPPED: {x or 'no output for 15 minutes'}")
    code = e.finish(a.scenario == "multi")
    if stop_guard:
        stop_guard.set()
    if code == 0 and a.move_cut and a.scenario in ("cutmove", "dscutmove"):
        if problem := get_cut_move_problem(Path(a.out)):
            with open(a.out, "a", encoding="utf-8") as out:
                out.write(f"STOPPED: {problem}\nexit 1\n")
            return 1
    return 1 if failed else code


class Engine:
    """Starts an engine with --serve, sends it command lines and copies its stdout to a file."""

    def __init__(self, cmd: list[str], env: dict[str, str], out_path: str):
        self.err_path = out_path + ".stderr"
        env = {**os.environ, **env, "PATH": str(Path(cmd[0]).parent) + os.pathsep + os.environ["PATH"]}
        with open(self.err_path, "w", encoding="utf-8") as err:
            self.p = subprocess.Popen(cmd, cwd=str(ROOT), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                                      text=True, encoding="utf-8", errors="replace", bufsize=1, env=env)
        self.lines: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=self.pump, daemon=True).start()
        self.out = open(out_path, "w", encoding="utf-8", newline="\n")
        self.note(f">>> start: {' '.join(map(str, cmd[1:]))}")

    def pump(self):
        for line in self.p.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def note(self, s: str):
        self.out.write(s + "\n")
        self.out.flush()

    def until(self, *ends: str) -> str:
        """Copies stdout lines to the file until one starts with one of `ends`, and returns that one."""
        while True:
            line = self.lines.get(timeout=900)
            if line is None:
                raise SystemExit("the engine ended early")
            self.out.write(line)
            if line.startswith(ends):
                self.out.flush()
                return line

    def send(self, cmd: str, *ends: str):
        self.note(f">>> {cmd}")
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()
        if ends:
            self.until(*ends)

    def gen(self, head: str, ids: list[int]) -> list[int]:
        """Sends `GEN <head> <ids>` and returns the prompt followed by the answer's tokens."""
        self.note(f">>> GEN {head} {describe(ids)}")
        self.p.stdin.write(f"GEN {head} {','.join(map(str, ids))}\n")
        self.p.stdin.flush()
        got = []
        while (line := self.until("T ", "DONE", "ERR")).startswith("T "):
            got.append(int(line.split()[1]))
        if line.startswith("ERR"):
            raise ValueError(line)
        return ids + got

    def note_settings(self, text: str):
        """Appends `text` to FILE.settings, which keeps the answers to CHANGE_SETTINGS out of the compared output."""
        with open(self.err_path.removesuffix(".stderr") + ".settings", "a", encoding="utf-8") as f:
            f.write(text)

    def record_memory(self, point: str):
        """Keeps allocation fields and driver process usage beside this run for VRAM accounting."""
        path = Path(self.err_path.removesuffix(".stderr") + ".memory")
        with path.open("a", encoding="utf-8") as out:
            out.write(f">>> {point}: GET_MEMORY_USE\n")
            out.writelines(self.ask_quietly("GET_MEMORY_USE", "END_MEMORY"))
            out.write(f">>> nvidia-smi processes; engine PID {self.p.pid}\n")
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv"],
                    capture_output=True, text=True, timeout=10, check=False)
                out.write(result.stdout + result.stderr)
                out.write(f"nvidia-smi exit {result.returncode}\n")
            except (OSError, subprocess.TimeoutExpired) as error:
                out.write(f"process VRAM unavailable: {error}\n")

    def ask_quietly(self, cmd: str, end: str) -> list[str]:
        """Sends `cmd` and returns its answer's lines up to `end` (or ERR) without copying them to the output."""
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()
        lines = []
        while (line := self.lines.get(timeout=900)) is not None:
            lines.append(line)
            if line.startswith((end, "ERR")):
                return lines
        raise SystemExit("the engine ended early")

    def bgen(self, slot: int, max_new: int, ids: list[int], changes: dict[int, str] | None = None) -> list[int]:
        """Sends a BGEN, waits for its slot to end, and returns the prompt followed by the answer's tokens.  `changes`:
        after how many tokens to send which CHANGE_SETTINGS fields (the engine takes them between two of the slot's
        steps); their answers go to the .settings file, not the output."""
        self.note(f">>> BGEN {slot} {max_new} {describe(ids)}")
        self.p.stdin.write(f"BGEN {slot} {max_new} {','.join(map(str, ids))}\n")
        self.p.stdin.flush()
        got, changes, answering = [], dict(changes or {}), False
        while True:
            line = self.lines.get(timeout=900)
            if line is None:
                raise SystemExit("the engine ended early")
            f = line.split() or [""]
            # the answer to a change comes between two steps, and tokens decoded before it may come first (and the
            # slot's end, when the move forgot its conversation)
            if answering and f[0] not in ("T", "BT", "BDONE"):
                self.note_settings(line)
                answering = f[0] not in ("END_MEMORY", "ERR")
                continue
            self.out.write(line)
            if f[0] == "T" or f[0] == "BT" and f[1] == str(slot):
                got.append(int(f[-1]))
                if (fields := changes.pop(len(got), None)) is not None:
                    self.note_settings(f">>> after {len(got)} tokens: CHANGE_SETTINGS {fields}\n")
                    self.p.stdin.write(f"CHANGE_SETTINGS {fields}\n")
                    self.p.stdin.flush()
                    answering = True
            elif f[0] == "ERR" or f[0] == "BDONE" and f[1] == str(slot) or f[0] == "BADM" and f[2] == "0":
                if answering:
                    while True:
                        reply = self.lines.get(timeout=900)
                        if reply is None:
                            raise SystemExit("the engine ended before its settings answer")
                        self.note_settings(reply)
                        if reply.startswith(("END_MEMORY", "ERR")):
                            break
                self.out.flush()
                return ids + got

    def admit(self, cmds: list[str]):
        """Sends BGEN lines together and copies the output until each of their slots has ended."""
        for c in cmds:
            self.send(c)
        left = len(cmds)
        while left:
            f = self.until("BDONE", "BADM", "ERR").split()
            if f[0] != "BADM" or f[2] == "0":
                left -= 1

    def finish(self, with_startup: bool) -> int:
        """Waits for the engine to end, writes the rest of its output (and the startup lines) and "exit <code>"."""
        self.p.stdin.close()
        try:
            while (rest := self.lines.get(timeout=300)) is not None:
                self.out.write(rest)
            code = self.p.wait(timeout=300)
        except (queue.Empty, subprocess.TimeoutExpired):
            self.p.kill()
            code = "killed"
        if with_startup:
            self.note(">>> startup lines")
            self.out.write(startup_lines(Path(self.err_path).read_text(encoding="utf-8", errors="replace")))
        self.note(f"exit {code}")
        self.out.close()
        return 0 if code == 0 else 1


def model_args(a) -> list[str]:
    """Gets the engine's model arguments for a scenario, from the files in --data (README.md lists them)."""
    d = Path(a.data)
    if a.scenario == "q36":
        gguf = next((d / "models/qwen36-UD-Q4_K_XL").glob("*.gguf"))
        return ["--native", str(gguf), "--max-context", "32768", "--prefill", "auto", "--mtp", "2", "--batch", "2"]
    if a.scenario in ("ds", "dsmemory", "dscutmove", "dsgroups"):
        return ["--pack", str(deepseek_pack(a)), "--max-context", "8192", "--eos", "1", "--dspark-conf", "0.65",
                *(["--batch", "1"] if a.scenario == "dscutmove" else [])]
    more = {"q38": ["--batch", "2"], "q38state": ["--conversation-cache-mib", "4096"], "cutmove": ["--batch", "1"]}
    return qwen38_args(d, drafting=a.scenario != "groups") + more.get(a.scenario, [])


def qwen38_args(d: Path, drafting: bool = True) -> list[str]:
    """Gets Qwen3.8 IQ3_S's arguments as setup writes them, at a 32K context; the drafter when mtp/rt is there."""
    shards = sorted((d / "models/IQ3_S").glob("*IQ3_S-0000?-of-00002.gguf"))
    args = ["--pack", d / "packs/iq3_s", "--native", shards[0], "--ple-gguf", shards[1], "--expert-profile",
            ROOT / "data/expert-profile.bin", "--expert-cache", "auto", "--prefill", "auto", "--max-context", "32768",
            "--kv", "int8"]
    if drafting and (d / "mtp/rt").is_dir():
        args += ["--spec", "4", "--spec-min-p", "0.5", "--mtp", d / "mtp/rt"]
    return [str(x) for x in args]


def deepseek_pack(a) -> Path:
    """Copies the DeepSeek pack beside the output, pointed at the GGUF folder of this PC and with the shipped
    hot-expert profile (each engine start rewrites the pack's profile)."""
    dst = Path(a.out).parent / "dspack"
    shutil.copytree(Path(a.data) / DS_PACK, dst, dirs_exist_ok=True)
    info = (dst / "ds4.txt").read_bytes()
    (dst / "ds4.txt").write_bytes(re.sub(rb"(?m)^model_dir .*?(\r?)$", lambda m: b"model_dir " +
                                         str(Path(a.deepseek)).encode() + m[1], info))
    shutil.copyfile(ROOT / "data/dsv41/profile-2.00.bin", dst / "expert-profile.bin")
    return dst


def q38(e: Engine, a):
    p = list(range(1000, 1040))
    e.send("LIST_CONVERSATIONS", "END_CONVERSATIONS")
    e.gen("24", p)
    e.gen("16 temperature=0 seed=3 cvec=1", p + [1, 2, 3])
    e.send("GET_CONTEXT_WINDOWS main", "END_CONTEXT_WINDOWS")
    for bad in ("GEN x 1,2", "GEN 4", "FOO bar", "BGEN 7 4 1,2", "BGEN 0 0 1,2", "GENI 4 C:/nonexistent.bin 1,2"):
        e.send(bad, "ERR")
    e.gen("8 logit_bias=bad", p)
    e.gen("8 top_k=1 penalty_repeat=1.0 pin=4 ckpt=0 unknown_key=7", p)
    for c in ("BSTOP 0", "BYIELD 1", "STOP"):
        e.send(c)
    e.admit([f"BGEN 0 12 {ids(p + [5])}"])
    e.admit([f"BGEN 0 20 {ids(p + [7])}", f"BGEN 1 20 {ids(p + [8, 9])}"])
    look = ["LIST_CONVERSATIONS"] + [f"GET_CONTEXT_WINDOWS {w}" for w in ("main", "slot0", "slot1")]
    x, y = e.bgen(0, 12, list(range(1000, 1200))), e.bgen(1, 12, list(range(2000, 2150)))
    looks(e, look)
    x = e.gen("8", x + [5])                    # continues slot 0's conversation: copied back from the slot
    x, y = e.bgen(0, 10, x + [6]), e.bgen(1, 10, y + [7])
    looks(e, look)
    e.gen("8", y + [8, 9])                     # slot 1's back into the main session
    looks(e, look)


def q38state(e: Engine, a):
    look = ["LIST_CONVERSATIONS", "GET_CONTEXT_WINDOWS main"]
    saved = a.out + ".ses"
    x = e.gen("16", list(range(1000, 1300)))
    x = e.gen("12", x + [5, 6, 7])             # prefix reuse
    y = e.gen("12", list(range(2000, 2260)))   # another conversation: the first is parked
    x = e.gen("12", x + [8, 9])                # and comes back from the cache
    looks(e, look)
    e.send(f"SAVE {saved}", "SAVED", "SERR", "ERR")
    e.gen("12", y + [1])
    for f, more in ((saved, [10]), (a.other or saved, [11])):
        e.send(f"RESTORE {f}", "RESTORED", "SERR", "ERR", "FATAL")
        looks(e, look)
        e.gen("12", x + more)
    e.send("RESTORE C:/nonexistent.ses", "RESTORED", "SERR", "ERR", "FATAL")


def q36(e: Engine, a):
    p = list(range(1000, 1040))
    e.send("LIST_CONVERSATIONS", "END_CONVERSATIONS")
    e.gen("24", p)
    e.gen("16 temperature=0 seed=3 top_k=5 img_keys=zz unknown=1", p + [1, 2, 3])
    e.gen("16 temperature=0.8 top_p=0.9 top_k=30 min_p=0.05 seed=7", p + [4])
    e.gen("16 penalty_last_n=64 penalty_repeat=1.2 penalty_freq=0.1 penalty_present=0.2", p + [4])
    e.send("GET_CONTEXT_WINDOWS slot0", "END_CONTEXT_WINDOWS")
    for bad in ("GET_CONTEXT_WINDOWS nosuch", "GEN x 1,2", "GEN 4", "FOO bar", "GENX 4 1,2", "BSTOP", "BGEN 7 4 1,2",
                "BGEN 0 0 1,2", "GEN 4 1,999999999", "GENI 4 C:/nonexistent.bin 1,2", "GENI 4 onlyfile",
                f"GEN 40000 {ids(p)}"):
        e.send(bad, "ERR", "END_CONTEXT_WINDOWS")
    for c in ("", "BSTOP 0", "BYIELD 1", "STOP"):
        e.send(c)
    e.gen("8", p + [9])
    e.admit([f"BGEN 0 12 {ids(p + [5])}"])
    e.admit([f"BGEN 1 12 temperature=0.7 seed=11 {ids(p + [6])}"])
    e.admit([f"BGEN 0 20 {ids(p + [7])}", f"BGEN 1 20 {ids(p + [8, 9])}"])
    e.send("LIST_CONVERSATIONS", "END_CONVERSATIONS")


def multi(e: Engine, a):
    tok = tokenizer(Path(a.data) / "packs/iq3_s")
    chat = lambda text: tok.encode(text, parse_special=True)  # noqa: E731
    doc = tok.encode((ROOT / "docs/ARCHITECTURE.md").read_text(encoding="utf-8"))[:1950]
    x = e.gen("128", chat(QWEN_USER + "Summarize this document in five bullet points.\n\n") + doc + chat(QWEN_ANSWER))
    e.gen("128", x + chat("<|im_end|>\n" + QWEN_USER + "Now name three open questions it leaves." + QWEN_ANSWER))


def check_groups(e: Engine, a):
    """Compares concurrent greedy answers with standalone and times decoding after the last admission."""
    deepseek = getattr(a, "scenario", "groups") == "dsgroups"
    tok = tokenizer(Path(a.data) / (DS_PACK if deepseek else "packs/iq3_s"))
    prefix = "The lighthouse keeper counts ships. " * 18 if deepseek else ""
    prompts = [tok.encode(render_deepseek_prompt(prefix + text) if deepseek else QWEN_USER + text + QWEN_ANSWER,
                          parse_special=True) for text in (
        "Write a Python function to compute Fibonacci numbers, with comments.",
        "Tell a long story about a sailor who discovers a hidden island.",
        "Count from one to one hundred, separated by commas.",
        "Explain how a bicycle works in detail.")]
    prompts = prompts[:getattr(a, "conversations", 4)]
    standalone = [e.gen("96 temperature=0", ids)[len(ids):] for ids in prompts]
    for round_index in range(2):
        answers = [[] for _ in prompts]
        done = set()
        remaining = 0
        for slot, ids in enumerate(prompts):
            e.send(f"BGEN {slot} 96 temperature=0 " + ",".join(map(str, ids)))
            while True:
                line = e.until("T ", "BT ", "BDONE ", "BADM ", "ERR")
                if line.startswith("ERR"):
                    raise ValueError(line)
                if line.startswith("T "):
                    answers[slot].append(int(line.split()[1]))
                elif line.startswith("BT "):
                    _, index, token = line.split()
                    answers[int(index)].append(int(token))
                elif line.startswith("BDONE "):
                    done.add(int(line.split()[1]))
                elif int(line.split()[1]) == slot:
                    break
        started = time.perf_counter()
        while len(done) < len(prompts):
            line = e.until("BT ", "BDONE ", "ERR")
            if line.startswith("ERR"):
                raise ValueError(line)
            if line.startswith("BT "):
                _, slot, token = line.split()
                answers[int(slot)].append(int(token))
                remaining += 1
            else:
                done.add(int(line.split()[1]))
        elapsed = time.perf_counter() - started
        e.note(f"SPEED groups decode={remaining / elapsed:.4f} prompt=0")
        different = [str(slot) for slot, (answer, expected) in enumerate(zip(answers, standalone))
                     if answer != expected]
        e.note("GROUP_IDENTITY differs slots=" + ",".join(different) if different else
               f"GROUP_IDENTITY passed slots={len(prompts)}")
        e.note(f"GROUP_TOKENS {round_index} " + json.dumps(answers, separators=(",", ":")))


def cutmove(e: Engine, a):
    deepseek = a.scenario == "dscutmove"
    tok = tokenizer(Path(a.data) / (DS_PACK if deepseek else "packs/iq3_s"))
    text = "Write a long, detailed story about a lighthouse keeper who counts ships."
    prompt = render_deepseek_prompt(text) if deepseek else QWEN_USER + text + QWEN_ANSWER
    story = tok.encode(prompt, parse_special=True)
    changes = {}
    if a.move_cut:
        report = "".join(e.ask_quietly("GET_MEMORY_USE", "END_MEMORY"))
        e.note_settings(">>> initial GET_MEMORY_USE\n" + report)
        match = re.search(r"memory_mib\.gpu0=(\d+)", report)
        if not match:
            raise SystemExit("GET_MEMORY_USE did not report memory_mib.gpu0")
        mib = int(match[1])
        changes = {24: f"memory_mib.gpu0={mib // 2}", 48: f"memory_mib.gpu0={mib}"}
    e.bgen(0, 96, story, changes)


def dsmemory(e: Engine, a):
    """Checks greedy tokens on the window and chunked prompt paths, including a continued conversation."""
    tok = tokenizer(Path(a.data) / DS_PACK)
    chat = lambda text: tok.encode(render_deepseek_prompt(text), parse_special=True)  # noqa: E731
    answer = e.gen("48 temperature=0", chat("Write an iterative Python Fibonacci function. Code only."))
    e.gen("24 temperature=0", answer)
    e.gen("32 temperature=0", chat("Summarize in one sentence: " + "The lighthouse keeper counts ships. " * 180))


def ds(e: Engine, a):
    tok = tokenizer(Path(a.data) / DS_PACK)
    chat = lambda text: tok.encode(render_deepseek_prompt(text), parse_special=True)  # noqa: E731
    code = chat("Write a Python function that returns the n-th Fibonacci number iteratively. Code only.")
    e.gen("48", code)
    e.gen("32 temperature=0.8 top_p=0.9 top_k=30 min_p=0.05 seed=7",
          chat("Tell me a two-sentence story about a lighthouse."))
    e.gen("64", chat("Count from 1 to 40, separated by commas."))
    e.gen("128", code)


def render_deepseek_prompt(text: str) -> str:
    """Formats one user turn with reasoning closed, as the DeepSeek smoke checks do."""
    return f"<｜begin▁of▁sentence｜><｜User｜>{text}<｜Assistant｜></think>"


SCENARIOS = {"q38": q38, "q38state": q38state, "q36": q36, "multi": multi, "groups": check_groups,
             "cutmove": cutmove, "ds": ds,
             "dsgroups": check_groups, "dsmemory": dsmemory, "dscutmove": cutmove}


def looks(e: Engine, cmds: list[str]):
    for c in cmds:
        e.send(c, "END_CONVERSATIONS" if c == "LIST_CONVERSATIONS" else "END_CONTEXT_WINDOWS", "ERR")


def ids(v: list[int]) -> str:
    return ",".join(map(str, v))


def describe(prompt: list[int]) -> str:
    """Gets a prompt's length and a digest of it for the output, so that report.py compares an answer only between
    runs that sent the same prompt."""
    return f"<{len(prompt)} ids {zlib.crc32(ids(prompt).encode()):08x}>"


class Sampler:
    """Records once a second, while a check runs: each card's clocks, temperature, power, load, memory, PCIe link
    and throttle reasons (nvidia-smi into FILE.gpu.csv, the columns of report.GPU_FIELDS) and the PC's CPU load, free
    RAM, page file use and disk reads (FILE.sys.csv)."""

    def __init__(self, out: str):
        self.out = out
        self.stop = threading.Event()
        self.smi: subprocess.Popen | None = None

    def __enter__(self):
        # drivers since 535 call the last column clocks_event_reasons; some still take the old name
        for reasons in ("clocks_throttle_reasons.active", "clocks_event_reasons.active"):
            if not shutil.which("nvidia-smi"):
                break
            fields = ",".join(GPU_FIELDS[:-1] + (reasons,))
            self.smi = subprocess.Popen(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits",
                                         "-lms", "1000", "-f", self.out + ".gpu.csv"], stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL)
            try:
                self.smi.wait(2)               # it exits at once on a column it does not know
            except subprocess.TimeoutExpired:
                break
        threading.Thread(target=self.sample_pc, daemon=True).start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        if self.smi and self.smi.poll() is None:
            self.smi.terminate()

    def sample_pc(self):
        import psutil
        read = lambda: getattr(psutil.disk_io_counters(), "read_bytes", 0)  # noqa: E731  (None without counters)
        start, last = time.time(), read()
        psutil.cpu_percent()
        with open(self.out + ".sys.csv", "w", encoding="utf-8", newline="\n") as f:
            f.write("seconds,cpu_pct,ram_free_gb,pagefile_pct,disk_read_mbs\n")
            while not self.stop.wait(1):
                now = read()
                f.write(f"{time.time() - start:.0f},{psutil.cpu_percent():.0f},"
                        f"{psutil.virtual_memory().available / 2**30:.1f},{psutil.swap_memory().percent:.0f},"
                        f"{(now - last) / 2**20:.0f}\n")
                f.flush()
                last = now


def serve(a, env: dict[str, str]) -> int:
    """Starts the server with this engine (its run config beside the output), asks it two greedy questions, writes
    the answers and their speed, and ends the server and its engine."""
    import psutil
    out = Path(a.out)
    out.with_suffix(".json").write_text(json.dumps(make_run_config(a, env), indent=1), encoding="utf-8")
    doc = (ROOT / "docs/ARCHITECTURE.md").read_text(encoding="utf-8")[:8500]
    prompts = {"short": "What is the capital of Australia? Answer in one sentence.",
               "long": "Read the following document and summarize its main rules in five bullet points.\n\n" + doc}
    lines, code, p = [], 1, None
    try:
        if is_port_taken():
            raise OSError(f"port {PORT} is taken")
        with open(str(out) + ".stderr", "w", encoding="utf-8") as err:
            p = subprocess.Popen([sys.executable, str(ROOT / "strata_server/server.py"), "--engine", "strata",
                                  "--config", str(out.with_suffix(".json")), "--port", str(PORT)], cwd=str(ROOT),
                                 stdout=err, stderr=err)
        wait_healthy(p, 1800)
        for name, text in prompts.items():
            body = {"model": "x", "messages": [{"role": "user", "content": text}], "temperature": 0,
                    "presence_penalty": 0, "max_tokens": 200, "chat_template_kwargs": {"enable_thinking": False}}
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as r:
                got = json.load(r)
            t = got.get("timings") or {}
            speed = f"decode={t.get('predicted_per_second') or 0} prompt={t.get('prompt_per_second') or 0}"
            lines += [f">>> {name}", got["choices"][0]["message"].get("content") or ""]
            if name == "long":                 # the short answer is too short to time
                lines.append(f"SPEED {name} {speed}")
        code = 0
    except (OSError, ValueError, KeyError) as x:
        lines.append(f"STOPPED: {x}")
    finally:
        try:
            for proc in psutil.Process(p.pid).children(recursive=True) + [psutil.Process(p.pid)] if p else []:
                proc.kill()
        except psutil.Error:
            pass
    out.write_text("\n".join(lines + [f"exit {code}"]) + "\n", encoding="utf-8", newline="\n")
    return code


def make_run_config(a, env: dict[str, str]) -> dict:
    """Gets a run config (strata-<model>.json, as setup writes it) for Qwen3.8 with this engine, the files in --data
    and --args, on --cards; its log goes beside --out."""
    return {"exe": a.exe, "args": qwen38_args(Path(a.data)) + a.args.split(), "cwd": str(ROOT), "port": PORT,
            "tokenizer": str(Path(a.data) / "packs/iq3_s/tokenizer"), "model_name": "qwen3.8-flash-next-iq3_s",
            "log": str(a.out) + ".engine.log", "lib_dirs": [str(Path(a.exe).parent)], "backend": "cuda", "env": env,
            "gpu": [int(c) for c in a.cards.split(",") if c]}


def is_port_taken() -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", PORT)) == 0


def wait_healthy(p: subprocess.Popen, timeout_s: int):
    """Waits until the server answers /health with 200 (it has loaded the engine); TimeoutError otherwise."""
    end = time.time() + timeout_s
    while time.time() < end and p.poll() is None:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                if r.status == 200:
                    wait_for_http_experts(f"http://127.0.0.1:{PORT}", timeout=max(0.1, end - time.time()))
                    return
        except OSError:
            pass
        time.sleep(5)
    raise TimeoutError(f"the server was not ready (exit code {p.poll()})")


if __name__ == "__main__":
    sys.exit(main())
