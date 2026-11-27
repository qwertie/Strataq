"""Old (strata-dsv41) versus new (strata-ds41) DeepSeek V4.1 decode speed, single conversation, greedy.

Repeats the method of bench/results/2026-11-05-old-local-vs-new: same three prompts (token IDs taken from that study),
256 greedy tokens each, a private copy of the pack (and its expert profile) per start, old/new alternating per start
with an engine-free idle gap, warm-up cycles until the GPU's expert count stops changing, then measured cycles.

    python ds_h2h.py --out DIR [--mode off,on] [--starts 3] [--measured 2] [--smoke]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time

sys.path.insert(0, r"C:\AI-models\Strataq\tools")

ROOT = r"C:\AI-models\Strataq"
PACK = r"C:\AI-models\Strata-data\packs\deepseek-v4.1-flash-ist-das-2.00bpw"
BIN = os.path.join(ROOT, "build-hip-win")
LIB = os.path.join(ROOT, "engine", "rocm", "bin")
OLD_STUDY = os.path.join(ROOT, "bench", "results", "2026-11-05-old-local-vs-new", "measurements.json")
NAMES = ["seasons", "pacman", "story"]
PACMAN = "Write a traditional Pac-Man clone in plain HTML with embedded JS (controls: arrow keys and WASD)"
COMMON = ["--max-context", "524288", "--eos", "1", "--dspark-conf", "0.65"]
ENV = {"STRATA_COMMIT_RESERVE_MIB": "0", "DS4_UMA_GPU_RESERVE_GB": "4"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="off,on", help="DSpark modes to run, in order")
    ap.add_argument("--starts", type=int, default=3, help="starts per engine per mode")
    ap.add_argument("--measured", type=int, default=2, help="measured cycles per start")
    ap.add_argument("--gap", type=float, default=20)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--gpu-mib", type=int, default=51200)
    ap.add_argument("--ram-mib", type=int, default=36864)
    ap.add_argument("--old-cap", type=float, default=None, help="DS4_COMMIT_CAP_GB for old (matches GPU experts)")
    ap.add_argument("--engines", default="old,new")
    ap.add_argument("--smoke", action="store_true", help="one start per engine, one warm cycle, no measuring")
    a = ap.parse_args()
    a.out = os.path.abspath(a.out)
    os.makedirs(a.out, exist_ok=True)
    prompts = load_prompts()
    results_path = os.path.join(a.out, "measurements.json")
    res = json.load(open(results_path)) if os.path.exists(results_path) else {"runs": []}
    res["binaries"] = {e: {"path": exe_of(e), "sha256": sha256(exe_of(e))} for e in ("old", "new")}
    res["prompts"] = prompts
    done = {r["name"] for r in res["runs"] if "error" not in r}
    for mode in a.mode.split(","):
        for rep in range(1, (1 if a.smoke else a.starts) + 1):
            for eng in a.engines.split(",") if rep % 2 else a.engines.split(",")[::-1]:
                name = f"{eng}-d{mode}-r{rep}"
                if name in done:
                    continue
                time.sleep(a.gap)
                run = run_start(a, name, eng, mode, prompts)
                res["runs"] = [r for r in res["runs"] if r["name"] != name] + [run]
                json.dump(res, open(results_path, "w"), indent=1)
                print(summarize(run), flush=True)


def run_start(a, name: str, eng: str, mode: str, prompts: list[list[int]]) -> dict:
    """Starts one engine on a private pack copy, warms it, measures it and stops it; gets everything it said."""
    pack = os.path.join(a.out, name + "-pack")
    shutil.rmtree(pack, ignore_errors=True)
    shutil.copytree(PACK, pack, ignore=shutil.ignore_patterns("*.tmp"))
    env = dict(os.environ, **ENV)
    env["PATH"] = LIB + os.pathsep + env["PATH"]
    args = ["--serve", "--pack", pack, *COMMON, "--dspark", "on" if mode == "on" else "none"]
    if eng == "new":
        args += ["--gpu-mib", str(a.gpu_mib), "--ram-mib", str(a.ram_mib), "--rows", "8", "--expert-cache", "auto",
                 "--batch", "3", "--concurrent", "3", "--remote-experts", "off", "--save-on-shutdown", "off",
                 "--pause-dir", pack + "-pause"]
    else:
        args += ["--ssd-cache", f"{a.ram_mib * 1048576 / 1e9:.9f}", "--vram-experts", "1", "--dspark-depth", "7"]
        if a.old_cap:
            env["DS4_COMMIT_CAP_GB"] = str(a.old_cap)
    run = {"name": name, "engine": eng, "dspark": mode, "exe": exe_of(eng), "args": args,
           "env": {k: env[k] for k in ENV.keys() | {"DS4_COMMIT_CAP_GB"} if k in env}, "started": time.time(),
           "warm": [], "measured": [], "snapshots": []}
    print(f"== {name}", flush=True)
    e = Engine([exe_of(eng), *args], env, os.path.join(a.out, name))
    try:
        t0 = time.monotonic()
        run["startup"] = e.until_ready()
        run["ready_s"] = time.monotonic() - t0
        run["fill_s"] = e.wait_fill()
        run["snapshots"].append(e.memory())
        last = gpu_experts(run["snapshots"][-1])
        for cycle in range(1 if a.smoke else 4):
            run["warm"].append([e.gen(p, a.tokens) for p in prompts])
            run["snapshots"].append(e.memory())
            now = gpu_experts(run["snapshots"][-1])
            print(f"   warm {cycle}: {[round(x['tok_s'], 2) for x in run['warm'][-1]]} gpu experts {now}", flush=True)
            if cycle >= 1 and now == last:
                break
            last = now
        for _ in range(0 if a.smoke else a.measured):
            run["measured"].append([e.gen(p, a.tokens) for p in prompts])
            print(f"   measured: {[round(x['tok_s'], 2) for x in run['measured'][-1]]}", flush=True)
        run["snapshots"].append(e.memory())
        run["gpu_experts"] = gpu_experts(run["snapshots"][-1])
    except Exception as ex:  # keep what was measured, and say why it stopped
        run["error"] = repr(ex)
        print("   ERROR", ex, flush=True)
    finally:
        run["exit"] = e.quit()
        run["stderr_tail"] = e.err_tail()
        shutil.rmtree(pack + "-pause", ignore_errors=True)
    return run


def gpu_experts(snapshot: list[str]):
    """Gets the GPU's trunk (non-drafter) routed experts from a GET_MEMORY_USE answer: [count, bytes]. The old engine
    reports only bytes there (its count is the startup INFO's expert_slots), so its count is None."""
    from urllib.parse import unquote
    count, size, old = 0, 0, None
    for line in snapshot:
        p = line.split()
        if line.startswith("MEMORY gpu0 ") and (m := re.search(r" experts=(\d+)", line)):
            old = int(m.group(1))
        elif line.startswith("MEMORY_BLOCK gpu0 ") and p[3] == "experts" and not unquote(p[5]).startswith("Drafter"):
            size += int(p[2])
            count += len(p[-1].split(",")) if re.fullmatch(r"[\d,]+", p[-1]) else 0
    return [None, old] if old is not None else [count, size]


def summarize(run: dict) -> str:
    rates = [[round(x["tok_s"], 2) for x in cyc] for cyc in run["measured"]]
    return f"{run['name']}: {rates} gpu_experts={run.get('gpu_experts')} {run.get('error', '')}"


def load_prompts() -> list[list[int]]:
    """Gets the Nov 5 study's no-thinking seasons and story prompts as token IDs
    (from its GEN commands), with the Pac-Man coding prompt in place of its repetition prompt."""
    m = json.load(open(OLD_STUDY))
    cmds = next(r for r in m["runs"] if r["name"].startswith("ds41-old"))["protocol_commands"]
    gens = [c for c in cmds if c.startswith("GEN ")][:3]
    ids = [[int(x) for x in g.split()[-1].split(",")] for g in gens]
    import strata_tokenizer
    tok = strata_tokenizer.load_folder(os.path.join(PACK, "tokenizer"))
    ids[1] = tok.encode(f"<｜begin▁of▁sentence｜><｜User｜>{PACMAN}<｜Assistant｜></think>", parse_special=True)
    return ids


def exe_of(eng: str) -> str:
    return os.path.join(BIN, "strata-dsv41.exe" if eng == "old" else "strata-ds41.exe")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Engine:
    """An engine started with --serve: stdout read line by line (copied to <base>.out), stderr in <base>.err."""

    def __init__(self, cmd: list[str], env: dict, base: str):
        self.err_path = base + ".err"
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=ROOT,
                                  stderr=open(self.err_path, "w", encoding="utf-8"), text=True, encoding="utf-8",
                                  errors="replace", bufsize=1, env=env)
        self.lines: queue.Queue = queue.Queue()
        self.rid = 0
        self.log = open(base + ".out", "w", encoding="utf-8", newline="\n")
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        for line in self.p.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def next(self, timeout: float = 1800) -> str:
        line = self.lines.get(timeout=timeout)
        if line is None:
            raise RuntimeError("engine ended")
        self.log.write(line)
        self.log.flush()
        if line.startswith("ERR"):
            raise RuntimeError(line.strip())
        return line.rstrip("\n")

    def send(self, cmd: str) -> None:
        self.log.write(f">>> {cmd[:160]}\n")
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()

    def until_ready(self) -> list[str]:
        info = []
        while not (line := self.next(3600)).startswith("READY"):
            info.append(line)
        return info + [line]

    def memory(self) -> list[str]:
        self.send("GET_MEMORY_USE")
        out = []
        while not (line := self.next(120)).startswith("END_MEMORY"):
            out.append(line)
        return out

    def wait_fill(self, timeout: float = 1800) -> float:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            fields = next((ln.split()[1:] for ln in self.memory() if ln.startswith("EXPERT_FILL ")), [])
            if dict(f.split("=", 1) for f in fields if "=" in f).get("loaded", "1") == "1":
                return time.monotonic() - t0
            time.sleep(0.5)
        raise TimeoutError("experts still loading")

    def gen(self, ids: list[int], n: int) -> dict:
        """Greedy-decodes up to `n` tokens after `ids`; gets the tokens, the rate from DONE's decode time, and DONE."""
        self.rid += 1
        tag = f"R r{self.rid} "
        self.send(f"RUN r{self.rid} {n} temperature=0 " + ",".join(map(str, ids)))
        toks = []
        while True:
            line = self.next()
            if not line.startswith(tag):
                continue
            line = line[len(tag):]
            if line.startswith("ERR"):
                raise RuntimeError(line)
            if line.startswith("T "):
                toks.append(int(line.split()[1]))
            elif line.startswith("DONE"):
                f = line.split()
                produced, decode_ms = int(f[1]), float(f[4])
                return {"tokens": toks, "produced": produced, "decode_ms": decode_ms,
                        "tok_s": produced * 1000 / decode_ms if decode_ms else 0.0, "right": int(f[6]),
                        "guessed": int(f[7]), "done": line, "at": time.time()}

    def err_tail(self, n: int = 60) -> list[str]:
        with open(self.err_path, encoding="utf-8", errors="replace") as f:
            return f.read().splitlines()[-n:]

    def quit(self):
        try:
            self.send("QUIT")
            self.p.stdin.close()
        except OSError:
            pass
        try:
            return self.p.wait(timeout=300)
        except subprocess.TimeoutExpired:
            self.p.kill()
            return "killed"
        finally:
            self.log.close()


if __name__ == "__main__":
    main()
