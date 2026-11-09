"""tools/measure_rows.py - strata-q38's decode speed by conversations at once and rows per step (docs/DETAILS.md,
"Rows per step, lookahead and conversations at once"; docs/MEMORY_MANAGER_PLAN.md, step M7).

One engine start runs every configuration (conversations at once x rows per step, with a lookahead) on story, code and
chat prompts, changing the settings live between them (CHANGE_SETTINGS); each round runs them all, the order
reversed every other round, after an unrecorded warm-up round that captures the verify windows' graphs.  Where the
experts live is set by the engine's arguments: run it once per memory case.

    python tools/measure_rows.py --config strata-iq3_s.json --exe engine/strata-q38.exe --case ram --rounds 3 \
        --out results/rows-ram -- --gpu-mib 20480

`--config`: a run config (strata-<model>.json: the model's arguments, the environment, the library folders, the
tokenizer); its arguments are used as they are (they need --mtp, the drafter), with 3 batch slots, 8 rows and no
suffix drafting added, and `--vision` left out.  The arguments after `--` go last (they win).  `--report A.json
B.json ...` prints the medians of earlier runs as Markdown tables instead.

A conversation alone decodes each prompt (GEN, its rate from the engine's DONE line); several decode at once in
batch slots (BGEN), their rate counted from the last admission to the first conversation that ends.  The guesses kept
come from the engine's stderr, and so do the experts the steps routed to (each counted once a step, and only where the
CPU sees the routing: with every expert on the GPU the device decides alone and none are counted).
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import statistics
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strata_tokenizer  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strata_common.expert_fill import wait_for_experts  # noqa: E402

PROMPTS = {
    "story": "Write a long, detailed story about a lighthouse keeper who counts ships. Use at least 600 words.",
    "code": "Write a Python module with a class for a bank account (deposit, withdraw, transfer, statement), with "
            "docstrings and unit tests. Use at least 150 lines.",
    "chat": "My partner and I are spending a long weekend in Lisbon in spring. We like food, old neighbourhoods and a "
            "bit of walking, not museums all day. How would you plan the three days?",
}
KINDS = list(PROMPTS)
# Name: (conversations at once, rows per step, lookahead).  Several conversations take story, code and chat in turn;
# 1x3 against 3x3 and 1x8 against 3x8 compare guesses with other conversations' tokens at the same rows.
CONFIGS = {"1x1": (1, 1, 0), "1x3": (1, 3, 2), "1x4": (1, 4, 3), "1x8": (1, 8, 7),
           "2x2": (2, 2, 0), "2x6": (2, 6, 7), "3x3": (3, 3, 0), "3x8": (3, 8, 7)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", help="the run config (strata-<model>.json)")
    ap.add_argument("--exe", help="strata-q38 (default: the run config's exe)")
    ap.add_argument("--case", default="default", help="a name for where the experts live, kept in the results")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=160, help="tokens each conversation decodes")
    ap.add_argument("--only", help="configurations to run, comma-separated (default: all)")
    ap.add_argument("--out", help="the results' path without .json; the engine's output goes beside it")
    ap.add_argument("--report", nargs="+", metavar="JSON", help="print the medians of these results and exit")
    ap.add_argument("engine_args", nargs=argparse.REMAINDER, help="-- then engine arguments that go last")
    a = ap.parse_args()
    if a.report:
        print_report([json.load(open(p, encoding="utf-8")) for p in a.report])
        return
    if not (a.config and a.out):
        ap.error("--config and --out are needed (or --report)")
    configs = {k: v for k, v in CONFIGS.items() if not a.only or k in a.only.split(",")}
    measure(a, configs, [x for x in a.engine_args if x != "--"])


def measure(a, configs: dict, extra: list[str]) -> None:
    """Starts the engine once, runs the warm-up and `a.rounds` rounds of `configs`, and writes `a.out`.json."""
    cfg = json.load(open(a.config, encoding="utf-8"))
    args = [x for x in cfg["args"] if x != "--vision"]
    args += ["--suffix-draft", "0", "--batch", "3", "--concurrent", "3", "--rows", "8", "--lookahead", "3", *extra]
    env = dict(os.environ, **cfg.get("env", {}))
    env["PATH"] = os.pathsep.join(cfg.get("lib_dirs", [])) + os.pathsep + env["PATH"]
    tok = strata_tokenizer.load_folder(cfg["tokenizer"])
    prompts = {k: tok.encode(f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                             parse_special=True) for k, text in PROMPTS.items()}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    e = Engine([a.exe or cfg["exe"], "--serve", *args], cfg.get("cwd"), env, a.out)
    e.until("READY")
    wait_for_experts(e.ask_memory)
    out = {"case": a.case, "args": args, "tokens": a.tokens, "memory_at_start": e.ask_memory(), "rounds": []}
    print("warm-up", flush=True)
    for name in configs:
        run_config(e, prompts, configs[name], 24)
    names = list(configs)
    for rnd in range(a.rounds):
        got = {}
        for name in names if rnd % 2 == 0 else names[::-1]:
            got[name] = run_config(e, prompts, configs[name], a.tokens)
            print(f"{rnd} {name}: {describe(got[name])}", flush=True)
        out["rounds"].append(got)
        json.dump(out, open(a.out + ".json", "w", encoding="utf-8"), indent=1)
    out["memory_at_end"] = e.ask_memory()
    json.dump(out, open(a.out + ".json", "w", encoding="utf-8"), indent=1)
    e.quit()
    print_report([out])


def run_config(e: "Engine", prompts: dict, config: tuple, n: int) -> dict:
    """Sets `config` live and decodes `n` tokens of each prompt (one conversation) or of the first c prompts at once;
    gets the rates and what the engine's stderr said meanwhile."""
    c, rows, lookahead = config
    e.ask(f"CHANGE_SETTINGS concurrent={c} rows={rows} lookahead={lookahead}", "END_MEMORY")
    at = e.err_size()
    res = {k: e.solo(prompts[k], n) for k in KINDS} if c == 1 else e.together([prompts[k] for k in KINDS[:c]], n)
    time.sleep(0.2)   # (the engine's last stderr lines of the run)
    res["err"] = parse_stderr(e.err_since(at))
    return res


def parse_stderr(text: str) -> dict:
    """Sums the guesses, the routing counts and the copies back that strata-q38's stderr reported in `text`."""
    def total(pattern, n):
        found = [m.groups() for m in re.finditer(pattern, text)]
        return [sum(float(g[i]) for g in found) for i in range(n)]
    right, guessed = total(r"(\d+) of (\d+) guesses right", 2)
    steps, rows, experts = total(r"(\d+) decode steps of (\d+) rows in all routed to (\d+) experts", 3)
    copied, copy_ms = total(r"(\d+) it evicted copied back to RAM \(([\d.]+) ms", 2)
    return {"right": int(right), "guessed": int(guessed), "steps": int(steps), "rows": int(rows),
            "experts": int(experts), "copied_back": int(copied), "copy_back_ms": copy_ms}


def describe(r: dict) -> str:
    """One line of a configuration's results, as the rounds go."""
    er = r["err"]
    rate = (" ".join(f"{k} {r[k]['tok_s']:.2f}" for k in KINDS) if "story" in r else
            f"together {r['tok_s']:.2f} (" + ", ".join(f"{v:.2f}" for v in r["each"]) + ")")
    per = f"; experts/step {er['experts'] / er['steps']:.0f}, rows/step {er['rows'] / er['steps']:.2f}" \
        if er["steps"] else ""
    return f"{rate} tok/s; guesses kept {er['right']}/{er['guessed']}{per}"


def print_report(runs: list[dict]) -> None:
    """Prints the medians of each run's rounds: one table of the rates, one of the guesses and the experts."""
    def med(values):
        values = list(values)
        return statistics.median(values) if values else float("nan")
    print("| case | config | " + " | ".join(KINDS) + " | together (each) |\n|---|---|" + "---:|" * (len(KINDS) + 1))
    for run in runs:
        for name in run["rounds"][0]:
            rs = [x[name] for x in run["rounds"]]
            if "story" in rs[0]:
                cells = [f"{med(x[k]['tok_s'] for x in rs):.2f}" for k in KINDS] + [""]
            else:
                each = ", ".join(f"{med(x['each'][i] for x in rs):.2f}" for i in range(len(rs[0]["each"])))
                cells = [""] * len(KINDS) + [f"{med(x['tok_s'] for x in rs):.2f} ({each})"]
            print(f"| {run['case']} | {name} | " + " | ".join(cells) + " |")
    print("\n| case | config | guesses kept | rows/step | experts/step | experts/row | copied back (ms) |\n"
          "|---|---|---:|---:|---:|---:|---:|")
    for run in runs:
        for name in run["rounds"][0]:
            ers = [x[name]["err"] for x in run["rounds"]]
            g, right = sum(x["guessed"] for x in ers), sum(x["right"] for x in ers)
            steps = [x for x in ers if x["steps"]]
            print(f"| {run['case']} | {name} | " + (f"{100 * right / g:.0f}%" if g else "-") + " | " +
                  (f"{med(x['rows'] / x['steps'] for x in steps):.2f} | "
                   f"{med(x['experts'] / x['steps'] for x in steps):.0f} | "
                   f"{med(x['experts'] / x['rows'] for x in steps):.0f}" if steps else "- | - | -") +
                  f" | {sum(x['copied_back'] for x in ers)} ({sum(x['copy_back_ms'] for x in ers):.1f}) |")


class Engine:
    """An engine started with --serve: its stdout lines one by one (copied to <out>.out), its stderr in
    <out>.out.stderr."""

    def __init__(self, cmd: list[str], cwd: str | None, env: dict, out: str):
        self.err_path = out + ".out.stderr"
        self.p = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=open(self.err_path, "w", encoding="utf-8"), text=True, encoding="utf-8",
                                  bufsize=1, env=env)
        self.lines: queue.Queue = queue.Queue()
        self.log = open(out + ".out", "w", encoding="utf-8", newline="\n")
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        for line in self.p.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def until(self, *starts: str) -> str:
        """Gets the next line that starts with one of `starts`; an ERR line ends the measurement."""
        while True:
            line = self.lines.get(timeout=1800)
            if line is None:
                raise SystemExit("the engine ended early")
            self.log.write(line)
            if line.startswith(starts):
                return line
            if line.startswith("ERR"):
                raise SystemExit(line)

    def send(self, command: str) -> None:
        """Writes `command` to the engine (and its first 200 characters to the log)."""
        self.log.write(f">>> {command[:200]}\n")
        self.log.flush()
        self.p.stdin.write(command + "\n")
        self.p.stdin.flush()

    def ask(self, command: str, end: str) -> list[str]:
        """Sends `command` and gets its answer's lines up to `end`."""
        self.send(command)
        lines = []
        while not (line := self.until("MEMORY", "EXPERT_FILL", "CONVERSATION", "SETTINGS",
                                      "SHORTFALL", end)).startswith(end):
            lines.append(line.strip())
        return lines

    def ask_memory(self) -> list[str]:
        """Gets the engine's GET_MEMORY_USE answer, its lines."""
        return self.ask("GET_MEMORY_USE", "END_MEMORY")

    def solo(self, ids: list[int], n: int) -> dict:
        """One conversation: a GEN of `n` tokens; its rate and guesses from DONE."""
        self.send(f"GEN {n} {','.join(map(str, ids))}")
        f = self.until("DONE").split()
        return {"tok_s": int(f[1]) * 1000 / float(f[4]), "right": int(f[6]), "guessed": int(f[7])}

    def together(self, prompts: list[list[int]], n: int) -> dict:
        """Several conversations: a BGEN of `n` tokens per prompt into slots 0.. (each read before the next); the
        tokens each slot gave from the last admission to the first BDONE, over that time."""
        for slot, ids in enumerate(prompts):
            self.send(f"BGEN {slot} {n} {','.join(map(str, ids))}")
            self.until("BADM")
        t0, counts, done, first = time.perf_counter(), [0] * len(prompts), set(), None
        while len(done) < len(prompts):
            f = self.until("BT ", "BDONE").split()
            if f[0] == "BT" and first is None:
                counts[int(f[1])] += 1
            elif f[0] == "BDONE":
                done.add(int(f[1]))
                first = first or time.perf_counter()
        dt = first - t0
        return {"tok_s": sum(counts) / dt, "each": [c / dt for c in counts]}

    def err_size(self) -> int:
        """Gets how far the engine's stderr file has grown, for err_since."""
        return os.path.getsize(self.err_path)

    def err_since(self, at: int) -> str:
        """Gets what the engine wrote to stderr after offset `at`."""
        with open(self.err_path, encoding="utf-8", errors="replace") as f:
            f.seek(at)
            return f.read()

    def quit(self) -> None:
        """Ends the engine and waits for it to exit."""
        self.send("QUIT")
        self.p.stdin.close()
        while self.lines.get(timeout=300) is not None:
            pass
        self.p.wait(timeout=300)
        self.log.close()


if __name__ == "__main__":
    main()
