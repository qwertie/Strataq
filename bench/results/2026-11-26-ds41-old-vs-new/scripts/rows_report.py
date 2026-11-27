"""Prints ds_rows.py's results as Markdown: median tok/s per configuration (single prompts, or all together and each),
guesses kept, rows and experts per step, and - for the run an io_sampler.py log covers - SSD reads per token and per
step.

    python rows_report.py rows/off.json rows/on.json [--ssd rows/on.out rows/on-io.jsonl]
"""
import bisect
import json
import re
import statistics
import sys

med = statistics.median


def main() -> None:
    args = sys.argv[1:]
    ssd_args = args[args.index("--ssd") + 1:] if "--ssd" in args else []
    runs = [json.load(open(p, encoding="utf-8")) for p in args[:len(args) - len(ssd_args) - bool(ssd_args)]]
    ssd = read_ssd(runs[-1], *ssd_args) if ssd_args else {}
    kinds = list(runs[0]["prompts"])
    print("| DSpark | config | " + " | ".join(kinds) + " | together (each) | guesses kept | rows/step | "
          "experts/step | SSD MB/token | SSD MB/step |\n|---|---|" + "---:|" * (len(kinds) + 6))
    for run in runs:
        for name in run["rounds"][0]:
            rs = [r[name] for r in run["rounds"]]
            if "each" in rs[0]:
                cells = [""] * len(kinds)
                each = ", ".join(f"{med(r['each'][i] for r in rs):.2f}" for i in range(len(rs[0]["each"])))
                together = f"{med(r['tok_s'] for r in rs):.2f} ({each})"
            else:
                cells = [f"{med(r[k]['tok_s'] for r in rs):.2f}" for k in kinds]
                together = ""
            ers = [r["err"] for r in rs]
            g, right = sum(e["guessed"] for e in ers), sum(e["right"] for e in ers)
            steps = sum(e["steps"] for e in ers)
            per_step = (f"{sum(e['rows'] for e in ers) / steps:.2f} | {sum(e['experts'] for e in ers) / steps:.0f}"
                        if steps else "- | -")
            s = ssd.get(name) if run is runs[-1] else None
            ssd_cells = f"{s['bytes'] / s['tokens'] / 1e6:.1f} | {s['bytes'] / s['steps'] / 1e6:.1f}" if s else "- | -"
            print(f"| {run['case'].removeprefix('dspark-')} | {name} | " + " | ".join(cells) +
                  f" | {together} | " + (f"{100 * right / g:.0f}%" if g else "-") + f" | {per_step} | {ssd_cells} |")


def read_ssd(run: dict, out_path: str, io_path: str) -> dict:
    """Sums, per configuration, the engine's read bytes over its measured rounds the sampler saw whole, with their
    tokens and steps from `run`. A configuration's span runs from its CHANGE_SETTINGS to the next one; the harness
    flushes its log only when it sends, so the log's size passes those offsets when they are sent."""
    text = open(out_path, "rb").read()
    marks = [(m.start(), int(m.group(1)), int(m.group(2))) for m in
             re.finditer(rb">>> ASK \S+ CHANGE_SETTINGS concurrent=(\d+) rows=(\d+) lookahead=\d+", text)]
    rules = [(m.start(), m.group(1)) for m in re.finditer(rb"CHANGE_SETTINGS draft_rule=(\w+)", text)]
    samples = [json.loads(line) for line in open(io_path)]
    sizes = [s["log_bytes"] for s in samples]
    seen, per = {}, {}
    for i, (at, c, r) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        rule = next((x for pos, x in reversed(rules) if pos < at), b"speculative")
        name = f"{c}x{r}" + (".7t" if rule == b"typical" else "")
        seen[name] = seen.get(name, 0) + 1
        rnd = seen[name] - 2   # (the first is the warm-up's)
        a, b = bisect.bisect_right(sizes, at), bisect.bisect_right(sizes, end)
        if rnd < 0 or a == 0 or b >= len(samples):   # (not seen whole)
            continue
        res = run["rounds"][rnd][name]
        tokens = sum(res["tokens"]) if "each" in res else sum(res[k]["tokens"] for k in run["prompts"])
        acc = per.setdefault(name, {"bytes": 0, "tokens": 0, "steps": 0})
        acc["bytes"] += samples[b]["read_bytes"] - samples[a - 1]["read_bytes"]
        acc["tokens"] += tokens
        acc["steps"] += res["err"]["steps"]
    return per


if __name__ == "__main__":
    main()
