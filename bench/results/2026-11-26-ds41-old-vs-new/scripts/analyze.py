"""Summarizes ds_h2h.py's measurements.json: pooled decode tok/s per engine, DSpark mode and prompt (with the range of
single answers), new versus old in percent, drafts kept/offered, GPU experts, and greedy old/new identity."""
import json
import sys

NAMES = ["seasons", "pacman", "story"]
res = json.load(open(sys.argv[1]))
runs = [r for r in res["runs"] if r["measured"]]
for r in res["runs"]:
    if "error" in r:
        print("ERROR in", r["name"], r["error"])


def cell(eng, mode, i=None):
    """Gets the pooled rate (produced / decode time), the single answers' rates, and drafts for one cell."""
    xs = [cyc[k] for r in runs if r["engine"] == eng and r["dspark"] == mode for cyc in r["measured"]
          for k in (range(3) if i is None else [i])]
    rate = sum(x["produced"] for x in xs) * 1000 / sum(x["decode_ms"] for x in xs) if xs else float("nan")
    return rate, [x["tok_s"] for x in xs], sum(x["right"] for x in xs), sum(x["guessed"] for x in xs)


print("| DSpark | prompt | old tok/s (range) | new tok/s (range) | new vs old | old drafts kept/offered | "
      "new drafts kept/offered |\n|---|---|---|---|---:|---|---|")
for mode in ("off", "on"):
    for i, name in [(0, "seasons"), (1, "pacman"), (2, "story"), (None, "all three")]:
        o, n = cell("old", mode, i), cell("new", mode, i)
        rng = lambda c: f"{c[0]:.2f} ({min(c[1]):.2f}-{max(c[1]):.2f})" if c[1] else "-"
        print(f"| {mode} | {name} | {rng(o)} | {rng(n)} | {100 * (n[0] / o[0] - 1):+.1f}% | {o[2]}/{o[3]} | "
              f"{n[2]}/{n[3]} |")

print("\nper start (all-three pooled tok/s per measured cycle, gpu experts, warm cycles, ready s):")
for r in sorted(runs, key=lambda r: r["started"]):
    per = [round(sum(x["produced"] for x in c) * 1000 / sum(x["decode_ms"] for x in c), 2) for c in r["measured"]]
    print(f"  {r['name']:14} {per} gpu={r.get('gpu_experts')} warm={len(r['warm'])} ready={r.get('ready_s', 0):.0f}")

print("\ngreedy identity, first differing token (1-based) of each measured answer vs the first old answer:")
for mode in ("off", "on"):
    ref = next(r for r in runs if r["engine"] == "old" and r["dspark"] == mode)["measured"][0]
    for eng in ("old", "new"):
        diffs = []
        for r in runs:
            if r["engine"] != eng or r["dspark"] != mode:
                continue
            for cyc in r["measured"]:
                d = []
                for k in range(3):
                    a, b = ref[k]["tokens"], cyc[k]["tokens"]
                    j = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), None)
                    d.append("match" if j is None and len(a) == len(b) else (j + 1 if j is not None else "len"))
                diffs.append(d)
        print(f"  {mode} {eng}: {diffs}")
