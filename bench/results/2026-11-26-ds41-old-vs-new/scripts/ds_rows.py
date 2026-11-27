"""strata-ds41: decode speed by conversations at once (1-4), rows per step (up to 8) and DSpark lookahead, using
tools/measure_rows.py's engine driver, prompts and report with DeepSeek's chat format. One start per DSpark mode;
the configurations change live (CHANGE_SETTINGS). Same pools as ds_h2h.py (50 GiB GPU, 36 GiB RAM).

    python ds_rows.py --mode on --out DIR/on [--rounds 3] [--tokens 256]
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, r"C:\AI-models\Strataq\tools")
sys.path.insert(0, r"C:\AI-models\Strataq")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure_rows as mr  # noqa: E402
import strata_tokenizer  # noqa: E402
from strata_common.expert_fill import wait_for_experts  # noqa: E402
import ds_h2h as h  # noqa: E402

SPRITES = "Propose sprites written in SVG for a Pac-Man clone"
PARSER = ("Write a recursive-descent parser and evaluator for floating-point math with standard unary/binary operators "
          "including + - * / % ^, in a modern C++ console app without 3rd-party libraries")
# Name: (conversations, rows per step, lookahead). A step gives each conversation a row and splits the rest evenly as
# guesses (docs/MEMORY_MANAGER_PLAN.md, "How rows are shared"), so while all of them decode the rows alone set the
# guesses; lookahead 7 (DSpark's most) only caps a conversation left alone, which the measured window never has.
CONFIGS = {
    "off": {"1x1": (1, 1, 0), "2x2": (2, 2, 0), "3x3": (3, 3, 0), "4x4": (4, 4, 0)},
    "on": {"1x2": (1, 2, 7), "1x3": (1, 3, 7), "1x4": (1, 4, 7), "1x6": (1, 6, 7), "1x8.7t": (1, 8, 7),
           "2x4": (2, 4, 7), "2x6": (2, 6, 7), "2x6.7t": (2, 6, 7), "2x8": (2, 8, 7),
           "3x6": (3, 6, 7), "3x6.7t": (3, 6, 7), "3x8": (3, 8, 7)},
}
# A ".7t" configuration samples at temperature 0.7 and keeps guesses by the lossy `typical` rule (run_config).


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["on", "off"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--only", help="configurations to run, comma-separated")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    configs = {k: v for k, v in CONFIGS[a.mode].items() if not a.only or k in a.only.split(",")}
    tok = strata_tokenizer.load_folder(os.path.join(h.PACK, "tokenizer"))
    old = json.load(open(h.OLD_STUDY))["prompts"]
    texts = {"seasons": old[0], "pacman": h.PACMAN, "sprites": SPRITES, "parser": PARSER}
    mr.KINDS = list(texts)   # (measure_rows' run_config reads its KINDS)
    prompts = {k: tok.encode(f"<｜begin▁of▁sentence｜><｜User｜>{text}<｜Assistant｜></think>", parse_special=True)
               for k, text in texts.items()}
    res_prompts = prompts
    pack = out + "-pack"
    shutil.rmtree(pack, ignore_errors=True)
    shutil.copytree(h.PACK, pack, ignore=shutil.ignore_patterns("*.tmp"))
    env = dict(os.environ, **h.ENV)
    env["PATH"] = h.LIB + os.pathsep + env["PATH"]
    args = ["--pack", pack, *h.COMMON, "--dspark", a.mode if a.mode == "on" else "none", "--gpu-mib", "51200",
            "--ram-mib", "36864", "--rows", "8", "--lookahead", "7" if a.mode == "on" else "0",
            "--expert-cache", "auto", "--batch", "4", "--concurrent", "4", "--vram-conversations", "4",
            "--remote-experts", "off", "--save-on-shutdown", "off", "--pause-dir", pack + "-pause"]
    e = mr.Engine([h.exe_of("new"), "--serve", *args], h.ROOT, env, out, [])
    e.until("READY")
    wait_for_experts(e.ask_memory)
    res = {"case": f"dspark-{a.mode}", "args": args, "keys": "", "tokens": a.tokens,
           "prompts": res_prompts, "memory_at_start": e.ask_memory(), "rounds": []}
    print("warm-up", flush=True)
    for name in configs:
        run_config(e, prompts, name, configs[name], 24, 0)
    names = list(configs)
    for rnd in range(a.rounds):
        got = {}
        for name in names if rnd % 2 == 0 else names[::-1]:
            got[name] = run_config(e, prompts, name, configs[name], a.tokens, rnd + 1)
            print(f"{rnd} {name}: {describe(got[name])}", flush=True)
        res["rounds"].append(got)
        json.dump(res, open(out + ".json", "w", encoding="utf-8"), indent=1)
    res["memory_at_end"] = e.ask_memory()
    json.dump(res, open(out + ".json", "w", encoding="utf-8"), indent=1)
    e.quit()
    shutil.rmtree(pack + "-pause", ignore_errors=True)



def run_config(e: mr.Engine, prompts: dict, name: str, config: tuple, n: int, seed: int) -> dict:
    """Runs measure_rows' run_config with `name`'s sampling: temperature 0.7 with `seed` (the round's number, so each
    round samples another answer) and the typical rule for a ".7t" name, greedy and the default (speculative) rule
    otherwise."""
    sampled = name.endswith(".7t")
    e.keys = ["temperature=0.7", f"seed={seed}"] if sampled else []
    rule = "typical" if sampled else "speculative"
    if getattr(e, "rule", "speculative") != rule:   # (only a change: the DSpark-off start never sends one)
        e.ask(f"CHANGE_SETTINGS draft_rule={rule}", "END_MEMORY")
        e.rule = rule
    return mr.run_config(e, prompts, config, n)


def describe(r: dict) -> str:
    """One line of a configuration's results (measure_rows.describe tells solo runs by a "story" key these prompts
    lack)."""
    rate = (f"together {r['tok_s']:.2f} (" + ", ".join(f"{v:.2f}" for v in r["each"]) + ")" if "each" in r else
            " ".join(f"{k} {r[k]['tok_s']:.2f}" for k in mr.KINDS))
    er = r["err"]
    per = f"; rows/step {er['rows'] / er['steps']:.2f}" if er["steps"] else ""
    short = f"; ENDED EARLY {r['short']}" if r["short"] else ""
    return f"{rate} tok/s; guesses kept {er['right']}/{er['guessed']}{per}{short}"


if __name__ == "__main__":
    main()
