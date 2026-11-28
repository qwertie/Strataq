"""16 tokens of a fixed prompt through strata-ds41 and a check of the TEXT (setup runs this once after the pack is built).

    python tools/dsv41/smoke.py --exe strata-ds41 --pack <pack> [--gpus 0] [--tokens 16]

A pack that loads can still give garbage (a wrong pack recipe does exactly that), so the answer is read, not just the exit code:
the model is asked for the capital of France and the reply must be words, with "Paris" in it.  Exit 0 and the text on success; exit 1
and one plain line (and the engine's last lines) when the reply is not sensible; exit 2 when the engine did not run at all.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

PROMPT = "<｜begin▁of▁sentence｜><｜User｜>What is the capital of France? Answer in one short sentence.<｜Assistant｜></think>"


def tokenizer(pack: Path):
    import strata_tokenizer as ST
    tpath = pack / "tokenizer"
    vocab = json.loads((tpath / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (tpath / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((tpath / "token_type.json").read_text())
    tcfg = {}
    try:
        tcfg = json.loads((tpath / "tokenizer.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    return ST.Tokenizer(tokens, merges, types, tcfg.get("pre") or "qwen35", tcfg.get("special_ids") or None)


def run_engine(exe: str, args: list[str], ids: list[int], n: int, timeout: float, env: dict | None = None) -> tuple[list[int], dict]:
    """Starts the serving engine `exe` with `args`, generates up to `n` tokens greedily after prompt `ids` and stops it.
    Returns the tokens and the figures of the engine's DONE line (StrataEngine.last).  Raises when the engine does not
    start or dies; a run longer than `timeout` seconds is cut short.  tune.py times its runs with this too."""
    from strata_common.engine_client import StrataEngine
    engine = StrataEngine(exe, [a for a in args if a != "--serve"], env=env)
    cancel = threading.Event()
    timer = threading.Timer(timeout, cancel.set)
    timer.start()
    try:
        out = [t for t in engine.generate(ids, n, {"temperature": 0}, cancel) if t is not None]
        return out, dict(engine.last)
    finally:
        timer.cancel()
        engine.close()


def sensible(text: str) -> tuple[bool, str]:
    """Words, not noise: letters in a normal share, more than two different words, no replacement characters, and Paris in it."""
    t = text.strip()
    words = re.findall(r"[A-Za-z]{2,}", t)
    if not t:
        return False, "the answer is empty"
    if "�" in t:
        return False, "the answer has broken characters"
    if len(set(w.lower() for w in words)) < 3:
        return False, "the answer is not made of words"
    letters = sum(c.isalpha() or c.isspace() or c in ".,'-" for c in t) / len(t)
    if letters < 0.8:
        return False, "the answer is mostly symbols"
    if "paris" not in t.lower():
        return False, "the answer to a question with a known answer (the capital of France) does not contain it"
    return True, "ok"


def main() -> int:
    for st in (sys.stdout, sys.stderr):       # the reply has DeepSeek's special tokens (fullwidth bars): a Windows console code page cannot print them
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--gpus")
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=5400)
    a = ap.parse_args()
    pack = Path(a.pack)
    tok = tokenizer(pack)
    ids = tok.encode(PROMPT, parse_special=True)
    args = ["--pack", str(pack), "--max-context", "1024", "--dspark", "auto"]
    if a.gpus:
        args += ["--gpus", a.gpus]
    try:
        out, _ = run_engine(a.exe, args, ids, a.tokens, a.timeout, dict(os.environ))
    except Exception as e:                                # noqa: BLE001 - any failure to start or run is the engine's
        print(f"the engine did not finish the test run: {e}")
        return 2
    text = tok.decode(out)
    good, why = sensible(text)
    print(text.strip())
    if not good:
        print(f"test question failed: {why}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
