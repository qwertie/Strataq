"""Write a run config from model files that are already prepared, when the installer (setup.py) cannot see the graphics card.

When setup stops because its card probe did not answer, and the model's files AND its prepared pack are already on disk (a box that was
prepared before, "use model files I already have"), the run config is simple enough to write here: the same arguments setup writes for the GSQ-RCO models,
with the engine named as `component:strata-engine/strata-q38`. Nothing is downloaded, prepared or compiled;
if a file is missing the person is told which.

Only the GSQ-RCO families (qwen, swift, coder) and Gyro (one GGUF with its n-gram table inside) are adopted this way; images stay off.
"""

from __future__ import annotations

import re
from pathlib import Path

from strata_common import cfg_paths

SHARD = re.compile(r"^(?P<stem>.+)-(?P<i>\d{5})-of-(?P<n>\d{5})\.gguf$")
FAMILIES = ("qwen", "swift", "coder", "gyro", "unsloth")
UNSUPPORTED = ("no nvidia gpu found", "no amd gpu", "nvidia gpu", "not supported", "cannot be used")


def setup_cannot_handle_this_card(message: str) -> bool:
    low = (message or "").lower()
    return "no nvidia gpu found" in low or "no amd gpu strata can use" in low or "nvidia-smi did not answer" in low


def find_shards(folder: Path) -> list[Path]:
    """The shards of one GGUF set in `folder` (-00001-of-0000N ... -0000N-of-0000N), sorted; [] if the set is incomplete."""
    found: dict[tuple[str, int], dict[int, Path]] = {}
    for p in sorted(Path(folder).glob("*.gguf")):
        m = SHARD.match(p.name)
        if m:
            found.setdefault((m.group("stem"), int(m.group("n"))), {})[int(m.group("i"))] = p
    for (_stem, n), items in found.items():
        if set(items) == set(range(1, n + 1)):
            return [items[i] for i in range(1, n + 1)]
    return []


def find_single(folder: Path) -> list[Path]:
    """The one model file of a Gyro folder (a single GGUF, not a picture reader); [] when there is none or several."""
    files = [p for p in sorted(Path(folder).glob("*.gguf")) if not p.name.lower().startswith("mmproj")]
    return files if len(files) == 1 else []


def plan(*, family: str, model: str, model_dir: Path, data_dir: Path, src: Path, context: int, kv: str | None, model_name: str, family_def: dict,
         log: Path, backend: str | None, pack_name: str | None = None) -> tuple[dict | None, list[str]]:
    """(config, problems): the config to write, or None with plain-words problems (what is missing). `pack_name`: the pack folder under
    packs/ (default the model's name in lower case; Gyro's is its tag, gyro-gyro-s)."""
    problems: list[str] = []
    if family not in FAMILIES:
        return None, [f"Strata can write the settings itself only for the Qwen3.8-Flash-Next, Swift, Coder and Gyro files, not for {family}"]
    shards = find_single(model_dir) if family == "gyro" else find_shards(model_dir)
    if not shards:
        problems.append(f"the model file is not in {model_dir}" if family == "gyro" else
                        f"the model files are not all in {model_dir} (every -0000N-of-0000N.gguf shard is needed)")
    pack = Path(data_dir) / "packs" / (pack_name or model.lower())
    if not (pack / "experts.bin").exists() and not (pack / "index.txt").exists():
        problems.append(f"the prepared pack {pack} is missing (this installer cannot prepare it for this graphics card)")
    tokenizer = pack / "tokenizer"
    if not tokenizer.exists():
        problems.append(f"the tokenizer {tokenizer} is missing")
    profile = src / "data" / (family_def.get("profile") or "expert-profile.bin")
    if not profile.exists():
        problems.append(f"the expert profile {profile} is missing")
    if problems:
        return None, problems
    mtp = Path(data_dir) / "mtp" / "rt"
    args = ["--pack", str(pack), "--native", str(shards[0])]
    if family != "unsloth" and (len(shards) > 1 or family == "gyro"):            # Gyro's one file holds the n-gram table too
        args += ["--ple-gguf", str(shards[-1])]      # (Unsloth's 3-6 shards: the engine finds the n-gram table by its tensor name, as setup.py writes the config; it is shard 2, 3 or the last one by set)
    args += ["--expert-profile", str(profile), "--expert-cache", "auto", "--prefill", "auto",
             "--lookahead", "4", "--spec-min-p", "0.5"]
    if mtp.exists():
        args += ["--mtp", str(mtp)]
    args += ["--max-context", str(context), "--kv", kv or ("int8" if context > 8192 else "fp16")]
    cfg = {"exe": "component:strata-engine/strata-q38", "args": args, "cwd": str(src),
           "tokenizer": str(tokenizer), "model_name": model_name,
           "log": str(log), "lib_dirs": [], "port": 8080}
    if backend:
        cfg["backend"] = backend
    return cfg, []


def write(path: Path, cfg: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(cfg_paths.dumps(cfg, path, indent=1) + "\n", encoding="utf-8")
