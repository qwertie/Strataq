"""strata_server/runconfig.py - #564: the web page's Settings view of the run config (strata-<model>.json).

A short list of documented keys can be read and changed from Strata's own page (GET / POST /config).  Everything
else in the file - the keys setup writes, the network and security keys (host, api_key, cors_origins,
trusted_origins, allowed_hosts), the MCP servers and before_load (which run programs), sampling keys not listed
here - is kept as it is: a change touches only the keys it names (as setup run again keeps the user's keys, #629).
The earlier file is kept as strata-<model>.json.bak.  The server reads the config when it starts, so a change is
used from the next start on.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

from strata_common import user_args
from strata_base import atomic as _atomic_mod

# (key, kind, help).  kind: "bool", "int>=0", "num>=0", ("enum", values), "names", or ("sampling", check) for a key
# of the "sampling" block, ("arg", flag) for an engine option kept in "args".
EDITABLE = [
    ("sampling.temperature", ("sampling", "num>=0"),
     "Default temperature for requests that send none (0 = greedy; without it the model card's: 1.0 thinking, 0.7 not)"),
    ("sampling.top_p", ("sampling", "0<x<=1"), "Default top_p for requests that send none"),
    ("sampling.top_k", ("sampling", "1..64"), "Default top_k for requests that send none (1-64)"),
    ("sampling.min_p", ("sampling", "0<=x<=1"), "Default min_p for requests that send none"),
    ("reasoning_budget_tokens", "int>=0", "Cap the thinking of every request at this many tokens (0 or empty: no cap)"),
    ("fit_max_tokens", "bool", "Shorten a max_tokens that does not fit the context (on by default); off: answer 400 like llama.cpp"),
    ("anthropic_thinking", ("enum", ["model", "on_request"]),
     "Anthropic requests that do not ask for thinking: think as the model does (model) or not (on_request)"),
    ("effort_position", ("enum", ["start", "end"]),
     "Where a non-default reasoning effort goes: start (the default) or end (keeps the cache when it changes)"),
    ("aliases", "names", "Other model names the server lists and answers to (comma-separated)"),
    ("idle_unload_s", "num>=0", "Unload the model after this many seconds without requests (0 or empty: never)"),
    ("mtp", "bool", "Draft layer (MTP): off starts the engine without it (about 1 GB less graphics memory, slower words per second); "
                    "empty or on: as installed"),
    ("lazy_load", "bool", "Start without loading the model; the first request loads it (text only)"),
    ("engine_silence_s", "num>=0", "End a request when the engine says nothing for this long (default 300 s, 0 = wait)"),
    ("api_monitor", "bool", "Keep the last 100 requests' prompts and answers in memory for /api-monitor"),
    ("open_browser", "bool", "Open the chat page in the browser when the model is ready"),
    ("vram_reserve_mib", ("arg", "--vram-reserve-mib"),
     "VRAM in MiB the engine leaves free for other programs (engine default 700)"),
]
SPEC = {k: kind for k, kind, _ in EDITABLE}

# The settings the Memory panel changes live (POST /v1/memory sends them to the engine as CHANGE_SETTINGS), each with
# the engine flag in "args" that holds its default.  Not in EDITABLE: the memory endpoint is their one writer.  A GPU
# other than the first (memory_mib.gpu<N>) can be changed live, but has no flag to save it to.
LIVE_FLAGS = {"memory_mib.gpu0": "--gpu-mib", "memory_mib.ram": "--ram-mib", "tokens_min": "--tokens-min",
              "context": "--conversation-tokens",
              "vram_conversations": "--vram-conversations",
              "conversation_idle_seconds": "--conversation-idle-seconds",
              "decode_share_percent": "--decode-share-percent", "prompt_turn_ms": "--prompt-turn-ms",
              "tokens_max": "--tokens-max", "ssd_pause_mib": "--ssd-pause-mib", "rows": "--rows",
              "suffix_draft": "--suffix-draft", "lookup_chain": "--lookup-chain",
              "lookahead": "--lookahead", "concurrent": "--concurrent", "drafter": "--drafter"}
# The numeric live settings whose range is not "at least 0": (least, most); the engine checks the rest (concurrent
# against its batch slots, lookahead against rows).
_LIVE_RANGE = {"rows": (1, 8), "concurrent": (1, None), "context": (1, None), "vram_conversations": (1, None)}


def _arg(cfg: dict, flag: str):
    a = cfg.get("args") if isinstance(cfg.get("args"), list) else []
    if flag in a[:-1]:
        v = str(a[a.index(flag) + 1])
        return int(v) if v.isdigit() else v
    return None


def value_of(cfg: dict, key: str):
    kind = SPEC[key]
    if isinstance(kind, tuple) and kind[0] == "sampling":
        s = cfg.get("sampling")
        return s.get(key.split(".", 1)[1]) if isinstance(s, dict) else None
    if isinstance(kind, tuple) and kind[0] == "arg":
        return _arg(cfg, kind[1])
    return cfg.get(key)


def view(cfg: dict, path: str | Path) -> dict:
    """GET /config: the editable keys with their values (None: not set, the default applies)."""
    out = []
    for key, kind, help_ in EDITABLE:
        k = kind[0] if isinstance(kind, tuple) else kind
        out.append({"key": key, "value": value_of(cfg, key), "help": help_,
                    "kind": "number" if k in ("sampling", "arg", "int>=0", "num>=0") else k,
                    **({"choices": kind[1]} if k == "enum" else {})})
    return {"file": Path(path).name, "keys": out,
            "note": "Saved to the run config; used from the next start of the model."}


def check_live(changes, to_default: bool, cards: list[str] | None = None) -> dict:
    """Checks POST /v1/memory's "set" ({setting: value}): gets the values as CHANGE_SETTINGS spells them (whole
    numbers; drafter "on" or "off"), or raises a ValueError naming the first bad setting or value.  `to_default`:
    the values are to be saved to the run config too, so only settings with a flag there are accepted; `cards`: the
    engine's GPU pool settings (get_card_settings), which --gpu-mib holds (one size per card)."""
    if not isinstance(changes, dict) or not changes:
        raise ValueError('send {"set": {"<setting>": <value>, ...}}')
    out = {}
    for key, v in changes.items():
        if key not in LIVE_FLAGS and not _is_card_setting(key):
            raise ValueError(f"{key!r} is not a memory setting (expected memory_mib.gpu<N>, "
                             f"memory_mib.gpu<N>.<card> or one of {', '.join(LIVE_FLAGS)})")
        if to_default and key not in LIVE_FLAGS and key not in (cards or []):
            raise ValueError(f"{key} has no default in the run config; change it for this session only")
        if key == "drafter":
            if not isinstance(v, bool) and v not in ("on", "off"):
                raise ValueError(f'drafter: expected true, false, "on" or "off", not {v!r}')
            out[key] = "on" if v in (True, "on") else "off"
        else:
            out[key] = _number(key, v, True, *_LIVE_RANGE.get(key, (0, None)))
    return out


def read_live_defaults(path: str | Path | None) -> dict:
    """Gets the defaults the run config at `path` holds for LIVE_FLAGS' settings ({setting: value}; None: not set,
    or no readable run config).  For GET /v1/memory."""
    try:
        cfg = load(path) if path else {}
    except (OSError, ValueError):
        cfg = {}
    return {key: _arg(cfg, flag) for key, flag in LIVE_FLAGS.items()}


def apply_live(cfg: dict, values: dict, settings: dict | None = None) -> tuple[dict, list[str]]:
    """Gets the config with check_live's `values` written as their engine flags in "args" (as the person's, as
    apply does), and the settings whose default changed.  `settings`: the engine's settings after the change; a GPU
    pool's size (get_card_settings) is saved as --gpu-mib's list of every card's pool, in the cards' order."""
    new = json.loads(json.dumps(cfg))                  # a deep copy: the caller's dict is left alone
    cards = get_card_settings(settings or {})
    changed = []
    if any(key in cards for key in values):
        sizes = ",".join(str(values.get(k, settings[k])) for k in cards)
        if str(_arg(new, "--gpu-mib")) != sizes:
            _set_arg(new, "--gpu-mib", sizes)
            changed = list(cards)
    for key, v in values.items():
        if key not in cards and _arg(new, LIVE_FLAGS[key]) != v:
            _set_arg(new, LIVE_FLAGS[key], v)
            changed.append(key)
    return new, changed


def get_card_settings(settings: dict) -> list[str]:
    """Gets the engine's GPU pool settings (memory_mib.gpu<N>, memory_mib.gpu<N>.<card>) in its cards' order."""
    return [key for key in settings if _is_card_setting(key)]


def _is_card_setting(key: str) -> bool:
    return re.fullmatch(r"memory_mib\.gpu[0-9]+(\.[0-9]+)?", key) is not None


def _number(key, v, whole=False, lo=0.0, hi=None, lo_open=False):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (whole and v != int(v)) or \
            (v <= lo if lo_open else v < lo) or (hi is not None and v > hi):
        rng = f"{'more than' if lo_open else 'at least'} {lo:g}" + (f" and at most {hi:g}" if hi is not None else "")
        raise ValueError(f"{key}: expected a {'whole ' if whole else ''}number, {rng}, not {v!r}")
    return int(v) if whole else v


def check(key: str, v, cfg: dict):
    """The value to store for `key`, or a ValueError naming what is expected.  None removes the key (its default)."""
    if key not in SPEC:
        raise ValueError(f"{key!r} cannot be changed here (only the keys the Settings view lists)")
    if v is None:
        return None
    kind = SPEC[key]
    k = kind[0] if isinstance(kind, tuple) else kind
    if k == "bool":
        if not isinstance(v, bool):
            raise ValueError(f"{key}: expected true or false, not {v!r}")
        if key == "lazy_load" and v and cfg.get("vision"):
            raise ValueError("lazy_load: lazy loading is text-only, and this model reads images (\"vision\")")
        return v
    if k == "enum":
        if v not in kind[1]:
            raise ValueError(f"{key}: expected one of {', '.join(kind[1])}, not {v!r}")
        return v
    if k == "names":
        if isinstance(v, str):
            v = [x.strip() for x in v.split(",") if x.strip()]
        if not isinstance(v, list) or not all(isinstance(x, str) and x.strip() and len(x) <= 128 for x in v):
            raise ValueError(f"{key}: expected a list of model names, not {v!r}")
        return v or None
    if k == "int>=0":
        return _number(key, v, whole=True)
    if k == "num>=0":
        return _number(key, v)
    if k == "arg":
        return _number(key, v, whole=True)
    rule = kind[1]                                     # a sampling key: the server's own rules
    if rule == "num>=0":
        return float(_number(key, v))
    if rule == "0<x<=1":
        return float(_number(key, v, hi=1, lo_open=True))
    if rule == "0<=x<=1":
        return float(_number(key, v, hi=1))
    return _number(key, v, whole=True, lo=1, hi=64)


def apply(cfg: dict, changes: dict) -> tuple[dict, list[str]]:
    """The config with `changes` ({key: value or None}) made, and the keys that changed.  Every value is checked
    first, so a request with one bad value changes nothing (ValueError)."""
    if not isinstance(changes, dict) or not changes:
        raise ValueError('send {"set": {"<key>": <value or null>, ...}}')
    checked = {k: check(k, v, cfg) for k, v in changes.items()}
    new = json.loads(json.dumps(cfg))                  # a deep copy: the caller's dict is left alone
    changed = []
    for key, v in checked.items():
        if value_of(new, key) == v:
            continue
        kind = SPEC[key]
        if isinstance(kind, tuple) and kind[0] == "sampling":
            s = dict(new.get("sampling") or {}) if isinstance(new.get("sampling"), dict) else {}
            sub = key.split(".", 1)[1]
            if v is None:
                s.pop(sub, None)
            else:
                s[sub] = v
            if s:
                new["sampling"] = s
            else:
                new.pop("sampling", None)
        elif isinstance(kind, tuple) and kind[0] == "arg":
            _set_arg(new, kind[1], v)
        elif v is None:
            new.pop(key, None)
        else:
            new[key] = v
        changed.append(key)
    return new, changed


def _set_arg(cfg: dict, flag: str, v) -> None:
    """Sets engine option `flag` to `v` (None: removes it) in cfg's "args", in place, as the person's own (user_args),
    so that it survives setup and the app writing "args" again."""
    a = list(cfg.get("args") or [])
    if flag in a[:-1]:
        i = a.index(flag)
        if v is None:
            del a[i:i + 2]
        else:
            a[i + 1] = str(v)
    elif v is not None:
        a += [flag, str(v)]
    cfg["args"] = a
    mine = user_args.set_flag(user_args.capture(cfg), flag, v)
    if mine:
        cfg[user_args.USER_KEY] = mine
    else:
        cfg.pop(user_args.USER_KEY, None)
    if user_args.WRITTEN_KEY in cfg:
        cfg[user_args.WRITTEN_KEY] = list(a)


def save(path: str | Path, cfg: dict) -> Path:
    """The config written whole (a temporary file moved over the old one), the earlier one kept as <name>.bak."""
    path = Path(path)
    bak = path.with_name(path.name + ".bak")
    shutil.copyfile(path, bak)
    _atomic_mod.write_text(path, json.dumps(cfg, indent=1))
    return bak


def load(path: str | Path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(cfg, dict):
        raise ValueError(f"{Path(path).name} is not a JSON object")
    return cfg
