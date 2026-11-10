"""The official DeepSeek-V4.1-Flash GGUF sets (ISTA-DAS 1.75 / 2.00 / 2.25 bpw), recognised by their pinned file names and sizes.

The Terminal's setup downloads them (catalog.MULTI_FILE_FAMILIES, DSV41_ISTA_FILES) and accepts the files of their earlier
layouts (DSV41_OLDER_FILES: ...-0000N-of-00004.gguf, model-2.00bpw-0000N-of-00004.gguf, engram-IQ4_NL.gguf). The app's
"I already have the model" preview used to know only the Qwen fingerprints, so a folder with such a set ended in "the folder holds more than one model" (the Engram is a second file) or, for one shard,
"this model type is not supported by the Strata engine". This module is how Control says what a folder of these files is, from the names and
sizes alone: no tensor is read and no 40 GB file is hashed in a preview (setup checks the SHA-256 once, when the set is adopted).

The tables are the catalog's (strata_common/catalog.py, never copied), so a new size added there is recognised here.  Shard 1 of
the 4-shard layout is also recognised at the size of ISTA-DASLab's 2026-10-26 re-publication (a larger chat template:
DSV41_NEW_SHARD1), and Unsloth's own low-bit sets (UD-IQ1_S, UD-IQ1_M) with the Engram as their shard 5.

The ISTA-DAS Coder (catalog.DSV41_CODER, data/dsv41/coder-set.json) is one more set of the same kind: its metadata shard and model
shards under the repository's names, matched by name AND size ("generic"), and ISTA-DAS's Engram file.
"""
from __future__ import annotations

import functools

from strata_common import catalog

FAMILY = "deepseek41"


@functools.cache
def tables() -> dict:
    """{"sets": {model: {"label", "generic", "layouts": [{name: bytes}, ...], "engram_alt": {name: bytes}}}, "engram": {name: bytes},
    "new1": {name: bytes}} for the sets setup packs (the official ones and Unsloth's low-bit ones); their layouts newest first
    (catalog.make_ista_set); "engram_alt": a set's own shard that is the Engram; "new1": the other size a shard 1 may have."""
    family = catalog.MULTI_FILE_FAMILIES[FAMILY]
    sizes = {name: pin[0] for name, pin in catalog.DSV41_ALL_FILES.items()}
    result: dict = {"sets": {}, "engram": {}, "new1": {name: pin[0] for name, pin in catalog.DSV41_NEW_SHARD1.items()}}
    for model, entry in family["models"].items():
        if entry.get("recipe") not in ("ista", "udlow", "coder"):
            continue                                # Unsloth's 508 GB set is not an "already have it" case of the app
        alt = entry.get("engram_shard")
        result["sets"][model] = {"label": entry.get("label", model), "generic": bool(entry.get("generic_names")),
                                 "layouts": [{name: sizes[name] for name in layout} for layout in entry["layouts"]],
                                 "engram_alt": {alt: sizes[alt]} if alt else {}}
    for name in (catalog.DSV41_ISTA_ENGRAM, catalog.DSV41_FIRST_NAMES[catalog.DSV41_ISTA_ENGRAM]):
        result["engram"][name] = sizes[name]
    return result


def match(files: dict[str, int]) -> list[dict]:
    """The official sets the files {name: bytes} of one folder belong to, best first: complete sets before partial ones.
    Each: {"model", "label", "complete", "present": [names], "missing": [names], "wrong_size": [names], "legacy": bool (an earlier
    layout), "engram": name|None, "newer_first": [the shards 1 at their re-published size]}, one for each layout of a set at least one
    of whose shards is there; names and sizes are the pinned ones.  A folder that mixes layouts is no complete set; one that holds two
    whole layouts of a set is that set once (the newest)."""
    known = tables()
    shared_engram = next((name for name in known["engram"] if files.get(name) == known["engram"][name]), None)
    found, whole = [], set()
    for model, entry in known["sets"].items():
        engram = shared_engram or next((n for n, size in entry["engram_alt"].items() if files.get(n) == size), None)
        for layout, pinned in enumerate(entry["layouts"]):
            present = [n for n in pinned if n in files]
            if not present:
                continue
            ok_sizes = {n: {pinned[n], known["new1"].get(n, pinned[n])} for n in pinned}
            if entry["generic"] and not any(files[n] in ok_sizes[n] for n in present):
                continue                                # a file of another size under these names is another model's, not a cut copy of this one
            wrong = [n for n in present if files[n] not in ok_sizes[n]]
            missing = [n for n in pinned if n not in files]
            if not missing and not wrong and model in whole:
                continue
            found.append({"model": model, "label": entry["label"], "legacy": layout > 0, "present": sorted(present), "missing": missing,
                          "wrong_size": wrong, "complete": not missing and not wrong, "engram": engram,
                          "newer_first": [n for n in present if n not in wrong and files[n] != pinned[n]],
                          "bytes": sum(files[n] for n in present if n not in wrong)
                                   + ((known["engram"].get(engram) or entry["engram_alt"].get(engram) or 0) if engram else 0)})
            if not missing and not wrong:
                whole.add(model)
    found.sort(key=lambda row: (not row["complete"], len(row["missing"]) + len(row["wrong_size"]), row["model"]))
    return found
