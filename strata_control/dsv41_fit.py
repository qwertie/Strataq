"""Where DeepSeek-V4.1-Flash goes on a computer: the real numbers of the model and its engine, in one place.

The model is 40 layers of 384 routed experts (top 6) beside dense attention weights.  Unsloth's UD-Q8_K_XL (the
file Strata runs, pinned in strata_common/catalog.py) is 508.4 GB in nine shards:

  routed experts, MXFP4      288.8 GB   15,360 experts of 18.8 MB (40 layers x 384)
  dense weights, BF16 / F32   16.8 GB   attention, router, shared experts, the head (the embedding table stays on the CPU)
  the n-gram table (Engram)  202.8 GB   MXFP8, shard 8: read from the SSD, a few rows a token, never held in memory

so the memory the model needs is the experts and the dense weights, 305.6 GB, with the n-gram table left out.  strata-dsv41
(src/models/deepseekv41) puts the dense weights and as many experts as the cards hold in graphics memory, one block of layers on each
card, and computes the rest of the experts on the CPU, reading them in place from the model's files (the page cache).

Measured (r9700, 4 x Radeon AI PRO R9700 32 GB, 251 GB RAM, 2026-10-20, f-dsv41-port 01389205e): per card 3.4-4.7 GB of dense
weights, 28.5-29.8 GB of experts (1,505-1,575 of its 3,840), ~0.6 GB free; 6,197 experts (116.5 GB) in graphics memory and
9,163 (172.3 GB) on the CPU lane; --tp 2: prompts 143-181 tokens a second (3K prompts), answers 25-31 tokens a second.  Pure functions, no I/O:
setup.py, tools/strata_mcp.py and strata_cli/launch import it.  Recommend, never force: nothing here refuses a computer.
"""

from __future__ import annotations

import os
from typing import Any

from strata_common import catalog, gpu_fit

GB = 1e9
GIB = 2 ** 30
_UD_SHARDS = [catalog.DSV41_SHARD.format(i=i) for i in range(1, 10)]   # UD-Q8_K_XL's shard names, 1 to 9
MAIN_BYTES = sum(catalog.DSV41_FILES[n][0] for n in _UD_SHARDS)   # the nine shards of UD-Q8_K_XL
ENGRAM_BYTES = catalog.DSV41_FILES[_UD_SHARDS[7]][0]               # shard 8: the n-gram table
EXPERT_BYTES = 288_777_830_400        # MXFP4: 543,581,798,400 values x 17/32 bytes
LAYERS = 40
EXPERTS_PER_LAYER = 384
EXPERTS = LAYERS * EXPERTS_PER_LAYER  # 15,360
EXPERT_GB = EXPERT_BYTES / GB / EXPERTS     # 18.8 MB each
WEIGHTS_GB = (MAIN_BYTES - ENGRAM_BYTES) / GB            # 305.6
EXPERTS_GB = EXPERT_BYTES / GB                           # 288.8
DENSE_GB = WEIGHTS_GB - EXPERTS_GB                       # 16.8
DENSE_ON_GPU_GB = 15.5                # measured: 3.72 + 3.69 + 3.40 + 4.70 (the embedding table, ~1.3 GB, stays on the CPU)
ENGRAM_GB = ENGRAM_BYTES / GB                            # 202.8
DOWNLOAD_GB = MAIN_BYTES / GB                            # 508.4
RESERVE_GB = 1.15                     # per card: the 600 MiB the engine leaves free (--reserve-mib) and ~0.5 GB of its buffers
                                      # (measured: dense weights + experts + free = 33.8 of an R9700's 34.4 GB)
REST_RAM_GB = 9.0                     # RAM the engine and the system keep besides the experts ("beside 9 GB for the rest")
SPARE_FITS_GB = 4.0                   # RAM left beyond the CPU lane's experts below which "fits" becomes "tight"
DISK_HEADROOM_GB = 6.0                # the pack folder (hash tables, tokenizer) and room to work
SMALLEST_CARD_GB = 8.0                # a card under this adds little: the dense weights alone are ~4 GB of it per card of four
MEASURED = ("4 x Radeon AI PRO R9700 (32 GB) with 251 GB of RAM: prompts of 3,000 tokens read at 143-181 tokens a second, answers "
            "at 25-31 tokens a second (--tp 2); one answer at a time")


# The two official ISTA-DAS sets (ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF, catalog.DSV41_ISTA_FILES): 5 shards each (the experts as IQ1_M / IQ2 / IQ3 types, the
# dense weights Q8_0 / Q5_K), plus ONE Engram (their shard 6: 110.6 GB, IQ4_NL, shared by the sets, stays on the SSD) and the DSpark drafter (8.0 GB).
# Measured 2026-10-23 (f-dsv41-rel): quality against the original model (final files, 4,995 positions: ppl +15.5% for 2.00 [code +8.6%, agent +5.3%], +7.3% for 2.25 [code +3.2%, agent +2.2%]
# on 2.25); speed on the owner PC (RTX 5070 12 GB, 64 GB RAM, 3.6 GB/s NVMe): answers ~4.7 tokens a second, prompts 121 tokens a second at 4K and
# 231 tokens a second at 8K; RTX 3060 12 GB with 46 GB of RAM (the engine held to 36 GB): answers ~7.2 tokens a second on 2.00; 4 x Radeon AI PRO
# R9700: answers ~31 tokens a second, prompts 330-373 tokens a second at 8K.
ISTA_ENGRAM_BYTES = catalog.DSV41_ISTA_FILES[catalog.DSV41_ISTA_ENGRAM][0]
ISTA_DSPARK_BYTES = catalog.DSV41_FILES[catalog.DSV41_DSPARK][0]


def _ista_set_bytes(model: str) -> int:
    """Adds up the pinned sizes of the shards setup downloads for a set it packs (an official one or Unsloth's low-bit ones).
    Defined above its callers: ISTA_SETS calls it."""
    shards = catalog.MULTI_FILE_FAMILIES["deepseek41"]["models"][model]["shards"]
    return sum(catalog.DSV41_ALL_FILES[n][0] for n in shards)


ISTA_SETS = {
    "IST-DAS-2.00bpw": {"title": "ISTA-DAS 2.00 bpw", "set_bytes": _ista_set_bytes("IST-DAS-2.00bpw"), "experts_gb": 132.07, "profile": "profile-2.00.bin",
                        "ppl": "15.5%", "ram_ok_gb": 40.0, "ram_great_gb": 160.0, "disk_total": 257.0,
                        "quality": "about 15.5% above the original model's perplexity (code and agent work about 5-9% above)"},
    # IQ1_M (1.75 bpw): 120,721,120,256 bytes of shards; experts = the shards minus the 6.16 GB of dense weights.  RAM class from the experts' size against
    # the 2.00 set's measured 40 GB (not measured on a computer yet); quality: the H100's 4,995-position score (KL 0.217, ppl 3.80, +21%; code +8.9%, agent +10.2%, English prose +53%).
    "IST-DAS-1.75bpw": {"title": "ISTA-DAS 1.75 bpw", "set_bytes": _ista_set_bytes("IST-DAS-1.75bpw"), "experts_gb": 114.56, "profile": "profile-2.00.bin",
                        "ppl": "21%", "ram_ok_gb": 34.0, "disk_total": 239.0,
                        "quality": "about 21% above the original model's perplexity: the smallest, fine for code and agent work, noticeably weaker "
                                   "on plain prose; pick 2.00 if it fits"},
    "IST-DAS-2.25bpw": {"title": "ISTA-DAS 2.25 bpw", "set_bytes": _ista_set_bytes("IST-DAS-2.25bpw"), "experts_gb": 149.06, "profile": "profile-2.25.bin",
                        "ppl": "7.3%", "ram_ok_gb": 44.0, "ram_great_gb": 176.0, "disk_total": 274.0,
                        "quality": "about 7.3% above the original model's perplexity (code and agent work about 2-3% above), "
                                   "the better of the two, 17 GB more disk"},
}
# Unsloth's UD-IQ1_S / UD-IQ1_M (2026-10-30, unsloth/DeepSeek-V4.1-Flash-1017-GGUF; catalog.DSV41_UD_FILES): shard 1 = metadata, shards 2-4 = the model, shard 5 = the
# Engram (the same IQ4_NL data as ISTA-DAS's file, so it is shared with them).  set_bytes = shards 1-4.  The dense weights are a little bigger than ISTA-DAS's
# (attention in Q6_K): 7.2 and 7.6 GB.  Quality is not measured on our side yet.
ISTA_SETS["UD-IQ1_S"] = {"title": "Unsloth UD-IQ1_S", "set_bytes": _ista_set_bytes("UD-IQ1_S"), "experts_gb": 115.23, "profile": "profile-2.00.bin",
                         "ppl": "?", "ram_ok_gb": 36.0, "disk_total": 241.0,
                         "quality": "Unsloth's smallest dynamic version, the size of ISTA-DAS 1.75 bpw; not measured here yet"}
ISTA_SETS["UD-IQ1_M"] = {"title": "Unsloth UD-IQ1_M", "set_bytes": _ista_set_bytes("UD-IQ1_M"), "experts_gb": 132.21, "profile": "profile-2.00.bin",
                         "ppl": "?", "ram_ok_gb": 40.0, "disk_total": 258.0,
                         "quality": "Unsloth's next dynamic version, the size of ISTA-DAS 2.00 bpw; not measured here yet"}
# The ISTA-DAS Coder (2026-11-01): ISTA-DAS's coding variant, "the layout of the IQ2_XXS set with experts removed": 192 routed experts a layer instead of 384 (read from the header of its
# shards: the experts, the router and the routing biases are half as wide; every other tensor is the 2.00 set's).  Where the repository, the commit and the files are is ONE
# file, data/dsv41/coder-set.json (tools/dsv41/pin_coder.py writes it; catalog.DSV41_CODER reads it); the sizes here come from it.  Shard 1 = the metadata, shards 2-5 = the model;
# the Engram is ISTA-DAS's own file, shared with them.  The tensors' bytes (from their offsets): experts 66.04 GB (7,680 of 8.60 MB, exactly half
# of 2.00's 132.07 GB), the other weights 6.08 GB.
CODER = catalog.DSV41_CODER or None
CODER_MODEL = catalog.DSV41_CODER_MODEL
if CODER:
    _coder_bytes = _ista_set_bytes(CODER_MODEL)
    ISTA_SETS[CODER_MODEL] = {
        "title": CODER.get("label", "ISTA-DAS Coder"), "set_bytes": _coder_bytes, "experts_gb": 66.04, "dense_gb": 6.08, "profile": CODER.get("profile", "profile-coder.bin"),
        "experts_per_layer": int(CODER["experts_per_layer"]), "ppl": "48%", "ram_ok_gb": 24.0,
        # measured 11-01 on r9700 (one Radeon AI PRO R9700, 251 GB RAM, this branch's engine, packaged Terminal, 128K context, 5 experts a word): the whole run in a user scope with MemoryMax
        # and no swap; decode median of 3 prompts x 200 tokens, one run each: no cap 25.6 tok/s, 40 GB 20.9, 32 GB 18.5, 24 GB 15.0, 16 GB 8.0; four cards, no cap: 59.0 (all experts in graphics memory)
        "measured": "one Radeon AI PRO R9700 (32 GB) with 251 GB of RAM: answers about 26 tokens a second, a 93,000-token prompt read in 78 s; four of them: about 59 tokens a second",
        "tight_words": "reads the rest from the SSD. Measured on one R9700 with the RAM capped for the whole run: about 21 tokens a second with 40 GB, 18 with 32 GB, 15 with 24 GB (26 with all the RAM it wanted).",
        "spill_words": "This computer has {ram:.0f} GB of RAM. This set was measured down to 16 GB (about 8 tokens a second on one R9700); with less, most experts come from the SSD while it answers: it works but is slower, and a word can take seconds. More RAM fixes it. It is still allowed.",
        "tight_short": "tight: runs, with the experts RAM cannot hold read from the SSD (about 21 to 15 tokens a second measured with 40 to 24 GB of RAM on one R9700)",
        "spill_short": "slow here: this computer has {ram:.0f} GB of RAM and the set was measured down to 16 GB (about 8 tokens a second on one R9700); it can still be installed",
        "unproven_uma": "Not run on a Ryzen AI Max (Strix Halo) yet: the sizes above are exact, the speed is not measured on a shared-memory computer.",
        "disk_total": round((_coder_bytes + ISTA_ENGRAM_BYTES + ISTA_DSPARK_BYTES) / GB),
        "quality": "ISTA-DAS's coding version, half of the model's experts removed (192 of 384 in every layer); against the original model perplexity +48% and KL 0.40 (code and agent work: KL 0.23 and 0.20; 2.00 bpw 0.10 and 0.11)"}
ISTA_DENSE_GB = 6.16                  # the dense weights in the shards (attention, router, shared experts, the head; the embedding stays on the CPU)
ISTA_DENSE_ON_GPU_GB = 4.5            # (the old plan's figure, kept for the Unsloth Q8 plan above; the official sets use the measured ones below)
# What the engine really puts on a card, from its own startup line ("CUDA0 layers [0, 39] + head: weights 5.86 GB, experts 25.29 GB (2566 of 15360 in
# VRAM), caches 0.02 GB, 576 MiB free") on f-042-maya-rc d93547aed, 11-01: the plan said 2,947 experts for one R9700 and the engine held 2,366-2,566;
# 13,180 for four and it held 12,147-12,192.  The old plan took 4.5 GB of dense weights and 1.15 GB of reserve and forgot the rest of a card's
# memory: the driver's context, the prompt path's buffers (they grow with the prompt chunk) and the dense weights each card repeats.
#   dense weights on the cards = base + CARD_DENSE_REPEAT_GB for each card (2.25 bpw: 5.86 GB on one card, 6.57 on four -> 5.62 + 0.24 per card;
#   boston's UD-IQ1_M: 7.61 on one and on two cards; p620's UD-IQ1_S: 6.58)
#   the engine keeps ENGINE_FREE_GB free on each card (--reserve-mib 600 and the small caches)
#   what is left over per card (the "other" of the accounting, measured against the card's nominal memory): AMD (HIP) 1.60 GB when the cards hold
#   half of the experts or more (the prompt path works in chunks of 1024: 4 x R9700), 2.58 GB when they hold less (one R9700: the experts the
#   card cannot hold are read in place and the prompt chunk is 32768 with the borrowed slots; the 8192 chunk of the older engine took 4.5 GB and
#   the card held 2,366); NVIDIA and others 3.6 GB (2 x A6000 3.7, an A4000 3.5; not tied to the chunk)
#   and the experts that stay on a card are a little bigger than the set's average (the hot ones: 9.86-10.0 MB against 9.70 for 2.25 bpw): x1.025
# ISTA-DAS Coder: the engine's startup lines on r9700 with this branch at 128K context, 11-01: one R9700 weights 5.70 GB, experts 24.91 GB (2,859 of 7,680); four R9700 weights
# 0.74 + 0.87 + 2.25 + 2.56 = 6.42 GB and all 7,680 experts in graphics memory -> base 5.46 + 0.24 for each card (5.70 on one, 6.42 on four).
ISTA_DENSE_GPU_BASE_GB = {CODER_MODEL: 5.46, "IST-DAS-2.00bpw": 5.62, "IST-DAS-2.25bpw": 5.62, "IST-DAS-1.75bpw": 5.62, "UD-IQ1_S": 6.34, "UD-IQ1_M": 7.37}
CARD_DENSE_REPEAT_GB = 0.24
ENGINE_FREE_GB = 0.62
CARD_OTHER_GB = {"amd": 1.60, "amd_low_share": 2.58, "other": 3.6}
RESIDENT_EXPERT_FACTOR = 1.025
ISTA_MEASURED = ("an RTX 5070 (12 GB) with 64 GB of RAM and a 3.6 GB/s NVMe drive: answers about 4.7 tokens a second, a 4,000-token prompt "
                 "read at 121 tokens a second; an RTX 3060 (12 GB) with 46 GB of RAM: about 7.2 tokens a second; four Radeon AI PRO R9700 cards: "
                 "about 31 tokens a second")
# A unified-memory APU (Ryzen AI Max "Strix Halo", gfx1151, Windows): the graphics chip and the CPU share one pool of RAM, so the
# engine's expert tiers are one pool too.  The measured 64 GB box: 2.00 bpw, Faster (5 experts), DSpark auto, Adrenalin 26.10.1.
UMA_RESERVE_GB = 12.0                 # kept for Windows, the desktop and the engine's buffers (the engine's own RAM store margin)
UMA_MEASURED = ("a Ryzen AI Max+ 395 (Radeon 8060S, 128 GB unified: a 64 GB BIOS carve-out plus 64 GB seen by Windows) on Windows: answers "
                "14-16 tokens a second on code in the default Faster mode (5 experts) and 10-11 on chat; a 4,000-token prompt 77-81 tokens a "
                "second, 8,000 tokens 108-111")


def variant(model: str | None) -> dict | None:
    """The official-set entry for a model name (case-insensitive), None for UD-Q8_K_XL and anything else."""
    if not model:
        return None
    return next((v for k, v in ISTA_SETS.items() if k.lower() == str(model).lower()), None)


#: The AMD cards the DeepSeek engine package holds code for (packaging/build_hip_linux.sh: strata-ds41 ships in the rx9070 build, gfx1200 and
#: gfx1201).  setup passes the installed package's own list when it has one.
DSV41_HIP_ARCHS = ("gfx1200", "gfx1201")


def _on_windows() -> bool:
    return os.name == "nt"


def run_cards(cards: list[dict] | None, vendor: str | None = None, *, windows: bool | None = None,
              covered=None, named: bool = False) -> tuple[list[dict], list[dict]]:
    """The cards one DeepSeek run uses when the person named none: (cards, left out with 'why_ignored'), in card order.  The ONE rule that
    `strata launch --plan`, setup and the app share, so the plan says what the run does.
    Every card that counts (gpu_fit.usable_cards: no integrated chip beside a card with memory of its own, no card under 8 GB).  AMD:
    on Windows one card only (several AMD cards sharing a model is Linux-only, setup.py), and only cards the engine package holds code for
    (`covered`, default DSV41_HIP_ARCHS): a PC with an RX 9070 and an RX 7900 uses the RX 9070 and says why the other is left out.
    `named`: the person named these cards: they are used as named (setup warns about one the package has no code for)."""
    mine = gpu_fit.for_vendor(list(cards or []), vendor)[0] if vendor else list(cards or [])
    keep, ignored = gpu_fit.usable_cards(mine)
    small = [c for c in keep if float(c.get("vram_gb") or 0) < SMALLEST_CARD_GB]
    keep = [c for c in keep if float(c.get("vram_gb") or 0) >= SMALLEST_CARD_GB]
    ignored = list(ignored) + [{**c, "why_ignored": f"only {float(c.get('vram_gb') or 0):.0f} GB of graphics memory"} for c in small]
    if not named and len(keep) > 1 and all(str(c.get("vendor") or "").lower() == "nvidia" for c in keep):
        # NVIDIA: setup's role rule (gpu_roles.recommend): a card of 8 GB or less starts Off beside a bigger one, so the plan counts what the run uses
        # (an AMD card of 8 GB counts in both the plan and the run)
        from strata_common import gpu_roles
        if any(round(float(c.get("vram_gb") or 0)) > gpu_roles.OFF_AT_OR_BELOW_GB for c in keep):
            off = [c for c in keep if round(float(c.get("vram_gb") or 0)) <= gpu_roles.OFF_AT_OR_BELOW_GB]
            ignored += [{**c, "why_ignored": f"a card with {gpu_roles.OFF_AT_OR_BELOW_GB:.0f} GB or less starts Off beside a bigger card (setup does the same)"}
                        for c in off]
            keep = [c for c in keep if c not in off]
    amd = not named and bool(keep) and all(str(c.get("vendor") or vendor or "").lower() == "amd" for c in keep)
    if amd:
        win = _on_windows() if windows is None else windows
        # the list of architectures: the package's own (`covered`), else the Linux package's (gfx1200, gfx1201).  Windows ships a different
        # DeepSeek package (hip-rdna3), so without a list of its own nothing is filtered there.
        archs = tuple(covered) if covered else (() if win else DSV41_HIP_ARCHS)
        if archs:
            known = [c for c in keep if c.get("arch")]
            ok = [c for c in keep if not c.get("arch") or c["arch"] in archs]
            if known and ok and len(ok) < len(keep):
                ignored += [{**c, "why_ignored": f"the DeepSeek engine has no code for {c['arch']} (it runs on {', '.join(archs)})"}
                            for c in keep if c not in ok]
                keep = ok
        if win and len(keep) > 1:
            # the card setup takes (setup.py amd_rank): a card with memory of its own before an APU, then the most memory, then the lower number
            first = min(keep, key=lambda c: (bool(c.get("uma") or c.get("integrated")), -round(float(c.get("vram_gb") or 0)), c.get("index", 0)))
            ignored += [{**c, "why_ignored": "on Windows a model runs on one AMD card"} for c in keep if c is not first]
            keep = [first]
    keep = sorted(keep, key=lambda c: c.get("index", 0))
    return keep, ignored



def _is_rdna4(card: dict) -> bool:
    return str(card.get("vendor") or "").lower() == "amd" and str(card.get("arch") or "").lower().startswith(DSV41_HIP_ARCHS)


def rdna4_faster(cards: list[dict] | None, model: str | None, vendor: str | None = None, named: bool = False) -> bool:
    """Does the engine's `--experts auto` give 5 on these RDNA4 cards (src/dsv41/ds4_experts_rule.hpp)?  One card the run uses: yes (the CPU lane does most
    of the experts: 1 x R9700 +16% decode, -24% agent loop, KL +0.004).  Two or more: yes when together they hold at least half of the experts, which is what
    this plan says for the set (4 x R9700: 12,260 of 15,360; 2 x R9700: about 40% of the 2.25 set, so 6).  The same cards `run_cards` picks (on Windows one
    AMD card).  False for anything else, so the screens never say Full quality where the engine runs 5, nor the reverse."""
    keep, _ = run_cards(cards, vendor, named=named)
    if not keep or not all(_is_rdna4(c) for c in keep):
        return False
    if len(keep) == 1:
        return True
    if not variant(model):
        return False
    p = plan(keep, None, None, model=model, vendor=vendor, named=True)
    return p["experts_in_vram"] >= 0.5 * p["experts_total"]


def uma_card(cards: list[dict] | None) -> dict | None:
    """The integrated Radeon a run uses, when the graphics memory is the PC's RAM (Ryzen AI Max, gfx1151): setup marks it `uma` (and
    `integrated`); a discrete card never is.  None on every other computer."""
    return next((c for c in cards or [] if c.get("uma") or c.get("integrated")), None)


def uma_total_gb(card: dict, ram_gb: float | None) -> float:
    """The one pool of a unified-memory PC, counted once: the RAM Windows sees plus the BIOS carve-out the GPU owns (the "shared"
    part of the GPU's figure is that same RAM).  On the owner's Strix Halo: 64 GB + a 64 GB carve-out = 128 GB.  Without the carve-out
    in the card's data, the GPU's figure less its shared part, or the larger of the two, is the best that can be said."""
    ram = float(ram_gb or 0.0)
    dedicated = float(card.get("dedicated_gb") or 0.0)
    if dedicated > 0:
        return ram + dedicated
    vram = float(card.get("vram_gb") or 0.0)
    shared = float(card.get("shared_gb") or 0.0)
    if vram > 0 and shared > 0:
        return ram + max(0.0, vram - shared)
    return max(ram, vram)


def variant_key(model: str | None) -> str | None:
    return next((k for k in ISTA_SETS if k.lower() == str(model or "").lower()), None)


def ista_download_bytes(model: str, have_engram: bool = False) -> int:
    v = variant(model)
    return v["set_bytes"] + ISTA_DSPARK_BYTES + (0 if have_engram else ISTA_ENGRAM_BYTES)


def recommend(ram_gb: float | None, disk_free_gb: float | None = None) -> str:
    """The official set that suits this computer: 2.00 bpw, unless its memory (under the measured 40 GB of RAM) or its disk (under 2.00's need but
    enough for the smaller one) make the 1.75 bpw set the better start.  Only a recommendation: every set can still be chosen."""
    big, small = "IST-DAS-2.00bpw", "IST-DAS-1.75bpw"
    if ram_gb is not None and float(ram_gb) < ISTA_SETS[big]["ram_ok_gb"]:
        return small
    if disk_free_gb is not None:
        need_big = ista_download_bytes(big) / GB + DISK_HEADROOM_GB
        need_small = ista_download_bytes(small) / GB + DISK_HEADROOM_GB
        if float(disk_free_gb) < need_big and float(disk_free_gb) >= need_small:
            return small
    return big


def _round1(x: float) -> float:
    return round(float(x) * 10) / 10


def _cap_gb(vram_gb: float) -> float:
    """Decimal GB a card of `vram_gb` (as the catalogs state it, GiB: 32 for an R9700) has."""
    return float(vram_gb) * GIB / GB


def split_layers(weights: list[float], layers: int = LAYERS) -> list[tuple[int, int]]:
    """The layer ranges (first, last) of consecutive cards, in proportion to `weights` (each card at least one layer).
    The engine balances its own split; on cards of one size this is what it picked (0-9, 10-19, 20-29, 30-39)."""
    n = len(weights)
    if n <= 1:
        return [(0, layers - 1)]
    total = float(sum(weights)) or 1.0
    raw = [w / total * layers for w in weights]
    counts = [max(1, int(r)) for r in raw]
    order = sorted(range(n), key=lambda i: raw[i] - int(raw[i]), reverse=True)
    k = 0
    while sum(counts) < layers:
        counts[order[k % n]] += 1
        k += 1
    while sum(counts) > layers:
        i = max(range(n), key=lambda j: counts[j])
        counts[i] -= 1
    out, first = [], 0
    for c in counts:
        out.append((first, first + c - 1))
        first += c
    return out


def tp_for(n_cards: int, cards: list[dict] | None = None) -> int:
    """--tp for this many cards.  Pairs of cards sharing each layer's matrices (--tp 2, the engine's default for two or more cards) were
    measured 2x faster for prompts than a plain layer split on the first rig; on four AMD Radeon AI PRO R9700 cards the plain layer split
    (--tp 1) is faster for both: 8K prompt 370 against 196 tokens a second and 32 against 29 for answers (2026-10-23, r9700), so the ISTA-DAS plans give four
    or more AMD cards 1 (cards passed); the Unsloth plan is unmeasured and keeps 2.  Two and three cards keep 2 (two R9700: prompts 220 against 140, answers slower; not changed).
    (The engine's own check refuses 4 - 576 is not a multiple of 128 - and drops to 2.)"""
    if n_cards >= 4 and cards and all(str(c.get("vendor") or "").lower() == "amd" for c in cards):
        return 1
    return 2 if n_cards >= 2 else 1


def _tp_args(n_cards: int, tp: int) -> list[str]:
    """The engine's --tp when it is not the engine's own pick: it takes 2 with two or more cards when told nothing, so 1 is written out."""
    return ["--tp", str(tp)] if tp > 1 or n_cards >= 2 else []


def _plan_ista(model: str, cards: list[dict] | None, ram_gb: float | None, disk_free_gb: float | None, context: int,
               vendor: str | None, named: bool = False) -> dict[str, Any]:
    """plan() for an official ISTA-DAS set (ISTA_SETS): the same keys, the measured speeds in plain words.  RAM classes: with RAM for every
    expert the CPU lane keeps ("fits"), with 40 GB or more the engine's RAM store keeps the hot experts and reads the rest from the SSD
    ("tight": measured 4.7 tokens a second on 64 GB, 7.2 on 46 GB), under 40 GB it is untested and slower ("spills").  Never a refusal."""
    v = variant(model)
    epl = int(v.get("experts_per_layer", EXPERTS_PER_LAYER))     # a layer's routed experts (192 in the ISTA-DAS Coder)
    n_experts = LAYERS * epl
    dense_gb_v = float(v.get("dense_gb", ISTA_DENSE_GB))
    measured_v = v.get("measured", ISTA_MEASURED)
    keep, ignored = run_cards(cards, vendor, named=named)
    n = len(keep)
    expert_gb = v["experts_gb"] / n_experts
    set_gb = v["set_bytes"] / GB
    download = ista_download_bytes(model) / GB
    out: dict[str, Any] = {"variant": model, "cards": [], "experts_total": n_experts, "experts_in_vram": 0, "vram_experts_gb": 0.0,
                           "cpu_experts_gb": v["experts_gb"], "weights_gb": _round1(set_gb), "experts_gb": _round1(v["experts_gb"]),
                           "dense_gb": _round1(dense_gb_v), "engram_gb": _round1(ISTA_ENGRAM_BYTES / GB), "download_gb": _round1(download),
                           "tp": tp_for(n, keep), "args": [], "measured": measured_v, "lines": [], "warnings": [], "ram_gb": ram_gb,
                           "context": context, "disk_total_gb": v["disk_total"]}
    need_disk = download + DISK_HEADROOM_GB
    out["disk_need_gb"] = _round1(need_disk)
    out["disk_ok"] = None if disk_free_gb is None else disk_free_gb >= need_disk
    lines, warns = out["lines"], out["warnings"]
    lines.append(f"DeepSeek-V4.1-Flash, {v['title']}: a {download:.0f} GB download ({set_gb:.0f} GB of model, {ISTA_ENGRAM_BYTES / GB:.0f} GB "
                 f"n-gram table, {ISTA_DSPARK_BYTES / GB:.0f} GB drafter). Quality: {v['quality']}.")
    lines.append("The n-gram table stays on the SSD and is read a few rows for each word: use an NVMe drive for the model files (a slow "
                 "disk makes every word slower). The same table serves every ISTA-DAS set, so installing another one downloads it once.")
    if n == 0:
        out["verdict"] = "no_card"
        warns.append("No graphics card Strata can use was found: this model needs one or more cards (8 GB or more; 12 GB is what it was "
                     "measured with).")
    else:
        amd = all(str(c.get("vendor") or vendor or "").lower() == "amd" for c in keep)
        per_dense = (ISTA_DENSE_GPU_BASE_GB.get(model, ISTA_DENSE_GPU_BASE_GB["IST-DAS-2.25bpw"]) + CARD_DENSE_REPEAT_GB * n) / n
        res_gb = expert_gb * RESIDENT_EXPERT_FACTOR     # what an expert that stays on a card takes
        ranges = split_layers([float(c.get("vram_gb") or 0) for c in keep])

        def place(other: float):
            cap = [max(0.0, _cap_gb(float(c.get("vram_gb") or 0)) - per_dense - ENGINE_FREE_GB - other) for c in keep]
            vram_exp = min(v["experts_gb"] * RESIDENT_EXPERT_FACTOR, sum(cap))
            counts = []
            for room, (a, b) in zip(cap, ranges):
                gb = min(room, vram_exp * (room / sum(cap))) if sum(cap) else 0.0
                counts.append(min(int(gb / res_gb), (b - a + 1) * epl))
            return cap, counts

        cap, counts = place(CARD_OTHER_GB["amd"] if amd else CARD_OTHER_GB["other"])
        if amd and sum(counts) < n_experts / 2:          # cards that hold under half of the experts: the engine's larger prompt buffers
            cap, counts = place(CARD_OTHER_GB["amd_low_share"])
        in_vram = 0
        for c, ne, (a, b) in zip(keep, counts, ranges):
            in_vram += ne
            out["cards"].append({"index": c.get("index"), "name": c.get("name"), "vram_gb": float(c.get("vram_gb") or 0), "layers": [a, b],
                                 "dense_gb": _round1(per_dense), "experts_gb": _round1(ne * res_gb), "experts": ne})
        out["experts_in_vram"] = in_vram
        out["vram_experts_gb"] = _round1(in_vram * res_gb)
        out["cpu_experts_gb"] = _round1(max(0.0, v["experts_gb"] - in_vram * res_gb))
        out["tp"] = tp_for(n, keep)
        out["args"] = _tp_args(n, out["tp"])
        names = " + ".join(f"{c['name'] or 'card ' + str(c['index'])} ({c['vram_gb']:.0f} GB)" for c in out["cards"])
        lines.append(f"Graphics memory: {names}. About {in_vram:,} of the {n_experts:,} experts ({out['vram_experts_gb']:.0f} GB) stay on "
                     f"the card{'s' if n > 1 else ''}; the other {n_experts - in_vram:,} ({out['cpu_experts_gb']:.0f} GB) are computed on the "
                     "CPU from RAM, and the ones RAM cannot hold are read from the SSD when a word needs them.")
        if n >= 2 and out["tp"] == 2:
            lines.append("Two cards share each layer's matrices (--tp 2): measured about twice as fast at reading prompts.")
        if ram_gb:
            all_cached = out["cpu_experts_gb"] + REST_RAM_GB
            out["ram_needed_gb"] = _round1(all_cached)
            out["spill_gb"] = _round1(max(0.0, all_cached - float(ram_gb)))
            out["spare_ram_gb"] = _round1(max(0.0, float(ram_gb) - all_cached))
            if float(ram_gb) >= all_cached:
                out["verdict"] = "fits"
                lines.append(f"That fits this computer: {ram_gb:.0f} GB of RAM keeps all {out['cpu_experts_gb']:.0f} GB of the CPU's experts cached.")
            elif float(ram_gb) >= v["ram_ok_gb"]:
                out["verdict"] = "tight"
                lines.append(f"This computer has {ram_gb:.0f} GB of RAM: Strata keeps the experts you use most in RAM (it sizes this itself) and "
                             + (v["tight_words"] if v.get("tight_words") else "reads the rest from the SSD. Measured: about 4.7 tokens a second on 64 GB of RAM and a 12 GB card, about 7 on "
                             "46 GB with a faster drive. More RAM or a faster NVMe drive makes it quicker."))
            else:
                out["verdict"] = "spills"
                warns.append(v["spill_words"].format(ram=ram_gb) if v.get("spill_words") else
                             f"This computer has {ram_gb:.0f} GB of RAM and this model was measured with 46 GB and 64 GB. With less, most "
                             "experts come from the SSD while it answers: it works but is slow (a word can take seconds). More RAM fixes "
                             "it. It is still allowed.")
        else:
            out["verdict"] = "unknown"
        short = [c for c in keep if _cap_gb(float(c.get("vram_gb") or 0)) < per_dense + RESERVE_GB + 1.0]
        if short:
            warns.append(f"The dense weights need ~{per_dense:.0f} GB of graphics memory and {short[0].get('name') or 'a card'} "
                         f"({float(short[0].get('vram_gb') or 0):.0f} GB) has hardly that: the engine may stop at the start. It is still allowed.")
        if ignored:
            lines.append("Not used: " + "; ".join(f"{c.get('name') or 'a card'} ({c.get('why_ignored')})" for c in ignored) + ".")
        if measured_v:
            lines.append("Measured on " + measured_v + ". Other computers are slower or faster by what their cards, RAM and drive can do.")
        else:
            lines.append("This version has not been timed on a computer like this one yet; the sizes above are exact.")
    if disk_free_gb is not None and disk_free_gb < need_disk:
        warns.append(f"The download needs about {need_disk:.0f} GB of free disk space and {disk_free_gb:.0f} GB are free where the files go. "
                     "Use --models-dir on a bigger drive, or --gguf-dir for files you already have.")
    out["words"] = " ".join(lines)
    return out


def _plan_uma(key: str, model: str, card: dict, ram_gb: float | None, disk_free_gb: float | None, context: int) -> dict[str, Any]:
    """plan() for an official ISTA-DAS set on a unified-memory APU (Ryzen AI Max, gfx1151, Windows): the graphics memory is the RAM,
    counted once.  The engine holds what fits beside the dense weights and the reserve and reads the rest from the SSD; the numbers in
    UMA_MEASURED are the measured 64 GB box.  Never a refusal."""
    v = ISTA_SETS[key]
    epl = int(v.get("experts_per_layer", EXPERTS_PER_LAYER))     # a layer's routed experts (192 in the ISTA-DAS Coder)
    n_experts = LAYERS * epl
    dense_gb_v = float(v.get("dense_gb", ISTA_DENSE_GB))
    measured_v = v.get("measured", ISTA_MEASURED)
    total = uma_total_gb(card, ram_gb)
    expert_gb = v["experts_gb"] / n_experts
    set_gb = v["set_bytes"] / GB
    download = ista_download_bytes(model) / GB
    room = max(0.0, total - dense_gb_v - UMA_RESERVE_GB)
    resident = min(float(v["experts_gb"]), room)
    streamed = max(0.0, float(v["experts_gb"]) - room)
    in_mem = int(resident / expert_gb)
    need_disk = download + DISK_HEADROOM_GB
    out: dict[str, Any] = {
        "variant": model, "cards": [{"index": card.get("index"), "name": card.get("name"), "vram_gb": _round1(total),
                                     "layers": [0, LAYERS - 1], "dense_gb": _round1(dense_gb_v),
                                     "experts_gb": _round1(resident), "experts": in_mem}],
        "experts_total": n_experts, "experts_in_vram": in_mem, "vram_experts_gb": _round1(resident), "cpu_experts_gb": 0.0,
        "weights_gb": _round1(set_gb), "experts_gb": _round1(v["experts_gb"]), "dense_gb": _round1(dense_gb_v),
        "engram_gb": _round1(ISTA_ENGRAM_BYTES / GB), "download_gb": _round1(download), "tp": 1, "args": [],
        "measured": "" if v.get("unproven_uma") else UMA_MEASURED, "lines": [], "warnings": [], "ram_gb": ram_gb, "context": context,
        "disk_total_gb": v["disk_total"], "disk_need_gb": _round1(need_disk),
        "disk_ok": None if disk_free_gb is None else disk_free_gb >= need_disk,
        "uma": True, "streamed_gb": _round1(streamed), "ram_needed_gb": _round1(total), "spill_gb": _round1(streamed),
        "spare_ram_gb": _round1(max(0.0, total - dense_gb_v - v["experts_gb"] - UMA_RESERVE_GB)),
    }
    lines, warns = out["lines"], out["warnings"]
    lines.append(f"DeepSeek-V4.1-Flash, {v['title']}: a {download:.0f} GB download ({set_gb:.0f} GB of model, "
                 f"{ISTA_ENGRAM_BYTES / GB:.0f} GB n-gram table, {ISTA_DSPARK_BYTES / GB:.0f} GB drafter). Quality: {v['quality']}.")
    lines.append("The n-gram table stays on the SSD and is read a few rows for each word: use an NVMe drive for the model files (a slow "
                 "disk makes every word slower). The same table serves every ISTA-DAS set, so installing another one downloads it once.")
    name = card.get("name") or "the built-in Radeon"
    lines.append(f"{name} shares this PC's memory: the RAM and the GPU's BIOS carve-out are one pool of {total:.0f} GB. About "
                 f"{resident:.0f} GB of the {float(v['experts_gb']):.0f} GB of experts stay in it beside {dense_gb_v:.1f} GB of "
                 f"dense weights and {UMA_RESERVE_GB:.0f} GB kept for Windows; the other {streamed:.0f} GB are read from the SSD while it answers.")
    if not v.get("unproven_uma"):                   # (these speeds are the 2.00 set's: a set not run there says so below instead)
        lines.append(f"Measured on this kind of PC: {UMA_MEASURED}.")
    if v.get("unproven_uma"):
        lines.append(v["unproven_uma"])
    if streamed > 0.5:
        lines.append("Keep the model files on the NVMe drive and the machine otherwise idle while it answers: every word reads experts from it.")
    if streamed <= 0.5:
        out["verdict"] = "fits"
    elif total >= float(v.get("ram_ok_gb") or 0):
        out["verdict"] = "tight"
        if total < 48:
            warns.append(f"This PC has {total:.0f} GB of memory shared with the graphics chip and the model was measured on 128 GB: with less, "
                         "more experts come from the SSD and each word takes longer. It still runs.")
    else:
        out["verdict"] = "spills"
        warns.append(f"This PC has {total:.0f} GB of memory shared with the graphics chip and the model was measured on 128 GB: most of its "
                     "experts are read from the SSD while it answers, so it is slow (a word can take seconds). The smaller 1.75 bpw set "
                     "suits this computer better. It is still allowed.")
    if disk_free_gb is not None and disk_free_gb < need_disk:
        warns.append(f"The download needs about {need_disk:.0f} GB of free disk space and {disk_free_gb:.0f} GB are free where the files go. "
                     "Use --models-dir on a bigger drive, or --gguf-dir for files you already have.")
    out["words"] = " ".join(lines)
    return out


def plan(cards: list[dict] | None, ram_gb: float | None, disk_free_gb: float | None = None, context: int = 8192,
         vendor: str | None = None, model: str | None = None, named: bool = False) -> dict[str, Any]:
    """Where the model goes on this computer.  `cards`: the graphics cards ({"index", "name", "vram_gb", ...}, as gpu_fit takes them),
    `ram_gb` the RAM, `disk_free_gb` the free space where the files go (None: not checked).  `model`: an official ISTA-DAS set
    (ISTA_SETS: "IST-DAS-2.00bpw" / "IST-DAS-2.25bpw" / "IST-DAS-1.75bpw") gets its own numbers; None or "UD-Q8_K_XL" is Unsloth's 508 GB set.

    Keys: cards (per card: index, name, vram_gb, layers, dense_gb, experts_gb, experts), experts_in_vram, experts_total, vram_experts_gb,
    cpu_experts_gb, ram_needed_gb, spill_gb, spare_ram_gb, verdict (fits / tight / spills / unknown / no_card), tp, args,
    download_gb, engram_gb, disk_need_gb, disk_ok, lines (plain words, one idea each), warnings (the risks, plainly), measured.
    On a unified-memory APU (Ryzen AI Max, gfx1151) the plan counts one pool: also `uma: True` and `streamed_gb` (the experts
    read from the SSD), tp 1 and no args (nothing to split between cards)."""
    if variant(model):
        mine = gpu_fit.for_vendor(list(cards or []), vendor)[0] if vendor else list(cards or [])
        card = uma_card(gpu_fit.usable_cards(mine)[0])   # an APU beside a card with memory of its own does not count (setup leaves it out too)
        if card is not None:
            return _plan_uma(variant_key(model), model, card, ram_gb, disk_free_gb, context)
        return _plan_ista(variant_key(model), cards, ram_gb, disk_free_gb, context, vendor, named)
    keep, ignored = run_cards(cards, vendor, named=named)                 # the engine takes the layers in card order
    n = len(keep)
    out: dict[str, Any] = {"cards": [], "experts_total": EXPERTS, "experts_in_vram": 0, "vram_experts_gb": 0.0,
                           "cpu_experts_gb": EXPERTS_GB, "weights_gb": _round1(WEIGHTS_GB), "experts_gb": _round1(EXPERTS_GB),
                           "dense_gb": _round1(DENSE_GB), "engram_gb": _round1(ENGRAM_GB), "download_gb": _round1(DOWNLOAD_GB),
                           "tp": tp_for(n), "args": [], "measured": MEASURED, "lines": [], "warnings": [],
                           "ram_gb": ram_gb, "context": context}
    need_disk = DOWNLOAD_GB + DISK_HEADROOM_GB
    out["disk_need_gb"] = _round1(need_disk)
    out["disk_ok"] = None if disk_free_gb is None else disk_free_gb >= need_disk
    lines, warns = out["lines"], out["warnings"]
    lines.append(f"DeepSeek-V4.1-Flash (UD-Q8_K_XL): a {DOWNLOAD_GB:.0f} GB download. In memory it needs {WEIGHTS_GB:.0f} GB: "
                 f"{EXPERTS_GB:.0f} GB of experts ({EXPERTS:,} of {EXPERT_GB * 1000:.1f} MB) and {DENSE_GB:.0f} GB of attention "
                 "and other dense weights.")
    lines.append(f"Its n-gram table ({ENGRAM_GB:.0f} GB of the download) stays on the SSD and is read a few rows for each word "
                 "it writes: it is not counted in RAM, but a slow disk (SATA, a network drive, a USB disk) makes every word "
                 "slower. Use an NVMe drive for the model files.")
    if n == 0:
        out["verdict"] = "no_card"
        warns.append("No graphics card Strata can use was found: this model needs one or more cards (it was run on AMD Radeon "
                     "AI PRO R9700 cards).")
    else:
        per_dense = DENSE_ON_GPU_GB / n
        cap = []
        for c in keep:
            room = _cap_gb(float(c.get("vram_gb") or 0)) - per_dense - RESERVE_GB
            cap.append(max(0.0, room))
        vram_exp = min(EXPERTS_GB, sum(cap))
        share = [x / sum(cap) if sum(cap) else 1.0 / n for x in cap]
        ranges = split_layers([float(c.get("vram_gb") or 0) for c in keep])
        in_vram = 0
        for c, room, (a, b), s in zip(keep, cap, ranges, share):
            gb = min(room, vram_exp * s) if sum(cap) else 0.0
            ne = int(gb / EXPERT_GB)
            ne = min(ne, (b - a + 1) * EXPERTS_PER_LAYER)
            in_vram += ne
            out["cards"].append({"index": c.get("index"), "name": c.get("name"), "vram_gb": float(c.get("vram_gb") or 0),
                                 "layers": [a, b], "dense_gb": _round1(per_dense), "experts_gb": _round1(ne * EXPERT_GB),
                                 "experts": ne})
        out["experts_in_vram"] = in_vram
        out["vram_experts_gb"] = _round1(in_vram * EXPERT_GB)
        out["cpu_experts_gb"] = _round1(EXPERTS_GB - in_vram * EXPERT_GB)
        out["tp"] = tp_for(n)
        out["args"] = ["--tp", str(out["tp"])] if out["tp"] > 1 else []
        if ram_gb:
            needed = out["cpu_experts_gb"] + REST_RAM_GB
            spare = float(ram_gb) - needed
            out["ram_needed_gb"] = _round1(needed)
            out["spare_ram_gb"] = _round1(max(0.0, spare))
            out["spill_gb"] = _round1(max(0.0, -spare))
            out["verdict"] = "spills" if spare < 0 else "tight" if spare < SPARE_FITS_GB else "fits"
        else:
            out["verdict"] = "unknown"
        # the placement, in words
        short = [c for c in keep if _cap_gb(float(c.get("vram_gb") or 0)) < per_dense + RESERVE_GB + 1.0]
        if short:
            warns.append(f"The dense weights need ~{per_dense:.0f} GB of graphics memory on {'each card' if n > 1 else 'the card'} "
                         f"and {short[0].get('name') or 'a card'} ({float(short[0].get('vram_gb') or 0):.0f} GB) has "
                         "hardly that: the engine may stop at the start (out of graphics memory). More or bigger cards fix it; "
                         "it is still allowed.")
        if n == 1:
            c = out["cards"][0]
            lines.append(f"One card ({c['name'] or 'card ' + str(c['index'])}, {c['vram_gb']:.0f} GB): the dense weights ({c['dense_gb']:.0f} GB) "
                         f"and {c['experts']:,} of the {EXPERTS:,} experts ({c['experts_gb']:.0f} GB) go in its memory, the other "
                         f"{EXPERTS - c['experts']:,} experts ({out['cpu_experts_gb']:.0f} GB) are computed on the CPU from RAM.")
        else:
            blocks = "  |  ".join(f"card {c['index']}: layers {c['layers'][0]}-{c['layers'][1]}" for c in out["cards"])
            lines.append(f"{n} cards, one block of layers on each ({blocks}); the engine balances the split. On each card "
                         f"~{out['cards'][0]['dense_gb']:.1f} GB of dense weights and ~{out['cards'][0]['experts_gb']:.0f} GB of experts "
                         f"(about {out['cards'][0]['experts']:,}); in all {in_vram:,} of {EXPERTS:,} experts ({out['vram_experts_gb']:.0f} GB) "
                         f"are in graphics memory.")
            lines.append("Two cards share each layer's matrices (--tp 2): measured about twice as fast at reading prompts as a plain "
                         "split of the layers.")
        lines.append(f"The other {EXPERTS - in_vram:,} experts ({out['cpu_experts_gb']:.0f} GB) are computed on the CPU, read in place "
                     "from the model's files: they need that much free RAM to stay cached "
                     f"({out.get('ram_needed_gb', out['cpu_experts_gb'] + REST_RAM_GB):.0f} GB with {REST_RAM_GB:.0f} GB for the system and the engine).")
        if out["verdict"] == "spills":
            warns.append(f"This computer has {ram_gb:.0f} GB of RAM and {out['ram_needed_gb']:.0f} GB would keep the CPU's share of the experts "
                         f"cached: about {out['spill_gb']:.0f} GB of them are read from the SSD while the model answers, which is slow "
                         "(a word can take seconds). More RAM or more graphics memory fixes it. It is still allowed.")
        elif out["verdict"] == "tight":
            warns.append(f"RAM is tight: {out['spare_ram_gb']:.0f} GB are left beside the experts; close other programs.")
        elif out["verdict"] == "fits":
            lines.append(f"That fits this computer ({ram_gb:.0f} GB of RAM, {out['spare_ram_gb']:.0f} GB to spare).")
        if n >= 2 and any(abs(float(c.get("vram_gb") or 0) - float(keep[0].get("vram_gb") or 0)) > 1 for c in keep):
            lines.append("The cards are not the same size: the layers are shared in proportion to their memory.")
        if ignored:
            lines.append("Not used: " + "; ".join(f"{c.get('name') or 'a card'} ({c.get('why_ignored')})" for c in ignored) + ".")
        lines.append("Measured on " + MEASURED + ". Other computers are slower or faster by what their cards and RAM can hold; "
                     "this engine has been run on AMD cards.")
    if disk_free_gb is not None and disk_free_gb < need_disk:
        warns.append(f"The download needs about {need_disk:.0f} GB of free disk space and {disk_free_gb:.0f} GB are free where the "
                     "files go. Use --models-dir on a bigger drive, or --gguf-dir for files you already have.")
    out["words"] = " ".join(lines)
    return out


def _option(key: str, cards: list[dict], ram_gb: float | None, ctx: int, vendor: str | None, model: str | None = None) -> dict[str, Any]:
    p = plan(cards, ram_gb, None, ctx, vendor, model)
    names = " + ".join(gpu_fit._name(c) for c in cards)
    label = f"{names} + RAM" if len(cards) == 1 else f"{names} + RAM (layers shared, --tp {p['tp']})"
    words = {"fits": f"fits: {p['experts_in_vram']:,} of {p['experts_total']:,} experts in graphics memory, the rest computed on the CPU "
                     f"({p.get('ram_needed_gb', 0):.0f} GB of RAM)",
             "tight": f"fits, but tight: RAM left over is under {SPARE_FITS_GB:.0f} GB",
             "spills": f"does not fit in memory: about {p.get('spill_gb', 0):.0f} GB of the experts are read from the SSD while answering (slow)",
             "unknown": "memory of this computer unknown", "no_card": "no usable card"}[p["verdict"]]
    return {"key": key, "cards": [c.get("index") for c in cards], "label": label, "verdict": p["verdict"], "summary": f"{label}: {words}",
            "words": words, "vram_gb": _round1(sum(float(c.get("vram_gb") or 0) for c in cards)), "spill_gb": p.get("spill_gb", 0.0),
            "ram_use_gb": p.get("cpu_experts_gb", 0.0), "spare_ram_gb": p.get("spare_ram_gb", 0.0),
            "notes": [l for l in p["lines"] if l.startswith(("One card", f"{len(cards)} cards", "Two cards share"))],
            "recommended": False, "plan": p}


def options(cards: list[dict], ram_gb: float | None, vendor: str | None = None, context: int = 8192, model: str | None = None) -> dict[str, Any]:
    """gpu_fit.options for this model: every card that counts (recommended: the model is bigger than any one card, so each card adds
    experts that stay off the CPU), and the largest card alone.  Same keys as gpu_fit.options."""
    keep, ignored = run_cards(cards, vendor)          # the same cards `strata launch --plan` and setup use (Windows: one AMD card; the package's own archs)
    out: dict[str, Any] = {"options": [], "recommended": None, "ignored": ignored, "cards": len(keep), "vendor": vendor,
                           "other_vendor_note": gpu_fit.other_vendor_note(list(cards or []), vendor)}
    if not keep:
        return out
    opts = [_option("all", sorted(keep, key=lambda c: c.get("index", 0)), ram_gb, context, vendor, model)]
    if len(keep) > 1:
        largest = min(keep, key=lambda c: (-round(float(c.get("vram_gb") or 0)), c.get("index", 0)))
        opts.append(_option("single", [largest], ram_gb, context, vendor, model))
    opts[0]["recommended"] = True
    out["options"], out["recommended"] = opts, "all"
    return out
