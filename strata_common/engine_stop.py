"""Why an engine start ended: the engine's own line saying a card ran out of memory. The launcher's final line and the
server's retry without a drafter (strata_common/engine_client.py) both read it."""

from __future__ import annotations

import re

ENGINE_STOP_PATTERNS = (
    re.compile(r"expert cache: no room", re.I),
    re.compile(r"no VRAM is left for the expert cache", re.I),
    re.compile(r"too little free VRAM", re.I),
    re.compile(r"cannot start the prompt path", re.I),
    re.compile(r"device buffers .* do not fit", re.I),
    re.compile(r"weight arena (does not fit|failed)|cudaMalloc\(\d+\) for the weight arena failed", re.I),
    re.compile(r"(session state|head|native dense projections|weights).*(out of memory|does not fit|failed)", re.I),
)
ENGINE_LINE = re.compile(r"^(?:\[strata\]\s*)?strata (?:generate|serve|mtp|prefill): ", re.I)
# a line that only reports what the walk did, never why a start stopped
_NOT_A_STOP = re.compile(r"short of GPU memory: |trying a smaller|trying \d+ of \d+ slots|a card this small|LOW: ", re.I)


def engine_stop_reason(lines) -> str | None:
    """Gets the last GPU memory failure in one start's log, without the engine prefix.

    Returns None when the start stopped for another reason.
    """
    for raw in reversed([str(x).strip() for x in lines or []]):
        if not ENGINE_LINE.match(raw) or _NOT_A_STOP.search(raw):
            continue
        if any(p.search(raw) for p in ENGINE_STOP_PATTERNS):
            return ENGINE_LINE.sub("", raw, count=1).strip()
    return None
