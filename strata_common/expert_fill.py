"""Waits until a paged engine (strata-q38, strata-ds41) has filled its expert pools, so that a speed test does not
time the fill. Those engines say READY, and the server answers, before the pools are full."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping

from strata_base import loopback


def wait_for_experts(read_memory: Callable[[], Mapping | Iterable[str]], timeout: float = 1800,
                     log: Callable[[str], object] = print) -> float:
    """Polls `read_memory` until the engine's expert pools hold every expert they can take now, and gets the seconds
    that took. `read_memory` gets either a parsed report (GET /v1/memory's, StrataEngine.memory's) or the lines of a
    GET_MEMORY_USE answer. A report without expert-fill progress, from an engine that loads its experts before READY,
    counts as loaded. Raises TimeoutError after `timeout` seconds."""
    start = time.monotonic()
    while True:
        report = read_memory()
        if isinstance(report, Mapping):
            loaded = report.get("expert_fill", {}).get("loaded", True)
        else:
            fields = next((line.split()[1:] for line in report if line.startswith("EXPERT_FILL ")), [])
            loaded = dict(field.split("=", 1) for field in fields).get("loaded", "1") == "1"
        elapsed = time.monotonic() - start
        if loaded:
            log(f"waited {elapsed:.1f} s for experts")
            return elapsed
        if elapsed >= timeout:
            raise TimeoutError(f"experts still loading after {elapsed:.1f} s")
        time.sleep(min(0.25, timeout - elapsed))


def wait_for_http_experts(base: str, key: str = "", timeout: float = 1800,
                          log: Callable[[str], object] = print) -> float:
    """Waits as wait_for_experts does, reading GET /v1/memory of the server at `base` with API key `key` (none when
    empty); a server without that path counts as loaded."""
    def read_memory():
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        request = urllib.request.Request(base.rstrip("/") + "/v1/memory", headers=headers)
        try:
            with loopback.urlopen(request, timeout=min(timeout, 30)) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return {}
            raise

    return wait_for_experts(read_memory, timeout, log)


def wait_for_protocol_experts(send: Callable[[str], object], read_line: Callable[[], str], timeout: float = 1800,
                              log: Callable[[str], object] = print) -> float:
    """Waits as wait_for_experts does, asking the engine itself on its --serve protocol: `send` writes one command
    line and `read_line` gets its next output line ("" at the end). The engine must be idle, so that every line
    belongs to the GET_MEMORY_USE answer."""
    def read_memory():
        send("GET_MEMORY_USE")
        lines = []
        while True:
            line = read_line()
            if not line or line.startswith("ERR"):
                raise RuntimeError("engine failed while waiting for expert loading")
            if line.startswith("END_MEMORY"):
                return lines
            lines.append(line)

    return wait_for_experts(read_memory, timeout, log)
