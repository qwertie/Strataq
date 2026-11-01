"""How much RAM this computer has, read on Windows, Linux and macOS without ever raising."""

from __future__ import annotations

import ctypes
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Ram:
    """System memory in bytes; None where the system would not say."""
    total: int | None = None
    available: int | None = None        # what a new process could use without swapping


def read_ram() -> Ram:
    """Reads the installed and the available RAM."""
    try:
        if sys.platform == "win32":
            status = windows_memory_status()
            return Ram(status.ullTotalPhys or None, status.ullAvailPhys or None)
        if sys.platform == "darwin":
            return Ram(int(_run(["sysctl", "-n", "hw.memsize"])), _mac_available())
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            info[key] = int(rest.split()[0]) * 1024
        return Ram(info.get("MemTotal"), info.get("MemAvailable"))
    except Exception:  # noqa: BLE001 - a probe must not raise
        return Ram()


def windows_memory_status():
    """Calls Windows' GlobalMemoryStatusEx, which also gives the commit limit (ullTotalPageFile = RAM + page file).
    Every field is 0 when the call fails."""
    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        status = MemoryStatusEx()
    return status


def _mac_available() -> int | None:
    """Gets the free, inactive and speculative pages that vm_stat reports, in bytes."""
    try:
        vm = _run(["vm_stat"])
        page = re.search(r"page size of (\d+)", vm)
        pages = sum(int(m.group(1)) for key in ("Pages free", "Pages inactive", "Pages speculative")
                    if (m := re.search(rf"{key}:\s+(\d+)", vm)))
        return pages * (int(page.group(1)) if page else 16384)
    except Exception:  # noqa: BLE001
        return None


def _run(argv: list[str]) -> str:
    return subprocess.run(argv, capture_output=True, text=True, errors="replace", timeout=5, check=True).stdout.strip()
