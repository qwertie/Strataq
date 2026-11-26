"""Terminal clients for the shared Strata server."""

import re
from pathlib import Path

SCHEMA_VERSION = 1

#: Identifies the Terminal release independently of the app and Control build.
PRODUCT_VERSION = "0.2.50"


def version() -> str:
    """Gets the Control build from VERSION, with CMake as a fallback for older source trees."""
    root = Path(__file__).resolve().parents[1]
    try:
        text = (root / "VERSION").read_text(encoding="utf-8").strip()
        if text:
            return text
    except OSError:
        pass
    try:
        m = re.search(r"^\s*project\([^)]*?VERSION\s+([0-9]+(?:\.[0-9]+)+)",
                      (root / "CMakeLists.txt").read_text(encoding="utf-8", errors="replace"), re.MULTILINE)
        if m:
            return m.group(1)
    except OSError:
        pass
    return "unknown"
