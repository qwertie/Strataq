"""Whether a model guesses ahead with its MTP draft layer. One place, read by the Terminal and the server.

The choice is the person's, and it is saved in the model's run config as ``"drafting"``: ``auto`` (the default), ``mtp`` or
``off``. ``strata launch --drafting ...`` writes it, and ``strata_server/server.py`` applies it at every start (like
``"mtp": false``, strata_common/draft_layer.py), so a setup run that rewrites the engine arguments cannot undo it.

  mtp      the model's own draft layer (``--mtp <folder>``)
  off      no draft layer (the same as ``"mtp": false``: the engine still looks ahead in the text)
  auto     the MTP

Recommend, never force: a draft layer that is not on this computer does not stop the start, and ``Decision.note`` says
why in one short line.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODES = ("auto", "mtp", "off")
DEFAULT_MODE = "auto"


@dataclass(frozen=True)
class Decision:
    asked: str                   # what the person chose (auto | mtp | off)
    drafter: str                 # what the engine is started with: mtp | off
    note: str = ""               # plain words when the choice could not be applied; "" otherwise


def normalize(value: Any) -> str:
    """Gets the drafting mode a saved value means; anything unknown (an older "dflash2" too) is Auto."""
    text = str(value if value is not None else DEFAULT_MODE).strip().lower()
    return text if text in MODES else DEFAULT_MODE


#: Why no drafter runs: each is a short reason in the line "Drafting: off (<reason>)".
NONE_HERE = "no drafter is downloaded yet"
SHORT_OF_VRAM = "the cards were short of memory"   # the engine ended before it was ready for lack of graphics memory: the start was repeated once with no drafter (strata_common/engine_client.py)


def line(drafter: str, reason: str = "") -> str:
    """Gets the line that says what drafts: ``Drafting: MTP``, ``Drafting: off (no drafter is downloaded yet)``."""
    word = {"mtp": "MTP", "off": "off"}.get(drafter, drafter)
    return f"Drafting: {word} ({reason})" if reason else f"Drafting: {word}"


def mtp_ready(args: list[Any]) -> bool | None:
    """Is the MTP layer this start would load on this computer? ``--mtp <folder>`` with its ``experts.bin`` (or ``mtp-q2_0.gguf`` beside the folder).
    None (not known, taken as there) when the arguments name no layer at all (a layer switched off) or a draft depth instead of a folder (Qwen3.6's
    ``--mtp 4``: that layer is its own)."""
    items = [str(a) for a in args or []]
    if "--mtp" not in items:
        return None
    value = _flag_value(args, "--mtp")
    if value is None or value.isdigit():
        return None
    folder = Path(value)
    try:
        return (folder / "experts.bin").is_file() or (folder.parent / "mtp-q2_0.gguf").is_file()
    except OSError:
        return False


def decide(mode: Any, mtp_here: bool | None = None) -> Decision:
    """Gets the drafter a start uses. Never raises. ``mtp_here`` says whether the MTP layer is on disk (None: not known, taken as there)."""
    asked = normalize(mode)
    if asked == "off":
        return Decision(asked, "off")
    return Decision(asked, "mtp") if mtp_here is not False else Decision(asked, "off", NONE_HERE)


def summary_line(decision: Decision) -> str:
    """The line the Terminal shows: what drafts."""
    return line(decision.drafter, decision.note) if decision.note else f"Drafting: {'MTP' if decision.drafter == 'mtp' else 'Off'}"


# --------------------------------------------------------------------------------------------------- what a start did

_pending_note = ""


def remember_note(note: str) -> None:
    """The server calls this when it builds the engine arguments: why the choice was not applied, for the engine that starts next."""
    global _pending_note
    _pending_note = note or ""


def take_note() -> str:
    global _pending_note
    note, _pending_note = _pending_note, ""
    return note


def engine_info(args: list[Any], note: str = "") -> dict[str, str]:
    """The facts of one engine start for /metrics: ``drafting`` (mtp | off) and ``drafting_note`` (plain words, or ""): the
    drafter this start loads, and why it is not the one asked for."""
    drafter = "mtp" if "--mtp" in [str(a) for a in args] else "off"
    return {"drafting": drafter, "drafting_note": line(drafter, note) if note else ""}


def _flag_value(args: list[Any], flag: str) -> str | None:
    items = [str(a) for a in args or []]
    return items[items.index(flag) + 1] if flag in items and items.index(flag) + 1 < len(items) else None
