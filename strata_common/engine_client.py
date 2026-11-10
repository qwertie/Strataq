"""The engine client: starting a Strata engine program, talking to it over its standard input and output, reading its
progress and errors, and building its command line and environment from a run config.  Also reads the API keys a config
names (key_list, api_key_of).

Used by the API servers (strata_server/server.py, sycl/serve/server_intel.py), by the Terminal (strata_cli) and by its
calibration and readiness checks (strata_control).
"""
from __future__ import annotations

from urllib.parse import unquote

import collections
import contextlib
import json
import math
import os
import queue
import re
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
from array import array
from pathlib import Path

from strata_common import draft_layer
from strata_common import drafting
from strata_common import engine_component
from strata_common import gpu_roles
from strata_common import lifeline
from strata_common import logit_bias as bias_api
from strata_common import user_args
from strata_common import paged_launch
from strata_common import vram_fit


ROOT = Path(__file__).resolve().parents[1]
IM_END = "<|im_end|>"


def popen(what: str, args: list, role: str = "engine", **kw):
    """#735: subprocess.Popen, but Windows' refusal to start a program (WinError 4551, or 1260 when a policy blocks it:
    Smart App Control on a clean Windows 11 blocks an exe that is not signed) is said in words instead of a bare
    OSError."""
    try:
        return lifeline.popen(args, role, **kw)       # a child that cannot outlive this server (strata_common/lifeline.py)
    except OSError as e:
        if getattr(e, "winerror", None) in (4551, 1260):
            raise OSError(f"Windows blocked {what} ({args[0]}): Smart App Control, or an application-control policy, does "
                          "not let a program that is not signed run. Turn Smart App Control off (Windows Security > App "
                          "& browser control > Smart App Control settings), or - for the image encoder only - run setup "
                          "with --vision no (see docs/TROUBLESHOOTING.md)") from e
        raise


# How long close() gives the engine at each step of ending it (seconds). Module values so that a test can shorten them.
ENGINE_QUIT_WAIT_S, ENGINE_TERM_WAIT_S, ENGINE_KILL_WAIT_S = 20, 5, 20
# More time after QUIT for an engine that writes its conversations to the SSD as it ends (INFO save_on_shutdown=1).
ENGINE_SAVE_WAIT_S = 120


def rttime_limit_note() -> str:
    """#3238: Linux only. A hard RLIMIT_RTTIME that is not unlimited (PipeWire's client module sets it to 0 when rtkit is
    missing; every program started from that desktop inherits it) makes the kernel SIGKILL a process as soon as it
    checks a thread with an armed CPU timer, with no OOM kill and no message. Returns one sentence naming the cause and
    the fix, or "" when the limit is unlimited, the platform has no such limit or it cannot be read."""
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_RTTIME)
    except (ImportError, AttributeError, OSError, ValueError):
        return ""
    if hard == resource.RLIM_INFINITY:
        return ""
    return (f"This session has a real-time CPU limit (RLIMIT_RTTIME hard limit {hard}; the desktop's audio system sets "
            "it to 0 when rtkit is not installed). Linux can then end a program with signal 9 for no visible reason. "
            "Install rtkit, or start Strata outside the desktop session's limits, for example: "
            "systemd-run --user --collect -p LimitRTTIME=infinity -- strata serve ...")


LEFTOVER_SETTLE_S = 120.0       # #4051: how long an engine ended at a start is waited for to give its memory back


def end_leftover_engines(exe: str) -> None:
    """Before an engine starts: end the ones a lost server left running (strata_common/procs.py). Only a process
    this install recorded whose starter is gone, or an orphan of exactly this executable running --serve, and only
    after its PID, start time and command line are checked again at the moment of the signal. Never raises.
    #4051: the new engine is not started while the old one still holds its memory (tens of GB take longer than the
    10 s this used to wait): a 40 GiB arena loaded beside 50 GiB the old engine had not returned put a 60 GiB PC into
    memory reclaim for good.  One still there after LEFTOVER_SETTLE_S is said in plain words, and the start goes on
    (the engine waits for the RAM itself and warns; nothing is refused)."""
    if os.environ.get("STRATA_KEEP_LEFTOVERS") == "1":
        return
    try:
        from strata_common import procs
        say = lambda text: print(text, flush=True)        # noqa: E731
        stuck = [r["pid"] for r in procs.end_leftovers(say=say, settle=LEFTOVER_SETTLE_S)["stuck"]]
        stuck += [r["pid"] for r in procs.end_orphans(exe, say=say, settle=LEFTOVER_SETTLE_S) if r["outcome"] == "stuck"]
        if stuck:
            print(f"[strata] WARNING: the engine from an earlier run (process {', '.join(map(str, stuck))}) was told to stop "
                  f"but still holds its memory after {LEFTOVER_SETTLE_S:.0f} s. Starting the next one now can leave this PC "
                  "short of RAM and very slow: wait until that process is gone (it is shown by `ps`/Task Manager), "
                  "or restart the PC if it does not go.", flush=True)
    except Exception as e:  # noqa: BLE001 - a failed check must not stop a start
        print(f"[strata] could not check for engines left by an earlier run: {type(e).__name__}", flush=True)


# #481: a running request whose engine prints nothing (no T, PP or any other line) for this long has lost step with the
# server (the engine's main thread waits, untimed, for its next command): the engine is ended and the request fails;
# the next request starts it again.  The config's "engine_silence_s" sets it (0: wait forever, as before).
ENGINE_SILENCE_S = 300.0
# Disk sessions: the longest allowance an engine may announce (`SWAIT <phase> <s>`) for one step that blocks in a
# single call (a file flush, the host->device transfer); the engine's own bound is the same (session_phase_limit_s).
SESSION_WAIT_MAX_S = 3600
# ... except while a prompt is read: a PP line comes once per chunk (up to 32768 tokens with --prefill auto, issue
# #282), and the slowest PCs read ~100 tok/s, so a first chunk can take minutes before the first line.  Until the first
# PP the wait adds the chunk's tokens at PP_FLOOR_TOK_S; after one, a chunk may take PP_SLACK x the last one's time.
PP_CHUNK_MAX = 32768
PP_FLOOR_TOK_S = 50.0
PP_SLACK = 3.0


def timeout_env(name: str, default: float) -> float | None:
    """Reads a timeout in seconds from the environment variable `name`; None (no limit) when it is 0 or less."""
    try:
        v = float(os.environ.get(name, default))
    except ValueError:
        v = default
    return None if v <= 0 else v


# #1317 part 2: the quick watchdog for a FROZEN engine.  engine_silence_s (above) is generous on purpose: a slow PC reads a
# prompt chunk for minutes without a line.  But an engine that prints nothing for ENGINE_STALL_S and also uses no CPU time and
# moves no bytes of I/O in that time is not slow, it is idle or frozen (a deadlock in a CUDA call, a driver stall, a lost
# step), and waiting longer cannot help: it is ended, and the next request starts it again.  An engine that is silent but
# still working is never ended by this, only reported once (the recommendation: engine_silence_s, or this variable).
# STRATA_ENGINE_STALL_S overrides it (0 = off).  Needs psutil (setup installs it); without it nothing is ended.
ENGINE_STALL_S = timeout_env("STRATA_ENGINE_STALL_S", 90.0)
PAR_COUNTERS = os.environ.get("STRATA_PAR_COUNTERS") == "1"   # research counters (queue wait for a slot), shown in /v1/status
# --batch: new requests take the control lines (one prompt read at a time) short prompts first: a prompt of at most
# STRATA_ADMIT_SHORT_FIRST tokens goes before longer ones (opt-in; unset or 0: whoever gets the lock first), and a
# longer one that waited STRATA_ADMIT_MAX_WAIT_S (default 30) counts as short.  A chat question then waits for the one
# read in progress, not for every agent read that came before it.
try:
    ADMIT_SHORT_FIRST = max(0, int(os.environ.get("STRATA_ADMIT_SHORT_FIRST") or 0))
except ValueError:
    ADMIT_SHORT_FIRST = 0
ADMIT_MAX_WAIT_S = timeout_env("STRATA_ADMIT_MAX_WAIT_S", 30.0) or 0.0
# STRATA_COALESCE_PREFILL=N (research, PAR-H; needs the engine's BCO / BCOK lines): a request that takes the control lines
# for its admission also sends the prompts of up to N-1 requests still waiting for them (BCO <slot> <ids>, into free slots
# it reserves for them); the engine reads all the prompts in ONE layer-major pass, so the experts a layer streams are
# read once for all of them, and each waiting request then finds its prompt in its slot.  0 / unset: off.
COALESCE_PREFILL = max(0, int(os.environ.get("STRATA_COALESCE_PREFILL") or 0))
COALESCE_TOKENS = max(256, int(os.environ.get("STRATA_COALESCE_TOKENS") or 16384))   # prompt tokens read together at most
COALESCE_WAIT_MS = max(0, int(os.environ.get("STRATA_COALESCE_WAIT_MS") or 0))       # a burst's later requests register within this long
# Adaptive: a read is shared (and the leader waits for its followers) only when this engine has seen at least
# COALESCE_MIN_REQS requests arrive within COALESCE_BURST_MS (the leader's own included).  Fewer than that - one or two
# requests per engine, as 8 requests on 4 replicas - read plainly, with no wait: nothing to share, and the wait and the
# longer pass only cost (measured 4 x 1 replicas at 8 requests: 168 against 196 tok/s with an unconditional wait).
COALESCE_MIN_REQS = max(2, int(os.environ.get("STRATA_COALESCE_MIN_REQS") or 3))
COALESCE_BURST_MS = max(10, int(os.environ.get("STRATA_COALESCE_BURST_MS") or 250))
STALL_CPU_EPS_S = 0.5       # CPU seconds the engine may use in the whole stall window and still count as frozen
STALL_IO_EPS_B = 1 << 20    # bytes read or written in that window


def engine_frozen(base: tuple[float, int], now: tuple[float, int]) -> bool:
    """#1317: True when two (cpu seconds, io bytes) samples of the engine show no work between them."""
    return (now[0] - base[0]) <= STALL_CPU_EPS_S and (now[1] - base[1]) <= STALL_IO_EPS_B


# #1317: how long a start may take before the engine says READY; a blocking read of that line that never returned
# held the request turn for good.  Generous on purpose: a cold model load or a slow disk is not a hang.  The config's
# "engine_ready_s" or STRATA_ENGINE_READY_S overrides it (0 = wait for ever).
ENGINE_READY_S = timeout_env("STRATA_ENGINE_READY_S", 900.0)


class MockEngine:
    """Replays a scripted completion (text) as token ids, one per step, then the end-of-turn token.  Given a list of
    scripts, each request gets the next one and the last one repeats (a tool call, then the answer after it)."""

    def __init__(self, tokenizer, script: str | list[str], max_context: int = 32768, delay_s: float = 0.0,
                 cycle: bool = False):
        self.tok, self.max_context, self.delay = tokenizer, max_context, delay_s
        end = tokenizer.encode(IM_END, parse_special=True)
        self.scripts = [tokenizer.encode(x, parse_special=True) + end for x in ([script] if isinstance(script, str)
                                                                                 else script)]
        self.script, self.turns = self.scripts[0], 0
        # `cycle` picks the script by request position instead of advancing to the last one and staying
        # there. A harness adds requests the script author did not plan for -- a session title, a re-ask
        # after a tool result -- and with the default behaviour one extra request shifts every later
        # reply, so a run looks like a model that simply chose differently.
        self.cycle = bool(cycle)
        self.last_prompt: list[int] = []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.last_prompt = list(ids)
        self.last_embeddings = embeddings
        if len(self.scripts) > 1:
            if self.cycle:
                self.script = self.scripts[self.turns % len(self.scripts)]
            else:
                self.script = self.scripts[min(self.turns, len(self.scripts) - 1)]
            self.turns += 1
        for t in self.script[:max_new]:
            if cancel.is_set():
                return
            if self.delay:
                time.sleep(self.delay)
            yield t


class EngineDied(RuntimeError):
    """The engine process ended in the middle of a request (issue #27: on Linux, the out-of-memory killer)."""
    restarts = True        # the engine is gone, so the next request starts it again; False for an error raised while the engine is left running (#3961)


def died_note(e: Exception) -> str:
    """What an API error says about an EngineDied: the cause, and that the next request restarts the engine - only where it really is ended."""
    return f"{e}; the next request restarts it" if getattr(e, "restarts", True) else str(e)


class QueueWaitExpired(RuntimeError):
    """An inference call waited too long for its first engine turn."""


class EngineSilent(EngineDied):
    """#481: the engine said nothing for too long during a request (or never acknowledged a STOP): the two sides lost
    step - the engine waiting for its next command, the server for this request's end - and the server ended it.  An
    EngineDied, so the request ends with an error and the next one starts the engine again."""


class EngineNonFinite(ValueError):
    """#879: the engine's DONE (or a slot's BDONE) said `nonfinite`: a window's logits held a NaN or an infinity.  The
    engine wrote none of that window's tokens and drops every cached state that may carry the poison, so its next
    request reads the prompt from token 0.  Service.run sends a request that had produced nothing yet once more, and
    ends one that had started (as "length").  A ValueError, so a stream that has started reports it as an error event."""


class SessionRefused(ValueError):
    """Disk sessions: the engine refused or failed a SAVE / RESTORE and is still in step (`SERR <kind> <published>
    <reason>`).  `kind` is the engine's category - invalid (400), storage (507), memory (503), io (500); `published`:
    a save whose new file already replaced the old one before a later step (the folder flush) failed."""

    STATUS = {"invalid": 400, "storage": 507, "memory": 503, "io": 500}

    def __init__(self, kind: str, message: str, published: bool = False):
        super().__init__(message)
        self.kind = kind if kind in self.STATUS else "io"
        self.published = published

    @property
    def status(self) -> int:
        return self.STATUS[self.kind]


class SettingsRefused(ValueError):
    """The engine refused a CHANGE_SETTINGS (`ERR <why>`, e.g. a GPU pool too small for the biggest active
    conversation) and changed none of it; the message is its why."""


class EngineStuck(RuntimeError):
    """The engine process did not end after QUIT, terminate and kill: the server keeps it (and says so) rather than
    reporting its GPU and RAM as given back."""


class GpuBusy(RuntimeError):
    """The model is unloaded and the GPU has less free VRAM than min_free_vram_mib: something else (a game, another
    model server) is using it, so the engine is not started into the little that is left.  `retry_after`: seconds
    for the Retry-After header of the 503."""

    def __init__(self, message="", retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


ENGINE_REQUEST = re.compile(
    # "+ 12288 of 98179 read": a request cancelled while its prompt was read (#471)
    r"prompt (?P<prompt>\d+) tokens = (?P<reused>\d+) reused \+ \d+(?: of \d+)? read in (?P<read>[\d.]+) ms "
    r"\((?P<pp>[\d.]+) tok/s\), "
    r"(?P<gen>\d+) generated in (?P<gen_ms>[\d.]+) ms \((?P<tg>[\d.]+) tok/s\)")


_echoing: set[str] = set()      # the logs echo_requests already follows (restart() runs StrataEngine.__init__ again)

DRAFT_HEAD_FAIL = "the draft head does not fit"
DRAFT_HEAD_HINT = ("a smaller draft vocabulary needs less VRAM: --draft-vocab cyrillic (English, code and the Cyrillic "
                   "script) or --draft-vocab en (English and code, ~215 MiB less than the default). Start once with "
                   "it - START-HERE.bat --draft-vocab en (Windows) or ./setup.sh --draft-vocab en - and the model "
                   "keeps it; or a smaller --context in setup.")


def start_failure_hint(log: str | None, offset: int) -> str:
    """#474: what to change when the engine stopped at the start because the MTP draft head did not fit the VRAM
    left: the engine's own `strata mtp:` lines after that failure (0.1.36+: what it needs, what is free, the smaller
    subsets), else the same advice in words for an older engine.  "" for any other failure: the log says why."""
    if not log:
        return ""
    try:
        with open(log, "rb") as f:
            f.seek(offset)
            text = f.read().decode("utf-8", "replace")
    except (OSError, ValueError):
        return ""
    if DRAFT_HEAD_FAIL not in text:
        return ""
    said = [x.strip()[len("strata mtp: "):] for x in text.splitlines()
            if x.strip().startswith("strata mtp: ") and ("draft head over" in x or "hint:" in x)]
    return ". mtp: " + DRAFT_HEAD_FAIL + ". " + (" ".join(said) if said else "Hint: " + DRAFT_HEAD_HINT)


def start_log_text(log: str | None, offset: int, limit: int = 1 << 20) -> str:
    """Everything this engine start has written to the log so far (at most `limit` bytes), "" when there is no log."""
    if not log:
        return ""
    try:
        with open(log, "rb") as f:
            f.seek(max(0, offset))
            return f.read(limit).decode("utf-8", "replace")
    except (OSError, ValueError):
        return ""


def start_log_tail(log: str | None, offset: int, n: int = 20) -> str:
    """#496: the last n lines this start wrote to the engine log, for the error when the engine ended before READY -
    whatever the failure, the engine's own reason is in them (people posted the traceback without the log).  "" when
    there is no log or nothing in it from this start."""
    if not log:
        return ""
    try:
        with open(log, "rb") as f:
            f.seek(0, 2)
            start = max(offset, f.tell() - 64 * 1024)   # enough for 20 lines, never an earlier start's
            f.seek(start)
            text = f.read().decode("utf-8", "replace")
    except (OSError, ValueError):
        return ""
    lines = text.splitlines()[1 if start > offset else 0:]   # not a line cut in half
    lines = [x.rstrip() for x in lines if x.strip()][-n:]
    if not lines:
        return ""
    return "\nthe engine log's last lines:\n" + "\n".join("  " + x for x in lines)


# Windows NTSTATUS exit codes of a process the system ended (a crash, not an error the engine reported)
_WIN_CRASH = {0xC000001D: "illegal instruction (the CPU lacks an instruction the program uses)",
              0xC0000005: "access violation", 0xC0000409: "stack buffer overrun / fast-fail",
              0xC0000135: "a DLL the program needs was not found", 0xC0000139: "entry point not found in a DLL",
              0xC0000142: "a DLL failed to initialize", 0xC000007B: "bad image (32-bit / 64-bit mix-up)",
              0xC00000FD: "stack overflow", 0xC0000094: "integer division by zero", 0xC0000096: "privileged instruction"}


def engine_exit_words(rc: int | None) -> str:
    """What a process's exit status says, in words: "killed by signal 4 (SIGILL, Illegal instruction)" (a negative status
    on Linux / macOS), a Windows NTSTATUS crash code, or "exit code N".  "" when there is no status."""
    if rc is None:
        return ""
    if rc < 0 and os.name != "nt":
        try:
            sig = signal.Signals(-rc)
            return f"killed by signal {-rc} ({sig.name}, {signal.strsignal(sig) or sig.name})"
        except (ValueError, AttributeError):
            return f"killed by signal {-rc}"
    code = rc & 0xFFFFFFFF
    if code in _WIN_CRASH:
        return f"exit code 0x{code:08X}: {_WIN_CRASH[code]}"
    if 0xC0000000 <= code < 0xD0000000:
        return f"exit code 0x{code:08X} (a system error status)"
    return f"exit code {rc}"


def engine_crashed(rc: int | None) -> bool:
    """True when the engine process was ended by the system (an illegal instruction, a bad memory access, the
    out-of-memory killer, a missing DLL ...) rather than leaving by itself: nothing a second try fixes."""
    if rc is None:
        return False
    if rc < 0 and os.name != "nt":
        return -rc in {int(getattr(signal, n)) for n in ("SIGILL", "SIGSEGV", "SIGBUS", "SIGFPE", "SIGKILL")
                       if hasattr(signal, n)}
    return 0xC0000000 <= (rc & 0xFFFFFFFF) < 0xD0000000      # an NTSTATUS error code


def cpu_isa_words() -> tuple:
    """(text, missing): the CPU's model and which of AVX / AVX2 / FMA / F16C it reports (Linux: /proc/cpuinfo), and the
    ones of those it lacks.  ("", []) where it cannot be read."""
    try:
        text = Path("/proc/cpuinfo").read_text(errors="replace")
    except OSError:
        return "", []
    model = next((x.split(":", 1)[1].strip() for x in text.splitlines() if x.lower().startswith("model name")), "")
    flags = next((x.split(":", 1)[1].split() for x in text.splitlines() if x.lower().startswith("flags")), [])
    have = ", ".join(f"{f} {'yes' if f in flags else 'NO'}" for f in ("avx", "avx2", "fma", "f16c", "avx512f"))
    missing = [f for f in ("avx", "avx2", "fma", "f16c") if f not in flags]
    return f"{model or 'this CPU'} ({have})", missing


def start_exit_note(rc: int | None, exe=None) -> str:
    """The sentence the start error carries about how the engine process ended (the engine log is silent when the
    system ends it): "" without a status, else a line, with the CPU's AVX flags after an illegal instruction."""
    words = engine_exit_words(rc)
    if not words:
        return ""
    note = f"\nthe engine process ended: {words}"
    try:                                            # 0xC0000135: say which DLL (the loader prints nothing)
        from strata_common.loader_diag import missing_note
        note += missing_note(rc, exe)
    except Exception:
        pass
    if rc is not None and ((rc < 0 and os.name != "nt" and -rc == int(getattr(signal, "SIGILL", 4))) or
                           (rc & 0xFFFFFFFF) == 0xC000001D):
        isa, missing = cpu_isa_words()
        note += "\n  An illegal instruction: the engine ran an instruction this CPU cannot execute"
        if isa and missing:
            note += f" ({', '.join(m.upper() for m in missing)} missing; {isa})"
        elif isa:
            note += f"; the CPU reports AVX, AVX2, FMA and F16C ({isa}), so this is a bug in the engine"
        note += ".  Please report it with the CPU model (github.com/Niko1221/Strata/issues)."
    return note


def start_log_marks(log: str | None, offset: int) -> list:
    """The GPU runtime failure texts (TRANSIENT_START_MARKS) this start wrote to the log: what lets a retry notice say the
    GPU runtime failed.  [] when none, or no log."""
    if not log:
        return []
    try:
        with open(log, "rb") as f:
            f.seek(offset)
            text = f.read(64 * 1024).decode("utf-8", "replace")
    except (OSError, ValueError):
        return []
    return [m for m in TRANSIENT_START_MARKS if m in text]


# A start that ends at once with the GPU runtime's own failure and none of the engine's lines: on ROCm, a start right
# after the previous engine exited can fail inside the runtime's device/queue set-up while the old process's queues are
# still being torn down ("terminate called after throwing an instance of 'rocr::AMD::hsa_exception'", or an exit with
# nothing in the log).  Such a start is tried again after a short wait (STRATA_START_RETRIES, default 3; 0: never).
TRANSIENT_START_MARKS = ("hsa_exception", "HSA_STATUS_ERROR", "hsaKmt")
START_RETRIES = max(0, int(os.environ.get("STRATA_START_RETRIES", "3") or 0))


def transient_start_failure(log: str | None, offset: int, elapsed_s: float, rc: int | None = None) -> bool:
    """True when the start that wrote the log from `offset` failed the transient way above: within 30 s, and its lines
    are only the launcher's own "engine started" line plus (optionally) a GPU runtime failure.  An exit with no line
    at all counts only when it was not the engine's own error exit (1: it said why on stdout)."""
    if not log or elapsed_s > 30.0:
        return False
    try:
        with open(log, "rb") as f:
            f.seek(offset)
            text = f.read(64 * 1024).decode("utf-8", "replace")
    except (OSError, ValueError):
        return False
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    engine_lines = [x for x in lines if not x.startswith("[strata]")]
    if any(m in text for m in TRANSIENT_START_MARKS):
        return all(x.startswith("terminate called") or x.startswith("what()") or
                   any(m in x for m in TRANSIENT_START_MARKS) for x in engine_lines)
    # it exited without writing a single line of its own - but not when the system ended it (an illegal instruction, a
    # bad memory access, the OOM killer, a missing DLL): that is a crash a second try repeats, and the retry notice
    # must not call it a GPU runtime start (#2324: an old CPU's illegal instruction was retried 3 times as "GPU
    # runtime" and the real reason never shown)
    return not engine_lines and rc != 1 and not engine_crashed(rc)


def echo_requests(log_path: str, offset: int) -> None:
    """STRATA_REQUEST_LINES=1: one stdout line per finished request, from the engine's own summary in its log.

    The engine's stderr goes to the log file (the start narrator reads it), so a supervisor that only sees this
    process's output - a tray, llama-swap - has no per-request numbers. This re-states the engine's line with the
    total the two times make: `request prompt P cached C output O prompt_read R ms total S ms prefill X tok/s decode Y
    tok/s` (prompt_read: the time the engine spent reading the prompt's new tokens, not a time to first token).
    """
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        f.seek(offset)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.2)
                continue
            m = ENGINE_REQUEST.search(line)
            if m:
                read_ms, gen_ms = float(m["read"]), float(m["gen_ms"])
                print("[strata] request prompt %s cached %s output %s prompt_read %.0f ms total %.0f ms prefill %s "
                      "tok/s decode %s tok/s" % (m["prompt"], m["reused"], m["gen"], read_ms, read_ms + gen_ms, m["pp"],
                                                 m["tg"]), flush=True)


def experts_loading_words(args: list, size: str) -> str:
    """#505: what the start does with the experts, by the engine's flags (generate.cpp's option parsing): a RAM budget
    copies the hottest N GiB into RAM (--resident-budget-gib), the resident low-RAM mode the ones the GPU does not hold
    (--resident-experts), plain --mmap-experts reads them from the model files through the OS file cache (nothing is
    loaded into RAM up front); otherwise all of them go into RAM."""
    if "--resident-budget-gib" in args:
        try:
            n = f"up to {float(args[args.index('--resident-budget-gib') + 1]):g} GiB"
        except (IndexError, ValueError):
            n = "a RAM budget"
        return (f"loading the most-used experts into RAM ({n}; the rest are read from the model files as needed) "
                "and locking part of them for the GPU.")
    if "--resident-experts" in args:
        return (f"loading the experts the GPU does not hold into RAM (of {size}) and locking part of them for the "
                "GPU.")
    if "--mmap-experts" in args:
        return (f"mapping the experts from the model files ({size}, --mmap-experts): they are not loaded into RAM - "
                "the OS file cache reads them as the GPU's expert cache fills and as requests need them.")
    return f"loading the experts into RAM ({size}) and locking part of them for the GPU."


OPENS_BROWSER = False        # set in main() when the server was started with --open: only then may a message say a browser opens


# STRATA_DS4_RESTORE_POINTS (4): the DeepSeek engine keeps restore points where a later prompt is likely to leave this
# one - its GEN line carries marks=<pos>,..., tail=<pos> and shared=<pos> - besides the end of the last prompt, the
# position before it and the anchor it keeps anyway.  N (1-16) turns them on, and is also the engine's cap a
# conversation (it reads the same setting); 0: no keys, every GEN line as before.  STRATA_DS4_RESTORE_TAIL (512): how
# far before the prompt's end the tail point sits (0: none).  (Project Maya #639, restore points.)
RESTORE_LEAST = 1024        # no point before this many tokens: such a start is read again in about the time a point costs


def restore_point_settings(env=None) -> tuple[bool, int]:
    """(on, tail) from STRATA_DS4_RESTORE_POINTS and STRATA_DS4_RESTORE_TAIL."""
    env = os.environ if env is None else env

    def whole(name, default):
        try:
            return max(0, int(str(env.get(name, default)).strip()))
        except ValueError:
            return default
    return whole("STRATA_DS4_RESTORE_POINTS", 4) > 0, whole("STRATA_DS4_RESTORE_TAIL", 512)


def deepseek_restore_points(ids, user_id, system_id=None, tail=512, least=RESTORE_LEAST) -> dict:
    """Where the DeepSeek engine keeps restore points for this prompt -> {"marks": [pos, ...], "tail": pos, "shared":
    pos} (any may be missing; {}: none).  From the prompt's ids as the template wrote them: a turn opens with its role's
    special token (<｜User｜>, a tool result's turn too; <｜System｜> for a system message after the first user turn), and
    text inside a message never encodes to one, so the positions are the template's own.

    - the start of the last user turn: its last message edited, or sent again after another answer, goes back there;
    - "tail": `tail` tokens before the end, when the last turn is longer than that: a new question after the same long
      document in the same message;
    - "shared": the end of the system prompt and its tools (the first user turn): another conversation behind the same
      start goes back there, and the engine gives that point up last.
    Positions under `least` are left out: a start that short is read again in about the time a point costs."""
    if user_id is None or len(ids) < 2:
        return {}
    first = last = None
    for i, t in enumerate(ids):
        if t == user_id:
            first = i if first is None else first
            last = i
        elif t == system_id and first is not None:
            last = i
    marks, out = [], {}
    if first is not None and first >= least:
        out["shared"] = first
    if last is not None and last >= least and last != first:
        marks.append(last)
    t = len(ids) - tail
    if tail > 0 and t >= least and (last is None or t > last):
        out["tail"] = t
    if marks:
        out["marks"] = sorted(marks)
    return out


def narrate_start(log_path: str, offset: int, args: list, done: threading.Event, heartbeat=20.0) -> None:
    """While the engine starts, say in the server window what it is doing, from its log: the start reads tens of GB
    into RAM and locks part of it for the GPU, and on many PCs everything is slow or frozen for a minute or more -
    people closed the window thinking it had hung.  The warning comes at that step, not after it."""
    gb = 0.0
    if "--native" in args:                              # about the size of the experts it will read
        try:
            gb = os.path.getsize(args[args.index("--native") + 1]) / 1e9
        except (OSError, IndexError):
            pass
    size = f"about {gb:.0f} GB" if gb >= 1 else "tens of GB"
    loading = experts_loading_words(args, size)
    t0 = last = time.time()
    said = set()

    def say(key, text):
        nonlocal last
        if key not in said:
            said.add(key)
            last = time.time()
            print(text, flush=True)

    say("weights", "[strata] starting the engine: reading the model's weights ...")
    pos = offset
    while not done.wait(0.5):
        try:
            with open(log_path, "rb") as f:
                f.seek(pos)
                chunk = f.read()
        except OSError:
            chunk = b""
        if chunk.count(b"\n"):
            cut = chunk.rfind(b"\n") + 1
            pos += cut
            for line in chunk[:cut].decode("utf-8", "replace").splitlines():
                if "PLE on" in line or "expert arena:" in line or "experts via mmap" in line:   # #505: mapped
                    say("arena", f"[strata] {loading}\n"
                                 "         YOUR PC CAN BE SLOW OR STOP RESPONDING FOR 1-3 MINUTES NOW - this is normal.\n"
                                 "         Please wait and don't close this window"
                                 + ("; the browser opens when it is ready." if OPENS_BROWSER else "; the server says when it is ready."))
                elif " loaded " in line and "GiB at" in line:
                    say("loaded", "[strata] experts loaded: " + line.split(" loaded ", 1)[1].strip() +
                        f" ({time.time() - t0:.0f} s so far)")
                elif " the page pool of " in line and " GiB (of " in line:   # the paged engines' expert store
                    gib = line.split("): ", 1)[-1].split(" (of ", 1)[0]
                    say("cache", f"[strata] filling the GPU's page pool for experts and conversations ({gib}) ...")
                elif "expert cache " in line and " slots, " in line and "auto" not in line:
                    n = line.split("expert cache ", 1)[1].split(";")[0].replace(" slots,", " experts,").strip()
                    say("cache", f"[strata] filling the GPU's expert cache ({n}) ...")
                elif "session is up" in line:
                    say("up", "[strata] almost ready ...")
        if time.time() - last > heartbeat:
            last = time.time()
            print(f"[strata] still starting ({time.time() - t0:.0f} s) - please wait ...", flush=True)


_BTRACE = bool(os.environ.get("STRATA_BATCH_TRACE"))


def btrace(*a):
    if _BTRACE:
        print("[batch-trace]", threading.get_ident() % 10000, *a, file=sys.stderr, flush=True)


EOS_IDS = {248044, 248046}   # <|endoftext|>, <|im_end|>: the engine's default --eos-ids


def get_request_priority(sampling: dict) -> int:
    """Gets a request's "priority" field (a whole number; higher goes first, and on an engine that can preempt it takes
    a busy slot from a request of lower priority), 0 when it has none."""
    p = sampling.get("priority")
    return p if isinstance(p, int) and not isinstance(p, bool) else 0


# The engine's own words for a verifier that cannot go on (its GPU waits were released, or an earlier window never
# finished, #267): the process may stay alive and even print DONE, but it must not take another request.
FATAL_PREFIXES = ("ERR verify: timed out at layer ", "ERR verify batch: timed out at layer ",
                  "ERR verify: an earlier window never finished", "ERR verify batch: an earlier window never finished")


def parse_engine_stage_line(line: str) -> list[dict] | None:
    """Accept only the engine's final, contiguous stage plan (inclusive layer ends)."""
    prefix = "strata serve: layer split: layers "
    suffix = ", one hand-off per window"
    if not line.startswith(prefix) or not line.rstrip("\r\n").endswith(suffix):
        return None
    body = line.rstrip("\r\n")[len(prefix):-len(suffix)]
    parts = body.split(", ")
    if not 2 <= len(parts) <= 16:
        return None
    stages, next_layer = [], 0
    for i, part in enumerate(parts):
        match = re.fullmatch(r"([0-9]{1,5})-([0-9]{1,5}) \(CUDA([0-9]{1,2})\)", part)
        if match is None:
            return None
        first, last, ordinal = map(int, match.groups())
        if first != next_layer or last < first or last > 65535 or ordinal > 15:
            return None
        stages.append({"stage": i, "layer_start": first, "layer_end": last,
                       "engine_ordinal": ordinal})
        next_layer = last + 1
    return stages


def read_engine_stage_log(path: str | None, start: int, limit: int = 2 << 20) -> list[dict]:
    """Read this engine start only; oversized/unavailable logs leave placement unknown."""
    if not path or start < 0:
        return []
    try:
        with open(path, "rb") as source:
            source.seek(start)
            data = source.read(limit + 1)
    except OSError:
        return []
    if len(data) > limit:
        return []
    for line in reversed(data.decode("utf-8", errors="replace").splitlines()):
        stages = parse_engine_stage_line(line)
        if stages is not None:
            return stages
    return []


def parse_batch_window_sample(line: str) -> dict | None:
    """Read an opt-in engine BSTAT sample; rows are verifier work, not emitted tokens."""
    if len(line) > 80:
        return None
    match = re.fullmatch(r"BSTAT ([1-9][0-9]{0,8}) ([1-9][0-9]{0,9}) "
                         r"([1-9][0-9]{0,9}) ([1-9][0-9]{0,9})\r?\n?", line)
    if match is None:
        return None
    windows, rows, active, elapsed_ms = map(int, match.groups())
    if rows > windows * 8 or active > rows or elapsed_ms > 86_400_000:
        return None
    return {"windows": windows, "processed_rows": rows, "active_rows": active,
            "elapsed_ms": elapsed_ms}


def parse_stage_in_flight_sample(line: str) -> dict | None:
    """Read bounded PSTAT host scheduling spans; these are not GPU utilization."""
    if len(line) > 256 or re.fullmatch(
            r"PSTAT (?:[2-9]|1[0-6]) [1-9][0-9]{0,10}(?: (?:0|[1-9][0-9]{0,10})){2,16}\r?\n?", line) is None:
        return None
    parts = line.strip().split(" ")
    stages, elapsed_us = int(parts[1]), int(parts[2])
    in_flight_us = [int(value) for value in parts[3:]]
    if len(in_flight_us) != stages or elapsed_us > 86_400_000_000 or \
            any(value > elapsed_us for value in in_flight_us):
        return None
    return {"elapsed_us": elapsed_us, "in_flight_us": in_flight_us}


def parse_conversation_line(f: list[str]) -> dict:
    """Gets a CONVERSATION line's state, optional last-token Unix time and token preview
    (LIST_CONVERSATIONS', and GET_MEMORY_USE's from strata-q38) as StrataEngine.list_conversations gives them."""
    fields = dict(v.split("=", 1) for v in f[8:] if "=" in v)
    preview = next((v for v in f[8:] if "=" not in v), "")
    return {"id": f[1], "place": f[2], "state": f[3], "count": int(f[4]), "capacity": int(f[5]), "bytes": int(f[6]),
            "used": int(f[7]), "preview": [int(x) for x in preview.split(",")] if preview else [],
            **({"last_tokens_at": int(fields["last_tokens_at"])} if "last_tokens_at" in fields else {})}


#: What a request may ask for that the all-cards split engine cannot do (its GEN parser has no such keys): answered with a 400 before anything is sent, not with an error in
#: the middle of a reply. logit_bias is refused the same way by logit_bias.validate_request (the split engine does not report logit_bias=1).
SPLIT_REFUSED_FEATURES = ("logprobs", "logit_bias")


def refuse_for_split(engine, req: dict) -> None:
    if not getattr(engine, "split_engine", False):
        return
    if top_logprobs(req) is not None:
        raise ValueError("logprobs: not supported by the all-cards split engine; start with --split-engine off")


def top_logprobs(req: dict):
    """The K for the engine's logprobs=K, or None when the request did not ask.

    OpenAI: "logprobs": true with "top_logprobs" 0..20; the legacy completions form,
    "logprobs": N, is taken as N."""
    lp = req.get("logprobs")
    if lp is True:
        k = req.get("top_logprobs")
        return max(0, min(20, k)) if isinstance(k, int) and not isinstance(k, bool) else 0
    if isinstance(lp, int) and not isinstance(lp, bool) and lp >= 0:
        return min(20, lp)
    return None


def allowed_ids(req: dict) -> list:
    """`allowed_token_ids`: the ids a caller named, or [] (vLLM's semantics, see LogprobEntries in strata_server/server.py)."""
    ids = req.get("allowed_token_ids")
    if not isinstance(ids, (list, tuple)):
        return []
    return [x for x in ids if isinstance(x, int) and not isinstance(x, bool) and x >= 0]


NO_LINE = object()                                   # StrataEngine.generate: no line held after a token


class LPLines(collections.deque):
    """One request's LP lines, tagged with its cancel Event (one per request, handed unchanged from the handler
    through Service.run to Engine.generate).  The last events of a request - the parser's and the stop strings'
    tail - come after it has left the queue, when the next request may already have replaced the engine's lp_lines
    with its own: LogprobEntries reads only the lines tagged with its own request's Event."""

    def __init__(self, owner=None):
        super().__init__()
        self.owner = owner


class _RequestOwned:
    """#4451: a per-request control field of a --batch engine.  With batch slots several requests run on their own
    threads, and the control lines serve one at a time; a field every request writes (`_ctl_mode`, `_ctl_result`,
    `_yielded`, `_last_done`, `_ctl_erred`) must not be seen by another request that gets the lines next.  So each
    thread has its own copy; one request at a time, it is the one attribute as before."""

    def __init__(self, default=None):
        self.default = default

    def __set_name__(self, owner, name):
        self.name = name

    def _store(self, obj):
        d = obj.__dict__
        if d.get("batch"):
            return d.setdefault("_own_tl", threading.local()).__dict__
        return d.setdefault("_own_plain", {})

    def __get__(self, obj, owner=None):
        if obj is None:
            return self
        return self._store(obj).get(self.name, self.default)

    def __set__(self, obj, value):
        self._store(obj)[self.name] = value


class StrataEngine:
    """Starts a resident engine and routes generation events and command replies by identity.

    Engines advertising `requests` use RUN/CANCEL and ASK envelopes. Released engines without that capability
    retain the legacy client path so updating the Python server does not require replacing every installed engine.
    """
    silence_s = ENGINE_SILENCE_S         # #481: main() sets the config's engine_silence_s (survives restart())
    silent_note = None                   # why the server ended the engine itself (#481, close()): death_note says it
    gpu_busy = None                      # #1317: () -> bool, "the GPU is working" (the telemetry's reading); a GPU at work is not frozen
    last_err = None                      # #997: an ERR that was its last stdout line (death_note says it)
    batch = 0                            # --batch: the engine's batch slots (0: one request at a time)
    _ctl_mode = _RequestOwned()          # the request that has the control lines: solo / batch (see _RequestOwned)
    _ctl_result = _RequestOwned()
    _yielded = _RequestOwned()
    _last_done = _RequestOwned()
    _ctl_erred = _RequestOwned(False)
    _ctl_held = _RequestOwned(False)     # this thread already holds the control lines (Service.vram takes them itself)
    CONTROL_WAIT_S = 300.0               # how long vram() / session_file() wait for the control lines of a batch engine
    DRAIN_TIMEOUT_S = 300.0              # how long an interrupted request waits, with no line from its engine, for DONE / BADM
    QUARANTINE_S = 600.0                 # ... and how long a slot or the control lines stay held after that with no line
    multi = 0                            # DeepSeek --parallel (READY ... multi=N): conversations the engine holds at once (batch = N too)

    # `last`: the figures of the engine's last DONE line.  With batch slots several requests run at once, each on its
    # own thread, so each one reads its own request's figures (Service.run records them); one at a time, the one dict.
    @property
    def last(self):
        if self.batch:
            tl = self.__dict__.setdefault("_tl", threading.local())
            if not hasattr(tl, "last"):
                tl.last = {}
            return tl.last
        return self.__dict__.get("_last", {})

    @last.setter
    def last(self, value):
        self.__dict__["_last"] = value
        if self.batch:
            tl = self.__dict__.setdefault("_tl", threading.local())
            tl.last = value

    def __init__(self, exe: str, args: list[str], cwd: str | None = None, log: str | None = None,
                 env: dict | None = None, lazy: bool = False, cpus: list[int] | None = None, _attempt: int = 0,
                 retry_args: list[str] | None = None, retry_vram_args: list[str] | None = None):
        # retry_args: the arguments to start again with, once, when this start ends before the engine is ready (an Auto start that picked DFlash2: mtp_retry_args)
        # retry_vram_args: the same start with no drafter (nodraft_retry_args), used instead when the engine ended because a card ran out of memory: swapping DFlash2 for
        # the MTP takes MORE memory from a card that is already short, so that direction is never taken then
        self.spawn = (exe, list(args), cwd, log, env, False, cpus)   # to start it again after it died (issue #27)
        # the all-cards split engine (sFNX, strata-split-N[-arch]): its GEN parser accepts fewer keys than Strata's engine (tests/split/test_split_gen_keys.py pins them)
        self.split_engine = os.path.basename(str(exe)).startswith("strata-split")
        self.session_save_reclaim = '--session-save-reclaim' in args
        paths = {k: v for k, v in zip(args, args[1:]) if k in ("--native", "--pack")}
        self.model_path = paths.get("--native") or paths.get("--pack", "pack/full")
        # The engine excludes a temporary effort turn from its reusable history boundary.
        self._tail_role_token = int(args[args.index("--tail-role-token") + 1]) if "--tail-role-token" in args else -1
        self.log_path = log
        self.proc, self.pump, self.log = None, None, None
        self.ended, self.unloaded = True, True
        self.max_context = int(args[args.index("--max-context") + 1]) if "--max-context" in args else 4096
        self.request_protocol = False
        self.can_stop = False            # the engine honours a STOP line mid-request (READY <ctx> stop)
        self.can_list = False            # it answers LIST_CONVERSATIONS and GET_CONTEXT_WINDOWS (... conversations)
        self.can_report_memory = False   # it answers GET_MEMORY_USE (... memory)
        self.can_preempt = False         # a BGEN of a higher priority ends a busy slot's request (... preempt)
        self.can_change_settings = False  # it answers CHANGE_SETTINGS (... settings)
        self.last = {}
        self.info = {}                   # INFO key=value facts (engine 0.1.8+): kv, expert slots, ... (Monitor tab)
        self.terminal_stages: list[dict] = []  # final layer ranges from this engine start's bounded stderr log
        self.terminal_batch_sample: dict | None = None
        self.terminal_stage_sample: dict | None = None
        self.prefill_tok_s_mean = None
        self.progress = None             # (read, total) prompt tokens while a prompt is read, from PP lines
        self.progress_ms = 0             # PP's own milliseconds since the prompt started (its third field)
        self.reused = 0                  # RESUME: prompt tokens not read again (a client takes them out of the work)
        self.silent_note = None
        self.last_err = None
        try:                            # a ready-made engine's BUILD.json says its version
            self.info["version"] = json.loads((Path(exe).parent / "BUILD.json").read_text()).get("version")
        except (OSError, ValueError):
            self.info["version"] = None
        if lazy:
            return
        self.unloaded = False            # `ended` stays True until READY (below): not alive while starting (#344)
        self.log = open(log, "a", encoding="utf-8") if log else subprocess.DEVNULL
        loading = threading.Event()                     # set once READY: the narrator below stops
        log_start = os.path.getsize(log) if log else 0  # where this start's lines begin (start_failure_hint)
        self.log_start = log_start                      # #596: the Monitor's conversation cache reads from here
        if log:
            threading.Thread(target=narrate_start, args=(log, os.path.getsize(log), args, loading),
                             daemon=True).start()
            # once per log: the follower keeps reading the same (appended) log across restarts and reloads
            if os.environ.get("STRATA_REQUEST_LINES") and os.path.abspath(log) not in _echoing:
                _echoing.add(os.path.abspath(log))
                threading.Thread(target=echo_requests, args=(log, os.path.getsize(log)), daemon=True).start()
        if log:                                          # #779: when an engine started (a crash report has the time)
            try:
                self.log.write(f"[strata] {time.strftime('%Y-%m-%d %H:%M:%S')} engine started: {exe} --serve "
                               f"{' '.join(args)}\n")
                self.log.flush()
            except OSError:
                pass
        self.max_context = 0                             # before the new process is visible: never its predecessor's
        pin = {}
        if cpus and hasattr(os, "sched_setaffinity"):    # replicas: this engine's own physical cores (strata_server/replicas.py)
            pin["preexec_fn"] = lambda: os.sched_setaffinity(0, cpus)
        nice = engine_nice(env)                          # Linux with built-in Radeon graphics: the desktop keeps its share of the CPU
        if nice:
            say_nice = pin.get("preexec_fn")
            pin["preexec_fn"] = lambda: (say_nice() if say_nice else None, os.nice(nice))[0]
            print(f"[strata] engine priority: nice +{nice} (this PC has built-in Radeon graphics that usually draw the desktop; "
                  "STRATA_ENGINE_NICE=0 turns it off)", flush=True)
        end_leftover_engines(exe)                        # one an earlier run lost still holds the GPU and the memory
        spawned_at = time.monotonic()
        try:
            self.proc = popen("the Strata engine", [exe, "--serve", *args], cwd=cwd, stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, stderr=self.log, text=True, encoding="utf-8", errors="replace",
                              bufsize=1, env=env, **pin)    # (#5044: a byte that is not UTF-8 killed the reader thread)
        except OSError:                                  # nothing started: the narrator stops, the log handle goes back
            loading.set()
            if log:
                self.log.close()
            raise
        proc = self.proc                                 # THIS start's process: a close() from another thread may set self.proc to None meanwhile (#4792)
        # the READY read has a timeout (#1317): the lines come from a thread, up to READY, so the rest of the stream
        # stays for _pump (one reader at a time)
        ready_q: queue.Queue = queue.Queue()
        threading.Thread(target=self._ready_pump, args=(self.proc, ready_q), daemon=True).start()
        ready_deadline = None if ENGINE_READY_S is None else time.monotonic() + ENGINE_READY_S
        timed_out = False
        while True:
            left = None if ready_deadline is None else ready_deadline - time.monotonic()
            if left is not None and left <= 0:
                timed_out = True
                break
            try:
                line = ready_q.get(timeout=left)
            except queue.Empty:
                continue
            if line is None:
                break
            if line.startswith("INFO "):
                for kv in line.split()[1:]:
                    k, _, v = kv.partition("=")
                    self.info[k] = int(v) if v.lstrip("-").isdigit() else v
            if line.startswith("READY"):
                f = line.split()
                self.max_context = int(f[1])
                self.can_stop = "stop" in f[2:]
                self.can_list = "conversations" in f[2:]
                self.can_report_memory = "memory" in f[2:]
                self.can_preempt = "preempt" in f[2:]
                self.can_change_settings = "settings" in f[2:]
                self.request_protocol = "requests" in f[2:]
                self.multi = next((int(x[6:]) for x in f[2:] if x.startswith("multi=") and x[6:].isdigit()), 0)
                break
        loading.set()
        if self.max_context <= 0:
            if timed_out:                               # still loading, or hung: end it, so the turn is not held for good
                try:
                    proc.kill()
                except OSError:
                    pass
            try:                                        # its pipes and our handle on its log (the log stays)
                proc.wait(timeout=5)
                proc.stdin.close()
                proc.stdout.close()
                if log:
                    self.log.close()
            except (OSError, subprocess.TimeoutExpired):
                pass
            if not timed_out and _attempt < START_RETRIES and \
                    transient_start_failure(log, log_start, time.monotonic() - spawned_at, proc.returncode):
                wait_s = 2.0 * (2 ** _attempt)              # 2, 4, 8 s: the previous process's GPU queues go away
                exit_words = engine_exit_words(proc.returncode)
                if start_log_marks(log, log_start):         # the runtime said so itself (hsa_exception ...)
                    print(f"[strata] the engine stopped while the GPU runtime was starting (most often right after "
                          f"another engine exited, whose GPU queues were still being released"
                          f"{'; ' + exit_words if exit_words else ''}): starting it again in {wait_s:.0f} s "
                          f"(try {_attempt + 2} of {START_RETRIES + 1})", flush=True)
                else:                                       # an exit with nothing in the log: say only what is known
                    print(f"[strata] the engine exited before it wrote anything to its log"
                          f"{' (' + exit_words + ')' if exit_words else ''}; it may be a GPU runtime that was not "
                          f"ready yet: starting it again in {wait_s:.0f} s (try {_attempt + 2} of {START_RETRIES + 1})",
                          flush=True)
                time.sleep(wait_s)
                StrataEngine.__init__(self, exe, args, cwd, log, env, lazy, cpus, _attempt + 1, retry_args, retry_vram_args)
                return
            short_reason = None if timed_out else vram_fit.engine_stop_reason(start_log_text(log, log_start).splitlines())
            if short_reason and retry_vram_args and retry_vram_args != list(args):
                # a card ran out of memory (the engine's own reason is in its log): one more start with no drafter, in one quiet line
                print(f"[strata] {drafting.line('off', drafting.SHORT_OF_VRAM)}: starting again without a drafter ({short_reason})", flush=True)
                try:
                    with open(log, "a", encoding="utf-8") as handle:
                        handle.write(f"[strata] {time.strftime('%Y-%m-%d %H:%M:%S')} the engine ended before it was ready because a card was short of memory "
                                     f"(exit code {proc.returncode}); starting it again without a drafter\n")
                except (OSError, TypeError):
                    pass
                drafting.remember_note(drafting.SHORT_OF_VRAM)
                StrataEngine.__init__(self, exe, retry_vram_args, cwd, log, env, lazy, cpus, 0, None, None)
                return
            if not timed_out and not short_reason and retry_args and retry_args != list(args):
                # Auto picked DFlash2 and the engine ended before it was ready (an engine that does not take one of its flags, a drafter file it cannot read...):
                # one more start with the MTP, said in one quiet line. The person named DFlash2 themselves: no retry, the start fails with its reason below.
                print(f"[strata] {drafting.line('mtp', drafting.DIDNT_START)}: starting again with the MTP", flush=True)
                try:
                    with open(log, "a", encoding="utf-8") as handle:
                        handle.write(f"[strata] {time.strftime('%Y-%m-%d %H:%M:%S')} the engine ended before it was ready with the DFlash2 drafter "
                                     f"(exit code {proc.returncode}); starting it again with the MTP\n")
                except (OSError, TypeError):
                    pass
                drafting.remember_note(drafting.DIDNT_START)
                StrataEngine.__init__(self, exe, retry_args, cwd, log, env, lazy, cpus, 0, None, None)
                return
            why = (f"the engine did not report READY within {ENGINE_READY_S:.0f} s" if timed_out else
                   "the engine exited before it was ready")
            # EngineDied, not bare RuntimeError: a request waiting on this load must end with a 503
            # through the request path's handler - a RuntimeError there left the turn with no answer (#1527)
            note = "" if timed_out else start_exit_note(proc.returncode, exe)
            tail = start_log_tail(log, log_start)
            lines = [x for x in tail.split("\n")[2:] if x.strip()]
            if tail and lines and all(x.strip().startswith("[strata]") for x in lines):
                tail += "\n  (those are the launcher's own lines: the engine itself wrote nothing)"
            raise EngineDied(why + (f" (see {log})" if log else "") + note +
                             start_failure_hint(log, log_start) + tail)
        self.known_ctx = self.max_context   # survives a failed restart: requests keep their limit and restart it
        # (from PR #41, midhatn) a locally built engine can sit next to another release's BUILD.json: engines that
        # report their own version (INFO engine=, 0.1.8+) win, the manifest stays the fallback for older ones
        if self.info.get("engine"):
            self.info["version"] = str(self.info["engine"])
        self.terminal_stages = read_engine_stage_log(log, log_start) if log else []
        try:                                            # which drafter runs, and in plain words why not the one that was asked for
            self.info.update(drafting.engine_info(args, start_log_text(log, log_start), drafting.take_note()))
        except Exception:                               # noqa: BLE001 - a note must never stop a start
            pass
        self.ended = False                              # READY: alive from here (restart() set it True, #344)
        # the engine's stdout on a thread, so a request can wait with a timeout (heartbeats, cancel checks)
        self.gen = self.__dict__.get("gen", 0) + 1      # which engine process this is (a request notes its own); first, so
        #                                                 a request that sees the new number sees the new queues (#4451)
        self.lines: queue.Queue = queue.Queue()
        # --batch: concurrent requests in the engine's batch slots (see generate_batched).  The engine says how many
        # it runs (INFO batch_slots=N: it may have fewer than asked, or none, when they do not fit).  Page-pool engines
        # expose their full step capacity (INFO request_capacity); --batch only seeds their scheduling.
        asked = next((int(args[args.index(k) + 1]) for k in ("--batch", "--slots", "--parallel") if k in args), 0)
        self.batch = int(self.info.get("request_capacity" if self.request_protocol else "batch_slots") or
                         (1 if self.request_protocol else 0))
        if self.multi:                                  # DeepSeek --parallel: several requests at once, in its own protocol
            self.batch = self.multi                     # (the server's parallel paths key on it: no FIFO, a request's own figures)
        if asked and self.batch != asked and not self.can_change_settings:
            print(f"[strata] parallel requests: {asked} asked, the engine runs {self.batch or 'one at a time'} "
                  "(its log says why)", flush=True)
        _bg = args[args.index("--batch-groups") + 1] if "--batch-groups" in args else "default"
        # no `--batch-groups` (the default on a layer split) or `auto`: the engine picks the groups and says so
        # (INFO batch_groups=G); a number is the group count asked for (1 = off)
        groups = int(self.info.get("batch_groups") or 1) if _bg in ("auto", "default") else int(_bg)
        groups = groups if self.batch and groups > 0 and self.batch % groups == 0 else 1
        gs = self.batch // groups if self.batch else 0
        # slots in the order that spreads requests over the pipeline's groups first: 0, gs, 2gs, .., 1, gs+1, ..
        self.slot_order = [g * gs + t for t in range(gs) for g in range(groups)]
        self.slot_group = [b // gs if gs else 0 for b in range(self.batch)]   # each slot's pipeline group
        self.slot_groups = groups
        # pipelined groups (the engine says how many it runs: INFO batch_groups; older engines: the args).  A pipelined
        # group's window spans its slots up to its last busy one, and the engine keeps no pipelined slot as a
        # conversation cache - see pick_slot
        groups_run = int(self.info.get("batch_groups") or groups)
        self.slot_gs = gs if groups > 1 and groups_run > 1 else 0
        self.slot_q = [queue.Queue() for _ in range(self.batch)]
        self.slot_busy = [False] * self.batch
        # #1603: who owns a busy flag.  slot_claims: slots a request of this server picked or reserved and has not
        # given back; slot_releasing: slots handed to the background BDONE wait.  A busy slot in neither has no owner
        # (its flag leaked) and would keep every later request waiting for a slot that nothing will free.
        self.slot_claims: set[int] = set()
        self.slot_releasing: set[int] = set()
        # slots a request of a higher priority is taking over (preempting): it waits for the event, which the request
        # it ends (or that one's background BDONE wait) sets once it has read the slot's BDONE
        self.slot_handover: dict[int, threading.Event] = {}
        self._orphan_seen: dict[int, float] = {}
        # what each slot's sessions hold (prompt + every token a window fed), so a conversation's next turn goes to
        # the slot that has its start (the engine checks it again); when the slot was last used
        self.slot_held: list[list[int]] = [[] for _ in range(self.batch)]
        self.slot_used = [0.0] * self.batch
        self.slot_live: list[dict | None] = [None] * self.batch   # /metrics: the request in each slot
        # #1012: the admission state belongs to the server, not to one engine process.  restart() runs this again
        # while requests still wait on it (they hold these very objects), so it is made once and kept: new ones
        # would leave the waiters on a lock and a condition nobody notifies, with their counts missing from the
        # new lists.
        if "slot_cv" not in self.__dict__:
            self.slot_cv = threading.Condition()
            self.waiting = 0                            # requests waiting for the control lines (ctl)
            self.admitting = 0                          # #1603: ... and those that gave the lines back to wait for a slot
            self.par_counters = {"queued": 0, "queue_wait_s": 0.0}
            self.wait_lens: list[list[int]] = []        # ... their prompt lengths (a long read gives way to short ones)
            self.arrivals = collections.deque(maxlen=64)   # monotonic times of the last requests that came for the control lines (STRATA_COALESCE_*)
            self.ctl_epoch = 0                          # how often the control lines were taken
            self.ctl = threading.Lock()                 # one admission or solo request on the control lines at a time
        if "rid_lock" not in self.__dict__:
            self.rid_lock = threading.Lock()
            self.rid_n = 0                              # --parallel requests' ids (rid=): never reused, a restart goes on from here
        self.rid_q: dict[int, queue.Queue] = {}         # this process's: rid -> that request's lines (T / PP / DONE / ERR, rid taken out)
        self.multi_stat = ""                            # the engine's last STAT line over everything it wrote
        self._ctl_erred = False                         # #1059: the control lines ended on an ERR
        self._yielded = None                            # (slot, tokens read): the last request on them gave way
        self.wlock = threading.Lock()                   # stdin writes from several request threads
        self.request_queues = {}
        self.request_lock = threading.Lock()
        self.request_command_lock = threading.Lock()
        self.request_activity = time.monotonic()
        self.request_sequence = 0
        self.request_permits = threading.BoundedSemaphore(max(1, self.batch))
        if self.request_protocol:
            self.slot_live = []
        self.pump = threading.Thread(target=self._pump, daemon=True)
        self.pump.start()

    def _pump(self):
        proc, lines = self.proc, self.lines             # this process's: a restart replaces both (#344)
        slot_q = self.slot_q
        requests = getattr(self, "request_queues", {})
        request_lock = getattr(self, "request_lock", threading.Lock())
        slot_busy = self.slot_busy                      # this process's (a restart makes new ones)
        rid_q = self.rid_q
        multi = self.multi
        line = None
        for line in proc.stdout:
            if self.proc is proc:
                self.request_activity = time.monotonic()
            # checked before batch routing: a fatal line is never a slot's own
            if line.startswith(FATAL_PREFIXES):
                # release_gpu_waits invalidates the verifier, even if the native
                # process stays alive and prints DONE afterwards.
                if self.proc is proc:
                    self.silent_note = "Unrecoverable native verification failure: " + line[4:].strip()
                    self.ended = True
                try:
                    proc.terminate()
                except OSError:
                    pass
                break                           # publish EOF, never the trailing DONE
            if line.startswith(("R ", "C ")):
                fields = line.split(" ", 2)
                if len(fields) == 3:
                    if fields[2].startswith(FATAL_PREFIXES):
                        if self.proc is proc:
                            self.silent_note = fields[2].strip()
                            self.ended = True
                        try:
                            proc.terminate()
                        except OSError:
                            pass
                        break
                    with request_lock:
                        target = requests.get(fields[1])
                        if target is not None:
                            target.put(fields[2])
                    continue
            if line.startswith("BSTAT"):
                sample = parse_batch_window_sample(line)
                if sample is not None and self.proc is proc:
                    self.terminal_batch_sample = {**sample, "observed_monotonic": time.monotonic()}
                continue                        # control telemetry never enters a request's token stream
            if line.startswith("PSTAT"):
                sample = parse_stage_in_flight_sample(line)
                if sample is not None and self.proc is proc:
                    self.terminal_stage_sample = {**sample, "observed_monotonic": time.monotonic()}
                continue
            if multi:                                   # DeepSeek --parallel: `T <rid> <id>`, `PP <rid> ...`, `DONE <rid> ...`, `ERR <rid> ...`
                head, _, rest = line.partition(" ")
                if head in ("T", "PP", "DONE", "ERR") and rest[:1].isdigit():
                    rid_s, _, tail = rest.partition(" ")
                    rq = rid_q.get(int(rid_s))
                    if rq is not None:                  # (a request that has ended: its late lines go nowhere)
                        rq.put(head + (" " + tail if tail else "\n"))
                    continue
                if head == "STAT":                      # over everything the engine wrote, not one request's
                    self.multi_stat = line
                    continue
            if line.startswith(("BT ", "BDONE ")) and slot_q:   # --batch: a batch slot's own lines
                try:
                    b = int(line.split()[1])
                    if 0 <= b < len(slot_q) and not slot_busy[b]:
                        # #4451: a frame for a slot that no request has claimed belongs to a request that is over (its
                        # slot was given back too early, or never claimed): the next request on the slot must not read it
                        self.stale_frames = getattr(self, "stale_frames", 0) + 1
                        btrace("stale frame dropped", line.strip()[:60])
                        continue
                    slot_q[b].put(line)
                    continue
                except (IndexError, ValueError):
                    pass
            lines.put(line)
        if self.proc is proc:                           # a killed engine's pump must not mark its successor dead
            self.ended = True                           # its output closed: it is gone, even before the OS says so
            if line and line.startswith("ERR"):         # #997 #890: why it exited, though no request may read it
                self.last_err = line[4:].strip()        # (a failed batch window: every request reads its slot)
        lines.put(None)
        with request_lock:
            for target in requests.values():
                target.put(None)
        for q in slot_q:
            q.put(None)
        for q in list(rid_q.values()):
            q.put(None)

    def death_note(self) -> str:
        """Why the engine most likely ended, from the end of its log: its own watchdog (issue #29), else RAM."""
        if getattr(self, "silent_note", None):          # #481: the server ended it, not the OS or the engine itself
            return self.silent_note
        tail = wide = ""
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 262144))
                wide = f.read().decode("utf-8", "replace")
            tail = wide[-4096:]
        except (OSError, TypeError):
            pass
        # #1705: STRATA_VERIFY_TRACE dumps are ~30 KB and push the watchdog's line out of a 4 KB tail
        for line in reversed(wide.splitlines()):
            if "issue #29" in line:
                return ("The engine stopped itself because it had stopped making progress - a hang it caught. Its log "
                        "line: " + line.strip() + " - please report it at github.com/Niko1221/Strata/issues.")
        if (self.last_err or "").startswith("fatal GPU error"):    # #4350: the engine ended itself; the next request starts it again
            return ("The graphics card reported a fatal error (" + self.last_err[len("fatal GPU error:"):].strip()[:160] + "), so the engine "
                    "ended. The next request starts it again. If this repeats, please report it at github.com/Niko1221/Strata/issues "
                    "with the log.")
        proc = self.proc                                 # None while a restart has not started the next one yet
        rc = proc.poll() if proc is not None else None
        if rc is not None and rc >= 0 and self.last_err:  # its own last words on stdout: they say why
            return (f"The engine exited (code {rc}) after it reported: {self.last_err} - please report it at "
                    "github.com/Niko1221/Strata/issues with the log.")
        last = next((x.strip() for x in reversed(tail.splitlines()) if x.strip().startswith(("strata", "ERR"))), "")
        if rc is not None and rc >= 0 and last:          # it ended by itself: its own last words say why (#215)
            return (f"The engine exited (code {rc}). Its last log line: {last} - if that does not explain it, please "
                    "report it at github.com/Niko1221/Strata/issues with the log.")
        rt_note = rttime_limit_note() if rc == -9 else ""
        if rt_note:                                       # #3238: the kernel killed it, but not for RAM
            return ("The engine was ended by signal 9. " + rt_note + " If dmesg (sudo dmesg | grep -i -E 'killed "
                    "process|out of memory') shows an out-of-memory kill instead, close other programs or use a smaller "
                    "model (Q2_0 / IQ2_XS).")
        return ("The usual cause is running out of RAM: Linux then ends the biggest program (check: sudo dmesg | "
                "grep -i -E 'killed process|out of memory'); Windows slows down instead. Close other programs or use a "
                "smaller model (Q2_0 / IQ2_XS).")

    def alive(self) -> bool:
        proc = self.proc                                 # read once: restart() / close() may set it to None meanwhile
        return proc is not None and not getattr(self, "ended", False) and proc.poll() is None

    def exit_code(self):
        if self.proc is None:
            return None
        try:
            return self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            return None

    def unload(self):
        """Release GPU/RAM between requests, retaining the spawn config for automatic reloading."""
        self.close()
        self.unloaded = True

    RESTART_RETRY_S = 15.0   # between the tries of restart(): a dying engine's VRAM may take a while to come back

    @staticmethod
    def _ready_pump(proc, out: queue.Queue) -> None:
        """The engine's first lines up to READY, for a read that can time out; None when its output closed first."""
        done = False
        try:
            for line in proc.stdout:
                out.put(line)
                if line.startswith("READY"):
                    done = True
                    return
        except (OSError, ValueError):                   # the pipe closed under us (the engine was killed)
            pass
        finally:
            if not done:
                out.put(None)

    def restart(self, tries: int = 3):
        """Start the engine again (the same command) after it died; the new process has its own line queue.
        close() ends and waits for the old process first (EngineStuck when it cannot be ended).  A start that still
        exits before READY - a dead engine's VRAM can take a while to come back, notably on ROCm - is retried
        (PR #637)."""
        info = dict(self.info)
        self.starting = True                     # prepare() answers 503 "starting" meanwhile (#344)
        try:
            self.close()
            for i in range(tries):
                # #344: not alive until READY - __init__ sets max_context to 0 and blocks until the engine says
                # READY, and a request that saw alive() in that window skipped load() and failed with "context
                # (0)".  __init__ clears `ended` itself once READY (before its pump thread can set it again).
                self.ended = True
                try:
                    self.__init__(*self.spawn)
                    break
                except RuntimeError:
                    try:
                        self.proc.wait(timeout=60)
                    except (subprocess.TimeoutExpired, OSError):
                        pass
                    if i == tries - 1:
                        raise
                    print(f"[strata] the engine did not start (try {i + 1} of {tries}); again in "
                          f"{self.RESTART_RETRY_S:g} s", flush=True)
                    time.sleep(self.RESTART_RETRY_S)
        finally:
            self.starting = False
            # a lazy first start never reached the post-READY __init__ where slot_cv is made (#1527)
            if "slot_cv" in self.__dict__:
                with self.slot_cv:               # #1012: wake the requests that waited through it: they go on with the
                    self.slot_cv.notify_all()    # new engine, or (it did not start) end with a clean EngineDied
        self.info = {**info, **self.info}

    def _parse_done(self, line):
        f = line.split()
        self.last = {"generated": int(f[1]), "prompt_tokens": int(f[2]), "prompt_ms": float(f[3]),
                     "decode_ms": float(f[4]), "finish": f[5]}
        if len(f) >= 9:                                   # the conversation cache's fields (engine 0.1.3+)
            self.last.update(drafts_accepted=int(f[6]), drafts_offered=int(f[7]), reused=int(f[8]))
        if len(f) >= 11:                                  # decode hit rate fields
            self.last.update(hits=int(f[9]), lookups=int(f[10]))
        if len(f) >= 14:                                  # the expert tiers (engine 0.1.31+): RAM / file blobs, file MB
            self.last.update(ram_blobs=int(f[11]), file_blobs=int(f[12]), file_mb=float(f[13]))
        if len(f) >= 15:                                  # #471 (engine 0.1.36+): the prompt tokens actually read
            self.last.update(prompt_read=int(f[14]))
        if len(f) >= 16:                                  # #588 (engine 0.1.39+): routed experts read over PCIe
            self.last.update(offloaded=int(f[15]))
        for tok in f[16:]:                                # key=value fields after the positional ones (the prompt's
            key, eq, value = tok.partition("=")           # first chunk and its streamed expert blobs)
            if eq and key in ("chunk", "experts_streamed"):
                try:
                    self.last[key] = int(value)
                except ValueError:
                    pass

    @contextlib.contextmanager
    def _exclusive_control(self):
        """#4451 (same class): on a batch engine the control lines carry one exchange at a time, and every request takes
        them (`ctl`) before it sends or reads anything there.  VRAM and SAVE / RESTORE read the same lines for their
        answer; run beside an admission they took its PP / T / BADM lines, and the admission its answer.  They take the
        lines like a request does.  One request at a time: nothing to take."""
        ctl = self.__dict__.get("ctl")
        if not self.batch or ctl is None or self._ctl_held:
            yield
            return
        if not ctl.acquire(timeout=self.CONTROL_WAIT_S):
            raise ValueError("a request is still running; try again when it has finished")
        try:
            yield
        finally:
            ctl.release()

    def vram(self, reserve_mib: int | None, timeout: float = 120.0) -> dict:
        with self._exclusive_control():
            return self._vram(reserve_mib, timeout)

    def _vram(self, reserve_mib: int | None, timeout: float = 120.0) -> dict:
        """#533: `VRAM <reserve_mib>` between requests (the caller holds the service's FIFO): the engine shrinks its
        expert cache until that much VRAM is free, or grows it back when more is free; None: back to the reserve it
        started with.  -> the engine's figures (expert_slots, expert_cache_mib, vram_free_mib, ...); raises ValueError
        with the engine's reason (e.g. it was not started with --vram-elastic), EngineDied when it ended."""
        if not self.alive():
            raise EngineDied("the engine is not running")
        self.proc.stdin.write("VRAM" + ("" if reserve_mib is None else f" {int(reserve_mib)}") + "\n")
        self.proc.stdin.flush()
        deadline = time.time() + timeout
        while True:
            try:
                line = self.lines.get(timeout=max(0.1, deadline - time.time()))
            except queue.Empty:
                raise EngineDied("the engine did not answer the VRAM command") from None
            if line is None:
                raise EngineDied("the engine ended")
            line = line.strip()
            if line.startswith("ERR"):
                raise ValueError(line[4:].strip() or "the engine refused the VRAM command")
            if line.startswith("VRAM "):
                out = {}
                for kv in line.split()[1:]:
                    k, _, v = kv.partition("=")
                    out[k] = int(v) if v.lstrip("-").isdigit() else v
                self.info.update({k: out[k] for k in ("expert_slots", "vram_free_mib") if k in out})
                self.info["vram"] = out
                return out
            if time.time() > deadline:
                raise EngineDied("the engine did not answer the VRAM command")

    def _ask(self, command: str, kind: str, timeout: float = 30.0, end: str | None = None,
             also: tuple[str, ...] = (), refused: type[Exception] | None = None) -> list[list[str]]:
        """Serializes command replies independently of RUN event streams."""
        if not getattr(self, "request_protocol", False):
            return self._ask_command(command, kind, timeout, end, also, refused)
        with self.request_command_lock:
            process = self.proc
            queues, lock = self.request_queues, self.request_lock
            channel = queue.Queue()
            with lock:
                self.request_sequence += 1
                identity = f"c{self.gen}_{self.request_sequence}"
                queues[identity] = channel
            try:
                return self._ask_command(command, kind, timeout, end, also, refused,
                                         channel=channel, identity=identity, process=process)
            finally:
                with lock:
                    queues.pop(identity, None)

    def _ask_command(self, command: str, kind: str, timeout: float = 30.0, end: str | None = None,
                     also: tuple[str, ...] = (), refused: type[Exception] | None = None,
                     *, channel=None, identity=None, process=None) -> list[list[str]]:
        """Send `command` and return the fields of each `kind` line (and `also` line) of its answer, up to its `end`
        line (default END_<kind>S; other lines on the way - a NOTE, BSTAT - are passed over).  Raises EngineDied when
        the engine ended, refused or said nothing; `refused`, when given, is raised with the engine's reason instead
        for a refusal (an `ERR` line)."""
        if not self.alive():
            raise EngineDied("the engine is not running")
        if identity is None:
            self._send(command)
        else:
            self._send(f"ASK {identity} {command}", process=process)
        replies = channel if channel is not None else self.lines
        name, end, out = command.split()[0], end or "END_" + kind + "S", []
        deadline = time.time() + timeout
        while True:
            try:
                line = replies.get(timeout=max(0.1, deadline - time.time()))
            except queue.Empty:
                raise EngineDied(f"the engine did not answer {name}") from None
            if line is None:
                raise EngineDied("the engine ended")
            f = line.split()
            if f[:1] == [end]:
                return out
            if f[:1] == ["ERR"]:
                why = line[4:].strip()
                raise refused(why) if refused else EngineDied(f"the engine refused {name}: {why}")
            if f and (f[0] == kind or f[0] in also):
                out.append(f)

    def list_conversations(self, marker=(), preview: int = 0) -> list[dict]:
        """The API monitor's Context view: `LIST_CONVERSATIONS` (an engine whose READY says "conversations";
        include/strata/common/state/token_listing.hpp) - each conversation the engine holds, in a session in VRAM or
        parked, without its tokens: [{"id", "place", "state", "count", "capacity", "bytes", "used", "preview"}]
        (count and capacity in tokens: what it holds and the most it can; bytes its physical size and used what its
        tokens take).  `marker` and `preview` ask for the
        first `preview` tokens after the first place a conversation holds `marker` (the user-turn header).  Between
        requests: the caller holds the service's FIFO and, with --batch, the control lines; the slots may decode
        meanwhile (the engine answers between their windows)."""
        command = "LIST_CONVERSATIONS" + (f" {','.join(map(str, marker))} {int(preview)}" if marker and preview else "")
        return [parse_conversation_line(f) for f in self._ask(command, "CONVERSATION",
                                                            timeout=self.CONTEXT_QUERY_TIMEOUT_S)]

    def context_windows(self, ids) -> dict:
        """`GET_CONTEXT_WINDOWS`: the tokens of the conversations `ids` names (as list_conversations gives them), as
        array("i"); None for one the engine no longer holds.  The same turn rules as list_conversations."""
        out = {}
        for f in self._ask("GET_CONTEXT_WINDOWS " + " ".join(ids), "CONTEXT_WINDOW",
                           timeout=self.CONTEXT_QUERY_TIMEOUT_S):
            # CONTEXT_WINDOW <id> <n> [<ids>]  or  CONTEXT_WINDOW <id> gone
            out[f[1]] = None if f[2] == "gone" else array("i", map(int, f[3].split(","))) if len(f) > 3 else array("i")
        return out

    MEMORY_QUERY_TIMEOUT_S = 30.0
    CONTEXT_QUERY_TIMEOUT_S = 300.0  # A single large prompt chunk may exceed the ordinary query deadline.

    def memory(self) -> dict:
        """GET /v1/memory: `GET_MEMORY_USE` (an engine whose READY says "memory";
        include/strata/common/serve/memory_report.hpp) - what the engine holds in each memory and its settings:
        {"memories": [{"id", "name", "total", "free", "unified", "uses": {use: bytes}, "paused_count",
        "paused_budget", "allocated_pct", "fragmented_pct", "setting", "smallest", "largest", "mapped"}],
        "settings": {key: number}}, and "conversations" (list_conversations' entries) from an engine that lists them
        there (strata-q38).  allocated_pct and fragmented_pct: the shares, in percent, of a page pool's pages that are
        allocated, and that are free in runs too short for any expert (strata-q38's pool).  setting: the pool's size
        in bytes now (what CHANGE_SETTINGS' memory_mib.<id> sets); smallest and largest: the range of sizes it
        accepts now.  mapped: the bytes of it the GPUs map (page-locked), which Windows counts in a GPU's shared use
        too.  A value the engine did not say is None (uses: absent).  The same turn rules as list_conversations."""
        return self._ask_memory_report("GET_MEMORY_USE")

    def change_settings(self, values: dict) -> dict:
        """POST /v1/memory: `CHANGE_SETTINGS k=v ...` (an engine whose READY says "settings") changes `values` live
        ({setting: value as the engine spells it}) and returns the engine's report after the change, as memory()
        gives it, with "shortfall": what the change got less of than asked, in the engine's words, when it did.
        Raises SettingsRefused with the engine's reason when it changed nothing.  The same turn rules as
        list_conversations."""
        return self._ask_memory_report("CHANGE_SETTINGS " + " ".join(f"{k}={v}" for k, v in values.items()),
                                       SettingsRefused)

    def _ask_memory_report(self, command: str, refused: type[Exception] | None = None) -> dict:
        """Sends `command` and parses its GET_MEMORY_USE-style answer for memory() and change_settings()."""
        memories, settings, held, shortfall, pause_directory = [], {}, [], None, None
        details, expert_fill = {}, {}
        for f in self._ask(command, "MEMORY", end="END_MEMORY",
                           timeout=self.MEMORY_QUERY_TIMEOUT_S,
                           also=("SETTINGS", "CONVERSATION", "SHORTFALL", "MEMORY_BLOCK",
                                 "PAUSE_DIRECTORY", "MEMORY_DETAIL", "EXPERT_FILL"),
                           refused=refused):
            # MEMORY <id> [key=value ...] [name=<text to the end>]  /  CONVERSATION ...  /  SETTINGS [key=value ...]  /
            # SHORTFALL <text>
            if f[0] == "EXPERT_FILL":
                values = dict(x.split("=", 1) for x in f[1:])
                expert_fill = {"loaded": bool(int(values["loaded"])), "pending": int(values["pending"])}
                continue
            if f[0] == "MEMORY_DETAIL":
                details[f[1]] = unquote(f[2])
                continue
            if f[0] == "PAUSE_DIRECTORY":
                pause_directory = " ".join(f[1:])
                continue
            if f[0] == "MEMORY_BLOCK":
                memory = next(m for m in memories if m["id"] == f[1])
                category, pool, label = ("" if v == "~" else unquote(v) for v in f[3:6])
                size = int(f[2])
                memory["uses"][category] = memory["uses"].get(category, 0) + size
                memory.setdefault("blocks", []).append({"category": category, "pool": pool,
                                                         "label": label, "bytes": size})
                if len(f) >= 8:
                    memory["blocks"][-1].update(identity_group=unquote(f[6]),
                                                 ids=[int(v) for v in f[7].split(",")])
                continue
            if f[0] == "CONVERSATION":
                held.append(parse_conversation_line(f))
                continue
            if f[0] == "SHORTFALL":
                shortfall = " ".join(f[1:])
                continue
            at = next((i for i, x in enumerate(f) if x.startswith("name=")), len(f))
            keys = {k: int(v) for k, _, v in (x.partition("=") for x in f[1 if f[0] == "SETTINGS" else 2:at])}
            if f[0] == "SETTINGS":
                settings = keys
                continue
            pool = keys.pop("pool_pages", 0)
            share = {k: round(100 * keys.pop(f"{k}_pages", 0) / pool, 2) if pool else None
                     for k in ("allocated", "fragmented")}
            memories.append({"id": f[1], "name": " ".join(f[at:])[5:] or None, "total": keys.pop("total", None),
                             "free": keys.pop("free", None), "unified": bool(keys.pop("unified", 0)),
                             "paused_count": keys.pop("paused_count", None),
                             "paused_budget": keys.pop("paused_budget", None),
                             "allocated_pct": share["allocated"], "fragmented_pct": share["fragmented"],
                             **{k: keys.pop(k, None) for k in
                                ("setting", "smallest", "largest", "mapped", "pool_offset")},
                             "uses": keys})
        for key in ("concurrent", "tokens_max", "vram_conversations", "conversation_idle_seconds",
                    "save_on_shutdown"):
            if key in self.info and key in settings:
                self.info[key] = settings[key]
        if "context_capacity" in self.info and "context" in settings:
            self.max_context = settings["context"]
        return {"memories": memories, "settings": settings, **({"details": details} if details else {}),
                **({"expert_fill": expert_fill} if expert_fill else {}),
                **({"conversations": held} if held else {}),
                **({"shortfall": shortfall} if shortfall else {}),
                **({"pause_directory": pause_directory} if pause_directory else {})}

    def pen_from_key(self, orig_len: int, prompt_len: int) -> str:
        """` pen_from=N` for a leg that continues a request (prompt + what it generated): where the reply began, Strata's penalty window start. The all-cards split engine
        (sFNX) counts its penalty window over the last penalty_last_n tokens of the whole context, so the continued prompt gives it the same window and the engine refuses the
        key (GEN: unknown key pen_from): it is not sent."""
        if self.split_engine or prompt_len <= orig_len:
            return ""
        return f" pen_from={orig_len - 1}"

    def gen_keys(self, sampling: dict | None) -> str:
        """Gets the request's key=value options for this engine's GEN / BGEN / RUN line."""
        return self.sampling_keys(sampling or {}, self.split_engine)

    @staticmethod
    def sampling_keys(sampling: dict, split_engine: bool = False) -> str:
        """Gets a request's key=value options for an engine's request line; `split_engine`: for the all-cards split
        engine, which refuses keys it does not know, so a request's priority (which only orders the server's queue
        there) is left out."""
        keys = ""
        t = sampling.get("temperature")
        if isinstance(t, (int, float)) and float(t) > 0.0:
            keys += f" temperature={float(t)!r}"
        tp = sampling.get("top_p")
        if isinstance(tp, (int, float)) and float(tp) < 1.0:
            keys += f" top_p={float(tp)!r}"
        tk = sampling.get("top_k")
        if isinstance(tk, int) and not isinstance(tk, bool) and tk >= 0:
            # the engine's sampled path keeps at most 64 candidates: 0 ("off") and wider lists get all 64
            keys += f" top_k={tk if 1 <= tk <= 64 else 64}"
        mp = sampling.get("min_p")
        if isinstance(mp, (int, float)) and 0.0 < float(mp) <= 1.0:
            keys += f" min_p={float(mp)!r}"
        rp = sampling.get("repetition_penalty")
        rp_on = isinstance(rp, (int, float)) and float(rp) != 1.0
        pf = sampling.get("frequency_penalty")
        pf_on = isinstance(pf, (int, float)) and float(pf) != 0.0
        pp = sampling.get("presence_penalty")
        pp_on = isinstance(pp, (int, float)) and float(pp) != 0.0
        if rp_on:
            keys += f" penalty_repeat={float(rp)!r}"
        if pf_on:
            keys += f" penalty_freq={float(pf)!r}"
        if pp_on:
            keys += f" penalty_present={float(pp)!r}"
        if rp_on or pf_on or pp_on:
            # a penalty without a window counts over nothing: the engine's default is the last 64 tokens
            pln = sampling.get("penalty_last_n")
            if isinstance(pln, int) and not isinstance(pln, bool) and pln > 0:
                keys += f" penalty_last_n={pln}"
            else:
                keys += " penalty_last_n=64"
        seed = sampling.get("seed")
        if isinstance(seed, int) and seed > 0:
            keys += f" seed={seed}"
        priority = get_request_priority(sampling)
        if priority != 0 and not split_engine:
            keys += f" priority={priority}"
        # setup's calibration (tools/calibrate.py): engine settings for this request only, measured without a restart
        tune = sampling.get("strata_tune")
        if isinstance(tune, dict):
            for k in ("pcie_frac", "spec_min_p"):
                v = tune.get(k)
                if isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= float(v) <= 1.0:
                    keys += f" {k}={float(v)!r}"
            # a layer split's prompt chunks (--prefill-pipe): 0 off, 1 evened out, > 1 the rig's b/a in tokens
            v = tune.get("prefill_pipe")
            if isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= float(v) <= 262144.0:
                keys += f" pipe={float(v)!r}"
            # --aux-cpus for this request: 1 puts the engine's threads that are neither pool workers nor the host on the
            # spare CPUs, 0 leaves them where they were created (true/false or 1/0; the engine started without a spare CPU
            # ignores a 1).  Anything else is not sent.
            ac = tune.get("aux_cpus")
            if isinstance(ac, bool) or (isinstance(ac, (int, float)) and ac in (0, 1)):
                keys += f" aux_cpus={int(ac)}"
        # "strata_checkpoint": false - a one-shot call (a classification, a probe) whose turn no later request
        # extends: no conversation checkpoint for it (#830).  It still reuses a cached prefix.  Absent = as before.
        if sampling.get("strata_checkpoint") is False:
            keys += " ckpt=0"
        # #2228: the pictures' keys (hex, in prompt order): the engine matches a picture by them, not by its embeddings
        ik = sampling.get("strata_img_keys")
        if isinstance(ik, list) and ik and all(isinstance(k, str) and k and all(c in "0123456789abcdef" for c in k)
                                               for k in ik):
            keys += " img_keys=" + ",".join(ik)
        # "strata_prefix" (resolved by Service.resolve_prefix to {"tokens": N}): the first N prompt tokens are a shared
        # prefix - the engine pins the checkpoint there.  Absent or unresolved = as before.
        prefix = sampling.get("strata_prefix")
        if isinstance(prefix, dict):
            pn = prefix.get("tokens")
            if isinstance(pn, int) and not isinstance(pn, bool) and pn > 0:
                keys += f" pin={pn}"
        # restore points (Service.prepare puts them in `_restore`; only the DeepSeek engine gets them)
        keys += StrataEngine.restore_keys(sampling)
        # logprobs: the engine writes an `LP` line after each `T` line (an engine without them ignores
        # the key, so an older engine answers exactly as it did).  Absent = no LP lines at all.
        k = top_logprobs(sampling)
        if k is not None:
            keys += f" logprobs={k}"
            ids = allowed_ids(sampling)
            if ids:
                # only these ids are printed; the engine's comma-separated list, renormalised below
                keys += " allowed=" + ",".join(str(i) for i in ids)
        return keys + StrataEngine.projection_key(sampling) + bias_api.engine_key(sampling.get("logit_bias"))

    @staticmethod
    def restore_keys(sampling: dict) -> str:
        """`marks=<pos>,...`, `tail=<pos>` and `shared=<pos>`: the DeepSeek engine's restore points for this prompt
        (Service.prepare puts them in `_restore` with STRATA_DS4_RESTORE_POINTS; no other engine gets them).  "" without:
        the line as before."""
        pts = sampling.get("_restore")
        if not isinstance(pts, dict):
            return ""

        def pos(v):
            return isinstance(v, int) and not isinstance(v, bool) and v > 0
        marks = [p for p in pts.get("marks") or () if pos(p)]
        tail, shared = pts.get("tail"), pts.get("shared")
        return ((" marks=" + ",".join(str(p) for p in marks) if marks else "") +
                (f" tail={tail}" if pos(tail) else "") + (f" shared={shared}" if pos(shared) else ""))

    @staticmethod
    def projection_key(sampling: dict) -> str:
        """`cvec=0|1`: the experimental-speed-projection control vector for this request, when the engine was
        started with one (--control-vector-scaled; an engine without one ignores the key).  Absent = on."""
        on = sampling.get("experimental_speed_projection")
        return f" cvec={int(on)}" if isinstance(on, bool) else ""

    def _send(self, text: str, process=None):
        """Writes to the expected process, refusing a replacement after a restart."""
        btrace("send>", text[:60])
        try:
            with self.wlock:
                destination = self.proc if process is None else process
                if process is not None and self.proc is not process:
                    raise EngineDied("the engine restarted before this command could be sent")
                destination.stdin.write(text + "\n")
                destination.stdin.flush()
        except (OSError, AttributeError, ValueError):
            if process is not None:
                raise EngineDied("the request's engine process ended") from None
            raise EngineDied(f"the engine stopped unexpectedly (exit code {self.exit_code()})") from None

    def _send_for(self, born: int, text: str):
        """#4451: a line meant for the engine process `born`.  After a restart the same slot number belongs to another
        request of the new process: a BSTOP or BYIELD that followed the old request would end that one."""
        if self.gen != born:
            raise EngineDied("the engine restarted while this request was running; it was not completed")
        self._send(text)

    def _quarantine_control(self, until: str, born: int, lines_q, slot, holding: bool) -> None:
        """#4451: a request left (cancel, disconnect, error) and the engine has not yet given the line that ends its
        exchange (`until`: DONE of a solo read, BADM of an admission) while it said nothing for DRAIN_TIMEOUT_S.  The control lines (and the
        admission's slot) stay with this exchange - nobody else is given them - until the engine confirms, so what it
        still prints can not reach a later request.  The engine silent for QUARANTINE_S more: it is ended.  (Only after a STOP or
        BSTOP, and only for silence: a generation that is still producing lines is never cut off.)"""
        if slot is not None:
            with self.slot_cv:
                self.slot_releasing.add(slot)           # busy stays: no request may take the slot
                self.slot_claims.discard(slot)

        def run():
            try:
                line = self._drain_control(until, timeout=self.QUARANTINE_S, born=born, lines_q=lines_q)
                if line is None:
                    self._end_out_of_step(born, "the engine did not end an interrupted request (STOP) in time")
                cont = bool(until == "BADM" and line and line.startswith("BADM ") and line.split()[2:3] == ["1"])
                if slot is not None and self.gen == born:
                    if cont and self.alive():
                        self._release_slot_when_done(slot, born=born)      # it went on in the slot: BSTOP, free at BDONE
                    else:
                        with self.slot_cv:
                            self.slot_busy[slot] = False
                            self.slot_releasing.discard(slot)
                            self.slot_cv.notify_all()
            finally:
                if holding:
                    self.ctl.release()
                with self.slot_cv:
                    self.slot_cv.notify_all()
        threading.Thread(target=run, daemon=True).start()

    def _control(self, cancel, on_token, stop_when=None, on_quiet=None, want_lp=False, lines_q=None):
        """Reads the control lines of the request on them (GEN / BGEN), yielding None heartbeats.  Calls on_token(id)
        for each `T`; returns ("done", None) at DONE, or ("badm", continues) at BADM (after DONE).  `stop_when()`
        true sends STOP once (the request is then read to its DONE).  `on_quiet()` runs whenever the engine has been
        quiet for half a second: between two PP lines of a read whose chunks take longer than that.

        `want_lp`: a batch admission's token is the only one this reader ever sees with an `LP` line after it (a
        slot's `BT` lines carry none), so a request asking for logprobs there is a `max_tokens` 1 one: read the
        line here, for the request itself (its own lines: the engine's is shared by the slots' readers)."""
        stopped = False
        beat = time.monotonic()
        held = NO_LINE                               # a line read while looking for a token's LP line
        lines_q = self.lines if lines_q is None else lines_q   # #4451: the request's own engine's lines (a restart makes new ones)

        def send_stop():
            if lines_q is self.lines:               # (a STOP to the engine that replaced this one would stop another request)
                self._send("STOP")
        while True:
            if held is not NO_LINE:
                line, held = held, NO_LINE
            else:
                try:
                    # A client can leave before the first PP/token.  Poll its Event as often as _take_control does;
                    # the ten-second HTTP heartbeat is independent of how promptly STOP reaches the engine.
                    line = lines_q.get(timeout=0.5)
                except queue.Empty:
                    if cancel.is_set() and not stopped:
                        send_stop()
                        stopped = True
                    if on_quiet is not None:
                        on_quiet()
                    if time.monotonic() - beat >= 10.0:
                        beat = time.monotonic()
                        yield None
                    continue
            beat = time.monotonic()
            if line is None:
                raise EngineDied(f"the engine stopped unexpectedly (exit code {self.exit_code()})")
            if not line.startswith(("PP ", "INFO")):
                btrace("ctl<", self._ctl_mode, line.strip()[:60])
            if line.startswith("T "):
                t = int(line[2:])
                if want_lp:
                    # the engine writes the token and its LP line in one flush: the next line is that one, or a
                    # line this reader must still see (held for the next turn, as generate() does)
                    try:
                        held = lines_q.get(timeout=0.5)
                    except queue.Empty:
                        held = NO_LINE
                    if isinstance(held, str) and held.startswith("LP "):
                        own = getattr(cancel, "_strata_lp_lines", None)
                        if own is not None:
                            own.append((t, held))
                        held = NO_LINE
                on_token(t)
                if not stopped and (cancel.is_set() or (stop_when is not None and stop_when())):
                    send_stop()
                    stopped = True
                yield False                              # a token is pending (not a heartbeat)
            elif line.startswith("PP "):
                f = line.split()
                if len(f) >= 3 and f[1].isdigit() and f[2].isdigit():
                    self.progress = (int(f[1]), int(f[2]))
                    self.progress_ms = float(f[3]) if len(f) >= 4 else 0.0
                    self.prefill_tok_s_mean = float(f[4]) if len(f) >= 5 else None
                if not stopped and cancel.is_set():
                    # #879: a Stop during the prompt read went out only after the first token (or after 10 s of
                    # silence, which a stream of PP lines never allows): the engine read the whole prompt first
                    send_stop()
                    stopped = True
                yield None
            elif line.startswith("DONE"):
                self._parse_done(line)
                self._last_done = line
            elif line.startswith("BADM "):
                f = line.split()
                self._ctl_result = ("badm", len(f) >= 3 and f[2] == "1")
                return
            elif line.startswith("BCOK "):               # STRATA_COALESCE_PREFILL: <slot> <tokens>: the slot holds a waiting prompt
                pass
            elif line.startswith("YIELDED "):            # the read gave way (BYIELD): <slot> <tokens read>
                f = line.split()
                if len(f) >= 3 and f[1].lstrip("-").isdigit() and f[2].isdigit():
                    self._yielded = (int(f[1]), int(f[2]))
            elif line.startswith("ERR"):
                self._ctl_erred = True                   # #1059: the engine refused it and is idle: no DONE follows
                self._raise_if_fatal_gpu(line)
                raise ValueError(line[4:].strip())
            if line.startswith("DONE") and self._ctl_mode == "solo":
                self._ctl_result = ("done", None)
                return

    def _done_finish(self) -> str | None:
        """The finish field of the DONE line the control lines read last (None: none since it was cleared)."""
        f = (getattr(self, "_last_done", None) or "").split()
        return f[5] if len(f) > 5 else None

    def _drain_control(self, until: str, timeout: float | None = None, born: int | None = None, lines_q=None):
        """After a consumer left early: read the control lines up to the next `until` line (DONE or BADM) so the next
        request does not read this one's leftovers.  Returns that line (None: the engine ended or never answered).
        `born`: the engine process (self.gen) the request started on; once that one is gone there is nothing to
        drain - its end line was read already (by the request's own EngineDied), and waiting for another one held
        a request that had failed for the whole 300 s (#1012), or read the lines of the engine that replaced it.
        `timeout` (DRAIN_TIMEOUT_S) is a NO-PROGRESS limit: it counts from the last line the engine wrote, so an engine that
        is still talking (PP / T lines of a read or a decode it was told to stop but has not yet) is never given up on;
        only one that says nothing for that long is."""
        wait_s = self.DRAIN_TIMEOUT_S if timeout is None else timeout
        end = time.monotonic() + wait_s
        lines_q = self.lines if lines_q is None else lines_q
        while time.monotonic() < end:
            if born is not None and (self.gen != born or not self.alive()):
                return None
            try:
                line = lines_q.get(timeout=max(0.1, min(1.0, end - time.monotonic())))
            except queue.Empty:
                continue
            if line is None:
                return None
            end = time.monotonic() + wait_s                  # any line is progress
            if line.startswith("DONE"):
                self._parse_done(line)
            if line.startswith(until) or line.startswith("ERR"):
                return line
        return None

    def _apply_bdone(self, f: list[str]) -> None:
        """BDONE counts the admission token too, but its clock starts after the admission's DONE."""
        if len(f) >= 5 and isinstance(self.last, dict):
            admission_ms = self.last.get("decode_ms")
            self.last = {**self.last, "finish": f[3],
                         "decode_ms": admission_ms + float(f[4]) if admission_ms is not None else None,
                         "generated": int(f[2]) if f[2].isdigit() else None}

    def _raise_if_fatal_gpu(self, line: str) -> None:
        """#4350: the engine's ERR line "fatal GPU error: ..." means it is ending (its GPU context is gone).  The request in flight fails
        as an engine death (a 503 the client can send again), not as a bad request (400), and the engine counts as ended, so the next
        request starts it again at once instead of finding a closed pipe."""
        text = line[4:].strip()
        if not text.startswith("fatal GPU error"):
            return
        self.last_err = text
        self.ended = True
        raise EngineDied("the GPU reported a fatal error and the engine ended: " + text[len("fatal GPU error:"):].strip()[:200])

    def _bdone_after_eos(self, slot: int, wait_s: float = 2.0, q=None):
        """#1869: a request in a slot whose consumer left at the stop token has its BDONE right behind it (the engine prints it
        in the same window).  Read it here, so the request's figures are the slot leg's (the background wait would eat it,
        and the request kept any admission tokens). None when it does not come in `wait_s`: then the background
        wait takes over."""
        if q is None and slot >= len(self.slot_q):
            return None
        q = self.slot_q[slot] if q is None else q
        end = time.monotonic() + wait_s
        while time.monotonic() < end:
            try:
                line = q.get(timeout=max(0.05, end - time.monotonic()))
            except queue.Empty:
                return None
            if line is None:
                return None
            if line.startswith("BDONE "):
                return line
        return None

    def _end_out_of_step(self, born: int, what: str) -> None:
        """#4451: the engine did not give a line it owes (a BDONE after BSTOP, a DONE / BADM after STOP) within the
        quarantine: what it may still print can not be told from a later request's lines, so the engine is ended (the
        same end as a silent one, #481) and the next request starts it again.  Only the engine process it concerns."""
        if self.gen == born and self.alive():
            self._silent(what)

    def _release_slot_when_done(self, slot: int, stream: list[int] | None = None,
                                taken: threading.Event | None = None, born: int | None = None):
        """A slot whose consumer left (a stop token, a stop string, a disconnect): BSTOP it and free it once the engine
        says BDONE (in the background).  `stream`: the prompt and every token of it so far - with the tokens still to
        come before BDONE, all but the last are what the slot holds then (the next turn of its conversation).
        `taken`: a request preempting this one takes the slot over: no BSTOP (its BGEN ends this request), and at the
        BDONE the event is set instead of the slot being freed (unless that request gave up meanwhile).
        `born`: the engine process (self.gen) the slot belongs to; None: the one running now."""
        if slot >= len(self.slot_q) or (born is not None and self.gen != born):
            # the engine restarted (#4451): the slot is gone, or the same number is a request's of the NEW engine,
            # which a BSTOP now would stop (and the wait below would read)
            if taken is not None:
                taken.set()
            return
        born = self.gen if born is None else born
        q, busy, held = self.slot_q[slot], self.slot_busy, self.slot_held   # this process's: restart() replaces them
        releasing, handover, claims, live = self.slot_releasing, self.slot_handover, self.slot_claims, self.slot_live
        if taken is None:
            try:
                self._send(f"BSTOP {slot}")
            except EngineDied:
                pass
            releasing.add(slot)

        def wait():
            end = time.monotonic() + self.QUARANTINE_S      # no-progress limit: every line of the slot starts it again
            tail = list(stream or [])
            stopped, confirmed = taken is None, False
            while time.monotonic() < end:
                if not stopped and handover.get(slot) is not taken:   # the request taking the slot over gave up:
                    stopped = True                                      # this one is not ended by it, so stop it
                    try:
                        self._send_for(born, f"BSTOP {slot}")
                    except EngineDied:
                        pass
                try:
                    line = q.get(timeout=0.5 if not stopped else 5.0)
                except queue.Empty:
                    if self.gen != born or not self.alive():
                        break
                    continue
                if line is None:
                    tail = []
                    break
                end = time.monotonic() + self.QUARANTINE_S
                if line.startswith("BT "):
                    try:
                        tail.append(int(line.split()[2]))
                    except (IndexError, ValueError):
                        tail = []
                if line.startswith("BDONE "):
                    confirmed = True
                    if line.split()[3:4] == ["nonfinite"]:   # #879: the engine dropped the slot's state
                        tail = []
                    break
            if not confirmed:
                tail = []
                # #4451: no BDONE in the quarantine from an engine that still runs: the slot is NOT handed on with its
                # lines still to come - the engine is ended instead (a dead engine's slots are free by being gone)
                self._end_out_of_step(born, "the engine did not end a stopped batch slot (BSTOP) in time")
            if taken is not None:
                with self.slot_cv:
                    # the request that was taking the slot over gave up (the engine refused its BGEN, so this request
                    # ran on), and nobody else frees the slot
                    if handover.get(slot) is not taken and slot not in releasing:
                        live[slot] = None
                        busy[slot] = False
                        claims.discard(slot)
                        self.slot_cv.notify_all()
                taken.set()
                return
            held[slot] = tail[:-1] if stream and tail else []
            with self.slot_cv:
                busy[slot] = False
                releasing.discard(slot)
                self.slot_cv.notify_all()
        threading.Thread(target=wait, daemon=True).start()

    BGEN_REFUSALS_MAX = 6    # #1994: how often one request tries another slot after the engine refused a BGEN
    SLOT_PARK_S = 2.0        # ... and how long a refused slot stays out of the picks

    def _park_slot(self, slot: int) -> None:
        """#1994: the engine refused a BGEN for `slot`.  Keep it busy here for SLOT_PARK_S (no request owns it), then
        give it back: the engine's view has caught up, or the next refusal parks it again."""
        with self.slot_cv:
            self.slot_claims.discard(slot)
            self.slot_busy[slot] = True
        busy = self.slot_busy

        def free():
            with self.slot_cv:
                if slot not in self.slot_claims and slot not in self.slot_releasing:
                    busy[slot] = False
                self.slot_cv.notify_all()
        t = threading.Timer(self.SLOT_PARK_S, free)
        t.daemon = True
        t.start()

    ORPHAN_SLOT_S = 3.0   # #1603: a busy flag nobody owns this long is a leak and is given back

    def _reap_orphan_slots(self) -> list[int]:
        """#1603: free the slots whose busy flag has no owner (no request holds or reserves them and no BDONE wait is
        running for them).  The caller holds slot_cv.  A flag that leaked kept every later request waiting for a slot
        that nothing would free - with nothing running, nothing queued and nothing for the frozen-engine check to see.
        A slot is only given back after it has looked ownerless for ORPHAN_SLOT_S, so a flag set a moment before its
        owner registers is never taken."""
        now = time.monotonic()
        freed = []
        for b, busy in enumerate(self.slot_busy):
            if busy and b not in self.slot_claims and b not in self.slot_releasing:
                first = self._orphan_seen.setdefault(b, now)
                if now - first >= self.ORPHAN_SLOT_S:
                    self.slot_busy[b] = False
                    self._orphan_seen.pop(b, None)
                    freed.append(b)
            else:
                self._orphan_seen.pop(b, None)
        if freed:
            print(f"[strata] parallel: slot(s) {', '.join(map(str, freed))} were marked busy with no request using them "
                  "(issue #1603); freed, the waiting request goes on", flush=True)
            self.slot_cv.notify_all()
        return freed

    def _pick_victim(self, priority: int) -> int | None:
        """Picks a slot to preempt for a request of `priority` (the caller holds slot_cv): one whose request decodes at
        a lower priority and is not being stopped or taken over, the lowest priority first, then the one that started
        last.  None: there is none."""
        victims = [b for b in self.slot_claims if b not in self.slot_handover and b not in self.slot_releasing and
                   (r := self.slot_live[b]) is not None and r["state"] == "decoding" and not r.get("stopping") and
                   r.get("priority", 0) < priority]
        return min(victims, key=lambda b: (self.slot_live[b].get("priority", 0), -self.slot_live[b]["started"]),
                   default=None)

    def _stop_slot(self, slot: int, live: dict, born: int) -> bool:
        """Sends BSTOP for this request's slot `slot` (`live`: its slot_live entry; `born`: the engine process it was
        admitted on, see _send_for) unless a request of a higher priority is taking the slot over, whose BGEN ends this
        request; tells whether it sent it."""
        with self.slot_cv:
            if slot in self.slot_handover:
                return False
            live["stopping"] = True
        self._send_for(born, f"BSTOP {slot}")
        return True

    def _await_handover(self, taken: threading.Event, born: int):
        """Yields None heartbeats until the request a preemption ended has read its BDONE (`taken` is set), or until the
        engine it was sent to is gone."""
        beat = time.monotonic()
        while not taken.wait(0.5) and self.gen == born and self.alive():
            if time.monotonic() - beat >= 10.0:
                beat = time.monotonic()
                yield None

    YIELDS_MAX = 2   # #656: how often one request's prompt read gives way to a shorter waiting one

    def _take_control(self, cancel, plen: int, after_epoch: int | None = None, new: bool = False, prompt=None):
        """Waits for the control lines (one prompt read at a time), yielding None heartbeats; False when cancelled.
        `after_epoch`: a request whose read gave way lets the requests waiting then go first.
        `new`: the request's first admission (STRATA_ADMIT_SHORT_FIRST orders only these)."""
        entry = [plen, time.monotonic(), new, prompt, cancel]           # (length, arrival, ordered: STRATA_ADMIT_SHORT_FIRST; the prompt and request, for STRATA_COALESCE_PREFILL)
        with self.slot_cv:
            self.waiting += 1
            self.wait_lens.append(entry)
            self.arrivals.append(entry[1])
        beat = time.monotonic()
        try:
            while True:
                deadline = getattr(cancel, "_strata_queue_deadline", None)
                if deadline is not None and time.monotonic() >= deadline:
                    raise QueueWaitExpired("the inference queue wait deadline expired; retry later")
                if not self.alive() and not getattr(self, "starting", False):
                    # #1012: the engine is gone and nobody is starting it again: this request was not sent, so it
                    # ends now (a clean 503) instead of waiting for control lines nothing will answer on.  While a
                    # restart is under way it keeps waiting and goes on with the new engine.
                    raise EngineDied("the engine stopped while this request waited; it was not sent")
                with self.slot_cv:
                    turn = after_epoch is None or self.ctl_epoch > after_epoch or self.waiting <= 1
                    if turn and new and ADMIT_SHORT_FIRST > 0:
                        turn = self._admit_next() is entry
                left = deadline - time.monotonic() if deadline is not None else 0.5
                if turn and left > 0 and self.ctl.acquire(timeout=min(0.5, left)):
                    if new and ADMIT_SHORT_FIRST > 0:
                        with self.slot_cv:              # a shorter one came while this one waited in acquire()
                            first = self._admit_next() is entry
                        if not first:
                            self.ctl.release()
                            with self.slot_cv:
                                self.slot_cv.notify_all()
                            continue
                    if self.alive():
                        if deadline is not None and time.monotonic() >= deadline:
                            self.ctl.release()
                            raise QueueWaitExpired("the inference queue wait deadline expired; retry later")
                        break
                    self.ctl.release()                  # it died (or is being restarted) while this one was getting
                    turn = False                        # the lines: not the engine this request was meant for
                if not turn:
                    with self.slot_cv:
                        # #4738 (PR #4738's approach): the turn was read, and the lock given up, before this wait: a
                        # notification sent in between is gone.  Read it again under the lock and only wait when this
                        # request still has to.
                        eligible = after_epoch is None or self.ctl_epoch > after_epoch or self.waiting <= 1
                        if eligible and new and ADMIT_SHORT_FIRST > 0:
                            eligible = self._admit_next() is entry
                        if not eligible or not self.alive():
                            self.slot_cv.wait(timeout=min(0.5, max(0, left)))
                if cancel.is_set():
                    return False
                if time.monotonic() - beat >= 10.0:     # a heartbeat every 10 s, as the engine's own wait
                    beat = time.monotonic()
                    yield None
        finally:
            with self.slot_cv:
                previous_waiting = self.waiting
                self.waiting = max(0, self.waiting - 1)
                if entry in self.wait_lens:
                    self.wait_lens.remove(entry)
                if (previous_waiting > 1 and self.waiting <= 1) or (new and ADMIT_SHORT_FIRST > 0):
                    # a waiter that leaves (cancelled, closed) can make another one the next in turn: tell it
                    self.slot_cv.notify_all()
        with self.slot_cv:
            self.ctl_epoch += 1
            self.slot_cv.notify_all()
        return True

    def _release_preslot(self, cancel):
        """STRATA_COALESCE_PREFILL: the request that was given a slot ahead of its turn left before using it."""
        slot = getattr(cancel, "_strata_preslot", None)
        if slot is None:
            return
        cancel._strata_preslot = None
        with self.slot_cv:
            self.slot_busy[slot] = False
            self.slot_claims.discard(slot)
            self.slot_cv.notify_all()

    def _coalesce(self, cancel, lead_prompt):
        """STRATA_COALESCE_PREFILL (the caller holds the control lines and sends its BGEN right after): reserve free slots
        for requests still waiting for the control lines and send their prompts as BCO lines; the engine reads them in the
        same layer-major pass as the BGEN's prompt.  Each follower finds its slot in cancel._strata_preslot."""
        picked = []
        total = len(lead_prompt)
        def seen() -> int:
            now = time.monotonic()
            with self.slot_cv:
                return max(sum(1 for t in self.arrivals if now - t <= COALESCE_BURST_MS / 1000.0), 1 + len(self.wait_lens))
        recent = seen()
        if recent < COALESCE_MIN_REQS and COALESCE_WAIT_MS > 0:
            time.sleep(0.010)                        # (the first request of a burst holds the lines before its neighbours have arrived)
            recent = seen()
        if recent < COALESCE_MIN_REQS:
            return                                   # one or two requests on this engine: a plain read, at most 10 ms lost
        if COALESCE_WAIT_MS > 0:
            time.sleep(COALESCE_WAIT_MS / 1000.0)    # the requests of a burst reach the server a few ms apart
        with self.slot_cv:
            for e in list(self.wait_lens):
                if len(picked) >= COALESCE_PREFILL - 1:
                    break
                prompt = e[3] if len(e) > 4 else None
                rq = e[4] if len(e) > 4 else None
                if prompt is None or rq is None or rq is cancel or getattr(rq, "_strata_preslot", None) is not None:
                    continue
                if getattr(rq, "_strata_co", False) or len(prompt) < 32 or total + len(prompt) > COALESCE_TOKENS:
                    continue
                slot = self.pick_slot(prompt)
                if slot is None:
                    break
                self.slot_busy[slot] = True
                self.slot_claims.add(slot)
                self.slot_held[slot] = []
                rq._strata_preslot = slot
                rq._strata_co = True
                total += len(prompt)
                picked.append((slot, prompt))
        for slot, prompt in picked:
            self._send(f"BCO {slot} {','.join(str(int(t)) for t in prompt)}")
        if picked:
            print(f"[strata] coalesce: {len(picked)} waiting prompt(s) ({total - len(lead_prompt)} tokens) read with this "
                  f"request's {len(lead_prompt)}", flush=True)

    def _admit_next(self):
        """STRATA_ADMIT_SHORT_FIRST=<tokens> (the caller holds slot_cv): the new request that takes the control lines
        next - one whose prompt is at most that long (it reads little) before a longer one, in arrival order within
        each; a longer one that has waited STRATA_ADMIT_MAX_WAIT_S counts as short (it is never held back for good).
        Off (unset or 0): whoever gets the lock first, as before.  Only the order changes, never what is read."""
        now = time.monotonic()
        ordered = [e for e in self.wait_lens if len(e) > 2 and e[2]]
        if not ordered:
            return None
        return min(ordered, key=lambda e: (0 if e[0] <= ADMIT_SHORT_FIRST or now - e[1] >= ADMIT_MAX_WAIT_S else 1,
                                           e[1]))

    SOLO_AGAIN_MAX = 2      # how often a request left alone in a slot goes back to the solo path
    SOLO_AGAIN_MIN_LEFT = 32  # ... only with at least this many tokens still allowed (max_tokens)

    def _may_go_solo(self, left: int, times: int, embeddings) -> bool:
        """A request decoding in a slot that is alone now (no other slot busy, nobody waiting) goes back to the solo
        path with its MTP drafts: the engine continues it from the slot's sessions (its slot cache; INFO
        slot_cache=1).  STRATA_PARALLEL_SOLO=0 keeps it in the slot."""
        if (embeddings or times >= self.SOLO_AGAIN_MAX or left < self.SOLO_AGAIN_MIN_LEFT or
                not (self.info or {}).get("slot_cache") or os.environ.get("STRATA_PARALLEL_SOLO") == "0"):
            return False
        with self.slot_cv:
            return sum(1 for b in self.slot_busy if b) == 1 and self.waiting == 0

    def _shorter_waiting(self, plen: int) -> bool:
        """#656: someone waits for the control lines with a prompt under half this one's: worth giving way to."""
        with self.slot_cv:
            return any(e[0] * 2 <= plen for e in self.wait_lens)

    def generate_request(self, ids, max_new, sampling, cancel, embeddings=None):
        """Streams one stable RUN identity without occupying the command/reply channel."""
        permits = None if self.info.get("request_queue") else self.request_permits
        born, process = self.gen, self.proc
        queues, lock, live_requests = self.request_queues, self.request_lock, self.slot_live
        beat = time.monotonic()
        while permits is not None and not permits.acquire(timeout=0.1):
            if cancel.is_set():
                return
            deadline = getattr(cancel, "_strata_queue_deadline", None)
            if deadline is not None and time.monotonic() >= deadline:
                raise QueueWaitExpired("the inference queue wait deadline expired; retry later")
            if self.proc is not process or self.gen != born or not self.alive():
                raise EngineDied("the engine ended while this request waited")
            if time.monotonic() - beat >= 10:
                beat = time.monotonic()
                yield None
        if self.proc is not process or self.gen != born or not self.alive():
            if permits is not None:
                permits.release()
            raise EngineDied("the engine ended while this request waited")
        channel = queue.Queue()
        live = {"request_id": getattr(cancel, "_strata_request_id", None), "_cancel": cancel,
                "state": "queued", "prompt_tokens": len(ids), "prompt_read": 0, "generated": 0,
                "started": time.time(), "first_token": None, "priority": get_request_priority(sampling or {})}
        with lock:
            if self.proc is not process or self.gen != born:
                if permits is not None:
                    permits.release()
                raise EngineDied("the engine restarted before this request could be sent")
            self.request_sequence += 1
            identity = f"r{born}_{self.request_sequence}"
            queues[identity] = channel
            live_requests.append(live)
        def release():
            with lock:
                queues.pop(identity, None)
                if live in live_requests:
                    live_requests.remove(live)
            if permits is not None:
                permits.release()

        def release_when_finished():
            deadline = time.monotonic() + 300
            try:
                while self.proc is process and self.gen == born and self.alive():
                    if time.monotonic() >= deadline:
                        self._silent("the engine did not finish a cancelled request")
                        break
                    try:
                        line = channel.get(timeout=1)
                    except queue.Empty:
                        continue
                    if line is None or line.startswith(("DONE ", "ERR ")):
                        break
            finally:
                release()

        complete, sent, stopped, admitted = False, False, False, False
        output, carry = [], {}
        prompt_ms, prompt_read = 0.0, None
        initial_reused = None
        self.last = {}
        silence = self.silence_s or 0.0
        allowance = max(silence, min(len(ids), PP_CHUNK_MAX) / PP_FLOOR_TOK_S) if silence > 0 else 0.0
        heard = time.monotonic()
        stall_state = {}
        try:
            if cancel.is_set():
                return
            deadline = getattr(cancel, "_strata_queue_deadline", None)
            if deadline is not None and time.monotonic() >= deadline:
                raise QueueWaitExpired("the inference queue wait deadline expired; retry later")
            head = "RUNI" if embeddings else "RUN"
            pictures = f" {embeddings}" if embeddings else ""
            self._send(f"{head} {identity} {int(max_new)}{self.gen_keys(sampling)}"
                       f"{pictures} {','.join(str(int(t)) for t in ids)}", process=process)
            sent = True
            while True:
                deadline = getattr(cancel, "_strata_queue_deadline", None)
                if not admitted and deadline is not None and time.monotonic() >= deadline:
                    raise QueueWaitExpired("the inference queue wait deadline expired; retry later")
                if cancel.is_set() and not stopped:
                    self._send(f"CANCEL {identity}", process=process)
                    stopped = True
                try:
                    line = channel.get(timeout=0.1)
                except queue.Empty:
                    if self.proc is not process or self.gen != born or not self.alive():
                        raise EngineDied("the engine ended during this request")
                    activity = max(heard, getattr(self, "request_activity", heard))
                    if allowance > 0 and time.monotonic() - activity >= allowance:
                        raise self._silent("the engine stopped reporting request progress")
                    if ENGINE_STALL_S:
                        self._stall_check(activity, stall_state, ENGINE_STALL_S)
                    if time.monotonic() - beat >= 10:
                        beat = time.monotonic()
                        yield None
                    continue
                if self.proc is not process or self.gen != born:
                    raise EngineDied("the engine restarted during this request")
                heard = beat = time.monotonic()
                stall_state.clear()
                if line is None:
                    complete = True
                    raise EngineDied("the engine ended during this request")
                fields = line.split()
                if not fields:
                    continue
                if fields[0] in ("STARTED", "PP", "T") and not admitted:
                    admitted = True
                    live["state"] = "reading"
                    cancel._strata_queue_deadline = None
                    started = getattr(cancel, "_strata_queue_started", None)
                    if started is not None:
                        started()
                if fields[0] == "T":
                    allowance = silence
                    live["state"] = "generating"
                    live["generated"] += 1
                    live["first_token"] = live["first_token"] or time.time()
                    output.append(int(fields[1]))
                    yield output[-1]
                elif fields[0] == "RESUME":
                    live["reused"] = int(fields[1])
                elif fields[0] == "PP":
                    live["prompt_read"] = int(fields[1])
                    self.progress = (int(fields[1]), int(fields[2]))
                    self.progress_ms = float(fields[3]) if len(fields) > 3 else 0
                    self.prefill_tok_s_mean = float(fields[4]) if len(fields) > 4 else None
                    yield None
                elif fields[0] == "DONE":
                    self._parse_done(line)
                elif fields[0] == "ERR":
                    complete = True
                    self._raise_if_fatal_gpu(line)
                    raise ValueError(" ".join(fields[1:]))
                if fields[0] == "DONE":
                    prompt_ms += float(self.last.get("prompt_ms") or 0.0)
                    if self.last.get("prompt_read") is not None:
                        prompt_read = (prompt_read or 0) + int(self.last["prompt_read"])
                    finish = self.last.get("finish")
                    if finish == "preempted" and not cancel.is_set() and len(output) < max_new:
                        if initial_reused is None:
                            initial_reused = self.last.get("reused")
                        carry_leg(carry, self.last)
                        prompt = [*ids, *output]
                        live["state"] = "queued"
                        self._send(f"{head} {identity} {int(max_new) - len(output)}"
                                   f"{self.gen_keys(sampling)}{pictures} " +
                                   ",".join(str(int(token)) for token in prompt), process=process)
                        continue
                    complete = True
                    if carry:
                        self.last = {**add_carry(carry, self.last), "prompt_tokens": len(ids), "prompt_ms": prompt_ms}
                        if prompt_read is not None:
                            self.last["prompt_read"] = prompt_read
                        if initial_reused is not None:
                            self.last = {**self.last, "reused": initial_reused}
                    if finish == "nonfinite":
                        raise EngineNonFinite("the engine's logits became non-finite")
                    return
        finally:
            if sent and not complete and self.proc is process and self.gen == born and self.alive():
                try:
                    if not stopped:
                        self._send(f"CANCEL {identity}", process=process)
                except EngineDied:
                    release()
                else:
                    threading.Thread(target=release_when_finished, daemon=True).start()
            else:
                release()

    def generate_batched(self, ids, max_new, sampling, cancel, embeddings=None):
        """Compatibility with released engines that do not advertise the request-ID protocol.

        Alone (no slot busy, nobody waiting): the solo path (GEN, with drafts: the fastest stream)
        - and when another request arrives meanwhile, this one is STOPped and continues in a batch slot (BGEN with
        its prompt + what it generated: the engine reuses that prefix).  Otherwise: BGEN into a free slot, then the
        slot's own BT lines until BDONE.  Several requests run at once; the control lines (prompt reading, admission)
        are taken one request at a time - and a long prompt read gives way at a chunk boundary to a waiting request
        with a much shorter prompt (#656: `BYIELD <slot>`; the part read waits in a slot and the read goes on after).
        Admission may hand off with zero tokens (DONE 0 then BADM <slot> 1); output begins with the slot's BT lines.
        A consumer that stops early leaves the engine in step: the solo request is STOPped and read to its DONE, an
        admission to its BADM, a slot is BSTOPped and freed at its BDONE."""
        if not self.alive():
            raise EngineDied("the engine is unavailable; this request was not sent")
        self.progress, self.progress_ms, self.reused = None, 0, 0
        born = self.gen                                 # the engine process this request is sent to
        keys = self.gen_keys(sampling)
        priority = get_request_priority(sampling or {})
        # logprobs: only an admission's token can have an LP line here (the slot's "BT" lines carry none), so the
        # server refuses these requests unless max_tokens is 1.  The lines go to the request itself, through its
        # cancel Event: several requests read control lines at once, and the engine's lp_lines is one attribute.
        want_lp = top_logprobs(sampling or {}) is not None
        if want_lp:
            cancel._strata_lp_lines = LPLines(cancel)
        out: list[int] = []
        pending: list[int] = []
        prompt, left = list(ids), int(max_new)
        orig_len = len(prompt)  # a leg that continues (prompt + what it generated) names where the reply began: pen_from, the penalty window's start
        reused0 = None      # the first read's reused tokens: a continued leg's count includes this request's own output
        carry = {}          # #1869: the figures of the legs that ended in a hand-back to the solo path or a preemption
        #                     (the final DONE adds to them)
        yielded_legs = []    # prompt reads that gave way before producing output
        ok = yield from self._take_control(cancel, len(prompt), new=True,
                                           prompt=prompt if (COALESCE_PREFILL > 1 and not embeddings) else None)
        if not ok:
            self._release_preslot(cancel)
            return
        holding, born = True, self.gen                  # (the engine it now has the control lines of)
        lines_q, sq = self.lines, self.slot_q           # #4451: ... and its queues; a restart makes new ones, and a request
        #                                                 of the old process must never read (or answer on) the new ones
        btrace("ctl acquired")
        slot, gen0, reserved = None, None, None
        live = None             # this request's entry in slot_live while it has a slot
        taken, victim_live = None, None   # preempting: the event slot_handover holds, the ended request's entry
        pre = getattr(cancel, "_strata_preslot", None)
        if pre is not None:                             # STRATA_COALESCE_PREFILL: a request before this one read the prompt into this slot
            slot, cancel._strata_preslot = pre, None
        phase = "none"            # solo -> (admit -> slot) ; "done" once the engine has finished with this request
        stop_sent = False
        admission_pending, admission_before = False, None   # the admission's DONE is partial until BADM 0 or BDONE
        yields, solo_again, refused = 0, 0, 0
        admit_wait, admit_since, admit_said = False, 0.0, False   # #1603: waiting for a slot, since when, said
        try:
            while True:   # a request in a slot that is left alone goes back to the solo path
                if not holding:
                    ok = yield from self._take_control(cancel, len(prompt))
                    if not ok:
                        return
                    holding, born = True, self.gen                  # (the engine it now has the control lines of)
                    lines_q, sq = self.lines, self.slot_q           # (it holds no slot of the engine before: nothing to lose)
                budget = self.info.get("tokens_max")
                if budget is not None and len(prompt) >= int(budget):
                    raise ValueError(f"the prompt has {len(prompt)} tokens; the shared GPU budget is {budget}")
                with self.slot_cv:
                    alone = not any(self.slot_busy) and self.waiting == 0
                if alone and left > 1:
                    pf = self.pen_from_key(orig_len, len(prompt))
                    head = f"GENI {left}{keys}{pf} {embeddings}" if embeddings else f"GEN {left}{keys}{pf}"
                    self._last_done = None                  # (#879: this request's DONE decides below)
                    deadline = getattr(cancel, "_strata_queue_deadline", None)
                    if deadline is not None and time.monotonic() >= deadline:
                        raise QueueWaitExpired("the inference queue wait deadline expired; retry later")
                    self._send(f"{head} {','.join(str(int(t)) for t in prompt)}")
                    if deadline is not None:
                        cancel._strata_queue_deadline = None
                    started = getattr(cancel, "_strata_queue_started", None)
                    if started is not None:
                        started()
                    phase = "solo"
                    self._ctl_mode, self._ctl_result, self._yielded = "solo", None, None
                    self._ctl_erred = False
                    def others():
                        with self.slot_cv:
                            return self.waiting > 0

                    def give_way():   # at a PP line, and between two far apart (the engine takes it at its next boundary)
                        nonlocal reserved
                        if (reserved is None and not out and not embeddings and yields < self.YIELDS_MAX and
                                self._shorter_waiting(len(prompt))):
                            with self.slot_cv:          # the slot the part read will wait in
                                reserved = self.pick_slot(prompt)
                                if reserved is not None:
                                    self.slot_busy[reserved] = True
                                    self.slot_claims.add(reserved)
                            if reserved is not None:
                                self._send_for(born, f"BYIELD {reserved}")
                    for x in self._control(cancel, pending.append, stop_when=others, on_quiet=give_way,
                                           want_lp=want_lp, lines_q=lines_q):
                        while pending:
                            t = pending.pop(0)
                            out.append(t)
                            yield t
                        if x is None:                       # a heartbeat (False: a token, flushed above)
                            give_way()
                            yield None
                    phase = "none"                          # its DONE is read
                    while pending:
                        t = pending.pop(0)
                        out.append(t)
                        yield t
                    if self._done_finish() == "nonfinite":    # #879 (never promoted to a slot)
                        raise EngineNonFinite("the engine's logits became non-finite (#879)")
                    if self._yielded is not None and reserved is not None and self._yielded[0] == reserved:
                        yielded_legs.append(self.last)
                        slot, reserved = reserved, None     # gave way: the read goes on in that slot (below)
                    elif reserved is not None:              # it did not give way: the slot is free again
                        with self.slot_cv:
                            self.slot_busy[reserved] = False
                            self.slot_claims.discard(reserved)
                            self.slot_cv.notify_all()
                        reserved = None
                    if slot is None:
                        finish = (self.last or {}).get("finish") if isinstance(self.last, dict) else None
                        left = int(max_new) - len(out)
                        if cancel.is_set() or left <= 0 or finish in ("stop", "length") or (out and out[-1] in EOS_IDS):
                            return
                        if reused0 is None:
                            reused0 = (self.last or {}).get("reused")
                        carry_leg(carry, self.last)         # #1869: the solo leg's figures; the slot's admission (a copy of
                                                            # the held prefix: prompt_read 1) and BDONE replace them
                        prompt = list(ids) + out            # promoted: it continues in a batch slot from here
                while True:
                    if slot is None:
                        # a free slot (they free themselves at BDONE, which needs no control lines): the one that holds
                        # the start of this prompt (its conversation's last turn), else the one used longest ago.
                        # With none free the control lines are given back while waiting: a slot can be held by a read
                        # that gave way (BYIELD) and needs them to go on, so waiting with them is a deadlock - long,
                        # long, short at 0.25 s steps under parallel: 2 (ENGINE_REVIEW finding 1, reproduced in 0.1.41).
                        while True:
                            deadline = getattr(cancel, "_strata_queue_deadline", None)
                            if deadline is not None and time.monotonic() >= deadline:
                                raise QueueWaitExpired("the inference queue wait deadline expired; retry later")
                            with self.slot_cv:
                                slot = self.pick_slot(prompt)
                                if (slot is None and self.can_preempt and not self.get_releasing_prefix(prompt) and
                                        self.can_admit_prompt(prompt, priority)):
                                    slot = self._pick_victim(priority)
                                    if slot is not None:
                                        taken = self.slot_handover[slot] = threading.Event()
                                        victim_live = self.slot_live[slot]
                                if slot is not None:
                                    self.slot_busy[slot] = True
                                    self.slot_claims.add(slot)
                                    if admit_wait:
                                        admit_wait = False
                                        self.admitting = max(0, self.admitting - 1)
                                        if PAR_COUNTERS:                # STRATA_PAR_COUNTERS=1: time spent waiting for a free slot
                                            self.par_counters["queued"] += 1
                                            self.par_counters["queue_wait_s"] += time.monotonic() - admit_since
                                    break
                                if not admit_wait:          # #1603: counted while it waits for a slot (it holds no
                                    admit_wait, admit_since = True, time.monotonic()   # control lines then)
                                    self.admitting += 1
                            if holding:
                                self.ctl.release()
                                holding = False
                            with self.slot_cv:
                                if self.pick_slot(prompt) is None:
                                    if (ENGINE_STALL_S or 0) > 0 and time.monotonic() - admit_since >= ENGINE_STALL_S:
                                        # the stall check of #1317 only watches a request on the control lines; one
                                        # waiting here is covered by looking for a slot nobody owns, and by saying so
                                        if not admit_said:
                                            admit_said = True
                                            busy = [b for b, v in enumerate(self.slot_busy) if v]
                                            print(f"[strata] parallel: a request has waited {time.monotonic() - admit_since:.0f} s "
                                                  f"for a free slot (busy: {busy}, in use: {sorted(self.slot_claims)}, "
                                                  f"ending: {sorted(self.slot_releasing)}) (issue #1603)", flush=True)
                                    self._reap_orphan_slots()
                                if self.pick_slot(prompt) is None:
                                    wait_remaining = deadline - time.monotonic() if deadline is not None else 1.0
                                    self.slot_cv.wait(timeout=min(1.0, max(0, wait_remaining)))
                            if cancel.is_set():
                                return
                            if not holding:
                                ok = yield from self._take_control(cancel, len(prompt))
                                if not ok:
                                    return
                                holding, born = True, self.gen
                                lines_q, sq = self.lines, self.slot_q   # (a slot is claimed below, on this engine)
                    if self._yielded is not None:           # it gave way: the others waiting then go first
                        self.slot_held[slot] = list(prompt[:self._yielded[1]])
                        self._yielded = None
                        yields += 1
                        self.ctl.release()
                        holding = False
                        with self.slot_cv:
                            epoch = self.ctl_epoch
                        ok = yield from self._take_control(cancel, len(prompt), after_epoch=epoch)
                        if not ok:
                            return
                        if self.gen != born:                    # #4451: the slot it holds was the old engine's
                            holding = True
                            raise EngineDied("the engine restarted while this request waited for it; it was not completed")
                        holding, born = True, self.gen                  # (the engine it now has the control lines of)
                    # (preempting: the slot's lines are those of the request it ends, which reads them)
                    while taken is None and not sq[slot].empty():
                        sq[slot].get_nowait()
                    pf = self.pen_from_key(orig_len, len(prompt))
                    head = f"BGENI {slot} {left}{keys}{pf} {embeddings}" if embeddings else f"BGEN {slot} {left}{keys}{pf}"
                    live = {"slot": slot, "request_id": getattr(cancel, "_strata_request_id", None),
                            "_cancel": cancel, "_prompt": prompt, "_generated_start": len(out),
                            "state": "reading", "prompt_tokens": len(prompt), "prompt_read": 0, "generated": 0,
                            "started": time.time(), "first_token": None, "priority": priority}
                    deadline = getattr(cancel, "_strata_queue_deadline", None)
                    if deadline is not None and time.monotonic() >= deadline:
                        raise QueueWaitExpired("the inference queue wait deadline expired; retry later")
                    self.slot_live[slot] = live
                    self._last_done = None                  # (#879: this admission's DONE decides below)
                    gen0 = len(out)
                    if COALESCE_PREFILL > 1 and not embeddings:
                        self._coalesce(cancel, prompt)
                    self._send(f"{head} {','.join(str(int(t)) for t in prompt)}")
                    if deadline is not None:
                        cancel._strata_queue_deadline = None
                    started = getattr(cancel, "_strata_queue_started", None)
                    if started is not None:
                        started()
                    self.slot_held[slot] = []               # the admission overwrites what the slot held
                    admission_pending, admission_before = True, self.last
                    phase = "admit"
                    self._ctl_mode, self._ctl_result, self._yielded = "batch", None, None
                    self._ctl_erred = False
                    asked = False

                    def ask():        # as give_way above, for the slot this admission reads into
                        nonlocal asked
                        if (not asked and not embeddings and yields < self.YIELDS_MAX and
                                self._shorter_waiting(len(prompt))):
                            self._send_for(born, f"BYIELD {slot}")
                            asked = True

                    def note_progress():
                        # the terminal layer's live prompt counter (the batch admission path)
                        if self.progress and live["state"] == "reading":
                            live["prompt_read"] = self.progress[0]
                    try:
                        for x in self._control(cancel, pending.append, on_quiet=lambda: (ask(), note_progress()),
                                               want_lp=want_lp, lines_q=lines_q):
                            while pending:
                                t = pending.pop(0)
                                out.append(t)
                                yield t
                            if x is None:
                                ask()
                                note_progress()
                                yield None
                    except ValueError as e:
                        # #1994: the engine says this slot is not free (its view and ours differ for a moment, or a
                        # slot it still holds).  That is not the request's fault and nothing was read: park the slot
                        # for a moment (busy here too), and wait for another one - never a 400 for the conversation.
                        if not (self._ctl_erred and str(e).startswith("BGEN:") and left >= 1 and not cancel.is_set()):
                            raise
                        refused += 1
                        print(f"[strata] parallel: the engine refused slot {slot} ({e}); trying another slot "
                              f"({refused}/{self.BGEN_REFUSALS_MAX}) (issue #1994)", flush=True)
                        if taken is not None:           # the request it would have ended keeps the slot
                            with self.slot_cv:
                                self.slot_handover.pop(slot, None)
                                self.slot_live[slot] = victim_live
                            taken, slot = None, None
                        if refused > self.BGEN_REFUSALS_MAX:
                            raise GpuBusy("the engine's batch slots are not free right now; try again in a moment",
                                      retry_after=5) from e
                        phase = "none"
                        self._ctl_erred = False
                        if slot is not None:
                            self.slot_live[slot] = None
                            self.slot_held[slot] = []
                            self._park_slot(slot)
                        slot = None
                        continue
                    cont = bool(self._ctl_result and self._ctl_result[1])
                    admission_pending = cont
                    phase = "slot" if cont else "none"
                    while pending:
                        t = pending.pop(0)
                        out.append(t)
                        yield t
                    if not cont and self._done_finish() == "nonfinite":   # #879
                        raise EngineNonFinite("the engine's logits became non-finite (#879)")
                    if not cont and self._yielded is not None and self._yielded[0] == slot and not cancel.is_set():
                        yielded_legs.append(self.last)
                        continue                            # gave way: again once the shorter request is in
                    break
                self.ctl.release()
                holding = False
                if taken is not None:   # the slot's lines are the ended request's up to its BDONE
                    yield from self._await_handover(taken, born)
                    with self.slot_cv:
                        self.slot_handover.pop(slot, None)
                    taken = None
                finish, going_solo = self._done_finish(), False
                if cont:
                    live.update(state="decoding", first_token=time.time() if len(out) > gen0 else None,
                                generated=len(out))
                    while True:
                        if cancel.is_set() and not stop_sent:
                            self._stop_slot(slot, live, born)
                            stop_sent = True
                        try:
                            line = sq[slot].get(timeout=10.0)
                        except queue.Empty:
                            if self.gen != born:            # #4451: the engine it was admitted on is gone (and so is its slot)
                                phase = "none"
                                raise EngineDied("the engine restarted while this request was running; it was not completed")
                            if cancel.is_set() and not stop_sent:
                                self._stop_slot(slot, live, born)
                                stop_sent = True
                            yield None
                            continue
                        if line is None:
                            phase = "none"
                            raise EngineDied(f"the engine stopped unexpectedly (exit code {self.exit_code()})")
                        if line.startswith("BT "):
                            t = int(line.split()[2])
                            out.append(t)
                            if live["first_token"] is None:
                                live["first_token"] = time.time()
                            live["generated"] = len(out)
                            if cancel.is_set():
                                if not stop_sent:
                                    self._stop_slot(slot, live, born)
                                    stop_sent = True
                                continue
                            yield t
                            if (not going_solo and not stop_sent and
                                    self._may_go_solo(int(max_new) - len(out), solo_again, embeddings)):
                                going_solo = self._stop_slot(slot, live, born)   # alone now: on with the drafts (below)
                        elif line.startswith("BDONE "):
                            phase = "none"
                            f = line.split()
                            self._apply_bdone(f)
                            admission_pending = False
                            with self.slot_cv:
                                taken_by = self.slot_handover.get(slot)
                                if taken_by is None:
                                    live["state"] = "ending"
                            if taken_by is not None:        # a request of a higher priority has the slot now
                                taken_by.set()
                                slot = None
                            else:   # The state holds every input and output except the next token to feed.
                                self.slot_held[slot] = (list(prompt) + out[gen0:])[:-1]
                            if f[3:4] == ["nonfinite"]:              # #879: the engine dropped the slot's state
                                if slot is not None:
                                    self.slot_held[slot] = []
                                raise EngineNonFinite("the engine's logits became non-finite (#879)")
                            finish = f[3] if len(f) > 3 else "stop"
                            break
                if ((finish == "preempted" or (going_solo and finish == "cancel")) and not cancel.is_set()
                        and len(out) < int(max_new) and not (out and out[-1] in EOS_IDS)):
                    # A preempted request waits again with its existing output, including a zero-token admission.
                    carry_leg(carry, self.last)         # #1869: the next leg's DONE replaces these figures
                    if slot is not None:
                        self.slot_live[slot] = None
                        self.slot_used[slot] = time.time()
                        with self.slot_cv:
                            self.slot_busy[slot] = False
                            self.slot_claims.discard(slot)
                            self.slot_cv.notify_all()
                    slot, gen0 = None, None
                    if reused0 is None:
                        reused0 = (self.last or {}).get("reused")
                    prompt, left = list(ids) + out, int(max_new) - len(out)
                    solo_again += going_solo
                    btrace("preempted" if finish == "preempted" else "back to the solo path")
                    continue
                return
        finally:
            # a consumer that left early (or an error): keep the engine and this server in step
            btrace("finally phase", phase, "slot", slot, "holding", holding)
            same = self.gen == born                     # #4451: the slot state below is this engine's; after a restart it is the NEW
            #                                             engine's lists and flags, which belong to other requests
            undrained = None                            # the line this request's exchange still owes, from an engine that runs
            try:
                if phase in ("solo", "admit") and (self.gen != born or not self.alive()):
                    phase = "none"                      # #1012: its engine is gone: nothing to stop or drain
                if phase in ("solo", "admit") and self._ctl_erred:
                    phase = "none"                      # #1059: it ended on the engine's ERR: no DONE to wait 300 s for
                if phase == "solo":
                    self._send("STOP")
                    if self._drain_control("DONE", born=born, lines_q=lines_q) is None and self.gen == born and self.alive():
                        undrained = "DONE"
                elif phase == "admit":
                    line = self._drain_control("BADM", born=born, lines_q=lines_q)
                    if line and line.startswith("BADM "):
                        admission_pending = line.split()[2:3] == ["1"]
                    if line and line.startswith("BADM ") and line.split()[2:3] == ["1"]:
                        phase = "slot"
                    elif line is None and self.gen == born and self.alive():
                        undrained = "BADM"
            except EngineDied:
                pass
            handed = False
            if undrained is not None:
                # #4451: the engine has not said it is done with this request; whatever it prints later must not be read
                # by the next one.  The control lines (and the admission's slot) stay quarantined until it does.
                self._quarantine_control(undrained, born, lines_q, slot if undrained == "BADM" else None, holding)
                holding, handed = False, True
            if taken is not None and same:      # it preempted, then failed before the hand-over
                with self.slot_cv:
                    self.slot_handover.pop(slot, None)
            if phase == "slot" and slot is not None and same and not handed:
                with self.slot_cv:
                    taken_by = self.slot_handover.get(slot)
                    if taken_by is None:
                        live["state"] = "ending"
                if taken_by is not None:        # a request of a higher priority takes the slot once its BDONE is read
                    self._release_slot_when_done(slot, taken=taken_by, born=born)
                    slot = None
            bdone_in = False
            if phase == "slot" and slot is not None and out and out[-1] in EOS_IDS and same:
                bd = self._bdone_after_eos(slot, q=sq[slot])    # (#1869) the slot's figures, not the admission's
                if bd is not None:
                    self._apply_bdone(bd.split())
                    admission_pending = False
                    bdone_in = True
            ended_on_yield = bool(yielded_legs and self.last is yielded_legs[-1])
            if admission_pending and isinstance(self.last, dict) and (self.last is not admission_before or carry):
                # A background drain owns the rest. The admission's one token and time are not this request's totals.
                self.last = {**self.last, "generated": None, "decode_ms": None}
            if reused0 is not None and isinstance(self.last, dict):
                self.last = {**self.last, "reused": reused0}
            if carry and isinstance(self.last, dict):
                self.last = add_carry(carry, self.last)
            earlier_reads = yielded_legs[:-1] if ended_on_yield else yielded_legs
            if earlier_reads and isinstance(self.last, dict):
                self.last = prefill_leg_stats(earlier_reads, self.last, len(ids))
            if holding:
                self.ctl.release()
            if admit_wait:
                with self.slot_cv:
                    self.admitting = max(0, self.admitting - 1)
            if reserved is not None and same:
                with self.slot_cv:
                    self.slot_busy[reserved] = False
                    self.slot_claims.discard(reserved)
                    self.slot_cv.notify_all()
            if slot is not None and same and not handed:
                if phase == "slot" and not bdone_in:
                    stream = list(prompt) + out[gen0:] if gen0 is not None and len(out) > gen0 else list(prompt)
                    with self.slot_cv:
                        self.slot_held[slot] = stream[:-1]
                        self.slot_releasing.add(slot)
                self.slot_live[slot] = None
                self.slot_used[slot] = time.time()
                if phase == "slot" and bdone_in:
                    stream = list(prompt) + out[gen0:] if gen0 is not None and len(out) > gen0 else None
                    self.slot_held[slot] = stream[:-1] if stream else []     # as the BDONE wait leaves it
                    with self.slot_cv:
                        self.slot_busy[slot] = False
                        self.slot_claims.discard(slot)
                        self.slot_cv.notify_all()
                elif phase == "slot":
                    self._release_slot_when_done(slot, stream, born=born)    # freed at its BDONE
                    self.slot_claims.discard(slot)                # (the BDONE wait owns it now)
                else:
                    with self.slot_cv:
                        self.slot_busy[slot] = False
                        self.slot_claims.discard(slot)
                        self.slot_cv.notify_all()

    slot_groups = 1         # --batch-groups (set when the engine starts)
    SLOT_PREFIX_MIN = 512   # a held prefix shorter than this is not worth unbalancing the pipeline's groups
    BRANCH_BOUNDARY_MIN = 8192   # the engine's message-boundary checkpoint (STRATA_CACHE_MESSAGE_BOUNDARY) starts at this many tokens
    TURN_ID = 248045        # <|im_start|>: the engine's default --turn-token (its turn checkpoint sits at the last one)

    def pick_slot(self, prompt: list[int]) -> int | None:
        """A free slot for `prompt` (the caller holds slot_cv): the one whose held tokens are the longest start of the
        prompt (the engine then reads only the rest), else an empty one, else the one used longest ago - so the
        conversations other slots hold stay for their next turns. None: all slots are busy, or a cancelling slot
        must finish before its longer cached prefix can be reused."""
        free = [b for b in self.slot_order if not self.slot_busy[b]]
        if not free or not self.can_admit_prompt(prompt):
            return None
        gs = getattr(self, "slot_gs", 0)
        if gs > 1:
            # pipelined groups of several slots: the engine keeps no idle slot below a busy one as a conversation cache
            # (its pad row writes the slot), so nothing is worth keeping free for a next turn.  A group's window spans
            # its slots up to its last busy one, so first fill a hole below a busy slot of the same group (it costs no
            # extra row), then follow the spreading order
            # (reported on #793: least-recently-used picks left leading holes, [_ B], that cost a whole row each)
            def fills_hole(b):
                g0 = b - b % gs
                return any(self.slot_busy[x] for x in range(b + 1, g0 + gs))
            return min(free, key=lambda b: (not fills_hole(b), self.slot_order.index(b)))
        agreed = {}
        def agree(b):
            # how many leading tokens slot b's held stream and the prompt share (slice compares: streams reach 100k tokens)
            if b not in agreed:
                h = self.slot_held[b]
                lo, hi = 0, min(len(h), len(prompt))
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if h[:mid] == prompt[:mid]:
                        lo = mid
                    else:
                        hi = mid - 1
                agreed[b] = lo
            return agreed[b]
        def held_prefix(b):
            # what of `prompt` the slot's cache serves: get_cached_prefix's whole stream or turn checkpoint (#2518), else
            # a branch: the same history with the last question replaced agrees with the held stream only up to an
            # EARLIER message boundary, where the engine may hold a checkpoint of this chat
            # (STRATA_CACHE_MESSAGE_BOUNDARY, prefixes of at least BRANCH_BOUNDARY_MIN tokens).  The engine finds that
            # source itself but copies it into the slot chosen here, so another chat's slot would be overwritten (#2518).
            h = self.slot_held[b]
            cached = self.get_cached_prefix(h, prompt)
            if cached or self.TURN_ID not in h:
                return cached
            last = len(h) - 1 - h[::-1].index(self.TURN_ID)
            i = min(agree(b), len(prompt) - 1, last - 1)
            if i < self.BRANCH_BOUNDARY_MIN:
                return 0
            window = h[self.BRANCH_BOUNDARY_MIN:i + 1]
            if self.TURN_ID not in window:
                return 0
            i -= window[::-1].index(self.TURN_ID)
            # a boundary every other held conversation shares as well (the system prompt and tools of one client) says
            # nothing about whose it is: the old order (an empty slot, then the least recent one) keeps all chats
            if not any(agree(c) < i for c in range(self.batch) if c != b and self.slot_held[c]):
                return 0
            return i
        best = max(free, key=held_prefix)
        prefix = held_prefix(best)
        if self.get_releasing_prefix(prompt) > prefix:
            return None
        # without --batch-groups a held prefix of any length is taken, as before
        if prefix >= (self.SLOT_PREFIX_MIN if self.slot_groups > 1 else 1):
            return best
        # with --batch-groups: the group with the fewest busy slots first, so the requests spread over the
        # pipeline's stages (two requests in one group leave the other stage idle); then empty, then least recent
        busy_in = {}
        for b in range(self.batch):
            if self.slot_busy[b]:
                busy_in[self.slot_group[b]] = busy_in.get(self.slot_group[b], 0) + 1
        return min(free, key=lambda b: (busy_in.get(self.slot_group[b], 0) if self.slot_groups > 1 else 0,
                                        bool(self.slot_held[b]), self.slot_used[b]))

    def can_admit_prompt(self, prompt: list[int], priority: int | None = None) -> bool:
        """Checks GPU residency and token allowances against the requests already active.

        A higher-priority admission may displace lower-priority requests. Idle cached conversations can be paused
        by the engine and do not reserve admission capacity. Callers hold slot_cv.
        """
        budget, limit = self.info.get("tokens_max"), self.info.get("vram_conversations")
        if budget is None and limit is None:
            return True
        if budget is not None and len(prompt) >= int(budget):
            return False
        used, active = len(prompt), 0
        for live in self.slot_live:
            if not live or live.get("state") == "ending":
                continue
            if priority is not None and live.get("priority", 0) < priority:
                continue
            active += 1
            if budget is not None:
                generated = max(0, live.get("generated", 0) - live.get("_generated_start", 0))
                used += live["prompt_tokens"] + generated - 1
        return (limit is None or active < int(limit)) and (budget is None or used <= int(budget))

    def get_releasing_prefix(self, prompt: list[int]) -> int:
        """Gets the longest cached-prefix hint in cancelling slots; callers hold slot_cv. Live forks do not wait."""
        if getattr(self, "slot_gs", 0) > 1:
            return 0
        best = 0
        for b, live in enumerate(self.slot_live):
            if b in self.slot_handover:
                continue
            cancel = live.get("_cancel") if live else None
            if b in self.slot_releasing:
                stream = self.slot_held[b]
            elif live and (live.get("state") == "ending" or (cancel and cancel.is_set())):
                stream = live.get("_prompt", [])
            else:
                continue
            best = max(best, self.get_cached_prefix(stream, prompt))
        return best

    def get_cached_prefix(self, held: list[int], prompt: list[int]) -> int:
        """Estimates cached tokens from a full prefix match or the engine's reusable history boundary."""
        if not held:
            return 0
        if len(held) < len(prompt) and prompt[:len(held)] == held:
            return len(held)
        boundary = len(held) - 1 - held[::-1].index(self.TURN_ID) if self.TURN_ID in held else 0
        if boundary > 0 and self._tail_role_token >= 0:
            previous = next((i for i in range(boundary - 1, 0, -1) if held[i] == self.TURN_ID), 0)
            if previous and held[previous + 1] == self._tail_role_token:
                boundary = previous
        return boundary if 0 < boundary < len(prompt) and prompt[:boundary] == held[:boundary] else 0

    def slots_view(self) -> list[dict]:
        """Gets active requests, or fixed execution slots for engines using the legacy protocol."""
        now, view = time.time(), []
        requests = getattr(self, "request_protocol", False)
        if requests:
            with self.request_lock:
                live = [dict(r) for r in self.slot_live if r is not None]
            held = []
        else:
            with self.slot_cv:
                live = [dict(r) if r else None for r in self.slot_live]
                held = [len(tokens) for tokens in self.slot_held]
        for b, r in enumerate(live):
            if r is None:
                view.append({"slot": b, "state": "idle", "held_tokens": held[b]})
                continue
            ft = r.get("first_token")
            view.append({"slot": b, "state": r["state"], "prompt_tokens": r["prompt_tokens"],
                         "prompt_read": r.get("prompt_read") if r["state"] == "reading" else None,
                         "generated": r["generated"], "elapsed_s": round(now - r["started"], 1),
                         "tok_s": round(r["generated"] / max(1e-6, now - ft), 1) if ft else None})
            if requests:
                view[-1].update(request_id=r["request_id"], reused=r.get("reused"))
        return view

    def generate(self, ids, max_new, sampling, cancel, embeddings=None, prefill_only=False):
        """Yields token ids, and None as a heartbeat every 10 s while the engine is quiet (reading a long prompt):
        the HTTP layer turns it into an SSE comment, which keeps clients' watchdogs calm and notices a client that
        has gone.  A consumer that stops early (or `cancel`) makes the engine STOP, so it does not run to max_new."""
        if not self.alive():
            raise EngineDied("the engine is unavailable; this request was not sent")
        bias_api.validate_request(sampling or {}, self, None)
        refuse_for_split(self, sampling or {})
        if getattr(self, "request_protocol", False):
            yield from self.generate_request(ids, max_new, sampling, cancel, embeddings)
            return
        if getattr(self, "batch", 0) and not self.multi:
            yield from self.generate_batched(ids, max_new, sampling, cancel, embeddings)
            return
        self.progress, self.progress_ms, self.reused = None, 0, 0
        self.prefill_tok_s_mean = None
        # DeepSeek --parallel: the engine holds several conversations; this request is one of them (GEN ... rid=<id>), its
        # lines come on a queue of their own and its STOP names it.  One at a time (self.multi 0): the one queue, as before.
        rid, lines_q, stop_line = None, self.lines, "STOP\n"
        if self.multi:
            with self.rid_lock:
                self.rid_n += 1
                rid = self.rid_n
            lines_q = queue.Queue()
            self.rid_q[rid] = lines_q
            stop_line = f"STOP {rid}\n"
        # logprobs: the engine writes "LP <logprob> [id:logprob ...]" right after each "T <id>" (the same
        # flush), read here before the token is yielded, so whoever takes the token finds its line in
        # lp_lines.  Not asked for = the deque stays empty and nothing on this path changes.
        want_lp = top_logprobs(sampling or {}) is not None
        self.lp_lines = LPLines(cancel)
        carry = NO_LINE                  # a line read for a token's LP line, held until it is used
        # an image request takes the same sampling keys as text (#75: it used to decode greedily whatever was asked)
        rid_key = f" rid={rid}" if rid is not None else ""
        head = f"GENI {int(max_new)}{self.gen_keys(sampling)}{rid_key} {embeddings}" if embeddings else \
            f"GEN {int(max_new)}{self.gen_keys(sampling)}{rid_key}"
        if prefill_only:                                 # split serving (strata_server/split.py): read the prompt, make no token
            if embeddings or int(max_new) != 0:
                raise ValueError("prefill_only is a text prompt and max_new 0")
            head = f"GEN 0 prefill=1{rid_key}"
        try:
            if rid is not None:                          # several requests write to the engine at once
                self._send(f"{head} {','.join(str(int(t)) for t in ids)}")
            else:
                self.proc.stdin.write(f"{head} {','.join(str(int(t)) for t in ids)}\n")
                self.proc.stdin.flush()
        except (OSError, EngineDied):                    # the pipe is gone: the engine died (not the client)
            if rid is not None:
                self.rid_q.pop(rid, None)
            raise EngineDied(f"the engine stopped unexpectedly (exit code {self.exit_code()})") from None
        done = False
        # #481: how long the engine may stay silent from here.  Until the first line: the request's first prompt
        # chunk at the slowest prompt reading on top of silence_s; a PP line resets it to its own chunk's time.
        silence = float(self.silence_s or 0)
        allow = silence + min(len(ids), PP_CHUNK_MAX) / PP_FLOOR_TOK_S if silence > 0 else 0.0
        heard, read_to = time.monotonic(), 0
        beat = heard
        stall_s = ENGINE_STALL_S or 0.0
        stall_state: dict = {}
        try:
            while True:
                wait = 10.0
                if allow > 0:
                    wait = min(wait, allow - (time.monotonic() - heard))
                    if wait <= 0:
                        done = True                       # no STOP and no drain: nothing is listening
                        raise self._silent(f"the engine said nothing for {time.monotonic() - heard:.0f} s during "
                                           "the request")
                try:
                    if carry is not NO_LINE:              # the line read after a token for its LP line
                        line, carry = carry, NO_LINE
                    else:
                        line = lines_q.get(timeout=min(wait, 0.5))
                except queue.Empty:
                    if cancel.is_set():
                        return
                    if stall_s > 0:
                        try:
                            self._stall_check(heard, stall_state, stall_s)
                        except EngineSilent:
                            done = True                   # as above: nothing is listening
                            raise
                    if allow > 0 and time.monotonic() - heard >= allow:
                        continue                          # enforce the expired deadline before another heartbeat
                    if time.monotonic() - beat >= 10.0:
                        beat = time.monotonic()
                        yield None                        # polling cancellation does not add HTTP heartbeats
                    continue
                if line is None:
                    done = True
                    raise EngineDied(f"the engine stopped unexpectedly (exit code {self.exit_code()})")
                heard = time.monotonic()                  # any line is output: T, PP, RESUME, INFO ...
                beat = heard
                if stall_state:
                    stall_state.clear()
                if line.startswith("T "):
                    allow = silence
                    if cancel.is_set():
                        return
                    t = int(line[2:])
                    if want_lp:
                        # the engine writes the token and its `LP` line in one flush, so the line is
                        # here now; an engine without logprobs leaves the next line a normal one
                        try:
                            carry = lines_q.get(timeout=0.5)
                        except queue.Empty:
                            carry = NO_LINE
                        if isinstance(carry, str) and carry.startswith("LP "):
                            self.lp_lines.append((t, carry))
                            carry = NO_LINE
                    yield t
                elif line.startswith("PP "):
                    f = line.split()
                    if len(f) >= 3 and f[1].isdigit() and f[2].isdigit():
                        self.progress = (int(f[1]), int(f[2]))             # prompt progress, one per chunk: also a heartbeat (the
                        self.progress_ms = float(f[3]) if len(f) >= 4 else 0.0
                        self.prefill_tok_s_mean = float(f[4]) if len(f) >= 5 else None
                        rate, chunk = self.prefill_tok_s_mean or 0.0, int(f[1]) - read_to
                        read_to = int(f[1])
                        if silence > 0 and rate > 0 and chunk > 0:   # #481: the next chunk, as long as this one
                            allow = max(silence, PP_SLACK * chunk / rate)
                    if cancel.is_set():                   # lines reset the 10 s wait, so without this a long prompt
                        return                            # would send no keep-alives at all)
                    yield None
                elif line.startswith("RESUME "):          # the reused tokens: the first chunk starts after them
                    try:
                        read_to = self.reused = int(line.split()[1])
                    except (IndexError, ValueError):
                        pass
                elif line.startswith("DONE"):
                    self._parse_done(line)
                    done = True
                    if self.last.get("finish") == "nonfinite":   # #879
                        raise EngineNonFinite("the engine's logits became non-finite (#879)")
                    return
                elif line.startswith("ERR"):
                    done = True
                    self._raise_if_fatal_gpu(line)
                    self._swallow_trailing_done(lines_q)  # an engine that ended the request on ERR may still print its DONE
                    raise ValueError(line[4:].strip())
        finally:
            if not done:                                  # the consumer stopped early: stop the engine, drain to DONE
                if self.can_stop:
                    try:
                        self.proc.stdin.write(stop_line)
                        self.proc.stdin.flush()
                    except OSError:
                        pass
                # #481: never an untimed wait here - it holds the request FIFO, and an engine that lost step never
                # answers.  An engine that honours STOP gets the current allowance in all (a STOP during a prompt
                # chunk is seen after it); an older one runs on to max_new, so each line only has to come in time.
                heard = time.monotonic()
                while True:
                    left = allow - (time.monotonic() - heard) if allow > 0 else None
                    try:
                        if carry is not NO_LINE:          # read already, after the last token
                            line, carry = carry, NO_LINE
                        elif left is not None and left <= 0:
                            raise queue.Empty
                        else:
                            line = lines_q.get(timeout=left)
                    except queue.Empty:
                        raise self._silent("the engine did not finish the request after it was stopped (STOP) "
                                           f"within {allow:.0f} s") from None
                    if line is None or line.startswith("ERR"):
                        break
                    if line.startswith("DONE"):
                        self._parse_done(line)
                        break
                    if not self.can_stop:
                        heard = time.monotonic()
            if rid is not None:                           # this request's lines are over (a late one goes nowhere)
                self.rid_q.pop(rid, None)

    @staticmethod
    def _swallow_trailing_done(lines_q, wait: float = 0.3) -> None:
        """A request that ends on `ERR ...` is over, but an engine that also prints the request's `DONE` after it (the DeepSeek
        engine did, after its NaN stop) would hand that DONE to the NEXT request as its own: 0 tokens at once, and every answer
        after it one request late.  So a short wait for a trailing DONE (or ERR) of the same request, taken off the queue and
        dropped, before the request is reported failed and the FIFO released.  An engine that prints only the ERR costs the wait."""
        end = time.monotonic() + wait
        while True:
            left = end - time.monotonic()
            if left <= 0:
                return
            try:
                late = lines_q.get(timeout=left)
            except queue.Empty:
                return
            if late is None or late.startswith(("DONE", "ERR")):
                return

    def _activity(self) -> tuple[float, int] | None:
        """#1317: (CPU seconds, I/O bytes) the engine process has used so far, or None when they cannot be read."""
        try:
            import psutil
            pr = psutil.Process(self.proc.pid)
            t = pr.cpu_times()
            io = pr.io_counters()
            return (t.user + t.system, io.read_bytes + io.write_bytes)
        except Exception:  # noqa: BLE001 - psutil missing, the process gone, no permission: no reading
            return None

    def _stall_check(self, heard: float, state: dict, stall_s: float) -> None:
        """#1317: called while the engine is silent.  Takes a baseline soon after the last line, and at stall_s of silence
        compares: no CPU and no I/O in all that time -> the engine is ended (EngineSilent); work going on -> one warning."""
        age = time.monotonic() - heard
        if state.get("base") is None:
            state["base"] = self._activity()
            return
        if age < state.get("next", stall_s) or state["base"] is None:
            return
        now = self._activity()
        try:
            gpu_working = bool(self.gpu_busy()) if self.gpu_busy is not None else False
        except Exception:  # noqa: BLE001 - a telemetry hiccup must not decide anything
            gpu_working = False
        if now is not None and not gpu_working and engine_frozen(state["base"], now):
            raise self._silent(f"the engine said nothing for {age:.0f} s and used no CPU or disk in that time (frozen; "
                               "STRATA_ENGINE_STALL_S sets this, 0 = off)")
        if now is not None:   # working: look again after another window from here, and say so once
            state["base"], state["next"] = now, age + stall_s
            if not state.get("warned"):
                state["warned"] = True
                print(f"[strata] the engine has said nothing for {age:.0f} s but is still working; it is not ended (the "
                      "limit is engine_silence_s). If it is stuck, \"engine_silence_s\" can be set lower.", flush=True)
        else:
            state["next"] = float("inf")   # no reading (psutil missing): nothing to decide on

    def _silent(self, what: str) -> EngineSilent:
        """#481: end an engine that lost step with the server (its main thread waits for a command the server never
        sends), so the next request starts it again: killed now, its GPU and RAM go back with the process."""
        self.silent_note = ("The engine and the server lost step (issue #481; a very slow PC can raise "
                            "\"engine_silence_s\" in the config, 0 = wait forever). If it happens again, please add "
                            "the end of the engine log to github.com/Niko1221/Strata/issues/481.")
        self.ended = True                               # not alive from now: the next request restarts it
        proc = self.proc
        try:
            proc.kill()
            proc.wait(timeout=20)                       # restart() -> close() handles one that is still exiting
        except (OSError, AttributeError, subprocess.TimeoutExpired):
            pass
        return EngineSilent(f"{what}; the server ended the engine")

    def end_for_restart(self) -> None:
        """End the engine now and mark it ended, so the next request starts it again (#3961: a run that stopped without its end event left the two sides
        out of step; the restart is what puts them back)."""
        self.ended = True
        try:
            self.proc.kill()
            self.proc.wait(timeout=20)
        except (OSError, AttributeError, subprocess.TimeoutExpired):
            pass

    def session_file(self, action: str, path: str) -> dict:
        with self._exclusive_control():
            return self._session_file(action, path)

    def _session_file(self, action: str, path: str) -> dict:
        """Disk sessions: `SAVE <path>` / `RESTORE <path>` between requests (the caller holds the service FIFO).
        -> {"tokens", "bytes", "ms"}.  SessionRefused (kind, published) when the engine refused or failed the file and
        is still in step (`SERR <kind> <0|1> <reason>`).  Any line this exchange does not allow - an unknown or
        malformed one, a bare ERR, an answer with negative counts or a non-finite time - means the two sides lost
        step: the engine is ended (EngineDied) and the next request starts it again.  EngineDied also when it ended
        by itself (a restore transfer that failed after the device state changed ends it on purpose: FATAL), and
        EngineSilent (#481) when it printed nothing valid for engine_silence_s - a SESSION progress line (one per
        block of the file moved, at most 16 MiB) counts only when its numbers are.  `SWAIT <phase> <seconds>` announces
        a step that blocks in one call (a flush, the device transfer): until the next line the wait is that step's
        explicit allowance (at most SESSION_WAIT_MAX_S) when it is longer than engine_silence_s - no more, so a step
        that never ends is still ended."""
        if action not in ("save", "restore") or any(c in path for c in "\r\n\0"):
            raise ValueError("invalid session command")
        try:
            self.proc.stdin.write(f"{'SAVE' if action == 'save' else 'RESTORE'} {path}\n")
            self.proc.stdin.flush()
        except OSError:
            raise EngineDied(f"the engine stopped unexpectedly (exit code {self.exit_code()})") from None
        silence = float(self.silence_s or 0)
        heard = time.monotonic()
        want = "SAVED" if action == "save" else "RESTORED"

        def out_of_step(line: str) -> EngineDied:
            self.ended = True                               # not alive from now: the next request restarts it
            try:
                self.proc.kill()
                self.proc.wait(timeout=20)
            except (OSError, AttributeError, subprocess.TimeoutExpired):
                pass
            return EngineDied(f"the engine answered a session {action} with {line.strip()[:200]!r}; "
                              "the server ended the engine")

        def count(s: str) -> int:
            if not s.isdigit():                             # no sign, no blank, no fraction
                raise ValueError
            return int(s)

        last_done = -1
        allow = 0.0                                         # the announced blocking step's allowance, seconds
        while True:
            limit = max(silence, allow)
            left = limit - (time.monotonic() - heard) if silence > 0 else None
            try:
                if left is not None and left <= 0:
                    raise queue.Empty
                line = self.lines.get(timeout=left)
            except queue.Empty:
                raise self._silent(f"the engine said nothing for {limit:.0f} s during a session {action}"
                                   + (f" (in a blocking step allowed {allow:.0f} s)" if allow > silence else "")) from None
            if line is None:
                raise EngineDied(f"the engine stopped unexpectedly (exit code {self.exit_code()})")
            f = line.split()
            head = f[0] if f else ""
            if head == "SESSION":                          # a large file moving: progress, and a heartbeat
                try:
                    if len(f) != 3:
                        raise ValueError
                    done, total = count(f[1]), count(f[2])
                    if done > total or done < last_done:
                        raise ValueError
                except ValueError:
                    raise out_of_step(line) from None
                last_done = done
                heard, allow = time.monotonic(), 0.0
                continue
            if head == "SWAIT":                            # a blocking step starts: its bounded allowance
                try:
                    if len(f) != 3 or not f[1].isalpha() or not f[1].isascii():
                        raise ValueError
                    seconds = count(f[2])
                    if not 0 < seconds <= SESSION_WAIT_MAX_S:
                        raise ValueError
                except ValueError:
                    raise out_of_step(line) from None
                heard, allow = time.monotonic(), float(seconds)
                continue
            if head == "FATAL":
                raise EngineDied(line[6:].strip() or "the engine ended during a session restore")
            if head == "SERR":
                parts = line.rstrip("\r\n").split(" ", 3)
                if len(parts) != 4 or parts[1] not in SessionRefused.STATUS or parts[2] not in ("0", "1"):
                    raise out_of_step(line)
                raise SessionRefused(parts[1], parts[3].strip(), parts[2] == "1")
            if head == want:
                try:
                    if len(f) != 4:
                        raise ValueError
                    tokens, size, ms = count(f[1]), count(f[2]), float(f[3])
                    if not math.isfinite(ms) or ms < 0:
                        raise ValueError
                except ValueError:
                    raise out_of_step(line) from None
                return {"tokens": tokens, "bytes": size, "ms": ms}
            if head == "INFO" or head == "WARN":          # engine log lines may interleave; they are not answers
                heard = time.monotonic()
                continue
            raise out_of_step(line)

    def close(self):
        """End the engine process: STOP + QUIT first (the engine frees its memory itself - unpinning tens of GB can take
        a while; ENGINE_QUIT_WAIT_S), then terminate (ENGINE_TERM_WAIT_S), then kill (ENGINE_KILL_WAIT_S).  Raises
        EngineStuck when it still runs after all three."""
        if self.proc is None:
            return
        self.silent_note = self.silent_note or ("The server ended it on purpose (Strata was stopping, unloading or "
                                                "restarting the engine); it did not crash.")
        try:
            if self.proc.poll() is None:
                try:
                    # STOP first: a request still running ends at its next step instead of being waited for (the
                    # engine reads stdin on its own thread; QUIT behind a long answer would sit in its queue)
                    self.proc.stdin.write("STOP\nQUIT\n")
                    self.proc.stdin.flush()
                    self.proc.stdin.close()  # Windows' detached stdin reader must see EOF before shutdown
                    self.proc.wait(timeout=ENGINE_QUIT_WAIT_S +
                                   (ENGINE_SAVE_WAIT_S if self.info.get("save_on_shutdown") == 1 else 0))
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    self.proc.terminate()
                    self.proc.wait(timeout=ENGINE_TERM_WAIT_S)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            try:
                self.proc.wait(timeout=ENGINE_KILL_WAIT_S)
            except subprocess.TimeoutExpired:
                raise EngineStuck("Strata is still releasing GPU/RAM; retry unloading after it exits") from None
        except OSError:
            pass
        finally:
            if self.proc.poll() is not None:
                if self.pump is not None and self.pump.ident is not None:   # a restart can stop before starting it
                    self.pump.join(timeout=2)
                try:
                    self.proc.stdin.close()             # buffered data may flush again after the engine has exited
                except OSError:
                    pass
                self.proc.stdout.close()
                if self.log not in (None, subprocess.DEVNULL):
                    self.log.close()
                self.proc = None
                self.ended = True
                self.progress, self.last = None, {}
                self.progress_ms, self.reused = 0, 0


def gpu_list(cfg: dict) -> list[int]:
    """The config's "gpu": one card (2), or several for a layer split ([0, 2] or "0,2"), numbered as nvidia-smi
    numbers them; [] when it names none."""
    roles, _ = gpu_roles.saved_roles(cfg)       # "gpu_roles" (Main / Helper / Off per card) wins unless "gpu" was edited by hand
    if roles is not None:
        legacy = gpu_roles.card_numbers(cfg.get("gpu"))
        return legacy if legacy else gpu_roles.engine_order(roles)     # Main LAST; Off cards left out; [] when every card is Off
    return gpu_roles.card_numbers(cfg.get("gpu"))


def gpu_roles_active(cfg: dict) -> bool:
    """True when the run settings carry usable "gpu_roles": the card order is the person's (Main last), never re-sorted by speed."""
    return gpu_roles.saved_roles(cfg)[0] is not None


def gpu_all_off(cfg: dict) -> bool:
    """Every card is set to Off ("gpu_roles" with no card on, or "gpu_none": true)."""
    roles, _ = gpu_roles.saved_roles(cfg)
    return roles is not None and not any(r != "off" for r in roles.values())


def layer_split_value(cfg: dict) -> str:
    """#644: the config's "layer_split" as the engine's --layer-split value: "auto" (the default), or the FIRST LAYER of
    each later GPU's share - one rising number per GPU after the first, e.g. "24,36,42" for 4 GPUs (layers 0-23 on the
    first card, 24-35, 36-41, 42 to the end) - as a string or a JSON list ([24, 36, 42]).  ValueError with the format
    and an example for anything else, such as per-card layer counts ("24,16,12,12")."""
    v = cfg.get("layer_split")
    if v is None or (isinstance(v, str) and v.strip().lower() in ("", "auto")):
        return "auto"
    n = len(gpu_list(cfg))
    items = v if isinstance(v, (list, tuple)) else str(v).split(",") if isinstance(v, (str, int)) else None
    vals = []
    for x in items or []:
        try:
            if isinstance(x, bool) or isinstance(x, float):
                raise ValueError
            vals.append(int(str(x).strip()))
        except ValueError:
            vals = None
            break
    want = max(n - 1, 1)
    example = ",".join(str(round(48 * (i + 1) / (want + 1))) for i in range(want))
    hint = (f'"layer_split" is the first layer of each later GPU, one rising number per GPU after the first '
            f'({want} for {n} GPUs), not a count of layers per card - e.g. "{example}" (or [{example.replace(",", ", ")}])'
            f' for an even share of 48 layers, or "auto" (the default) to place them by each card\'s free VRAM')
    if not vals:
        raise ValueError(f"{hint}; got {v!r}")
    if n > 1 and len(vals) != n - 1:
        raise ValueError(f"{hint}; got {len(vals)} number(s) ({v!r}) for {n} GPUs")
    if vals[0] < 2 or any(b <= a for a, b in zip(vals, vals[1:])):
        raise ValueError(f"{hint}; got {v!r}, which does not rise from 2 or more")
    return ",".join(str(x) for x in vals)


def layer_split_of(cfg: dict) -> bool:
    """Several GPUs in the config are a layer split, unless the args put the later card(s) to another use: with
    --peer-device the second card is an expert-cache tier, and the engine refuses that beside --layer-split (#665)."""
    return len(gpu_list(cfg)) > 1 and "--peer-device" not in cfg["args"]


_CARD_CLASS: list = []


def drafting_card_class() -> str:
    """The class of this computer's graphics card for the drafter table (strata_common/drafting.py), looked up once."""
    if not _CARD_CLASS:
        archs: list[str] = []
        try:
            from strata_common import engine_component
            archs = list(engine_component.cuda_archs_present()) + list(engine_component.hip_archs_present())
        except Exception:                                    # noqa: BLE001 - an unknown card is the "other" class, never an error
            pass
        _CARD_CLASS.append(drafting.card_class(archs))
    return _CARD_CLASS[0]


def drafting_vram_mib(cfg: dict, backend: str | None, archs) -> float | None:
    """MiB of graphics memory of the ONE card DFlash2 would run on, for the drafter's confidence (drafting.expected_slots); None for several cards, a
    chip that shares the computer's memory (Strix Halo, the Spark: their experts are all "resident") or a card the driver does not answer for."""
    try:
        gl = gpu_list(cfg)
        arch = str((list(archs or [""]) or [""])[0]).lower()
        if len(gl) > 1 or arch in HIP_IGPU_ARCHS or arch == "sm_121":
            return None
        got = gpu_vram_mib(gl or [0], hip=(backend == "hip"))
        return float(next(iter(got.values()))) if got else None
    except Exception:                                        # noqa: BLE001 - an unreadable card keeps the plain 0.35, never stops a start
        return None


def mtp_retry_args(cfg: dict, exe: str | None, extra: list[str] | None = None) -> list[str] | None:
    """The arguments to start again with when an Auto start that picked DFlash2 ends before the engine is ready: the same start with the MTP (or no drafter, when
    the MTP layer is not on this computer). None when there is nothing to retry: the person named DFlash2 themselves (that start fails with its reason, as it
    should), the choice was not DFlash2, or the MTP arguments are the same. Builds the arguments without any line or note (engine_args announce=False)."""
    if drafting.normalize(cfg.get("drafting")) != "auto" or "args" not in cfg:
        return None
    try:
        first = engine_args(cfg, exe, announce=False)
        if "--draft" not in first or first[first.index("--draft") + 1:first.index("--draft") + 2] != ["dflash2"]:
            return None
        alt = engine_args({**cfg, "drafting": "mtp"}, exe, announce=False) + list(extra or [])
    except Exception:                                  # noqa: BLE001 - a retry that cannot be worked out is no retry
        return None
    return alt if alt != first + list(extra or []) else None


def nodraft_retry_args(cfg: dict, exe: str | None, extra: list[str] | None = None) -> list[str] | None:
    """The arguments to start again with when an Auto start ends before the engine is ready because a card ran out of memory: the same start with no drafter
    (the draft layer and DFlash2's buffers are the biggest optional part on a card that is short), never the MTP in place of DFlash2.  None when the drafting
    was not Auto (the person named a drafter: that start fails with the engine's reason), when the first start has no drafter anyway, or when nothing differs."""
    if drafting.normalize(cfg.get("drafting")) != "auto" or "args" not in cfg:
        return None
    try:
        first = engine_args(cfg, exe, announce=False)
        if "--draft" not in first and "--mtp" not in first:
            return None
        alt = engine_args({**cfg, "drafting": "off"}, exe, announce=False) + list(extra or [])
    except Exception:                                  # noqa: BLE001 - a retry that cannot be worked out is no retry
        return None
    return alt if alt != first + list(extra or []) else None


def engine_args(cfg: dict, exe: str | None = None, *, announce: bool = True) -> list[str]:
    """The engine's arguments: the config's, and with several GPUs the layer split across them ("layer_split" in the
    config: "auto" by default, or the first layer of each later GPU's share, e.g. "18" or "16,32"; see
    layer_split_value). `exe` is the engine program that will start: the drafter's flags follow what that program takes
    (strata_common/drafting.py engine_form). `announce=False` builds the arguments without the "[strata] drafting" lines
    and without leaving a note for the next engine start (the retry arguments of mtp_retry_args)."""
    args = list(cfg["args"])
    # "user_args" (strata_common/user_args.py): the options the person set by hand, over the generated ones at every start,
    # so a setup run or the app rewriting "args" cannot undo them (#3216 #3215 #3270)
    if cfg.get("user_args"):
        args = user_args.merge(args, cfg["user_args"])
    if paged_launch.is_paged_engine(cfg):
        if layer_split_of(cfg) and "--layer-split" not in args:
            args += ["--layer-split", layer_split_value(cfg)]
        args += parallel_args(cfg, args)
        return paged_launch.translate_args(cfg, args)
    # "drafting": auto | mtp | dflash2 | off (strata_common/drafting.py; the Models screen and `strata launch --drafting` write it): which drafter
    # guesses ahead. ONE drafter per run: DFlash2 alone (no --mtp, --draft-fallback none) or the MTP alone (no DFlash2 option). Auto is DFlash2 where the engine
    # has it, on one card and on a layer split over several cards. A choice that cannot be applied (no drafter file yet, several requests at once) starts the other one if it is on this computer,
    # else none, and says why in the engine's /metrics ("drafting_note"). The start never downloads.
    shaped = {**cfg, "args": args}
    backend, archs = drafting.detect_hardware()
    parallel_n = cfg.get("parallel")
    decision = drafting.apply(shaped, klass=drafting_card_class(), data_dir=drafting.data_dir_of(args), backend=backend, archs=archs,
                              vram_mib=drafting_vram_mib(cfg, backend, archs),
                              slots=parallel_n if isinstance(parallel_n, int) and not isinstance(parallel_n, bool) else 0, engine=exe)
    args = shaped["args"]
    if announce:
        drafting.remember_note(decision.note)
        if decision.note:
            print(f"[strata] {drafting.line(decision.drafter, decision.note)}", flush=True)
            hint = drafting.fetch_hint(decision, model_name=cfg.get("model_name"), backend=backend, archs=archs, data_dir=drafting.data_dir_of(args), args=args,
                                       vram_mib=drafting_vram_mib(cfg, backend, archs))
            if hint:
                print(f"[strata] {hint}", flush=True)
        elif decision.flags:
            print(f"[strata] drafting: DFlash2 alone ({' '.join(decision.flags[:2])}, --draft-conf {args[args.index('--draft-conf') + 1] if '--draft-conf' in args else '-'})", flush=True)
    # A card that is short of VRAM for this start (a single request on a small card, or "parallel" 2 or more, every request slot a session of its own) gets work moved off
    # the GPU, never a smaller context and never fewer requests (owner, 2026-11-02): a smaller KV window on CUDA, a smaller prompt chunk and reserve on Arc, and the
    # engine sizes the rest; one plain note says it is slower (strata_common/vram_fit.py fit_start).  The person's own values are kept.  Nothing changes without one
    # card's memory, on a layer split, or for DeepSeek.
    try:
        args, fit_parallel, fit_note = vram_fit.fit_start(cfg, args, drafting_vram_mib(cfg, backend, archs), backend)
    except Exception:                                        # noqa: BLE001 - an estimate that fails decides nothing; the engine still refuses what does not fit
        fit_parallel, fit_note = None, ""
    if fit_note and announce:                                # (announce=False builds the arguments without any line)
        print(f"[strata] {fit_note}", flush=True)
    if fit_parallel is not None:
        cfg = {**cfg, "parallel": fit_parallel}
    # "mtp": false (the Models settings, `strata launch --mtp off`, or by hand): no draft layer, whatever the args say.  A setup run
    # that writes --mtp back into the args cannot undo it, because this is applied at every start. (A DFlash2 start keeps its own --mtp DIR, which only names
    # the vocabulary for an engine that takes it that way.)
    if draft_layer.is_off(cfg) and not decision.flags:
        args, removed = draft_layer.strip(args)
        if removed:
            print(f'[strata] "mtp": false - starting without the draft layer ({" ".join(removed)} left out)', flush=True)
    # #1322: a config with a "vision" section but without --vision in its args (written by an older setup run, or edited by
    # hand) advertised images and then refused every picture ("this engine was started without --vision").  The section
    # says images are wanted: start the engine with them, and keep the encoder's VRAM free as setup does for a GPU encoder
    # (only when the config has no reserve of its own).
    if isinstance(cfg.get("vision"), dict) and "--vision" not in args:
        args.append("--vision")
        note = "added --vision"
        if cfg["vision"].get("gpu") and "--vram-reserve-mib" not in args:
            args += ["--vram-reserve-mib", "700"]
            note += " and --vram-reserve-mib 700"
        print(f'[strata] the config has a "vision" section but no --vision in its args: {note} (run setup again with '
              "--vision to write them)", flush=True)
    if layer_split_of(cfg) and "--layer-split" not in args:
        args += ["--layer-split", layer_split_value(cfg)]
    # opt-in: an auto split runs on the first card alone when it holds every profiled expert and the KV
    if layer_split_of(cfg) and cfg.get("split_skip_if_fits") and "--split-skip-if-fits" not in args:
        args.append("--split-skip-if-fits")
    # #533 (opt-in): "vram_elastic": true - the expert cache in segments, so POST /v1/vram can give VRAM back to other
    # programs and take it back; "vram_segment_mib" sets the segment size (the engine's default: 512)
    if cfg.get("vram_elastic") is True and "--vram-elastic" not in args:
        args.append("--vram-elastic")
        seg = cfg.get("vram_segment_mib")
        if isinstance(seg, int) and not isinstance(seg, bool) and seg > 0 and "--vram-segment-mib" not in args:
            args += ["--vram-segment-mib", str(seg)]
    args += parallel_args(cfg, args)
    args += pool_cores_args(cfg, args)
    return learned_profile_args(cfg, args)


PARALLEL_MAX = 8      # without a layer split the engine's batch window holds at most 8 rows (kVerifyNarrowT); --batch-mtp waves more through it


SPLIT_MAX_LANES = 8   # the all-cards split engine runs 1..8 lanes (strata_control/split_engine.py MAX_LANES)


def parallel_flag_conflicts(cfg: dict) -> bool:
    """`--parallel N` next to the saved engine args' own --batch/--slots is refused, except for the all-cards split engine's run config (split_config's
    "split_engine_note"): its --batch is its lane count, and "parallel" sets it (parallel_args). The app's Concurrent requests reaches the server this way."""
    args = cfg.get("args", [])
    return ("--batch" in args or "--slots" in args) and not cfg.get("split_engine_note")


def parallel_args(cfg: dict, args: list[str]) -> list[str]:
    """#465 (opt-in): "parallel": N in the config - up to N requests decode together; more wait their turn.  It reaches
    the engine as --batch N: the engine's batch slots, or a page-pool engine's first scheduling limit (it can change it
    while it runs); the old DeepSeek engine takes --parallel N.  1 or absent: one request at a time, as before.  A value
    the engine cannot use is said and passed on: the engine warns and adjusts (recommend, never force)."""
    ds = (cfg.get("arch") or "") == "deepseek41" and not paged_launch.is_paged_engine(cfg)
    if cfg.get("split_engine_note") and "--batch" in args:
        # the all-cards split engine: its run config carries --batch N (its lanes); "parallel" (the Terminal's --parallel, the app's Concurrent requests) sets N
        n = cfg.get("parallel")
        if isinstance(n, int) and not isinstance(n, bool) and n >= 1:
            args[args.index("--batch") + 1] = str(min(n, SPLIT_MAX_LANES))
        return []
    if "--batch" in args or "--slots" in args or (ds and "--parallel" in args):
        return []
    n = cfg.get("parallel")
    if n is None or n is False:
        return []
    if isinstance(n, bool) or not isinstance(n, int):
        print(f'[strata] "parallel" must be a whole number of requests (2..{PARALLEL_MAX}), not {n!r}: ignored',
              flush=True)
        return []
    if n <= 1:
        return []
    if ds:
        # DeepSeek: --parallel N (the engine holds N conversations at once, each its own rings and KV in VRAM, taken out of the
        # experts' room; their tokens are decoded together).  Not --batch: that is the Qwen engine's slots.
        # An engine package from before this flag would stop at the start on it ("unknown option"): asked of the engine itself, and said.
        try:
            from strata_common import dsv41_flags
            old_engine = dsv41_flags.takes_parallel(cfg.get("exe")) is False
        except Exception:                                # noqa: BLE001 - a probe that fails decides nothing
            old_engine = False
        if old_engine:
            print(f'[strata] "parallel": {n} - this DeepSeek engine ({cfg.get("exe")}) is from before --parallel, so it answers one request at a '
                  "time. Update the engine (strata update) to answer several at once.", flush=True)
            return []
        if n > PARALLEL_MAX:
            print(f'[strata] "parallel": {n} - the DeepSeek engine runs at most {PARALLEL_MAX} conversations at once; it will use '
                  f"{PARALLEL_MAX}", flush=True)
        return ["--parallel", str(min(n, PARALLEL_MAX))]
    # (on a layer split the pipelined groups each hold up to 8 slots - 16 in a HIP engine with wide windows - so more than 8 is
    # not a limit there: the engine says how many slots it runs, INFO batch_slots)
    if (n > PARALLEL_MAX and "--batch-mtp" not in args and os.environ.get("STRATA_BATCH_MTP", "0") in ("", "0")
            and not layer_split_of(cfg)):
        print(f'[strata] "parallel": {n} - the engine runs at most {PARALLEL_MAX} requests together; it will use '
              f"{PARALLEL_MAX}", flush=True)
    return ["--batch", str(n)]


_POOL_CORES_OK = re.compile(r"^[0-9]+(-[0-9]+(:[0-9]+)?)?(,[0-9]+(-[0-9]+(:[0-9]+)?)?)*$")


def pool_cores_args(cfg: dict, args: list[str]) -> list[str]:
    """Experimental, off by default: "pool_cores": "0,2,4,6" (or "2-14:2", or a list of numbers) in the config pins the CPU
    expert workers to exactly those logical processors (--pool-cores; docs/DETAILS.md, "Which logical processors the CPU
    workers use").  The engine warns about two entries on one core or the host's own and leaves out processors the process may
    not use; a text this cannot read is said here and ignored (recommend, never force).  An engine option already in the args wins."""
    value = cfg.get("pool_cores")
    if value is None or value is False or value == "" or value == []:
        return []
    if "--pool-cores" in args:
        return []
    if isinstance(value, list) and value and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in value):
        text = ",".join(str(v) for v in value)
    elif isinstance(value, str):
        text = value.replace(" ", "")
    else:
        text = ""
    if not _POOL_CORES_OK.match(text):
        print(f'[strata] "pool_cores" must be logical processor numbers like "0,2,4,6" or "2-14:2", not {value!r}: ignored '
              "(the CPU workers keep their default placement)", flush=True)
        return []
    return ["--pool-cores", text]


def profile_shape(path: str) -> tuple[int, int] | None:
    """An expert profile's (layers, experts per layer), from its header (tools/make_profile.py's format), or None
    when the file is missing, is not one or is shorter than the pairs its header promises."""
    try:
        with open(path, "rb") as f:
            head = f.read(24)
            size = os.fstat(f.fileno()).st_size
    except OSError:
        return None
    if len(head) < 24 or head[:4] != b"STRP":
        return None
    _, nl, ne, _, n = struct.unpack("<5I", head[4:])
    return (nl, ne) if size >= 24 + 4 * n else None


def learned_profile_args(cfg: dict, args: list[str]) -> list[str]:
    """#477 (opt-in): "expert_profile_save": "<path>" in the config has the engine (0.1.36+) save what its adaptive
    tier learned there - on QUIT and every "expert_profile_save_every" minutes (10 by default, 0 = at QUIT only) -
    and the next start begins from it instead of the config's --expert-profile, when it is a profile of the same
    model (its header's layers and experts match); otherwise from the config's own, as before.  A relative path is
    the engine's (the config's "cwd").  Without the key, the arguments are the config's, unchanged."""
    save = cfg.get("expert_profile_save")
    if not isinstance(save, str) or not save.strip() or "--expert-profile-save" in args:
        return args
    args = args + ["--expert-profile-save", save]
    every = cfg.get("expert_profile_save_every")
    if isinstance(every, (int, float)) and not isinstance(every, bool) and every >= 0:
        args += ["--expert-profile-save-every", str(every)]
    if "--expert-profile" in args[:-1]:
        i = args.index("--expert-profile") + 1
        here = cfg.get("cwd") or "."
        learned = profile_shape(save if os.path.isabs(save) else os.path.join(here, save))
        base = profile_shape(args[i] if os.path.isabs(args[i]) else os.path.join(here, args[i]))
        if learned is not None and learned == base:
            args[i] = save
    return args


def hip_visible(cfg: dict) -> list[int]:
    """AMD: the devices the engine should see, as the HIP runtime numbers them (HIP_VISIBLE_DEVICES).

    On Linux setup's KFD order is HIP's order, so the config's "gpu" is it.  On Windows setup finds the cards in the
    display-adapter order, and an integrated Radeon that HIP also enumerates takes ordinal 0 and pushes the discrete
    card to 1 (#325): setup records the ordinal `strata-device --list-devices` gave the card as "hip_ordinal", which
    wins for a one-card config.  Without it (a config from before), the config's "gpu"."""
    ordinal = cfg.get("hip_ordinal")
    if ordinal is not None and str(ordinal).strip() != "" and len(gpu_list(cfg)) <= 1:
        try:
            return [int(str(ordinal).strip())]
        except ValueError:
            pass
    return ordered_gpus(cfg)


def gpu_speed_scores(indices: list[int]) -> dict[int, float] | None:
    """Multiprocessors x max clock for each NVIDIA card (gpu_roles.nvidia_speeds); None when it cannot say for every
    one of `indices`."""
    got = gpu_roles.nvidia_speeds(timeout=30)
    return got if all(i in got for i in indices) else None


def hip_speed_scores(indices: list[int], root: str = "/sys/class/kfd/kfd/topology/nodes") -> dict[int, float] | None:
    """AMD on Linux: SIMDs x max engine clock of each card, from the KFD topology (the order setup and the engine use
    for "gpu" there: the GPU nodes in node order).  None when the files are not there (Windows, no ROCm driver)."""
    try:
        nodes = sorted((int(n) for n in os.listdir(root) if n.isdigit()))
    except OSError:
        return None
    cards = []
    for n in nodes:
        try:
            with open(os.path.join(root, str(n), "properties")) as fh:
                props = dict(line.split() for line in fh if len(line.split()) == 2)
            simd, clk = int(props.get("simd_count", 0)), int(props.get("max_engine_clk_fcompute", 0))
        except (OSError, ValueError):
            continue
        if simd > 0:
            cards.append(float(simd) * float(clk or 1))
    got = dict(enumerate(cards))
    return got if all(i in got for i in indices) else None


def engine_nice(env: dict | None = None, archs: dict[int, str] | None = None, platform: str | None = None) -> int:
    """Linux only: the nice value the engine starts with.  The engine's CPU pool spins at 100% on every physical core but one and its host
    thread on another; with the desktop drawn by a built-in Radeon (HIP_IGPU_ARCHS: Strix Halo and the other APUs) the compositor then
    competes with them for the CPU, which reads as a frozen screen.  +5 gives the desktop its share and costs nothing when nothing else
    runs.  STRATA_ENGINE_NICE=N sets it (0 = off, also for any PC without such a chip).  Not on Windows / macOS."""
    if not (platform or sys.platform).startswith("linux"):
        return 0
    asked = (env or os.environ).get("STRATA_ENGINE_NICE")
    if asked is not None and str(asked).strip() != "":
        try:
            return max(0, min(19, int(str(asked).strip())))
        except ValueError:
            return 0
    ar = archs if archs is not None else hip_gpu_archs()
    return 5 if ar and any(a in HIP_IGPU_ARCHS for a in ar.values()) else 0


def hip_gpu_archs(root: str = "/sys/class/kfd/kfd/topology/nodes") -> dict[int, str] | None:
    """AMD on Linux: {HIP ordinal: "gfx1151"} from the KFD topology (GPU nodes in node order, as hip_speed_scores and setup number them).
    None when the files are not there (Windows, no ROCm driver)."""
    try:
        nodes = sorted(int(n) for n in os.listdir(root) if n.isdigit())
    except OSError:
        return None
    out: dict[int, str] = {}
    for n in nodes:
        try:
            with open(os.path.join(root, str(n), "properties")) as fh:
                props = dict(line.split() for line in fh if len(line.split()) == 2)
            ver, simd = int(props.get("gfx_target_version", 0)), int(props.get("simd_count", 0))
        except (OSError, ValueError):
            continue
        if simd > 0 and ver > 0:
            out[len(out)] = f"gfx{ver // 10000}{(ver // 100) % 100:x}{ver % 100:x}"
    return out or None


def igpu_cards(indices: list[int], archs: dict[int, str] | None) -> set[int]:
    """The cards of `indices` that are integrated (their memory is the system RAM, and they usually draw the desktop)."""
    return {i for i in indices if archs and archs.get(i) in HIP_IGPU_ARCHS}


def gpu_vram_mib(indices: list[int], hip: bool = False) -> dict[int, int] | None:
    """Total VRAM of each card in MiB (NVML / amdgpu sysfs); None when any of them cannot be read."""
    try:
        from strata_common.telemetry import gpu_reader
    except ImportError:
        return None
    out = {}
    for i in indices:
        g = gpu_reader(i, amd=hip)
        if not g.ok():
            return None
        total = g.read().get("mem_total")
        if total is None:
            return None
        out[i] = int(total) >> 20
    return out


def ordered_gpus(cfg: dict, scores=None, vrams=None, archs=None) -> list[int]:
    """#1352: the cards of a layer split in the order the engine stages them.  With "layer_split": "auto" (the
    default) the faster card goes LAST - the last stage runs the head, the draft layer and the verify, and a prompt
    chunk waits on it (the reporter's 4070 Ti SUPER + 5060 Ti: a 6K prompt took 50 s one way round and 15 s the
    other); "faster" is multiprocessors x max clock.  Equal cards - scores within 5% - keep the config's order, and
    so does anything we cannot measure.  "gpu_order": "as_given" keeps the config's order whatever the cards.  A
    manual "layer_split" ("24") also keeps it: the user placed the layers.  #1576: the reorder is skipped when it
    would land the last stage on a card with less total VRAM than the one the config put there - that stage's head,
    draft and verify have a floor the smaller card cannot afford.  scores/vrams: {index: value} (tests); None asks
    the driver - injected scores without injected vrams means no VRAM check (a test seam is a test seam)."""
    gl = gpu_list(cfg)
    if len(gl) < 2 or cfg.get("gpu_order") == "as_given" or gpu_roles_active(cfg):
        return gl
    args = cfg.get("args") or []
    if "--peer-device" in args or "--layer-split" in args or layer_split_value(cfg) != "auto":
        return gl
    hip = cfg.get("backend") == "hip"
    if hip:
        # An integrated Radeon (Strix Halo beside an R9700) is never the LAST stage: that stage runs the head, the draft layer and the
        # check of the guesses, and the iGPU usually draws the desktop.  Its "VRAM" is the BIOS carve-out (up to 96 GB), which made the
        # #1576 memory guard below keep it last.  It goes first, the cards with memory of their own follow in speed order.
        # archs: {index: "gfx1151"} (tests); None asks the KFD topology, unless scores were injected (a test seam).
        ar = archs if archs is not None else (hip_gpu_archs() if scores is None else None)
        ig = igpu_cards(gl, ar)
        if ig and len(ig) < len(gl):
            sc0 = scores if scores is not None else hip_speed_scores(gl)
            rest = [i for i in gl if i not in ig]
            if sc0 and all(i in sc0 for i in rest):
                rest = sorted(rest, key=lambda i: sc0[i]) if max(sc0[i] for i in rest) >= 1.05 * min(sc0[i] for i in rest) else rest
            out = [i for i in gl if i in ig] + rest
            if out != gl:
                print(f"[strata] layer split: card order {','.join(map(str, out))} (GPU {','.join(str(i) for i in sorted(ig))} is built-in "
                      f"graphics: it takes the first layers, never the last stage; \"gpu_order\": \"as_given\" keeps {','.join(map(str, gl))})",
                      flush=True)
            return out
    sc = scores if scores is not None else (hip_speed_scores(gl) if hip else gpu_speed_scores(gl))
    if not sc or any(i not in sc for i in gl):
        return gl
    if max(sc[i] for i in gl) < 1.05 * min(sc[i] for i in gl):
        return gl     # #1760: cards within 5% of each other are the same card (two 5070 Ti read 1.2% apart on the clock): keep the order
    out = sorted(gl, key=lambda i: sc[i])
    if out != gl:
        vram = vrams if vrams is not None else (None if scores is not None else gpu_vram_mib(gl, hip))
        if vram is not None and all(i in vram for i in gl) and vram[out[-1]] < vram[gl[-1]]:
            print(f"[strata] layer split: keeping card order {','.join(map(str, gl))} - the faster card has "
                  f"less VRAM than the one the split ends on ({vram[out[-1]]} vs {vram[gl[-1]]} MiB, #1576)",
                  flush=True)
            return gl
        print(f"[strata] layer split: card order {','.join(map(str, out))} (the faster card last; "
              f'"gpu_order": "as_given" keeps {",".join(map(str, gl))}, #1352)', flush=True)
    return out


#: AMD chips that share the system memory (an iGPU): never preferred over a card that has memory of its own.
HIP_IGPU_ARCHS = ("gfx1035", "gfx1036", "gfx1037", "gfx1103", "gfx1150", "gfx1151", "gfx1152", "gfx90c")


def parse_hip_devices(text: str) -> list[dict]:
    """`strata-device --list-devices`: [{"index", "name", "arch", "vram_gb", "integrated", "cannot_run"}], numbered as the HIP runtime numbers them."""
    found: list[dict] = []
    for line in text.splitlines():
        s = line.strip()
        m = re.match(r"device\s+(\d+):\s*(.*)$", s)
        if m:
            found.append({"index": int(m.group(1)), "name": m.group(2).strip(), "arch": "", "vram_gb": 0.0, "integrated": False, "cannot_run": None})
            continue
        if not found:
            continue
        a = re.match(r"arch\s+(gfx[0-9a-f]+)\s*,\s*([\d.]+)\s*GiB(.*)$", s)
        if a and not found[-1]["arch"]:
            found[-1]["arch"], found[-1]["vram_gb"] = a.group(1), float(a.group(2))
            found[-1]["integrated"] = "integrated" in a.group(3)      # a newer engine says so itself
        elif s.startswith("cannot run:"):
            found[-1]["cannot_run"] = s[len("cannot run:"):].strip()
    for d in found:
        d["integrated"] = d["integrated"] or d["arch"] in HIP_IGPU_ARCHS
    return found


def hip_no_card_message(devices: list[dict]) -> str:
    """The plain line when the engine has code for none of the computer's cards."""
    cards = ", ".join(f"{d['name'] or 'graphics card'} ({d['arch'] or 'unknown'}{', integrated' if d['integrated'] else ''})" for d in devices)
    built = next((m.group(1) for d in devices for m in [re.search(r"compiled for \(([^)]*)\)", d["cannot_run"] or "")] if m), "")
    text = f"[strata] none of this computer's graphics cards has code in this Strata engine{f' (it is built for {built})' if built else ''}. Cards found: {cards}."
    note = engine_component.no_package_note([d["arch"] for d in devices])
    return text + (f" {note}" if note else "")


def pick_hip_devices(devices: list[dict]) -> tuple[list[int] | None, str | None]:
    """Which card an "Automatic" start runs on: (HIP_VISIBLE_DEVICES to set, or None to leave the runtime's own order; a plain problem or
    None).  Cards this engine has no code for are skipped, and so are integrated chips while a card of its own can run (#3203: a mini PC's
    Radeon 760M is HIP's device 0 and the engine died on it; the RX 6800 XT behind it was never tried).  When device 0 is already the right
    kind nothing changes."""
    if not devices:
        return None, None
    runnable = [d for d in devices if not d["cannot_run"]]
    if not runnable:
        return None, hip_no_card_message(devices)
    pool = [d for d in runnable if not d["integrated"]] or runnable
    if devices[0] in pool:
        return None, None
    best = max(pool, key=lambda d: (round(d["vram_gb"]), -d["index"]))
    return [best["index"]], None


def hip_auto_devices(cfg: dict, env: dict) -> list[int] | None:
    """HIP_VISIBLE_DEVICES for a start that names no card (see pick_hip_devices), asked of the engine package's own `strata-device`.
    None leaves everything as it was: a card the config or the environment names, no probe next to the engine, or a probe that fails.
    SystemExit with the plain line when no card of this computer can run."""
    if cfg.get("backend") != "hip" or gpu_list(cfg) or gpu_all_off(cfg) or str(cfg.get("hip_ordinal") if cfg.get("hip_ordinal") is not None else "").strip() \
            or env.get("HIP_VISIBLE_DEVICES") or env.get("ROCR_VISIBLE_DEVICES"):
        return None
    exe = cfg.get("exe")
    if not isinstance(exe, str) or not exe or engine_component.is_reference(exe):
        return None
    probe = Path(exe).with_name("strata-device.exe" if os.name == "nt" else "strata-device")
    if not probe.is_file():
        return None
    try:
        done = subprocess.run([str(probe), "--list-devices"], capture_output=True, text=True, timeout=60, cwd=str(probe.parent), env=env,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    devices = parse_hip_devices(done.stdout)
    visible, problem = pick_hip_devices(devices)
    if problem:
        raise SystemExit(problem)
    if visible:
        d = next(x for x in devices if x["index"] == visible[0])
        print(f"[strata] automatic GPU choice: GPU {d['index']} ({d['name']}, {d['arch']}); the engine skips cards it has no code for and "
              f"integrated chips while a card of its own can run", flush=True)
    return visible


DS41_SLOT_KEEP_DEFAULT = "1"      # the engine keeps a displaced conversation in RAM and in the slot folder (see below)
DS41_SLOT_GB_DEFAULT = "2"        # the folder's cap in GB (a 6.5K-token conversation is about 17 MB)


def deepseek_slot_env(env: dict, cfg: dict) -> None:
    """DeepSeek-V4.1 only.  The engine sets a conversation aside when another request takes the model and takes it back
    at its checkpoints (agents and sub-agents that take turns do not read their prompts again).  STRATA_DS4_SLOT_KEEP
    switches it (the default above; a config's env or the shell wins); the folder is `ds4-slots` next to the run log, in
    Strata's data folder (STRATA_DS4_SLOT_DIR names another), and the engine caps it (STRATA_DS4_SLOT_GB, 2 here, 16 in the engine) and
    removes what is older than a week or not this model's.  On by default; STRATA_DS4_SLOT_KEEP=0 turns it off."""
    env.setdefault("STRATA_DS4_SLOT_KEEP", DS41_SLOT_KEEP_DEFAULT)
    env.setdefault("STRATA_DS4_SLOT_GB", DS41_SLOT_GB_DEFAULT)
    log = cfg.get("log")
    if "STRATA_DS4_SLOT_DIR" not in env and log:
        env["STRATA_DS4_SLOT_DIR"] = str(Path(log).parent / "ds4-slots")


# 0.2.10: the two parts that need the card's free VRAM are the ENGINE's own, never an environment variable of ours: the FP16 weight
# cache (bit-exact; takes only VRAM beyond 3 GiB kept free) and no loan of cache slots (only when every expert is resident and the
# prompt path's own buffers fit beside 3 GiB; KL 0.006, QUAL-PAR2).  STRATA_PF_W16_CACHE_MIB=0 and STRATA_AUTO_NOBORROW=0 turn them off.
# (The parallel-request stack, the GDN prompt speedups, the Q8_0 hyper-connection read and the fp16 decode windows are the engine's own defaults now:
# src/core/arch_defaults.cpp, generate.cpp, metal window.cpp.  Nothing here sets them.)


def parallel_slots(cfg: dict) -> int:
    """How many batch slots the config asks for (0 = none): the args' own --batch/--slots, else "parallel"."""
    args = list(cfg.get("args") or [])
    for k in ("--batch", "--slots"):
        if k in args:
            try:
                return int(args[args.index(k) + 1])
            except (IndexError, ValueError):
                return 0
    n = cfg.get("parallel")
    return n if isinstance(n, int) and not isinstance(n, bool) and n > 1 else 0


def child_env(cfg: dict, *, offline: bool = False) -> dict:
    """The engine's environment: the CUDA libraries setup installed (pip's nvidia packages, or the toolkit that
    compiled it) first on the library search path."""
    env = dict(os.environ)
    if gpu_all_off(cfg):                             # every card Off: the engine sees none (a CPU engine runs; the others say so at start)
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["HIP_VISIBLE_DEVICES"] = ""
    elif hip_visible(cfg) and cfg.get("backend") == "hip":   # AMD: numbered as HIP numbers them (hip_visible)
        env["HIP_VISIBLE_DEVICES"] = ",".join(str(i) for i in hip_visible(cfg))
    elif gpu_list(cfg):                              # issue #51: the GPU(s) to run on, numbered as nvidia-smi does; CUDA's
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"      # own default order (fastest first) can number the cards otherwise
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in ordered_gpus(cfg))
    for k, v in (cfg.get("env") or {}).items():      # engine settings the config carries (AMD: the GEMM tuning table)
        env[str(k)] = str(v)
    if (cfg.get("arch") or "") == "deepseek41":      # (only this engine reads these; Qwen's environment is untouched)
        deepseek_slot_env(env, cfg)
    # Strict offline reaches children through the environment, because a tool subprocess or a model
    # loader is code this server does not own. Applied last so a profile's own `env` cannot remove
    # the policy by naming the same variable.
    if offline:
        try:
            from strata_common import offline as offline_policy
            env.update(offline_policy.OFFLINE_ENV)
        except ImportError:
            # The policy module is part of the application; without it, refuse rather than silently
            # running a child with network access under a flag that promised otherwise.
            env["STRATA_OFFLINE"] = "1"
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
    dirs = [d for d in cfg.get("lib_dirs") or [] if Path(d).is_dir()]
    if dirs:
        var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        env[var] = os.pathsep.join(dirs + ([env[var]] if env.get(var) else []))
    if cfg.get("backend") == "hip":
        hip_env_guard(env, cfg)
        note = sh_stream_warning(env, cfg)
        if note:
            print(note, flush=True)
        auto = hip_auto_devices(cfg, env)
        if auto:
            env["HIP_VISIBLE_DEVICES"] = ",".join(str(i) for i in auto)
    return env


def _dir_writable(p: str) -> bool:
    try:
        with tempfile.TemporaryFile(dir=p):
            return True
    except OSError:
        return False


def sh_stream_warning(env: dict, cfg: dict) -> str | None:
    """#3895: the shared-expert fork (STRATA_SH_STREAM=1) together with KV streaming (--kv-resident) made one Windows RX 6900 XT fail on the first
    request ("verify: unspecified launch failure"); each alone, or Linux, did not. A warning, not a refusal: the person set the variable, the
    engine runs as asked, and the line says what to take out."""
    if os.name != "nt" or str(env.get("STRATA_SH_STREAM", "")).strip() in ("", "0"):
        return None
    args = [str(a) for a in cfg.get("args") or []]
    if "--kv-resident" not in args:
        return None
    return ("[strata] WARNING: STRATA_SH_STREAM=1 with --kv-resident (KV streaming) failed on the first request on one Windows RX 6000 PC "
            "(\"verify: unspecified launch failure\", #3895). If the engine stops that way, take STRATA_SH_STREAM out of the config's env "
            "(AMD runs without it by default) or run without --kv-resident, and tell us the card and the driver.")


def hip_env_guard(env: dict, cfg: dict) -> list[str]:
    """#654: two things in an AMD user's environment that crash the HIP runtime before it prints a line.  (1) HIP_PATH,
    HIP_DEVICE_LIB_PATH or LLVM_PATH naming a folder that is gone (a ROCm build that was deleted): the bundled runtime
    follows them.  They are dropped, and only when the folder does not exist.  (2) A TEMP / TMP / TMPDIR the engine cannot
    create a file in: the runtime's comgr JIT builds its blit kernels through temporary files and the first call
    crashes in amdhip64 (0xC0000005).  A writable folder next to the engine's log takes its place.  Returns what it
    changed (said once, on stderr); a healthy environment is left exactly as it was."""
    said: list[str] = []
    for k in ("HIP_PATH", "HIP_DEVICE_LIB_PATH", "LLVM_PATH"):
        v = env.get(k)
        if v and not Path(v).exists():
            del env[k]
            said.append(f"{k}={v} does not exist: not passed to the engine")
    bad = [k for k in ("TEMP", "TMP", "TMPDIR") if env.get(k) and not _dir_writable(env[k])]
    if bad:
        base = Path(cfg.get("log") or cfg.get("cwd") or ROOT).resolve()
        for d in (base.parent / "tmp", ROOT / "tmp"):
            try:
                d.mkdir(parents=True, exist_ok=True)
            except OSError:
                continue
            if _dir_writable(str(d)):
                for k in bad:
                    said.append(f"{k}={env[k]} is not writable: using {d}")
                    env[k] = str(d)
                break
        else:
            said.append(f"{', '.join(bad)} not writable (AMD's runtime needs a writable temporary folder)")
    for line in said:
        print(f"strata serve: AMD: {line}", file=sys.stderr, flush=True)
    return said


def carry_leg(carry: dict, leg: dict) -> None:
    """Keeps a finished leg's figures for the request's totals (add_carry): a request ends with another leg's DONE,
    which replaces the engine's figures - #1869's batch-slot leg that went back to the solo path, and a page-pool
    engine's request that a preemption paused and that continued later.  A figure one leg did not give makes the
    total unknown (None)."""
    for key in ("decode_ms", "generated"):
        previous, value = carry.get(key, 0), leg.get(key)
        carry[key] = previous + value if previous is not None and value is not None else None
    carry.setdefault("prompt_ms", float(leg.get("prompt_ms") or 0.0))
    if leg.get("prompt_read") is not None:
        carry.setdefault("prompt_read", int(leg["prompt_read"]))


def add_carry(carry: dict, last: dict) -> dict:
    """The request's totals: the earlier legs' decode time and tokens plus the final one's; the prompt figures of the leg
    that read the prompt (the solo continuation reads next to nothing)."""
    out = {**last, "prompt_ms": max(float(last.get("prompt_ms") or 0.0), carry.get("prompt_ms", 0.0))}
    for key in ("decode_ms", "generated"):
        previous, value = carry.get(key, 0), last.get(key)
        out[key] = previous + value if previous is not None and value is not None else None
    if "prompt_read" in carry:
        out["prompt_read"] = carry["prompt_read"]
    return out


def request_stats(segments: list[dict]) -> dict:
    """Keep input/cache accounting at the original API boundary and total native work across continuations."""
    if not segments:
        return {}
    result = dict(segments[-1])
    for key in ("reused", "prompt_read", "chunk"):
        if key in segments[0]:
            result[key] = segments[0][key]
        else:
            result.pop(key, None)
    for key in ("prompt_ms", "decode_ms", "generated", "drafts_offered", "drafts_accepted",
                "hits", "lookups", "offloaded", "ram_blobs", "file_blobs", "file_mb", "experts_streamed"):
        if key in ("decode_ms", "generated") and any(segment.get(key) is None for segment in segments):
            result[key] = None
        elif any(segment.get(key) is not None for segment in segments):
            result[key] = sum(segment.get(key) or 0 for segment in segments)
    return result


def prefill_leg_stats(earlier: list[dict], last: dict, prompt_tokens: int) -> dict:
    """Total yielded reads once, retaining only reuse from before this API request."""
    legs = [*earlier, last]
    total = request_stats(legs)
    if all(leg.get("prompt_read") is not None for leg in legs):
        fresh = max(0, prompt_tokens - int(total.get("reused") or 0))
        total["prompt_read"] = min(fresh, sum(int(leg["prompt_read"]) for leg in legs))
    return total


def key_list(value) -> list[str]:
    """#1344: the keys a server accepts.  --api-key / STRATA_API_KEY / "api_key" take llama.cpp's form, several keys
    separated by commas ("k1,k2"; spaces around each are dropped), or a list in the config ("api_key": ["k1", "k2"],
    the way to give a key that contains a comma).  A list is taken as it is."""
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return [str(k) for k in value if str(k)]
    return [k.strip() for k in str(value).split(",") if k.strip()]


def explicit_no_key(value) -> bool:
    """The owner chose to serve without a key: "api_key": "none" in the config, STRATA_API_KEY=none or --api-key none
    (or --no-api-key, handled by the caller).  Never a key called "none"."""
    return isinstance(value, str) and value.strip().lower() == "none"


def api_key_of(value):
    """The key as a client can send it (#725).  An HTTP header loses the spaces and line ends around its value, so a
    key kept with them (a config file or an environment file with CRLF line ends, a quoted " key ") matched no
    request: every client got 401 with the right key.  A value that is only such characters raises ValueError, it
    never means "no key" (#213).  #1344: a config list of keys returns the list (each key stripped, none empty); a
    string stays a string (its commas separate keys, see key_list)."""
    if isinstance(value, (list, tuple)):
        keys = [str(k).strip() for k in value]
        if not keys or any(not k for k in keys):
            raise ValueError("an API key was given but it is empty")
        return keys
    key = "" if value is None else str(value)
    if key and not key_list(key):
        raise ValueError("an API key was given but it is empty")
    return key.strip()
