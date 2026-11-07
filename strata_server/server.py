"""strata_server/server.py - plan v0.3 P8: OpenAI and Anthropic endpoints over any engine that maps token ids to tokens.

    python -m strata_server.server --engine mock --port 8095            (a scripted engine, for clients and tests)
    python -m strata_server.server --engine strata --config strata.json --port 8080   (the real engine, resident)

Endpoints: POST /v1/chat/completions (OpenAI, stream and non-stream), POST /v1/messages (Anthropic, stream and
non-stream), GET /v1/models, GET /models, GET /props, GET /slots, GET /health, GET /mcp. One sequence at a time behind a FIFO (plan: one resident sequence).
Image generation (optional, when the config has an "images" section): POST /v1/images/generations (OpenAI's shape,
plus seed and steps) runs stable-diffusion.cpp's sd-cli for Qwen-Image-2.1-Turbo, one job at a time on the GPU the
text engine uses (strata_server/images.py).
Tools from MCP servers (strata_server/mcp.py, `"mcp_servers"` in the config or --mcp-config) are offered only to requests that
ask for them with `"strata_mcp": true` - the web app does; other clients see exactly the API they always saw.
Images (optional, when the config has a "vision" entry): OpenAI image_url parts and Anthropic image blocks (base64
data, http(s) URLs or local file paths) go through `strata-vision` (the model's mmproj file) and reach the engine as
embeddings (`GENI`).  JPEG/PNG/BMP/GIF go straight in; WebP, TIFF, AVIF, ... (agents like omp send WebP) are
converted to PNG first with Pillow.
A request whose prompt alone does not fit the engine's context is REJECTED with 400, never truncated.  When the prompt
fits but prompt plus max tokens does not (Claude Code asks for 32000, the Harness for 16384), max tokens is shortened to
the room left (less CTX_SLACK) and the thinking budget follows it; `--no-fit-max-tokens` / `"fit_max_tokens": false`
brings back the strict 400 of llama.cpp.
An unset (or 0, or -1) max tokens means "unlimited": whatever the prompt leaves of the context.

The engine boundary is `Engine.generate(prompt_ids, max_new, sampling, cancel) -> iterator of token ids`.
`StrataEngine` keeps one `strata --serve` process resident (weights, expert arena and VRAM tier load once) and
talks to it over stdin/stdout; `MockEngine` is a scripted stand-in that makes every API path testable without a GPU.
Both are in strata_common/engine_client.py.
"""
from __future__ import annotations

import argparse
import contextlib
import collections
import base64
import hashlib
import hmac
import codecs
import ctypes
import json
import math
import os
import queue
import re
import select
import socket
import shutil
import subprocess
import struct
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from http.client import HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from functools import cached_property
from typing import Iterator, Protocol
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))   # run as a script (run-<model>.bat) as well as a module
from strata_server.frontend import (THINK_END, ChatTemplate, Event, OutputParser, anthropic_to_messages,  # noqa: E402
                            forced_call, images_of, literal_tags, mark_think_literals, openai_to_messages, require_messages,
                            tool_choice_of, unmark_think_literals)
from strata_server.mcp import McpCancelled, hub_from_config  # noqa: E402
from strata_server import runconfig  # noqa: E402
from strata_common import engine_component  # noqa: E402
from strata_base import tls  # noqa: E402

tls.install()       # #3722: image URLs and tool servers over HTTPS are checked against the operating system's certificates
from strata_common import hipblaslt_tuning  # noqa: E402
from strata_common import vision_reader  # noqa: E402
from strata_common import gpu_roles  # noqa: E402
from strata_common import ready as ready_file  # noqa: E402
from strata_server import owner_guard  # noqa: E402
from strata_common import launch_identity  # noqa: E402
from strata_common import logit_bias as bias_api  # noqa: E402
from strata_common import engine_client  # noqa: E402
from strata_common import paged_launch  # noqa: E402
from strata_common import lifeline  # noqa: E402
from strata_server.structured import StructuredOutputError, prepare_format, validated_json  # noqa: E402
from strata_server import responses as responses_api  # noqa: E402
from strata_server.responses import ResponsesError, error_body as responses_error_body  # noqa: E402
from strata_server import images as images_api  # noqa: E402

IM_END = "<|im_end|>"
IMAGE_PAD = "<|image_pad|>"


def listen_problem(host: str, port: int, e: OSError) -> str:
    """#769: why the server cannot listen.  Only an address that is taken says "already in use"; anything else (a port
    Windows keeps reserved, an address this PC does not have, no permission) says what the OS said."""
    import errno
    code = getattr(e, "winerror", None) or e.errno
    if code in (errno.EADDRINUSE, 10048):
        return (f"port {port} is already in use - is Strata (or another server) already running? "
                f"Close it, or start this one with a different --port")
    what = e.strerror or str(e)
    if code in (10013, errno.EACCES):
        hint = ("Windows may keep this port for itself (see `netsh int ipv4 show excludedportrange protocol=tcp`) "
                "or a firewall rule forbids it: start this one with a different --port")
    elif code in (10049, errno.EADDRNOTAVAIL):
        hint = f"{host} is not an address of this PC: check \"host\" in the model's strata-<model>.json or --host"
    else:
        hint = "check --host and --port"
    return f"cannot listen on {host}:{port}: {what} [{code}]. {hint}"


def combined_embeddings_path(vision_dir: Path, nbytes: int) -> Path:
    """#874: where one request's combined image-embeddings file goes.  Every request with images rewrites it (it is
    deleted after the request), which on a disk is steady heavy writing: on Linux it goes to /dev/shm when that has
    room (the engine reads it back at once), on Windows the file is opened short-lived (the cache is not forced out
    to the drive).  Anywhere else, or when /dev/shm is too small (Docker's default is 64 MB), it stays in the encoder's
    own directory, as before."""
    name = f"req-{uuid.uuid4().hex[:12]}.sve"
    shm = Path("/dev/shm")
    if os.name != "nt" and shm.is_dir() and os.access(shm, os.W_OK):
        try:
            if shutil.disk_usage(shm).free > 2 * nbytes + (16 << 20):
                return shm / f"strata-vision-{os.getpid()}-{name}"
        except OSError:
            pass
    return Path(vision_dir) / name


def write_temporary(path: Path, parts: list) -> None:
    """#874: write `parts` (files) one after the other into `path`; on Windows with FILE_ATTRIBUTE_TEMPORARY
    (os.O_SHORT_LIVED), so the file normally never reaches the drive before it is deleted."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0) | getattr(os, "O_SHORT_LIVED", 0)
    with os.fdopen(os.open(path, flags, 0o600), "wb") as f:
        for part in parts:
            f.write(Path(part).read_bytes())
# #458: the trailing effort turn ("effort_position": "end"); low is the template's own sentence, medium (which the
# template says nothing for: no xhigh sentence is medium there) says it, since the xhigh one stays at the top
EFFORT_TURN = "<|im_start|>system\n{}<|im_end|>\n"
EFFORT_TEXT = {"low": "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the "
                      "conclusion without unnecessary elaboration.",
               "medium": "Reasoning effort is set to medium. Think as much as the task needs, without unnecessary "
                         "elaboration.",
               # the template's own xhigh sentence (chat_template.jinja), which #728's recovery swaps for the low one
               "xhigh": "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
                        "assumptions, consider plausible alternatives, and prioritize correctness, consistency, and "
                        "clarity in the final answer."}
VISION_START = "<|vision_start|>"
# #606: a reply that repeats one token this many times in a row is ended there ("length"): a model in a loop, or a
# broken state that answers one token forever (an issue saw 36,689 tokens of "!"). The config's "repeat_stop_tokens"
# sets it; 0 turns it off.
REPEAT_STOP_TOKENS = 256
ANSWER_ROOM_MIN_CAP = 1024    # 0.1.45: from this max_tokens on, thinking without a budget is closed early enough to leave room to answer
DS41_DEFAULT_BUDGET = 32768   # DeepSeek-V4.1 only: the thinking budget when neither the request nor the config names one
DS41_SMALL_CAP_MIN = 48       # DeepSeek-V4.1 only: from this max_tokens up to ANSWER_ROOM_MIN_CAP, thinking is capped at half of it
ANSWER_RESERVE_MIN = 512     # #984: the tokens kept for the answer when a thinking budget is cut down to fit max_tokens
# #123: what closes the thinking when it reaches reasoning_budget_tokens (the model's own end-of-thinking tag after it)
# #1053 (opt-in "reasoning_close_retry": true): a reply that ends on its stop token still inside <think>, with no answer
# and no call, is continued once with the thinking closed (the same way the budget's wrap-up closes it)
REASONING_CLOSE = "\n</think>\n\n"
REASONING_WRAP_UP = "\n\nI have thought about this long enough; time to give my answer.\n</think>\n\n"
# #728: opt-in handling of reasoning that repeats whole passages (which the single-token guard above cannot see).
# "reasoning_loop_recovery": "stop" ends the reply there; "recover" (or true) goes on from the same output with the
# low-effort instruction in place of the xhigh one.  Both off by default.
HIGH_EFFORT = EFFORT_TEXT["xhigh"]
LOW_EFFORT = EFFORT_TEXT["low"]
LOOP_CHECK_EVERY = 512           # output tokens between two looks at the reasoning (at a clean parser boundary)
LOOP_COVERAGE = 0.25             # the share of the last 2,000 words inside 12-word passages seen three times
LOOP_HISTORY_WORDS = 30000       # how far back the passages are counted (bounds the cost of a look)
LOOP_TAIL_CHARS = 2048           # chars at the end of the reasoning also checked for a short period (#1753)
LOOP_TAIL_PERIOD = 64            # a period at most this long in that tail reads as a fully repeated window


def reasoning_repeat_coverage(text):
    """Coverage of recent words by 12-word passages seen at least three times.

    Only reasoning is supplied. The history (the last LOOP_HISTORY_WORDS words) detects repeated verification passes
    separated by long code drafts; the recent window excludes old repetitions.  #1753: a loop can also be one long
    word (a 24k-digit string cycling an 11-digit pattern): the word split makes it a single word and no word count
    grows, and #606's token run never grows either (the tokenizer has no multi-digit tokens), so before the word
    count the last LOOP_TAIL_CHARS chars are checked for a period of at most LOOP_TAIL_PERIOD, and a periodic tail
    reads as full coverage: it is all of the recent output, repeated.
    """
    tail = text[-LOOP_TAIL_CHARS:].lower()
    if len(tail) == LOOP_TAIL_CHARS and any(tail[p:] == tail[:-p] for p in range(1, LOOP_TAIL_PERIOD + 1)):
        return 1.0
    words = re.findall(r"\w+|[^\w\s]", text.lower())[-LOOP_HISTORY_WORDS:]
    width, window = 12, 2000
    if len(words) < window:
        return 0.0
    grams = [tuple(words[i:i + width]) for i in range(len(words) - width + 1)]
    counts = collections.Counter(grams)
    start = max(0, len(words) - window)
    covered = set()
    for i in range(start, len(grams)):
        if counts[grams[i]] >= 3:
            covered.update(range(i, i + width))
    return len(covered) / (len(words) - start)


def focused_recovery_prompt(tok, ids, generated):
    """The prompt's ids with the xhigh effort sentence of the first system message replaced by the low one, then
    every token generated so far.  A splice of token ids: the prompt is never decoded and encoded again, so a literal
    `</think>` or vision marker in it (#537, #554) stays what it was.  None when the sentence is not there as whole
    tokens in the first system message (a user's text never counts).  No answer or end-of-thinking is inserted."""
    end = tok.encode(IM_END, parse_special=True)
    if len(end) != 1 or end[0] not in ids:
        return None
    head = ids[:ids.index(end[0])]
    if not tok.decode(head).startswith("<|im_start|>system\n"):
        return None
    old, new = tok.encode(HIGH_EFFORT, parse_special=False), tok.encode(LOW_EFFORT, parse_special=False)
    for i in range(len(head) - len(old) + 1):
        if head[i:i + len(old)] == old:
            return ids[:i] + new + ids[i + len(old):] + list(generated)
    return None


LOOPBACK_NAMES = ("localhost", "127.0.0.1", "::1")
QUEUE_BEAT_S = 10.0         # a request waiting for the single turn sends a keep-alive this often (#1619)
KEEPALIVE_S = 10.0          # a held-back tool argument leaves the stream quiet: ping at least this often
# Anthropic clients add /v1/messages to their base URL themselves. A base URL that already ends in /v1 (copied from an OpenAI-style
# setup) makes them post /v1/v1/messages; these two paths mean the same as the real ones. Nothing else is aliased.
V1_V1_ALIASES = {"/v1/v1/messages": "/v1/messages", "/v1/v1/messages/count_tokens": "/v1/messages/count_tokens"}
CTX_SLACK = 8               # `strata --serve` rejects prompt + max_new + 8 > context: keep the same margin here
# The live tok/s is a rate over a window, not a mean since the first token: a mean reads ~1/elapsed at the first
# token (the Monitor showed five-digit numbers) and then undershoots for the first second of every answer.
RATE_WINDOW_S = 2.0
RATE_MIN_SPAN_S = 0.25      # younger than this there is no rate yet: the mean so far, with the span floored here
# Reads that had no timeout (#1317).  A blocking read of the image encoder's pipe that never returned held the request
# turn for good: the engine then sat idle in its command loop, so no watchdog could fire and every later request
# queued behind it (the "lost step" class of #481).  Generous on purpose: a slow CPU encode is not a hang.  Only an
# encoder that says nothing at all for this long is ended; the next request starts it again.  The engine's READY line
# has its own: engine_client.ENGINE_READY_S.
# STRATA_VISION_READY_S / STRATA_VISION_ENCODE_S override them (0 = wait for ever, as before).
VISION_READY_S = engine_client.timeout_env("STRATA_VISION_READY_S", 300.0)
VISION_ENCODE_S = engine_client.timeout_env("STRATA_VISION_ENCODE_S", 300.0)


# ------------------------------------------------------------------------------------------------ engines
class Engine(Protocol):
    max_context: int
    def generate(self, ids: list[int], max_new: int, sampling: dict, cancel: threading.Event) -> Iterator[int]: ...


class EngineStarting(RuntimeError):
    """The engine is (re)starting and has not said READY yet (#344): no context size to plan a request with - a 503,
    not a 400 about the prompt."""


class ClientGone(Exception):
    """#1999: the client hung up while its request was being prepared (an image URL downloading, a picture encoding or
    waiting for its turn): nothing is left to answer."""


class ModelBusy(RuntimeError):
    """Explicit model controls must not interrupt active or queued requests."""


class QueueCancelled(RuntimeError):
    """The client left before its first serial engine turn."""


def unload_wait(value, default: float) -> float:
    """#1997: the `wait` of POST /unload?wait=...: absent / 0 / false: 0 (answer "busy" at once, as before); 1 / true /
    yes: `default` seconds; a number of seconds above 1: that long (at most an hour)."""
    if value is None:
        return 0.0
    v = str(value).strip().lower()
    if v in ("", "0", "false", "no", "off"):
        return 0.0
    if v in ("1", "true", "yes", "on"):
        return default
    try:
        return min(3600.0, max(0.0, float(v)))
    except ValueError:
        return default


class ConvCacheLog:
    """#596: the engine's conversation cache as its log tells it (the engine writes "strata serve: conversation
    cache: parked N tokens ...; parked=P bytes=B evictions=E" and "restored N tokens ...; parked=P bytes=B" to stderr,
    which is the log): read on from where it was last read, from the start of the engine's current run."""
    EVENT = re.compile(r"conversation cache: (parked|skipped|restored) (\d+) tokens.*?parked=(\d+) bytes=(\d+)"
                       r"(?: evictions=(\d+))?")
    DROPPED = re.compile(r"conversation cache: dropped \d+ superseded .*?parked=(\d+)")
    READ_MAX = 1 << 20                                  # at most the last MiB of new lines per read

    def __init__(self):
        self.key, self.pos = None, 0
        self.reset()

    def reset(self):
        self.state = {"parked": 0, "bytes": 0, "evictions": 0, "parks": 0, "restores": 0, "last_event": None,
                      "last_tokens": None, "last_at": None}

    def poll(self, path, start) -> dict:
        """The state after the log's new lines; `start` is where the engine's current run began in it."""
        if not path or start is None:
            return dict(self.state)
        if self.key != (path, start):                   # another start of the engine: its cache starts empty
            self.key, self.pos = (path, start), start
            self.reset()
        try:
            size = os.path.getsize(path)
            if size < self.pos:                         # the log was emptied or replaced
                self.pos = 0
            if size - self.pos > self.READ_MAX:
                self.pos = size - self.READ_MAX
            with open(path, "rb") as f:
                f.seek(self.pos)
                data = f.read(size - self.pos)
        except OSError:
            return dict(self.state)
        end = data.rfind(b"\n") + 1                     # whole lines only: the rest is read next time
        self.pos += end
        now = time.time()
        for line in data[:end].decode("utf-8", "replace").splitlines():
            if "conversation cache:" not in line:
                continue
            m = self.EVENT.search(line)
            if m:
                st = self.state
                st["parked"], st["bytes"] = int(m.group(3)), int(m.group(4))
                if m.group(5) is not None:
                    st["evictions"] = int(m.group(5))
                if m.group(1) != "skipped":
                    st["parks" if m.group(1) == "parked" else "restores"] += 1
                    st["last_event"], st["last_tokens"], st["last_at"] = m.group(1), int(m.group(2)), now
                continue
            m = self.DROPPED.search(line)
            if m:
                self.state["parked"] = int(m.group(1))
        return dict(self.state)


def conversation_slots(engine) -> int | None:
    """How many conversations the engine's cache holds: its --conversation-cache-slots argument, which is known
    before the engine has started (a lazy start has reported no INFO line yet), else the INFO line it reported."""
    args = engine.spawn[1] if hasattr(engine, "spawn") else []
    if "--conversation-cache-slots" in args[:-1]:
        try:
            return int(args[args.index("--conversation-cache-slots") + 1])
        except ValueError:
            pass
    slots = (getattr(engine, "info", None) or {}).get("conversation_cache_slots")
    return slots if isinstance(slots, int) else None


def conversation_cache_view(info: dict, hist: list, totals: dict, parked: dict) -> dict:
    """#596: the Monitor's Conversation cache card: the parked conversations (the engine's opt-in
    --conversation-cache-mib: budget, slots, what its log says) and how much of the prompts the cache gave back."""
    mib = info.get("conversation_cache_mib")
    last = hist[-1] if hist else None
    return {"enabled": isinstance(mib, int) and mib > 0, "budget_mib": mib if isinstance(mib, int) else None,
            "slots": info.get("conversation_cache_slots"), **parked,
            "requests": len(hist), "requests_reused": sum(1 for r in hist if (r.get("reused") or 0) > 0),
            "reused_tokens": totals.get("reused", 0), "prompt_tokens": totals.get("prompt_tokens", 0),
            "last_reused": last.get("reused") if last else None,
            "last_prompt": last.get("prompt_tokens") if last else None}


def visible_device_order(env: dict | None, backend: str | None) -> list[int]:
    """Map engine ordinals to OS GPU indices only when the visibility list is unambiguous."""
    if backend not in (None, "cuda", "hip"):
        return []
    if backend != "hip" and (env or {}).get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        return []
    name = "HIP_VISIBLE_DEVICES" if backend == "hip" else "CUDA_VISIBLE_DEVICES"
    value = (env or {}).get(name)
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{1,5}(?:,[0-9]{1,5}){0,15}", value):
        return []
    devices = [int(part) for part in value.split(",")]
    return devices if all(0 <= device <= 65535 for device in devices) and len(set(devices)) == len(devices) else []


def configure_batch_telemetry_env(env: dict, cfg: dict, override: bool | None) -> None:
    """Only an explicit setting changes the inherited opt-in engine environment."""
    choice = override if override is not None else cfg.get("batch_telemetry")
    if choice is True:
        env["STRATA_BATCH_TELEMETRY"] = "1"
    elif choice is False:
        env.pop("STRATA_BATCH_TELEMETRY", None)


IMAGE_URL_MAX = 32 << 20        # an image URL is read up to this (the web app attaches pictures of up to 20 MB)


def network_path(path: str) -> bool:
    r"""A path to another computer: \\host\share or //host/share (either slash), \\?\UNC\host\..., \\.\UNC\host\...
    or the NT form \??\UNC\host\...  Windows connects to that host at the first look at the path (os.path.isfile is
    enough) and signs in with the user's NTLM credentials.  \\?\ and \??\ name local files too; no client needs them."""
    p = path.replace("/", "\\")
    return p.startswith("\\\\") or p.startswith("\\??\\")


def vision_log_tail(log=None, limit: int = 16384) -> str:
    """The last `limit` bytes of the encoder's stderr file (`log` is the file object it was started with; "" when there
    is none or it cannot be read)."""
    try:
        name = getattr(log, "name", None)
        if name:
            with open(name, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - limit))
                return f.read().decode("utf-8", "replace")
    except (OSError, ValueError):
        pass
    return ""


# What the CUDA / HIP runtimes print when the reader has no code for the card it runs on (a binary built without that
# architecture: #3669, the Windows cuda12 package's reader on a Pascal card).
_VISION_ARCH_SIGNS = ("device kernel image is invalid", "no kernel image is available", "im2col failed",
                      "unsupported gpu architecture", "invalid device function", "unknown gpu architecture")


def vision_start_error(line: str, log=None, code: int | None = None, waited: float | None = None) -> str:
    """The message when the image encoder answered something other than READY.  #1445: a card in the Exclusive_Process
    compute mode (or one another process holds) refuses the encoder's CUDA context with "busy or unavailable"; that
    cause is named instead of leaving the user the first line of a traceback.  `log` is the encoder's stderr file, whose
    tail is searched too (the CUDA error is printed there).  An encoder that dies before it prints anything (#3645, #3444,
    #3669, #3697) used to give a blank sentence: now the last lines of its stderr, its exit status (`code`) and, for a
    kernel-image error, the card/architecture hint come with it.  `waited`: it said nothing for that many seconds."""
    text = line.strip()
    tail = vision_log_tail(log)
    both = (text + chr(10) + tail).lower()
    if "busy or unavailable" in both:
        return ("the vision encoder did not start: the GPU refused a second CUDA context (\"CUDA-capable device(s) "
                "is/are busy or unavailable\"). This is usually the compute mode Exclusive_Process: only one process "
                "may use the card, and the engine already holds it. Check with: nvidia-smi --query-gpu=compute_mode "
                "--format=csv; set it back with: sudo nvidia-smi -c DEFAULT (or run the encoder on another card). "
                "First line from the encoder: " + (text or "(none)"))
    lines = [x.strip() for x in tail.splitlines() if x.strip()]
    if any(sign in both for sign in _VISION_ARCH_SIGNS):
        evidence = next((x for x in reversed(lines) if any(sign in x.lower() for sign in _VISION_ARCH_SIGNS)), text)
        return ("the vision encoder did not start: it has no code for this graphics card (a kernel-image error: the "
                "picture reader in this engine package was built for other card generations than yours). Update the "
                "engine, or read pictures on the CPU (Models, Read pictures, Image encoder device: CPU; or run setup "
                "again with --vision cpu). From the encoder: " + evidence[:300])
    if text:
        msg = "the vision encoder did not start: " + text
    elif waited is not None:
        msg = f"the vision encoder did not start: it said nothing for {waited:g} s (it never reported READY)"
    else:
        msg = "the vision encoder did not start: it ended without saying anything"
    extra = []
    if code is not None:
        words = engine_client.engine_exit_words(code)
        if words:
            extra.append(words)
    shown = [x for x in lines[-3:] if x != text]
    if shown:
        extra.append("last lines of its log: " + " | ".join(x[:200] for x in shown))
    if extra:
        msg += " (" + "; ".join(extra) + ")"
    return msg


class Vision:
    """The resident image encoder: `strata-vision` (llama.cpp mtmd + the mmproj file) reads `ENC <image> <out>`
    lines and writes each image's embeddings; results are cached by the image's hash, so a conversation that
    sends the same picture again (every turn, with most clients) encodes it once."""

    @staticmethod
    def work_dir() -> Path:
        """#480: the directory for the encoder's files.  The engine reads the embeddings file named in a `GENI` line
        and the encoder the output named in an `ENC <image> <out>` line, both split at spaces, so a space in the path
        (a %TEMP% under a user name with a space) broke every image.  The system temp directory is used while its path
        has no space, then the working directory, then the system drive's root; the plain temp directory is the last
        resort (the encoder's own files are then named relative to it, see `_start`)."""
        for parent in (None, Path.cwd(), Path(os.environ.get("SystemDrive", "C:") + os.sep)):
            try:
                d = Path(tempfile.mkdtemp(prefix="strata-vision-", dir=parent))
            except OSError:
                continue                                       # not writable, or no such drive
            if " " not in str(d):
                return d
            try:
                d.rmdir()                                      # just created, so still empty
            except OSError:
                pass
        return Path(tempfile.mkdtemp(prefix="strata-vision-"))

    def __init__(self, cfg: dict, log=None, env: dict | None = None, lazy: bool = False):
        args = self.reader_args(cfg)
        self.dir = self.work_dir()
        self.spawn = (args, log, env)                   # to start it again after an unload
        # AMD: the GPU reader (strata-vision-hip) has the CPU reader (strata-vision, same program, no GPU runtime) to
        # fall back on when it does not start; _start switches to it by itself
        cpu = vision_reader.cpu_fallback(cfg)
        self.fallback = (self.reader_args(cpu), log, env) if cpu else None
        self.proc = None
        self.stopped = True
        self.lock = threading.Lock()
        self.cache: dict[str, tuple[Path, int]] = {}
        # opt-in (the vision section's "elastic": true, with the config's "vram_elastic"): the encoder starts with the
        # first picture and stops after elastic_idle_s without one; while it runs, the engine's expert cache keeps
        # elastic_mib of VRAM free for it (Service._vision_up / _vision_idle_down)
        self.max_tokens = self.token_limit(cfg)
        self.elastic = cfg.get("elastic") is True
        self.elastic_mib = int(cfg.get("elastic_mib") or 2048)
        self.elastic_idle_s = float(cfg.get("elastic_idle_s") or 120)
        if not lazy and not self.elastic:
            self._start()

    # What a picture may become when the config does not say (setup writes the same numbers, setup.VISION): an old, hand-edited or migrated
    # config without vision.max_tokens left the reader unclamped, and #3690 saw one picture become 93,700 image tokens (I7).  A key that is
    # there is the person's: 0 means no limit, as before.
    DEFAULT_MAX_TOKENS = {"gpu": 1024, "cpu": 300}

    @staticmethod
    def token_limit(cfg: dict) -> int:
        """The --max-tokens of a reader config: its max_tokens, else the default of its kind (GPU 1024, CPU 300)."""
        if cfg.get("max_tokens") is not None:
            try:
                return max(0, int(cfg["max_tokens"]))
            except (TypeError, ValueError):
                pass
        return Vision.DEFAULT_MAX_TOKENS["gpu" if cfg.get("gpu") else "cpu"]

    @staticmethod
    def reader_args(cfg: dict) -> list:
        absolute = lambda p: os.path.abspath(p) if os.path.dirname(str(p)) else p   # the encoder runs in its own dir (#480)  # noqa: E731
        args = [absolute(cfg["exe"]), "--mmproj", absolute(cfg["mmproj"]), "--model", absolute(cfg["model"])]
        if cfg.get("gpu"):
            args.append("--gpu")
        if cfg.get("threads"):
            args += ["--threads", str(cfg["threads"])]
        limit = Vision.token_limit(cfg)
        if limit:
            args += ["--max-tokens", str(limit)]
        if cfg.get("min_tokens"):                       # #767: mtmd's image_min_tokens (a hand-edited key)
            args += ["--min-tokens", str(cfg["min_tokens"])]
        return args

    def _ensure_dir(self):
        """#3370: a temp cleaner (Windows Storage Sense) removes the empty work folder while the encoder is unloaded; the next
        start then failed with WinError 267 for as long as the server lived.  Make it again; pictures cached in it are gone."""
        if not self.dir.is_dir():
            self.dir.mkdir(parents=True, exist_ok=True)
            self.cache.clear()

    def _start(self):
        """Start the reader.  When a GPU reader that has a CPU reader to fall back on does not start (the GPU runtime does not
        load, the card is not in its build, the card is busy), the CPU reader is started instead, for good, with one log line
        and no question."""
        try:
            self._start_reader()
        except (RuntimeError, OSError) as e:
            fallback = getattr(self, "fallback", None)
            if fallback is None:
                raise
            why = " ".join(str(e).split())[:240]
            print(f"[strata] the GPU picture reader did not start ({why}); pictures are read on the CPU", flush=True)
            self.spawn, self.fallback = fallback, None
            self._start_reader()

    def _start_reader(self):
        args, log, env = self.spawn
        self._ensure_dir()
        # cwd is the encoder's directory: the ENC lines name the files relative to it (#480: no space in the line)
        self.proc = engine_client.popen("the image encoder", args, role="vision", stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=log or subprocess.DEVNULL, text=True,
                                        encoding="utf-8", bufsize=1, env=env, cwd=self.dir)
        try:
            line = self._readline(VISION_READY_S, "the vision encoder did not start")
        except RuntimeError as e:                       # silent for VISION_READY_S: it was ended, and it is not left half-started
            self.proc, self.stopped = None, True
            if VISION_READY_S:
                raise RuntimeError(vision_start_error("", log, waited=VISION_READY_S)) from None
            raise RuntimeError(str(e)) from None
        if not line.startswith("READY"):
            proc, self.proc = self.proc, None
            self.stopped = True
            code = proc.poll() if hasattr(proc, "poll") else None   # an encoder that died by itself has its status already
            try:
                proc.kill()
                proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass
            if code is None:
                code = getattr(proc, "returncode", None)
            raise RuntimeError(vision_start_error(line, log, code))
        self.stopped = False

    def _readline(self, timeout: float, what: str) -> str:
        """One line from the encoder, waiting at most `timeout` s (#1317).  On a timeout the encoder is killed, so the
        next request starts a fresh one, and the read raises RuntimeError instead of holding the request turn."""
        proc = self.proc
        got: queue.Queue = queue.Queue()

        def read():
            try:
                got.put(proc.stdout.readline())
            except (OSError, ValueError):               # killed under the read
                got.put("")

        threading.Thread(target=read, daemon=True).start()
        try:
            return got.get(timeout=timeout)
        except queue.Empty:
            try:
                proc.kill()
            except OSError:
                pass
            raise RuntimeError(f"{what}: it said nothing for {timeout:.0f} s") from None

    def alive(self) -> bool:
        return self.proc is not None and not self.stopped and self.proc.poll() is None

    def unload(self):
        """Stop the encoder process (its VRAM or RAM goes back); the encoded images stay cached on disk."""
        self.close()
        self.stopped = True

    def restart(self):
        """Start the encoder again after an unload (or if it died); the cache of encoded images is kept."""
        if self.proc is not None:
            try:
                self.proc.kill()
            except OSError:
                pass
        self.proc = None
        self.stopped = True
        self._start()

    @staticmethod
    def load(source: str) -> bytes:
        if source.startswith("data:"):
            try:
                return base64.b64decode(source.split(",", 1)[1])
            except (IndexError, ValueError) as e:                # no comma, or not base64
                raise ValueError(f"the image's data: URL could not be read ({e})") from None
        if source.startswith(("http://", "https://")):
            return Vision.download(source)
        path = source[7:] if source.startswith("file://") else source
        # refused before anything looks at the path: file://host/share/x.png is \\host\share\x.png ("file://C:/x.png",
        # a drive, is a client's spelling of file:///C:/x.png)
        host = re.split(r"[/\\]", path, maxsplit=1)[0] if source.startswith("file://") else ""
        if network_path(path) or host and not re.fullmatch(r"[A-Za-z]:|localhost", host, re.IGNORECASE):
            raise ValueError("an image file must be on this computer: network paths (\\\\host\\share, //host/share, "
                             "file://host/...) are not read")
        if path and os.path.isfile(path):
            try:
                return Path(path).read_bytes()
            except OSError as e:
                raise ValueError(f"the image could not be read from {path} ({e})") from None
        raise ValueError("an image must be a data: URL, an http(s) URL or a local file path")

    @staticmethod
    def download(url: str) -> bytes:
        """An image URL's bytes, at most IMAGE_URL_MAX of them: the whole response was read, so a huge or endless one
        filled the memory.  One that cannot be read is a ValueError (a 400 that says so), not a dropped connection."""
        too_big = f"the image URL's file is over {IMAGE_URL_MAX >> 20} MiB"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "strata"}), timeout=60) as r:
                size = r.headers.get("Content-Length") or ""
                if size.isdigit() and int(size) > IMAGE_URL_MAX:
                    raise ValueError(too_big)
                data = r.read(IMAGE_URL_MAX + 1)
        except (OSError, HTTPException) as e:
            raise ValueError(f"the image URL could not be read: {e}") from None
        if len(data) > IMAGE_URL_MAX:
            raise ValueError(too_big)
        return data

    @staticmethod
    def _exif_rotated(data: bytes) -> bool:
        """A JPEG whose EXIF Orientation says it is stored turned (phones do this for portrait photos): the decoder
        (stb_image) ignores the tag.  False without Pillow, or when the tag is missing, 1 or unreadable (#1229)."""
        try:
            import io
            from PIL import Image
            with Image.open(io.BytesIO(data)) as im:
                return im.getexif().get(0x0112, 1) not in (None, 0, 1)
        except Exception:  # noqa: BLE001 - any failure leaves the image as it was sent
            return False

    @staticmethod
    def normalize(data: bytes) -> bytes:
        """The formats strata-vision's decoder (stb_image) reads pass through; anything else is converted to PNG.
        A JPEG with an EXIF Orientation other than 1 is turned upright first (needs Pillow)."""
        jpeg = data[:3] == b"\xff\xd8\xff"
        if (jpeg and not Vision._exif_rotated(data)) or data[:8] == b"\x89PNG\r\n\x1a\n" or data[:2] == b"BM" or \
                data[:6] in (b"GIF87a", b"GIF89a"):
            return data
        try:
            import io
            from PIL import Image
        except ImportError:
            raise ValueError("this image format needs Pillow (python -m pip install pillow); JPEG, PNG, BMP and "
                             "GIF work without it") from None
        try:
            im = Image.open(io.BytesIO(data))
            im.load()
            from PIL import ImageOps
            im = ImageOps.exif_transpose(im)             # the camera's orientation tag, applied (#1229)
        except Exception as e:
            raise ValueError(f"the image could not be read ({e})") from None
        if im.mode in ("RGBA", "LA", "P") and "transparency" in im.info or im.mode in ("RGBA", "LA"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))   # transparent areas become white, not black
            bg.paste(im, mask=im.split()[-1])
            im = bg
        elif im.mode != "RGB":
            im = im.convert("RGB")
        out = io.BytesIO()
        im.save(out, format="PNG")
        return out.getvalue()

    def encode_all(self, sources: list, cancel=None) -> list[tuple[Path, int]]:
        """encode() for every image of one request: the files of this request's earlier images are never evicted to
        make room for a later one (#1072: with more than 64 unique images the first was deleted before the request
        had read it).  `cancel` (#1999): a request whose client left does not encode its next pictures."""
        out: list[tuple[Path, int]] = []
        for src in sources:
            if cancel is not None and cancel.is_set():
                raise ClientGone()
            out.append(self.encode(src, keep=[p for p, _ in out]))
        return out

    def encode(self, source: str | bytes, keep=()) -> tuple[Path, int]:
        """-> (embeddings file, number of image tokens).  `source`: what load() reads, or the image's bytes.
        `keep`: files the caller still needs (never evicted by this call)."""
        data = self.normalize(source if isinstance(source, bytes) else self.load(source))
        key = hashlib.sha256(data).hexdigest()[:32]
        with self.lock:
            self._ensure_dir()
            if key in self.cache:
                self.cache[key] = self.cache.pop(key)                  # most recently used last
                return self.cache[key]
            if self.proc is None or self.proc.stdin is None:           # an elastic encoder not started yet, or one that was unloaded
                raise ValueError("the picture reader is not running")
            img, out = self.dir / f"{key}.img", self.dir / f"{key}.sve"
            img.write_bytes(data)
            try:
                self.proc.stdin.write(f"ENC {img.name} {out.name}\n")      # relative to the encoder's cwd (#480)
                self.proc.stdin.flush()
                try:
                    line = self._readline(VISION_ENCODE_S, "the image could not be read").strip()
                except RuntimeError as e:                 # silent for VISION_ENCODE_S: the encoder was ended (#1317)
                    self.stopped = True                   # alive() is False, so the next request starts it again
                    raise ValueError(str(e)) from None
            finally:                                                   # #352: also when the encoder's pipe is gone
                img.unlink(missing_ok=True)
            if not line.startswith("OK"):
                raise ValueError("the image could not be read: " + (line[4:] if line.startswith("ERR") else
                                                                     "the vision encoder stopped"))
            self.cache[key] = (out, int(line.split()[1]))
            self.warn_if_over_limit(self.cache[key][1])
            held = {str(p) for p in keep} | {str(out)}
            for old in list(self.cache):                               # oldest first, but never one still in use
                if len(self.cache) <= 64:
                    break
                if str(self.cache[old][0]) not in held:
                    self.cache.pop(old)[0].unlink(missing_ok=True)
            return self.cache[key]

    def warn_if_over_limit(self, tokens: int) -> None:
        """I7: one picture that became more than 4 times the limit is logged with the numbers, so the next report has them (the reader's own
        stderr has the warm-up line).  A reader built before the limit was enforced inside it (strata-vision refuses these now) still lets them by."""
        limit = getattr(self, "max_tokens", 0)
        if limit and tokens > 4 * limit:
            print(f"[strata] images: one picture became {tokens} image tokens, more than 4 times the limit of {limit} (vision.max_tokens); "
                  "the picture reader is not applying its limit. Update the engine, and send the line 'strata-vision: warmed up at N image "
                  "tokens' from the reader's log with the report", flush=True)

    def close(self):
        proc, self.proc = self.proc, None
        if proc is None:
            self.stopped = True
            return
        try:
            proc.stdin.write("QUIT\n")
            proc.stdin.flush()
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except OSError:
                pass
            try:
                proc.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                pass
        self.stopped = True

    def shutdown(self):
        """#914: close() for good: the server ends, so the encoder's directory (one ~10 MB .sve per image, on a tmpfs
        /tmp that is RAM) goes with it instead of piling up with every start.  unload() keeps it for the restart."""
        self.close()
        shutil.rmtree(self.dir, ignore_errors=True)


def card_roles_lines(cfg: dict) -> list[str]:
    """The plain lines printed at start: the roles (cards numbered as nvidia-smi numbers them), a hand-edited "gpu" that beat them, and the
    all-Off case.  SystemExit when every card is Off and the engine cannot start without one (the CUDA, HIP and SYCL engines)."""
    roles, problem = gpu_roles.saved_roles(cfg)
    out = [f"[strata] {problem}"] if problem else []
    if roles is None:
        return out
    if engine_client.gpu_all_off(cfg):
        if cfg.get("backend") in ("cpu", "metal") or sys.platform == "darwin":
            out.append("[strata] every graphics card is Off: " + gpu_roles.CPU_ONLY_LINE)
            return out
        raise SystemExit("[strata] every graphics card is set to Off, and this engine cannot start without one. Turn a card on "
                         "(Main or Helper) in the app or with strata launch --gpu-roles.")
    word = {"main": "Main", "helper": "Helper", "off": "Off"}
    out.append("[strata] card roles: " + ", ".join(f"{word[r]} GPU {i}" for r in ("main", "helper", "off")
                                                    for i in sorted(roles) if roles[i] == r) + " (numbers as nvidia-smi gives them)")
    return out


def engine_binary(exe: str, cfg: dict | None = None) -> str:
    """The file that holds the engine's own option strings, for a feature check that reads it.  The Intel launcher
    (sycl/serve/strata-sycl.sh) is a shell script that runs its repo's STRATA_SYCL_BIN (default build-sycl-aot/strata)
    in a container, so the check reads that binary, not the script.  Any other `exe` is the engine itself."""
    p = Path(exe)
    if p.name == "strata-sycl.sh":
        rel = ((cfg or {}).get("env") or {}).get("STRATA_SYCL_BIN") or os.environ.get("STRATA_SYCL_BIN") \
            or "build-sycl-aot/strata"
        b = p.resolve().parents[2] / rel
        if b.is_file():
            return str(b)
    return exe


def effort_end_args(cfg: dict, exe: str, tok) -> list[str] | None:
    """#458 (opt-in): the engine arguments for "effort_position": "end" - the id of "system" as --tail-role-token, so
    the engine checkpoints in front of the trailing effort turn - or None when the config leaves it at the top (the
    default) or the engine is too old for it (said once; the prompt stays the default one).  ValueError for a value
    other than "start" / "end"."""
    pos = cfg.get("effort_position", "start")
    if pos not in ("start", "end"):
        raise ValueError(f'"effort_position" must be "start" (the default) or "end", not {pos!r}')
    if pos == "start":
        return None
    try:
        with open(engine_binary(exe, cfg), "rb") as f:
            known = b"--tail-role-token" in f.read()
    except OSError:
        known = False
    role = tok.encode("system", parse_special=True)
    if not known or len(role) != 1:
        print("[strata] effort_position \"end\" needs engine 0.1.39 or newer (--tail-role-token): the reasoning "
              "effort stays at the top of the prompt", flush=True)
        return None
    print("[strata] effort_position end: a request's non-default reasoning effort goes right before the answer, so "
          "changing it keeps the cached conversation", flush=True)
    return ["--tail-role-token", str(role[0])]


def engine_silence_s(cfg: dict) -> float:
    """#481: the config's "engine_silence_s" - seconds an engine may print nothing during a request before the server
    ends it (default ENGINE_SILENCE_S; 0 = wait forever).  ValueError for anything but a number >= 0."""
    v = cfg.get("engine_silence_s")
    if v is None:
        return engine_client.ENGINE_SILENCE_S
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
        raise ValueError(f'"engine_silence_s" must be a number of seconds >= 0 (0 = no limit), not {v!r}')
    return float(v)


def apply_ready_timeout(cfg: dict) -> None:
    """The config's "engine_ready_s": how long a start may take before READY (default ENGINE_READY_S, 900).  A model of 300+ GB read
    from a slow disk (DeepSeek-V4.1-Flash: setup writes 3600) needs more.  ValueError for anything but a number > 0."""
    v = cfg.get("engine_ready_s")
    if v is None:
        return
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not v > 0:
        raise ValueError(f'"engine_ready_s" must be a number of seconds above 0, not {v!r}')
    engine_client.ENGINE_READY_S = float(v)


def igpu_split_warning(cfg: dict, archs: dict[int, str] | None = None) -> str | None:
    """A plain line when the layers are split over built-in Radeon graphics AND a card with memory of its own.  A recommendation, not a
    refusal: the split runs as asked.  The iGPU's memory is the RAM, it usually draws the desktop, and compute on it can freeze the screen."""
    if cfg.get("backend") != "hip":
        return None
    gl = engine_client.gpu_list(cfg)
    if len(gl) < 2:
        return None
    ar = archs if archs is not None else engine_client.hip_gpu_archs()
    ig = engine_client.igpu_cards(gl, ar)
    if not ig or len(ig) == len(gl):
        return None
    return ("[strata] note: GPU " + ", ".join(str(i) for i in sorted(ig)) + " is built-in graphics (its memory is the system RAM, and it "
            "usually draws the desktop). Running part of the model on it can freeze the screen and uses RAM twice; the cards with memory of "
            "their own are faster alone (docs/STRIX_HALO.md). Turn it Off in the card roles, or name only the other card, to leave it out")


def card_note(cfg: dict, env: dict) -> str | None:
    """One plain line that ties the engine's own numbering to the person's: the engine counts the cards it is given from 0 ("GPU 0",
    "CUDA0" in its log), so a person who chose GPU 1 would read "GPU 0" and think the choice was lost.  Said only when the two differ
    (NVIDIA, numbered as nvidia-smi numbers them); None otherwise."""
    if cfg.get("backend") == "hip" or not engine_client.gpu_list(cfg):
        return None
    mine = [x.strip() for x in str((env or {}).get("CUDA_VISIBLE_DEVICES", "")).split(",") if x.strip() != ""]
    if not mine or mine == [str(i) for i in range(len(mine))]:
        return None
    names = [f"CUDA{i}" for i in range(len(mine))]
    if len(mine) == 1:
        return f"[strata] running on GPU {mine[0]} (as nvidia-smi numbers it); the engine's log calls it GPU 0 / CUDA0"
    return (f"[strata] cards in stage order: GPU {', then GPU '.join(mine)} (as nvidia-smi numbers them); the engine's log calls them "
            f"{', '.join(names)} - the first runs the first layers and the picture reader, the last runs the head and the draft layer")


def vision_card_off_igpu(env: dict, archs: dict[int, str] | None = None) -> int | None:
    """AMD: the picture reader is a separate program that sees the engine's HIP_VISIBLE_DEVICES and runs on the FIRST visible card.  With a
    layer split that starts on a built-in Radeon (Strix Halo beside an R9700) that is the iGPU: the card that draws the desktop and whose
    memory is the RAM the experts live in.  Returns the HIP ordinal of the last visible card with memory of its own (the Main stage), or
    None when nothing needs changing (one card, no integrated chip among them, or the topology is unknown)."""
    vis = [x.strip() for x in str(env.get("HIP_VISIBLE_DEVICES") or "").split(",") if x.strip().isdigit()]
    if len(vis) < 2:
        return None
    idx = [int(x) for x in vis]
    ar = archs if archs is not None else engine_client.hip_gpu_archs()
    ig = engine_client.igpu_cards(idx, ar)
    own = [i for i in idx if i not in ig and ar and i in ar]
    if not ig or not own:
        return None
    print(f"[strata] the picture reader runs on GPU {own[-1]}, not on the built-in graphics (GPU {','.join(str(i) for i in sorted(ig))}) "
          "that draws the desktop; \"vision\": {\"cuda_device\": N} chooses another card", flush=True)
    return own[-1]


def vision_env(cfg: dict, env: dict, archs: dict[int, str] | None = None) -> dict:
    """The image encoder's environment: the engine's, unless the config's vision section names its own "cuda_device"
    (numbered like nvidia-smi) - then the encoder runs on that card alone, so a spare GPU can hold it while the engine
    keeps all of its own cards' VRAM (#408, Efs-O).  Without it, nothing changes."""
    dev = (cfg.get("vision") or {}).get("cuda_device")
    if dev is None and cfg.get("backend") == "hip":
        dev = vision_card_off_igpu(env, archs)
    if dev is None:
        return env
    env = dict(env)
    if cfg.get("backend") == "hip":
        env["HIP_VISIBLE_DEVICES"] = str(dev)
    else:
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(dev)
    return env


class ByteTokenizer:
    """Tiny stand-in tokenizer for tests without the pack: one id per UTF-8 byte, specials as ids >= 256."""
    SPECIALS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|vision_start|>", "<|image_pad|>", "<|vision_end|>"]

    ALWAYS = ()                                     # specials matched without parse_special (type 4, as <think>)

    @property
    def max_special_len(self):
        return max(len(s) for s in self.SPECIALS)

    @property
    def control_tokens(self):
        return [s for s in self.SPECIALS if s not in self.ALWAYS]

    def encode(self, text, parse_special=False, plain=()):
        return self.encode_marked(text, parse_special, plain)[0]

    def encode_marked(self, text, parse_special=False, plain=()):
        """encode() and its resume points (after each special), as strata_tokenizer's for PromptEncoder."""
        out, marks, i = [], [], 0
        while i < len(text):
            for k, s in enumerate(self.SPECIALS):
                if (parse_special or s in self.ALWAYS) and text.startswith(s, i) and not any(
                        a <= i < b for a, b in plain):
                    out.append(256 + k)
                    i += len(s)
                    marks.append((i, len(out)))
                    break
            else:
                out.extend(text[i].encode("utf-8"))
                i += 1
        return out, marks

    def decode(self, ids, errors="replace"):
        raw = bytearray()
        for t in ids:
            raw += self.SPECIALS[t - 256].encode() if t >= 256 else bytes([t])
        return raw.decode("utf-8", errors=errors)


def token_pieces(tok, ids) -> list[str]:
    """Each token's text, so that the pieces join to the decoded text: a character split across tokens belongs to
    the token that completes it (the others get "").  An id outside the vocabulary is shown as <?id>."""
    raw = getattr(tok, "token_bytes", None)
    dec = codecs.getincrementaldecoder("utf-8")("replace")
    pieces = []
    for t in ids:
        try:
            b = raw(t) if raw else tok.decode([t], errors="surrogateescape").encode("utf-8", "surrogateescape")
        except (IndexError, KeyError):
            pieces.append(dec.decode(b"", final=True) + f"<?{t}>")
            continue
        pieces.append(dec.decode(b))
    tail = dec.decode(b"", final=True)
    if tail and pieces:
        pieces[-1] += tail
    return pieces


# ------------------------------------------------------------------------------------------------ core
class Detokenizer:
    """Incremental decode: each token's bytes go through an incremental UTF-8 decoder, which emits the complete
    characters and holds a multi-byte character split across tokens until it is complete (invalid bytes become
    U+FFFD, as a whole decode with errors="replace" makes them).  Constant time per token - the old re-decode of
    every generated id cost 2 ms per token after 8K tokens and 4 ms after 16K (perf-review F-1).  A tokenizer
    without `token_bytes` (the tests' byte tokenizer) keeps the re-decode."""

    def __init__(self, tok):
        self.tok, self.ids, self.sent = tok, [], 0
        self.inc = codecs.getincrementaldecoder("utf-8")(errors="replace") if hasattr(tok, "token_bytes") else None

    def pending(self) -> bool:
        """A character is split across the tokens so far: its first bytes are held."""
        if self.inc is not None:
            return bool(self.inc.getstate()[0])
        return self.tok.decode(self.ids).endswith("\ufffd")

    def push(self, t: int) -> str:
        if self.inc is not None:
            return self.inc.decode(self.tok.token_bytes(t))
        self.ids.append(t)
        text = self.tok.decode(self.ids)
        if text.endswith("�"):
            return ""
        delta, self.sent = text[self.sent:], len(text)
        return delta


OPENAI_MAX_STOP = 4                                  # OpenAI's limit on `stop`


def stop_strings(req: dict) -> list[str]:
    """The request's stop strings: OpenAI's `stop` (a string or a list of up to 4) or Anthropic's `stop_sequences`
    (a list).  Empty strings are left out.  A bad value is a ValueError, so a 400 before anything is generated."""
    req = req or {}
    if req.get("stop") is not None:
        stop = req["stop"]
        stop = [stop] if isinstance(stop, str) else stop
        if not isinstance(stop, list) or not all(isinstance(x, str) for x in stop):
            raise ValueError("stop must be a string or a list of strings")
        if len(stop) > OPENAI_MAX_STOP:
            raise ValueError(f"stop takes at most {OPENAI_MAX_STOP} strings")
    elif req.get("stop_sequences") is not None:
        stop = req["stop_sequences"]
        if not isinstance(stop, list) or not all(isinstance(x, str) for x in stop):
            raise ValueError("stop_sequences must be a list of strings")
    else:
        return []
    return [x for x in stop if x]


class StopMatcher:
    """Cuts the answer's text at the first stop string.  Text that could still be the start of a stop string is held
    back (only that tail, never more), so a stop string split across tokens or chunks is still found and never sent."""

    def __init__(self, stops: list[str]):
        self.stops, self.held, self.hit = stops, "", None

    def push(self, text: str) -> str:
        """-> the text that is safe to send now; after a match, self.hit is the stop string and nothing more comes."""
        if self.hit is not None:
            return ""
        buf = self.held + text
        first = None
        for s in self.stops:
            i = buf.find(s)
            if i >= 0 and (first is None or i < first[0]):
                first = (i, s)
        if first is not None:
            self.held, self.hit = "", first[1]
            return buf[:first[0]]
        keep = 0                                      # the longest end of buf that a stop string starts with
        for s in self.stops:
            for k in range(min(len(s) - 1, len(buf)), keep, -1):
                if buf.endswith(s[:k]):
                    keep = k
                    break
        self.held = buf[len(buf) - keep:] if keep else ""
        return buf[:len(buf) - keep]

    def flush(self) -> str:
        held, self.held = self.held, ""
        return held


# Windows names that open a device whatever their extension or folder (Microsoft, "Naming Files, Paths, and Namespaces")
_WIN_DEVICES = {"con", "prn", "aux", "nul", "conin$", "conout$", *(f"com{i}" for i in "123456789\u00b9\u00b2\u00b3"),
                *(f"lpt{i}" for i in "123456789\u00b9\u00b2\u00b3")}


def slot_filename_problem(name) -> str | None:
    """/slots/0 {"filename": NAME}: why NAME is not one plain file name inside the slot save path, on Linux and on
    Windows alike (None when it is).  No separator or drive (`D:x`), no stream (`a:b`), no device (`NUL.txt`), no
    trailing dot or space (Windows drops them), no control character, no leading dot (temporary files use it)."""
    if not isinstance(name, str) or not name:
        return "must be a non-empty string"
    if len(name.encode("utf-8", "surrogatepass")) > 200:
        return "is too long (at most 200 bytes)"
    if any(ord(c) < 32 or ord(c) == 127 for c in name):
        return "must not contain control characters"
    if any(c in name for c in '/\\:*?"<>|'):
        return "must be a plain file name (no / \\ : * ? \" < > |)"
    if name.startswith("."):
        return "must not start with a dot"
    if name.endswith((".", " ")):
        return "must not end with a dot or a space"
    if name.split(".")[0].rstrip(" ").lower() in _WIN_DEVICES:
        return "names a Windows device"
    try:
        name.encode("utf-8")
    except UnicodeEncodeError:
        return "is not valid Unicode"
    return None


def slot_save_dir(value, base: str | None = None) -> str:
    """The slot save path as one absolute, normalized directory (relative to `base`, else the server's working
    directory), created private to this user when missing.  ValueError for anything else."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("slot_save_path must be a directory path (a string)")
    if any(ord(c) < 32 for c in value):
        raise ValueError("slot_save_path must not contain control characters")
    path = os.path.abspath(os.path.join(base, value) if base and not os.path.isabs(value) else value)
    os.makedirs(path, mode=0o700, exist_ok=True)
    if not os.path.isdir(path):
        raise ValueError(f"slot_save_path {path} is not a directory")
    return path

class Service:
    def __init__(self, engine: Engine, tokenizer, template: ChatTemplate, model_name: str = "qwen3.8-flash-next",
                 vision: Vision | None = None, sampling_defaults: dict | None = None,
                 fit_max_tokens: bool = True):
        self.engine, self.tok, self.template, self.model, self.vision = engine, tokenizer, template, model_name, vision
        self.literals = literal_tags(getattr(tokenizer, "control_tokens", ()))   # texts that stay text inside a message
        self.fit_max_tokens = fit_max_tokens          # default: clamp the output cap to the room left; False (--no-fit-max-tokens) = 400
        self.aliases: list[str] = []                  # #297: other names of the model (the config's `aliases`)
        self.answer_room = True                       # 0.1.45: see ANSWER_ROOM_MIN_CAP (the config's "answer_room": false turns it off)
        self.reasoning_close_retry = False            # #1053 (opt-in): close the thinking once when a reply ends inside it
        self.codex_compaction_cache = False           # #924 (opt-in): a Codex compaction is rendered with its conversation's tools
        self.codex_thread_titles = False              # #923 (opt-in): answer Codex's thread-title turns without the engine
        self.sampling_defaults = dict(sampling_defaults or {})   # the run config's `sampling` block
        self.model_arch = None                        # the config's family ("qwen4exp", "qwen35moe"): None = no model-card defaults
        self.shared = {}                              # the web app's Chat settings for every client (POST /settings)
        self.shared_path = None                       # where they are kept between starts (next to the config)
        self.config_effort = None                     # #1641: the config's "reasoning_effort": the level for requests with none
        self.keep_awake = None                        # #1727 (opt-in): strata_server.keepawake.KeepAwake, the config's "prevent_sleep"
        self.fifo = threading.Lock()
        self.slot_save_path = None                    # --slot-save-path: /slots/0?action=save|restore (off when None)
        self.embeddings = threading.local()           # the current request's image embeddings file (GENI)
        self.api_key = ""                              # when set, /v1/* needs it (Bearer or x-api-key)
        # #458 (opt-in, the config's "effort_position": "end"): a non-default reasoning effort goes in a short system
        # turn before the answer instead of the top of the prompt, so switching it keeps the cached conversation
        self.effort_end = False
        self.conv_log = ConvCacheLog()                  # #596: the parked conversations, from the engine's log
        self.config_path = None                         # #564: the run config the web page's Settings view edits
        self.config_lock = threading.Lock()
        # #321: browser pages of these origins may call /v1/* (CORS; "*" = any page - only with an api_key that
        # matters); empty = no CORS headers at all, as before
        self.cors_origins: list[str] = []
        # #321: origins that count as Strata's own page for /settings and MCP tools, e.g. the web app reached through a
        # reverse proxy or tunnel whose Host differs ("https://strata.example.com"); never a wildcard
        self.trusted_origins: list[str] = []
        # DNS rebinding: extra Host names this server answers to (the config's allowed_hosts, $STRATA_ALLOWED_HOSTS;
        # "*" = any), and every name it answers to, which serve() works out from the address it listens on
        self.allowed_hosts: list[str] = []
        self.host_names: set[str] = set(LOOPBACK_NAMES)
        self.status = {"busy": False, "queued": 0}      # GET /status: what the model is doing right now
        self.rate = collections.deque(maxlen=32)        # (time, generated) samples for the live tok/s window
        # "parallel" (the engine's batch slots): every running request's own status and rate window; self.status
        # then says busy while any runs and shows the newest one
        self.live_reqs: dict[int, tuple[dict, collections.deque]] = {}
        # Privacy-safe correlation for terminal telemetry. No prompts, client identity or tool text is stored.
        self.terminal_requests: dict[str, dict] = {}
        self.terminal_events = collections.deque(maxlen=128)  # fixed-shape request transitions only
        self.terminal_event_seq = 0
        # HTTP admission is reserved before generation or SSE headers. This bounds pending inference calls even
        # while a model is loading or a request is between MCP rounds; the engine's own slots remain unchanged.
        self.max_queue: int | None = None
        self.queue_timeout_s: float | None = None
        self.admission_reserved = 0
        self.admission_rejected = 0
        self.queue_timeouts = 0
        self.client_disconnect_cancels = 0
        self.draining = False
        self.active_cancels: set[threading.Event] = set()
        # #332: the API request monitor (/api-monitor) keeps the last 100 requests' prompts and answers in memory,
        # so it is off unless the config's "api_monitor" (or --api-monitor) turns it on
        self.api_monitor = False
        self.context_list = None                        # the Context view's last list (context_view)
        self.context_shown = None                       # ... and the context window it read last (context_window)
        # the engine's last GET_MEMORY_USE answer (memory_view)
        self.memory_last = {"at": None, "memories": [], "settings": {}}
        self.api_requests = collections.deque(maxlen=100)  # bounded I/O in memory; no headers or API keys
        self.request_trace = threading.local()
        self.history = collections.deque(maxlen=500)    # the last finished requests, newest last (GET /metrics)
        # since the server started (the Monitor's totals, issue #35)
        self.totals = {"since": time.time(), "requests": 0, "prompt_tokens": 0, "reused": 0, "output_tokens": 0,
                       "prompt_ms": 0.0, "decode_ms": 0.0,
                       "drafts_offered": 0, "drafts_accepted": 0}   # #457: the MTP drafts, summed where reported
        self.last_timings = None                         # the last finished request's, llama.cpp's names (/v1/status)
        # /metrics/prometheus' two throughput gauges need the totals as they stood at the last scrape, because
        # llama.cpp averages them over the window between two scrapes (#2341).
        self.scrape_bucket = (0, 0.0, 0, 0.0)
        from strata_server.prometheus import Latencies
        self.latencies = Latencies()                     # GET /metrics in Prometheus text: the latency histograms
        self.last_request_at = None                      # when a request last started or finished
        self.started_at = time.time()
        self.instance_id = uuid.uuid4().hex
        self.status_lock = threading.Lock()
        self.admission_changed = threading.Condition(self.status_lock)
        self.mcp = None                                  # strata_server/mcp.py's McpHub when MCP servers are configured
        self.offline = False                             # no image URL fetch or MCP subprocess in strict local mode
        self.telemetry_key = ""                         # optional read-only key, valid only on /v1/telemetry
        # sharing the GPU with other programs (all off by default): unload the engine after this many idle seconds,
        # only start it again when this much VRAM is free, and run this command first (e.g. to unload another
        # server's model); the next request after an unload starts the engine again
        self.idle_unload_s = 0
        self.min_free_vram_mib = 0
        self.before_load = None
        self.before_load_timeout_s = 120.0               # #1997: how long the before_load command may run
        self.before_load_ignore_failure = False          # #1997: load even when it fails, without a VRAM check
        self.vram_free_wait_s = 15.0                     # how long a full card is waited on (memory given back)
        self.before_load_strict = False                  # #1997: refuse (503) whenever it fails
        self.vram_reserve = None                         # #533: the last POST /v1/vram reserve (None: the start's)
        self.vram_reserve_asked = False                  # a POST /v1/vram was made (vram_reserve None is then asked)
        self.vision_held = False                         # an elastic encoder holds VRAM the expert cache gave it
        self.last_image_at = 0.0                         # when an image turn last ended (the elastic encoder's idle)
        self.reasoning_loop_recovery = False
        self.reasoning_budget_configured = False         # the config named reasoning_budget_tokens (0 included): no family default
        self.reasoning_budget_tokens = 0                 # #123: the config's default thinking budget (0: none)
        self.repeat_stop_tokens = REPEAT_STOP_TOKENS     # #606: one token this many times in a row ends a reply (0: off)
        self.anthropic_think_unasked = True               # #278: "anthropic_thinking": "on_request" -> False
        self.tool_call_recovery = False                   # opt-in: tool calls in forms next to the template's
        self.images = None                                # strata_server/images.py's ImageGen when the config has "images"
        self.restore_after_load = None                    # a session file an image job saved before it unloaded
        self.stop_ids = set(tokenizer.encode(IM_END, parse_special=True) +
                            tokenizer.encode("<|endoftext|>", parse_special=True))
        self.family = None                              # the model family ("deepseek41"): set_family() below
        self.parser_cls = OutputParser                  # reads the model's output (reasoning, content, tool calls)
        # #567: a prompt re-encodes only what follows the last special token it shares with a recent prompt (the
        # same ids as a full encode: tools/strata_tokenizer.py PromptEncoder).  Tokenizers without resume points
        # encode in full.
        self.prompts = None
        if hasattr(tokenizer, "encode_marked") and hasattr(tokenizer, "max_special_len"):
            from strata_tokenizer import PromptEncoder
            # one kept prompt per conversation the engine's cache can hold (--conversation-cache-slots); the
            # encoder's own default without that setting
            slots = conversation_slots(engine)
            self.prompts = PromptEncoder(tokenizer, keep=slots) if slots is not None and slots > 0 \
                else PromptEncoder(tokenizer)

    def set_family(self, family: str | None) -> None:
        """A family whose text differs from the Qwen form this server was written for (DeepSeek-V4.1: its own end
        token, DSML tool calls, reasoning-effort words).  The others keep the defaults."""
        self.family = family
        if family == "deepseek41":
            from strata_server import dsml
            eos = (getattr(self.tok, "special_ids", None) or {}).get("eos", 1)
            self.stop_ids = {int(eos)}
            self.parser_cls = dsml.DsmlOutputParser
            self._force_text = dsml.to_dsml_force
        import strata_server.frontend as _frontend
        _frontend.use_family(family)

    _force_text = None

    def force_text(self, force):
        """forced_call's opening of the call the reply must make, in this model's spelling."""
        return self._force_text(force) if self._force_text is not None else force

    def loaded(self) -> bool:
        return not hasattr(self.engine, "alive") or self.engine.alive()

    def admission_limit(self) -> int | None:
        if self.max_queue is None:
            return None
        return max(1, int(getattr(self.engine, "batch", 0) or 0)) + self.max_queue

    def _terminal_event_locked(self, request_id: str, event: str, outcome: str | None = None) -> None:
        """Append one redacted transition while holding status_lock; never include request content."""
        if event not in ("queued", "admitted", "reading", "first_token", "finished", "cancelled",
                         "disconnected", "timeout", "error"):
            return
        if outcome is not None and outcome not in ("stop", "length", "cancel", "disconnect", "timeout", "error"):
            outcome = "error"
        self.terminal_event_seq += 1
        row = {"seq": self.terminal_event_seq, "at": time.time(),
               "request_id": request_id, "event": event}
        if outcome is not None:
            row["outcome"] = outcome
        self.terminal_events.append(row)

    def reserve_inference(self) -> bool | None:
        """Reserve one bounded HTTP inference call before response headers are written."""
        with self.status_lock:
            if self.draining:
                return None
            limit = self.admission_limit()
            if limit is not None and self.admission_reserved >= limit:
                self.admission_rejected += 1
                return False
            self.admission_reserved += 1
            return True

    def release_inference(self) -> None:
        with self.admission_changed:
            self.admission_reserved -= 1
            self.admission_changed.notify_all()

    def note_client_disconnect(self, cancel: threading.Event) -> None:
        """Cancel once; count only work that was still queued or active when the client left."""
        with self.status_lock:
            if not cancel.is_set():
                request_id = getattr(cancel, "_strata_request_id", None)
                active = request_id in self.terminal_requests
                cancel.set()
                if active:
                    self.client_disconnect_cancels += 1

    @contextlib.contextmanager
    def _inference_fifo(self, deadline: float | None, cancel: threading.Event | None = None):
        if deadline is None:
            with self.fifo:
                yield
            return
        while True:
            if cancel is not None and cancel.is_set():
                raise QueueCancelled("the client disconnected while waiting for the engine")
            left = deadline - time.monotonic()
            if left <= 0:
                raise engine_client.QueueWaitExpired("the inference queue wait deadline expired; retry later")
            if self.fifo.acquire(timeout=min(0.1, left)):
                break
        try:
            if cancel is not None and cancel.is_set():
                raise QueueCancelled("the client disconnected while waiting for the engine")
            if time.monotonic() >= deadline:
                raise engine_client.QueueWaitExpired("the inference queue wait deadline expired; retry later")
            yield
        finally:
            self.fifo.release()

    def begin_drain(self) -> int:
        """Stop new inference admission and return the number of owned calls to drain."""
        with self.admission_changed:
            self.draining = True
            self.admission_changed.notify_all()
            return self.admission_reserved

    def wait_for_drain(self, timeout: float) -> int:
        """Wait for reserved inference calls; return the number still in flight."""
        deadline = time.monotonic() + max(0.0, timeout)
        with self.admission_changed:
            while self.admission_reserved:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                self.admission_changed.wait(left)
            return self.admission_reserved

    def cancel_owned_requests(self) -> int:
        """Cancel only requests admitted by this Service, never unrelated GPU work."""
        with self.status_lock:
            active = list(self.active_cancels)
        for cancel in active:
            cancel.set()
        return len(active)

    def slot_action(self, slot: str, action: str, filename) -> tuple[int, dict]:
        """llama-server's POST /slots/{id}?action=save|restore {"filename": ...}: the conversation the engine holds,
        to or from a file in slot_save_path.  -> (HTTP status, body)."""
        def error(code, message):
            return code, {"error": {"code": code, "message": message,
                                    "type": "invalid_request_error" if code < 500 else "server_error"}}
        if not self.slot_save_path or not hasattr(self.engine, "session_file"):
            return error(501, "slot save/restore is disabled (start the server with --slot-save-path DIR)")
        capacity = int(getattr(self.engine, "batch", 0) or 0)
        if capacity and (capacity > 1 or not getattr(self.engine, "request_protocol", False)):
            # This API names one model state; it cannot select among concurrently executing contexts.
            return error(501, "slot save/restore is not available with parallel requests (\"parallel\" / --batch)")
        if slot != "0":
            return error(400, "this server has one slot: 0")
        if action not in ("save", "restore"):
            return error(400, "action must be save or restore")
        why = slot_filename_problem(filename)
        if why:
            return error(400, f"filename {why}")
        path = os.path.join(self.slot_save_path, filename)
        if action == "restore":
            try:
                os.lstat(path)                              # the engine opens it without following a link
            except OSError:
                return error(404, f"no saved session named {filename}")
        self.last_request_at = time.time()                  # the idle unload counts this as activity
        with self.status_lock:
            self.status["queued"] += 1
        waiting = True
        try:
            with self.fifo:
                with self.status_lock:
                    self.status["queued"] -= 1
                waiting = False
                if not self.loaded():
                    return error(503, "the model is not loaded")
                with self.status_lock:
                    self.status.update(busy=True, phase="saving the session" if action == "save"
                                       else "restoring a session", started=time.time(), first_token=None,
                                       prompt_tokens=None, generated=None, max_tokens=None)
                try:
                    if action == 'save' and getattr(self.engine, 'session_save_reclaim', False):
                        from strata_server.session_save_retry import session_file
                        r = session_file(self, action, path, engine_client.SessionRefused)
                    else:
                        r = self.engine.session_file(action, path)
                except engine_client.SessionRefused as e:
                    body = error(e.status, str(e))
                    body[1]["error"]["kind"] = e.kind
                    if e.published:
                        body[1]["error"]["published"] = True
                    return body
                except ValueError as e:                     # the request itself (an invalid command)
                    return error(400, str(e))
                finally:
                    with self.status_lock:
                        self.status["busy"] = False
                    self.last_request_at = time.time()
        finally:
            if waiting:                                     # never reached the FIFO (an exception while waiting)
                with self.status_lock:
                    self.status["queued"] -= 1
        if action == "save":
            return 200, {"id_slot": 0, "filename": filename, "n_saved": r["tokens"], "n_written": r["bytes"],
                         "timings": {"save_ms": r["ms"]}}
        return 200, {"id_slot": 0, "filename": filename, "n_restored": r["tokens"], "n_read": r["bytes"],
                     "timings": {"restore_ms": r["ms"]}}

    def set_aliases(self, aliases) -> None:
        """#297: the config's `aliases` - a list of names (or one comma-separated string), like llama-server's --alias.
        ValueError for anything else."""
        if aliases is None:
            aliases = []
        if isinstance(aliases, str):
            aliases = aliases.split(",")
        if not isinstance(aliases, list) or not all(isinstance(x, str) for x in aliases):
            raise ValueError(f"aliases={aliases!r}: expected a list of model names")
        names = []
        for x in (x.strip() for x in aliases):
            if x and x != self.model and x not in names:
                names.append(x)
        self.aliases = names

    def model_names(self) -> list[str]:
        return [self.model, *self.aliases]

    def model_for(self, req) -> str:
        """The name to answer with: the request's own when it is the model's name or an alias (#297), else the model's.
        Other names are still served, as before."""
        asked = req.get("model") if isinstance(req, dict) else None
        return asked if isinstance(asked, str) and asked in self.aliases else self.model

    def reported_ctx(self) -> int:
        """The context the endpoints report (/v1/status, /v1/models, /props, /health, /slots, /metrics): the engine's,
        or the last one it reported while it restarts (max_context is 0 until READY, and a client that sizes its
        prompt from it would send nothing, #351).  Reporting only: requests are checked in prepare()."""
        return int(getattr(self.engine, "max_context", 0) or getattr(self.engine, "known_ctx", 0) or 0)

    def reasoning_budget(self, req) -> int | None:
        """#123: the most tokens this request may think, or None: the request's `reasoning_budget_tokens` (or, as
        llama.cpp reads it, `thinking_budget_tokens`), else the config's.  0 (or less) means no budget, so a request can
        turn a configured one off.  ValueError (a 400) for anything that is not a whole number."""
        req = req if isinstance(req, dict) else {}
        value = req.get("reasoning_budget_tokens")
        if value is None:                                # a null does not hide the other name
            value = req.get("thinking_budget_tokens")
        if value is None:
            value = self.reasoning_budget_tokens
            if not value and self.family == "deepseek41" and not self.reasoning_budget_configured:
                value = DS41_DEFAULT_BUDGET          # DeepSeek-V4.1 only, as Maya ships it: no budget anywhere = this one
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"reasoning_budget_tokens={value!r}: expected a whole number of tokens (0: no budget)")
        return value if value > 0 else None

    def _vision_down(self) -> bool:
        """The encoder should run and does not (the engine's load starts it again).  An elastic encoder is down
        between pictures by design: it starts in the image turn (_vision_up), not with the engine."""
        return (self.vision is not None and hasattr(self.vision, "alive") and not self.vision.alive()
                and not getattr(self.vision, "elastic", False))

    def _engine_reserve(self) -> int | None:
        """The VRAM the engine keeps free for other programs outside an elastic encoder's turn: the last POST /v1/vram
        reserve, or 0 when none was asked (the cache grows back to its full size; None would leave the reserve the engine
        started with, a segment or more below the size it had)."""
        return self.vram_reserve if self.vram_reserve_asked else 0

    def _vision_up(self):
        """The image turn's start for an elastic encoder that is not running (the caller holds self.fifo): the expert
        cache first keeps the encoder's elastic_mib of VRAM free on top of the reserve, then the encoder starts (load and
        warm-up, a few seconds).  If the cache cannot shrink (an engine without --vram-elastic, the resident low-RAM mode)
        the encoder starts anyway, as a non-elastic one would."""
        v = self.vision
        if not getattr(v, "elastic", False) or v.alive():
            return
        if hasattr(self.engine, "vram") and self.loaded():
            try:
                out = self.engine.vram((self._engine_reserve() or 0) + v.elastic_mib)
                self.vision_held = True
                print(f"[strata] images: the expert cache keeps {v.elastic_mib} MiB free for the image encoder "
                      f"({out.get('expert_cache_mib')} of {out.get('expert_cache_full_mib')} MiB left)", flush=True)
            except ValueError as e:
                print(f"[strata] images: the expert cache did not make room for the image encoder ({e}); starting "
                      "it anyway", flush=True)
        if self.vision_held:
            time.sleep(0.2)          # the driver's free-VRAM counter shows the cache's release a moment later
        free0, t0 = self.free_vram_mib(), time.time()
        v.restart()
        free1 = self.free_vram_mib()
        took = f", {free0 - free1} MiB of VRAM" if free0 is not None and free1 is not None else ""
        print(f"[strata] images: the image encoder started in {time.time() - t0:.1f} s{took}", flush=True)

    def _vision_idle_down(self, force: bool = False) -> str:
        """An elastic encoder idle for elastic_idle_s (or force) stops, and the expert cache takes its VRAM back:
        "down", "busy" (a request holds the turn, or a picture came more recently) or "nothing" (not elastic or not
        holding anything)."""
        v = self.vision
        if not getattr(v, "elastic", False) or not (self.vision_held or v.alive()):
            return "nothing"
        if not force and time.time() - self.last_image_at < v.elastic_idle_s:
            return "busy"
        if not self.fifo.acquire(blocking=False):
            return "busy"
        try:
            v.unload()
            if self.vision_held and hasattr(self.engine, "vram") and self.loaded():
                out = self.engine.vram(self._engine_reserve())
                print(f"[strata] images: the image encoder stopped after {v.elastic_idle_s:g} s without a picture; "
                      f"the expert cache took its VRAM back ({out.get('expert_cache_mib')} of "
                      f"{out.get('expert_cache_full_mib')} MiB)", flush=True)
            self.vision_held = False
            return "down"
        finally:
            self.fifo.release()

    def start_vision_idle(self):
        v = self.vision
        if not getattr(v, "elastic", False):
            return
        print(f"[strata] images: the image encoder starts with the first picture and stops after {v.elastic_idle_s:g} s "
              f"without one; while it runs the expert cache keeps {v.elastic_mib} MiB free for it", flush=True)

        def loop():
            while True:
                time.sleep(max(1.0, min(30.0, v.elastic_idle_s / 4)))
                try:
                    self._vision_idle_down()
                except Exception as e:                 # tried again at the next turn; the thread keeps running
                    print(f"[strata] images: stopping the idle image encoder: {e}", flush=True)
        threading.Thread(target=loop, daemon=True).start()

    def free_vram_mib(self) -> int | None:
        """Free VRAM on the engine's (first) GPU, from the backend's reader - NVML, the amdgpu sysfs files (#301) or
        IOKit on macOS (Metal backend); None when it can't be read (then nothing is refused)."""
        try:
            from strata_common.telemetry import gpu_reader
            g = gpu_reader(int(getattr(self, "gpu_index", 0) or 0), amd=getattr(self, "backend", None) == "hip")
            if not g.ok():
                return None
            r = g.read()
            if r.get("mem_total") is None or r.get("mem_used") is None:
                return None
            return int((r["mem_total"] - r["mem_used"]) >> 20)
        except Exception:
            return None

    def ensure_loaded(self):
        """Start the engine if it is not running (unloaded, or it died - issue #27), after the before_load hook and
        the free-VRAM check.  The caller holds self.fifo."""
        if self.loaded() and not self._vision_down():
            return
        hook_failed = None
        if self.before_load:
            cmd = self.before_load
            print(f"[strata] before loading: {cmd if isinstance(cmd, str) else ' '.join(map(str, cmd))}", flush=True)
            why = None
            try:
                r = subprocess.run(cmd, shell=isinstance(cmd, str), timeout=self.before_load_timeout_s,
                                   stdin=subprocess.DEVNULL)
                if r.returncode != 0:
                    why = f"it exited with code {r.returncode}"
            except subprocess.TimeoutExpired:
                why = f"it did not finish in {self.before_load_timeout_s:g} s (before_load_timeout_s)"
            except (OSError, subprocess.SubprocessError) as e:
                why = f"it could not run ({e})"
            if why is not None:
                # #1997: a failing hook (e.g. the other server's /unload that is still busy) may mean the GPU was not
                # given back.  A peer that is simply not running also makes `curl` fail (exit 7), so the exit code alone
                # does not refuse: the card's free VRAM decides (min_free_vram_mib, checked just below); "strict" always
                # refuses, "ignore" always loads.
                if self.before_load_strict and not self.before_load_ignore_failure:
                    print(f"[strata] the before_load command failed ({why}); the model stays unloaded and this "
                          "request is answered 503 - try again (before_load_strict)", flush=True)
                    raise engine_client.GpuBusy(f"the before_load command failed ({why}): the GPU was not given back, "
                                                "so the model was not loaded; try again in a moment", retry_after=15)
                hook_failed = why
                if self.before_load_ignore_failure or not self.min_free_vram_mib:
                    print(f"[strata] the before_load command failed ({why}); loading anyway"
                          + ("" if self.before_load_ignore_failure else
                             " (set min_free_vram_mib to have the free VRAM checked, or before_load_strict to refuse)"),
                          flush=True)
                else:
                    print(f"[strata] the before_load command failed ({why}); checking the free VRAM", flush=True)
        if self.min_free_vram_mib and not (hook_failed and self.before_load_ignore_failure):
            free = self.free_vram_mib()
            deadline = time.time() + self.vram_free_wait_s   # memory another process just gave back can take a moment
            while free is not None and free < self.min_free_vram_mib and time.time() < deadline:
                time.sleep(0.5)
                free = self.free_vram_mib()
            if free is not None and free < self.min_free_vram_mib:
                if hook_failed and not self.before_load_ignore_failure:
                    print(f"[strata] the before_load command failed ({hook_failed}) and only {free} MiB of VRAM is "
                          f"free (the model needs {self.min_free_vram_mib}): not loading; 503", flush=True)
                raise engine_client.GpuBusy(f"the GPU is in use by another program: {free} MiB of VRAM free, the model "
                                            f"needs {self.min_free_vram_mib} (min_free_vram_mib) - it stays unloaded "
                                            "until that is free"
                                            + (f"; the before_load command failed ({hook_failed})" if hook_failed else ""),
                                            retry_after=15)
            if hook_failed and not self.before_load_ignore_failure:
                print(f"[strata] the before_load command failed ({hook_failed}), but {free} MiB of VRAM is free: "
                      "loading", flush=True)
        vision_started = False
        if self._vision_down():                         # first, as at a start: a GPU encoder takes its VRAM before
            print("[strata] starting the vision encoder ...", flush=True)   # the engine sizes its cache
            try:
                self.vision.restart()
            except Exception as e:
                raise engine_client.EngineDied(f"the vision encoder failed to start: {e}") from e
            vision_started = True
        if self.loaded():
            return
        if getattr(self.engine, "unloaded", False):
            print("[strata] loading the model again (it was unloaded) ...", flush=True)
        else:
            code = self.engine.exit_code() if hasattr(self.engine, "exit_code") else None
            print(f"[strata] the engine had stopped (exit code {code}); starting it again "
                  "(a minute or two) ...", flush=True)
        try:
            self.engine.restart()
        except Exception as e:
            if vision_started:
                self.vision.unload()
            # the program could not be started at all (missing, not executable, or blocked by Windows - #735): the
            # request path answers EngineDied with a 503, but takes a bare OSError for a client that went away (#2496)
            if isinstance(e, OSError):
                raise engine_client.EngineDied(f"the engine could not be started: {e}") from e
            raise
        print("[strata] the engine is running again", flush=True)
        if self.restore_after_load:                      # an image job unloaded it: its conversation comes back
            images_api.restore_session(self)
        if self.vram_reserve is not None and hasattr(self.engine, "vram"):   # #533: the reserve asked for last
            try:
                self.engine.vram(self.vram_reserve)
            except (ValueError, engine_client.EngineDied) as e:
                print(f"[strata] the VRAM reserve ({self.vram_reserve} MiB) was not applied: {e}", flush=True)

    vram_wait_s = 300.0                                  # #533: how long POST /v1/vram waits for a running request

    def vram(self, reserve_mib: int | None) -> dict:
        """#533, POST /v1/vram: keep `reserve_mib` of VRAM free for other programs (None: the reserve the engine
        started with), applied between requests - a request that is running finishes first (up to wait_s).  Only an
        engine started with --vram-elastic (the config's "vram_elastic": true) can do it; it never resizes on its own.
        An unloaded engine applies it when it loads.  -> {"status": ..., and the engine's figures}."""
        if not hasattr(self.engine, "vram"):
            raise ValueError("this engine cannot resize its VRAM use")
        if not self.fifo.acquire(timeout=self.vram_wait_s):
            raise ModelBusy("a request is still running; try again when it has finished")
        # parallel requests do not hold the fifo: the engine's control lines (a prompt being admitted) are taken too,
        # and the engine refuses a resize while a slot decodes
        ctl = getattr(self.engine, "ctl", None) if getattr(self.engine, "batch", 0) else None
        if ctl is not None and not ctl.acquire(timeout=self.vram_wait_s):
            self.fifo.release()
            raise ModelBusy("a request is still running; try again when it has finished")
        try:
            if not self.loaded():
                self.vram_reserve, self.vram_reserve_asked = reserve_mib, True
                return {"status": "not loaded", "reserve_mib": reserve_mib,
                        "note": "applied when the model loads"}
            # an elastic encoder that runs keeps its VRAM on top of the asked reserve
            out = self.engine.vram((reserve_mib or 0) + self.vision.elastic_mib if self.vision_held else reserve_mib)
            self.vram_reserve, self.vram_reserve_asked = reserve_mib, True
            print(f"[strata] VRAM: {out.get('vram_free_mib')} MiB free, expert cache {out.get('expert_cache_mib')} of "
                  f"{out.get('expert_cache_full_mib')} MiB ({out.get('expert_slots')} experts)", flush=True)
            return {"status": "ok", **out}
        finally:
            if ctl is not None:
                ctl.release()
            self.fifo.release()

    def _say_died(self, e: Exception) -> None:
        """The server window's line for an engine that died (or was ended, #481) in the middle of a request."""
        note = self.engine.death_note() if hasattr(self.engine, "death_note") else ""
        log = getattr(self.engine, "log_path", None)
        print(f"[strata] {e}. {note} The next request starts the engine again."
              f"{' Its log: ' + log if log else ''}", flush=True)

    def load(self):
        """POST /load and every generation request: start the engine now if it is unloaded (raises GpuBusy)."""
        # a request is on its way: the idle thread must not unload between this and the request's own start
        self.last_request_at = time.time()
        if self.loaded() and not self._vision_down():
            return
        trace = getattr(self.request_trace, "record", None)
        waiting = time.perf_counter()
        with self.status_lock:                          # #1603: waiting for the fifo to start the engine is queued, not nothing
            self.status["queued"] += 1
        counted = True
        try:
            with self.fifo:
                with self.status_lock:
                    self.status["queued"] -= 1
                counted = False
                loading = time.perf_counter()
                if trace is not None:
                    with self.status_lock:
                        trace["queue_s"] += round(loading - waiting, 3)
                        trace["state"] = "loading"
                try:
                    self.ensure_loaded()
                finally:
                    if trace is not None:
                        with self.status_lock:
                            trace["load_s"] += round(time.perf_counter() - loading, 3)
                            trace["state"] = "queued"
        finally:
            if counted:
                with self.status_lock:
                    self.status["queued"] -= 1

    UNLOAD_WAIT_S = 300.0       # #1997: POST /unload?wait=1 waits this long for the running turn (wait=<s> to choose)

    def unload_waiting(self, wait_s: float) -> str:
        """#1997: unload(), but a running or queued request is waited for (up to wait_s) instead of answering "busy":
        a peer server that needs this GPU can then free the card on the server's terms.  "busy" after wait_s."""
        deadline = time.monotonic() + max(0.0, wait_s)
        while True:
            r = self.unload()
            if r != "busy" or time.monotonic() >= deadline:
                return r
            time.sleep(0.25)

    def unload(self, idle_for: float | None = None) -> str:
        """Stop the engine between requests: "unloaded", "not loaded", "busy" (a request is running or waiting, or
        with idle_for: one ran more recently than that) or "unsupported"."""
        if not hasattr(self.engine, "unload"):
            return "unsupported"
        if not self.fifo.acquire(blocking=False):
            return "busy"
        try:
            if not self.engine.alive():
                return "not loaded"
            with self.status_lock:
                if self.status.get("busy") or self.status.get("queued"):
                    return "busy"
            if idle_for is not None and time.time() - (self.last_request_at or self.started_at) < idle_for:
                return "busy"
            self.engine.unload()
            if self.vision is not None and hasattr(self.vision, "unload"):
                self.vision.unload()
            self.vision_held = False
            print(f"[strata] model unloaded{f' after {idle_for:.0f} s idle' if idle_for else ''}; "
                  "the next request loads it again", flush=True)
            return "unloaded"
        finally:
            self.fifo.release()

    def start_idle_unload(self):
        if not self.idle_unload_s or not hasattr(self.engine, "unload"):
            return
        print(f"[strata] the model unloads after {self.idle_unload_s} s without requests", flush=True)

        def loop():
            while True:
                time.sleep(max(1.0, min(30.0, self.idle_unload_s / 4)))
                try:
                    self.unload(idle_for=self.idle_unload_s)
                except engine_client.EngineStuck as e:   # tried again at the next turn; the thread keeps running
                    print(f"[strata] idle unload: {e}", flush=True)
        threading.Thread(target=loop, daemon=True).start()

    def set_shared(self, defaults) -> dict:
        """The Chat settings every client gets for what it leaves out; {} / None = clients use their own again."""
        self.shared = clean_shared_defaults(defaults)
        if self.shared_path:
            try:
                if self.shared:
                    Path(self.shared_path).write_text(json.dumps(self.shared, indent=1), encoding="utf-8")
                else:
                    Path(self.shared_path).unlink(missing_ok=True)
            except OSError as e:
                print(f"[strata] could not save the shared settings: {e}", flush=True)
        return self.shared

    def effort_default(self) -> str | None:
        """The thinking level of a request that names none: the Chat settings shared with apps first, then the config's
        "reasoning_effort" (#1641).  Every API reads it here (the Responses API did not: 0.1.44.1)."""
        return self.shared.get("reasoning_effort") or self.config_effort

    def with_shared(self, req: dict, api: str) -> dict:
        """The request with the shared thinking level and max tokens filled in where it has none of its own."""
        s = self.shared
        effort = self.effort_default()
        if not s and not effort:
            return req
        req = dict(req)
        if "max_tokens" in s and not req.get("max_tokens") and not req.get("max_completion_tokens"):
            req["max_tokens"] = s["max_tokens"]
        if effort:
            if api == "openai":
                ctk = req.get("chat_template_kwargs") if isinstance(req.get("chat_template_kwargs"), dict) else {}
                if req.get("reasoning_effort") in (None, "") and not req.get("reasoning") and \
                        "enable_thinking" not in ctk and "reasoning_effort" not in ctk:
                    req["reasoning_effort"] = effort
            elif not req.get("thinking") and not req.get("output_config"):
                if effort == "none":
                    req["thinking"] = {"type": "disabled"}
                else:
                    req["output_config"] = {"effort": effort}
        return req

    def start_telemetry(self):
        """The hardware sampler behind GET /metrics (strata_common/telemetry.py), recording this server's tok/s too."""
        if getattr(self, "telemetry", None) is None:
            from strata_common.telemetry import Telemetry
            self.telemetry = Telemetry(extra=lambda: {"tok_s": self._tok_s(), "tok_s_mean": self._tok_s_mean(),
                                                    "prefill_tok_s_mean": self._prefill_tok_s_mean()},
                                       gpu_index=int(getattr(self, "gpu_index", 0) or 0),
                                       gpu_indices=getattr(self, "gpu_indices", None),
                                       amd=getattr(self, "backend", None) == "hip")
            if getattr(self, "engine", None) is not None and hasattr(self.engine, "gpu_busy"):
                # #1317: the quick frozen-engine check does not end an engine whose GPU is busy (a long prompt chunk on a
                # slow card keeps the GPU at work while the host thread sleeps)
                self.engine.gpu_busy = lambda: (self.telemetry.snapshot()["now"].get("gpu_util") or 0) >= 10

    def _tok_s(self):
        """tok/s over the last RATE_WINDOW_S seconds.  Returns 0.0 while nothing is generating.  With parallel requests:
        the sum of theirs."""
        with self.status_lock:
            many = [(dict(st), list(rt)) for st, rt in self.live_reqs.values()]
        if many:
            return sum(self._rate_of(st, rt) for st, rt in many)
        with self.status_lock:
            s = dict(self.status)
            rate = list(self.rate)
        return self._rate_of(s, rate)

    @staticmethod
    def _rate_of(s, rate):
        if not s.get("busy") or not s.get("first_token"):
            return 0.0
        now = time.time()
        newest = rate[-1] if rate else None
        oldest = next(((t, g) for t, g in rate if now - t <= RATE_WINDOW_S), None)
        if newest and oldest and newest[0] - oldest[0] >= RATE_MIN_SPAN_S:
            return max(0.0, (newest[1] - oldest[1]) / (newest[0] - oldest[0]))
        return s["generated"] / max(RATE_MIN_SPAN_S, now - s["first_token"])

    def _tok_s_mean(self):
        """The whole-request mean since the first token (the old formula), kept so the two can be compared."""
        with self.status_lock:
            many = [dict(st) for st, _ in self.live_reqs.values()] or [dict(self.status)]
        return sum(s["generated"] / max(1e-6, time.time() - s["first_token"]) for s in many
                   if s.get("busy") and s.get("first_token"))

    def _prefill_tok_s_mean(self):
        """Engine-reported mean over newly read tokens, excluding the cached prefix."""
        with self.status_lock:
            reading = self.status.get("busy") and self.status.get("first_token") is None
        return getattr(self.engine, "prefill_tok_s_mean", None) if reading else 0.0

    def begin_request(self, path, req):
        """#332: a monitor record for this request, or None when the monitor is off (nothing is kept then)."""
        if not self.api_monitor:
            return None
        raw = json.dumps(req, ensure_ascii=False, indent=2)
        record = {"id": uuid.uuid4().hex[:12], "path": path, "model": req.get("model") or self.model,
                  "started_at": time.time(), "state": "queued", "stream": bool(req.get("stream")),
                  "response_format": (req.get("response_format") or {}).get("type")
                  if isinstance(req.get("response_format"), dict) else None,
                  "input": raw[:262144], "input_truncated": len(raw) > 262144,
                  "output": "", "reasoning": "", "output_truncated": False, "reasoning_truncated": False,
                  "queue_s": 0.0, "load_s": 0.0, "first_token_s": None,
                  "_clock": time.perf_counter()}
        with self.status_lock:
            self.api_requests.append(record)
        self.request_trace.record = record
        return record

    CONTEXT_WAIT_S = 2.0      # how long GET /api/context waits for the engine's turn before it answers with what it has
    SETTINGS_WAIT_S = 30.0    # how long POST /v1/memory waits for the engine's turn before it answers 503
    PREVIEW_TOKENS = 24       # the start of each conversation's first user message in the list

    @contextlib.contextmanager
    def _engine_turn(self, wait: float | None = None):
        """Serializes command replies and model lifecycle operations. RUN streams use independent queues, so a
        generation never owns this turn. Older released engines may still hold it until generation ends."""
        wait = self.CONTEXT_WAIT_S if wait is None else wait
        if not self.fifo.acquire(timeout=wait):
            yield False
            return
        ctl = getattr(self.engine, "ctl", None) if getattr(self.engine, "batch", 0) else None
        try:
            if ctl is not None and not ctl.acquire(timeout=wait):
                yield False
                return
            try:
                yield True
            finally:
                if ctl is not None:
                    ctl.release()
        finally:
            self.fifo.release()

    def _user_marker(self) -> list[int]:
        """The tokens that open a user turn in this model's chat template: the preview starts after them."""
        try:
            return list(self.tok.encode("<｜User｜>" if self.family == "deepseek41" else "<|im_start|>user\n",
                                        parse_special=True))
        except Exception:                               # noqa: BLE001 - a tokenizer without specials: no preview
            return []

    def _preview_text(self, ids) -> str:
        text = "".join(token_pieces(self.tok, ids)).split(IM_END)[0]
        return " ".join(text.split())

    def context_view(self) -> dict:
        """GET /api/context (API monitor): each conversation the engine holds - its own session, the batch slots', then
        the parked ones, the next to be evicted first (StrataEngine.list_conversations) - with the start of its first
        user message, but not its tokens (context_window).  While a request has the engine's turn (one without --batch,
        or one alone on the solo path) it answers with the last list, marked busy."""
        eng, base = self.engine, {"loaded": self.loaded(), "max_context": self.reported_ctx()}
        if not self.loaded():
            return {"available": False, "reason": "unloaded", **base}
        if not getattr(eng, "can_list", False):
            return {"available": False, "reason": "engine", **base}
        busy, error = False, None
        try:
            with self._engine_turn() as turn:
                busy = not turn
                if turn:
                    held = eng.list_conversations(self._user_marker(), self.PREVIEW_TOKENS)
                    self.context_list = {"at": time.time(), "entries": self._context_entries(held)}
        except engine_client.EngineDied as e:
            error = str(e)
        got = self.context_list
        return {"available": True, **base, "busy": busy, "error": error, "at": got and got["at"],
                "entries": got["entries"] if got else []}

    def memory_view(self, answered: dict | None = None) -> dict:
        """GET /v1/memory (the web page's Memory panel): what the engine holds in each memory and its settings
        (StrataEngine.memory), and the machine's own view - its RAM and what the hardware sampler reads of the GPUs;
        "can_change": the engine changes settings live (POST /v1/memory), "defaults": the run config's values for
        them (runconfig.read_live_defaults), "config": there is a run config to save defaults to.  `answered`: the
        engine's report, read already (POST /v1/memory's change_settings), so the engine is not asked again.
        The engine's `ram` gets the machine's size and free RAM. Each memory with a size gets "other_apps":
        what is in use beyond the engine's accounted allocations. Where Windows identifies the engine's own
        dedicated usage, unnamed runtime bytes stay under its Other category. A GPU whose memory is system RAM
        (`unified`) is shown as the BIOS divides it, where read_shared_gpu_memory can read that: its total
        and free are the BIOS share's, and what it borrows beyond the share counts against the RAM (the RAM's
        `gpu_borrowed`).  The engine's RAM the GPU maps (the RAM's `mapped`: strata-q38's page-locked pool) is in the
        GPU's borrowed use too, as Windows and Linux count it, and is not borrowed again: the RAM's bar has it as the
        engine's.  While a request has the engine's turn: the last answer, marked busy."""
        from strata_base.memory import read_ram
        from strata_common.telemetry import read_shared_gpu_memory
        eng, ram = self.engine, read_ram()
        tel = self.telemetry.snapshot() if getattr(self, "telemetry", None) else {"now": {}, "static": {}}
        now = tel["now"]
        gpus = now.get("gpus") or ([{"index": int(getattr(self, "gpu_index", 0) or 0),
                                     "name": tel["static"].get("gpu_name"), "mem_used": now.get("gpu_mem_used"),
                                     "mem_total": now.get("gpu_mem_total")}] if now.get("gpu_mem_total") else [])
        base = {"loaded": self.loaded(), "machine": {"ram_total": ram.total, "ram_free": ram.available, "gpus": gpus},
                "can_change": self.loaded() and getattr(eng, "can_change_settings", False),
                "defaults": runconfig.read_live_defaults(self.config_path), "config": bool(self.config_path)}
        if not self.loaded():
            return {"available": False, "reason": "unloaded", **base}
        if not getattr(eng, "can_report_memory", False):
            return {"available": False, "reason": "engine", **base}
        busy, error = False, None
        try:
            if answered is None:
                with self._engine_turn() as turn:
                    busy = not turn
                    answered = eng.memory() if turn else None
            if answered is not None:
                self.memory_last = {"at": time.time(), **answered}
        except engine_client.EngineDied as e:
            error = str(e)
        memories = [{**m, "uses": dict(m["uses"]), "blocks": list(m.get("blocks", []))}
                    for m in self.memory_last["memories"]]
        borrowed = 0                             # what GPUs whose memory is system RAM hold beyond their BIOS share
        mapped = next((m.get("mapped") or 0 for m in memories if m["id"] == "ram"), 0)   # left to take off "borrowed"
        for m in memories:
            # (gpu<N>.<card>: another card of strata-q38's layer split on GPU N, whose share gpu<N> reports)
            split = m["unified"] and "." not in m["id"] and read_shared_gpu_memory(m["name"], int(m["id"][3:]),
                                                            getattr(getattr(eng, "proc", None), "pid", None))
            if split:
                used, beyond = sum(m["uses"].values()), split["borrowed"]
                if beyond is not None:
                    own = min(mapped, beyond)
                    beyond, mapped = beyond - own, mapped - own
                m.update(runtime_total=m["total"], runtime_free=m["free"], total=split["share"],
                         free=max(0, split["share"] - split["share_used"]), borrowed=beyond,
                         borrow_limit=split["borrow_limit"], borrowed_by=split["borrowed_by"],
                         in_share=min(used, split["share_used"]) if split["in_share"] is None else split["in_share"])
                borrowed += beyond or 0
        for m in memories:
            if m["id"] == "ram":
                if m["total"] is None:
                    m["total"], m["free"] = ram.total, ram.available
                m["gpu_borrowed"] = borrowed
            if m["total"] is not None and m["free"] is not None:
                used = sum(m["uses"].values())
                runtime = max(0, m.get("in_share", 0) - used)
                if runtime:
                    m["uses"]["other"] = m["uses"].get("other", 0) + runtime
                    m["blocks"].append({"category": "other", "pool": "Graphics runtime",
                                        "label": "Runtime allocations and allocation padding", "bytes": runtime})
                named = min(used + runtime, m.get("in_share", math.inf)) + m.get("gpu_borrowed", 0)
                m["other_apps"] = max(0, m["total"] - m["free"] - named)
        return {"available": True, **base, "busy": busy, "error": error, **self.memory_last, "memories": memories}

    def change_memory_settings(self, raw: bytes) -> tuple[int, dict]:
        """POST /v1/memory (the Memory panel's controls): `{"set": {setting: value}, "save": "session" | "default"}`
        changes the settings live (StrataEngine.change_settings) and, with "default", saves them to the run config's
        "args" too (runconfig.apply_live).  -> (HTTP status, body): memory_view's body with "changed" (the settings
        sent), "saved" and, when the engine got less than asked, "shortfall" (its words); 409 when the engine refused
        (nothing saved) or cannot change settings live."""
        def fail(status, kind, message):
            return status, {"error": {"type": kind, "message": message}}
        try:
            req = json.loads(raw or b"{}")
            save = req.get("save", "session") if isinstance(req, dict) else None
            if save not in ("session", "default"):
                raise ValueError('send {"set": {...}, "save": "session" or "default"}')
            if save == "default" and not self.config_path:
                raise ValueError("this server was started without a run config: there is no default to save to")
            values = runconfig.check_live(req.get("set"), save == "default",
                                          runconfig.get_card_settings(self.memory_last["settings"]))
        except ValueError as e:                        # (a JSONDecodeError is one)
            return fail(400, "invalid_request_error", str(e))
        if not self.loaded():
            return fail(503, "unavailable", "no model is loaded")
        if not getattr(self.engine, "can_change_settings", False):
            return fail(409, "unsupported", "this engine cannot change its memory settings while it runs")
        if "context" in values and "context_capacity" not in getattr(self.engine, "info", {}):
            return fail(409, "unsupported", "this engine cannot change its per-conversation limit while it runs")
        if "vram_conversations" in values and "vram_conversations" not in getattr(self.engine, "info", {}):
            return fail(409, "unsupported", "this engine cannot limit the number of conversations held in VRAM")
        if ("conversation_idle_seconds" in values and
                "conversation_idle_seconds" not in getattr(self.engine, "info", {})):
            return fail(409, "unsupported", "this engine cannot change its idle conversation eviction age")
        try:
            with self._engine_turn(self.SETTINGS_WAIT_S) as turn:
                answered = self.engine.change_settings(values) if turn else None
        except engine_client.SettingsRefused as e:
            return fail(409, "refused", str(e))
        except engine_client.EngineDied as e:
            return fail(503, "unavailable", str(e))
        if answered is None:
            return fail(503, "unavailable", "a request kept the engine busy; try again when it has finished")
        if save == "default":
            try:
                with self.config_lock:
                    new, changed = runconfig.apply_live(runconfig.load(self.config_path), values,
                                                        answered.get("settings"))
                    if changed:
                        bak = runconfig.save(self.config_path, new)
            except (OSError, ValueError) as e:
                return fail(500, "server_error",
                            f"changed for this session, but the run config cannot be written: {e}")
            if changed:
                print(f"[strata] the Memory panel saved {', '.join(changed)} as defaults in "
                      f"{Path(self.config_path).name} (the earlier file: {bak.name})", flush=True)
        shortfall = answered.pop("shortfall", None)   # (this answer's: not kept in memory_last)
        return 200, {**self.memory_view(answered), "changed": list(values), "saved": save == "default",
                     **({"shortfall": shortfall} if shortfall else {})}

    def _context_entries(self, held: list[dict]) -> list[dict]:
        """Names occupied contexts uniformly while retaining the engine's protocol addresses for token reads."""
        eng = self.engine
        used = getattr(eng, "slot_used", None) or []
        parked = [h for h in held if h["id"].startswith("parked")]
        entries = []
        for h in held:
            if not h["count"] and h["state"] == "empty":
                continue
            e = {k: h[k] for k in ("id", "state", "count", "capacity", "bytes", "used")} | \
                {"where": h["place"], "preview": self._preview_text(h["preview"])}
            if h["id"].startswith("parked"):
                e |= {"kind": "parked", "slot": None,
                      "name": f"Parked conversation {parked.index(h) + 1} of {len(parked)}"}
            else:
                slot = int(h["id"][4:]) if h["id"].startswith("slot") else None
                e |= {"kind": "session", "slot": slot,
                      "name": f"Conversation {len(entries) + 1}",
                      "last_used": used[slot] if slot is not None and slot < len(used) else None}
            entries.append(e)
        return entries

    def context_window(self, cid: str) -> tuple[int, dict]:
        """GET /api/context?id=<id>: the tokens of the conversation the engine holds under that id
        (StrataEngine.context_windows), with each token's text.  While a request has the engine's turn: the one read
        last, when it is this conversation's (marked busy).  -> (HTTP status, body)."""
        if not getattr(self.engine, "can_list", False) or not self.loaded():
            return 404, {"error": {"message": "the engine does not list its conversations"}}
        try:
            with self._engine_turn() as turn:
                got = self.engine.context_windows([cid]).get(cid) if turn else None
        except engine_client.EngineDied as e:
            return 503, {"error": {"message": str(e)}}
        if turn:
            if got is None:
                return 404, {"error": {"message": "the engine no longer holds this conversation; press Refresh"}}
            self.context_shown = {"id": cid, "at": time.time(), "tokens": got}
        shown = self.context_shown
        if shown is None or shown["id"] != cid:
            return 503, {"error": {"message": "a request has the engine's turn; try again when it has finished"}}
        return 200, {"id": cid, "busy": not turn, "at": shown["at"], "tokens": shown["tokens"].tolist(),
                     "pieces": token_pieces(self.tok, shown["tokens"])}

    def request_records(self, request_id=None):
        with self.status_lock:
            records = list(self.api_requests)
            if request_id:
                record = next((r for r in records if r["id"] == request_id), None)
                if record is None:
                    return None
                return {k: v for k, v in record.items() if not k.startswith("_")}
            return [{k: v for k, v in r.items()
                     if k not in ("input", "output", "reasoning", "response") and not k.startswith("_")}
                    | {"wallclock_s": r.get("wallclock_s", round(time.perf_counter() - r["_clock"], 3))}
                    for r in reversed(records)]

    def metrics(self, all_requests=False) -> dict:
        """GET /metrics: what the Monitor tab shows - the engine's facts, what it is doing, the last requests, and
        the hardware (with a minute of history per series)."""
        with self.status_lock:
            s = dict(self.status)
            if self.live_reqs:                          # parallel requests: the newest one's status
                s = {**s, **dict(list(self.live_reqs.values())[-1][0]), "queued": s.get("queued", 0)}
            hist = list(self.history)
            totals = dict(self.totals)
        now = time.time()
        progress = getattr(self.engine, "progress", None)
        if s.get("busy") and s.get("first_token") is None:
            state = "reading"
        elif s.get("busy"):
            state = "generating"
        elif not self.loaded():
            state = "unloaded"
        else:
            state = "idle"
        live = {"state": state, "queued": s.get("queued", 0), "phase": s.get("phase") if s.get("busy") else None,
                "prompt_tokens": s.get("prompt_tokens") if s.get("busy") else None,
                "prompt_read": None, "prompt_total": None, "generated": s.get("generated") if s.get("busy") else None,
                "max_tokens": s.get("max_tokens") if s.get("busy") else None,
                "elapsed_s": round(now - s["started"], 1) if s.get("busy") and s.get("started") else None,
                "tok_s": round(self._tok_s(), 1) if state == "generating" else None,
                "tok_s_mean": round(self._tok_s_mean(), 1) if state == "generating" else None,
                "prefill_tok_s_mean": getattr(self.engine, "prefill_tok_s_mean", None) if s.get("busy") else None,
                "tok_s_window_s": RATE_WINDOW_S if state == "generating" else None}
        if state == "reading" and progress:
            live["prompt_read"], live["prompt_total"] = progress
        par = int(getattr(self.engine, "batch", 0) or 0)
        if par:                                         # #465: the requests running together, slot by slot
            with self.status_lock:
                running = len(self.live_reqs)
            slots = self.engine.slots_view() if hasattr(self.engine, "slots_view") else []
            # Requests waiting in the server have no engine-side metrics entry yet.
            in_slots = sum(1 for x in slots if x["state"] != "idle")
            live.update(parallel=par, running=running, slots=slots, outside_slots=max(0, running - in_slots),
                        waiting=int(getattr(self.engine, "waiting", 0) or 0),
                        admitting=int(getattr(self.engine, "admitting", 0) or 0))
            if running and state == "idle":
                live["state"] = "generating"
        if hasattr(self.engine, "replica_view"):        # opt-in replicas: each one's state and requests routed to it
            live["replicas"] = self.engine.replica_view()
        engine = {"model": self.model, "max_context": self.reported_ctx(), "images": self.vision is not None,
                  **dict(getattr(self.engine, "info", {}) or {}), "model_details": self.model_details}
        tel = self.telemetry.snapshot() if getattr(self, "telemetry", None) else {"now": {}, "history": {}, "static": {}}
        parked = self.conv_log.poll(getattr(self.engine, "log_path", None), getattr(self.engine, "log_start", None))
        return {"engine": engine, "live": live, "requests": hist[::-1][:None if all_requests else 12],
                "conversation_cache": conversation_cache_view(engine, hist, totals, parked),
                "requests_kept": len(hist), "totals": totals, "hardware": tel["now"],
                "hardware_static":
                tel["static"], "history": tel["history"], "time": now}

    def prometheus(self) -> str:
        """GET /metrics/prometheus: the totals and what is running in Prometheus' text format, under llama-server's
        metric names (llamacpp:*) so its dashboards work unchanged; the drafts and requests as strata:*."""
        with self.status_lock:
            s, t = dict(self.status), dict(self.totals)
            running = len(self.live_reqs) or int(bool(s.get("busy")))
            prompt_n = t["prompt_tokens"] - t["reused"]               # what was read, as llama.cpp counts it
            prompt_s, decode_s = t["prompt_ms"] / 1000, t["decode_ms"] / 1000
            # llama.cpp reads these two gauges from a bucket it resets on every scrape (server_metrics::bucket,
            # "the gauges are averaged over the window between two scrapes"), so an idle server reports 0 rather
            # than its lifetime average. Keep the same window here, or the names lie to llama.cpp dashboards.
            was_prompt_n, was_prompt_s, was_predicted_n, was_decode_s = self.scrape_bucket
            self.scrape_bucket = (prompt_n, prompt_s, t["output_tokens"], decode_s)
        d_prompt_n, d_prompt_s = prompt_n - was_prompt_n, prompt_s - was_prompt_s
        d_predicted_n, d_decode_s = t["output_tokens"] - was_predicted_n, decode_s - was_decode_s
        rows = [
            ("llamacpp:prompt_tokens_total", "counter", "Prompt tokens read", prompt_n),
            ("llamacpp:prompt_tokens_cached_total", "counter", "Prompt tokens the conversation cache already held",
             t["reused"]),
            ("llamacpp:prompt_seconds_total", "counter", "Prompt read time", prompt_s),
            ("llamacpp:tokens_predicted_total", "counter", "Tokens generated", t["output_tokens"]),
            ("llamacpp:tokens_predicted_seconds_total", "counter", "Generation time", decode_s),
            ("llamacpp:prompt_tokens_seconds", "gauge", "Average prompt read throughput in tokens/s, since the last scrape",
             d_prompt_n / d_prompt_s if d_prompt_s > 0 else 0),
            ("llamacpp:predicted_tokens_seconds", "gauge", "Average generation throughput in tokens/s, since the last scrape",
             d_predicted_n / d_decode_s if d_decode_s > 0 else 0),
            ("llamacpp:requests_processing", "gauge", "Requests running", running),
            ("llamacpp:requests_deferred", "gauge", "Requests waiting their turn", s.get("queued") or 0),
            ("strata:requests_total", "counter", "Requests finished", t["requests"]),
            ("strata:drafts_offered_total", "counter", "Speculative draft tokens offered", t["drafts_offered"]),
            ("strata:drafts_accepted_total", "counter", "Speculative draft tokens accepted", t["drafts_accepted"]),
            ("strata:model_loaded", "gauge", "1 while the model is loaded", int(self.loaded())),
        ]
        return "".join(f"# HELP {name} {help_}\n# TYPE {name} {kind}\n{name} {value}\n"
                       for name, kind, help_, value in rows)

    def v1_status(self) -> dict:
        """GET /v1/status: what this server is and does, for a client that would rather ask than guess (a front-end
        that polls its OpenAI-compatible server's status, collabosm's for one): the model and its window, images,
        the APIs, what is running, and the last request's timings in llama.cpp's names.  /metrics has the rest."""
        with self.status_lock:
            s, totals = dict(self.status), dict(self.totals)
            last_t, last_at = (dict(self.last_timings) if self.last_timings else None), self.last_request_at
            running = len(self.live_reqs) if self.live_reqs else int(bool(s.get("busy")))
        tel = self.telemetry.snapshot() if getattr(self, "telemetry", None) else {"now": {}, "static": {}}
        hw, static = tel.get("now") or {}, tel.get("static") or {}

        def scaled(v, unit, digits=0):
            return round(v / unit, digits) if isinstance(v, (int, float)) else None

        busy, ctx = bool(s.get("busy")), self.reported_ctx()
        images = self.vision is not None
        return {
            "service": "strata", "model": self.model, "model_details": self.model_details,
            "loaded": self.loaded(), "auto_load": hasattr(self.engine, "restart"),
            "structured_output": {"formats": ["json_object", "json_schema"], "method": "prompt_and_validate",
                                  "constrained_decoding": False, "stream_buffered": True},
            "engine": (getattr(self.engine, "info", {}) or {}).get("version"),
            "started": int(self.started_at), "uptime_s": int(time.time() - self.started_at),
            "cache_max_tokens": ctx,
            "context": {"native": ctx, "max_positions": ctx},
            # Paged engines report their live scheduling limit; older engines use their fixed slot count.
            "concurrency": {"serving": int((getattr(self.engine, "info", {}) or {}).get("concurrent") or
                                            max(1, int(getattr(self.engine, "batch", 0) or 0))),
                            "requested": getattr(self, "requested_slots", max(1, int(getattr(self.engine, "batch", 0) or 0)))},
            "dialects": ["/v1/chat/completions", "/v1/messages", "/v1/responses"],
            "vision": {"enabled": images, "available": images, "error": None},
            "image_generation": self.images.status() if self.images is not None else {"enabled": False},
            "activity": {"requests": totals["requests"] + running,
                         "in_flight": running + int(s.get("queued") or 0),
                         "last_request_at": int(last_at) if last_at else None},
            "last_timings": last_t,
            "vram": dict((getattr(self.engine, "info", {}) or {}).get("vram") or {},
                         elastic=bool((getattr(self.engine, "info", {}) or {}).get("vram_elastic"))),
            "machine": {
                "at": int(time.time()),
                "gpu": {"name": static.get("gpu_name"), "used_mib": scaled(hw.get("gpu_mem_used"), 2 ** 20),
                        "total_mib": scaled(hw.get("gpu_mem_total"), 2 ** 20), "util_pct": hw.get("gpu_util"),
                        "temp_c": hw.get("gpu_temp"), "power_w": hw.get("gpu_power")} if static.get("gpu_name") else None,
                "ram": {"used_gib": scaled(hw.get("ram_used"), 2 ** 30, 1),
                        "total_gib": scaled(hw.get("ram_total"), 2 ** 30, 1)} if hw.get("ram_total") else None}}

    @cached_property
    def model_details(self) -> dict:
        """Gets display metadata from the model header without reading tensor data."""
        from strata_common.gguf_reader import GGUFFile

        spawn = getattr(self.engine, "spawn", None)
        root = Path(spawn[2] or ".") if spawn else Path.cwd()
        source = getattr(self.engine, "model_path", None)
        path = (root / source).resolve() if source else None
        if path and path.is_dir():
            try:
                metadata = dict(line.split(" ", 1) for line in
                                (path / "ds4.txt").read_text(encoding="utf-8").splitlines() if " " in line)
                with (path / "tensors.txt").open(encoding="utf-8") as manifest:
                    first = next(line[7:].strip() for line in manifest if line.startswith("file 0 "))
                path = (root / metadata["model_dir"] / first).resolve()
            except (OSError, KeyError, StopIteration):
                pass
        name = self.model
        if path and path.suffix.lower() == ".gguf":
            try:
                name = GGUFFile(path).metadata.get("general.name") or name
            except (OSError, ValueError, KeyError, struct.error):
                pass
        return {"name": name, "engine": Path(spawn[0]).name if spawn else
                (getattr(self.engine, "info", {}) or {}).get("version"),
                "config_path": str(Path(self.config_path).resolve()) if self.config_path else None,
                "model_path": str(path) if path else None}

    def render_prompt(self, messages, tools, kwargs) -> str:
        """The template rendered.  #458: with `effort_end`, a request with a non-default effort (low, medium or no
        thinking) is rendered as a default one up to the answer - the same prompt start, so the conversation cache
        keeps it - and its effort follows in a short system turn right before the answer (thinking off: the template's
        empty thinking block).  The engine (--tail-role-token) checkpoints in front of that turn."""
        effort = kwargs.get("reasoning_effort")
        off = kwargs.get("enable_thinking") is False
        if not self.effort_end or (not off and effort in (None, "", "xhigh", "high")):
            return self.template.render(messages, tools=tools, **kwargs)
        base = {k: v for k, v in kwargs.items() if k not in ("reasoning_effort", "enable_thinking")}
        history = self.template.render(messages, tools=tools, add_generation_prompt=False, **base)
        full = self.template.render(messages, tools=tools, add_generation_prompt=True, **kwargs)
        mine = self.template.render(messages, tools=tools, add_generation_prompt=False, **kwargs)
        if not full.startswith(mine):                  # a template this cannot take apart: rendered as asked
            return full
        tail = "" if off else EFFORT_TURN.format(EFFORT_TEXT.get(effort, EFFORT_TEXT["medium"]))
        return history + tail + full[len(mine):]

    def encode_prompt(self, messages, tools, kwargs, store: bool = True) -> list[int]:
        """The request's prompt: the template rendered and tokenized.  #537: a <think> / </think> written inside a
        message's text is encoded as the text it is, not as the model's reasoning markers (the template's own are).
        A control token's text (<|im_start|>, <|im_end|>, <|endoftext|>, ...) inside a message is text as well: only
        the control tokens the template writes are control tokens.  `store=False` (a token count) reuses the kept
        prompts without adding or replacing one."""
        marked, marked_tools, changed = mark_think_literals(messages, tools, self.literals)
        prompt = self.render_prompt(marked, marked_tools, kwargs)
        plain = ()
        if changed:
            prompt, plain = unmark_think_literals(prompt, self.literals)
        if self.prompts is not None:            # #567: only what follows the last special shared with a recent prompt
            return self.prompts.encode(prompt, plain, store)
        return self.tok.encode(prompt, parse_special=True, plain=plain) if plain else \
            self.tok.encode(prompt, parse_special=True)

    @contextlib.contextmanager
    def _turn(self, cancel=None):
        """The engine's single turn (self.fifo) for an image encode.  #1999: a client that leaves while this request
        waits for it gives its place up (ClientGone), as a queued generation does (#1619)."""
        if cancel is None:
            self.fifo.acquire()
        else:
            while not self.fifo.acquire(timeout=0.25):
                if cancel.is_set():
                    raise ClientGone()
        try:
            yield
        finally:
            self.fifo.release()

    def _note_unreadable_tool_images(self, messages, cancel=None):
        """A picture a tool returned (Claude Code's Read of an image file) that this server cannot read - it has no
        image encoder, or the encoder refuses that picture - becomes a short note in its place instead of a 400.
        Clients resend the whole history every turn, so one unreadable tool picture would fail every later request of
        the conversation; a picture the user sends keeps the 400.  A picture that encodes stays in the cache, so the
        prompt's own encode below finds it."""
        fetched = {}                     # a URL's bytes, so prepare() does not download the same picture again
        for m in messages:
            content = m.get("content")
            if m.get("role") != "tool" or not isinstance(content, list):
                continue
            for n, item in enumerate(content):
                if not isinstance(item, dict) or item.get("type") != "image":
                    continue
                why = "this server has no image encoder" if self.vision is None else None
                if why is None:
                    src = item["source"]
                    try:
                        if cancel is not None and cancel.is_set():
                            raise ClientGone()
                        if self.offline and src.startswith(("http://", "https://")):
                            raise ValueError("offline mode cannot download an image URL")
                        if src.startswith(("http://", "https://")):
                            src = fetched[item["source"]] = Vision.download(src)   # outside the FIFO, as in prepare()
                        with self._turn(cancel):  # the encoder takes its turn with the requests (see below)
                            self._vision_up()         # an elastic encoder is not running until a picture needs it
                            self.vision.encode(src)
                    except (ValueError, OSError, RuntimeError) as e:    # RuntimeError: the encoder did not start
                        why = str(e)
                if why is not None:
                    print(f"[strata] a picture a tool returned was left out of the prompt: {why}", flush=True)
                    content[n] = {"type": "text", "text": f"[image omitted: {why}]"}
        return fetched

    def forget_prompt(self, ids):
        if self.prompts is not None:
            self.prompts.forget(ids)


    def prepare(self, messages, tools, kwargs, max_new=None, force=None, req=None, cancel=None):
        """-> (ids, thinking, max_new). An unset or non-positive max_new (some clients send -1) means "unlimited":
        the rest of the context.  `force` (forced_call): without thinking the reply starts with it, so it ends the
        prompt; with thinking, Service.run writes it once the thinking is over."""
        # `cancel` (#1999): set when the client hangs up; the downloads, the wait for the turn and the encodes stop
        # between pictures and raise ClientGone
        fetched = self._note_unreadable_tool_images(messages, cancel)
        ids = encoded_ids = self.encode_prompt(messages, tools, kwargs)
        if force and kwargs.get("enable_thinking", True) is False:
            ids = ids + self.tok.encode(force, parse_special=True)
        self.embeddings.path = None
        self.embeddings.keys = None
        images = images_of(messages)
        if images:
            if self.offline and any(src.startswith(("http://", "https://")) for src in images):
                raise ValueError("offline mode cannot download an image URL; send a data: image instead")
            if self.vision is None:
                raise ValueError("this server was started without the vision encoder (run setup again and choose "
                                 "'vision'), so it cannot read images")
            pad = self.tok.encode(IMAGE_PAD, parse_special=True)[0]
            start = self.tok.encode(VISION_START, parse_special=True)[0]
            # An image URL is downloaded first, outside the FIFO: under it, a slow server held every other request
            # for as long as urlopen waited.
            def fetch(src):
                if cancel is not None and cancel.is_set():
                    raise ClientGone()
                return fetched[src] if src in fetched else Vision.download(src) \
                    if src.startswith(("http://", "https://")) else src
            images = [fetch(src) for src in images]
            # Encode only while the engine is idle: the engine and the image encoder (a separate process) must not
            # run on the GPU at the same time - an encode during a running request left that request stuck at
            # "reading the prompt" with CPU and GPU busy, for good (reproduced).  So encoding takes its turn in the
            # same FIFO as the requests.
            with self._turn(cancel):
                self._vision_up()
                try:
                    encoded = (self.vision.encode_all(images, cancel) if hasattr(self.vision, "encode_all")
                               else [self.vision.encode(src) for src in images])
                finally:
                    self.last_image_at = time.time()
            # one <|image_pad|> per image -> one per image token.  Only the markers the template writes for an image
            # (right after <|vision_start|>) are images: the same text inside a message (an agent reading these docs,
            # #150) is kept as plain text, or it took an image's place and the counts no longer matched.
            literal = self.tok.encode(IMAGE_PAD, parse_special=False)
            out, k = [], 0
            for j, t in enumerate(ids):
                if t == pad and j > 0 and ids[j - 1] == start:
                    # more pairs than images: text parts that cut both markers apart (the literal marks keep whole ones text)
                    if k == len(encoded):
                        raise ValueError("the prompt and its images do not match")
                    out += [pad] * encoded[k][1]
                    k += 1
                elif t == pad:
                    out += literal
                else:
                    out.append(t)
            if k != len(encoded):
                raise ValueError("the prompt and its images do not match")
            ids = out
        ctx = self.engine.max_context
        if ctx <= 0:
            if getattr(self.engine, "starting", False):   # #344: (re)starting, not a prompt that is too long
                raise EngineStarting("the engine is starting (a minute or two); try again shortly")
            # a failed restart leaves max_context 0: check against the last known context, so the request reaches
            # run() - which starts the engine again - instead of failing with "exceeds the context (0)" forever
            ctx = getattr(self.engine, "known_ctx", 0)
            if ctx <= 0:
                raise EngineStarting("the engine is starting (a minute or two); try again shortly")
        room = ctx - CTX_SLACK - len(ids)
        if max_new is None or max_new <= 0 or (self.fit_max_tokens and room < 1):
            if room < 1:
                # A prompt refused for the context is not kept (#567): the client cannot resend it unchanged.  The
                # other refusals (engine starting, images) keep it: the same prompt comes back and its encode is reused.
                self.forget_prompt(encoded_ids)
                raise ValueError("request exceeds the context window: "
                                 f"prompt ({len(ids)} tokens) leaves no room to answer in the context "
                                 f"({ctx}); requests are never truncated")
            max_new = room
        elif max_new > room:
            if not self.fit_max_tokens:
                self.forget_prompt(encoded_ids)
                raise ValueError("request exceeds the context window: "
                                 f"prompt ({len(ids)} tokens) + max tokens ({max_new}) exceeds the context "
                                 f"({ctx}); requests are never truncated. Send a smaller "
                                 f"max_tokens (at most {max(0, room)} here), or add \"fit_max_tokens\": true to the "
                                 "model's strata-<model>.json to shorten it to the room left (#545)")
            print(f"[strata] max_tokens {max_new} shortened to {max(1, room)}: the prompt ({len(ids)} tokens) leaves that much "
                  f"of the context ({ctx}); \"fit_max_tokens\": false answers 400 instead", flush=True)
            max_new = max(1, room)          # a shorter completion beats a 400 (the default)
        if images:
            # The request's images in one file for GENI (~10 MB a picture), written once nothing above refuses the
            # request: one refused after it (the engine starting, no room) left it in the vision directory for good,
            # one more for every retry of a 503.  run() deletes it; drop_embeddings() if run() never starts.
            combined = combined_embeddings_path(self.vision.dir, sum(Path(p).stat().st_size for p, _ in encoded))
            self.embeddings.path = combined             # first, so a half-written one is found as well
            # #2228: each picture's key is its normalized bytes' hash (the encoder's file name), so the engine's
            # conversation cache matches a picture by what it is, not by its embedding floats: a picture encoded again
            # (an encoder restart, the cache's eviction) with other bits still matches its checkpoints
            self.embeddings.keys = [Path(p).stem[:16] for p, _ in encoded]
            write_temporary(combined, [p for p, _ in encoded])
        if self.family == "deepseek41" and req is not None and req.get("strata_prefix") is None and not images:
            lead = self.deepseek_shared_start(messages)
            if lead:
                req["strata_prefix"] = {"messages": lead}
        if req is not None and req.get("strata_prefix") is not None:
            req["strata_prefix"] = self.resolve_prefix(req["strata_prefix"], messages, tools, kwargs, ids, bool(images))
        return ids, kwargs.get("enable_thinking", True) is not False, max_new

    @staticmethod
    def deepseek_shared_start(messages) -> int:
        """DeepSeek-V4.1 only: how many leading messages are the system prompt (its tools are rendered with them).  An
        agent and its sub-agents that work behind the same system prompt and tools share exactly this start, so the
        engine keeps a checkpoint of its state there (pin=N): the next agent behind the same start goes back to it
        instead of reading it again.  0: none (STRATA_DS4_AUTO_PIN=0, no system message, or nothing after it)."""
        if os.environ.get("STRATA_DS4_AUTO_PIN", "1") == "0":
            return 0
        n = 0
        for m in messages:
            if isinstance(m, dict) and m.get("role") in ("system", "developer"):
                n += 1
            else:
                break
        return n if 0 < n < len(messages) else 0

    PREFIX_END = "\u0001strata-prefix-end\u0001"          # marks where a message's shared part ends in the rendered prompt

    def resolve_prefix(self, spec, messages, tools, kwargs, ids, has_images=False) -> dict:
        """R1: `"strata_prefix"` marks the start of the prompt as a shared read-only prefix: a long document that many
        short questions follow.  Three ways to say where it ends: `{"messages": K}` (the first K messages), `{"message":
        I, "chars": N}` (the first N characters of message I's text: a document and its question in one message) or
        `{"tokens": N}`.  The engine reads the prompt in two parts there, keeps the checkpoint at the boundary pinned
        (never evicted) and does not park the branch it leaves, so N questions cost one prefix.  Returns
        {"tokens": N} (what the engine's `pin=N` takes: always a prefix of the prompt's ids, the tokenizer's merge across
        the boundary included), or {} when the prefix cannot be marked (said in the log: the request is never refused for
        it).  A malformed field is a 400."""
        keys = ("messages", "message", "chars", "tokens")
        if (not isinstance(spec, dict) or not spec or any(k not in keys for k in spec) or
                set(spec) not in ({"messages"}, {"tokens"}, {"message", "chars"})):
            raise ValueError('strata_prefix: {"messages": K} (the first K messages), {"message": I, "chars": N} (the first '
                             'N characters of message I) or {"tokens": N}')
        for k, v in spec.items():
            if not isinstance(v, int) or isinstance(v, bool) or v < (0 if k == "message" else 1):
                raise ValueError(f"strata_prefix.{k}: a whole number ({'0' if k == 'message' else '1'} or more)")
        if has_images:
            print("[strata] strata_prefix ignored: the request has images", flush=True)
            return {}
        what = ",".join(f"{k}={v}" for k, v in spec.items())

        def lcp(a, b):
            n = 0
            for x, y in zip(a, b):
                if x != y:
                    break
                n += 1
            return n

        def head_ids(msgs):
            """The ids the rendered `msgs` start with (a message's trailing marker cut at PREFIX_END when it is there)."""
            marked, marked_tools, changed = mark_think_literals(msgs, tools, self.literals)
            effort = kwargs.get("reasoning_effort")
            off = kwargs.get("enable_thinking") is False        # render_prompt's rule for #458's effort_end
            kw = ({k: v for k, v in kwargs.items() if k not in ("reasoning_effort", "enable_thinking")}
                  if self.effort_end and (off or effort not in (None, "", "xhigh", "high")) else dict(kwargs))
            kw["add_generation_prompt"] = False
            text = self.template.render(marked, tools=marked_tools, **kw)
            cut = text.find(self.PREFIX_END)
            if cut >= 0:
                text = text[:cut]
            if changed:
                text, plain = unmark_think_literals(text, self.literals)
                return self.tok.encode(text, parse_special=True, plain=plain)
            return self.tok.encode(text, parse_special=True)

        if "tokens" in spec:
            n = spec["tokens"]
        elif "messages" in spec:
            k = spec["messages"]
            if k >= len(messages):
                print(f"[strata] strata_prefix.messages={k} ignored: the request has {len(messages)} message(s) and the "
                      "prefix must leave at least the last one", flush=True)
                return {}
            head = head_ids(messages[:k])
            n = lcp(head, ids)
        else:
            i, chars = spec["message"], spec["chars"]
            text = messages[i].get("content") if 0 <= i < len(messages) else None
            if not isinstance(text, str) or chars > len(text):
                print(f"[strata] strata_prefix ignored: message {i} has no text of {chars} characters", flush=True)
                return {}
            cut = messages[:i] + [{**messages[i], "content": text[:chars] + self.PREFIX_END}]
            head = head_ids(cut)
            n = lcp(head, ids)
        if n < 1 or n >= len(ids) - 1:
            print(f"[strata] strata_prefix {what} ignored: it leaves no usable prefix of the {len(ids)}-token prompt "
                  f"(common start {n})", flush=True)
            return {}
        return {"tokens": n}

    def drop_embeddings(self) -> None:
        """Delete the combined image file prepare() wrote when no run() took it over (a run deletes its own as it
        ends): a request can end before its answer starts - its client gone, anything raised after prepare() - and
        then nothing else would.  The handler calls this when a request is done."""
        path, self.embeddings.path = getattr(self.embeddings, "path", None), None
        if path is not None:
            Path(path).unlink(missing_ok=True)

    def _note(self, n, evs, st=None, rate=None):
        with self.status_lock:
            s = self.status if st is None else st
            s["generated"] = n
            if s.get("first_token") is None:
                s["first_token"] = time.time()
            (self.rate if rate is None else rate).append((time.time(), n))   # the live rate's window (RATE_WINDOW_S)
            for ev in evs:
                if ev.kind == "reasoning":
                    s["phase"] = "thinking"
                elif ev.kind == "content":
                    s["phase"] = "answering"
                elif ev.kind == "tool_start":
                    s["phase"], s["tool"] = f"writing a tool call: {ev.call.name}", ev.call.name
                elif ev.kind == "tool_call":
                    s["phase"] = "tool call complete"
                s["tail"] = ((s.get("tail") or "") + (ev.text or ""))[-600:]

    def _progress(self, last_print, every=1.0, st=None):
        """A progress line in the server window every `every` seconds while a request runs."""
        now = time.time()
        if now - last_print < every:
            return last_print
        with self.status_lock:
            s = dict(self.status if st is None else st)
        el = now - s.get("started", now)
        if s.get("first_token") is None:
            pr = getattr(self.engine, "progress", None)   # (position reached, prompt tokens): a reused prefix counts
            done = f"{pr[0]:,} of {pr[1]:,}" if pr and pr[1] else f"{s.get('prompt_tokens', 0):,}"   # as read (#29)
            print(f"[strata] reading the prompt: {done} tokens, {el:.0f} s so far", flush=True)
        else:
            rate = s["generated"] / max(RATE_MIN_SPAN_S, now - s["first_token"])
            # #984: while a thinking budget applies, the thinking line shows it, not the request's max_tokens
            limit = (f"budget {s['thinking_budget']} (max {s.get('max_tokens')})"
                     if s["phase"] == "thinking" and s.get("thinking_budget") else f"max {s.get('max_tokens')}")
            print(f"[strata] {s['phase']}: {s['generated']} of {limit} tokens, {rate:.1f} tok/s, "
                  f"{el:.0f} s", flush=True)
        return now

    def run(self, ids, thinking, tools, max_new, sampling, cancel, force=None,
            declared_only=False) -> Iterator[tuple[str, object]]:
        """The request's events (see _run); with "prevent_sleep" in the config, the PC stays awake while it runs or waits."""
        ka = self.keep_awake
        if ka is not None:
            ka.acquire()
        try:
            yield from self._run(ids, thinking, tools, max_new, sampling, cancel, force, declared_only)
        finally:
            if ka is not None:
                ka.release()

    def _run(self, ids, thinking, tools, max_new, sampling, cancel, force=None,
             declared_only=False) -> Iterator[tuple[str, object]]:
        """Yields ("event", Event) as text arrives, then ("done", {"finish": .., "completion_tokens": ..}).
        `force` (forced_call): the opening of the call the reply must make - see prepare()."""
        # the model card's sampling for this thinking state, then the config's block, then the Chat settings shared with
        # apps; the request's own fields win over all of them
        defaults = {**model_card_sampling(self.model_arch, thinking), **self.sampling_defaults, **self.shared}
        if defaults:                   # the request's own fields win (explicit 0 stays greedy)
            req_values = {k: v for k, v in (sampling or {}).items() if v is not None}
            sampling = {**defaults, **req_values}
        # #123: read after the merge, so a budget shared through POST /settings is seen like the other keys
        budget = self.reasoning_budget(sampling) if thinking else None
        if thinking and not budget and self.answer_room and max_new and max_new >= ANSWER_ROOM_MIN_CAP:
            # 0.1.45: max_tokens counts the thinking too.  A client that caps the reply at 8192 (the desktop agent's
            # harness sends max_completion_tokens: 8192 on every request) and a task the model thinks long about got
            # the cut-off in the middle of the thinking and NO answer.  So thinking is closed early enough to leave
            # the answer its quarter of the cap (the budget's wrap-up: a short closing sentence, </think>, the answer).
            # A request with a budget of its own, or max_tokens below ANSWER_ROOM_MIN_CAP, is left as it was.
            budget = max_new - min(max(ANSWER_RESERVE_MIN, max_new // 4), max_new // 2)
            if max_new <= 32768: print(f"[strata] thinking limited to {budget} of max_tokens {max_new} so the answer has room "
                  "(\"answer_room\": false in the config turns this off)", flush=True)
        if (thinking and not budget and self.answer_room and self.family == "deepseek41"
                and max_new and DS41_SMALL_CAP_MIN <= max_new < ANSWER_ROOM_MIN_CAP):
            # DeepSeek-V4.1 only: a short max_tokens (/v1/messages and /v1/chat/completions with 200-600) was spent
            # entirely on the thinking, so no text block ever arrived.  With no budget of the client's own, the thinking
            # is capped at half of max_tokens; then the wrap-up, </think> and the answer follow in the other half.
            budget = max_new // 2
            print(f"[strata] thinking limited to {budget} of max_tokens {max_new} so the answer has room "
                  "(\"answer_room\": false in the config turns this off)", flush=True)
        if budget and max_new and max_new > 0:
            # #984: a budget at or near max_tokens never fires before the reply is cut off while still thinking, so
            # it has no answer.  Close the thinking early enough to leave room for one.
            reserve = min(max(ANSWER_RESERVE_MIN, max_new // 4), max_new // 2)
            if budget > max_new - reserve:
                print(f"[strata] thinking capped at {max_new - reserve} of max_tokens {max_new} to leave room for "
                      f"the answer (budget {budget})", flush=True)
                budget = max(1, max_new - reserve)
        parser = self.parser_cls(thinking=thinking, tools=tools, stream_tools=True, recover=self.tool_call_recovery,
                                 declared_only=declared_only)
        detok, n, finish = Detokenizer(self.tok), 0, "length"
        run_tok, run_len, repeated = None, 0, False     # #606: the current run of one repeated token
        thinking_n = 0                                  # tokens written while thinking (Responses' reasoning_tokens)
        stop_list = stop_strings(sampling)
        stops = StopMatcher(stop_list) if stop_list else None

        def cut(evs):
            """The events with the answer's text cut at a stop string (OpenAI stop, Anthropic stop_sequences)."""
            if stops is None:
                return evs
            out = []
            for ev in evs:
                if stops.hit is not None:
                    break                               # nothing after the stop string is sent
                if ev.kind == "content":
                    text = stops.push(ev.text or "")
                    if text:
                        out.append(Event("content", text))
                    continue
                held = stops.flush()                    # a stop string does not run across a tool call or thinking
                if held:
                    out.append(Event("content", held))
                out.append(ev)
            return out
        opening = []                                    # without thinking the prompt ends with the forced opening:
        if force and not thinking:                      # the parser reads it as if the model had written it
            opening, force = parser.feed(force), None
        tail = ""                                       # the last characters written (the newlines before a call)
        answered, close_retried = False, False          # #1053: content or a call came out; the thinking was closed
        nf_retried = False                              # #879: the request was sent again after non-finite logits
        timings, before = None, None                    # this request's timings; the engine's `last` before it
        recovery_count, reasoning_text, repeat_coverage = 0, "", 0.0
        looped, next_loop_check = False, LOOP_CHECK_EVERY       # #728: reasoning that repeats whole passages
        raw_ids = []                                    # every generated id (STRATA_DEBUG: dump raw model text)
        emb, self.embeddings.path = getattr(self.embeddings, "path", None), None   # this run's to delete now
        img_keys, self.embeddings.keys = getattr(self.embeddings, "keys", None), None
        if emb and img_keys:
            sampling = {**(sampling or {}), "strata_img_keys": img_keys}   # #2228: sent beside the embeddings file
        # Only a DONE line replaces engine.last, so a request that died, errored or was disconnected must not have
        # the PREVIOUS request's decode figures recorded as its own: each pass keeps its own DONE (a segment) if
        # one arrived.  A reasoning-budget continuation is another native generation, not another API request,
        # so the generated reasoning must not become this request's cached input (#615).
        segments = []
        timed_out = False
        trace = getattr(self.request_trace, "record", None)
        waiting = time.perf_counter()
        first_engine_token_s = None  # this request, from entering the engine queue; includes queue/prefill
        # #465: with "parallel" (the engine's batch slots) requests run at once: each keeps its own status and rate
        # window (self.status says busy while any runs); one at a time they are self.status / self.rate, as before
        par = bool(getattr(self.engine, "batch", 0))
        st = {} if par else self.status
        rate = collections.deque(maxlen=32) if par else self.rate
        request_id = uuid.uuid4().hex[:12]
        setattr(cancel, "_strata_request_id", request_id)
        deadline = time.monotonic() + self.queue_timeout_s if self.queue_timeout_s is not None else None
        if deadline is not None:
            setattr(cancel, "_strata_queue_deadline", deadline)
        def queue_started():
            with self.status_lock:
                row = self.terminal_requests.get(request_id)
                if row is not None and row["state"] != "reading":
                    row["state"] = "reading"
                    self._terminal_event_locked(request_id, "reading")
        setattr(cancel, "_strata_queue_started", queue_started)
        def replica_selected(index: int) -> None:
            with self.status_lock:
                row = self.terminal_requests.get(request_id)
                if row is not None:
                    row["replica"] = index
        setattr(cancel, "_strata_replica_selected", replica_selected)
        with self.status_lock:
            self.status["queued"] += 1
            self.terminal_requests[request_id] = {"id": request_id, "state": "queued",
                                                    "queued_at": time.time(), "admitted_at": None,
                                                    "prompt_tokens": len(ids)}
            self._terminal_event_locked(request_id, "queued")
        queued_counted = True
        try:
            # --batch: the engine runs several requests at once (StrataEngine.generate_batched orders them)
            turn = contextlib.ExitStack()
            if not par:
                # #1619: a request waiting for the single turn sends a keep-alive every QUEUE_BEAT_S, so a client with
                # a stream idle timeout does not abort a healthy queued request (and its retry read the prompt again);
                # a client that left while queued gives its place up instead of taking the turn only to cancel there
                got = False
                while not got:
                    if cancel.is_set():
                        raise QueueCancelled("the client disconnected while waiting for the engine")
                    left = deadline - time.monotonic() if deadline is not None else QUEUE_BEAT_S
                    if left <= 0:
                        raise engine_client.QueueWaitExpired("the inference queue wait deadline expired; retry later")
                    got = self.fifo.acquire(timeout=min(QUEUE_BEAT_S, left))
                    if got:
                        if cancel.is_set():
                            self.fifo.release()
                            raise QueueCancelled("the client disconnected while waiting for the engine")
                        if deadline is not None and time.monotonic() >= deadline:
                            self.fifo.release()
                            raise engine_client.QueueWaitExpired(
                                "the inference queue wait deadline expired; retry later")
                        turn.callback(self.fifo.release)
                    else:
                        yield "ping", None
            with turn:
                try:
                    with self.status_lock:
                        if trace is not None:
                            trace["queue_s"] += round(time.perf_counter() - waiting, 3)
                            trace["state"] = "generating"
                        if not par:
                            self.status["queued"] -= 1
                            queued_counted = False
                            self.terminal_requests[request_id].update(state="admitted", admitted_at=time.time())
                            self._terminal_event_locked(request_id, "admitted")
                    # issue #27: it died in an earlier request (or was unloaded) - start it again instead of failing
                    # (parallel requests do not hold the fifo: one of them starts it, the others wait for that).
                    # #1603: a parallel request stays counted as queued until it is registered as running below -
                    # waiting here for the fifo (an image encode, a /v1/vram resize) left it neither - and one on a
                    # loaded engine has nothing to start, so it does not wait for the fifo at all
                    if not par:
                        self.ensure_loaded()
                    elif not (self.loaded() and not self._vision_down()):
                        with self._inference_fifo(deadline, cancel):
                            self.ensure_loaded()
                    if not par and getattr(self.engine, "request_protocol", False):
                        turn.close()
                        par = True
                    with self.status_lock:
                        if par and queued_counted:
                            self.status["queued"] -= 1
                            queued_counted = False
                            self.terminal_requests[request_id].update(state="waiting", admitted_at=time.time())
                            self._terminal_event_locked(request_id, "admitted")
                        st.update(busy=True, phase="reading the prompt", prompt_tokens=len(ids),
                                  generated=0, started=time.time(), first_token=None, tool=None, tail="",
                                  max_tokens=max_new, reasoning_recoveries=0, thinking_budget=budget)
                        if par:
                            self.live_reqs[id(st)] = (st, rate)
                            self.status.update(st)
                        self.last_request_at = time.time()
                        rate.clear()                    # the previous request's samples must not leak into this one
                    before = getattr(self.engine, "last", None)
                    last_print = time.time()
                    prompt, thought = ids, 0            # thought: the reasoning tokens so far (the budget's count)
                    for ev in opening:
                        yield "event", ev
                    while True:
                        segment_before = getattr(self.engine, "last", None)
                        segments_before = len(segments)     # (#879: a pass sent again leaves no segment)
                        gen = self.engine.generate(prompt, max_new - n, sampling, cancel, embeddings=emb) if emb \
                            else self.engine.generate(prompt, max_new - n, sampling, cancel)
                        recover_prompt = None
                        seg, wrap, leaving = [], False, False   # this pass's tokens; the budget is reached; closed
                        nonfinite = False               # #879: the engine ended this pass on non-finite logits
                        opens = False                   # the thinking is over: write the forced call's opening
                        try:
                            # A value the frontend holds back (a tool call's array or object argument is sent
                            # whole, once complete) can keep the stream silent for minutes while tokens are
                            # generated; a client's idle timeout then ends the request.  Send the same keep-alive
                            # the engine's heartbeat sends when nothing has gone out for KEEPALIVE_S.
                            last_out = time.perf_counter()
                            for t in gen:
                                if t is None:               # heartbeat while the engine is quiet
                                    last_print = self._progress(last_print, st=st)
                                    last_out = time.perf_counter()
                                    yield "ping", None
                                    continue
                                n += 1
                                if first_engine_token_s is None:
                                    first_engine_token_s = round(time.perf_counter() - waiting, 3)
                                    with self.status_lock:
                                        row = self.terminal_requests.get(request_id)
                                        if row is not None:
                                            row["first_engine_token_s"] = first_engine_token_s
                                            self._terminal_event_locked(request_id, "first_token")
                                if trace is not None and trace["first_token_s"] is None:
                                    with self.status_lock:
                                        trace["first_token_s"] = round(time.perf_counter() - trace["_clock"], 3)
                                if t in self.stop_ids:
                                    if force and parser.state == "reasoning" and parser.buf == THINK_END:
                                        opens = True    # #537: the held </think> was the end: the call opens there
                                        break
                                    finish = "stop"
                                    raw_ids.append(t)
                                    break
                                raw_ids.append(t)
                                seg.append(t)
                                thinking_n += parser.state in ("reasoning", "rcall")
                                run_len = run_len + 1 if t == run_tok else 1
                                run_tok = t
                                if self.repeat_stop_tokens and run_len >= self.repeat_stop_tokens:
                                    repeated = True     # #606: a degenerate output, not an answer: end it here
                                    break
                                piece = detok.push(t)
                                tail = (tail + piece)[-2:]
                                evs = cut(parser.feed(piece))
                                self._note(n, evs, st, rate)
                                last_print = self._progress(last_print, st=st)
                                for ev in evs:
                                    if self.reasoning_loop_recovery and ev.kind == "reasoning":
                                        reasoning_text += ev.text or ""
                                    if ev.kind in ("content", "tool_start", "tool_call"):
                                        answered = True
                                    yield "event", ev
                                if evs:
                                    last_out = time.perf_counter()
                                elif time.perf_counter() - last_out >= KEEPALIVE_S:
                                    last_out = time.perf_counter()
                                    yield "ping", None
                                if stops is not None and stops.hit is not None:
                                    finish = "stop"         # gen.close() below STOPs the engine, as for a stop token
                                    break
                                if (self.reasoning_loop_recovery and not recovery_count and parser.state == "reasoning"
                                        and n >= next_loop_check and not parser.buf and not parser.pending
                                        and not detok.pending()):
                                    next_loop_check = n + LOOP_CHECK_EVERY
                                    repeat_coverage = reasoning_repeat_coverage(reasoning_text)
                                    if repeat_coverage >= LOOP_COVERAGE:
                                        if self.reasoning_loop_recovery == "stop":
                                            looped = True        # #728: end the reply here, as for a repeated token
                                            break
                                        if not emb:              # "recover": not for a request with pictures
                                            recover_prompt = focused_recovery_prompt(self.tok, ids, raw_ids)
                                            if recover_prompt is not None:
                                                break
                                if budget and parser.state == "reasoning":
                                    thought += 1
                                    # at a clean point: no tag held back, no character split across tokens
                                    if thought >= budget and parser.clean and not detok.pending():
                                        wrap = True
                                        break
                                if force:
                                    if any(ev.kind in ("tool_start", "tool_call") for ev in evs):
                                        force = None    # the model is writing a call of its own
                                    elif parser.state == "content" and parser.clean and not detok.pending():
                                        opens = True
                                        break
                        except engine_client.EngineNonFinite:
                            nonfinite = True
                        except engine_client.EngineDied as e:
                            finish = "error"
                            self._say_died(e)
                            raise
                        except engine_client.QueueWaitExpired:
                            finish = "error"
                            timed_out = True
                            raise
                        except ValueError as e:             # the engine's ERR line (it may have ended after it)
                            finish = "error"
                            print(f"[strata] the engine reported an error: {e}", flush=True)
                            raise
                        except GeneratorExit:               # the client went away: an engine that never acknowledges
                            leaving = True                  # the STOP below is ended, but no error replaces this
                            raise
                        finally:
                            try:
                                gen.close()             # STOP+drain to THIS request's DONE while still holding the
                                #                         fifo, so a stop-token break can't leave the shared engine
                                #                         queue mid-drain for the next request to read as its own DONE
                            except engine_client.EngineSilent as e:
                                # #481: the STOP was never acknowledged: the engine is ended
                                finish = "error"
                                self._say_died(e)
                                if not leaving and not cancel.is_set():
                                    raise
                            segment_done = getattr(self.engine, "last", None)
                            if segment_done is not None and segment_done is not segment_before:
                                segments.append(dict(segment_done))
                        if nonfinite and not cancel.is_set():
                            # #879: the engine emitted none of the poisoned window's tokens and reads its next prompt
                            # from token 0.  Nothing sent yet: the same request once more, unseen by the client
                            if n == 0 and not nf_retried:
                                nf_retried = True
                                # the retry reads from token 0: its DONE, not this one's, says what was reused
                                del segments[segments_before:]
                                print("[strata] the engine's logits became non-finite before the first token (#879); it "
                                      "dropped its cached state: sending the request again once", flush=True)
                                continue
                            if n == 0:
                                finish = "error"
                                print("[strata] the engine's logits became non-finite again on the retry (#879): the "
                                      "request ends with an error", flush=True)
                                raise engine_client.EngineNonFinite("the model's output became non-finite (NaN) twice "
                                                                    "in a row (#879); the engine has dropped its cached "
                                                                    "state - send the request again")
                            print(f"[strata] the engine's logits became non-finite after {n} tokens (#879): the reply "
                                  "ends here as \"length\"; the engine dropped its cached state, so the next request "
                                  "reads its prompt from the start", flush=True)
                            finish = "length"
                            break
                        if recover_prompt is not None and not cancel.is_set() and n < max_new:
                            recovery_count += 1
                            # Only the two settings that keep the same words coming are raised (temperature to at
                            # least 1.0, presence penalty to at least 1.5): a client's top_p and top_k are never
                            # touched, and its own request dict is copied, not changed.
                            want = (max(1.0, float((sampling or {}).get("temperature") or 0)),
                                    max(1.5, float((sampling or {}).get("presence_penalty") or 0)))
                            if want != tuple(float((sampling or {}).get(k) or 0)
                                             for k in ("temperature", "presence_penalty")):
                                sampling = {**(sampling or {}), "temperature": want[0], "presence_penalty": want[1]}
                            prompt = recover_prompt
                            with self.status_lock:
                                st["reasoning_recoveries"] = recovery_count
                            if trace is not None:
                                trace["reasoning_recoveries"] = recovery_count
                                trace["reasoning_repeat_coverage"] = round(repeat_coverage, 3)
                            print(f"[strata] repeated reasoning detected at {n} tokens "
                                  f"(coverage={repeat_coverage:.3f}); resuming the same output with the low-effort "
                                  "instruction (reasoning_loop_recovery)", flush=True)
                            continue
                        if (self.reasoning_close_retry and thinking and finish == "stop" and not close_retried
                                and not answered and not wrap and not opens and parser.state == "reasoning"
                                and parser.buf != THINK_END         # #537: not after a held </think>, the end
                                and (stops is None or stops.hit is None) and not cancel.is_set()):
                            # #1053: the model wrote its reasoning and stopped before </think>: the client would get
                            # an empty answer.  Close the thinking once and let it answer.
                            close_retried = True
                            extra = self.tok.encode(REASONING_CLOSE, parse_special=True)
                            if max_new - n - len(extra) >= 1:
                                print("[strata] the reply ended inside its thinking with no answer: closing the "
                                      "thinking once and continuing (reasoning_close_retry)", flush=True)
                                for t in extra:
                                    n += 1
                                    raw_ids.append(t)
                                    thinking_n += parser.state in ("reasoning", "rcall")
                                    evs = cut(parser.feed(detok.push(t)))
                                    self._note(n, evs, st, rate)
                                    for ev in evs:
                                        yield "event", ev
                                prompt = prompt + seg + extra
                                finish = "length"
                                continue
                        if not (wrap or opens) or cancel.is_set():
                            break
                        # #123: the thinking reached reasoning_budget_tokens.  Close it the way the model would (a
                        # short wrap-up and </think>) and let it answer: the next pass's prompt is this one plus what
                        # was generated plus the wrap-up, so the engine continues from the prefix it already holds.
                        # A forced call (tool_choice) is opened the same way: after the wrap-up, or where the
                        # thinking ended, after the blank line the template puts before a call.
                        if wrap:
                            budget = st["thinking_budget"] = None
                            text = REASONING_WRAP_UP + (force or "")
                        else:
                            text = "\n" * (2 - (len(tail) - len(tail.rstrip("\n")))) + force
                        force = None
                        extra = self.tok.encode(text, parse_special=True)
                        if max_new - n - len(extra) < 1:
                            break                       # no room left to answer: "length", as without a budget
                        if wrap:
                            print(f"[strata] thinking budget reached ({thought} tokens): wrapping up the thinking",
                                  flush=True)
                        for t in extra:
                            n += 1
                            raw_ids.append(t)
                            thinking_n += parser.state in ("reasoning", "rcall")
                            evs = cut(parser.feed(detok.push(t)))
                            self._note(n, evs, st, rate)
                            for ev in evs:
                                yield "event", ev
                        if stops is not None and stops.hit is not None:
                            finish = "stop"
                            break
                        prompt = prompt + seg + extra
                    if cancel.is_set():
                        finish = "cancel"
                    elif looped:
                        print(f"[strata] the thinking repeated the same passages (coverage={repeat_coverage:.3f}) at "
                              f"{n} tokens: ended as \"length\" (reasoning_loop_recovery: \"stop\" in "
                              "strata-<model>.json; remove it to turn this off)", flush=True)
                    elif repeated:
                        print(f"[strata] the reply repeated one token ({self.tok.decode([run_tok])!r}) "
                              f"{run_len} times in a row: ended as \"length\" (repeat_stop_tokens in "
                              "strata-<model>.json; 0 turns this off). If a new request with a short prompt does the "
                              "same, restart the server and report it (#606)", flush=True)
                except GeneratorExit:                   # the client disconnected mid-stream
                    finish = "disconnect"
                    raise
                finally:
                    # #266: settle this request's status, history and totals while still holding the fifo: once
                    # it is released the next request sets its own status, which this must not record or clear
                    with self.status_lock:
                        if st.get("busy") and not timed_out:
                            last = request_stats(segments)          # only this request's own DONEs count
                            started = st.get("started", time.time())
                            cvec = (getattr(self.engine, "info", {}) or {}).get("cvec", 0)
                            loaded = str(cvec) not in ("0", "", "None")
                            hit_rate = round(last["hits"] / last["lookups"], 3) if last.get("lookups") else None
                            # #588: the share the GPU read over PCIe (--pcie-frac) or another GPU computed - not in
                            # the hit rate's lookups, which is the VRAM share of the rest (None: an older engine)
                            routed = (last.get("lookups") or 0) + (last.get("offloaded") or 0)
                            pcie_share = round(last["offloaded"] / routed, 3) \
                                if last.get("offloaded") is not None and routed else None
                            seen = prompt_tokens_seen(len(ids), last)   # #471: < len(ids) when cancelled mid-read
                            self.history.append({
                                "id": request_id,
                                "projection": (sampling or {}).get("experimental_speed_projection") is not False
                                if loaded else None,
                                "time": started, "duration_s": round(time.time() - started, 1), "finish": finish,
                                "prompt_tokens": seen, "reused": last.get("reused"), "output_tokens": n,
                                "first_engine_token_s": first_engine_token_s,
                                # the request's whole prompt, and the tokens read of it (None: an older engine)
                                "prompt_total": len(ids), "prompt_read": last.get("prompt_read"),
                                "reasoning_recoveries": recovery_count,
                                "reasoning_repeat_coverage": round(repeat_coverage, 3),
                                "engine_generated": last.get("generated"),
                                "prompt_ms": last.get("prompt_ms"), "decode_ms": last.get("decode_ms"),
                                "decode_tok_s": round(last["generated"] / (last["decode_ms"] / 1000), 1)
                                if n and (last.get("generated") or 0) >= 2 and last.get("decode_ms") else None,
                                "hit_rate": hit_rate, "pcie_share": pcie_share, "ram_blobs": last.get("ram_blobs"),
                                "file_blobs": last.get("file_blobs"), "file_mb": last.get("file_mb"),
                                # #457: the speculative drafts from the DONE line (None: the engine did not say)
                                "drafts_offered": last.get("drafts_offered"),
                                "drafts_accepted": last.get("drafts_accepted")})
                            t = self.totals
                            t["requests"] += 1
                            t["prompt_tokens"] += seen
                            t["reused"] += last.get("reused") or 0
                            t["output_tokens"] += n
                            t["prompt_ms"] += last.get("prompt_ms") or 0.0
                            t["decode_ms"] += last.get("decode_ms") or 0.0
                            t["drafts_offered"] += last.get("drafts_offered") or 0
                            t["drafts_accepted"] += last.get("drafts_accepted") or 0
                            ft0 = st.get("first_token")             # this request's own (#465: one per request)
                            self.latencies.observe(ft0 - started if ft0 else None, n,
                                                   (last.get("decode_ms") or 0.0) / 1000, time.time() - started)
                            fresh = getattr(self.engine, "last", None)
                            if fresh is not None and fresh is not before:      # the engine's clock for THIS request
                                timings = request_timings(seen, n, last)
                                self.last_timings = dict(timings, at=int(time.time())) if timings else None
                            self.last_request_at = time.time()
                            now = time.time()
                            el = now - st.get("started", now)
                            ft = st.get("first_token")
                            rate = answer_speed(last, n, now - ft if ft else None)
                            hit_msg = f", expert cache {hit_rate*100:.1f}% hit" if hit_rate is not None else ""
                            if hit_msg and pcie_share:
                                hit_msg += f" (+{pcie_share*100:.1f}% of the routed experts over PCIe)"
                            rate_txt = f"{rate:.1f} tok/s" if rate is not None else "no speed"
                            print(f"[strata] done: {n} tokens in {el:.0f} s ({rate_txt}) "
                                  f"({finish}, cancel={cancel.is_set()}){hit_msg}", flush=True)
                            if finish == "length" and parser.state in ("reasoning", "rcall"):   # #530
                                print("[strata] the reply reached max tokens while still thinking, so it has no "
                                      "answer: a thinking budget (reasoning_budget_tokens, in the request or in "
                                      "strata-<model>.json for every request) leaves room to answer", flush=True)
                            if os.environ.get("STRATA_DEBUG") and raw_ids:
                                print(f"[strata] raw: {self.tok.decode(raw_ids)!r}", flush=True)
                                print(f"[strata] raw ids: {list(raw_ids)}", flush=True)
                        st["busy"] = False
                        st.pop("tail", None)                     # #212: the answer's end is not kept once it is done
                        st.pop("tool", None)
                        if par:
                            self.live_reqs.pop(id(st), None)
                            if self.live_reqs:              # the newest request still running
                                self.status.update(list(self.live_reqs.values())[-1][0])
                            else:
                                self.status.update(busy=False)
                                self.status.pop("tail", None)
                                self.status.pop("tool", None)
        except QueueCancelled:
            finish = "cancel"
            yield "done", {"finish": "cancel", "completion_tokens": 0, "reused": 0, "timings": None,
                           "reasoning_tokens": 0, "reasoning_recoveries": 0}
            return
        except engine_client.QueueWaitExpired:
            finish = "timeout"
            with self.status_lock:
                self.queue_timeouts += 1
            raise
        except Exception:
            finish = "error"
            raise
        finally:
            with self.status_lock:
                if queued_counted:
                    self.status["queued"] -= 1
                self.terminal_requests.pop(request_id, None)
                event = {"stop": "finished", "length": "finished", "cancel": "cancelled",
                         "disconnect": "disconnected", "timeout": "timeout", "error": "error"}.get(finish, "error")
                self._terminal_event_locked(request_id, event, finish)
            if emb:
                Path(emb).unlink(missing_ok=True)
        for ev in cut(parser.finish(finish)):
            yield "event", ev
        if parser.rescued or parser.refused:
            self.totals["tool_calls_from_reasoning"] = self.totals.get("tool_calls_from_reasoning", 0) + parser.rescued
            print(f"[strata] tool calls inside the thinking: {parser.rescued} read as calls, {parser.refused} kept as "
                  f"reasoning (quoted, or the turn did not end on a stop)", flush=True)
        if parser.unread:
            named = getattr(parser, "unread_names", None)
            print(f"[strata] {parser.unread} <tool_call> block(s) in the answer did not read as a call of a declared "
                  "tool and went back as text (a name the request did not declare, or a form the template does "
                  "not use)" + (f"; the names the model wrote: {', '.join(named)}" if named else ""), flush=True)
        if stops is not None and stops.hit is None and stops.held:
            yield "event", Event("content", stops.flush())     # the held tail was not a stop string after all
        done = {"finish": finish, "completion_tokens": n, "reused": (timings or {}).get("cache_n", 0),
                "timings": timings, "reasoning_tokens": thinking_n,
                "reasoning_recoveries": recovery_count}
        if stops is not None and stops.hit is not None:
            done["stop_sequence"] = stops.hit
        yield "done", done


def prompt_tokens_seen(prompt_tokens: int, last: dict) -> int:
    """#471: the prompt tokens a request got through - all of them, unless the engine's DONE line says a cancel stopped
    its prompt read part-way (then the reused ones plus those read).  /metrics' history and totals count these, so a
    cancelled read is neither recorded as the whole prompt nor given a rate from tokens it never read.  An engine
    before 0.1.36 does not say (no `prompt_read`): the whole prompt, as before."""
    read = last.get("prompt_read")
    if read is None or last.get("finish") != "cancel":
        return prompt_tokens
    return min(prompt_tokens, int(last.get("reused") or 0) + int(read))


def answer_speed(last: dict, generated: int, wall_s: float | None = None) -> float | None:
    """The one definition of "answer speed": every token the model produced (thinking and answer, the tokens it kept - not
    the MTP drafts it threw away) over the decode time, first generated token to last.  The engine's DONE line holds both
    numbers; /metrics (`decode_tok_s`), the timings (`predicted_per_second`), the Terminal monitor and the app all use them.
    Only an engine that sent no DONE falls back to the server's own clock from the first token, and a clock younger than
    RATE_MIN_SPAN_S says nothing (it printed 1000000 tok/s for a one-token reply).  Prompt reading is never in it."""
    produced, decode_ms = last.get("generated"), last.get("decode_ms")
    if produced and produced >= 2 and decode_ms and decode_ms > 0:       # one token has no first-to-last span (a 1-token probe printed 1,250 tok/s)
        return round(produced / (decode_ms / 1000), 1)
    if generated and wall_s is not None and wall_s >= RATE_MIN_SPAN_S:
        return round(generated / wall_s, 1)
    return None


def request_timings(prompt_tokens: int, generated: int, last: dict) -> dict | None:
    """One request's `timings` in llama.cpp's names (what its clients show as speed), from the engine's own clock
    (StrataEngine.last): prompt_n is what was read, cache_n what the conversation cache already held.  None when the
    engine keeps no clock (MockEngine)."""
    if last.get("prompt_ms") is None:
        return None
    cache_n = int(last.get("reused") or 0)
    if last.get("prompt_read") is not None and last.get("finish") != "cancel":
        # #1870: the engine says how many prompt tokens it read: the slot's own session may have held the rest
        # (not the conversation cache), so the rate is not made of tokens it never read
        cache_n = max(cache_n, max(0, prompt_tokens - int(last["prompt_read"])))
    prompt_n, prompt_ms, decode_ms = max(0, prompt_tokens - cache_n), float(last["prompt_ms"]), float(last.get("decode_ms") or 0)
    decoded = int(last.get("generated") or generated)          # the engine's count gives its rate, as /metrics does
    return {"cache_n": cache_n, "prompt_n": prompt_n, "prompt_ms": round(prompt_ms, 1),
            "prompt_per_token_ms": round(prompt_ms / prompt_n, 3) if prompt_n else None,
            "prompt_per_second": round(prompt_n / (prompt_ms / 1000), 1) if prompt_n and prompt_ms > 0 else None,
            "predicted_n": generated, "predicted_ms": round(decode_ms, 1),
            "predicted_per_token_ms": round(decode_ms / decoded, 3) if decoded else None,
            "predicted_per_second": round(decoded / (decode_ms / 1000), 1) if decoded and decode_ms > 0 else None,
            # the speculative drafts, as llama.cpp names them (from PR #83, @mikicvi): only when the engine reported them
            **({"draft_n": int(last["drafts_offered"]), "draft_n_accepted": int(last["drafts_accepted"])}
               if last.get("drafts_offered") is not None else {})}


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
        if any(segment.get(key) is not None for segment in segments):
            result[key] = sum(segment.get(key) or 0 for segment in segments)
    return result


def _debug_req(api, req, messages, tools, max_new, thinking, prompt_tokens):
    """One compact line per request while diagnosing blank/empty turns. Set STRATA_DEBUG=1 to enable."""
    if not os.environ.get("STRATA_DEBUG"):
        return
    last = messages[-1] if messages else {}
    body = last.get("content")
    if isinstance(body, list):
        body = " ".join(p.get("text", "") for p in body if isinstance(p, dict))
    preview = (str(body or "")[:80]).replace("\n", " ")
    print(f"[strata] req {api}: msgs={len(messages)} tools={len(tools or [])} "
          f"max_tokens_raw={req.get('max_tokens')!r}/{req.get('max_completion_tokens')!r} "
          f"max_new={max_new} thinking={thinking} stream={bool(req.get('stream'))} "
          f"prompt_tokens={prompt_tokens} last={last.get('role')!r}:{preview!r}", flush=True)


# ------------------------------------------------------------------------------------------------ MCP tool loop
def run_with_mcp(svc: Service, hub, messages, tools, kw, ids, thinking, max_new, max_req, sampling, cancel,
                 mcp_names):
    """Service.run with the MCP tools executed here: the model writes a call to an MCP tool, the server runs it, adds
    the call and its result to the conversation and lets the model continue - up to `max_rounds` times.  Yields what
    Service.run yields (text, thinking, the request's own tool calls) plus ("mcp", {...}) for the tool activity, and
    one ("done", ...) at the very end with the output tokens of every round.

    `mcp_names`: the MCP tools this request offered; any other call is one of the request's own tools and ends the
    turn as always (the client answers it).  MCP calls written in the same answer are then not run (their results
    could not reach the model before the client's)."""
    max_rounds = int(hub.settings["max_rounds"])
    total, rounds, done = 0, 0, None
    messages = list(messages)
    while True:
        text, reasoning, calls, own_calls = [], [], [], 0
        for kind, x in svc.run(ids, thinking, tools, max_new, sampling, cancel, declared_only=True):
            if kind == "done":
                done = x
                continue
            if kind == "event":
                ev: Event = x
                if ev.call is not None and ev.call.name in mcp_names:
                    if ev.kind == "tool_start":          # the model has started writing a call: say so at once
                        yield "mcp", {"event": "start", "id": ev.call.id, "name": ev.call.name}
                    elif ev.kind == "tool_call":
                        calls.append(ev.call)
                    continue                             # its argument pieces are not streamed to the client
                if ev.kind == "tool_call":
                    own_calls += 1
                elif ev.kind == "content":
                    text.append(ev.text)
                elif ev.kind == "reasoning":
                    reasoning.append(ev.text)
            yield kind, x
        total += done["completion_tokens"]
        run_them = calls and not own_calls and done["finish"] == "stop" and not cancel.is_set()
        if run_them and rounds >= max_rounds:
            yield "mcp", {"event": "limit", "max_rounds": max_rounds}
            run_them = False
        if not run_them:
            for c in calls:                              # announced, never run: close them in the client's view
                yield "mcp", {"event": "result", "id": c.id, "ok": False, "skipped": True, "text": "not run",
                              "chars": 0, "truncated": False, "ms": 0}
            break
        rounds += 1
        results = []
        for c in calls:
            s, tool = hub.routes().get(c.name, (None, c.name))
            yield "mcp", {"event": "call", "id": c.id, "name": c.name, "server": s.name if s else None,
                          "tool": tool, "arguments": c.arguments, "round": rounds}
            # The call runs on a thread while this generator keeps yielding heartbeats: they reach the client as
            # keep-alives, which is how a client that went away (the web app's Stop) is noticed during a slow tool.
            box = {}

            def work(c=c, box=box):
                try:
                    box["r"] = hub.call(c.name, c.arguments, cancel)
                except McpCancelled:
                    box["cancelled"] = True
            worker = threading.Thread(target=work, daemon=True)
            worker.start()
            try:
                while worker.is_alive():
                    worker.join(1.0)
                    if worker.is_alive():
                        yield "ping", None
            except GeneratorExit:
                cancel.set()                             # the client is gone: stop the tool too
                raise
            if "r" not in box:
                break
            r = box["r"]
            print(f"[strata] tool {c.name}: {'ok' if r['ok'] else 'error'}, {r['chars']:,} characters in "
                  f"{r['ms'] / 1000:.1f} s{' (truncated for the model)' if r['truncated'] else ''}", flush=True)
            results.append(r["text"])
            yield "mcp", {"event": "result", "id": c.id, **{k: r[k] for k in ("ok", "text", "chars", "truncated", "ms")}}
        if cancel.is_set() or len(results) < len(calls):
            done = {**done, "finish": "cancel"}
            break
        messages.append({"role": "assistant", "content": "".join(text).strip(),
                         **({"reasoning_content": "".join(reasoning).strip()} if reasoning else {}),
                         "tool_calls": [{"function": {"name": c.name, "arguments": c.arguments}} for c in calls]})
        messages += [{"role": "tool", "content": r} for r in results]
        try:
            ids, thinking, max_new = svc.prepare(messages, tools, kw, max_req, cancel=cancel)
        except ClientGone:                                      # #1999: the client left while a picture was prepared
            done = {**done, "finish": "cancel"}
            break
    yield "done", {**done, "completion_tokens": total, "prompt_tokens": len(ids)}


# ------------------------------------------------------------------------------------------------ OpenAI
# The engine's last PP line stops short of the prompt: up to `--short-read` tokens (generate.cpp's `short_read`, 64 by
# default) are read through the verify windows instead of the batched path, so the batched path never reports the whole
# prompt.  A PP that close to the end says the prompt is read.  It matters because a prompt shorter than one `--prefill`
# chunk sends exactly one PP, that one: a client would see a bar that appears already full.  Such a line is dropped, and
# the answer's first token ends the progress for the client anyway.
PP_DONE_TAIL = 64


def prompt_progress(svc: Service):
    """The last PP line as llama.cpp's `prompt_progress`, so a client written for llama.cpp reads both servers: the
    same four fields, `time_ms` elapsed since prompt reading started, and `processed` counting from token 0 so that a
    client takes the reused prefix out with `cache` (llama.cpp's rule: the timed progress is
    `(processed-cache)/(total-cache)`).  None when there is nothing to say: no PP yet, or the line that says the prompt
    is read.  An engine too old to print RESUME gives `cache` 0, which only makes a reused prefix count as work."""
    engine = svc.engine
    if getattr(engine, "batch", 0):      # --batch: the engine's progress is one global line, not this request's
        return None
    progress = getattr(engine, "progress", None)
    if not progress:
        return None
    read, total = progress
    if total <= 0 or total - read <= PP_DONE_TAIL:
        return None
    return {"total": int(total), "cache": int(getattr(engine, "reused", 0) or 0), "processed": int(read),
            "time_ms": int(getattr(engine, "progress_ms", 0) or 0)}


def openai_chunks(svc: Service, req: dict, ids, thinking, tools, max_new, cancel, run=None, force=None):
    """`run`: the events to send instead of Service.run's (run_with_mcp); its ("mcp", {...}) items become chunks with
    an empty delta and a `strata_mcp` field, which only the web app reads.  `force`: see Service.run."""
    cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())
    model = svc.model_for(req)
    # llama.cpp's flag, and its default: a client that wants prompt progress asks for it, and nothing changes for one
    # that does not.  Strata already sends a keep-alive comment on every PP line, so this only fills that line in.
    want_progress = bool(req.get("return_progress"))

    def chunk(delta, finish=None):
        return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    yield chunk({"role": "assistant", "content": ""})
    calls = 0
    streamed = {}                                  # tool call id -> index, for calls sent piece by piece
    finished = set()                               # ... and the ones whose final tool_call came (#211)
    for kind, x in run if run is not None else svc.run(ids, thinking, tools, max_new, req, cancel, force=force, declared_only=True):
        if kind == "ping":
            # a PP line: the prompt is being read and there is nothing to say yet.  It used to be only the SSE comment;
            # now it carries the progress when the client asked, and stays the comment when there is nothing to report.
            pp = prompt_progress(svc) if want_progress else None
            yield None if pp is None else {**chunk({}), "prompt_progress": pp}
        elif kind == "mcp":
            c = chunk({})
            c["strata_mcp"] = x
            yield c
        elif kind == "event":
            ev: Event = x
            if ev.kind == "reasoning" and ev.text:
                yield chunk({"reasoning_content": ev.text})
            elif ev.kind == "content" and ev.text:
                yield chunk({"content": ev.text})
            elif ev.kind == "tool_start":
                streamed[ev.call.id] = calls
                calls += 1
                yield chunk({"tool_calls": [{"index": streamed[ev.call.id], "id": ev.call.id, "type": "function",
                                             "function": {"name": ev.call.name, "arguments": ""}}]})
            elif ev.kind == "tool_args":
                yield chunk({"tool_calls": [{"index": streamed[ev.call.id], "function": {"arguments": ev.text}}]})
            elif ev.kind == "tool_call" and ev.call.id in streamed:
                finished.add(ev.call.id)
            elif ev.kind == "tool_call":
                yield chunk({"tool_calls": [{"index": calls, "id": ev.call.id, "type": "function",
                                             "function": {"name": ev.call.name,
                                                          "arguments": json.dumps(ev.call.arguments, ensure_ascii=False)}}]})
                calls += 1
        else:
            # a streamed call without its final tool_call is one the output ended inside: not "tool_calls" (#211)
            whole = calls and streamed.keys() <= finished
            finish = "tool_calls" if whole and x["finish"] == "stop" else {"cancel": "stop"}.get(x["finish"], x["finish"])
            last = chunk({}, finish)
            pt = x.get("prompt_tokens", len(ids))     # after MCP rounds: the last round's prompt
            last["usage"] = {"prompt_tokens": pt, "completion_tokens": x["completion_tokens"],
                             "total_tokens": pt + x["completion_tokens"],
                             # the part of the prompt the conversation cache already held (OpenAI's field)
                             "prompt_tokens_details": {"cached_tokens": x.get("reused") or 0}}
            if x.get("timings"):
                last["timings"] = x["timings"]          # llama.cpp's field: the speed its clients show
            yield last


def _is_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except ValueError:
        return False


def openai_collect(chunks) -> dict:
    content, reasoning, by_index, last, mcp = [], [], {}, None, []
    for c in chunks:
        if c is None:                              # a heartbeat
            continue
        if c.get("strata_mcp"):
            mcp.append(c["strata_mcp"])
        d = c["choices"][0]["delta"]
        content.append(d.get("content") or "")
        reasoning.append(d.get("reasoning_content") or "")
        for tc in d.get("tool_calls") or []:       # streamed calls arrive in pieces: merge them by index
            cur = by_index.setdefault(tc.get("index", len(by_index)), {"id": None, "type": "function",
                                                                        "function": {"name": "", "arguments": ""}})
            cur["id"] = tc.get("id") or cur["id"]
            fn = tc.get("function") or {}
            cur["function"]["name"] += fn.get("name") or ""
            cur["function"]["arguments"] += fn.get("arguments") or ""
        last = c
    calls = [by_index[i] for i in sorted(by_index)]
    if last["choices"][0]["finish_reason"] != "tool_calls":
        calls = [c for c in calls if _is_json(c["function"]["arguments"])]   # a call the output ended inside (#211)
    msg = {"role": "assistant", "content": "".join(content) or None}
    if "".join(reasoning).strip() and "".join(reasoning).strip() == "".join(content).strip():
        reasoning = []              # a turn with no </think> (strata_server/dsml.py): the text is the answer; it is not also reasoning
    if "".join(reasoning):
        msg["reasoning_content"] = "".join(reasoning)
    if calls:
        msg["tool_calls"] = calls
    if mcp:
        msg["strata_mcp"] = mcp
    out = {"id": last["id"], "object": "chat.completion", "created": last["created"], "model": last["model"],
           "choices": [{"index": 0, "message": msg, "finish_reason": last["choices"][0]["finish_reason"]}],
           "usage": last["usage"]}
    if last.get("timings"):
        out["timings"] = last["timings"]
    return out


def structured_chunks(chunks, validator):
    """Buffer structured streams so a client never receives unvalidated content."""
    buffered = []
    heartbeat = time.monotonic()
    try:
        for chunk in chunks:
            progress = chunk is not None and "prompt_progress" in chunk
            if chunk is not None and not progress:
                buffered.append(chunk)      # a progress chunk carries no content: there is nothing in it to validate
            if chunk is None or progress or time.monotonic() - heartbeat >= 1:
                heartbeat = time.monotonic()
                yield chunk if progress else None
        result = openai_collect(buffered)
        choice = result["choices"][0]
        content = validated_json(choice["message"]["content"], validator, choice["finish_reason"])
        yield buffered[0]
        delta = {"content": content}
        if choice["message"].get("reasoning_content"):
            delta["reasoning_content"] = choice["message"]["reasoning_content"]
        yield {**buffered[0], "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
        yield buffered[-1]
    finally:
        chunks.close()


# ------------------------------------------------------------------------------------------------ Anthropic
def anthropic_events(svc: Service, req: dict, ids, thinking, tools, max_new, cancel, force=None):
    mid = "msg_" + uuid.uuid4().hex[:24]
    yield "message_start", {"type": "message_start", "message": {
        "id": mid, "type": "message", "role": "assistant", "model": svc.model_for(req), "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": len(ids), "output_tokens": 0}}}
    index, open_kind, used_tool = -1, None, False

    def close():
        return ("content_block_stop", {"type": "content_block_stop", "index": index})

    streamed, finished = set(), set()              # calls sent piece by piece; those whose final tool_call came (#211)
    for kind, x in svc.run(ids, thinking, tools, max_new, req, cancel, force=force, declared_only=True):
        if kind == "ping":
            yield None
            continue
        if kind == "event":
            ev: Event = x
            if ev.kind == "tool_args":
                yield "content_block_delta", {"type": "content_block_delta", "index": index,
                                              "delta": {"type": "input_json_delta", "partial_json": ev.text}}
                continue
            if ev.kind == "tool_call" and ev.call.id in streamed:
                finished.add(ev.call.id)
                continue
            want = {"reasoning": "thinking", "content": "text", "tool_call": "tool_use", "tool_start": "tool_use"}[ev.kind]
            if ev.kind not in ("tool_call", "tool_start") and not ev.text:
                continue
            if ev.kind == "tool_start":
                streamed.add(ev.call.id)
                used_tool = True
            if open_kind != want or want == "tool_use":
                if open_kind is not None:
                    yield close()
                index += 1
                open_kind = want
                block = {"thinking": {"type": "thinking", "thinking": "", "signature": ""},
                         "text": {"type": "text", "text": ""},
                         "tool_use": {"type": "tool_use", "id": ev.call.id if ev.call else "", "name":
                                      ev.call.name if ev.call else "", "input": {}}}[want]
                yield "content_block_start", {"type": "content_block_start", "index": index, "content_block": block}
            if want == "thinking":
                yield "content_block_delta", {"type": "content_block_delta", "index": index,
                                              "delta": {"type": "thinking_delta", "thinking": ev.text}}
            elif want == "text":
                yield "content_block_delta", {"type": "content_block_delta", "index": index,
                                              "delta": {"type": "text_delta", "text": ev.text}}
            elif ev.kind == "tool_start":
                pass                                # its input follows as tool_args pieces
            else:
                used_tool = True
                yield "content_block_delta", {"type": "content_block_delta", "index": index, "delta": {
                    "type": "input_json_delta", "partial_json": json.dumps(ev.call.arguments, ensure_ascii=False)}}
        else:
            if open_kind is not None:
                yield close()
            stop = "stop_sequence" if x.get("stop_sequence") is not None else \
                "tool_use" if used_tool and streamed <= finished and x["finish"] == "stop" else \
                {"stop": "end_turn", "length": "max_tokens", "cancel": "end_turn"}[x["finish"]]
            # the final counts (message_start could only say the whole prompt): what was read now stays for the next
            # request, a cache write; input_tokens keeps the last token, as 0 makes Claude Code count the prompt twice (#1013)
            reused = max(0, min(x.get("reused") or 0, len(ids) - 1))
            read_now = len(ids) - reused
            yield "message_delta", {"type": "message_delta", "delta": {"stop_reason": stop,
                                                                       "stop_sequence": x.get("stop_sequence")},
                                    "usage": {"input_tokens": 1, "cache_creation_input_tokens": read_now - 1,
                                              "cache_read_input_tokens": reused,
                                              "output_tokens": x["completion_tokens"]}}
            yield "message_stop", {"type": "message_stop"}


def anthropic_collect(events) -> dict:
    msg, blocks = None, []
    for item in events:
        if item is None:                           # a heartbeat
            continue
        name, e = item
        if name == "message_start":
            msg = e["message"]
        elif name == "content_block_start":
            blocks.append(dict(e["content_block"]))
        elif name == "content_block_delta":
            d, b = e["delta"], blocks[-1]
            if d["type"] == "text_delta":
                b["text"] += d["text"]
            elif d["type"] == "thinking_delta":
                b["thinking"] += d["thinking"]
            else:                                  # input_json_delta pieces: parsed when complete
                b["_json"] = b.get("_json", "") + d["partial_json"]
        elif name == "content_block_stop" and blocks and "_json" in blocks[-1]:
            b = blocks[-1]
            try:
                b["input"] = json.loads(b.pop("_json") or "{}")
            except ValueError:                     # a call the output ended inside (#211): it has no input to give
                blocks.pop()
        elif name == "message_delta":
            msg["stop_reason"] = e["delta"]["stop_reason"]
            msg["stop_sequence"] = e["delta"].get("stop_sequence")
            msg["usage"].update(e["usage"])
    for i in range(len(blocks) - 1):    # a turn with no </think> (strata_server/dsml.py): the text is the answer; it is not also a thinking block
        a, b = blocks[i], blocks[i + 1]
        if a.get("type") == "thinking" and b.get("type") == "text" and a.get("thinking", "").strip() == b.get("text", "").strip():
            del blocks[i]
            break
    msg["content"] = blocks
    return msg


# ------------------------------------------------------------------------------------------------ HTTP
CHUNKED_BODY_MAX = 256 << 20                # #893: the most a Transfer-Encoding: chunked body may hold (read into memory)


def body_limit() -> int:
    """The most a request body may hold, in bytes (read into memory): 256 MiB, which is a million-token conversation
    with room to spare; STRATA_MAX_BODY_MIB changes it (0 or an unreadable value: the default).  A larger one is
    answered 413 before it is read (a Content-Length of 10 TB used to be read until the client gave up)."""
    try:
        mib = int(os.environ.get("STRATA_MAX_BODY_MIB", "0"))
    except ValueError:
        mib = 0
    return mib << 20 if mib > 0 else CHUNKED_BODY_MAX


class BadBody(Exception):
    """A request body that cannot be read (a malformed or oversized chunked body): the status and the sentence."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def make_handler(svc: Service):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"                       # SSE ends by closing the connection

        record = None                                       # #332: this request's monitor record, if kept
        watch_done = None                                   # #430 #431: stops this request's disconnect watcher
        cancel_event = None                                 # active generation owned by this handler
        body_read = False                                   # whether a handler took this request's body
        DRAIN_SECONDS = 5                                   # the longest an unread body is read and dropped

        def log_message(self, fmt, *args):
            pass

        def handle_one_request(self):
            self.answer_started = False
            super().handle_one_request()
            self._drain_body()

        def send_response(self, code, message=None):
            self.answer_started = True                 # a malformed-request answer can only replace one not yet begun
            super().send_response(code, message)

        def _chunked(self) -> bool:
            """#893: a body sent as Transfer-Encoding: chunked (a relay or proxy that does not buffer it).  By RFC 9112
            it wins over a Content-Length."""
            te = self.headers.get("Transfer-Encoding", "") or ""
            return te.split(",")[-1].strip().lower() == "chunked"

        def _read_chunked(self, limit: int, keep: bool = True, deadline: float | None = None) -> bytes:
            """Decode a chunked body: at most `limit` bytes (BadBody 413 beyond it - the size is checked before a chunk
            is read, so a huge announced chunk is never allocated), BadBody 400 for a malformed one.  keep=False reads
            and drops it (the drain); `deadline` (monotonic) ends the read early."""
            parts, total = [], 0
            while True:
                if deadline is not None:
                    wait = deadline - time.monotonic()
                    if wait <= 0:
                        return b"".join(parts)
                    self.connection.settimeout(wait)
                line = self.rfile.readline(1025)
                if not line.endswith(b"\n"):
                    raise BadBody(400, "malformed chunked request body")
                try:
                    size = int(line.split(b";", 1)[0].strip(), 16)
                except ValueError:
                    raise BadBody(400, "malformed chunk size in the request body") from None
                if size < 0:
                    raise BadBody(400, "malformed chunk size in the request body")
                if size == 0:
                    for _ in range(64):                       # the trailers, including the final blank line
                        trailer = self.rfile.readline(8193)
                        if trailer == b"\r\n":
                            return b"".join(parts)
                        if not trailer.endswith(b"\r\n"):
                            raise BadBody(400, "malformed or incomplete chunked request trailers")
                    raise BadBody(400, "too many chunked request trailers")
                total += size
                if total > limit:
                    self.close_connection = True
                    raise BadBody(413, f"the request body is larger than {limit >> 20} MiB")
                left = size
                while left > 0:
                    piece = self.rfile.read(min(left, 1 << 20))
                    if not piece:
                        raise BadBody(400, "the chunked request body ended early")
                    left -= len(piece)
                    if keep:
                        parts.append(piece)
                if self.rfile.read(2) != b"\r\n":
                    raise BadBody(400, "malformed chunked request body")

        def _body(self) -> bytes:
            self.body_read = True
            limit = body_limit()
            if self._chunked():
                return self._read_chunked(limit)
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                raise BadBody(400, "invalid Content-Length") from None
            if length < 0:
                raise BadBody(400, "invalid Content-Length")
            if length > limit:
                self.close_connection = True               # the body is not read: the connection ends with the answer
                raise BadBody(413, f"the request body is larger than {limit >> 20} MiB "
                                   "(STRATA_MAX_BODY_MIB raises the limit)")
            return self.rfile.read(length)

        def _drain_body(self):
            """An answer sent before the body was read (a 401, a 403, /load, a method with no handler) must not close
            the connection on unread bytes: the close then sends a reset, and a client that sends its body after the
            headers (http.client, urllib, requests) gets a connection error instead of the answer.  So the body is
            read and dropped here, once, after an answer, in pieces so that its size is never held in memory.  What
            has not arrived DRAIN_SECONDS later is left unread: the limit is time, so that a conversation of many
            megabytes, at any speed the client has, still gets its answer."""
            headers = getattr(self, "headers", None)
            if self.body_read or headers is None:
                return
            if self._chunked():
                try:
                    self._read_chunked(body_limit(), keep=False, deadline=time.monotonic() + self.DRAIN_SECONDS)
                except (BadBody, OSError):
                    self.close_connection = True
                return
            try:
                left = int(headers.get("Content-Length", 0))
            except ValueError:
                return
            if left <= 0:
                return
            deadline = time.monotonic() + self.DRAIN_SECONDS
            try:
                while left > 0 and (wait := deadline - time.monotonic()) > 0:
                    self.connection.settimeout(wait)         # a client that never sends what it announced
                    piece = self.rfile.read1(min(left, 1 << 20))   # one socket read: read() would wait for it all
                    if not piece:
                        break
                    left -= len(piece)
            except OSError:
                pass

        def parse_request(self):
            """Without an API key, every request (any method) first passes the Host check: DNS rebinding protection
            (host_allowed).  With a key a rebinding page cannot authenticate, so the check is skipped: tunnels and
            proxies that pass their own name on keep working."""
            if not super().parse_request():
                return False
            host = self.headers.get("Host")
            if svc.api_key or host_allowed(host, svc.host_names, "*" in svc.allowed_hosts):
                return True
            print(f"[strata] refused a request for Host {host!r} from {self.client_address[0]}: not a name this server "
                  f"answers to (add it to \"allowed_hosts\" in the config or STRATA_ALLOWED_HOSTS, or set an API key)",
                  flush=True)
            self._json(403, {"error": {"type": "forbidden", "message":
                             f"Host {host!r} is not allowed (DNS rebinding protection). Reaching Strata under this "
                             f"name on purpose? Add it to \"allowed_hosts\" in the config (strata-<model>.json) or to "
                             f"the STRATA_ALLOWED_HOSTS environment variable, or set an API key (\"api_key\"), which "
                             f"turns this check off"}})
            return False

        def _foreign_page(self) -> bool:
            """Without an API key, a /v1 POST from a browser page of another site (any site can POST text/plain
            there without a CORS preflight) would burn GPU time: an Origin header must name an allowed page, and
            then the body must be JSON.  No Origin (curl, the SDKs, other servers): any content type, as before."""
            origin = self.headers.get("Origin")
            if svc.api_key or not origin:
                return False
            if not origin_allowed(origin, self.headers.get("Host"), svc.host_names,
                                  [*svc.trusted_origins, *svc.cors_origins]):
                print(f"[strata] refused an API request from the web page {origin!r} (no API key; add its host to "
                      f"\"allowed_hosts\" or its origin to \"cors_origins\" in the config)", flush=True)
                self._json(403, {"error": {"type": "forbidden", "message":
                                 f"web pages of {origin} may not use this server without an API key; set \"api_key\", "
                                 f"or add the page's host to \"allowed_hosts\" (or its origin to \"cors_origins\") "
                                 f"in the config"}})
                return True
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                self._json(415, {"error": {"message": "send application/json"}})
                return True
            return False

        def end_headers(self):
            # HTTP/1.0 without keep-alive: the server closes the connection after every response.  Say so, or a
            # client that pools connections (.NET's SocketsHttpHandler) can put its next request on this socket
            # before the close reaches it, and that request fails with "response ended prematurely" without the
            # server ever reading it (a .NET agent app, 2026-10-03: 25 of ~1000 requests, 1-24 ms after the previous answer).
            self.send_header("Connection", "close")
            super().end_headers()

        def _watch_client(self, cancel: threading.Event) -> None:
            """Watches for connection EOF while this handler owns the request. Socket readiness wakes the watcher
            on disconnect. The HTTP/1.0 request body is fully read here; unexpected extra bytes are not EOF."""
            done = self.watch_done = threading.Event()
            self.cancel_event = cancel
            with svc.status_lock:
                svc.active_cancels.add(cancel)
            sock = self.connection

            def watch():
                while not done.is_set() and not cancel.is_set():
                    try:
                        readable, _, _ = select.select([sock], [], [], 0.5)
                        gone = bool(readable) and sock.recv(1, socket.MSG_PEEK) == b""
                    except (ConnectionError, TimeoutError):
                        gone = True
                    except (OSError, ValueError):            # the socket was closed here: the request has ended
                        return
                    if gone:
                        if not done.is_set():
                            self._note(outcome="disconnected")
                            svc.note_client_disconnect(cancel)
                        return
                    if readable:
                        done.wait(0.5)

            threading.Thread(target=watch, daemon=True, name="strata-client-watch").start()

        def _note(self, **values):
            """#332: what the monitor shows about this request (nothing when the monitor is off)."""
            if self.record is not None:
                with svc.status_lock:
                    self.record.update(values)

        def _cors(self):
            """#321: CORS headers for an API path (/v1/*) and an origin the config lists in cors_origins - nothing
            otherwise, so a browser keeps every other page away from the API, /settings, /unload and the MCP tools."""
            if not svc.cors_origins or not self.path.split("?")[0].startswith("/v1/"):
                return
            origin = (self.headers.get("Origin") or "").rstrip("/")
            if "*" in svc.cors_origins:
                allow = "*"
            elif origin and origin in svc.cors_origins:
                allow = origin
                self.send_header("Vary", "Origin")
            else:
                return
            self.send_header("Access-Control-Allow-Origin", allow)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            # the headers the preflight asks for (SDKs add their own; "*" does not cover Authorization)
            asked = self.headers.get("Access-Control-Request-Headers")
            self.send_header("Access-Control-Allow-Headers",
                             asked or "Authorization, Content-Type, x-api-key, anthropic-version, anthropic-beta")
            self.send_header("Access-Control-Max-Age", "600")

        def do_OPTIONS(self):
            # a CORS preflight: no credentials come with it, so no API key; the headers only for cors_origins
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _json(self, code, obj, headers=None, retry_after=None):
            if self.record is not None:
                with svc.status_lock:
                    self.record["http_status"] = code
                    raw = json.dumps(obj, ensure_ascii=False, indent=2)
                    self.record.update(response=raw[:262144], response_truncated=len(raw) > 262144)
                    for key in ("error", "usage", "timings"):
                        if key in obj:
                            self.record[key] = obj[key]
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            for k, v in (headers or {}).items():
                self.send_header(k, str(v))
            if retry_after is not None:
                self.send_header("Retry-After", str(retry_after))
            self._cors()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            if not svc.api_key:
                return True
            auth = self.headers.get("Authorization", "")
            given = auth[7:].strip() if auth.lower().startswith("bearer ") else self.headers.get("x-api-key", "")
            if key_matches(given, svc.api_key):
                return True
            self._json(401, {"error": {"type": "authentication_error", "message": "missing or wrong API key"}})
            return False

        def _authorized_telemetry(self) -> bool:
            if not svc.api_key and not svc.telemetry_key:
                return True
            auth = self.headers.get("Authorization", "")
            given = auth[7:].strip() if auth.lower().startswith("bearer ") else self.headers.get("x-api-key", "")
            if (svc.api_key and key_matches(given, svc.api_key)) or \
                    (svc.telemetry_key and key_matches(given, svc.telemetry_key)):
                return True
            self._json(401, {"error": {"type": "authentication_error", "message": "missing or wrong telemetry key"}})
            return False

        def do_GET(self):
            path = self.path.split("?")[0].rstrip("/")
            if path.startswith("/fonts/"):
                # the web app's font (Outfit, OFL: strata_server/web/fonts); the page falls back to the system font
                name = path[len("/fonts/"):]
                f = ROOT / "strata_server" / "web" / "fonts" / name
                if "/" in name or "\\" in name or not name.endswith(".woff2") or not f.is_file():
                    self._json(404, {"error": {"message": "not found"}})
                    return
                body = f.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "font/woff2")
                self.send_header("Cache-Control", "max-age=86400")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path.startswith("/web/"):
                # the web app's own files (strata_server/web): styles, script, icon sprite - same origin, no CDN
                name = path[len("/web/"):]
                types = {".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                         ".svg": "image/svg+xml"}
                f = ROOT / "strata_server" / "web" / name
                ext = os.path.splitext(name)[1]
                if "/" in name or "\\" in name or ext not in types or not f.is_file():
                    self._json(404, {"error": {"message": "not found"}})
                    return
                body = f.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", types[ext])
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/metrics/prometheus":
                if self._authorized():
                    body = svc.prometheus().encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                return
            if path == "/metrics":
                from strata_server import prometheus
                if prometheus.wants_prometheus(self.headers.get("Accept", ""), urlsplit(self.path).query):
                    if self._authorized():   # a scraper: the text format, vLLM's names (strata_server/prometheus.py)
                        body = prometheus.render(svc.metrics(), svc.latencies.snapshot()).encode()
                        self.send_response(200)
                        self.send_header("Content-Type", prometheus.CONTENT_TYPE)
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                    return
                if self._authorized():
                    # the last 12 requests; `?requests=all` every one kept (the Monitor's "Show all", issue #35)
                    self._json(200, svc.metrics(all_requests="requests=all" in self.path))
                return
            if path == "/v1/telemetry":
                if self._authorized_telemetry():
                    from strata_common.client import normalize_legacy
                    snapshot = normalize_legacy(svc.metrics(), svc.v1_status())
                    snapshot["source"] = "server-v1"
                    snapshot["work"]["slots_requested"] = getattr(svc, "requested_slots", None)
                    # Keep /metrics' established slot shape intact; enrich only this versioned contract.
                    replica_engines = (getattr(svc.engine, "engines", None)
                                       if hasattr(svc.engine, "replica_view") else None)
                    if replica_engines:
                        groups, live_slots, replica_by_slot = [], [], []
                        group_base = 0
                        all_batched = True
                        for replica, child in enumerate(replica_engines):
                            count = int(getattr(child, "batch", 0) or 0)
                            all_batched = all_batched and count > 0
                            count = max(1, count)
                            child_groups = getattr(child, "slot_group", [])
                            child_live = getattr(child, "slot_live", [])
                            for local in range(count):
                                group = child_groups[local] if local < len(child_groups) else None
                                groups.append(group_base + group if type(group) is int and group >= 0 else None)
                                live_slots.append(child_live[local] if local < len(child_live) else None)
                                replica_by_slot.append(replica)
                            group_base += int(getattr(child, "slot_groups", 0) or 0)
                        snapshot["work"]["groups"] = group_base if all_batched and group_base > 0 else None
                    else:
                        groups = getattr(svc.engine, "slot_group", [])
                        live_slots = getattr(svc.engine, "slot_live", [])
                        replica_by_slot = []
                    for slot in snapshot["work"]["slots"]:
                        i = slot.get("slot")
                        if type(i) is int and 0 <= i < len(groups):
                            slot["group"] = groups[i]
                        if type(i) is int and 0 <= i < len(replica_by_slot):
                            slot["replica"] = replica_by_slot[i]
                        if type(i) is int and 0 <= i < len(live_slots) and live_slots[i]:
                            slot["request_id"] = live_slots[i].get("request_id")
                    slot_by_id = {slot.get("request_id"): slot for slot in snapshot["work"]["slots"]
                                  if slot.get("request_id")}
                    with svc.status_lock:
                        active = list(svc.terminal_requests.values())
                        snapshot["activity"] = {"last_seq": svc.terminal_event_seq,
                                                "events": list(svc.terminal_events)[-32:]}
                        draining = svc.draining
                        admission_reserved = svc.admission_reserved
                        admission_rejected = svc.admission_rejected
                        queue_timeouts = svc.queue_timeouts
                        client_disconnect_cancels = svc.client_disconnect_cancels
                        in_flight_tokens = (sum(max(0, int(st.get("generated") or 0))
                                                for st, _ in svc.live_reqs.values())
                                            if svc.live_reqs else
                                            max(0, int(svc.status.get("generated") or 0))
                                            if svc.status.get("busy") else 0)
                        output_counter = int(svc.totals.get("output_tokens") or 0) + in_flight_tokens
                    snapshot["instance"]["id"] = svc.instance_id
                    snapshot["instance"]["state"] = "draining" if draining else "ready"
                    if getattr(svc, "launch_attestation", None) is not None:
                        snapshot["instance"]["launch_attestation"] = svc.launch_attestation
                    snapshot["rates"]["output_tokens_server_counter"] = output_counter
                    snapshot["work"]["requests_omitted"] = max(0, len(active) - 50)
                    snapshot["work"].update(max_queue=svc.max_queue, queue_timeout_s=svc.queue_timeout_s,
                                            queue_timeouts=queue_timeouts,
                                            client_disconnect_cancels=client_disconnect_cancels,
                                            admission_limit=svc.admission_limit(),
                                            admission_reserved=admission_reserved,
                                            admission_rejected=admission_rejected,
                                            oldest_queue_s=max((max(0, time.time() - row["queued_at"])
                                                                for row in active if row["state"] in ("queued", "waiting")),
                                                               default=0))
                    snapshot["work"]["requests"] = [
                        {**row, "state": slot_by_id[row["id"]]["state"] if row["id"] in slot_by_id else row["state"],
                         "slot": slot_by_id[row["id"]]["slot"] if row["id"] in slot_by_id else None,
                         "group": slot_by_id[row["id"]]["group"] if row["id"] in slot_by_id else None,
                         "replica": slot_by_id[row["id"]]["replica"] if row["id"] in slot_by_id else row.get("replica"),
                         "generated": slot_by_id[row["id"]].get("generated") if row["id"] in slot_by_id else None,
                         "prompt_read": slot_by_id[row["id"]].get("prompt_read") if row["id"] in slot_by_id else None}
                        for row in active[:50]]
                    snapshot["quality"]["request_correlation"] = "reported"
                    batch_sample = (getattr(svc.engine, "terminal_batch_sample", None)
                                    if svc.loaded() and not replica_engines else None)
                    if replica_engines:
                        replica_batches = []
                        for replica, child in enumerate(replica_engines[:16]):
                            row = {"replica": replica}
                            sample = (getattr(child, "terminal_batch_sample", None)
                                      if child.alive() else None)
                            if sample:
                                row.update(batch_sample_windows=sample["windows"],
                                           batch_sample_rows=sample["processed_rows"],
                                           batch_sample_active_rows=sample["active_rows"],
                                           batch_sample_span_s=sample["elapsed_ms"] / 1000.0,
                                           batch_sample_age_s=max(0.0, time.monotonic() -
                                                                  sample["observed_monotonic"]))
                            replica_batches.append(row)
                        snapshot["work"]["replica_batches"] = replica_batches
                    if batch_sample:
                        windows = batch_sample["windows"]
                        snapshot["work"].update(effective_rows_window=batch_sample["processed_rows"] / windows,
                                                batch_active_rows_window=batch_sample["active_rows"] / windows,
                                                batch_sample_windows=windows,
                                                batch_sample_rows=batch_sample["processed_rows"],
                                                batch_sample_active_rows=batch_sample["active_rows"],
                                                batch_sample_span_s=batch_sample["elapsed_ms"] / 1000.0,
                                                batch_sample_age_s=max(0.0, time.monotonic() -
                                                                       batch_sample["observed_monotonic"]))
                        snapshot["quality"]["batch_window_rows"] = "measured"
                        snapshot["quality"]["batch_window_source"] = "engine_bstat"
                    stage_sample = (getattr(svc.engine, "terminal_stage_sample", None)
                                    if svc.loaded() and not replica_engines else None)
                    if replica_engines:
                        replica_stage_samples = []
                        for replica, child in enumerate(replica_engines[:16]):
                            row = {"replica": replica}
                            sample = (getattr(child, "terminal_stage_sample", None)
                                      if child.alive() else None)
                            if sample:
                                row.update(stage_sample_elapsed_us=sample["elapsed_us"],
                                           stage_in_flight_us=sample["in_flight_us"],
                                           stage_sample_age_s=max(0.0, time.monotonic() -
                                                                  sample["observed_monotonic"]))
                            replica_stage_samples.append(row)
                        snapshot["work"]["replica_stage_samples"] = replica_stage_samples
                    if stage_sample:
                        snapshot["work"].update(stage_sample_elapsed_us=stage_sample["elapsed_us"],
                                                stage_in_flight_us=stage_sample["in_flight_us"],
                                                stage_sample_age_s=max(0.0, time.monotonic() -
                                                                       stage_sample["observed_monotonic"]))
                        snapshot["quality"]["stage_in_flight_source"] = "engine_pstat_host_span"
                    indices = list(getattr(svc, "gpu_indices", []) or [])
                    visibility = list(getattr(svc, "terminal_visibility_order", []) or [])
                    reported = (list(getattr(svc.engine, "terminal_stages", []) or [])
                                if svc.loaded() and not replica_engines else [])
                    stages = [{**stage, "device_index":
                               visibility[stage["engine_ordinal"]]
                               if stage["engine_ordinal"] < len(visibility) else None}
                              for stage in reported]
                    snapshot["topology"]["selected_devices"] = indices if replica_engines else visibility or indices
                    replica_devices = getattr(svc, "replica_gpu_groups", []) if replica_engines else []
                    snapshot["topology"]["replica_devices"] = [
                        {"replica": i, "devices": list(devices)}
                        for i, devices in enumerate(replica_devices[:16])]
                    replica_stages = []
                    for replica, child in enumerate((replica_engines or [])[:16]):
                        child_reported = (list(getattr(child, "terminal_stages", []) or [])
                                          if child.alive() else [])
                        spawn = getattr(child, "spawn", None)
                        child_env = spawn[4] if isinstance(spawn, tuple) and len(spawn) > 4 else None
                        child_visibility = visible_device_order(child_env, getattr(svc, "backend", None))
                        configured_cards = (set(replica_devices[replica])
                                            if replica < len(replica_devices) else set())
                        child_stages = []
                        for stage in child_reported:
                            ordinal = stage["engine_ordinal"]
                            card = (child_visibility[ordinal]
                                    if type(ordinal) is int and 0 <= ordinal < len(child_visibility) else None)
                            child_stages.append({**stage, "device_index": card if card in configured_cards else None})
                        replica_stages.append({"replica": replica, "stages": child_stages})
                    snapshot["topology"]["replica_stages"] = replica_stages
                    snapshot["topology"]["mode"] = ("layer split (engine reported)" if stages else
                                                     getattr(svc, "terminal_topology_mode", None))
                    snapshot["topology"]["configured_split"] = getattr(svc, "terminal_configured_split", None)
                    snapshot["topology"]["stages"] = stages
                    snapshot["topology"]["resolved"] = bool(stages) and all(
                        stage["device_index"] is not None for stage in stages)
                    snapshot["topology"]["source"] = "engine_start_log" if stages else None
                    snapshot["quality"]["placement"] = ("reported" if snapshot["topology"]["resolved"] else
                                                         "partial" if stages else "unavailable")
                    present = {d.get("index") for d in snapshot["devices"]}
                    for index in indices:
                        if index not in present:
                            snapshot["devices"].append({"index": index, "name": None, "mem_used": None,
                                                        "mem_total": None, "util": None, "temp": None, "power": None})
                    self._json(200, snapshot)
                return
            if path == "/api/context" and svc.api_monitor:
                if self._authorized():
                    cid = parse_qs(urlsplit(self.path).query).get("id", [None])[0]
                    if cid is None:
                        self._json(200, svc.context_view())
                    elif not re.fullmatch(r"[a-z]+[0-9]*", cid):     # main, slot<b>, parked<serial>
                        self._json(400, {"error": {"message": "invalid conversation id"}})
                    else:
                        self._json(*svc.context_window(cid))
                return
            if path == "/v1/memory":                       # the Memory panel (docs/MEMORY_MANAGER_PLAN.md, M0)
                if self._authorized():
                    self._json(200, svc.memory_view())
                return
            if path == "/api/requests" and svc.api_monitor:
                if self._authorized():
                    request_id = parse_qs(urlsplit(self.path).query).get("id", [None])[0]
                    records = svc.request_records(request_id)
                    self._json(404 if records is None else 200,
                               {"error": {"message": "request no longer retained"}} if records is None else
                               records if request_id else {"requests": records, "retention": 100, "persistent": False,
                                                          "loaded": svc.loaded(), "auto_load": hasattr(svc.engine, "restart")})
                return
            if path == "/settings":
                if self._authorized():
                    self._json(200, {"shared": bool(svc.shared), "defaults": svc.shared})
                return
            if path == "/config":                            # #564: the Settings view's keys of the run config
                if self._authorized():
                    self._config_get()
                return
            if path == "/mcp":
                # the MCP servers, their state and tools (the web app's switch and Monitor card)
                if self._authorized():
                    self._json(200, svc.mcp.status() if svc.mcp else {"servers": [], "tools": 0})
                return
            if path == "" or (path == "/api-monitor" and svc.api_monitor):
                body = (ROOT / "strata_server" / "web" / ("monitor.html" if path else "index.html")).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path in ("/health", "/api/health"):
                self._json(200, {"status": "ok", "max_context": svc.reported_ctx(), "model": svc.model,
                                 "images": svc.vision is not None, "api_key": bool(svc.api_key),
                                 "image_generation": svc.images is not None,
                                 "loaded": svc.loaded(), "service": "strata",
                                 **({"api_monitor": True} if svc.api_monitor else {})})
            elif path.startswith("/v1/images/files/"):
                # response_format url: the name (32 random hex digits) is the access, as an OpenAI image URL is
                f = svc.images.file(path[len("/v1/images/files/"):]) if svc.images is not None else None
                if f is None:
                    self._json(404, {"error": {"message": "not found (the server keeps the newest images only)"}})
                    return
                body = f.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Cache-Control", "private, max-age=86400")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/status":
                if not self._authorized():                  # #212: it shows the end of the last answer
                    return
                with svc.status_lock:
                    s = dict(svc.status)
                now = time.time()
                if s.get("busy"):
                    s["elapsed_s"] = round(now - s["started"], 1)
                    if s.get("first_token"):
                        s["tokens_per_s"] = round(svc._tok_s(), 1)
                        s["tokens_per_s_mean"] = round(svc._tok_s_mean(), 1)
                for k in ("started", "first_token"):
                    s.pop(k, None)
                self._json(200, s)
            elif path in ("/v1/models", "/models"):
                if self._authorized():
                    loaded = svc.loaded()
                    model = {"id": svc.model, "object": "model", "status": {"value": "loaded"},
                             "meta": {"n_ctx": svc.reported_ctx()},
                             "architecture": {"input_modalities": ["text", "image"] if svc.vision is not None else ["text"],
                                              "output_modalities": ["text"]}}
                    if not loaded and (svc.idle_unload_s or getattr(svc.engine, "unloaded", False)):
                        model["status"] = {"value": "unloaded"}   # like llama-server's router: listed, loads on use
                        loaded = True
                    if svc.aliases:                       # #297: the aliases, and each one listed under its own id
                        model["aliases"] = list(svc.aliases)
                    data = [model, *({**model, "id": x, "alias_of": svc.model} for x in svc.aliases)]
                    data = data if loaded else []
                    if svc.images is not None:            # the image model: POST /v1/images/generations
                        data.append({"id": svc.images.model, "object": "model", "status": {"value": "loaded"},
                                     "architecture": {"input_modalities": ["text"], "output_modalities": ["image"]}})
                    self._json(200, {"object": "list", "data": data})
            elif path == "/props":
                if self._authorized():
                    self._props()
            elif path == "/slots":
                if self._authorized():
                    loaded = not hasattr(svc.engine, "alive") or svc.engine.alive()
                    with svc.status_lock:
                        busy = bool(svc.status.get("busy"))
                        # llama.cpp's name for what the slot's context holds: the running request's prompt size,
                        # kept after it ends (the conversation cache carries it on).  A front-end's context meter
                        # divides it by n_ctx; without the key it reads 0 % against Strata forever (#562).
                        in_use = int(svc.status.get("prompt_tokens") or 0)
                    if getattr(svc.engine, "batch", 0) and hasattr(svc.engine, "slots_view"):
                        # --batch: one entry per slot (the same view /metrics has), not one for the whole engine
                        n_ctx = svc.reported_ctx()
                        slots = [{"id": s["slot"], "n_ctx": n_ctx, "is_processing": s["state"] != "idle",
                                  "n_prompt_tokens": int(s.get("prompt_tokens") or s.get("held_tokens") or 0)}
                                 for s in svc.engine.slots_view()]
                    else:
                        slots = [{"id": 0, "n_ctx": svc.reported_ctx(), "is_processing": busy,
                                  "n_prompt_tokens": in_use}]
                    self._json(200, slots if loaded else [])
            elif path == "/v1/status":
                if self._authorized():
                    self._json(200, svc.v1_status())
            else:
                self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            try:
                self._post()
            except BadBody as e:                              # #893: a malformed or oversized chunked body
                self._json(e.status, {"error": {"type": "invalid_request_error", "message": str(e)}})

        def _post(self):
            if not self._authorized():
                return
            path = self.path.split("?")[0].rstrip("/")   # issue #55: Claude Code posts /v1/messages?beta=true
            path = V1_V1_ALIASES.get(path, path)         # an Anthropic client given ".../v1" as its base posts /v1/v1/messages
            if path.startswith("/v1/") and self._foreign_page():
                return
            if path == "/settings":
                self._settings()
                return
            if path == "/config":
                self._config_post()
                return
            if path == "/v1/memory":                       # the Memory panel's controls (MEMORY_MANAGER_PLAN.md, M6)
                body = self._body()
                if self._own_page("memory settings can be changed"):
                    self._json(*svc.change_memory_settings(body))
                return
            if path in ("/unload", "/load") and not self._control_body():
                return
            # JSON from Strata's own page only, as /settings: else a plain form POST from any site unloads the model
            if path in ("/unload", "/load") and not self._own_page("the model can be loaded or unloaded"):
                return
            # the same for the slot files: else any site could overwrite a saved conversation or replace the live one
            if path.startswith("/slots/") and not self._own_page("conversations can be saved or restored"):
                return
            if path == "/unload":                            # give the GPU back now (between requests)
                try:
                    wait_s = unload_wait(parse_qs(urlsplit(self.path).query).get("wait", [None])[0],
                                         svc.UNLOAD_WAIT_S)
                    r = svc.unload_waiting(wait_s) if wait_s else svc.unload()
                except engine_client.EngineStuck as e:
                    self._json(503, {"error": {"type": "server_error", "message": str(e)}})
                    return
                self._json(409 if r == "busy" else 200, {"status": r})
                return
            if path == "/load":                              # load now, e.g. ahead of a request
                try:
                    svc.load()
                    self._json(200, {"status": "loaded"})
                except engine_client.GpuBusy as e:
                    ra = getattr(e, "retry_after", None)
                    self._json(503, {"error": {"type": "server_error", "message": str(e)}},
                               headers={"Retry-After": int(ra)} if ra else None)
                return
            reserved = False
            try:
                if path in ("/v1/chat/completions", "/v1/messages", "/v1/responses"):
                    reserved = svc.reserve_inference()
                    if reserved is None:
                        error = {"type": "server_error", "code": "server_draining",
                                 "message": "this instance is draining; retry against the next instance"}
                        body = responses_error_body(error["message"], "server_error", code=error["code"]) \
                            if path == "/v1/responses" else {"error": error}
                        self._json(503, body, retry_after=1)
                        return
                    if not reserved:
                        error = {"type": "rate_limit_error", "code": "admission_full",
                                 "message": "the inference admission limit is full; retry later"}
                        body = responses_error_body(error["message"], "server_error", code=error["code"]) \
                            if path == "/v1/responses" else {"error": error}
                        self._json(429, body, retry_after=1)
                        return
                req = json.loads(self._body() or b"{}")
                if not isinstance(req, dict):
                    raise ValueError("send a JSON object")
                if path.startswith("/v1/responses/"):        # retrieve/delete/cancel/compact: nothing is stored
                    self._json(404, responses_error_body("this server keeps no responses (stateless); send the "
                                                         "whole conversation to POST /v1/responses", code="not_found"))
                    return
                if path in ("/v1/load", "/v1/unload"):
                    if not self._own_page("the model can be loaded or unloaded"):
                        return
                    if req.get("model") not in (None, svc.model):
                        self._json(404, {"error": {"message": "model not found"}})
                        return
                    if path == "/v1/unload":
                        wait_s = unload_wait(parse_qs(urlsplit(self.path).query).get("wait", [None])[0],
                                             svc.UNLOAD_WAIT_S)
                        result = svc.unload_waiting(wait_s) if wait_s else svc.unload()
                        if result == "busy":
                            raise ModelBusy("a request is running or queued")
                    else:
                        if not svc.fifo.acquire(blocking=False):
                            raise ModelBusy("a request is running or queued")
                        try:
                            with svc.status_lock:
                                if svc.status.get("busy") or svc.status.get("queued"):
                                    raise ModelBusy("a request is running or queued")
                            svc.ensure_loaded()
                            svc.last_request_at = time.time()
                        finally:
                            svc.fifo.release()
                        result = "loaded"
                    self._json(200, {"status": result, **svc.v1_status()})
                    return
                if path == "/v1/vram":                       # #533: give VRAM back to other programs, or take it
                    if not self._own_page("the VRAM reserve can be changed"):
                        return
                    r = req.get("reserve_mib")
                    if r is not None and (isinstance(r, bool) or not isinstance(r, int) or r < 0):
                        raise ValueError("reserve_mib: a whole number of MiB (0 or more), or null for the start's")
                    try:
                        self._json(200, svc.vram(r))
                    except engine_client.EngineDied as e:
                        self._json(503, {"error": {"type": "server_error", "message": str(e)}})
                    return
                if path in ("/v1/chat/completions", "/v1/messages", "/v1/responses"):
                    self.record = svc.begin_request(path, req)
                if path == "/v1/responses":
                    self._responses(req)
                elif path.startswith("/slots/"):
                    query = parse_qs(self.path.partition("?")[2])
                    try:
                        code, body = svc.slot_action(path[len("/slots/"):], (query.get("action") or [""])[0],
                                                     req.get("filename"))
                    except engine_client.EngineDied as e:                  # EngineSilent included: the engine was ended
                        code, body = 500, {"error": {"code": 500, "message": str(e) + "; the next request starts "
                                                     "the engine again", "type": "server_error"}}
                    self._json(code, body)
                elif path == "/v1/chat/completions":
                    self._openai(req)
                elif path == "/v1/messages":
                    self._anthropic(req)
                elif path == "/v1/messages/count_tokens":
                    self._count_tokens(req)
                elif path == "/v1/images/generations":
                    self._images(req)
                else:
                    self._json(404, {"error": {"message": "not found"}})
            except engine_client.EngineNonFinite as e:                    # #879, twice: not the request's fault
                print(f"[strata] 503: {e}", flush=True)
                self._json(503, responses_error_body(str(e), "server_error") if path == "/v1/responses"
                           else {"error": {"type": "server_error", "message": str(e)}})
            except ValueError as e:
                print(f"[strata] 400 invalid request: {e}", flush=True)
                if path == "/v1/responses":
                    self._json(400, responses_error_body(str(e)))
                else:
                    self._json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
            except ModelBusy as e:
                self._json(409, {"error": {"type": "model_busy", "message": str(e)}})
            except engine_client.QueueWaitExpired as e:
                error = {"type": "rate_limit_error", "code": "queue_wait_timeout", "message": str(e)}
                self._note(error=error)
                body = responses_error_body(str(e), "server_error", code="queue_wait_timeout") \
                    if path == "/v1/responses" else {"error": error}
                self._json(429, body, retry_after=1)
            except StructuredOutputError as e:
                self._json(502, {"error": {"type": "structured_output_failed", "code": "structured_output_failed",
                                          "message": str(e)}})
            except (engine_client.GpuBusy, EngineStarting) as e:
                ra = getattr(e, "retry_after", None)       # #1997 / #1994: a refusal that is worth retrying
                self._json(503, {"error": {"type": "server_error", "message": str(e)}},
                           headers={"Retry-After": int(ra)} if ra else None)
            except (TypeError, KeyError, AttributeError, IndexError, UnicodeError) as e:
                # a body that is JSON but the wrong shape ("messages": 5, a content part that is a number): a 400 that
                # says so, with the traceback in the log, instead of a dropped connection - unless the answer began
                if getattr(self, "answer_started", False):
                    raise
                import traceback
                print(f"[strata] 400 malformed request ({type(e).__name__}: {e}):\n" + traceback.format_exc(),
                      flush=True)
                body = {"error": {"type": "invalid_request_error",
                                  "message": f"the request is malformed ({type(e).__name__}: {e})"}}
                self._json(400, responses_error_body(body["error"]["message"]) if path == "/v1/responses" else body)
            except engine_client.EngineDied as e:                          # before the answer started (not streamed)
                self._json(503, {"error": {"type": "server_error", "message": f"{e}; the next request restarts it"}})
            except engine_client.EngineStuck as e:   # an unload or restart that could not end the engine
                self._json(503, {"error": {"type": "server_error", "message": str(e)}})

            except ClientGone:                               # #1999: the client left while the request was prepared
                self._note(outcome="disconnected")
            except OSError:
                self._note(outcome="disconnected")
                raise                                        # as before #332: the server's own handling
            finally:
                if reserved:
                    svc.release_inference()
                svc.drop_embeddings()                        # the images' file of a request that never got to run()
                if self.watch_done is not None:
                    self.watch_done.set()
                if self.cancel_event is not None:
                    with svc.status_lock:
                        svc.active_cancels.discard(self.cancel_event)
                record = self.record
                if record is not None:
                    with svc.status_lock:
                        record["wallclock_s"] = round(time.perf_counter() - record["_clock"], 3)
                        record["finished_at"] = time.time()
                        record["state"] = "error" if record.get("error") else record.get("outcome", "completed")
                    svc.request_trace.record = None

        def _images(self, req):
            """POST /v1/images/generations (strata_server/images.py): one sd-cli job in the GPU's turn."""
            gen = svc.images
            if gen is None:
                self._json(404, {"error": {"type": "invalid_request_error", "code": "images_not_set_up",
                                           "message": "image generation is not set up on this server (run setup "
                                                      "with --images qwen-image)"}})
                return
            job = images_api.parse_request(req, gen.cfg)        # ImageRequestError (a ValueError): 400
            svc.last_request_at = time.time()
            cancel = threading.Event()
            self._watch_client(cancel)
            print(f"[strata] image: {job['width']}x{job['height']}, n={job['n']}, {job['steps']} steps, "
                  f"seed {job['seed']}", flush=True)
            try:
                with images_api.gpu_turn(svc, gen, job, cancel) as policy:
                    images, figures = gen.run(job, cancel)
            except images_api.ImageJobFailed as e:
                self._json(500, {"error": {"type": "server_error", "code": "image_generation_failed",
                                           "message": str(e)}})
                return
            figures["gpu_policy"] = policy
            print(f"[strata] image done in {figures['seconds']:.1f} s ({policy})", flush=True)
            host = self.headers.get("Host") or f"{self.server.server_address[0]}:{self.server.server_address[1]}"
            scheme = "https" if (self.headers.get("X-Forwarded-Proto") or "").lower() == "https" else "http"
            self._json(200, gen.response(images, job, figures, f"{scheme}://{host}"))

        def _props(self):
            model = parse_qs(urlsplit(self.path).query).get("model", [svc.model])[0]
            if model not in svc.model_names():
                self._json(404, {"error": {"message": "model not found"}})
                return
            if not svc.loaded() and not getattr(svc.engine, "unloaded", False):
                self._json(503, {"error": {"message": "the engine is not running"}})
                return
            defaults = {**model_card_sampling(svc.model_arch, True), **svc.sampling_defaults, **svc.shared}
            names = {"repetition_penalty": "repeat_penalty", "penalty_last_n": "repeat_last_n"}
            params = {names.get(k, k): v for k, v in defaults.items()
                      if k in ("temperature", "top_p", "top_k", "min_p", "seed", "repetition_penalty",
                               "presence_penalty", "frequency_penalty", "penalty_last_n")}
            params["n_predict"] = svc.shared.get("max_tokens", -1)
            props = {"default_generation_settings": {"n_ctx": svc.reported_ctx(), "params": params},
                     "total_slots": max(1, int(getattr(svc.engine, "batch", 0) or 0)),   # #1004: the batch slots
                     "model_alias": svc.model, "chat_template": svc.template.source,
                     "chat_template_caps": svc.template.caps,
                     "modalities": {"vision": svc.vision is not None}, "models_autoload": hasattr(svc.engine, "restart"),
                     "is_sleeping": not svc.loaded()}
            if getattr(svc.engine, "model_path", None):
                props["model_path"] = svc.engine.model_path
            version = getattr(svc.engine, "info", {}).get("version")
            if version:
                props["build_info"] = "Strata " + str(version)
            self._json(200, props)

        def _control_body(self) -> bool:
            """Consume the unused control body before replying/closing (Windows otherwise sends a TCP reset)."""
            if self._chunked():
                self.body_read = True
                try:
                    self.connection.settimeout(2.0)
                    self._read_chunked(65536, keep=False)
                except BadBody as e:
                    self._json(e.status, {"error": {"message": f"control request: {e}"}})
                    return False
                except OSError:
                    self._json(400, {"error": {"message": "incomplete control request body"}})
                    return False
                return True
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._json(400, {"error": {"message": "invalid Content-Length"}})
                return False
            if not 0 <= length <= 65536:
                self._json(413, {"error": {"message": "control request body is limited to 64 KiB"}})
                return False
            timeout = self.connection.gettimeout()
            try:
                self.connection.settimeout(2.0)
                complete = len(self.rfile.read(length)) == length
            except OSError:
                complete = False
            finally:
                self.connection.settimeout(timeout)
                self.body_read = True                    # nothing left for the drain to wait for (#594)
            if not complete:
                self._json(400, {"error": {"message": "incomplete control request body"}})
            return complete

        def _own_page(self, what) -> bool:
            """Only JSON (a form or a "simple" cross-site request can't send it without a CORS preflight, which this
            server never grants) and no foreign Origin: a web page elsewhere must not change settings or run tools."""
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                self._json(415, {"error": {"message": "send application/json"}})
                return False
            if self._foreign_origin():
                self._json(403, {"error": {"message": f"{what} only from Strata's own page (or an origin in the "
                                                      f"config's trusted_origins)"}})
                return False
            return True

        def _config_get(self):
            if not svc.config_path:
                self._json(404, {"error": {"message": "this server was started without a run config"}})
                return
            try:
                cfg = runconfig.load(svc.config_path)
            except (OSError, ValueError) as e:
                self._json(500, {"error": {"type": "server_error", "message": f"the run config cannot be read: {e}"}})
                return
            self._json(200, runconfig.view(cfg, svc.config_path))

        def _config_post(self):
            """#564: change a few documented keys of the run config - JSON from Strata's own page only, as
            /settings (the key is checked before); every other key of the file stays as it is."""
            body = self._body()
            if not self._own_page("the run config can be changed"):
                return
            if not svc.config_path:
                self._json(404, {"error": {"message": "this server was started without a run config"}})
                return
            try:
                req = json.loads(body or b"{}")
                with svc.config_lock:
                    cfg = runconfig.load(svc.config_path)
                    new, changed = runconfig.apply(cfg, req.get("set") if isinstance(req, dict) else None)
                    if changed:
                        bak = runconfig.save(svc.config_path, new)
            except ValueError as e:
                self._json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
                return
            except OSError as e:
                self._json(500, {"error": {"type": "server_error", "message": f"the run config cannot be written: {e}"}})
                return
            if changed:
                print(f"[strata] the Settings view changed {', '.join(changed)} in {Path(svc.config_path).name} "
                      f"(the earlier file: {bak.name}); used from the next start", flush=True)
            self._json(200, {**runconfig.view(new, svc.config_path), "changed": changed})
        def _foreign_origin(self) -> bool:
            """A web page of another origin sent this.  The Origin must be this server's own address (host and port),
            or an origin the config trusts (trusted_origins: the web app behind a reverse proxy or tunnel); no Origin
            is a client that is no browser.  Headers a proxy adds (X-Forwarded-*, CF-Ray, CF-Connecting-IP) prove
            nothing about the page that sent the request, so they open nothing."""
            origin = (self.headers.get("Origin") or "").rstrip("/")
            return bool(origin) and origin.split("://", 1)[-1] != self.headers.get("Host", "") and \
                origin not in svc.trusted_origins

        def _no_local_images(self, messages):
            """A web page of another origin must not have a file on this computer read as an image.  Without an
            api_key, _foreign_page refuses such a page unless cors_origins lets it in - and a page cors_origins lets
            in (["*"]: every page) could name any file and, CORS allowing, read what the model says about it.  data:
            and http(s) images are the page's own to send; files come from Strata's own page, a trusted origin or a
            client that is no browser.  ValueError (a 400)."""
            if self._foreign_origin() and any(not src.startswith(("data:", "http://", "https://"))
                                              for src in images_of(messages)):
                raise ValueError("a web page of another origin cannot have a file on this computer read as an "
                                 "image: send it as a data: URL (or add the page's origin to the config's "
                                 "trusted_origins)")

        def _settings(self):
            # They change what every client gets, so only the app's own page may set them
            body = self._body()
            if not self._own_page("settings can be changed"):
                return
            try:
                req = json.loads(body or b"{}")
                if not isinstance(req, dict) or "defaults" not in req:
                    self._json(400, {"error": {"type": "invalid_request_error",
                                              "message": 'the body must be {"defaults": {...}} to set them, '
                                                         'or {"defaults": null} to clear them'}})
                    return
                shared = svc.set_shared(req["defaults"])
            except ValueError as e:
                self._json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
                return
            print("[strata] other apps now use the Chat settings: " + ", ".join(f"{k}={v}" for k, v in shared.items())
                  if shared else "[strata] other apps use their own settings again", flush=True)
            self._json(200, {"shared": bool(shared), "defaults": shared})

        def _sse(self):
            self._note(http_status=200)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")   # nginx and similar proxies pass each event at once
            self._cors()
            self.end_headers()

        def _capture(self, items, api):
            """Retain bounded input/output for the monitor without changing the API response (the items as they
            are when the monitor is off)."""
            if self.record is None:
                return items
            return self._captured(items, api)

        def _captured(self, items, api):
            try:
                for item in items:
                    if item is not None:
                        if api == "openai":
                            delta = item["choices"][0]["delta"]
                            content, reasoning = delta.get("content", ""), delta.get("reasoning_content", "")
                            usage, timings = item.get("usage"), item.get("timings")
                        elif api == "responses":
                            kind = item["type"]
                            content = item["delta"] if kind == "response.output_text.delta" else ""
                            reasoning = item["delta"] if kind == "response.reasoning_text.delta" else ""
                            usage, timings = (item.get("response") or {}).get("usage"), None
                        else:
                            _, event = item
                            delta = event.get("delta", {})
                            content, reasoning = delta.get("text", ""), delta.get("thinking", "")
                            usage, timings = event.get("usage"), None
                        with svc.status_lock:
                            for name, value in (("output", content), ("reasoning", reasoning)):
                                if value:
                                    combined = self.record[name] + value
                                    self.record[name] = combined[:262144]
                                    self.record[name + "_truncated"] |= len(combined) > 262144
                            if usage:
                                self.record["usage"] = usage
                            if timings:
                                self.record["timings"] = timings
                    yield item
            finally:
                items.close()

        def _openai(self, req):
            req = svc.with_shared(req, "openai")
            bias_api.normalize(req.get("logit_bias"), len(getattr(svc.tok, "tokens", ())) or None)
            messages, tools, kw = openai_to_messages(req)
            require_messages(messages)                # an empty conversation is a 400, as OpenAI answers it, not a reply to nothing
            self._no_local_images(messages)
            if tool_choice_of(req.get("tool_choice"))[0] == "none":   # as the Responses route: no tools are offered
                tools = None
            force = svc.force_text(forced_call(req.get("tool_choice"), tools))      # a bad value is a 400 before anything is sent
            messages, validator = prepare_format(req.get("response_format"), messages)
            if validator is not None and (tools or req.get("strata_mcp")):
                raise ValueError("structured response_format with tools/MCP is not supported")
            svc.load()
            max_req = max_new = int(req.get("max_completion_tokens") or req.get("max_tokens") or 0)   # 0/-1: the rest
            bias_api.validate_request(req, svc.engine, len(getattr(svc.tok, "tokens", ())) or None)
            use_mcp = req.get("strata_mcp") is True and svc.mcp is not None      # the web app's opt-in (strata_server/mcp.py)
            own = {t.get("name") for t in tools or [] if isinstance(t, dict)}   # #592: a second line of defence
            if use_mcp:
                if not self._own_page("MCP tools can be used"):   # tools run with the user's rights on this PC
                    return
                svc.mcp.wait(10)                                  # servers still starting (only right after start)
                extra = svc.mcp.template_tools(exclude=own)       # the request's own tools win a name clash
                use_mcp = bool(extra)
                tools = (tools or []) + extra or None
            if force and use_mcp:
                raise ValueError("a forced tool_choice with MCP tools is not supported")
            svc.reasoning_budget(req)                         # a bad value is a 400 before anything is sent
            stop_strings(req)                                 # ... and so is a bad stop / stop_sequences
            check_request_sampling(req)                       # ... and a sampling field of the wrong type
            cancel = threading.Event()
            self._watch_client(cancel)                       # #430 #431 (#1999: before prepare, which can take long)
            ids, thinking, max_new = svc.prepare(messages, tools, kw, max_new, force=force, req=req, cancel=cancel)
            _debug_req("openai", req, messages, tools, max_new, thinking, len(ids))
            run = run_with_mcp(svc, svc.mcp, messages, tools, kw, ids, thinking, max_new, max_req, req, cancel,
                               {t["name"] for t in extra}) if use_mcp else None
            chunks = openai_chunks(svc, req, ids, thinking, tools, max_new, cancel, run=run, force=force)
            if validator is not None:
                chunks = structured_chunks(chunks, validator)
            chunks = self._capture(chunks, "openai")
            if not req.get("stream"):
                return self._json(200, openai_collect(chunks))
            self._sse()
            try:
                for c in chunks:
                    if c is None:
                        self.wfile.write(b": keep-alive\n\n")      # an SSE comment: clients ignore it
                    else:
                        self.wfile.write(b"data: " + json.dumps(c, ensure_ascii=False).encode() + b"\n\n")
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
            except OSError:
                self._note(outcome="disconnected")
                svc.note_client_disconnect(cancel)           # client went away: stop the engine
                chunks.close()
            except engine_client.EngineDied as e:   # mid-stream: say so, then end the stream properly
                err = {"error": {"type": "server_error", "message": f"{e}; the next request restarts it"}}
                self._note(error=err["error"])
                self.wfile.write(b"data: " + json.dumps(err).encode() + b"\n\ndata: [DONE]\n\n")
            except engine_client.QueueWaitExpired as e:
                err = {"error": {"type": "rate_limit_error", "code": "queue_wait_timeout", "message": str(e)}}
                self._note(error=err["error"])
                self.wfile.write(b"data: " + json.dumps(err).encode() + b"\n\ndata: [DONE]\n\n")
            except StructuredOutputError as e:
                err = {"error": {"type": "structured_output_failed", "code": "structured_output_failed", "message": str(e)}}
                self._note(error=err["error"])
                self.wfile.write(b"data: " + json.dumps(err).encode() + b"\n\ndata: [DONE]\n\n")
            except ValueError as e:                          # the engine's ERR after the stream started: the
                err = {"error": {"type": "server_error", "message": str(e)}}   # headers are sent, so no 400 now
                self._note(error=err["error"])
                self.wfile.write(b"data: " + json.dumps(err).encode() + b"\n\ndata: [DONE]\n\n")

        def _responses_ready(self, req, events):
            """Send a finished Responses result that did not run on the engine."""
            if not req.get("stream"):
                return self._json(200, responses_api.collect(events))
            self._sse()
            try:
                for e in events:
                    self.wfile.write(f"event: {e['type']}\n".encode() + b"data: " +
                                     json.dumps(e, ensure_ascii=False).encode() + b"\n\n")
                    self.wfile.flush()
            except OSError:
                self._note(outcome="disconnected")

        def _responses(self, req):
            """#451: POST /v1/responses - OpenAI's Responses API, stateless (strata_server/responses.py), on the chat path.
            Errors use the Responses format; once the stream has started they arrive as a response.failed event."""
            try:
                title = (responses_api.thread_title_events(req, svc.model_for(req))
                         if svc.codex_thread_titles else None)
            except ResponsesError as e:
                return self._json(e.status, e.body())
            if title is not None:
                return self._responses_ready(req, title)
            cancel = threading.Event()
            self._watch_client(cancel)                       # #430 #431 (#1999: before prepare, which can take long)
            try:
                ids, thinking, tools, max_new, asm, validator, req = self._responses_prepare(req, cancel)
            except ClientGone:
                self._note(outcome="disconnected")
                return
            except ResponsesError as e:
                return self._json(e.status, e.body())
            except ModelBusy as e:
                return self._json(409, responses_error_body(str(e), "server_error", code="model_busy"))
            except (engine_client.GpuBusy, EngineStarting, engine_client.EngineStuck, engine_client.EngineDied) as e:
                return self._json(503, responses_error_body(str(e), "server_error", code="server_error"))

            def check(text, finish):
                return validated_json(text, validator, finish)

            def events():
                yield from asm.start()
                done = None
                for kind, x in svc.run(ids, thinking, tools, max_new, req, cancel):
                    if kind == "ping":
                        yield None
                    elif kind == "event":
                        yield from asm.feed(x)
                    elif kind == "done":
                        done = x
                if done is not None and done["finish"] != "cancel" and not cancel.is_set():
                    yield from asm.finish(done, check)
            items = self._capture(events(), "responses")
            if not req.get("stream"):
                try:
                    result = responses_api.collect(items)
                except StructuredOutputError as e:
                    return self._json(502, responses_error_body(str(e), "server_error",
                                                                code="structured_output_failed"))
                except engine_client.QueueWaitExpired as e:
                    return self._json(429, responses_error_body(str(e), "server_error",
                                                                code="queue_wait_timeout"), retry_after=1)
                except engine_client.EngineDied as e:
                    return self._json(503, responses_error_body(f"{e}; the next request restarts it", "server_error",
                                                                code="server_error"))
                except engine_client.EngineNonFinite as e:                 # #879, twice: not the request's fault
                    return self._json(503, responses_error_body(str(e), "server_error", code="server_error"))
                except ValueError as e:                      # the engine's ERR line
                    return self._json(500, responses_error_body(str(e), "server_error", code="server_error"))
                return self._json(200, result)
            self._sse()
            last = time.monotonic()

            def send(e):
                self.wfile.write(f"event: {e['type']}\n".encode() + b"data: " +
                                 json.dumps(e, ensure_ascii=False).encode() + b"\n\n")

            try:
                for e in items:
                    if e is None:
                        # Codex's idle timeout (5 minutes by default) counts events, not SSE comments: while a long
                        # prompt is read, a response.in_progress every 15 s tells it the server is still working
                        if time.monotonic() - last >= 15:
                            send(asm.in_progress())
                            last = time.monotonic()
                        else:
                            self.wfile.write(b": keep-alive\n\n")
                    else:
                        send(e)
                        last = time.monotonic()
                    self.wfile.flush()
            except OSError:
                self._note(outcome="disconnected")
                svc.note_client_disconnect(cancel)           # client went away: stop the engine
                items.close()
            except engine_client.EngineDied as e:   # mid-stream: response.failed, then the stream ends
                self._responses_failed(asm, f"{e}; the next request restarts it", "server_error", send)
            except engine_client.QueueWaitExpired as e:
                self._responses_failed(asm, str(e), "queue_wait_timeout", send)
            except StructuredOutputError as e:
                self._responses_failed(asm, str(e), "structured_output_failed", send)
            except ValueError as e:                          # the engine's ERR after the stream started
                self._responses_failed(asm, str(e), "server_error", send)

        def _responses_failed(self, asm, message, code, send):
            e = asm.failed(message, code)
            self._note(error=e["response"]["error"])
            try:
                send(e)
                self.wfile.flush()
            except OSError:
                pass

        def _responses_prepare(self, req, cancel=None):
            responses_api.check_request(req)
            messages = responses_api.input_messages(req)
            tools, names, skipped = responses_api.request_tools(req)
            kw = responses_api.template_kwargs(req, {"reasoning_effort": svc.effort_default()})
            try:
                self._no_local_images(messages)              # as on the chat route: no file read for a foreign page
            except ValueError as e:
                raise ResponsesError(str(e), "input") from None
            try:
                messages, validator = prepare_format(responses_api.text_format(req), messages, with_tools=bool(tools))
            except ValueError as e:
                raise ResponsesError(str(e).replace("response_format", "text.format"), "text.format") from None
            noted = svc.__dict__.setdefault("hosted_tools_noted", set())   # said once per tool, not per request
            if set(skipped) - noted:
                print(f"[strata] /v1/responses: left out the hosted tools {', '.join(sorted(set(skipped) - noted))} "
                      f"(this server cannot run them)", flush=True)
                noted.update(skipped)
            if req.get("max_output_tokens") is None and "max_tokens" in svc.shared:   # the shared Chat settings
                req = {**req, "max_output_tokens": svc.shared["max_tokens"]}
            try:
                svc.reasoning_budget(req)                    # a bad value is a 400 before anything is sent
            except ValueError as e:
                raise ResponsesError(str(e), "reasoning_budget_tokens") from None
            try:
                stop_strings(req)
            except ValueError as e:
                raise ResponsesError(str(e), "stop") from None
            try:
                check_request_sampling(req)                   # the request is also the sampling dict (see below)
            except ValueError as e:                           # its message starts with the field's name: the param
                raise ResponsesError(str(e), str(e).partition(":")[0]) from None
            svc.load()
            # #924 (opt-in): a Codex compaction request is rendered with its conversation's tools, so its prompt
            # starts as the cached one did; the parser and the response keep the request's own tools
            shown = responses_api.prompt_tools(req, tools) if svc.codex_compaction_cache else tools
            try:
                try:
                    ids, thinking, max_new = svc.prepare(messages, shown, kw, req.get("max_output_tokens") or 0, req=req,
                                                          cancel=cancel)
                except ValueError:
                    if shown is tools:
                        raise
                    shown = tools                            # too long with the kept tools: the request as sent
                    ids, thinking, max_new = svc.prepare(messages, shown, kw, req.get("max_output_tokens") or 0, req=req,
                                                          cancel=cancel)
            except ResponsesError:
                raise
            except ValueError as e:                          # too long for the context, an image without vision
                raise ResponsesError(str(e), "input", "context_length_exceeded" if "context" in str(e) else None) \
                    from None
            if svc.codex_compaction_cache:
                responses_api.prompt_made(req, shown)
            _debug_req("responses", req, messages, tools, max_new, thinking, len(ids))
            include = req.get("include") if isinstance(req.get("include"), list) else []
            asm = responses_api.Assembler(req, svc.model_for(req), len(ids), names,
                                          "reasoning.encrypted_content" in include, validator is not None)
            # the request is also the sampling dict, as on the chat path (temperature, top_p, top_k, seed, ...)
            return ids, thinking, tools, max_new, asm, validator, req

        def _count_tokens(self, req):
            """Anthropic's token count, which Claude Code asks for its context figures: the prompt this server would
            read for the same request, rendered and tokenized - the model does not run."""
            req = svc.with_shared(req, "anthropic")
            messages, tools, kw = anthropic_to_messages(req, svc.anthropic_think_unasked)
            stop_strings(req)                                 # the same 400 as the request itself would get
            check_request_sampling(req)                       # ... likewise a sampling field of the wrong type
            self._json(200, {"input_tokens": len(svc.encode_prompt(messages, tools, kw, store=False))})

        def _anthropic(self, req):
            svc.load()
            req = svc.with_shared(req, "anthropic")
            messages, tools, kw = anthropic_to_messages(req, svc.anthropic_think_unasked)
            self._no_local_images(messages)
            if tool_choice_of(req.get("tool_choice"))[0] == "none":   # Anthropic's {"type": "none"}: no tools offered
                tools = None
            force = svc.force_text(forced_call(req.get("tool_choice"), tools))        # "any" / {"type": "tool", "name": N}
            max_new = int(req.get("max_tokens") or 0)                  # 0/-1: the rest of the context
            svc.reasoning_budget(req)                         # a bad value is a 400 before anything is sent
            stop_strings(req)                                 # ... and so is a bad stop / stop_sequences
            check_request_sampling(req)                       # ... and a sampling field of the wrong type
            cancel = threading.Event()
            self._watch_client(cancel)                       # #430 #431 (#1999: before prepare, which can take long)
            ids, thinking, max_new = svc.prepare(messages, tools, kw, max_new, force=force, req=req, cancel=cancel)
            _debug_req("anthropic", req, messages, tools, max_new, thinking, len(ids))
            events = anthropic_events(svc, req, ids, thinking, tools, max_new, cancel, force=force)
            events = self._capture(events, "anthropic")
            if not req.get("stream"):
                return self._json(200, anthropic_collect(events))
            self._sse()
            try:
                for item in events:
                    if item is None:
                        self.wfile.write(b": keep-alive\n\n")
                    else:
                        name, e = item
                        self.wfile.write(f"event: {name}\n".encode() + b"data: " +
                                         json.dumps(e, ensure_ascii=False).encode() + b"\n\n")
                    self.wfile.flush()
            except OSError:
                self._note(outcome="disconnected")
                svc.note_client_disconnect(cancel)
                events.close()
            except engine_client.EngineDied as e:                          # mid-stream: Anthropic's error event
                err = {"type": "error", "error": {"type": "api_error", "message": f"{e}; the next request restarts it"}}
                self._note(error=err["error"])
                self.wfile.write(b"event: error\ndata: " + json.dumps(err).encode() + b"\n\n")
            except engine_client.QueueWaitExpired as e:
                err = {"type": "error", "error": {"type": "rate_limit_error", "code": "queue_wait_timeout",
                                                   "message": str(e)}}
                self._note(error=err["error"])
                self.wfile.write(b"event: error\ndata: " + json.dumps(err).encode() + b"\n\n")
            except ValueError as e:                          # the engine's ERR after the stream started
                err = {"type": "error", "error": {"type": "api_error", "message": str(e)}}
                self._note(error=err["error"])
                self.wfile.write(b"event: error\ndata: " + json.dumps(err).encode() + b"\n\n")

    return Handler


class Server(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a second server bind a port that is already serving, and requests then land on
    # either one (a forgotten second start of run-<model>.bat).  Without it the second start fails loudly instead.
    allow_reuse_address = os.name != "nt"
    # socketserver listens with a backlog of 5: a burst of 30-40 clients at once got "connection reset by peer" on the
    # first ones (measured on 4 x R9700, also with one request at a time); the requests wait in the server, not the kernel
    # (STRATA_HTTP_BACKLOG overrides it)
    request_queue_size = max(5, int(os.environ.get("STRATA_HTTP_BACKLOG") or 256))

    def handle_error(self, request, client_address):
        if not isinstance(sys.exc_info()[1], ConnectionError):   # a client that hangs up needs no stack trace
            super().handle_error(request, client_address)

    def server_close(self):
        # the hardware sampler serve() started for this server stops with it (test isolation: every Service a test
        # started used to leave its sampler running, 21 of them after test_responses, which slowed test_parallel's timing)
        tel = getattr(getattr(self, "svc", None), "telemetry", None)
        if tel is not None and hasattr(tel, "close"):
            tel.close()
        super().server_close()


def warn_tight_ram(arena_mib) -> None:
    """The model's experts live in RAM (INFO arena_mib, engine 0.1.10+).  With less than ~6 GB left beside them for the
    system, the engine and this server, Linux ends the engine mid-answer when memory runs out (issue #27) and Windows
    pages to disk; say so at start instead of after a lost answer."""
    if not isinstance(arena_mib, int) or arena_mib <= 0:
        return
    try:
        import psutil
        total = psutil.virtual_memory().total
    except Exception:  # noqa: BLE001 - psutil is optional here
        return
    left = total / 2**30 - arena_mib / 1024
    if left < 6:
        print(f"[strata] WARNING: RAM is tight - the model's experts take {arena_mib / 1024:.1f} GB of this PC's "
              f"{total / 2**30:.0f} GB, leaving {left:.1f} GB for everything else. "
              + ("Linux may stop the engine in the middle of an answer. " if os.name != "nt" else
                 "Windows will slow down (paging to disk). ")
              + "Close other programs, or run START-HERE --setup and pick a smaller size (Q2_0 / IQ2_XS).", flush=True)


def warn_budget_over_ram(args) -> str | None:
    """#1080: an explicit `--resident-budget-gib N` bigger than the RAM this PC has free (or than its total beside
    ~6 GB for everything else) pushes experts into swap or the file tier: decode that is "dead slow" until the number is
    removed.  A warning at start, never a refusal (recommend, never force).  -> the sentence printed, or None."""
    try:
        budget = float(args[list(args).index("--resident-budget-gib") + 1])
    except (ValueError, IndexError):
        return None
    try:
        import psutil
        vm = psutil.virtual_memory()
        total, free = vm.total / 2**30, vm.available / 2**30
    except Exception:  # noqa: BLE001 - psutil is optional here
        return None
    if budget <= free and budget <= total - 6:
        return None
    msg = (f"WARNING: the RAM budget for experts is {budget:g} GiB (--resident-budget-gib), but this PC has "
           f"{free:.0f} GiB free of {total:.0f} GiB: what does not fit goes to swap or is read from disk, which can make "
           "decoding very slow. Lower it (setup sizes it for this PC) or remove the argument to let the engine choose.")
    print("[strata] " + msg, flush=True)
    return msg


DESKTOP_FREE_MIB = 2048          # #560 #516: below this, an AMD card that also drives a Linux desktop can run out
DESKTOP_RESERVE_MIB = 3072       # what kept KDE/Wayland alive beside a full expert cache in both reports


def linux_desktop(env=None) -> bool:
    """A graphical session on Linux (Wayland or X): its compositor, browser and apps take VRAM after the model has."""
    env = os.environ if env is None else env
    return os.name != "nt" and sys.platform != "darwin" and bool(env.get("WAYLAND_DISPLAY") or env.get("DISPLAY"))


def desktop_vram_note(backend, vram_free_mib, args: list, desktop: bool) -> str:
    """#560 #516: on Linux, when the desktop needs VRAM the AMD card does not have, amdgpu moves GPU memory (the expert
    cache, ~24 GB) to system RAM, which the experts already fill - the OOM killer then ends the compositor.  The
    default reserve (700 MiB) is sized for a card without a desktop.  A recommendation, nothing changes: "" when it
    does not apply."""
    if backend != "hip" or not desktop or not isinstance(vram_free_mib, int) or vram_free_mib >= DESKTOP_FREE_MIB:
        return ""
    try:
        reserve = int(args[args.index("--vram-reserve-mib") + 1]) if "--vram-reserve-mib" in args else 700
    except (ValueError, IndexError):
        reserve = 700
    if reserve >= DESKTOP_RESERVE_MIB:
        return ""
    return (f"[strata] note: {vram_free_mib} MiB of VRAM free with the model loaded. If this AMD card also drives your "
            "desktop and the desktop or apps crash after the start (the driver moves the expert cache to RAM and "
            "the OOM killer ends the session), keep more VRAM free: ./setup.sh --vram-reserve-mib "
            f"{DESKTOP_RESERVE_MIB} (remembered; the expert cache gets "
            f"{(DESKTOP_RESERVE_MIB - reserve) / 1024:.1f} GB less, a few % of speed)")


def lan_addresses() -> list[str]:
    """This PC's IPv4 addresses on its networks (what another device types in), without loopback/link-local."""
    import socket
    first, ips = None, set()
    try:                                                # the address of the default route; sends nothing (UDP)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            first = s.getsockname()[0]
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    ok = lambda ip: ip and not ip.startswith(("127.", "169.254.", "0."))
    return ([first] if ok(first) else []) + sorted(ip for ip in ips if ok(ip) and ip != first)


def host_name(value) -> str:
    """The name in a Host header (or an origin's host[:port]): "Example.com:8080" -> "example.com",
    "[::1]:8095" -> "::1"; "" when it is malformed."""
    v = (value or "").strip().lower()
    if v.startswith("["):
        name, sep, rest = v[1:].partition("]")
        return name if sep and (not rest or (rest[:1] == ":" and rest[1:].isdigit())) else ""
    if v.count(":") == 1:
        v, port = v.split(":")
        if not port.isdigit():
            return ""
    elif ":" in v:                                       # a bare IPv6 address (a config entry)
        return v
    v = v.rstrip(".")
    return v if v and all(c.isalnum() or c in "-._" for c in v) else ""


def _is_ip(name: str) -> bool:
    import ipaddress
    try:
        ipaddress.ip_address(name)
        return True
    except ValueError:
        return False


def _name_in(name: str, names) -> bool:
    """`name` is one of `names`, or below an entry that starts with a dot (".example.com")."""
    return name in names or any(n.startswith(".") and (name.endswith(n) or name == n[1:]) for n in names)


def allowed_hosts_of(value, env: str = "") -> list[str]:
    """The config's allowed_hosts (a name or a list) plus $STRATA_ALLOWED_HOSTS (comma-separated): host names, "*" or
    ".example.com" (it and every name below it).  A scheme, port or path is dropped ("https://a.example.com:8443/"
    -> "a.example.com"); a wrong entry stops the start (ValueError)."""
    items = [] if value in (None, "") else [value] if isinstance(value, str) else value
    if not isinstance(items, list) or not all(isinstance(x, str) for x in items):
        raise ValueError("allowed_hosts: expected a host name or a list of them")
    out = []
    for raw in items + [x for x in env.split(",") if x.strip()]:
        x = raw.strip().lower()
        if x == "*":
            out.append(x)
            continue
        x = x.split("://", 1)[-1].split("/", 1)[0]
        dot = x.startswith(".")
        name = host_name(x[1:] if dot else x)
        if not name:
            raise ValueError(f"allowed_hosts: {raw!r} is not a host name like strata.example.com")
        out.append("." + name if dot else name)
    return list(dict.fromkeys(out))


def host_names_for(bind_host: str, allowed_hosts=(), trusted_origins=()) -> set[str]:
    """Every name this server answers to besides an IP address: localhost, the address it listens on, the config's
    allowed_hosts and trusted_origins' hosts, and - listening beyond this PC (0.0.0.0 or a LAN address) - this PC's
    name and LAN addresses (the LAN addresses matter for the Origin check, which takes no IP on trust)."""
    names = set(LOOPBACK_NAMES)
    bind = host_name(bind_host)
    if bind and bind not in ("0.0.0.0", "::"):
        names.add(bind)
    if bind not in LOOPBACK_NAMES:
        try:
            pc = socket.gethostname().lower()
            names.update((pc, pc + ".local"))
        except OSError:
            pass
        names.update(("host.docker.internal", *lan_addresses()))
    names.update(x for x in allowed_hosts if x != "*")
    names.update(filter(None, (host_name(o.split("://", 1)[-1]) for o in trusted_origins)))
    return names


def host_allowed(host, names, any_host=False) -> bool:
    """DNS rebinding: a web page of another site whose name its DNS points at 127.0.0.1 reaches this server as the
    same origin, so the browser lets it read every answer.  Its requests carry that site's name in Host, so only
    the names this server answers to pass.  An IP address passes (a page served from an IP is that IP's own page;
    rebinding needs a name), as does "*.localhost" (browsers never ask DNS for it) and a request without a Host
    header (HTTP/1.0 clients; browsers always send one)."""
    if any_host or not (host or "").strip():
        return True
    name = host_name(host)
    return bool(name) and (_is_ip(name) or name == "localhost" or name.endswith(".localhost")
                           or _name_in(name, names))


def open_server_warning(port, public=False, tools=False) -> str:
    """The one plain warning for a server on the network with no API key (recommend, never force)."""
    msg = f"no API key: anyone who can reach port {port} can use the model" + (" and its tools" if tools else "") + "."
    if tools:
        msg += " The tools can read and change files on this computer."
    if public:
        msg += " This address is on the public internet, so anyone who finds it can use it."
    return msg + " Add an API key to keep others out."


def key_matches(given: str, key) -> bool:
    """given: a header's value as http.server read it, each byte one character.  A key with characters outside ASCII
    arrives as UTF-8 from most clients and as Latin-1 from some; the right key passes in both forms (#725: the
    bytes were re-encoded before, so a UTF-8 key never matched).  Constant-time (#213).  #1344: `key` may name several
    keys (see engine_client.key_list); the given one must match any, and every key is compared whatever the outcome."""
    raw = given.encode("latin-1", "replace")
    ok = False
    for k in engine_client.key_list(key):
        ok |= hmac.compare_digest(raw, k.encode())
        try:
            ok |= hmac.compare_digest(raw, k.encode("latin-1"))
        except UnicodeEncodeError:
            pass
    return ok


def origin_allowed(origin: str, host, names, origins=()) -> bool:
    """A browser page's Origin that may use the model without an API key: this server's own page (the Origin is the
    request's own Host), a page on one of the names this server answers to (any port), or an origin the config lists
    (trusted_origins, cors_origins).  Unlike the Host check no IP passes on trust: a page served from any other IP is
    another site.  "null" (a sandboxed frame, a file:// page) does not pass: any site can send it.  An origin of another
    scheme (chrome-extension://, moz-extension://, an Electron app's app://) does: no web site can send one."""
    origin = (origin or "").strip().rstrip("/")
    if origin in origins or "*" in origins:             # cors_origins ["*"]: the config lets every page in
        return True
    scheme, sep, rest = origin.lower().partition("://")
    if not sep or not scheme:
        return False
    if scheme not in ("http", "https"):
        return True
    if host and rest == host.strip().lower():
        return True
    name = host_name(rest)
    return bool(name) and (name in LOOPBACK_NAMES or name.endswith(".localhost") or _name_in(name, names))


def serve(svc: Service, host="127.0.0.1", port=8095) -> ThreadingHTTPServer:
    svc.host_names = host_names_for(host, svc.allowed_hosts, svc.trusted_origins)
    svc.start_telemetry()
    httpd = Server((host, port), make_handler(svc))
    httpd.svc = svc
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def origins_of(value, key: str, wildcard: bool) -> list[str]:
    """#321: a config's origin list ("https://chat.example.com" or a list of them) - scheme://host[:port], no path;
    "*" only where `wildcard` allows it.  A wrong entry stops the start rather than opening less or more than meant."""
    if value is None or value == "" or value == []:
        return []
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, list) or not all(isinstance(x, str) for x in items):
        raise SystemExit(f"[strata] {key}: expected an origin or a list of origins")
    out = []
    for x in items:
        x = x.strip().rstrip("/")
        if x == "*" and wildcard:
            out.append(x)
            continue
        scheme, sep, rest = x.partition("://")
        if scheme not in ("http", "https") or not sep or not rest or "/" in rest or "*" in rest:
            raise SystemExit(f"[strata] {key}: {x!r} is not an origin like https://chat.example.com"
                             + ("" if wildcard else " (no wildcards here)"))
        out.append(x)
    return out


SHARED_KEYS = ("reasoning_effort", "reasoning_budget_tokens", "temperature", "top_p", "top_k", "seed", "max_tokens",
               "experimental_speed_projection")


def clean_shared_defaults(d) -> dict:
    """The Chat settings other apps get (POST /settings): only known keys, each checked; ValueError names a bad one."""
    if d is None:
        return {}
    if not isinstance(d, dict):
        raise ValueError("defaults must be an object")
    out = {}
    for key, value in d.items():
        if value is None or value == "":
            continue
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if key == "reasoning_effort":
            if value not in ("none", "low", "medium", "high"):
                raise ValueError("reasoning_effort: none, low, medium or high")
        elif key == "temperature":
            if not number or not 0 <= value <= 2:
                raise ValueError("temperature: 0..2")
        elif key == "top_p":
            if not number or not 0 < value <= 1:
                raise ValueError("top_p: 0 < top_p <= 1")
        elif key == "top_k":
            if not number or value != int(value) or not 1 <= value <= 64:
                raise ValueError("top_k: an integer 1..64")
            value = int(value)
        elif key in ("seed", "max_tokens"):
            if not number or value != int(value) or value <= 0:
                raise ValueError(f"{key}: a positive integer")
            value = int(value)
        elif key == "reasoning_budget_tokens":                 # #123: 0 means no budget, so 0 is allowed here
            if not number or value != int(value) or value < 0:
                raise ValueError("reasoning_budget_tokens: a whole number of tokens (0: no budget)")
            value = int(value)
        elif key == "experimental_speed_projection":
            if not isinstance(value, bool):
                raise ValueError("experimental_speed_projection: true or false")
        else:
            raise ValueError(f"unknown setting {key!r}")
        out[key] = float(value) if key in ("temperature", "top_p") else value
    return out


def split_joined_args(cfg: dict) -> None:
    """#2267: an engine option written as ONE element of the config's "args" - "--ple-io mmap" or "--ple-io=mmap" - reaches
    the engine as a single argument ("unknown argument: --ple-io mmap"; the usual form is two elements, "--ple-io",
    "mmap").  Recommend, never force: such an element is split into the flag and its value, and said at startup.  Only an
    element that starts with a flag name (--letters-digits-dashes) followed by a space or "=" is touched, and never one
    that sits in the value place of the flag before it."""
    args = cfg.get("args")
    if not isinstance(args, list):
        return
    out, changed = [], []
    for a in args:
        m = re.match(r"^(--[A-Za-z0-9][A-Za-z0-9-]*)(?:\s+|=)(\S.*)$", a) if isinstance(a, str) else None
        prev = out[-1] if out else None
        in_value_place = isinstance(prev, str) and re.match(r"^--[A-Za-z0-9][A-Za-z0-9-]*$", prev) is not None
        if m is None or in_value_place:
            out.append(a)
            continue
        out += [m.group(1), m.group(2).strip()]
        changed.append(a)
    if changed:
        cfg["args"] = out
        print("[strata] the config's args hold " + ", ".join(repr(c) for c in changed) + " as one element each: split into the "
              'flag and its value (write them as two elements, e.g. "--ple-io", "mmap")', flush=True)


def read_config(path) -> dict:
    """The run config's JSON.  A mistake in it (a missing comma between two args, a trailing comma) is said with its
    place in the file instead of a traceback (#2267: editing "args" by hand is where it happens)."""
    return parse_config_text(Path(path).read_text(encoding="utf-8-sig"), path)   # Notepad adds a BOM


def parse_config_text(text: str, path) -> dict:
    """read_config on text already read (the owner-launched server hashes the bytes it parses, launch attestation)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise SystemExit(f"[strata] {path} is not valid JSON: {e.msg} at line {e.lineno}, column {e.colno}. Check the commas "
                         'between the entries of "args" (each option and each value is its own quoted entry, e.g. '
                         '"--ple-io", "mmap",) and that no entry is followed by a comma at the end of a list.') from None


def lift_reasoning_effort_arg(cfg: dict) -> None:
    """#1641: "--reasoning-effort high" in a config's "args" is meant for the thinking level, which is not an engine
    option (the engine stopped at the unknown flag).  It is the config's "reasoning_effort" (none, low, medium or high:
    the level for requests that name none), so the flag is taken out of the engine's arguments and becomes that key
    unless the config already has one; said at startup."""
    args = cfg.get("args")
    if not isinstance(args, list):
        return
    out, value, i = [], None, 0
    while i < len(args):
        a = args[i]
        if a == "--reasoning-effort" and i + 1 < len(args):
            value, i = args[i + 1], i + 2
        elif isinstance(a, str) and a.startswith("--reasoning-effort="):
            value, i = a.partition("=")[2], i + 1
        else:
            out.append(a)
            i += 1
    if value is None:
        return
    cfg["args"] = out
    if cfg.get("reasoning_effort") is None:
        cfg["reasoning_effort"] = value
    print("[strata] '--reasoning-effort' in the config's args is not an engine option: taken out of them. The thinking "
          "level is the config's 'reasoning_effort' key (outside 'args'); it is used as " +
          f"{cfg['reasoning_effort']!r}", flush=True)


# #1819: llama.cpp clients call the repetition penalty and its window repeat_penalty / repeat_last_n; they are the
# same settings as repetition_penalty / penalty_last_n (the /props answer already reports them under those names)
SAMPLING_ALIASES = {"repeat_penalty": "repetition_penalty", "repeat_last_n": "penalty_last_n"}


def apply_saved_tuning(cfg: dict, *, say=None) -> dict:
    """The run config with the active saved tuning for its choices put on top (strata_common/presets.py), with no measurement.
    Every start path runs this (the Terminal, `runtime start`, the app's load, a restart of the app, an unload and a load), so the
    tuning is applied every time and never re-measured. A tune's own worker sets STRATA_NO_PRESET so that it measures the
    configuration it was given. Never fails a start: a store that cannot be read means the configuration as it is."""
    if os.environ.get("STRATA_NO_PRESET"):
        return cfg
    say = say or (lambda line: print(line, flush=True))
    try:
        from strata_common import presets as _presets
        out, applied = _presets.overlay(cfg)
        if applied.get("applied"):
            changes = {**(applied.get("args") or {}), **(applied.get("env") or {})}
            note = applied.get("notice") or ""
            say(f"[strata] saved tuning '{applied['preset']['name']}' applied: "
                f"{' '.join(f'{k} {v}' for k, v in changes.items()) or 'the standard settings (nothing to change)'}"
                f"{'; ' + note if note and 'identical' not in note else ''}")
            if applied.get("state") == "soft":
                say("[strata] this PC or engine changed since it was saved (" + "; ".join(applied["reasons"]) +
                    "): a short re-check is available (strata presets recheck)")
        elif applied.get("why_not"):
            say(f"[strata] saved tuning '{applied['preset']['name']}' {applied['why_not']}; re-tune with strata tune auto")
        return out
    except Exception as e:  # noqa: BLE001
        say(f"[strata] saved tuning could not be read ({type(e).__name__}); starting with the configuration as it is")
        return cfg


def sampling_defaults_from_config(cfg: dict) -> dict:
    """The run config's optional `sampling` block: defaults for the sampling fields a request leaves out, so
    a plain client gets configured sampling instead of greedy.  Supported: temperature, top_p, top_k, min_p,
    presence_penalty, repetition_penalty (also written repeat_penalty), frequency_penalty, penalty_last_n (also
    repeat_last_n), seed.  The request's own fields
    always win - an explicit temperature=0 still means greedy, a field set to null falls back to the default.
    A bad value refuses to start the server (a typo'd config should not quietly change sampling); unknown keys
    are named at startup and ignored."""
    out = {}
    block = dict(cfg.get("sampling") or {})
    for alias, canon in SAMPLING_ALIASES.items():      # #1819: llama.cpp's names for the same two fields
        if block.get(alias) is not None:
            if block.get(canon) is not None:
                raise SystemExit(f"[strata] config sampling: both {alias!r} and {canon!r} are set; they are the same "
                                 f"setting, keep one")
            block[canon] = block.pop(alias)
    for key, value in block.items():
        if value is None:
            continue
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if key == "temperature":
            if not number or value < 0:
                raise SystemExit(f"[strata] config sampling.temperature={value!r}: expected a number >= 0 (0 = greedy)")
            out[key] = float(value)
        elif key == "top_p":
            if not number or not 0 < value <= 1:
                raise SystemExit(f"[strata] config sampling.top_p={value!r}: expected 0 < top_p <= 1")
            out[key] = float(value)
        elif key == "min_p":
            if not number or not 0 <= value <= 1:
                raise SystemExit(f"[strata] config sampling.min_p={value!r}: expected 0 <= min_p <= 1")
            out[key] = float(value)
        elif key == "top_k":
            if not number or value != int(value) or not 1 <= value <= 64:
                raise SystemExit(f"[strata] config sampling.top_k={value!r}: the sampled path takes an integer 1..64")
            out[key] = int(value)
        elif key == "presence_penalty":
            if not number or value < 0:
                raise SystemExit(f"[strata] config sampling.presence_penalty={value!r}: expected a number >= 0")
            out[key] = float(value)
        elif key == "frequency_penalty":
            if not number or value < 0:
                raise SystemExit(f"[strata] config sampling.frequency_penalty={value!r}: expected a number >= 0")
            out[key] = float(value)
        elif key == "repetition_penalty":
            if not number or value <= 0:
                raise SystemExit(f"[strata] config sampling.repetition_penalty={value!r}: expected a number > 0 (1 = off)")
            out[key] = float(value)
        elif key == "penalty_last_n":
            if not number or value != int(value) or value < 0:
                raise SystemExit(f"[strata] config sampling.penalty_last_n={value!r}: expected a non-negative integer")
            out[key] = int(value)
        elif key == "seed":
            if not number or value != int(value) or value <= 0:
                raise SystemExit(f"[strata] config sampling.seed={value!r}: expected a positive integer")
            out[key] = int(value)
        elif key == "strata_tune":           # a saved tuning's per-request settings (strata_common/presets.py)
            if not isinstance(value, dict):
                raise SystemExit(f"[strata] config sampling.strata_tune={value!r}: expected an object")
            out[key] = {k: v for k, v in value.items() if k in ("pcie_frac", "spec_min_p", "aux_cpus")}
        elif key == "experimental_speed_projection":
            if not isinstance(value, bool):
                raise SystemExit(f"[strata] config sampling.experimental_speed_projection={value!r}: expected true or "
                                 "false (the default for requests that leave it out, when the engine has the vector)")
            out[key] = value
        else:
            print(f"[strata] config sampling.{key}={value!r}: unknown key, ignored", flush=True)
    return out


def start_replicas(cfg: dict, groups: list[list[int]], exe: str, effort_end: list[str], lazy: bool,
                   *, offline: bool = False):
    """Opt-in data-parallel replicas (strata_server/replicas.py): one StrataEngine per card group, each with its own cards, log
    and CPU cores, started together; the ReplicaEngine in front spreads the requests."""
    from strata_server import replicas as rp
    n = len(groups)
    cpus = rp.cpu_sets(n, cfg.get("replica_cpus") is not False)
    print(f"[strata] {n} replicas on the cards {' | '.join(','.join(map(str, g)) for g in groups)}"
          f"{' (own CPU cores each)' if cpus else ''}: requests go to the least busy one, a conversation back to its own",
          flush=True)
    engines: list = [None] * n
    errors: list = []

    def one(i: int):
        rc = rp.replica_cfg(cfg, i, groups[i])
        try:
            engines[i] = engine_client.StrataEngine(exe, engine_client.engine_args(rc) + effort_end, cwd=rc.get("cwd"),
                                                    log=rc.get("log"), env=engine_client.child_env(rc, offline=offline),
                                                    lazy=lazy, cpus=cpus[i] if cpus else None)
        except BaseException as e:        # noqa: BLE001
            errors.append((i, e))
    rc0 = rp.replica_cfg(cfg, 0, groups[0])
    warn_budget_over_ram(engine_client.engine_args(rc0) if "args" in rc0 else [])
    if os.environ.get("STRATA_REPLICA_START", "") == "serial":
        for i in range(n):
            one(i)
    else:
        ts = [threading.Thread(target=one, args=(i,)) for i in range(n)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    if errors:
        for e in engines:
            if e is not None:
                try:
                    e.close()
                except Exception:        # noqa: BLE001
                    pass
        i, e = errors[0]
        raise SystemExit(f"[strata] replica {i} did not start: {e}")
    load = int(cfg.get("replica_load_tokens") or rp.LOAD_TOKENS)
    return rp.ReplicaEngine(engines, load_tokens=load)


def check_request_sampling(req) -> None:
    """A request's sampling field of the wrong JSON type ("temperature": "0.7", "top_k": 40.0, "seed": true) is a 400
    that names the field, as the same value is through POST /settings or the run config.  Unchecked, the value
    replaced the configured default in run()'s merge and sampling_keys dropped it, so the reply was greedy.  Only
    types are checked: a number (int or float, never bool) for temperature, top_p, min_p and the three penalties; an
    integer (never bool or float) for top_k, penalty_last_n, seed and priority.  Ranges behave as before (a
    temperature <= 0 is greedy, top_k 0 or above 64 is clamped, a seed <= 0 is unset), and an absent or null field is
    not checked."""
    if isinstance(req, dict):      # #1819: llama.cpp's names; the OpenAI-style field wins when both are sent
        if req.get("repeat_penalty") is not None and req.get("repetition_penalty") is None:
            req["repetition_penalty"] = req["repeat_penalty"]
        rl = req.get("repeat_last_n")
        if req.get("penalty_last_n") is None and isinstance(rl, int) and not isinstance(rl, bool) and rl > 0:
            req["penalty_last_n"] = rl      # 0 and -1 (off / the whole context) keep the engine's default window
    for key, integer in (("temperature", False), ("top_p", False), ("min_p", False), ("presence_penalty", False),
                         ("frequency_penalty", False), ("repetition_penalty", False), ("top_k", True),
                         ("penalty_last_n", True), ("seed", True), ("priority", True)):
        value = (req or {}).get(key)
        if value is None:
            continue
        if isinstance(value, str) and value.strip():      # some clients send numbers as strings: read and used
            try:
                num = float(value)
                num = int(num) if integer and num == int(num) else (num if not integer else None)
                if integer and num is None:
                    raise ValueError
                if not integer and re.fullmatch(r"\s*[+-]?\d+\s*", value):
                    num = int(value)
            except (ValueError, OverflowError):
                raise ValueError(f"{key}: {'an integer' if integer else 'a number'}") from None
            if num != num or num in (float("inf"), float("-inf")):
                raise ValueError(f"{key}: {'an integer' if integer else 'a number'}")
            req[key] = value = num
        if isinstance(value, bool) or not isinstance(value, int if integer else (int, float)):
            raise ValueError(f"{key}: {'an integer' if integer else 'a number'}")


# the API's model name per model family (the GGUF architecture, "arch" in the config) when the config names none
FAMILY_MODEL_NAMES = {"qwen4exp": "qwen3.8-flash-next", "qwen35moe": "qwen3.6-35b-a3b",
                      "deepseek41": "deepseek-v4.1-flash"}


# The sampling each model card recommends ("Best Practices"), used when neither the request, the Chat settings nor the
# run config's `sampling` block says otherwise.  Greedy is only for a request or config that asks for temperature 0.
# min_p 0, repetition_penalty 1 and presence_penalty 0 are the engine's own "off", so they are not listed.
MODEL_CARD_SAMPLING = {
    # Qwen3.8-Flash-Next (and Gyro, the same base model)
    "qwen4exp": {"thinking": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
                 "instruct": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}},
    # Qwen3.6-35B-A3B: the general-task sets (the card's precise-coding set is temperature 0.6, presence_penalty 0)
    "qwen35moe": {"thinking": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5},
                  "instruct": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}},
    # DeepSeek-V4.1-Flash: temperature 1.0, top_p 1.0 for local use (thinking or not), no penalties
    "deepseek41": {"thinking": {"temperature": 1.0, "top_p": 1.0}, "instruct": {"temperature": 1.0, "top_p": 1.0}},
}


def model_card_sampling(arch, thinking) -> dict:
    """The model card's sampling for the family `arch` and the thinking state ({} for an unknown or unset family)."""
    sets = MODEL_CARD_SAMPLING.get(arch) if arch else None
    return dict(sets["thinking" if thinking else "instruct"]) if sets else {}


def default_model_name(cfg: dict) -> str:
    """The config's "model_name", else its family's name ("arch": qwen4exp, qwen35moe), else Flash-Next's."""
    return cfg.get("model_name") or FAMILY_MODEL_NAMES.get(cfg.get("arch") or "qwen4exp", "qwen3.8-flash-next")


def with_current_reader(cfg: dict, package: Path) -> dict:
    """The picture reader (vision "exe", "cpu_exe") is written into the run config as an absolute path inside the engine package's folder,
    and that folder is named after the engine version: after an engine update, or in a container made for another card, the path is gone
    and the server would stop with "server_exited". Use the same file of the package that is installed now (the reader ships in the package)."""
    vision = cfg.get("vision")
    if not isinstance(vision, dict):
        return cfg
    fixed = dict(vision)
    for key in ("exe", "cpu_exe"):
        old = fixed.get(key)
        if isinstance(old, str) and old and not engine_component.is_reference(old) and os.path.isabs(old) and not os.path.exists(old):
            new = package / old.replace("\\", "/").rsplit("/", 1)[-1]
            if new.is_file():
                print(f"[strata] the picture reader {old} is not there any more; using {new} from the engine package in use", flush=True)
                fixed[key] = str(new)
    if fixed == vision:
        return cfg
    return {**cfg, "vision": fixed}


def with_component_engine(cfg: dict) -> dict:
    """A run config that names "component:strata-engine" gets the engine package for this PC: the path in `exe`, and
    the package's own folder first on the library path (the GPU runtime libraries ship beside the binary)."""
    if not engine_component.is_reference(cfg.get("exe")):
        return cfg
    cfg = dict(cfg)
    cfg["exe"] = _component_exe(cfg["exe"], cfg)
    cfg["lib_dirs"] = [*engine_component.package_lib_dirs(Path(cfg["exe"]).parent), *(cfg.get("lib_dirs") or [])]
    cfg = with_current_reader(cfg, Path(cfg["exe"]).parent)
    if cfg.get("backend") == "hip":     # AMD: the package's own hipBLASLt tuning table for this card, when its hipBLASLt version matches
        cfg = hipblaslt_tuning.apply(cfg, Path(cfg["exe"]).parent, say=lambda line: print(line, flush=True))
    return cfg


def _component_exe(reference: str, cfg: dict) -> str:
    """The engine a run config names as "component:strata-engine". No engine for this PC is a clear notice and exit 5;
    the server never goes looking on the internet for one (strata_common/engine_component.py)."""
    try:
        return engine_component.expand(reference, cfg)
    except engine_component.EngineMissing as e:
        print(f"[strata] {e}", file=sys.stderr, flush=True)
        raise SystemExit(e.exit_code) from None


def main(argv=None, *, lifecycle_event=None, interrupt_exit_code: int = 0) -> int:
    try:                                               # files made by a run as administrator still belong to the user
        from strata_base import private_files as _pf
        _pf.adopt_user_defaults()
    except Exception:
        pass
    event_sink = lifecycle_event

    def notify_lifecycle(kind: str, payload: dict) -> None:
        nonlocal event_sink
        if event_sink is None:
            return
        try:
            event_sink(kind, payload)
        except (OSError, ValueError):
            # A closed event consumer must not strand the owned engine or API.
            event_sink = None

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["mock", "strata"], default="mock")
    ap.add_argument("--config", help="strata engine config (JSON: exe, args, cwd, tokenizer, model_name), "
                                     "written by setup.py")
    ap.add_argument("--host", default=None,
                    help="the address to listen on: 127.0.0.1 = this PC only (the default), 0.0.0.0 = also other devices "
                         "on your network (set an API key); also \"host\" in the config")
    ap.add_argument("--script-cycle", action="store_true",
                    help="with --script given more than once: repeat the scripts by position instead of advancing "
                         "to the last one and staying there, so extra requests a harness makes (a session title, "
                         "a re-ask after a tool result) cannot shift the sequence")
    ap.add_argument("--script-delay", type=float, default=0.0,
                    help="seconds to wait between the mock engine's tokens, so a turn stays live long enough for a "
                         "client to cancel it; the mock honours cancellation between tokens")
    ap.add_argument("--script", action="append",
                    help="the mock engine's answer (default: a short greeting); given more than once, requests get "
                         "them in turn and the last one repeats")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--ready-file", help="write a JSON instance record once the API is bound and model state is known")
    ap.add_argument("--owner-fd", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--shutdown-timeout", type=float, default=None,
                    help="opt-in graceful drain deadline in seconds (0..300) before owned requests are cancelled")
    ap.add_argument("--parallel", type=int, default=None,
                    help="request this many simultaneous engine slots without changing the saved config")
    ap.add_argument("--batch-groups", default=None,
                    help="pipeline groups: auto or a positive integer, without changing the saved config")
    ap.add_argument("--max-queue", type=int, default=None,
                    help="maximum inference calls waiting beyond effective engine slots; 0 refuses overload")
    ap.add_argument("--queue-timeout", type=float, default=None,
                    help="maximum seconds waiting for the first engine turn (0.1..600; off by default)")
    ap.add_argument("--batch-telemetry", action=argparse.BooleanOptionalAction, default=None,
                    help="opt in to low-rate engine decode-window samples for /v1/telemetry")
    ap.add_argument("--offline", action="store_true",
                    help="do not fetch remote image URLs or start MCP/tool subprocesses")
    ap.add_argument("--gpu", help="the GPU to run on, as nvidia-smi numbers them, or several for a layer split "
                                  "(\"0,2\"; also \"gpu\" in the config)")
    ap.add_argument("--tokenizer", default=str(ROOT / "pack/full/tokenizer"),
                    help="pack tokenizer directory (falls back to a byte tokenizer if absent)")
    ap.add_argument("--open", action="store_true", help="open the local page in the browser once the model is ready")
    ap.add_argument("--fit-max-tokens", action="store_true",
                    help="clamp max_tokens to the remaining context instead of rejecting the request "
                         "(the default; kept so older scripts keep working)")
    ap.add_argument("--no-fit-max-tokens", action="store_true",
                    help="answer 400 when prompt + max_tokens does not fit the context, like llama.cpp "
                         "(also \"fit_max_tokens\": false in the config). A prompt that alone does not fit is always 400")
    ap.add_argument("--api-key", default=os.environ.get("STRATA_API_KEY", ""),
                    help="require this key on /v1/* (Authorization: Bearer ... or x-api-key); also $STRATA_API_KEY. "
                         "Several keys: separate them with commas (key1,key2), as llama.cpp does")
    ap.add_argument("--no-api-key", action="store_true",
                    help="serve on the network without an API key (anyone who can reach the port can use the model; "
                         "also \"api_key\": \"none\" in the config). Warns once, never refuses")
    ap.add_argument("--mcp-config", help="a JSON file with MCP servers in Claude Desktop's format ({\"mcpServers\": "
                                         "{...}}); the web app's chat can use their tools (also \"mcp_servers\" in "
                                         "the config)")
    ap.add_argument("--lazy", action="store_true", help="start the API unloaded (the engine and the vision encoder start with the first request)")
    ap.add_argument("--api-monitor", action="store_true",
                    help="the API request monitor at /api-monitor: keeps the last 100 requests' prompts and answers in "
                         "memory (also \"api_monitor\": true in the config; off by default)")
    ap.add_argument("--idle-unload", type=float, default=None, metavar="SECONDS",
                    help="unload the model after this many seconds without requests, so other programs (games, other "
                         "model servers) can use the VRAM; the next request loads it again (also \"idle_unload_s\" "
                         "in the config; default: never)")
    ap.add_argument("--min-free-vram-mib", type=int, default=None,
                    help="load an unloaded model only when this much VRAM is free, else answer 503 (also "
                         "\"min_free_vram_mib\" in the config; default: always load)")
    ap.add_argument("--before-load", help="a command run before the model is loaded again (e.g. to unload another "
                                          "server's model; also \"before_load\" in the config, a string or a list)")
    ap.add_argument("--slot-save-path", default=None, metavar="DIR",
                    help="enable POST /slots/0?action=save|restore {\"filename\": NAME} (llama-server's API): the "
                         "conversation the engine holds, to or from DIR/NAME (also \"slot_save_path\" in the config)")
    ap.add_argument("--replicas", type=int, default=None, metavar="N",
                    help="data-parallel replicas (opt-in): cut the \"gpu\" list into N equal groups and run one engine on "
                         "each, requests spread over them (also \"replicas\" in the config: a number, or "
                         "[{\"gpus\": [0, 1]}, {\"gpus\": [2, 3]}])")
    launch_args = list(sys.argv[1:] if argv is None else argv)
    a = ap.parse_args(launch_args)
    # #3238: say it before the first unexplained -9
    if a.engine == "strata" and (_rt := engine_client.rttime_limit_note()):
        print("[strata] WARNING: " + _rt, flush=True)
    engine_client.OPENS_BROWSER = bool(a.open)
    # Every ordinary way of being asked to stop takes Ctrl+C's path (QUIT to the engine, then exit): SIGTERM (#96 docker
    # stop; systemd), SIGHUP (the terminal closed, ssh dropped), and on Windows Ctrl+Break and the console's close,
    # logoff and shutdown events. They are installed before anything slow starts, so one that arrives while the model
    # loads is not the default "die at once". SIGINT keeps Python's own handler (a second Ctrl+C stops waiting).
    termination = "sigint"
    finished = threading.Event()                        # set when the clean stop is over (a Windows console handler waits for it)

    def request_stop(reason: str):
        nonlocal termination
        termination = reason
        if threading.current_thread() is threading.main_thread():
            raise KeyboardInterrupt
        import _thread
        _thread.interrupt_main()
    lifeline.install_stop_handlers(request_stop, finished)
    if a.owner_fd is not None:
        try:
            owner_guard.start(a.owner_fd, grace=(a.shutdown_timeout or 0) + 40)
        except (OSError, ValueError) as exc:
            ap.error(str(exc))
    if a.shutdown_timeout is not None and not 0 <= a.shutdown_timeout <= 300:
        ap.error("--shutdown-timeout must be in 0..300 seconds")
    config_bytes = Path(a.config).read_bytes() if a.config else b""
    cfg = parse_config_text(config_bytes.decode("utf-8-sig"), a.config) if a.config else {}   # Notepad adds a BOM
    split_joined_args(cfg)
    if a.config and a.engine == "strata":
        cfg = apply_saved_tuning(cfg)
    attestation_key = os.environ.pop("STRATA_OWNER_ATTEST_KEY", None)
    launch_attestation = None
    if a.owner_fd is not None:
        if attestation_key is None or not re.fullmatch(r"[0-9a-f]{64}", attestation_key):
            ap.error("owned server needs a private launch attestation key")
        launch_attestation = launch_identity.attestation(bytes.fromhex(attestation_key),
                                                          config_bytes, launch_args)
    if "batch_telemetry" in cfg and type(cfg["batch_telemetry"]) is not bool:
        ap.error("batch_telemetry must be true or false")
    try:
        ready_path = ready_file.preflight(a.ready_file) if a.ready_file else None
    except ValueError as e:
        ap.error(str(e))
    max_queue = a.max_queue if a.max_queue is not None else cfg.get("max_queue")
    if max_queue is not None and (isinstance(max_queue, bool) or not isinstance(max_queue, int)
                                  or not 0 <= max_queue <= 4096):
        ap.error("--max-queue / max_queue must be a whole number in 0..4096")
    queue_timeout = a.queue_timeout if a.queue_timeout is not None else cfg.get("queue_timeout")
    if queue_timeout is not None and (isinstance(queue_timeout, bool) or not isinstance(queue_timeout, (int, float))
                                      or not math.isfinite(queue_timeout) or not 0.1 <= queue_timeout <= 600):
        ap.error("--queue-timeout / queue_timeout must be finite seconds in 0.1..600")
    if queue_timeout is not None and max_queue is None:
        ap.error("--queue-timeout requires --max-queue / max_queue to bound pending calls")
    if a.parallel is not None:
        if a.parallel < 1 or a.parallel > 1024:
            ap.error("--parallel must be in 1..1024; the engine may reduce it to the supported count")
        if "--batch" in cfg.get("args", []) or "--slots" in cfg.get("args", []):
            ap.error("--parallel conflicts with an explicit --batch/--slots in the saved engine args")
        cfg["parallel"] = a.parallel
    if a.batch_groups is not None:
        if a.batch_groups != "auto" and (not a.batch_groups.isdecimal() or int(a.batch_groups) < 1):
            ap.error("--batch-groups must be auto or a positive integer")
        if not cfg or "args" not in cfg:
            ap.error("--batch-groups needs an engine config")
        if "--batch-groups" in cfg["args"]:
            ap.error("--batch-groups conflicts with the saved engine args")
        cfg["args"] = [*cfg["args"], "--batch-groups", a.batch_groups]
    lift_reasoning_effort_arg(cfg)
    if a.replicas is not None:
        cfg["replicas"] = a.replicas
    if a.gpu is not None:
        cfg["gpu"] = int(a.gpu) if a.gpu.strip().isdigit() else a.gpu
    a.host = a.host or cfg.get("host") or "127.0.0.1"   # issue #26: the run scripts pass no --host, the config can
    supplied_args = argv if argv is not None else sys.argv[1:]
    if ("STRATA_API_KEY" in os.environ and not os.environ["STRATA_API_KEY"].strip()) or \
            any((x == "--api-key" and i + 1 < len(supplied_args) and not supplied_args[i + 1].strip()) or
                x.strip() == "--api-key=" for i, x in enumerate(supplied_args)):
        print("[strata] an API key was given but it is empty: set a key, or leave it out", file=sys.stderr)
        return 2
    raw_key = a.api_key or cfg.get("api_key", "")
    no_key_chosen = bool(getattr(a, "no_api_key", False)) or engine_client.explicit_no_key(raw_key)
    if engine_client.explicit_no_key(raw_key):
        raw_key = ""
    try:
        preflight_api_key = engine_client.api_key_of(raw_key)
    except ValueError as e:
        print(f"[strata] {e}: set a key, or leave it out", file=sys.stderr)
        return 2
    # A key is recommended on a network, never required (owner decision 2026-10-20): the banner below warns once.
    try:                                                # before the minutes of loading: is the port free?
        Server((a.host, a.port), BaseHTTPRequestHandler).server_close()
    except OSError as e:
        ap.error(listen_problem(a.host, a.port, e))
    if cfg.get("tokenizer"):
        a.tokenizer = cfg["tokenizer"]
    tok = ByteTokenizer()
    tpath = Path(a.tokenizer)
    if a.engine == "strata" and not (tpath / "vocab.json").exists():
        ap.error(f"the model's tokenizer is missing ({tpath / 'vocab.json'}); run setup again")
    if (tpath / "vocab.json").exists():
        import strata_tokenizer as ST
        vocab = json.loads((tpath / "vocab.json").read_text(encoding="utf-8"))
        tokens = [None] * len(vocab)
        for t, i in vocab.items():
            tokens[i] = t
        merges = (tpath / "merges.txt").read_text(encoding="utf-8").split("\n")
        types = json.loads((tpath / "token_type.json").read_text())
        tcfg = {}
        try:                                            # the pack's pre-tokenizer name and special ids (DeepSeek's differ)
            tcfg = json.loads((tpath / "tokenizer.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        tok = ST.Tokenizer(tokens, merges, types, tcfg.get("pre") or "qwen35", tcfg.get("special_ids") or None)
    hub = hub_from_config(cfg, a.mcp_config)            # before the minutes of loading: a bad entry stops here
    if a.offline and (hub is not None or cfg.get("before_load") or a.before_load):
        ap.error("--offline requires MCP servers and before_load commands to be disabled in this profile")
    if a.offline:
        # The same decision, reported through the policy module so the two cannot drift: a profile
        # that permits a network source is refused here rather than on its first request.
        try:
            from strata_common import offline as offline_policy
            blocked = offline_policy.ensure_strictly_offline(cfg)
        except ImportError:
            blocked = []
        if blocked:
            ap.error("--offline cannot be strict for this profile: " + "; ".join(blocked))
    lazy = False
    if a.engine == "strata":
        if not cfg:
            ap.error("--engine strata needs --config")
        cfg = with_component_engine(cfg)
        vision = None
        # The engine's own environment. It carries the offline policy too: an engine that fetched a
        # missing shard or a plugin would break the promise the flag makes.
        env = engine_client.child_env(cfg, offline=a.offline)
        configure_batch_telemetry_env(env, cfg, a.batch_telemetry)
        sampling_defaults = sampling_defaults_from_config(cfg)
        if sampling_defaults:
            pretty = ", ".join(f"{k}={v}" for k, v in sampling_defaults.items())
            print(f"[strata] sampling defaults from the config: {pretty}", flush=True)
        lazy = a.lazy or cfg.get("lazy_load") is True
        for line in card_roles_lines(cfg):             # the card roles: said first; every card Off stops here, before anything touches a card
            print(line, flush=True)
        if cfg.get("vision"):
            elastic = cfg["vision"].get("elastic") is True and cfg.get("vram_elastic") is True
            print("the vision encoder starts with the first picture (elastic) ..." if elastic else
                  "loading the vision encoder ..." if not lazy else
                  "vision encoder unloaded; it starts with the model ...", flush=True)
            # relative paths are the config's cwd's, as for the engine below
            vcfg = {k: (os.path.abspath(os.path.join(cfg.get("cwd") or ".", v))
                        if k in ("exe", "mmproj", "model") and isinstance(v, str) and not os.path.isabs(v) else v)
                    for k, v in cfg["vision"].items() if not (k == "exe" and engine_component.is_reference(v))}
            if engine_component.is_reference(cfg["vision"].get("exe")):     # the engine from the component folder
                vcfg["exe"] = _component_exe(cfg["vision"]["exe"], cfg)
            if vcfg.get("elastic") is True and cfg.get("vram_elastic") is not True:
                print('[strata] the vision section\'s "elastic" needs "vram_elastic": true (the expert cache gives the '
                      "encoder its VRAM): the encoder starts with the model instead", flush=True)
                vcfg["elastic"] = False
            vision = Vision(vcfg, log=open(cfg["log"], "a", encoding="utf-8") if cfg.get("log") else None,
                            env=vision_env(cfg, env), lazy=lazy)
        print("model unloaded; the first request loads it ..." if lazy else
              "loading the model (the first start takes a minute or two) ...", flush=True)
        warn_budget_over_ram(engine_client.engine_args(cfg) if "args" in cfg else [])      # #1080
        note = card_note(cfg, env)
        if note:
            print(note, flush=True)
        igpu_line = igpu_split_warning(cfg)
        if igpu_line:
            print(igpu_line, flush=True)
        if engine_client.layer_split_of(cfg):
            try:
                split = engine_client.layer_split_value(cfg)          # #644: before the (minutes-long) start
            except ValueError as e:
                raise SystemExit(f"[strata] config {e}")
            print(f"[strata] layer split across GPUs {engine_client.gpu_list(cfg)} ({split})", flush=True)
        elif len(engine_client.gpu_list(cfg)) > 1:
            print(f"[strata] GPUs {engine_client.gpu_list(cfg)}: the later card(s) serve as the peer expert tier "
                  "(--peer-device)", flush=True)
        # a relative "exe" is the config's cwd's: Windows' CreateProcess resolves "engine/strata.exe" against nothing
        # it is told about (WinError 2), so it is made absolute here
        if engine_component.is_reference(cfg["exe"]):      # "component:strata-engine": the engine package for this PC
            exe = _component_exe(cfg["exe"], cfg)
        else:
            exe = cfg["exe"] if os.path.isabs(cfg["exe"]) else os.path.abspath(os.path.join(cfg.get("cwd") or ".", cfg["exe"]))
        try:
            silence = engine_silence_s(cfg)             # #481: checked before the (minutes-long) start
            apply_ready_timeout(cfg)
        except ValueError as e:
            raise SystemExit(f"[strata] config {e}")
        try:
            effort_end = effort_end_args(cfg, exe, tok)  # #458
        except ValueError as e:
            raise SystemExit(f"[strata] config {e}")
        from strata_server import replicas as rp
        try:
            groups = rp.gpu_groups(cfg)
        except rp.ReplicaConfigError as e:
            raise SystemExit(f"[strata] config {e}")
        if groups:
            engine = start_replicas(cfg, groups, exe, effort_end or [], lazy, offline=a.offline)
        else:
            engine = engine_client.StrataEngine(exe, engine_client.engine_args(cfg) + (effort_end or []),
                                                cwd=cfg.get("cwd"), log=cfg.get("log"), env=env, lazy=lazy)
        engine.silence_s = silence                      # an attribute of its own: restart() keeps it
        warn_tight_ram(engine.info.get("arena_mib"))
        note = desktop_vram_note(cfg.get("backend"), engine.info.get("vram_free_mib"), engine.spawn[1],
                                 linux_desktop()) if not groups else None
        if note:                                        # #560 #516: before --open starts a browser on that card
            print(note, flush=True)
    else:
        effort_end = None
        engine, vision, sampling_defaults = engine_client.MockEngine(tok, a.script or [
            "Thinking about it.</think>\n\nHello from the mock engine."],
            cycle=getattr(a, "script_cycle", False),
            delay_s=getattr(a, "script_delay", 0.0) or 0.0), None, {}
    # the model's own chat template (exported with its tokenizer), else the original model's
    tpl = tpath / "chat_template.jinja"
    if (cfg.get("arch") or "") == "deepseek41":
        from strata_server import dsml
        tpl, swapped = dsml.pick_template(tpl)
        if swapped:
            print("[strata] this pack's chat template spells the tool-call block differently from the V4.1 model's; "
                  "using the V4.1 template so tool calls come back as real calls", flush=True)
    svc = Service(engine, tok, ChatTemplate(tpl if tpl.exists() else ROOT / "strata_server/chat_template.jinja"),
                  model_name=default_model_name(cfg), vision=vision,
                  sampling_defaults=sampling_defaults,
                  fit_max_tokens=a.fit_max_tokens or (not a.no_fit_max_tokens and cfg.get("fit_max_tokens") is not False))
    svc.launch_attestation = launch_attestation
    svc.offline = a.offline
    svc.max_queue = max_queue
    svc.queue_timeout_s = float(queue_timeout) if queue_timeout is not None else None
    if a.engine == "strata":                            # the model card's sampling when nothing else is said (see MODEL_CARD_SAMPLING)
        svc.model_arch = cfg.get("arch") or "qwen4exp"
        svc.set_family(svc.model_arch if svc.model_arch == "deepseek41" else None)
        requested = cfg.get("parallel")
        if requested is None:
            args = cfg.get("args", [])
            for flag in ("--batch", "--slots"):
                if flag in args and args.index(flag) + 1 < len(args):
                    try:
                        requested = int(args[args.index(flag) + 1])
                    except ValueError:
                        pass
        per_replica = requested if isinstance(requested, int) and not isinstance(requested, bool) and requested > 0 else 1
        svc.requested_slots = per_replica * len(groups) if groups else per_replica
    svc.answer_room = cfg.get("answer_room") is not False
    svc.reasoning_close_retry = cfg.get("reasoning_close_retry") is True    # #1053: opt-in, off by default
    if cfg.get("images"):                               # image generation (strata_server/images.py), set up by setup --images
        icfg = {k: (os.path.abspath(os.path.join(cfg.get("cwd") or ".", v))
                    if k in ("exe", "diffusion_model", "vae", "llm", "out_dir") and isinstance(v, str)
                    and not os.path.isabs(v) else v)
                for k, v in cfg["images"].items()}
        try:
            svc.images = images_api.ImageGen(icfg, log=open(cfg["log"], "a", encoding="utf-8") if cfg.get("log") else None)
        except ValueError as e:
            raise SystemExit(f"[strata] config: {e}")
        gone = svc.images.missing()
        if gone:
            print(f"[strata] image generation: missing {', '.join(gone)} - requests fail until setup --images has run "
                  "again", flush=True)
        print(f"[strata] image generation: {svc.images.model} {icfg.get('quant') or ''} on "
              f"{icfg.get('backend') or '?'} (POST /v1/images/generations; GPU policy {svc.images.policy})", flush=True)
    if cfg.get("prevent_sleep") is not None and not isinstance(cfg["prevent_sleep"], bool):
        raise SystemExit(f"[strata] config \"prevent_sleep\" must be true or false, not {cfg['prevent_sleep']!r}")
    from strata_server.keepawake import KeepAwake
    svc.keep_awake = KeepAwake.create(cfg.get("prevent_sleep") is True)      # #1727: opt-in, Windows only
    if cfg.get("reasoning_effort") is not None:         # #1641: the thinking level of requests that name none
        try:
            svc.config_effort = clean_shared_defaults({"reasoning_effort": cfg["reasoning_effort"]}).get("reasoning_effort")
        except ValueError as e:
            raise SystemExit(f"[strata] config {e}")
        if svc.config_effort:
            print(f"[strata] thinking level: {svc.config_effort} (reasoning_effort in the config; a request, or the "
                  "Chat settings shared with other apps, can set its own)", flush=True)
    svc.codex_thread_titles = cfg.get("codex_thread_titles") is True    # #923: opt-in, off by default
    svc.codex_compaction_cache = cfg.get("codex_compaction_cache") is True   # #924: opt-in, off by default
    try:
        svc.set_aliases(cfg.get("aliases"))             # #297: other names the model answers to
    except ValueError as e:
        raise SystemExit(f"[strata] config {e}")
    if svc.aliases:
        print(f"[strata] model aliases: {', '.join(svc.aliases)}", flush=True)
    svc.api_key = preflight_api_key
    try:
        svc.telemetry_key = engine_client.api_key_of(os.environ.get("STRATA_TELEMETRY_KEY")
                                                     or cfg.get("telemetry_read_key", ""))
    except ValueError as e:
        print(f"[strata] invalid telemetry_read_key: {e}", file=sys.stderr)
        return 2
    if set(engine_client.key_list(svc.api_key)) & set(engine_client.key_list(svc.telemetry_key)):
        print("[strata] telemetry_read_key must differ from every inference API key", file=sys.stderr)
        return 2
    if not svc.api_key and "api_key" in cfg and not no_key_chosen:             # #569: written, but empty: the server has none (a warning, not a stop)
        print("[strata] the api_key in the config is empty: this server has no API key (anyone who can reach it can "
              "use it); set one, or leave api_key out", file=sys.stderr, flush=True)
    svc.cors_origins = origins_of(cfg.get("cors_origins"), "cors_origins", wildcard=True)
    svc.trusted_origins = origins_of(cfg.get("trusted_origins"), "trusted_origins", wildcard=False)
    try:
        svc.allowed_hosts = allowed_hosts_of(cfg.get("allowed_hosts"), os.environ.get("STRATA_ALLOWED_HOSTS", ""))
    except ValueError as e:
        raise SystemExit(f"[strata] config {e}")
    if svc.allowed_hosts:
        print("[strata] Host check off: any name reaches this server (allowed_hosts \"*\")" if "*" in svc.allowed_hosts
              else f"[strata] also answers to the host names {', '.join(svc.allowed_hosts)} (allowed_hosts)", flush=True)
    svc.api_monitor = a.api_monitor or cfg.get("api_monitor") is True
    if svc.api_monitor:
        print("[strata] API request monitor on (/api-monitor): the last 100 requests' prompts and answers are kept in "
              "memory" + ("" if svc.api_key else "; anyone who can reach this server can read them (no API key)"),
              flush=True)
    if svc.cors_origins:
        print(f"[strata] CORS on /v1/* for {', '.join(svc.cors_origins)}"
              + ("" if svc.api_key or "*" not in svc.cors_origins else
                 " - WARNING: any web page may use the model (no API key)"), flush=True)
    svc.idle_unload_s = a.idle_unload if a.idle_unload is not None else float(cfg.get("idle_unload_s") or 0)
    svc.min_free_vram_mib = a.min_free_vram_mib if a.min_free_vram_mib is not None else \
        int(cfg.get("min_free_vram_mib") or 0)
    svc.before_load = a.before_load or cfg.get("before_load") or None
    try:
        svc.before_load_timeout_s = float(cfg.get("before_load_timeout_s") or svc.before_load_timeout_s)
    except (TypeError, ValueError):
        print("[strata] \"before_load_timeout_s\" in the config is not a number; using "
              f"{svc.before_load_timeout_s:g} s", flush=True)
    svc.before_load_ignore_failure = cfg.get("before_load_ignore_failure") is True
    svc.before_load_strict = cfg.get("before_load_strict") is True
    # --slot-save-path is relative to the working directory the server was started in; the config's slot_save_path
    # to the config's "cwd" (the engine's folder) when it has one.  Either way the engine gets an absolute path.
    if a.slot_save_path or cfg.get("slot_save_path"):
        try:
            svc.slot_save_path = slot_save_dir(a.slot_save_path) if a.slot_save_path else \
                slot_save_dir(cfg["slot_save_path"], cfg.get("cwd"))
        except (ValueError, OSError) as e:
            raise SystemExit(f"[strata] {e}")
        print(f"[strata] slot save/restore on: {svc.slot_save_path} (session files are kept until deleted: "
              "about 1.2 GB per 63K-token conversation; restore only files this server wrote)", flush=True)
    mode = str(cfg.get("anthropic_thinking") or "model")   # #278: "on_request" = only when the request asks
    if mode not in ("model", "on_request"):
        raise SystemExit(f"[strata] config anthropic_thinking must be \"model\" or \"on_request\", not {mode!r}")
    svc.anthropic_think_unasked = mode == "model"
    recovery = cfg.get("tool_call_recovery", False)     # opt-in: see OutputParser's recover
    if not isinstance(recovery, bool):
        raise SystemExit(f"[strata] config \"tool_call_recovery\" must be true or false, not {recovery!r}")
    svc.tool_call_recovery = recovery
    etu = cfg.get("env_to_user", False)                # opt-in: see frontend._env_to_user
    if not isinstance(etu, bool):
        raise SystemExit(f"[strata] config \"env_to_user\" must be true or false, not {etu!r}")
    import strata_server.frontend as _frontend
    _frontend.ENV_TO_USER = etu
    rs = cfg.get("repeat_stop_tokens", REPEAT_STOP_TOKENS)   # #606: opt-out with 0
    if isinstance(rs, bool) or not isinstance(rs, int) or rs < 0:
        raise SystemExit(f"[strata] config \"repeat_stop_tokens\" must be a whole number >= 0 (0 = off), not {rs!r}")
    svc.repeat_stop_tokens = rs
    svc.effort_end = bool(effort_end)                   # #458: "effort_position": "end" with an engine that has it
    if cfg.get("reasoning_budget_tokens") is not None:  # #123: a default thinking budget for every request
        svc.reasoning_budget_configured = True
        try:
            svc.reasoning_budget_tokens = cfg["reasoning_budget_tokens"]
            budget = svc.reasoning_budget({})
        except ValueError as e:
            raise SystemExit(f"[strata] config {e}")
        if budget:
            print(f"[strata] thinking budget: {budget} tokens (reasoning_budget_tokens; a request can set its own)",
                  flush=True)
    recovery = cfg.get("reasoning_loop_recovery", False)   # #728: false (default) | "stop" | "recover" (true)
    if recovery is True:
        recovery = "recover"
    if recovery is not False and recovery not in ("stop", "recover"):
        raise SystemExit("[strata] config \"reasoning_loop_recovery\" must be false, \"stop\" or \"recover\", "
                         f"not {recovery!r}")
    svc.reasoning_loop_recovery = recovery
    svc.gpu_index = (engine_client.gpu_list(cfg) or [0])[0]  # the Monitor reads the card the engine runs on (issue #51)
    svc.gpu_indices = engine_client.gpu_list(cfg)                     # ... or every card of a layer split (issue #112)
    if a.engine == "strata" and groups:                 # replicas: every card of every group
        svc.gpu_indices = [c for g in groups for c in g]
        svc.gpu_index = svc.gpu_indices[0]
        svc.replica_gpu_groups = groups
    svc.terminal_visibility_order = visible_device_order(env, cfg.get("backend")) if a.engine == "strata" else []
    has_layer_split = a.engine == "strata" and engine_client.layer_split_of(cfg)
    svc.terminal_topology_mode = ("replicas (configured)" if a.engine == "strata" and groups
                                  else "layer split (configured)" if has_layer_split
                                  else "single GPU or CPU" if a.engine == "strata" else "mock")
    svc.terminal_configured_split = engine_client.layer_split_value(cfg) if has_layer_split else None
    svc.backend = cfg.get("backend")                    # "hip": the AMD cards' readings come from sysfs (#301)
    if a.config:
        svc.config_path = a.config                      # #564: the web page's Settings view
    if a.config:                                        # the Chat settings shared with other apps, from last time
        svc.shared_path = str(Path(a.config).with_suffix("")) + ".shared-settings.json"
        try:
            svc.shared = clean_shared_defaults(json.loads(Path(svc.shared_path).read_text(encoding="utf-8")))
            if svc.shared:
                print("[strata] other apps use the Chat settings: " +
                      ", ".join(f"{k}={v}" for k, v in svc.shared.items()), flush=True)
        except (OSError, ValueError):
            svc.shared = {}
    if hub is not None:
        import atexit
        svc.mcp = hub
        print(f"[strata] starting {len(hub.servers)} MCP server{'s' * (len(hub.servers) != 1)} for the web app's "
              f"chat: {', '.join(hub.servers)}", flush=True)
        hub.start()
        atexit.register(hub.close)                      # the servers Strata started end with it
    httpd = serve(svc, host=a.host, port=a.port)
    bound_port = httpd.server_address[1]               # --port 0: report the port the OS actually assigned
    svc.start_idle_unload()
    svc.start_vision_idle()
    here = "127.0.0.1" if a.host in ("0.0.0.0", "", "::") else a.host
    print(f"ready: http://{here}:{bound_port}/v1  (OpenAI: /v1/chat/completions, Anthropic: /v1/messages, "
          f"context {engine.max_context} tokens{', images on' if vision else ''}"
          f"{', API key required' if svc.api_key else ''})", flush=True)
    print(f"       open http://{here}:{bound_port}/ in a browser to chat; close this window to stop the model", flush=True)
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        # issue #26: reachable from other devices - say at which address, and what can still block it
        ips = lan_addresses()
        for ip in ips:
            print(f"       from other devices: http://{ip}:{bound_port}/   (API: http://{ip}:{bound_port}/v1)", flush=True)
        if not ips:
            print("       from other devices: http://<this PC's IP address>:" + str(bound_port) + "/", flush=True)
        if not svc.api_key:
            print("       WARNING: " + open_server_warning(bound_port, tools=svc.mcp is not None),
                  flush=True)
        if os.name == "nt":
            print("       nothing arrives? Windows Firewall blocks it until allowed: accept its prompt for Python, or run "
                  "in an admin PowerShell:\n         New-NetFirewallRule -DisplayName \"Strata " + str(bound_port) + "\" "
                  "-Direction Inbound -Protocol TCP -LocalPort " + str(bound_port) + " -Action Allow -Profile Private\n"
                  "       (and set this network to Private in Windows' network settings)", flush=True)
    if a.open and cfg.get("open_browser") is not False:   # #609: the config's "open_browser": false wins (an older
        import webbrowser                                  # run-<model>.bat still passes --open)
        # on its own thread: webbrowser.open waits for the browser launcher, which on Linux can run for as long as
        # the browser does, and the main thread must be free to take Ctrl+C and SIGTERM (#1872)
        threading.Thread(target=webbrowser.open, daemon=True, name="strata-open-browser",
                         args=(f"http://{'127.0.0.1' if a.host in ('0.0.0.0', '') else a.host}:{bound_port}/",)).start()
    if ready_path is not None or event_sink is not None:
        ready_host = f"[{here}]" if ":" in here and not here.startswith("[") else here
        record = {"schema_version": 1, "instance_id": svc.instance_id, "pid": os.getpid(),
                  "api_url": f"http://{ready_host}:{bound_port}", "bound_host": a.host, "bound_port": bound_port,
                  "model_id": svc.model, "model_state": "unloaded" if lazy else "ready",
                  "api_state": "ready", "created_at": time.time()}
    if ready_path is not None:
        try:
            ready_file.publish(ready_path, record)
        except OSError as e:
            print(f"[strata] could not write ready file ({type(e).__name__})", file=sys.stderr)
            for close in (httpd.shutdown, getattr(engine, "close", None), vision.shutdown if vision else None,
                          hub.close if hub is not None else None):
                if close:
                    close()
            return 6
    if event_sink is not None:
        notify_lifecycle("ready", record)
    interrupted = False
    drained = True
    try:
        while True:
            time.sleep(1)                               # Windows never delivers Ctrl+C to an untimed Event.wait()
    except KeyboardInterrupt:
        interrupted = True
        notify_lifecycle("stopping", {"reason": termination, "instance_id": svc.instance_id})
        print("\n[strata] stopping (Ctrl+C again to end the engine at once) ...", flush=True)
    finally:
        cancelled = 0
        if a.shutdown_timeout is not None:
            pending = svc.begin_drain()
            notify_lifecycle("draining", {"instance_id": svc.instance_id,
                                          "pending_requests": pending,
                                          "deadline_s": a.shutdown_timeout})
            if pending:
                print(f"[strata] draining {pending} inference call(s) for up to {a.shutdown_timeout:g} s ...",
                      flush=True)
            pending = svc.wait_for_drain(a.shutdown_timeout)
            if pending:
                cancelled = svc.cancel_owned_requests()
                print(f"[strata] drain deadline expired; cancelling {cancelled} owned request(s) ...", flush=True)
                svc.wait_for_drain(5)
                drained = False
        closers = [httpd.shutdown, getattr(engine, "close", None), vision.shutdown if vision else None,
                   hub.close if hub is not None else None]
        try:
            for close in filter(None, closers):
                try:
                    close()
                except KeyboardInterrupt:               # a second Ctrl+C: don't wait for the engine to free its memory
                    if getattr(engine, "proc", None):
                        engine.proc.kill()
                        try:
                            from strata_server import cache_drop
                            cache_drop.drop_after_exit(engine)       # opt-in (STRATA_DROP_CACHE_ON_EXIT=1), default unchanged
                        except Exception as e:  # noqa: BLE001 - never block the exit
                            print(f"[strata] could not release the model files' cache: {e}", flush=True)
        finally:
            if ready_path is not None:
                ready_file.remove_if_owned(ready_path, svc.instance_id)
            finished.set()
        if interrupted:
            print("[strata] stopped", flush=True)
    exit_code = (8 if not drained else
                 interrupt_exit_code if interrupted and termination == "sigint" else 0)
    notify_lifecycle("stopped", {"instance_id": svc.instance_id, "reason": termination if interrupted else "normal",
                                 "drained": drained, "cancelled_requests": cancelled,
                                 "exit_code": exit_code})
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
