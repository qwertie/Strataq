#!/usr/bin/env python3
"""End-to-end agent bridge test: Strata API worker -> DeepSeek Harness -> durable session.

This exercises the integration that the handoff calls the difficult part, without needing model
weights. It proves the *bridge*: that the pinned Harness boots with the Strata profile, that an
agent turn reaches the local Strata API worker, that the assistant message records the Strata
provider and model as its source, and that the turn reaches a terminal state in a durable session.

It does NOT prove inference quality. The worker runs its built-in `MockEngine`, because no Strata
pack fits this development Mac (see history/docs/work/IMPLEMENTATION_STATUS.md section 3.2). Every reported
fact is about the path, not the model; the qualification kit replaces the mock engine with a real
one on a machine that can hold a pack.

Usage:
    python integrations/harness/bridge_e2e.py [--dsh /path/to/dsh] [--port 8095] [--keep]

Exit codes:
    0  the turn completed and the assistant message names the Strata provider and model
    1  the bridge failed; the durable session events are printed to help diagnose it
    2  a prerequisite was missing (the dsh CLI, the worker, or the profile)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bridge import (  # noqa: E402
    DEFAULT_DSH,
    DEFAULT_PYTHON,
    PROFILE_PATCH,
    WORKSPACE_MARKER,
    BridgeError,
    gateway_call,
    http_json,
    log,
    read_session_events,
    start_agent,
    start_worker,
    wait_for_agent_url,
    wait_for_worker,
)

PROMPT = "Say hello in exactly five words."


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsh", default=os.environ.get("DSH_CLI", DEFAULT_DSH))
    ap.add_argument("--python", default=str(DEFAULT_PYTHON))
    ap.add_argument("--worker-port", type=int, default=8095)
    ap.add_argument("--agent-port", type=int, default=0, help="0 lets the OS choose")
    ap.add_argument("--keep", action="store_true", help="leave the processes and temp dir in place")
    args = ap.parse_args()

    dsh = Path(args.dsh)
    python = Path(args.python)
    if not dsh.exists():
        print(f"prerequisite missing: the dsh CLI is not at {dsh}", file=sys.stderr)
        return 2
    if not python.exists():
        print(f"prerequisite missing: no interpreter at {python}", file=sys.stderr)
        return 2
    if not PROFILE_PATCH.exists():
        print(f"prerequisite missing: the Strata profile patch is not at {PROFILE_PATCH}", file=sys.stderr)
        return 2

    tmp = Path(tempfile.mkdtemp(prefix=f"{WORKSPACE_MARKER}-"))
    (tmp / "ws").mkdir()
    worker = agent = None
    failure: str | None = None
    session_id = None

    print("strata agent bridge — end to end")
    print(f"  profile : {PROFILE_PATCH}")
    print(f"  dsh     : {dsh}")
    print(f"  temp    : {tmp}")
    print()

    try:
        print("[1/5] Strata API worker")
        worker, worker_log = start_worker(python, args.worker_port, tmp)
        health = wait_for_worker(args.worker_port)
        log(f"ready: model={health.get('model')} context={health.get('max_context')}")

        print("[2/5] DeepSeek Harness with the Strata profile")
        agent, dsh_log = start_agent(dsh, args.agent_port, tmp)
        text = wait_for_agent_url(dsh_log, agent)
        m = re.search(r"http://127\.0\.0\.1:(\d+)/\?token=([A-Za-z0-9_-]+)", text)
        assert m, "URL was matched by the caller but not re-matched here"
        agent_port, token = int(m.group(1)), m.group(2)
        log(f"ready on 127.0.0.1:{agent_port}")

        print("[3/5] Operator session with approval answering enabled")
        # `interactions=answer` is required: it defaults to 'none', and a peer without it cannot
        # answer approvals, so every ask would fail closed as 'unavailable'.
        _, _, set_cookie = http_json("GET", "127.0.0.1", agent_port,
                                     f"/?token={token}&interactions=answer")
        cookie = set_cookie.split(";", 1)[0]
        if not cookie.startswith("dsh-auth-"):
            raise BridgeError(f"no operator cookie was issued (got {set_cookie[:80]!r})")
        log("authenticated as owner with interactions=answer")

        print("[4/5] Session, then one agent turn")
        created = gateway_call("127.0.0.1", agent_port, cookie, "c1", "session/create",
                               {"request": {"cwd": str(tmp / "ws")}})
        session_id = created["sessionId"]
        log(f"session {session_id}")
        gateway_call("127.0.0.1", agent_port, cookie, "p1", "session/prompt",
                     {"request": {"requestId": "req-1", "sessionId": session_id,
                                  "mode": "queue",
                                  "content": [{"type": "text", "text": PROMPT}]}})
        log("prompt accepted")

        print("[5/5] Waiting for the turn to reach a terminal state")
        events = []
        deadline = time.time() + 120
        while time.time() < deadline:
            events = read_session_events(session_id)
            if any(e.get("type") == "turn/end" for e in events):
                break
            time.sleep(1.0)

        turn_end = next((e for e in events if e.get("type") == "turn/end"), None)
        if turn_end is None:
            raise BridgeError("the turn never reached turn/end within 120s")
        reason = (turn_end.get("data") or {}).get("reason") or {}
        if reason.get("kind") != "completed":
            raise BridgeError(f"the turn ended as {json.dumps(reason)[:400]}")

        assistant = next((e for e in events if e.get("type") == "assistant/message"), None)
        if assistant is None:
            raise BridgeError("the turn completed but committed no assistant message")
        source = ((assistant.get("data") or {}).get("message") or {}).get("source") or {}
        if source.get("provider") != "strata-local":
            raise BridgeError(f"the assistant message names provider {source.get('provider')!r}, "
                              f"not 'strata-local'")

        # Independent confirmation that the request actually reached the worker, not just that the
        # session recorded a plausible-looking source.
        _, raw, _ = http_json("GET", "127.0.0.1", args.worker_port, "/v1/status")
        activity = (json.loads(raw).get("activity") or {})
        requests = activity.get("requests") or 0
        if requests < 1:
            raise BridgeError("the session completed but the Strata worker recorded no requests")

        print()
        print("PASS")
        print(f"  turn            : {reason.get('kind')}")
        print(f"  provider/model  : {source.get('provider')} / {source.get('model')}")
        print(f"  worker requests : {requests}")
        print(f"  turn statements : {', '.join(e.get('type','?') for e in events if e.get('type','').startswith('turn/'))}")
        return 0

    except BridgeError as e:
        failure = str(e)
    except Exception as e:  # noqa: BLE001 - a test harness reports, it does not crash
        failure = f"{type(e).__name__}: {e}"
    finally:
        if not args.keep:
            for proc in (agent, worker):
                if proc and proc.poll() is None:
                    try:
                        proc.send_signal(signal.SIGTERM)
                        proc.wait(timeout=8)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
        else:
            print(f"\nkeeping processes; temp dir: {tmp}")

    print()
    print("FAIL")
    print(f"  {failure}")
    if session_id:
        evs = read_session_events(session_id)
        print(f"  durable session events ({len(evs)}):")
        for e in evs[-18:]:
            t = e.get("type")
            d = e.get("data") or {}
            extra = ""
            if t == "turn/end":
                extra = json.dumps(d.get("reason"))[:220]
            elif t == "assistant/message":
                extra = json.dumps((d.get("message") or {}).get("source"))[:160]
            print(f"    {e.get('seq')} {t} {extra}")
    if not args.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        print(f"  logs left in {tmp}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
