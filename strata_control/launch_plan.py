"""Starting a saved serving profile: which settings apply and what the server and engine would run.

Used by the `strata` command (strata_cli/cli.py) and by Control's runtime, management and benchmarking.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
from pathlib import Path

from strata_common import profiles
from strata_common.profiles import ProfileError


def read_source(ref: str) -> tuple[Path, dict | None, str | None, dict]:
    """Gets a saved serving profile's run-config path, saved profile (None for a plain run config), revision and run
    config."""
    path, saved, revision = profiles.resolve(ref)
    return path, saved, revision, profiles.read_run_config(path)


def redact_argv(args: list[str]) -> list[str]:
    """Gets a copy of a command line with the values of key, token, secret, password and credential options hidden."""
    out = []
    conceal_next = False
    for item in args:
        if conceal_next:
            out.append("<redacted>")
            conceal_next = False
            continue
        if item.startswith("-") and any(word in item.lower()
                                        for word in ("key", "token", "secret", "password", "credential")):
            if "=" in item:
                out.append(item.partition("=")[0] + "=<redacted>")
            else:
                out.append(item)
                conceal_next = True
        else:
            out.append(item)
    return out


def get_effective(cfg: dict, saved: dict | None, cli: argparse.Namespace | None = None) -> tuple[dict, dict]:
    """Gets the serving settings of a profile and where each came from (command, saved profile, run config,
    environment or default); raises ProfileError when they contradict each other or the run config."""
    overrides = saved["overrides"] if saved else {}
    values = {}
    origins = {}
    for key, default in (("host", "127.0.0.1"), ("port", 8095), ("parallel", None),
                         ("batch_groups", None), ("max_queue", None), ("queue_timeout", None),
                         ("offline", False), ("batch_telemetry", False)):
        explicit = getattr(cli, key, None) if cli is not None else None
        if key == "offline" and explicit is False:
            explicit = None
        if explicit is not None:
            value, origin = explicit, "command"
        elif key in overrides:
            value, origin = overrides[key], "saved_profile"
        elif key in cfg:
            value, origin = cfg[key], "run_config"
        elif key == "batch_telemetry" and "STRATA_BATCH_TELEMETRY" in os.environ:
            value, origin = os.environ["STRATA_BATCH_TELEMETRY"] == "1", "environment"
        else:
            value, origin = default, "default"
        values[key], origins[key] = value, origin
    profiles.validate_overrides({key: value for key, value in values.items() if key in
                                  ("host", "port", "parallel", "batch_groups", "max_queue", "queue_timeout",
                                   "offline", "batch_telemetry")
                                  and value is not None})
    if values["queue_timeout"] is not None and values["max_queue"] is None:
        raise ProfileError("queue_timeout requires max_queue to bound pending calls")
    if values["parallel"] is not None and any(flag in cfg["args"] for flag in ("--batch", "--slots")):
        if origins["parallel"] != "run_config":
            raise ProfileError("parallel override conflicts with --batch/--slots in run config args")
    if values["batch_groups"] is not None and "--batch-groups" in cfg["args"]:
        raise ProfileError("batch_groups override conflicts with --batch-groups in run config args")
    if values["offline"] and (cfg.get("mcp_servers") or cfg.get("before_load")):
        raise ProfileError("offline serving requires MCP servers and before_load to be absent")
    return values, origins


def make_launch_plan(ref: str, cli: argparse.Namespace | None = None) -> tuple[dict, list[str]]:
    """Gets what starting a saved serving profile would run: the plan shown to the person (effective settings,
    engine command line with secrets hidden) and the server's arguments."""
    path, saved, revision, cfg = read_source(ref)
    values, origins = get_effective(cfg, saved, cli)
    host = values["host"]
    from strata_common.engine_client import api_key_of, engine_args, explicit_no_key, key_list
    raw_key = os.environ.get("STRATA_API_KEY") or cfg.get("api_key", "")
    if not explicit_no_key(raw_key):
        try:
            key_list(api_key_of(raw_key))
        except ValueError:
            raise ProfileError("the inference API key is invalid", 5) from None
    # A key is recommended on a network, never required: the server warns once at start (owner decision 2026-10-20).
    server_args = ["--engine", getattr(cli, "engine", "strata"), "--config", str(path), "--host", str(host),
                   "--port", str(values["port"])]
    if cli is not None:
        server_args += ["--shutdown-timeout", str(cli.shutdown_timeout)]
    for key, flag in (("parallel", "--parallel"), ("batch_groups", "--batch-groups"),
                      ("max_queue", "--max-queue"), ("queue_timeout", "--queue-timeout")):
        if values[key] is not None and origins[key] != "run_config":
            server_args += [flag, str(values[key])]
    if values["offline"]:
        server_args.append("--offline")
    if origins["batch_telemetry"] not in ("run_config", "default"):
        server_args.append("--batch-telemetry" if values["batch_telemetry"] else "--no-batch-telemetry")
    ready_path = None
    if cli is not None and cli.ready_file:
        from strata_common.ready import preflight as ready_preflight
        ready_path = str(ready_preflight(cli.ready_file))
        server_args += ["--ready-file", ready_path]
    effective_cfg = {**cfg, "args": list(cfg["args"])}
    if values["parallel"] is not None:
        effective_cfg["parallel"] = values["parallel"]
    if values["batch_groups"] is not None:
        effective_cfg["args"] += ["--batch-groups", str(values["batch_groups"])]
    notes = io.StringIO()
    with contextlib.redirect_stdout(notes):
        engine_argv = engine_args(effective_cfg)
    plan = {"profile": saved["id"] if saved else None, "profile_revision": revision,
            "source_config": str(path), "model_name": cfg.get("model_name"),
            "effective": values, "origin": origins, "server_argv": server_args,
            "ready_file": ready_path,
            "engine_exe": cfg["exe"], "engine_argv": redact_argv(engine_argv),
            "notes": [line for line in notes.getvalue().splitlines() if line]}
    return plan, server_args
