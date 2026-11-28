"""Drive each entry point for one (machine class, model set) and return what the engine would be started with, normalized.

A record has the same shape for every entry point:

  entry            which one
  command          what the person asked for, in words
  setup_args       what the entry point hands to setup.py (Terminal launch only)
  config_file      the run config the entry point leaves on disk (normalized)
  engine           {exe, argv, env_set, notes}: what serve/server.py would start: engine_args(cfg) and the variables child_env(cfg) adds
"""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
from unittest import mock

from .matrix import ModelSet
from .normalize import normalize
from .world import World, context_for


def run_cli(argv: list[str]) -> tuple[int, str, str]:
    from strata_cli.cli import main
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = main(argv)
        except SystemExit as e:
            code = int(e.code or 0) if not isinstance(e.code, str) else 1
    return code, out.getvalue(), err.getvalue()

def engine_start(world: World, cfg: dict) -> dict:
    """What serve/server.py hands to the engine for this run config: the arguments (engine_args) and the variables child_env adds to the environment."""
    from strata_common import engine_client as server
    baseline = dict(os.environ)
    notes = io.StringIO()
    with contextlib.redirect_stdout(notes):
        argv = server.engine_args(dict(cfg))
        env = server.child_env(dict(cfg))
    env_set = {k: v for k, v in sorted(env.items()) if baseline.get(k) != v}
    return {"exe": cfg["exe"], "argv": [str(a) for a in argv], "env_set": env_set,
            "notes": [l for l in notes.getvalue().splitlines() if l.strip()]}


def drive_launch(world: World) -> dict:
    """`strata launch --model <set> --yes --no-tui` through the real command, with the engine start, the model download and the tuning replaced."""
    from strata_cli.launch import catalog as C, runner as R
    seen: dict = {}
    cat = C.Catalog(root=world.install, hardware=world.hardware_report())

    def ensure_model(self):
        seen["setup_args"] = self.setup_args()
        return world.config_path

    with mock.patch.object(C, "Catalog", lambda *a, **k: cat), \
            mock.patch.object(R.Runner, "ensure_engine", lambda self: None), \
            mock.patch.object(R.Runner, "ensure_model", ensure_model), \
            mock.patch.object(R.Runner, "tune", lambda self: None), \
            mock.patch.object(R.Runner, "start", lambda self: seen.setdefault("started", True) and None), \
            mock.patch.object(R.Runner, "open_access", lambda self: None), \
            mock.patch.object(R.Runner, "check_serving", lambda self: None):
        command = ["launch", "--model", world.mset.model, "--yes", "--no-tui", "--context", str(context_for(world.machine, world.mset))]
        orig_run = R.Runner.run

        def run(self, **kw):
            seen["runner"] = self
            return orig_run(self, **kw)

        with mock.patch.object(R.Runner, "run", run):
            code, out, err = run_cli(command)
    runner = seen.get("runner")
    if code != 0 or runner is None or runner.config_path is None:
        return {"entry": "launch", "command": "strata " + " ".join(command), "failed": {"exit": code, "stderr": err[-400:], "stdout": out[-400:]}}
    cfg = json.loads(Path(runner.config_path).read_text(encoding="utf-8"))
    return {"entry": "launch", "command": "strata " + " ".join(command),
            "setup_args": seen.get("setup_args"), "config_file": cfg, "engine": engine_start(world, cfg)}


def drive(machine_key: str, set_key: str, entry: str, root: Path | None = None) -> dict:
    """One record, normalized.  A fresh scratch world every time."""
    from .matrix import machine as get_machine, model_set as get_set
    world = World(get_machine(machine_key), get_set(set_key), root).build()
    try:
        if entry == "setup":
            from .setup_driver import drive_setup
            rec = drive_setup(world)
        elif entry == "launch":
            with world.patched():
                rec = drive_launch(world)
        else:
            raise ValueError(entry)
        return normalize(rec, world)
    finally:
        world.close()
