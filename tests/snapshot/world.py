"""A scratch world for one (machine class, model set): files, a stand-in for setup's output, and the patches that make the probes answer as that machine.

Everything lives under one temporary folder that is also HOME, the data folder, the install folder and every per-user folder, so nothing real is read or written.
No model file is needed (the model files are empty), no engine runs (the engine is a text stub that carries the strings the program's own checks read), nothing is
downloaded, and nothing touches the network.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

from .matrix import MACHINES, Machine, ModelSet

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: The engine stub: a text file in place of the engine program (it is never started).
ENGINE_STUB = b"STRATA ENGINE STUB (config snapshot)\n"

#: Environment variables a developer or CI may have set that would leak into a start; the world starts from a clean list.
CLEAR = ("STRATA_SETTINGS_FILE", "STRATA_CONNECTION_DIR", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "STRATA_API_KEY",
         "STRATA_ENGINE_BACKEND", "STRATA_UPDATE_FEED", "STRATA_NO_ACQUIRE", "STRATA_COMPONENTS_DIR")


def context_for(machine: Machine, model_set: ModelSet) -> int:
    if model_set.multi_file:
        return 131072
    if machine.key == "nv_12gb_streaming":
        return 65536
    return 131072


class World:
    def __init__(self, machine: Machine, mset: ModelSet, root: Path | None = None):
        self._tmp = None
        if root is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="strata-snap-")
            root = Path(self._tmp.name)
        self.machine, self.mset, self.root = machine, mset, Path(root).resolve()
        r = self.root
        self.home, self.install, self.data = r / "home", r / "install", r / "data"
        self.engine_dir = r / "engine"
        self.models = self.data / "models"
        self.pack = self.data / "packs" / mset.key
        self.mtp = self.data / "mtp" / "rt"
        self.gguf_dir = self.models / mset.model
        self.exe = self.engine_dir / ("strata-ds41" if mset.multi_file else "strata-q38")
        self.tag = (("deepseek-v4.1-flash-" if mset.multi_file else {"qwen": "", "unsloth": "unsloth-"}[mset.family]) + mset.model).lower()
        self.config_path = self.install / f"strata-{self.tag}.json"
        self.env = {
            "HOME": str(self.home), "USERPROFILE": str(self.home),
            "APPDATA": str(r / "roaming"), "LOCALAPPDATA": str(r / "local"),
            "XDG_CONFIG_HOME": str(r / "xdg-config"), "XDG_STATE_HOME": str(r / "state"), "XDG_DATA_HOME": str(r / "xdg-data"),
            "STRATA_PROFILE_DIR": str(r / "config" / "profiles"), "STRATA_PRESET_DIR": str(r / "config" / "presets"),
            "STRATA_LAUNCH_PRESET_DIR": str(r / "config" / "launch-presets"),
            "STRATA_INSTALL_DIR": str(self.install), "STRATA_RUNTIME_DIR": str(r / "runtime"),
            "TMPDIR": str(r / "tmp"), "TEMP": str(r / "tmp"), "TMP": str(r / "tmp"),
            "STRATA_SETTINGS_FILE": str(r / "settings.json"),
        }

    # ------------------------------------------------------------------ files
    def build(self) -> "World":
        for d in (self.home, self.install, self.data, self.engine_dir, self.models, self.gguf_dir, self.pack / "tokenizer", self.mtp,
                  self.root / "tmp", self.root / "roaming", self.root / "local", self.root / "xdg-config", self.root / "state", self.root / "runtime"):
            d.mkdir(parents=True, exist_ok=True)
        self.exe.write_bytes(ENGINE_STUB)
        for shard in self.mset.shards:
            (self.gguf_dir / shard).write_bytes(b"GGUF")
        (self.pack / "native_experts.txt").write_text("x")
        (self.mtp / "experts.bin").write_bytes(b"e")
        (self.mtp / "draft_vocab.bin").write_bytes(b"v")
        (self.data / "mtp" / "mtp-q2_0.gguf").write_bytes(b"m")
        (self.root / "settings.json").write_text(json.dumps({"data_dir": str(self.data), "installs": [str(self.install)]}), encoding="utf-8")
        self.config_path.write_text(json.dumps(self.base_config(), indent=1), encoding="utf-8")
        return self

    def close(self) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()

    # ------------------------------------------------------------------ the stand-in for setup.py's output
    def base_config(self) -> dict:
        """What setup.py writes for this class and set, rebuilt from setup.py's own rules (the qwen block around line 10975-11200 and dsv41 around 9498).
        It is a STAND-IN: setup.py itself cannot be driven without a model download, an engine and a GPU probe (see RESULTS-STEP0.md), so the launch
        entry point starts from this file."""
        m, s = self.machine, self.mset
        ctx = context_for(m, s)
        if s.multi_file:
            args = ["--pack", str(self.pack), "--max-context", str(ctx), "--eos", "1", "--vision", "on",
                    "--dspark", "auto", "--dspark-conf", "0.65"]
            cfg = {"exe": str(self.exe), "args": args, "cwd": str(ROOT), "tokenizer": str(self.pack / "tokenizer"),
                   "model_name": "deepseek-v4.1-flash", "arch": "deepseek41", "log": str(self.install / f"strata-{self.tag}.log"),
                   "lib_dirs": [], "port": 8080, "engine_ready_s": 3600}
            if m.unified and m.backend == "hip":
                cfg["env"] = {"DS4_ASYNC_T": "4"}
        else:
            shard1 = self.gguf_dir / s.shards[0]
            ple = self.gguf_dir / s.shards[min(1, len(s.shards) - 1)]
            args = ["--pack", str(self.pack), "--native", str(shard1)]
            if len(s.shards) <= 2:
                args += ["--ple-gguf", str(ple)]
            args += ["--expert-profile", str(ROOT / "data" / "expert-profile.bin"), "--expert-cache", "auto", "--prefill", "auto", "--spec", "4",
                     "--spec-min-p", "0.5", "--mtp", str(self.mtp), "--max-context", str(ctx)]
            if ctx > 8192:
                args += ["--kv", "int8"]
            if ctx >= 65536 and m.key == "nv_12gb_streaming":
                args += ["--kv-resident", "32768"]
            if s.family == "unsloth":
                args += ["--resident-budget-gib", "24" if m.ram_gb < 100 else "48"]
            cfg = {"exe": str(self.exe), "args": args, "cwd": str(ROOT), "tokenizer": str(self.pack / "tokenizer"),
                   "model_name": s.model_name, "log": str(self.install / f"strata-{self.tag}.log"), "lib_dirs": [], "port": 8080}
        if m.backend == "hip":
            cfg["backend"] = m.backend
        if m.cards > 1:
            cfg["gpu"] = list(range(m.cards))
        return cfg

    # ------------------------------------------------------------------ what the probes answer
    def hardware_report(self) -> dict:
        """The hardware report the Terminal's catalog reads (the shape tests/terminal/launch_support.py uses)."""
        m = self.machine
        card = {"index": 0, "name": m.card, "vendor": m.vendor, "vram_gb": m.vram_gb, "usable": True}
        if m.arch_field is not None:
            card["arch"] = m.arch_field
        if m.unified:
            card["unified_gb"] = m.ram_gb
            card["uma"] = True
        hw = {"ram_gb": m.ram_gb, "os": {"windows": "Windows 11", "linux": "Linux"}[m.os], "disk_free_gb": 2000.0,
              "cpu": {"avx2": m.machine in ("x86_64",) or None, "name": "snapshot cpu", "cores": 16}, "gpus": [card], "engine_vendor": m.vendor}
        return hw

    def vram_mib(self) -> float | None:
        """What the telemetry reader would answer for the ONE card (None for a shared-memory chip)."""
        return self.machine.vram_mib or None

    @contextlib.contextmanager
    def patched(self):
        """The Python side of the machine class: engine family, cards, card memory, platform, environment.  Nothing else is replaced."""
        from strata_common import engine_component
        m = self.machine
        archs = list(m.archs)
        vram = self.vram_mib()
        saved_env = dict(os.environ)

        class Reader:
            def ok(self_):                                       # noqa: N805
                return vram is not None

            def read(self_):                                     # noqa: N805
                return {"mem_total": int(vram * (1 << 20)) if vram else 0}

        with contextlib.ExitStack() as st:
            st.enter_context(mock.patch.dict(os.environ, self.env))
            for k in CLEAR:
                if k not in self.env:
                    os.environ.pop(k, None)
            os.environ["PATH"] = str(self.engine_dir)           # nothing on PATH is probed; the engine is named by the config
            st.enter_context(mock.patch.object(engine_component, "detect_backend", lambda: (m.backend, "snapshot")))
            st.enter_context(mock.patch.object(engine_component, "cuda_archs_present", lambda: archs if m.backend == "cuda" else []))
            st.enter_context(mock.patch.object(engine_component, "hip_archs_present", lambda: archs if m.backend == "hip" else []))
            with contextlib.suppress(ImportError):
                from strata_common import telemetry
                st.enter_context(mock.patch.object(telemetry, "gpu_reader", lambda *a, **k: Reader()))
            try:
                from strata_common import engine_client as server
                st.enter_context(mock.patch.object(server, "gpu_vram_mib", lambda gl, hip=False: {} if m.unified or vram is None else {int((gl or [0])[0]): vram}))
            except ImportError:                                  # serve needs its packages: a world without them still plans
                pass
            yield self
        os.environ.clear()
        os.environ.update(saved_env)


def worlds(machine_keys=None, set_keys=None):
    for m in MACHINES:
        if machine_keys and m.key not in machine_keys:
            continue
        from .matrix import MODEL_SETS
        for s in MODEL_SETS:
            if set_keys and s.key not in set_keys:
                continue
            yield m, s
