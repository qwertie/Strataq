#!/usr/bin/env python3
"""Builds Strata's Windows AMD (HIP) engines for a release with setup.py's own HIP build (its toolchain: AMD's TheRock
ROCm wheels, MSVC's headers and linker, ROCm's clang for host and device code), then packages the Strata engine as
dist/strata-windows-x64-hip.zip (tools/hip/package_windows.py).

    .venv\\Scripts\\python.exe tools\\hip\\build_release.py [--tests]

Run it with Strata's own Python (the .venv START-HERE.bat makes): ROCm's wheels (~10 GB) are installed into that
environment unless ROCM_PATH names a TheRock root you already have (`rocm-sdk path --root`).  --tests also builds the
HIP unit tests (ctest in the build folder; they need an AMD GPU).  The build folder also holds strata-dsv41.exe and
strata-ds41.exe for the DeepSeek package (packaging/README.md).  Settings (environment variables, all optional):
  STRATA_HIP_ARCHS   gfx1100;gfx1101;gfx1102;gfx1200;gfx1201;gfx1030;gfx1031;gfx1151 (gfx1151 = Strix Halo).
                     The release packages are built one per card family: rx9070 = gfx1200;gfx1201, rdna3 = gfx1100;gfx1101;gfx1102;
                     gfx1151, rdna2 = gfx1030;gfx1031; each then goes through package_engine.py --variant <name>
  STRATA_ROCM_VERSION  another TheRock version than setup.py's WIN_ROCM_VERSION
  BUILD_DIR          <repo>\\build-hip-win        DIST_DIR   <repo>\\dist
  STRATA_NO_VISION   1 = without the two picture readers (strata-vision.exe for the CPU, strata-vision-hip.exe for
                     the GPU)
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

ARCHS = "gfx1100;gfx1101;gfx1102;gfx1200;gfx1201;gfx1030;gfx1031;gfx1151"
PROGRAMS = ("strata.exe", "strata-device.exe", setup.Q38EXE, setup.DS41EXE, setup.DSV41EXE)


def main() -> int:
    """Builds the Windows release programs and readers, then collects their runtime files into a zip."""
    archs = os.environ.get("STRATA_HIP_ARCHS", ARCHS).split(";")
    bdir = Path(os.environ.get("BUILD_DIR") or ROOT / "build-hip-win")
    dist = Path(os.environ.get("DIST_DIR") or ROOT / "dist")
    tests = "--tests" in sys.argv[1:]
    llama = setup.get_llama_cpp()
    root, dirs = setup.use_rocm(archs)
    setup.compile_hip(ROOT, bdir, "all" if tests else setup.cmake_targets(PROGRAMS), root, dirs,
                      [*setup.hip_engine_defs(archs, llama, PROGRAMS), "-DSTRATA_PORTABLE=ON",
                       *(["-DSTRATA_BUILD_TESTS=ON"] if tests else [])], "build-strata-hip-release.bat")
    readers = []
    if os.environ.get("STRATA_NO_VISION") != "1":
        setup.compile_cpu_reader(ROOT / "build-vision-cpu-win", llama, portable=True)
        setup.compile_hip_reader(ROOT / "build-vision-hip-win", archs, root, dirs, llama, portable=True)
        readers = ["--vision-cpu", str(ROOT / "build-vision-cpu-win" / "bin" / setup.VEXE),
                   "--vision-hip", str(ROOT / "build-vision-hip-win" / "bin" / setup.HVEXE)]
    # The builder's vcvars runs in a child shell; packaging also needs its C++ redistributable location.
    vcvars = setup.find_vcvars()
    # cmd.exe needs literal quotes; Python's argument-list quoting escapes them for C runtimes instead.
    environment = subprocess.check_output(f'cmd /d /c call "{vcvars}" >nul && set', text=True)
    package_env = dict(line.split("=", 1) for line in environment.splitlines()
                       if "=" in line and not line.startswith("="))
    subprocess.run([sys.executable, str(ROOT / "tools" / "hip" / "package_windows.py"), "--build", str(bdir),
                    "--rocm", str(root), "--archs", ";".join(archs),
                    "--rocm-version", setup.ROCM_VERSION_OVERRIDE or setup.WIN_ROCM_VERSION, "--out", str(dist),
                    *readers], check=True, env=package_env)
    print(f"Done: {dist / setup.WIN_HIP_ASSET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
