"""Starts strata-dsv41 with each DS4_COMMIT_CAP_GB given and prints its GPU expert slots and RAM store, then quits."""
import os, re, subprocess, sys, shutil, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ds_h2h as h
mode = sys.argv[1]
for cap in sys.argv[2:]:
    pack = os.path.abspath(f"calib/old-cap{cap}-{mode}-pack")
    shutil.rmtree(pack, ignore_errors=True)
    shutil.copytree(h.PACK, pack, ignore=shutil.ignore_patterns("*.tmp"))
    env = dict(os.environ, **h.ENV, DS4_COMMIT_CAP_GB=cap)
    env["PATH"] = h.LIB + os.pathsep + env["PATH"]
    args = ["--serve", "--pack", pack, *h.COMMON, "--dspark", "on" if mode == "on" else "none",
            "--ssd-cache", f"{36864 * 1048576 / 1e9:.9f}", "--vram-experts", "1", "--dspark-depth", "7"]
    err = pack + ".err"
    p = subprocess.Popen([h.exe_of("old"), *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=open(err, "w"), text=True, cwd=h.ROOT, env=env)
    slots = None
    for line in p.stdout:
        if m := re.search(r"expert_slots=(\d+)", line):
            slots = int(m.group(1))
        if line.startswith(("READY", "ERR")):
            break
    p.stdin.write("QUIT\n"); p.stdin.close(); p.wait(300)
    store = [l.strip() for l in open(err) if "RAM store: up to" in l or "experts (" in l and "in VRAM" in l]
    print(cap, slots, store, flush=True)
    time.sleep(5)
