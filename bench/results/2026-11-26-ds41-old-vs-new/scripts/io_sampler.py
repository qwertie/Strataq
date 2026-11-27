"""Samples a running strata-ds41's read bytes (Windows' per-process I/O counters: SSD and file reads, unbuffered
included) every 0.5 s with the size of its stdout log, so reads can be assigned to the commands the log shows.

    python io_sampler.py LOG.out OUT.jsonl
"""
import json, os, sys, time
import psutil
log, out = sys.argv[1], sys.argv[2]
p = next(q for q in psutil.process_iter(["name"]) if q.info["name"] == "strata-ds41.exe")
with open(out, "a") as f:
    while p.is_running():
        try:
            io = p.io_counters()
        except psutil.Error:
            break
        f.write(json.dumps({"t": time.time(), "read_bytes": io.read_bytes, "read_count": io.read_count,
                            "log_bytes": os.path.getsize(log)}) + "\n")
        f.flush()
        time.sleep(0.5)
