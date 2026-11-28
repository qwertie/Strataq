"""An Auto start that ended before the engine was ready because a card ran out of memory is started once more without a
drafter, in one quiet line. A real process and a real pipe (a fake engine program), like strata_server/test_fatal_recovery.py."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

from strata_common.engine_client import EngineDied, StrataEngine
from strata_common import drafting

FAKE = r'''
import sys
from pathlib import Path
args = sys.argv[1:]
Path(args[args.index('--starts') + 1]).open('a').write(' '.join(args) + '\n')
if '--die-short' in args and '--mtp' in args:
    sys.stderr.write('ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected'+chr(10))
    sys.stderr.write('strata generate: layer split, CUDA1 expert cache: no room'+chr(10))
    sys.stderr.flush()
    raise SystemExit(1)
if '--die-always' in args:
    sys.stderr.write('strata generate: something else is wrong\n')
    sys.stderr.flush()
    raise SystemExit(2)
print('READY 4096 stop', flush=True)
for line in sys.stdin:
    if line.startswith('QUIT'):
        break
'''


class AutoRetriesWithoutADrafter(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        py = self.root / "native.py"
        if os.name == "nt":
            py.write_text(FAKE)
            self.script = self.root / "native.cmd"
            self.script.write_text(f'@"{sys.executable}" "{py}" %*\r\n')
        else:
            self.script = py
            py.write_text("#!" + sys.executable + "\n" + FAKE)
            py.chmod(0o700)
        self.starts = self.root / "starts.txt"
        self.log = str(self.root / "engine.log")
        self.first = ["--pack", "/d/packs/g", "--starts", str(self.starts), "--die-short", "--mtp", "/d/mtp/rt", "--spec", "4"]
        self.nodraft = ["--pack", "/d/packs/g", "--starts", str(self.starts), "--die-short"]
        drafting.take_note()
        self.addCleanup(drafting.take_note)

    def started(self):
        return self.starts.read_text().splitlines() if self.starts.exists() else []

    def test_a_start_that_ran_out_of_memory_is_repeated_with_no_drafter(self):
        engine = StrataEngine(str(self.script), self.first, log=self.log, retry_vram_args=self.nodraft)
        self.addCleanup(engine.close)
        self.assertTrue(engine.alive())
        starts = self.started()
        self.assertEqual(len(starts), 2, "one start with the MTP, one without a drafter")
        self.assertNotIn("--mtp", starts[1])
        self.assertEqual(engine.spawn[1], self.nodraft, "a later restart of this engine uses the arguments that worked")
        self.assertEqual(engine.info["drafting"], "off")
        self.assertEqual(engine.info["drafting_note"], "Drafting: off (the cards were short of memory)")
        text = Path(self.log).read_text(encoding="utf-8")
        self.assertIn("short of memory", text)
        self.assertIn("expert cache: no room", text, "the engine's own reason stays in the log")

    def test_a_memory_stop_without_a_retry_fails_with_the_engines_reason(self):
        with self.assertRaises(EngineDied) as caught:
            StrataEngine(str(self.script), self.first, log=self.log)
        self.assertIn("expert cache: no room", str(caught.exception))
        self.assertEqual(len(self.started()), 1)

    def test_another_stop_is_not_retried(self):
        with self.assertRaises(EngineDied) as caught:
            StrataEngine(str(self.script), ["--starts", str(self.starts), "--mtp", "/d/mtp/rt", "--die-always"], log=self.log,
                         retry_vram_args=self.nodraft)
        self.assertEqual(len(self.started()), 1)
        self.assertIn("exited before it was ready", str(caught.exception))

    def test_a_start_that_works_is_not_started_twice(self):
        ok = ["--pack", "/d/packs/g", "--starts", str(self.starts), "--mtp", "/d/mtp/rt"]
        engine = StrataEngine(str(self.script), ok, log=self.log, retry_vram_args=self.nodraft)
        self.addCleanup(engine.close)
        self.assertEqual(len(self.started()), 1)
        self.assertEqual(engine.info.get("drafting_note", ""), "")
        self.assertEqual(engine.info["drafting"], "mtp")


if __name__ == "__main__":
    unittest.main()
