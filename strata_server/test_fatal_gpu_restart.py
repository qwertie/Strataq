"""strata_server/test_fatal_gpu_restart.py: #4350.  After a fatal GPU error the DeepSeek engine ends itself (ERR fatal GPU error: ...) and the
server starts it again on the next request; three such deaths within ten minutes stop the restarts with a plain 503 text.
CPU only, no engine: the server's bookkeeping is called on a stand-in."""
import sys
import threading
import time
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strata_server import server as S  # noqa: E402
from strata_common import engine_client  # noqa: E402


def stand_in(pid, last_err):
    return types.SimpleNamespace(engine=types.SimpleNamespace(last_err=last_err, proc=types.SimpleNamespace(pid=pid)))


def request_engine(*lines):
    """Gets an engine of the request-ID protocol (strata-q38, strata-ds41) that answers its RUN line with `lines`."""
    e = engine_client.StrataEngine.__new__(engine_client.StrataEngine)
    e.proc, e.gen, e.info, e.silence_s = types.SimpleNamespace(poll=lambda: None), 1, {"request_queue": 1}, 0
    e.request_queues, e.request_lock, e.slot_live, e.request_sequence = {}, threading.Lock(), [], 0
    e.last_err, e.ended = None, False

    def send(text, process=None):
        if text.startswith("RUN "):
            for line in lines:
                e.request_queues[text.split()[1]].put(line)
    e._send = send
    return e


class FatalGpuRestart(unittest.TestCase):
    def note(self, svc, code=3):
        S.Service._note_fatal_gpu_exit.__get__(svc)(code)

    def make(self, pid, err="fatal GPU error: cudaStreamSynchronize(s): an illegal memory access was encountered"):
        svc = stand_in(pid, err)
        svc.FATAL_GPU_MAX = S.Service.FATAL_GPU_MAX
        svc.FATAL_GPU_WINDOW_S = S.Service.FATAL_GPU_WINDOW_S
        return svc

    def test_one_fatal_death_is_restarted_with_one_plain_line(self):
        svc = self.make(100)
        self.note(svc)                      # no exception: the caller goes on to restart the engine

    def test_an_ordinary_death_is_not_counted(self):
        svc = self.make(100, err="the engine ran out of memory")
        for pid in range(10):
            svc.engine.proc.pid = pid
            self.note(svc)
        self.assertFalse(svc.__dict__.get("_fatal_gpu_times"))

    def test_the_same_death_seen_by_many_requests_counts_once(self):
        svc = self.make(100)
        for _ in range(8):
            self.note(svc)
        self.assertEqual(len(svc._fatal_gpu_times), 1)

    def test_three_fatal_deaths_in_ten_minutes_stop_the_restarts_and_say_so(self):
        svc = self.make(1)
        self.note(svc)
        svc.engine.proc.pid = 2
        self.note(svc)
        svc.engine.proc.pid = 3
        with self.assertRaises(engine_client.EngineDied) as cm:
            self.note(svc)
        self.assertIn("3 times in 10 minutes", str(cm.exception))
        self.assertIn("not started again", str(cm.exception))
        with self.assertRaises(engine_client.EngineDied):          # the same dead engine asked again: still refused
            self.note(svc)

    def test_after_the_window_it_tries_again(self):
        svc = self.make(1)
        self.note(svc)
        svc.engine.proc.pid = 2
        self.note(svc)
        svc._fatal_gpu_times[:] = [t - S.Service.FATAL_GPU_WINDOW_S - 1 for t in svc._fatal_gpu_times]
        svc.engine.proc.pid = 3
        self.note(svc)                      # the two old ones aged out: only this one counts

    def test_the_request_in_flight_fails_as_an_engine_death_not_a_bad_request(self):
        eng = types.SimpleNamespace(last_err=None, ended=False)
        with self.assertRaises(engine_client.EngineDied) as cm:
            engine_client.StrataEngine._raise_if_fatal_gpu(eng, "ERR fatal GPU error: cudaStreamSynchronize(s): an illegal memory access was encountered")
        self.assertIn("illegal memory access", str(cm.exception))
        self.assertTrue(eng.ended)                                  # the next request starts the engine at once
        self.assertTrue(eng.last_err.startswith("fatal GPU error"))

    def test_a_paged_engines_fatal_line_ends_its_request_as_an_engine_death(self):
        eng = request_engine("STARTED", "PP 3 3 10 300.0", "ERR fatal GPU error: unspecified launch failure")
        with self.assertRaises(engine_client.EngineDied) as cm:
            list(eng.generate_request([1, 2, 3], 8, {}, threading.Event()))
        self.assertIn("unspecified launch failure", str(cm.exception))
        self.assertTrue(eng.ended)
        eng = request_engine("STARTED", "ERR the conversation's memory is full")
        with self.assertRaises(ValueError):
            list(eng.generate_request([1, 2, 3], 8, {}, threading.Event()))
        self.assertFalse(eng.ended)

    def test_an_ordinary_engine_error_stays_a_bad_request(self):
        eng = types.SimpleNamespace(last_err=None, ended=False)
        engine_client.StrataEngine._raise_if_fatal_gpu(eng, "ERR the prompt (9 tokens) and max_new 5 are more than the context 8")   # no exception
        self.assertFalse(eng.ended)


if __name__ == "__main__":
    unittest.main()
