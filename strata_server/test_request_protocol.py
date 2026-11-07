"""Exercises the Python client against the real C++ command loop without loading a model."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import json
import subprocess
import urllib.request
import threading
import time

import pytest

from strata_common.engine_client import EngineDied, QueueWaitExpired, StrataEngine


@pytest.fixture
def start_engine():
    root = Path(__file__).resolve().parents[1]
    override = os.environ.get("STRATA_REQUEST_PROTOCOL_DRIVER")
    candidates = [Path(override)] if override else [root / "build-hip-win/request_protocol_driver.exe",
                                                   root / "build/request_protocol_driver"]
    executable = next((path for path in candidates if path.is_file()), None)
    if executable is None:
        pytest.skip("build request_protocol_driver or set STRATA_REQUEST_PROTOCOL_DRIVER")
    engines = []

    def start(mode="native", lazy=False):
        engine = StrataEngine(str(executable), [mode], cwd=str(root), lazy=lazy)
        engines.append(engine)
        return engine

    yield start
    for engine in engines:
        engine.close()


def collect(engine, token, maximum, first=None, cancel=None):
    """Collects one stream and signals when its first token becomes observable."""
    out = []
    for item in engine.generate([token, token], maximum, {}, cancel or threading.Event()):
        if item is not None:
            out.append(item)
            if first is not None:
                first.set()
    return out, engine.last


@pytest.mark.parametrize("mode", ["native", "legacy", "serial"])
def test_every_request_uses_run_even_when_alone(start_engine, mode):
    engine = start_engine(mode)
    sent, send = [], engine._send
    engine._send = lambda line, **kwargs: (sent.append(line), send(line, **kwargs))[-1]
    assert engine.slots_view() == []
    out, last = collect(engine, 31, 6)
    assert out == [31] * 6
    assert last["generated"] == 6
    assert last["finish"] == "length"
    assert sent[0].startswith("RUN r")
    assert not any(line.startswith(("GEN ", "BGEN ", "STOP", "BSTOP")) for line in sent)


@pytest.mark.parametrize("mode", ["native", "legacy"])
def test_settings_reply_arrives_before_the_running_request_finishes(start_engine, mode):
    engine = start_engine(mode)
    first = threading.Event()
    with ThreadPoolExecutor() as pool:
        future = pool.submit(collect, engine, 31, 400, first)
        assert first.wait(5)
        active = engine.slots_view()
        assert len(active) == 1
        assert active[0]["generated"] > 0
        assert active[0]["state"] == "generating"
        report = engine.change_settings({"rows": 3})
        assert report["settings"]["rows"] == 3
        assert not future.done()
        assert future.result(timeout=10)[0] == [31] * 400


@pytest.mark.parametrize("mode", ["native", "legacy"])
def test_interleaved_streams_and_commands_cannot_consume_each_others_events(start_engine, mode):
    engine = start_engine(mode)
    first = threading.Event()
    with ThreadPoolExecutor() as pool:
        a = pool.submit(collect, engine, 31, 50, first)
        assert first.wait(5)
        b = pool.submit(collect, engine, 41, 60)
        reports = list(pool.map(lambda _: engine.memory(), range(6)))
        assert all(report["settings"]["tokens_max"] == 8192 for report in reports)
        assert a.result(timeout=10)[0] == [31] * 50
        assert b.result(timeout=10)[0] == [41] * 60


@pytest.mark.parametrize("mode", ["native", "legacy", "serial"])
def test_closing_a_consumer_cancels_and_releases_its_request(start_engine, mode):
    engine = start_engine(mode)
    stream = engine.generate([31, 31], 400, {}, threading.Event())
    assert next(item for item in stream if item is not None) == 31
    stream.close()
    out, _ = collect(engine, 41, 5)
    assert out == [41] * 5
    deadline = time.monotonic() + 5
    while engine.request_queues and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not engine.request_queues


def test_preemption_resumes_the_same_identity_without_truncating_output(start_engine):
    engine = start_engine("preempt")
    sent, send = [], engine._send
    engine._send = lambda line, **kwargs: (sent.append(line), send(line, **kwargs))[-1]
    out, last = collect(engine, 31, 10)
    assert out == [31] * 10
    requests = [line.split() for line in sent if line.startswith("RUN ")]
    assert len(requests) == 2
    assert requests[0][1] == requests[1][1]
    assert [int(line[2]) for line in requests] == [10, 7]
    assert last["generated"] == 10
    assert last["prompt_tokens"] == 2


def test_engine_exit_unblocks_all_request_queues(start_engine):
    engine = start_engine()
    first = threading.Event()
    with ThreadPoolExecutor() as pool:
        a = pool.submit(collect, engine, 31, 400, first)
        assert first.wait(5)
        b = pool.submit(collect, engine, 41, 400)
        engine.proc.kill()
        for future in (a, b):
            with pytest.raises(EngineDied):
                future.result(timeout=5)


@pytest.mark.parametrize("lazy", [False, True])
def test_http_settings_apply_during_generation_including_first_lazy_request(start_engine, lazy):
    from strata_server.frontend import ChatTemplate
    from strata_server.server import ByteTokenizer, Service, serve

    engine = start_engine(lazy=lazy)
    first = threading.Event()
    generate = engine.generate

    def observe(*args, **kwargs):
        for token in generate(*args, **kwargs):
            if token is not None:
                first.set()
            yield token

    engine.generate = observe
    service = Service(engine, ByteTokenizer(), ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
    server = serve(service, port=0)
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def post(path, body):
        request = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)

    try:
        with ThreadPoolExecutor() as pool:
            future = pool.submit(post, "/v1/chat/completions",
                                 {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 160})
            assert first.wait(5)
            with urllib.request.urlopen(base + "/metrics", timeout=5) as response:
                metrics = json.load(response)
            assert len(metrics["live"]["slots"]) == 1
            assert metrics["live"]["slots"][0]["generated"] > 0
            report = post("/v1/memory", {"set": {"rows": 3}})
            assert report["settings"]["rows"] == 3
            assert not future.done()
            assert future.result(timeout=10)["choices"]
    finally:
        server.shutdown()
        server.server_close()


def test_ready_does_not_admit_parallel_work_before_initial_memory_is_cached(start_engine):
    from strata_server.frontend import ChatTemplate
    from strata_server.server import ByteTokenizer, Service, serve

    engine = start_engine(lazy=True)
    service = Service(engine, ByteTokenizer(), ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
    reading, release, generating = threading.Event(), threading.Event(), threading.Event()
    memory, generate = engine.memory, engine.generate

    def wait_for_memory():
        reading.set()
        assert release.wait(5)
        return memory()

    def observe(*args, **kwargs):
        generating.set()
        yield from generate(*args, **kwargs)

    def load():
        with service.fifo:
            service.ensure_loaded()

    def post():
        request = urllib.request.Request(base + "/v1/chat/completions",
                                         data=json.dumps({"messages": [{"role": "user", "content": "Hi"}],
                                                          "max_tokens": 3}).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)

    engine.memory, engine.generate = wait_for_memory, observe
    server = serve(service, port=0)
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with ThreadPoolExecutor() as pool:
            loading = pool.submit(load)
            try:
                assert reading.wait(30)
                assert engine.alive()
                request = pool.submit(post)
                assert not generating.wait(0.1)
            finally:
                release.set()
            loading.result(timeout=5)
            assert service.memory_last["at"] is not None
            assert service.memory_last["settings"]["rows"] == 2
            assert not service.initializing_engine
            assert request.result(timeout=5)["choices"]
    finally:
        release.set()
        server.shutdown()
        server.server_close()


def test_native_admission_does_not_hide_requests_behind_a_python_capacity_semaphore(start_engine):
    engine = start_engine()
    one, two, three = threading.Event(), threading.Event(), threading.Event()
    with ThreadPoolExecutor() as pool:
        a = pool.submit(collect, engine, 31, 300, one)
        assert one.wait(5)
        b = pool.submit(collect, engine, 41, 300, two)
        assert two.wait(5)
        c = pool.submit(collect, engine, 51, 3, three)
        assert three.wait(1)
        assert not a.done() and not b.done()
        assert c.result(timeout=5)[0] == [51] * 3
        assert len(a.result(timeout=10)[0]) == len(b.result(timeout=10)[0]) == 300


@pytest.mark.parametrize("mode", ["native", "legacy", "serial"])
def test_all_backends_expose_the_same_event_shapes(start_engine, mode):
    engine = start_engine(mode)
    result = subprocess.run([engine.spawn[0], mode], input="RUN external 2 31,31\n",
                            capture_output=True, text=True, timeout=5, check=True)
    records = result.stdout.splitlines()
    events = [line.split(maxsplit=2)[2] for line in records if line.startswith("R external ")]
    assert [event for event in events if event.startswith("T ")] == ["T 31", "T 31"]
    assert len([event for event in events if event.startswith("DONE ")]) == 1
    assert all(event.split()[0] in {"STARTED", "PP", "T", "DONE"} for event in events)
    assert not any(line.startswith(("BT ", "BDONE ", "BADM ", "DONE ", "T ")) for line in records)


def test_timed_out_command_reply_cannot_be_mistaken_for_the_next_reply(start_engine):
    engine = start_engine("slow")
    with pytest.raises(EngineDied, match="did not answer"):
        engine._ask("GET_MEMORY_USE", "MEMORY", timeout=0.02, end="END_MEMORY", also=("SETTINGS",))
    report = engine.change_settings({"rows": 3})
    assert report["settings"]["rows"] == 3


def test_inspection_during_prefill_preserves_generation_and_defers_settings(start_engine):
    engine = start_engine("prefill")
    first = threading.Event()
    with ThreadPoolExecutor() as pool:
        generation = pool.submit(collect, engine, 31, 5, first)
        deadline = time.monotonic() + 5
        while engine.progress is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert engine.progress is not None
        engine._send("ASK deferred CHANGE_SETTINGS rows=3")
        records = engine._ask("GET_CONTEXT_WINDOWS conversation0", "CONTEXT_WINDOW", timeout=0.5)
        assert records == [["CONTEXT_WINDOW", "conversation0", "2", "31,32"]]
        assert engine.list_conversations()[0]["state"] == "reading"
        assert engine.memory()["settings"]["rows"] == 2
        assert not first.is_set()
        assert generation.result(timeout=5)[0] == [31] * 5
        assert engine.memory()["settings"]["rows"] == 3


def test_context_read_waits_for_a_chunk_longer_than_the_ordinary_query_budget(start_engine):
    engine = start_engine("prefill")
    ask = engine._ask
    engine._ask = lambda command, kind, timeout=0.02, **kwargs: ask(command, kind, timeout=timeout, **kwargs)
    engine.CONTEXT_QUERY_TIMEOUT_S = 0.5
    with ThreadPoolExecutor() as pool:
        generation = pool.submit(collect, engine, 31, 5)
        deadline = time.monotonic() + 5
        while engine.progress is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert engine.progress is not None
        assert engine.context_windows(["conversation0"])["conversation0"].tolist() == [31, 32]
        assert generation.result(timeout=5)[0] == [31] * 5


def test_settings_interrupt_prefill_and_resume_the_same_request(start_engine):
    engine = start_engine("settings-prefill")
    first = threading.Event()
    sent, send = [], engine._send
    engine._send = lambda line, **kwargs: (sent.append(line), send(line, **kwargs))[-1]
    with ThreadPoolExecutor() as pool:
        generation = pool.submit(collect, engine, 31, 5, first)
        deadline = time.monotonic() + 5
        while engine.progress is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert engine.progress is not None
        assert engine.change_settings({"rows": 3})["settings"]["rows"] == 3
        assert not first.is_set()
        output, last = generation.result(timeout=5)
        assert output == [31] * 5
        assert last["generated"] == 5
        assert last["prompt_ms"] == 2
        assert last["prompt_read"] == 2
        runs = [line.split()[1] for line in sent if line.startswith("RUN ")]
        assert len(runs) == 2 and runs[0] == runs[1]


@pytest.mark.parametrize("mode", ["native", "legacy", "serial"])
def test_started_retires_queue_deadline_once_before_a_longer_generation(start_engine, mode):
    engine = start_engine(mode)
    cancel = threading.Event()
    started = []
    cancel._strata_queue_deadline = time.monotonic() + 0.15
    cancel._strata_queue_started = lambda: started.append(True)
    out, _ = collect(engine, 31, 80, cancel=cancel)
    assert out == [31] * 80
    assert started == [True]
    assert cancel._strata_queue_deadline is None


def test_native_queue_deadline_covers_wait_until_started(start_engine):
    engine = start_engine("slow")
    asked = threading.Event()
    send = engine._send

    def observe(line, **kwargs):
        send(line, **kwargs)
        if line.startswith("ASK "):
            asked.set()

    engine._send = observe
    cancel = threading.Event()
    with ThreadPoolExecutor() as pool:
        report = pool.submit(engine.memory)
        assert asked.wait(5)
        cancel._strata_queue_deadline = time.monotonic() + 0.02
        with pytest.raises(QueueWaitExpired):
            collect(engine, 31, 80, cancel=cancel)
        assert report.result(timeout=5)["settings"]["rows"] == 2
    deadline = time.monotonic() + 5
    while engine.request_queues and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not engine.request_queues
