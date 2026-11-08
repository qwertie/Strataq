"""Checks prompt cancellation handoff without loading a model."""
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from strata_server import server


class DisconnectWatcher(unittest.TestCase):
    """Handler._watch_client: when a client that hung up cancels its request."""

    def test_disconnect_watcher_checks_socket_readiness_before_sleeping(self):
        cancel, done, select = run_watcher()
        assert select.call_args.args[-1] == 0.5
        done.wait.assert_not_called()
        assert cancel.is_set()

    def test_completed_request_does_not_become_a_disconnect(self):
        for connection_error in (False, True):
            with self.subTest(connection_error=connection_error):
                cancel, done, _ = run_watcher(completed=True, connection_error=connection_error)
                done.wait.assert_not_called()
                assert not cancel.is_set()

    def test_unexpected_extra_data_does_not_spin_or_cancel(self):
        cancel, done, select = run_watcher(data=b"extra")
        assert select.call_count == 1
        done.wait.assert_called_once_with(0.5)
        assert not cancel.is_set()


def run_watcher(*, completed=False, connection_error=False, data=b""):
    """Runs one watcher iteration synchronously, including completion while socket readiness is checked."""
    cancel = threading.Event()
    service = SimpleNamespace(status_lock=threading.Lock(), active_cancels=set(),
                              note_client_disconnect=lambda event: event.set())
    handler = object.__new__(server.make_handler(service))
    handler.connection = mock.Mock()
    handler.connection.recv.return_value = data
    done = mock.Mock()
    done.is_set.return_value = False
    done.wait.return_value = True

    def readiness(*args):
        if completed:
            done.is_set.return_value = True
        if connection_error:
            raise ConnectionResetError()
        if data:
            done.is_set.side_effect = [True]
        return [handler.connection], [], []

    with mock.patch.object(server.threading, "Event", return_value=done), \
            mock.patch.object(server.threading, "Thread") as thread, \
            mock.patch.object(server.select, "select", side_effect=readiness) as select:
        handler._watch_client(cancel)
        thread.call_args.kwargs["target"]()
    select.assert_called_once_with([handler.connection], [], [], 0.5)
    return cancel, done, select


if __name__ == "__main__":
    unittest.main()
