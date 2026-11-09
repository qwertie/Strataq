"""Benchmark waits distinguish request readiness from expert loading."""
from unittest.mock import patch

import pytest

from strata_common.expert_fill import wait_for_experts, wait_for_protocol_experts


def test_legacy_report_is_loaded():
    messages = []
    assert wait_for_experts(lambda: {"memories": []}, log=messages.append) >= 0
    assert "for experts" in messages[0]


def test_waits_for_explicit_completion():
    reports = iter([{"expert_fill": {"loaded": False}}, {"expert_fill": {"loaded": True, "pending": 9}}])
    with patch("strata_common.expert_fill.time.sleep") as sleep:
        wait_for_experts(lambda: next(reports), log=lambda message: None)
    sleep.assert_called_once()


def test_protocol_reports_and_timeout():
    reports = iter(["EXPERT_FILL loaded=0 pending=9", "END_MEMORY",
                    "EXPERT_FILL loaded=1 pending=0", "END_MEMORY"])
    sent = []
    with patch("strata_common.expert_fill.time.sleep"):
        wait_for_protocol_experts(sent.append, lambda: next(reports), log=lambda message: None)
    assert sent == ["GET_MEMORY_USE", "GET_MEMORY_USE"]
    with pytest.raises(TimeoutError):
        wait_for_experts(lambda: {"expert_fill": {"loaded": False}}, timeout=0)
