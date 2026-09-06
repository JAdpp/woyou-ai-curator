"""Late diagnostic writes must not contaminate the next evaluated question."""
from contextvars import copy_context

from scripts.run_retrieval_evaluation import InMemoryTraceCapture


def test_late_previous_question_trace_is_ignored():
    capture = InMemoryTraceCapture()
    capture.reset()
    timed_out_worker_context = copy_context()
    capture.reset()  # The next question has started; the old worker still runs.
    timed_out_worker_context.run(
        capture.write,
        stage_latency_ms={"dense": 9000},
        warnings=["dense_query_failed"],
    )
    result = capture.pop_trace()
    assert result["warnings"] == []
    assert result["stageLatencyMs"] == {}


def test_current_worker_context_is_captured():
    capture = InMemoryTraceCapture()
    capture.reset()
    worker_context = copy_context()
    worker_context.run(capture.write, stage_latency_ms={"dense": 4.5}, warnings=["sample"])
    assert capture.pop_trace() == {
        "stageLatencyMs": {"dense": 4.5}, "warnings": ["sample"], "candidateResults": [],
    }


def test_late_consumed_stage_cannot_replace_current_stage():
    capture = InMemoryTraceCapture()
    capture.reset()
    initial_context = copy_context()
    capture.write(stage_latency_ms={"dense": 3})
    capture.pop_trace()
    capture.write(stage_latency_ms={"audit_search": 7})
    initial_context.run(capture.write, stage_latency_ms={"dense": 999})
    assert capture.pop_trace()["stageLatencyMs"] == {"audit_search": 7.0}
