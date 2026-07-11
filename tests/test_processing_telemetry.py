from __future__ import annotations

import json
import sys
import threading
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from whodoirunlike import processing_telemetry


ATTEMPT_ID = "11111111-1111-4111-8111-111111111111"


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0
        self.wall_start = datetime(2026, 7, 9, 12, 0, tzinfo=timezone.utc)

    def monotonic(self) -> float:
        return self.value

    def wall(self) -> datetime:
        return self.wall_start + timedelta(seconds=self.value - 100.0)

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _telemetry(
    tmp_path: Path,
    clock: FakeClock,
    *,
    sender: Any = None,
    local_path: Path | None = None,
) -> processing_telemetry.ProcessingTelemetry:
    ids = iter(f"00000000-0000-4000-8000-{index:012d}" for index in range(1, 100))
    return processing_telemetry.ProcessingTelemetry(
        run_id="12345678-1234-4234-9234-123456789abc",
        attempt_id=ATTEMPT_ID,
        local_path=local_path or tmp_path / "processing_events.jsonl",
        input_metadata={
            "size_bytes": 2048,
            "source_url": "https://private.example/video.mp4",
            "filename": "private.mp4",
        },
        runtime_metadata={"backend": "test", "secret": "never-store-this"},
        monotonic_clock=clock.monotonic,
        wall_clock=clock.wall,
        event_id_factory=lambda: next(ids),
        resource_sampler=lambda: {"peak_rss_bytes": 4096},
        event_sender=sender,
        asynchronous_delivery=False,
    )


def test_boundary_events_have_stable_contract_exact_elapsed_and_append_only_file(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    delivered: list[dict[str, Any]] = []
    telemetry = _telemetry(tmp_path, clock, sender=delivered.append)

    telemetry.attempt_started()
    with telemetry.stage("target_tracking", runtime={"model": "yolo"}) as stage:
        clock.advance(1.125)
        with telemetry.span("target_tracking", "model_load"):
            clock.advance(2.375)
        stage.add_measurements({"frame_count": 30})
        clock.advance(0.5)
    telemetry.attempt_completed({"artifact_count": 3})

    events = [json.loads(line) for line in telemetry.local_path.read_text().splitlines()]
    assert events == delivered
    assert [event["sequence"] for event in events] == list(range(1, 7))
    assert [event["event_id"] for event in events] == [
        f"00000000-0000-4000-8000-{index:012d}" for index in range(1, 7)
    ]
    assert all(event["schema_version"] == 1 for event in events)
    assert all(event["run_id"] == telemetry.run_id for event in events)
    assert all(event["attempt_id"] == ATTEMPT_ID for event in events)
    assert all(event["event_time"].endswith("Z") for event in events)
    assert events[0]["input"] == {"size_bytes": 2048}
    assert events[0]["runtime"]["backend"] == "test"
    assert "secret" not in events[0]["runtime"]
    assert events[0]["resources"] == {"peak_rss_bytes": 4096}

    span_complete = next(event for event in events if event["event_type"] == "span_completed")
    stage_complete = next(event for event in events if event["event_type"] == "stage_completed")
    assert span_complete["elapsed_seconds"] == 2.375
    assert stage_complete["elapsed_seconds"] == 4.0
    assert stage_complete["measurements"]["frame_count"] == 30
    for event in events:
        processing_telemetry.validate_event(event)


def test_runner_mask_result_exports_predictor_cache_measurements(tmp_path: Path) -> None:
    clock = FakeClock()
    telemetry = _telemetry(tmp_path, clock)

    with telemetry.stage("runner_mask") as stage:
        stage.set_result(
            {
                "cache_hit": True,
                "model_build_seconds": 0.0,
                "predictor_lock_wait_seconds": 0.004,
                "frame_count": 260,
                "elapsed_seconds": 40.0,
            }
        )

    events = [json.loads(line) for line in telemetry.local_path.read_text().splitlines()]
    completed = next(event for event in events if event["event_type"] == "stage_completed")
    assert completed["measurements"] == {
        "cache_hit": True,
        "elapsed_seconds": 40.0,
        "frame_count": 260,
        "milliseconds_per_frame": 153.84615384615384,
        "model_build_seconds": 0.0,
        "predictor_lock_wait_seconds": 0.004,
    }


def test_progress_reporter_throttles_to_five_seconds_and_tracks_phase_spans(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    telemetry = _telemetry(tmp_path, clock)
    reporter = telemetry.progress_reporter(
        stage="runner_mask",
        phase_spans={"loading_model": "model_load", "running": "inference"},
    )

    reporter({"phase": "loading_model", "processed_frames": 0, "total_frames": 100})
    clock.advance(4.9)
    reporter({"phase": "loading_model", "processed_frames": 0, "total_frames": 100})
    clock.advance(0.1)
    reporter({"phase": "loading_model", "processed_frames": 0, "total_frames": 100})
    clock.advance(2.0)
    reporter({"phase": "running", "processed_frames": 10, "total_frames": 100})
    clock.advance(3.25)
    reporter.close()

    events = [json.loads(line) for line in telemetry.local_path.read_text().splitlines()]
    progress = [event for event in events if event["event_type"] == "progress_sampled"]
    assert [(event["span"], event["progress"]["phase"]) for event in progress] == [
        ("model_load", "loading_model"),
        ("model_load", "loading_model"),
        ("inference", "running"),
    ]
    completed = [event for event in events if event["event_type"] == "span_completed"]
    assert [(event["span"], event["elapsed_seconds"]) for event in completed] == [
        ("model_load", 7.0),
        ("inference", 3.25),
    ]


def test_failed_boundary_emits_sanitized_classified_error(tmp_path: Path) -> None:
    clock = FakeClock()
    telemetry = _telemetry(tmp_path, clock)

    with pytest.raises(RuntimeError, match="model failed"):
        with telemetry.stage("pose_sequence"):
            clock.advance(1.75)
            raise RuntimeError(
                "model failed at https://private.example/path "
                "Bearer abc.def and token=super-secret in /tmp/private/clip.mp4"
            )

    event = json.loads(telemetry.local_path.read_text().splitlines()[-1])
    assert event["event_type"] == "stage_failed"
    assert event["elapsed_seconds"] == 1.75
    assert event["error"]["exception_type"] == "RuntimeError"
    assert event["error"]["class"] == "RuntimeError"
    assert event["error"]["code"] == "processing.runtime_error"
    assert event["error"]["category"] == "processing"
    assert event["error"]["retryable"] is False
    assert "private.example" not in event["error"]["message"]
    assert "abc.def" not in event["error"]["message"]
    assert "super-secret" not in event["error"]["message"]
    assert "/tmp/private" not in event["error"]["message"]
    assert "<url>" in event["error"]["message"]
    assert "<redacted>" in event["error"]["message"]
    assert "<path>" in event["error"]["message"]


def test_skipped_stage_uses_valid_complete_status_and_preserves_outcome(tmp_path: Path) -> None:
    clock = FakeClock()
    telemetry = _telemetry(tmp_path, clock)

    with telemetry.stage("densepose_body_map") as boundary:
        boundary.set_result({"status": "skipped"})

    event = json.loads(telemetry.local_path.read_text().splitlines()[-1])
    assert event["event_type"] == "stage_completed"
    assert event["status"] == "complete"
    assert event["measurements"]["outcome"] == "skipped"


def test_milestones_use_attempt_age_while_boundaries_keep_scope_duration(tmp_path: Path) -> None:
    clock = FakeClock()
    telemetry = processing_telemetry.ProcessingTelemetry(
        run_id="12345678-1234-4234-9234-123456789abc",
        attempt_id=ATTEMPT_ID,
        local_path=tmp_path / "events.jsonl",
        monotonic_clock=clock.monotonic,
        wall_clock=clock.wall,
        resource_sampler=lambda: {},
        asynchronous_delivery=False,
        attempt_elapsed_offset_seconds=30.0,
    )

    with telemetry.stage("source_download"):
        clock.advance(5.0)
    telemetry.analysis_completed()
    clock.advance(2.0)
    telemetry.result_ready()
    clock.advance(3.0)
    telemetry.attempt_completed()

    events = [json.loads(line) for line in telemetry.local_path.read_text().splitlines()]
    assert next(event for event in events if event["event_type"] == "stage_completed")[
        "elapsed_seconds"
    ] == 5.0
    assert next(event for event in events if event["event_type"] == "analysis_completed")[
        "elapsed_seconds"
    ] == 35.0
    assert next(event for event in events if event["event_type"] == "result_ready")[
        "elapsed_seconds"
    ] == 37.0
    assert events[-1]["event_type"] == "attempt_completed"
    assert events[-1]["elapsed_seconds"] == 40.0


def test_contract_rejects_out_of_range_sequence_elapsed_and_naive_time(tmp_path: Path) -> None:
    clock = FakeClock()
    telemetry = _telemetry(tmp_path, clock)
    telemetry.attempt_started()
    stored = json.loads(telemetry.local_path.read_text().splitlines()[-1])

    with pytest.raises(ValueError, match="sequence"):
        processing_telemetry.validate_event({**stored, "sequence": 0})
    with pytest.raises(ValueError, match="elapsed_seconds"):
        processing_telemetry.validate_event({**stored, "elapsed_seconds": 31_536_001})
    with pytest.raises(ValueError, match="timezone"):
        processing_telemetry.validate_event({**stored, "event_time": "2026-07-09T12:00:00"})


def test_local_sender_and_resource_failures_never_escape(tmp_path: Path) -> None:
    clock = FakeClock()
    unwritable_path = tmp_path / "events-directory"
    unwritable_path.mkdir()

    def broken_sender(_: dict[str, Any]) -> None:
        raise TimeoutError("analytics endpoint unavailable")

    telemetry = _telemetry(
        tmp_path,
        clock,
        sender=broken_sender,
        local_path=unwritable_path,
    )
    telemetry._resource_sampler = lambda: (_ for _ in ()).throw(RuntimeError("no sampler"))

    event = telemetry.emit(
        "stage_started",
        stage="quality_control",
        elapsed_seconds=0.0,
        status="running",
    )

    assert event is not None
    assert event["resources"] == {}
    assert telemetry.local_write_failures == 1
    assert telemetry.delivery_failures == 1


def test_worker_event_post_is_authenticated_and_local_write_happens_first(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    local_path = tmp_path / "events.jsonl"
    requests: list[Any] = []
    observed_timeouts: list[float] = []

    class FakeResponse:
        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_: Any) -> None:
            return None

    def fake_urlopen(request: Any, *, timeout: float) -> FakeResponse:
        assert local_path.read_text(encoding="utf-8").strip()
        observed_timeouts.append(timeout)
        requests.append(request)
        return FakeResponse()

    monkeypatch.delenv("WHODOIRUNLIKE_TELEMETRY_DELIVERY_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setattr(processing_telemetry.urllib.request, "urlopen", fake_urlopen)
    telemetry = processing_telemetry.ProcessingTelemetry(
        run_id="12345678-1234-4234-9234-123456789abc",
        attempt_id=ATTEMPT_ID,
        local_path=local_path,
        callback_base_url="https://api.whodoirunlike.com/",
        auth_token="shared-secret",
        monotonic_clock=clock.monotonic,
        wall_clock=clock.wall,
        resource_sampler=lambda: {},
        asynchronous_delivery=False,
    )

    telemetry.attempt_started()

    assert len(requests) == 1
    assert observed_timeouts == [10.0]
    request = requests[0]
    assert request.full_url.endswith(
        "/v1/jobs/12345678-1234-4234-9234-123456789abc/events"
    )
    assert request.get_header("Authorization") == "Bearer shared-secret"
    posted = json.loads(request.data)
    assert posted["event_type"] == "attempt_started"
    assert posted["schema_version"] == 1


def test_worker_event_post_retries_transient_failures_with_bounded_attempts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = 0

    class FakeResponse:
        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_: Any) -> None:
            return None

    def flaky_urlopen(request: Any, *, timeout: float) -> FakeResponse:
        nonlocal calls
        calls += 1
        assert timeout == 0.25
        if calls < 3:
            raise urllib.error.URLError("temporary telemetry failure")
        return FakeResponse()

    monkeypatch.setenv("WHODOIRUNLIKE_TELEMETRY_DELIVERY_TIMEOUT_SECONDS", "0.25")
    monkeypatch.setenv("WHODOIRUNLIKE_TELEMETRY_DELIVERY_ATTEMPTS", "3")
    monkeypatch.setenv("WHODOIRUNLIKE_TELEMETRY_DELIVERY_BACKOFF_SECONDS", "0")
    monkeypatch.setattr(processing_telemetry.urllib.request, "urlopen", flaky_urlopen)
    telemetry = processing_telemetry.ProcessingTelemetry(
        run_id="12345678-1234-4234-9234-123456789abc",
        attempt_id=ATTEMPT_ID,
        local_path=tmp_path / "events.jsonl",
        callback_base_url="https://api.whodoirunlike.com",
        auth_token="secret",
        resource_sampler=lambda: {},
        asynchronous_delivery=False,
    )

    telemetry.attempt_started()

    assert calls == 3
    assert telemetry.delivery_measurements()["telemetry_delivery_failures"] == 0


def test_async_sender_keeps_delivery_off_caller_thread(tmp_path: Path) -> None:
    caller_thread = threading.current_thread().name
    sender_threads: list[str] = []
    delivered = threading.Event()

    def sender(_: dict[str, Any]) -> None:
        sender_threads.append(threading.current_thread().name)
        delivered.set()

    telemetry = processing_telemetry.ProcessingTelemetry(
        run_id="12345678-1234-4234-9234-123456789abc",
        attempt_id="22222222-2222-4222-8222-222222222222",
        local_path=tmp_path / "events.jsonl",
        event_sender=sender,
        asynchronous_delivery=True,
        resource_sampler=lambda: {},
    )

    telemetry.attempt_started()
    assert delivered.wait(timeout=1.0)
    assert telemetry.close(timeout=1.0) is True
    assert sender_threads == ["processing-telemetry-sender"]
    assert sender_threads[0] != caller_thread


def test_async_sender_close_drains_slow_terminal_success(tmp_path: Path) -> None:
    sender_started = threading.Event()
    release_sender = threading.Event()
    close_started = threading.Event()
    delivered: list[str] = []
    close_results: list[bool] = []

    def slow_sender(event: dict[str, Any]) -> None:
        sender_started.set()
        assert release_sender.wait(timeout=1.0)
        delivered.append(event["event_type"])

    telemetry = processing_telemetry.ProcessingTelemetry(
        run_id="12345678-1234-4234-9234-123456789abc",
        attempt_id=ATTEMPT_ID,
        local_path=tmp_path / "events.jsonl",
        event_sender=slow_sender,
        asynchronous_delivery=True,
        resource_sampler=lambda: {},
    )
    telemetry.attempt_completed()
    assert sender_started.wait(timeout=1.0)

    def close_telemetry() -> None:
        close_started.set()
        close_results.append(telemetry.close(timeout=180.0))

    close_thread = threading.Thread(target=close_telemetry)
    close_thread.start()
    assert close_started.wait(timeout=1.0)
    assert close_thread.is_alive()

    release_sender.set()
    close_thread.join(timeout=1.0)

    assert close_thread.is_alive() is False
    assert close_results == [True]
    assert delivered == ["attempt_completed"]


def test_async_sender_opens_circuit_and_counts_drops_after_repeated_failures(
    tmp_path: Path,
) -> None:
    def broken_sender(_: dict[str, Any]) -> None:
        raise TimeoutError("endpoint down")

    telemetry = processing_telemetry.ProcessingTelemetry(
        run_id="12345678-1234-4234-9234-123456789abc",
        attempt_id=ATTEMPT_ID,
        local_path=tmp_path / "events.jsonl",
        event_sender=broken_sender,
        asynchronous_delivery=True,
        delivery_failure_threshold=2,
        resource_sampler=lambda: {},
    )
    for _ in range(5):
        telemetry.emit(
            "stage_started",
            stage="source_download",
            elapsed_seconds=0.0,
            status="running",
        )

    assert telemetry.flush_delivery(timeout=1.0) is True
    measurements = telemetry.delivery_measurements()
    assert measurements["telemetry_delivery_failures"] == 2
    assert measurements["telemetry_delivery_dropped"] == 3
    assert measurements["telemetry_delivery_pending"] == 0
    assert telemetry.close(timeout=1.0) is True


def test_default_resource_and_runtime_metadata_expose_dashboard_gpu_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        memory_allocated=lambda: 2 * 1024 * 1024,
        memory_reserved=lambda: 3 * 1024 * 1024,
        max_memory_allocated=lambda: 4 * 1024 * 1024,
        current_device=lambda: 0,
        get_device_name=lambda _: "Test GPU",
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))
    monkeypatch.setattr(
        processing_telemetry,
        "_current_rss_bytes",
        lambda: 5 * 1024 * 1024,
    )

    resources = processing_telemetry.default_resource_sample()
    runtime = processing_telemetry._gpu_runtime_metadata()

    assert resources["rss_mb"] == 5.0
    assert resources["peak_rss_mb"] >= 0
    assert resources["cuda_allocated_mb"] == 2.0
    assert resources["cuda_reserved_mb"] == 3.0
    assert resources["cuda_peak_mb"] == 4.0
    assert runtime["gpu_type"] == "Test GPU"


@pytest.mark.parametrize(
    ("value", "accepted"),
    [
        ("33333333-3333-4333-8333-333333333333", True),
        ("bad attempt with spaces", False),
        (None, False),
    ],
)
def test_attempt_id_is_accepted_when_safe_and_generated_as_fallback(
    value: str | None,
    accepted: bool,
) -> None:
    attempt_id = processing_telemetry.ensure_attempt_id(value)

    if accepted:
        assert attempt_id == value
    else:
        assert attempt_id != value
        assert len(attempt_id) == 36
