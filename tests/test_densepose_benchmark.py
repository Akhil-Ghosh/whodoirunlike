from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from whodoirunlike import densepose_benchmark, densepose_benchmark_serverless
from whodoirunlike.densepose_benchmark import (
    BenchmarkContractError,
    BenchmarkRequest,
    CudaRuntime,
    LabelEvidenceCollector,
)


def _patch_mask_contract(monkeypatch: pytest.MonkeyPatch, raw: bytes) -> str:
    encoded = base64.b64encode(raw).decode("ascii")
    monkeypatch.setattr(densepose_benchmark, "RUNNER_MASK_BYTES", len(raw))
    monkeypatch.setattr(densepose_benchmark, "RUNNER_MASK_BASE64_CHARS", len(encoded))
    monkeypatch.setattr(
        densepose_benchmark,
        "RUNNER_MASK_SHA256",
        hashlib.sha256(raw).hexdigest(),
    )
    return encoded


def _request_payload(
    encoded_mask: str,
    *,
    batch_sizes: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "type": densepose_benchmark.BENCHMARK_TYPE,
        "schema_version": densepose_benchmark.BENCHMARK_SCHEMA_VERSION,
        "fixture_id": densepose_benchmark.CANONICAL_FIXTURE_ID,
        "batch_sizes": batch_sizes or [1, 2, 4, 8],
        "assets": {
            "baseline_runner_mask_mp4": {
                "encoding": "base64",
                "sha256": densepose_benchmark.RUNNER_MASK_SHA256,
                "data": encoded_mask,
            }
        },
    }


def _densepose_row(frame_index: int, *, score: float = 0.9) -> dict[str, Any]:
    return {
        "frame_index": frame_index,
        "usable": True,
        "drop_reason": None,
        "part_ids": [1, 2],
        "part_centroids": {
            "1": {"x": 0.25, "y": 0.4},
            "2": {"x": 0.3, "y": 0.5},
        },
        "densepose_coverage": 0.3,
        "mask_overlap": 0.8,
        "bbox": [10, 20, 30, 40],
        "score": score,
        "inference_input": {
            "target_crop_enabled": True,
            "crop_bbox": [0, 0, 50, 60],
            "width": 50,
            "height": 60,
        },
    }


def _write_video(
    path: Path,
    *,
    frame_count: int = 3,
    width: int = 64,
    height: int = 48,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (width, height),
    )
    assert writer.isOpened()
    for index in range(frame_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[8:30, 10 + index : 28 + index] = (40, 120, 220)
        writer.write(frame)
    writer.release()


def test_registered_canonical_fixture_contract_is_exact() -> None:
    assert densepose_benchmark.SOURCE_SHA256 == (
        "a8146591119c5439cc01168df63fa6144a7a55ff6817726946e1e8f5bc381617"
    )
    assert densepose_benchmark.RUNNER_MASK_SHA256 == (
        "f7bb2d1ed00767ed2866c5b3a57b47361a591f1dbf090a5089d187f9ae410ef7"
    )
    assert densepose_benchmark.RUNNER_MASK_BYTES == 526_609
    assert densepose_benchmark.RUNNER_MASK_BASE64_CHARS == 702_148
    assert densepose_benchmark.MAX_REQUEST_BYTES == 1024 * 1024
    assert densepose_benchmark.MAX_RESPONSE_BYTES == 256 * 1024
    assert densepose_benchmark.ALLOWED_BATCH_SIZES == (1, 2, 4, 8)


def test_request_validation_accepts_only_canonical_mask_and_normalizes_batch_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = b"canonical-mask"
    encoded = _patch_mask_contract(monkeypatch, raw)
    payload = _request_payload(encoded, batch_sizes=[8, 1, 4])

    request = densepose_benchmark.validate_request(payload)

    assert request.batch_sizes == (1, 4, 8)
    assert request.runner_mask_bytes == raw
    assert request.request_bytes == densepose_benchmark.serialized_size(payload)


@pytest.mark.parametrize(
    ("batch_sizes", "message"),
    [
        ([], "nonempty"),
        ([2, 4], "include"),
        ([1, 1], "duplicates"),
        ([1, 3], "unsupported"),
        ([1, True], "integers"),
        ([1, 2, 4, 8, 8], "too many"),
    ],
)
def test_request_validation_rejects_invalid_batch_matrices(
    monkeypatch: pytest.MonkeyPatch,
    batch_sizes: list[Any],
    message: str,
) -> None:
    encoded = _patch_mask_contract(monkeypatch, b"mask")
    payload = _request_payload(encoded)
    payload["batch_sizes"] = batch_sizes

    with pytest.raises(BenchmarkContractError, match=message):
        densepose_benchmark.validate_request(payload)


def test_request_validation_rejects_extra_fields_and_asset_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded = _patch_mask_contract(monkeypatch, b"mask")
    payload = _request_payload(encoded)
    payload["callback_base_url"] = "https://api.whodoirunlike.com"

    with pytest.raises(BenchmarkContractError, match="unexpected fields"):
        densepose_benchmark.validate_request(payload)

    payload = _request_payload(encoded)
    payload["assets"]["source"] = {"data": "arbitrary"}
    with pytest.raises(BenchmarkContractError, match="exact fixture asset"):
        densepose_benchmark.validate_request(payload)


def test_request_validation_rejects_noncanonical_or_modified_base64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded = _patch_mask_contract(monkeypatch, b"mask")
    payload = _request_payload(encoded)
    payload["assets"]["baseline_runner_mask_mp4"]["data"] = "!" * len(encoded)

    with pytest.raises(BenchmarkContractError, match="strict base64"):
        densepose_benchmark.validate_request(payload)

    payload = _request_payload(encoded)
    payload["assets"]["baseline_runner_mask_mp4"]["sha256"] = "0" * 64
    with pytest.raises(BenchmarkContractError, match="SHA-256"):
        densepose_benchmark.validate_request(payload)


def test_request_validation_enforces_whole_request_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded = _patch_mask_contract(monkeypatch, b"mask")
    payload = _request_payload(encoded)
    payload["assets"]["baseline_runner_mask_mp4"]["data"] = "A" * (
        densepose_benchmark.MAX_REQUEST_BYTES + 1
    )

    with pytest.raises(BenchmarkContractError, match="1 MiB"):
        densepose_benchmark.validate_request(payload)


def test_response_limit_fails_closed_without_truncating() -> None:
    with pytest.raises(densepose_benchmark.BenchmarkResponseTooLarge, match="exceeds"):
        densepose_benchmark.ensure_bounded_response(
            {"status": "complete", "evidence": "x" * densepose_benchmark.MAX_RESPONSE_BYTES}
        )


def test_bounded_failure_never_contains_message_path_or_traceback() -> None:
    failure = densepose_benchmark.bounded_failure(
        "benchmark_execution_failed",
        exception_type="RuntimeError",
    )

    serialized = json.dumps(failure)
    assert failure["status"] == "failed"
    assert failure["response_bytes"] <= densepose_benchmark.MAX_RESPONSE_BYTES
    assert "message" not in serialized
    assert "traceback" not in serialized.lower()
    assert "/" not in serialized


def test_serverless_handler_is_disabled_by_default_and_has_no_production_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(densepose_benchmark_serverless.ENABLE_ENV, raising=False)

    with pytest.raises(RuntimeError, match="disabled"):
        densepose_benchmark_serverless.handler(
            {"input": {"type": densepose_benchmark.BENCHMARK_TYPE}}
        )
    with pytest.raises(ValueError, match="accepts only"):
        densepose_benchmark_serverless.handler(
            {"input": {"type": "process_clip", "run_id": "must-not-run"}}
        )


def test_serverless_health_is_shallow_bounded_and_advertises_disabled_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(densepose_benchmark_serverless.ENABLE_ENV, raising=False)
    monkeypatch.delenv("WHODOIRUNLIKE_PROCESSOR_VERSION", raising=False)
    monkeypatch.delenv("WHODOIRUNLIKE_BENCHMARK_IMAGE_DIGEST", raising=False)

    result = densepose_benchmark_serverless.handler({"input": {"type": "health"}})

    assert result["benchmark_enabled"] is False
    assert result["runtime_identity_pinned"] is False
    assert result["allowed_batch_sizes"] == [1, 2, 4, 8]
    assert result["isolation"] == "fresh_spawned_process_per_matrix"
    assert result["response_bytes"] <= densepose_benchmark.MAX_RESPONSE_BYTES


def test_enabled_handler_validates_then_runs_only_the_isolated_benchmark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = b"mask"
    encoded = _patch_mask_contract(monkeypatch, raw)
    payload = _request_payload(encoded, batch_sizes=[1, 4])
    captured: list[BenchmarkRequest] = []
    monkeypatch.setenv(densepose_benchmark_serverless.ENABLE_ENV, "true")
    monkeypatch.setattr(
        densepose_benchmark_serverless,
        "run_benchmark_isolated",
        lambda request: (
            captured.append(request)
            or densepose_benchmark.ensure_bounded_response(
                {"status": "complete", "type": densepose_benchmark.BENCHMARK_RESULT_TYPE}
            )
        ),
    )

    result = densepose_benchmark_serverless.handler({"input": payload})

    assert result["status"] == "complete"
    assert captured[0].batch_sizes == (1, 4)
    assert captured[0].runner_mask_bytes == raw


def test_child_failure_is_bounded_and_does_not_return_exception_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Sender:
        def __init__(self) -> None:
            self.value: dict[str, Any] | None = None

        def send(self, value: dict[str, Any]) -> None:
            self.value = value

        def close(self) -> None:
            return None

    sender = Sender()
    request = BenchmarkRequest((1,), b"mask", "a" * 64, 100)
    monkeypatch.setattr(
        densepose_benchmark_serverless,
        "run_benchmark",
        lambda _request: (_ for _ in ()).throw(
            RuntimeError("secret token at /private/fixture should never escape")
        ),
    )

    densepose_benchmark_serverless._child_entry(request, sender)  # type: ignore[arg-type]

    assert sender.value is not None
    serialized = json.dumps(sender.value)
    assert sender.value["error"] == {
        "code": "benchmark_execution_failed",
        "exception_type": "RuntimeError",
    }
    assert "secret token" not in serialized
    assert "/private" not in serialized


def test_spawned_matrix_process_returns_a_bounded_failure_on_unpinned_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WHODOIRUNLIKE_PROCESSOR_VERSION", raising=False)
    monkeypatch.delenv("WHODOIRUNLIKE_BENCHMARK_IMAGE_DIGEST", raising=False)
    request = BenchmarkRequest((1,), b"mask", "a" * 64, 100)

    result = densepose_benchmark_serverless.run_benchmark_isolated(
        request,
        timeout_seconds=15,
    )

    assert result["status"] == "failed"
    assert result["error"] == {
        "code": "benchmark_execution_failed",
        "exception_type": "BenchmarkContractError",
    }
    assert result["response_bytes"] <= densepose_benchmark.MAX_RESPONSE_BYTES


def test_production_container_command_and_batch_default_are_unchanged() -> None:
    dockerfile = (Path(__file__).resolve().parents[1] / "Dockerfile.runpod").read_text(
        encoding="utf-8"
    )

    assert 'CMD ["python", "-m", "whodoirunlike.runpod_serverless"]' in dockerfile
    assert "densepose_benchmark_serverless" not in dockerfile
    assert "DENSEPOSE_BATCH_SIZE=" not in dockerfile
    assert densepose_benchmark.run_densepose.__kwdefaults__["batch_size"] == 1


def test_label_evidence_returns_hashes_and_aggregates_but_never_raw_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(densepose_benchmark, "EXPECTED_FRAME_COUNT", 3)
    collector = LabelEvidenceCollector()
    collector(0, _densepose_row(0), np.asarray([[0, 1], [2, 2]], dtype=np.uint8))
    collector(1, _densepose_row(1), np.asarray([[0, 1], [1, 1]], dtype=np.uint8))
    collector(
        2,
        {"frame_index": 2, "usable": False, "drop_reason": "densepose_missing"},
        None,
    )

    summary = collector.summary()

    assert summary["frame_count"] == 3
    assert len(summary["frame_hashes"]) == 3
    assert all(len(value) == 64 for value in summary["frame_hashes"])
    assert summary["shape_counts"] == {"2x2": 2, "missing": 1}
    assert summary["dtype_counts"] == {"missing": 1, "uint8": 2}
    assert summary["part_histogram"] == {"0": 2, "1": 4, "2": 2}
    serialized = json.dumps(summary)
    assert "labels" not in serialized
    assert "array(" not in serialized


def test_label_gate_reports_hash_mismatches_and_caps_mismatch_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame_count = 20
    monkeypatch.setattr(densepose_benchmark, "EXPECTED_FRAME_COUNT", frame_count)
    control = LabelEvidenceCollector()
    candidate = LabelEvidenceCollector()
    for frame_index in range(frame_count):
        control(
            frame_index,
            _densepose_row(frame_index),
            np.asarray([[0, 1], [1, 2]], dtype=np.uint8),
        )
        candidate(
            frame_index,
            _densepose_row(frame_index),
            np.asarray([[0, 2], [2, 1]], dtype=np.uint8),
        )

    gate = densepose_benchmark.compare_label_evidence(control, candidate)

    assert gate["passed"] is False
    assert gate["diagnostics"]["raw_label_hashes_exact"] is False
    assert gate["measurements"]["mismatch_count"] == frame_count
    assert len(gate["measurements"]["mismatches"]) == 16
    assert gate["measurements"]["mismatches_truncated"] is True


def test_label_gate_allows_nonexact_hashes_when_declared_iou_thresholds_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(densepose_benchmark, "EXPECTED_FRAME_COUNT", 1)
    control = LabelEvidenceCollector()
    candidate = LabelEvidenceCollector()
    control_labels = np.ones((100, 100), dtype=np.uint8)
    candidate_labels = control_labels.copy()
    candidate_labels[0, 0] = 0
    control(0, _densepose_row(0), control_labels)
    candidate(0, _densepose_row(0), candidate_labels)

    gate = densepose_benchmark.compare_label_evidence(control, candidate)

    assert gate["passed"] is True
    assert gate["diagnostics"]["raw_label_hashes_exact"] is False
    assert gate["measurements"]["part_iou_mean"] == pytest.approx(0.9999)
    assert gate["measurements"]["part_iou_p05"] == pytest.approx(0.9999)


def test_phase_transition_timing_synchronizes_once_per_first_phase() -> None:
    sync_calls: list[str] = []
    timestamps: dict[str, float] = {}
    clock_values = iter((10.0, 11.25, 14.0))
    cuda = CudaRuntime(
        synchronize=lambda: sync_calls.append("sync"),
        reset_peak_memory_stats=lambda: None,
        max_memory_allocated=lambda: 0,
        max_memory_reserved=lambda: 0,
        gpu_name="NVIDIA A40",
        torch_version="test",
        cuda_version="12.8",
    )

    for phase in ("running_densepose", "running_densepose", "encoding", "writing_outputs"):
        densepose_benchmark._record_phase_transition(
            {"phase": phase, "elapsed_seconds": 999.9},
            cuda=cuda,
            timestamps=timestamps,
            clock=lambda: next(clock_values),
        )

    assert sync_calls == ["sync", "sync", "sync"]
    assert timestamps == {
        "running_densepose": 10.0,
        "encoding": 11.25,
        "writing_outputs": 14.0,
    }
    assert densepose_benchmark._phase_durations(timestamps) == {
        "inference_loop": 1.25,
        "browser_encode": 2.75,
    }


def test_densepose_row_gate_passes_exact_rows_and_rejects_new_unusable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(densepose_benchmark, "EXPECTED_FRAME_COUNT", 2)
    rows = [_densepose_row(0), _densepose_row(1)]

    exact = densepose_benchmark.compare_densepose_rows(rows, json.loads(json.dumps(rows)))
    changed = json.loads(json.dumps(rows))
    changed[1]["usable"] = False
    changed[1]["drop_reason"] = "densepose_missing"
    failed = densepose_benchmark.compare_densepose_rows(rows, changed)

    assert exact["passed"] is True
    assert failed["passed"] is False
    assert failed["checks"]["usable_exact"] is False
    assert failed["checks"]["no_new_unusable_frames"] is False


def test_qa_gate_decodes_and_accepts_identical_videos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(densepose_benchmark, "EXPECTED_FRAME_COUNT", 3)
    control = tmp_path / "control.mp4"
    candidate = tmp_path / "candidate.mp4"
    _write_video(control)
    candidate.write_bytes(control.read_bytes())

    gate = densepose_benchmark.compare_qa_videos(control, candidate)

    assert gate["passed"] is True
    assert gate["measurements"]["decoded_frame_count"] == 3
    assert gate["measurements"]["ssim_p05"] == pytest.approx(1.0)
    assert gate["measurements"]["normalized_mae"] == 0.0


def test_fixture_verification_checks_hash_dimensions_and_frame_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mp4"
    mask = tmp_path / "mask.mp4"
    _write_video(source)
    _write_video(mask)
    monkeypatch.setattr(
        densepose_benchmark, "SOURCE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(
        densepose_benchmark,
        "RUNNER_MASK_SHA256",
        hashlib.sha256(mask.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(densepose_benchmark, "EXPECTED_FRAME_COUNT", 3)
    monkeypatch.setattr(densepose_benchmark, "EXPECTED_WIDTH", 64)
    monkeypatch.setattr(densepose_benchmark, "EXPECTED_HEIGHT", 48)
    monkeypatch.setattr(densepose_benchmark, "EXPECTED_FPS", 10.0)

    result = densepose_benchmark.verify_fixture(source, mask)

    assert result["frame_count"] == 3
    assert result["width"] == 64
    mask.write_bytes(b"modified")
    with pytest.raises(BenchmarkContractError, match="runner-mask"):
        densepose_benchmark.verify_fixture(source, mask)


def test_runtime_configuration_fails_closed_on_unpinned_setting() -> None:
    runtime = {
        "device": "cuda",
        "input_min_size_test": 512,
        "input_max_size_test": 960,
        "target_crop_enabled": False,
        "target_crop_padding_ratio": 0.2,
        "target_crop_padding_pixels": 16,
    }

    with pytest.raises(BenchmarkContractError, match="target_crop_enabled"):
        densepose_benchmark.validate_runtime_configuration(runtime)


def test_runtime_identity_requires_commit_and_immutable_image_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WHODOIRUNLIKE_PROCESSOR_VERSION", raising=False)
    monkeypatch.delenv("WHODOIRUNLIKE_BENCHMARK_IMAGE_DIGEST", raising=False)
    with pytest.raises(BenchmarkContractError, match="runtime identity"):
        densepose_benchmark.runtime_identity()

    monkeypatch.setenv("WHODOIRUNLIKE_PROCESSOR_VERSION", "a" * 40)
    monkeypatch.setenv("WHODOIRUNLIKE_BENCHMARK_IMAGE_DIGEST", "sha256:" + "b" * 64)
    assert densepose_benchmark.runtime_identity() == {
        "processor_version": "a" * 40,
        "image_digest": "sha256:" + "b" * 64,
    }


def test_performance_gate_selects_smallest_candidate_within_five_percent_of_fastest() -> None:
    aggregates = {
        "1": {
            "median_inference_loop_seconds": 1.0,
            "max_peak_cuda_reserved_bytes": 10 * 1024**3,
        },
        "2": {
            "median_inference_loop_seconds": 0.8,
            "max_peak_cuda_reserved_bytes": 12 * 1024**3,
        },
        "4": {
            "median_inference_loop_seconds": 0.76,
            "max_peak_cuda_reserved_bytes": 13 * 1024**3,
        },
        "8": {
            "median_inference_loop_seconds": 0.74,
            "max_peak_cuda_reserved_bytes": 14 * 1024**3,
        },
    }
    comparisons = {str(size): {"passed": True} for size in (1, 2, 4, 8)}

    result = densepose_benchmark.evaluate_performance(
        aggregates,
        comparisons,
        batch_sizes=(1, 2, 4, 8),
    )

    assert result["eligible_batch_sizes"] == [2, 4, 8]
    assert result["selected_batch_size"] == 4
    assert result["gates"]["2"]["measurements"]["inference_speedup"] == 0.2


def test_four_way_canonical_hash_evidence_fits_bounded_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(densepose_benchmark, "EXPECTED_FRAME_COUNT", 260)
    collector = LabelEvidenceCollector()
    labels = np.arange(25, dtype=np.uint8).reshape(5, 5)
    for frame_index in range(260):
        collector(frame_index, _densepose_row(frame_index), labels)
    summary = collector.summary()
    mismatch = {
        "frame_index": 0,
        "control_sha256": "a" * 64,
        "candidate_sha256": "b" * 64,
        "control_shape": [5, 5],
        "candidate_shape": [5, 5],
    }
    payload = {
        "type": densepose_benchmark.BENCHMARK_RESULT_TYPE,
        "status": "complete",
        "label_evidence": {str(size): summary for size in (1, 2, 4, 8)},
        "comparisons": {
            str(size): {"raw_labels": {"mismatches": [mismatch] * 16}} for size in (1, 2, 4, 8)
        },
    }

    bounded = densepose_benchmark.ensure_bounded_response(payload)

    assert bounded["response_bytes"] < densepose_benchmark.MAX_RESPONSE_BYTES


def test_full_matrix_orchestration_is_warm_balanced_bounded_and_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mp4"
    mask = tmp_path / "mask.mp4"
    _write_video(source)
    _write_video(mask)
    source_bytes = source.read_bytes()
    mask_bytes = mask.read_bytes()
    monkeypatch.setattr(densepose_benchmark, "SOURCE_PATH", source)
    monkeypatch.setattr(
        densepose_benchmark, "SOURCE_SHA256", hashlib.sha256(source_bytes).hexdigest()
    )
    monkeypatch.setattr(
        densepose_benchmark,
        "RUNNER_MASK_SHA256",
        hashlib.sha256(mask_bytes).hexdigest(),
    )
    monkeypatch.setattr(densepose_benchmark, "EXPECTED_FRAME_COUNT", 3)
    monkeypatch.setattr(densepose_benchmark, "EXPECTED_WIDTH", 64)
    monkeypatch.setattr(densepose_benchmark, "EXPECTED_HEIGHT", 48)
    monkeypatch.setattr(densepose_benchmark, "EXPECTED_FPS", 10.0)
    monkeypatch.setattr(densepose_benchmark, "clear_densepose_backend_cache", lambda: None)
    monkeypatch.setattr(densepose_benchmark, "load_densepose_backend", lambda **_: object())
    monkeypatch.setattr(
        densepose_benchmark,
        "_densepose_runtime_kwargs",
        lambda: {
            "config_path": Path("cfg.yaml"),
            "weights_path": "weights.pkl",
            "device": "cuda",
            "input_min_size_test": 512,
            "input_max_size_test": 960,
            "target_crop_enabled": True,
            "target_crop_padding_ratio": 0.2,
            "target_crop_padding_pixels": 16,
            "batch_size": 1,
        },
    )
    cuda = CudaRuntime(
        synchronize=lambda: None,
        reset_peak_memory_stats=lambda: None,
        max_memory_allocated=lambda: 1_000,
        max_memory_reserved=lambda: 2_000,
        gpu_name="NVIDIA A40",
        torch_version="test",
        cuda_version="12.8",
    )
    monkeypatch.setattr(densepose_benchmark, "load_cuda_runtime", lambda: cuda)
    calls: list[int] = []

    def fake_run_densepose(**kwargs: Any) -> dict[str, Any]:
        batch_size = int(kwargs["batch_size"])
        calls.append(batch_size)
        run_dir = Path(kwargs["run_dir"])
        rows = [_densepose_row(index) for index in range(3)]
        (run_dir / "densepose.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        _write_video(run_dir / "qa_overlay.mp4")
        callback = kwargs.get("benchmark_evidence_callback")
        if callback is not None:
            for row in rows:
                callback(
                    row["frame_index"],
                    row,
                    np.asarray([[0, 1], [1, 2]], dtype=np.uint8),
                )
        progress = kwargs["progress_callback"]
        for phase, elapsed in (
            ("loading_model", 0.0),
            ("decoding", 0.1),
            ("running_densepose", 0.2),
            ("encoding", 1.0 / batch_size + 0.2),
            ("writing_outputs", 1.0 / batch_size + 0.3),
            ("completed", 1.0 / batch_size + 0.4),
        ):
            progress({"phase": phase, "elapsed_seconds": elapsed})
        return {
            "status": "complete",
            "frame_count": 3,
            "usable_frames": 3,
            "elapsed_seconds": 1.0 / batch_size + 0.4,
            "inference_settings": {
                "batch_size": batch_size,
                "batched_inference_enabled": batch_size > 1,
            },
        }

    monkeypatch.setattr(densepose_benchmark, "run_densepose", fake_run_densepose)
    monkeypatch.setenv("WHODOIRUNLIKE_PROCESSOR_VERSION", "a" * 40)
    monkeypatch.setenv("WHODOIRUNLIKE_BENCHMARK_IMAGE_DIGEST", "sha256:" + "b" * 64)
    request = BenchmarkRequest(
        batch_sizes=(1, 2, 4, 8),
        runner_mask_bytes=mask_bytes,
        request_sha256="c" * 64,
        request_bytes=700_000,
    )

    result = densepose_benchmark.run_benchmark(request)

    assert result["status"] == "complete"
    assert result["passed"] is True
    assert calls[:4] == [1, 2, 4, 8]
    assert calls[4:] == [1, 2, 4, 8, 4, 8, 1, 2, 8, 4, 2, 1]
    assert result["runtime"]["warmup_runs_per_batch"] == 1
    assert result["runtime"]["measured_repetitions"] == 3
    assert result["comparisons"]["8"]["passed"] is True
    assert len(result["label_evidence"]["8"]["frame_hashes"]) == 3
    assert result["response_bytes"] <= densepose_benchmark.MAX_RESPONSE_BYTES
    serialized = json.dumps(result)
    assert str(tmp_path) not in serialized
    assert "weights.pkl" not in serialized
