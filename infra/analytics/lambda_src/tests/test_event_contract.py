from __future__ import annotations

import sys
import unittest
from pathlib import Path


LAMBDA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAMBDA_ROOT))

from event_contract import ContractError, event_partition, flatten_event, validate_event  # noqa: E402


def event_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "event_id": "01952dd0-31c3-7ca1-9a7a-f7080b16bfa1",
        "run_id": "6acc0d87-38cc-49f5-b268-db7e7bac2771",
        "attempt_id": "01952dd0-31c3-7ca1-9a7a-f7080b16bfa2",
        "sequence": 4,
        "event_type": "span_completed",
        "event_time": "2026-07-09T21:03:04.123Z",
        "stage": "runner_mask",
        "span": "inference",
        "status": "complete",
        "elapsed_seconds": 12.5,
        "progress": {"processed_frames": 50, "total_frames": 100, "percent": 0.5},
        "input": {
            "duration_seconds": 8.2,
            "duration_bucket": "5_10s",
            "resolution_bucket": "hd",
            "frame_count": 246,
            "width": 1280,
            "height": 720,
        },
        "runtime": {
            "service": "whodoirunlike-processor",
            "environment": "production",
            "execution_environment": "runpod",
            "runpod_endpoint_id": "endpoint-old",
            "attempt_number": 2,
            "backend": "sam31_gpu",
            "gpu_type": "A40",
            "cold_start": False,
        },
        "resources": {
            "rss_mb": 384.0,
            "peak_rss_bytes": 536870912,
            "cuda_peak_mb": 4321.0,
        },
        "measurements": {
            "artifact_type": "fused_overlay",
            "bytes": 4096,
            "milliseconds_per_frame": 50.8,
            "timing_basis": "runpod_delay_time",
            "cache_hit": True,
            "model_build_seconds": 0.0,
            "predictor_lock_wait_seconds": 0.004,
            "data_ready_seconds": 78.25,
            "presentation_tail_seconds": 16.75,
        },
    }
    payload.update(overrides)
    return payload


class EventContractTests(unittest.TestCase):
    def test_validates_and_flattens_event(self) -> None:
        event = validate_event(event_payload())
        flat = flatten_event(event, ingested_at="2026-07-09T21:03:05Z")
        self.assertEqual(flat["stage"], "runner_mask")
        self.assertEqual(flat["processed_frames"], 50)
        self.assertEqual(flat["gpu_type"], "A40")
        self.assertEqual(flat["duration_bucket"], "5_10s")
        self.assertEqual(flat["resolution_bucket"], "hd")
        self.assertEqual(flat["attempt_number"], 2)
        self.assertEqual(flat["runpod_endpoint_id"], "endpoint-old")
        self.assertEqual(flat["rss_mb"], 384.0)
        self.assertEqual(flat["peak_rss_mb"], 512.0)
        self.assertEqual(flat["artifact_type"], "fused_overlay")
        self.assertEqual(flat["artifact_size_bytes"], 4096)
        self.assertEqual(flat["milliseconds_per_frame"], 50.8)
        self.assertEqual(flat["timing_basis"], "runpod_delay_time")
        self.assertIs(flat["cache_hit"], True)
        self.assertEqual(flat["model_build_seconds"], 0.0)
        self.assertEqual(flat["predictor_lock_wait_seconds"], 0.004)
        self.assertEqual(flat["data_ready_seconds"], 78.25)
        self.assertEqual(flat["presentation_tail_seconds"], 16.75)
        self.assertEqual(event_partition(event["event_time"]), ("2026-07-09", "21"))

    def test_rejects_unknown_field(self) -> None:
        with self.assertRaisesRegex(ContractError, "unknown event fields"):
            validate_event(event_payload(secret="nope"))

    def test_rejects_missing_span(self) -> None:
        with self.assertRaisesRegex(ContractError, "span is required"):
            validate_event(event_payload(span=None))

    def test_rejects_nan(self) -> None:
        with self.assertRaisesRegex(ContractError, "finite"):
            validate_event(event_payload(elapsed_seconds=float("nan")))

    def test_rejects_out_of_range_sequence_status_and_elapsed(self) -> None:
        with self.assertRaisesRegex(ContractError, "sequence"):
            validate_event(event_payload(sequence=0))
        with self.assertRaisesRegex(ContractError, "status"):
            validate_event(event_payload(status="completed"))
        with self.assertRaisesRegex(ContractError, "elapsed_seconds"):
            validate_event(event_payload(elapsed_seconds=31_536_001))

    def test_failed_event_requires_sanitized_error(self) -> None:
        with self.assertRaisesRegex(ContractError, "require sanitized error"):
            validate_event(
                event_payload(event_type="stage_failed", span=None, error=None)
            )


if __name__ == "__main__":
    unittest.main()
