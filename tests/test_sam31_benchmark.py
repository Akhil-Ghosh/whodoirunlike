from __future__ import annotations

import base64
import gzip

import numpy as np
import pytest

from whodoirunlike import sam31_benchmark, sam31_benchmark_serverless


def test_decode_asset_verifies_gzip_payload_after_decompression(monkeypatch) -> None:
    raw = b'{"frame_index":0}\n'
    spec = sam31_benchmark.AssetSpec(
        encoding="gzip+base64",
        sha256=sam31_benchmark._sha256(raw),
        max_decoded_bytes=1024,
    )
    monkeypatch.setitem(sam31_benchmark.ASSET_SPECS, "tracklets_jsonl", spec)

    decoded = sam31_benchmark._decode_asset(
        "tracklets_jsonl",
        {
            "encoding": "gzip+base64",
            "sha256": spec.sha256,
            "data": base64.b64encode(gzip.compress(raw)).decode("ascii"),
        },
    )

    assert decoded == raw


def test_decode_asset_rejects_hash_mismatch(monkeypatch) -> None:
    raw = b"prompt"
    spec = sam31_benchmark.AssetSpec(
        encoding="base64",
        sha256="0" * 64,
        max_decoded_bytes=1024,
    )
    monkeypatch.setitem(sam31_benchmark.ASSET_SPECS, "person_prompt_json", spec)

    with pytest.raises(ValueError, match="SHA-256"):
        sam31_benchmark._decode_asset(
            "person_prompt_json",
            {
                "encoding": "base64",
                "sha256": spec.sha256,
                "data": base64.b64encode(raw).decode("ascii"),
            },
        )


def test_quality_comparison_reports_exact_agreement() -> None:
    first = np.zeros((20, 30), dtype=np.uint8)
    first[3:12, 5:15] = 1
    second = np.zeros((20, 30), dtype=np.uint8)
    second[4:13, 6:16] = 1
    masks = [first, second]
    track_boxes = {
        0: np.array([4, 2, 16, 13], dtype=np.float32),
        1: np.array([5, 3, 17, 14], dtype=np.float32),
    }

    quality = sam31_benchmark.compare_masks_to_production_baseline(
        masks,
        masks,
        track_boxes=track_boxes,
    )

    assert quality["candidate_nonempty_frames"] == 2
    assert quality["iou"] == {"mean": 1.0, "median": 1.0, "p05": 1.0}
    assert quality["dice_mean"] == 1.0
    assert quality["target_box_centroid_rate"]["candidate"] == 1.0
    assert quality["strict_mask_agreement_gate"]["passed"] is True


def test_serverless_handler_rejects_benchmark_when_disabled(monkeypatch) -> None:
    monkeypatch.delenv("WHODOIRUNLIKE_ENABLE_SAM31_BENCHMARK", raising=False)

    with pytest.raises(RuntimeError, match="disabled"):
        sam31_benchmark_serverless.handler(
            {"input": {"type": "sam31_benchmark"}}
        )


def test_serverless_handler_has_no_production_job_fallback() -> None:
    with pytest.raises(ValueError, match="accepts only"):
        sam31_benchmark_serverless.handler(
            {"input": {"type": "process_clip", "run_id": "should-not-run"}}
        )


def test_serverless_handler_runs_only_dedicated_benchmark(monkeypatch) -> None:
    monkeypatch.setenv("WHODOIRUNLIKE_ENABLE_SAM31_BENCHMARK", "true")
    monkeypatch.setattr(
        sam31_benchmark_serverless,
        "run_benchmark",
        lambda payload: {"variant_id": payload["variant_id"]},
    )

    result = sam31_benchmark_serverless.handler(
        {
            "input": {
                "type": "sam31_benchmark",
                "variant_id": "preseed_single_pass",
            }
        }
    )

    assert result == {"variant_id": "preseed_single_pass"}
