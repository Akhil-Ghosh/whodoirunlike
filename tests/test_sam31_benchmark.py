from __future__ import annotations

import base64
from dataclasses import replace
import gzip
import json
from pathlib import Path
from types import MappingProxyType

import cv2
import numpy as np
import pytest

from whodoirunlike import sam31_benchmark, sam31_benchmark_serverless
from whodoirunlike.mask_artifacts import write_masks_jsonl_from_video
from whodoirunlike.pipeline_parity import materialize_pipeline_fixture
from whodoirunlike.sam31_parity import (
    CANONICAL_FRAME130_FIXTURE_ID,
    get_parity_fixture,
)


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
    assert quality["strict_mask_agreement_gate"]["thresholds"] == {
        "expected_frame_count": 2,
        "iou_mean_min": 0.985,
        "iou_p05_min": 0.975,
        "boundary_f1_mean_min": 0.99,
        "centroid_error_normalized_mean_max": 0.002,
        "temporal_iou_absolute_delta_max": 0.01,
        "fallback_used": False,
        "box_loss_frame_count_max": 0,
        "identity_switch_count_max": 0,
    }


def test_mask_benchmark_exposes_only_the_public_production_entrypoint() -> None:
    source = Path(sam31_benchmark.__file__).read_text(encoding="utf-8")

    assert sam31_benchmark.BENCHMARK_VARIANT_IDS == ("production_candidate_public_entrypoint",)
    assert "from whodoirunlike.sam31_gpu_runner import run_sam31_gpu_mask" in source
    assert "_collect_sam31_masks" not in source
    assert "_load_identity_track_boxes" not in source
    assert "_synchronize_cuda" not in source


def test_request_validation_selects_the_canonical_frame130_fixture(monkeypatch) -> None:
    fixture = get_parity_fixture(CANONICAL_FRAME130_FIXTURE_ID)
    prompt = fixture.prompt.canonical_bytes()
    tracklets = b'{"frame_index":0}\n'
    baseline_mask = b"mask-video"
    specs = {
        "person_prompt_json": sam31_benchmark.AssetSpec(
            encoding="base64",
            sha256=sam31_benchmark._sha256(prompt),
            max_decoded_bytes=16 * 1024,
        ),
        "tracklets_jsonl": sam31_benchmark.AssetSpec(
            encoding="gzip+base64",
            sha256=sam31_benchmark._sha256(tracklets),
            max_decoded_bytes=1024,
        ),
        "baseline_runner_mask_mp4": sam31_benchmark.AssetSpec(
            encoding="base64",
            sha256=sam31_benchmark._sha256(baseline_mask),
            max_decoded_bytes=1024,
        ),
    }
    monkeypatch.setattr(sam31_benchmark, "ASSET_SPECS", specs)

    fixture_id, variant_id, decoded = sam31_benchmark._validate_request(
        {
            "type": sam31_benchmark.BENCHMARK_TYPE,
            "schema_version": sam31_benchmark.BENCHMARK_SCHEMA_VERSION,
            "fixture_id": CANONICAL_FRAME130_FIXTURE_ID,
            "variant_id": sam31_benchmark.BENCHMARK_VARIANT_ID,
            "assets": {
                "person_prompt_json": {
                    "encoding": "base64",
                    "sha256": specs["person_prompt_json"].sha256,
                    "data": base64.b64encode(prompt).decode("ascii"),
                },
                "tracklets_jsonl": {
                    "encoding": "gzip+base64",
                    "sha256": specs["tracklets_jsonl"].sha256,
                    "data": base64.b64encode(gzip.compress(tracklets)).decode("ascii"),
                },
                "baseline_runner_mask_mp4": {
                    "encoding": "base64",
                    "sha256": specs["baseline_runner_mask_mp4"].sha256,
                    "data": base64.b64encode(baseline_mask).decode("ascii"),
                },
            },
        }
    )

    assert fixture_id == CANONICAL_FRAME130_FIXTURE_ID
    assert variant_id == sam31_benchmark.BENCHMARK_VARIANT_ID
    assert decoded == {
        "person_prompt_json": prompt,
        "tracklets_jsonl": tracklets,
        "baseline_runner_mask_mp4": baseline_mask,
    }


def test_serverless_handler_rejects_benchmark_when_disabled(monkeypatch) -> None:
    monkeypatch.delenv("WHODOIRUNLIKE_ENABLE_SAM31_BENCHMARK", raising=False)

    with pytest.raises(RuntimeError, match="disabled"):
        sam31_benchmark_serverless.handler({"input": {"type": "sam31_benchmark"}})


def test_serverless_health_advertises_canonical_frame130_fixture() -> None:
    health = sam31_benchmark_serverless.handler({"input": {"type": "health"}})

    assert CANONICAL_FRAME130_FIXTURE_ID in health["fixture_ids"]


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


def test_serverless_handler_routes_full_scope_to_pipeline_parity(monkeypatch) -> None:
    monkeypatch.setenv("WHODOIRUNLIKE_ENABLE_SAM31_BENCHMARK", "true")
    monkeypatch.setattr(
        sam31_benchmark_serverless,
        "run_full_pipeline_benchmark",
        lambda payload: {"scope": payload["scope"]},
    )

    result = sam31_benchmark_serverless.handler(
        {
            "input": {
                "type": "sam31_benchmark",
                "scope": "full",
            }
        }
    )

    assert result == {"scope": "full"}


def test_serverless_health_advertises_full_profiles_and_scopes() -> None:
    health = sam31_benchmark_serverless.handler({"input": {"type": "health"}})

    assert health["scope_ids"] == ["mask", "full"]
    assert health["default_full_profile_ids"] == [
        "downstream_baseline_control",
        "downstream_candidate_control",
    ]
    assert "production_control" in health["full_profile_ids"]
    assert "production_candidate_schedule_only" in health["full_profile_ids"]
    assert "production_final_candidate" in health["full_profile_ids"]


def test_serverless_final_candidate_advertises_only_the_safe_full_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WHODOIRUNLIKE_BENCHMARK_IMAGE_ROLE", "final_candidate")
    monkeypatch.delenv("WHODOIRUNLIKE_ENFORCE_BASE_CONTRACT", raising=False)
    monkeypatch.setenv("WHODOIRUNLIKE_SAM31_GPU_EXACT_CV2_LOADER", "true")
    monkeypatch.setenv("WHODOIRUNLIKE_SAM31_GPU_EXACT_CV2_CHUNK_FRAMES", "8")
    monkeypatch.setenv("WHODOIRUNLIKE_SAM31_GPU_EXACT_CV2_MAX_FRAMES", "600")
    monkeypatch.setenv(
        "WHODOIRUNLIKE_SAM31_GPU_EXACT_CV2_MAX_DESTINATION_BYTES",
        str(8 * 1024**3),
    )
    monkeypatch.setenv("WHODOIRUNLIKE_PROCESSOR_CONCURRENCY", "1")
    monkeypatch.setattr(
        sam31_benchmark_serverless,
        "verify_densepose_weights",
        lambda: {"passed": True},
    )

    health = sam31_benchmark_serverless.handler({"input": {"type": "health"}})

    assert health["scope_ids"] == ["full"]
    assert health["full_profile_ids"] == ["production_final_candidate"]
    assert health["default_full_profile_ids"] == ["production_final_candidate"]
    assert health["sam31_input_loader"]["mode"] == "exact_cv2"
    assert health["sam31_input_loader"]["concurrency_ready"] is True
    assert health["densepose_weights"]["passed"] is True


def test_serverless_fails_closed_when_exact_base_contract_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WHODOIRUNLIKE_ENABLE_SAM31_BENCHMARK", "true")
    monkeypatch.setenv("WHODOIRUNLIKE_ENFORCE_BASE_CONTRACT", "true")
    monkeypatch.setenv("WHODOIRUNLIKE_BENCHMARK_IMAGE_ROLE", "schedule_only")
    monkeypatch.setattr(
        sam31_benchmark_serverless,
        "verify_non_overlay_production_files",
        lambda _role: {
            "passed": False,
            "mismatches": ["full_pipeline.py"],
        },
    )

    with pytest.raises(RuntimeError, match="full_pipeline.py"):
        sam31_benchmark_serverless.handler(
            {
                "input": {
                    "type": "sam31_benchmark",
                    "scope": "full",
                }
            }
        )


def test_serverless_fails_closed_when_final_candidate_identity_is_not_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WHODOIRUNLIKE_ENABLE_SAM31_BENCHMARK", "true")
    monkeypatch.setenv("WHODOIRUNLIKE_ENFORCE_BASE_CONTRACT", "true")
    monkeypatch.setenv("WHODOIRUNLIKE_BENCHMARK_IMAGE_ROLE", "final_candidate")
    monkeypatch.setenv("WHODOIRUNLIKE_CANDIDATE_COMMIT", "not-a-commit")
    digest = "sha256:" + "b" * 64
    monkeypatch.setenv("WHODOIRUNLIKE_CANDIDATE_IMAGE_DIGEST", digest)
    monkeypatch.setenv(
        "WHODOIRUNLIKE_FINAL_CANDIDATE_IMAGE_REFERENCE",
        "ghcr.io/akhil-ghosh/whodoirunlike-runpod-processor@" + digest,
    )
    monkeypatch.setattr(
        sam31_benchmark_serverless,
        "verify_non_overlay_production_files",
        lambda _role: {"passed": True, "mismatches": []},
    )

    with pytest.raises(RuntimeError, match="commit_is_exact"):
        sam31_benchmark_serverless.handler(
            {
                "input": {
                    "type": "sam31_benchmark",
                    "scope": "full",
                }
            }
        )


def test_serverless_final_candidate_rejects_nonfinal_profiles_and_missing_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WHODOIRUNLIKE_ENABLE_SAM31_BENCHMARK", "true")
    monkeypatch.delenv("WHODOIRUNLIKE_ENFORCE_BASE_CONTRACT", raising=False)
    monkeypatch.setenv("WHODOIRUNLIKE_BENCHMARK_IMAGE_ROLE", "final_candidate")

    with pytest.raises(ValueError, match="exactly production_final_candidate"):
        sam31_benchmark_serverless.handler(
            {
                "input": {
                    "type": "sam31_benchmark",
                    "scope": "full",
                    "profile_ids": ["production_control"],
                    "artifact_sink": {},
                }
            }
        )

    with pytest.raises(ValueError, match="control handoff sink"):
        sam31_benchmark_serverless.handler(
            {
                "input": {
                    "type": "sam31_benchmark",
                    "scope": "full",
                    "profile_ids": ["production_final_candidate"],
                }
            }
        )


def _write_test_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (64, 48),
    )
    assert writer.isOpened()
    for index in range(3):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[10:35, 15 + index : 30 + index] = 255
        writer.write(frame)
    writer.release()


def test_public_candidate_mask_stage_persists_mask_for_downstream_arms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mp4"
    baseline = tmp_path / "baseline.mp4"
    _write_test_video(source)
    _write_test_video(baseline)
    base_fixture = get_parity_fixture(CANONICAL_FRAME130_FIXTURE_ID)
    fixture = replace(
        base_fixture,
        frame_count=3,
        width=64,
        height=48,
        source_sha256=sam31_benchmark._file_sha256(source),
        asset_sha256=MappingProxyType(
            {
                **base_fixture.asset_sha256,
                "baseline_runner_mask_mp4": sam31_benchmark._file_sha256(baseline),
            }
        ),
    )
    monkeypatch.setattr(sam31_benchmark, "_FIXTURE", fixture)
    run_dir = tmp_path / "run"
    materialize_pipeline_fixture(
        run_dir=run_dir,
        source_path=source,
        assets={
            "person_prompt_json": base_fixture.prompt.canonical_bytes(),
            "tracklets_jsonl": b'{"frame_index":0,"is_target":true}\n',
            "baseline_runner_mask_mp4": baseline.read_bytes(),
        },
        profile_id="candidate",
    )

    def fake_public_runner(candidate_run_dir: Path) -> dict[str, object]:
        mask_path = candidate_run_dir / "runner_mask.mp4"
        shutil_source = baseline.read_bytes()
        mask_path.write_bytes(shutil_source)
        write_masks_jsonl_from_video(mask_path, candidate_run_dir / "masks.jsonl")
        return {
            "backend": "sam31_gpu",
            "frame_count": 3,
            "prompting": {"identity_filter": {"rejected_frames": 0}},
            "fallback": None,
        }

    result = sam31_benchmark.run_candidate_mask_stage(
        run_dir,
        mask_runner=fake_public_runner,
    )

    assert result["quality_vs_production_baseline"]["strict_mask_agreement_gate"]["passed"] is True
    assert (run_dir / "baseline_runner_mask.mp4").is_file()
    assert (run_dir / "runner_mask.mp4").is_file()
    assert (run_dir / "masks.jsonl").is_file()
    assert "/" not in result["persisted_artifacts"]["runner_mask"]["name"]
    assert str(tmp_path) not in json.dumps(result)
