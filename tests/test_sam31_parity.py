from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

from whodoirunlike import sam31_parity
from whodoirunlike.sam31_parity import (
    CANONICAL_FRAME130_FIXTURE_ID,
    MaskQualityMeasurements,
    PARITY_FIXTURES,
    evaluate_strict_mask_gate,
    get_parity_fixture,
    load_local_fixture_assets,
    validate_prompt_for_fixture,
    verify_non_overlay_production_files,
)


def test_canonical_frame130_fixture_reproduces_audited_prompt_hash() -> None:
    fixture = get_parity_fixture(CANONICAL_FRAME130_FIXTURE_ID)

    prompt_bytes = fixture.prompt.canonical_bytes()
    validation = validate_prompt_for_fixture(prompt_bytes, fixture=fixture)

    assert hashlib.sha256(prompt_bytes).hexdigest() == fixture.prompt.raw_sha256
    assert validation == {
        "raw_sha256": fixture.prompt.raw_sha256,
        "raw_hash_matches": True,
        "semantic_sha256": fixture.prompt.semantic_sha256,
        "semantic_hash_matches": True,
    }


def test_canonical_prompt_accepts_a_local_image_path_without_weakening_semantics() -> None:
    fixture = get_parity_fixture(CANONICAL_FRAME130_FIXTURE_ID)
    prompt = json.loads(fixture.prompt.canonical_bytes())
    prompt["frame"]["image_path"] = "/private/local-fixtures/prompt_frame.jpg"
    prompt_bytes = (json.dumps(prompt, indent=2) + "\n").encode("utf-8")

    validation = validate_prompt_for_fixture(prompt_bytes, fixture=fixture)

    assert validation["raw_hash_matches"] is False
    assert validation["semantic_hash_matches"] is True


@pytest.mark.parametrize(
    ("field_path", "replacement"),
    [
        (("source",), "hosted_upload_user_prompt_v1"),
        (("frame", "frame_index"), 129),
        (("frame", "time_seconds"), 4.3),
        (("selection", "box", "x"), 0.5),
        (("selection", "type"), "box"),
        (("subject", "profile_id"), "another_runner"),
    ],
)
def test_canonical_prompt_rejects_any_behavioral_semantic_change(
    field_path: tuple[str, ...],
    replacement: object,
) -> None:
    fixture = get_parity_fixture(CANONICAL_FRAME130_FIXTURE_ID)
    prompt = json.loads(fixture.prompt.canonical_bytes())
    target = prompt
    for field in field_path[:-1]:
        target = target[field]
    target[field_path[-1]] = replacement

    with pytest.raises(ValueError, match="semantics"):
        validate_prompt_for_fixture(
            (json.dumps(prompt, indent=2) + "\n").encode("utf-8"),
            fixture=fixture,
        )


def test_canonical_frame130_fixture_registers_all_audited_asset_hashes() -> None:
    fixture = get_parity_fixture(CANONICAL_FRAME130_FIXTURE_ID)

    assert fixture.source_sha256 == (
        "a8146591119c5439cc01168df63fa6144a7a55ff6817726946e1e8f5bc381617"
    )
    assert fixture.asset_sha256 == {
        "person_prompt_json": fixture.prompt.raw_sha256,
        "tracklets_jsonl": "d886295534908392b43b7ac8d17e1df98efc5b566622b985f107cb05606f96d9",
        "baseline_runner_mask_mp4": (
            "f7bb2d1ed00767ed2866c5b3a57b47361a591f1dbf090a5089d187f9ae410ef7"
        ),
    }
    assert (fixture.frame_count, fixture.width, fixture.height) == (260, 960, 540)
    assert fixture.baseline_run_id == "2fa255e3-aefe-4fb2-b41e-f36c73c09546"
    assert fixture.baseline_attempt_id == "46f7dbe7-6899-46ef-a971-9a7261a37480"
    assert fixture.baseline_processor_version == "8b33d07cd129cb6878f4630af133bf30c291914b"
    assert fixture.local_asset_paths == {
        "person_prompt_json": Path("person_prompt.json"),
        "tracklets_jsonl": Path("tracklets.jsonl"),
        "baseline_runner_mask_mp4": Path("runner_mask.mp4"),
    }


def test_local_fixture_loader_verifies_ignored_assets_and_generates_missing_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_fixture = get_parity_fixture(CANONICAL_FRAME130_FIXTURE_ID)
    tracklets = b'{"frame_index":0}\n'
    baseline_mask = b"local-mask-fixture"
    fixture = replace(
        base_fixture,
        fixture_id="test-local-frame130",
        asset_sha256=MappingProxyType(
            {
                "person_prompt_json": base_fixture.prompt.raw_sha256,
                "tracklets_jsonl": hashlib.sha256(tracklets).hexdigest(),
                "baseline_runner_mask_mp4": hashlib.sha256(baseline_mask).hexdigest(),
            }
        ),
    )
    monkeypatch.setitem(PARITY_FIXTURES, fixture.fixture_id, fixture)
    (tmp_path / "tracklets.jsonl").write_bytes(tracklets)
    (tmp_path / "runner_mask.mp4").write_bytes(baseline_mask)

    assets = load_local_fixture_assets(fixture.fixture_id, tmp_path)

    assert assets == {
        "person_prompt_json": fixture.prompt.canonical_bytes(),
        "tracklets_jsonl": tracklets,
        "baseline_runner_mask_mp4": baseline_mask,
    }


def test_local_fixture_loader_rejects_a_modified_binary_asset(
    tmp_path: Path,
) -> None:
    fixture = get_parity_fixture(CANONICAL_FRAME130_FIXTURE_ID)
    (tmp_path / "tracklets.jsonl").write_bytes(b"modified-tracklets")
    (tmp_path / "runner_mask.mp4").write_bytes(b"modified-mask")

    with pytest.raises(ValueError, match="tracklets_jsonl"):
        load_local_fixture_assets(fixture.fixture_id, tmp_path)


def test_strict_mask_gate_passes_at_every_registered_quality_boundary() -> None:
    gate = evaluate_strict_mask_gate(_passing_measurements(), expected_frame_count=260)

    assert gate["passed"] is True
    assert all(gate["checks"].values())


def _passing_measurements() -> MaskQualityMeasurements:
    return MaskQualityMeasurements(
        reference_frame_count=260,
        candidate_frame_count=260,
        candidate_nonempty_frame_count=260,
        iou_mean=0.985,
        iou_p05=0.975,
        boundary_f1_mean=0.99,
        centroid_error_normalized_mean=0.002,
        temporal_iou_absolute_delta=0.01,
        fallback_used=False,
        box_loss_frame_count=0,
        identity_switch_count=0,
    )


@pytest.mark.parametrize(
    ("field", "violating_value", "check"),
    [
        ("reference_frame_count", 259, "reference_frame_count_exact"),
        ("candidate_frame_count", 259, "candidate_frame_count_exact"),
        (
            "candidate_nonempty_frame_count",
            259,
            "candidate_nonempty_frame_count_exact",
        ),
        ("iou_mean", 0.984999, "iou_mean"),
        ("iou_p05", 0.974999, "iou_p05"),
        ("boundary_f1_mean", 0.989999, "boundary_f1_mean"),
        (
            "centroid_error_normalized_mean",
            0.002001,
            "centroid_error_normalized_mean",
        ),
        (
            "temporal_iou_absolute_delta",
            0.010001,
            "temporal_iou_absolute_delta",
        ),
        ("fallback_used", True, "no_fallback"),
        ("box_loss_frame_count", 1, "no_box_loss"),
        ("identity_switch_count", 1, "no_identity_switches"),
    ],
)
def test_each_strict_mask_gate_rejects_its_own_violation(
    field: str,
    violating_value: object,
    check: str,
) -> None:
    measurements = replace(_passing_measurements(), **{field: violating_value})

    gate = evaluate_strict_mask_gate(measurements, expected_frame_count=260)

    assert gate["passed"] is False
    assert gate["checks"][check] is False


def test_non_overlay_base_contract_hashes_every_registered_production_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {
        "full_pipeline.py": hashlib.sha256(b"control-full").hexdigest(),
        "sam31_gpu_runner.py": hashlib.sha256(b"control-mask").hexdigest(),
    }
    monkeypatch.setattr(
        sam31_parity,
        "NON_OVERLAY_PRODUCTION_SHA256",
        MappingProxyType({"control": MappingProxyType(expected)}),
    )
    (tmp_path / "full_pipeline.py").write_bytes(b"control-full")
    (tmp_path / "sam31_gpu_runner.py").write_bytes(b"control-mask")

    result = verify_non_overlay_production_files("control", module_root=tmp_path)

    assert result["passed"] is True
    assert result["checked_file_count"] == 2
    assert result["mismatches"] == []


def test_non_overlay_base_contract_fails_closed_on_one_changed_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sam31_parity,
        "NON_OVERLAY_PRODUCTION_SHA256",
        MappingProxyType(
            {
                "candidate": MappingProxyType(
                    {"full_pipeline.py": hashlib.sha256(b"expected").hexdigest()}
                )
            }
        ),
    )
    (tmp_path / "full_pipeline.py").write_bytes(b"changed")

    result = verify_non_overlay_production_files("candidate", module_root=tmp_path)

    assert result["passed"] is False
    assert result["mismatches"] == ["full_pipeline.py"]
