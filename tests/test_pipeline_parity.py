from __future__ import annotations

import base64
from dataclasses import replace
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
import threading
from types import MappingProxyType

import cv2
import numpy as np
import pytest

from whodoirunlike import pipeline_parity
from whodoirunlike.pipeline_parity import (
    DensePoseParityMeasurements,
    FusionParityMeasurements,
    FeatureParityMeasurements,
    ArtifactParityMeasurements,
    QcParityMeasurements,
    VideoParityMeasurements,
    PIPELINE_BENCHMARK_PROFILES,
    PipelineStageFunctions,
    PoseParityMeasurements,
    compare_densepose_rows,
    compare_fusion_rows,
    compare_feature_artifacts,
    compare_artifact_contracts,
    compare_qc_payloads,
    compare_runner_mask_videos,
    compare_video_contracts,
    compare_pipeline_runs,
    compare_pose_rows,
    evaluate_densepose_parity,
    evaluate_exact_loader_gate,
    evaluate_fusion_parity,
    evaluate_feature_parity,
    evaluate_artifact_parity,
    evaluate_qc_parity,
    evaluate_video_parity,
    resolve_pipeline_profiles,
    materialize_pipeline_fixture,
    run_pipeline_profile,
    run_full_pipeline_benchmark,
    evaluate_pose_parity,
    validate_full_benchmark_request,
)
from whodoirunlike.sam31_parity import (
    CANONICAL_FRAME130_FIXTURE_ID,
    FINAL_CANDIDATE_PRODUCTION_FILES,
    PARITY_FIXTURES,
    get_parity_fixture,
)


def _passing_pose_measurements() -> PoseParityMeasurements:
    return PoseParityMeasurements(
        control_frame_count=260,
        candidate_frame_count=260,
        schema_match=True,
        control_schema_preserved=True,
        required_fields_present=True,
        aligned_frame_count=260,
        usable_agreement_rate=0.99,
        new_unusable_frame_count=1,
        common_visible_point_count=1,
        pck_at_001_diagonal=0.99,
        joint_error_median_normalized=0.002,
        joint_error_p95_normalized=0.01,
        visibility_mae=0.01,
    )


def test_pose_parity_passes_at_all_registered_boundaries() -> None:
    gate = evaluate_pose_parity(_passing_pose_measurements(), expected_frame_count=260)

    assert gate["passed"] is True
    assert all(gate["checks"].values())


@pytest.mark.parametrize(
    ("field", "value", "check"),
    [
        ("control_frame_count", 259, "control_frame_count_exact"),
        ("candidate_frame_count", 259, "candidate_frame_count_exact"),
        ("control_schema_preserved", False, "control_schema_preserved"),
        ("required_fields_present", False, "required_fields_present"),
        ("aligned_frame_count", 259, "frame_indices_aligned"),
        ("usable_agreement_rate", 0.989999, "usable_agreement"),
        ("new_unusable_frame_count", 2, "new_unusable_frames"),
        ("common_visible_point_count", 0, "common_visible_point_evidence"),
        ("pck_at_001_diagonal", 0.989999, "pck_at_001_diagonal"),
        (
            "joint_error_median_normalized",
            0.002001,
            "joint_error_median_normalized",
        ),
        (
            "joint_error_p95_normalized",
            0.010001,
            "joint_error_p95_normalized",
        ),
        ("visibility_mae", 0.010001, "visibility_mae"),
    ],
)
def test_each_pose_gate_rejects_its_own_violation(
    field: str,
    value: object,
    check: str,
) -> None:
    measurements = replace(_passing_pose_measurements(), **{field: value})

    gate = evaluate_pose_parity(measurements, expected_frame_count=260)

    assert gate["passed"] is False
    assert gate["checks"][check] is False


def _pose_row(frame_index: int, *, x: float = 0.25) -> dict[str, object]:
    return {
        "frame_index": frame_index,
        "usable": True,
        "visibility_mean": 0.9,
        "landmarks": [
            {
                "index": 0,
                "name": "nose",
                "x": x,
                "y": 0.4,
                "visibility": 0.9,
                "missing": False,
            }
        ],
    }


def test_pose_row_comparison_measures_pck_and_visibility_on_common_points() -> None:
    control = [_pose_row(0), _pose_row(1)]
    candidate = [_pose_row(0, x=0.251), _pose_row(1, x=0.249)]

    gate = compare_pose_rows(control, candidate, expected_frame_count=2)

    assert gate["passed"] is True
    assert gate["measurements"]["common_visible_point_count"] == 2
    assert gate["measurements"]["pck_at_001_diagonal"] == 1.0


def test_pose_row_comparison_allows_additive_candidate_metadata_but_rejects_type_changes() -> None:
    control = [_pose_row(0)]
    additive_candidate = [
        {
            **_pose_row(0),
            "inference_settings": {"device": "cuda", "detector_enabled": False},
        }
    ]

    additive_gate = compare_pose_rows(control, additive_candidate, expected_frame_count=1)

    assert additive_gate["passed"] is True
    assert additive_gate["measurements"]["schema_match"] is False
    assert additive_gate["measurements"]["control_schema_preserved"] is True
    assert additive_gate["schema_compatibility"]["candidate_only_path_count"] > 0

    changed_type_candidate = [{**_pose_row(0), "visibility_mean": "0.9"}]
    changed_type_gate = compare_pose_rows(
        control,
        changed_type_candidate,
        expected_frame_count=1,
    )

    assert changed_type_gate["passed"] is False
    assert changed_type_gate["measurements"]["control_schema_preserved"] is False
    assert changed_type_gate["schema_compatibility"]["type_change_count"] == 1


def _passing_densepose_measurements() -> DensePoseParityMeasurements:
    return DensePoseParityMeasurements(
        control_frame_count=260,
        candidate_frame_count=260,
        schema_match=True,
        control_schema_preserved=True,
        required_fields_present=True,
        aligned_frame_count=260,
        usable_agreement_rate=0.99,
        new_unusable_frame_count=1,
        common_usable_frame_count=1,
        part_jaccard_mean=0.99,
        part_jaccard_p05=0.95,
        centroid_error_normalized_mean=0.005,
        centroid_error_normalized_p95=0.015,
        bbox_iou_p05=0.95,
        coverage_mae=0.01,
        mask_overlap_mae=0.01,
    )


def test_densepose_parity_passes_at_all_registered_boundaries() -> None:
    gate = evaluate_densepose_parity(
        _passing_densepose_measurements(),
        expected_frame_count=260,
    )

    assert gate["passed"] is True
    assert all(gate["checks"].values())


@pytest.mark.parametrize(
    ("field", "value", "check"),
    [
        ("control_frame_count", 259, "control_frame_count_exact"),
        ("candidate_frame_count", 259, "candidate_frame_count_exact"),
        ("control_schema_preserved", False, "control_schema_preserved"),
        ("required_fields_present", False, "required_fields_present"),
        ("aligned_frame_count", 259, "frame_indices_aligned"),
        ("usable_agreement_rate", 0.989999, "usable_agreement"),
        ("new_unusable_frame_count", 2, "new_unusable_frames"),
        ("common_usable_frame_count", 0, "common_usable_frame_evidence"),
        ("part_jaccard_mean", 0.989999, "part_jaccard_mean"),
        ("part_jaccard_p05", 0.949999, "part_jaccard_p05"),
        (
            "centroid_error_normalized_mean",
            0.005001,
            "centroid_error_normalized_mean",
        ),
        (
            "centroid_error_normalized_p95",
            0.015001,
            "centroid_error_normalized_p95",
        ),
        ("bbox_iou_p05", 0.949999, "bbox_iou_p05"),
        ("coverage_mae", 0.010001, "coverage_mae"),
        ("mask_overlap_mae", 0.010001, "mask_overlap_mae"),
    ],
)
def test_each_densepose_gate_rejects_its_own_violation(
    field: str,
    value: object,
    check: str,
) -> None:
    measurements = replace(_passing_densepose_measurements(), **{field: value})

    gate = evaluate_densepose_parity(measurements, expected_frame_count=260)

    assert gate["passed"] is False
    assert gate["checks"][check] is False


def _densepose_row(frame_index: int, *, centroid_x: float = 0.25) -> dict[str, object]:
    return {
        "frame_index": frame_index,
        "usable": True,
        "part_ids": [1, 2],
        "part_centroids": {
            "1": {"x": centroid_x, "y": 0.4},
            "2": {"x": 0.3, "y": 0.5},
        },
        "densepose_coverage": 0.3,
        "mask_overlap": 0.8,
        "bbox": [10, 20, 30, 40],
    }


def test_densepose_row_comparison_measures_parts_centroids_and_coverage() -> None:
    control = [_densepose_row(0), _densepose_row(1)]
    candidate = [_densepose_row(0, centroid_x=0.251), _densepose_row(1, centroid_x=0.249)]

    gate = compare_densepose_rows(control, candidate, expected_frame_count=2)

    assert gate["passed"] is True
    assert gate["measurements"]["common_usable_frame_count"] == 2
    assert gate["measurements"]["part_jaccard_mean"] == 1.0


def test_densepose_row_comparison_allows_inference_metadata_but_rejects_type_changes() -> None:
    control = [_densepose_row(0)]
    additive_candidate = [
        {
            **_densepose_row(0),
            "inference_input": {
                "target_crop_enabled": True,
                "crop_bbox": [0, 0, 960, 540],
            },
        }
    ]

    additive_gate = compare_densepose_rows(
        control,
        additive_candidate,
        expected_frame_count=1,
    )

    assert additive_gate["passed"] is True
    assert additive_gate["measurements"]["schema_match"] is False
    assert additive_gate["measurements"]["control_schema_preserved"] is True

    changed_type_candidate = [{**_densepose_row(0), "densepose_coverage": "0.3"}]
    changed_type_gate = compare_densepose_rows(
        control,
        changed_type_candidate,
        expected_frame_count=1,
    )

    assert changed_type_gate["passed"] is False
    assert changed_type_gate["measurements"]["control_schema_preserved"] is False


def _passing_fusion_measurements() -> FusionParityMeasurements:
    return FusionParityMeasurements(
        control_frame_count=260,
        candidate_frame_count=260,
        schema_match=True,
        required_fields_present=True,
        aligned_frame_count=260,
        frame_state_agreement_rate=0.99,
        risk_state_increase_count=0,
        usable_agreement_rate=0.99,
        confidence_mae=0.01,
        confidence_mean_drop=0.01,
        common_joint_weight_count=1,
        joint_weight_mae=0.01,
        joint_weight_p95_error=0.03,
    )


def test_fusion_parity_passes_at_all_registered_boundaries() -> None:
    gate = evaluate_fusion_parity(
        _passing_fusion_measurements(),
        expected_frame_count=260,
    )

    assert gate["passed"] is True
    assert all(gate["checks"].values())


@pytest.mark.parametrize(
    ("field", "value", "check"),
    [
        ("control_frame_count", 259, "control_frame_count_exact"),
        ("candidate_frame_count", 259, "candidate_frame_count_exact"),
        ("schema_match", False, "schema_match"),
        ("required_fields_present", False, "required_fields_present"),
        ("aligned_frame_count", 259, "frame_indices_aligned"),
        ("frame_state_agreement_rate", 0.989999, "frame_state_agreement"),
        ("risk_state_increase_count", 1, "no_new_risk_states"),
        ("usable_agreement_rate", 0.989999, "usable_agreement"),
        ("confidence_mae", 0.010001, "confidence_mae"),
        ("confidence_mean_drop", 0.010001, "confidence_mean_drop"),
        ("common_joint_weight_count", 0, "common_joint_weight_evidence"),
        ("joint_weight_mae", 0.010001, "joint_weight_mae"),
        ("joint_weight_p95_error", 0.030001, "joint_weight_p95_error"),
    ],
)
def test_each_fusion_gate_rejects_its_own_violation(
    field: str,
    value: object,
    check: str,
) -> None:
    measurements = replace(_passing_fusion_measurements(), **{field: value})

    gate = evaluate_fusion_parity(measurements, expected_frame_count=260)

    assert gate["passed"] is False
    assert gate["checks"][check] is False


def _fusion_row(frame_index: int, *, weight: float = 0.8) -> dict[str, object]:
    return {
        "frame_index": frame_index,
        "frame_state": "usable",
        "usable": True,
        "frame_confidence": 0.8,
        "joint_weights": [{"index": 0, "name": "nose", "weight": weight}],
    }


def test_fusion_row_comparison_measures_state_confidence_and_joint_weights() -> None:
    control = [_fusion_row(0), _fusion_row(1)]
    candidate = [_fusion_row(0, weight=0.801), _fusion_row(1, weight=0.799)]

    gate = compare_fusion_rows(control, candidate, expected_frame_count=2)

    assert gate["passed"] is True
    assert gate["measurements"]["common_joint_weight_count"] == 2


def test_fusion_row_comparison_keeps_exact_schema_gate() -> None:
    control = [_fusion_row(0)]
    candidate = [{**_fusion_row(0), "inference_settings": {"parallel": True}}]

    gate = compare_fusion_rows(control, candidate, expected_frame_count=1)

    assert gate["passed"] is False
    assert gate["checks"]["schema_match"] is False
    assert gate["schema_compatibility"]["control_preserved"] is True


def _passing_feature_measurements() -> FeatureParityMeasurements:
    return FeatureParityMeasurements(
        control_frame_count=260,
        candidate_frame_count=260,
        npz_keys_match=True,
        npz_shapes_match=True,
        npz_dtypes_match=True,
        array_schema_match=True,
        comparable_array_count=1,
        array_max_abs_delta=0.001,
        valid_frame_loss_count=1,
        joint_angle_common_value_count=1,
        joint_angle_median_abs_error=0.5,
        joint_angle_p95_abs_error=2.0,
        runner_metric_keys_match=True,
        comparable_runner_metric_count=1,
        runner_metric_max_abs_delta=1.0,
        runner_metrics_within_tolerance=True,
    )


def test_feature_parity_passes_at_all_registered_boundaries() -> None:
    gate = evaluate_feature_parity(
        _passing_feature_measurements(),
        expected_frame_count=260,
    )

    assert gate["passed"] is True
    assert all(gate["checks"].values())


@pytest.mark.parametrize(
    ("field", "value", "check"),
    [
        ("control_frame_count", 259, "control_frame_count_exact"),
        ("candidate_frame_count", 259, "candidate_frame_count_exact"),
        ("npz_keys_match", False, "npz_keys_match"),
        ("npz_shapes_match", False, "npz_shapes_match"),
        ("npz_dtypes_match", False, "npz_dtypes_match"),
        ("array_schema_match", False, "array_schema_match"),
        ("comparable_array_count", 0, "comparable_array_evidence"),
        ("array_max_abs_delta", 0.001001, "array_max_abs_delta"),
        ("valid_frame_loss_count", 2, "valid_frame_loss"),
        ("joint_angle_common_value_count", 0, "joint_angle_evidence"),
        ("joint_angle_median_abs_error", 0.500001, "joint_angle_median_abs_error"),
        ("joint_angle_p95_abs_error", 2.000001, "joint_angle_p95_abs_error"),
        ("runner_metric_keys_match", False, "runner_metric_keys_match"),
        ("comparable_runner_metric_count", 0, "runner_metric_evidence"),
        ("runner_metrics_within_tolerance", False, "runner_metrics_within_tolerance"),
    ],
)
def test_each_feature_gate_rejects_its_own_violation(
    field: str,
    value: object,
    check: str,
) -> None:
    measurements = replace(_passing_feature_measurements(), **{field: value})

    gate = evaluate_feature_parity(measurements, expected_frame_count=260)

    assert gate["passed"] is False
    assert gate["checks"][check] is False


def test_feature_artifact_comparison_measures_npz_and_runner_metric_deltas(
    tmp_path: Path,
) -> None:
    control_npz = tmp_path / "control.npz"
    candidate_npz = tmp_path / "candidate.npz"
    np.savez_compressed(
        control_npz,
        pose_xy=np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        valid_frames=np.asarray([True, True]),
        joint_angles=np.asarray([[90.0], [100.0]], dtype=np.float32),
    )
    np.savez_compressed(
        candidate_npz,
        pose_xy=np.asarray([[0.1001, 0.2], [0.3, 0.4]], dtype=np.float32),
        valid_frames=np.asarray([True, True]),
        joint_angles=np.asarray([[90.1], [99.9]], dtype=np.float32),
    )
    control_metadata = {
        "frame_count": 2,
        "array_schema": {"joint_names": ["nose"]},
        "summary_features": {"stride_rhythm_proxy": 1.0, "knee_lift_proxy": 0.5},
    }
    candidate_metadata = {
        "frame_count": 2,
        "array_schema": {"joint_names": ["nose"]},
        "summary_features": {"stride_rhythm_proxy": 1.001, "knee_lift_proxy": 0.5},
    }

    gate = compare_feature_artifacts(
        control_metadata,
        candidate_metadata,
        control_npz,
        candidate_npz,
        expected_frame_count=2,
    )

    assert gate["passed"] is True
    assert gate["measurements"]["comparable_array_count"] == 2
    assert gate["measurements"]["comparable_runner_metric_count"] == 2


def _passing_qc_measurements() -> QcParityMeasurements:
    return QcParityMeasurements(
        schema_match=True,
        required_components_present=True,
        categorical_match=True,
        numeric_field_count=1,
        numeric_max_abs_delta=0.01,
        identity_exact=True,
        mask_churn_abs_delta=0.01,
        uncertainty_increase=0.01,
    )


def test_qc_parity_passes_at_all_registered_boundaries() -> None:
    gate = evaluate_qc_parity(_passing_qc_measurements())

    assert gate["passed"] is True
    assert all(gate["checks"].values())


@pytest.mark.parametrize(
    ("field", "value", "check"),
    [
        ("schema_match", False, "schema_match"),
        ("required_components_present", False, "required_components_present"),
        ("categorical_match", False, "categorical_match"),
        ("numeric_field_count", 0, "numeric_field_evidence"),
        ("numeric_max_abs_delta", 0.010001, "numeric_max_abs_delta"),
        ("identity_exact", False, "identity_exact"),
        ("mask_churn_abs_delta", 0.010001, "mask_churn_abs_delta"),
        ("uncertainty_increase", 0.010001, "uncertainty_increase"),
    ],
)
def test_each_qc_gate_rejects_its_own_violation(
    field: str,
    value: object,
    check: str,
) -> None:
    measurements = replace(_passing_qc_measurements(), **{field: value})

    gate = evaluate_qc_parity(measurements)

    assert gate["passed"] is False
    assert gate["checks"][check] is False


def test_qc_comparison_normalizes_timestamps_paths_and_candidate_ids() -> None:
    control = {
        "version": 1,
        "candidate_id": "control",
        "updated_at": "2026-01-01T00:00:00Z",
        "identity": {"frame_count": 260, "state": "stable"},
        "mask": {
            "frame_count": 260,
            "mean_mask_churn": 0.2,
            "output_path": "/control/masks.jsonl",
        },
        "pose": {"frame_count": 260, "pose_available": True},
        "fused": {"frame_count": 260, "fused_available": True},
        "uncertainty_score": 0.1,
    }
    candidate = {
        **control,
        "candidate_id": "candidate",
        "updated_at": "2026-01-02T00:00:00Z",
        "mask": {
            "frame_count": 260,
            "mean_mask_churn": 0.2,
            "output_path": "/candidate/masks.jsonl",
        },
    }

    gate = compare_qc_payloads(control, candidate)

    assert gate["passed"] is True


def test_qc_comparison_keeps_exact_schema_gate_for_unplanned_additions() -> None:
    control = {
        "identity": {"frame_count": 260},
        "mask": {"mean_mask_churn": 0.2},
        "pose": {"frame_count": 260},
        "fused": {"frame_count": 260},
        "uncertainty_score": 0.1,
    }
    candidate = {**control, "inference_settings": {"parallel": True}}

    gate = compare_qc_payloads(control, candidate)

    assert gate["passed"] is False
    assert gate["checks"]["schema_match"] is False


def _passing_artifact_measurements() -> ArtifactParityMeasurements:
    return ArtifactParityMeasurements(
        control_required_artifacts_present=True,
        candidate_required_artifacts_present=True,
        inventory_match=True,
        control_inventory_preserved=True,
        schema_artifact_count=1,
        json_schema_match=True,
        json_control_schema_preserved=True,
        parquet_schema_match=True,
        parquet_control_schema_preserved=True,
        parquet_row_counts_match=True,
    )


def test_artifact_parity_passes_when_inventory_and_schemas_match() -> None:
    gate = evaluate_artifact_parity(_passing_artifact_measurements())

    assert gate["passed"] is True
    assert all(gate["checks"].values())


@pytest.mark.parametrize(
    ("field", "value", "check"),
    [
        (
            "control_required_artifacts_present",
            False,
            "control_required_artifacts_present",
        ),
        (
            "candidate_required_artifacts_present",
            False,
            "candidate_required_artifacts_present",
        ),
        (
            "control_inventory_preserved",
            False,
            "control_inventory_preserved",
        ),
        ("schema_artifact_count", 0, "schema_artifact_evidence"),
        (
            "json_control_schema_preserved",
            False,
            "json_control_schema_preserved",
        ),
        (
            "parquet_control_schema_preserved",
            False,
            "parquet_control_schema_preserved",
        ),
        ("parquet_row_counts_match", False, "parquet_row_counts_match"),
    ],
)
def test_each_artifact_gate_rejects_its_own_violation(
    field: str,
    value: object,
    check: str,
) -> None:
    measurements = replace(_passing_artifact_measurements(), **{field: value})

    gate = evaluate_artifact_parity(measurements)

    assert gate["passed"] is False
    assert gate["checks"][check] is False


def test_artifact_contract_comparison_checks_jsonl_and_parquet_schemas(
    tmp_path: Path,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    control_dir = tmp_path / "control"
    candidate_dir = tmp_path / "candidate"
    control_dir.mkdir()
    candidate_dir.mkdir()
    for directory, value in ((control_dir, 1.0), (candidate_dir, 1.001)):
        (directory / "pose_landmarks.jsonl").write_text(
            json.dumps({"frame_index": 0, "usable": True, "visibility_mean": value}) + "\n",
            encoding="utf-8",
        )
        pq.write_table(
            pa.Table.from_pylist([{"frame_index": 0, "usable": True}]),
            directory / "poses.parquet",
        )

    gate = compare_artifact_contracts(
        control_dir,
        candidate_dir,
        required_artifacts={"pose_landmarks.jsonl", "poses.parquet"},
    )

    assert gate["passed"] is True
    assert gate["measurements"]["schema_artifact_count"] == 2
    assert gate["inventory"] == ["pose_landmarks.jsonl", "poses.parquet"]


def test_artifact_contract_allows_additive_candidate_files_fields_and_columns(
    tmp_path: Path,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    control_dir = tmp_path / "control"
    candidate_dir = tmp_path / "candidate"
    control_dir.mkdir()
    candidate_dir.mkdir()
    (control_dir / "densepose.jsonl").write_text(
        json.dumps({"frame_index": 0, "usable": True, "coverage": 0.4}) + "\n",
        encoding="utf-8",
    )
    (candidate_dir / "densepose.jsonl").write_text(
        json.dumps(
            {
                "frame_index": 0,
                "usable": True,
                "coverage": 0.4,
                "inference_input": {"width": 960, "height": 540},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    pq.write_table(
        pa.Table.from_pylist([{"frame_index": 0, "usable": True}]),
        control_dir / "densepose.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist([{"frame_index": 0, "usable": True, "inference_width": 960}]),
        candidate_dir / "densepose.parquet",
    )
    (candidate_dir / "pose_qa_overlay.mp4").write_bytes(b"candidate-only")

    gate = compare_artifact_contracts(
        control_dir,
        candidate_dir,
        required_artifacts={"densepose.jsonl", "densepose.parquet"},
    )

    assert gate["passed"] is True
    assert gate["measurements"] == {
        "control_required_artifacts_present": True,
        "candidate_required_artifacts_present": True,
        "inventory_match": False,
        "control_inventory_preserved": True,
        "schema_artifact_count": 2,
        "json_schema_match": False,
        "json_control_schema_preserved": True,
        "parquet_schema_match": False,
        "parquet_control_schema_preserved": True,
        "parquet_row_counts_match": True,
    }
    assert gate["inventory_only_in_candidate"] == ["pose_qa_overlay.mp4"]


def test_artifact_contract_rejects_removed_control_artifact_and_type_changes(
    tmp_path: Path,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    control_dir = tmp_path / "control"
    candidate_dir = tmp_path / "candidate"
    control_dir.mkdir()
    candidate_dir.mkdir()
    (control_dir / "densepose.jsonl").write_text(
        json.dumps({"frame_index": 0, "usable": True, "coverage": 0.4}) + "\n",
        encoding="utf-8",
    )
    (candidate_dir / "densepose.jsonl").write_text(
        json.dumps({"frame_index": 0, "usable": True, "coverage": "0.4"}) + "\n",
        encoding="utf-8",
    )
    pq.write_table(
        pa.Table.from_pylist([{"frame_index": 0, "usable": True}]),
        control_dir / "densepose.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist([{"frame_index": "0", "usable": True}]),
        candidate_dir / "densepose.parquet",
    )
    (control_dir / "legacy-control-evidence.txt").write_text("required\n", encoding="utf-8")

    gate = compare_artifact_contracts(
        control_dir,
        candidate_dir,
        required_artifacts={"densepose.jsonl", "densepose.parquet"},
    )

    assert gate["passed"] is False
    assert gate["checks"]["control_inventory_preserved"] is False
    assert gate["checks"]["json_control_schema_preserved"] is False
    assert gate["checks"]["parquet_control_schema_preserved"] is False
    assert gate["inventory_only_in_control"] == ["legacy-control-evidence.txt"]


def test_multi_artifact_compatibility_diagnostics_fit_full_response_cap(
    tmp_path: Path,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    control_dir = tmp_path / "control"
    candidate_dir = tmp_path / "candidate"
    control_dir.mkdir()
    candidate_dir.mkdir()
    required_artifacts: set[str] = set()
    candidate_additions = {
        f"candidate_only_metadata_field_{index:02d}": index for index in range(48)
    }
    for index in range(12):
        name = f"evidence_{index:02d}.json"
        required_artifacts.add(name)
        (control_dir / name).write_text(
            json.dumps({"frame_index": index, "usable": True}) + "\n",
            encoding="utf-8",
        )
        (candidate_dir / name).write_text(
            json.dumps(
                {
                    "frame_index": index,
                    "usable": True,
                    **candidate_additions,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    for index in range(6):
        name = f"evidence_{index:02d}.parquet"
        required_artifacts.add(name)
        pq.write_table(
            pa.Table.from_pylist([{"frame_index": index, "usable": True}]),
            control_dir / name,
        )
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "frame_index": index,
                        "usable": True,
                        **candidate_additions,
                    }
                ]
            ),
            candidate_dir / name,
        )

    gate = compare_artifact_contracts(
        control_dir,
        candidate_dir,
        required_artifacts=required_artifacts,
    )
    encoded = json.dumps(
        {"comparisons": {"artifacts": gate}},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    assert gate["passed"] is True
    assert len(encoded) < min(
        pipeline_parity.FULL_BENCHMARK_RESPONSE_MAX_BYTES,
        64 * 1024,
    )
    assert (
        max(
            len(details.get("candidate_only_paths", []))
            for details in gate["schema_compatibility"]["json"].values()
        )
        == pipeline_parity._SCHEMA_DIAGNOSTIC_MAX_PATHS
    )
    assert (
        max(
            len(details.get("candidate_only_fields", []))
            for details in gate["schema_compatibility"]["parquet"].values()
        )
        == pipeline_parity._SCHEMA_DIAGNOSTIC_MAX_PATHS
    )


def _passing_video_measurements() -> VideoParityMeasurements:
    return VideoParityMeasurements(
        control_required_videos_present=True,
        candidate_required_videos_present=True,
        decoded_video_count=8,
        all_videos_playable=True,
        no_blank_frames=True,
        dimensions_exact=True,
        frame_counts_exact=True,
        profile_metadata_match=True,
        fps_expected_match=True,
        fps_max_abs_delta=0.01,
    )


def test_video_parity_passes_at_all_registered_boundaries() -> None:
    gate = evaluate_video_parity(_passing_video_measurements(), expected_video_count=8)

    assert gate["passed"] is True
    assert all(gate["checks"].values())


@pytest.mark.parametrize(
    ("field", "value", "check"),
    [
        ("control_required_videos_present", False, "control_required_videos_present"),
        (
            "candidate_required_videos_present",
            False,
            "candidate_required_videos_present",
        ),
        ("decoded_video_count", 7, "decoded_video_count_exact"),
        ("all_videos_playable", False, "all_videos_playable"),
        ("no_blank_frames", False, "no_blank_frames"),
        ("dimensions_exact", False, "dimensions_exact"),
        ("frame_counts_exact", False, "frame_counts_exact"),
        ("profile_metadata_match", False, "profile_metadata_match"),
        ("fps_expected_match", False, "fps_expected_match"),
        ("fps_max_abs_delta", 0.010001, "fps_max_abs_delta"),
    ],
)
def test_each_video_gate_rejects_its_own_violation(
    field: str,
    value: object,
    check: str,
) -> None:
    measurements = replace(_passing_video_measurements(), **{field: value})

    gate = evaluate_video_parity(measurements, expected_video_count=8)

    assert gate["passed"] is False
    assert gate["checks"][check] is False


def _write_test_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (64, 48),
    )
    assert writer.isOpened()
    for index in range(3):
        frame = np.full((48, 64, 3), 30 + index * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _write_test_mask_video(
    path: Path,
    *,
    rectangle_width: int = 20,
    frame_size: tuple[int, int] = (64, 48),
) -> None:
    width, height = frame_size
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (width, height),
    )
    assert writer.isOpened()
    for index in range(3):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        x1 = 10 + index
        cv2.rectangle(frame, (x1, 10), (x1 + rectangle_width, 35), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


def test_runner_mask_comparison_decodes_and_gates_framewise_binary_agreement(
    tmp_path: Path,
) -> None:
    control = tmp_path / "control.mp4"
    candidate = tmp_path / "candidate.mp4"
    _write_test_mask_video(control)
    shutil.copy2(control, candidate)

    gate = compare_runner_mask_videos(
        control,
        candidate,
        expected_width=64,
        expected_height=48,
        expected_frame_count=3,
    )

    assert gate["passed"] is True
    assert gate["measurements"]["iou_mean"] == 1.0
    assert gate["measurements"]["iou_p05"] == 1.0
    assert gate["measurements"]["coverage_mae"] == 0.0
    assert gate["measurements"]["mask_churn_abs_delta"] == 0.0
    assert gate["thresholds"]["iou_mean_min"] == 0.985
    assert gate["thresholds"]["iou_p05_min"] == 0.975


def test_runner_mask_comparison_rejects_coverage_and_geometry_drift(
    tmp_path: Path,
) -> None:
    control = tmp_path / "control.mp4"
    candidate = tmp_path / "candidate.mp4"
    _write_test_mask_video(control, rectangle_width=20)
    _write_test_mask_video(candidate, rectangle_width=32)

    gate = compare_runner_mask_videos(
        control,
        candidate,
        expected_width=64,
        expected_height=48,
        expected_frame_count=3,
    )

    assert gate["passed"] is False
    assert gate["checks"]["iou_mean"] is False
    assert gate["checks"]["iou_p05"] is False
    assert gate["checks"]["coverage_mae"] is False


def test_video_contract_comparison_decodes_every_frame_in_both_profiles(
    tmp_path: Path,
) -> None:
    control_dir = tmp_path / "control"
    candidate_dir = tmp_path / "candidate"
    control_dir.mkdir()
    candidate_dir.mkdir()
    _write_test_video(control_dir / "runner_mask.mp4")
    _write_test_video(candidate_dir / "runner_mask.mp4")

    gate = compare_video_contracts(
        control_dir,
        candidate_dir,
        required_videos={"runner_mask.mp4"},
        expected_width=64,
        expected_height=48,
        expected_frame_count=3,
        expected_fps=10.0,
    )

    assert gate["passed"] is True
    assert gate["measurements"]["decoded_video_count"] == 2
    assert gate["videos"]["runner_mask.mp4"]["control"]["decoded_frames"] == 3


def test_pipeline_comparison_reports_missing_evidence_without_weakening_gate(
    tmp_path: Path,
) -> None:
    control_dir = tmp_path / "control"
    candidate_dir = tmp_path / "candidate"
    control_dir.mkdir()
    candidate_dir.mkdir()

    result = compare_pipeline_runs(
        control_dir,
        candidate_dir,
        expected_width=960,
        expected_height=540,
        expected_frame_count=260,
        expected_fps=29.97,
    )

    assert result["passed"] is False
    assert set(result["comparisons"]) == {
        "runner_mask",
        "pose",
        "densepose",
        "fusion",
        "features",
        "qc",
        "artifacts",
        "videos",
    }
    assert result["comparisons"]["pose"]["availability"] == "unavailable"
    assert result["comparisons"]["pose"]["passed"] is False


def test_full_profile_matrix_defaults_to_sam_mask_control_comparison() -> None:
    profiles = resolve_pipeline_profiles(None)

    assert [profile.profile_id for profile in profiles] == [
        "downstream_baseline_control",
        "downstream_candidate_control",
    ]
    assert profiles[0].execution_mode == "fixed_mask_downstream"
    assert profiles[0].mask_source == "baseline"
    assert profiles[0].parallel_pose_densepose is False
    assert profiles[1].mask_source == "candidate"
    assert profiles[1].parallel_pose_densepose is False
    assert profiles[1].parallel_pose_densepose is False
    assert profiles[1].parallel_post_fusion is False


def test_full_profile_registry_includes_direct_production_pipeline_timing_modes() -> None:
    assert set(PIPELINE_BENCHMARK_PROFILES) == {
        "downstream_baseline_control",
        "downstream_candidate_control",
        "production_control",
        "production_candidate_schedule_only",
        "production_final_candidate",
    }
    assert dict(
        PIPELINE_BENCHMARK_PROFILES["production_candidate_schedule_only"].environment_overrides
    ) == {
        "WHODOIRUNLIKE_PARALLEL_MASK_PRESENTATION": "true",
        "WHODOIRUNLIKE_PARALLEL_POSE_DENSEPOSE": "true",
        "WHODOIRUNLIKE_PARALLEL_POST_FUSION": "true",
        "WHODOIRUNLIKE_PARALLEL_ARTIFACT_PUBLISH": "true",
        "MMPOSE_DEVICE": "cpu",
        "RTMW_RUNTIME_BACKEND": "onnxruntime",
    }
    assert (
        PIPELINE_BENCHMARK_PROFILES["production_final_candidate"].environment_overrides
        == PIPELINE_BENCHMARK_PROFILES[
            "production_candidate_schedule_only"
        ].environment_overrides
    )


def test_full_profile_matrix_rejects_more_than_three_executions() -> None:
    with pytest.raises(ValueError, match="at most three"):
        resolve_pipeline_profiles(
            [
                "downstream_baseline_control",
                "downstream_candidate_control",
                "production_control",
                "production_candidate_schedule_only",
            ]
        )


def test_pipeline_fixture_materializes_exact_inputs_as_a_running_clip_run(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    mask = tmp_path / "mask.mp4"
    _write_test_video(source)
    _write_test_video(mask)
    run_dir = tmp_path / "run"

    manifest_path = materialize_pipeline_fixture(
        run_dir=run_dir,
        source_path=source,
        assets={
            "person_prompt_json": json.dumps(
                {"frame": {"frame_index": 0}, "selection": {"box": None}}
            ).encode(),
            "tracklets_jsonl": b'{"frame_index":0,"is_target":true}\n',
            "baseline_runner_mask_mp4": mask.read_bytes(),
        },
        profile_id="downstream_baseline_control",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert (run_dir / "source_segment.mp4").read_bytes() == source.read_bytes()
    assert (run_dir / "runner_mask.mp4").read_bytes() == mask.read_bytes()
    assert (run_dir / "masks.jsonl").is_file()
    assert manifest["candidate_id"] == "benchmark-downstream-baseline-control"
    assert manifest["paths"]["pose_landmarks"].endswith("/pose_landmarks.jsonl")


def test_safe_parallel_profile_overlaps_analysis_and_post_fusion_stages(
    tmp_path: Path,
) -> None:
    analysis_barrier = threading.Barrier(2)
    post_barrier = threading.Barrier(3)
    calls: list[str] = []
    call_lock = threading.Lock()

    def stage(name: str, barrier: threading.Barrier | None = None):
        def run(_run_dir: Path) -> dict[str, object]:
            with call_lock:
                calls.append(name)
            if barrier is not None:
                barrier.wait(timeout=2)
            return {"status": "complete", "frame_count": 260}

        return run

    functions = PipelineStageFunctions(
        pose=stage("pose", analysis_barrier),
        densepose=stage("densepose", analysis_barrier),
        fusion=stage("fusion"),
        features=stage("features", post_barrier),
        tables=stage("tables", post_barrier),
        qc=stage("qc", post_barrier),
        full_pipeline=stage("full_pipeline"),
    )

    result = run_pipeline_profile(
        tmp_path,
        replace(
            PIPELINE_BENCHMARK_PROFILES["downstream_candidate_control"],
            profile_id="safe-parallel-probe",
            parallel_pose_densepose=True,
            parallel_post_fusion=True,
        ),
        stage_functions=functions,
    )

    assert result["status"] == "complete"
    assert set(result["timings_seconds"]) >= {
        "pose",
        "densepose",
        "analysis_wall",
        "fusion",
        "features",
        "tables",
        "qc",
        "post_fusion_wall",
        "total",
    }
    assert calls.index("fusion") > max(calls.index("pose"), calls.index("densepose"))


def test_production_profile_reports_stage_timings_without_paths(tmp_path: Path) -> None:
    def unused(_run_dir: Path) -> dict[str, object]:
        raise AssertionError("fixed-mask stage should not run")

    def full(_run_dir: Path) -> dict[str, object]:
        return {
            "steps": [
                {
                    "stage": "mask",
                    "result": {
                        "status": "complete",
                        "elapsed_seconds": 12.5,
                        "frame_count": 260,
                        "exact_cv2_loader_enabled": True,
                        "exact_cv2_loader_attempted": True,
                        "exact_cv2_loader_used": True,
                        "exact_cv2_loader_seconds": 1.25,
                        "exact_cv2_loader_required_concurrency": 1,
                        "exact_cv2_loader_configured_concurrency": 1,
                        "exact_cv2_loader_concurrency_ready": True,
                        "runner_mask": "/private/run/runner_mask.mp4",
                    },
                }
            ]
        }

    result = run_pipeline_profile(
        tmp_path,
        PIPELINE_BENCHMARK_PROFILES["production_final_candidate"],
        stage_functions=PipelineStageFunctions(
            pose=unused,
            densepose=unused,
            fusion=unused,
            features=unused,
            tables=unused,
            qc=unused,
            full_pipeline=full,
        ),
    )

    assert result["timings_seconds"]["pipeline_mask"] == 12.5
    assert result["stages"]["production_full_pipeline"]["steps"] == [
        {
            "stage": "mask",
            "status": "complete",
            "elapsed_seconds": 12.5,
            "frame_count": 260,
            "exact_cv2_loader_enabled": True,
            "exact_cv2_loader_attempted": True,
            "exact_cv2_loader_used": True,
            "exact_cv2_loader_seconds": 1.25,
            "exact_cv2_loader_required_concurrency": 1,
            "exact_cv2_loader_configured_concurrency": 1,
            "exact_cv2_loader_concurrency_ready": True,
        }
    ]
    assert "/private" not in json.dumps(result)


def test_exact_loader_gate_requires_attempt_use_positive_timing_and_safe_concurrency() -> None:
    passing = {
        "stages": {
            "production_full_pipeline": {
                "steps": [
                    {
                        "stage": "mask",
                        "exact_cv2_loader_enabled": True,
                        "exact_cv2_loader_attempted": True,
                        "exact_cv2_loader_used": True,
                        "exact_cv2_loader_seconds": 1.25,
                        "exact_cv2_loader_required_concurrency": 1,
                        "exact_cv2_loader_configured_concurrency": 1,
                        "exact_cv2_loader_concurrency_ready": True,
                    }
                ]
            }
        }
    }

    gate = evaluate_exact_loader_gate(passing)

    assert gate["passed"] is True
    assert all(gate["checks"].values())

    passing["stages"]["production_full_pipeline"]["steps"][0][
        "exact_cv2_loader_seconds"
    ] = 0.0
    failed = evaluate_exact_loader_gate(passing)
    assert failed["passed"] is False
    assert failed["checks"]["nonzero_timing"] is False


def test_production_profiles_apply_only_safe_schedule_configs_and_restore_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from whodoirunlike import full_pipeline

    calls: list[dict[str, object]] = []

    def fake_full_pipeline(
        *,
        run_dir: Path,
        pose_backend: str,
        mask_backend: str,
        skip_densepose: bool,
        parallel_mask_presentation: bool,
        parallel_pose_densepose: bool,
        parallel_post_fusion: bool,
    ) -> dict[str, object]:
        calls.append(
            {
                "run_dir": run_dir,
                "pose_backend": pose_backend,
                "mask_backend": mask_backend,
                "skip_densepose": skip_densepose,
                "parallel_mask_presentation": parallel_mask_presentation,
                "parallel_pose_densepose": parallel_pose_densepose,
                "parallel_post_fusion": parallel_post_fusion,
                "device": os.getenv("MMPOSE_DEVICE"),
                "runtime_backend": os.getenv("RTMW_RUNTIME_BACKEND"),
                "use_detector": os.getenv("MMPOSE_USE_DETECTOR"),
                "densepose_min": os.getenv("DENSEPOSE_INPUT_MIN_SIZE_TEST"),
                "densepose_max": os.getenv("DENSEPOSE_INPUT_MAX_SIZE_TEST"),
                "densepose_crop": os.getenv("DENSEPOSE_TARGET_CROP_ENABLED"),
                "parallel_publish": os.getenv("WHODOIRUNLIKE_PARALLEL_ARTIFACT_PUBLISH"),
            }
        )
        return {"status": "complete", "steps": []}

    monkeypatch.setattr(full_pipeline, "run_full_cv_pipeline", fake_full_pipeline)
    monkeypatch.setenv("MMPOSE_USE_DETECTOR", "outside")
    (tmp_path / "control").mkdir()
    (tmp_path / "schedule-only").mkdir()
    (tmp_path / "final").mkdir()

    run_pipeline_profile(
        tmp_path / "control",
        PIPELINE_BENCHMARK_PROFILES["production_control"],
    )
    run_pipeline_profile(
        tmp_path / "schedule-only",
        PIPELINE_BENCHMARK_PROFILES["production_candidate_schedule_only"],
    )
    run_pipeline_profile(
        tmp_path / "final",
        PIPELINE_BENCHMARK_PROFILES["production_final_candidate"],
    )

    assert calls[0] | {"run_dir": None} == {
        "run_dir": None,
        "pose_backend": "mmpose_rtmpose_l_384",
        "mask_backend": "sam31_gpu",
        "skip_densepose": False,
        "parallel_mask_presentation": False,
        "parallel_pose_densepose": False,
        "parallel_post_fusion": False,
        "device": "cpu",
        "runtime_backend": "onnxruntime",
        "use_detector": "outside",
        "densepose_min": None,
        "densepose_max": None,
        "densepose_crop": None,
        "parallel_publish": "false",
    }
    assert calls[1] | {"run_dir": None} == {
        "run_dir": None,
        "pose_backend": "mmpose_rtmpose_l_384",
        "mask_backend": "sam31_gpu",
        "skip_densepose": False,
        "parallel_mask_presentation": True,
        "parallel_pose_densepose": True,
        "parallel_post_fusion": True,
        "device": "cpu",
        "runtime_backend": "onnxruntime",
        "use_detector": "outside",
        "densepose_min": None,
        "densepose_max": None,
        "densepose_crop": None,
        "parallel_publish": "true",
    }
    assert calls[2] | {"run_dir": None} == {
        "run_dir": None,
        "pose_backend": "mmpose_rtmpose_l_384",
        "mask_backend": "sam31_gpu",
        "skip_densepose": False,
        "parallel_mask_presentation": True,
        "parallel_pose_densepose": True,
        "parallel_post_fusion": True,
        "device": "cpu",
        "runtime_backend": "onnxruntime",
        "use_detector": "outside",
        "densepose_min": None,
        "densepose_max": None,
        "densepose_crop": None,
        "parallel_publish": "true",
    }
    assert os.environ["MMPOSE_USE_DETECTOR"] == "outside"


def test_artifact_sink_validation_rejects_non_scratch_origins_and_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch_origin = "https://parity-scratch.example.com"
    valid = {
        "callback_base_url": scratch_origin,
        "run_id": "11c51cf1-c4d0-42ef-a2e1-cb9e2605ef1b",
        "attempt_id": "5ec9566e-cda9-4113-9100-9a4b2a248f6f",
    }

    monkeypatch.delenv("WHODOIRUNLIKE_PARITY_SINK_ORIGIN", raising=False)
    with pytest.raises(RuntimeError, match="must name the exact scratch sink"):
        pipeline_parity._validate_artifact_sink(valid)

    monkeypatch.setenv("WHODOIRUNLIKE_PARITY_SINK_ORIGIN", scratch_origin)
    assert pipeline_parity._validate_artifact_sink(valid) == valid
    with pytest.raises(ValueError, match="production or staging"):
        pipeline_parity._validate_artifact_sink(
            {**valid, "callback_base_url": "https://api.whodoirunlike.com"}
        )
    with pytest.raises(ValueError, match="production or staging"):
        pipeline_parity._validate_artifact_sink(
            {**valid, "callback_base_url": "https://staging-api.whodoirunlike.com"}
        )
    with pytest.raises(ValueError, match="exact HTTPS origin"):
        pipeline_parity._validate_artifact_sink(
            {
                **valid,
                "callback_base_url": f"{scratch_origin}/private",
            }
        )
    with pytest.raises(ValueError, match="does not match"):
        pipeline_parity._validate_artifact_sink(
            {**valid, "callback_base_url": "https://other-scratch.example.com"}
        )

    monkeypatch.setenv(
        "WHODOIRUNLIKE_PARITY_SINK_ORIGIN",
        "https://api.whodoirunlike.com",
    )
    with pytest.raises(RuntimeError, match="production or staging"):
        pipeline_parity._validate_artifact_sink(valid)


def test_r2_handoff_bundle_round_trips_with_provenance_and_file_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "control-run"
    (run_dir / "nested").mkdir(parents=True)
    (run_dir / "cv_run_manifest.json").write_text("{}\n", encoding="utf-8")
    source_bytes = b"canonical-source"
    prompt_bytes = b'{"canonical":"prompt"}\n'
    (run_dir / "source_segment.mp4").write_bytes(source_bytes)
    (run_dir / "person_prompt.json").write_bytes(prompt_bytes)
    (run_dir / "nested" / "pose.jsonl").write_text('{"frame_index":0}\n', encoding="utf-8")
    base_fixture = get_parity_fixture(CANONICAL_FRAME130_FIXTURE_ID)
    fixture = replace(
        base_fixture,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        prompt=replace(
            base_fixture.prompt,
            raw_sha256=hashlib.sha256(prompt_bytes).hexdigest(),
        ),
    )
    runtime = {
        "base_processor_commit": pipeline_parity.EXACT_CONTROL_COMMIT,
        "base_processor_image_digest": pipeline_parity.EXACT_CONTROL_IMAGE_DIGEST,
        "gpu_name": "NVIDIA A40",
        "base_contract": {
            "passed": True,
            "image_role": "control",
            "checked_file_count": 12,
            "mismatches": [],
        },
    }
    bundle, manifest_path, manifest = pipeline_parity._create_handoff_bundle(
        workspace=tmp_path / "workspace",
        run_dir=run_dir,
        profile_id="production_control",
        image_role="control",
        runtime=runtime,
        fixture=fixture,
    )
    second_bundle, _, _ = pipeline_parity._create_handoff_bundle(
        workspace=tmp_path / "second-workspace",
        run_dir=run_dir,
        profile_id="production_control",
        image_role="control",
        runtime=runtime,
        fixture=fixture,
    )
    sources = {bundle.name: bundle, manifest_path.name: manifest_path}

    def fake_download(
        _sink: dict[str, str],
        *,
        name: str,
        destination: Path,
        max_bytes: int,
    ) -> None:
        assert sources[name].stat().st_size <= max_bytes
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sources[name], destination)

    monkeypatch.setattr(pipeline_parity, "_download_sink_artifact", fake_download)
    profile_dir, provenance = pipeline_parity._load_control_handoff(
        sink={
            "callback_base_url": "https://parity-scratch.example.com",
            "run_id": "11c51cf1-c4d0-42ef-a2e1-cb9e2605ef1b",
            "attempt_id": "5ec9566e-cda9-4113-9100-9a4b2a248f6f",
        },
        workspace=tmp_path / "candidate-workspace",
        runtime={"gpu_name": "NVIDIA A40"},
        fixture=fixture,
    )

    assert manifest["bundle"]["sha256"] == hashlib.sha256(bundle.read_bytes()).hexdigest()
    assert second_bundle.read_bytes() == bundle.read_bytes()
    assert (profile_dir / "cv_run_manifest.json").read_text(encoding="utf-8") == "{}\n"
    assert (profile_dir / "nested" / "pose.jsonl").is_file()
    assert provenance["provenance_checks"] == {
        "role": True,
        "commit": True,
        "digest": True,
        "fixture": True,
        "profile": True,
        "base_contract": True,
        "gpu": True,
    }


def test_safe_handoff_extraction_rejects_path_traversal(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "unsafe.tar.gz"
    payload = b"escape"
    with tarfile.open(bundle, "w:gz") as archive:
        info = tarfile.TarInfo("../escape.txt")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    with pytest.raises(ValueError, match="unsafe"):
        pipeline_parity._safe_extract_handoff_bundle(
            bundle_path=bundle,
            destination=tmp_path / "extract",
            manifest={
                "profile_id": "production_control",
                "files": [
                    {
                        "path": "../escape.txt",
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ],
            },
        )


def test_full_request_validation_decodes_canonical_assets_and_profile_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_fixture = get_parity_fixture(CANONICAL_FRAME130_FIXTURE_ID)
    prompt = base_fixture.prompt.canonical_bytes()
    tracklets = b'{"frame_index":0}\n'
    mask = b"mask-video"
    fixture = replace(
        base_fixture,
        asset_sha256=MappingProxyType(
            {
                "person_prompt_json": hashlib.sha256(prompt).hexdigest(),
                "tracklets_jsonl": hashlib.sha256(tracklets).hexdigest(),
                "baseline_runner_mask_mp4": hashlib.sha256(mask).hexdigest(),
            }
        ),
    )
    monkeypatch.setitem(PARITY_FIXTURES, CANONICAL_FRAME130_FIXTURE_ID, fixture)
    payload = {
        "type": "sam31_benchmark",
        "schema_version": 1,
        "scope": "full",
        "fixture_id": CANONICAL_FRAME130_FIXTURE_ID,
        "profile_ids": [
            "downstream_baseline_control",
            "downstream_candidate_control",
        ],
        "assets": {
            "person_prompt_json": {
                "encoding": "base64",
                "sha256": hashlib.sha256(prompt).hexdigest(),
                "data": base64.b64encode(prompt).decode(),
            },
            "tracklets_jsonl": {
                "encoding": "gzip+base64",
                "sha256": hashlib.sha256(tracklets).hexdigest(),
                "data": base64.b64encode(gzip.compress(tracklets)).decode(),
            },
            "baseline_runner_mask_mp4": {
                "encoding": "base64",
                "sha256": hashlib.sha256(mask).hexdigest(),
                "data": base64.b64encode(mask).decode(),
            },
        },
    }

    profiles, decoded = validate_full_benchmark_request(payload)

    assert [profile.profile_id for profile in profiles] == [
        "downstream_baseline_control",
        "downstream_candidate_control",
    ]
    assert decoded == {
        "person_prompt_json": prompt,
        "tracklets_jsonl": tracklets,
        "baseline_runner_mask_mp4": mask,
    }


def test_full_benchmark_response_is_bounded_and_contains_no_artifact_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mp4"
    mask_path = tmp_path / "mask.mp4"
    _write_test_video(source)
    _write_test_video(mask_path)
    base_fixture = get_parity_fixture(CANONICAL_FRAME130_FIXTURE_ID)
    prompt = base_fixture.prompt.canonical_bytes()
    tracklets = b'{"frame_index":0,"is_target":true}\n'
    mask = mask_path.read_bytes()
    fixture = replace(
        base_fixture,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        frame_count=3,
        width=64,
        height=48,
        asset_sha256=MappingProxyType(
            {
                "person_prompt_json": hashlib.sha256(prompt).hexdigest(),
                "tracklets_jsonl": hashlib.sha256(tracklets).hexdigest(),
                "baseline_runner_mask_mp4": hashlib.sha256(mask).hexdigest(),
            }
        ),
    )
    monkeypatch.setitem(PARITY_FIXTURES, CANONICAL_FRAME130_FIXTURE_ID, fixture)
    payload = {
        "type": "sam31_benchmark",
        "schema_version": 1,
        "scope": "full",
        "fixture_id": CANONICAL_FRAME130_FIXTURE_ID,
        "profile_ids": ["production_control", "production_candidate_schedule_only"],
        "assets": {
            "person_prompt_json": {
                "encoding": "base64",
                "sha256": fixture.asset_sha256["person_prompt_json"],
                "data": base64.b64encode(prompt).decode(),
            },
            "tracklets_jsonl": {
                "encoding": "gzip+base64",
                "sha256": fixture.asset_sha256["tracklets_jsonl"],
                "data": base64.b64encode(gzip.compress(tracklets)).decode(),
            },
            "baseline_runner_mask_mp4": {
                "encoding": "base64",
                "sha256": fixture.asset_sha256["baseline_runner_mask_mp4"],
                "data": base64.b64encode(mask).decode(),
            },
        },
    }

    result = run_full_pipeline_benchmark(
        payload,
        source_path=source,
        profile_runner=lambda run_dir, profile: {
            "profile_id": profile.profile_id,
            "execution_mode": profile.execution_mode,
            "status": "complete",
            "config": {},
            "timings_seconds": {"total": 0.01},
            "stages": {},
            "artifact_inventory": sorted(path.name for path in run_dir.iterdir()),
        },
    )

    encoded = json.dumps(result, allow_nan=False).encode()
    assert result["scope"] == "full"
    assert len(encoded) <= 256 * 1024
    assert result["response_bytes"] <= 256 * 1024
    assert payload["assets"]["baseline_runner_mask_mp4"]["data"] not in encoded.decode()


def _small_full_benchmark_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    profile_ids: list[str],
) -> tuple[Path, dict[str, object]]:
    source = tmp_path / "small-source.mp4"
    mask_path = tmp_path / "small-mask.mp4"
    _write_test_video(source)
    _write_test_video(mask_path)
    base_fixture = get_parity_fixture(CANONICAL_FRAME130_FIXTURE_ID)
    prompt = base_fixture.prompt.canonical_bytes()
    tracklets = b'{"frame_index":0,"is_target":true}\n'
    mask = mask_path.read_bytes()
    fixture = replace(
        base_fixture,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        frame_count=3,
        width=64,
        height=48,
        asset_sha256=MappingProxyType(
            {
                "person_prompt_json": hashlib.sha256(prompt).hexdigest(),
                "tracklets_jsonl": hashlib.sha256(tracklets).hexdigest(),
                "baseline_runner_mask_mp4": hashlib.sha256(mask).hexdigest(),
            }
        ),
    )
    monkeypatch.setitem(PARITY_FIXTURES, CANONICAL_FRAME130_FIXTURE_ID, fixture)
    return source, {
        "type": "sam31_benchmark",
        "schema_version": 1,
        "scope": "full",
        "fixture_id": CANONICAL_FRAME130_FIXTURE_ID,
        "profile_ids": profile_ids,
        "assets": {
            "person_prompt_json": {
                "encoding": "base64",
                "sha256": fixture.asset_sha256["person_prompt_json"],
                "data": base64.b64encode(prompt).decode(),
            },
            "tracklets_jsonl": {
                "encoding": "gzip+base64",
                "sha256": fixture.asset_sha256["tracklets_jsonl"],
                "data": base64.b64encode(gzip.compress(tracklets)).decode(),
            },
            "baseline_runner_mask_mp4": {
                "encoding": "base64",
                "sha256": fixture.asset_sha256["baseline_runner_mask_mp4"],
                "data": base64.b64encode(mask).decode(),
            },
        },
    }


def test_default_sam_mask_control_runner_generates_candidate_mask_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, payload = _small_full_benchmark_payload(
        tmp_path,
        monkeypatch,
        profile_ids=list(pipeline_parity.DEFAULT_PIPELINE_PROFILE_MATRIX),
    )
    mask_calls: list[Path] = []
    observed_masks: dict[str, str] = {}

    def candidate_mask(run_dir: Path) -> dict[str, object]:
        mask_calls.append(run_dir)
        return {
            "status": "complete",
            "quality_vs_production_baseline": {"strict_mask_agreement_gate": {"passed": True}},
        }

    def profile_runner(
        run_dir: Path,
        profile: pipeline_parity.PipelineBenchmarkProfile,
    ) -> dict[str, object]:
        observed_masks[profile.profile_id] = hashlib.sha256(
            (run_dir / "runner_mask.mp4").read_bytes()
        ).hexdigest()
        return {
            "profile_id": profile.profile_id,
            "execution_mode": profile.execution_mode,
            "status": "complete",
            "config": {},
            "timings_seconds": {"total": 0.01},
            "stages": {},
            "artifact_inventory": [],
        }

    monkeypatch.setattr(
        pipeline_parity,
        "compare_pipeline_runs",
        lambda *_args, **_kwargs: {"passed": True, "comparisons": {}},
    )
    result = run_full_pipeline_benchmark(
        payload,
        source_path=source,
        profile_runner=profile_runner,
        candidate_mask_runner=candidate_mask,
    )

    assert len(mask_calls) == 1
    assert set(observed_masks) == {
        "downstream_baseline_control",
        "downstream_candidate_control",
    }
    assert set(result["comparisons"]) == {"profile_comparison"}
    assert result["parity_passed"] is True


def test_full_benchmark_persists_complete_scratch_artifacts_when_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, payload = _small_full_benchmark_payload(
        tmp_path,
        monkeypatch,
        profile_ids=["production_control"],
    )
    output_root = tmp_path / "persisted"
    monkeypatch.setenv("WHODOIRUNLIKE_SAM31_BENCHMARK_OUTPUT_ROOT", str(output_root))

    result = run_full_pipeline_benchmark(
        payload,
        source_path=source,
        profile_runner=lambda run_dir, profile: {
            "profile_id": profile.profile_id,
            "execution_mode": profile.execution_mode,
            "status": "complete",
            "config": {},
            "timings_seconds": {"total": 0.01},
            "stages": {},
            "artifact_inventory": sorted(path.name for path in run_dir.iterdir()),
        },
    )

    bundle_dir = output_root / result["artifact_bundle"]["id"]
    assert result["artifact_bundle"]["persisted"] is True
    assert (bundle_dir / "production_control" / "cv_run_manifest.json").is_file()
    assert (bundle_dir / "benchmark_result.json").is_file()


@pytest.mark.parametrize(
    (
        "profile_id",
        "image_role",
        "base_commit",
        "base_digest",
        "code_commit",
        "code_source",
        "code_reference_digest",
        "dependency_role",
        "dependency_commit",
        "dependency_digest",
    ),
    [
        (
            "production_candidate_schedule_only",
            "schedule_only",
            pipeline_parity.EXACT_CONTROL_COMMIT,
            pipeline_parity.EXACT_CONTROL_IMAGE_DIGEST,
            pipeline_parity.EXACT_CANDIDATE_COMMIT,
            "git_commit",
            pipeline_parity.EXACT_CANDIDATE_IMAGE_DIGEST,
            "control",
            pipeline_parity.EXACT_CONTROL_COMMIT,
            pipeline_parity.EXACT_CONTROL_IMAGE_DIGEST,
        ),
        (
            "production_final_candidate",
            "final_candidate",
            "a" * 40,
            "sha256:" + "b" * 64,
            "a" * 40,
            "base_image",
            "sha256:" + "b" * 64,
            "final_candidate",
            "a" * 40,
            "sha256:" + "b" * 64,
        ),
    ],
)
def test_candidate_handoff_adds_authoritative_cross_image_gate_without_blobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile_id: str,
    image_role: str,
    base_commit: str,
    base_digest: str,
    code_commit: str,
    code_source: str,
    code_reference_digest: str,
    dependency_role: str,
    dependency_commit: str,
    dependency_digest: str,
) -> None:
    source, payload = _small_full_benchmark_payload(
        tmp_path,
        monkeypatch,
        profile_ids=[profile_id],
    )
    payload["artifact_sink"] = {
        "callback_base_url": "https://parity-scratch.example.com",
        "run_id": "11c51cf1-c4d0-42ef-a2e1-cb9e2605ef1b",
        "attempt_id": "5ec9566e-cda9-4113-9100-9a4b2a248f6f",
    }
    monkeypatch.setenv(
        "WHODOIRUNLIKE_PARITY_SINK_ORIGIN",
        "https://parity-scratch.example.com",
    )
    monkeypatch.setenv("WHODOIRUNLIKE_BENCHMARK_IMAGE_ROLE", image_role)
    monkeypatch.setenv(
        "WHODOIRUNLIKE_BASE_PROCESSOR_COMMIT",
        base_commit,
    )
    monkeypatch.setenv(
        "WHODOIRUNLIKE_BASE_PROCESSOR_IMAGE_DIGEST",
        base_digest,
    )
    monkeypatch.setenv("WHODOIRUNLIKE_CODE_OVERLAY_COMMIT", code_commit)
    monkeypatch.setenv("WHODOIRUNLIKE_CODE_OVERLAY_SOURCE", code_source)
    monkeypatch.setenv(
        "WHODOIRUNLIKE_CODE_OVERLAY_REFERENCE_IMAGE_DIGEST",
        code_reference_digest,
    )
    monkeypatch.setenv("WHODOIRUNLIKE_DEPENDENCY_BASE_ROLE", dependency_role)
    monkeypatch.setenv("WHODOIRUNLIKE_DEPENDENCY_BASE_COMMIT", dependency_commit)
    monkeypatch.setenv(
        "WHODOIRUNLIKE_DEPENDENCY_BASE_IMAGE_DIGEST",
        dependency_digest,
    )
    if image_role == "final_candidate":
        monkeypatch.setenv("WHODOIRUNLIKE_CANDIDATE_COMMIT", code_commit)
        monkeypatch.setenv("WHODOIRUNLIKE_CANDIDATE_IMAGE_DIGEST", base_digest)
        monkeypatch.setenv("WHODOIRUNLIKE_PROCESSOR_VERSION", code_commit)
        monkeypatch.setenv(
            "WHODOIRUNLIKE_FINAL_CANDIDATE_IMAGE_REFERENCE",
            "ghcr.io/akhil-ghosh/whodoirunlike-runpod-processor@" + base_digest,
        )
        monkeypatch.setenv("WHODOIRUNLIKE_ENFORCE_BASE_CONTRACT", "true")
        monkeypatch.setenv("WHODOIRUNLIKE_MASK_BACKEND", "sam31_gpu")
        monkeypatch.setenv("MMPOSE_DEVICE", "cpu")
        monkeypatch.setenv("RTMW_RUNTIME_BACKEND", "onnxruntime")
        monkeypatch.setenv(
            "DENSEPOSE_WEIGHTS",
            str(pipeline_parity.DENSEPOSE_WEIGHTS_PATH),
        )
        for name in pipeline_parity.FINAL_FORBIDDEN_QUALITY_ENVIRONMENT:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(
            pipeline_parity,
            "verify_densepose_weights",
            lambda: {"passed": True},
        )
        monkeypatch.setattr(
            pipeline_parity,
            "verify_non_overlay_production_files",
            lambda role: {
                "passed": role == "final_candidate",
                "image_role": role,
                "checked_file_count": len(FINAL_CANDIDATE_PRODUCTION_FILES),
                "mismatches": [],
            },
        )
    control_dir = tmp_path / "downloaded-control"
    control_dir.mkdir()
    monkeypatch.setattr(
        pipeline_parity,
        "_load_control_handoff",
        lambda **_kwargs: (
            control_dir,
            {"status": "verified", "provenance_checks": {"gpu": True}},
        ),
    )
    monkeypatch.setattr(
        pipeline_parity,
        "compare_pipeline_runs",
        lambda *_args, **_kwargs: {"passed": True, "comparisons": {}},
    )
    monkeypatch.setattr(
        pipeline_parity,
        "_publish_handoff_bundle",
        lambda **_kwargs: {"status": "published", "bundle": {"name": "candidate"}},
    )
    monkeypatch.setattr(
        pipeline_parity,
        "_publish_gate_result",
        lambda **_kwargs: {"name": "parity_gates.json", "sha256": "a" * 64},
    )
    monkeypatch.setattr(pipeline_parity, "_finalize_sink", lambda *_args, **_kwargs: None)

    result = run_full_pipeline_benchmark(
        payload,
        source_path=source,
        profile_runner=lambda run_dir, profile: {
            "profile_id": profile.profile_id,
            "execution_mode": profile.execution_mode,
            "status": "complete",
            "config": {},
            "timings_seconds": {"total": 0.01},
            "stages": (
                {
                    "production_full_pipeline": {
                        "steps": [
                            {
                                "stage": "mask",
                                "exact_cv2_loader_enabled": True,
                                "exact_cv2_loader_attempted": True,
                                "exact_cv2_loader_used": True,
                                "exact_cv2_loader_seconds": 1.0,
                                "exact_cv2_loader_required_concurrency": 1,
                                "exact_cv2_loader_configured_concurrency": 1,
                                "exact_cv2_loader_concurrency_ready": True,
                            }
                        ]
                    }
                }
                if profile.profile_id == "production_final_candidate"
                else {}
            ),
            "artifact_inventory": sorted(path.name for path in run_dir.iterdir()),
        },
    )

    assert result["comparisons"]["authoritative_cross_image"]["passed"] is True
    assert result["comparisons"]["authoritative_cross_image"]["candidate_profile_id"] == profile_id
    assert result["parity_passed"] is True
    if image_role == "final_candidate":
        assert result["runtime_gates"]["exact_cv2_loader"]["passed"] is True
    assert result["handoff"]["gates"]["name"] == "parity_gates.json"
    assert payload["assets"]["baseline_runner_mask_mp4"]["data"] not in json.dumps(result)
