from __future__ import annotations

import base64
from dataclasses import dataclass
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import statistics
import tempfile
import threading
import time
from typing import Any, Callable

import cv2
import numpy as np

from whodoirunlike.mask_artifacts import iter_mask_video, mask_iou
from whodoirunlike.running_clip_run import RunningClipRun
from whodoirunlike.sam31_parity import (
    CANONICAL_FRAME130_FIXTURE_ID,
    EXACT_CANDIDATE_COMMIT,
    EXACT_CANDIDATE_IMAGE_DIGEST,
    MaskQualityMeasurements,
    evaluate_strict_mask_gate,
    get_parity_fixture,
    validate_prompt_for_fixture,
)
from whodoirunlike.sam2_runner import inspect_video


BENCHMARK_SCHEMA_VERSION = 1
BENCHMARK_TYPE = "sam31_benchmark"
BENCHMARK_FIXTURE_ID = CANONICAL_FRAME130_FIXTURE_ID
BENCHMARK_VARIANT_ID = "production_candidate_public_entrypoint"
BENCHMARK_VARIANT_IDS = (BENCHMARK_VARIANT_ID,)
MAX_RESPONSE_BYTES = 256 * 1024


@dataclass(frozen=True)
class AssetSpec:
    encoding: str
    sha256: str
    max_decoded_bytes: int


_FIXTURE = get_parity_fixture(BENCHMARK_FIXTURE_ID)
ASSET_SPECS = {
    "person_prompt_json": AssetSpec(
        encoding="base64",
        sha256=_FIXTURE.asset_sha256["person_prompt_json"],
        max_decoded_bytes=16 * 1024,
    ),
    "tracklets_jsonl": AssetSpec(
        encoding="gzip+base64",
        sha256=_FIXTURE.asset_sha256["tracklets_jsonl"],
        max_decoded_bytes=2 * 1024 * 1024,
    ),
    "baseline_runner_mask_mp4": AssetSpec(
        encoding="base64",
        sha256=_FIXTURE.asset_sha256["baseline_runner_mask_mp4"],
        max_decoded_bytes=2 * 1024 * 1024,
    ),
}
FIXTURE_ASSET_SPECS = {BENCHMARK_FIXTURE_ID: ASSET_SPECS}
BENCHMARK_FIXTURE_IDS = (BENCHMARK_FIXTURE_ID,)
_BENCHMARK_LOCK = threading.Lock()


def _round(value: float | int | None, digits: int = 6) -> float | None:
    return round(float(value), digits) if value is not None else None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_gzip_bounded(data: bytes, *, max_bytes: int) -> bytes:
    with gzip.GzipFile(fileobj=io.BytesIO(data), mode="rb") as stream:
        decoded = stream.read(max_bytes + 1)
    if len(decoded) > max_bytes:
        raise ValueError("Compressed benchmark asset exceeds its decoded size limit.")
    return decoded


def _decode_asset(
    name: str,
    payload: Any,
    *,
    asset_specs: dict[str, AssetSpec] | None = None,
) -> bytes:
    spec = (asset_specs or ASSET_SPECS)[name]
    if not isinstance(payload, dict):
        raise ValueError(f"Benchmark asset {name} must be an object.")
    if payload.get("encoding") != spec.encoding:
        raise ValueError(f"Benchmark asset {name} has an unsupported encoding.")
    if payload.get("sha256") != spec.sha256:
        raise ValueError(f"Benchmark asset {name} does not match the fixed fixture hash.")
    encoded = payload.get("data")
    if not isinstance(encoded, str):
        raise ValueError(f"Benchmark asset {name} must contain base64 data.")
    max_encoded_bytes = ((spec.max_decoded_bytes + 2) // 3) * 4 + 4096
    if len(encoded) > max_encoded_bytes:
        raise ValueError(f"Benchmark asset {name} exceeds its encoded size limit.")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Benchmark asset {name} is not valid base64.") from exc
    if spec.encoding == "gzip+base64":
        try:
            raw = _read_gzip_bounded(raw, max_bytes=spec.max_decoded_bytes)
        except (EOFError, OSError) as exc:
            raise ValueError(f"Benchmark asset {name} is not valid gzip data.") from exc
    elif len(raw) > spec.max_decoded_bytes:
        raise ValueError(f"Benchmark asset {name} exceeds its decoded size limit.")
    if _sha256(raw) != spec.sha256:
        raise ValueError(f"Benchmark asset {name} failed SHA-256 verification.")
    return raw


def _fixture_source_path() -> Path:
    configured = os.getenv("WHODOIRUNLIKE_SAM31_BENCHMARK_SOURCE", "").strip()
    repository_root = Path(__file__).resolve().parents[2]
    candidates = [Path(configured)] if configured else []
    candidates.extend(
        [
            repository_root / "site/public/assets/demos/cole-source.mp4",
            Path("/app/site/public/assets/demos/cole-source.mp4"),
        ]
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        if _file_sha256(candidate) != _FIXTURE.source_sha256:
            raise ValueError("Baked benchmark source failed SHA-256 verification.")
        return candidate
    raise FileNotFoundError("The baked SAM 3.1 benchmark source clip is unavailable.")


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _binary_mask(mask: np.ndarray | None, *, height: int, width: int) -> np.ndarray:
    if mask is None:
        return np.zeros((height, width), dtype=np.uint8)
    binary = (np.squeeze(mask) > 0).astype(np.uint8)
    if binary.shape != (height, width):
        binary = cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)
    return binary


def _dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    denominator = int(a.sum()) + int(b.sum())
    if denominator == 0:
        return 1.0
    return float(2 * np.logical_and(a, b).sum() / denominator)


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return None
    return float(xs.mean()), float(ys.mean())


def _centroid_in_box(
    centroid: tuple[float, float] | None,
    box: np.ndarray | None,
) -> bool:
    if centroid is None or box is None:
        return False
    x, y = centroid
    x1, y1, x2, y2 = [float(value) for value in box]
    return x1 <= x <= x2 and y1 <= y <= y2


def _boundary_f1(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    tolerance_pixels: int = 2,
) -> float:
    kernel = np.ones((3, 3), dtype=np.uint8)
    edge_a = np.logical_xor(mask_a > 0, cv2.erode(mask_a, kernel) > 0)
    edge_b = np.logical_xor(mask_b > 0, cv2.erode(mask_b, kernel) > 0)
    count_a = int(edge_a.sum())
    count_b = int(edge_b.sum())
    if count_a == 0 and count_b == 0:
        return 1.0
    if count_a == 0 or count_b == 0:
        return 0.0
    tolerance_kernel = np.ones(
        (tolerance_pixels * 2 + 1, tolerance_pixels * 2 + 1),
        dtype=np.uint8,
    )
    near_a = cv2.dilate(edge_a.astype(np.uint8), tolerance_kernel) > 0
    near_b = cv2.dilate(edge_b.astype(np.uint8), tolerance_kernel) > 0
    precision = float(np.logical_and(edge_a, near_b).sum() / count_a)
    recall = float(np.logical_and(edge_b, near_a).sum() / count_b)
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def _mean_temporal_iou(masks: list[np.ndarray]) -> float | None:
    if len(masks) < 2:
        return None
    return float(
        statistics.fmean(mask_iou(previous, current) for previous, current in zip(masks, masks[1:]))
    )


def compare_masks_to_production_baseline(
    candidate_masks: list[np.ndarray],
    reference_masks: list[np.ndarray],
    *,
    track_boxes: dict[int, np.ndarray],
    fallback_used: bool = False,
    box_loss_frame_count: int = 0,
    expected_frame_count: int | None = None,
) -> dict[str, Any]:
    if not reference_masks:
        raise ValueError("Production baseline mask video contains no frames.")
    height, width = reference_masks[0].shape[:2]
    candidates = [
        _binary_mask(
            candidate_masks[index] if index < len(candidate_masks) else None,
            height=height,
            width=width,
        )
        for index in range(len(reference_masks))
    ]
    references = [_binary_mask(mask, height=height, width=width) for mask in reference_masks]
    diagonal = max(float(np.hypot(width, height)), 1.0)
    ious: list[float] = []
    dices: list[float] = []
    boundary_scores: list[float] = []
    centroid_errors: list[float] = []
    candidate_target_hits: list[bool] = []
    reference_target_hits: list[bool] = []
    for frame_index, (candidate, reference) in enumerate(zip(candidates, references)):
        ious.append(mask_iou(candidate, reference))
        dices.append(_dice(candidate, reference))
        boundary_scores.append(_boundary_f1(candidate, reference))
        candidate_centroid = _centroid(candidate)
        reference_centroid = _centroid(reference)
        if candidate_centroid is not None and reference_centroid is not None:
            centroid_errors.append(
                float(
                    np.hypot(
                        candidate_centroid[0] - reference_centroid[0],
                        candidate_centroid[1] - reference_centroid[1],
                    )
                    / diagonal
                )
            )
        track_box = track_boxes.get(frame_index)
        if track_box is not None:
            candidate_target_hits.append(_centroid_in_box(candidate_centroid, track_box))
            reference_target_hits.append(_centroid_in_box(reference_centroid, track_box))

    candidate_temporal = _mean_temporal_iou(candidates)
    reference_temporal = _mean_temporal_iou(references)
    temporal_delta = (
        abs(candidate_temporal - reference_temporal)
        if candidate_temporal is not None and reference_temporal is not None
        else None
    )
    centroid_error = float(statistics.fmean(centroid_errors)) if centroid_errors else None
    identity_switches = sum(
        1
        for previous, current in zip(
            candidate_target_hits,
            candidate_target_hits[1:],
        )
        if previous and not current
    )
    target_box_misses = sum(not hit for hit in candidate_target_hits)
    measurements = MaskQualityMeasurements(
        reference_frame_count=len(references),
        candidate_frame_count=len(candidate_masks),
        candidate_nonempty_frame_count=sum(int(mask.any()) for mask in candidates),
        iou_mean=float(statistics.fmean(ious)),
        iou_p05=_percentile(ious, 5),
        boundary_f1_mean=float(statistics.fmean(boundary_scores)),
        centroid_error_normalized_mean=centroid_error,
        temporal_iou_absolute_delta=temporal_delta,
        fallback_used=bool(fallback_used),
        box_loss_frame_count=max(int(box_loss_frame_count), target_box_misses),
        identity_switch_count=identity_switches,
    )
    return {
        "label": "agreement_vs_production_baseline",
        "baseline_is_lossy_mp4_not_ground_truth": True,
        "reference_frames": measurements.reference_frame_count,
        "candidate_frames": measurements.candidate_frame_count,
        "candidate_nonempty_frames": measurements.candidate_nonempty_frame_count,
        "iou": {
            "mean": _round(measurements.iou_mean),
            "median": _round(statistics.median(ious)),
            "p05": _round(measurements.iou_p05),
        },
        "dice_mean": _round(statistics.fmean(dices)),
        "boundary_f1_2px_mean": _round(measurements.boundary_f1_mean),
        "centroid_error_normalized_mean": _round(centroid_error),
        "temporal_iou": {
            "candidate_mean": _round(candidate_temporal),
            "reference_mean": _round(reference_temporal),
            "absolute_delta": _round(temporal_delta),
        },
        "tracked_frames": len(candidate_target_hits),
        "target_box_centroid_rate": {
            "candidate": _round(statistics.fmean(candidate_target_hits))
            if candidate_target_hits
            else None,
            "reference": _round(statistics.fmean(reference_target_hits))
            if reference_target_hits
            else None,
        },
        "identity_switch_count": identity_switches,
        "fallback_used": bool(fallback_used),
        "target_box_miss_count": target_box_misses,
        "box_loss_frame_count": measurements.box_loss_frame_count,
        "worst_frame_indices": sorted(range(len(ious)), key=lambda index: ious[index])[:10],
        "strict_mask_agreement_gate": evaluate_strict_mask_gate(
            measurements,
            expected_frame_count=(
                int(expected_frame_count) if expected_frame_count is not None else len(references)
            ),
        ),
    }


def _validate_request(payload: Any) -> tuple[str, str, dict[str, bytes]]:
    if not isinstance(payload, dict):
        raise ValueError("SAM 3.1 benchmark input must be an object.")
    if payload.get("type") != BENCHMARK_TYPE:
        raise ValueError("Unsupported benchmark request type.")
    if payload.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise ValueError("Unsupported SAM 3.1 benchmark schema version.")
    fixture_id = payload.get("fixture_id")
    if fixture_id != BENCHMARK_FIXTURE_ID:
        raise ValueError("Unsupported SAM 3.1 benchmark fixture.")
    variant_id = payload.get("variant_id")
    if variant_id != BENCHMARK_VARIANT_ID:
        raise ValueError("Unsupported SAM 3.1 benchmark variant.")
    assets = payload.get("assets")
    if not isinstance(assets, dict) or set(assets) != set(ASSET_SPECS):
        raise ValueError("SAM 3.1 benchmark requires the exact comparison assets.")
    decoded = {name: _decode_asset(name, assets[name]) for name in ASSET_SPECS}
    prompt_validation = validate_prompt_for_fixture(
        decoded["person_prompt_json"],
        fixture=_FIXTURE,
    )
    if not prompt_validation["raw_hash_matches"]:
        raise ValueError("Canonical frame-130 prompt failed raw SHA-256 verification.")
    return str(fixture_id), str(variant_id), decoded


PublicMaskRunner = Callable[[Path], dict[str, Any]]


def _run_public_mask_entrypoint(run_dir: Path) -> dict[str, Any]:
    from whodoirunlike.sam31_gpu_runner import run_sam31_gpu_mask

    return run_sam31_gpu_mask(run_dir=run_dir)


def _safe_runner_summary(result: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "backend",
        "model",
        "frame_count",
        "prompt_frame",
        "detected_frames",
        "cache_hit",
        "model_build_seconds",
        "predictor_lock_wait_seconds",
        "data_ready_seconds",
        "elapsed_seconds",
        "mask_summary",
        "fallback",
    )
    return {field: result[field] for field in fields if field in result}


def run_candidate_mask_stage(
    run_dir: Path,
    *,
    mask_runner: PublicMaskRunner | None = None,
) -> dict[str, Any]:
    """Run the pinned candidate's public mask entrypoint and retain its artifacts."""

    run_dir = Path(run_dir)
    run = RunningClipRun(run_dir)
    manifest = run.read_manifest()
    source_path = run.artifact_path("source_segment", manifest)
    baseline_path = run_dir / "baseline_runner_mask.mp4"
    original_mask_path = run.artifact_path("runner_mask", manifest)
    if not baseline_path.is_file():
        shutil.copy2(original_mask_path, baseline_path)

    source_meta = inspect_video(source_path)
    baseline_meta, baseline_masks = iter_mask_video(baseline_path)
    if (
        int(source_meta.get("width") or 0) != _FIXTURE.width
        or int(source_meta.get("height") or 0) != _FIXTURE.height
        or int(source_meta.get("frame_count") or 0) != _FIXTURE.frame_count
        or int(baseline_meta.get("frame_count") or 0) != _FIXTURE.frame_count
    ):
        raise ValueError("Candidate mask run does not match the canonical fixture metadata.")

    started_at = time.perf_counter()
    runner_result = (mask_runner or _run_public_mask_entrypoint)(run_dir)
    elapsed_seconds = time.perf_counter() - started_at
    if not isinstance(runner_result, dict):
        raise TypeError("Public SAM 3.1 mask entrypoint must return an object.")

    completed_manifest = run.read_manifest()
    candidate_mask_path = run.artifact_path("runner_mask", completed_manifest)
    masks_jsonl_path = run.artifact_path("masks_jsonl", completed_manifest)
    if not candidate_mask_path.is_file() or not masks_jsonl_path.is_file():
        raise RuntimeError("Public SAM 3.1 entrypoint did not persist its mask contract.")
    candidate_meta, candidate_masks = iter_mask_video(candidate_mask_path)
    if (
        int(candidate_meta.get("width") or 0) != _FIXTURE.width
        or int(candidate_meta.get("height") or 0) != _FIXTURE.height
        or int(candidate_meta.get("frame_count") or 0) != _FIXTURE.frame_count
    ):
        raise ValueError("Candidate mask output does not match the canonical fixture metadata.")

    from whodoirunlike.sam31_mlx_runner import load_track_boxes

    track_boxes = load_track_boxes(
        {
            "tracklets_jsonl": str(run.artifact_path("tracklets_jsonl", completed_manifest)),
            "tracklets": str(run.artifact_path("tracklets", completed_manifest)),
        },
        width=_FIXTURE.width,
        height=_FIXTURE.height,
    )
    prompting = runner_result.get("prompting")
    identity_filter = (
        prompting.get("identity_filter")
        if isinstance(prompting, dict) and isinstance(prompting.get("identity_filter"), dict)
        else {}
    )
    quality = compare_masks_to_production_baseline(
        candidate_masks,
        baseline_masks,
        track_boxes=track_boxes,
        fallback_used=bool(runner_result.get("fallback")),
        box_loss_frame_count=int(identity_filter.get("rejected_frames") or 0),
        expected_frame_count=_FIXTURE.frame_count,
    )
    return {
        "status": "complete",
        "entrypoint": "whodoirunlike.sam31_gpu_runner.run_sam31_gpu_mask",
        "elapsed_seconds": _round(elapsed_seconds),
        "runner": _safe_runner_summary(runner_result),
        "quality_vs_production_baseline": quality,
        "persisted_artifacts": {
            "runner_mask": {
                "name": candidate_mask_path.name,
                "sha256": _file_sha256(candidate_mask_path),
                "bytes": candidate_mask_path.stat().st_size,
            },
            "masks_jsonl": {
                "name": masks_jsonl_path.name,
                "sha256": _file_sha256(masks_jsonl_path),
                "bytes": masks_jsonl_path.stat().st_size,
            },
        },
    }


def _runtime_metadata() -> dict[str, Any]:
    image_role = os.getenv("WHODOIRUNLIKE_BENCHMARK_IMAGE_ROLE", "candidate")
    result: dict[str, Any] = {
        "processor_version": os.getenv("WHODOIRUNLIKE_PROCESSOR_VERSION", "unknown"),
        "benchmark_version": os.getenv("WHODOIRUNLIKE_BENCHMARK_VERSION", "unknown"),
        "candidate_commit": EXACT_CANDIDATE_COMMIT,
        "candidate_image_digest": EXACT_CANDIDATE_IMAGE_DIGEST,
        "base_image_role": image_role,
        "base_processor_commit": os.getenv(
            "WHODOIRUNLIKE_BASE_PROCESSOR_COMMIT",
            EXACT_CANDIDATE_COMMIT,
        ),
        "base_processor_image_digest": os.getenv(
            "WHODOIRUNLIKE_BASE_PROCESSOR_IMAGE_DIGEST",
            EXACT_CANDIDATE_IMAGE_DIGEST,
        ),
    }
    try:
        import torch

        result["torch_version"] = getattr(torch, "__version__", "unknown")
        result["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            result["gpu_name"] = torch.cuda.get_device_name(0)
    except ModuleNotFoundError:
        result["cuda_available"] = False
    return result


def _run_benchmark_locked(
    payload: dict[str, Any],
    *,
    source_path: Path | None = None,
    mask_runner: PublicMaskRunner | None = None,
) -> dict[str, Any]:
    fixture_id, variant_id, assets = _validate_request(payload)
    source_path = Path(source_path) if source_path is not None else _fixture_source_path()
    if _file_sha256(source_path) != _FIXTURE.source_sha256:
        raise ValueError("Canonical benchmark source failed SHA-256 verification.")

    from whodoirunlike.pipeline_parity import materialize_pipeline_fixture

    with tempfile.TemporaryDirectory(prefix="wdirl-sam31-public-benchmark-") as temp_name:
        run_dir = Path(temp_name) / "candidate"
        materialize_pipeline_fixture(
            run_dir=run_dir,
            source_path=source_path,
            assets=assets,
            profile_id=variant_id,
        )
        candidate = run_candidate_mask_stage(
            run_dir,
            mask_runner=mask_runner,
        )

    result = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "type": "sam31_public_mask_parity",
        "fixture": {
            "id": fixture_id,
            "source_sha256": _FIXTURE.source_sha256,
            "prompt_sha256": _FIXTURE.prompt.raw_sha256,
            "tracklets_sha256": _FIXTURE.asset_sha256["tracklets_jsonl"],
            "baseline_mask_sha256": _FIXTURE.asset_sha256["baseline_runner_mask_mp4"],
            "frame_count": _FIXTURE.frame_count,
            "width": _FIXTURE.width,
            "height": _FIXTURE.height,
        },
        "variant_id": variant_id,
        "runtime": _runtime_metadata(),
        "candidate": candidate,
        "parity_passed": bool(
            candidate["quality_vs_production_baseline"]["strict_mask_agreement_gate"]["passed"]
        ),
        "response_bytes": 0,
    }
    encoded = json.dumps(
        result,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    result["response_bytes"] = len(encoded)
    encoded = json.dumps(
        result,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    result["response_bytes"] = len(encoded)
    if len(encoded) > MAX_RESPONSE_BYTES:
        raise RuntimeError("SAM 3.1 benchmark response exceeded its size limit.")
    return result


def run_benchmark(payload: dict[str, Any]) -> dict[str, Any]:
    with _BENCHMARK_LOCK:
        return _run_benchmark_locked(payload)
