from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import resource
import shutil
import statistics
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from whodoirunlike.cv_flow import write_json
from whodoirunlike.densepose_runner import (
    clear_densepose_backend_cache,
    load_densepose_backend,
    run_densepose,
)
from whodoirunlike.full_pipeline import _densepose_runtime_kwargs
from whodoirunlike.sam2_runner import inspect_video


BENCHMARK_TYPE = "densepose_batch_benchmark"
BENCHMARK_RESULT_TYPE = "densepose_batch_benchmark_result"
BENCHMARK_SCHEMA_VERSION = 1
CANONICAL_FIXTURE_ID = "cole-frame130-production-v1"
BENCHMARK_PROFILE_ENV = "WHODOIRUNLIKE_DENSEPOSE_BENCHMARK_PROFILE"
TARGET_CROP_PROFILE_ID = "target-crop-512-960-v1"
LIVE_CONTROL_PROFILE_ID = "live-control-no-resize-override-v1"
BENCHMARK_PROFILE_IDS = frozenset({TARGET_CROP_PROFILE_ID, LIVE_CONTROL_PROFILE_ID})
ALLOWED_BATCH_SIZES = (1, 2, 4, 8)
MEASURED_REPETITIONS = 3
MAX_REQUEST_BYTES = 1024 * 1024
MAX_RESPONSE_BYTES = 256 * 1024
MAX_MISMATCH_RECORDS = 16
SOURCE_SHA256 = "a8146591119c5439cc01168df63fa6144a7a55ff6817726946e1e8f5bc381617"
RUNNER_MASK_SHA256 = "f7bb2d1ed00767ed2866c5b3a57b47361a591f1dbf090a5089d187f9ae410ef7"
RUNNER_MASK_BYTES = 526_609
RUNNER_MASK_BASE64_CHARS = 702_148
EXPECTED_FRAME_COUNT = 260
EXPECTED_WIDTH = 960
EXPECTED_HEIGHT = 540
EXPECTED_FPS = 29.97
EXPECTED_GPU_NAME = "NVIDIA A40"
A40_MEMORY_BYTES = 48 * 1024**3
MIN_INFERENCE_SPEEDUP = 0.15
MAX_RESERVED_MEMORY_FRACTION = 0.80
MAX_RESERVED_MEMORY_DELTA_BYTES = 8 * 1024**3
FASTEST_WINDOW_FRACTION = 0.05
REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "site/public/assets/demos/cole-source.mp4"

DENSEPOSE_THRESHOLDS = {
    "part_jaccard_mean_min": 0.99,
    "part_jaccard_p05_min": 0.95,
    "centroid_error_normalized_mean_max": 0.005,
    "centroid_error_normalized_p95_max": 0.015,
    "bbox_iou_p05_min": 0.95,
    "coverage_mae_max": 0.01,
    "mask_overlap_mae_max": 0.01,
    "score_mae_max": 0.005,
    "label_part_iou_mean_min": 0.999,
    "label_part_iou_p05_min": 0.995,
    "qa_ssim_p05_min": 0.995,
    "qa_normalized_mae_max": 0.005,
    "qa_fps_abs_delta_max": 0.01,
}

_REQUEST_KEYS = frozenset({"type", "schema_version", "fixture_id", "batch_sizes", "assets"})
_ASSET_KEYS = frozenset({"baseline_runner_mask_mp4"})
_ENCODED_ASSET_KEYS = frozenset({"encoding", "sha256", "data"})
_REQUIRED_DENSEPOSE_FIELDS = frozenset(
    {
        "frame_index",
        "usable",
        "drop_reason",
        "part_ids",
        "part_centroids",
        "densepose_coverage",
        "mask_overlap",
        "bbox",
        "score",
        "inference_input",
    }
)


class BenchmarkContractError(ValueError):
    pass


class BenchmarkResponseTooLarge(RuntimeError):
    pass


@dataclass(frozen=True)
class BenchmarkRequest:
    batch_sizes: tuple[int, ...]
    runner_mask_bytes: bytes
    request_sha256: str
    request_bytes: int


@dataclass
class LabelFrameEvidence:
    frame_index: int
    digest: str
    shape: tuple[int, ...] | None
    dtype: str | None
    nonzero_pixels: int
    histogram: dict[int, int]
    labels: np.ndarray | None


@dataclass
class BenchmarkRun:
    batch_size: int
    sample: dict[str, Any]
    rows: list[dict[str, Any]]
    labels: "LabelEvidenceCollector | None"
    qa_overlay_path: Path


@dataclass(frozen=True)
class CudaRuntime:
    synchronize: Callable[[], None]
    reset_peak_memory_stats: Callable[[], None]
    max_memory_allocated: Callable[[], int]
    max_memory_reserved: Callable[[], int]
    gpu_name: str
    torch_version: str
    cuda_version: str | None


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def serialized_size(value: Any) -> int:
    return len(_json_bytes(value))


def ensure_bounded_response(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["response_limit_bytes"] = MAX_RESPONSE_BYTES
    result["response_bytes"] = 0
    for _ in range(3):
        result["response_bytes"] = serialized_size(result)
    if result["response_bytes"] > MAX_RESPONSE_BYTES:
        raise BenchmarkResponseTooLarge(
            f"DensePose benchmark response exceeds {MAX_RESPONSE_BYTES} bytes"
        )
    return result


def bounded_failure(
    code: str,
    *,
    batch_size: int | None = None,
    exception_type: str | None = None,
) -> dict[str, Any]:
    failure: dict[str, Any] = {"code": code}
    if batch_size in ALLOWED_BATCH_SIZES:
        failure["batch_size"] = batch_size
    if exception_type and len(exception_type) <= 96 and exception_type.replace("_", "").isalnum():
        failure["exception_type"] = exception_type
    return ensure_bounded_response(
        {
            "type": BENCHMARK_RESULT_TYPE,
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "status": "failed",
            # RunPod treats a top-level `error` key as an SDK control flag and removes it
            # from otherwise successful handler output. Keep bounded diagnostics under a
            # neutral field so callers can inspect the failure code and exception class.
            "failure": failure,
        }
    )


def validate_request(payload: Any) -> BenchmarkRequest:
    if not isinstance(payload, dict):
        raise BenchmarkContractError("DensePose benchmark input must be an object")
    try:
        raw_request = _json_bytes(payload)
    except (TypeError, ValueError) as exc:
        raise BenchmarkContractError("DensePose benchmark input must be bounded JSON") from exc
    if len(raw_request) > MAX_REQUEST_BYTES:
        raise BenchmarkContractError("DensePose benchmark input exceeds 1 MiB")
    if set(payload) != _REQUEST_KEYS:
        raise BenchmarkContractError("DensePose benchmark input has unexpected fields")
    if payload.get("type") != BENCHMARK_TYPE:
        raise BenchmarkContractError("Unsupported DensePose benchmark request type")
    if payload.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise BenchmarkContractError("Unsupported DensePose benchmark schema version")
    if payload.get("fixture_id") != CANONICAL_FIXTURE_ID:
        raise BenchmarkContractError("DensePose benchmark requires the canonical fixture")

    raw_batch_sizes = payload.get("batch_sizes")
    if not isinstance(raw_batch_sizes, list) or not raw_batch_sizes:
        raise BenchmarkContractError("batch_sizes must be a nonempty list")
    if len(raw_batch_sizes) > len(ALLOWED_BATCH_SIZES):
        raise BenchmarkContractError("batch_sizes contains too many entries")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in raw_batch_sizes):
        raise BenchmarkContractError("batch_sizes must contain integers")
    if any(value not in ALLOWED_BATCH_SIZES for value in raw_batch_sizes):
        raise BenchmarkContractError("batch_sizes contains an unsupported batch size")
    if len(set(raw_batch_sizes)) != len(raw_batch_sizes):
        raise BenchmarkContractError("batch_sizes must not contain duplicates")
    if 1 not in raw_batch_sizes:
        raise BenchmarkContractError("batch_sizes must include the batch=1 control")
    batch_sizes = tuple(value for value in ALLOWED_BATCH_SIZES if value in raw_batch_sizes)

    assets = payload.get("assets")
    if not isinstance(assets, dict) or set(assets) != _ASSET_KEYS:
        raise BenchmarkContractError("DensePose benchmark requires the exact fixture asset")
    encoded = assets["baseline_runner_mask_mp4"]
    if not isinstance(encoded, dict) or set(encoded) != _ENCODED_ASSET_KEYS:
        raise BenchmarkContractError("Runner-mask fixture has unexpected fields")
    if encoded.get("encoding") != "base64":
        raise BenchmarkContractError("Runner-mask fixture must use base64")
    if encoded.get("sha256") != RUNNER_MASK_SHA256:
        raise BenchmarkContractError("Runner-mask fixture SHA-256 does not match")
    data = encoded.get("data")
    if not isinstance(data, str) or len(data) != RUNNER_MASK_BASE64_CHARS:
        raise BenchmarkContractError("Runner-mask fixture has an unexpected encoded length")
    try:
        runner_mask_bytes = base64.b64decode(data, validate=True)
    except (TypeError, ValueError) as exc:
        raise BenchmarkContractError("Runner-mask fixture is not strict base64") from exc
    if len(runner_mask_bytes) != RUNNER_MASK_BYTES:
        raise BenchmarkContractError("Runner-mask fixture has an unexpected decoded length")
    if base64.b64encode(runner_mask_bytes).decode("ascii") != data:
        raise BenchmarkContractError("Runner-mask fixture base64 is not canonical")
    if hashlib.sha256(runner_mask_bytes).hexdigest() != RUNNER_MASK_SHA256:
        raise BenchmarkContractError("Runner-mask fixture failed SHA-256 verification")

    return BenchmarkRequest(
        batch_sizes=batch_sizes,
        runner_mask_bytes=runner_mask_bytes,
        request_sha256=hashlib.sha256(raw_request).hexdigest(),
        request_bytes=len(raw_request),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_fixture(source_path: Path, runner_mask_path: Path) -> dict[str, Any]:
    if not source_path.is_file() or _file_sha256(source_path) != SOURCE_SHA256:
        raise BenchmarkContractError("Canonical source fixture failed SHA-256 verification")
    if not runner_mask_path.is_file() or _file_sha256(runner_mask_path) != RUNNER_MASK_SHA256:
        raise BenchmarkContractError("Canonical runner-mask fixture failed SHA-256 verification")
    source_meta = inspect_video(source_path)
    mask_meta = inspect_video(runner_mask_path)
    for label, metadata in (("source", source_meta), ("runner mask", mask_meta)):
        if (
            int(metadata.get("frame_count") or 0) != EXPECTED_FRAME_COUNT
            or int(metadata.get("width") or 0) != EXPECTED_WIDTH
            or int(metadata.get("height") or 0) != EXPECTED_HEIGHT
            or abs(float(metadata.get("fps") or 0.0) - EXPECTED_FPS) > 0.01
        ):
            raise BenchmarkContractError(f"Canonical {label} fixture metadata does not match")
    return {
        "fixture_id": CANONICAL_FIXTURE_ID,
        "source_sha256": SOURCE_SHA256,
        "runner_mask_sha256": RUNNER_MASK_SHA256,
        "frame_count": EXPECTED_FRAME_COUNT,
        "width": EXPECTED_WIDTH,
        "height": EXPECTED_HEIGHT,
        "source_fps": round(float(source_meta["fps"]), 6),
        "runner_mask_fps": round(float(mask_meta["fps"]), 6),
    }


def benchmark_profile() -> str:
    profile_id = os.getenv(BENCHMARK_PROFILE_ENV, "").strip() or TARGET_CROP_PROFILE_ID
    if profile_id not in BENCHMARK_PROFILE_IDS:
        raise BenchmarkContractError("DensePose benchmark profile is unsupported")
    return profile_id


def validate_runtime_configuration(
    runtime_kwargs: dict[str, Any],
    *,
    profile_id: str | None = None,
) -> None:
    selected_profile = profile_id or benchmark_profile()
    if selected_profile == TARGET_CROP_PROFILE_ID:
        numerical_settings = {
            "input_min_size_test": 512,
            "input_max_size_test": 960,
            "target_crop_enabled": True,
        }
    elif selected_profile == LIVE_CONTROL_PROFILE_ID:
        numerical_settings = {
            "input_min_size_test": None,
            "input_max_size_test": None,
            "target_crop_enabled": False,
        }
    else:
        raise BenchmarkContractError("DensePose benchmark profile is unsupported")
    expected = {
        "device": "cuda",
        "target_crop_padding_ratio": 0.2,
        "target_crop_padding_pixels": 16,
        **numerical_settings,
    }
    mismatches = [key for key, value in expected.items() if runtime_kwargs.get(key) != value]
    if mismatches:
        raise BenchmarkContractError(
            "DensePose benchmark runtime is not pinned: " + ", ".join(mismatches)
        )


def load_cuda_runtime() -> CudaRuntime:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise BenchmarkContractError("DensePose benchmark requires one CUDA GPU")
    return CudaRuntime(
        synchronize=torch.cuda.synchronize,
        reset_peak_memory_stats=torch.cuda.reset_peak_memory_stats,
        max_memory_allocated=lambda: int(torch.cuda.max_memory_allocated()),
        max_memory_reserved=lambda: int(torch.cuda.max_memory_reserved()),
        gpu_name=str(torch.cuda.get_device_name(0)),
        torch_version=str(torch.__version__),
        cuda_version=str(torch.version.cuda) if torch.version.cuda is not None else None,
    )


def _current_rss_bytes() -> int:
    status_path = Path("/proc/self/status")
    if status_path.is_file():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                fields = line.split()
                if len(fields) >= 2 and fields[1].isdigit():
                    return int(fields[1]) * 1024
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(usage if os.uname().sysname == "Darwin" else usage * 1024)


class LabelEvidenceCollector:
    def __init__(self) -> None:
        self.frames: dict[int, LabelFrameEvidence] = {}

    def __call__(
        self,
        frame_index: int,
        row: dict[str, Any],
        labels: np.ndarray | None,
    ) -> None:
        if frame_index in self.frames:
            raise RuntimeError("DensePose benchmark received duplicate frame evidence")
        if labels is None:
            metadata = {
                "frame_index": int(frame_index),
                "state": "missing",
                "usable": bool(row.get("usable")),
                "drop_reason": str(row.get("drop_reason") or ""),
            }
            digest = hashlib.sha256(_json_bytes(metadata)).hexdigest()
            evidence = LabelFrameEvidence(
                frame_index=frame_index,
                digest=digest,
                shape=None,
                dtype=None,
                nonzero_pixels=0,
                histogram={},
                labels=None,
            )
        else:
            contiguous = np.ascontiguousarray(labels)
            values, counts = np.unique(contiguous, return_counts=True)
            histogram = {
                int(value): int(count) for value, count in zip(values.tolist(), counts.tolist())
            }
            metadata = {
                "frame_index": int(frame_index),
                "shape": [int(value) for value in contiguous.shape],
                "dtype": str(contiguous.dtype),
                "nonzero_pixels": int(np.count_nonzero(contiguous)),
                "histogram": [[key, histogram[key]] for key in sorted(histogram)],
            }
            digest_builder = hashlib.sha256()
            digest_builder.update(_json_bytes(metadata))
            digest_builder.update(b"\0")
            digest_builder.update(contiguous.tobytes(order="C"))
            evidence = LabelFrameEvidence(
                frame_index=frame_index,
                digest=digest_builder.hexdigest(),
                shape=tuple(int(value) for value in contiguous.shape),
                dtype=str(contiguous.dtype),
                nonzero_pixels=metadata["nonzero_pixels"],
                histogram=histogram,
                labels=contiguous.copy(),
            )
        self.frames[frame_index] = evidence

    def summary(self) -> dict[str, Any]:
        ordered = [self.frames[index] for index in sorted(self.frames)]
        shape_counts: Counter[str] = Counter()
        dtype_counts: Counter[str] = Counter()
        histogram: Counter[int] = Counter()
        for frame in ordered:
            shape_key = "missing" if frame.shape is None else "x".join(map(str, frame.shape))
            shape_counts[shape_key] += 1
            dtype_counts[frame.dtype or "missing"] += 1
            histogram.update(frame.histogram)
        sequence = "\n".join(f"{frame.frame_index}:{frame.digest}" for frame in ordered).encode(
            "ascii"
        )
        return {
            "frame_count": len(ordered),
            "frame_indices_sha256": hashlib.sha256(
                _json_bytes([frame.frame_index for frame in ordered])
            ).hexdigest(),
            "frame_hashes": [frame.digest for frame in ordered],
            "sequence_sha256": hashlib.sha256(sequence).hexdigest(),
            "shape_counts": dict(sorted(shape_counts.items())),
            "dtype_counts": dict(sorted(dtype_counts.items())),
            "nonzero_pixels": sum(frame.nonzero_pixels for frame in ordered),
            "part_histogram": {str(key): histogram[key] for key in sorted(histogram)},
        }


def _part_label_iou(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return 1.0 if left is None and right is None else 0.0
    if left.shape != right.shape or left.dtype != right.dtype:
        return 0.0
    part_ids = sorted(
        set(int(value) for value in np.unique(left) if int(value) > 0)
        | set(int(value) for value in np.unique(right) if int(value) > 0)
    )
    if not part_ids:
        return 1.0 if np.array_equal(left, right) else 0.0
    values = []
    for part_id in part_ids:
        left_part = left == part_id
        right_part = right == part_id
        union = int(np.count_nonzero(left_part | right_part))
        intersection = int(np.count_nonzero(left_part & right_part))
        values.append(intersection / union if union else 1.0)
    return float(statistics.fmean(values))


def compare_label_evidence(
    control: LabelEvidenceCollector,
    candidate: LabelEvidenceCollector,
) -> dict[str, Any]:
    indices = sorted(set(control.frames) | set(candidate.frames))
    exact_matches = 0
    shape_matches = 0
    dtype_matches = 0
    ious: list[float] = []
    mismatches: list[dict[str, Any]] = []
    for frame_index in indices:
        left = control.frames.get(frame_index)
        right = candidate.frames.get(frame_index)
        exact = left is not None and right is not None and left.digest == right.digest
        exact_matches += int(exact)
        shape_matches += int(left is not None and right is not None and left.shape == right.shape)
        dtype_matches += int(left is not None and right is not None and left.dtype == right.dtype)
        iou = _part_label_iou(
            left.labels if left is not None else None,
            right.labels if right is not None else None,
        )
        ious.append(iou)
        if not exact and len(mismatches) < MAX_MISMATCH_RECORDS:
            mismatches.append(
                {
                    "frame_index": frame_index,
                    "control_sha256": left.digest if left is not None else None,
                    "candidate_sha256": right.digest if right is not None else None,
                    "control_shape": list(left.shape) if left and left.shape else None,
                    "candidate_shape": list(right.shape) if right and right.shape else None,
                }
            )
    denominator = max(len(indices), 1)
    measurements = {
        "aligned_frame_count": len(set(control.frames) & set(candidate.frames)),
        "union_frame_count": len(indices),
        "exact_hash_match_rate": exact_matches / denominator,
        "shape_match_rate": shape_matches / denominator,
        "dtype_match_rate": dtype_matches / denominator,
        "part_iou_mean": float(statistics.fmean(ious)) if ious else None,
        "part_iou_p05": float(np.percentile(np.asarray(ious), 5)) if ious else None,
        "mismatch_count": len(indices) - exact_matches,
        "mismatches": mismatches,
        "mismatches_truncated": len(indices) - exact_matches > len(mismatches),
    }
    checks = {
        "frame_count_exact": len(control.frames) == len(candidate.frames) == EXPECTED_FRAME_COUNT,
        "frame_indices_exact": set(control.frames)
        == set(candidate.frames)
        == set(range(EXPECTED_FRAME_COUNT)),
        "label_shapes_exact": measurements["shape_match_rate"] == 1.0,
        "label_dtypes_exact": measurements["dtype_match_rate"] == 1.0,
        "label_part_iou_mean": measurements["part_iou_mean"] is not None
        and measurements["part_iou_mean"] >= DENSEPOSE_THRESHOLDS["label_part_iou_mean_min"],
        "label_part_iou_p05": measurements["part_iou_p05"] is not None
        and measurements["part_iou_p05"] >= DENSEPOSE_THRESHOLDS["label_part_iou_p05_min"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "diagnostics": {
            "raw_label_hashes_exact": measurements["exact_hash_match_rate"] == 1.0,
        },
        "measurements": measurements,
        "thresholds": {
            "shape_match_rate": 1.0,
            "dtype_match_rate": 1.0,
            "label_part_iou_mean_min": DENSEPOSE_THRESHOLDS["label_part_iou_mean_min"],
            "label_part_iou_p05_min": DENSEPOSE_THRESHOLDS["label_part_iou_p05_min"],
        },
    }


def _rows_by_frame(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row.get("frame_index", -1)): row for row in rows}


def _row_schema(rows: list[dict[str, Any]]) -> frozenset[str]:
    return frozenset(str(key) for row in rows for key in row)


def _xywh_iou(left: Any, right: Any) -> float | None:
    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
        return None
    if len(left) != 4 or len(right) != 4:
        return None
    try:
        ax, ay, aw, ah = (float(value) for value in left)
        bx, by, bw, bh = (float(value) for value in right)
    except (TypeError, ValueError):
        return None
    intersection = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0.0, min(ay + ah, by + bh) - max(ay, by)
    )
    union = max(0.0, aw * ah) + max(0.0, bw * bh) - intersection
    return intersection / union if union else None


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def compare_densepose_rows(
    control_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    control_by_frame = _rows_by_frame(control_rows)
    candidate_by_frame = _rows_by_frame(candidate_rows)
    indices = sorted(set(control_by_frame) & set(candidate_by_frame))
    usable_matches: list[bool] = []
    drop_reason_matches: list[bool] = []
    part_jaccards: list[float] = []
    centroid_errors: list[float] = []
    bbox_ious: list[float] = []
    coverage_errors: list[float] = []
    overlap_errors: list[float] = []
    score_errors: list[float] = []
    new_unusable = 0
    common_usable = 0
    for frame_index in indices:
        control = control_by_frame[frame_index]
        candidate = candidate_by_frame[frame_index]
        control_usable = bool(control.get("usable"))
        candidate_usable = bool(candidate.get("usable"))
        usable_matches.append(control_usable == candidate_usable)
        drop_reason_matches.append(control.get("drop_reason") == candidate.get("drop_reason"))
        new_unusable += int(control_usable and not candidate_usable)
        if not (control_usable and candidate_usable):
            continue
        common_usable += 1
        control_parts = {int(value) for value in control.get("part_ids") or []}
        candidate_parts = {int(value) for value in candidate.get("part_ids") or []}
        union = control_parts | candidate_parts
        part_jaccards.append(len(control_parts & candidate_parts) / len(union) if union else 1.0)
        control_centroids = control.get("part_centroids") or {}
        candidate_centroids = candidate.get("part_centroids") or {}
        for part_id in sorted(set(control_centroids) & set(candidate_centroids)):
            left = control_centroids.get(part_id) or {}
            right = candidate_centroids.get(part_id) or {}
            left_x = _finite_float(left.get("x"))
            left_y = _finite_float(left.get("y"))
            right_x = _finite_float(right.get("x"))
            right_y = _finite_float(right.get("y"))
            if None not in (left_x, left_y, right_x, right_y):
                centroid_errors.append(math.hypot(left_x - right_x, left_y - right_y))
        bbox_iou = _xywh_iou(control.get("bbox"), candidate.get("bbox"))
        if bbox_iou is not None:
            bbox_ious.append(bbox_iou)
        for key, target in (
            ("densepose_coverage", coverage_errors),
            ("mask_overlap", overlap_errors),
            ("score", score_errors),
        ):
            left = _finite_float(control.get(key))
            right = _finite_float(candidate.get(key))
            if left is not None and right is not None:
                target.append(abs(left - right))

    measurements = {
        "control_frame_count": len(control_rows),
        "candidate_frame_count": len(candidate_rows),
        "aligned_frame_count": len(indices),
        "schema_match": _row_schema(control_rows) == _row_schema(candidate_rows),
        "required_fields_present": _REQUIRED_DENSEPOSE_FIELDS.issubset(_row_schema(control_rows))
        and _REQUIRED_DENSEPOSE_FIELDS.issubset(_row_schema(candidate_rows)),
        "usable_agreement_rate": float(statistics.fmean(usable_matches))
        if usable_matches
        else None,
        "drop_reason_agreement_rate": (
            float(statistics.fmean(drop_reason_matches)) if drop_reason_matches else None
        ),
        "new_unusable_frame_count": new_unusable,
        "common_usable_frame_count": common_usable,
        "part_jaccard_mean": float(statistics.fmean(part_jaccards)) if part_jaccards else None,
        "part_jaccard_p05": (
            float(np.percentile(np.asarray(part_jaccards), 5)) if part_jaccards else None
        ),
        "centroid_error_normalized_mean": (
            float(statistics.fmean(centroid_errors)) if centroid_errors else None
        ),
        "centroid_error_normalized_p95": (
            float(np.percentile(np.asarray(centroid_errors), 95)) if centroid_errors else None
        ),
        "bbox_iou_p05": float(np.percentile(np.asarray(bbox_ious), 5)) if bbox_ious else None,
        "coverage_mae": float(statistics.fmean(coverage_errors)) if coverage_errors else None,
        "mask_overlap_mae": float(statistics.fmean(overlap_errors)) if overlap_errors else None,
        "score_mae": float(statistics.fmean(score_errors)) if score_errors else None,
    }
    checks = {
        "control_frame_count_exact": len(control_rows) == EXPECTED_FRAME_COUNT,
        "candidate_frame_count_exact": len(candidate_rows) == EXPECTED_FRAME_COUNT,
        "frame_indices_exact": list(control_by_frame) == list(range(EXPECTED_FRAME_COUNT))
        and list(candidate_by_frame) == list(range(EXPECTED_FRAME_COUNT)),
        "schema_match": measurements["schema_match"],
        "required_fields_present": measurements["required_fields_present"],
        "usable_exact": measurements["usable_agreement_rate"] == 1.0,
        "drop_reason_exact": measurements["drop_reason_agreement_rate"] == 1.0,
        "no_new_unusable_frames": new_unusable == 0,
        "common_usable_frame_evidence": common_usable > 0,
        "part_jaccard_mean": measurements["part_jaccard_mean"] is not None
        and measurements["part_jaccard_mean"] >= DENSEPOSE_THRESHOLDS["part_jaccard_mean_min"],
        "part_jaccard_p05": measurements["part_jaccard_p05"] is not None
        and measurements["part_jaccard_p05"] >= DENSEPOSE_THRESHOLDS["part_jaccard_p05_min"],
        "centroid_error_normalized_mean": measurements["centroid_error_normalized_mean"] is not None
        and measurements["centroid_error_normalized_mean"]
        <= DENSEPOSE_THRESHOLDS["centroid_error_normalized_mean_max"],
        "centroid_error_normalized_p95": measurements["centroid_error_normalized_p95"] is not None
        and measurements["centroid_error_normalized_p95"]
        <= DENSEPOSE_THRESHOLDS["centroid_error_normalized_p95_max"],
        "bbox_iou_p05": measurements["bbox_iou_p05"] is not None
        and measurements["bbox_iou_p05"] >= DENSEPOSE_THRESHOLDS["bbox_iou_p05_min"],
        "coverage_mae": measurements["coverage_mae"] is not None
        and measurements["coverage_mae"] <= DENSEPOSE_THRESHOLDS["coverage_mae_max"],
        "mask_overlap_mae": measurements["mask_overlap_mae"] is not None
        and measurements["mask_overlap_mae"] <= DENSEPOSE_THRESHOLDS["mask_overlap_mae_max"],
        "score_mae": measurements["score_mae"] is not None
        and measurements["score_mae"] <= DENSEPOSE_THRESHOLDS["score_mae_max"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "measurements": measurements,
        "thresholds": {
            **DENSEPOSE_THRESHOLDS,
            "usable_agreement_rate": 1.0,
            "drop_reason_agreement_rate": 1.0,
            "new_unusable_frame_count": 0,
        },
    }


def _ssim(left: np.ndarray, right: np.ndarray) -> float:
    left_float = left.astype(np.float64)
    right_float = right.astype(np.float64)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_left = cv2.GaussianBlur(left_float, (11, 11), 1.5)
    mu_right = cv2.GaussianBlur(right_float, (11, 11), 1.5)
    mu_left_sq = mu_left * mu_left
    mu_right_sq = mu_right * mu_right
    mu_product = mu_left * mu_right
    sigma_left_sq = cv2.GaussianBlur(left_float * left_float, (11, 11), 1.5) - mu_left_sq
    sigma_right_sq = cv2.GaussianBlur(right_float * right_float, (11, 11), 1.5) - mu_right_sq
    sigma_product = cv2.GaussianBlur(left_float * right_float, (11, 11), 1.5) - mu_product
    numerator = (2 * mu_product + c1) * (2 * sigma_product + c2)
    denominator = (mu_left_sq + mu_right_sq + c1) * (sigma_left_sq + sigma_right_sq + c2)
    return float(np.mean(numerator / np.maximum(denominator, 1e-12)))


def compare_qa_videos(control_path: Path, candidate_path: Path) -> dict[str, Any]:
    control = cv2.VideoCapture(str(control_path))
    candidate = cv2.VideoCapture(str(candidate_path))
    if not control.isOpened() or not candidate.isOpened():
        control.release()
        candidate.release()
        return {
            "passed": False,
            "checks": {"videos_opened": False},
            "measurements": {},
            "thresholds": dict(DENSEPOSE_THRESHOLDS),
        }
    control_fps = float(control.get(cv2.CAP_PROP_FPS) or 0.0)
    candidate_fps = float(candidate.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = 0
    shapes_match = True
    ssim_values: list[float] = []
    normalized_errors: list[float] = []
    while True:
        control_ok, control_frame = control.read()
        candidate_ok, candidate_frame = candidate.read()
        if not control_ok or not candidate_ok:
            both_ended = not control_ok and not candidate_ok
            break
        frame_count += 1
        if control_frame.shape != candidate_frame.shape:
            shapes_match = False
            continue
        ssim_values.append(_ssim(control_frame, candidate_frame))
        normalized_errors.append(
            float(
                np.mean(
                    np.abs(control_frame.astype(np.float32) - candidate_frame.astype(np.float32))
                )
                / 255.0
            )
        )
    control.release()
    candidate.release()
    measurements = {
        "decoded_frame_count": frame_count,
        "both_streams_ended_together": both_ended,
        "shapes_match": shapes_match,
        "fps_abs_delta": abs(control_fps - candidate_fps),
        "ssim_mean": float(statistics.fmean(ssim_values)) if ssim_values else None,
        "ssim_p05": float(np.percentile(np.asarray(ssim_values), 5)) if ssim_values else None,
        "normalized_mae": (
            float(statistics.fmean(normalized_errors)) if normalized_errors else None
        ),
    }
    checks = {
        "videos_opened": True,
        "frame_count_exact": frame_count == EXPECTED_FRAME_COUNT,
        "streams_ended_together": both_ended,
        "dimensions_exact": shapes_match,
        "fps_match": measurements["fps_abs_delta"] <= DENSEPOSE_THRESHOLDS["qa_fps_abs_delta_max"],
        "ssim_p05": measurements["ssim_p05"] is not None
        and measurements["ssim_p05"] >= DENSEPOSE_THRESHOLDS["qa_ssim_p05_min"],
        "normalized_mae": measurements["normalized_mae"] is not None
        and measurements["normalized_mae"] <= DENSEPOSE_THRESHOLDS["qa_normalized_mae_max"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "measurements": measurements,
        "thresholds": {
            "expected_frame_count": EXPECTED_FRAME_COUNT,
            "fps_abs_delta_max": DENSEPOSE_THRESHOLDS["qa_fps_abs_delta_max"],
            "ssim_p05_min": DENSEPOSE_THRESHOLDS["qa_ssim_p05_min"],
            "normalized_mae_max": DENSEPOSE_THRESHOLDS["qa_normalized_mae_max"],
        },
    }


def _materialize_run(
    run_dir: Path,
    *,
    source_path: Path,
    runner_mask_path: Path,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    source_target = run_dir / "source_segment.mp4"
    mask_target = run_dir / "runner_mask.mp4"
    shutil.copyfile(source_path, source_target)
    shutil.copyfile(runner_mask_path, mask_target)
    write_json(
        run_dir / "cv_run_manifest.json",
        {
            "version": 1,
            "candidate_id": "densepose-batch-benchmark",
            "paths": {
                "source_segment": str(source_target),
                "runner_mask": str(mask_target),
                "densepose": str(run_dir / "densepose.jsonl"),
                "qa_overlay": str(run_dir / "qa_overlay.mp4"),
            },
            "stages": {"densepose": {"status": "pending"}},
        },
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


_TIMED_PHASES = frozenset(
    {
        "loading_model",
        "decoding",
        "running_densepose",
        "encoding",
        "writing_outputs",
        "completed",
    }
)


def _record_phase_transition(
    event: dict[str, Any],
    *,
    cuda: CudaRuntime,
    timestamps: dict[str, float],
    clock: Callable[[], float] = time.perf_counter,
) -> None:
    phase = str(event.get("phase") or "")
    if phase not in _TIMED_PHASES or phase in timestamps:
        return
    cuda.synchronize()
    timestamps[phase] = clock()


def _phase_durations(timestamps: dict[str, float]) -> dict[str, float]:
    pairs = {
        "model_cache_lookup": ("loading_model", "decoding"),
        "decode_open": ("decoding", "running_densepose"),
        "inference_loop": ("running_densepose", "encoding"),
        "browser_encode": ("encoding", "writing_outputs"),
        "write_outputs": ("writing_outputs", "completed"),
    }
    return {
        name: round(max(0.0, timestamps[end] - timestamps[start]), 6)
        for name, (start, end) in pairs.items()
        if start in timestamps and end in timestamps
    }


def _run_once(
    *,
    root: Path,
    name: str,
    source_path: Path,
    runner_mask_path: Path,
    runtime_kwargs: dict[str, Any],
    batch_size: int,
    cuda: CudaRuntime,
    capture_evidence: bool,
) -> BenchmarkRun:
    run_dir = root / name
    _materialize_run(
        run_dir,
        source_path=source_path,
        runner_mask_path=runner_mask_path,
    )
    phase_timestamps: dict[str, float] = {}
    collector = LabelEvidenceCollector() if capture_evidence else None
    cuda.synchronize()
    cuda.reset_peak_memory_stats()
    started_at = time.perf_counter()
    result = run_densepose(
        run_dir=run_dir,
        batch_size=batch_size,
        write_qa_overlay=True,
        progress_callback=lambda event: _record_phase_transition(
            event,
            cuda=cuda,
            timestamps=phase_timestamps,
        ),
        benchmark_evidence_callback=collector,
        **runtime_kwargs,
    )
    cuda.synchronize()
    wall_seconds = time.perf_counter() - started_at
    if result.get("status") != "complete":
        raise RuntimeError("DensePose benchmark run did not complete")
    inference_settings = result.get("inference_settings") or {}
    if inference_settings.get("batch_size") != batch_size:
        raise RuntimeError("DensePose benchmark did not apply the requested batch size")
    rows = _read_jsonl(run_dir / "densepose.jsonl")
    if collector is not None and len(collector.frames) != len(rows):
        raise RuntimeError("DensePose benchmark label evidence is incomplete")
    sample = {
        "wall_seconds": round(wall_seconds, 6),
        "reported_seconds": round(float(result["elapsed_seconds"]), 6),
        "milliseconds_per_frame": round(wall_seconds * 1000.0 / max(len(rows), 1), 6),
        "phase_seconds": _phase_durations(phase_timestamps),
        "peak_cuda_allocated_bytes": cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": cuda.max_memory_reserved(),
        "host_rss_bytes": _current_rss_bytes(),
        "usable_frames": int(result.get("usable_frames") or 0),
        "frame_count": len(rows),
    }
    return BenchmarkRun(
        batch_size=batch_size,
        sample=sample,
        rows=rows,
        labels=collector,
        qa_overlay_path=run_dir / "qa_overlay.mp4",
    )


def _round_orders(batch_sizes: Sequence[int]) -> list[list[int]]:
    preferred = (
        (1, 2, 4, 8),
        (4, 8, 1, 2),
        (8, 4, 2, 1),
    )
    selected = set(batch_sizes)
    return [[value for value in order if value in selected] for order in preferred]


def _median_sample(samples: list[dict[str, Any]]) -> dict[str, Any]:
    inference_values = [
        float(sample["phase_seconds"]["inference_loop"])
        for sample in samples
        if "inference_loop" in sample["phase_seconds"]
    ]
    return {
        "sample_count": len(samples),
        "median_wall_seconds": round(
            float(statistics.median(float(sample["wall_seconds"]) for sample in samples)),
            6,
        ),
        "median_milliseconds_per_frame": round(
            float(statistics.median(float(sample["milliseconds_per_frame"]) for sample in samples)),
            6,
        ),
        "median_inference_loop_seconds": (
            round(float(statistics.median(inference_values)), 6) if inference_values else None
        ),
        "max_peak_cuda_allocated_bytes": max(
            int(sample["peak_cuda_allocated_bytes"]) for sample in samples
        ),
        "max_peak_cuda_reserved_bytes": max(
            int(sample["peak_cuda_reserved_bytes"]) for sample in samples
        ),
        "max_host_rss_bytes": max(int(sample["host_rss_bytes"]) for sample in samples),
        "samples": samples,
    }


def _safe_identity(value: str, pattern: str) -> str | None:
    import re

    stripped = value.strip()
    return stripped if re.fullmatch(pattern, stripped) else None


def runtime_identity() -> dict[str, str]:
    processor_version = _safe_identity(
        os.getenv("WHODOIRUNLIKE_PROCESSOR_VERSION", ""),
        r"[0-9a-f]{40}",
    )
    image_digest = _safe_identity(
        os.getenv("WHODOIRUNLIKE_BENCHMARK_IMAGE_DIGEST", ""),
        r"sha256:[0-9a-f]{64}",
    )
    missing = [
        name
        for name, value in (
            ("WHODOIRUNLIKE_PROCESSOR_VERSION", processor_version),
            ("WHODOIRUNLIKE_BENCHMARK_IMAGE_DIGEST", image_digest),
        )
        if value is None
    ]
    if missing:
        raise BenchmarkContractError(
            "DensePose benchmark runtime identity is not pinned: " + ", ".join(missing)
        )
    return {
        "processor_version": processor_version,
        "image_digest": image_digest,
    }


def evaluate_performance(
    aggregates: dict[str, dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
    *,
    batch_sizes: Sequence[int],
) -> dict[str, Any]:
    control = aggregates["1"]
    control_inference = _finite_float(control.get("median_inference_loop_seconds"))
    control_reserved = int(control.get("max_peak_cuda_reserved_bytes") or 0)
    max_absolute_reserved = int(A40_MEMORY_BYTES * MAX_RESERVED_MEMORY_FRACTION)
    gates: dict[str, Any] = {}
    eligible: list[int] = []
    for batch_size in batch_sizes:
        if batch_size == 1:
            continue
        aggregate = aggregates[str(batch_size)]
        candidate_inference = _finite_float(aggregate.get("median_inference_loop_seconds"))
        speedup = (
            1.0 - candidate_inference / control_inference
            if control_inference and candidate_inference is not None
            else None
        )
        candidate_reserved = int(aggregate.get("max_peak_cuda_reserved_bytes") or 0)
        checks = {
            "parity_passed": bool(comparisons[str(batch_size)].get("passed")),
            "minimum_inference_speedup": speedup is not None and speedup >= MIN_INFERENCE_SPEEDUP,
            "reserved_memory_below_a40_limit": candidate_reserved <= max_absolute_reserved,
            "reserved_memory_delta_bounded": candidate_reserved
            <= control_reserved + MAX_RESERVED_MEMORY_DELTA_BYTES,
        }
        passed = all(checks.values())
        if passed:
            eligible.append(batch_size)
        gates[str(batch_size)] = {
            "passed": passed,
            "checks": checks,
            "measurements": {
                "inference_speedup": round(speedup, 6) if speedup is not None else None,
                "peak_reserved_bytes": candidate_reserved,
                "peak_reserved_delta_vs_batch1_bytes": candidate_reserved - control_reserved,
            },
        }

    selected: int | None = None
    fastest_seconds: float | None = None
    if eligible:
        fastest_seconds = min(
            float(aggregates[str(batch_size)]["median_inference_loop_seconds"])
            for batch_size in eligible
        )
        near_fastest = [
            batch_size
            for batch_size in eligible
            if float(aggregates[str(batch_size)]["median_inference_loop_seconds"])
            <= fastest_seconds * (1.0 + FASTEST_WINDOW_FRACTION)
        ]
        selected = min(near_fastest)
    return {
        "selected_batch_size": selected,
        "eligible_batch_sizes": eligible,
        "fastest_inference_seconds": round(fastest_seconds, 6)
        if fastest_seconds is not None
        else None,
        "gates": gates,
        "thresholds": {
            "minimum_inference_speedup": MIN_INFERENCE_SPEEDUP,
            "maximum_reserved_memory_bytes": max_absolute_reserved,
            "maximum_reserved_memory_delta_bytes": MAX_RESERVED_MEMORY_DELTA_BYTES,
            "fastest_window_fraction": FASTEST_WINDOW_FRACTION,
        },
    }


def run_benchmark(request: BenchmarkRequest) -> dict[str, Any]:
    identity = runtime_identity()
    profile_id = benchmark_profile()
    runtime_kwargs = _densepose_runtime_kwargs()
    runtime_kwargs.pop("batch_size", None)
    validate_runtime_configuration(runtime_kwargs, profile_id=profile_id)
    cuda = load_cuda_runtime()
    if cuda.gpu_name != EXPECTED_GPU_NAME:
        raise BenchmarkContractError(
            f"DensePose benchmark requires {EXPECTED_GPU_NAME}; observed another GPU"
        )

    with tempfile.TemporaryDirectory(prefix="wdirl-densepose-batch-benchmark-") as temp_name:
        temp_root = Path(temp_name)
        runner_mask_path = temp_root / "runner_mask.mp4"
        runner_mask_path.write_bytes(request.runner_mask_bytes)
        fixture = verify_fixture(SOURCE_PATH, runner_mask_path)

        clear_densepose_backend_cache()
        cuda.synchronize()
        model_started_at = time.perf_counter()
        load_densepose_backend(
            config_path=runtime_kwargs["config_path"],
            weights_path=runtime_kwargs["weights_path"],
            device=runtime_kwargs["device"],
            input_min_size_test=runtime_kwargs["input_min_size_test"],
            input_max_size_test=runtime_kwargs["input_max_size_test"],
        )
        cuda.synchronize()
        model_load_seconds = time.perf_counter() - model_started_at

        for batch_size in request.batch_sizes:
            _run_once(
                root=temp_root,
                name=f"warmup-b{batch_size}",
                source_path=SOURCE_PATH,
                runner_mask_path=runner_mask_path,
                runtime_kwargs=runtime_kwargs,
                batch_size=batch_size,
                cuda=cuda,
                capture_evidence=False,
            )

        samples: dict[int, list[dict[str, Any]]] = {
            batch_size: [] for batch_size in request.batch_sizes
        }
        evidence_runs: dict[int, BenchmarkRun] = {}
        for round_index, order in enumerate(_round_orders(request.batch_sizes)):
            for batch_size in order:
                run = _run_once(
                    root=temp_root,
                    name=f"measured-r{round_index}-b{batch_size}",
                    source_path=SOURCE_PATH,
                    runner_mask_path=runner_mask_path,
                    runtime_kwargs=runtime_kwargs,
                    batch_size=batch_size,
                    cuda=cuda,
                    capture_evidence=batch_size not in evidence_runs,
                )
                samples[batch_size].append(run.sample)
                if run.labels is not None:
                    evidence_runs[batch_size] = run

        if any(len(values) != MEASURED_REPETITIONS for values in samples.values()):
            raise RuntimeError("DensePose benchmark timing matrix is incomplete")
        if set(evidence_runs) != set(request.batch_sizes):
            raise RuntimeError("DensePose benchmark evidence matrix is incomplete")

        control = evidence_runs[1]
        comparisons: dict[str, Any] = {}
        evidence: dict[str, Any] = {}
        for batch_size in request.batch_sizes:
            candidate = evidence_runs[batch_size]
            assert control.labels is not None and candidate.labels is not None
            row_gate = compare_densepose_rows(control.rows, candidate.rows)
            label_gate = compare_label_evidence(control.labels, candidate.labels)
            qa_gate = compare_qa_videos(control.qa_overlay_path, candidate.qa_overlay_path)
            comparisons[str(batch_size)] = {
                "passed": row_gate["passed"] and label_gate["passed"] and qa_gate["passed"],
                "rows": row_gate,
                "raw_labels": label_gate,
                "qa_overlay": qa_gate,
            }
            evidence[str(batch_size)] = candidate.labels.summary()

        aggregates = {
            str(batch_size): _median_sample(values) for batch_size, values in samples.items()
        }
        control_inference = aggregates["1"]["median_inference_loop_seconds"]
        for batch_size in request.batch_sizes:
            candidate_inference = aggregates[str(batch_size)]["median_inference_loop_seconds"]
            speedup = None
            if control_inference and candidate_inference is not None:
                speedup = 1.0 - float(candidate_inference) / float(control_inference)
            aggregates[str(batch_size)]["inference_speedup_vs_batch1"] = (
                round(speedup, 6) if speedup is not None else None
            )

        performance = evaluate_performance(
            aggregates,
            comparisons,
            batch_sizes=request.batch_sizes,
        )

        result = {
            "type": BENCHMARK_RESULT_TYPE,
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "status": "complete",
            "passed": all(comparison["passed"] for comparison in comparisons.values()),
            "fixture": fixture,
            "request": {
                "sha256": request.request_sha256,
                "bytes": request.request_bytes,
                "batch_sizes": list(request.batch_sizes),
            },
            "runtime": {
                **identity,
                "benchmark_profile": profile_id,
                "gpu_name": cuda.gpu_name,
                "torch_version": cuda.torch_version,
                "cuda_version": cuda.cuda_version,
                "opencv_version": cv2.__version__,
                "numpy_version": np.__version__,
                "model_load_seconds": round(model_load_seconds, 6),
                "warmup_runs_per_batch": 1,
                "measured_repetitions": MEASURED_REPETITIONS,
                "round_orders": _round_orders(request.batch_sizes),
                "densepose": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in runtime_kwargs.items()
                    if key not in {"weights_path", "config_path"}
                },
            },
            "timings": aggregates,
            "comparisons": comparisons,
            "performance": performance,
            "label_evidence": evidence,
        }
        return ensure_bounded_response(result)
