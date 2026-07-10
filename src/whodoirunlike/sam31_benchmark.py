from __future__ import annotations

import base64
from dataclasses import dataclass
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import tempfile
import threading
import time
from typing import Any

import cv2
import numpy as np

from whodoirunlike.mask_artifacts import iter_mask_video, mask_iou, write_masks_jsonl_from_video
from whodoirunlike.sam2_runner import (
    extract_video_frames,
    inspect_video,
    load_prompt,
    write_mask_outputs,
)
from whodoirunlike.sam31_gpu_runner import (
    DEFAULT_SAM31_GPU_OBJ_ID,
    SAM31_GPU_STRATEGY_PRESEED_SINGLE_PASS,
    SAM31_GPU_STRATEGY_PRODUCTION_CONTROL,
    _collect_sam31_masks,
    _configure_interactive_tracker_for_user_prompt,
    _filter_masks_to_track_boxes,
    _load_identity_track_boxes,
    _patch_multiplex_init_state_kwargs,
    _synchronize_cuda,
)


BENCHMARK_SCHEMA_VERSION = 1
BENCHMARK_TYPE = "sam31_benchmark"
BENCHMARK_FIXTURE_ID = "cole-8.68s-260f-v1"
BENCHMARK_SOURCE_SHA256 = "a8146591119c5439cc01168df63fa6144a7a55ff6817726946e1e8f5bc381617"
BENCHMARK_SAM_GIT_SHA = "5dd401d1c5c1d5c3eedff06d41b77af824517619"
MAX_RESPONSE_BYTES = 256 * 1024


@dataclass(frozen=True)
class AssetSpec:
    encoding: str
    sha256: str
    max_decoded_bytes: int


ASSET_SPECS = {
    "person_prompt_json": AssetSpec(
        encoding="base64",
        sha256="66d0760138febcd8fee2b7d944aedd68bd7ada665cc47c7253de376cf065e26c",
        max_decoded_bytes=16 * 1024,
    ),
    "tracklets_jsonl": AssetSpec(
        encoding="gzip+base64",
        sha256="47dea72891c0de6b95e7a255506c1afac1f7ee6525c2d1afe4589544fd760010",
        max_decoded_bytes=2 * 1024 * 1024,
    ),
    "baseline_runner_mask_mp4": AssetSpec(
        encoding="base64",
        sha256="0edf35fb0837d4083f0f73103631b10972c69347cc13ffccafa2cb78634c443f",
        max_decoded_bytes=2 * 1024 * 1024,
    ),
}


@dataclass(frozen=True)
class VariantConfig:
    strategy: str
    resource: str = "video"
    offload_video_to_cpu: bool = False
    max_num_objects: int = 16
    compile: bool = False
    warm_up: bool = False
    render_outputs: bool = True


VARIANTS = {
    "production_control": VariantConfig(strategy=SAM31_GPU_STRATEGY_PRODUCTION_CONTROL),
    "preseed_single_pass": VariantConfig(strategy=SAM31_GPU_STRATEGY_PRESEED_SINGLE_PASS),
    "preseed_single_pass_frame_dir": VariantConfig(
        strategy=SAM31_GPU_STRATEGY_PRESEED_SINGLE_PASS,
        resource="frame_dir",
    ),
    "preseed_single_pass_offload_video_cpu": VariantConfig(
        strategy=SAM31_GPU_STRATEGY_PRESEED_SINGLE_PASS,
        offload_video_to_cpu=True,
    ),
    "preseed_single_pass_max_objects_1": VariantConfig(
        strategy=SAM31_GPU_STRATEGY_PRESEED_SINGLE_PASS,
        max_num_objects=1,
    ),
}


_BENCHMARK_LOCK = threading.Lock()
_PREDICTOR_CACHE: tuple[tuple[Any, ...], Any, dict[str, Any]] | None = None


def _round(value: float | int | None, digits: int = 6) -> float | None:
    return round(float(value), digits) if value is not None else None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_gzip_bounded(data: bytes, *, max_bytes: int) -> bytes:
    import io

    with gzip.GzipFile(fileobj=io.BytesIO(data), mode="rb") as stream:
        decoded = stream.read(max_bytes + 1)
    if len(decoded) > max_bytes:
        raise ValueError("Compressed benchmark asset exceeds its decoded size limit.")
    return decoded


def _decode_asset(name: str, payload: Any) -> bytes:
    spec = ASSET_SPECS[name]
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
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Benchmark asset {name} is not valid base64.") from exc
    if spec.encoding == "gzip+base64":
        try:
            raw = _read_gzip_bounded(raw, max_bytes=spec.max_decoded_bytes)
        except (OSError, EOFError) as exc:
            raise ValueError(f"Benchmark asset {name} is not valid gzip data.") from exc
    elif len(raw) > spec.max_decoded_bytes:
        raise ValueError(f"Benchmark asset {name} exceeds its decoded size limit.")
    if _sha256(raw) != spec.sha256:
        raise ValueError(f"Benchmark asset {name} failed SHA-256 verification.")
    return raw


def _fixture_source_path() -> Path:
    configured = os.getenv("WHODOIRUNLIKE_SAM31_BENCHMARK_SOURCE", "").strip()
    candidates = [Path(configured)] if configured else []
    repository_root = Path(__file__).resolve().parents[2]
    candidates.extend(
        [
            repository_root / "site/public/assets/demos/cole-source.mp4",
            Path("/app/site/public/assets/demos/cole-source.mp4"),
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            data = candidate.read_bytes()
            if _sha256(data) != BENCHMARK_SOURCE_SHA256:
                raise ValueError("Baked benchmark source failed SHA-256 verification.")
            return candidate
    raise FileNotFoundError("The baked SAM 3.1 benchmark source clip is unavailable.")


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _cuda_memory(torch_module: Any) -> dict[str, Any]:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        return {}
    return {
        "allocated_gib": _round(cuda.memory_allocated() / (1024**3), 4),
        "reserved_gib": _round(cuda.memory_reserved() / (1024**3), 4),
        "peak_allocated_gib": _round(cuda.max_memory_allocated() / (1024**3), 4),
        "peak_reserved_gib": _round(cuda.max_memory_reserved() / (1024**3), 4),
    }


def _runtime_metadata(torch_module: Any) -> dict[str, Any]:
    cuda = getattr(torch_module, "cuda", None)
    cuda_available = bool(cuda is not None and cuda.is_available())
    result: dict[str, Any] = {
        "processor_version": os.getenv("WHODOIRUNLIKE_PROCESSOR_VERSION", "unknown"),
        "sam_git_sha": BENCHMARK_SAM_GIT_SHA,
        "torch_version": getattr(torch_module, "__version__", "unknown"),
        "torch_cuda_version": getattr(getattr(torch_module, "version", None), "cuda", None),
        "cuda_available": cuda_available,
    }
    if cuda_available:
        properties = cuda.get_device_properties(0)
        result.update(
            {
                "gpu_name": cuda.get_device_name(0),
                "gpu_total_memory_gib": _round(properties.total_memory / (1024**3), 3),
                "compute_capability": f"{properties.major}.{properties.minor}",
            }
        )
    return result


class _NvmlSampler:
    def __init__(self, interval_seconds: float = 0.25) -> None:
        self.interval_seconds = interval_seconds
        self._samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pynvml: Any = None
        self._handle: Any = None
        self.error: str | None = None

    def start(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as exc:  # pragma: no cover - depends on RunPod host driver
            self.error = type(exc).__name__
            return
        self._thread = threading.Thread(target=self._run, name="sam31-nvml-sampler", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                utilization = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                memory = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                try:
                    power_watts = self._pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                except Exception:
                    power_watts = 0.0
                self._samples.append(
                    {
                        "gpu_utilization_percent": float(utilization.gpu),
                        "memory_used_gib": float(memory.used) / (1024**3),
                        "power_watts": float(power_watts),
                    }
                )
            except Exception as exc:  # pragma: no cover - depends on RunPod host driver
                self.error = type(exc).__name__
                return

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 4))
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
        gpu = [sample["gpu_utilization_percent"] for sample in self._samples]
        memory = [sample["memory_used_gib"] for sample in self._samples]
        power = [sample["power_watts"] for sample in self._samples]
        return {
            "sample_count": len(self._samples),
            "interval_seconds": self.interval_seconds,
            "gpu_utilization_percent": {
                "mean": _round(statistics.fmean(gpu), 2) if gpu else None,
                "p50": _round(_percentile(gpu, 50), 2),
                "p95": _round(_percentile(gpu, 95), 2),
                "max": _round(max(gpu), 2) if gpu else None,
            },
            "memory_used_gib_max": _round(max(memory), 4) if memory else None,
            "power_watts_max": _round(max(power), 2) if power else None,
            "error": self.error,
        }


def _predictor_key(config: VariantConfig, checkpoint_path: str | None) -> tuple[Any, ...]:
    return (
        checkpoint_path,
        config.max_num_objects,
        config.compile,
        config.warm_up,
        False,
        0.5,
    )


def _get_predictor(config: VariantConfig, *, torch_module: Any) -> tuple[Any, dict[str, Any]]:
    global _PREDICTOR_CACHE

    from sam3.model_builder import build_sam3_multiplex_video_predictor

    checkpoint_path = os.getenv("WHODOIRUNLIKE_SAM31_GPU_CHECKPOINT", "").strip() or None
    key = _predictor_key(config, checkpoint_path)
    if _PREDICTOR_CACHE is not None and _PREDICTOR_CACHE[0] == key:
        return _PREDICTOR_CACHE[1], {
            "cache_hit": True,
            "model_build": 0.0,
            "tracker_config": _PREDICTOR_CACHE[2],
        }

    if _PREDICTOR_CACHE is not None:
        previous_predictor = _PREDICTOR_CACHE[1]
        bf16_context = getattr(previous_predictor, "bf16_context", None)
        context_exit = getattr(bf16_context, "__exit__", None)
        if callable(context_exit):
            context_exit(None, None, None)
        del previous_predictor
    _PREDICTOR_CACHE = None
    gc.collect()
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
        torch_module.cuda.reset_peak_memory_stats()
    _synchronize_cuda(torch_module)
    started_at = time.perf_counter()
    predictor = build_sam3_multiplex_video_predictor(
        checkpoint_path=checkpoint_path,
        max_num_objects=config.max_num_objects,
        multiplex_count=16,
        use_fa3=False,
        compile=config.compile,
        warm_up=config.warm_up,
        default_output_prob_thresh=0.5,
        async_loading_frames=True,
    )
    _patch_multiplex_init_state_kwargs(predictor)
    tracker_config = _configure_interactive_tracker_for_user_prompt(predictor)
    _synchronize_cuda(torch_module)
    model_build = time.perf_counter() - started_at
    _PREDICTOR_CACHE = (key, predictor, tracker_config)
    return predictor, {
        "cache_hit": False,
        "model_build": _round(model_build),
        "tracker_config": tracker_config,
    }


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


def _centroid_in_box(centroid: tuple[float, float] | None, box: np.ndarray | None) -> bool:
    if centroid is None or box is None:
        return False
    x, y = centroid
    x1, y1, x2, y2 = [float(value) for value in box]
    return x1 <= x <= x2 and y1 <= y <= y2


def _leakage_outside_track_box(mask: np.ndarray, box: np.ndarray, padding_ratio: float = 0.08) -> float:
    mask_area = int((mask > 0).sum())
    if mask_area == 0:
        return 1.0
    height, width = mask.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in box]
    pad_x = max(2.0, (x2 - x1) * padding_ratio)
    pad_y = max(2.0, (y2 - y1) * padding_ratio)
    left = max(0, int(np.floor(x1 - pad_x)))
    top = max(0, int(np.floor(y1 - pad_y)))
    right = min(width, int(np.ceil(x2 + pad_x)))
    bottom = min(height, int(np.ceil(y2 + pad_y)))
    inside_area = int((mask[top:bottom, left:right] > 0).sum())
    return float(max(0, mask_area - inside_area) / mask_area)


def _boundary_f1(mask_a: np.ndarray, mask_b: np.ndarray, tolerance_pixels: int = 2) -> float:
    kernel = np.ones((3, 3), dtype=np.uint8)
    edge_a = np.logical_xor(mask_a > 0, cv2.erode(mask_a, kernel) > 0)
    edge_b = np.logical_xor(mask_b > 0, cv2.erode(mask_b, kernel) > 0)
    count_a = int(edge_a.sum())
    count_b = int(edge_b.sum())
    if count_a == 0 and count_b == 0:
        return 1.0
    if count_a == 0 or count_b == 0:
        return 0.0
    tolerance_kernel = np.ones((tolerance_pixels * 2 + 1, tolerance_pixels * 2 + 1), np.uint8)
    near_a = cv2.dilate(edge_a.astype(np.uint8), tolerance_kernel) > 0
    near_b = cv2.dilate(edge_b.astype(np.uint8), tolerance_kernel) > 0
    precision = float(np.logical_and(edge_a, near_b).sum() / count_a)
    recall = float(np.logical_and(edge_b, near_a).sum() / count_b)
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def _mean_temporal_iou(masks: list[np.ndarray]) -> float | None:
    if len(masks) < 2:
        return None
    return float(statistics.fmean(mask_iou(previous, current) for previous, current in zip(masks, masks[1:])))


def compare_masks_to_production_baseline(
    candidate_masks: list[np.ndarray],
    reference_masks: list[np.ndarray],
    *,
    track_boxes: dict[int, np.ndarray],
) -> dict[str, Any]:
    if not reference_masks:
        raise ValueError("Production baseline mask video contains no frames.")
    height, width = reference_masks[0].shape[:2]
    normalized_candidates = [
        _binary_mask(candidate_masks[index] if index < len(candidate_masks) else None, height=height, width=width)
        for index in range(len(reference_masks))
    ]
    normalized_references = [
        _binary_mask(reference, height=height, width=width) for reference in reference_masks
    ]

    ious: list[float] = []
    dices: list[float] = []
    boundary_f1s: list[float] = []
    area_ratio_deltas: list[float] = []
    centroid_errors: list[float] = []
    candidate_target_hits: list[bool] = []
    reference_target_hits: list[bool] = []
    candidate_leakage: list[float] = []
    reference_leakage: list[float] = []
    diagonal = max(float(np.hypot(width, height)), 1.0)
    for frame_index, (candidate, reference) in enumerate(
        zip(normalized_candidates, normalized_references)
    ):
        ious.append(mask_iou(candidate, reference))
        dices.append(_dice(candidate, reference))
        boundary_f1s.append(_boundary_f1(candidate, reference))
        area_ratio_deltas.append(abs(float(candidate.mean()) - float(reference.mean())))
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
            candidate_leakage.append(_leakage_outside_track_box(candidate, track_box))
            reference_leakage.append(_leakage_outside_track_box(reference, track_box))

    candidate_temporal_iou = _mean_temporal_iou(normalized_candidates)
    reference_temporal_iou = _mean_temporal_iou(normalized_references)
    candidate_target_rate = (
        float(statistics.fmean(candidate_target_hits)) if candidate_target_hits else None
    )
    reference_target_rate = (
        float(statistics.fmean(reference_target_hits)) if reference_target_hits else None
    )
    identity_switches = sum(
        1
        for previous, current in zip(candidate_target_hits, candidate_target_hits[1:])
        if previous and not current
    )
    worst_frames = sorted(range(len(ious)), key=lambda index: ious[index])[:10]
    result = {
        "label": "agreement_vs_production_baseline",
        "baseline_is_lossy_mp4_not_ground_truth": True,
        "reference_frames": len(normalized_references),
        "candidate_frames": len(candidate_masks),
        "candidate_nonempty_frames": sum(int(mask.any()) for mask in normalized_candidates),
        "reference_nonempty_frames": sum(int(mask.any()) for mask in normalized_references),
        "iou": {
            "mean": _round(statistics.fmean(ious)),
            "median": _round(statistics.median(ious)),
            "p05": _round(_percentile(ious, 5)),
        },
        "dice_mean": _round(statistics.fmean(dices)),
        "boundary_f1_2px_mean": _round(statistics.fmean(boundary_f1s)),
        "absolute_area_ratio_delta_mean": _round(statistics.fmean(area_ratio_deltas)),
        "centroid_error_normalized_mean": (
            _round(statistics.fmean(centroid_errors)) if centroid_errors else None
        ),
        "temporal_iou": {
            "candidate_mean": _round(candidate_temporal_iou),
            "reference_mean": _round(reference_temporal_iou),
            "absolute_delta": (
                _round(abs(candidate_temporal_iou - reference_temporal_iou))
                if candidate_temporal_iou is not None and reference_temporal_iou is not None
                else None
            ),
        },
        "tracked_frames": len(candidate_target_hits),
        "target_box_centroid_rate": {
            "candidate": _round(candidate_target_rate),
            "reference": _round(reference_target_rate),
        },
        "leakage_outside_padded_track_box_mean": {
            "candidate": _round(statistics.fmean(candidate_leakage)) if candidate_leakage else None,
            "reference": _round(statistics.fmean(reference_leakage)) if reference_leakage else None,
        },
        "identity_switch_count": identity_switches,
        "worst_frame_indices": worst_frames,
    }
    temporal_delta = result["temporal_iou"]["absolute_delta"]
    candidate_leakage_mean = result["leakage_outside_padded_track_box_mean"]["candidate"]
    reference_leakage_mean = result["leakage_outside_padded_track_box_mean"]["reference"]
    result["strict_mask_agreement_gate"] = {
        "passed": bool(
            result["candidate_nonempty_frames"] == result["reference_frames"]
            and result["iou"]["mean"] is not None
            and result["iou"]["mean"] >= 0.95
            and result["iou"]["median"] is not None
            and result["iou"]["median"] >= 0.98
            and result["iou"]["p05"] is not None
            and result["iou"]["p05"] >= 0.9
            and temporal_delta is not None
            and temporal_delta <= 0.03
            and candidate_target_rate is not None
            and reference_target_rate is not None
            and candidate_target_rate >= reference_target_rate
            and identity_switches == 0
            and candidate_leakage_mean is not None
            and reference_leakage_mean is not None
            and candidate_leakage_mean <= reference_leakage_mean + 0.02
        ),
        "thresholds": {
            "candidate_nonempty_frames": result["reference_frames"],
            "iou_mean_min": 0.95,
            "iou_median_min": 0.98,
            "iou_p05_min": 0.9,
            "temporal_iou_absolute_delta_max": 0.03,
            "target_box_centroid_rate": "no_worse_than_reference",
            "identity_switch_count_max": 0,
            "leakage_outside_padded_track_box_mean": "reference_plus_0.02_max",
        },
    }
    return result


def _validate_request(payload: Any) -> tuple[str, dict[str, bytes]]:
    if not isinstance(payload, dict):
        raise ValueError("SAM 3.1 benchmark input must be an object.")
    if payload.get("type") != BENCHMARK_TYPE:
        raise ValueError("Unsupported benchmark request type.")
    if payload.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise ValueError("Unsupported SAM 3.1 benchmark schema version.")
    if payload.get("fixture_id") != BENCHMARK_FIXTURE_ID:
        raise ValueError("Unsupported SAM 3.1 benchmark fixture.")
    variant_id = payload.get("variant_id")
    if variant_id not in VARIANTS:
        raise ValueError("Unsupported SAM 3.1 benchmark variant.")
    assets = payload.get("assets")
    if not isinstance(assets, dict) or set(assets) != set(ASSET_SPECS):
        raise ValueError("SAM 3.1 benchmark requires the exact fixed comparison assets.")
    return str(variant_id), {name: _decode_asset(name, assets[name]) for name in ASSET_SPECS}


def _run_benchmark_locked(payload: dict[str, Any]) -> dict[str, Any]:
    import torch

    overall_started_at = time.perf_counter()
    variant_id, assets = _validate_request(payload)
    config = VARIANTS[variant_id]
    source_path = _fixture_source_path()
    runtime = _runtime_metadata(torch)
    if not runtime["cuda_available"]:
        raise RuntimeError("SAM 3.1 benchmark requires CUDA.")

    sampler = _NvmlSampler()
    sampler.start()
    timings: dict[str, float | None] = {}
    memory: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    try:
        with tempfile.TemporaryDirectory(prefix="wdirl-sam31-benchmark-") as temp_name:
            temp_dir = Path(temp_name)
            prompt_path = temp_dir / "person_prompt.json"
            tracklets_path = temp_dir / "tracklets.jsonl"
            baseline_mask_path = temp_dir / "baseline_runner_mask.mp4"
            prompt_path.write_bytes(assets["person_prompt_json"])
            tracklets_path.write_bytes(assets["tracklets_jsonl"])
            baseline_mask_path.write_bytes(assets["baseline_runner_mask_mp4"])

            preprocessing_started_at = time.perf_counter()
            video_meta = inspect_video(source_path)
            frame_dir = temp_dir / "frames"
            frame_paths = extract_video_frames(source_path, frame_dir, force=True)
            if (
                video_meta["width"] != 960
                or video_meta["height"] != 540
                or len(frame_paths) != 260
            ):
                raise ValueError("Baked benchmark source metadata does not match the fixed fixture.")
            prompt = load_prompt(prompt_path, video_meta["width"], video_meta["height"])
            track_boxes = _load_identity_track_boxes(
                paths={"tracklets_jsonl": str(tracklets_path), "tracklets": None},
                width=video_meta["width"],
                height=video_meta["height"],
            )
            timings["source_decode_and_frame_extract"] = _round(
                time.perf_counter() - preprocessing_started_at
            )
            baseline_decode_started_at = time.perf_counter()
            baseline_meta, baseline_masks = iter_mask_video(baseline_mask_path)
            if (
                baseline_meta["width"] != video_meta["width"]
                or baseline_meta["height"] != video_meta["height"]
                or len(baseline_masks) != len(frame_paths)
            ):
                raise ValueError("Production baseline mask metadata does not match the fixed fixture.")
            timings["quality_baseline_decode"] = _round(
                time.perf_counter() - baseline_decode_started_at
            )

            predictor, predictor_info = _get_predictor(config, torch_module=torch)
            timings["model_build"] = predictor_info["model_build"]
            diagnostics["predictor_cache_hit"] = predictor_info["cache_hit"]
            diagnostics["tracker_config"] = predictor_info["tracker_config"]
            memory["after_model_load"] = _cuda_memory(torch)

            torch.cuda.reset_peak_memory_stats()
            resource_path = frame_dir if config.resource == "frame_dir" else source_path
            inference_started_at = time.perf_counter()
            masks_by_frame, sam_diagnostics = _collect_sam31_masks(
                predictor=predictor,
                video_path=resource_path,
                prompt=prompt,
                width=video_meta["width"],
                height=video_meta["height"],
                frame_count=len(frame_paths),
                obj_id=DEFAULT_SAM31_GPU_OBJ_ID,
                track_boxes=track_boxes,
                strategy=config.strategy,
                offload_video_to_cpu=config.offload_video_to_cpu,
                strict_obj_id=True,
            )
            timings["sam_session_and_propagation"] = _round(
                time.perf_counter() - inference_started_at
            )
            for phase_name, elapsed in sam_diagnostics.get("timings_seconds", {}).items():
                timings[f"sam_{phase_name}"] = _round(elapsed)
            for index, propagation in enumerate(sam_diagnostics.get("propagation", []), start=1):
                label = str(propagation.get("pass") or "propagation")
                direction = str(propagation.get("direction") or "unknown")
                timings[f"sam_propagation_{index}_{label}_{direction}"] = _round(
                    propagation.get("elapsed_seconds")
                )
            memory["sam_session_and_propagation"] = _cuda_memory(torch)

            filter_started_at = time.perf_counter()
            masks_by_frame, identity_filter = _filter_masks_to_track_boxes(
                masks_by_frame,
                track_boxes,
            )
            timings["identity_filter"] = _round(time.perf_counter() - filter_started_at)

            output_paths = {
                "runner_mask": temp_dir / "candidate_runner_mask.mp4",
                "masked_runner": temp_dir / "candidate_masked_runner.mp4",
                "qa_overlay": temp_dir / "candidate_qa_overlay.mp4",
                "metadata": temp_dir / "candidate_runner_mask_metadata.jsonl",
                "masks_jsonl": temp_dir / "candidate_masks.jsonl",
            }
            if config.render_outputs:
                render_started_at = time.perf_counter()
                write_mask_outputs(
                    frame_paths=frame_paths,
                    masks_by_frame=masks_by_frame,
                    fps=video_meta["fps"],
                    runner_mask_path=output_paths["runner_mask"],
                    masked_runner_path=output_paths["masked_runner"],
                    qa_overlay_path=output_paths["qa_overlay"],
                    metadata_path=output_paths["metadata"],
                )
                timings["render_and_encode"] = _round(time.perf_counter() - render_started_at)
                rle_started_at = time.perf_counter()
                output_summary = write_masks_jsonl_from_video(
                    output_paths["runner_mask"],
                    output_paths["masks_jsonl"],
                )
                timings["rle_export"] = _round(time.perf_counter() - rle_started_at)
                _, candidate_masks = iter_mask_video(output_paths["runner_mask"])
            else:
                candidate_masks = [
                    _binary_mask(
                        masks_by_frame.get(index),
                        height=video_meta["height"],
                        width=video_meta["width"],
                    )
                    for index in range(len(frame_paths))
                ]
                output_summary = {
                    "frame_count": len(candidate_masks),
                    "nonempty_frames": sum(int(mask.any()) for mask in candidate_masks),
                }

            quality_started_at = time.perf_counter()
            quality = compare_masks_to_production_baseline(
                candidate_masks,
                baseline_masks,
                track_boxes=track_boxes,
            )
            timings["quality_compare"] = _round(time.perf_counter() - quality_started_at)
            timings["measured_mask_stage"] = _round(
                sum(
                    float(timings.get(name) or 0.0)
                    for name in (
                        "source_decode_and_frame_extract",
                        "model_build",
                        "sam_session_and_propagation",
                        "identity_filter",
                        "render_and_encode",
                        "rle_export",
                    )
                )
            )
            timings["total_handler"] = _round(time.perf_counter() - overall_started_at)

            propagation_seconds = sum(
                float(item.get("elapsed_seconds") or 0.0)
                for item in sam_diagnostics.get("propagation", [])
            )
            result: dict[str, Any] = {
                "schema_version": BENCHMARK_SCHEMA_VERSION,
                "fixture": {
                    "id": BENCHMARK_FIXTURE_ID,
                    "source_sha256": BENCHMARK_SOURCE_SHA256,
                    "duration_seconds": _round(
                        len(frame_paths) / max(float(video_meta["fps"]), 1e-9)
                    ),
                    "frame_count": len(frame_paths),
                    "width": video_meta["width"],
                    "height": video_meta["height"],
                    "fps": _round(video_meta["fps"]),
                    "tracked_frame_count": len(track_boxes),
                    "production_baseline_frame_count": baseline_meta["frame_count"],
                },
                "variant_id": variant_id,
                "effective_config": {
                    "strategy": config.strategy,
                    "resource": config.resource,
                    "offload_video_to_cpu": config.offload_video_to_cpu,
                    "offload_state_to_cpu": False,
                    "max_num_objects": config.max_num_objects,
                    "multiplex_count": 16,
                    "compile": config.compile,
                    "warm_up": config.warm_up,
                    "use_fa3": False,
                    "strict_object_id": True,
                    "render_outputs": config.render_outputs,
                },
                "runtime": runtime,
                "timings_seconds": timings,
                "throughput": {
                    "propagation_frames_per_second": _round(
                        len(frame_paths) / propagation_seconds if propagation_seconds else None,
                        3,
                    ),
                    "measured_mask_stage_frames_per_second": _round(
                        len(frame_paths) / float(timings["measured_mask_stage"])
                        if timings["measured_mask_stage"]
                        else None,
                        3,
                    ),
                    "propagation_response_count": sum(
                        int(item.get("responses") or 0)
                        for item in sam_diagnostics.get("propagation", [])
                    ),
                },
                "memory": memory,
                "quality_vs_production_baseline": quality,
                "diagnostics": {
                    **diagnostics,
                    "sam": sam_diagnostics,
                    "identity_filter": identity_filter,
                    "output_summary": {
                        "frame_count": output_summary.get("frame_count"),
                        "nonempty_frames": output_summary.get("nonempty_frames"),
                        "mean_temporal_iou": output_summary.get("mean_temporal_iou"),
                    },
                },
            }
    finally:
        memory["nvml"] = sampler.stop()

    result["memory"] = memory
    encoded_result = json.dumps(result, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if len(encoded_result) > MAX_RESPONSE_BYTES:
        raise RuntimeError("SAM 3.1 benchmark response exceeded its fixed size limit.")
    result["response_bytes"] = len(encoded_result)
    return result


def run_benchmark(payload: dict[str, Any]) -> dict[str, Any]:
    with _BENCHMARK_LOCK:
        return _run_benchmark_locked(payload)
