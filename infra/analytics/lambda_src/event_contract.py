from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from typing import Any


SCHEMA_VERSION = 1
EVENT_TYPES = frozenset(
    {
        "attempt_started",
        "stage_started",
        "span_started",
        "progress_sampled",
        "span_completed",
        "span_failed",
        "stage_completed",
        "stage_failed",
        "result_ready",
        "analysis_completed",
        "attempt_completed",
        "attempt_failed",
    }
)
PIPELINE_STAGES = frozenset(
    {
        "source_ingest",
        "processor_enqueue",
        "processor_queue",
        "source_download",
        "run_preparation",
        "target_tracking",
        "runner_mask",
        "pose_sequence",
        "densepose_body_map",
        "fused_form_signal",
        "form_feature_compilation",
        "artifact_table_export",
        "quality_control",
        "artifact_publish",
        "result_ready",
        "analysis_complete",
    }
)
PROCESSING_SPANS = frozenset(
    {"model_load", "decode", "preprocess", "inference", "postprocess", "render", "encode", "write", "publish"}
)
EVENT_STATUSES = frozenset({"queued", "running", "complete", "failed"})
TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "event_id",
        "run_id",
        "attempt_id",
        "sequence",
        "event_type",
        "event_time",
        "stage",
        "span",
        "status",
        "elapsed_seconds",
        "progress",
        "input",
        "runtime",
        "resources",
        "measurements",
        "error",
        "attributes",
    }
)
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


class ContractError(ValueError):
    pass


def _object(value: Any, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ContractError(f"{field} must be an object")
    return value


def _optional_number(value: Any, field: str) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ContractError(f"{field} must be a finite number")
    return value


def _iso_datetime(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value or len(value) > 40:
        raise ContractError(f"{field} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError(f"{field} must be an ISO-8601 string") from exc
    if parsed.tzinfo is None:
        raise ContractError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def validate_event(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ContractError("event must be an object")
    unknown = sorted(set(payload) - TOP_LEVEL_FIELDS)
    if unknown:
        raise ContractError(f"unknown event fields: {', '.join(unknown)}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ContractError("schema_version must be 1")

    event_id = payload.get("event_id")
    attempt_id = payload.get("attempt_id")
    run_id = payload.get("run_id")
    if not isinstance(event_id, str) or not UUID_PATTERN.fullmatch(event_id):
        raise ContractError("event_id must be a UUID")
    if not isinstance(attempt_id, str) or not UUID_PATTERN.fullmatch(attempt_id):
        raise ContractError("attempt_id must be a UUID")
    if not isinstance(run_id, str) or not ID_PATTERN.fullmatch(run_id):
        raise ContractError("run_id has an invalid format")

    sequence = payload.get("sequence")
    if (
        isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or sequence < 1
        or sequence > 1_000_000_000
    ):
        raise ContractError("sequence must be an integer from 1 to 1000000000")
    event_type = payload.get("event_type")
    if event_type not in EVENT_TYPES:
        raise ContractError("event_type is not recognized")
    _iso_datetime(payload.get("event_time"), "event_time")

    stage = payload.get("stage")
    span = payload.get("span")
    if stage is not None and stage not in PIPELINE_STAGES:
        raise ContractError("stage is not recognized")
    if span is not None and span not in PROCESSING_SPANS:
        raise ContractError("span is not recognized")
    if event_type.startswith(("stage_", "span_")) or event_type == "progress_sampled":
        if stage is None:
            raise ContractError("stage is required for stage, span, and progress events")
    if event_type.startswith("span_") and span is None:
        raise ContractError("span is required for span events")
    if span is not None and stage is None:
        raise ContractError("span requires stage")

    status = payload.get("status")
    if status is not None and status not in EVENT_STATUSES:
        raise ContractError("status is not recognized")
    elapsed_seconds = _optional_number(payload.get("elapsed_seconds"), "elapsed_seconds")
    if elapsed_seconds is not None and not 0 <= elapsed_seconds <= 31_536_000:
        raise ContractError("elapsed_seconds must be between 0 and 31536000")
    for field in (
        "progress",
        "input",
        "runtime",
        "resources",
        "measurements",
        "error",
        "attributes",
    ):
        _object(payload.get(field), field)
    if event_type.endswith("failed") and not payload.get("error"):
        raise ContractError("failed events require sanitized error metadata")

    try:
        encoded = json.dumps(payload, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ContractError("event must contain only finite JSON values") from exc
    if len(encoded.encode("utf-8")) > 65_536:
        raise ContractError("event exceeds 65536 bytes")
    return dict(payload)


def _value(mapping: dict[str, Any], key: str) -> Any:
    value = mapping.get(key)
    return value if value is not None else None


def _megabytes(mapping: dict[str, Any], direct_key: str, bytes_key: str) -> Any:
    direct = _value(mapping, direct_key)
    if direct is not None:
        return direct
    byte_value = _value(mapping, bytes_key)
    if isinstance(byte_value, (int, float)) and not isinstance(byte_value, bool):
        return float(byte_value) / (1024 * 1024)
    return None


def flatten_event(event: dict[str, Any], *, ingested_at: str) -> dict[str, Any]:
    progress = _object(event.get("progress"), "progress")
    input_data = _object(event.get("input"), "input")
    runtime = _object(event.get("runtime"), "runtime")
    resources = _object(event.get("resources"), "resources")
    measurements = _object(event.get("measurements"), "measurements")
    error = _object(event.get("error"), "error")
    return {
        "schema_version": event["schema_version"],
        "event_id": event["event_id"],
        "run_id": event["run_id"],
        "attempt_id": event["attempt_id"],
        "sequence": event["sequence"],
        "event_type": event["event_type"],
        "event_time": event["event_time"],
        "ingested_at": ingested_at,
        "stage": event.get("stage"),
        "span": event.get("span"),
        "status": event.get("status"),
        "elapsed_seconds": event.get("elapsed_seconds"),
        "processed_frames": _value(progress, "processed_frames"),
        "total_frames": _value(progress, "total_frames"),
        "progress_percent": _value(progress, "percent"),
        "eta_seconds": _value(progress, "eta_seconds"),
        "clip_duration_seconds": _value(input_data, "duration_seconds"),
        "clip_frame_count": _value(input_data, "frame_count"),
        "clip_width": _value(input_data, "width"),
        "clip_height": _value(input_data, "height"),
        "clip_fps": _value(input_data, "fps"),
        "clip_size_bytes": _value(input_data, "size_bytes"),
        "duration_bucket": _value(input_data, "duration_bucket"),
        "resolution_bucket": _value(input_data, "resolution_bucket"),
        "environment": _value(runtime, "environment"),
        "service": _value(runtime, "service"),
        "execution_environment": _value(runtime, "execution_environment"),
        "attempt_number": _value(runtime, "attempt_number"),
        "processor_version": _value(runtime, "processor_version"),
        "backend": _value(runtime, "backend"),
        "model": _value(runtime, "model"),
        "gpu_type": _value(runtime, "gpu_type"),
        "cold_start": _value(runtime, "cold_start"),
        "cache_hit": _value(runtime, "cache_hit"),
        "rss_mb": _value(resources, "rss_mb"),
        "peak_rss_mb": _megabytes(resources, "peak_rss_mb", "peak_rss_bytes"),
        "cuda_allocated_mb": _megabytes(
            resources, "cuda_allocated_mb", "gpu_memory_allocated_bytes"
        ),
        "cuda_reserved_mb": _megabytes(
            resources, "cuda_reserved_mb", "gpu_memory_reserved_bytes"
        ),
        "cuda_peak_mb": _megabytes(
            resources, "cuda_peak_mb", "gpu_peak_memory_allocated_bytes"
        ),
        "gpu_utilization_pct": _value(resources, "gpu_utilization_pct"),
        "error_class": _value(error, "class") or _value(error, "exception_type"),
        "error_code": _value(error, "code") or _value(error, "category"),
        "artifact_type": _value(measurements, "artifact_type"),
        "artifact_size_bytes": _value(measurements, "bytes"),
        "milliseconds_per_frame": _value(measurements, "milliseconds_per_frame"),
        "timing_basis": _value(measurements, "timing_basis"),
        "measurements_json": json.dumps(event.get("measurements") or {}, separators=(",", ":"), sort_keys=True),
        "attributes_json": json.dumps(event.get("attributes") or {}, separators=(",", ":"), sort_keys=True),
    }


def event_partition(event_time: str) -> tuple[str, str]:
    parsed = _iso_datetime(event_time, "event_time")
    return parsed.strftime("%Y-%m-%d"), parsed.strftime("%H")
