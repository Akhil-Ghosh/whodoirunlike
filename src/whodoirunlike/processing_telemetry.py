from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import queue
import re
import resource
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


SCHEMA_VERSION = 1
PROCESSING_EVENTS_FILENAME = "processing_events.jsonl"
DEFAULT_PROGRESS_INTERVAL_SECONDS = 5.0
DEFAULT_DELIVERY_QUEUE_SIZE = 2048
# A Worker callback performs several durable R2 writes before returning. The
# sender runs off the processing thread, so allowing 10 seconds avoids treating
# slow successful writes as failures. During a real outage, three attempts plus
# backoff can occupy the sender for roughly 30 seconds before one event fails.
DEFAULT_DELIVERY_TIMEOUT_SECONDS = 10.0
DEFAULT_DELIVERY_ATTEMPTS = 3
DEFAULT_DELIVERY_BACKOFF_SECONDS = 0.1
DEFAULT_DELIVERY_FAILURE_THRESHOLD = 3
DEFAULT_DELIVERY_WORKERS = 6
MAX_DELIVERY_WORKERS = 8

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
    {
        "model_load",
        "decode",
        "preprocess",
        "inference",
        "postprocess",
        "render",
        "encode",
        "write",
        "publish",
    }
)
EVENT_STATUSES = frozenset({"queued", "running", "complete", "failed"})
_STAGE_EVENT_TYPES = frozenset(
    {
        "stage_started",
        "stage_completed",
        "stage_failed",
        "span_started",
        "span_completed",
        "span_failed",
        "progress_sampled",
    }
)
_SPAN_EVENT_TYPES = frozenset({"span_started", "span_completed", "span_failed"})
_FAILED_EVENT_TYPES = frozenset({"span_failed", "stage_failed", "attempt_failed"})

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_URL_PATTERN = re.compile(r"https?://[^\s)\]}>,;]+", re.IGNORECASE)
_BEARER_PATTERN = re.compile(r"(?i)Bearer\s+[A-Za-z0-9._~+/=-]+")
_SECRET_PATTERN = re.compile(
    r"(?i)(token|secret|password|authorization|api[_-]?key)(\s*[:=]\s*)([^\s,;]+)"
)
_ABSOLUTE_PATH_PATTERN = re.compile(r"(?<![\w.-])/(?:[^\s/:]+/)+[^\s,:;]+")
_SENSITIVE_KEYS = frozenset(
    {
        "authorization",
        "callback_base_url",
        "filename",
        "key",
        "password",
        "path",
        "prompt",
        "secret",
        "source_path",
        "source_url",
        "token",
        "traceback",
        "url",
    }
)

Clock = Callable[[], float]
WallClock = Callable[[], datetime]
EventSender = Callable[[dict[str, Any]], None]
ResourceSampler = Callable[[], Mapping[str, Any]]
BoundaryKind = Literal["stage", "span"]


def ensure_attempt_id(value: str | None) -> str:
    candidate = (value or "").strip()
    if candidate and _UUID_PATTERN.fullmatch(candidate):
        return candidate
    return str(uuid.uuid4())


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_event_time(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if result != result or result in {float("inf"), float("-inf")}:
        return None
    return result


def _safe_json_value(value: Any, *, key: str | None = None) -> Any:
    if key and key.lower() in _SENSITIVE_KEYS:
        return None
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return _safe_float(value)
    if isinstance(value, Path):
        return None
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            child_key = str(raw_key)
            cleaned = _safe_json_value(raw_value, key=child_key)
            if cleaned is not None:
                output[child_key] = cleaned
        return output
    if isinstance(value, (list, tuple, set, frozenset)):
        output_list = []
        for item in value:
            cleaned = _safe_json_value(item)
            if cleaned is not None:
                output_list.append(cleaned)
        return output_list
    return str(value)[:500]


def sanitize_error(error: BaseException | str) -> dict[str, Any]:
    if isinstance(error, BaseException):
        error_type = type(error).__name__
        raw_message = str(error)
        status_code = getattr(error, "code", None)
    else:
        error_type = "ProcessingError"
        raw_message = str(error)
        status_code = None

    message = raw_message.replace("\r", " ").replace("\n", " ")
    message = _BEARER_PATTERN.sub("Bearer <redacted>", message)
    message = _SECRET_PATTERN.sub(lambda match: f"{match.group(1)}{match.group(2)}<redacted>", message)
    message = _URL_PATTERN.sub("<url>", message)
    message = _ABSOLUTE_PATH_PATTERN.sub("<path>", message)
    message = " ".join(message.split())[:500] or error_type

    retryable = isinstance(error, (TimeoutError, ConnectionError, urllib.error.URLError))
    if isinstance(error, ValueError):
        category = "invalid_input"
    elif isinstance(error, TimeoutError):
        category = "timeout"
    elif isinstance(error, urllib.error.HTTPError):
        category = "upstream_http"
        retryable = int(error.code) == 429 or int(error.code) >= 500
    elif isinstance(error, (ConnectionError, urllib.error.URLError)):
        category = "upstream_connection"
    elif isinstance(error, OSError):
        category = "io"
    elif isinstance(error, RuntimeError):
        category = "processing"
    else:
        category = "internal"

    fingerprint_source = f"{error_type}:{category}:{message}".encode("utf-8", errors="replace")
    normalized_type = re.sub(r"(?<!^)(?=[A-Z])", "_", error_type).lower()
    payload: dict[str, Any] = {
        "class": error_type,
        "code": f"{category}.{normalized_type}",
        "exception_type": error_type,
        "category": category,
        "message": message,
        "retryable": retryable,
        "fingerprint": hashlib.sha256(fingerprint_source).hexdigest()[:16],
    }
    if isinstance(status_code, int):
        payload["http_status"] = status_code
    return payload


def default_runtime_metadata() -> dict[str, Any]:
    configured_version = os.getenv("WHODOIRUNLIKE_PROCESSOR_VERSION", "").strip()
    if not configured_version:
        try:
            configured_version = importlib.metadata.version("whodoirunlike")
        except importlib.metadata.PackageNotFoundError:
            configured_version = "unknown"
    return {
        "service": "whodoirunlike-processor",
        "environment": os.getenv("WHODOIRUNLIKE_ENVIRONMENT", "development").strip()
        or "development",
        "processor_version": configured_version[:100],
        "python_version": platform.python_version(),
        "platform": platform.system().lower(),
        "machine": platform.machine().lower(),
        "pid": os.getpid(),
    }


def _gpu_runtime_metadata() -> dict[str, Any]:
    torch_module = sys.modules.get("torch")
    try:
        cuda = getattr(torch_module, "cuda", None)
        if cuda is not None and bool(cuda.is_available()):
            return {"gpu_type": str(cuda.get_device_name(cuda.current_device()))[:200]}
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
    return {}


def _current_rss_bytes() -> int | None:
    if sys.platform.startswith("linux"):
        try:
            fields = Path("/proc/self/statm").read_text(encoding="utf-8").split()
            if len(fields) >= 2:
                return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError, IndexError):
            pass
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def default_resource_sample() -> dict[str, Any]:
    sample: dict[str, Any] = {"cpu_count": os.cpu_count()}
    current_rss = _current_rss_bytes()
    if current_rss is not None:
        sample.update(
            {
                "rss_bytes": current_rss,
                "rss_mb": round(current_rss / (1024 * 1024), 3),
            }
        )
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        peak_rss = int(usage.ru_maxrss)
        if sys.platform != "darwin":
            peak_rss *= 1024
        sample.update(
            {
                "peak_rss_bytes": peak_rss,
                "peak_rss_mb": round(peak_rss / (1024 * 1024), 3),
                "user_cpu_seconds": float(usage.ru_utime),
                "system_cpu_seconds": float(usage.ru_stime),
            }
        )
    except (OSError, ValueError):
        pass

    torch_module = sys.modules.get("torch")
    try:
        cuda = getattr(torch_module, "cuda", None)
        if cuda is not None and bool(cuda.is_available()):
            sample.update(
                {
                    "gpu_memory_allocated_bytes": int(cuda.memory_allocated()),
                    "gpu_memory_reserved_bytes": int(cuda.memory_reserved()),
                    "gpu_peak_memory_allocated_bytes": int(cuda.max_memory_allocated()),
                    "cuda_allocated_mb": round(cuda.memory_allocated() / (1024 * 1024), 3),
                    "cuda_reserved_mb": round(cuda.memory_reserved() / (1024 * 1024), 3),
                    "cuda_peak_mb": round(cuda.max_memory_allocated() / (1024 * 1024), 3),
                }
            )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
    return sample


def input_metadata_from_video(
    video_meta: Mapping[str, Any],
    *,
    size_bytes: int | None = None,
    content_type: str | None = None,
) -> dict[str, Any]:
    width = max(0, int(video_meta.get("width") or 0))
    height = max(0, int(video_meta.get("height") or 0))
    frame_count = max(0, int(video_meta.get("frame_count") or 0))
    fps = max(0.0, float(video_meta.get("fps") or 0.0))
    duration_seconds = frame_count / fps if frame_count and fps else 0.0
    max_dimension = max(width, height)
    if max_dimension <= 720:
        resolution_bucket = "sd_or_lower"
    elif max_dimension <= 1280:
        resolution_bucket = "hd"
    elif max_dimension <= 1920:
        resolution_bucket = "full_hd"
    else:
        resolution_bucket = "above_full_hd"
    if duration_seconds <= 5:
        duration_bucket = "0_5s"
    elif duration_seconds <= 10:
        duration_bucket = "5_10s"
    elif duration_seconds <= 20:
        duration_bucket = "10_20s"
    else:
        duration_bucket = "over_20s"
    payload: dict[str, Any] = {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": duration_seconds,
        "duration_bucket": duration_bucket,
        "resolution_bucket": resolution_bucket,
    }
    if size_bytes is not None:
        payload["size_bytes"] = max(0, int(size_bytes))
    if content_type:
        payload["content_type"] = content_type[:100]
    return payload


def validate_event(event: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "event_id",
        "run_id",
        "attempt_id",
        "sequence",
        "event_type",
        "event_time",
        "stage",
        "span",
        "elapsed_seconds",
        "status",
    }
    missing = sorted(required.difference(event))
    if missing:
        raise ValueError("Processing telemetry event is missing: " + ", ".join(missing))
    if event["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported processing telemetry schema: {event['schema_version']}")
    if event["event_type"] not in EVENT_TYPES:
        raise ValueError(f"Unknown processing telemetry event_type: {event['event_type']}")
    for identifier_key in ("event_id", "attempt_id"):
        identifier = event[identifier_key]
        if not isinstance(identifier, str) or not _UUID_PATTERN.fullmatch(identifier):
            raise ValueError(f"Processing telemetry {identifier_key} must be a UUID")
    if not isinstance(event["run_id"], str) or not _IDENTIFIER_PATTERN.fullmatch(event["run_id"]):
        raise ValueError("Processing telemetry run_id must be a bounded identifier")
    if event["stage"] is not None and event["stage"] not in PIPELINE_STAGES:
        raise ValueError(f"Unknown processing telemetry stage: {event['stage']}")
    if event["span"] is not None and event["span"] not in PROCESSING_SPANS:
        raise ValueError(f"Unknown processing telemetry span: {event['span']}")
    if event["event_type"] in _STAGE_EVENT_TYPES and event["stage"] is None:
        raise ValueError("This processing telemetry event_type requires stage")
    if event["event_type"] in _SPAN_EVENT_TYPES and event["span"] is None:
        raise ValueError("This processing telemetry event_type requires span")
    if event["span"] is not None and event["stage"] is None:
        raise ValueError("Processing telemetry span requires stage")
    if event["status"] not in EVENT_STATUSES:
        raise ValueError(f"Unknown processing telemetry status: {event['status']}")
    if (
        not isinstance(event["sequence"], int)
        or isinstance(event["sequence"], bool)
        or not 1 <= event["sequence"] <= 1_000_000_000
    ):
        raise ValueError("Processing telemetry sequence must be from 1 to 1000000000")
    elapsed_seconds = _safe_float(event["elapsed_seconds"])
    if elapsed_seconds is None or not 0 <= elapsed_seconds <= 31_536_000:
        raise ValueError(
            "Processing telemetry elapsed_seconds must be between 0 and 31536000"
        )
    event_time = event["event_time"]
    if not isinstance(event_time, str) or len(event_time) > 40:
        raise ValueError("Processing telemetry event_time must be an RFC 3339 timestamp")
    try:
        parsed_event_time = datetime.fromisoformat(event_time.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            "Processing telemetry event_time must be an RFC 3339 timestamp"
        ) from exc
    if parsed_event_time.tzinfo is None:
        raise ValueError("Processing telemetry event_time must include a timezone")
    if event["event_type"] in _FAILED_EVENT_TYPES and "error" not in event:
        raise ValueError("Failed processing telemetry events require error metadata")


class _AsyncEventSender:
    def __init__(
        self,
        sender: EventSender,
        *,
        max_queue_size: int = DEFAULT_DELIVERY_QUEUE_SIZE,
        failure_threshold: int = DEFAULT_DELIVERY_FAILURE_THRESHOLD,
        worker_count: int = DEFAULT_DELIVERY_WORKERS,
    ) -> None:
        self._sender = sender
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(
            maxsize=max(1, int(max_queue_size))
        )
        self._condition = threading.Condition()
        self._pending = 0
        self._closed = False
        self._circuit_open = False
        self._worker_count = max(1, min(int(worker_count), MAX_DELIVERY_WORKERS))
        self._failure_threshold = max(1, int(failure_threshold)) * self._worker_count
        self._consecutive_failures = 0
        self.failures = 0
        self.dropped = 0
        self._threads = [
            threading.Thread(
                target=self._run,
                name=f"processing-telemetry-sender-{index + 1}",
                daemon=True,
            )
            for index in range(self._worker_count)
        ]
        try:
            for thread in self._threads:
                thread.start()
        except RuntimeError:
            with self._condition:
                self._closed = True
                self._condition.notify_all()
            raise

    def submit(self, event: dict[str, Any]) -> None:
        with self._condition:
            if self._closed or self._circuit_open:
                self.dropped += 1
                return
            self._pending += 1
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                self._pending -= 1
                self.dropped += 1
                return

    def _run(self) -> None:
        while True:
            try:
                event = self._queue.get(timeout=0.1)
            except queue.Empty:
                with self._condition:
                    if self._closed and self._pending == 0:
                        return
                continue
            opened_circuit = False
            with self._condition:
                circuit_open = self._circuit_open
            if circuit_open:
                with self._condition:
                    self.dropped += 1
                    self._pending -= 1
                    self._condition.notify_all()
                self._queue.task_done()
                continue
            try:
                self._sender(event)
                with self._condition:
                    if not self._circuit_open:
                        self._consecutive_failures = 0
            except BaseException:
                with self._condition:
                    self.failures += 1
                    self._consecutive_failures += 1
                    if (
                        not self._circuit_open
                        and self._consecutive_failures >= self._failure_threshold
                    ):
                        self._circuit_open = True
                        opened_circuit = True
            finally:
                with self._condition:
                    self._pending -= 1
                    self._condition.notify_all()
                self._queue.task_done()
            if opened_circuit:
                self._drop_queued_events()

    def _drop_queued_events(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return
            with self._condition:
                self.dropped += 1
                self._pending -= 1
                self._condition.notify_all()
            self._queue.task_done()

    def flush(self, timeout: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        with self._condition:
            while self._pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(timeout=remaining)
        return True

    def close(self, timeout: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        flushed = self.flush(max(0.0, deadline - time.monotonic()))
        for thread in self._threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        return flushed and all(not thread.is_alive() for thread in self._threads)

    def measurements(self) -> dict[str, int]:
        with self._condition:
            return {
                "pending": self._pending,
                "failures": self.failures,
                "dropped": self.dropped,
            }


class ProcessingTelemetry:
    def __init__(
        self,
        *,
        run_id: str,
        attempt_id: str | None,
        local_path: Path | None,
        callback_base_url: str | None = None,
        auth_token: str | None = None,
        input_metadata: Mapping[str, Any] | None = None,
        runtime_metadata: Mapping[str, Any] | None = None,
        progress_interval_seconds: float = DEFAULT_PROGRESS_INTERVAL_SECONDS,
        monotonic_clock: Clock = time.monotonic,
        wall_clock: WallClock = _utc_now,
        event_id_factory: Callable[[], str] | None = None,
        resource_sampler: ResourceSampler = default_resource_sample,
        event_sender: EventSender | None = None,
        asynchronous_delivery: bool = True,
        delivery_queue_size: int = DEFAULT_DELIVERY_QUEUE_SIZE,
        delivery_failure_threshold: int = DEFAULT_DELIVERY_FAILURE_THRESHOLD,
        delivery_workers: int | None = None,
        start_heartbeat: bool = False,
        sequence_start: int = 1,
        attempt_elapsed_offset_seconds: float = 0.0,
    ) -> None:
        self.run_id = str(run_id)
        self.attempt_id = ensure_attempt_id(attempt_id)
        self.local_path = Path(local_path) if local_path is not None else None
        self.progress_interval_seconds = max(0.01, float(progress_interval_seconds))
        self._clock = monotonic_clock
        self._wall_clock = wall_clock
        self._event_id_factory = event_id_factory or (lambda: str(uuid.uuid4()))
        self._resource_sampler = resource_sampler
        self._started_at = self._safe_clock()
        self._attempt_elapsed_offset_seconds = max(
            0.0,
            _safe_float(attempt_elapsed_offset_seconds) or 0.0,
        )
        self._sequence = max(1, int(sequence_start)) - 1
        self._lock = threading.RLock()
        self._emit_lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._last_progress_at: dict[tuple[str, str | None], float] = {}
        self._active_stage: dict[str, Any] | None = None
        self._active_span: dict[str, Any] | None = None
        self._latest_progress: dict[str, Any] = {"phase": "running"}
        self._closed = False
        self.local_write_failures = 0
        self.delivery_failures = 0
        self.delivery_dropped = 0
        self._input = _safe_json_value(dict(input_metadata or {})) or {}
        self._runtime = {
            **default_runtime_metadata(),
            **(_safe_json_value(dict(runtime_metadata or {})) or {}),
        }

        sender = event_sender
        if sender is None and callback_base_url and auth_token:
            sender = self._http_sender(
                callback_base_url=callback_base_url,
                auth_token=auth_token,
            )
        self._direct_sender = sender if sender is not None and not asynchronous_delivery else None
        self._async_sender = (
            _AsyncEventSender(
                sender,
                max_queue_size=delivery_queue_size,
                failure_threshold=delivery_failure_threshold,
                worker_count=(
                    delivery_workers
                    if delivery_workers is not None
                    else _env_int(
                        "WHODOIRUNLIKE_TELEMETRY_DELIVERY_WORKERS",
                        DEFAULT_DELIVERY_WORKERS,
                    )
                ),
            )
            if sender is not None and asynchronous_delivery
            else None
        )
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        if start_heartbeat:
            try:
                self._heartbeat_thread = threading.Thread(
                    target=self._heartbeat_loop,
                    name="processing-telemetry-heartbeat",
                    daemon=True,
                )
                self._heartbeat_thread.start()
            except RuntimeError:
                self._heartbeat_thread = None

    def _safe_clock(self) -> float:
        try:
            return float(self._clock())
        except BaseException:
            return time.monotonic()

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, self._safe_clock() - self._started_at)

    @property
    def attempt_elapsed_seconds(self) -> float:
        return self._attempt_elapsed_offset_seconds + self.elapsed_seconds

    def update_input(self, values: Mapping[str, Any]) -> None:
        try:
            cleaned = _safe_json_value(dict(values)) or {}
            with self._lock:
                self._input.update(cleaned)
        except BaseException:
            return

    def update_runtime(self, values: Mapping[str, Any]) -> None:
        try:
            cleaned = _safe_json_value(dict(values)) or {}
            with self._lock:
                self._runtime.update(cleaned)
        except BaseException:
            return

    def _http_sender(self, *, callback_base_url: str, auth_token: str) -> EventSender:
        endpoint = f"{callback_base_url.rstrip('/')}/v1/jobs/{self.run_id}/events"
        timeout = _env_float(
            "WHODOIRUNLIKE_TELEMETRY_DELIVERY_TIMEOUT_SECONDS",
            DEFAULT_DELIVERY_TIMEOUT_SECONDS,
        )
        attempts = max(
            1,
            _env_int("WHODOIRUNLIKE_TELEMETRY_DELIVERY_ATTEMPTS", DEFAULT_DELIVERY_ATTEMPTS),
        )
        backoff = max(
            0.0,
            _env_float(
                "WHODOIRUNLIKE_TELEMETRY_DELIVERY_BACKOFF_SECONDS",
                DEFAULT_DELIVERY_BACKOFF_SECONDS,
            ),
        )

        def send(event: dict[str, Any]) -> None:
            body = json.dumps(event, separators=(",", ":"), allow_nan=False).encode("utf-8")
            for attempt_index in range(attempts):
                request = urllib.request.Request(
                    endpoint,
                    data=body,
                    method="POST",
                    headers={
                        "Authorization": f"Bearer {auth_token}",
                        "Accept": "application/json",
                        "Content-Type": "application/json; charset=utf-8",
                        "Content-Length": str(len(body)),
                        "User-Agent": "whodoirunlike-processor/1.0",
                    },
                )
                try:
                    with urllib.request.urlopen(request, timeout=timeout):
                        return
                except urllib.error.HTTPError as error:
                    retryable = error.code in {408, 425, 429} or error.code >= 500
                    if not retryable or attempt_index + 1 >= attempts:
                        raise
                except (TimeoutError, OSError, urllib.error.URLError):
                    if attempt_index + 1 >= attempts:
                        raise
                if backoff:
                    time.sleep(min(backoff * (2**attempt_index), 1.0))

        return send

    def _next_event(
        self,
        *,
        event_type: str,
        stage: str | None,
        span: str | None,
        elapsed_seconds: float,
        status: str,
        progress: Mapping[str, Any] | None,
        runtime: Mapping[str, Any] | None,
        measurements: Mapping[str, Any] | None,
        error: BaseException | str | None,
    ) -> dict[str, Any]:
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
            input_payload = dict(self._input)
            runtime_payload = dict(self._runtime)
        if runtime:
            runtime_payload.update(_safe_json_value(dict(runtime)) or {})
        runtime_payload.update(
            {
                key: value
                for key, value in _gpu_runtime_metadata().items()
                if key not in runtime_payload
            }
        )
        try:
            resources = _safe_json_value(dict(self._resource_sampler())) or {}
        except BaseException:
            resources = {}
        try:
            event_time = _format_event_time(self._wall_clock())
        except BaseException:
            event_time = _format_event_time(_utc_now())
        try:
            event_id = str(self._event_id_factory())
        except BaseException:
            event_id = str(uuid.uuid4())
        event: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "event_id": event_id,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "sequence": sequence,
            "event_type": event_type,
            "event_time": event_time,
            "stage": stage,
            "span": span,
            "elapsed_seconds": max(0.0, float(elapsed_seconds)),
            "status": status,
            "input": input_payload,
            "runtime": runtime_payload,
            "resources": resources,
            "measurements": _safe_json_value(dict(measurements or {})) or {},
        }
        if progress is not None:
            event["progress"] = _safe_json_value(dict(progress)) or {}
        if error is not None:
            event["error"] = sanitize_error(error)
        validate_event(event)
        return event

    def emit(
        self,
        event_type: str,
        *,
        stage: str | None = None,
        span: str | None = None,
        elapsed_seconds: float | None = None,
        status: str,
        progress: Mapping[str, Any] | None = None,
        runtime: Mapping[str, Any] | None = None,
        measurements: Mapping[str, Any] | None = None,
        error: BaseException | str | None = None,
    ) -> dict[str, Any] | None:
        with self._emit_lock:
            try:
                event = self._next_event(
                    event_type=event_type,
                    stage=stage,
                    span=span,
                    elapsed_seconds=self.elapsed_seconds if elapsed_seconds is None else elapsed_seconds,
                    status=status,
                    progress=progress,
                    runtime=runtime,
                    measurements=measurements,
                    error=error,
                )
            except BaseException:
                return None

            self._append_local(event)
            self._deliver(event)
            return event

    def _append_local(self, event: dict[str, Any]) -> None:
        if self.local_path is None:
            return
        try:
            line = json.dumps(event, separators=(",", ":"), allow_nan=False) + "\n"
            with self._write_lock:
                self.local_path.parent.mkdir(parents=True, exist_ok=True)
                with self.local_path.open("a", encoding="utf-8") as output:
                    output.write(line)
                    output.flush()
        except BaseException:
            with self._lock:
                self.local_write_failures += 1

    def _deliver(self, event: dict[str, Any]) -> None:
        if self._async_sender is not None:
            self._async_sender.submit(event)
            return
        if self._direct_sender is not None:
            try:
                self._direct_sender(event)
            except BaseException:
                with self._lock:
                    self.delivery_failures += 1

    def attempt_started(self) -> None:
        self.emit("attempt_started", status="running", elapsed_seconds=0.0)

    def attempt_completed(self, measurements: Mapping[str, Any] | None = None) -> None:
        self.emit(
            "attempt_completed",
            status="complete",
            elapsed_seconds=self.attempt_elapsed_seconds,
            measurements=measurements,
        )

    def attempt_failed(
        self,
        error: BaseException | str,
        measurements: Mapping[str, Any] | None = None,
    ) -> None:
        self.emit(
            "attempt_failed",
            status="failed",
            elapsed_seconds=self.attempt_elapsed_seconds,
            measurements=measurements,
            error=error,
        )

    def result_ready(self, measurements: Mapping[str, Any] | None = None) -> None:
        self.emit(
            "result_ready",
            stage="result_ready",
            status="complete",
            elapsed_seconds=self.attempt_elapsed_seconds,
            measurements=measurements,
        )

    def analysis_completed(self, measurements: Mapping[str, Any] | None = None) -> None:
        self.emit(
            "analysis_completed",
            stage="analysis_complete",
            status="complete",
            elapsed_seconds=self.attempt_elapsed_seconds,
            measurements=measurements,
        )

    def stage(
        self,
        stage: str,
        *,
        runtime: Mapping[str, Any] | None = None,
        measurements: Mapping[str, Any] | None = None,
    ) -> _Boundary:
        return _Boundary(
            self,
            kind="stage",
            stage=stage,
            span=None,
            runtime=runtime,
            measurements=measurements,
        )

    def span(
        self,
        stage: str,
        span: str,
        *,
        runtime: Mapping[str, Any] | None = None,
        measurements: Mapping[str, Any] | None = None,
    ) -> _Boundary:
        return _Boundary(
            self,
            kind="span",
            stage=stage,
            span=span,
            runtime=runtime,
            measurements=measurements,
        )

    def progress_reporter(
        self,
        *,
        stage: str,
        phase_spans: Mapping[str, str | None] | None = None,
        default_span: str | None = "inference",
        runtime: Mapping[str, Any] | None = None,
    ) -> ProgressReporter:
        return ProgressReporter(
            self,
            stage=stage,
            phase_spans=phase_spans,
            default_span=default_span,
            runtime=runtime,
        )

    def sample_progress(
        self,
        *,
        stage: str,
        span: str | None,
        progress: Mapping[str, Any],
        elapsed_seconds: float | None = None,
        force: bool = False,
        runtime: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        now = self._safe_clock()
        key = (stage, span)
        with self._state_lock:
            self._latest_progress = _safe_json_value(dict(progress)) or {}
            last = self._last_progress_at.get(key)
            if not force and last is not None and now - last < self.progress_interval_seconds:
                return None
            self._last_progress_at[key] = now
        return self.emit(
            "progress_sampled",
            stage=stage,
            span=span,
            elapsed_seconds=(
                max(0.0, now - self._started_at)
                if elapsed_seconds is None
                else max(0.0, float(elapsed_seconds))
            ),
            status="running",
            progress=progress,
            runtime=runtime,
        )

    def _activate_boundary(self, boundary: _Boundary) -> None:
        state = {
            "stage": boundary.stage,
            "span": boundary.span,
            "started_at": boundary.started_at,
            "runtime": dict(boundary.runtime),
        }
        with self._state_lock:
            if boundary.kind == "stage":
                self._active_stage = state
                self._latest_progress = {"phase": "running"}
            else:
                self._active_span = state

    def _deactivate_boundary(self, boundary: _Boundary) -> None:
        with self._state_lock:
            if boundary.kind == "stage":
                if self._active_stage and self._active_stage.get("stage") == boundary.stage:
                    self._active_stage = None
                    self._active_span = None
            elif self._active_span and self._active_span.get("span") == boundary.span:
                self._active_span = None

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(self.progress_interval_seconds):
            with self._state_lock:
                active = dict(self._active_span or self._active_stage or {})
                progress = dict(self._latest_progress)
            if not active:
                continue
            now = self._safe_clock()
            progress["heartbeat"] = True
            self.sample_progress(
                stage=str(active["stage"]),
                span=active.get("span"),
                progress=progress,
                elapsed_seconds=max(0.0, now - float(active["started_at"])),
                runtime=active.get("runtime"),
            )

    def close(self, *, timeout: float = 2.0) -> bool:
        if self._closed:
            return True
        self._closed = True
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=min(max(timeout, 0.0), 0.1))
        delivered = True
        if self._async_sender is not None:
            delivered = self._async_sender.close(timeout)
        return delivered

    def flush_delivery(self, *, timeout: float) -> bool:
        if self._async_sender is None:
            return True
        return self._async_sender.flush(max(0.0, timeout))

    def delivery_measurements(self) -> dict[str, int]:
        async_values = (
            self._async_sender.measurements()
            if self._async_sender is not None
            else {"pending": 0, "failures": 0, "dropped": 0}
        )
        with self._lock:
            return {
                "telemetry_delivery_pending": async_values["pending"],
                "telemetry_delivery_failures": self.delivery_failures
                + async_values["failures"],
                "telemetry_delivery_dropped": self.delivery_dropped + async_values["dropped"],
                "telemetry_local_write_failures": self.local_write_failures,
            }


class _Boundary:
    def __init__(
        self,
        telemetry: ProcessingTelemetry,
        *,
        kind: BoundaryKind,
        stage: str,
        span: str | None,
        runtime: Mapping[str, Any] | None,
        measurements: Mapping[str, Any] | None,
    ) -> None:
        self.telemetry = telemetry
        self.kind = kind
        self.stage = stage
        self.span = span
        self.runtime = dict(runtime or {})
        self.measurements = dict(measurements or {})
        self.status = "running"
        self.error: BaseException | str | None = None
        self.started_at = telemetry._safe_clock()
        self._entered = False
        self._closed = False

    def __enter__(self) -> _Boundary:
        self._entered = True
        self.started_at = self.telemetry._safe_clock()
        self.telemetry._activate_boundary(self)
        self.telemetry.emit(
            f"{self.kind}_started",
            stage=self.stage,
            span=self.span,
            elapsed_seconds=0.0,
            status="running",
            runtime=self.runtime,
            measurements=self.measurements,
        )
        return self

    def add_measurements(self, values: Mapping[str, Any]) -> None:
        try:
            self.measurements.update(_safe_json_value(dict(values)) or {})
        except BaseException:
            return

    def set_result(self, result: Mapping[str, Any] | None) -> None:
        if not isinstance(result, Mapping):
            return
        result_status = str(result.get("status") or "complete").lower()
        self.status = result_status
        self.add_measurements(_measurements_from_result(result))
        if result_status in {"failed", "failure", "unavailable", "error"}:
            self.error = str(result.get("error") or f"{self.stage} returned {result_status}")

    def close(self, error: BaseException | str | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        if error is not None:
            self.error = error
            if self.status in {"running", "complete"}:
                self.status = "failed"
        elapsed = max(0.0, self.telemetry._safe_clock() - self.started_at)
        self.telemetry._deactivate_boundary(self)
        failed = self.error is not None or self.status in {
            "failed",
            "failure",
            "unavailable",
            "error",
        }
        if self.status not in {"running", "complete", "failed"}:
            self.measurements.setdefault("outcome", self.status)
        self.telemetry.emit(
            f"{self.kind}_{'failed' if failed else 'completed'}",
            stage=self.stage,
            span=self.span,
            elapsed_seconds=elapsed,
            status="failed" if failed else "complete",
            runtime=self.runtime,
            measurements=self.measurements,
            error=self.error if failed else None,
        )

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        self.close(exc)
        return False


class ProgressReporter:
    def __init__(
        self,
        telemetry: ProcessingTelemetry,
        *,
        stage: str,
        phase_spans: Mapping[str, str | None] | None,
        default_span: str | None,
        runtime: Mapping[str, Any] | None,
    ) -> None:
        self.telemetry = telemetry
        self.stage = stage
        self.phase_spans = dict(phase_spans or {})
        self.default_span = default_span
        self.runtime = dict(runtime or {})
        self._active_span_name: str | None = None
        self._active_boundary: _Boundary | None = None
        self._started_at = telemetry._safe_clock()
        self._closed = False

    def __call__(self, progress: dict[str, Any]) -> None:
        if self._closed:
            return
        try:
            phase = str(progress.get("phase") or "running")
            span = self.phase_spans.get(phase, self.default_span)
            if span != self._active_span_name:
                if self._active_boundary is not None:
                    self._active_boundary.close()
                    self._active_boundary = None
                self._active_span_name = span
                if span is not None:
                    boundary = self.telemetry.span(self.stage, span, runtime=self.runtime)
                    boundary.__enter__()
                    self._active_boundary = boundary
            self.telemetry.sample_progress(
                stage=self.stage,
                span=span,
                progress=progress,
                elapsed_seconds=max(0.0, self.telemetry._safe_clock() - self._started_at),
                runtime=self.runtime,
            )
        except BaseException:
            return

    def close(self, error: BaseException | str | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        if self._active_boundary is not None:
            self._active_boundary.close(error)
            self._active_boundary = None


def _measurements_from_result(result: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "artifact_count",
        "artifacts_uploaded",
        "cache_hit",
        "detected_frames",
        "elapsed_seconds",
        "exports",
        "frame_count",
        "processed_frames",
        "model_build_seconds",
        "predictor_lock_wait_seconds",
        "row_count",
        "track_gated_frames",
        "usable_frame_count",
        "usable_frames",
    }
    measurements: dict[str, Any] = {}
    for key in allowed:
        value = result.get(key)
        if key == "artifacts_uploaded" and isinstance(value, list):
            measurements["artifact_count"] = len(value)
        elif key == "exports" and isinstance(value, Mapping):
            measurements["export_count"] = len(value)
            measurements["row_count"] = sum(
                int(item.get("row_count") or 0)
                for item in value.values()
                if isinstance(item, Mapping)
            )
        elif isinstance(value, (int, float, bool)):
            measurements[key] = value
    frame_count = _safe_float(measurements.get("frame_count"))
    elapsed = _safe_float(measurements.get("elapsed_seconds"))
    if frame_count and elapsed is not None and frame_count > 0:
        measurements["milliseconds_per_frame"] = elapsed * 1000.0 / frame_count
    return measurements


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def create_hosted_telemetry(
    *,
    run_id: str,
    attempt_id: str | None,
    run_dir: Path,
    callback_base_url: str,
    auth_token: str,
    input_metadata: Mapping[str, Any] | None = None,
    runtime_metadata: Mapping[str, Any] | None = None,
    sequence_start: int = 100,
    attempt_elapsed_offset_seconds: float = 0.0,
) -> ProcessingTelemetry:
    interval = _env_float(
        "WHODOIRUNLIKE_TELEMETRY_PROGRESS_INTERVAL_SECONDS",
        DEFAULT_PROGRESS_INTERVAL_SECONDS,
    )
    try:
        return ProcessingTelemetry(
            run_id=run_id,
            attempt_id=attempt_id,
            local_path=run_dir / PROCESSING_EVENTS_FILENAME,
            callback_base_url=callback_base_url,
            auth_token=auth_token,
            input_metadata=input_metadata,
            runtime_metadata=runtime_metadata,
            progress_interval_seconds=interval,
            asynchronous_delivery=True,
            delivery_failure_threshold=max(
                1,
                _env_int(
                    "WHODOIRUNLIKE_TELEMETRY_DELIVERY_FAILURE_THRESHOLD",
                    DEFAULT_DELIVERY_FAILURE_THRESHOLD,
                ),
            ),
            start_heartbeat=True,
            sequence_start=sequence_start,
            attempt_elapsed_offset_seconds=attempt_elapsed_offset_seconds,
        )
    except BaseException:
        return ProcessingTelemetry(
            run_id=run_id,
            attempt_id=attempt_id,
            local_path=run_dir / PROCESSING_EVENTS_FILENAME,
            input_metadata=input_metadata,
            runtime_metadata=runtime_metadata,
            progress_interval_seconds=interval,
            asynchronous_delivery=False,
            start_heartbeat=False,
            sequence_start=sequence_start,
            attempt_elapsed_offset_seconds=attempt_elapsed_offset_seconds,
        )
