from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Any, Mapping


SAM31_EXACT_CV2_LOADER_ENV = "WHODOIRUNLIKE_SAM31_GPU_EXACT_CV2_LOADER"
SAM31_EXACT_CV2_CHUNK_FRAMES_ENV = "WHODOIRUNLIKE_SAM31_GPU_EXACT_CV2_CHUNK_FRAMES"
SAM31_EXACT_CV2_MAX_FRAMES_ENV = "WHODOIRUNLIKE_SAM31_GPU_EXACT_CV2_MAX_FRAMES"
SAM31_EXACT_CV2_MAX_DESTINATION_BYTES_ENV = (
    "WHODOIRUNLIKE_SAM31_GPU_EXACT_CV2_MAX_DESTINATION_BYTES"
)
PROCESSOR_CONCURRENCY_ENV = "WHODOIRUNLIKE_PROCESSOR_CONCURRENCY"

DEFAULT_SAM31_EXACT_CV2_CHUNK_FRAMES = 8
DEFAULT_SAM31_EXACT_CV2_MAX_FRAMES = 600
DEFAULT_SAM31_EXACT_CV2_MAX_DESTINATION_BYTES = 8 * 1024**3
REQUIRED_SAM31_EXACT_CV2_CONCURRENCY = 1

MAX_SAM31_EXACT_CV2_CHUNK_FRAMES = 64
ABSOLUTE_MAX_SAM31_EXACT_CV2_FRAMES = 3600
ABSOLUTE_MAX_SAM31_EXACT_CV2_DESTINATION_BYTES = 32 * 1024**3
MAX_PROCESSOR_CONCURRENCY = 64


@dataclass(frozen=True)
class Sam31ExactCv2LoaderSettings:
    enabled: bool
    chunk_frames: int
    max_frames: int
    max_destination_bytes: int
    configured_concurrency: int

    @property
    def mode(self) -> str:
        return "exact_cv2" if self.enabled else "upstream"

    @property
    def concurrency_ready(self) -> bool:
        return self.configured_concurrency == REQUIRED_SAM31_EXACT_CV2_CONCURRENCY

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        return {
            "mode": self.mode,
            "enabled": values["enabled"],
            "chunk_frames": values["chunk_frames"],
            "max_frames": values["max_frames"],
            "max_destination_bytes": values["max_destination_bytes"],
            "required_concurrency": REQUIRED_SAM31_EXACT_CV2_CONCURRENCY,
            "configured_concurrency": values["configured_concurrency"],
            "concurrency_ready": self.concurrency_ready,
        }


def _bool_value(values: Mapping[str, object], name: str, default: bool) -> bool:
    raw = values.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _bounded_int(
    values: Mapping[str, object],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = values.get(name)
    try:
        parsed = default if raw is None else int(str(raw).strip())
    except (TypeError, ValueError, OverflowError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def sam31_exact_cv2_loader_settings(
    values: Mapping[str, object] | None = None,
) -> Sam31ExactCv2LoaderSettings:
    environment: Mapping[str, object] = os.environ if values is None else values
    return Sam31ExactCv2LoaderSettings(
        enabled=_bool_value(environment, SAM31_EXACT_CV2_LOADER_ENV, False),
        chunk_frames=_bounded_int(
            environment,
            SAM31_EXACT_CV2_CHUNK_FRAMES_ENV,
            DEFAULT_SAM31_EXACT_CV2_CHUNK_FRAMES,
            minimum=1,
            maximum=MAX_SAM31_EXACT_CV2_CHUNK_FRAMES,
        ),
        max_frames=_bounded_int(
            environment,
            SAM31_EXACT_CV2_MAX_FRAMES_ENV,
            DEFAULT_SAM31_EXACT_CV2_MAX_FRAMES,
            minimum=1,
            maximum=ABSOLUTE_MAX_SAM31_EXACT_CV2_FRAMES,
        ),
        max_destination_bytes=_bounded_int(
            environment,
            SAM31_EXACT_CV2_MAX_DESTINATION_BYTES_ENV,
            DEFAULT_SAM31_EXACT_CV2_MAX_DESTINATION_BYTES,
            minimum=1,
            maximum=ABSOLUTE_MAX_SAM31_EXACT_CV2_DESTINATION_BYTES,
        ),
        configured_concurrency=_bounded_int(
            environment,
            PROCESSOR_CONCURRENCY_ENV,
            REQUIRED_SAM31_EXACT_CV2_CONCURRENCY,
            minimum=1,
            maximum=MAX_PROCESSOR_CONCURRENCY,
        ),
    )
