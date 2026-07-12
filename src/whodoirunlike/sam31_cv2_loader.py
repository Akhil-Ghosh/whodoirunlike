from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
import threading
import time
from typing import Any, Callable, Iterator

import cv2
import numpy as np

from whodoirunlike.sam31_loader_config import (
    DEFAULT_SAM31_EXACT_CV2_MAX_DESTINATION_BYTES,
    DEFAULT_SAM31_EXACT_CV2_MAX_FRAMES,
)


@dataclass(frozen=True)
class ExactCv2LoaderDiagnostics:
    frame_capacity: int
    decoded_frames: int
    image_size: int
    chunk_frames: int
    max_buffered_frames: int
    peak_host_staging_bytes: int
    max_frames: int
    max_destination_bytes: int
    destination_bytes: int
    output_tensor_bytes: int
    output_shape: tuple[int, ...]
    output_stride: tuple[int, ...]
    output_dtype: str
    output_device: str
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExactCv2LoaderUnavailable(RuntimeError):
    """The bounded loader cannot safely preallocate an exact output tensor."""


class ExactCv2LoaderSafetyLimitExceeded(RuntimeError):
    """The input exceeds a configured hard bound for the exact loader."""


@dataclass
class ExactCv2LoaderProbe:
    attempted: bool = False
    used: bool = False
    diagnostics: ExactCv2LoaderDiagnostics | None = None
    fallback_reason: str | None = None


_VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm"})
_UPSTREAM_LOADER_PATCH_LOCK = threading.RLock()


@contextmanager
def scoped_sam31_exact_cv2_loader(
    *,
    tracking_module: Any,
    enabled: bool,
    chunk_frames: int,
    max_frames: int = DEFAULT_SAM31_EXACT_CV2_MAX_FRAMES,
    max_destination_bytes: int = DEFAULT_SAM31_EXACT_CV2_MAX_DESTINATION_BYTES,
) -> Iterator[ExactCv2LoaderProbe]:
    """Substitute SAM's imported loader only for one serialized session start.

    SAM imports ``load_resource_as_video_frames`` into its tracking module, so
    replacing that one module attribute is the narrowest hook that preserves
    the rest of upstream ``init_state``. The caller already serializes predictor
    use; this lock additionally prevents overlapping scoped substitutions.
    """
    probe = ExactCv2LoaderProbe()
    if not enabled:
        yield probe
        return

    original_loader = getattr(tracking_module, "load_resource_as_video_frames", None)
    if not callable(original_loader):
        probe.fallback_reason = "tracking_module_loader_unavailable"
        yield probe
        return

    def replacement_loader(**kwargs: Any) -> Any:
        resource_path = kwargs.get("resource_path")
        video_loader_type = str(kwargs.get("video_loader_type") or "cv2").lower()
        is_supported_video = (
            isinstance(resource_path, str)
            and Path(resource_path).suffix.lower() in _VIDEO_EXTENSIONS
        )
        if not is_supported_video or video_loader_type != "cv2":
            return original_loader(**kwargs)

        probe.attempted = True
        try:
            result = load_video_frames_exact_cv2(
                video_path=resource_path,
                image_size=int(kwargs["image_size"]),
                offload_video_to_cpu=bool(kwargs["offload_video_to_cpu"]),
                img_mean=tuple(kwargs.get("img_mean") or (0.5, 0.5, 0.5)),
                img_std=tuple(kwargs.get("img_std") or (0.5, 0.5, 0.5)),
                chunk_frames=chunk_frames,
                max_frames=max_frames,
                max_destination_bytes=max_destination_bytes,
                diagnostics_callback=lambda sample: setattr(probe, "diagnostics", sample),
            )
        except (ExactCv2LoaderUnavailable, ExactCv2LoaderSafetyLimitExceeded) as exc:
            probe.used = False
            probe.fallback_reason = str(exc)
            return original_loader(**kwargs)
        probe.used = True
        return result

    with _UPSTREAM_LOADER_PATCH_LOCK:
        setattr(tracking_module, "load_resource_as_video_frames", replacement_loader)
        try:
            yield probe
        finally:
            if getattr(tracking_module, "load_resource_as_video_frames", None) is replacement_loader:
                setattr(tracking_module, "load_resource_as_video_frames", original_loader)


def load_video_frames_exact_cv2(
    *,
    video_path: str,
    image_size: int,
    offload_video_to_cpu: bool,
    img_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    img_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    chunk_frames: int = 8,
    max_frames: int = DEFAULT_SAM31_EXACT_CV2_MAX_FRAMES,
    max_destination_bytes: int = DEFAULT_SAM31_EXACT_CV2_MAX_DESTINATION_BYTES,
    diagnostics_callback: Callable[[ExactCv2LoaderDiagnostics], None] | None = None,
) -> tuple[Any, int, int]:
    """Load SAM frames with upstream-equivalent CV2 preprocessing.

    Unlike SAM's upstream loader, this implementation never retains every
    resized uint8 frame alongside a stacked uint8 copy and a full float32 copy.
    It preallocates the final NHWC float32 tensor on the destination device and
    transfers at most ``chunk_frames`` decoded frames at a time.
    """
    import torch

    started_at = time.perf_counter()
    effective_chunk_frames = max(1, int(chunk_frames))
    effective_max_frames = max(1, int(max_frames))
    effective_max_destination_bytes = max(1, int(max_destination_bytes))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_capacity = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_capacity <= 0:
            raise ExactCv2LoaderUnavailable(
                "OpenCV did not report a positive frame count for bounded preallocation"
            )

        destination_bytes = (
            frame_capacity * image_size * image_size * 3 * np.dtype(np.float32).itemsize
        )
        if frame_capacity > effective_max_frames:
            raise ExactCv2LoaderSafetyLimitExceeded(
                "SAM exact CV2 loader refused "
                f"{frame_capacity} frames because the configured maximum is "
                f"{effective_max_frames}"
            )
        if destination_bytes > effective_max_destination_bytes:
            raise ExactCv2LoaderSafetyLimitExceeded(
                "SAM exact CV2 loader refused a "
                f"{destination_bytes}-byte destination because the configured maximum is "
                f"{effective_max_destination_bytes}"
            )

        device = "cpu" if offload_video_to_cpu else "cuda"
        destination = torch.empty(
            (frame_capacity, image_size, image_size, 3),
            dtype=torch.float32,
            device=device,
        )
        staging_capacity = min(frame_capacity, effective_chunk_frames)
        uint8_staging = np.empty(
            (staging_capacity, image_size, image_size, 3),
            dtype=np.uint8,
        )

        decoded_frames = 0
        max_buffered_frames = 0
        peak_host_staging_bytes = int(uint8_staging.nbytes)
        reached_end = False
        while not reached_end:
            buffered_frames = 0
            while buffered_frames < staging_capacity:
                ok, frame = cap.read()
                if not ok:
                    reached_end = True
                    break
                if decoded_frames + buffered_frames >= frame_capacity:
                    raise ExactCv2LoaderUnavailable(
                        "OpenCV decoded more frames than CAP_PROP_FRAME_COUNT reported"
                    )
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                uint8_staging[buffered_frames] = cv2.resize(
                    frame_rgb,
                    (image_size, image_size),
                    interpolation=cv2.INTER_CUBIC,
                )
                buffered_frames += 1

            if buffered_frames == 0:
                continue
            max_buffered_frames = max(max_buffered_frames, buffered_frames)
            float32_staging = uint8_staging[:buffered_frames].astype(np.float32)
            peak_host_staging_bytes = max(
                peak_host_staging_bytes,
                int(uint8_staging.nbytes + float32_staging.nbytes),
            )
            destination[decoded_frames : decoded_frames + buffered_frames].copy_(
                torch.from_numpy(float32_staging)
            )
            decoded_frames += buffered_frames
            del float32_staging

        if decoded_frames == 0:
            raise RuntimeError(
                f"No frames could be decoded from video: {video_path}. "
                "The file may be corrupted, empty, or encoded with an unsupported codec."
            )
        if decoded_frames != frame_capacity:
            raise ExactCv2LoaderUnavailable(
                "OpenCV decoded fewer frames than CAP_PROP_FRAME_COUNT reported"
            )

        video_tensor = destination[:decoded_frames].permute(0, 3, 1, 2)
        mean_tensor = torch.tensor(
            img_mean,
            dtype=torch.float16,
            device=video_tensor.device,
        ).view(1, 3, 1, 1)
        std_tensor = torch.tensor(
            img_std,
            dtype=torch.float16,
            device=video_tensor.device,
        ).view(1, 3, 1, 1)
        video_tensor -= mean_tensor
        video_tensor /= std_tensor

        if diagnostics_callback is not None:
            diagnostics_callback(
                ExactCv2LoaderDiagnostics(
                    frame_capacity=frame_capacity,
                    decoded_frames=decoded_frames,
                    image_size=int(image_size),
                    chunk_frames=effective_chunk_frames,
                    max_buffered_frames=max_buffered_frames,
                    peak_host_staging_bytes=peak_host_staging_bytes,
                    max_frames=effective_max_frames,
                    max_destination_bytes=effective_max_destination_bytes,
                    destination_bytes=destination_bytes,
                    output_tensor_bytes=(
                        decoded_frames * image_size * image_size * 3 * np.dtype(np.float32).itemsize
                    ),
                    output_shape=tuple(int(value) for value in video_tensor.shape),
                    output_stride=tuple(int(value) for value in video_tensor.stride()),
                    output_dtype=str(video_tensor.dtype),
                    output_device=str(video_tensor.device),
                    elapsed_seconds=round(time.perf_counter() - started_at, 6),
                )
            )
        return video_tensor, original_height, original_width
    except ExactCv2LoaderUnavailable:
        # The scoped adapter may invoke the upstream loader after this frame
        # unwinds. Drop the preallocated destination before that fallback so
        # the two full-frame tensors cannot overlap.
        if "destination" in locals():
            del destination
        if "uint8_staging" in locals():
            del uint8_staging
        if str(locals().get("device", "")).startswith("cuda"):
            try:
                torch.cuda.empty_cache()
            except (AttributeError, RuntimeError):
                pass
        raise
    finally:
        cap.release()
