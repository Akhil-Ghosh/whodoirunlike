from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import types
import weakref

import cv2
import numpy as np
import pytest

from whodoirunlike.sam31_cv2_loader import (
    ExactCv2LoaderDiagnostics,
    load_video_frames_exact_cv2,
)

torch = pytest.importorskip(
    "torch",
    reason="Exact SAM CV2 loader equivalence requires the RunPod Torch runtime",
)


def _write_synthetic_video(path: Path) -> None:
    width, height = 46, 34
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        12.0,
        (width, height),
        True,
    )
    assert writer.isOpened()
    rng = np.random.default_rng(20260711)
    for frame_index in range(7):
        frame = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        frame[:, :, frame_index % 3] = np.arange(width, dtype=np.uint8)[None, :]
        writer.write(frame)
    writer.release()


def _load_pinned_upstream_io_utils(monkeypatch: pytest.MonkeyPatch):
    source_root = os.getenv("WHODOIRUNLIKE_SAM3_SOURCE")
    if not source_root:
        pytest.skip("Set WHODOIRUNLIKE_SAM3_SOURCE to the pinned SAM 3.1 checkout")
    module_path = Path(source_root) / "sam3" / "model" / "io_utils.py"
    if not module_path.is_file():
        pytest.skip(f"Pinned SAM 3.1 io_utils.py is unavailable at {module_path}")

    sam3_package = types.ModuleType("sam3")
    sam3_package.__path__ = [str(Path(source_root) / "sam3")]
    logger_module = types.ModuleType("sam3.logger")
    logger_module.get_logger = lambda _name: types.SimpleNamespace(
        info=lambda *_args, **_kwargs: None,
        warning=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "sam3", sam3_package)
    monkeypatch.setitem(sys.modules, "sam3.logger", logger_module)
    spec = importlib.util.spec_from_file_location("sam3.model.io_utils", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_cv2_loader_is_bitwise_identical_to_pinned_upstream(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "synthetic.mp4"
    _write_synthetic_video(video_path)
    upstream = _load_pinned_upstream_io_utils(monkeypatch)

    expected, expected_height, expected_width = (
        upstream.load_video_frames_from_video_file_using_cv2(
            video_path=str(video_path),
            image_size=32,
            offload_video_to_cpu=True,
        )
    )
    actual, actual_height, actual_width = load_video_frames_exact_cv2(
        video_path=str(video_path),
        image_size=32,
        offload_video_to_cpu=True,
        chunk_frames=3,
    )

    assert (actual_height, actual_width) == (expected_height, expected_width)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.stride() == expected.stride()
    assert torch.equal(actual, expected)


def test_exact_cv2_loader_bounds_host_staging_to_configured_chunk(tmp_path: Path) -> None:
    video_path = tmp_path / "synthetic.mp4"
    _write_synthetic_video(video_path)
    diagnostics: list[ExactCv2LoaderDiagnostics] = []

    frames, _height, _width = load_video_frames_exact_cv2(
        video_path=str(video_path),
        image_size=32,
        offload_video_to_cpu=True,
        chunk_frames=2,
        diagnostics_callback=diagnostics.append,
    )

    assert len(diagnostics) == 1
    sample = diagnostics[0]
    bytes_per_frame = 32 * 32 * 3
    assert sample.decoded_frames == frames.shape[0] == 7
    assert sample.max_buffered_frames <= 2
    assert sample.peak_host_staging_bytes <= 2 * bytes_per_frame * (1 + 4)
    assert sample.destination_bytes == 7 * bytes_per_frame * 4


def test_scoped_sam_loader_uses_exact_cv2_and_restores_upstream(tmp_path: Path) -> None:
    from whodoirunlike.sam31_cv2_loader import scoped_sam31_exact_cv2_loader

    video_path = tmp_path / "synthetic.mp4"
    _write_synthetic_video(video_path)
    original_calls: list[str] = []

    def original_loader(**kwargs):
        original_calls.append(str(kwargs["resource_path"]))
        return "upstream"

    tracking_module = types.SimpleNamespace(load_resource_as_video_frames=original_loader)

    with scoped_sam31_exact_cv2_loader(
        tracking_module=tracking_module,
        enabled=True,
        chunk_frames=2,
    ) as probe:
        frames, height, width = tracking_module.load_resource_as_video_frames(
            resource_path=str(video_path),
            image_size=32,
            offload_video_to_cpu=True,
            img_mean=(0.5, 0.5, 0.5),
            img_std=(0.5, 0.5, 0.5),
            async_loading_frames=True,
            video_loader_type="cv2",
        )
        assert frames.shape == (7, 3, 32, 32)
        assert (height, width) == (34, 46)
        assert probe.used is True
        assert probe.diagnostics is not None
        assert original_calls == []

    assert tracking_module.load_resource_as_video_frames is original_loader


def test_frame_count_mismatch_releases_destination_before_upstream_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from whodoirunlike import sam31_cv2_loader
    from whodoirunlike.sam31_cv2_loader import scoped_sam31_exact_cv2_loader

    class FakeCapture:
        def __init__(self, _path: str) -> None:
            self.frames = [
                np.zeros((4, 6, 3), dtype=np.uint8),
                np.ones((4, 6, 3), dtype=np.uint8),
            ]

        def isOpened(self) -> bool:
            return True

        def get(self, property_id: int) -> float:
            if property_id == cv2.CAP_PROP_FRAME_HEIGHT:
                return 4
            if property_id == cv2.CAP_PROP_FRAME_WIDTH:
                return 6
            if property_id == cv2.CAP_PROP_FRAME_COUNT:
                return 1
            return 0

        def read(self):
            if not self.frames:
                return False, None
            return True, self.frames.pop(0)

        def release(self) -> None:
            return None

    destination_refs: list[weakref.ReferenceType] = []
    real_empty = torch.empty

    def recording_empty(*args, **kwargs):
        destination = real_empty(*args, **kwargs)
        destination_refs.append(weakref.ref(destination))
        return destination

    monkeypatch.setattr(sam31_cv2_loader.cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(torch, "empty", recording_empty)

    def original_loader(**_kwargs):
        assert destination_refs[0]() is None
        return "upstream"

    tracking_module = types.SimpleNamespace(load_resource_as_video_frames=original_loader)

    with scoped_sam31_exact_cv2_loader(
        tracking_module=tracking_module,
        enabled=True,
        chunk_frames=1,
    ) as probe:
        result = tracking_module.load_resource_as_video_frames(
            resource_path="clip.mp4",
            image_size=8,
            offload_video_to_cpu=True,
            img_mean=(0.5, 0.5, 0.5),
            img_std=(0.5, 0.5, 0.5),
            async_loading_frames=True,
            video_loader_type="cv2",
        )

    assert result == "upstream"
    assert probe.used is True
    assert probe.fallback_reason == (
        "OpenCV decoded more frames than CAP_PROP_FRAME_COUNT reported"
    )
