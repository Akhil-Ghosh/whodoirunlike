from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

import whodoirunlike.video_io as video_io
from whodoirunlike.video_io import make_browser_playable_mp4, make_browser_playable_mp4s


def _write_mp4v_fixture(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (48, 32), True)
    assert writer.isOpened()
    for index in range(4):
        frame = np.zeros((32, 48, 3), dtype=np.uint8)
        frame[:, :, 2] = 60 + index * 30
        writer.write(frame)
    writer.release()


def _ffmpeg_video_line(path: Path) -> str:
    run = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return "\n".join(line.strip() for line in run.stderr.splitlines() if "Video:" in line)


def test_make_browser_playable_mp4_rewrites_opencv_mp4v_to_h264(tmp_path: Path) -> None:
    video_path = tmp_path / "artifact.mp4"
    _write_mp4v_fixture(video_path)

    make_browser_playable_mp4(video_path)

    video_line = _ffmpeg_video_line(video_path)
    assert "Video: h264" in video_line
    assert "yuv420p" in video_line


def test_make_browser_playable_mp4s_encodes_independent_files_concurrently(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = [tmp_path / f"artifact-{index}.mp4" for index in range(3)]
    active = 0
    peak_active = 0
    lock = threading.Lock()

    def encode(path: Path, *, crf: int = 20) -> None:
        nonlocal active, peak_active
        assert crf == 17
        assert path in paths
        with lock:
            active += 1
            peak_active = max(peak_active, active)
        time.sleep(0.03)
        with lock:
            active -= 1

    monkeypatch.setattr(video_io, "make_browser_playable_mp4", encode)

    make_browser_playable_mp4s(paths, crf=17)

    assert peak_active == len(paths)
