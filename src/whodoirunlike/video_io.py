from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

import imageio_ffmpeg


def make_browser_playable_mp4(path: Path, *, crf: int = 20) -> None:
    """Rewrite an MP4 as browser-safe H.264.

    OpenCV's MP4 writer commonly emits `mp4v`, which QuickTime may play locally but
    browser `<video>` elements often reject. Keep generation simple with OpenCV, then
    normalize the finished artifact through ffmpeg.
    """
    if not path.exists():
        raise FileNotFoundError(f"Video artifact not found: {path}")

    path = path.resolve()
    with tempfile.NamedTemporaryFile(
        delete=False,
        dir=str(path.parent),
        prefix=f".{path.stem}.",
        suffix=".h264.mp4",
    ) as temp_file:
        temp_path = Path(temp_file.name)

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-i",
        str(path),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temp_path),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        temp_path.replace(path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def make_browser_playable_mp4s(paths: Iterable[Path], *, crf: int = 20) -> None:
    queued_paths = list(paths)
    if len(queued_paths) < 2:
        for path in queued_paths:
            make_browser_playable_mp4(path, crf=crf)
        return

    with ThreadPoolExecutor(
        max_workers=min(3, len(queued_paths)),
        thread_name_prefix="video-encode",
    ) as executor:
        futures = [
            executor.submit(make_browser_playable_mp4, path, crf=crf)
            for path in queued_paths
        ]
        for future in futures:
            future.result()
