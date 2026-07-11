#!/usr/bin/env python3
"""PROTOTYPE: isolate OpenCV decode/resize/mask throughput from model inference."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2


def run_once(video: Path, width: int) -> tuple[int, float]:
    capture = cv2.VideoCapture(str(video))
    started = time.perf_counter()
    frames = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            target_height = max(1, round(frame.shape[0] * width / frame.shape[1]))
            resized = cv2.resize(frame, (width, target_height), interpolation=cv2.INTER_LINEAR)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)[1]
            cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            frames += 1
    finally:
        capture.release()
    return frames, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    runs = [run_once(args.video, args.width) for _ in range(args.iterations)]
    frame_count = runs[-1][0]
    seconds = [duration for _, duration in runs]
    median = statistics.median(seconds)
    print(
        json.dumps(
            {
                "prototype": True,
                "question": "Does OpenCV 5 materially change our non-model frame path?",
                "opencv": cv2.__version__,
                "video": str(args.video.resolve()),
                "frames": frame_count,
                "iterations": args.iterations,
                "seconds": [round(value, 4) for value in seconds],
                "median_seconds": round(median, 4),
                "median_frames_per_second": round(frame_count / median, 2),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
