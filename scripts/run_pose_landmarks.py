#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console

from whodoirunlike.cv_flow import DEFAULT_CV_RUN_ROOT
from whodoirunlike.pose_runner import run_pose_landmarks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MediaPipe pose extraction for one CV run.")
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_CV_RUN_ROOT)
    parser.add_argument("--model-dir", type=Path, default=Path("models/mediapipe"))
    parser.add_argument("--model-variant", choices=["lite", "full", "heavy"], default="heavy")
    parser.add_argument(
        "--input-mode",
        choices=["auto", "source", "masked"],
        default="auto",
        help="auto prefers masked_runner.mp4 when present, then falls back to source_segment.mp4.",
    )
    args = parser.parse_args()

    result = run_pose_landmarks(
        run_dir=args.run_root / args.candidate_id,
        model_dir=args.model_dir,
        model_variant=args.model_variant,
        input_mode=args.input_mode,
    )
    Console().print_json(json.dumps(result))


if __name__ == "__main__":
    main()
