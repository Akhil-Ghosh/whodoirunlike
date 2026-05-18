#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console

from whodoirunlike.cv_flow import DEFAULT_CV_RUN_ROOT
from whodoirunlike.sam31_mlx_runner import DEFAULT_SAM31_MLX_MODEL, run_sam31_mlx_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAM 3.1 MLX runner tracking for one CV run.")
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_CV_RUN_ROOT)
    parser.add_argument("--model", default=DEFAULT_SAM31_MLX_MODEL)
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Text prompt to detect. Can be repeated. Defaults to runner/person prompts.",
    )
    parser.add_argument("--threshold", type=float, default=0.18)
    parser.add_argument(
        "--quality-mode",
        choices=["max", "native", "fast"],
        default="native",
        help="SAM 3.1 input resolution preset: max=source-sized, native=1008, fast=224.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        help="Custom square input resolution. Overrides --quality-mode.",
    )
    parser.add_argument("--force-frames", action="store_true", help="Re-extract source frames.")
    args = parser.parse_args()

    result = run_sam31_mlx_mask(
        run_dir=args.run_root / args.candidate_id,
        model_path=args.model,
        prompts=tuple(args.prompt) if args.prompt else ("a runner", "a person"),
        quality_mode=args.quality_mode,
        threshold=args.threshold,
        resolution=args.resolution,
        force_frames=args.force_frames,
    )
    Console().print_json(json.dumps(result))


if __name__ == "__main__":
    main()
