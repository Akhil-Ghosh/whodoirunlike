#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console

from whodoirunlike.cv_flow import DEFAULT_CV_RUN_ROOT
from whodoirunlike.sam2_runner import run_sam2_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAM 2 whole-runner tracking for one CV run.")
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_CV_RUN_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/sam2/sam2.1_hiera_tiny.pt"))
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], help="Override torch device.")
    parser.add_argument("--force-frames", action="store_true", help="Re-extract source frames.")
    args = parser.parse_args()

    result = run_sam2_mask(
        run_dir=args.run_root / args.candidate_id,
        checkpoint=args.checkpoint,
        model_cfg=args.model_cfg,
        device=args.device,
        force_frames=args.force_frames,
    )
    Console().print_json(json.dumps(result))


if __name__ == "__main__":
    main()
