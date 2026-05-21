#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console

from whodoirunlike.cv_flow import DEFAULT_CV_RUN_ROOT
from whodoirunlike.full_pipeline import run_full_cv_pipeline
from whodoirunlike.identity_runner import DEFAULT_IDENTITY_BACKEND


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full identity-stable CV pipeline for one CV run.")
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_CV_RUN_ROOT)
    parser.add_argument("--identity-backend", default=DEFAULT_IDENTITY_BACKEND)
    parser.add_argument("--pose-backend", default="mmpose_rtmpose_l_384")
    parser.add_argument("--mask-quality-mode", default="native")
    parser.add_argument("--skip-densepose", action="store_true")
    args = parser.parse_args()
    result = run_full_cv_pipeline(
        run_dir=args.run_root / args.candidate_id,
        identity_backend=args.identity_backend,
        pose_backend=args.pose_backend,
        mask_quality_mode=args.mask_quality_mode,
        skip_densepose=args.skip_densepose,
    )
    Console().print_json(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
