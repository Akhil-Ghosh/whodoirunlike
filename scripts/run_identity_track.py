#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console

from whodoirunlike.cv_flow import DEFAULT_CV_RUN_ROOT
from whodoirunlike.identity_runner import (
    DEFAULT_DETECTOR_MODEL,
    DEFAULT_IDENTITY_BACKEND,
    DEFAULT_REID_WEIGHTS,
    run_identity_tracking,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run prompt-seeded target identity tracking.")
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_CV_RUN_ROOT)
    parser.add_argument("--backend", default=DEFAULT_IDENTITY_BACKEND)
    parser.add_argument("--detector-model", default=DEFAULT_DETECTOR_MODEL)
    parser.add_argument("--reid-weights", default=DEFAULT_REID_WEIGHTS)
    parser.add_argument("--device")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--detector-confidence", type=float, default=0.25)
    parser.add_argument("--detector-iou", type=float, default=0.7)
    parser.add_argument("--detector-imgsz", type=int, default=960)
    args = parser.parse_args()

    result = run_identity_tracking(
        run_dir=args.run_root / args.candidate_id,
        backend=args.backend,
        detector_model=args.detector_model,
        reid_weights=args.reid_weights,
        device=args.device,
        half=args.half,
        detector_confidence=args.detector_confidence,
        detector_iou=args.detector_iou,
        detector_imgsz=args.detector_imgsz,
    )
    Console().print_json(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
