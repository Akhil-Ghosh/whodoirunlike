#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console

from whodoirunlike.cv_flow import DEFAULT_CV_RUN_ROOT
from whodoirunlike.densepose_runner import run_densepose


def main() -> None:
    parser = argparse.ArgumentParser(description="Run optional Detectron2 DensePose for one CV run.")
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_CV_RUN_ROOT)
    parser.add_argument("--config", type=Path, help="DensePose Detectron2 config YAML.")
    parser.add_argument("--weights", help="DensePose model weights path or URL.")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu", help="Detectron2 device, for example cpu or cuda.")
    parser.add_argument("--no-qa-overlay", action="store_true", help="Skip writing qa_overlay.mp4.")
    args = parser.parse_args()

    result = run_densepose(
        run_dir=args.run_root / args.candidate_id,
        config_path=args.config,
        weights_path=args.weights,
        confidence_threshold=args.confidence_threshold,
        device=args.device,
        write_qa_overlay=not args.no_qa_overlay,
    )
    Console().print_json(json.dumps(result))
    if result.get("status") == "failed":
        sys.exit(2)


if __name__ == "__main__":
    main()
