#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console

from whodoirunlike.cv_flow import DEFAULT_CV_RUN_ROOT
from whodoirunlike.fusion_runner import run_fused_form


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse pose, runner mask, and DensePose into form QA outputs.")
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_CV_RUN_ROOT)
    args = parser.parse_args()

    result = run_fused_form(run_dir=args.run_root / args.candidate_id)
    Console().print_json(json.dumps(result))


if __name__ == "__main__":
    main()
