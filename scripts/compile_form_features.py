#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console

from whodoirunlike.cv_flow import DEFAULT_CV_RUN_ROOT
from whodoirunlike.form_features import compile_form_features


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile searchable pose-sequence form features for one CV run."
    )
    parser.add_argument("--candidate-id")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_CV_RUN_ROOT)
    args = parser.parse_args()

    if not args.candidate_id and not args.run_dir:
        parser.error("Provide either --candidate-id or --run-dir")

    run_dir = args.run_dir or args.run_root / str(args.candidate_id)
    result = compile_form_features(run_dir=run_dir)
    Console().print_json(json.dumps(result))


if __name__ == "__main__":
    main()
