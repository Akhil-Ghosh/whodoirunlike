from __future__ import annotations

import argparse
import json
from pathlib import Path

from whodoirunlike.cv_flow import DEFAULT_CV_RUN_ROOT
from whodoirunlike.openpose_runner import run_openpose_comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run optional OpenPose BODY_25 and compare it against MediaPipe landmarks."
    )
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--cv-run-root", type=Path, default=DEFAULT_CV_RUN_ROOT)
    parser.add_argument("--binary", type=Path, default=None, help="Path to openpose/openpose.bin")
    parser.add_argument("--model-folder", type=Path, default=None, help="Path to OpenPose models folder")
    args = parser.parse_args()

    result = run_openpose_comparison(
        run_dir=args.cv_run_root / args.candidate_id,
        binary_path=args.binary,
        model_folder=args.model_folder,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
