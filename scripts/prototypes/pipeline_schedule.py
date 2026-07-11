#!/usr/bin/env python3
"""PROTOTYPE: expose the critical path implied by real production timings."""

from __future__ import annotations

import argparse
import json


OBSERVED = {
    "warm": {
        "front_matter": 8.870,
        "target_tracking": 2.274,
        "runner_mask": 35.317,
        "pose_sequence": 39.603,
        "densepose_body_map": 23.437,
        "post_join_analysis": 3.653,
        "publish_to_result_ready": 20.909,
        "observed_result_ready": 134.066,
    },
    "cold": {
        "front_matter": 23.336,
        "target_tracking": 5.706,
        "runner_mask": 54.569,
        "pose_sequence": 36.978,
        "densepose_body_map": 26.081,
        "post_join_analysis": 3.752,
        "publish_to_result_ready": 21.261,
        "observed_result_ready": 171.703,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=sorted(OBSERVED), default="warm")
    parser.add_argument(
        "--parallel-pose-densepose",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--unified-track-mask-seconds",
        type=float,
        help="Replace target_tracking + runner_mask with a measured unified YOLO duration.",
    )
    parser.add_argument(
        "--pose-seconds",
        type=float,
        help="Override pose duration with a measured provider/model prototype.",
    )
    parser.add_argument(
        "--densepose-seconds",
        type=float,
        help="Override DensePose duration with a measured prototype.",
    )
    parser.add_argument(
        "--publish-to-ready-seconds",
        type=float,
        help="Override required-artifact publishing time.",
    )
    args = parser.parse_args()

    observed = dict(OBSERVED[args.scenario])
    pose = args.pose_seconds if args.pose_seconds is not None else observed["pose_sequence"]
    densepose = (
        args.densepose_seconds
        if args.densepose_seconds is not None
        else observed["densepose_body_map"]
    )
    track_mask = (
        args.unified_track_mask_seconds
        if args.unified_track_mask_seconds is not None
        else observed["target_tracking"] + observed["runner_mask"]
    )
    fork_join = max(pose, densepose) if args.parallel_pose_densepose else pose + densepose
    publish = (
        args.publish_to_ready_seconds
        if args.publish_to_ready_seconds is not None
        else observed["publish_to_result_ready"]
    )
    simulated = (
        observed["front_matter"]
        + track_mask
        + fork_join
        + observed["post_join_analysis"]
        + publish
    )
    state = {
        "prototype": True,
        "question": "How does a different dependency graph change result-ready latency?",
        "scenario": args.scenario,
        "inputs_seconds": {
            "front_matter": observed["front_matter"],
            "track_and_mask": round(track_mask, 3),
            "pose": round(pose, 3),
            "densepose": round(densepose, 3),
            "post_join_analysis": observed["post_join_analysis"],
            "publish_to_ready": round(publish, 3),
        },
        "schedule": {
            "pose_densepose": "fork_join" if args.parallel_pose_densepose else "serial",
            "pose_densepose_critical_path_seconds": round(fork_join, 3),
        },
        "observed_result_ready_seconds": observed["observed_result_ready"],
        "simulated_result_ready_seconds": round(simulated, 3),
        "seconds_saved": round(observed["observed_result_ready"] - simulated, 3),
        "speedup": round(observed["observed_result_ready"] / simulated, 3),
    }
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
