#!/usr/bin/env python3
"""Emit a secret-free compatibility fingerprint for the RunPod processor image."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, distributions, version
from pathlib import Path
from typing import Any


EXPECTED_DISTRIBUTIONS = {
    "boxmot": "21.0.0",
    "detectron2": "0.6",
    "onnxruntime-gpu": "1.26.0",
    "opencv-contrib-python-headless": "4.11.0.86",
    "rtmlib": "0.0.15",
    "timm": "1.0.28",
    "torch": "2.9.1+cu128",
    "torchvision": "0.24.1",
    "ultralytics": "8.4.92",
}
EXPECTED_ASSETS = {
    "/app/yolo26n-seg.pt": {
        "size": 6_719_965,
        "sha256": "361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f",
    },
    (
        "/opt/rtmlib-cache/hub/checkpoints/"
        "yolox_m_8xb8-300e_humanart-c2c7a14a.onnx"
    ): {
        "size": 101_400_344,
        "sha256": "3dea6513388889f0fff4b77bf7a26013600321b9eb9ceb0e9a400a82572f5f23",
    },
    (
        "/opt/rtmlib-cache/hub/checkpoints/"
        "rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.onnx"
    ): {
        "size": 134_399_323,
        "sha256": "8cfecfc2226d8e14b510c2fd28442226c518b4d60690535753187877411d4005",
    },
}
EXPECTED_REVISIONS = {
    "detectron2": "02b5c4e295e990042a714712c21dc79b731e8833",
    "sam3": "5dd401d1c5c1d5c3eedff06d41b77af824517619",
}
EXPECTED_DISTRIBUTION_FINGERPRINT = (
    "875c0ab9cdf529cb8fef26e117c95c212544d5f8e5fc61b1b8326c462a380615"
)
EXPECTED_TREE_FINGERPRINTS = {
    "app": "cb1321f756974b3b2f2c59c4cea9fbbb68c81aaeb38abf4cce770016e9fc085a",
    "detectron2": "eb6f3c496af0f85faae1b9e63d8aed8c7f7657b385c9fc966c1bcc42a06698ee",
    "sam3": "1b3a2309e831f5cf45689dcc613a47282a82f9de5060a78f8c0b2b73226aa133",
}
TREE_ROOTS = {
    "app": Path("/app"),
    "detectron2": Path("/opt/detectron2"),
    "sam3": Path("/opt/sam3"),
}
IGNORED_TREE_PARTS = {".git", "__pycache__"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_fingerprint(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    if not root.is_dir():
        return {"exists": False, "files": 0, "bytes": 0, "sha256": None}

    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in IGNORED_TREE_PARTS for part in relative.parts):
            continue
        if not path.is_file() or path.suffix == ".pyc":
            continue
        file_digest = _sha256(path)
        size = path.stat().st_size
        digest.update(relative.as_posix().encode())
        digest.update(b"\0")
        digest.update(str(size).encode())
        digest.update(b"\0")
        digest.update(file_digest.encode())
        digest.update(b"\n")
        file_count += 1
        total_bytes += size
    return {
        "exists": True,
        "files": file_count,
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def _distribution_versions() -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    for distribution in distributions():
        name = str(distribution.metadata.get("Name") or "").strip()
        if not name:
            continue
        found.append({"name": name.lower().replace("_", "-"), "version": distribution.version})
    return sorted(found, key=lambda item: (item["name"], item["version"]))


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _module_origin(name: str) -> str | None:
    spec = importlib.util.find_spec(name)
    return str(spec.origin) if spec and spec.origin else None


def _revision(repo_root: Path, name: str) -> str | None:
    immutable_provenance = Path("/opt/whodoirunlike-provenance") / f"{name}.revision"
    if immutable_provenance.is_file():
        return immutable_provenance.read_text(encoding="utf-8").strip()
    provenance = repo_root / "WHODOIRUNLIKE_REVISION"
    if provenance.is_file():
        return provenance.read_text(encoding="utf-8").strip()
    head = repo_root / ".git" / "HEAD"
    if head.is_file():
        value = head.read_text(encoding="utf-8").strip()
        if not value.startswith("ref:"):
            return value
    return None


def build_snapshot() -> dict[str, Any]:
    import cv2
    import onnxruntime
    import torch
    import torchvision
    from boxmot.trackers.tracker_zoo import create_tracker
    from detectron2 import _C as detectron2_native
    from torchvision.ops import nms
    from whodoirunlike.runpod_serverless import handler

    importlib.import_module("densepose")
    importlib.import_module("rtmlib")
    importlib.import_module("sam3")

    # Exercise the Torchvision native extension without requiring a GPU.
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 9.0, 9.0]])
    scores = torch.tensor([0.9, 0.8])
    kept = nms(boxes, scores, 0.5).tolist()
    shallow_health = handler({"input": {"type": "health"}})

    assets: dict[str, Any] = {}
    for raw_path in EXPECTED_ASSETS:
        path = Path(raw_path)
        assets[raw_path] = {
            "exists": path.is_file(),
            "size": path.stat().st_size if path.is_file() else None,
            "sha256": _sha256(path) if path.is_file() else None,
        }

    installed_distributions = _distribution_versions()
    distribution_json = json.dumps(installed_distributions, separators=(",", ":"), sort_keys=True)
    distribution_fingerprint = hashlib.sha256((distribution_json + "\n").encode()).hexdigest()

    return {
        "schema": 1,
        "platform": {
            "machine": platform.machine(),
            "python_implementation": platform.python_implementation(),
            "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        },
        "runtime": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "torchvision": torchvision.__version__,
            "opencv": cv2.__version__,
            "onnxruntime": onnxruntime.__version__,
            "onnxruntime_providers": onnxruntime.get_available_providers(),
            "torchvision_nms_result": kept,
            "detectron2_native_module": str(getattr(detectron2_native, "__file__", "")),
            "boxmot_factory_callable": callable(create_tracker),
        },
        "distributions": installed_distributions,
        "distribution_fingerprint": distribution_fingerprint,
        "critical_distributions": {
            name: _version(name) for name in sorted(EXPECTED_DISTRIBUTIONS)
        },
        "module_origins": {
            name: _module_origin(name)
            for name in (
                "boxmot",
                "cv2",
                "densepose",
                "detectron2",
                "onnxruntime",
                "rtmlib",
                "sam3",
                "torch",
                "torchvision",
                "ultralytics",
                "whodoirunlike",
            )
        },
        "assets": assets,
        "trees": {name: _tree_fingerprint(root) for name, root in TREE_ROOTS.items()},
        "revisions": {
            "detectron2": _revision(Path("/opt/detectron2"), "detectron2"),
            "sam3": _revision(Path("/opt/sam3"), "sam3"),
        },
        "handler": {
            "status": shallow_health.get("status"),
            "ready_for_invocation": shallow_health.get("health", {}).get(
                "ready_for_invocation"
            ),
            "identity_backend": shallow_health.get("health", {}).get("identity_backend"),
            "pose_backend": shallow_health.get("health", {}).get("pose_backend"),
            "mask_backend": shallow_health.get("health", {}).get("mask_backend"),
        },
    }


def verify_snapshot(snapshot: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if snapshot["platform"]["machine"] != "x86_64":
        errors.append(f"expected x86_64, got {snapshot['platform']['machine']}")
    if snapshot["platform"]["python_major_minor"] != "3.12":
        errors.append(
            "expected Python 3.12, got " + snapshot["platform"]["python_major_minor"]
        )
    if snapshot["distribution_fingerprint"] != EXPECTED_DISTRIBUTION_FINGERPRINT:
        errors.append(
            "installed distribution set changed: expected "
            f"{EXPECTED_DISTRIBUTION_FINGERPRINT}, got "
            f"{snapshot['distribution_fingerprint']}"
        )
    actual_tree_fingerprints = {
        name: details.get("sha256") for name, details in snapshot["trees"].items()
    }
    if actual_tree_fingerprints != EXPECTED_TREE_FINGERPRINTS:
        errors.append(
            "runtime source trees changed: expected "
            f"{EXPECTED_TREE_FINGERPRINTS}, got {actual_tree_fingerprints}"
        )
    for name, expected in EXPECTED_DISTRIBUTIONS.items():
        actual = snapshot["critical_distributions"].get(name)
        if actual != expected:
            errors.append(f"{name}: expected {expected}, got {actual}")
    runtime = snapshot["runtime"]
    if runtime["torch_cuda"] != "12.8":
        errors.append(f"expected Torch CUDA 12.8, got {runtime['torch_cuda']}")
    if "CUDAExecutionProvider" not in runtime["onnxruntime_providers"]:
        errors.append("onnxruntime CUDAExecutionProvider is unavailable")
    if runtime["torchvision_nms_result"] != [0]:
        errors.append(
            "Torchvision native NMS contract changed: "
            f"{runtime['torchvision_nms_result']!r}"
        )
    if not runtime["boxmot_factory_callable"]:
        errors.append("BoxMOT tracker factory is not callable")
    for path, expected in EXPECTED_ASSETS.items():
        actual = snapshot["assets"].get(path, {})
        if actual.get("size") != expected["size"]:
            errors.append(f"{path}: expected {expected['size']} bytes, got {actual.get('size')}")
        if actual.get("sha256") != expected["sha256"]:
            errors.append(
                f"{path}: expected SHA-256 {expected['sha256']}, got {actual.get('sha256')}"
            )
    if snapshot["revisions"] != EXPECTED_REVISIONS:
        errors.append(
            f"source revisions changed: expected {EXPECTED_REVISIONS}, got {snapshot['revisions']}"
        )
    if snapshot["handler"] != {
        "status": "ok",
        "ready_for_invocation": True,
        "identity_backend": "boxmot_bytetrack",
        "pose_backend": "mmpose_rtmpose_l_384",
        "mask_backend": "sam31_gpu",
    }:
        errors.append(f"shallow handler health changed: {snapshot['handler']!r}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    snapshot = build_snapshot()
    encoded = json.dumps(snapshot, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    if args.verify:
        errors = verify_snapshot(snapshot)
        if errors:
            raise SystemExit("RunPod runtime contract failed:\n- " + "\n- ".join(errors))


if __name__ == "__main__":
    main()
