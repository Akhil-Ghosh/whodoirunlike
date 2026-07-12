#!/usr/bin/env python3
"""Capture and verify the slim RunPod processor runtime contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import site
import sys
from importlib.metadata import PackageNotFoundError, distributions, version
from pathlib import Path
from typing import Any


SCHEMA = 2
DEPENDENCY_TREE_ROOTS = {
    "detectron2": Path("/opt/detectron2"),
    "sam3": Path("/opt/sam3"),
    "rtmlib_cache": Path("/opt/rtmlib-cache"),
}
IGNORED_TREE_PARTS = {".git", "__pycache__"}
MODEL_ASSETS = {
    "/opt/densepose-weights/model_final_162be9.pkl": {
        "size": 255_757_821,
        "sha256": "b8a7382001b16e453bad95ca9dbc68ae8f2b839b304cf90eaf5c27fbdb4dae91",
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
EXPECTED_ENVIRONMENT = {
    "WHODOIRUNLIKE_IDENTITY_BACKEND": "boxmot_bytetrack",
    "WHODOIRUNLIKE_POSE_BACKEND": "mmpose_rtmpose_l_384",
    "WHODOIRUNLIKE_MASK_BACKEND": "sam31_gpu",
    "WHODOIRUNLIKE_SAM31_GPU_USE_FA3": "false",
    "WHODOIRUNLIKE_SAM31_GPU_CACHE_PREDICTOR": "true",
    "WHODOIRUNLIKE_SAM31_GPU_PRESEED_ANCHORS": "true",
    "WHODOIRUNLIKE_SAM31_GPU_EXACT_CV2_LOADER": "true",
    "WHODOIRUNLIKE_SAM31_GPU_EXACT_CV2_CHUNK_FRAMES": "8",
    "WHODOIRUNLIKE_SAM31_GPU_EXACT_CV2_MAX_FRAMES": "600",
    "WHODOIRUNLIKE_SAM31_GPU_EXACT_CV2_MAX_DESTINATION_BYTES": "8589934592",
    "WHODOIRUNLIKE_PROCESSOR_CONCURRENCY": "1",
    "WHODOIRUNLIKE_PARALLEL_MASK_PRESENTATION": "true",
    "WHODOIRUNLIKE_PARALLEL_POSE_DENSEPOSE": "true",
    "WHODOIRUNLIKE_PARALLEL_POST_FUSION": "true",
    "WHODOIRUNLIKE_PARALLEL_ARTIFACT_PUBLISH": "true",
    "WHODOIRUNLIKE_SKIP_DENSEPOSE": "false",
    "MMPOSE_DEVICE": "cpu",
    "RTMW_RUNTIME_BACKEND": "onnxruntime",
    "DENSEPOSE_WEIGHTS": "/opt/densepose-weights/model_final_162be9.pkl",
    "DENSEPOSE_DEVICE": "cuda",
}
FORBIDDEN_ENVIRONMENT = (
    "MMPOSE_USE_DETECTOR",
    "DENSEPOSE_TARGET_CROP",
    "DENSEPOSE_TARGET_CROP_ENABLED",
    "DENSEPOSE_INPUT_MIN_SIZE_TEST",
    "DENSEPOSE_INPUT_MAX_SIZE_TEST",
)
DEPENDENCY_MODULES = (
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
)


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
        size = path.stat().st_size
        digest.update(relative.as_posix().encode())
        digest.update(b"\0")
        digest.update(str(size).encode())
        digest.update(b"\0")
        digest.update(_sha256(path).encode())
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
        if name:
            found.append(
                {
                    "name": name.lower().replace("_", "-"),
                    "version": distribution.version,
                }
            )
    return sorted(found, key=lambda item: (item["name"], item["version"]))


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _module_origin(name: str) -> str | None:
    spec = importlib.util.find_spec(name)
    return str(spec.origin) if spec and spec.origin else None


def _revision(name: str) -> str | None:
    path = Path("/opt/whodoirunlike-provenance") / f"{name}.revision"
    return path.read_text(encoding="utf-8").strip() if path.is_file() else None


def _asset_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for raw_path in MODEL_ASSETS:
        path = Path(raw_path)
        snapshot[raw_path] = {
            "exists": path.is_file(),
            "size": path.stat().st_size if path.is_file() else None,
            "sha256": _sha256(path) if path.is_file() else None,
        }
    return snapshot


def capture_dependencies() -> dict[str, Any]:
    import cv2
    import onnxruntime
    import torch
    import torchvision
    from boxmot.trackers.tracker_zoo import create_tracker
    from detectron2 import _C as detectron2_native
    from torchvision.ops import nms

    for name in ("densepose", "rtmlib", "sam3"):
        importlib.import_module(name)

    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 9.0, 9.0]])
    scores = torch.tensor([0.9, 0.8])
    installed_distributions = _distribution_versions()
    distribution_json = json.dumps(
        installed_distributions,
        separators=(",", ":"),
        sort_keys=True,
    )
    return {
        "schema": SCHEMA,
        "platform": {
            "machine": platform.machine(),
            "python_implementation": platform.python_implementation(),
            "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
            "site_packages": site.getsitepackages(),
        },
        "runtime": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "torchvision": torchvision.__version__,
            "opencv": cv2.__version__,
            "onnxruntime": onnxruntime.__version__,
            "onnxruntime_providers": onnxruntime.get_available_providers(),
            "torchvision_nms_result": nms(boxes, scores, 0.5).tolist(),
            "detectron2_native_module": str(getattr(detectron2_native, "__file__", "")),
            "boxmot_factory_callable": callable(create_tracker),
        },
        "distributions": installed_distributions,
        "distribution_fingerprint": hashlib.sha256(
            (distribution_json + "\n").encode()
        ).hexdigest(),
        "critical_distributions": {
            name: _version(name)
            for name in (
                "boxmot",
                "detectron2",
                "onnxruntime",
                "onnxruntime-gpu",
                "opencv-contrib-python-headless",
                "rtmlib",
                "torch",
                "torchvision",
                "ultralytics",
            )
        },
        "module_origins": {name: _module_origin(name) for name in DEPENDENCY_MODULES},
        "assets": _asset_snapshot(),
        "trees": {
            name: _tree_fingerprint(root) for name, root in DEPENDENCY_TREE_ROOTS.items()
        },
        "revisions": {
            "detectron2": _revision("detectron2"),
            "sam3": _revision("sam3"),
        },
    }


def _forbidden_yolo26_paths() -> list[str]:
    found: list[str] = []
    for root in (Path("/app"), Path("/opt"), Path("/root/.cache")):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if "yolo26" in path.name.lower():
                found.append(str(path))
    return sorted(found)


def capture_application() -> dict[str, Any]:
    from whodoirunlike.runpod_serverless import handler

    health = handler({"input": {"type": "health"}})
    return {
        "module_origin": _module_origin("whodoirunlike"),
        "processor_version": os.getenv("WHODOIRUNLIKE_PROCESSOR_VERSION"),
        "environment": {
            name: os.environ.get(name)
            for name in (*EXPECTED_ENVIRONMENT, *FORBIDDEN_ENVIRONMENT)
        },
        "health": health,
        "yolo26_paths": _forbidden_yolo26_paths(),
        "source_tree": _tree_fingerprint(Path("/app/src/whodoirunlike")),
    }


def _verify_dependency_baseline(
    baseline: dict[str, Any],
    current: dict[str, Any],
) -> list[str]:
    def distribution_labels(value: Any) -> set[str]:
        if not isinstance(value, list):
            return set()
        return {
            f"{item['name']}=={item['version']}"
            for item in value
            if isinstance(item, dict)
            and isinstance(item.get("name"), str)
            and isinstance(item.get("version"), str)
        }

    def summarize(labels: set[str]) -> str:
        ordered = sorted(labels)
        visible = ordered[:20]
        suffix = f", +{len(ordered) - len(visible)} more" if len(ordered) > len(visible) else ""
        return ",".join(visible) + suffix if visible else "none"

    errors: list[str] = []
    for key in (
        "schema",
        "platform",
        "runtime",
        "distributions",
        "distribution_fingerprint",
        "critical_distributions",
        "module_origins",
        "assets",
        "trees",
        "revisions",
    ):
        if current.get(key) == baseline.get(key):
            continue
        if key == "distributions":
            expected = distribution_labels(baseline.get(key))
            observed = distribution_labels(current.get(key))
            errors.append(
                "dependency contract changed for distributions "
                f"(missing={summarize(expected - observed)}; "
                f"unexpected={summarize(observed - expected)})"
            )
        else:
            errors.append(f"dependency contract changed for {key}")
    return errors


def _verify_static_dependencies(snapshot: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    platform_snapshot = snapshot["platform"]
    if platform_snapshot["machine"] != "x86_64":
        errors.append(f"expected x86_64, got {platform_snapshot['machine']}")
    if platform_snapshot["python_major_minor"] != "3.12":
        errors.append(
            f"expected Python 3.12, got {platform_snapshot['python_major_minor']}"
        )

    runtime = snapshot["runtime"]
    if runtime["torch_cuda"] != "12.8":
        errors.append(f"expected Torch CUDA 12.8, got {runtime['torch_cuda']}")
    providers = runtime["onnxruntime_providers"]
    if "CPUExecutionProvider" not in providers:
        errors.append("ONNX Runtime CPUExecutionProvider is unavailable")
    for forbidden_provider in ("CUDAExecutionProvider", "TensorrtExecutionProvider"):
        if forbidden_provider in providers:
            errors.append(f"unexpected ONNX Runtime provider: {forbidden_provider}")
    if snapshot["critical_distributions"].get("onnxruntime-gpu") is not None:
        errors.append("onnxruntime-gpu must not be installed")
    if snapshot["critical_distributions"].get("onnxruntime") is None:
        errors.append("CPU onnxruntime distribution is missing")
    if runtime["torchvision_nms_result"] != [0]:
        errors.append(
            "Torchvision native NMS contract changed: "
            f"{runtime['torchvision_nms_result']!r}"
        )
    if not runtime["boxmot_factory_callable"]:
        errors.append("BoxMOT tracker factory is not callable")
    if not runtime["detectron2_native_module"]:
        errors.append("Detectron2 native extension is unavailable")

    for path, expected in MODEL_ASSETS.items():
        actual = snapshot["assets"].get(path, {})
        if actual.get("size") != expected["size"]:
            errors.append(f"{path}: unexpected size {actual.get('size')}")
        if actual.get("sha256") != expected["sha256"]:
            errors.append(f"{path}: unexpected SHA-256 {actual.get('sha256')}")
    for name, details in snapshot["trees"].items():
        if not details.get("exists") or not details.get("sha256"):
            errors.append(f"dependency tree is missing: {name}")
    for name, revision in snapshot["revisions"].items():
        if not revision:
            errors.append(f"source revision is missing: {name}")
    return errors


def _verify_application(snapshot: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    module_origin = str(snapshot.get("module_origin") or "")
    if not module_origin.startswith("/app/src/whodoirunlike/"):
        errors.append(f"application imported from unexpected path: {module_origin!r}")
    if not snapshot.get("processor_version"):
        errors.append("WHODOIRUNLIKE_PROCESSOR_VERSION is empty")
    environment = snapshot["environment"]
    for name, expected in EXPECTED_ENVIRONMENT.items():
        if environment.get(name) != expected:
            errors.append(f"{name}: expected {expected!r}, got {environment.get(name)!r}")
    for name in FORBIDDEN_ENVIRONMENT:
        if environment.get(name) is not None:
            errors.append(f"forbidden quality override is present: {name}")

    health = snapshot["health"]
    health_details = health.get("health", {})
    if health.get("status") != "ok":
        errors.append(f"shallow handler health failed: {health!r}")
    if health_details.get("ready_for_invocation") is not True:
        errors.append(f"processor is not ready for invocation: {health_details!r}")
    loader_health = health_details.get("sam31_input_loader", {})
    if loader_health.get("mode") != "exact_cv2":
        errors.append(f"exact SAM loader is not enabled: {loader_health!r}")
    if loader_health.get("configured_concurrency") != 1:
        errors.append(f"SAM loader concurrency is not one: {loader_health!r}")
    if loader_health.get("concurrency_ready") is not True:
        errors.append(f"SAM loader concurrency policy is not ready: {loader_health!r}")
    for key, expected in (
        ("identity_backend", "boxmot_bytetrack"),
        ("pose_backend", "mmpose_rtmpose_l_384"),
        ("mask_backend", "sam31_gpu"),
    ):
        if health_details.get(key) != expected:
            errors.append(f"handler {key}: expected {expected!r}, got {health_details.get(key)!r}")
    if snapshot["yolo26_paths"]:
        errors.append(f"YOLO26 artifacts are forbidden: {snapshot['yolo26_paths']!r}")
    if not snapshot["source_tree"].get("sha256"):
        errors.append("application source tree is missing")
    return errors


def _write(snapshot: dict[str, Any], output: Path | None) -> None:
    encoded = json.dumps(snapshot, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(encoded, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dependencies", action="store_true")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    dependencies = capture_dependencies()
    if args.capture_dependencies:
        if args.baseline or args.verify:
            parser.error("--capture-dependencies cannot be combined with --baseline or --verify")
        _write(dependencies, args.output)
        return

    application = capture_application()
    result = {
        "schema": SCHEMA,
        "dependencies": dependencies,
        "application": application,
    }
    errors = [*_verify_static_dependencies(dependencies), *_verify_application(application)]
    if args.baseline is not None:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        errors.extend(_verify_dependency_baseline(baseline, dependencies))
    elif args.verify:
        errors.append("--verify requires --baseline")

    result["verified"] = not errors
    result["errors"] = errors
    _write(result, args.output)
    if args.verify and errors:
        raise SystemExit("RunPod runtime contract failed:\n- " + "\n- ".join(errors))


if __name__ == "__main__":
    main()
