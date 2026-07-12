from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SLIM_DOCKERFILE = ROOT / "Dockerfile.runpod-slim"
CONTRACT_SCRIPT = ROOT / "scripts/runpod_runtime_contract.py"
WORKFLOW = ROOT / ".github/workflows/build-runpod-slim-lab.yml"
DONOR_DIGEST = "sha256:47d776f83ae3e2e1c7f1fa935b0019a9abd82a324ada4ab3d98746b3d75216fc"
RUNTIME_DIGEST = "sha256:9175fa92f96de35a8cfb9493f0dfcf9435c7a597e9d95ad41d2cae382a95e3f9"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_contract_module():
    spec = importlib.util.spec_from_file_location("runpod_runtime_contract", CONTRACT_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_slim_image_uses_only_immutable_amd64_images() -> None:
    dockerfile = _text(SLIM_DOCKERFILE)

    assert DONOR_DIGEST in dockerfile
    assert RUNTIME_DIGEST in dockerfile
    assert dockerfile.count("FROM --platform=linux/amd64") == 2
    assert ":latest" not in dockerfile
    assert "sha256:52392c71" not in dockerfile


def test_slim_image_copies_dependencies_but_not_donor_application() -> None:
    dockerfile = _text(SLIM_DOCKERFILE)

    for required in (
        "/usr/local/bin/ /usr/local/bin/",
        "/usr/local/lib/python3.12/ /usr/local/lib/python3.12/",
        "/usr/lib/python3/dist-packages/ /usr/lib/python3/dist-packages/",
        "/opt/detectron2/ /opt/detectron2/",
        "/opt/sam3/ /opt/sam3/",
        "/opt/rtmlib-cache/ /opt/rtmlib-cache/",
        "COPY pyproject.toml README.md ./",
        "COPY src ./src",
        "COPY site/public/assets/demos ./site/public/assets/demos",
    ):
        assert required in dockerfile
    assert "COPY --from=dependency-runtime /app" not in dockerfile
    assert "yolo26" not in dockerfile.lower()
    reset_marker = "rm -rf \\\n        /usr/local/lib/python3.12 \\\n        /usr/lib/python3/dist-packages"
    assert reset_marker in dockerfile
    assert dockerfile.index(reset_marker) < dockerfile.index(
        "COPY --from=dependency-runtime /usr/local/lib/python3.12/"
    )


def test_slim_image_preserves_quality_environment_without_override_knobs() -> None:
    dockerfile = _text(SLIM_DOCKERFILE)

    assert "MMPOSE_DEVICE=cpu" in dockerfile
    assert "RTMW_RUNTIME_BACKEND=onnxruntime" in dockerfile
    assert "DENSEPOSE_DEVICE=cuda" in dockerfile
    for required in (
        "WHODOIRUNLIKE_SAM31_GPU_EXACT_CV2_LOADER=true",
        "WHODOIRUNLIKE_SAM31_GPU_EXACT_CV2_CHUNK_FRAMES=8",
        "WHODOIRUNLIKE_SAM31_GPU_EXACT_CV2_MAX_FRAMES=600",
        "WHODOIRUNLIKE_SAM31_GPU_EXACT_CV2_MAX_DESTINATION_BYTES=8589934592",
        "WHODOIRUNLIKE_PROCESSOR_CONCURRENCY=1",
        "WHODOIRUNLIKE_PARALLEL_MASK_PRESENTATION=true",
        "WHODOIRUNLIKE_PARALLEL_POSE_DENSEPOSE=true",
        "WHODOIRUNLIKE_PARALLEL_POST_FUSION=true",
        "WHODOIRUNLIKE_PARALLEL_ARTIFACT_PUBLISH=true",
    ):
        assert required in dockerfile
    for forbidden in (
        "MMPOSE_USE_DETECTOR",
        "DENSEPOSE_TARGET_CROP",
        "DENSEPOSE_INPUT_MIN_SIZE_TEST",
        "DENSEPOSE_INPUT_MAX_SIZE_TEST",
        "onnxruntime-gpu",
    ):
        assert forbidden not in dockerfile


def test_slim_final_stage_has_runtime_libraries_without_build_toolchain() -> None:
    final = _text(SLIM_DOCKERFILE).split(" AS runtime\n", maxsplit=1)[1]

    for package in (
        "ca-certificates",
        "ffmpeg",
        "libgl1",
        "libglib2.0-0",
        "libgomp1",
        "python3.12",
    ):
        assert package in final
    for forbidden in (
        "build-essential",
        "git clone",
        "git fetch",
        "ninja-build",
        "python3.12-dev",
        "pip install",
        "curl ",
        "wget ",
    ):
        assert forbidden not in final


def test_contract_detects_dependency_and_application_drift() -> None:
    contract = _load_contract_module()
    baseline = {
        "schema": 2,
        "platform": {"python_major_minor": "3.12"},
        "runtime": {"torch_cuda": "12.8"},
        "distributions": [],
        "distribution_fingerprint": "same",
        "critical_distributions": {},
        "module_origins": {},
        "assets": {},
        "trees": {},
        "revisions": {},
    }

    assert contract._verify_dependency_baseline(baseline, dict(baseline)) == []
    changed = dict(baseline)
    changed["distribution_fingerprint"] = "different"
    assert contract._verify_dependency_baseline(baseline, changed) == [
        "dependency contract changed for distribution_fingerprint"
    ]

    application = {
        "module_origin": "/app/src/whodoirunlike/__init__.py",
        "processor_version": "test-sha",
        "environment": {
            **contract.EXPECTED_ENVIRONMENT,
            **{name: None for name in contract.FORBIDDEN_ENVIRONMENT},
        },
        "health": {
            "status": "ok",
            "health": {
                "ready_for_invocation": True,
                "identity_backend": "boxmot_bytetrack",
                "pose_backend": "mmpose_rtmpose_l_384",
                "mask_backend": "sam31_gpu",
                "sam31_input_loader": {
                    "mode": "exact_cv2",
                    "configured_concurrency": 1,
                    "concurrency_ready": True,
                },
            },
        },
        "yolo26_paths": [],
        "source_tree": {"sha256": "source"},
    }
    assert contract._verify_application(application) == []
    application["environment"]["MMPOSE_USE_DETECTOR"] = "false"
    assert "forbidden quality override is present: MMPOSE_USE_DETECTOR" in (
        contract._verify_application(application)
    )


def test_manual_lab_workflow_never_tags_or_deploys_production() -> None:
    workflow = _text(WORKFLOW)

    assert "workflow_dispatch:" in workflow
    assert "Dockerfile.runpod-slim" in workflow
    assert "platforms: linux/amd64" in workflow
    assert "${{ steps.build.outputs.digest }}" in workflow
    assert (
        "LAB_IMAGE: ghcr.io/akhil-ghosh/whodoirunlike-runpod-processor@"
        "${{ needs.build.outputs.digest }}"
    ) in workflow
    assert "LAB_IMAGE: ${{ env.IMAGE_NAME }}" not in workflow
    assert "--network none" in workflow
    assert DONOR_DIGEST in workflow
    assert 'docker manifest inspect --verbose "${DONOR_IMAGE}"' in workflow
    assert 'donor_bytes="$(jq' in workflow
    assert "DONOR_COMPRESSED_BYTES" not in workflow
    assert ":latest" not in workflow
    for forbidden in ("runpodctl", "serverless update", "wrangler deploy", "production"):
        assert forbidden not in workflow.lower()
