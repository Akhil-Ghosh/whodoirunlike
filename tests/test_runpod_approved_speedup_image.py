from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile.runpod-approved-speedup"
WORKFLOW = ROOT / ".github/workflows/build-runpod-approved-speedup.yml"
PRODUCTION_DIGEST = (
    "sha256:47d776f83ae3e2e1c7f1fa935b0019a9abd82a324ada4ab3d98746b3d75216fc"
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_release_image_overlays_only_approved_source_on_production_runtime() -> None:
    dockerfile = _text(DOCKERFILE)

    assert f"whodoirunlike-runpod-processor@{PRODUCTION_DIGEST}" in dockerfile
    assert "COPY src/whodoirunlike/ /app/src/whodoirunlike/" in dockerfile
    for forbidden in (
        "COPY requirements-runpod-processor.txt",
        "pip install",
        "apt-get",
        "DENSEPOSE_INPUT_MIN_SIZE_TEST",
        "DENSEPOSE_INPUT_MAX_SIZE_TEST",
        "DENSEPOSE_TARGET_CROP",
        "MMPOSE_USE_DETECTOR",
        "yolo26",
    ):
        assert forbidden.lower() not in dockerfile.lower()


def test_release_image_enables_only_the_approved_speed_paths() -> None:
    dockerfile = _text(DOCKERFILE)

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


def test_release_workflow_builds_and_smokes_only_an_immutable_digest() -> None:
    workflow = _text(WORKFLOW)

    assert "workflow_dispatch:" in workflow
    assert "Dockerfile.runpod-approved-speedup" in workflow
    assert "platforms: linux/amd64" in workflow
    assert "${{ steps.build.outputs.digest }}" in workflow
    assert PRODUCTION_DIGEST in workflow
    assert ":latest" not in workflow
    assert "runpodctl" not in workflow
    assert "wrangler deploy" not in workflow
    assert "production" not in workflow.lower().replace("production_digest", "")
