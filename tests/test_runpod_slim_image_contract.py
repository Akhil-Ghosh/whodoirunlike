from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SLIM_DOCKERFILE = REPO_ROOT / "Dockerfile.runpod-slim"
CURRENT_DOCKERFILE = REPO_ROOT / "Dockerfile.runpod"
CANDIDATE_DIGEST = (
    "sha256:52392c71b9a44d804f0ba7fcd894247988319dbe98e6160cb05d70a13894714a"
)
RUNTIME_DIGEST = (
    "sha256:9175fa92f96de35a8cfb9493f0dfcf9435c7a597e9d95ad41d2cae382a95e3f9"
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _final_stage(text: str) -> str:
    marker = " AS runtime\n"
    assert marker in text
    return text.split(marker, maxsplit=1)[1]


def _environment_contract(text: str) -> set[str]:
    names = {
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONUNBUFFERED",
        "HF_HOME",
        "TORCH_HOME",
        "WHODOIRUNLIKE_HOSTED_RUN_ROOT",
        "WHODOIRUNLIKE_IDENTITY_BACKEND",
        "WHODOIRUNLIKE_POSE_BACKEND",
        "WHODOIRUNLIKE_MASK_BACKEND",
        "WHODOIRUNLIKE_SAM31_GPU_USE_FA3",
        "WHODOIRUNLIKE_SAM31_GPU_CACHE_PREDICTOR",
        "WHODOIRUNLIKE_SAM31_GPU_PRESEED_ANCHORS",
        "WHODOIRUNLIKE_SKIP_DENSEPOSE",
        "WHODOIRUNLIKE_PROCESSOR_VERSION",
        "MMPOSE_DEVICE",
        "RTMW_RUNTIME_BACKEND",
        "DENSEPOSE_CONFIG",
        "DENSEPOSE_WEIGHTS",
        "DENSEPOSE_DEVICE",
    }
    found: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip().removesuffix("\\").strip()
        for name in names:
            if stripped.startswith(f"{name}="):
                found.add(stripped)
    return found


def test_slim_image_uses_immutable_amd64_candidate_and_runtime() -> None:
    text = _text(SLIM_DOCKERFILE)

    assert f"whodoirunlike-runpod-processor@{CANDIDATE_DIGEST}" in text
    assert f"nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04@{RUNTIME_DIGEST}" in text
    assert text.count("FROM --platform=linux/amd64") == 2
    assert ":latest" not in text


def test_slim_image_preserves_handler_environment_and_command() -> None:
    current = _text(CURRENT_DOCKERFILE)
    slim = _text(SLIM_DOCKERFILE)

    assert _environment_contract(slim) == _environment_contract(current)
    assert 'CMD ["python", "-m", "whodoirunlike.runpod_serverless"]' in slim
    assert "WORKDIR /app" in slim


def test_slim_image_copies_every_runtime_and_model_contract() -> None:
    text = _text(SLIM_DOCKERFILE)
    required_copies = {
        "/usr/local/bin/ /usr/local/bin/",
        "/usr/local/lib/python3.12/ /usr/local/lib/python3.12/",
        "/usr/lib/python3/dist-packages/ /usr/lib/python3/dist-packages/",
        "/app/ /app/",
        "/opt/detectron2/ /opt/detectron2/",
        "/opt/sam3/ /opt/sam3/",
        "/opt/rtmlib-cache/ /opt/rtmlib-cache/",
        "/opt/whodoirunlike-provenance/ /opt/whodoirunlike-provenance/",
    }

    assert all(copy in text for copy in required_copies)
    assert "dist-packages/[" not in text
    assert "dist-packages/*" not in text
    assert "rm -rf /opt/detectron2/.git /opt/sam3/.git" in text
    assert "runtime-contract.py" in text
    assert "--verify" in text


def test_final_stage_has_runtime_libraries_but_no_build_toolchain_or_fetcher() -> None:
    final = _final_stage(_text(SLIM_DOCKERFILE))
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


def test_image_definition_does_not_bake_credentials() -> None:
    text = _text(SLIM_DOCKERFILE)
    forbidden_assignments = (
        "HF_TOKEN=",
        "HUGGING_FACE_HUB_TOKEN=",
        "WHODOIRUNLIKE_PROCESSOR_SHARED_SECRET=",
        "AWS_ACCESS_KEY_ID=",
        "AWS_SECRET_ACCESS_KEY=",
        "RUNPOD_API_KEY=",
    )

    assert not any(assignment in text for assignment in forbidden_assignments)
