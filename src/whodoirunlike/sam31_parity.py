from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping


CANONICAL_FRAME130_FIXTURE_ID = "cole-frame130-production-v1"
EXACT_CANDIDATE_COMMIT = "657d36588f3ca073554bfd40071ff747e0e750bb"
EXACT_CANDIDATE_IMAGE_DIGEST = (
    "sha256:52392c71b9a44d804f0ba7fcd894247988319dbe98e6160cb05d70a13894714a"
)
EXACT_CONTROL_COMMIT = "fd35d9cf56e9f1271380575149a3e72afec31344"
EXACT_CONTROL_IMAGE_DIGEST = (
    "sha256:47d776f83ae3e2e1c7f1fa935b0019a9abd82a324ada4ab3d98746b3d75216fc"
)
FINAL_CANDIDATE_IMAGE_REPOSITORY = (
    "ghcr.io/akhil-ghosh/whodoirunlike-runpod-processor"
)
FINAL_CANDIDATE_CONTRACT_ENV = "WHODOIRUNLIKE_FINAL_PRODUCTION_SHA256_B64"
DENSEPOSE_WEIGHTS_PATH = Path("/opt/densepose-weights/model_final_162be9.pkl")
DENSEPOSE_WEIGHTS_SIZE = 255_757_821
DENSEPOSE_WEIGHTS_SHA256 = (
    "b8a7382001b16e453bad95ca9dbc68ae8f2b839b304cf90eaf5c27fbdb4dae91"
)
FINAL_FORBIDDEN_QUALITY_ENVIRONMENT = (
    "MMPOSE_USE_DETECTOR",
    "DENSEPOSE_TARGET_CROP",
    "DENSEPOSE_TARGET_CROP_ENABLED",
    "DENSEPOSE_TARGET_CROP_PADDING_RATIO",
    "DENSEPOSE_TARGET_CROP_PADDING_PIXELS",
    "DENSEPOSE_INPUT_MIN_SIZE_TEST",
    "DENSEPOSE_INPUT_MAX_SIZE_TEST",
)
_EXACT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_EXACT_OCI_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_EXACT_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_BASE_NON_OVERLAY_PRODUCTION_SHA256 = MappingProxyType(
    {
        "control": MappingProxyType(
            {
                "full_pipeline.py": "519de206085ea60bdc76652bcdfd1228fe38eb380af10d5f4e2ad61e8250b8ff",
                "sam31_gpu_runner.py": "cc12ccadcd209451e7fa106bfde58a6a10f100d48528862de8262f0d494710fa",
                "sam31_mlx_runner.py": "5235da2c165cc8fff8df7cfacde229f5e6df750b58a107193cf76d7c52b540ed",
                "sam2_runner.py": "71a07013f98c8335606df3e7cffa0c7d8b33c1ee16d79a5684ff1b7246e4ef20",
                "mask_artifacts.py": "682286c9a59c7338579e80c903f1657917d5e99a50a6202ef040d8ddb4999714",
                "mmpose_runner.py": "4bc266c122a6a83d4fa604b83243d1823ddc8208064b6fc1a5a93b9ab1c99797",
                "densepose_runner.py": "7cdd4b5714b98a08ece74a2f6a29241630129b06ae578db1e748addb37cdca8c",
                "fusion_runner.py": "a50e0fe9e18ae90ea76fb716fc1af0d7a9892547ea5ba9e77cd0faeb6cb4f67f",
                "form_features.py": "cfdae5b2757016e7c98a0f77140b4568a62b27477335b84aa98da5c2bb0d3d89",
                "artifact_tables.py": "d0d055b0159b966ddf0a7d46e24dc2f4d7891e57407c4638ca5fb99fa66de23d",
                "qc.py": "6b1358d911c63ab0fda4da3c8f28189e8bdb3d9617ed4f9306bc2d8649b8bf13",
                "running_clip_run.py": "2a97a98b6ab4bcb8664d84c6e47ce7992437347bd2ea8f942601b12e43d3a25e",
            }
        ),
        "candidate": MappingProxyType(
            {
                "full_pipeline.py": "0238de7bb37d960c41c202a001525c71ccdad1b799a31834da6fb41eac913afb",
                "sam31_gpu_runner.py": "593822f192689bbd55c5727cfa4693e6e4e1deccbe90344d87034f03dc0b9c9e",
                "sam31_mlx_runner.py": "5235da2c165cc8fff8df7cfacde229f5e6df750b58a107193cf76d7c52b540ed",
                "sam2_runner.py": "813be60802a9aea5137fdc2922f02070a35b2cf18d64c3cfb14c75fc01ca5667",
                "mask_artifacts.py": "682286c9a59c7338579e80c903f1657917d5e99a50a6202ef040d8ddb4999714",
                "mmpose_runner.py": "d32fc621c8dfcdd065f8ccfdb97250f6ada5df55885701d2e766e9388ed7036e",
                "densepose_runner.py": "e84ba8623db1244b493a3a17c2edf92359045aa4f1223047f5971ca35a881f9b",
                "fusion_runner.py": "a50e0fe9e18ae90ea76fb716fc1af0d7a9892547ea5ba9e77cd0faeb6cb4f67f",
                "form_features.py": "cfdae5b2757016e7c98a0f77140b4568a62b27477335b84aa98da5c2bb0d3d89",
                "artifact_tables.py": "d0d055b0159b966ddf0a7d46e24dc2f4d7891e57407c4638ca5fb99fa66de23d",
                "qc.py": "4f21561145d427d295b149cbf4e5dfaf146e206f423151833beb580bfba04a91",
                "running_clip_run.py": "931435c93c67c34453603615e6f3b5259e614038134dd4e210bf1ed79ea204de",
            }
        ),
    }
)
NON_OVERLAY_PRODUCTION_SHA256 = MappingProxyType(
    {
        "control": _BASE_NON_OVERLAY_PRODUCTION_SHA256["control"],
        # The schedule-only image uses the exact live control dependency/model base,
        # then copies the immutable 657 candidate Python source tree into /app/src.
        "schedule_only": _BASE_NON_OVERLAY_PRODUCTION_SHA256["candidate"],
    }
)
FINAL_CANDIDATE_PRODUCTION_FILES = tuple(
    dict.fromkeys(
        (
            *_BASE_NON_OVERLAY_PRODUCTION_SHA256["candidate"],
            "identity_runner.py",
            "processing_telemetry.py",
            "sam31_cv2_loader.py",
            "sam31_loader_config.py",
            "video_io.py",
        )
    )
)


def verify_densepose_weights(
    path: Path = DENSEPOSE_WEIGHTS_PATH,
) -> dict[str, Any]:
    exists = path.is_file()
    observed_size = path.stat().st_size if exists else None
    observed_sha256: str | None = None
    if exists:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        observed_sha256 = digest.hexdigest()
    checks = {
        "path_exact": path == DENSEPOSE_WEIGHTS_PATH,
        "file_exists": exists,
        "size_exact": observed_size == DENSEPOSE_WEIGHTS_SIZE,
        "sha256_exact": observed_sha256 == DENSEPOSE_WEIGHTS_SHA256,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "path": str(path),
        "expected_size": DENSEPOSE_WEIGHTS_SIZE,
        "observed_size": observed_size,
        "expected_sha256": DENSEPOSE_WEIGHTS_SHA256,
        "observed_sha256": observed_sha256,
    }


def validate_final_candidate_identity(
    *,
    commit: str,
    image_digest: str,
    image_reference: str,
) -> dict[str, Any]:
    expected_reference = f"{FINAL_CANDIDATE_IMAGE_REPOSITORY}@{image_digest}"
    checks = {
        "commit_is_exact": bool(_EXACT_COMMIT_PATTERN.fullmatch(commit)),
        "digest_is_exact": bool(_EXACT_OCI_DIGEST_PATTERN.fullmatch(image_digest)),
        "reference_is_digest_pinned": image_reference == expected_reference,
        "reference_has_no_tag": ":" not in image_reference.rsplit("/", 1)[-1].split("@", 1)[0],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "commit": commit,
        "image_digest": image_digest,
        "image_reference": image_reference,
        "expected_image_reference": expected_reference,
    }


def _final_candidate_production_contract(encoded_contract: str) -> Mapping[str, str]:
    if not encoded_contract:
        raise ValueError(f"{FINAL_CANDIDATE_CONTRACT_ENV} is required for final parity.")
    try:
        raw_contract = base64.b64decode(
            encoded_contract.encode("ascii"),
            altchars=b"-_",
            validate=True,
        )
        payload = json.loads(raw_contract)
    except (UnicodeEncodeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Final candidate production hash contract is malformed.") from exc
    if not isinstance(payload, dict) or set(payload) != set(FINAL_CANDIDATE_PRODUCTION_FILES):
        raise ValueError("Final candidate production hash contract has an invalid file set.")
    contract: dict[str, str] = {}
    for name in FINAL_CANDIDATE_PRODUCTION_FILES:
        digest = payload.get(name)
        if not isinstance(digest, str) or not _EXACT_SHA256_PATTERN.fullmatch(digest):
            raise ValueError("Final candidate production hash contract has an invalid digest.")
        contract[name] = digest
    return MappingProxyType(contract)


def verify_non_overlay_production_files(
    image_role: str,
    *,
    module_root: Path | None = None,
) -> dict[str, Any]:
    if image_role == "final_candidate":
        import os

        expected = _final_candidate_production_contract(
            os.getenv(FINAL_CANDIDATE_CONTRACT_ENV, "")
        )
        contract_source = "workflow_candidate_commit"
    else:
        try:
            expected = NON_OVERLAY_PRODUCTION_SHA256[image_role]
        except KeyError as exc:
            raise ValueError(f"Unsupported benchmark image role: {image_role}") from exc
        contract_source = "registered_fixture"
    root = Path(module_root) if module_root is not None else Path(__file__).resolve().parent
    observed: dict[str, str | None] = {}
    mismatches: list[str] = []
    for name, expected_sha256 in expected.items():
        path = root / name
        digest = _sha256(path.read_bytes()) if path.is_file() else None
        observed[name] = digest
        if digest != expected_sha256:
            mismatches.append(name)
    return {
        "passed": not mismatches,
        "image_role": image_role,
        "contract_source": contract_source,
        "contract_sha256": _sha256(
            json.dumps(dict(expected), separators=(",", ":"), sort_keys=True).encode("utf-8")
        ),
        "checked_file_count": len(expected),
        "mismatches": mismatches,
        "expected_sha256": dict(expected),
        "observed_sha256": observed,
    }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class PromptSpec:
    payload: dict[str, Any]
    raw_sha256: str
    semantic_sha256: str

    def canonical_bytes(self) -> bytes:
        return (json.dumps(self.payload, indent=2) + "\n").encode("utf-8")


@dataclass(frozen=True)
class ParityFixture:
    fixture_id: str
    baseline_run_id: str
    baseline_attempt_id: str
    baseline_processor_version: str
    prompt: PromptSpec
    source_sha256: str
    asset_sha256: Mapping[str, str]
    frame_count: int
    width: int
    height: int
    local_asset_paths: Mapping[str, Path]


@dataclass(frozen=True)
class MaskQualityMeasurements:
    reference_frame_count: int
    candidate_frame_count: int
    candidate_nonempty_frame_count: int
    iou_mean: float | None
    iou_p05: float | None
    boundary_f1_mean: float | None
    centroid_error_normalized_mean: float | None
    temporal_iou_absolute_delta: float | None
    fallback_used: bool
    box_loss_frame_count: int
    identity_switch_count: int


STRICT_MASK_GATE_THRESHOLDS = MappingProxyType(
    {
        "iou_mean_min": 0.985,
        "iou_p05_min": 0.975,
        "boundary_f1_mean_min": 0.99,
        "centroid_error_normalized_mean_max": 0.002,
        "temporal_iou_absolute_delta_max": 0.01,
        "fallback_used": False,
        "box_loss_frame_count_max": 0,
        "identity_switch_count_max": 0,
    }
)


def evaluate_strict_mask_gate(
    measurements: MaskQualityMeasurements,
    *,
    expected_frame_count: int,
) -> dict[str, Any]:
    if expected_frame_count <= 0:
        raise ValueError("Strict mask gate expected_frame_count must be positive.")

    def at_least(value: float | None, minimum: float) -> bool:
        return value is not None and value >= minimum

    def at_most(value: float | None, maximum: float) -> bool:
        return value is not None and value <= maximum

    checks = {
        "reference_frame_count_exact": measurements.reference_frame_count == expected_frame_count,
        "candidate_frame_count_exact": measurements.candidate_frame_count == expected_frame_count,
        "candidate_nonempty_frame_count_exact": measurements.candidate_nonempty_frame_count
        == expected_frame_count,
        "iou_mean": at_least(
            measurements.iou_mean,
            float(STRICT_MASK_GATE_THRESHOLDS["iou_mean_min"]),
        ),
        "iou_p05": at_least(
            measurements.iou_p05,
            float(STRICT_MASK_GATE_THRESHOLDS["iou_p05_min"]),
        ),
        "boundary_f1_mean": at_least(
            measurements.boundary_f1_mean,
            float(STRICT_MASK_GATE_THRESHOLDS["boundary_f1_mean_min"]),
        ),
        "centroid_error_normalized_mean": at_most(
            measurements.centroid_error_normalized_mean,
            float(STRICT_MASK_GATE_THRESHOLDS["centroid_error_normalized_mean_max"]),
        ),
        "temporal_iou_absolute_delta": at_most(
            measurements.temporal_iou_absolute_delta,
            float(STRICT_MASK_GATE_THRESHOLDS["temporal_iou_absolute_delta_max"]),
        ),
        "no_fallback": not measurements.fallback_used,
        "no_box_loss": measurements.box_loss_frame_count == 0,
        "no_identity_switches": measurements.identity_switch_count == 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "expected_frame_count": expected_frame_count,
            **dict(STRICT_MASK_GATE_THRESHOLDS),
        },
    }


_FRAME130_PROMPT = {
    "version": 1,
    "source": "hosted_upload_demo_profile_v1",
    "selection": {
        "type": "reference_box",
        "positive_points": [],
        "negative_points": [],
        "box": {
            "x": 0.624283,
            "y": 0.175162,
            "width": 0.182908,
            "height": 0.772011,
        },
    },
    "frame": {
        "frame_index": 130,
        "time_seconds": 4.338,
        "image_path": (
            "/runpod-volume/whodoirunlike/runs/"
            "2fa255e3-aefe-4fb2-b41e-f36c73c09546/prompt_frame.jpg"
        ),
        "height": 540,
        "width": 960,
    },
    "subject": {
        "runner_name": "Cole Hocker",
        "profile_id": "cole_hocker_reference_v1",
    },
    "notes": "Seeded from the validated local reference run for this public Cole Hocker demo clip.",
}


def _prompt_semantic_bytes(payload: dict[str, Any]) -> bytes:
    semantic_payload = json.loads(json.dumps(payload))
    frame = semantic_payload.get("frame")
    if isinstance(frame, dict):
        frame.pop("image_path", None)
    return json.dumps(semantic_payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


_FRAME130_FIXTURE = ParityFixture(
    fixture_id=CANONICAL_FRAME130_FIXTURE_ID,
    baseline_run_id="2fa255e3-aefe-4fb2-b41e-f36c73c09546",
    baseline_attempt_id="46f7dbe7-6899-46ef-a971-9a7261a37480",
    baseline_processor_version="8b33d07cd129cb6878f4630af133bf30c291914b",
    prompt=PromptSpec(
        payload=_FRAME130_PROMPT,
        raw_sha256="2aac067e69cdeb6840f510ac0f34c5249b8f33dd0764e251ac6c4d8b4c37fb40",
        semantic_sha256="254205b56f3373e0f522d596d1e93358622498f4e656c5a5e9c63b92a2066ed1",
    ),
    source_sha256="a8146591119c5439cc01168df63fa6144a7a55ff6817726946e1e8f5bc381617",
    asset_sha256=MappingProxyType(
        {
            "person_prompt_json": (
                "2aac067e69cdeb6840f510ac0f34c5249b8f33dd0764e251ac6c4d8b4c37fb40"
            ),
            "tracklets_jsonl": ("d886295534908392b43b7ac8d17e1df98efc5b566622b985f107cb05606f96d9"),
            "baseline_runner_mask_mp4": (
                "f7bb2d1ed00767ed2866c5b3a57b47361a591f1dbf090a5089d187f9ae410ef7"
            ),
        }
    ),
    frame_count=260,
    width=960,
    height=540,
    local_asset_paths=MappingProxyType(
        {
            "person_prompt_json": Path("person_prompt.json"),
            "tracklets_jsonl": Path("tracklets.jsonl"),
            "baseline_runner_mask_mp4": Path("runner_mask.mp4"),
        }
    ),
)

PARITY_FIXTURES = {CANONICAL_FRAME130_FIXTURE_ID: _FRAME130_FIXTURE}


def get_parity_fixture(fixture_id: str) -> ParityFixture:
    try:
        return PARITY_FIXTURES[fixture_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported SAM 3.1 parity fixture: {fixture_id}") from exc


def validate_prompt_for_fixture(
    prompt_bytes: bytes,
    *,
    fixture: ParityFixture,
) -> dict[str, str | bool]:
    try:
        payload = json.loads(prompt_bytes)
    except (TypeError, ValueError) as exc:
        raise ValueError("Parity fixture prompt is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Parity fixture prompt must be a JSON object.")

    raw_sha256 = _sha256(prompt_bytes)
    semantic_sha256 = _sha256(_prompt_semantic_bytes(payload))
    if semantic_sha256 != fixture.prompt.semantic_sha256:
        raise ValueError("Parity fixture prompt semantics do not match the registered fixture.")
    return {
        "raw_sha256": raw_sha256,
        "raw_hash_matches": raw_sha256 == fixture.prompt.raw_sha256,
        "semantic_sha256": semantic_sha256,
        "semantic_hash_matches": True,
    }


def load_local_fixture_assets(
    fixture_id: str,
    fixture_root: Path,
) -> dict[str, bytes]:
    fixture = get_parity_fixture(fixture_id)
    root = Path(fixture_root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"SAM 3.1 parity fixture root is unavailable: {root}")

    assets: dict[str, bytes] = {}
    for asset_name, relative_path in fixture.local_asset_paths.items():
        path = root / relative_path
        if asset_name == "person_prompt_json" and not path.is_file():
            data = fixture.prompt.canonical_bytes()
        else:
            if not path.is_file():
                raise FileNotFoundError(
                    f"SAM 3.1 parity fixture asset is unavailable: {asset_name} ({path})"
                )
            data = path.read_bytes()
            if asset_name == "person_prompt_json":
                validate_prompt_for_fixture(data, fixture=fixture)
                # Local prompt paths are intentionally allowed, but the serverless
                # request stays byte-identical to the audited production prompt.
                data = fixture.prompt.canonical_bytes()

        expected_sha256 = fixture.asset_sha256[asset_name]
        if _sha256(data) != expected_sha256:
            raise ValueError(
                f"SAM 3.1 parity fixture asset failed SHA-256 verification: {asset_name}"
            )
        assets[asset_name] = data
    return assets
