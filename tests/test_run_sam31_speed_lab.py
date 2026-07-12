from __future__ import annotations

import base64
from dataclasses import replace
import gzip
import hashlib
import importlib.util
from pathlib import Path
from types import MappingProxyType

import pytest

from whodoirunlike.sam31_parity import (
    CANONICAL_FRAME130_FIXTURE_ID,
    PARITY_FIXTURES,
    get_parity_fixture,
)


def _load_speed_lab_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts/run_sam31_speed_lab.py"
    spec = importlib.util.spec_from_file_location("run_sam31_speed_lab", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_builds_selected_frame130_payload_from_local_ignored_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load_speed_lab_script()
    base_fixture = get_parity_fixture(CANONICAL_FRAME130_FIXTURE_ID)
    tracklets = b'{"frame_index":130}\n'
    baseline_mask = b"fixture-mask"
    fixture = replace(
        base_fixture,
        fixture_id="test-cli-frame130",
        asset_sha256=MappingProxyType(
            {
                "person_prompt_json": base_fixture.prompt.raw_sha256,
                "tracklets_jsonl": hashlib.sha256(tracklets).hexdigest(),
                "baseline_runner_mask_mp4": hashlib.sha256(baseline_mask).hexdigest(),
            }
        ),
    )
    monkeypatch.setitem(PARITY_FIXTURES, fixture.fixture_id, fixture)
    (tmp_path / "tracklets.jsonl").write_bytes(tracklets)
    (tmp_path / "runner_mask.mp4").write_bytes(baseline_mask)

    payload = script._build_assets(
        fixture_id=fixture.fixture_id,
        fixture_root=tmp_path,
    )

    assert base64.b64decode(payload["person_prompt_json"]["data"]) == (
        fixture.prompt.canonical_bytes()
    )
    assert gzip.decompress(base64.b64decode(payload["tracklets_jsonl"]["data"])) == tracklets
    assert base64.b64decode(payload["baseline_runner_mask_mp4"]["data"]) == baseline_mask
    assert payload["tracklets_jsonl"]["sha256"] == hashlib.sha256(tracklets).hexdigest()


def test_cli_resolves_full_profile_selection_and_named_matrix() -> None:
    script = _load_speed_lab_script()

    assert script._resolve_full_profile_ids(
        profile_id="production_candidate",
        profile_matrix=None,
    ) == ["production_candidate"]
    assert script._resolve_full_profile_ids(
        profile_id=None,
        profile_matrix="three-arm",
    ) == [
        "downstream_baseline_control",
        "downstream_candidate_control",
        "downstream_candidate_optimized",
    ]
    assert script._resolve_full_profile_ids(
        profile_id=None,
        profile_matrix="production-reversed",
    ) == ["production_candidate", "production_control"]
    assert script._resolve_full_profile_ids(
        profile_id=None,
        profile_matrix="authoritative-control",
    ) == ["production_control"]
    assert script._resolve_full_profile_ids(
        profile_id=None,
        profile_matrix="authoritative-candidate",
    ) == ["production_candidate"]


def test_cli_rejects_profile_and_matrix_together() -> None:
    script = _load_speed_lab_script()

    with pytest.raises(ValueError, match="either"):
        script._resolve_full_profile_ids(
            profile_id="downstream_candidate",
            profile_matrix="three-arm",
        )


def test_cli_creates_private_reusable_r2_sink_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load_speed_lab_script()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    sink_file = tmp_path / "private" / "parity-sink.json"
    expected = {
        "callback_base_url": "https://parity-scratch.example.com",
        "run_id": "11c51cf1-c4d0-42ef-a2e1-cb9e2605ef1b",
        "attempt_id": "5ec9566e-cda9-4113-9100-9a4b2a248f6f",
    }
    calls: list[Path] = []

    def fake_create(*, api_base_url: str, source_clip: Path):
        assert api_base_url == expected["callback_base_url"]
        calls.append(source_clip)
        return expected

    monkeypatch.setattr(script, "_create_artifact_sink", fake_create)

    first = script._load_or_create_artifact_sink(
        sink_file=sink_file,
        api_base_url=expected["callback_base_url"],
        source_clip=source,
    )
    second = script._load_or_create_artifact_sink(
        sink_file=sink_file,
        api_base_url=expected["callback_base_url"],
        source_clip=source,
    )

    assert first == second == expected
    assert calls == [source]
    assert sink_file.stat().st_mode & 0o777 == 0o600
    assert "secret" not in sink_file.read_text(encoding="utf-8").lower()


@pytest.mark.parametrize(
    "origin",
    (
        "https://api.whodoirunlike.com",
        "https://staging-api.whodoirunlike.com",
    ),
)
def test_cli_rejects_production_and_staging_artifact_sinks(origin: str) -> None:
    script = _load_speed_lab_script()

    with pytest.raises(RuntimeError, match="production or staging"):
        script._validate_cli_sink_origin(origin)


def test_cli_requires_existing_descriptor_to_match_explicit_scratch_origin(
    tmp_path: Path,
) -> None:
    script = _load_speed_lab_script()
    sink_file = tmp_path / "parity-sink.json"
    sink_file.write_text(
        """{
  "callback_base_url": "https://first-scratch.example.com",
  "run_id": "11c51cf1-c4d0-42ef-a2e1-cb9e2605ef1b",
  "attempt_id": "5ec9566e-cda9-4113-9100-9a4b2a248f6f"
}\n""",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="does not match"):
        script._load_or_create_artifact_sink(
            sink_file=sink_file,
            api_base_url="https://second-scratch.example.com",
            source_clip=tmp_path / "unused.mp4",
        )
