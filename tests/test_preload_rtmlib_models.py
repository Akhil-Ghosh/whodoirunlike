from __future__ import annotations

import hashlib
import importlib.util
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/preload_rtmlib_models.py"
SPEC = importlib.util.spec_from_file_location("preload_rtmlib_models", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
ModelArchive = MODULE.ModelArchive
preload_asset = MODULE.preload_asset


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_archive(path: Path, onnx_payload: bytes) -> bytes:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("release/model/end2end.onnx", onnx_payload)
    return path.read_bytes()


def test_preload_asset_writes_rtmlib_cache_path_and_reuses_verified_model(
    tmp_path: Path,
) -> None:
    onnx_payload = b"deterministic-onnx-model"
    archive_path = tmp_path / "fake-model.zip"
    archive_payload = _write_archive(archive_path, onnx_payload)
    asset = ModelArchive(
        url=archive_path.as_uri(),
        archive_sha256=_sha256(archive_payload),
        archive_size=len(archive_payload),
        onnx_sha256=_sha256(onnx_payload),
        onnx_size=len(onnx_payload),
    )

    output = preload_asset(asset, cache_root=tmp_path / "cache", attempts=1)

    assert output == tmp_path / "cache/hub/checkpoints/fake-model.onnx"
    assert output.read_bytes() == onnx_payload

    archive_path.unlink()
    assert preload_asset(asset, cache_root=tmp_path / "cache", attempts=1) == output


def test_preload_asset_rejects_non_zip_without_poisoning_cache(tmp_path: Path) -> None:
    archive_path = tmp_path / "broken-model.zip"
    archive_payload = b"truncated upstream response"
    archive_path.write_bytes(archive_payload)
    asset = ModelArchive(
        url=archive_path.as_uri(),
        archive_sha256=_sha256(archive_payload),
        archive_size=len(archive_payload),
        onnx_sha256=_sha256(b"expected-model"),
        onnx_size=len(b"expected-model"),
    )

    with pytest.raises(RuntimeError, match="Could not preload verified RTMLib model"):
        preload_asset(asset, cache_root=tmp_path / "cache", attempts=1)

    assert not (tmp_path / "cache/hub/checkpoints/broken-model.onnx").exists()
