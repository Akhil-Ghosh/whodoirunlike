#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class ModelArchive:
    url: str
    archive_sha256: str
    archive_size: int
    onnx_sha256: str
    onnx_size: int

    @property
    def cache_name(self) -> str:
        return f"{Path(urlparse(self.url).path).stem}.onnx"


PRODUCTION_ASSETS = (
    ModelArchive(
        url=(
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
            "yolox_m_8xb8-300e_humanart-c2c7a14a.zip"
        ),
        archive_sha256="a000224fd8ba283202bc62d4a5fcdfe353adb9f468777dbac1ea2ada2093adde",
        archive_size=94_223_081,
        onnx_sha256="3dea6513388889f0fff4b77bf7a26013600321b9eb9ceb0e9a400a82572f5f23",
        onnx_size=101_400_344,
    ),
    ModelArchive(
        url=(
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
            "rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.zip"
        ),
        archive_sha256="b06f1bc598ddbdb5008a61f4a411aa11ed331370b4f7f438126d6d99e168d147",
        archive_size=125_375_567,
        onnx_sha256="8cfecfc2226d8e14b510c2fd28442226c518b4d60690535753187877411d4005",
        onnx_size=134_399_323,
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified(path: Path, *, size: int, sha256: str) -> bool:
    return path.is_file() and path.stat().st_size == size and _sha256(path) == sha256


def cached_model_path(asset: ModelArchive, *, cache_root: Path) -> Path:
    return cache_root.expanduser().resolve() / "hub" / "checkpoints" / asset.cache_name


def _download_archive(asset: ModelArchive, destination: Path, *, timeout_seconds: int) -> None:
    request = Request(asset.url, headers={"User-Agent": "whodoirunlike-image-builder/1.0"})
    with urlopen(request, timeout=timeout_seconds) as response, destination.open("wb") as output:
        advertised = response.headers.get("Content-Length")
        shutil.copyfileobj(response, output, length=1024 * 1024)

    actual_size = destination.stat().st_size
    if advertised is not None and actual_size != int(advertised):
        raise ValueError(
            f"Truncated RTMLib archive: expected HTTP length {advertised}, got {actual_size}"
        )
    if actual_size != asset.archive_size:
        raise ValueError(
            f"Unexpected RTMLib archive size: expected {asset.archive_size}, got {actual_size}"
        )
    actual_sha256 = _sha256(destination)
    if actual_sha256 != asset.archive_sha256:
        raise ValueError(
            "Unexpected RTMLib archive SHA-256: "
            f"expected {asset.archive_sha256}, got {actual_sha256}"
        )


def _extract_model(asset: ModelArchive, archive_path: Path, output_path: Path) -> None:
    temporary_output: Path | None = None
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            members = [name for name in archive.namelist() if Path(name).name == "end2end.onnx"]
            if len(members) != 1:
                raise ValueError(
                    f"Expected one end2end.onnx in {archive_path.name}, found {len(members)}"
                )
            with archive.open(members[0], "r") as source:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix=f".{output_path.name}.",
                    suffix=".tmp",
                    dir=output_path.parent,
                    delete=False,
                ) as destination:
                    temporary_output = Path(destination.name)
                    shutil.copyfileobj(source, destination, length=1024 * 1024)

        if not _verified(
            temporary_output,
            size=asset.onnx_size,
            sha256=asset.onnx_sha256,
        ):
            raise ValueError(f"Extracted RTMLib model failed integrity checks: {asset.cache_name}")
        os.replace(temporary_output, output_path)
        temporary_output = None
    finally:
        if temporary_output is not None:
            temporary_output.unlink(missing_ok=True)


def preload_asset(
    asset: ModelArchive,
    *,
    cache_root: Path,
    attempts: int = 3,
    timeout_seconds: int = 300,
) -> Path:
    if attempts < 1:
        raise ValueError("attempts must be at least 1")

    output_path = cached_model_path(asset, cache_root=cache_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if _verified(output_path, size=asset.onnx_size, sha256=asset.onnx_sha256):
        return output_path
    output_path.unlink(missing_ok=True)

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        with tempfile.NamedTemporaryFile(
            prefix=f".{Path(urlparse(asset.url).path).name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary:
            archive_path = Path(temporary.name)
        try:
            _download_archive(asset, archive_path, timeout_seconds=timeout_seconds)
            _extract_model(asset, archive_path, output_path)
            return output_path
        except (EOFError, OSError, ValueError, zipfile.BadZipFile) as error:
            last_error = error
            output_path.unlink(missing_ok=True)
            if attempt < attempts:
                time.sleep(min(attempt, 5))
        finally:
            archive_path.unlink(missing_ok=True)

    raise RuntimeError(
        f"Could not preload verified RTMLib model after {attempts} attempt(s): {asset.url}"
    ) from last_error


def default_cache_root() -> Path:
    torch_home = os.getenv("TORCH_HOME", "").strip()
    if torch_home:
        return Path(torch_home).expanduser()
    xdg_cache = Path(os.getenv("XDG_CACHE_HOME", "~/.cache")).expanduser()
    return xdg_cache / "rtmlib"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preload checksum-verified RTMLib ONNX models into its offline cache."
    )
    parser.add_argument("--cache-root", type=Path, default=default_cache_root())
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args()

    for asset in PRODUCTION_ASSETS:
        output = preload_asset(asset, cache_root=args.cache_root, attempts=args.attempts)
        print(f"Preloaded {output.name} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
