from __future__ import annotations

import json
import os
import tempfile
import threading
import weakref
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


MANIFEST_FILENAME = "cv_run_manifest.json"

CANONICAL_ARTIFACT_FILENAMES: Mapping[str, str] = {
    "source_segment": "source_segment.mp4",
    "prompt_frame": "prompt_frame.jpg",
    "person_prompt": "person_prompt.json",
    "target_prompt": "person_prompt.json",
    "track_seed": "track_seed.json",
    "view_bucket": "view_bucket.json",
    "tracklets": "tracklets.parquet",
    "tracklets_jsonl": "tracklets.jsonl",
    "reid": "reid.parquet",
    "reid_jsonl": "reid.jsonl",
    "masks_jsonl": "masks.jsonl",
    "mask_logits": "mask_logits.zarr",
    "poses": "poses.parquet",
    "pose_landmarks": "pose_landmarks.jsonl",
    "runner_mask": "runner_mask.mp4",
    "densepose": "densepose.jsonl",
    "densepose_parquet": "densepose.parquet",
    "fused_form": "fused_form.jsonl",
    "fused_form_parquet": "fused_form.parquet",
    "skeleton_render": "skeleton_render.mp4",
    "masked_runner": "masked_runner.mp4",
    "pose_qa_overlay": "pose_qa_overlay.mp4",
    "qa_overlay": "qa_overlay.mp4",
    "fused_overlay": "fused_overlay.mp4",
    "qc_metrics": "qc_metrics.json",
    "features": "features.json",
    "form_features": "form_features.json",
    "form_feature_arrays": "form_features.npz",
    "mmpose_landmarks": "mmpose_landmarks.jsonl",
    "openpose_landmarks": "openpose_landmarks.jsonl",
    "openpose_skeleton_render": "openpose_skeleton_render.mp4",
    "openpose_qa_overlay": "openpose_qa_overlay.mp4",
    "pose_comparison": "pose_comparison.json",
    "runner_mask_metadata": "runner_mask_metadata.jsonl",
    "hosted_pipeline_result": "hosted_pipeline_result.json",
}

_AUXILIARY_ARTIFACT_KEYS = frozenset({"runner_mask_metadata", "hosted_pipeline_result"})
MANIFEST_ARTIFACT_KEYS = tuple(
    key for key in CANONICAL_ARTIFACT_FILENAMES if key not in _AUXILIARY_ARTIFACT_KEYS
)

_ARTIFACT_ALIASES = {"target_prompt": "person_prompt"}
_MISSING = object()
_MANIFEST_LOCKS_GUARD = threading.Lock()
_MANIFEST_LOCKS: weakref.WeakValueDictionary[Path, threading.RLock] = (
    weakref.WeakValueDictionary()
)


def _manifest_lock(path: Path) -> threading.RLock:
    key = path.resolve(strict=False)
    with _MANIFEST_LOCKS_GUARD:
        lock = _MANIFEST_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _MANIFEST_LOCKS[key] = lock
        return lock


def _manifest_copy(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise ValueError("Running Clip Run manifest must be an object")

    copied = dict(manifest)
    for field in ("paths", "stages"):
        if field not in copied:
            continue
        value = copied[field]
        if not isinstance(value, Mapping):
            raise ValueError(f"Running Clip Run manifest '{field}' must be an object")
        copied[field] = dict(value)
    return copied


def _alias_value(paths: Mapping[str, Any], key: str) -> Any:
    target = _ARTIFACT_ALIASES.get(key)
    if target is not None and target in paths:
        return paths[target]

    for alias, alias_target in _ARTIFACT_ALIASES.items():
        if key == alias_target and alias in paths:
            return paths[alias]
    return _MISSING


class RunningClipRun:
    """Local manifest and artifact coordination for one Running Clip Run."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self._lock = _manifest_lock(self.manifest_path)

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / MANIFEST_FILENAME

    def canonical_paths(self, keys: Iterable[str] | None = None) -> dict[str, str]:
        selected_keys = MANIFEST_ARTIFACT_KEYS if keys is None else keys
        return {key: str(self.run_dir / CANONICAL_ARTIFACT_FILENAMES[key]) for key in selected_keys}

    def read_manifest(self) -> dict[str, Any]:
        with self._lock:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            return _manifest_copy(payload)

    def write_manifest(self, manifest: Mapping[str, Any]) -> Path:
        with self._lock:
            payload = _manifest_copy(manifest)
            self.run_dir.mkdir(parents=True, exist_ok=True)

            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    "w",
                    delete=False,
                    dir=self.run_dir,
                    encoding="utf-8",
                    prefix=f".{MANIFEST_FILENAME}.",
                    suffix=".tmp",
                ) as temporary_file:
                    temporary_path = Path(temporary_file.name)
                    json.dump(payload, temporary_file, indent=2)
                    temporary_file.write("\n")
                    temporary_file.flush()
                    os.fsync(temporary_file.fileno())
                os.replace(temporary_path, self.manifest_path)
            except Exception:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
                raise

        return self.manifest_path

    def ensure_paths(
        self,
        manifest: Mapping[str, Any],
        keys: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        updated = _manifest_copy(manifest)
        paths = dict(updated.get("paths", {}))
        selected_keys = CANONICAL_ARTIFACT_FILENAMES if keys is None else keys

        for key in selected_keys:
            if key in paths:
                continue
            alias_value = _alias_value(paths, key)
            if alias_value is not _MISSING:
                paths[key] = alias_value
            else:
                paths[key] = str(self.run_dir / CANONICAL_ARTIFACT_FILENAMES[key])

        updated["paths"] = paths
        return updated

    def artifact_path(
        self,
        key: str,
        manifest: Mapping[str, Any] | None = None,
    ) -> Path:
        if manifest is None:
            current = self.read_manifest() if self.manifest_path.is_file() else {}
        else:
            current = _manifest_copy(manifest)

        paths = current.get("paths", {})
        if key in paths:
            return Path(str(paths[key]))

        alias_value = _alias_value(paths, key)
        if alias_value is not _MISSING:
            return Path(str(alias_value))

        return Path(self.canonical_paths((key,))[key])

    def update_stage(
        self,
        stage: str,
        values: Mapping[str, Any],
        manifest: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.update_stages({stage: values}, manifest)

    def update_stages(
        self,
        updates: Mapping[str, Mapping[str, Any]],
        manifest: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(updates, Mapping):
            raise ValueError("Running Clip Run stage updates must be an object")

        with self._lock:
            supplied = _manifest_copy(manifest) if manifest is not None else None
            if self.manifest_path.is_file():
                updated = self.read_manifest()
                if supplied is not None:
                    for key, value in supplied.items():
                        if key not in {"paths", "stages"}:
                            if (
                                key == "updated_at"
                                and isinstance(value, str)
                                and isinstance(updated.get(key), str)
                                and updated[key] > value
                            ):
                                continue
                            updated[key] = value
                    supplied_paths = supplied.get("paths", {})
                    paths = dict(updated.get("paths", {}))
                    paths.update(supplied_paths)
                    updated["paths"] = paths
            else:
                updated = supplied or {}

            stages = dict(updated.get("stages", {}))
            supplied_stages = dict(supplied.get("stages", {})) if supplied is not None else {}
            for stage, values in updates.items():
                if not isinstance(values, Mapping):
                    raise ValueError("Running Clip Run stage values must be an object")
                existing_stage = supplied_stages.get(stage, stages.get(stage, {}))
                if not isinstance(existing_stage, Mapping):
                    raise ValueError(f"Running Clip Run stage '{stage}' must be an object")

                merged_stage = dict(existing_stage)
                merged_stage.update(values)
                stages[stage] = merged_stage

            updated["stages"] = stages
            self.write_manifest(updated)
            return updated

    def existing_artifacts(
        self,
        keys: Iterable[str] | None = None,
        extra_names: Iterable[str] = (),
    ) -> list[Path]:
        current = self.read_manifest() if self.manifest_path.is_file() else {}
        selected_keys = CANONICAL_ARTIFACT_FILENAMES if keys is None else keys
        candidates = [self.artifact_path(key, current) for key in selected_keys]
        candidates.extend(self.run_dir / name for name in extra_names)

        existing: list[Path] = []
        seen: set[Path] = set()
        for path in candidates:
            if path in seen:
                continue
            seen.add(path)
            if path.is_file():
                existing.append(path)
        return existing
