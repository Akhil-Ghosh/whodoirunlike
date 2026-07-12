from __future__ import annotations

from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import sys
import types

import numpy as np

import whodoirunlike.sam31_gpu_runner as sam31_gpu_runner
from whodoirunlike.sam31_gpu_runner import (
    _borrow_sam31_gpu_predictor,
    _build_track_box_fallback_masks,
    _collect_sam31_masks,
    _configure_interactive_tracker_for_user_prompt,
    _filter_masks_to_track_boxes,
    _first_nonempty_output_object_id,
    _has_prompt_points,
    _mask_from_outputs,
    _patch_multiplex_init_state_kwargs,
    _prompt_points_with_box_support,
    _sam31_gpu_session_autocast,
    _seed_points_for_frame,
    _support_points_from_box,
    _track_prompt_anchors,
    update_manifest_after_sam31_gpu,
)


def _clear_predictor_cache() -> None:
    sam31_gpu_runner._SAM31_GPU_CACHED_PREDICTOR = None
    sam31_gpu_runner._SAM31_GPU_CACHED_CONFIG = None


def test_predictor_cache_builds_once_for_same_config_and_rebuilds_for_change() -> None:
    class ExitRecorder:
        def __init__(self) -> None:
            self.exits = 0

        def __exit__(self, *_args: object) -> None:
            self.exits += 1

    class Predictor:
        def __init__(self) -> None:
            self.bf16_context = ExitRecorder()
            self.model = types.SimpleNamespace(
                tracker=types.SimpleNamespace(bf16_context=ExitRecorder())
            )

    built: list[Predictor] = []

    def builder(**_kwargs: object) -> Predictor:
        predictor = Predictor()
        built.append(predictor)
        return predictor

    _clear_predictor_cache()
    try:
        with _borrow_sam31_gpu_predictor(
            builder=builder,
            build_kwargs={"checkpoint_path": None, "use_fa3": False},
            cache_enabled=True,
        ) as (first, first_metadata):
            assert first_metadata["hit"] is False
            assert first_metadata["autocast_contexts_unwound"] == 2

        with _borrow_sam31_gpu_predictor(
            builder=builder,
            build_kwargs={"checkpoint_path": None, "use_fa3": False},
            cache_enabled=True,
        ) as (second, second_metadata):
            assert second_metadata["hit"] is True

        with _borrow_sam31_gpu_predictor(
            builder=builder,
            build_kwargs={"checkpoint_path": "alternate.pt", "use_fa3": False},
            cache_enabled=True,
        ) as (third, third_metadata):
            assert third_metadata["hit"] is False

        assert first is second
        assert third is not first
        assert len(built) == 2
        assert built[0].bf16_context.exits == 1
        assert built[0].model.tracker.bf16_context.exits == 1
        assert built[1].bf16_context.exits == 1
        assert built[1].model.tracker.bf16_context.exits == 1
    finally:
        _clear_predictor_cache()


def test_sam31_session_autocast_is_entered_per_use(monkeypatch) -> None:
    events: list[str] = []

    class FakeAutocast:
        def __enter__(self) -> None:
            events.append("enter")

        def __exit__(self, *_args: object) -> None:
            events.append("exit")

    fake_torch = types.SimpleNamespace(
        bfloat16="bfloat16",
        autocast=lambda **_kwargs: FakeAutocast(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with _sam31_gpu_session_autocast():
        events.append("session-1")
    with _sam31_gpu_session_autocast():
        events.append("session-2")

    assert events == ["enter", "session-1", "exit", "enter", "session-2", "exit"]


def test_reclaim_gpu_memory_is_safe_without_torch(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)

    sam31_gpu_runner._reclaim_sam31_gpu_memory()


def test_runner_mask_readiness_fires_after_fallback_and_before_presentation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = {
        "source_segment": str(tmp_path / "source_segment.mp4"),
        "person_prompt": str(tmp_path / "person_prompt.json"),
        "runner_mask": str(tmp_path / "runner_mask.mp4"),
        "masked_runner": str(tmp_path / "masked_runner.mp4"),
        "qa_overlay": str(tmp_path / "qa_overlay.mp4"),
        "runner_mask_metadata": str(tmp_path / "runner_mask_metadata.jsonl"),
        "masks_jsonl": str(tmp_path / "masks.jsonl"),
        "tracklets_jsonl": str(tmp_path / "tracklets.jsonl"),
        "tracklets": str(tmp_path / "tracklets.json"),
    }
    (tmp_path / "cv_run_manifest.json").write_text(
        json.dumps({"candidate_id": "candidate-1", "paths": paths, "stages": {}}),
        encoding="utf-8",
    )
    for name in ("source_segment", "person_prompt", "tracklets_jsonl", "tracklets"):
        Path(paths[name]).touch()
    frame_paths = [tmp_path / "00000.jpg", tmp_path / "00001.jpg"]
    for frame_path in frame_paths:
        frame_path.touch()

    sam3_package = types.ModuleType("sam3")
    model_builder = types.ModuleType("sam3.model_builder")
    model_builder.build_sam3_multiplex_video_predictor = lambda **_: object()
    sam3_package.model_builder = model_builder
    monkeypatch.setitem(sys.modules, "sam3", sam3_package)
    monkeypatch.setitem(sys.modules, "sam3.model_builder", model_builder)

    predictor = object()

    @contextmanager
    def borrow_predictor(**_kwargs: object):
        yield predictor, {
            "hit": True,
            "model_build_seconds": 0.0,
            "lock_wait_seconds": 0.0,
        }

    monkeypatch.setattr(sam31_gpu_runner, "_borrow_sam31_gpu_predictor", borrow_predictor)
    monkeypatch.setattr(sam31_gpu_runner, "_sam31_gpu_session_autocast", nullcontext)
    monkeypatch.setattr(
        sam31_gpu_runner,
        "inspect_video",
        lambda _path: {"width": 10, "height": 8, "fps": 30.0},
    )
    monkeypatch.setattr(
        sam31_gpu_runner,
        "extract_video_frames",
        lambda *_args, **_kwargs: frame_paths,
    )
    monkeypatch.setattr(
        sam31_gpu_runner,
        "load_prompt",
        lambda *_args, **_kwargs: {
            "frame_index": 0,
            "box_source": "test",
            "positive_points": [],
        },
    )
    track_boxes = {0: (1, 1, 8, 7), 1: (1, 1, 8, 7)}
    monkeypatch.setattr(
        sam31_gpu_runner,
        "_load_identity_track_boxes",
        lambda **_kwargs: track_boxes,
    )
    monkeypatch.setattr(sam31_gpu_runner, "_track_prompt_anchors", lambda **_kwargs: [0])
    monkeypatch.setattr(
        sam31_gpu_runner,
        "_configure_interactive_tracker_for_user_prompt",
        lambda _predictor: {"applied": True},
    )
    initial_masks = {0: np.ones((8, 10), dtype=np.uint8)}
    monkeypatch.setattr(
        sam31_gpu_runner,
        "_collect_sam31_masks",
        lambda **_kwargs: (initial_masks, {"active_obj_id": 1}),
    )
    monkeypatch.setattr(
        sam31_gpu_runner,
        "_filter_masks_to_track_boxes",
        lambda masks, _boxes: (masks, {"enabled": True}),
    )

    events: list[str] = []

    def write_data(**_kwargs: object) -> None:
        events.append("data")
        Path(paths["runner_mask"]).touch()
        Path(paths["runner_mask_metadata"]).touch()

    summaries = iter(({"nonempty_frames": 0}, {"nonempty_frames": 2}))

    def write_masks(_mask_path: Path, output_path: Path) -> dict[str, int]:
        events.append("summary")
        output_path.touch()
        return next(summaries)

    fallback_masks = {
        0: np.ones((8, 10), dtype=np.uint8),
        1: np.ones((8, 10), dtype=np.uint8),
    }

    def build_fallback(**_kwargs: object):
        events.append("fallback")
        return fallback_masks, {"method": "track_box"}

    progress_phases: list[str] = []

    def ready() -> None:
        assert progress_phases[-1] == "analytical_mask_ready"
        events.append("ready")

    def write_presentation(**kwargs: object) -> None:
        assert kwargs["render_qa_overlay"] is False
        events.append("presentation")

    monkeypatch.setattr(sam31_gpu_runner, "write_runner_mask_data_outputs", write_data)
    monkeypatch.setattr(sam31_gpu_runner, "write_masks_jsonl_from_video", write_masks)
    monkeypatch.setattr(
        sam31_gpu_runner,
        "_identity_track_box_fallback_masks",
        build_fallback,
    )
    monkeypatch.setattr(
        sam31_gpu_runner,
        "write_mask_presentation_outputs",
        write_presentation,
    )
    monkeypatch.setattr(sam31_gpu_runner, "update_manifest_after_sam31_gpu", lambda *_a, **_k: None)

    result = sam31_gpu_runner.run_sam31_gpu_mask(
        run_dir=tmp_path,
        progress_callback=lambda payload: progress_phases.append(str(payload["phase"])),
        runner_mask_ready_callback=ready,
        render_qa_overlay=False,
    )

    assert events == [
        "data",
        "summary",
        "fallback",
        "data",
        "summary",
        "ready",
        "presentation",
    ]
    assert result["fallback"]["reason"] == "sam31_gpu_empty_mask"
    assert result["mask_summary"]["nonempty_frames"] == 2
    assert result["data_ready_seconds"] <= result["elapsed_seconds"]


def test_predictor_cache_is_invalidated_after_session_error(monkeypatch) -> None:
    class Predictor:
        _whodoirunlike_autocast_unwound = True
        model = types.SimpleNamespace()

        def __init__(self) -> None:
            self._all_inference_states = {"partial-session": {"state": object()}}

    built: list[Predictor] = []
    reclaimed: list[bool] = []
    monkeypatch.setattr(
        sam31_gpu_runner,
        "_reclaim_sam31_gpu_memory",
        lambda: reclaimed.append(True),
    )

    def builder(**_kwargs: object) -> Predictor:
        predictor = Predictor()
        built.append(predictor)
        return predictor

    _clear_predictor_cache()
    try:
        try:
            with _borrow_sam31_gpu_predictor(
                builder=builder,
                build_kwargs={"checkpoint_path": None},
                cache_enabled=True,
            ) as (_predictor, metadata):
                raise RuntimeError("session failed")
        except RuntimeError as exc:
            assert str(exc) == "session failed"
        else:
            raise AssertionError("expected session failure")

        assert metadata["invalidated_after_error"] is True
        assert built[0]._all_inference_states == {}
        assert reclaimed == [True]
        with _borrow_sam31_gpu_predictor(
            builder=builder,
            build_kwargs={"checkpoint_path": None},
            cache_enabled=True,
        ) as (_predictor, retry_metadata):
            assert retry_metadata["hit"] is False
        assert len(built) == 2
    finally:
        _clear_predictor_cache()


def test_mask_from_outputs_selects_requested_object() -> None:
    outputs = {
        "out_obj_ids": np.array([7, 1]),
        "out_binary_masks": np.array(
            [
                [[0, 1], [0, 0]],
                [[1, 1], [0, 0]],
            ],
            dtype=np.uint8,
        ),
    }

    mask = _mask_from_outputs(outputs, obj_id=1)

    assert mask is not None
    assert mask.tolist() == [[1, 1], [0, 0]]


def test_mask_from_outputs_falls_back_to_first_nonempty_mask() -> None:
    outputs = {
        "out_obj_ids": np.array([3]),
        "out_binary_masks": np.array([[[0, 0], [1, 0]]], dtype=np.uint8),
    }

    mask = _mask_from_outputs(outputs, obj_id=1)

    assert mask is not None
    assert mask.tolist() == [[0, 0], [1, 0]]


def test_first_nonempty_output_object_id_uses_mask_content() -> None:
    outputs = {
        "out_obj_ids": np.array([3, 7]),
        "out_binary_masks": np.array(
            [
                [[0, 0], [0, 0]],
                [[0, 1], [0, 0]],
            ],
            dtype=np.uint8,
        ),
    }

    assert _first_nonempty_output_object_id(outputs) == 7


def test_patch_multiplex_init_state_kwargs_filters_unknown_kwargs() -> None:
    class Model:
        def init_state(self, video_path: str) -> dict[str, str]:
            return {"video_path": video_path}

    class Predictor:
        model = Model()

    predictor = Predictor()

    _patch_multiplex_init_state_kwargs(predictor)

    assert predictor.model.init_state(
        video_path="clip.mp4",
        offload_state_to_cpu=True,
        offload_video_to_cpu=True,
    ) == {"video_path": "clip.mp4"}


def test_configure_interactive_tracker_disables_demo_suppression() -> None:
    class Model:
        masklet_confirmation_enable = True
        hotstart_delay = 15
        hotstart_unmatch_thresh = 8
        hotstart_dup_thresh = 8
        suppress_unmatched_only_within_hotstart = False
        suppress_overlapping_based_on_recent_occlusion_threshold = 0.7

    class Predictor:
        model = Model()

    result = _configure_interactive_tracker_for_user_prompt(Predictor())

    assert result["applied"] is True
    assert Predictor.model.masklet_confirmation_enable is False
    assert Predictor.model.hotstart_delay == 0
    assert Predictor.model.suppress_unmatched_only_within_hotstart is True
    assert Predictor.model.suppress_overlapping_based_on_recent_occlusion_threshold == 1.1


def test_prompt_points_with_box_support_keeps_user_point_and_adds_runner_body_points() -> None:
    prompt = {
        "points": np.array([[20, 30]], dtype=np.float32),
        "labels": np.array([1], dtype=np.int32),
    }

    points, labels = _prompt_points_with_box_support(
        prompt,
        box=np.array([10, 10, 30, 50], dtype=np.float32),
    )

    assert points is not None
    assert labels is not None
    assert points.shape == (4, 2)
    assert labels.tolist() == [1, 1, 1, 1]
    assert points[0].tolist() == [20, 30]
    assert points[1:].tolist() == _support_points_from_box(
        np.array([10, 10, 30, 50], dtype=np.float32)
    ).tolist()


def test_seed_points_for_frame_uses_track_box_points_when_seed_frame_differs() -> None:
    prompt = {
        "points": np.array([[90, 90]], dtype=np.float32),
        "labels": np.array([1], dtype=np.int32),
    }
    box = np.array([10, 10, 30, 50], dtype=np.float32)

    points, labels = _seed_points_for_frame(
        prompt=prompt,
        box=box,
        seed_frame=0,
        prompt_frame=99,
    )

    assert points is not None
    assert labels is not None
    assert points.tolist() == _support_points_from_box(box).tolist()
    assert labels.tolist() == [1, 1, 1]


def test_has_prompt_points_handles_numpy_arrays() -> None:
    assert _has_prompt_points({"points": np.array([[20, 30]], dtype=np.float32)}) is True
    assert _has_prompt_points({"points": np.zeros((0, 2), dtype=np.float32)}) is False
    assert _has_prompt_points({}) is False


def test_track_prompt_anchors_choose_prompt_start_middle_and_end_boxes() -> None:
    boxes = {
        0: np.array([0, 0, 10, 10], dtype=np.float32),
        50: np.array([1, 0, 11, 10], dtype=np.float32),
        99: np.array([2, 0, 12, 10], dtype=np.float32),
        150: np.array([3, 0, 13, 10], dtype=np.float32),
        199: np.array([4, 0, 14, 10], dtype=np.float32),
    }

    anchors = _track_prompt_anchors(
        prompt_frame=99,
        frame_count=200,
        track_boxes=boxes,
        max_anchors=4,
    )

    assert 99 in anchors
    assert anchors[0] == 0
    assert len(anchors) <= 4


def test_filter_masks_to_track_boxes_rejects_off_target_masks() -> None:
    on_target = np.zeros((30, 40), dtype=np.uint8)
    on_target[5:20, 10:25] = 1
    off_target = np.zeros((30, 40), dtype=np.uint8)
    off_target[5:20, 28:38] = 1

    filtered, summary = _filter_masks_to_track_boxes(
        {0: on_target, 1: off_target},
        {
            0: np.array([9, 4, 26, 21], dtype=np.float32),
            1: np.array([9, 4, 26, 21], dtype=np.float32),
        },
    )

    assert sorted(filtered) == [0]
    assert summary["enabled"] is True
    assert summary["accepted_frames"] == 1
    assert summary["rejected_frames"] == 1


def test_track_box_fallback_masks_follow_target_boxes_and_fill_short_gaps() -> None:
    masks, info = _build_track_box_fallback_masks(
        {
            0: np.array([10, 5, 20, 25], dtype=np.float32),
            2: np.array([14, 5, 24, 25], dtype=np.float32),
        },
        width=40,
        height=30,
        frame_count=3,
        max_interpolation_gap=4,
    )

    assert sorted(masks) == [0, 1, 2]
    assert info["backend"] == "identity_track_box"
    assert info["generated_frames"] == 3
    assert info["track_box_frames"] == 2
    assert info["interpolated_frames"] == 1
    assert masks[0][15, 15] == 1
    assert masks[1][15, 17] == 1
    assert masks[2][15, 19] == 1
    assert masks[1][1, 1] == 0


def test_update_manifest_after_sam31_gpu_records_fallback(tmp_path) -> None:
    manifest_path = tmp_path / "cv_run_manifest.json"
    metadata_path = tmp_path / "runner_mask_metadata.jsonl"
    masks_jsonl_path = tmp_path / "masks.jsonl"
    configured_masks_path = tmp_path / "custom-layout" / "runner-masks.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "candidate_id": "clip-1",
                "custom_top_level": {"keep": True},
                "paths": {"masks_jsonl": str(configured_masks_path)},
                "stages": {
                    "whole_runner_mask": {"error": "old failure", "custom_stage_value": 7},
                    "future_stage": {"status": "vendor-specific"},
                },
            }
        ),
        encoding="utf-8",
    )

    update_manifest_after_sam31_gpu(
        manifest_path,
        metadata_path,
        masks_jsonl_path,
        checkpoint_path=None,
        elapsed_seconds=1.25,
        mask_summary={"nonempty_frames": 3},
        fallback={
            "backend": "identity_track_box",
            "reason": "sam31_gpu_empty_mask",
            "generated_frames": 3,
        },
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage = manifest["stages"]["whole_runner_mask"]
    assert stage["status"] == "complete"
    assert stage["fallback"]["reason"] == "sam31_gpu_empty_mask"
    assert stage["mask_summary"]["nonempty_frames"] == 3
    assert "error" not in stage
    assert stage["custom_stage_value"] == 7
    assert manifest["paths"]["masks_jsonl"] == str(configured_masks_path)
    assert manifest["custom_top_level"] == {"keep": True}
    assert manifest["stages"]["future_stage"] == {"status": "vendor-specific"}


def test_collect_sam31_masks_tracks_visual_box_object_id(monkeypatch, tmp_path: Path) -> None:
    class FakeInferenceMode:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: object) -> None:
            return None

    fake_torch = types.SimpleNamespace(
        float32="float32",
        int32="int32",
        tensor=lambda data, **_kwargs: np.asarray(data),
        inference_mode=lambda: FakeInferenceMode(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakePredictor:
        def __init__(self) -> None:
            self.add_prompts: list[dict[str, object]] = []
            self.stream_requests: list[dict[str, object]] = []

        def handle_request(self, *, request: dict[str, object]) -> dict[str, object]:
            request_type = request["type"]
            if request_type == "start_session":
                return {"session_id": "session-1"}
            if request_type == "close_session":
                return {}
            if request_type == "add_prompt":
                self.add_prompts.append(request)
                if "bounding_boxes" in request:
                    return {
                        "outputs": {
                            "out_obj_ids": np.array([7]),
                            "out_binary_masks": np.array([[[1, 0], [0, 0]]], dtype=np.uint8),
                        }
                    }
                return {
                    "outputs": {
                        "out_obj_ids": np.array([7]),
                        "out_binary_masks": np.array([[[0, 1], [0, 0]]], dtype=np.uint8),
                    }
                }
            raise AssertionError(f"unexpected request: {request_type}")

        def handle_stream_request(self, *, request: dict[str, object]) -> list[dict[str, object]]:
            assert request["type"] == "propagate_in_video"
            self.stream_requests.append(request)
            return [
                {
                    "frame_index": 1,
                    "outputs": {
                        "out_obj_ids": np.array([7]),
                        "out_binary_masks": np.array([[[1, 1], [0, 0]]], dtype=np.uint8),
                    },
                }
            ]

    predictor = FakePredictor()
    masks, diagnostics = _collect_sam31_masks(
        predictor=predictor,
        video_path=tmp_path / "clip.mp4",
        prompt={
            "frame_index": 0,
            "box": np.array([0, 0, 2, 2], dtype=np.float32),
            "points": np.array([[1, 1]], dtype=np.float32),
            "labels": np.array([1], dtype=np.int32),
        },
        width=2,
        height=2,
        frame_count=2,
        obj_id=1,
    )

    point_prompt = predictor.add_prompts[1]
    assert point_prompt["obj_id"] == 7
    assert diagnostics["visual_box_prompt"] is True
    assert diagnostics["visual_box_obj_id"] == 7
    assert diagnostics["active_obj_id"] == 7
    assert [request["propagation_direction"] for request in predictor.stream_requests] == ["forward"]
    assert {request["start_frame_index"] for request in predictor.stream_requests} == {0}
    assert sorted(masks) == [0, 1]
    assert masks[1].tolist() == [[1, 1], [0, 0]]
    assert set(diagnostics["timings"]) == {
        "start_session_seconds",
        "box_prompt_seconds",
        "point_prompt_seconds",
        "initial_prompt_seconds",
        "preseed_anchors_seconds",
        "propagation_seconds",
        "close_session_seconds",
    }
    assert all(value >= 0 for value in diagnostics["timings"].values())


def test_collect_sam31_masks_scopes_exact_cv2_loader_to_session_start(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FakeInferenceMode:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: object) -> None:
            return None

    fake_torch = types.SimpleNamespace(
        float32="float32",
        int32="int32",
        tensor=lambda data, **_kwargs: np.asarray(data),
        inference_mode=lambda: FakeInferenceMode(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    tracking_module = types.ModuleType("test_sam31_tracking_module")
    monkeypatch.setitem(sys.modules, tracking_module.__name__, tracking_module)

    class Model:
        pass

    Model.__module__ = tracking_module.__name__

    class Predictor:
        model = Model()

        def handle_request(self, *, request: dict[str, object]) -> dict[str, object]:
            if request["type"] == "start_session":
                return {"session_id": "session-1"}
            if request["type"] == "close_session":
                return {}
            if request["type"] == "add_prompt":
                return {
                    "outputs": {
                        "out_obj_ids": np.array([1]),
                        "out_binary_masks": np.array([[[1, 0], [0, 0]]], dtype=np.uint8),
                    }
                }
            raise AssertionError(f"unexpected request: {request['type']}")

        def handle_stream_request(self, *, request: dict[str, object]):
            assert request["type"] == "propagate_in_video"
            return []

    scoped_calls: list[dict[str, object]] = []

    @contextmanager
    def scoped_loader(**kwargs: object):
        scoped_calls.append(kwargs)
        yield types.SimpleNamespace(
            attempted=False,
            used=False,
            diagnostics=None,
            fallback_reason=None,
        )

    monkeypatch.setattr(
        sam31_gpu_runner,
        "scoped_sam31_exact_cv2_loader",
        scoped_loader,
    )

    _collect_sam31_masks(
        predictor=Predictor(),
        video_path=tmp_path / "clip.mp4",
        prompt={
            "frame_index": 0,
            "points": np.array([[1, 1]], dtype=np.float32),
            "labels": np.array([1], dtype=np.int32),
        },
        width=2,
        height=2,
        frame_count=1,
        obj_id=1,
        exact_cv2_loader_enabled=True,
        exact_cv2_chunk_frames=3,
        exact_cv2_max_frames=21,
        exact_cv2_max_destination_bytes=987654,
    )

    assert scoped_calls == [
        {
            "tracking_module": tracking_module,
            "enabled": True,
            "chunk_frames": 3,
            "max_frames": 21,
            "max_destination_bytes": 987654,
        }
    ]


def test_collect_sam31_masks_seeds_from_first_target_track_frame(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FakeInferenceMode:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: object) -> None:
            return None

    fake_torch = types.SimpleNamespace(
        float32="float32",
        int32="int32",
        tensor=lambda data, **_kwargs: np.asarray(data),
        inference_mode=lambda: FakeInferenceMode(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakePredictor:
        def __init__(self) -> None:
            self.add_prompts: list[dict[str, object]] = []
            self.stream_requests: list[dict[str, object]] = []

        def handle_request(self, *, request: dict[str, object]) -> dict[str, object]:
            request_type = request["type"]
            if request_type == "start_session":
                return {"session_id": "session-1"}
            if request_type == "close_session":
                return {}
            if request_type == "add_prompt":
                self.add_prompts.append(request)
                return {
                    "outputs": {
                        "out_obj_ids": np.array([1]),
                        "out_binary_masks": np.array([[[1, 0], [0, 0]]], dtype=np.uint8),
                    }
                }
            raise AssertionError(f"unexpected request: {request_type}")

        def handle_stream_request(self, *, request: dict[str, object]) -> list[dict[str, object]]:
            self.stream_requests.append(request)
            return []

    predictor = FakePredictor()
    _masks, diagnostics = _collect_sam31_masks(
        predictor=predictor,
        video_path=tmp_path / "clip.mp4",
        prompt={
            "frame_index": 99,
            "box": np.array([70, 70, 90, 100], dtype=np.float32),
            "points": np.array([[80, 90]], dtype=np.float32),
            "labels": np.array([1], dtype=np.int32),
        },
        width=100,
        height=100,
        frame_count=120,
        obj_id=1,
        track_boxes={
            0: np.array([10, 10, 30, 50], dtype=np.float32),
            99: np.array([70, 70, 90, 100], dtype=np.float32),
        },
    )

    assert diagnostics["seed_frame"] == 0
    assert diagnostics["seed_source"] == "target_track_first_visible_frame"
    assert predictor.add_prompts[0]["frame_index"] == 0
    assert predictor.add_prompts[1]["frame_index"] == 0
    point_prompt = predictor.add_prompts[1]
    assert np.asarray(point_prompt["points"]).shape == (3, 2)
    expected_points = _support_points_from_box(np.array([10, 10, 30, 50], dtype=np.float32))
    expected_points[:, 0] /= 100
    expected_points[:, 1] /= 100
    assert np.asarray(point_prompt["points"]).tolist() == expected_points.tolist()
    assert predictor.stream_requests[0]["start_frame_index"] == 0
    assert predictor.stream_requests[0]["propagation_direction"] == "forward"


def test_collect_sam31_masks_preseeds_anchors_before_one_full_pass(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FakeInferenceMode:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: object) -> None:
            return None

    fake_torch = types.SimpleNamespace(
        float32="float32",
        int32="int32",
        tensor=lambda data, **_kwargs: np.asarray(data),
        inference_mode=lambda: FakeInferenceMode(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakePredictor:
        def __init__(self) -> None:
            self.events: list[tuple[str, int | None]] = []

        def handle_request(self, *, request: dict[str, object]) -> dict[str, object]:
            request_type = str(request["type"])
            if request_type == "start_session":
                self.events.append((request_type, None))
                return {"session_id": "session-1"}
            if request_type == "close_session":
                self.events.append((request_type, None))
                return {}
            if request_type == "add_prompt":
                frame_index = int(request["frame_index"])
                self.events.append((request_type, frame_index))
                return {
                    "outputs": {
                        "out_obj_ids": np.array([1]),
                        "out_binary_masks": np.array([[[1, 0], [0, 0]]], dtype=np.uint8),
                    }
                }
            raise AssertionError(f"unexpected request: {request_type}")

        def handle_stream_request(self, *, request: dict[str, object]) -> list[dict[str, object]]:
            self.events.append(("propagate_in_video", int(request["start_frame_index"])))
            return [
                {
                    "frame_index": frame_index,
                    "outputs": {
                        "out_obj_ids": np.array([1]),
                        "out_binary_masks": np.array([[[1, 1], [0, 0]]], dtype=np.uint8),
                    },
                }
                for frame_index in range(6)
            ]

    predictor = FakePredictor()
    boxes = {
        frame_index: np.array([0, 0, 2, 2], dtype=np.float32) for frame_index in range(6)
    }
    masks, diagnostics = _collect_sam31_masks(
        predictor=predictor,
        video_path=tmp_path / "clip.mp4",
        prompt={
            "frame_index": 0,
            "box": np.array([0, 0, 2, 2], dtype=np.float32),
            "points": np.array([[1, 1]], dtype=np.float32),
            "labels": np.array([1], dtype=np.int32),
        },
        width=2,
        height=2,
        frame_count=6,
        obj_id=1,
        track_boxes=boxes,
    )

    propagation_positions = [
        index for index, event in enumerate(predictor.events) if event[0] == "propagate_in_video"
    ]
    anchor_positions = [
        index
        for index, event in enumerate(predictor.events)
        if event[0] == "add_prompt" and event[1] in {1, 2, 3, 4, 5}
    ]
    assert len(propagation_positions) == 1
    assert anchor_positions
    assert max(anchor_positions) < propagation_positions[0]
    assert diagnostics["preseed_anchor_frames"] == [0, 1, 2, 3, 4, 5]
    assert diagnostics["anchor_refinement_triggered"] is False
    assert [item["pass"] for item in diagnostics["propagation"]] == ["primary_prompt"]
    assert sorted(masks) == list(range(6))


def test_collect_sam31_masks_retries_once_when_first_pass_is_sparse(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FakeInferenceMode:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: object) -> None:
            return None

    fake_torch = types.SimpleNamespace(
        float32="float32",
        int32="int32",
        tensor=lambda data, **_kwargs: np.asarray(data),
        inference_mode=lambda: FakeInferenceMode(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakePredictor:
        def __init__(self) -> None:
            self.stream_requests = 0

        def handle_request(self, *, request: dict[str, object]) -> dict[str, object]:
            if request["type"] == "start_session":
                return {"session_id": "session-1"}
            if request["type"] == "close_session":
                return {}
            if request["type"] == "add_prompt":
                return {"outputs": {}}
            raise AssertionError(f"unexpected request: {request['type']}")

        def handle_stream_request(self, *, request: dict[str, object]) -> list[dict[str, object]]:
            self.stream_requests += 1
            if self.stream_requests == 2:
                return [
                    {
                        "frame_index": frame_index,
                        "outputs": {
                            "out_obj_ids": np.array([1]),
                            "out_binary_masks": np.array(
                                [[[1, 0], [0, 0]]],
                                dtype=np.uint8,
                            ),
                        },
                    }
                    for frame_index in range(20)
                ]
            return [
                {
                    "frame_index": 0,
                    "outputs": {
                        "out_obj_ids": np.array([1]),
                        "out_binary_masks": np.array([[[1, 0], [0, 0]]], dtype=np.uint8),
                    },
                }
            ]

    predictor = FakePredictor()
    masks, diagnostics = _collect_sam31_masks(
        predictor=predictor,
        video_path=tmp_path / "clip.mp4",
        prompt={
            "frame_index": 0,
            "box": np.array([0, 0, 2, 2], dtype=np.float32),
            "points": np.array([[1, 1]], dtype=np.float32),
            "labels": np.array([1], dtype=np.int32),
        },
        width=2,
        height=2,
        frame_count=20,
        obj_id=1,
        track_boxes={
            frame_index: np.array([0, 0, 2, 2], dtype=np.float32)
            for frame_index in range(20)
        },
    )

    assert predictor.stream_requests == 2
    assert sorted(masks) == list(range(20))
    assert diagnostics["anchor_refinement_triggered"] is True
    assert [item["pass"] for item in diagnostics["propagation"]] == [
        "primary_prompt",
        "sparse_safety_retry",
    ]


def test_collect_sam31_masks_closes_session_when_propagation_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FakeInferenceMode:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: object) -> None:
            return None

    fake_torch = types.SimpleNamespace(
        float32="float32",
        int32="int32",
        tensor=lambda data, **_kwargs: np.asarray(data),
        inference_mode=lambda: FakeInferenceMode(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakePredictor:
        closed = False

        def handle_request(self, *, request: dict[str, object]) -> dict[str, object]:
            if request["type"] == "start_session":
                return {"session_id": "session-1"}
            if request["type"] == "close_session":
                self.closed = True
                return {}
            if request["type"] == "add_prompt":
                return {"outputs": {}}
            raise AssertionError(f"unexpected request: {request['type']}")

        def handle_stream_request(self, *, request: dict[str, object]) -> list[dict[str, object]]:
            raise ValueError("inference failed")

    predictor = FakePredictor()
    try:
        _collect_sam31_masks(
            predictor=predictor,
            video_path=tmp_path / "clip.mp4",
            prompt={
                "frame_index": 0,
                "box": np.array([0, 0, 2, 2], dtype=np.float32),
                "points": np.array([[1, 1]], dtype=np.float32),
                "labels": np.array([1], dtype=np.int32),
            },
            width=2,
            height=2,
            frame_count=1,
            obj_id=1,
        )
    except ValueError as exc:
        assert str(exc) == "inference failed"
    else:
        raise AssertionError("expected propagation failure")

    assert predictor.closed is True
