# Single-Clip CV Flow

Use one approved clip as the proving ground before scaling the corpus. The goal is to produce inspectable artifacts first, then automate only after the artifacts look right.

The current implementation target is identity-first: choose a good window, identify the
target runner once, preserve that target with tracking/ReID, then run SAM 3.1-style mask
propagation and pose extraction on the selected runner.

Prepare a run folder with:

```bash
python scripts/prepare_single_clip_cv_run.py --candidate-id <candidate_id>
```

## Inputs

- `candidate_id`
- reviewed `start_seconds` and `end_seconds`
- reviewed `camera_angle`: `side`, `diagonal`, `front`, `rear`, `mixed`, or `unknown`
- local source video path from the review manifest

## Artifact Contract

For one clip, write everything under:

```text
artifacts/cv_runs/<candidate_id>/
  source_segment.mp4
  prompt_frame.jpg
  person_prompt.json
  track_seed.json
  view_bucket.json
  tracklets.parquet
  reid.parquet
  masks.jsonl
  poses.parquet
  pose_landmarks.jsonl
  runner_mask.mp4
  densepose.jsonl
  qc_metrics.json
  fused_form.jsonl
  skeleton_render.mp4
  masked_runner.mp4
  qa_overlay.mp4
  fused_overlay.mp4
  features.json
```

## Flow

1. Trim the reviewed interval into `source_segment.mp4`.
2. Extract a clean prompt frame near the middle of the interval.
3. Show the prompt frame in an internal UI and let the reviewer click the target runner or draw a loose box.
4. Use detector/tracker/ReID logic to lock onto the intended person through the segment.
5. Run SAM 3.1 or a future Cutie-style propagator on the known target identity, preferably on a dynamic crop.
6. Run pose estimation across the segment and save per-frame landmarks, confidence, bounding boxes, and dropped-frame reasons. The pose runner should hard-mask each source frame with `runner_mask.mp4` before inference when a mask is available, then reject pose candidates that do not overlap the runner mask.
7. Generate a normalized skeleton-render video for Gemini pose-render embeddings.
8. Run DensePose as the body-region confidence layer when available.
9. Fuse pose, runner mask, and DensePose into `fused_form.jsonl`.
10. Write `qc_metrics.json` with identity stability, occlusion recovery, temporal mask churn, and pose visibility metrics.
11. Render `fused_overlay.mp4` with source/DensePose video, mask edge, skeleton, frame confidence, and rejected-frame markers.
12. Review the artifacts. If the target person switches, limbs disappear, or angle metadata is wrong, fix the prompt/segment before scaling.

## Tooling Recommendation

Primary matching should be pose-sequence based. Segmentation is still valuable, but mostly as target isolation, QA, body-region confidence, and the masked-video embedding experiment.

The practical order for the first clip:

1. Ranked clip-window proposal with PySceneDetect plus cheap motion/pose scoring.
2. Click or box selection on one frame to disambiguate the target runner.
3. Person detection plus BoT-SORT/OSNet-style target tracking.
4. SAM 3.1 mask generation on the chosen target identity.
5. RTMPose/RTMW pose extraction, with OpenPose/MediaPipe as baselines.
6. DensePose body-region pass as a confidence and visual QA layer.
7. Fused form signal for per-frame and per-joint weighting.
8. Skeleton render and masked-runner render as downstream embedding inputs.

SAM/ZIM-style segmentation should not be the source of body-part semantics by itself. For legs, torso, arms, and head, pose landmarks are the first reliable motion structure, while DensePose provides the body-region confidence layer once the whole-person mask is stable.

## Angle Policy

Camera angle should be explicit metadata, not inferred silently.

- `side`: best for stride mechanics and first MVP comparisons.
- `diagonal`: usable, but compare against diagonal clips or downweight angle-sensitive features.
- `front` / `rear`: useful for entertainment matches, weaker for side-view mechanics.
- `mixed`: needs either a shorter segment or subsegments by angle.
- `unknown`: allowed during review, but not ideal for approved reference segments.

For matching, start by comparing clips within the same angle bucket. Once we have enough examples, add a cross-angle embedding experiment rather than pretending all views are equivalent.
