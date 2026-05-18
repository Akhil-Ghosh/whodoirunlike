# Similarity Search Implementation Plan

This plan turns processed **Running Clips** and approved **Reference Segments** into **Form Matches**. The guiding decision is already captured in ADR 0001: **Pose Sequence** is the motion truth, while the **Fused Form Signal** supplies confidence, occlusion handling, body-region features, and visual QA.

## Progress Tracker

- [x] Phase 1: Define the searchable feature artifact.
- [x] Phase 2: Compile `form_features` for one processed clip.
- [ ] Phase 3: Compile a local reference feature library.
- [ ] Phase 4: Implement exact weighted sequence matching.
- [ ] Phase 5: Aggregate reference-segment matches into runner matches.
- [ ] Phase 6: Generate explanation tags and matched-clip output.
- [ ] Phase 7: Add leave-one-clip-out validation.
- [ ] Phase 8: Add the internal similarity-search UI.
- [ ] Phase 9: Add embedding experiments after the pose-first matcher is working.

Current build note: `scripts/compile_form_features.py` and the Subject UI `Run Features` action now write
`form_features.json` and `form_features.npz`. The first compiled clip is
`artifacts/cv_runs/d6ee6cd75cd04b95/form_features.json`.

## Target Outcome

Given one query **Running Clip**, return:

- top **Form Match**
- top 3 alternative runners
- the matched **Reference Segment** for each result
- match confidence as `high`, `medium`, or `low`, not as a percentage match
- explanation tags such as `similar torso lean`, `similar compact arm swing`, `similar knee recovery path`, and `similar stride rhythm`
- side-by-side source/fused/skeleton playback for the query and matched reference segment

## Non-Goals For This Stage

- Do not train a learned model yet.
- Do not use raw video embeddings as the primary matcher.
- Do not claim coaching, injury, identity, or biomechanical precision.
- Do not require DensePose for every clip; use it when available and degrade gracefully to pose-only confidence.
- Do not introduce FAISS/vector infrastructure until exact search becomes too slow for the corpus size.

## Artifact Contract

Each processed clip should eventually contain:

```text
artifacts/cv_runs/<candidate_id>/
  pose_landmarks.jsonl
  runner_mask.mp4
  densepose.jsonl
  fused_form.jsonl
  form_features.json
  form_features.npz
  skeleton_render.mp4
  fused_overlay.mp4
```

`form_features.json` stores metadata and summary features. `form_features.npz` stores dense arrays for sequence matching.

Recommended `form_features.json` shape:

```json
{
  "version": 1,
  "candidate_id": "clip-id",
  "runner_name": "Cole Hocker",
  "camera_angle": "side",
  "event_bucket": "800_1500",
  "frame_count": 260,
  "usable_frame_count": 236,
  "fps": 29.97,
  "duration_seconds": 8.68,
  "feature_files": {
    "arrays": "form_features.npz"
  },
  "quality": {
    "usable_rate": 0.9077,
    "fused_confidence_mean": 0.8722,
    "pose_visibility_mean": 0.8084
  },
  "summary_features": {
    "torso_lean_mean": 0.0,
    "arm_swing_amplitude": 0.0,
    "knee_lift_proxy": 0.0,
    "lower_leg_recovery_arc": 0.0,
    "vertical_oscillation_proxy": 0.0,
    "stride_rhythm_proxy": 0.0
  }
}
```

Recommended `form_features.npz` arrays:

- `pose_xy`: `[frames, joints, 2]`, normalized landmark coordinates
- `pose_world`: `[frames, joints, 3]`, MediaPipe world landmarks when available
- `joint_weights`: `[frames, joints]`, from `fused_form.jsonl`
- `frame_weights`: `[frames]`, from fused frame confidence and frame state
- `bone_vectors`: `[frames, bones, 2]`
- `joint_angles`: `[frames, named_angles]`
- `angular_velocity`: `[frames, named_angles]`
- `densepose_groups`: `[frames, groups]`
- `valid_frames`: `[frames]`
- `time_seconds`: `[frames]`

## Feature Compiler

Create a compiler that reads:

- `pose_landmarks.jsonl`
- `fused_form.jsonl`
- `densepose.jsonl` when present
- `cv_run_manifest.json`

and writes:

- `form_features.json`
- `form_features.npz`

### Pose Sequence Features

The primary sequence should include:

- centered landmark coordinates
- scale-normalized landmark coordinates
- bone vectors
- torso axis
- shoulder axis
- hip axis
- upper-arm and lower-arm vectors
- thigh and shin vectors
- ankle/heel/foot-index trajectories

Normalization rules:

- Center each frame around the hip midpoint when hips are reliable.
- Scale by shoulder-to-hip torso length or shoulder/hip box size.
- Keep body-proportion signal available separately; do not erase all morphology because height/build can be part of entertainment resemblance.
- Do not mirror left/right automatically in the first version; instead record the camera direction/mirroring decision as metadata later.
- Compare only within compatible camera-angle buckets first.

### Joint Angles

Compute these angle series:

- left/right elbow angle
- left/right shoulder angle relative to torso
- left/right hip angle relative to torso
- left/right knee angle
- left/right ankle-foot angle when foot landmarks are reliable
- torso lean relative to vertical image axis
- thigh/shin segment angle relative to horizontal and vertical axes

### DensePose Body-Region Features

DensePose should contribute:

- torso coverage over time
- upper-leg and lower-leg coverage over time
- upper-arm and lower-arm coverage over time
- foot/hand/head visibility over time
- coverage rhythm and disappearance patterns
- DensePose confidence and mask overlap

These features are auxiliary. They should help explain and weight the match, not dominate it.

### Fused Weights

Use `fused_form.jsonl` to build:

- `frame_weights`
- `joint_weights`
- `valid_frames`
- per-joint questionable markers

Initial frame weighting:

```text
usable                 => frame_confidence
short_occlusion        => frame_confidence * 0.45
pose_rejected          => 0
densepose_missing      => pose-only confidence * 0.75
identity_risk          => 0
cutaway                => 0
```

Initial joint weighting:

```text
joint_weight = fused joint weight
```

If DensePose is missing, fall back to:

```text
joint_weight = MediaPipe visibility * mask containment proxy
```

## Stride Windows

The matcher should compare useful windows, not necessarily whole clips.

First implementation:

- Use all `valid_frames`.
- Require at least 60 valid frames or about 2 seconds.
- Allow clips with partial bad regions if valid frames are contiguous enough.

Second implementation:

- Detect stride rhythm from ankle/foot-index trajectories.
- Segment into 1-3 stride-cycle windows.
- Store window metadata in `form_features.json`.

Window shape:

```json
{
  "window_id": "clip-id:w0",
  "start_frame": 20,
  "end_frame": 140,
  "usable_rate": 0.94,
  "confidence_mean": 0.89,
  "stride_cycles_estimated": 2
}
```

## Reference Library

Create:

```text
artifacts/search/reference_index.json
artifacts/search/features/<candidate_id>.form_features.json
artifacts/search/features/<candidate_id>.form_features.npz
```

Reference index entry:

```json
{
  "candidate_id": "clip-id",
  "runner_name": "Faith Kipyegon",
  "runner_slug": "faith-kipyegon",
  "camera_angle": "side",
  "event_bucket": "800_1500",
  "quality": "good",
  "usable_rate": 0.92,
  "feature_json": "artifacts/search/features/clip-id.form_features.json",
  "feature_npz": "artifacts/search/features/clip-id.form_features.npz",
  "source_segment": "artifacts/cv_runs/clip-id/source_segment.mp4",
  "fused_overlay": "artifacts/cv_runs/clip-id/fused_overlay.mp4",
  "skeleton_render": "artifacts/cv_runs/clip-id/skeleton_render.mp4"
}
```

Inclusion rules:

- reviewed quality must be `good`
- `pose_landmarks.jsonl` must exist
- `fused_form.jsonl` should exist
- usable rate should be at least `0.75`
- camera angle should not be `mixed` unless the segment is manually accepted

## Matching Algorithm

Use exact search first. With a corpus of roughly 100-200 reference segments, exact matching is simpler and inspectable.

### Candidate Filtering

Filter reference segments by:

- mode: distance running first
- camera angle: same bucket first
- quality: `good`
- minimum usable rate
- optional event bucket filter: `800_1500`, `5k_10k`, `marathon`

Fallback policy:

- If fewer than 10 compatible references exist, allow adjacent angle buckets with a penalty.
- If still sparse, return low-confidence results and say the reference pool is thin.

### Weighted Pose DTW

Compare query and reference sequences using dynamic time warping over weighted frame distances.

Frame distance:

```text
frame_distance(q, r) =
  weighted_joint_xy_distance
+ weighted_bone_vector_distance
+ weighted_joint_angle_distance
```

Joint weights:

```text
pair_joint_weight = sqrt(query_joint_weight * reference_joint_weight)
```

Frame weights:

```text
pair_frame_weight = sqrt(query_frame_weight * reference_frame_weight)
```

DTW cost:

```text
weighted_dtw_cost = sum(pair_frame_weight * frame_distance) / sum(pair_frame_weight)
```

Use `dtaidistance`, `tslearn`, or a small custom NumPy implementation. Start custom if weighted frames/joints are awkward in libraries.

### Summary Feature Distance

Compute a normalized distance over summary features:

- torso lean mean and range
- arm swing amplitude
- elbow angle range
- knee angle range
- ankle/foot trajectory range
- hip vertical oscillation proxy
- stride rhythm proxy

This should produce explanation tags and stabilize results when DTW overfits one noisy window.

### DensePose Auxiliary Distance

Compare body-region series:

- torso coverage
- upper/lower arm coverage
- upper/lower leg coverage
- foot visibility
- DensePose confidence rhythm

This should be capped so it cannot dominate the match.

Initial scoring formula:

```text
clip_distance =
  0.70 * weighted_pose_dtw
+ 0.20 * summary_feature_distance
+ 0.10 * densepose_region_distance
```

If DensePose is unavailable:

```text
clip_distance =
  0.80 * weighted_pose_dtw
+ 0.20 * summary_feature_distance
```

Convert distance into match confidence only after calibration. Until then, use rank and qualitative confidence.

## Runner Aggregation

The search operates over reference segments, but the product returns runners.

Do not average all clips for a runner. That punishes runners with one bad segment.

Initial aggregation:

```text
runner_distance = median(top 2 segment distances for that runner)
```

If a runner has only one reference segment:

```text
runner_distance = best segment distance + sparse_reference_penalty
```

Return:

- best runner
- best matching reference segment for that runner
- top 3 runners
- top 5 matching reference segments for debugging

## Explanation Tags

Generate tags from feature deltas, not from a language model.

Potential tags:

- `similar torso lean`
- `similar arm carriage`
- `similar compact arm swing`
- `similar knee recovery path`
- `similar lower-leg angle`
- `similar foot path`
- `similar vertical rhythm`
- `similar stride timing`

Rule:

- A tag is eligible when the relevant feature distance is in the best 30 percent of that match's feature groups.
- Do not show a tag when either clip has low confidence for the involved joints/frames.
- Show 2-4 tags per result.

## Result Artifact

Write:

```text
artifacts/search/results/<query_candidate_id>.match_results.json
```

Shape:

```json
{
  "version": 1,
  "query_candidate_id": "query-id",
  "created_at": "2026-05-18T00:00:00Z",
  "mode": "running",
  "camera_angle_policy": "same_bucket_first",
  "top_runner": {
    "runner_name": "Jakob Ingebrigtsen",
    "runner_slug": "jakob-ingebrigtsen",
    "confidence": "medium",
    "distance": 0.42,
    "explanation_tags": ["similar torso lean", "similar knee recovery path"],
    "best_segment": {
      "candidate_id": "ref-id",
      "source_segment": "...",
      "fused_overlay": "...",
      "skeleton_render": "..."
    }
  },
  "runner_results": [],
  "segment_debug_results": []
}
```

## Validation Plan

### Leave-One-Clip-Out

For each runner with at least 2 approved reference segments:

1. Remove one segment from the reference index.
2. Use it as the query.
3. Search the remaining reference index.
4. Record whether the same runner appears in top 1, top 3, and top 5.

Metrics:

- same-runner top-1 rate
- same-runner top-3 rate
- same-event-bucket top-3 rate
- median rank of same runner
- failure cases grouped by angle, occlusion, and clip quality

### Plausibility Review

For early entertainment quality, same-runner retrieval is not enough.

Create blind comparisons:

- system match vs random same-angle match
- system match vs nearest event-bucket-only match
- pose-only match vs fused-weight match

Reviewer question:

```text
Which comparison feels more like the same running style?
```

Success threshold for MVP:

- system match preferred over random at least 70 percent of the time
- fused-weight match preferred over pose-only on clips with occlusion or pack traffic
- top-3 result feels shareable even when top-1 is imperfect

## UI Plan

Add a local search page after the CLI works:

```text
http://127.0.0.1:8765/search.html
```

Controls:

- choose query CV run
- choose reference pool
- angle policy: same angle, compatible angles, all
- event bucket filter
- run search

Result view:

- top runner card
- top 3 alternatives
- query fused overlay beside matched fused overlay
- query skeleton beside matched skeleton
- explanation tags
- debug table of segment distances

## Implementation Phases

### Phase 1: Feature Schema

- [ ] Add `src/whodoirunlike/form_features.py`.
- [ ] Define joint/bone/angle names.
- [ ] Define `form_features.json` metadata schema in code.
- [ ] Define `form_features.npz` arrays.
- [ ] Add tests for normalization, angle computation, and densepose group extraction.

Done when:

- one existing CV run can compile deterministic `form_features.json` and `form_features.npz`
- tests cover missing DensePose fallback

### Phase 2: Feature Compiler CLI

- [ ] Add `scripts/compile_form_features.py`.
- [ ] Read one CV run folder.
- [ ] Write feature artifacts into the run folder.
- [ ] Update `cv_run_manifest.json`.
- [ ] Add the feature artifact to `subject.html`.

Done when:

- the Cole Hocker run compiles successfully
- feature summaries look sane
- artifact appears in the UI

### Phase 3: Reference Index Builder

- [ ] Add `scripts/build_reference_index.py`.
- [ ] Scan approved CV runs.
- [ ] Compile missing feature artifacts.
- [ ] Write `artifacts/search/reference_index.json`.
- [ ] Copy or reference feature files under `artifacts/search/features/`.

Done when:

- at least the current processed clips can be indexed
- index entries include runner, angle, event bucket, quality, and feature paths

### Phase 4: Exact Matcher

- [ ] Add `src/whodoirunlike/similarity.py`.
- [ ] Implement weighted frame distance.
- [ ] Implement weighted DTW.
- [ ] Implement summary feature distance.
- [ ] Implement DensePose auxiliary distance.
- [ ] Add `scripts/search_form_matches.py`.

Done when:

- one query clip can search a local reference index
- output includes segment distances and runner aggregation

### Phase 5: Explanations

- [ ] Add feature-group distance breakdown.
- [ ] Convert lowest-distance feature groups into explanation tags.
- [ ] Suppress tags for low-confidence joints/features.
- [ ] Include tags in `match_results.json`.

Done when:

- each result has 2-4 deterministic tags
- tags can be traced back to numeric feature groups

### Phase 6: Validation

- [ ] Add `scripts/evaluate_similarity_leave_one_out.py`.
- [ ] Compute top-k retrieval metrics.
- [ ] Save `artifacts/search/evaluation/leave_one_out.json`.
- [ ] Save failure-case CSV for review.

Done when:

- we can compare pose-only vs fused-weight vs fused-plus-DensePose configurations

### Phase 7: Internal Search UI

- [ ] Add `review_ui/search.html`.
- [ ] Add API endpoints for reference index status and running search.
- [ ] Show side-by-side query/match videos.
- [ ] Show explanation tags and debug distances.

Done when:

- a reviewer can pick a query run and inspect top matches without touching the CLI

### Phase 8: Embedding Experiments

- [ ] Generate normalized skeleton-render videos.
- [ ] Generate fused/DensePose render videos.
- [ ] Run Gemini embedding experiment on skeleton renders.
- [ ] Run Gemini embedding experiment on masked/fused render videos.
- [ ] Add optional reranker, never primary scorer until validated.

Done when:

- embedding results can be compared against exact pose search on the same leave-one-out suite

## First Build Slice

Build the smallest useful slice in this order:

1. `form_features.py`
2. `scripts/compile_form_features.py`
3. tests for feature arrays and angle summaries
4. compile features for `d6ee6cd75cd04b95`
5. add `form_features` artifact visibility to the UI

This gives us a concrete feature object to inspect before implementing search math.

## Open Questions

- Should body-proportion features be included as a small positive signal, or only kept for explanations?
- Should `unknown` camera-angle clips be excluded from the first reference index?
- Should diagonal clips be compared with side clips at a penalty, or kept separate until the library is larger?
- How many approved reference segments per runner are required before that runner appears in public results?
- Should DensePose be run on Modal for all reference segments before search validation, or only for clips where local CPU output is already promising?
