# Segmentation and DensePose Plan

This pipeline separates four jobs that are easy to blur together:

1. Target selection: identify which person in the clip is the runner we care about.
2. Whole-runner tracking: follow that selected runner through the reviewed segment.
3. Pose/form features: extract landmarks over time for matching.
4. Body-surface/detail layer: add DensePose body-region outputs when the runner mask and pose are already trustworthy.
5. Fused form signal: combine pose, mask, and DensePose evidence into confidence-weighted form QA.

## Recommended Stack

### Whole-Runner Segmentation

Use SAM 3.1 as the current SAM-style whole-runner mask backend. The mask pass should run
after target identity is seeded by the prompt and guarded by detector/tracker/ReID logic;
it should not be treated as the identity engine by itself.

Inputs:

- `source_segment.mp4`
- `person_prompt.json` with positive point, optional negative points, or a loose box

Outputs:

- `runner_mask.mp4`
- `masks.jsonl` once RLE mask storage is added
- per-frame mask metadata: area, centroid, confidence/proxy score, dropped-frame reason

### Pose Features

Use RTMPose/RTMW through RTMLib as the preferred production pose path. Keep OpenPose and
MediaPipe Pose Landmarker as baselines and fallback options. The matching feature should
be pose-sequence based, not mask-pixel based.

When `runner_mask.mp4` exists, pose extraction should treat it as a hard target constraint: black out non-runner pixels before MediaPipe inference, then prefer or reject pose candidates by overlap with the mask. This prevents crowded race footage from snapping to a clearer neighboring runner.

Outputs:

- `pose_landmarks.jsonl`
- `skeleton_render.mp4`
- frame-level quality metrics

### DensePose Layer

Use Detectron2 `projects/DensePose` as the first real DensePose layer. DensePose maps human pixels to a body surface representation, which is useful for body-region QA and maybe future body-part-aware features. It should run after target selection, and should be cropped or masked to the target runner so nearby runners do not pollute the result.

Outputs:

- `densepose.jsonl`
- per-frame part coverage, part pixels, UV summary stats, detection confidence, and mask overlap
- optional part/surface overlay inside `qa_overlay.mp4`

Do not make DensePose a hard dependency for MVP matching. Treat it as an enhancement layer because install/runtime requirements are heavier than the pose-first path.

### Fused Form Signal

Use MediaPipe as the motion truth and DensePose as the body-region confidence layer.
The fused stage should not replace pose-sequence matching; it should produce frame and
joint weights that tell the matcher which pose samples to trust.

Inputs:

- `pose_landmarks.jsonl`
- `runner_mask.mp4`
- `densepose.jsonl`
- source or DensePose overlay video for rendering

Outputs:

- `fused_form.jsonl`
- `fused_overlay.mp4`

The fused JSON should include per-frame confidence, frame state, DensePose body-region
coverage, per-joint weights, and questionable joints. The fused overlay should show the
runner mask edge, DensePose body-region rendering when available, MediaPipe skeleton,
red/yellow questionable joints, and a confidence badge.

## Single-Clip Sequence

1. Prepare a run folder:

   ```bash
   python scripts/prepare_single_clip_cv_run.py --candidate-id <candidate_id>
   ```

2. Open `prompt_frame.jpg` and select the target runner.
3. Write the selection into `person_prompt.json`.
4. Run detector/tracker/ReID target seeding when available.
5. Run SAM 3.1 over the target identity to produce `runner_mask.mp4`.
6. Run pose extraction on the segment, constrained by the runner mask when possible.
7. Run DensePose on masked/cropped frames.
8. Run the fused form stage.
9. Render `fused_overlay.mp4`.
10. Inspect the overlay before adding the clip to the scaled corpus.

## Prompt Selection

The first selection UI should be intentionally simple:

- show `prompt_frame.jpg`
- click one positive point near the torso/hip of the target runner
- optionally draw a loose bounding box when people overlap
- optionally add negative clicks on nearby non-target runners
- save normalized coordinates in `person_prompt.json`

Normalized coordinates are important so prompts survive resized previews.

## Occlusion Policy

Occlusions are normal in broadcast running footage. The pipeline should classify them, not fail blindly.

Frame states:

- `usable`: target mask and key pose landmarks are stable
- `short_occlusion`: a few frames are partially blocked, but identity is continuous
- `long_occlusion`: target is blocked or missing long enough to split the segment
- `identity_risk`: mask/pose likely jumped to another runner
- `cutaway`: camera cut or replay graphic interrupts the segment

Handling:

- Short occlusion: keep the segment, interpolate pose only for very short gaps, and downweight those frames.
- Long occlusion: split into subsegments or ask for another prompt after reappearance.
- Identity risk: stop propagation and reprompt. Do not let the tracker quietly switch runners.
- Cutaway: trim around it or mark the clip `mixed`/not usable for this pass.

Signals to monitor:

- pose landmark confidence
- missing hips/shoulders/ankles
- mask area jumps
- mask centroid velocity spikes
- DensePose part coverage disappearing
- fused form confidence dropping below the acceptance threshold
- scene cuts from PySceneDetect

Matching should use confidence-weighted frame aggregation. A clip can still be good if 80 percent of the selected segment is clean and the bad frames are marked.

## Scaling Gate

Do not process all reviewed clips until one segment can repeatedly produce:

- target mask without identity switching
- pose landmarks through at least one full stride cycle
- clear dropped-frame reasons
- skeleton render that looks like the runner
- fused overlay that makes failures obvious
