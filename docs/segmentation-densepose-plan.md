# Segmentation and DensePose Plan

This pipeline separates four jobs that are easy to blur together:

1. Target selection: identify which person in the clip is the runner we care about.
2. Whole-runner tracking: follow that selected runner through the reviewed segment.
3. Pose/form features: extract landmarks over time for matching.
4. Body-surface/detail layer: add DensePose or body-part outputs when the runner mask and pose are already trustworthy.

## Recommended Stack

### Whole-Runner Segmentation

Use SAM 2.1 video prediction for the whole-runner mask. Meta documents SAM 2 as promptable with click, box, or mask inputs on an image or video frame, and its video predictor is designed to propagate a selected object through a video.

Inputs:

- `source_segment.mp4`
- `person_prompt.json` with positive point, optional negative points, or a loose box

Outputs:

- `runner_mask.mp4`
- per-frame mask metadata: area, centroid, confidence/proxy score, dropped-frame reason

### Pose Features

Use MediaPipe Pose Landmarker first because it is fast and already in the repo. The matching feature should be pose-sequence based, not mask-pixel based.

Outputs:

- `pose_landmarks.jsonl`
- `skeleton_render.mp4`
- frame-level quality metrics

### DensePose Layer

Use Detectron2 `projects/DensePose` as the first real DensePose layer. DensePose maps human pixels to a body surface representation, which is useful for body-region QA and maybe future body-part-aware features. It should run after target selection, and should be cropped or masked to the target runner so nearby runners do not pollute the result.

Outputs:

- `densepose.jsonl`
- optional part/surface overlay inside `qa_overlay.mp4`

Do not make DensePose a hard dependency for MVP matching. Treat it as an enhancement layer because install/runtime requirements are heavier than the pose-first path.

## Single-Clip Sequence

1. Prepare a run folder:

   ```bash
   python scripts/prepare_single_clip_cv_run.py --candidate-id <candidate_id>
   ```

2. Open `prompt_frame.jpg` and select the target runner.
3. Write the selection into `person_prompt.json`.
4. Run SAM 2.1 over `source_segment.mp4` to produce `runner_mask.mp4`.
5. Run pose extraction on the segment, constrained by the runner mask when possible.
6. Run DensePose on masked/cropped frames.
7. Render `qa_overlay.mp4`.
8. Inspect the overlay before adding the clip to the scaled corpus.

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
- scene cuts from PySceneDetect

Matching should use confidence-weighted frame aggregation. A clip can still be good if 80 percent of the selected segment is clean and the bad frames are marked.

## Scaling Gate

Do not process all reviewed clips until one segment can repeatedly produce:

- target mask without identity switching
- pose landmarks through at least one full stride cycle
- clear dropped-frame reasons
- skeleton render that looks like the runner
- QA overlay that makes failures obvious
