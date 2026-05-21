# Identity-Stable Runner Pipeline Plan

This is the implementation goal derived from the GPT 5.5 Pro research doc, with the SAM
2.1 recommendations intentionally ignored. The repo should use SAM 3.1 where a SAM-style
mask backend is needed, and the mask backend should be gated by explicit identity logic
rather than treated as the identity engine.

## Goal

Build an offline-first pipeline that turns long running videos into short, reviewable,
identity-stable runner clips, then extracts pose-first form features from the chosen
runner.

The pipeline target is:

```text
proxy ingest
shot detection
ranked clip-window proposal
human accepts window and clicks target once
person detection and tracking
ReID-gated target track
SAM 3.1/Cutie-style mask propagation on the known target
RTMPose/RTMW pose extraction on isolated crops
optional DensePose confidence layer
fusion, QC metrics, and retrieval features
```

## Architecture Principles

- Curation is cheap and broad: run shot detection and short-window scoring before any
  expensive mask work.
- Identity is explicit: the reviewer identifies the target runner once with a click or
  loose box; tracking and ReID preserve that identity through overlap.
- Segmentation refines a chosen identity: SAM 3.1 and future Cutie support masks after the
  target track is known.
- Pose remains the matching truth: RTMPose/RTMW becomes the preferred production pose path,
  with OpenPose and MediaPipe kept as baselines.
- DensePose stays downstream: use it for body-region coverage and QA after the target mask
  is stable.
- Multi-view logic has two modes: synchronized overlapping cameras use calibration and
  geometry; unrelated clips use angle buckets plus ReID/pose-cycle retrieval.

## Repo Implementation Phases

### Phase 1: Clip Proposal And Review

Implemented foundation:

- `src/whodoirunlike/curation.py`
- `scripts/run_clip_curation.py`
- `schemas/clip_window.schema.json`

Run:

```bash
python scripts/run_clip_curation.py \
  clips/raw/review_best/example.mp4 \
  --out artifacts/curation/clip_windows.json \
  --top-k 12 \
  --write-thumbnails
```

The current scorer is a motion-proxy first pass. It writes the same manifest shape that a
later detector/pose scorer can enrich with person visibility, track continuity, view
bucket, runningness, and occlusion penalties.

### Phase 2: Target Prompt And Identity Seed

Implemented foundation:

- `person_prompt.json` remains the current UI-compatible target prompt file.
- `target_prompt` is added as a manifest alias for the same identity prompt.
- `track_seed.json` records the intended detector, tracker, and ReID thresholds.
- `view_bucket.json` records the coarse view policy for single-view vs future multi-view
  matching.

The reviewer should save one positive torso/hip point, optional negative points on nearby
runners, and an optional loose box only when overlap makes the click ambiguous.

### Phase 3: Detector, Tracker, And ReID

Implemented foundation:

- `src/whodoirunlike/identity_runner.py`
- `scripts/run_identity_track.py`
- Writes `tracklets.parquet`, `tracklets.jsonl`, `reid.parquet`, `reid.jsonl`
- Updates `track_seed.json`, `cv_run_manifest.json`, and identity metrics in
  `qc_metrics.json`

Run:

```bash
python scripts/run_identity_track.py --candidate-id <candidate_id>
```

The current local backend is `prompt_template_tracker_v1`: a prompt-seeded template tracker
with HSV-histogram ReID. It is intentionally small and deterministic so the artifact
contract can be exercised before the heavier production backend lands.

Next implementation target:

- Replace or augment the baseline with BoT-SORT plus Torchreid OSNet embeddings.
- Keep Deep OC-SORT as the hard-occlusion A/B test and ByteTrack as the fastest baseline.
- Mark identity-risk intervals instead of smoothing through them.

Recommended initial thresholds are stored in `track_seed.json` when a CV run is prepared.

### Phase 4: Mask Propagation

Current default:

- SAM 3.1 MLX via `scripts/run_sam31_mlx_mask.py`

Future production target:

- Run SAM 3.1 or Cutie on a dynamic crop around the selected target track.
- Write `masks.jsonl` as COCO RLE-style frame records once pycocotools is introduced.
- Keep `runner_mask.mp4`, `masked_runner.mp4`, and `qa_overlay.mp4` as review artifacts.

Reset or request review when identity similarity drops, target area jumps, centroid motion
spikes, or two candidate tracks have near-tied ReID similarity.

### Phase 5: Pose, Fusion, And QC

Current and next pose path:

- RTMLib RTMPose/RTMW is available through `mmpose_runner.py`.
- OpenPose and MediaPipe remain baseline runners.
- `poses.parquet` is reserved for analytics-friendly pose tables.

QC should report:

- target identity stability rate
- overlap-only target identity stability rate
- occlusion recovery latency
- temporal mask churn
- pose visibility and missing-gap counts
- cross-view association F1 only when multi-camera labels exist

## Artifact Contract Additions

Each `artifacts/cv_runs/<candidate_id>/` folder now reserves:

```text
target_prompt -> person_prompt.json
track_seed.json
view_bucket.json
tracklets.parquet
reid.parquet
masks.jsonl
mask_logits.zarr
poses.parquet
densepose.parquet
fused_form.parquet
qc_metrics.json
```

The existing video/json artifacts stay in place so the current review UI and runners keep
working while the identity-first pieces are added.
