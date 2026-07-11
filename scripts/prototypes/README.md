# Pipeline speed prototypes

These are throwaway experiments on the `perf-prototypes` branch. They do not
change the production pipeline or deployment.

## Critical-path simulator

```bash
python scripts/prototypes/pipeline_schedule.py --scenario warm
```

Pass measured prototype durations to explore a new schedule:

```bash
python scripts/prototypes/pipeline_schedule.py \
  --scenario warm \
  --unified-track-mask-seconds 5 \
  --pose-seconds 10
```

## YOLO mask adapter

The experiment associates each YOLO person instance with the already selected
target track, then compares that mask with an existing SAM mask and the pose
keypoints consumed downstream.

```bash
.prototype-venv/bin/python scripts/prototypes/yolo_mask_adapter.py \
  /absolute/path/to/an/existing/run \
  --segment-model yolo11n-seg.pt \
  --detector-model yolo11n.pt \
  --imgsz 960
```

The `.prototype-venv` environment intentionally pins OpenCV 5 so compatibility
is measured separately from production's OpenCV 4 build.

## OpenCV 4 vs 5 frame-path control

```bash
.prototype-venv/bin/python scripts/prototypes/opencv_io_benchmark.py /path/to/source.mp4
.prototype-cv4/bin/python scripts/prototypes/opencv_io_benchmark.py /path/to/source.mp4
```
