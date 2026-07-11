# Prototype findings — pipeline speed without output loss

Question: can the current production behavior (select one runner, preserve that
identity, and produce the same analysis/artifact contract) become materially
faster without changing production while we experiment?

This worktree is isolated on `perf-prototypes`. Nothing here is deployed.

## Production baseline (2026-07-11)

The current `fd35d9c` processor has two completed 260-frame controls. The one
actually routed through the current production endpoint is the warm attempt;
the cold attempt was a pre-route validation run. The sample is intentionally
treated as low-confidence.

| Boundary | Warm seconds | Cold seconds |
| --- | ---: | ---: |
| Result ready | 134.066 | 171.703 |
| Target tracking | 2.274 | 5.706 |
| Runner mask | 35.317 | 54.569 |
| Pose sequence | 39.603 | 36.978 |
| DensePose | 23.437 | 26.081 |
| Publish before result ready | 20.909 | 21.261 |

The warm mask hit the process cache; the cold mask spent 18.300 seconds
building SAM. Warm SAM inference was 27.304 seconds. Pose inference was 38.051
seconds. DensePose inference was 22.308 seconds.

An older endpoint processed two attempts at the same time and showed a severe
tail: pose rose to 184.650/201.162 seconds and DensePose to 59.582/70.972
seconds. This is not part of the current endpoint cohort, but the current
endpoint also allows two workers, so a scratch two-job stress test is required.

## Dependency result

The code is serial, but the data graph is not:

```text
tracking -> mask -> [pose || DensePose] -> fusion
                                  -> [features || tables || QC]
                                  -> analysis -> artifact publish
```

The schedule simulator predicts that only the pose/DensePose fork saves 23.440
seconds warm and 26.101 seconds cold, reducing result-ready to 110.626 and
145.602 seconds respectively.

It is not safe to add threads around the existing functions. Pose and
DensePose both write `qa_overlay.mp4`, both perform stale read/modify/write
manifest updates, and telemetry heartbeat state represents only one active
stage/span. Those contracts must be isolated before the fork/join experiment.

## YOLO mask adapter result

The adapter uses the existing selected target track, chooses the YOLO person
instance with maximum box IoU, applies the same identity-risk blanking policy as
SAM, and compares against existing artifacts. A 5-pixel dilation was tested to
avoid clipping thin limbs. This is an adapter-quality result, not yet an inline
tracker implementation.

### Crowded 1080p clip — 146 frames

`yolo26n-seg.pt`, 960 input, 5-pixel dilation:

- target association: 146/146 frames;
- identity-risk/nonempty policy: exactly 143 nonempty frames for both YOLO and SAM;
- mask IoU vs SAM: mean 0.7294, p10 0.5994;
- mask recall vs SAM: mean 0.9562;
- temporal IoU: 0.6311 YOLO vs 0.5833 SAM;
- visible pose keypoints inside mask: 0.9149 YOLO vs 0.9162 SAM (-0.13 points);
- OpenCV 5 + CPU PyTorch: 40.88 ms/frame p50, 41.66 ms/frame p95.

### Hosted 540p clip — 260 frames

`yolo26n-seg.pt`, 960 input, 5-pixel dilation:

- target association: 260/260 frames;
- identity-risk/nonempty policy: exactly 228 nonempty frames for both YOLO and SAM;
- mask IoU vs SAM: mean 0.6562, p10 0.5351;
- mask recall vs SAM: mean 0.9532;
- temporal IoU: 0.5237 YOLO vs 0.4953 SAM;
- visible pose keypoints inside mask: 0.9097 YOLO vs 0.8830 SAM (+2.67 points);
- OpenCV 5 + CPU PyTorch: 40.58 ms/frame p50, 41.58 ms/frame p95.

The result supports a scratch TensorRT experiment. It does not yet prove final
feature or runner-ranking equivalence, and local CPU timings must not be used as
a RunPod latency forecast.

## OpenCV 5 control

On the 260-frame production fixture, five local decode + resize + grayscale +
mask morphology iterations produced:

- OpenCV 4.11 median: 0.1939 seconds (1340.65 frames/s)
- OpenCV 5.0 median: 0.1390 seconds (1871.04 frames/s)

That is a meaningful relative improvement but only 54.9 milliseconds across
the whole clip. OpenCV 5 is a compatible supporting upgrade, not the main
end-to-end speed lever.

## Next prototype gates

1. Freeze 20–30 explicit prompt/clip baselines and compare decoded artifacts,
   normalized JSON/JSONL, form arrays, QC, and final runner rankings.
2. In a scratch image, expose the actual RTMLib ONNX providers. Compare current
   CPU, CUDA ONNX Runtime, TensorRT, and direct tracked-box top-down pose.
3. Implement output-safe pose/DensePose fork/join and independently test model
   caches. Do not combine these changes in one measurement.
4. Replace YOLO11 detection with YOLO26 segmentation inside identity tracking,
   retaining frame-ordered tracker updates, identity-risk blanking, dilation,
   temporal sanity checks, and track-box/SAM fallback.
5. Export YOLO26n-seg to fixed-shape TensorRT FP16 and run warm, cold, and two-job
   scratch endpoint tests.
6. Upload `fused_overlay.mp4` first and teach the Worker/UI a nonterminal
   `result_ready` state; continue serial artifact uploads until the existing
   final artifact contract is complete.
7. Once pose is faster than DensePose, prototype padded target crops for
   DensePose and remap outputs to source coordinates. Do not skip DensePose for
   this target because it would change current fused output semantics.

Promotion requires zero new target switches, mask retention within one point,
pose and DensePose usable rates within two points, stable feature vectors,
at least 90% top-five runner overlap, the same artifact names/schemas, and a
material warm/cold p50/p95 latency improvement.
