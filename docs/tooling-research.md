# Tooling Research Notes

Current recommendation for the ingestion/evaluation pipeline:

1. **Search/discovery**
   - Primary: `yt-dlp` search and metadata extraction for YouTube.
   - Secondary: Scrapling for static/dynamic page extraction and resilient parsing.
   - Secondary: Camoufox for browser-backed search when pages are JS-heavy, rate-limited, or anti-bot-sensitive.

2. **Candidate triage**
   - Metadata relevance score first, before downloading video.
   - Visual/CV score second, on only the best candidates.

3. **Video processing**
   - `imageio-ffmpeg` supplies a local ffmpeg binary without requiring Homebrew/system install.
   - OpenCV reads low-res downloaded candidate videos and samples frames.
   - PySceneDetect detects camera cuts once we are trimming approved segments.

4. **Pose / form extraction**
   - MediaPipe Pose Landmarker is the first-pass evaluator because it is lightweight, local, and outputs 33 landmarks including feet plus optional segmentation masks.
   - Ultralytics YOLO Pose/Track is the next runner-detection/tracking layer, especially when multiple people are in frame.
   - MMPose/RTMPose is the later high-accuracy whole-body/foot-keypoint path if MediaPipe/YOLO are not enough.

5. **Segmentation / masks**
   - SAM 3 is the right person-instance segmentation/tracking candidate for polished masks.
   - ZIM can refine masks/mattes for visual output, but its CC BY-NC 4.0 license means it is private-alpha friendly and must be revisited before commercial use.

6. **Similarity**
   - Primary: normalized pose-sequence similarity.
   - Experiment A: Gemini Embedding 2 over canonical skeleton/body-map render videos.
   - Experiment B: Gemini Embedding 2 over masked runner videos.
   - Avoid raw video embeddings as the primary signal.

