# Who Do I Run Like

This context describes the language for the running-form similarity product and its internal CV ingestion pipeline.

## Language

**Running Clip**:
A short video segment containing one runner whose movement can be compared against the reference library.
_Avoid_: raw video, source video when referring to the reviewed movement sample

**Reference Segment**:
A human-approved running clip from a known athlete in the internal comparison library.
_Avoid_: scrape, download, candidate once it has been approved

**Target Runner**:
The person in a running clip whose form is being analyzed.
_Avoid_: subject, person, object when discussing product behavior

**Runner Mask**:
A per-frame whole-body mask that isolates the target runner from the rest of the video.
_Avoid_: segmentation when the intended meaning is whole-runner isolation

**Pose Sequence**:
The target runner's landmarks over time, used as the primary motion representation for similarity.
_Avoid_: skeleton video when referring to the underlying matching data

**DensePose Body Map**:
A per-frame body-surface map that labels visible target-runner pixels by anatomical region.
_Avoid_: pose, skeleton, segmentation when referring to DensePose body-region output

**Fused Form Signal**:
A confidence-weighted form representation that combines pose sequence, runner mask, and DensePose body map evidence.
_Avoid_: score, percentage match

**Form Match**:
An entertainment-oriented resemblance result between a running clip and one or more reference segments.
_Avoid_: diagnosis, coaching assessment, biometric identification

## Relationships

- A **Running Clip** has exactly one **Target Runner**.
- A **Target Runner** has one **Runner Mask** per processed clip.
- A **Pose Sequence** belongs to one **Target Runner** and is the primary input to a **Form Match**.
- A **DensePose Body Map** belongs to one **Target Runner** and refines confidence, occlusion handling, and visual QA.
- A **Fused Form Signal** combines one **Pose Sequence**, one **Runner Mask**, and optionally one **DensePose Body Map**.
- A **Form Match** compares a user **Running Clip** against approved **Reference Segments**.

## Example Dialogue

> **Dev:** "Should the **DensePose Body Map** decide who the runner matches?"
> **Domain expert:** "No. The **Pose Sequence** drives the **Form Match**. The **DensePose Body Map** tells us which joints and frames to trust and makes the QA overlay more convincing."

## Flagged Ambiguities

- "segmentation" was used for both whole-runner isolation and body-part labeling. Resolved: use **Runner Mask** for whole-runner isolation and **DensePose Body Map** for anatomical body-region output.
- "similarity score" risks sounding clinical or precise. Resolved: use **Form Match** and **Fused Form Signal** internally, and avoid percentage-match language in product copy for now.
