# Pose Sequence Is Primary, DensePose Is a Confidence Layer

The matching pipeline uses the **Pose Sequence** as the primary motion representation and treats the **DensePose Body Map** as a secondary confidence, occlusion, body-region feature, and visual QA layer. DensePose is richer at the pixel/body-part level, but it is slower, heavier to run, more sensitive to body shape and visible surface area, and easier to misuse as appearance similarity; pose sequences are cheaper, more interpretable, and closer to the product's intended form-resemblance signal.
