export type DisplayMetric = {
  label: string;
  value: string;
  note: string;
};

export type ClipOutput = {
  label: string;
  href: string;
  note: string;
};

export type ProcessedClip = {
  label: string;
  context: string;
  duration: string;
  quality: string;
  sourceUrl: string;
  strideNotes: string[];
  metrics: DisplayMetric[];
  outputs: ClipOutput[];
};

export type AnalyzedRunner = {
  slug: string;
  name: string;
  event: string;
  image: string;
  imageAlt: string;
  clipCount: number;
  cardBlurb: string;
  detailBlurb: string;
  stridePattern: string;
  metricSummary: DisplayMetric[];
  clips: ProcessedClip[];
};

export const analyzedRunners: AnalyzedRunner[] = [
  {
    slug: "cole-hocker",
    name: "Cole Hocker",
    event: "1500m track samples",
    image: "/assets/gallery/runners/cole-hocker-finish.webp",
    imageAlt: "Cole Hocker finishing ahead of Josh Kerr and Jakob Ingebrigtsen",
    clipCount: 2,
    cardBlurb: "Fast 1500m samples with a compact carriage, quick rhythm, and higher knee sweep near the finish.",
    detailBlurb:
      "Cole is the best current demo case because the clips force the pipeline to follow one runner through a pack and then through a fast finishing segment.",
    stridePattern: "Compact upper body, quick turnover, and a bigger knee-angle sweep when the finish segment opens up.",
    metricSummary: [
      { label: "Knee sweep", value: "124-136 deg", note: "Average left/right knee angle range across the two processed clips." },
      { label: "Hip bounce", value: "5.9-7.1%", note: "Normalized hip vertical oscillation proxy from pose landmarks." },
      { label: "Readable frames", value: "91-100%", note: "Frames where the pipeline kept enough pose signal to use the stride." },
    ],
    clips: [
      {
        label: "Pack tracking segment",
        context: "A longer 1500m excerpt with camera motion, broadcast graphics, and pack traffic.",
        duration: "8.66s",
        quality: "236 of 260 usable frames",
        sourceUrl: "https://www.youtube.com/watch?v=2sb32uxUO10",
        strideNotes: [
          "The tracker keeps enough continuity to read the stride even when the pack tightens around him.",
          "Knee-angle sweep sits around 124 deg, with hip bounce just under 6%.",
          "Feet are the weakest read in this clip, so the stride note leans more on knees, hips, and torso.",
        ],
        metrics: [
          { label: "Stride rhythm proxy", value: "7.75", note: "Peak spacing in the lower-body motion signal." },
          { label: "Knee lift proxy", value: "32%", note: "Relative vertical knee travel in the normalized pose." },
          { label: "Pose visibility", value: "81%", note: "Mean landmark visibility across the usable clip." },
        ],
        outputs: [
          { label: "Runner isolation", href: "/assets/gallery/clips/cole-hocker/pack-isolation.mp4", note: "The selected runner mask over the source segment." },
          { label: "Stride skeleton", href: "/assets/gallery/clips/cole-hocker/pack-skeleton.mp4", note: "Pose landmarks rendered frame by frame." },
        ],
      },
      {
        label: "Finish-phase segment",
        context: "A shorter 1500m sample where the runner is easier to see and the stride opens up.",
        duration: "3.98s",
        quality: "120 of 120 usable frames",
        sourceUrl: "https://www.youtube.com/watch?v=Ct7MBxmwZ7M",
        strideNotes: [
          "The cleaner view produces a stronger pose read and a larger knee-angle sweep.",
          "The arm swing proxy rises sharply, which matches the late-race drive in the clip.",
          "This is the best sample for showing the pipeline at full confidence right now.",
        ],
        metrics: [
          { label: "Stride rhythm proxy", value: "7.99", note: "Peak spacing in the lower-body motion signal." },
          { label: "Knee lift proxy", value: "41%", note: "Relative vertical knee travel in the normalized pose." },
          { label: "Pose visibility", value: "85%", note: "Mean landmark visibility across the usable clip." },
        ],
        outputs: [
          { label: "Runner isolation", href: "/assets/gallery/clips/cole-hocker/finish-isolation.mp4", note: "The selected runner mask over the source segment." },
          { label: "Stride skeleton", href: "/assets/gallery/clips/cole-hocker/finish-skeleton.mp4", note: "Pose landmarks rendered frame by frame." },
        ],
      },
    ],
  },
  {
    slug: "josh-kerr",
    name: "Josh Kerr",
    event: "Championship 1500m sample",
    image: "/assets/gallery/runners/josh-kerr.webp",
    imageAlt: "Josh Kerr racing on a track",
    clipCount: 1,
    cardBlurb: "A dense championship clip with high rhythm signal and a large knee-angle sweep.",
    detailBlurb:
      "Kerr's sample is useful because the clip is crowded but still readable. The model has to keep choosing the right runner instead of the easiest body in frame.",
    stridePattern: "Upright pack running, high rhythm proxy, and a strong knee-angle sweep from the usable frames.",
    metricSummary: [
      { label: "Knee sweep", value: "147 deg", note: "Average left/right knee angle range in the processed clip." },
      { label: "Hip bounce", value: "8.7%", note: "Normalized hip vertical oscillation proxy from pose landmarks." },
      { label: "Readable frames", value: "100%", note: "Frames where the pipeline kept enough pose signal to use the stride." },
    ],
    clips: [
      {
        label: "Championship pack segment",
        context: "A 1500m race clip with multiple runners close together and a broadcast camera angle.",
        duration: "5.83s",
        quality: "146 of 146 usable frames",
        sourceUrl: "https://www.youtube.com/watch?v=cMwaUbgf2Zs",
        strideNotes: [
          "The rhythm proxy is the highest in the current gallery set.",
          "Knee sweep is large, especially on the right-side read.",
          "Pose visibility is lower than the cleaner clips, which makes this a useful identity and tracking test.",
        ],
        metrics: [
          { label: "Stride rhythm proxy", value: "9.42", note: "Peak spacing in the lower-body motion signal." },
          { label: "Knee lift proxy", value: "30%", note: "Relative vertical knee travel in the normalized pose." },
          { label: "Pose visibility", value: "72%", note: "Mean landmark visibility across the usable clip." },
        ],
        outputs: [
          { label: "Runner isolation", href: "/assets/gallery/clips/josh-kerr/championship-isolation.mp4", note: "The selected runner mask over the source segment." },
          { label: "Stride skeleton", href: "/assets/gallery/clips/josh-kerr/championship-skeleton.mp4", note: "Pose landmarks rendered frame by frame." },
        ],
      },
    ],
  },
  {
    slug: "keely-hodgkinson",
    name: "Keely Hodgkinson",
    event: "800m track sample",
    image: "/assets/gallery/runners/keely-hodgkinson.webp",
    imageAlt: "Keely Hodgkinson racing indoors",
    clipCount: 1,
    cardBlurb: "A cleaner side-view 800m read with low hip bounce and strong knee lift.",
    detailBlurb:
      "Hodgkinson's clip gives the model a clearer side-view stride. It is the best current sample for checking lower-body timing without much pack confusion.",
    stridePattern: "Low hip bounce, strong knee-lift proxy, and a clean enough side angle to read the lower body.",
    metricSummary: [
      { label: "Knee sweep", value: "135 deg", note: "Average left/right knee angle range in the processed clip." },
      { label: "Hip bounce", value: "2.4%", note: "Normalized hip vertical oscillation proxy from pose landmarks." },
      { label: "Readable frames", value: "100%", note: "Frames where the pipeline kept enough pose signal to use the stride." },
    ],
    clips: [
      {
        label: "Indoor 800m segment",
        context: "A side-view indoor clip with a clearer look at stride mechanics.",
        duration: "7.85s",
        quality: "235 of 235 usable frames",
        sourceUrl: "https://www.youtube.com/watch?v=3oVZ5B9Pw_s",
        strideNotes: [
          "Hip vertical movement is the lowest in the gallery set.",
          "Knee lift proxy is high while the skeleton remains stable across the full clip.",
          "The side-view angle makes this a good sample for testing hip, knee, and ankle timing.",
        ],
        metrics: [
          { label: "Stride rhythm proxy", value: "7.52", note: "Peak spacing in the lower-body motion signal." },
          { label: "Knee lift proxy", value: "41%", note: "Relative vertical knee travel in the normalized pose." },
          { label: "Pose visibility", value: "88%", note: "Mean landmark visibility across the usable clip." },
        ],
        outputs: [
          { label: "Runner isolation", href: "/assets/gallery/clips/keely-hodgkinson/indoor-isolation.mp4", note: "The selected runner mask over the source segment." },
          { label: "Stride skeleton", href: "/assets/gallery/clips/keely-hodgkinson/indoor-skeleton.mp4", note: "Pose landmarks rendered frame by frame." },
        ],
      },
    ],
  },
  {
    slug: "kelvin-kiptum",
    name: "Kelvin Kiptum",
    event: "Marathon road sample",
    image: "/assets/gallery/runners/kelvin-kiptum.webp",
    imageAlt: "Kelvin Kiptum running a road marathon",
    clipCount: 1,
    cardBlurb: "A road-running sample with a harder camera angle and a noisier stride read.",
    detailBlurb:
      "Kiptum's clip moves the system away from track footage. The pose read is harder, but that makes it useful for testing road angles and pacing groups.",
    stridePattern: "Road-race rhythm, higher hip-motion proxy, and lower pose visibility because the camera angle is less controlled.",
    metricSummary: [
      { label: "Knee sweep", value: "124 deg", note: "Average left/right knee angle range in the processed clip." },
      { label: "Hip bounce", value: "15.9%", note: "Normalized hip vertical oscillation proxy from pose landmarks." },
      { label: "Readable frames", value: "96%", note: "Frames where the pipeline kept enough pose signal to use the stride." },
    ],
    clips: [
      {
        label: "Road marathon segment",
        context: "A road clip with a pacing pack and a less stable broadcast angle.",
        duration: "3.56s",
        quality: "103 of 107 usable frames",
        sourceUrl: "https://www.youtube.com/watch?v=oC3_2RTzb0Q",
        strideNotes: [
          "The camera angle lowers pose visibility, especially compared with the track samples.",
          "The hip-motion proxy is high, so the page flags this as a clip to review rather than a clean comparison target.",
          "This is the best current sample for testing whether the pipeline can leave the track setting.",
        ],
        metrics: [
          { label: "Stride rhythm proxy", value: "1.16", note: "Peak spacing in the lower-body motion signal." },
          { label: "Knee lift proxy", value: "79%", note: "Relative vertical knee travel in the normalized pose." },
          { label: "Pose visibility", value: "46%", note: "Mean landmark visibility across the usable clip." },
        ],
        outputs: [
          { label: "Runner isolation", href: "/assets/gallery/clips/kelvin-kiptum/road-isolation.mp4", note: "The selected runner mask over the source segment." },
          { label: "Stride skeleton", href: "/assets/gallery/clips/kelvin-kiptum/road-skeleton.mp4", note: "Pose landmarks rendered frame by frame." },
        ],
      },
    ],
  },
  {
    slug: "jakob-ingebrigtsen",
    name: "Jakob Ingebrigtsen",
    event: "Middle-distance reference",
    image: "/assets/gallery/runners/jakob-ingebrigtsen.webp",
    imageAlt: "Jakob Ingebrigtsen racing on a track",
    clipCount: 1,
    cardBlurb: "A mid-distance reference with moderate knee lift and a less forgiving source clip.",
    detailBlurb:
      "Ingebrigtsen's sample is not the cleanest file in the set. It is useful because the page can show where a reference clip is still worth keeping, but should be treated with caution.",
    stridePattern: "Efficient carriage with moderate knee lift, a stable rhythm proxy, and weaker foot visibility than the cleaner clips.",
    metricSummary: [
      { label: "Knee sweep", value: "120 deg", note: "Average left/right knee angle range in the processed clip." },
      { label: "Hip bounce", value: "6.1%", note: "Normalized hip vertical oscillation proxy from pose landmarks." },
      { label: "Readable frames", value: "100%", note: "Frames where the pipeline kept enough pose signal to use the stride." },
    ],
    clips: [
      {
        label: "Middle-distance reference segment",
        context: "A track clip that works as a reference, but not a pristine one.",
        duration: "4.42s",
        quality: "132 of 132 usable frames",
        sourceUrl: "https://www.youtube.com/watch?v=Hqu1WQR9qX4",
        strideNotes: [
          "The clip keeps a readable skeleton, but foot visibility is weaker than the main track samples.",
          "Knee sweep is the lowest in the gallery set, which makes it a useful contrast case.",
          "The torso range is wider than the cleanest samples, so this page treats it as a tolerance test.",
        ],
        metrics: [
          { label: "Stride rhythm proxy", value: "7.04", note: "Peak spacing in the lower-body motion signal." },
          { label: "Knee lift proxy", value: "33%", note: "Relative vertical knee travel in the normalized pose." },
          { label: "Pose visibility", value: "77%", note: "Mean landmark visibility across the usable clip." },
        ],
        outputs: [
          { label: "Runner isolation", href: "/assets/gallery/clips/jakob-ingebrigtsen/reference-isolation.mp4", note: "The selected runner mask over the source segment." },
          { label: "Stride skeleton", href: "/assets/gallery/clips/jakob-ingebrigtsen/reference-skeleton.mp4", note: "Pose landmarks rendered frame by frame." },
        ],
      },
    ],
  },
];
