"use client";

import {
  ArrowRight,
  BoundingBox,
  CaretDown,
  CheckCircle,
  Clock,
  Copy,
  LockKey,
  SpinnerGap,
  UploadSimple,
  WarningCircle,
  X,
} from "@phosphor-icons/react";
import { motion } from "framer-motion";
import Image from "next/image";
import {
  ChangeEvent,
  CSSProperties,
  DragEvent,
  PointerEvent as ReactPointerEvent,
  useEffect,
  useRef,
  useState,
} from "react";

const apiBaseUrl =
  process.env.NEXT_PUBLIC_API_BASE_URL ??
  (process.env.NODE_ENV === "production" ? "https://api.whodoirunlike.com" : "http://127.0.0.1:8000");
const uploadApiMode =
  process.env.NEXT_PUBLIC_UPLOAD_API_MODE ?? (process.env.NODE_ENV === "production" ? "async" : "sync");
const maxBytes = 75 * 1024 * 1024;
const asyncPollIntervalMs = 5_000;
const asyncPollTimeoutMs = 20 * 60 * 1_000;

type UploadState = "idle" | "ready" | "error" | "loading" | "queued" | "complete";

type RunnerBox = {
  x: number;
  y: number;
  width: number;
  height: number;
};

type PromptPreview = {
  imageUrl: string;
  width: number;
  height: number;
  timeSeconds: number;
};

type TargetPromptPayload = {
  version: 1;
  source: "upload_ui_box_v1";
  selection: {
    type: "box";
    positive_points: Array<{ x: number; y: number; label: string }>;
    negative_points: Array<{ x: number; y: number; label: string }>;
    box: RunnerBox;
  };
  frame: {
    time_seconds: number;
    width: number;
    height: number;
  };
};

type ClipProcessResponse = {
  run_id: string;
  status: string;
  quality: {
    pose_hit_rate?: number;
    usable_rate?: number;
    visibility_mean?: number;
  };
  summary_features: Record<string, number>;
  artifacts: {
    skeleton_render: string;
    qa_overlay: string;
    form_features: string;
  };
};

type WorkerJobResponse = {
  run_id: string;
  status: "uploaded" | "queued" | "running" | "complete" | "failed";
  progress?: {
    phase?: string;
    elapsed_seconds?: number;
  } | null;
  error?: string | null;
  message?: string;
  processor_configured?: boolean;
  artifacts: Record<
    string,
    {
      href: string;
      content_type: string;
      size_bytes: number;
      updated_at?: string;
    }
  >;
};

type UploadResult =
  | { mode: "sync"; data: ClipProcessResponse }
  | { mode: "async"; data: WorkerJobResponse };

const processingSteps = [
  { phase: "uploaded", label: "Upload received" },
  { phase: "queued_on_runpod", label: "GPU worker queued" },
  { phase: "running_full_cv_pipeline", label: "Runner analysis" },
  { phase: "uploading_artifacts", label: "Saving overlays" },
  { phase: "complete", label: "Preview ready" },
];

function sleep(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

async function parseJsonResponse<T>(response: Response): Promise<T> {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message =
      typeof payload === "object" && payload && "detail" in payload
        ? String(payload.detail)
        : typeof payload === "object" && payload && "error" in payload
          ? String(payload.error)
          : "Clip processing failed.";
    throw new Error(message);
  }
  return payload as T;
}

function phaseLabel(job: WorkerJobResponse) {
  const phase = job.progress?.phase?.replaceAll("_", " ");
  const elapsed = job.progress?.elapsed_seconds;
  const elapsedLabel =
    typeof elapsed === "number" && Number.isFinite(elapsed) && elapsed > 0
      ? ` ${Math.round(elapsed)}s elapsed.`
      : "";
  if (job.status === "complete") return "Full pipeline complete.";
  if (job.status === "failed") return job.error || "Pipeline failed.";
  if (phase) return `Pipeline ${phase}.${elapsedLabel}`;
  if (job.message) return job.message;
  return "Clip is queued for the full CV pipeline.";
}

function clamp(value: number, min = 0, max = 1) {
  return Math.max(min, Math.min(max, value));
}

function roundedUnit(value: number) {
  return Math.round(value * 1_000_000) / 1_000_000;
}

function boxCenter(box: RunnerBox) {
  return {
    x: roundedUnit(clamp(box.x + box.width / 2)),
    y: roundedUnit(clamp(box.y + box.height / 2)),
  };
}

function boxStyle(box: RunnerBox): CSSProperties {
  return {
    left: `${box.x * 100}%`,
    top: `${box.y * 100}%`,
    width: `${box.width * 100}%`,
    height: `${box.height * 100}%`,
  };
}

function defaultBoxAround(point: { x: number; y: number }): RunnerBox {
  const width = 0.2;
  const height = 0.62;
  return {
    x: roundedUnit(clamp(point.x - width / 2, 0.02, 1 - width - 0.02)),
    y: roundedUnit(clamp(point.y - height * 0.45, 0.04, 1 - height - 0.04)),
    width,
    height,
  };
}

function boxFromPoints(start: { x: number; y: number }, end: { x: number; y: number }): RunnerBox {
  const x1 = clamp(Math.min(start.x, end.x));
  const y1 = clamp(Math.min(start.y, end.y));
  const x2 = clamp(Math.max(start.x, end.x));
  const y2 = clamp(Math.max(start.y, end.y));
  const width = x2 - x1;
  const height = y2 - y1;

  if (width < 0.04 || height < 0.08) {
    return defaultBoxAround(end);
  }

  return {
    x: roundedUnit(x1),
    y: roundedUnit(y1),
    width: roundedUnit(width),
    height: roundedUnit(height),
  };
}

function promptFromBox(preview: PromptPreview, box: RunnerBox): TargetPromptPayload {
  const center = boxCenter(box);
  return {
    version: 1,
    source: "upload_ui_box_v1",
    selection: {
      type: "box",
      positive_points: [{ ...center, label: "target_runner_center" }],
      negative_points: [],
      box,
    },
    frame: {
      time_seconds: preview.timeSeconds,
      width: preview.width,
      height: preview.height,
    },
  };
}

async function capturePromptPreview(file: File): Promise<PromptPreview> {
  const objectUrl = URL.createObjectURL(file);
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.preload = "metadata";

  try {
    await new Promise<void>((resolve, reject) => {
      video.onloadedmetadata = () => resolve();
      video.onerror = () => reject(new Error("Could not load video metadata."));
      video.src = objectUrl;
      video.load();
    });

    const duration = Number.isFinite(video.duration) ? video.duration : 0;
    const timeSeconds = duration > 0.4 ? Math.min(Math.max(duration * 0.38, 0.2), duration - 0.08) : 0;

    if (timeSeconds > 0.01) {
      await new Promise<void>((resolve, reject) => {
        const timeout = window.setTimeout(() => reject(new Error("Could not seek video preview.")), 3500);
        video.onseeked = () => {
          window.clearTimeout(timeout);
          resolve();
        };
        video.onerror = () => {
          window.clearTimeout(timeout);
          reject(new Error("Could not seek video preview."));
        };
        video.currentTime = timeSeconds;
      });
    } else if (video.readyState < 2) {
      await new Promise<void>((resolve, reject) => {
        const timeout = window.setTimeout(() => reject(new Error("Could not load video frame.")), 3500);
        video.onloadeddata = () => {
          window.clearTimeout(timeout);
          resolve();
        };
        video.onerror = () => {
          window.clearTimeout(timeout);
          reject(new Error("Could not load video frame."));
        };
      });
    }

    const width = video.videoWidth || 960;
    const height = video.videoHeight || 540;
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");
    if (!context) {
      throw new Error("Could not create preview canvas.");
    }
    context.drawImage(video, 0, 0, width, height);
    return {
      imageUrl: canvas.toDataURL("image/jpeg", 0.88),
      width,
      height,
      timeSeconds: Number(timeSeconds.toFixed(3)),
    };
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

function asyncArtifact(result: UploadResult | null) {
  const asyncResult = result?.mode === "async" ? result.data : null;
  return (
    asyncResult?.artifacts["fused_overlay.mp4"] ??
    asyncResult?.artifacts["qa_overlay.mp4"] ??
    asyncResult?.artifacts["skeleton_render.mp4"] ??
    null
  );
}

function phaseIndex(job: WorkerJobResponse | null) {
  if (!job) return 0;
  if (job.status === "complete") return processingSteps.length - 1;
  if (job.status === "failed") return -1;
  const phase = job.progress?.phase;
  const index = processingSteps.findIndex((step) => step.phase === phase);
  if (index >= 0) return index;
  if (job.status === "queued") return 1;
  if (job.status === "running") return 2;
  return 0;
}

function RunnerPromptDialog({
  open,
  preview,
  initialBox,
  onCancel,
  onConfirm,
}: {
  open: boolean;
  preview: PromptPreview | null;
  initialBox: RunnerBox | null;
  onCancel: () => void;
  onConfirm: (box: RunnerBox) => void;
}) {
  const [selectedBox, setSelectedBox] = useState<RunnerBox | null>(initialBox);
  const frameRef = useRef<HTMLDivElement | null>(null);
  const liveOverlayRef = useRef<HTMLDivElement | null>(null);
  const dragStartRef = useRef<{ x: number; y: number } | null>(null);

  useEffect(() => {
    if (open) setSelectedBox(initialBox);
  }, [initialBox, open]);

  useEffect(() => {
    if (!liveOverlayRef.current) return;
    if (selectedBox) {
      Object.assign(liveOverlayRef.current.style, boxStyle(selectedBox), { opacity: "1" });
    } else {
      liveOverlayRef.current.style.opacity = "0";
    }
  }, [selectedBox]);

  if (!open || !preview) return null;

  function unitPoint(event: ReactPointerEvent<HTMLDivElement>) {
    const rect = frameRef.current?.getBoundingClientRect();
    if (!rect) return { x: 0.5, y: 0.5 };
    return {
      x: clamp((event.clientX - rect.left) / rect.width),
      y: clamp((event.clientY - rect.top) / rect.height),
    };
  }

  function paintDraft(box: RunnerBox) {
    if (!liveOverlayRef.current) return;
    Object.assign(liveOverlayRef.current.style, boxStyle(box), { opacity: "1" });
  }

  function handlePointerDown(event: ReactPointerEvent<HTMLDivElement>) {
    event.currentTarget.setPointerCapture(event.pointerId);
    const point = unitPoint(event);
    dragStartRef.current = point;
    paintDraft(defaultBoxAround(point));
  }

  function handlePointerMove(event: ReactPointerEvent<HTMLDivElement>) {
    const start = dragStartRef.current;
    if (!start) return;
    paintDraft(boxFromPoints(start, unitPoint(event)));
  }

  function handlePointerUp(event: ReactPointerEvent<HTMLDivElement>) {
    const start = dragStartRef.current;
    if (!start) return;
    const nextBox = boxFromPoints(start, unitPoint(event));
    dragStartRef.current = null;
    setSelectedBox(nextBox);
    paintDraft(nextBox);
  }

  return (
    <div className="fixed inset-0 z-[80] grid place-items-center bg-[rgba(16,17,19,0.42)] px-4 py-6 backdrop-blur-md" role="dialog" aria-modal="true" aria-label="Select runner to analyze">
      <motion.div
        className="max-h-[94dvh] w-full max-w-[860px] overflow-hidden rounded-[24px] border border-white/55 bg-[#fbfaf7] text-[var(--ink)] shadow-[0_30px_90px_rgba(0,0,0,0.28)]"
        initial={{ opacity: 0, y: 18, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.28, ease: [0.16, 1, 0.3, 1] }}
      >
        <div className="flex items-start justify-between gap-4 border-b border-[var(--line)] px-5 py-4 sm:px-6">
          <div>
            <p className="text-[11px] font-bold uppercase tracking-[0.08em] text-[var(--accent-deep)]">Runner Prompt</p>
            <h3 className="mt-1 text-[25px] font-medium leading-none">Choose the runner</h3>
          </div>
          <button className="focus-ring grid h-10 w-10 place-items-center rounded-full border border-[var(--line)] bg-white/70 text-[var(--ink)] transition hover:bg-white" type="button" onClick={onCancel} aria-label="Close runner selector">
            <X size={18} weight="bold" />
          </button>
        </div>

        <div className="grid gap-5 p-5 sm:p-6 lg:grid-cols-[1fr_230px]">
          <div>
            <div
              ref={frameRef}
              className="relative aspect-video touch-none overflow-hidden rounded-[18px] border border-[rgba(23,23,25,0.12)] bg-[#111214]"
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              onPointerCancel={() => {
                dragStartRef.current = null;
              }}
            >
              <img className="h-full w-full select-none object-contain" src={preview.imageUrl} alt="Selected video frame" draggable={false} />
              <div ref={liveOverlayRef} className="pointer-events-none absolute rounded-[14px] border-2 border-[#5ce0c3] bg-[#5ce0c3]/12 opacity-0 shadow-[0_0_0_9999px_rgba(0,0,0,0.18)]">
                <span className="absolute -left-0.5 -top-8 rounded-full bg-[#101113] px-3 py-1 text-[11px] font-semibold text-white shadow-[0_10px_28px_rgba(0,0,0,0.24)]">
                  Analyze this runner
                </span>
              </div>
            </div>
            <p className="mt-3 text-[13px] leading-[1.45] text-[var(--muted)]">
              Drag a box around the runner you want analyzed. A quick click will place a runner-sized box that you can adjust by dragging again.
            </p>
          </div>

          <aside className="rounded-[18px] border border-[var(--line)] bg-white/66 p-4">
            <BoundingBox size={26} weight="regular" className="text-[var(--accent-deep)]" />
            <p className="mt-4 text-[15px] font-semibold leading-tight">This becomes the first model prompt.</p>
            <p className="mt-2 text-[13px] leading-[1.45] text-[var(--muted)]">
              The backend stores your box as normalized coordinates and uses it to lock onto the right athlete.
            </p>
            <dl className="mt-5 grid grid-cols-2 gap-3 text-[12px]">
              <div className="rounded-xl bg-[#f5f0e8] p-3">
                <dt className="text-[var(--muted)]">Frame</dt>
                <dd className="mt-1 font-semibold text-[var(--ink)]">{preview.timeSeconds.toFixed(1)}s</dd>
              </div>
              <div className="rounded-xl bg-[#f5f0e8] p-3">
                <dt className="text-[var(--muted)]">Size</dt>
                <dd className="mt-1 font-semibold text-[var(--ink)]">{preview.width}x{preview.height}</dd>
              </div>
            </dl>
          </aside>
        </div>

        <div className="flex flex-col-reverse gap-3 border-t border-[var(--line)] px-5 py-4 sm:flex-row sm:justify-end sm:px-6">
          <button className="focus-ring h-11 rounded-full border border-[rgba(23,23,25,0.14)] bg-white/70 px-5 text-[14px] font-medium text-[var(--ink)] transition hover:bg-white" type="button" onClick={onCancel}>
            Cancel
          </button>
          <button className="focus-ring inline-flex h-11 items-center justify-center gap-2 rounded-full bg-[var(--charcoal)] px-5 text-[14px] font-medium text-white transition hover:bg-[#202124] disabled:cursor-not-allowed disabled:opacity-50" type="button" disabled={!selectedBox} onClick={() => selectedBox && onConfirm(selectedBox)}>
            <CheckCircle size={18} weight="regular" />
            Use this runner
          </button>
        </div>
      </motion.div>
    </div>
  );
}

function ProcessingDialog({
  open,
  status,
  message,
  result,
  onClose,
}: {
  open: boolean;
  status: UploadState;
  message: string;
  result: UploadResult | null;
  onClose: () => void;
}) {
  const [showDetails, setShowDetails] = useState(false);
  const [copied, setCopied] = useState(false);
  const job = result?.mode === "async" ? result.data : null;
  const artifact = asyncArtifact(result);
  const currentIndex = phaseIndex(job);
  const isFailed = status === "error" || job?.status === "failed";
  const isComplete = status === "complete" || job?.status === "complete";
  const canCopy = Boolean(job?.run_id && typeof navigator !== "undefined" && navigator.clipboard);
  const helperText = isComplete
    ? "The cloud run finished. Open the overlay when you are ready."
    : isFailed
      ? "Review the error, then try another upload or adjust the selected runner."
      : "The clip is being processed in the cloud. Keep this tab open to watch the result land here.";

  useEffect(() => {
    if (!open) setCopied(false);
  }, [open]);

  if (!open) return null;

  async function copyRunId() {
    if (!job?.run_id || !canCopy) return;
    await navigator.clipboard.writeText(job.run_id);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  }

  return (
    <div className="fixed inset-0 z-[90] grid place-items-center bg-[rgba(16,17,19,0.44)] px-4 py-6 backdrop-blur-md" role="dialog" aria-modal="true" aria-label="Processing status">
      <motion.div
        className="w-full max-w-[620px] overflow-hidden rounded-[24px] border border-white/55 bg-[#fbfaf7] text-[var(--ink)] shadow-[0_30px_90px_rgba(0,0,0,0.28)]"
        initial={{ opacity: 0, y: 18, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.28, ease: [0.16, 1, 0.3, 1] }}
      >
        <div className="flex items-start justify-between gap-4 border-b border-[var(--line)] px-5 py-4 sm:px-6">
          <div>
            <p className="text-[11px] font-bold uppercase tracking-[0.08em] text-[var(--accent-deep)]">Cloud Processing</p>
            <h3 className="mt-1 text-[26px] font-medium leading-none">
              {isComplete ? "Overlay is ready" : isFailed ? "Processing stopped" : "Clip is still processing"}
            </h3>
          </div>
          <button className="focus-ring grid h-10 w-10 place-items-center rounded-full border border-[var(--line)] bg-white/70 text-[var(--ink)] transition hover:bg-white" type="button" onClick={onClose} aria-label="Close processing dialog">
            <X size={18} weight="bold" />
          </button>
        </div>

        <div className="p-5 sm:p-6">
          <div className="rounded-[18px] border border-[var(--line)] bg-white/70 p-4">
            <div className="flex items-start gap-3">
              <span className="mt-0.5 grid h-10 w-10 shrink-0 place-items-center rounded-full bg-[#f3ede3] text-[var(--accent-deep)]">
                {isComplete ? <CheckCircle size={23} weight="regular" /> : isFailed ? <WarningCircle size={23} weight="regular" /> : <SpinnerGap size={23} weight="regular" className="animate-spin" />}
              </span>
              <div>
                <p className="text-[15px] font-semibold leading-tight">{message}</p>
                <p className="mt-1 text-[13px] leading-[1.45] text-[var(--muted)]">{helperText}</p>
              </div>
            </div>
          </div>

          <ol className="mt-5 grid gap-3">
            {processingSteps.map((step, index) => {
              const done = currentIndex >= index && !isFailed;
              const active = currentIndex === index && !isComplete && !isFailed;
              return (
                <li className="grid grid-cols-[34px_1fr] items-center gap-3" key={step.phase}>
                  <span
                    className={[
                      "grid h-[34px] w-[34px] place-items-center rounded-full border text-[14px]",
                      done ? "border-[var(--accent-deep)] bg-[var(--accent-deep)] text-white" : "border-[#dfd7ca] bg-white text-[var(--muted)]",
                      active ? "border-[var(--accent-deep)] text-[var(--accent-deep)]" : "",
                    ].join(" ")}
                  >
                    {done ? <CheckCircle size={18} weight="regular" /> : active ? <Clock size={17} weight="regular" /> : index + 1}
                  </span>
                  <span className={["text-[14px] font-medium", active ? "text-[var(--ink)]" : "text-[var(--muted)]"].join(" ")}>
                    {step.label}
                  </span>
                </li>
              );
            })}
          </ol>

          {job ? (
            <div className="mt-5 rounded-[16px] border border-[var(--line)] bg-[#f8f4ee] p-4">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-[12px] font-bold uppercase tracking-[0.08em] text-[var(--accent-deep)]">Run ID</p>
                  <p className="mt-1 max-w-[280px] truncate text-[13px] font-medium text-[var(--ink)]">{job.run_id}</p>
                </div>
                <button className="focus-ring inline-flex h-10 items-center gap-2 rounded-full border border-[rgba(23,23,25,0.12)] bg-white/70 px-4 text-[13px] font-medium text-[var(--ink)] transition hover:bg-white disabled:opacity-50" type="button" onClick={copyRunId} disabled={!canCopy}>
                  <Copy size={15} weight="regular" />
                  {copied ? "Copied" : "Copy"}
                </button>
              </div>
            </div>
          ) : null}

          <button className="focus-ring mt-4 flex w-full items-center justify-between rounded-[14px] border border-[var(--line)] bg-white/60 px-4 py-3 text-left text-[13px] font-medium text-[var(--ink)] transition hover:bg-white" type="button" onClick={() => setShowDetails((value) => !value)}>
            <span>Processing details</span>
            <CaretDown size={16} weight="bold" className={showDetails ? "rotate-180 transition" : "transition"} />
          </button>

          {showDetails ? (
            <div className="mt-3 rounded-[14px] border border-[var(--line)] bg-white/70 p-4 text-[13px] leading-[1.5] text-[var(--muted)]">
              <p>Status: {job?.status ?? status}</p>
              <p>Phase: {job?.progress?.phase?.replaceAll("_", " ") ?? "waiting for first update"}</p>
              {typeof job?.progress?.elapsed_seconds === "number" ? <p>Elapsed: {Math.round(job.progress.elapsed_seconds)} seconds</p> : null}
              {artifact ? <p>Primary artifact: fused overlay</p> : null}
            </div>
          ) : null}

          <div className="mt-5 flex flex-col-reverse gap-3 sm:flex-row sm:justify-end">
            <button className="focus-ring h-11 rounded-full border border-[rgba(23,23,25,0.14)] bg-white/70 px-5 text-[14px] font-medium text-[var(--ink)] transition hover:bg-white" type="button" onClick={onClose}>
              {isComplete || isFailed ? "Close" : "Hide dialog"}
            </button>
            {artifact ? (
              <a className="focus-ring inline-flex h-11 items-center justify-center gap-2 rounded-full bg-[var(--charcoal)] px-5 text-[14px] font-medium text-white transition hover:bg-[#202124]" href={artifact.href} target="_blank" rel="noreferrer">
                Open overlay
                <ArrowRight size={17} weight="regular" />
              </a>
            ) : null}
          </div>
        </div>
      </motion.div>
    </div>
  );
}

export function UploadCard() {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const previewRequestRef = useRef(0);
  const [file, setFile] = useState<File | null>(null);
  const [fileName, setFileName] = useState("");
  const [status, setStatus] = useState<UploadState>("idle");
  const [message, setMessage] = useState("MP4, MOV or WebM, max 75MB");
  const [isDragging, setIsDragging] = useState(false);
  const [hasConsent, setHasConsent] = useState(false);
  const [result, setResult] = useState<UploadResult | null>(null);
  const [promptPreview, setPromptPreview] = useState<PromptPreview | null>(null);
  const [promptBox, setPromptBox] = useState<RunnerBox | null>(null);
  const [targetPrompt, setTargetPrompt] = useState<TargetPromptPayload | null>(null);
  const [isPromptDialogOpen, setPromptDialogOpen] = useState(false);
  const [isProcessingDialogOpen, setProcessingDialogOpen] = useState(false);
  const [promptStatus, setPromptStatus] = useState<"idle" | "loading" | "ready" | "error">("idle");

  useEffect(() => setResult(null), [file]);

  async function preparePromptPreview(nextFile: File) {
    const requestId = previewRequestRef.current + 1;
    previewRequestRef.current = requestId;
    setPromptStatus("loading");
    setPromptPreview(null);
    setPromptBox(null);
    setTargetPrompt(null);

    try {
      const preview = await capturePromptPreview(nextFile);
      if (previewRequestRef.current !== requestId) return;
      setPromptPreview(preview);
      setPromptStatus("ready");
      setMessage("Choose the runner you want analyzed.");
      setPromptDialogOpen(true);
    } catch {
      if (previewRequestRef.current !== requestId) return;
      setPromptStatus("error");
      setStatus("error");
      setMessage("Could not load a preview frame. Try another clip.");
    }
  }

  function acceptFile(file?: File) {
    if (!file) return;

    if (file.size > maxBytes) {
      setFile(null);
      setFileName("");
      setStatus("error");
      setMessage("Choose a clip under 75MB.");
      return;
    }

    setFile(file);
    setFileName(file.name);
    setStatus("ready");
    setMessage("Preparing a frame so you can select the runner.");
    void preparePromptPreview(file);
  }

  function handleChange(event: ChangeEvent<HTMLInputElement>) {
    acceptFile(event.target.files?.[0]);
  }

  function handleDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setIsDragging(false);
    acceptFile(event.dataTransfer.files?.[0]);
  }

  function confirmPrompt(box: RunnerBox) {
    if (!promptPreview) return;
    setPromptBox(box);
    setTargetPrompt(promptFromBox(promptPreview, box));
    setPromptDialogOpen(false);
    setStatus("ready");
    setMessage("Runner selected. Ready for the hosted pipeline.");
  }

  async function handleSyncAnalyze(file: File) {
    const form = new FormData();
    form.append("file", file);
    form.append("model_variant", "lite");

    const response = await fetch(`${apiBaseUrl}/v1/clips`, {
      method: "POST",
      body: form,
    });
    const payload = await parseJsonResponse<ClipProcessResponse>(response);
    setResult({ mode: "sync", data: payload });
    setStatus("complete");
    setMessage("Processed. Skeleton, QA overlay, and form features are ready.");
  }

  async function pollAsyncJob(runId: string) {
    const deadline = Date.now() + asyncPollTimeoutMs;

    while (Date.now() < deadline) {
      await sleep(asyncPollIntervalMs);
      const response = await fetch(`${apiBaseUrl}/v1/jobs/${runId}`);
      const job = await parseJsonResponse<WorkerJobResponse>(response);
      setResult({ mode: "async", data: job });
      setMessage(phaseLabel(job));

      if (job.status === "complete") {
        setStatus("complete");
        setProcessingDialogOpen(true);
        return;
      }
      if (job.status === "failed") {
        throw new Error(job.error || "Pipeline failed.");
      }
    }

    setStatus("queued");
    setMessage("Uploaded. Processing is still running in the cloud; keep this tab open or save the run ID.");
  }

  async function handleAsyncAnalyze(file: File, prompt: TargetPromptPayload) {
    setProcessingDialogOpen(true);
    setMessage("Uploading clip to the hosted pipeline.");
    const uploadResponse = await fetch(`${apiBaseUrl}/v1/uploads`, {
      method: "POST",
      headers: {
        "Content-Type": file.type || "application/octet-stream",
        "X-Original-Filename": file.name,
        "X-Clip-Consent": "volunteer-pipeline-test",
        "X-Runner-Prompt": encodeURIComponent(JSON.stringify(prompt)),
      },
      body: file,
    });
    const upload = await parseJsonResponse<WorkerJobResponse>(uploadResponse);
    setResult({ mode: "async", data: upload });
    setStatus("queued");
    setMessage("Uploaded. Starting the full CV pipeline.");

    const startResponse = await fetch(`${apiBaseUrl}/v1/jobs/${upload.run_id}/start`, {
      method: "POST",
    });
    const started = await parseJsonResponse<WorkerJobResponse>(startResponse);
    setResult({ mode: "async", data: started });

    if (started.processor_configured === false) {
      setStatus("queued");
      setMessage("Uploaded. The processing service still needs to be connected.");
      return;
    }

    setStatus("loading");
    setMessage(phaseLabel(started));
    await pollAsyncJob(upload.run_id);
  }

  async function handleAnalyze() {
    if (!file) {
      setStatus("error");
      setMessage("Upload a short running clip first.");
      return;
    }
    if (!targetPrompt) {
      setStatus("error");
      setMessage(promptStatus === "loading" ? "Runner selector is still preparing." : "Select the runner to analyze first.");
      if (promptPreview) setPromptDialogOpen(true);
      return;
    }
    if (!hasConsent) {
      setStatus("error");
      setMessage("Confirm you are comfortable helping test the pipeline.");
      return;
    }

    setStatus("loading");
    setMessage(
      uploadApiMode === "async"
        ? "Sending clip to the hosted pipeline."
        : "Processing clip through the local pose API.",
    );

    try {
      if (uploadApiMode === "async") {
        await handleAsyncAnalyze(file, targetPrompt);
      } else {
        await handleSyncAnalyze(file);
      }
    } catch (error) {
      setStatus("error");
      setProcessingDialogOpen(true);
      setMessage(error instanceof Error ? error.message : "Clip processing failed.");
    }
  }

  const isError = status === "error";
  const isPositive = status === "ready" || status === "queued" || status === "complete";
  const syncResult = result?.mode === "sync" ? result.data : null;
  const asyncResult = result?.mode === "async" ? result.data : null;
  const artifact = asyncArtifact(result);
  const isBusy = status === "loading" || status === "queued";

  return (
    <>
      <motion.aside
        id="upload"
        className="upload-glass relative mx-auto w-full max-w-[360px] rounded-[24px] border border-[rgba(27,27,28,0.07)] bg-white/82 p-6 text-[var(--ink)] backdrop-blur-xl sm:p-8 lg:max-w-[300px] 2xl:max-w-[306px] 2xl:p-[29px]"
        aria-label="Volunteer a clip"
        whileHover={{ y: -3 }}
        transition={{ type: "spring", stiffness: 140, damping: 20 }}
      >
        <p className="mb-4 text-[11px] font-bold uppercase tracking-[0.08em] text-[var(--accent)]">Volunteer a Clip</p>
        <h2 className="max-w-[230px] text-[31px] font-medium leading-[0.98] sm:text-[33px]">
          Test the
          <br />
          Pipeline
        </h2>
        <Image
          className="absolute right-12 top-14 w-9 sm:right-14 sm:top-14 sm:w-10"
          src="/assets/ui/accent-rays.svg"
          alt=""
          width={52}
          height={44}
          aria-hidden="true"
        />
        <p className="mt-3.5 hidden text-[14px] leading-[1.46] text-[var(--muted)] sm:block">
          Have a 5-10 second running clip? Share it if you&apos;re comfortable helping test the pipeline.
        </p>

        <label
          className={[
            "focus-ring mt-5 grid min-h-[116px] place-items-center rounded-[18px] border border-dashed p-4 text-center transition duration-300 sm:min-h-[148px] sm:p-5",
            isDragging ? "border-[var(--accent)] bg-[#f8f1e8]" : "border-[#d8cfc2] bg-[#faf7f2]/70",
            isError ? "border-[#b77265] bg-[#fff8f6]" : "",
          ].join(" ")}
          onDragOver={(event) => {
            event.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
        >
          <input
            ref={inputRef}
            className="sr-only"
            type="file"
            accept="video/mp4,video/quicktime,video/webm"
            onChange={handleChange}
          />
          <span className="grid h-11 w-11 place-items-center rounded-full border border-[#d9cec0] bg-white/60 text-[var(--accent)]">
            {isError ? <WarningCircle size={24} weight="regular" /> : isPositive ? <CheckCircle size={24} weight="regular" /> : <UploadSimple size={24} weight="regular" />}
          </span>
          <span className="mt-3 block max-w-full truncate text-[15px] font-medium">
            {fileName || "Upload video"}
          </span>
          <span className={["mt-1 block text-[12px]", isError ? "text-[#935548]" : "text-[var(--muted)]"].join(" ")} aria-live="polite">
            {message}
          </span>
        </label>

        {file ? (
          <button
            className="focus-ring mt-3 flex w-full items-center justify-between rounded-[16px] border border-[var(--line)] bg-[#faf7f2]/80 px-4 py-3 text-left transition hover:bg-white"
            type="button"
            onClick={() => promptPreview && setPromptDialogOpen(true)}
            disabled={!promptPreview}
          >
            <span>
              <span className="block text-[12px] font-bold uppercase tracking-[0.08em] text-[var(--accent-deep)]">Target runner</span>
              <span className="mt-1 block text-[13px] font-medium text-[var(--ink)]">
                {targetPrompt ? "Runner selected" : promptStatus === "loading" ? "Preparing selector" : "Choose runner"}
              </span>
            </span>
            <BoundingBox size={22} weight="regular" className={targetPrompt ? "text-[var(--accent-deep)]" : "text-[var(--muted)]"} />
          </button>
        ) : null}

        <label className="mt-4 grid grid-cols-[18px_1fr] gap-3 text-[12px] leading-[1.35] text-[var(--muted)]">
          <input
            className="mt-0.5 h-[15px] w-[15px] accent-[var(--accent-deep)]"
            type="checkbox"
            checked={hasConsent}
            onChange={(event) => setHasConsent(event.target.checked)}
          />
          <span>I&apos;m comfortable with this clip being processed to test the pipeline.</span>
        </label>

        <motion.button
          className="focus-ring mt-5 flex h-[52px] w-full items-center justify-center gap-4 rounded-full bg-[var(--charcoal)] px-5 text-[15px] font-medium text-white shadow-[inset_0_-1px_0_rgba(255,255,255,0.15),0_12px_28px_rgba(0,0,0,0.16)] transition-colors duration-300 hover:bg-[#202124] disabled:cursor-not-allowed disabled:opacity-60 sm:h-[58px]"
          type="button"
          onClick={handleAnalyze}
          disabled={isBusy}
          whileTap={{ scale: 0.98, y: 1 }}
          transition={{ type: "spring", stiffness: 220, damping: 24 }}
        >
          <span>{status === "loading" ? "Processing..." : status === "queued" ? "Queued" : "Process Clip"}</span>
          <ArrowRight size={21} weight="regular" />
        </motion.button>

        {syncResult ? (
          <div className="mt-5 rounded-lg border border-[#e7ded2] bg-[#faf7f2]/74 p-4">
            <p className="text-[12px] font-bold uppercase text-[var(--accent-deep)]">
              Run {syncResult.run_id.slice(0, 8)}
            </p>
            <p className="mt-2 text-[13px] leading-[1.35] text-[var(--muted)]">
              Pose hit rate {Math.round((syncResult.quality.pose_hit_rate ?? 0) * 100)}%.
              Artifacts are available from the API response.
            </p>
            <a
              className="mt-3 inline-flex text-[13px] font-semibold text-[var(--ink)] underline underline-offset-4"
              href={syncResult.artifacts.qa_overlay}
              target="_blank"
              rel="noreferrer"
            >
              Open QA overlay
            </a>
          </div>
        ) : null}

        {asyncResult ? (
          <div className="mt-5 rounded-lg border border-[#e7ded2] bg-[#faf7f2]/74 p-4">
            <p className="text-[12px] font-bold uppercase text-[var(--accent-deep)]">
              Run {asyncResult.run_id.slice(0, 8)}
            </p>
            <p className="mt-2 text-[13px] leading-[1.35] text-[var(--muted)]">
              {phaseLabel(asyncResult)}
            </p>
            <div className="mt-3 flex flex-wrap gap-3">
              <button className="text-[13px] font-semibold text-[var(--ink)] underline underline-offset-4" type="button" onClick={() => setProcessingDialogOpen(true)}>
                View status
              </button>
              {artifact ? (
                <a
                  className="text-[13px] font-semibold text-[var(--ink)] underline underline-offset-4"
                  href={artifact.href}
                  target="_blank"
                  rel="noreferrer"
                >
                  Open overlay
                </a>
              ) : null}
            </div>
          </div>
        ) : null}

        <p className="mt-4 flex items-center gap-2 text-[12px] text-[var(--muted)]">
          <LockKey size={15} weight="regular" className="text-[var(--accent)]" />
          The API processes short clips and stores review artifacts for this preview.
        </p>
      </motion.aside>

      <RunnerPromptDialog
        open={isPromptDialogOpen}
        preview={promptPreview}
        initialBox={promptBox}
        onCancel={() => setPromptDialogOpen(false)}
        onConfirm={confirmPrompt}
      />
      <ProcessingDialog
        open={isProcessingDialogOpen}
        status={status}
        message={message}
        result={result}
        onClose={() => setProcessingDialogOpen(false)}
      />
    </>
  );
}
