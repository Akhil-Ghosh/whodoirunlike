"use client";

import { ArrowRight, CheckCircle, LockKey, UploadSimple, WarningCircle } from "@phosphor-icons/react";
import { motion } from "framer-motion";
import Image from "next/image";
import { ChangeEvent, DragEvent, useEffect, useRef, useState } from "react";

const apiBaseUrl =
  process.env.NEXT_PUBLIC_API_BASE_URL ??
  (process.env.NODE_ENV === "production" ? "https://api.whodoirunlike.com" : "http://127.0.0.1:8000");
const uploadApiMode =
  process.env.NEXT_PUBLIC_UPLOAD_API_MODE ?? (process.env.NODE_ENV === "production" ? "async" : "sync");
const maxBytes = 75 * 1024 * 1024;

type UploadState = "idle" | "ready" | "error" | "loading" | "queued" | "complete";

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
    }
  >;
};

type UploadResult =
  | { mode: "sync"; data: ClipProcessResponse }
  | { mode: "async"; data: WorkerJobResponse };

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
  if (job.status === "complete") return "Full pipeline complete.";
  if (job.status === "failed") return job.error || "Pipeline failed.";
  if (phase) return `Pipeline ${phase}.`;
  if (job.message) return job.message;
  return "Clip is queued for the full CV pipeline.";
}

export function UploadCard() {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [fileName, setFileName] = useState("");
  const [status, setStatus] = useState<UploadState>("idle");
  const [message, setMessage] = useState("MP4, MOV or WebM, max 75MB");
  const [isDragging, setIsDragging] = useState(false);
  const [hasConsent, setHasConsent] = useState(false);
  const [result, setResult] = useState<UploadResult | null>(null);

  useEffect(() => setResult(null), [file]);

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
    setMessage("Ready to send to the pose pipeline.");
  }

  function handleChange(event: ChangeEvent<HTMLInputElement>) {
    acceptFile(event.target.files?.[0]);
  }

  function handleDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setIsDragging(false);
    acceptFile(event.dataTransfer.files?.[0]);
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
    for (let attempt = 0; attempt < 40; attempt += 1) {
      await sleep(3000);
      const response = await fetch(`${apiBaseUrl}/v1/jobs/${runId}`);
      const job = await parseJsonResponse<WorkerJobResponse>(response);
      setResult({ mode: "async", data: job });
      setMessage(phaseLabel(job));

      if (job.status === "complete") {
        setStatus("complete");
        return;
      }
      if (job.status === "failed") {
        throw new Error(job.error || "Pipeline failed.");
      }
    }

    setStatus("queued");
    setMessage("Uploaded. Processing is still running; keep the run ID for review.");
  }

  async function handleAsyncAnalyze(file: File) {
    setMessage("Uploading clip to the hosted pipeline.");
    const uploadResponse = await fetch(`${apiBaseUrl}/v1/uploads`, {
      method: "POST",
      headers: {
        "Content-Type": file.type || "application/octet-stream",
        "X-Original-Filename": file.name,
        "X-Clip-Consent": "volunteer-pipeline-test",
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
        await handleAsyncAnalyze(file);
      } else {
        await handleSyncAnalyze(file);
      }
    } catch (error) {
      setStatus("error");
      setMessage(error instanceof Error ? error.message : "Clip processing failed.");
    }
  }

  const isError = status === "error";
  const isPositive = status === "ready" || status === "queued" || status === "complete";
  const syncResult = result?.mode === "sync" ? result.data : null;
  const asyncResult = result?.mode === "async" ? result.data : null;
  const asyncArtifact =
    asyncResult?.artifacts["fused_overlay.mp4"] ??
    asyncResult?.artifacts["qa_overlay.mp4"] ??
    asyncResult?.artifacts["skeleton_render.mp4"];

  return (
    <motion.aside
      id="upload"
      className="upload-glass relative mx-auto w-full max-w-[360px] rounded-[24px] border border-[rgba(27,27,28,0.07)] bg-white/82 p-6 text-[var(--ink)] backdrop-blur-xl sm:p-8 lg:max-w-[300px] 2xl:max-w-[306px] 2xl:p-[29px]"
      aria-label="Volunteer a clip"
      whileHover={{ y: -3 }}
      transition={{ type: "spring", stiffness: 140, damping: 20 }}
    >
      <p className="mb-4 text-[11px] font-bold uppercase text-[var(--accent)]">Volunteer a Clip</p>
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
        className="focus-ring mt-5 flex h-[52px] w-full items-center justify-center gap-4 rounded-full bg-[var(--charcoal)] px-5 text-[15px] font-medium text-white shadow-[inset_0_-1px_0_rgba(255,255,255,0.15),0_12px_28px_rgba(0,0,0,0.16)] transition-colors duration-300 hover:bg-[#202124] sm:h-[58px]"
        type="button"
        onClick={handleAnalyze}
        disabled={status === "loading"}
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
          {asyncArtifact ? (
            <a
              className="mt-3 inline-flex text-[13px] font-semibold text-[var(--ink)] underline underline-offset-4"
              href={asyncArtifact.href}
              target="_blank"
              rel="noreferrer"
            >
              Open pipeline overlay
            </a>
          ) : null}
        </div>
      ) : null}

      <p className="mt-4 flex items-center gap-2 text-[12px] text-[var(--muted)]">
        <LockKey size={15} weight="regular" className="text-[var(--accent)]" />
        The API processes short clips and stores review artifacts for this preview.
      </p>
    </motion.aside>
  );
}
