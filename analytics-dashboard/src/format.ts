const STAGE_LABELS: Record<string, string> = {
  source_ingest: "Source ingest",
  processor_enqueue: "Processor enqueue",
  processor_queue: "Queue",
  source_download: "Source download",
  run_preparation: "Run preparation",
  target_tracking: "Target tracking",
  runner_mask: "Runner mask",
  pose_sequence: "Pose sequence",
  densepose_body_map: "DensePose",
  fused_form_signal: "Fused form signal",
  form_feature_compilation: "Feature compilation",
  artifact_table_export: "Table export",
  quality_control: "Quality control",
  artifact_publish: "Artifact publish",
  result_ready: "Result ready",
  analysis_complete: "Analysis complete",
};

const SPAN_LABELS: Record<string, string> = {
  model_load: "Model load",
  decode: "Decode",
  preprocess: "Preprocess",
  inference: "Inference",
  postprocess: "Postprocess",
  render: "Render",
  encode: "Encode",
  write: "Write",
  publish: "Publish",
};

export function stageLabel(value: string | null | undefined): string {
  if (!value) return "Unknown";
  return STAGE_LABELS[value] ?? value.replaceAll("_", " ").replace(/^./, (letter) => letter.toUpperCase());
}

export function spanLabel(value: string | null | undefined): string {
  if (!value) return "Unknown";
  return SPAN_LABELS[value] ?? stageLabel(value);
}

export function seconds(value: number | null | undefined, digits = 1): string {
  return value == null || !Number.isFinite(value) ? "—" : `${value.toFixed(digits)}s`;
}

export function percent(value: number | null | undefined, digits = 1): string {
  return value == null || !Number.isFinite(value) ? "—" : `${(value * 100).toFixed(digits)}%`;
}

export function shortId(value: string | null | undefined): string {
  return value ? value.slice(0, 8) : "—";
}

export function compactNumber(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(value);
}

export function utcTime(value: string | null | undefined): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) return "—";
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
    timeZone: "UTC",
  }).format(date);
}

export function ageLabel(secondsAgo: number | null | undefined): string {
  if (secondsAgo == null || !Number.isFinite(secondsAgo)) return "Awaiting first event";
  if (secondsAgo < 60) return `Updated ${Math.max(0, Math.round(secondsAgo))}s ago`;
  if (secondsAgo < 3_600) return `Updated ${Math.round(secondsAgo / 60)}m ago`;
  return `Updated ${Math.round(secondsAgo / 3_600)}h ago`;
}
