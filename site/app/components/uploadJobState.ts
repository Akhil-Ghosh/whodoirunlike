export type ResultReadyJob = {
  result_ready?: boolean;
  artifacts: Record<string, unknown>;
};

export function jobResultReady(job: ResultReadyJob | null | undefined): boolean {
  if (!job) return false;
  if (typeof job.result_ready === "boolean") return job.result_ready;
  return Boolean(job.artifacts["fused_overlay.mp4"]);
}
