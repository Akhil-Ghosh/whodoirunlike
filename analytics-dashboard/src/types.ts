export type QueryName =
  | "overview"
  | "stage_latency"
  | "span_latency"
  | "attempts"
  | "attempt_detail"
  | "failures"
  | "stalls"
  | "freshness";

export type QueryValue = string | number | boolean;
export type QueryParameters = Record<string, QueryValue>;

export interface QueryStartResponse {
  query: QueryName;
  query_execution_id: string;
  state: "QUEUED";
  poll_after_ms: number;
}

export interface QueryResultResponse<Row> {
  query: QueryName;
  query_execution_id: string;
  state: "QUEUED" | "RUNNING" | "SUCCEEDED" | "FAILED" | "CANCELLED";
  poll_after_ms?: number;
  rows?: Row[];
  row_count?: number;
  statistics?: {
    engine_execution_ms?: number | null;
    queue_ms?: number | null;
    data_scanned_bytes?: number | null;
    reused_previous_result?: boolean;
  };
  error?: { message: string; category?: number; type?: number; retryable?: boolean };
}

export interface OverviewRow {
  attempts: number;
  result_ready_attempts: number;
  completed_attempts: number;
  failed_attempts: number;
  terminal_attempts: number;
  success_rate: number | null;
  failure_rate: number | null;
  p50_result_ready_seconds: number | null;
  p90_result_ready_seconds: number | null;
  p95_result_ready_seconds: number | null;
  average_result_ready_seconds: number | null;
  bottleneck_stage: string | null;
  bottleneck_samples: number | null;
  bottleneck_p95_seconds: number | null;
}

export interface StageLatencyRow {
  stage: string;
  samples: number;
  failures: number;
  p50_seconds: number | null;
  p90_seconds: number | null;
  p95_seconds: number | null;
  average_ms_per_frame: number | null;
  confidence: "low" | "stable";
}

export interface SpanLatencyRow {
  stage: string;
  span: string;
  samples: number;
  occurrences: number;
  failures: number;
  p50_seconds: number | null;
  p90_seconds: number | null;
  p95_seconds: number | null;
  confidence: "low" | "stable";
}

export interface AttemptRow {
  run_id: string;
  attempt_id: string;
  first_event_at: string;
  last_event_at: string;
  status: "complete" | "failed" | "running";
  environment: string | null;
  backend: string | null;
  gpu_type: string | null;
  processor_version: string | null;
  cold_start: boolean | null;
  duration_bucket: string | null;
  resolution_bucket: string | null;
  clip_duration_seconds: number | null;
  clip_frame_count: number | null;
  clip_width: number | null;
  clip_height: number | null;
  result_ready_seconds: number | null;
  analysis_complete_seconds: number | null;
  attempt_complete_seconds: number | null;
  observed_stage_seconds: number;
  unattributed_seconds: number;
  bottleneck_stage: string | null;
  bottleneck_seconds: number | null;
  last_stage: string | null;
  last_span: string | null;
  last_event_type: string;
}

export interface AttemptEventRow {
  run_id: string;
  attempt_id: string;
  sequence: number;
  event_type: string;
  stage: string | null;
  span: string | null;
  status: string | null;
  event_time: string;
  elapsed_seconds: number | null;
  start_offset_seconds: number;
  end_offset_seconds: number;
  timing_basis: string | null;
  artifact_type: string | null;
  error_class: string | null;
  error_code: string | null;
}

export interface FailureRow {
  stage: string | null;
  span: string | null;
  error_class: string | null;
  error_code: string | null;
  failures: number;
  affected_attempts: number;
  p50_time_to_failure_seconds: number | null;
  p95_time_to_failure_seconds: number | null;
  most_recent_at: string;
}

export interface StallRow {
  run_id: string;
  attempt_id: string;
  first_event_at: string;
  last_event_at: string;
  stale_seconds: number;
  last_stage: string | null;
  last_span: string | null;
  last_event_type: string;
}

export interface FreshnessRow {
  latest_event_at: string | null;
  latest_ingested_at: string | null;
  event_age_seconds: number | null;
  latest_ingestion_lag_seconds: number | null;
  events_last_24_hours: number;
  attempts_last_24_hours: number;
}

export interface DashboardFilters {
  rangeDays: 1 | 7 | 14 | 30 | 90;
  environment: string;
  durationBucket: string;
  gpuType: string;
  backend: string;
  coldStart: "all" | "cold" | "warm";
}

export interface DashboardData {
  overview: OverviewRow | null;
  stages: StageLatencyRow[];
  spans: SpanLatencyRow[];
  attempts: AttemptRow[];
  failures: FailureRow[];
  stalls: StallRow[];
  freshness: FreshnessRow | null;
  coldStages: StageLatencyRow[];
  warmStages: StageLatencyRow[];
}
