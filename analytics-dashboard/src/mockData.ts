import type { AttemptEventRow, DashboardData } from "./types";

const attemptOne = "9f3c2b7a-01bf-4a25-961e-d459be5ee034";

export const mockData: DashboardData = {
  overview: {
    attempts: 126,
    result_ready_attempts: 119,
    completed_attempts: 119,
    failed_attempts: 7,
    terminal_attempts: 126,
    success_rate: 119 / 126,
    failure_rate: 7 / 126,
    p50_result_ready_seconds: 83.4,
    p90_result_ready_seconds: 176.2,
    p95_result_ready_seconds: 214.7,
    average_result_ready_seconds: 101.8,
    bottleneck_stage: "runner_mask",
    bottleneck_samples: 42,
    bottleneck_p95_seconds: 108.2,
  },
  stages: [
    { stage: "runner_mask", samples: 42, failures: 2, p50_seconds: 24.1, p90_seconds: 81.4, p95_seconds: 108.2, average_ms_per_frame: 254.2, confidence: "stable" },
    { stage: "pose_sequence", samples: 118, failures: 1, p50_seconds: 18.6, p90_seconds: 43.8, p95_seconds: 57.3, average_ms_per_frame: 132.1, confidence: "stable" },
    { stage: "target_tracking", samples: 118, failures: 1, p50_seconds: 11.2, p90_seconds: 24.7, p95_seconds: 29.7, average_ms_per_frame: 77.4, confidence: "stable" },
    { stage: "densepose_body_map", samples: 117, failures: 0, p50_seconds: 9.7, p90_seconds: 19.2, p95_seconds: 22.8, average_ms_per_frame: 63.2, confidence: "stable" },
    { stage: "source_download", samples: 126, failures: 1, p50_seconds: 6.1, p90_seconds: 10.8, p95_seconds: 12.4, average_ms_per_frame: null, confidence: "stable" },
    { stage: "artifact_publish", samples: 125, failures: 2, p50_seconds: 2.3, p90_seconds: 4.1, p95_seconds: 4.6, average_ms_per_frame: null, confidence: "stable" },
  ],
  spans: [
    { stage: "runner_mask", span: "inference", samples: 42, occurrences: 42, failures: 1, p50_seconds: 16.8, p90_seconds: 55.9, p95_seconds: 71.6, confidence: "stable" },
    { stage: "runner_mask", span: "render", samples: 42, occurrences: 42, failures: 0, p50_seconds: 7.2, p90_seconds: 15.3, p95_seconds: 18.7, confidence: "stable" },
    { stage: "runner_mask", span: "model_load", samples: 42, occurrences: 42, failures: 0, p50_seconds: 4.2, p90_seconds: 11.5, p95_seconds: 14.5, confidence: "stable" },
    { stage: "runner_mask", span: "encode", samples: 42, occurrences: 42, failures: 0, p50_seconds: 5.8, p90_seconds: 9.6, p95_seconds: 11.5, confidence: "stable" },
  ],
  attempts: [
    { run_id: "4f39b1e5-199b-4f19-886f-447b9a4aaed4", attempt_id: attemptOne, first_event_at: "2026-07-09T21:12:13Z", last_event_at: "2026-07-09T21:15:21Z", status: "complete", environment: "production", backend: "sam31_gpu", gpu_type: "NVIDIA L4", processor_version: "f45ff8c", cold_start: true, duration_bucket: "5_10s", resolution_bucket: "hd", clip_duration_seconds: 8.1, clip_frame_count: 243, clip_width: 1280, clip_height: 720, result_ready_seconds: 187.6, analysis_complete_seconds: 187.6, attempt_complete_seconds: 190.7, observed_stage_seconds: 184.5, unattributed_seconds: 3.1, bottleneck_stage: "runner_mask", bottleneck_seconds: 116.3, last_stage: "analysis_complete", last_span: null, last_event_type: "attempt_completed" },
    { run_id: "aad39bf0-32fb-4b29-a34a-a62776e81287", attempt_id: "a83c2b7e-37aa-4a36-a81c-2a09ff13bbb0", first_event_at: "2026-07-09T20:43:12Z", last_event_at: "2026-07-09T20:45:33Z", status: "failed", environment: "production", backend: "sam31_gpu", gpu_type: "NVIDIA L4", processor_version: "f45ff8c", cold_start: false, duration_bucket: "5_10s", resolution_bucket: "hd", clip_duration_seconds: 7.4, clip_frame_count: 222, clip_width: 1280, clip_height: 720, result_ready_seconds: null, analysis_complete_seconds: null, attempt_complete_seconds: null, observed_stage_seconds: 92.1, unattributed_seconds: 0, bottleneck_stage: "runner_mask", bottleneck_seconds: 92.1, last_stage: "runner_mask", last_span: "inference", last_event_type: "attempt_failed" },
    { run_id: "eb47f711-0ad5-46f9-8fbc-02f03ca0c256", attempt_id: "bb49be37-148d-4453-865c-bd3350243a02", first_event_at: "2026-07-09T19:18:12Z", last_event_at: "2026-07-09T19:20:21Z", status: "complete", environment: "production", backend: "sam31_gpu", gpu_type: "NVIDIA L4", processor_version: "f45ff8c", cold_start: false, duration_bucket: "5_10s", resolution_bucket: "hd", clip_duration_seconds: 6.2, clip_frame_count: 186, clip_width: 1280, clip_height: 720, result_ready_seconds: 129.2, analysis_complete_seconds: 132.1, attempt_complete_seconds: 134.0, observed_stage_seconds: 126.7, unattributed_seconds: 7.3, bottleneck_stage: "runner_mask", bottleneck_seconds: 61.2, last_stage: "analysis_complete", last_span: null, last_event_type: "attempt_completed" },
  ],
  failures: [
    { stage: "runner_mask", span: "inference", error_class: "CudaOutOfMemory", error_code: "CUDA_OUT_OF_MEMORY", failures: 3, affected_attempts: 3, p50_time_to_failure_seconds: 92.1, p95_time_to_failure_seconds: 120.5, most_recent_at: "2026-07-09T20:45:33Z" },
    { stage: "source_download", span: null, error_class: "SourceUnavailable", error_code: "SOURCE_NOT_FOUND", failures: 2, affected_attempts: 2, p50_time_to_failure_seconds: 12.4, p95_time_to_failure_seconds: 14.1, most_recent_at: "2026-07-09T19:58:11Z" },
    { stage: "target_tracking", span: "inference", error_class: "TrackerLost", error_code: "TRACKER_LOST", failures: 1, affected_attempts: 1, p50_time_to_failure_seconds: 24.7, p95_time_to_failure_seconds: 24.7, most_recent_at: "2026-07-09T18:21:19Z" },
  ],
  stalls: [],
  freshness: { latest_event_at: new Date(Date.now() - 120_000).toISOString(), latest_ingested_at: new Date(Date.now() - 116_000).toISOString(), event_age_seconds: 120, latest_ingestion_lag_seconds: 4, events_last_24_hours: 5821, attempts_last_24_hours: 126 },
  coldStages: [
    { stage: "runner_mask", samples: 28, failures: 1, p50_seconds: 82.4, p90_seconds: 121.2, p95_seconds: 136.2, average_ms_per_frame: 310, confidence: "stable" },
    { stage: "run_preparation", samples: 28, failures: 0, p50_seconds: 47.1, p90_seconds: 58.2, p95_seconds: 62.5, average_ms_per_frame: null, confidence: "stable" },
    { stage: "artifact_publish", samples: 28, failures: 0, p50_seconds: 14.3, p90_seconds: 19.2, p95_seconds: 21.7, average_ms_per_frame: null, confidence: "stable" },
  ],
  warmStages: [
    { stage: "runner_mask", samples: 98, failures: 1, p50_seconds: 23.7, p90_seconds: 54.3, p95_seconds: 67.9, average_ms_per_frame: 142, confidence: "stable" },
    { stage: "run_preparation", samples: 98, failures: 0, p50_seconds: 5.8, p90_seconds: 8.1, p95_seconds: 9.2, average_ms_per_frame: null, confidence: "stable" },
    { stage: "artifact_publish", samples: 98, failures: 1, p50_seconds: 5.7, p90_seconds: 8.4, p95_seconds: 9.6, average_ms_per_frame: null, confidence: "stable" },
  ],
};

export const mockAttemptEvents: AttemptEventRow[] = [
  { run_id: mockData.attempts[0].run_id, attempt_id: attemptOne, sequence: 7, event_type: "stage_completed", stage: "processor_queue", span: null, status: "completed", event_time: "2026-07-09T21:12:18Z", elapsed_seconds: 4.8, start_offset_seconds: 0, end_offset_seconds: 4.8, timing_basis: "runpod_delay_time", artifact_type: null, error_class: null, error_code: null },
  { run_id: mockData.attempts[0].run_id, attempt_id: attemptOne, sequence: 111, event_type: "stage_completed", stage: "source_download", span: null, status: "completed", event_time: "2026-07-09T21:12:30Z", elapsed_seconds: 11.9, start_offset_seconds: 4.8, end_offset_seconds: 16.7, timing_basis: "monotonic", artifact_type: null, error_class: null, error_code: null },
  { run_id: mockData.attempts[0].run_id, attempt_id: attemptOne, sequence: 121, event_type: "stage_completed", stage: "target_tracking", span: null, status: "completed", event_time: "2026-07-09T21:12:50Z", elapsed_seconds: 20.0, start_offset_seconds: 16.7, end_offset_seconds: 36.7, timing_basis: "monotonic", artifact_type: null, error_class: null, error_code: null },
  { run_id: mockData.attempts[0].run_id, attempt_id: attemptOne, sequence: 131, event_type: "stage_completed", stage: "runner_mask", span: null, status: "completed", event_time: "2026-07-09T21:14:46Z", elapsed_seconds: 116.3, start_offset_seconds: 36.7, end_offset_seconds: 153.0, timing_basis: "monotonic", artifact_type: null, error_class: null, error_code: null },
  { run_id: mockData.attempts[0].run_id, attempt_id: attemptOne, sequence: 123, event_type: "span_completed", stage: "runner_mask", span: "model_load", status: "completed", event_time: "2026-07-09T21:13:04Z", elapsed_seconds: 14.5, start_offset_seconds: 36.7, end_offset_seconds: 51.2, timing_basis: "monotonic", artifact_type: null, error_class: null, error_code: null },
  { run_id: mockData.attempts[0].run_id, attempt_id: attemptOne, sequence: 125, event_type: "span_completed", stage: "runner_mask", span: "inference", status: "completed", event_time: "2026-07-09T21:14:15Z", elapsed_seconds: 71.6, start_offset_seconds: 51.2, end_offset_seconds: 122.8, timing_basis: "monotonic", artifact_type: null, error_class: null, error_code: null },
  { run_id: mockData.attempts[0].run_id, attempt_id: attemptOne, sequence: 127, event_type: "span_completed", stage: "runner_mask", span: "render", status: "completed", event_time: "2026-07-09T21:14:34Z", elapsed_seconds: 18.7, start_offset_seconds: 122.8, end_offset_seconds: 141.5, timing_basis: "monotonic", artifact_type: null, error_class: null, error_code: null },
  { run_id: mockData.attempts[0].run_id, attempt_id: attemptOne, sequence: 129, event_type: "span_completed", stage: "runner_mask", span: "encode", status: "completed", event_time: "2026-07-09T21:14:45Z", elapsed_seconds: 11.5, start_offset_seconds: 141.5, end_offset_seconds: 153.0, timing_basis: "monotonic", artifact_type: null, error_class: null, error_code: null },
  { run_id: mockData.attempts[0].run_id, attempt_id: attemptOne, sequence: 141, event_type: "stage_completed", stage: "pose_sequence", span: null, status: "completed", event_time: "2026-07-09T21:14:59Z", elapsed_seconds: 13.1, start_offset_seconds: 156.1, end_offset_seconds: 169.2, timing_basis: "monotonic", artifact_type: null, error_class: null, error_code: null },
  { run_id: mockData.attempts[0].run_id, attempt_id: attemptOne, sequence: 151, event_type: "stage_completed", stage: "densepose_body_map", span: null, status: "completed", event_time: "2026-07-09T21:15:10Z", elapsed_seconds: 10.4, start_offset_seconds: 169.2, end_offset_seconds: 179.6, timing_basis: "monotonic", artifact_type: null, error_class: null, error_code: null },
  { run_id: mockData.attempts[0].run_id, attempt_id: attemptOne, sequence: 201, event_type: "stage_completed", stage: "artifact_publish", span: null, status: "completed", event_time: "2026-07-09T21:15:18Z", elapsed_seconds: 8.0, start_offset_seconds: 179.6, end_offset_seconds: 187.6, timing_basis: "monotonic", artifact_type: null, error_class: null, error_code: null },
];
