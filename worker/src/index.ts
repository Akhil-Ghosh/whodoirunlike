import {
  deliverAnalyticsOutboxItem,
  parseTelemetryRequest,
  persistTelemetryEvent,
  readTelemetryTimeline,
  reconcileAnalyticsOutbox,
  retryAnalyticsOutbox,
  TelemetryConflictError,
  TelemetryPayloadTooLargeError,
} from "./telemetry";
import type { ProcessingTelemetryEvent } from "./telemetry";

type JobStatus = "uploaded" | "queued" | "running" | "complete" | "failed";

type RunnerPromptPoint = {
  x: number;
  y: number;
  label?: string;
};

type RunnerPromptBox = {
  x: number;
  y: number;
  width: number;
  height: number;
};

type RunnerTargetPrompt = {
  version: 1;
  source: "hosted_upload_user_prompt_v1";
  selection: {
    type: "box" | "point";
    positive_points: RunnerPromptPoint[];
    negative_points: RunnerPromptPoint[];
    box?: RunnerPromptBox;
  };
  frame: {
    time_seconds?: number;
    frame_index?: number;
    width?: number;
    height?: number;
  };
  notes: string;
};

type ArtifactRecord = {
  key: string;
  attempt_id?: string;
  content_type: string;
  object_version?: string;
  size_bytes: number;
  updated_at: string;
};

type ArtifactFinalizeItem = {
  name: string;
  content_type: string;
  object_version: string;
  size_bytes: number;
};

type ProcessingAttemptRecord = {
  attempt_id: string;
  attempt_number: number;
  created_at: string;
  processor_enqueued_at?: string;
  processor_queue_started_at?: string;
  processor_started_at?: string;
  runpod_endpoint_id?: string;
  runpod_job_id?: string;
};

type JobRecord = {
  version: 1;
  run_id: string;
  status: JobStatus;
  created_at: string;
  updated_at: string;
  upload_completed_at?: string;
  upload: {
    key: string;
    filename: string | null;
    content_type: string;
    size_bytes: number;
    consent_scope: string | null;
  };
  target_prompt?: RunnerTargetPrompt | null;
  attempt_id?: string;
  attempt_number?: number;
  processor_enqueued_at?: string;
  processor_queue_started_at?: string;
  processor_started_at?: string;
  runpod_endpoint_id?: string;
  runpod_job_id?: string;
  processing_attempts?: ProcessingAttemptRecord[];
  progress?: unknown;
  summary?: unknown;
  error?: string;
  error_code?: string;
  artifacts: Record<string, ArtifactRecord>;
};

type ProcessorPayload = {
  run_id: string;
  attempt_id: string;
  attempt_number: number;
  attempt_started_at: string;
  processor_enqueued_at: string;
  telemetry_sequence_start: number;
  source: {
    url: string;
    key: string;
    filename: string | null;
    content_type: string;
    size_bytes: number;
  };
  target_prompt?: RunnerTargetPrompt | null;
  callback_base_url: string;
};

type EnqueuedJobRecord = JobRecord & {
  attempt_id: string;
  attempt_number: number;
  processor_enqueued_at: string;
};

class JobMutationConflictError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "JobMutationConflictError";
  }
}

type RunPodStartResponse = {
  id?: string;
  status?: string;
  error?: string;
};

const DEFAULT_MAX_UPLOAD_BYTES = 75 * 1024 * 1024;
const MAX_ARTIFACT_FINALIZE_BODY_BYTES = 64 * 1024;
const MAX_ARTIFACT_FINALIZE_COUNT = 64;
const MAX_R2_OBJECT_BYTES = 5 * 1024 * 1024 * 1024;
const JSON_HEADERS = {
  "Content-Type": "application/json; charset=utf-8",
};
const RESULT_READY_ARTIFACT_NAME = "fused_overlay.mp4";

export default {
  async fetch(request, env, ctx): Promise<Response> {
    return handleRequest(request, env, ctx);
  },
  async scheduled(controller, env): Promise<void> {
    await reconcileAnalyticsOutbox(env, controller.scheduledTime);
    await retryAnalyticsOutbox(env, controller.scheduledTime);
  },
} satisfies ExportedHandler<Env>;

async function handleRequest(
  request: Request,
  env: Env,
  ctx: ExecutionContext,
): Promise<Response> {
  if (request.method === "OPTIONS") {
    return new Response(null, { status: 204, headers: corsHeaders(request, env) });
  }

  const url = new URL(request.url);
  const segments = url.pathname.split("/").filter(Boolean);

  try {
    if (request.method === "GET" && (url.pathname === "/health" || url.pathname === "/v1/health")) {
      return jsonResponse(request, env, {
        status: "ok",
        service: "whodoirunlike-worker",
        environment: env.ENVIRONMENT,
      });
    }

    if (segments[0] !== "v1") {
      return notFound(request, env);
    }

    if (request.method === "POST" && segments.length === 2 && segments[1] === "uploads") {
      return handleUpload(request, env, ctx);
    }

    if (segments[1] === "jobs" && segments[2]) {
      const runId = normalizeRunId(segments[2]);
      if (!runId) {
        return errorResponse(request, env, 400, "Invalid run_id.");
      }

      if (request.method === "GET" && segments.length === 3) {
        return handleGetJob(request, env, runId);
      }

      if (request.method === "POST" && segments.length === 4 && segments[3] === "start") {
        return handleStartJob(request, env, ctx, runId);
      }

      if (request.method === "POST" && segments.length === 4 && segments[3] === "events") {
        return handlePostTelemetryEvent(request, env, ctx, runId);
      }

      if (request.method === "GET" && segments.length === 4 && segments[3] === "events") {
        return handleGetTelemetryEvents(request, env, runId);
      }

      if (request.method === "GET" && segments.length === 4 && segments[3] === "source") {
        return handleGetSource(request, env, runId);
      }

      if (
        request.method === "PUT" &&
        segments.length === 5 &&
        segments[3] === "artifacts"
      ) {
        const artifactName = normalizeArtifactName(segments[4]);
        if (!artifactName) {
          return errorResponse(request, env, 400, "Invalid artifact name.");
        }
        return handlePutArtifact(request, env, runId, artifactName);
      }

      if (
        request.method === "POST" &&
        segments.length === 5 &&
        segments[3] === "artifacts" &&
        segments[4] === "finalize"
      ) {
        return handleFinalizeArtifacts(request, env, runId);
      }

      if (request.method === "POST" && segments.length === 4 && segments[3] === "report") {
        return handleReport(request, env, ctx, runId);
      }
    }

    if (request.method === "GET" && segments[1] === "artifacts" && segments[2] && segments[3]) {
      const runId = normalizeRunId(segments[2]);
      const artifactName = normalizeArtifactName(segments[3]);
      if (!runId || !artifactName) {
        return errorResponse(request, env, 400, "Invalid artifact URL.");
      }
      return handleGetArtifact(request, env, runId, artifactName);
    }

    return notFound(request, env);
  } catch (error) {
    console.error(
      JSON.stringify({
        level: "error",
        message: error instanceof Error ? error.message : "Unhandled Worker error",
        path: url.pathname,
      }),
    );
    return errorResponse(request, env, 500, "Unexpected API error.");
  }
}

async function handleUpload(
  request: Request,
  env: Env,
  ctx: ExecutionContext,
): Promise<Response> {
  const maxBytes = maxUploadBytes(env);
  const declaredLength = parseContentLength(request);
  if (declaredLength !== null && declaredLength > maxBytes) {
    return errorResponse(request, env, 413, `Clip is too large. Max upload size is ${Math.floor(maxBytes / 1024 / 1024)} MB.`);
  }
  if (!request.body) {
    return errorResponse(request, env, 400, "Upload a video file as the request body.");
  }

  const contentType = request.headers.get("content-type") ?? "application/octet-stream";
  const originalFilename = request.headers.get("x-original-filename");
  const extension = extensionForUpload(contentType, originalFilename);
  if (!extension) {
    return errorResponse(request, env, 415, "Upload an MP4, MOV, M4V, or WebM running clip.");
  }
  const promptResult = parseRunnerPromptHeader(request.headers.get("x-runner-prompt"));
  if (!promptResult.ok) {
    return errorResponse(request, env, 400, promptResult.error);
  }

  const runId = crypto.randomUUID();
  const createdAt = new Date().toISOString();
  const initialAttempt = createProcessingAttempt(1, createdAt);
  const uploadKey = `uploads/${runId}/source${extension}`;
  const uploaded = await env.CLIPS.put(uploadKey, request.body, {
    httpMetadata: { contentType },
    customMetadata: {
      run_id: runId,
      original_filename: originalFilename ?? "",
      consent_scope: request.headers.get("x-clip-consent") ?? "",
    },
  });

  if (uploaded.size > maxBytes) {
    await env.CLIPS.delete(uploadKey);
    return errorResponse(request, env, 413, `Clip is too large. Max upload size is ${Math.floor(maxBytes / 1024 / 1024)} MB.`);
  }

  const uploadCompletedAt = new Date().toISOString();

  const job: JobRecord = {
    version: 1,
    run_id: runId,
    status: "uploaded",
    created_at: createdAt,
    updated_at: uploadCompletedAt,
    upload_completed_at: uploadCompletedAt,
    upload: {
      key: uploadKey,
      filename: originalFilename,
      content_type: contentType,
      size_bytes: uploaded.size,
      consent_scope: request.headers.get("x-clip-consent"),
    },
    target_prompt: promptResult.prompt,
    attempt_id: initialAttempt.attempt_id,
    attempt_number: initialAttempt.attempt_number,
    processing_attempts: [initialAttempt],
    artifacts: {},
  };
  await writeJob(env, job);
  await recordWorkerLifecycleEvent(
    env,
    ctx,
    workerLifecycleEvent(env, job, {
      sequence: 1,
      event_type: "attempt_started",
      event_time: createdAt,
      status: "running",
      elapsed_seconds: 0,
    }),
  );
  await recordWorkerLifecycleEvent(
    env,
    ctx,
    workerLifecycleEvent(env, job, {
      sequence: 2,
      event_type: "stage_started",
      event_time: createdAt,
      stage: "source_ingest",
      status: "running",
      elapsed_seconds: 0,
    }),
  );
  await recordWorkerLifecycleEvent(
    env,
    ctx,
    workerLifecycleEvent(env, job, {
      sequence: 3,
      event_type: "stage_completed",
      event_time: uploadCompletedAt,
      stage: "source_ingest",
      status: "complete",
      elapsed_seconds: elapsedSeconds(createdAt, uploadCompletedAt),
      input: {
        size_bytes: uploaded.size,
        content_type: contentType.slice(0, 100),
      },
    }),
  );

  let responseJob = job;
  if (new URL(request.url).searchParams.get("start") === "1" && processorConfigured(env)) {
    const queued = enqueueJob(job);
    await writeJob(env, queued);
    ctx.waitUntil(notifyProcessorSafely(request, env, ctx, queued));
    responseJob = queued;
  }

  const response = publicJob(responseJob, publicBaseUrl(request, env));
  if (responseJob.status === "queued") {
    response.message = "Uploaded. Processor notification started.";
  }
  return jsonResponse(request, env, response, 201);
}

async function handleGetJob(request: Request, env: Env, runId: string): Promise<Response> {
  const snapshot = await readJobSnapshot(env, runId);
  if (!snapshot) {
    return errorResponse(request, env, 404, "Job not found.");
  }
  const job = snapshot.job;
  return jsonResponse(request, env, publicJob(job, publicBaseUrl(request, env)));
}

async function handleStartJob(
  request: Request,
  env: Env,
  ctx: ExecutionContext,
  runId: string,
): Promise<Response> {
  const snapshot = await readJobSnapshot(env, runId);
  if (!snapshot) {
    return errorResponse(request, env, 404, "Job not found.");
  }
  const job = snapshot.job;

  if (job.status === "complete" || job.status === "running" || job.status === "queued") {
    return jsonResponse(request, env, publicJob(job, publicBaseUrl(request, env)), 202);
  }

  if (!processorConfigured(env)) {
    return jsonResponse(
      request,
      env,
      {
        ...publicJob(job, publicBaseUrl(request, env)),
        processor_configured: false,
        message: processorConfigurationMessage(env),
      },
      202,
    );
  }

  if (!env.PROCESSOR_SHARED_SECRET) {
    return errorResponse(request, env, 503, "Processor secret is not configured.");
  }

  const queued = enqueueJob(job);
  if (!(await writeJobIfUnchanged(env, queued, snapshot.etag))) {
    return errorResponse(request, env, 409, "Job changed while the attempt was being enqueued.");
  }
  if (queued.attempt_id !== job.attempt_id) {
    const attemptStartedAt = currentProcessingAttempt(queued)?.created_at ?? queued.processor_enqueued_at;
    await recordWorkerLifecycleEvent(
      env,
      ctx,
      workerLifecycleEvent(env, queued, {
        sequence: 1,
        event_type: "attempt_started",
        event_time: attemptStartedAt,
        status: "running",
        elapsed_seconds: 0,
      }),
    );
  }
  await notifyProcessor(request, env, ctx, queued);
  return jsonResponse(request, env, publicJob(queued, publicBaseUrl(request, env)), 202);
}

async function handlePostTelemetryEvent(
  request: Request,
  env: Env,
  ctx: ExecutionContext,
  runId: string,
): Promise<Response> {
  if (!(await hasProcessorAuth(request, env))) {
    return errorResponse(request, env, 401, "Processor authorization required.");
  }
  const job = await readJob(env, runId);
  if (!job) {
    return errorResponse(request, env, 404, "Job not found.");
  }

  const parsed = await parseTelemetryRequest(request, runId);
  if (!parsed.ok) {
    return errorResponse(request, env, parsed.status, parsed.error);
  }
  if (!jobHasAttempt(job, parsed.value.attempt_id)) {
    return errorResponse(request, env, 409, "Processing attempt does not belong to this job.");
  }
  const requestedTerminalStatus = parsed.value.event_type === "attempt_completed"
    ? "complete"
    : parsed.value.event_type === "attempt_failed"
      ? "failed"
      : null;
  if (
    job.attempt_id === parsed.value.attempt_id &&
    requestedTerminalStatus &&
    (job.status === "complete" || job.status === "failed") &&
    job.status !== requestedTerminalStatus
  ) {
    return errorResponse(request, env, 409, "Attempt already has the opposite terminal state.");
  }

  try {
    const persisted = await persistTelemetryEvent(env, parsed.value);
    if (persisted.outbox_key && persisted.outbox_body) {
      ctx.waitUntil(
        deliverAnalyticsOutboxItem(env, persisted.outbox_key, persisted.outbox_body),
      );
    }
    let currentJob = await readJob(env, runId);
    if (currentJob?.attempt_id === parsed.value.attempt_id) {
      if (!currentJob.runpod_job_id && !currentJob.processor_started_at) {
        currentJob = await mutateCurrentJob(
          env,
          runId,
          parsed.value.attempt_id,
          (current) => markProcessorStarted(current, parsed.value.event_time),
        );
        ctx.waitUntil(
          recordDirectQueueCompletion(env, ctx, currentJob, parsed.value.event_time),
        );
      }
      if (
        parsed.value.event_type === "attempt_completed" ||
        parsed.value.event_type === "attempt_failed"
      ) {
        const terminalStatus =
          parsed.value.event_type === "attempt_completed" ? "complete" : "failed";
        currentJob = await transitionJobStatus(env, currentJob, terminalStatus);
        currentJob = await mutateCurrentJob(
          env,
          runId,
          parsed.value.attempt_id,
          (current) => ({
            ...updateJob(current, current.status),
            error_code: terminalStatus === "failed"
              ? "PROCESSING_ATTEMPT_FAILED"
              : undefined,
          }),
        );
        if (currentJob.runpod_job_id) {
          ctx.waitUntil(recordProviderQueueCompletion(env, ctx, currentJob));
        }
      }
    }
    return jsonResponse(
      request,
      env,
      {
        schema_version: 1,
        run_id: runId,
        attempt_id: parsed.value.attempt_id,
        event_id: parsed.value.event_id,
        status: persisted.duplicate ? "already_stored" : "stored",
        aws_export_queued: persisted.outbox_key !== null,
      },
      persisted.duplicate ? 200 : 202,
    );
  } catch (error) {
    if (error instanceof TelemetryConflictError) {
      return errorResponse(request, env, 409, error.message);
    }
    if (error instanceof TelemetryPayloadTooLargeError) {
      return errorResponse(request, env, 413, error.message);
    }
    if (error instanceof JobMutationConflictError) {
      return errorResponse(request, env, 409, error.message);
    }
    throw error;
  }
}

async function handleGetTelemetryEvents(
  request: Request,
  env: Env,
  runId: string,
): Promise<Response> {
  if (!(await hasProcessorAuth(request, env))) {
    return errorResponse(request, env, 401, "Processor authorization required.");
  }
  const job = await readJob(env, runId);
  if (!job) {
    return errorResponse(request, env, 404, "Job not found.");
  }

  const url = new URL(request.url);
  const unknownParameter = [...url.searchParams.keys()].find(
    (key) => key !== "attempt_id" && key !== "limit" && key !== "cursor",
  );
  if (unknownParameter) {
    return errorResponse(request, env, 400, `Unknown timeline parameter: ${unknownParameter}.`);
  }
  const attemptId = normalizeAttemptId(url.searchParams.get("attempt_id") ?? job.attempt_id ?? "");
  if (!attemptId || !jobHasAttempt(job, attemptId)) {
    return errorResponse(request, env, 400, "A valid attempt_id is required.");
  }
  const rawLimit = url.searchParams.get("limit");
  const limit = rawLimit === null ? 200 : Number.parseInt(rawLimit, 10);
  if (!Number.isInteger(limit) || limit < 1 || limit > 200) {
    return errorResponse(request, env, 400, "Timeline limit must be between 1 and 200.");
  }
  const cursor = url.searchParams.get("cursor") ?? undefined;
  if (cursor && cursor.length > 1024) {
    return errorResponse(request, env, 400, "Timeline cursor is too large.");
  }

  const timeline = await readTelemetryTimeline(env, runId, attemptId, limit, cursor);
  return jsonResponse(request, env, timeline);
}

async function handleGetSource(request: Request, env: Env, runId: string): Promise<Response> {
  if (!(await hasProcessorAuth(request, env))) {
    return errorResponse(request, env, 401, "Processor authorization required.");
  }

  const job = await readJob(env, runId);
  if (!job) {
    return errorResponse(request, env, 404, "Job not found.");
  }

  const object = await env.CLIPS.get(job.upload.key);
  if (!object) {
    return errorResponse(request, env, 404, "Source upload not found.");
  }

  return objectResponse(request, env, object, {
    "Content-Type": job.upload.content_type,
    "Content-Disposition": contentDisposition(job.upload.filename ?? `source-${runId}`),
  });
}

async function handlePutArtifact(
  request: Request,
  env: Env,
  runId: string,
  artifactName: string,
): Promise<Response> {
  if (!(await hasProcessorAuth(request, env))) {
    return errorResponse(request, env, 401, "Processor authorization required.");
  }
  if (!request.body) {
    return errorResponse(request, env, 400, "Artifact body is required.");
  }

  const snapshot = await readJobSnapshot(env, runId);
  if (!snapshot) {
    return errorResponse(request, env, 404, "Job not found.");
  }
  const job = snapshot.job;
  const attemptId = normalizeAttemptId(request.headers.get("x-processing-attempt-id") ?? "");
  if (!attemptId) {
    return errorResponse(request, env, 400, "X-Processing-Attempt-Id is required.");
  }
  if (job.attempt_id !== attemptId) {
    return errorResponse(request, env, 409, "Processing attempt is stale.");
  }

  const deferIndex = new URL(request.url).searchParams.get("defer_index") === "1";
  if (deferIndex && artifactName === RESULT_READY_ARTIFACT_NAME) {
    return errorResponse(request, env, 400, "The result-ready artifact must be indexed immediately.");
  }
  const requestedContentType = request.headers.get("content-type") ?? "application/octet-stream";
  const contentType = deferIndex
    ? normalizeArtifactContentType(requestedContentType)
    : requestedContentType;
  if (!contentType) {
    return errorResponse(request, env, 400, "Invalid artifact content type.");
  }
  const artifactKey = `artifacts/${runId}/${attemptId}/${artifactName}`;
  const stored = await env.CLIPS.put(artifactKey, request.body, {
    httpMetadata: { contentType },
    customMetadata: { run_id: runId, attempt_id: attemptId, artifact_name: artifactName },
  });

  if (deferIndex) {
    if (stored.size > MAX_R2_OBJECT_BYTES) {
      logAbandonedDeferredArtifact(runId, attemptId, artifactName, "invalid_size");
      return errorResponse(request, env, 400, "Artifact size is invalid.");
    }
    const latestJob = await readJob(env, runId);
    if (!latestJob || latestJob.attempt_id !== attemptId) {
      logAbandonedDeferredArtifact(runId, attemptId, artifactName, "stale_attempt");
      return errorResponse(request, env, 409, "Processing attempt became stale.");
    }
    return jsonResponse(request, env, {
      run_id: runId,
      attempt_id: attemptId,
      artifact: artifactName,
      status: "stored_unindexed",
      content_type: contentType,
      object_version: stored.version,
      size_bytes: stored.size,
    });
  }

  const updated = updateJob(job, job.status);
  updated.artifacts[artifactName] = {
    key: artifactKey,
    attempt_id: attemptId,
    content_type: contentType,
    object_version: stored.version,
    size_bytes: stored.size,
    updated_at: updated.updated_at,
  };
  if (!(await writeJobIfUnchanged(env, updated, snapshot.etag))) {
    return errorResponse(request, env, 409, "Job changed during artifact upload.");
  }

  return jsonResponse(request, env, {
    run_id: runId,
    attempt_id: attemptId,
    artifact: artifactName,
    status: "stored",
    size_bytes: stored.size,
  });
}

async function handleFinalizeArtifacts(
  request: Request,
  env: Env,
  runId: string,
): Promise<Response> {
  if (!(await hasProcessorAuth(request, env))) {
    return errorResponse(request, env, 401, "Processor authorization required.");
  }
  const declaredLength = parseContentLength(request);
  if (declaredLength !== null && declaredLength > MAX_ARTIFACT_FINALIZE_BODY_BYTES) {
    return errorResponse(request, env, 413, "Artifact finalization body is too large.");
  }

  const boundedBody = await readBoundedRequestText(
    request,
    MAX_ARTIFACT_FINALIZE_BODY_BYTES,
  );
  if (!boundedBody.ok) {
    return errorResponse(request, env, 413, "Artifact finalization body is too large.");
  }
  const rawBody = boundedBody.value;
  let rawPayload: unknown;
  try {
    rawPayload = JSON.parse(rawBody);
  } catch {
    return errorResponse(request, env, 400, "Artifact finalization must be valid JSON.");
  }
  const parsed = parseArtifactFinalizePayload(rawPayload);
  if (!parsed.ok) {
    return errorResponse(request, env, 400, parsed.error);
  }
  const headerAttemptId = normalizeAttemptId(
    request.headers.get("x-processing-attempt-id") ?? "",
  );
  if (!headerAttemptId) {
    return errorResponse(request, env, 400, "X-Processing-Attempt-Id is required.");
  }
  if (headerAttemptId !== parsed.attemptId) {
    return errorResponse(request, env, 400, "Processing attempt identifiers do not match.");
  }

  const job = await readJob(env, runId);
  if (!job) {
    return errorResponse(request, env, 404, "Job not found.");
  }
  if (job.attempt_id !== parsed.attemptId) {
    return errorResponse(request, env, 409, "Processing attempt is stale.");
  }

  const records = await Promise.all(
    parsed.artifacts.map(async (artifact): Promise<[string, ArtifactRecord] | null> => {
      const key = `artifacts/${runId}/${parsed.attemptId}/${artifact.name}`;
      const stored = await env.CLIPS.head(key);
      if (
        !stored ||
        stored.size !== artifact.size_bytes ||
        stored.version !== artifact.object_version ||
        stored.httpMetadata?.contentType !== artifact.content_type ||
        stored.customMetadata?.run_id !== runId ||
        stored.customMetadata?.attempt_id !== parsed.attemptId ||
        stored.customMetadata?.artifact_name !== artifact.name
      ) {
        return null;
      }
      return [
        artifact.name,
        {
          key,
          attempt_id: parsed.attemptId,
          content_type: artifact.content_type,
          object_version: artifact.object_version,
          size_bytes: artifact.size_bytes,
          updated_at: stored.uploaded.toISOString(),
        },
      ];
    }),
  );
  if (records.some((record) => record === null)) {
    return errorResponse(request, env, 409, "A deferred artifact is missing or changed.");
  }

  try {
    await mutateCurrentJob(env, runId, parsed.attemptId, (current) => {
      const updated = updateJob(current, current.status);
      for (const record of records) {
        if (record) updated.artifacts[record[0]] = record[1];
      }
      return updated;
    });
  } catch (error) {
    if (error instanceof JobMutationConflictError) {
      return errorResponse(request, env, 409, error.message);
    }
    throw error;
  }

  return jsonResponse(request, env, {
    run_id: runId,
    attempt_id: parsed.attemptId,
    status: "indexed",
    artifacts: parsed.artifacts.map((artifact) => artifact.name),
  });
}

async function handleReport(
  request: Request,
  env: Env,
  ctx: ExecutionContext,
  runId: string,
): Promise<Response> {
  if (!(await hasProcessorAuth(request, env))) {
    return errorResponse(request, env, 401, "Processor authorization required.");
  }
  const declaredLength = parseContentLength(request);
  if (declaredLength !== null && declaredLength > 64 * 1024) {
    return errorResponse(request, env, 413, "Report body is too large.");
  }

  const job = await readJob(env, runId);
  if (!job) {
    return errorResponse(request, env, 404, "Job not found.");
  }

  const payload = (await request.json().catch(() => null)) as {
    attempt_id?: string;
    status?: JobStatus;
    progress?: unknown;
    summary?: unknown;
    error?: string;
  } | null;
  if (!payload) {
    return errorResponse(request, env, 400, "Report must be valid JSON.");
  }

  const attemptId = normalizeAttemptId(payload.attempt_id ?? "");
  if (!attemptId) {
    return errorResponse(request, env, 400, "Report attempt_id is required.");
  }
  if (job.attempt_id !== attemptId) {
    return errorResponse(request, env, 409, "Processing attempt is stale.");
  }

  const latestJob = await readJob(env, runId);
  if (!latestJob || latestJob.attempt_id !== attemptId) {
    return errorResponse(request, env, 409, "Processing attempt became stale.");
  }
  const status = normalizeReportStatus(payload.status);
  if (payload.status !== undefined && !status) {
    return errorResponse(request, env, 400, "Invalid report status.");
  }
  const startedAtCandidate = status === "running" && !latestJob.processor_started_at
    ? new Date().toISOString()
    : null;
  let updated: JobRecord;
  try {
    if (status) await transitionJobStatus(env, latestJob, status);
    updated = await mutateCurrentJob(env, runId, attemptId, (current) => {
      const next = updateJob(current, current.status);
      if (startedAtCandidate && !next.processor_started_at) {
        next.processor_started_at = startedAtCandidate;
        next.processing_attempts = (next.processing_attempts ?? []).map((attempt) =>
          attempt.attempt_id === attemptId
            ? { ...attempt, processor_started_at: startedAtCandidate }
            : attempt,
        );
      }
      if ("progress" in payload) next.progress = sanitizePublicProgress(payload.progress);
      if (status === "failed") next.error_code = "PROCESSING_ATTEMPT_FAILED";
      if (status === "complete") next.error_code = undefined;
      return next;
    });
  } catch (error) {
    if (error instanceof JobMutationConflictError) {
      return errorResponse(request, env, 409, error.message);
    }
    throw error;
  }

  if (startedAtCandidate && updated.processor_started_at === startedAtCandidate && !updated.runpod_job_id) {
    ctx.waitUntil(recordDirectQueueCompletion(env, ctx, updated, startedAtCandidate));
  }
  if (status === "complete" || status === "failed") {
    ctx.waitUntil(recordProviderQueueCompletion(env, ctx, updated));
  }

  return jsonResponse(request, env, publicJob(updated, publicBaseUrl(request, env)));
}

async function handleGetArtifact(
  request: Request,
  env: Env,
  runId: string,
  artifactName: string,
): Promise<Response> {
  const job = await readJob(env, runId);
  if (!job) {
    return errorResponse(request, env, 404, "Job not found.");
  }
  const artifact = job.artifacts[artifactName];
  if (!artifact || !artifactBelongsToCurrentAttempt(job, artifact)) {
    return errorResponse(request, env, 404, "Artifact not found.");
  }
  const object = await env.CLIPS.get(artifact.key);
  if (!object) {
    return errorResponse(request, env, 404, "Artifact object not found.");
  }
  if (artifact.object_version && object.version !== artifact.object_version) {
    return errorResponse(request, env, 404, "Artifact object version changed.");
  }
  return objectResponse(request, env, object, {
    "Content-Type": artifact.content_type,
    "Content-Disposition": contentDisposition(artifactName),
    "Cache-Control": "private, max-age=300",
  });
}

async function notifyProcessorSafely(
  request: Request,
  env: Env,
  ctx: ExecutionContext,
  job: EnqueuedJobRecord,
): Promise<void> {
  try {
    await notifyProcessor(request, env, ctx, job);
  } catch (error) {
    console.error(
      JSON.stringify({
        level: "error",
        message: "processor_enqueue_failed",
        run_id: job.run_id,
        attempt_id: job.attempt_id,
        error: error instanceof Error ? error.message : "Unknown processor enqueue error",
      }),
    );
  }
}

async function notifyProcessor(
  request: Request,
  env: Env,
  ctx: ExecutionContext,
  job: EnqueuedJobRecord,
): Promise<void> {
  if (!env.PROCESSOR_SHARED_SECRET) return;
  const base = publicBaseUrl(request, env);
  const payload: ProcessorPayload = {
    run_id: job.run_id,
    attempt_id: job.attempt_id,
    attempt_number: job.attempt_number,
    attempt_started_at: currentProcessingAttempt(job)?.created_at ?? job.processor_enqueued_at,
    processor_enqueued_at: job.processor_enqueued_at,
    telemetry_sequence_start: 100,
    source: {
      url: `${base}/v1/jobs/${job.run_id}/source`,
      key: job.upload.key,
      filename: job.upload.filename,
      content_type: job.upload.content_type,
      size_bytes: job.upload.size_bytes,
    },
    target_prompt: job.target_prompt ?? null,
    callback_base_url: base,
  };

  const enqueueStartedAt = new Date().toISOString();
  await recordWorkerLifecycleEvent(
    env,
    ctx,
    workerLifecycleEvent(env, job, {
      sequence: 4,
      event_type: "stage_started",
      event_time: enqueueStartedAt,
      stage: "processor_enqueue",
      status: "running",
      elapsed_seconds: 0,
    }),
  );

  try {
    if (asyncRunPodConfigured(env)) {
      await notifyRunPod(env, job, payload);
    } else if (env.PROCESSOR_URL) {
      const response = await fetch(`${trimTrailingSlash(env.PROCESSOR_URL)}/v1/processor/jobs`, {
        method: "POST",
        headers: {
          ...JSON_HEADERS,
          Authorization: `Bearer ${env.PROCESSOR_SHARED_SECRET}`,
        },
        body: JSON.stringify(payload),
      });
      if (response.body) await response.body.cancel();
      if (!response.ok) {
        const failed = updateJob(job, "failed");
        failed.error = `Processor rejected job with HTTP ${response.status}`;
        await writeJob(env, failed);
        throw new Error(failed.error);
      }
      console.log(
        JSON.stringify({
          level: "info",
          message: "processor_enqueue_accepted",
          run_id: job.run_id,
          attempt_id: job.attempt_id,
          processor: "configured_url",
        }),
      );
    }

    const completedAt = new Date().toISOString();
    const acceptedJob = await markProcessorQueueStarted(env, job, completedAt);
    await recordWorkerLifecycleEvent(
      env,
      ctx,
      workerLifecycleEvent(env, acceptedJob, {
        sequence: 5,
        event_type: "stage_completed",
        event_time: completedAt,
        stage: "processor_enqueue",
        status: "complete",
        elapsed_seconds: elapsedSeconds(enqueueStartedAt, completedAt),
      }),
    );
    await recordWorkerLifecycleEvent(
      env,
      ctx,
      workerLifecycleEvent(env, acceptedJob, {
        sequence: 6,
        event_type: "stage_started",
        event_time: completedAt,
        stage: "processor_queue",
        status: "running",
        elapsed_seconds: 0,
      }),
    );
  } catch (error) {
    const failedAt = new Date().toISOString();
    const failedJob = updateJob(job, "failed");
    failedJob.error = "Processor enqueue failed.";
    await writeJob(env, failedJob);
    await recordWorkerLifecycleEvent(
      env,
      ctx,
      workerLifecycleEvent(env, job, {
        sequence: 5,
        event_type: "stage_failed",
        event_time: failedAt,
        stage: "processor_enqueue",
        status: "failed",
        elapsed_seconds: elapsedSeconds(enqueueStartedAt, failedAt),
        error: {
          class: "ProcessorEnqueueError",
          code: "upstream_enqueue.rejected",
          category: "upstream_enqueue",
          message: "Processor enqueue failed.",
          retryable: true,
        },
      }),
    );
    const attemptStartedAt = currentProcessingAttempt(job)?.created_at ?? job.processor_enqueued_at;
    await recordWorkerLifecycleEvent(
      env,
      ctx,
      workerLifecycleEvent(env, job, {
        sequence: 6,
        event_type: "attempt_failed",
        event_time: failedAt,
        status: "failed",
        elapsed_seconds: elapsedSeconds(attemptStartedAt, failedAt),
        error: {
          class: "ProcessorEnqueueError",
          code: "upstream_enqueue.rejected",
          category: "upstream_enqueue",
          message: "Processor enqueue failed.",
          retryable: true,
        },
      }),
    );
    throw error;
  }
}

async function notifyRunPod(
  env: Env,
  job: EnqueuedJobRecord,
  payload: ProcessorPayload,
): Promise<void> {
  const endpointId = runPodEndpointId(env);
  if (env.RUNPOD_RUNSYNC === "1") {
    throw new Error("RUNPOD_RUNSYNC_UNSUPPORTED");
  }
  const response = await fetch(`https://api.runpod.ai/v2/${endpointId}/run`, {
    method: "POST",
    headers: {
      ...JSON_HEADERS,
      Authorization: `Bearer ${env.RUNPOD_API_KEY}`,
    },
    body: JSON.stringify({ input: payload }),
  });
  const body = (await response.json().catch(() => null)) as RunPodStartResponse | null;
  if (!response.ok || body?.error) {
    const failed = updateJob(job, "failed");
    failed.error = body?.error || `RunPod rejected job with HTTP ${response.status}`;
    await writeJob(env, failed);
    throw new Error(failed.error);
  }

  const runpodJobId = typeof body?.id === "string" ? body.id.slice(0, 200) : undefined;
  await mutateCurrentJob(env, job.run_id, job.attempt_id, (current) => {
    const updated = updateJob(current, current.status);
    updated.runpod_endpoint_id = endpointId;
    if (runpodJobId) {
      updated.runpod_job_id = runpodJobId;
    }
    updated.processing_attempts = (updated.processing_attempts ?? []).map((attempt) =>
      attempt.attempt_id === job.attempt_id
        ? {
            ...attempt,
            runpod_endpoint_id: endpointId,
            ...(runpodJobId ? { runpod_job_id: runpodJobId } : {}),
          }
        : attempt,
    );
    updated.progress = {
      phase: "queued_on_runpod",
      runpod_job_id: runpodJobId ?? null,
      runpod_status: body?.status ?? null,
    };
    return updated;
  });
  console.log(
    JSON.stringify({
      level: "info",
      message: "processor_enqueue_accepted",
      run_id: job.run_id,
      attempt_id: job.attempt_id,
      processor: "runpod",
      runpod_job_id: runpodJobId ?? null,
    }),
  );
}

function processorConfigured(env: Env): boolean {
  const hasSharedSecret = Boolean(env.PROCESSOR_SHARED_SECRET);
  const hasRunPod = asyncRunPodConfigured(env);
  const hasUrlProcessor = Boolean(env.PROCESSOR_URL);
  if (
    env.RUNPOD_RUNSYNC === "1" &&
    runPodEndpointId(env) &&
    env.RUNPOD_API_KEY &&
    !hasUrlProcessor
  ) {
    console.error(
      JSON.stringify({
        level: "error",
        message: "runpod_runsync_unsupported",
        required_mode: "run",
      }),
    );
  }
  return hasSharedSecret && (hasRunPod || hasUrlProcessor);
}

function processorConfigurationMessage(env: Env): string {
  if (!env.PROCESSOR_SHARED_SECRET) {
    return "Upload stored. Configure PROCESSOR_SHARED_SECRET before jobs can run.";
  }
  if (
    env.RUNPOD_RUNSYNC === "1" &&
    runPodEndpointId(env) &&
    env.RUNPOD_API_KEY &&
    !env.PROCESSOR_URL
  ) {
    return "Upload stored. RUNPOD_RUNSYNC=1 is unsupported; configure asynchronous RunPod /run mode.";
  }
  if (!runPodEndpointId(env) || !env.RUNPOD_API_KEY) {
    return "Upload stored. Configure RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY before jobs can run.";
  }
  return "Upload stored. Processor is not configured.";
}

function asyncRunPodConfigured(env: Env): boolean {
  return Boolean(
    env.RUNPOD_RUNSYNC !== "1" && runPodEndpointId(env) && env.RUNPOD_API_KEY,
  );
}

function runPodEndpointId(env: Env): string {
  return (env.RUNPOD_ENDPOINT_ID ?? "").trim();
}

function publicJob(job: JobRecord, baseUrl: string): Record<string, unknown> {
  const resultReadyArtifact = job.artifacts[RESULT_READY_ARTIFACT_NAME];
  const resultReady = Boolean(
    resultReadyArtifact &&
    artifactBelongsToCurrentAttempt(job, resultReadyArtifact),
  );
  const artifacts = Object.fromEntries(
    Object.entries(job.artifacts)
      .filter(([, artifact]) => artifactBelongsToCurrentAttempt(job, artifact))
      .map(([name, artifact]) => [
        name,
        {
          href: `${baseUrl}/v1/artifacts/${job.run_id}/${encodeURIComponent(name)}`,
          content_type: artifact.content_type,
          size_bytes: artifact.size_bytes,
          updated_at: artifact.updated_at,
        },
      ]),
  );

  return {
    run_id: job.run_id,
    status: job.status,
    attempt_id: job.attempt_id ?? null,
    attempt_number: job.attempt_number ?? null,
    processor_enqueued_at: job.processor_enqueued_at ?? null,
    processing_attempt_count: job.processing_attempts?.length ?? (job.attempt_id ? 1 : 0),
    created_at: job.created_at,
    upload_completed_at: job.upload_completed_at ?? null,
    updated_at: job.updated_at,
    result_ready: resultReady,
    result_ready_at: resultReady ? resultReadyArtifact.updated_at : null,
    upload: {
      filename: job.upload.filename,
      content_type: job.upload.content_type,
      size_bytes: job.upload.size_bytes,
      consent_scope: job.upload.consent_scope,
    },
    target_prompt: job.target_prompt
      ? {
          source: job.target_prompt.source,
          selection_type: job.target_prompt.selection.type,
          frame: job.target_prompt.frame,
        }
      : null,
    progress: sanitizePublicProgress(job.progress),
    summary: null,
    error: job.error_code ?? (job.status === "failed" ? "PROCESSING_ATTEMPT_FAILED" : null),
    artifacts,
    links: {
      job: `${baseUrl}/v1/jobs/${job.run_id}`,
      start: `${baseUrl}/v1/jobs/${job.run_id}/start`,
      timeline: job.attempt_id
        ? `${baseUrl}/v1/jobs/${job.run_id}/events?attempt_id=${encodeURIComponent(job.attempt_id)}`
        : null,
    },
  };
}

function artifactBelongsToCurrentAttempt(
  job: JobRecord,
  artifact: ArtifactRecord,
): boolean {
  return Boolean(job.attempt_id && artifact.attempt_id === job.attempt_id);
}

type WorkerLifecycleDetails = Pick<
  ProcessingTelemetryEvent,
  "sequence" | "event_type" | "event_time"
> &
  Partial<
    Pick<
      ProcessingTelemetryEvent,
      "stage" | "span" | "status" | "elapsed_seconds" | "input" | "measurements" | "error"
    >
  >;

function workerLifecycleEvent(
  env: Env,
  job: JobRecord,
  details: WorkerLifecycleDetails,
): ProcessingTelemetryEvent {
  if (!job.attempt_id || !job.attempt_number) {
    throw new Error("Worker lifecycle event requires a Processing Attempt.");
  }
  return {
    schema_version: 1,
    event_id: crypto.randomUUID(),
    run_id: job.run_id,
    attempt_id: job.attempt_id,
    runtime: {
      service: "whodoirunlike-worker",
      environment: env.ENVIRONMENT.slice(0, 64),
      attempt_number: job.attempt_number,
      processor_enqueued_at: job.processor_enqueued_at ?? null,
      runpod_endpoint_id: persistedRunPodEndpointId(job) ?? null,
    },
    ...details,
  };
}

async function recordWorkerLifecycleEvent(
  env: Env,
  ctx: ExecutionContext,
  event: ProcessingTelemetryEvent,
): Promise<void> {
  try {
    const persisted = await persistTelemetryEvent(env, event, "worker");
    if (persisted.outbox_key && persisted.outbox_body) {
      ctx.waitUntil(
        deliverAnalyticsOutboxItem(env, persisted.outbox_key, persisted.outbox_body),
      );
    }
  } catch (error) {
    console.error(
      JSON.stringify({
        level: "error",
        message: "worker_lifecycle_event_failed",
        run_id: event.run_id,
        attempt_id: event.attempt_id,
        event_type: event.event_type,
        stage: event.stage ?? null,
        error: error instanceof Error ? error.message : "Unknown lifecycle persistence error",
      }),
    );
  }
}

function elapsedSeconds(start: string, end: string): number {
  return Math.max(0, (Date.parse(end) - Date.parse(start)) / 1000);
}

function enqueueJob(job: JobRecord): EnqueuedJobRecord {
  const processorEnqueuedAt = new Date().toISOString();
  const priorAttempts = job.processing_attempts ?? [];
  const reuseInitialAttempt =
    job.status === "uploaded" &&
    Boolean(job.attempt_id) &&
    Number.isInteger(job.attempt_number);
  const nextAttemptNumber = reuseInitialAttempt
    ? job.attempt_number ?? 1
    : Math.max(
        job.attempt_number ?? 0,
        ...priorAttempts.map((attempt) => attempt.attempt_number),
      ) + 1;
  const attempt: ProcessingAttemptRecord = reuseInitialAttempt
    ? {
        attempt_id: job.attempt_id ?? crypto.randomUUID(),
        attempt_number: nextAttemptNumber,
        created_at:
          priorAttempts.find((candidate) => candidate.attempt_id === job.attempt_id)?.created_at ??
          job.created_at,
        processor_enqueued_at: processorEnqueuedAt,
      }
    : {
        ...createProcessingAttempt(nextAttemptNumber, processorEnqueuedAt),
        processor_enqueued_at: processorEnqueuedAt,
      };
  const attempts = reuseInitialAttempt
    ? priorAttempts.some((candidate) => candidate.attempt_id === attempt.attempt_id)
      ? priorAttempts.map((candidate) =>
          candidate.attempt_id === attempt.attempt_id ? attempt : candidate,
        )
      : [...priorAttempts, attempt]
    : [...priorAttempts, attempt];
  const queued = updateJob(job, "queued");
  const enqueued: EnqueuedJobRecord = {
    ...queued,
    attempt_id: attempt.attempt_id,
    attempt_number: attempt.attempt_number,
    processor_enqueued_at: processorEnqueuedAt,
    processing_attempts: attempts,
    artifacts: reuseInitialAttempt ? queued.artifacts : {},
    error_code: undefined,
    runpod_endpoint_id: undefined,
    runpod_job_id: undefined,
  };
  console.log(
    JSON.stringify({
      level: "info",
      message: "processing_attempt_enqueued",
      run_id: job.run_id,
      attempt_id: attempt.attempt_id,
      attempt_number: attempt.attempt_number,
      processor_enqueued_at: attempt.processor_enqueued_at,
    }),
  );
  return enqueued;
}

function createProcessingAttempt(
  attemptNumber: number,
  createdAt: string,
): ProcessingAttemptRecord {
  return {
    attempt_id: crypto.randomUUID(),
    attempt_number: attemptNumber,
    created_at: createdAt,
  };
}

function jobHasAttempt(job: JobRecord, attemptId: string): boolean {
  if (job.attempt_id === attemptId) return true;
  return (job.processing_attempts ?? []).some((attempt) => attempt.attempt_id === attemptId);
}

function currentProcessingAttempt(job: JobRecord): ProcessingAttemptRecord | undefined {
  return (job.processing_attempts ?? []).find(
    (attempt) => attempt.attempt_id === job.attempt_id,
  );
}

function markProcessorStarted(job: JobRecord, startedAt: string): JobRecord {
  const updated = updateJob(job, job.status === "queued" ? "running" : job.status);
  updated.processor_started_at = startedAt;
  updated.processing_attempts = (updated.processing_attempts ?? []).map((attempt) =>
    attempt.attempt_id === updated.attempt_id
      ? { ...attempt, processor_started_at: startedAt }
      : attempt,
  );
  return updated;
}

async function transitionJobStatus(
  env: Env,
  job: JobRecord,
  target: "running" | "complete" | "failed",
): Promise<JobRecord> {
  const expectedAttemptId = job.attempt_id;
  for (let retry = 0; retry < 6; retry += 1) {
    const snapshot = await readJobSnapshot(env, job.run_id);
    if (!snapshot || snapshot.job.attempt_id !== expectedAttemptId) {
      throw new JobMutationConflictError("Processing attempt became stale.");
    }
    const current = snapshot.job;
    if (current.status === target) return current;
    if (current.status === "complete" || current.status === "failed") {
      throw new JobMutationConflictError("Job already has a different terminal state.");
    }

    const nextStatus: JobStatus = current.status === "uploaded"
      ? "queued"
      : current.status === "queued"
        ? "running"
        : target;
    const updated = updateJob(current, nextStatus);
    if (await writeJobIfUnchanged(env, updated, snapshot.etag)) {
      if (nextStatus === target) return updated;
    }
  }
  throw new JobMutationConflictError("Job changed too many times during transition.");
}

async function mutateCurrentJob(
  env: Env,
  runId: string,
  attemptId: string,
  mutate: (job: JobRecord) => JobRecord,
): Promise<JobRecord> {
  for (let retry = 0; retry < 6; retry += 1) {
    const snapshot = await readJobSnapshot(env, runId);
    if (!snapshot || snapshot.job.attempt_id !== attemptId) {
      throw new JobMutationConflictError("Processing attempt became stale.");
    }
    const updated = mutate(snapshot.job);
    if (await writeJobIfUnchanged(env, updated, snapshot.etag)) return updated;
  }
  throw new JobMutationConflictError("Job changed too many times during update.");
}

async function recordProviderQueueCompletion(
  env: Env,
  ctx: ExecutionContext,
  job: JobRecord,
): Promise<void> {
  const endpointId = persistedRunPodEndpointId(job) ?? runPodEndpointId(env);
  if (!job.runpod_job_id || !env.RUNPOD_API_KEY || !endpointId) return;
  try {
    const response = await fetch(
      `https://api.runpod.ai/v2/${endpointId}/status/${encodeURIComponent(job.runpod_job_id)}`,
      { headers: { Authorization: `Bearer ${env.RUNPOD_API_KEY}` } },
    );
    const payload = (await response.json().catch(() => null)) as {
      delayTime?: unknown;
      status?: unknown;
    } | null;
    if (!response.ok) {
      console.error(
        JSON.stringify({
          level: "error",
          message: "runpod_queue_timing_rejected",
          run_id: job.run_id,
          attempt_id: job.attempt_id ?? null,
          http_status: response.status,
        }),
      );
      return;
    }
    const delayTime = typeof payload?.delayTime === "number" ? payload.delayTime : Number.NaN;
    if (!Number.isFinite(delayTime) || delayTime < 0 || delayTime > 31_536_000_000) {
      console.error(
        JSON.stringify({
          level: "error",
          message: "runpod_queue_timing_missing",
          run_id: job.run_id,
          attempt_id: job.attempt_id ?? null,
        }),
      );
      return;
    }
    const queueStartedAt = job.processor_queue_started_at ?? job.processor_enqueued_at;
    if (!queueStartedAt) return;
    const completedAt = new Date(Date.parse(queueStartedAt) + delayTime).toISOString();
    await recordQueueCompletionEvent(env, ctx, job, completedAt, delayTime / 1000, {
      timing_basis: "runpod_delay_time",
      delay_time_ms: delayTime,
      runpod_endpoint_id: endpointId,
    });
  } catch (error) {
    console.error(
      JSON.stringify({
        level: "error",
        message: "runpod_queue_timing_failed",
        run_id: job.run_id,
        attempt_id: job.attempt_id ?? null,
        error: error instanceof Error ? error.message : "Unknown RunPod status error",
      }),
    );
  }
}

function persistedRunPodEndpointId(job: JobRecord): string | null {
  const attemptEndpointId = currentProcessingAttempt(job)?.runpod_endpoint_id?.trim();
  if (attemptEndpointId) return attemptEndpointId;
  const jobEndpointId = job.runpod_endpoint_id?.trim();
  return jobEndpointId || null;
}

async function recordDirectQueueCompletion(
  env: Env,
  ctx: ExecutionContext,
  job: JobRecord,
  firstProcessorEventTime: string,
): Promise<void> {
  const queueStartedAt = job.processor_queue_started_at ?? job.processor_enqueued_at;
  if (!queueStartedAt || job.runpod_job_id) return;
  const elapsed = elapsedSeconds(queueStartedAt, firstProcessorEventTime);
  await recordQueueCompletionEvent(env, ctx, job, firstProcessorEventTime, elapsed, {
    timing_basis: "worker_dispatch_to_start_estimate",
  });
}

async function recordQueueCompletionEvent(
  env: Env,
  ctx: ExecutionContext,
  job: JobRecord,
  eventTime: string,
  elapsed: number,
  measurements: Record<string, string | number>,
): Promise<void> {
  if (!job.attempt_id) return;
  const event = workerLifecycleEvent(env, job, {
    sequence: 7,
    event_type: "stage_completed",
    event_time: eventTime,
    stage: "processor_queue",
    status: "complete",
    elapsed_seconds: elapsed,
    measurements,
  });
  event.event_id = await deterministicTelemetryEventId(job.attempt_id, "processor_queue_completed");
  await recordWorkerLifecycleEvent(env, ctx, event);
}

async function deterministicTelemetryEventId(attemptId: string, name: string): Promise<string> {
  const digest = new Uint8Array(
    await crypto.subtle.digest("SHA-256", new TextEncoder().encode(`${attemptId}:${name}`)),
  );
  digest[6] = (digest[6] & 0x0f) | 0x50;
  digest[8] = (digest[8] & 0x3f) | 0x80;
  const hex = Array.from(digest.slice(0, 16), (byte) =>
    byte.toString(16).padStart(2, "0")
  ).join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

async function markProcessorQueueStarted(
  env: Env,
  fallbackJob: EnqueuedJobRecord,
  startedAt: string,
): Promise<JobRecord> {
  return mutateCurrentJob(env, fallbackJob.run_id, fallbackJob.attempt_id, (current) => {
    const updated = updateJob(current, current.status);
    updated.processor_queue_started_at = startedAt;
    updated.processing_attempts = (updated.processing_attempts ?? []).map((attempt) =>
      attempt.attempt_id === fallbackJob.attempt_id
        ? { ...attempt, processor_queue_started_at: startedAt }
        : attempt,
    );
    return updated;
  });
}

function updateJob(job: JobRecord, status: JobStatus): JobRecord {
  return {
    ...job,
    status,
    updated_at: new Date().toISOString(),
    artifacts: { ...job.artifacts },
    processing_attempts: job.processing_attempts
      ? job.processing_attempts.map((attempt) => ({ ...attempt }))
      : undefined,
  };
}

async function readJob(env: Env, runId: string): Promise<JobRecord | null> {
  return (await readJobSnapshot(env, runId))?.job ?? null;
}

async function readJobSnapshot(
  env: Env,
  runId: string,
): Promise<{ job: JobRecord; etag: string } | null> {
  const object = await env.CLIPS.get(jobKey(runId));
  if (!object) return null;
  return { job: (await object.json()) as JobRecord, etag: object.etag };
}

async function writeJob(env: Env, job: JobRecord): Promise<void> {
  await env.CLIPS.put(jobKey(job.run_id), JSON.stringify(job, null, 2), {
    httpMetadata: { contentType: "application/json; charset=utf-8" },
  });
}

async function writeJobIfUnchanged(
  env: Env,
  job: JobRecord,
  etag: string,
): Promise<boolean> {
  const stored = await env.CLIPS.put(jobKey(job.run_id), JSON.stringify(job, null, 2), {
    onlyIf: { etagMatches: etag },
    httpMetadata: { contentType: "application/json; charset=utf-8" },
  });
  return stored !== null;
}

function jobKey(runId: string): string {
  return `jobs/${runId}.json`;
}

function jsonResponse(
  request: Request,
  env: Env,
  payload: unknown,
  status = 200,
): Response {
  return new Response(JSON.stringify(payload, null, 2) + "\n", {
    status,
    headers: {
      ...JSON_HEADERS,
      ...corsHeaders(request, env),
    },
  });
}

function errorResponse(request: Request, env: Env, status: number, message: string): Response {
  return jsonResponse(request, env, { error: message }, status);
}

function notFound(request: Request, env: Env): Response {
  return errorResponse(request, env, 404, "Not found.");
}

function objectResponse(
  request: Request,
  env: Env,
  object: R2ObjectBody,
  headers: Record<string, string>,
): Response {
  return new Response(object.body, {
    headers: {
      ...headers,
      ETag: object.httpEtag,
      ...corsHeaders(request, env),
    },
  });
}

function corsHeaders(request: Request, env: Env): Record<string, string> {
  const origin = request.headers.get("origin");
  const origins = allowedOrigins(env);
  const allowedOrigin = origin && (origins.includes("*") || origins.includes(origin)) ? origin : origins[0];

  return {
    "Access-Control-Allow-Origin": allowedOrigin,
    "Access-Control-Allow-Methods": "GET, POST, PUT, OPTIONS",
    "Access-Control-Allow-Headers": "Authorization, Content-Type, X-Clip-Consent, X-Original-Filename, X-Processing-Attempt-Id, X-Runner-Prompt",
    "Access-Control-Max-Age": "86400",
    Vary: "Origin",
  };
}

type PromptParseResult =
  | { ok: true; prompt: RunnerTargetPrompt | null }
  | { ok: false; error: string };

function parseRunnerPromptHeader(rawHeader: string | null): PromptParseResult {
  if (!rawHeader || !rawHeader.trim()) {
    return { ok: true, prompt: null };
  }

  if (rawHeader.length > 4096) {
    return { ok: false, error: "Runner prompt is too large." };
  }

  let decoded = rawHeader;
  try {
    decoded = decodeURIComponent(rawHeader);
  } catch {
    decoded = rawHeader;
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(decoded);
  } catch {
    return { ok: false, error: "Runner prompt must be valid JSON." };
  }

  if (!parsed || typeof parsed !== "object") {
    return { ok: false, error: "Runner prompt must be an object." };
  }

  const prompt = parsed as Record<string, unknown>;
  const selection = prompt.selection;
  if (!selection || typeof selection !== "object") {
    return { ok: false, error: "Runner prompt selection is required." };
  }

  const selectionRecord = selection as Record<string, unknown>;
  const box = sanitizePromptBox(selectionRecord.box);
  const positivePoints = sanitizePromptPoints(selectionRecord.positive_points);
  const negativePoints = sanitizePromptPoints(selectionRecord.negative_points);

  if (!box && positivePoints.length === 0) {
    return { ok: false, error: "Select a runner with a box or a point before processing." };
  }

  const frame = sanitizePromptFrame(prompt.frame);
  return {
    ok: true,
    prompt: {
      version: 1,
      source: "hosted_upload_user_prompt_v1",
      selection: {
        type: box ? "box" : "point",
        positive_points: positivePoints,
        negative_points: negativePoints,
        ...(box ? { box } : {}),
      },
      frame,
      notes: "Selected in the upload UI.",
    },
  };
}

function sanitizePromptBox(value: unknown): RunnerPromptBox | undefined {
  if (!value || typeof value !== "object") return undefined;
  const raw = value as Record<string, unknown>;
  const x = normalizedNumber(raw.x);
  const y = normalizedNumber(raw.y);
  const width = normalizedNumber(raw.width);
  const height = normalizedNumber(raw.height);
  if (x === null || y === null || width === null || height === null) return undefined;
  const boundedX = Math.min(x, 0.98);
  const boundedY = Math.min(y, 0.96);
  const boundedWidth = Math.min(width, 1 - boundedX);
  const boundedHeight = Math.min(height, 1 - boundedY);
  if (boundedWidth < 0.02 || boundedHeight < 0.04) return undefined;
  return {
    x: roundUnit(boundedX),
    y: roundUnit(boundedY),
    width: roundUnit(boundedWidth),
    height: roundUnit(boundedHeight),
  };
}

function sanitizePromptPoints(value: unknown): RunnerPromptPoint[] {
  if (!Array.isArray(value)) return [];
  return value.slice(0, 4).flatMap((point): RunnerPromptPoint[] => {
    if (!point || typeof point !== "object") return [];
    const raw = point as Record<string, unknown>;
    const x = normalizedNumber(raw.x);
    const y = normalizedNumber(raw.y);
    if (x === null || y === null) return [];
    const label = typeof raw.label === "string" ? raw.label.slice(0, 80) : undefined;
    return [{ x: roundUnit(x), y: roundUnit(y), ...(label ? { label } : {}) }];
  });
}

function sanitizePromptFrame(value: unknown): RunnerTargetPrompt["frame"] {
  if (!value || typeof value !== "object") return {};
  const raw = value as Record<string, unknown>;
  const frame: RunnerTargetPrompt["frame"] = {};
  const timeSeconds = finiteNumber(raw.time_seconds);
  if (timeSeconds !== null && timeSeconds >= 0 && timeSeconds <= 600) {
    frame.time_seconds = Number(timeSeconds.toFixed(3));
  }
  const frameIndex = finiteNumber(raw.frame_index);
  if (frameIndex !== null && frameIndex >= 0) {
    frame.frame_index = Math.floor(frameIndex);
  }
  const width = finiteNumber(raw.width);
  const height = finiteNumber(raw.height);
  if (width !== null && width > 0) frame.width = Math.round(width);
  if (height !== null && height > 0) frame.height = Math.round(height);
  return frame;
}

function normalizedNumber(value: unknown): number | null {
  const parsed = finiteNumber(value);
  if (parsed === null || parsed < 0 || parsed > 1) return null;
  return parsed;
}

function finiteNumber(value: unknown): number | null {
  const parsed = typeof value === "number" ? value : typeof value === "string" ? Number(value) : Number.NaN;
  return Number.isFinite(parsed) ? parsed : null;
}

function roundUnit(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000;
}

function allowedOrigins(env: Env): string[] {
  const raw = env.PUBLIC_ORIGINS || "https://whodoirunlike.com";
  return raw
    .split(",")
    .map((origin) => origin.trim())
    .filter(Boolean);
}

function extensionForUpload(contentType: string, filename: string | null): string | null {
  const suffix = filename?.toLowerCase().match(/\.(mp4|mov|m4v|webm)$/)?.[0];
  if (suffix) return suffix;

  const normalized = contentType.split(";")[0].trim().toLowerCase();
  if (normalized === "video/mp4") return ".mp4";
  if (normalized === "video/quicktime" || normalized === "video/mov") return ".mov";
  if (normalized === "video/x-m4v") return ".m4v";
  if (normalized === "video/webm") return ".webm";
  if (normalized === "application/octet-stream") return ".mp4";
  return null;
}

function maxUploadBytes(env: Env): number {
  const parsed = Number.parseInt(env.MAX_UPLOAD_BYTES || "", 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_MAX_UPLOAD_BYTES;
}

function parseContentLength(request: Request): number | null {
  const raw = request.headers.get("content-length");
  if (!raw) return null;
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) ? parsed : null;
}

type BoundedRequestTextResult =
  | { ok: true; value: string }
  | { ok: false };

async function readBoundedRequestText(
  request: Request,
  maxBytes: number,
): Promise<BoundedRequestTextResult> {
  if (!request.body) return { ok: true, value: "" };
  const reader = request.body.getReader();
  const chunks: Uint8Array[] = [];
  let byteLength = 0;
  try {
    while (true) {
      const chunk = await reader.read();
      if (chunk.done) break;
      byteLength += chunk.value.byteLength;
      if (byteLength > maxBytes) {
        await reader.cancel("request body exceeds configured limit");
        return { ok: false };
      }
      chunks.push(chunk.value);
    }
  } finally {
    reader.releaseLock();
  }

  const body = new Uint8Array(byteLength);
  let offset = 0;
  for (const chunk of chunks) {
    body.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return { ok: true, value: new TextDecoder().decode(body) };
}

function normalizeRunId(value: string): string | null {
  const decoded = decodeURIComponent(value);
  return /^[a-f0-9-]{32,36}$/i.test(decoded) ? decoded : null;
}

function normalizeAttemptId(value: string): string | null {
  return /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/.test(value) ? value : null;
}

function normalizeArtifactName(value: string): string | null {
  const decoded = decodeURIComponent(value);
  return /^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}$/.test(decoded) ? decoded : null;
}

function normalizeArtifactContentType(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const normalized = value.trim();
  if (
    normalized.length < 3 ||
    normalized.length > 200 ||
    !/^[a-zA-Z0-9!#$&^_.+-]+\/[a-zA-Z0-9!#$&^_.+-]+(?:\s*;[^\r\n]*)?$/.test(normalized)
  ) {
    return null;
  }
  return normalized;
}

type ArtifactFinalizeParseResult =
  | { ok: true; attemptId: string; artifacts: ArtifactFinalizeItem[] }
  | { ok: false; error: string };

function parseArtifactFinalizePayload(value: unknown): ArtifactFinalizeParseResult {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return { ok: false, error: "Artifact finalization must be an object." };
  }
  const payload = value as Record<string, unknown>;
  const attemptId = normalizeAttemptId(
    typeof payload.attempt_id === "string" ? payload.attempt_id : "",
  );
  if (!attemptId) {
    return { ok: false, error: "Artifact finalization attempt_id is required." };
  }
  if (
    !Array.isArray(payload.artifacts) ||
    payload.artifacts.length < 1 ||
    payload.artifacts.length > MAX_ARTIFACT_FINALIZE_COUNT
  ) {
    return {
      ok: false,
      error: `Artifact finalization requires 1 to ${MAX_ARTIFACT_FINALIZE_COUNT} artifacts.`,
    };
  }

  const artifacts: ArtifactFinalizeItem[] = [];
  const seenNames = new Set<string>();
  for (const value of payload.artifacts) {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      return { ok: false, error: "Artifact finalization metadata is invalid." };
    }
    const raw = value as Record<string, unknown>;
    const name = normalizeArtifactName(typeof raw.name === "string" ? raw.name : "");
    const contentType = normalizeArtifactContentType(raw.content_type);
    const objectVersion = normalizeR2ObjectVersion(raw.object_version);
    const sizeBytes = raw.size_bytes;
    if (
      !name ||
      name === RESULT_READY_ARTIFACT_NAME ||
      !contentType ||
      !objectVersion ||
      typeof sizeBytes !== "number" ||
      !Number.isSafeInteger(sizeBytes) ||
      sizeBytes < 0 ||
      sizeBytes > MAX_R2_OBJECT_BYTES
    ) {
      return { ok: false, error: "Artifact finalization metadata is invalid." };
    }
    if (seenNames.has(name)) {
      return { ok: false, error: "Artifact finalization names must be unique." };
    }
    seenNames.add(name);
    artifacts.push({
      name,
      content_type: contentType,
      object_version: objectVersion,
      size_bytes: sizeBytes,
    });
  }
  return { ok: true, attemptId, artifacts };
}

function normalizeR2ObjectVersion(value: unknown): string | null {
  return typeof value === "string" && /^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$/.test(value)
    ? value
    : null;
}

function logAbandonedDeferredArtifact(
  runId: string,
  attemptId: string,
  artifactName: string,
  reason: "invalid_size" | "stale_attempt",
): void {
  console.warn(
    JSON.stringify({
      level: "warn",
      message: "deferred_artifact_abandoned",
      run_id: runId,
      attempt_id: attemptId,
      artifact: artifactName,
      reason,
      cleanup: "retained_unindexed_to_avoid_deleting_a_newer_write",
    }),
  );
}

function normalizeReportStatus(value: unknown): "running" | "complete" | "failed" | null {
  return value === "running" || value === "complete" || value === "failed" ? value : null;
}

function sanitizePublicProgress(value: unknown): Record<string, string | number> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const record = value as Record<string, unknown>;
  const output: Record<string, string | number> = {};
  if (typeof record.phase === "string" && /^[a-z][a-z0-9_.-]{0,63}$/.test(record.phase)) {
    output.phase = record.phase;
  }
  const numericFields = [
    "elapsed_seconds",
    "processed_frames",
    "total_frames",
    "percent",
    "eta_seconds",
  ] as const;
  for (const field of numericFields) {
    const numeric = record[field];
    if (typeof numeric === "number" && Number.isFinite(numeric) && numeric >= 0) {
      output[field] = numeric;
    }
  }
  return Object.keys(output).length ? output : null;
}

function publicBaseUrl(request: Request, env: Env): string {
  return trimTrailingSlash(env.PUBLIC_API_BASE_URL || new URL(request.url).origin);
}

function trimTrailingSlash(value: string): string {
  return value.replace(/\/+$/, "");
}

function contentDisposition(filename: string): string {
  const fallback = filename.replace(/[^a-zA-Z0-9_.-]/g, "_").slice(0, 120) || "artifact";
  return `inline; filename="${fallback}"`;
}

async function hasProcessorAuth(request: Request, env: Env): Promise<boolean> {
  const expected = env.PROCESSOR_SHARED_SECRET;
  if (!expected) return false;
  const auth = request.headers.get("authorization") ?? "";
  if (!auth.startsWith("Bearer ")) return false;
  const supplied = auth.slice("Bearer ".length);
  return timingSafeEqual(supplied, expected);
}

async function timingSafeEqual(left: string, right: string): Promise<boolean> {
  const [leftHash, rightHash] = await Promise.all([sha256(left), sha256(right)]);
  return crypto.subtle.timingSafeEqual(leftHash, rightHash);
}

async function sha256(value: string): Promise<Uint8Array> {
  const encoded = new TextEncoder().encode(value);
  const digest = await crypto.subtle.digest("SHA-256", encoded);
  return new Uint8Array(digest);
}
