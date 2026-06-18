type WorkerEnv = Env & {
  PROCESSOR_SHARED_SECRET?: string;
  RUNPOD_API_KEY?: string;
  RUNPOD_ENDPOINT_ID?: string;
  RUNPOD_RUNSYNC?: string;
};

type JobStatus = "uploaded" | "queued" | "running" | "complete" | "failed";

type ArtifactRecord = {
  key: string;
  content_type: string;
  size_bytes: number;
  updated_at: string;
};

type JobRecord = {
  version: 1;
  run_id: string;
  status: JobStatus;
  created_at: string;
  updated_at: string;
  upload: {
    key: string;
    filename: string | null;
    content_type: string;
    size_bytes: number;
    consent_scope: string | null;
  };
  progress?: unknown;
  summary?: unknown;
  error?: string;
  artifacts: Record<string, ArtifactRecord>;
};

type ProcessorPayload = {
  run_id: string;
  source: {
    url: string;
    key: string;
    filename: string | null;
    content_type: string;
    size_bytes: number;
  };
  callback_base_url: string;
};

type RunPodStartResponse = {
  id?: string;
  status?: string;
  error?: string;
};

const DEFAULT_MAX_UPLOAD_BYTES = 75 * 1024 * 1024;
const JSON_HEADERS = {
  "Content-Type": "application/json; charset=utf-8",
};

export default {
  async fetch(request, env, ctx): Promise<Response> {
    return handleRequest(request, env as WorkerEnv, ctx);
  },
} satisfies ExportedHandler<Env>;

async function handleRequest(
  request: Request,
  env: WorkerEnv,
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
        return handleStartJob(request, env, runId);
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

      if (request.method === "POST" && segments.length === 4 && segments[3] === "report") {
        return handleReport(request, env, runId);
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
  env: WorkerEnv,
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

  const runId = crypto.randomUUID();
  const createdAt = new Date().toISOString();
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

  const job: JobRecord = {
    version: 1,
    run_id: runId,
    status: "uploaded",
    created_at: createdAt,
    updated_at: createdAt,
    upload: {
      key: uploadKey,
      filename: originalFilename,
      content_type: contentType,
      size_bytes: uploaded.size,
      consent_scope: request.headers.get("x-clip-consent"),
    },
    artifacts: {},
  };
  await writeJob(env, job);

  const response = publicJob(job, publicBaseUrl(request, env));
  if (new URL(request.url).searchParams.get("start") === "1" && processorConfigured(env)) {
    ctx.waitUntil(notifyProcessor(request, env, job));
    response.status = "queued";
    response.message = "Uploaded. Processor notification started.";
  }

  return jsonResponse(request, env, response, 201);
}

async function handleGetJob(request: Request, env: WorkerEnv, runId: string): Promise<Response> {
  const job = await readJob(env, runId);
  if (!job) {
    return errorResponse(request, env, 404, "Job not found.");
  }
  return jsonResponse(request, env, publicJob(job, publicBaseUrl(request, env)));
}

async function handleStartJob(request: Request, env: WorkerEnv, runId: string): Promise<Response> {
  const job = await readJob(env, runId);
  if (!job) {
    return errorResponse(request, env, 404, "Job not found.");
  }

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

  const queued = updateJob(job, "queued");
  await writeJob(env, queued);
  await notifyProcessor(request, env, queued);
  return jsonResponse(request, env, publicJob(queued, publicBaseUrl(request, env)), 202);
}

async function handleGetSource(request: Request, env: WorkerEnv, runId: string): Promise<Response> {
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
  env: WorkerEnv,
  runId: string,
  artifactName: string,
): Promise<Response> {
  if (!(await hasProcessorAuth(request, env))) {
    return errorResponse(request, env, 401, "Processor authorization required.");
  }
  if (!request.body) {
    return errorResponse(request, env, 400, "Artifact body is required.");
  }

  const job = await readJob(env, runId);
  if (!job) {
    return errorResponse(request, env, 404, "Job not found.");
  }

  const contentType = request.headers.get("content-type") ?? "application/octet-stream";
  const artifactKey = `artifacts/${runId}/${artifactName}`;
  const stored = await env.CLIPS.put(artifactKey, request.body, {
    httpMetadata: { contentType },
    customMetadata: { run_id: runId, artifact_name: artifactName },
  });

  const updated = updateJob(job, job.status);
  updated.artifacts[artifactName] = {
    key: artifactKey,
    content_type: contentType,
    size_bytes: stored.size,
    updated_at: updated.updated_at,
  };
  await writeJob(env, updated);

  return jsonResponse(request, env, {
    run_id: runId,
    artifact: artifactName,
    status: "stored",
    size_bytes: stored.size,
  });
}

async function handleReport(request: Request, env: WorkerEnv, runId: string): Promise<Response> {
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
    status?: JobStatus;
    progress?: unknown;
    summary?: unknown;
    error?: string;
  } | null;
  if (!payload) {
    return errorResponse(request, env, 400, "Report must be valid JSON.");
  }

  const status = normalizeStatus(payload.status) ?? job.status;
  const updated = updateJob(job, status);
  if ("progress" in payload) updated.progress = payload.progress;
  if ("summary" in payload) updated.summary = payload.summary;
  if (payload.error) updated.error = String(payload.error).slice(0, 2000);
  await writeJob(env, updated);

  return jsonResponse(request, env, publicJob(updated, publicBaseUrl(request, env)));
}

async function handleGetArtifact(
  request: Request,
  env: WorkerEnv,
  runId: string,
  artifactName: string,
): Promise<Response> {
  const job = await readJob(env, runId);
  if (!job) {
    return errorResponse(request, env, 404, "Job not found.");
  }
  const artifact = job.artifacts[artifactName];
  if (!artifact) {
    return errorResponse(request, env, 404, "Artifact not found.");
  }
  const object = await env.CLIPS.get(artifact.key);
  if (!object) {
    return errorResponse(request, env, 404, "Artifact object not found.");
  }
  return objectResponse(request, env, object, {
    "Content-Type": artifact.content_type,
    "Content-Disposition": contentDisposition(artifactName),
    "Cache-Control": "private, max-age=300",
  });
}

async function notifyProcessor(request: Request, env: WorkerEnv, job: JobRecord): Promise<void> {
  if (!env.PROCESSOR_SHARED_SECRET) return;
  const base = publicBaseUrl(request, env);
  const payload: ProcessorPayload = {
    run_id: job.run_id,
    source: {
      url: `${base}/v1/jobs/${job.run_id}/source`,
      key: job.upload.key,
      filename: job.upload.filename,
      content_type: job.upload.content_type,
      size_bytes: job.upload.size_bytes,
    },
    callback_base_url: base,
  };

  if (runPodEndpointId(env) && env.RUNPOD_API_KEY) {
    await notifyRunPod(env, job, payload);
    return;
  }

  if (!env.PROCESSOR_URL) return;
  const response = await fetch(`${trimTrailingSlash(env.PROCESSOR_URL)}/v1/processor/jobs`, {
    method: "POST",
    headers: {
      ...JSON_HEADERS,
      Authorization: `Bearer ${env.PROCESSOR_SHARED_SECRET}`,
    },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const failed = updateJob(job, "failed");
    failed.error = `Processor rejected job with HTTP ${response.status}`;
    await writeJob(env, failed);
    throw new Error(failed.error);
  }
}

async function notifyRunPod(
  env: WorkerEnv,
  job: JobRecord,
  payload: ProcessorPayload,
): Promise<void> {
  const endpointId = runPodEndpointId(env);
  const runMode = env.RUNPOD_RUNSYNC === "1" ? "runsync" : "run";
  const response = await fetch(`https://api.runpod.ai/v2/${endpointId}/${runMode}`, {
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

  const queued = updateJob(job, "queued");
  queued.progress = {
    phase: "queued_on_runpod",
    runpod_job_id: body?.id ?? null,
    runpod_status: body?.status ?? null,
  };
  await writeJob(env, queued);
}

function processorConfigured(env: WorkerEnv): boolean {
  const hasSharedSecret = Boolean(env.PROCESSOR_SHARED_SECRET);
  const hasRunPod = Boolean(runPodEndpointId(env) && env.RUNPOD_API_KEY);
  const hasUrlProcessor = Boolean(env.PROCESSOR_URL);
  return hasSharedSecret && (hasRunPod || hasUrlProcessor);
}

function processorConfigurationMessage(env: WorkerEnv): string {
  if (!env.PROCESSOR_SHARED_SECRET) {
    return "Upload stored. Configure PROCESSOR_SHARED_SECRET before jobs can run.";
  }
  if (!runPodEndpointId(env) || !env.RUNPOD_API_KEY) {
    return "Upload stored. Configure RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY before jobs can run.";
  }
  return "Upload stored. Processor is not configured.";
}

function runPodEndpointId(env: WorkerEnv): string {
  return (env.RUNPOD_ENDPOINT_ID ?? "").trim();
}

function publicJob(job: JobRecord, baseUrl: string): Record<string, unknown> {
  const artifacts = Object.fromEntries(
    Object.entries(job.artifacts).map(([name, artifact]) => [
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
    created_at: job.created_at,
    updated_at: job.updated_at,
    upload: {
      filename: job.upload.filename,
      content_type: job.upload.content_type,
      size_bytes: job.upload.size_bytes,
      consent_scope: job.upload.consent_scope,
    },
    progress: job.progress ?? null,
    summary: job.summary ?? null,
    error: job.error ?? null,
    artifacts,
    links: {
      job: `${baseUrl}/v1/jobs/${job.run_id}`,
      start: `${baseUrl}/v1/jobs/${job.run_id}/start`,
    },
  };
}

function updateJob(job: JobRecord, status: JobStatus): JobRecord {
  return {
    ...job,
    status,
    updated_at: new Date().toISOString(),
    artifacts: { ...job.artifacts },
  };
}

async function readJob(env: WorkerEnv, runId: string): Promise<JobRecord | null> {
  const object = await env.CLIPS.get(jobKey(runId));
  if (!object) return null;
  return (await object.json()) as JobRecord;
}

async function writeJob(env: WorkerEnv, job: JobRecord): Promise<void> {
  await env.CLIPS.put(jobKey(job.run_id), JSON.stringify(job, null, 2), {
    httpMetadata: { contentType: "application/json; charset=utf-8" },
  });
}

function jobKey(runId: string): string {
  return `jobs/${runId}.json`;
}

function jsonResponse(
  request: Request,
  env: WorkerEnv,
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

function errorResponse(request: Request, env: WorkerEnv, status: number, message: string): Response {
  return jsonResponse(request, env, { error: message }, status);
}

function notFound(request: Request, env: WorkerEnv): Response {
  return errorResponse(request, env, 404, "Not found.");
}

function objectResponse(
  request: Request,
  env: WorkerEnv,
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

function corsHeaders(request: Request, env: WorkerEnv): Record<string, string> {
  const origin = request.headers.get("origin");
  const origins = allowedOrigins(env);
  const allowedOrigin = origin && (origins.includes("*") || origins.includes(origin)) ? origin : origins[0];

  return {
    "Access-Control-Allow-Origin": allowedOrigin,
    "Access-Control-Allow-Methods": "GET, POST, PUT, OPTIONS",
    "Access-Control-Allow-Headers": "Authorization, Content-Type, X-Clip-Consent, X-Original-Filename",
    "Access-Control-Max-Age": "86400",
    Vary: "Origin",
  };
}

function allowedOrigins(env: WorkerEnv): string[] {
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

function maxUploadBytes(env: WorkerEnv): number {
  const parsed = Number.parseInt(env.MAX_UPLOAD_BYTES || "", 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_MAX_UPLOAD_BYTES;
}

function parseContentLength(request: Request): number | null {
  const raw = request.headers.get("content-length");
  if (!raw) return null;
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) ? parsed : null;
}

function normalizeRunId(value: string): string | null {
  const decoded = decodeURIComponent(value);
  return /^[a-f0-9-]{32,36}$/i.test(decoded) ? decoded : null;
}

function normalizeArtifactName(value: string): string | null {
  const decoded = decodeURIComponent(value);
  return /^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}$/.test(decoded) ? decoded : null;
}

function normalizeStatus(value: unknown): JobStatus | null {
  if (
    value === "uploaded" ||
    value === "queued" ||
    value === "running" ||
    value === "complete" ||
    value === "failed"
  ) {
    return value;
  }
  return null;
}

function publicBaseUrl(request: Request, env: WorkerEnv): string {
  return trimTrailingSlash(env.PUBLIC_API_BASE_URL || new URL(request.url).origin);
}

function trimTrailingSlash(value: string): string {
  return value.replace(/\/+$/, "");
}

function contentDisposition(filename: string): string {
  const fallback = filename.replace(/[^a-zA-Z0-9_.-]/g, "_").slice(0, 120) || "artifact";
  return `inline; filename="${fallback}"`;
}

async function hasProcessorAuth(request: Request, env: WorkerEnv): Promise<boolean> {
  const expected = env.PROCESSOR_SHARED_SECRET;
  if (!expected) return false;
  const auth = request.headers.get("authorization") ?? "";
  if (!auth.startsWith("Bearer ")) return false;
  const supplied = auth.slice("Bearer ".length);
  return timingSafeEqual(supplied, expected);
}

async function timingSafeEqual(left: string, right: string): Promise<boolean> {
  const [leftHash, rightHash] = await Promise.all([sha256(left), sha256(right)]);
  let diff = 0;
  for (let index = 0; index < leftHash.length; index += 1) {
    diff |= leftHash[index] ^ rightHash[index];
  }
  return diff === 0;
}

async function sha256(value: string): Promise<Uint8Array> {
  const encoded = new TextEncoder().encode(value);
  const digest = await crypto.subtle.digest("SHA-256", encoded);
  return new Uint8Array(digest);
}
