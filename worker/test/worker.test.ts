import { env, exports } from "cloudflare:workers";
import {
  createExecutionContext,
  reset,
  waitOnExecutionContext,
} from "cloudflare:test";
import { afterEach, describe, expect, it, vi } from "vitest";

import worker from "../src/index";
import {
  buildTelemetryEventKey,
  deliverAnalyticsOutboxItem,
  reconcileAnalyticsOutbox,
} from "../src/telemetry";
import type { ProcessingTelemetryEvent } from "../src/telemetry";

const PROCESSOR_SECRET = "worker-test-processor-secret";

type UploadedJob = {
  run_id: string;
  attempt_id: string;
  attempt_number: number;
  status: string;
};

afterEach(async () => {
  vi.restoreAllMocks();
  await reset();
});

describe("processing attempt boundaries", () => {
  it("does not double-enqueue when start requests race", async () => {
    const job = await uploadClip();
    const runpodFetch = vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
      Response.json({ id: "runpod-race-job", status: "IN_QUEUE" })
    );
    const firstCtx = createExecutionContext();
    const secondCtx = createExecutionContext();
    const [first, second] = await Promise.all([
      worker.fetch(
        asIncomingRequest(new Request(`https://example.test/v1/jobs/${job.run_id}/start`, {
          method: "POST",
        })),
        env,
        firstCtx,
      ),
      worker.fetch(
        asIncomingRequest(new Request(`https://example.test/v1/jobs/${job.run_id}/start`, {
          method: "POST",
        })),
        env,
        secondCtx,
      ),
    ]);
    await Promise.all([
      waitOnExecutionContext(firstCtx),
      waitOnExecutionContext(secondCtx),
    ]);
    expect([first.status, second.status].every((status) => status === 202 || status === 409)).toBe(true);
    expect(runpodFetch).toHaveBeenCalledTimes(1);
  });

  it("rejects stale reports and artifacts and stores current artifacts by attempt", async () => {
    const job = await uploadClip();
    const staleAttemptId = crypto.randomUUID();

    const staleArtifact = await exports.default.fetch(
      `https://example.test/v1/jobs/${job.run_id}/artifacts/fused_overlay.mp4`,
      {
        method: "PUT",
        headers: processorHeaders({
          "Content-Type": "video/mp4",
          "X-Processing-Attempt-Id": staleAttemptId,
        }),
        body: new Uint8Array([1, 2, 3]),
      },
    );
    expect(staleArtifact.status).toBe(409);

    const staleReport = await exports.default.fetch(
      `https://example.test/v1/jobs/${job.run_id}/report`,
      {
        method: "POST",
        headers: processorHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({
          attempt_id: staleAttemptId,
          status: "failed",
          error: "must not reach the current job",
        }),
      },
    );
    expect(staleReport.status).toBe(409);

    const acceptedArtifact = await exports.default.fetch(
      `https://example.test/v1/jobs/${job.run_id}/artifacts/fused_overlay.mp4`,
      {
        method: "PUT",
        headers: processorHeaders({
          "Content-Type": "video/mp4",
          "X-Processing-Attempt-Id": job.attempt_id,
        }),
        body: new Uint8Array([4, 5, 6]),
      },
    );
    expect(acceptedArtifact.status).toBe(200);
    expect(
      await env.CLIPS.head(
        `artifacts/${job.run_id}/${job.attempt_id}/fused_overlay.mp4`,
      ),
    ).not.toBeNull();
    expect(
      await env.CLIPS.head(
        `artifacts/${job.run_id}/${staleAttemptId}/fused_overlay.mp4`,
      ),
    ).toBeNull();
  });

  it("moves current terminal telemetry through a safe public terminal state", async () => {
    const job = await uploadClip();
    const running = await postReport(job, {
      status: "running",
      progress: {
        phase: "running_full_cv_pipeline",
        processed_frames: 12,
        runpod_job_id: "must-not-be-public",
      },
      summary: { run_dir: "/tmp/private-run", checkpoint_path: "/models/private.pt" },
      error: "raw processor error",
    });
    expect(running.status).toBe(200);

    const terminal = telemetryEvent(job, 100, "attempt_completed");
    const accepted = await postEvent(job.run_id, terminal);
    expect(accepted.status).toBe(202);

    const response = await exports.default.fetch(
      `https://example.test/v1/jobs/${job.run_id}`,
    );
    expect(response.status).toBe(200);
    const publicJob = await response.json<Record<string, unknown>>();
    expect(publicJob.status).toBe("complete");
    expect(publicJob.summary).toBeNull();
    expect(publicJob.error).toBeNull();
    expect(JSON.stringify(publicJob)).not.toContain("private-run");
    expect(JSON.stringify(publicJob)).not.toContain("private.pt");
    expect(JSON.stringify(publicJob)).not.toContain("raw processor error");
    expect(JSON.stringify(publicJob)).not.toContain("must-not-be-public");

    const contradictory = telemetryEvent(job, 101, "attempt_failed");
    contradictory.error = {
      class: "ContradictoryFailure",
      code: "test.contradiction",
      category: "test",
      message: "must be rejected",
      retryable: false,
    };
    const rejected = await postEvent(job.run_id, contradictory);
    expect(rejected.status).toBe(409);
    expect(
      await env.CLIPS.head(buildTelemetryEventKey(contradictory)),
    ).toBeNull();
  });
});

describe("telemetry ordering and recovery", () => {
  it("records direct queue completion from the first running report exactly once", async () => {
    const job = await uploadClip();
    const queueStartedAt = new Date(Date.now() - 1500).toISOString();
    await seedOperationalJob(job, {
      status: "queued",
      processor_enqueued_at: queueStartedAt,
      processor_queue_started_at: queueStartedAt,
    });
    const request = new Request(`https://example.test/v1/jobs/${job.run_id}/report`, {
      method: "POST",
      headers: processorHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({
        attempt_id: job.attempt_id,
        status: "running",
        progress: { phase: "downloading_upload" },
      }),
    });
    const ctx = createExecutionContext();
    const response = await worker.fetch(asIncomingRequest(request), env, ctx);
    await waitOnExecutionContext(ctx);
    expect(response.status).toBe(200);

    await postEvent(job.run_id, telemetryEvent(job, 100, "stage_started", "source_download"));
    const queueEvents = await waitForStoredSequence(job, 7);
    expect(queueEvents).toHaveLength(1);
    expect(queueEvents[0]).toMatchObject({
      stage: "processor_queue",
      measurements: { timing_basis: "worker_dispatch_to_start_estimate" },
    });
  });

  it("records one authoritative RunPod queue completion from provider delayTime", async () => {
    const job = await uploadClip();
    const queueStartedAt = "2026-07-09T20:00:00.000Z";
    await seedOperationalJob(job, {
      status: "running",
      processor_enqueued_at: queueStartedAt,
      processor_queue_started_at: queueStartedAt,
      runpod_job_id: "runpod-job-test",
    });
    const statusFetch = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      Response.json({ delayTime: 2500, status: "COMPLETED" }),
    );

    const event = telemetryEvent(job, 100, "attempt_completed");
    const request = new Request(`https://example.test/v1/jobs/${job.run_id}/events`, {
      method: "POST",
      headers: processorHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(event),
    });
    const duplicateRequest = request.clone();
    const ctx = createExecutionContext();
    const response = await worker.fetch(asIncomingRequest(request), env, ctx);
    await waitOnExecutionContext(ctx);
    expect(response.status).toBe(202);
    const completions = await waitForStoredSequence(job, 7);
    expect(statusFetch).toHaveBeenCalledWith(
      expect.stringContaining("/status/runpod-job-test"),
      expect.any(Object),
    );
    expect(completions).toHaveLength(1);
    expect(completions[0]).toMatchObject({
      event_type: "stage_completed",
      stage: "processor_queue",
      elapsed_seconds: 2.5,
      measurements: {
        timing_basis: "runpod_delay_time",
        delay_time_ms: 2500,
      },
    });
    const secondCtx = createExecutionContext();
    const duplicateResponse = await worker.fetch(
      asIncomingRequest(duplicateRequest),
      env,
      secondCtx,
    );
    await waitOnExecutionContext(secondCtx);
    expect(duplicateResponse.status).toBe(200);
    expect(await waitForStoredSequence(job, 7)).toHaveLength(1);
    statusFetch.mockRestore();
  });

  it("uses the endpoint that accepted the attempt after configuration changes and keeps it private", async () => {
    const job = await uploadClip();
    const originalEndpointId = "endpoint-old-123";
    const replacementEndpointId = "endpoint-new-456";
    const runpodJobId = "runpod-provenance-job";
    const runpodFetch = vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof Request ? input.url : String(input);
      if (url === `https://api.runpod.ai/v2/${originalEndpointId}/run`) {
        return Response.json({ id: runpodJobId, status: "IN_QUEUE" });
      }
      if (url === `https://api.runpod.ai/v2/${originalEndpointId}/status/${runpodJobId}`) {
        return Response.json({ delayTime: 1250, status: "COMPLETED" });
      }
      return new Response("Unexpected RunPod endpoint", { status: 500 });
    });

    const startCtx = createExecutionContext();
    const started = await worker.fetch(
      asIncomingRequest(new Request(`https://example.test/v1/jobs/${job.run_id}/start`, {
        method: "POST",
      })),
      withRunPodEndpoint(originalEndpointId),
      startCtx,
    );
    await waitOnExecutionContext(startCtx);
    expect(started.status).toBe(202);

    const stored = await env.CLIPS.get(`jobs/${job.run_id}.json`);
    expect(stored).not.toBeNull();
    const internalJob = await stored!.json<{
      runpod_endpoint_id?: string;
      processing_attempts?: Array<{ attempt_id: string; runpod_endpoint_id?: string }>;
    }>();
    expect(internalJob.runpod_endpoint_id).toBe(originalEndpointId);
    expect(
      internalJob.processing_attempts?.find((attempt) => attempt.attempt_id === job.attempt_id)
        ?.runpod_endpoint_id,
    ).toBe(originalEndpointId);

    const terminal = telemetryEvent(job, 100, "attempt_completed");
    const terminalCtx = createExecutionContext();
    const terminalResponse = await worker.fetch(
      asIncomingRequest(new Request(`https://example.test/v1/jobs/${job.run_id}/events`, {
        method: "POST",
        headers: processorHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify(terminal),
      })),
      withRunPodEndpoint(replacementEndpointId),
      terminalCtx,
    );
    await waitOnExecutionContext(terminalCtx);
    expect(terminalResponse.status).toBe(202);
    expect(runpodFetch).toHaveBeenCalledWith(
      `https://api.runpod.ai/v2/${originalEndpointId}/status/${runpodJobId}`,
      expect.any(Object),
    );
    expect(
      runpodFetch.mock.calls.some(([input]) => String(input).includes(replacementEndpointId)),
    ).toBe(false);

    const completions = await waitForStoredSequence(job, 7);
    expect(completions).toHaveLength(1);
    expect(completions[0]).toMatchObject({
      runtime: { runpod_endpoint_id: originalEndpointId },
      measurements: { runpod_endpoint_id: originalEndpointId },
    });

    const publicCtx = createExecutionContext();
    const publicResponse = await worker.fetch(
      asIncomingRequest(new Request(`https://example.test/v1/jobs/${job.run_id}`)),
      withRunPodEndpoint(replacementEndpointId),
      publicCtx,
    );
    await waitOnExecutionContext(publicCtx);
    const publicJob = await publicResponse.json<Record<string, unknown>>();
    expect(JSON.stringify(publicJob)).not.toContain("runpod_endpoint_id");
    expect(JSON.stringify(publicJob)).not.toContain(originalEndpointId);
  });

  it("keeps endpoint provenance isolated across a failed attempt and retry", async () => {
    const firstJob = await uploadClip();
    const firstEndpointId = "endpoint-first-123";
    const secondEndpointId = "endpoint-second-456";
    const firstRunPodJobId = "runpod-first-attempt";
    const secondRunPodJobId = "runpod-second-attempt";
    const requestedRunPodUrls: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof Request ? input.url : String(input);
      requestedRunPodUrls.push(url);
      if (url === `https://api.runpod.ai/v2/${firstEndpointId}/run`) {
        return Response.json({ id: firstRunPodJobId, status: "IN_QUEUE" });
      }
      if (
        url ===
          `https://api.runpod.ai/v2/${firstEndpointId}/status/${firstRunPodJobId}`
      ) {
        return Response.json({ delayTime: 1100, status: "FAILED" });
      }
      if (url === `https://api.runpod.ai/v2/${secondEndpointId}/run`) {
        return Response.json({ id: secondRunPodJobId, status: "IN_QUEUE" });
      }
      if (
        url ===
          `https://api.runpod.ai/v2/${secondEndpointId}/status/${secondRunPodJobId}`
      ) {
        return Response.json({ delayTime: 2200, status: "COMPLETED" });
      }
      throw new Error(`Unexpected RunPod request: ${url}`);
    });

    const firstStartCtx = createExecutionContext();
    const firstStart = await worker.fetch(
      asIncomingRequest(new Request(`https://example.test/v1/jobs/${firstJob.run_id}/start`, {
        method: "POST",
      })),
      withRunPodEndpoint(firstEndpointId),
      firstStartCtx,
    );
    await waitOnExecutionContext(firstStartCtx);
    expect(firstStart.status).toBe(202);

    const firstFailure = telemetryEvent(firstJob, 100, "attempt_failed");
    firstFailure.status = "failed";
    firstFailure.error = {
      class: "FirstAttemptFailure",
      code: "test.first_attempt_failed",
      category: "test",
      message: "First attempt failed for retry coverage.",
      retryable: true,
    };
    const firstTerminalCtx = createExecutionContext();
    const firstTerminal = await worker.fetch(
      asIncomingRequest(new Request(`https://example.test/v1/jobs/${firstJob.run_id}/events`, {
        method: "POST",
        headers: processorHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify(firstFailure),
      })),
      withRunPodEndpoint(secondEndpointId),
      firstTerminalCtx,
    );
    await waitOnExecutionContext(firstTerminalCtx);
    expect(firstTerminal.status).toBe(202);
    expect(await waitForStoredSequence(firstJob, 7)).toEqual([
      expect.objectContaining({
        runtime: expect.objectContaining({ runpod_endpoint_id: firstEndpointId }),
        measurements: expect.objectContaining({ runpod_endpoint_id: firstEndpointId }),
      }),
    ]);

    const retryCtx = createExecutionContext();
    const retryResponse = await worker.fetch(
      asIncomingRequest(new Request(`https://example.test/v1/jobs/${firstJob.run_id}/start`, {
        method: "POST",
      })),
      withRunPodEndpoint(secondEndpointId),
      retryCtx,
    );
    await waitOnExecutionContext(retryCtx);
    expect(retryResponse.status).toBe(202);
    const secondJob = await retryResponse.json<UploadedJob>();
    expect(secondJob.attempt_id).not.toBe(firstJob.attempt_id);
    expect(secondJob.attempt_number).toBe(2);

    const staleReportCtx = createExecutionContext();
    const staleReport = await worker.fetch(
      asIncomingRequest(new Request(`https://example.test/v1/jobs/${firstJob.run_id}/report`, {
        method: "POST",
        headers: processorHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({ attempt_id: firstJob.attempt_id, status: "complete" }),
      })),
      withRunPodEndpoint(firstEndpointId),
      staleReportCtx,
    );
    await waitOnExecutionContext(staleReportCtx);
    expect(staleReport.status).toBe(409);

    const staleTerminal = telemetryEvent(firstJob, 101, "attempt_completed");
    const staleTerminalCtx = createExecutionContext();
    const staleTerminalResponse = await worker.fetch(
      asIncomingRequest(new Request(`https://example.test/v1/jobs/${firstJob.run_id}/events`, {
        method: "POST",
        headers: processorHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify(staleTerminal),
      })),
      withRunPodEndpoint(firstEndpointId),
      staleTerminalCtx,
    );
    await waitOnExecutionContext(staleTerminalCtx);
    expect(staleTerminalResponse.status).toBe(202);

    const afterStale = await readInternalJob(firstJob.run_id);
    expect(afterStale).toMatchObject({
      status: "queued",
      attempt_id: secondJob.attempt_id,
      runpod_endpoint_id: secondEndpointId,
      runpod_job_id: secondRunPodJobId,
    });

    const secondCompletion = telemetryEvent(secondJob, 100, "attempt_completed");
    const secondTerminalCtx = createExecutionContext();
    const secondTerminal = await worker.fetch(
      asIncomingRequest(new Request(`https://example.test/v1/jobs/${secondJob.run_id}/events`, {
        method: "POST",
        headers: processorHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify(secondCompletion),
      })),
      withRunPodEndpoint(firstEndpointId),
      secondTerminalCtx,
    );
    await waitOnExecutionContext(secondTerminalCtx);
    expect(secondTerminal.status).toBe(202);
    expect(await waitForStoredSequence(secondJob, 7)).toEqual([
      expect.objectContaining({
        runtime: expect.objectContaining({ runpod_endpoint_id: secondEndpointId }),
        measurements: expect.objectContaining({ runpod_endpoint_id: secondEndpointId }),
      }),
    ]);

    const finalJob = await readInternalJob(firstJob.run_id);
    expect(finalJob).toMatchObject({
      status: "complete",
      attempt_id: secondJob.attempt_id,
      runpod_endpoint_id: secondEndpointId,
      runpod_job_id: secondRunPodJobId,
    });
    expect(finalJob.processing_attempts).toEqual([
      expect.objectContaining({
        attempt_id: firstJob.attempt_id,
        runpod_endpoint_id: firstEndpointId,
        runpod_job_id: firstRunPodJobId,
      }),
      expect.objectContaining({
        attempt_id: secondJob.attempt_id,
        runpod_endpoint_id: secondEndpointId,
        runpod_job_id: secondRunPodJobId,
      }),
    ]);
    expect(requestedRunPodUrls).toEqual([
      `https://api.runpod.ai/v2/${firstEndpointId}/run`,
      `https://api.runpod.ai/v2/${firstEndpointId}/status/${firstRunPodJobId}`,
      `https://api.runpod.ai/v2/${secondEndpointId}/run`,
      `https://api.runpod.ai/v2/${secondEndpointId}/status/${secondRunPodJobId}`,
    ]);
  });

  it("uses sequence-prefixed keys and paginates a globally ordered timeline", async () => {
    const job = await uploadClip();
    await postEvent(job.run_id, telemetryEvent(job, 100, "stage_started", "source_download"));
    await postEvent(job.run_id, telemetryEvent(job, 101, "stage_completed", "source_download"));
    await postEvent(job.run_id, telemetryEvent(job, 102, "stage_started", "run_preparation"));

    const prefix = `telemetry/v1/events/${job.run_id}/${job.attempt_id}/`;
    const objects = await env.CLIPS.list({ prefix });
    expect(objects.objects.length).toBeGreaterThanOrEqual(6);
    for (const object of objects.objects) {
      expect(object.key.slice(prefix.length)).toMatch(/^\d{10}-[0-9a-f-]{36}\.json$/);
    }

    const sequences: number[] = [];
    let cursor: string | null = null;
    do {
      const query = new URLSearchParams({ attempt_id: job.attempt_id, limit: "2" });
      if (cursor) query.set("cursor", cursor);
      const response = await exports.default.fetch(
        `https://example.test/v1/jobs/${job.run_id}/events?${query}`,
        { headers: processorHeaders() },
      );
      expect(response.status).toBe(200);
      const page = await response.json<{
        events: Array<{ sequence: number }>;
        cursor: string | null;
      }>();
      sequences.push(...page.events.map((event) => event.sequence));
      cursor = page.cursor;
    } while (cursor);

    expect(sequences).toEqual([...sequences].sort((left, right) => left - right));
    expect(sequences).toEqual(expect.arrayContaining([1, 2, 3, 100, 101, 102]));
  });

  it("reconciles immutable events that are missing analytics outbox entries", async () => {
    const job = await uploadClip();
    const prefix = `telemetry/v1/events/${job.run_id}/${job.attempt_id}/`;
    const listed = await env.CLIPS.list({ prefix, limit: 1 });
    const stored = await env.CLIPS.get(listed.objects[0].key);
    expect(stored).not.toBeNull();
    const event = await stored!.json<ProcessingTelemetryEvent>();
    const outboxKey = `telemetry/v1/outbox/${job.run_id}/${job.attempt_id}/${event.event_id}.json`;
    expect(await env.CLIPS.head(outboxKey)).toBeNull();

    const reconciliationEnv: Env = {
      CLIPS: env.CLIPS,
      ENVIRONMENT: env.ENVIRONMENT,
      PUBLIC_API_BASE_URL: env.PUBLIC_API_BASE_URL,
      PUBLIC_ORIGINS: env.PUBLIC_ORIGINS,
      MAX_UPLOAD_BYTES: env.MAX_UPLOAD_BYTES,
      PROCESSOR_URL: env.PROCESSOR_URL,
      RUNPOD_RUNSYNC: env.RUNPOD_RUNSYNC,
      AWS_ANALYTICS_INGEST_URL: "https://analytics.example.test/events",
      PROCESSOR_SHARED_SECRET: env.PROCESSOR_SHARED_SECRET,
      RUNPOD_API_KEY: env.RUNPOD_API_KEY,
      RUNPOD_ENDPOINT_ID: env.RUNPOD_ENDPOINT_ID,
      AWS_ANALYTICS_SHARED_SECRET: env.AWS_ANALYTICS_SHARED_SECRET,
    };
    const first = await reconcileAnalyticsOutbox(reconciliationEnv, Date.now());
    expect(first.repaired).toBeGreaterThan(0);
    expect(await env.CLIPS.head(outboxKey)).not.toBeNull();

    const second = await reconcileAnalyticsOutbox(reconciliationEnv, Date.now());
    expect(second.repaired).toBe(0);

    const exportFetch = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(null, { status: 202 }),
    );
    expect(await deliverAnalyticsOutboxItem(reconciliationEnv, outboxKey)).toBe(true);
    expect(exportFetch).toHaveBeenCalledWith(
      "https://analytics.example.test/events",
      expect.objectContaining({ redirect: "manual" }),
    );
    expect(await env.CLIPS.head(outboxKey)).toBeNull();
    const receiptKey = `telemetry/v1/delivered/${job.run_id}/${job.attempt_id}/${event.event_id}.json`;
    expect(await env.CLIPS.head(receiptKey)).not.toBeNull();

    const afterDelivery = await reconcileAnalyticsOutbox(reconciliationEnv, Date.now());
    expect(afterDelivery.repaired).toBe(0);
    expect(await env.CLIPS.head(outboxKey)).toBeNull();
  });

  it("builds lexicographically sortable event keys", () => {
    const runId = crypto.randomUUID();
    const attemptId = crypto.randomUUID();
    const eventId = crypto.randomUUID();
    expect(
      buildTelemetryEventKey({ run_id: runId, attempt_id: attemptId, sequence: 7, event_id: eventId }),
    ).toBe(`telemetry/v1/events/${runId}/${attemptId}/0000000007-${eventId}.json`);
  });
});

async function uploadClip(): Promise<UploadedJob> {
  const response = await exports.default.fetch("https://example.test/v1/uploads", {
    method: "POST",
    headers: { "Content-Type": "video/mp4", "X-Original-Filename": "test.mp4" },
    body: new Uint8Array([0, 1, 2, 3]),
  });
  expect(response.status).toBe(201);
  return response.json<UploadedJob>();
}

async function postReport(
  job: UploadedJob,
  values: Record<string, unknown>,
): Promise<Response> {
  return exports.default.fetch(`https://example.test/v1/jobs/${job.run_id}/report`, {
    method: "POST",
    headers: processorHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ attempt_id: job.attempt_id, ...values }),
  });
}

async function postEvent(runId: string, event: ProcessingTelemetryEvent): Promise<Response> {
  return exports.default.fetch(`https://example.test/v1/jobs/${runId}/events`, {
    method: "POST",
    headers: processorHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(event),
  });
}

function telemetryEvent(
  job: UploadedJob,
  sequence: number,
  eventType: ProcessingTelemetryEvent["event_type"],
  stage?: ProcessingTelemetryEvent["stage"],
): ProcessingTelemetryEvent {
  return {
    schema_version: 1,
    event_id: crypto.randomUUID(),
    run_id: job.run_id,
    attempt_id: job.attempt_id,
    sequence,
    event_type: eventType,
    event_time: new Date(Date.now() + sequence).toISOString(),
    ...(stage ? { stage } : {}),
    status: eventType.endsWith("completed") ? "complete" : "running",
    elapsed_seconds: sequence / 100,
    runtime: { service: "worker-test-processor" },
    resources: {},
    measurements: {},
  };
}

function processorHeaders(extra: Record<string, string> = {}): Headers {
  return new Headers({
    Authorization: `Bearer ${PROCESSOR_SECRET}`,
    ...extra,
  });
}

function withRunPodEndpoint(endpointId: string): Env {
  return {
    CLIPS: env.CLIPS,
    ENVIRONMENT: env.ENVIRONMENT,
    PUBLIC_API_BASE_URL: env.PUBLIC_API_BASE_URL,
    PUBLIC_ORIGINS: env.PUBLIC_ORIGINS,
    MAX_UPLOAD_BYTES: env.MAX_UPLOAD_BYTES,
    PROCESSOR_URL: env.PROCESSOR_URL,
    RUNPOD_RUNSYNC: env.RUNPOD_RUNSYNC,
    PROCESSOR_SHARED_SECRET: env.PROCESSOR_SHARED_SECRET,
    RUNPOD_API_KEY: env.RUNPOD_API_KEY,
    RUNPOD_ENDPOINT_ID: endpointId,
    AWS_ANALYTICS_INGEST_URL: env.AWS_ANALYTICS_INGEST_URL,
    AWS_ANALYTICS_SHARED_SECRET: env.AWS_ANALYTICS_SHARED_SECRET,
  };
}

async function readInternalJob(runId: string): Promise<Record<string, unknown>> {
  const stored = await env.CLIPS.get(`jobs/${runId}.json`);
  expect(stored).not.toBeNull();
  return stored!.json<Record<string, unknown>>();
}

async function seedOperationalJob(
  job: UploadedJob,
  fields: Record<string, unknown>,
): Promise<void> {
  const key = `jobs/${job.run_id}.json`;
  const object = await env.CLIPS.get(key);
  expect(object).not.toBeNull();
  const record = await object!.json<Record<string, unknown>>();
  Object.assign(record, fields);
  if (Array.isArray(record.processing_attempts)) {
    record.processing_attempts = record.processing_attempts.map((attempt) =>
      attempt && typeof attempt === "object"
        ? { ...(attempt as Record<string, unknown>), ...fields }
        : attempt
    );
  }
  await env.CLIPS.put(key, JSON.stringify(record), {
    httpMetadata: { contentType: "application/json; charset=utf-8" },
  });
}

async function storedEvents(job: UploadedJob): Promise<Array<Record<string, unknown>>> {
  const prefix = `telemetry/v1/events/${job.run_id}/${job.attempt_id}/`;
  const listed = await env.CLIPS.list({ prefix });
  return Promise.all(
    listed.objects.map(async (object) => {
      const stored = await env.CLIPS.get(object.key);
      expect(stored).not.toBeNull();
      return stored!.json<Record<string, unknown>>();
    }),
  );
}

async function waitForStoredSequence(
  job: UploadedJob,
  sequence: number,
): Promise<Array<Record<string, unknown>>> {
  for (let attempt = 0; attempt < 20; attempt += 1) {
    const matches = (await storedEvents(job)).filter((event) => event.sequence === sequence);
    if (matches.length) return matches;
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  return [];
}

function asIncomingRequest(
  request: Request,
): Request<unknown, IncomingRequestCfProperties> {
  return request as Request<unknown, IncomingRequestCfProperties>;
}
