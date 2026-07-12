const TELEMETRY_SCHEMA_VERSION = 1;
export const TELEMETRY_MAX_BODY_BYTES = 64 * 1024;

const EVENT_TYPES = [
  "attempt_started",
  "stage_started",
  "span_started",
  "progress_sampled",
  "span_completed",
  "span_failed",
  "stage_completed",
  "stage_failed",
  "result_ready",
  "analysis_completed",
  "attempt_completed",
  "attempt_failed",
] as const;

const PIPELINE_STAGES = [
  "source_ingest",
  "processor_enqueue",
  "processor_queue",
  "source_download",
  "run_preparation",
  "target_tracking",
  "runner_mask",
  "pose_sequence",
  "densepose_body_map",
  "fused_form_signal",
  "form_feature_compilation",
  "artifact_table_export",
  "quality_control",
  "artifact_publish",
  "result_ready",
  "analysis_complete",
] as const;

const PROCESSING_SPANS = [
  "model_load",
  "decode",
  "preprocess",
  "inference",
  "postprocess",
  "render",
  "encode",
  "write",
  "publish",
] as const;

const EVENT_STATUSES = ["queued", "running", "complete", "failed"] as const;

const REQUIRED_EVENT_KEYS = [
  "schema_version",
  "event_id",
  "run_id",
  "attempt_id",
  "sequence",
  "event_type",
  "event_time",
] as const;

const OPTIONAL_EVENT_KEYS = [
  "stage",
  "span",
  "status",
  "elapsed_seconds",
  "progress",
  "input",
  "runtime",
  "resources",
  "measurements",
  "error",
  "attributes",
] as const;

const EVENT_KEYS = [...REQUIRED_EVENT_KEYS, ...OPTIONAL_EVENT_KEYS] as const;
const STAGE_EVENT_TYPES = [
  "stage_started",
  "stage_completed",
  "stage_failed",
  "span_started",
  "span_completed",
  "span_failed",
  "progress_sampled",
] as const;
const SPAN_EVENT_TYPES = ["span_started", "span_completed", "span_failed"] as const;
const FAILED_EVENT_TYPES = ["span_failed", "stage_failed", "attempt_failed"] as const;

const OUTBOX_PREFIX = "telemetry/v1/outbox/";
const DELIVERED_PREFIX = "telemetry/v1/delivered/";
const EVENT_PREFIX = "telemetry/v1/events/";
const EVENT_ID_INDEX_PREFIX = "telemetry/v1/event-ids/";
const OUTBOX_CURSOR_KEY = "telemetry/v1/state/outbox-retry-cursor.txt";
const RECONCILIATION_CURSOR_KEY = "telemetry/v1/state/reconciliation-cursor.txt";
const OUTBOX_BATCH_SIZE = 100;
const OUTBOX_CONCURRENCY = 10;
const RECONCILIATION_BATCH_SIZE = 50;
const MAX_OUTBOX_OBJECT_BYTES = TELEMETRY_MAX_BODY_BYTES + 16 * 1024;
const MAX_METADATA_PROPERTIES = 64;

const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const METADATA_KEY_PATTERN = /^[A-Za-z][A-Za-z0-9_.-]{0,63}$/;
const SENSITIVE_METADATA_KEYS = [
  "authorization",
  "callback_base_url",
  "filename",
  "key",
  "password",
  "path",
  "prompt",
  "secret",
  "source_path",
  "source_url",
  "token",
  "traceback",
  "url",
] as const;
const SENSITIVE_VALUE_PATTERNS = [
  /https?:\/\//i,
  /\bBearer\s+\S+/i,
  /(?:^|\s)\/(?:[^\s/]+\/)+[^\s/]+/,
  /\b[A-Za-z]:\\(?:[^\s\\]+\\)+[^\s\\]+/,
] as const;

type EventType = (typeof EVENT_TYPES)[number];
type PipelineStage = (typeof PIPELINE_STAGES)[number];
type ProcessingSpan = (typeof PROCESSING_SPANS)[number];
type EventStatus = (typeof EVENT_STATUSES)[number];
type JsonScalar = string | number | boolean | null;
type BoundedJsonValue = JsonScalar | BoundedJsonValue[] | { [key: string]: BoundedJsonValue };
type BoundedJsonObject = { [key: string]: BoundedJsonValue };

export type ProcessingTelemetryEvent = {
  schema_version: 1;
  event_id: string;
  run_id: string;
  attempt_id: string;
  sequence: number;
  event_type: EventType;
  event_time: string;
  stage?: PipelineStage | null;
  span?: ProcessingSpan | null;
  status?: EventStatus;
  elapsed_seconds?: number;
  progress?: BoundedJsonObject;
  input?: BoundedJsonObject;
  runtime?: BoundedJsonObject;
  resources?: BoundedJsonObject;
  measurements?: BoundedJsonObject;
  error?: BoundedJsonObject;
  attributes?: BoundedJsonObject;
};

type ParseResult =
  | { ok: true; value: ProcessingTelemetryEvent }
  | { ok: false; error: string; status: 400 | 413 };

export type PersistTelemetryResult = {
  event: ProcessingTelemetryEvent;
  event_key: string;
  outbox_key: string | null;
  outbox_body: string | null;
  duplicate: boolean;
};

export type TimelineResult = {
  run_id: string;
  attempt_id: string;
  events: unknown[];
  truncated: boolean;
  cursor: string | null;
};

export class TelemetryConflictError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "TelemetryConflictError";
  }
}

export class TelemetryPayloadTooLargeError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "TelemetryPayloadTooLargeError";
  }
}

export async function parseTelemetryRequest(
  request: Request,
  expectedRunId: string,
): Promise<ParseResult> {
  const declaredLength = parseContentLength(request);
  if (declaredLength !== null && declaredLength > TELEMETRY_MAX_BODY_BYTES) {
    return { ok: false, error: "Telemetry event body is too large.", status: 413 };
  }
  if (!request.body) {
    return { ok: false, error: "Telemetry event body is required.", status: 400 };
  }

  const readResult = await readBoundedBody(request.body, TELEMETRY_MAX_BODY_BYTES);
  if (!readResult.ok) return readResult;

  let parsed: unknown;
  try {
    parsed = JSON.parse(
      new TextDecoder("utf-8", { fatal: true, ignoreBOM: false }).decode(readResult.value),
    );
  } catch {
    return { ok: false, error: "Telemetry event must be valid UTF-8 JSON.", status: 400 };
  }

  return validateTelemetryEvent(parsed, expectedRunId);
}

export async function persistTelemetryEvent(
  env: Env,
  event: ProcessingTelemetryEvent,
  source: "processor" | "worker" = "processor",
): Promise<PersistTelemetryResult> {
  const storedEvent: ProcessingTelemetryEvent = {
    ...event,
    runtime: {
      ...(event.runtime ?? {}),
      environment: boundedEnvironment(env.ENVIRONMENT),
    },
  };
  const candidateBody = `${JSON.stringify(storedEvent)}\n`;
  if (new TextEncoder().encode(candidateBody).byteLength > TELEMETRY_MAX_BODY_BYTES) {
    throw new TelemetryPayloadTooLargeError("Telemetry event exceeds 65536 bytes after enrichment.");
  }
  const payloadSha256 = await sha256Hex(candidateBody);
  const receivedAt = new Date().toISOString();
  const eventKey = buildTelemetryEventKey(storedEvent);
  const eventIdIndexKey = telemetryEventIdIndexKey(storedEvent);
  const eventIdIndexBody = `${JSON.stringify({
    version: 1,
    event_key: eventKey,
    payload_sha256: payloadSha256,
  })}\n`;
  const indexCreated = await env.CLIPS.put(eventIdIndexKey, eventIdIndexBody, {
    onlyIf: new Headers({ "If-None-Match": "*" }),
    httpMetadata: { contentType: "application/json; charset=utf-8" },
    customMetadata: telemetryMetadata(storedEvent, source, payloadSha256, receivedAt, env.ENVIRONMENT),
  });

  let storedBody = candidateBody;
  let duplicate = !indexCreated;
  if (!indexCreated) {
    const existingIndex = await env.CLIPS.get(eventIdIndexKey);
    if (!existingIndex) {
      throw new Error("Telemetry event identity conflicted but could not be read.");
    }
    const identity = parseEventIdIndex(await existingIndex.text());
    if (
      !identity ||
      identity.event_key !== eventKey ||
      identity.payload_sha256 !== payloadSha256
    ) {
      throw new TelemetryConflictError("event_id is already used by a different event.");
    }
  }

  const eventCreated = await env.CLIPS.put(eventKey, candidateBody, {
    onlyIf: new Headers({ "If-None-Match": "*" }),
    httpMetadata: { contentType: "application/json; charset=utf-8" },
    customMetadata: telemetryMetadata(storedEvent, source, payloadSha256, receivedAt, env.ENVIRONMENT),
  });
  if (!eventCreated) {
    const existing = await env.CLIPS.get(eventKey);
    if (!existing) {
      throw new Error("Telemetry event conflicted but could not be read.");
    }
    const existingBody = await existing.text();
    const existingSha256 =
      existing.customMetadata?.payload_sha256 ?? await sha256Hex(existingBody);
    if (existingSha256 !== payloadSha256) {
      throw new TelemetryConflictError("event_id is already used by a different event.");
    }
    storedBody = existingBody;
    duplicate = true;
  }

  let outboxKey: string | null = null;
  let outboxBody: string | null = null;
  if (analyticsExportEnabled(env)) {
    outboxKey = analyticsOutboxKey(storedEvent);
    outboxBody = storedBody;
    await env.CLIPS.put(outboxKey, storedBody, {
      onlyIf: new Headers({ "If-None-Match": "*" }),
      httpMetadata: { contentType: "application/json; charset=utf-8" },
      customMetadata: telemetryMetadata(storedEvent, source, payloadSha256, receivedAt, env.ENVIRONMENT),
    });
  }

  logInfo("telemetry_event_persisted", {
    run_id: event.run_id,
    attempt_id: event.attempt_id,
    event_id: event.event_id,
    event_type: event.event_type,
    duplicate,
    aws_export_queued: outboxKey !== null,
  });

  return {
    event: storedEvent,
    event_key: eventKey,
    outbox_key: outboxKey,
    outbox_body: outboxBody,
    duplicate,
  };
}

export async function deliverAnalyticsOutboxItem(
  env: Env,
  outboxKey: string,
  providedBody?: string,
): Promise<boolean> {
  const ingestUrl = env.AWS_ANALYTICS_INGEST_URL.trim();
  const sharedSecret = env.AWS_ANALYTICS_SHARED_SECRET;
  if (!ingestUrl || !sharedSecret) {
    logError("analytics_export_not_configured", {
      outbox_key: outboxKey,
      ingest_url_configured: Boolean(ingestUrl),
      shared_secret_configured: Boolean(sharedSecret),
    });
    return false;
  }

  const receiptKey = analyticsDeliveredReceiptKey(outboxKey);
  if (await env.CLIPS.head(receiptKey)) {
    await env.CLIPS.delete(outboxKey);
    return true;
  }

  let body = providedBody;
  if (body === undefined) {
    const object = await env.CLIPS.get(outboxKey);
    if (!object) return true;
    if (object.size > MAX_OUTBOX_OBJECT_BYTES) {
      logError("analytics_outbox_item_too_large", {
        outbox_key: outboxKey,
        size_bytes: object.size,
      });
      return false;
    }
    body = await object.text();
  }

  try {
    const timestamp = String(Math.floor(Date.now() / 1000));
    const signature = await hmacSha256Hex(sharedSecret, `${timestamp}.${body}`);
    const response = await fetch(ingestUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json; charset=utf-8",
        "X-WDIRL-Timestamp": timestamp,
        "X-WDIRL-Signature": signature,
      },
      body,
      redirect: "manual",
    });

    if (response.body) await response.body.cancel();
    if (response.status < 200 || response.status >= 300) {
      logError("analytics_export_rejected", {
        outbox_key: outboxKey,
        http_status: response.status,
      });
      return false;
    }

    const deliveredAt = new Date().toISOString();
    await env.CLIPS.put(
      receiptKey,
      `${JSON.stringify({ version: 1, outbox_key: outboxKey, delivered_at: deliveredAt })}\n`,
      {
        onlyIf: new Headers({ "If-None-Match": "*" }),
        httpMetadata: { contentType: "application/json; charset=utf-8" },
        customMetadata: { delivered_at: deliveredAt },
      },
    );
    await env.CLIPS.delete(outboxKey);
    logInfo("analytics_export_delivered", {
      outbox_key: outboxKey,
      http_status: response.status,
    });
    return true;
  } catch (error) {
    logError("analytics_export_failed", {
      outbox_key: outboxKey,
      error: error instanceof Error ? error.message : "Unknown analytics export error",
    });
    return false;
  }
}

export async function reconcileAnalyticsOutbox(
  env: Env,
  scheduledTime: number,
): Promise<{ scanned: number; repaired: number }> {
  if (!analyticsExportEnabled(env)) return { scanned: 0, repaired: 0 };

  const objects = await listReconciliationBatch(env, scheduledTime);
  let repaired = 0;
  for (let offset = 0; offset < objects.length; offset += OUTBOX_CONCURRENCY) {
    const batch = objects.slice(offset, offset + OUTBOX_CONCURRENCY);
    const outcomes = await Promise.all(
      batch.map(async (object): Promise<boolean> => {
        try {
          if (object.size > MAX_OUTBOX_OBJECT_BYTES) return false;
          const stored = await env.CLIPS.get(object.key);
          if (!stored) return false;
          const body = await stored.text();
          const event = parseStoredTelemetryEvent(body);
          if (!event) {
            logError("telemetry_reconciliation_invalid_event", { event_key: object.key });
            return false;
          }
          const outboxKey = analyticsOutboxKey(event);
          const receiptKey = analyticsDeliveredReceiptKey(outboxKey);
          const [outbox, receipt] = await Promise.all([
            env.CLIPS.head(outboxKey),
            env.CLIPS.head(receiptKey),
          ]);
          if (outbox || receipt) return false;
          const payloadSha256 = await sha256Hex(body);
          const metadata = stored.customMetadata ?? telemetryMetadata(
            event,
            "processor",
            payloadSha256,
            new Date().toISOString(),
            env.ENVIRONMENT,
          );
          const created = await env.CLIPS.put(outboxKey, body, {
            onlyIf: new Headers({ "If-None-Match": "*" }),
            httpMetadata: { contentType: "application/json; charset=utf-8" },
            customMetadata: metadata,
          });
          return created !== null;
        } catch (error) {
          logError("telemetry_reconciliation_failed", {
            event_key: object.key,
            error: error instanceof Error ? error.message : "Unknown reconciliation error",
          });
          return false;
        }
      }),
    );
    repaired += outcomes.filter(Boolean).length;
  }

  logInfo("telemetry_reconciliation_completed", { scanned: objects.length, repaired });
  return { scanned: objects.length, repaired };
}

export async function retryAnalyticsOutbox(env: Env, scheduledTime: number): Promise<void> {
  if (!env.AWS_ANALYTICS_INGEST_URL.trim()) {
    logInfo("analytics_outbox_retry_skipped", { reason: "ingest_url_not_configured" });
    return;
  }
  if (!env.AWS_ANALYTICS_SHARED_SECRET) {
    logError("analytics_outbox_retry_skipped", { reason: "shared_secret_not_configured" });
    return;
  }

  const objects = await listOutboxRetryBatch(env, scheduledTime);
  let delivered = 0;
  let retained = 0;

  for (let offset = 0; offset < objects.length; offset += OUTBOX_CONCURRENCY) {
    const batch = objects.slice(offset, offset + OUTBOX_CONCURRENCY);
    const outcomes = await Promise.all(
      batch.map((object) => deliverAnalyticsOutboxItem(env, object.key)),
    );
    delivered += outcomes.filter(Boolean).length;
    retained += outcomes.filter((outcome) => !outcome).length;
  }

  logInfo("analytics_outbox_retry_completed", {
    selected: objects.length,
    delivered,
    retained,
  });
}

export async function readTelemetryTimeline(
  env: Env,
  runId: string,
  attemptId: string,
  limit: number,
  cursor?: string,
): Promise<TimelineResult> {
  const listed = await env.CLIPS.list({
    prefix: telemetryAttemptPrefix(runId, attemptId),
    limit,
    ...(cursor ? { cursor } : {}),
  });
  const events = await Promise.all(
    listed.objects.map(async (object): Promise<unknown | null> => {
      if (object.size > MAX_OUTBOX_OBJECT_BYTES) return null;
      const stored = await env.CLIPS.get(object.key);
      if (!stored) return null;
      try {
        return JSON.parse(await stored.text()) as unknown;
      } catch {
        logError("telemetry_event_unreadable", { event_key: object.key });
        return null;
      }
    }),
  );
  const presentEvents = events.filter((event): event is unknown => event !== null);
  presentEvents.sort(compareTimelineEvents);
  return {
    run_id: runId,
    attempt_id: attemptId,
    events: presentEvents,
    truncated: listed.truncated,
    cursor: listed.truncated ? listed.cursor : null,
  };
}

function validateTelemetryEvent(value: unknown, expectedRunId: string): ParseResult {
  const record = asRecord(value);
  if (!record) {
    return { ok: false, error: "Telemetry event must be a JSON object.", status: 400 };
  }

  const unknownKey = Object.keys(record).find((key) => !includes(EVENT_KEYS, key));
  if (unknownKey) {
    return { ok: false, error: `Unknown telemetry field: ${unknownKey}.`, status: 400 };
  }
  const missingKey = REQUIRED_EVENT_KEYS.find((key) => !(key in record));
  if (missingKey) {
    return { ok: false, error: `Telemetry event is missing ${missingKey}.`, status: 400 };
  }
  if (record.schema_version !== TELEMETRY_SCHEMA_VERSION) {
    return { ok: false, error: "Unsupported telemetry schema_version.", status: 400 };
  }

  const eventId = uuidIdentifier(record.event_id);
  const attemptId = uuidIdentifier(record.attempt_id);
  if (!eventId || !attemptId) {
    return { ok: false, error: "event_id and attempt_id must be UUIDs.", status: 400 };
  }
  if (record.run_id !== expectedRunId) {
    return { ok: false, error: "Telemetry run_id does not match the route.", status: 400 };
  }
  if (!isEventType(record.event_type)) {
    return { ok: false, error: "Unknown telemetry event_type.", status: 400 };
  }
  if (!isPositiveInteger(record.sequence, 1_000_000_000)) {
    return { ok: false, error: "Telemetry sequence must be a positive integer.", status: 400 };
  }
  if (!isEventTime(record.event_time)) {
    return { ok: false, error: "Telemetry event_time must be an RFC 3339 timestamp.", status: 400 };
  }

  const stage = record.stage === undefined || record.stage === null ? null : record.stage;
  const span = record.span === undefined || record.span === null ? null : record.span;
  if (stage !== null && !isPipelineStage(stage)) {
    return { ok: false, error: "Unknown telemetry stage.", status: 400 };
  }
  if (span !== null && !isProcessingSpan(span)) {
    return { ok: false, error: "Unknown telemetry span.", status: 400 };
  }
  if (includes(STAGE_EVENT_TYPES, record.event_type) && stage === null) {
    return { ok: false, error: "This telemetry event_type requires stage.", status: 400 };
  }
  if (includes(SPAN_EVENT_TYPES, record.event_type) && span === null) {
    return { ok: false, error: "This telemetry event_type requires span.", status: 400 };
  }
  if (span !== null && stage === null) {
    return { ok: false, error: "Telemetry span requires stage.", status: 400 };
  }

  const status = record.status;
  if (status !== undefined && !isEventStatus(status)) {
    return { ok: false, error: "Unknown telemetry status.", status: 400 };
  }
  const elapsedSeconds = record.elapsed_seconds;
  if (
    elapsedSeconds !== undefined &&
    (!isFiniteNumber(elapsedSeconds) || elapsedSeconds < 0 || elapsedSeconds > 31_536_000)
  ) {
    return { ok: false, error: "Telemetry elapsed_seconds is out of bounds.", status: 400 };
  }

  const nestedFields = [
    "progress",
    "input",
    "runtime",
    "resources",
    "measurements",
    "error",
    "attributes",
  ] as const;
  const nestedValues: Partial<Record<(typeof nestedFields)[number], BoundedJsonObject>> = {};
  for (const field of nestedFields) {
    if (record[field] === undefined) continue;
    const nested = validateBoundedObject(record[field], field);
    if (!nested.ok) return { ok: false, error: nested.error, status: 400 };
    nestedValues[field] = nested.value;
  }
  if (includes(FAILED_EVENT_TYPES, record.event_type) && !nestedValues.error) {
    return { ok: false, error: "Failed telemetry events require error metadata.", status: 400 };
  }

  const event: ProcessingTelemetryEvent = {
    schema_version: 1,
    event_id: eventId,
    run_id: expectedRunId,
    attempt_id: attemptId,
    sequence: record.sequence,
    event_type: record.event_type,
    event_time: record.event_time,
    ...(record.stage !== undefined ? { stage } : {}),
    ...(record.span !== undefined ? { span } : {}),
    ...(status !== undefined ? { status } : {}),
    ...(elapsedSeconds !== undefined ? { elapsed_seconds: elapsedSeconds } : {}),
    ...nestedValues,
  };
  return { ok: true, value: event };
}

function validateBoundedObject(
  value: unknown,
  field: string,
): { ok: true; value: BoundedJsonObject } | { ok: false; error: string } {
  const record = asRecord(value);
  if (!record) return { ok: false, error: `Telemetry ${field} must be an object.` };
  const budget = { nodes: 0 };
  const validated = validateBoundedRecord(record, field, 0, budget);
  return validated;
}

function validateBoundedRecord(
  record: Record<string, unknown>,
  field: string,
  depth: number,
  budget: { nodes: number },
): { ok: true; value: BoundedJsonObject } | { ok: false; error: string } {
  const entries = Object.entries(record);
  if (entries.length > MAX_METADATA_PROPERTIES) {
    return { ok: false, error: `Telemetry ${field} has too many properties.` };
  }
  const output: BoundedJsonObject = {};
  for (const [key, rawValue] of entries) {
    if (!METADATA_KEY_PATTERN.test(key)) {
      return { ok: false, error: `Telemetry ${field} has an invalid metadata key.` };
    }
    if (includes(SENSITIVE_METADATA_KEYS, key.toLowerCase())) {
      return { ok: false, error: `Telemetry ${field} contains a prohibited metadata key.` };
    }
    const validated = validateBoundedValue(rawValue, field, depth + 1, budget);
    if (!validated.ok) return validated;
    output[key] = validated.value;
  }
  return { ok: true, value: output };
}

function validateBoundedValue(
  value: unknown,
  field: string,
  depth: number,
  budget: { nodes: number },
): { ok: true; value: BoundedJsonValue } | { ok: false; error: string } {
  budget.nodes += 1;
  if (budget.nodes > 256) {
    return { ok: false, error: `Telemetry ${field} is too complex.` };
  }
  if (depth > 3) {
    return { ok: false, error: `Telemetry ${field} is nested too deeply.` };
  }
  if (value === null || typeof value === "boolean") return { ok: true, value };
  if (typeof value === "string") {
    if (value.length > 500) {
      return { ok: false, error: `Telemetry ${field} contains an oversized string.` };
    }
    if (SENSITIVE_VALUE_PATTERNS.some((pattern) => pattern.test(value))) {
      return { ok: false, error: `Telemetry ${field} contains a prohibited string value.` };
    }
    return { ok: true, value };
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value) || Math.abs(value) > 1_000_000_000_000_000) {
      return { ok: false, error: `Telemetry ${field} contains an invalid number.` };
    }
    return { ok: true, value };
  }
  if (Array.isArray(value)) {
    if (value.length > 32) {
      return { ok: false, error: `Telemetry ${field} contains an oversized array.` };
    }
    const output: BoundedJsonValue[] = [];
    for (const item of value) {
      const validated = validateBoundedValue(item, field, depth + 1, budget);
      if (!validated.ok) return validated;
      output.push(validated.value);
    }
    return { ok: true, value: output };
  }
  const record = asRecord(value);
  if (!record) {
    return { ok: false, error: `Telemetry ${field} contains an unsupported value.` };
  }
  return validateBoundedRecord(record, field, depth, budget);
}

async function readBoundedBody(
  body: ReadableStream<Uint8Array>,
  maxBytes: number,
): Promise<{ ok: true; value: Uint8Array } | { ok: false; error: string; status: 413 }> {
  const reader = body.getReader();
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    while (true) {
      const result = await reader.read();
      if (result.done) break;
      size += result.value.byteLength;
      if (size > maxBytes) {
        await reader.cancel("Telemetry event body is too large.");
        return { ok: false, error: "Telemetry event body is too large.", status: 413 };
      }
      chunks.push(result.value);
    }
  } finally {
    reader.releaseLock();
  }

  const value = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    value.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return { ok: true, value };
}

async function listOutboxRetryBatch(env: Env, scheduledTime: number): Promise<R2Object[]> {
  void scheduledTime;
  return listWithDurableCursor(env, OUTBOX_PREFIX, OUTBOX_BATCH_SIZE, OUTBOX_CURSOR_KEY);
}

async function listReconciliationBatch(env: Env, scheduledTime: number): Promise<R2Object[]> {
  void scheduledTime;
  return listWithDurableCursor(
    env,
    EVENT_PREFIX,
    RECONCILIATION_BATCH_SIZE,
    RECONCILIATION_CURSOR_KEY,
  );
}

async function listWithDurableCursor(
  env: Env,
  prefix: string,
  limit: number,
  cursorKey: string,
): Promise<R2Object[]> {
  const storedCursor = await env.CLIPS.get(cursorKey);
  const cursor = storedCursor && storedCursor.size <= 2048
    ? (await storedCursor.text()).trim()
    : "";
  const listed = await env.CLIPS.list({
    prefix,
    limit,
    ...(cursor ? { cursor } : {}),
  });
  if (listed.truncated) {
    await env.CLIPS.put(cursorKey, listed.cursor, {
      httpMetadata: { contentType: "text/plain; charset=utf-8" },
    });
  } else {
    await env.CLIPS.delete(cursorKey);
  }
  return listed.objects;
}

function parseStoredTelemetryEvent(body: string): ProcessingTelemetryEvent | null {
  try {
    const parsed: unknown = JSON.parse(body);
    const record = asRecord(parsed);
    if (!record || typeof record.run_id !== "string") return null;
    const validated = validateTelemetryEvent(record, record.run_id);
    return validated.ok ? validated.value : null;
  } catch {
    return null;
  }
}

function compareTimelineEvents(left: unknown, right: unknown): number {
  const leftRecord = asRecord(left);
  const rightRecord = asRecord(right);
  const leftSequence = leftRecord && typeof leftRecord.sequence === "number" ? leftRecord.sequence : 0;
  const rightSequence = rightRecord && typeof rightRecord.sequence === "number" ? rightRecord.sequence : 0;
  if (leftSequence !== rightSequence) return leftSequence - rightSequence;
  const leftTime = leftRecord && typeof leftRecord.event_time === "string" ? leftRecord.event_time : "";
  const rightTime = rightRecord && typeof rightRecord.event_time === "string" ? rightRecord.event_time : "";
  return leftTime.localeCompare(rightTime);
}

function telemetryMetadata(
  event: ProcessingTelemetryEvent,
  source: "processor" | "worker",
  payloadSha256: string,
  receivedAt: string,
  environment: string,
): Record<string, string> {
  return {
    schema_version: String(event.schema_version),
    run_id: event.run_id,
    attempt_id: event.attempt_id,
    event_id: event.event_id,
    event_type: event.event_type,
    source,
    payload_sha256: payloadSha256,
    received_at: receivedAt,
    environment: boundedEnvironment(environment),
  };
}

export function buildTelemetryEventKey(event: Pick<
  ProcessingTelemetryEvent,
  "run_id" | "attempt_id" | "sequence" | "event_id"
>): string {
  const sequence = String(event.sequence).padStart(10, "0");
  return `${telemetryAttemptPrefix(event.run_id, event.attempt_id)}${sequence}-${event.event_id}.json`;
}

function telemetryAttemptPrefix(runId: string, attemptId: string): string {
  return `${EVENT_PREFIX}${runId}/${attemptId}/`;
}

function telemetryEventIdIndexKey(event: ProcessingTelemetryEvent): string {
  return `${EVENT_ID_INDEX_PREFIX}${event.run_id}/${event.attempt_id}/${event.event_id}.json`;
}

function parseEventIdIndex(
  body: string,
): { event_key: string; payload_sha256: string } | null {
  try {
    const parsed: unknown = JSON.parse(body);
    const record = asRecord(parsed);
    if (
      !record ||
      record.version !== 1 ||
      typeof record.event_key !== "string" ||
      typeof record.payload_sha256 !== "string" ||
      !/^[a-f0-9]{64}$/.test(record.payload_sha256)
    ) {
      return null;
    }
    return { event_key: record.event_key, payload_sha256: record.payload_sha256 };
  } catch {
    return null;
  }
}

function analyticsOutboxKey(event: ProcessingTelemetryEvent): string {
  return `${OUTBOX_PREFIX}${event.run_id}/${event.attempt_id}/${event.event_id}.json`;
}

function analyticsDeliveredReceiptKey(outboxKey: string): string {
  if (!outboxKey.startsWith(OUTBOX_PREFIX)) {
    throw new Error("Invalid analytics outbox key.");
  }
  return `${DELIVERED_PREFIX}${outboxKey.slice(OUTBOX_PREFIX.length)}`;
}

function analyticsExportEnabled(env: Env): boolean {
  return Boolean(env.AWS_ANALYTICS_INGEST_URL.trim());
}

function boundedEnvironment(value: string): string {
  const trimmed = value.trim();
  return trimmed.slice(0, 64) || "unknown";
}

async function hmacSha256Hex(secret: string, message: string): Promise<string> {
  const encoder = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw",
    encoder.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const signature = await crypto.subtle.sign("HMAC", key, encoder.encode(message));
  return bytesToHex(new Uint8Array(signature));
}

async function sha256Hex(value: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
  return bytesToHex(new Uint8Array(digest));
}

function bytesToHex(bytes: Uint8Array): string {
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function parseContentLength(request: Request): number | null {
  const raw = request.headers.get("content-length");
  if (!raw) return null;
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : null;
}

function uuidIdentifier(value: unknown): string | null {
  return typeof value === "string" && UUID_PATTERN.test(value) ? value : null;
}

function isPositiveInteger(value: unknown, maximum: number): value is number {
  return Number.isInteger(value) && typeof value === "number" && value >= 1 && value <= maximum;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isEventTime(value: unknown): value is string {
  if (typeof value !== "string" || value.length > 40) return false;
  if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|[+-]\d{2}:\d{2})$/.test(value)) {
    return false;
  }
  return Number.isFinite(Date.parse(value));
}

function isEventType(value: unknown): value is EventType {
  return includes(EVENT_TYPES, value);
}

function isPipelineStage(value: unknown): value is PipelineStage {
  return includes(PIPELINE_STAGES, value);
}

function isProcessingSpan(value: unknown): value is ProcessingSpan {
  return includes(PROCESSING_SPANS, value);
}

function isEventStatus(value: unknown): value is EventStatus {
  return includes(EVENT_STATUSES, value);
}

function includes<const Values extends readonly unknown[]>(
  values: Values,
  candidate: unknown,
): candidate is Values[number] {
  return values.includes(candidate);
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function logInfo(message: string, fields: Record<string, unknown>): void {
  console.log(JSON.stringify({ level: "info", message, ...fields }));
}

function logError(message: string, fields: Record<string, unknown>): void {
  console.error(JSON.stringify({ level: "error", message, ...fields }));
}
