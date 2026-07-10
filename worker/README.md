# Who Do I Run Like Worker

Cloudflare Worker API for the public upload flow.

The Worker coordinates the hosted processing path:

- stores volunteer clips in R2
- keeps a JSON job record in R2
- hands the job to RunPod Serverless when `RUNPOD_ENDPOINT_ID` and `RUNPOD_API_KEY` are configured
- serves completed artifacts
- assigns a new Processing Attempt ID for every enqueue or retry
- persists authenticated Processor telemetry as immutable R2 objects
- asynchronously exports telemetry through an R2 outbox when the AWS analytics adapter is configured

It does not run SAM, DensePose, identity tracking, or video encoding. Those stay in the RunPod processor.

The RunPod payload includes `attempt_id`, `attempt_number`, `attempt_started_at`, `processor_enqueued_at`, and `telemetry_sequence_start`. It never includes the AWS analytics shared secret.

The Worker owns telemetry sequence numbers 1–99: `attempt_started` is 1, initial-upload `source_ingest` start/completion are 2/3, `processor_enqueue` start/completion (or failure) are 4/5, `processor_queue` start is 6, and its authoritative completion is 7. RunPod queue duration uses the provider's `delayTime`; the direct-processor fallback is explicitly labeled as an estimate. The Processor begins its analysis events at `telemetry_sequence_start` 100 with `source_download`. The initial attempt ID is created with the upload and reused by its first `/start`; restarting a failed job creates the next attempt ID and attempt number (with upload events omitted for that retry).

RunPod must use asynchronous `/run`. `RUNPOD_RUNSYNC=1` is rejected because a synchronous response would arrive after Processor telemetry and break lifecycle ordering.

## Local setup

```bash
cd worker
npm ci
cp .dev.vars.example .dev.vars
npm run dev
```

Set the same `PROCESSOR_SHARED_SECRET` in both `worker/.dev.vars` and the RunPod processor environment.

The production `AWS_ANALYTICS_INGEST_URL` is checked into `wrangler.jsonc` because it is not secret. `.dev.vars.example` overrides it with an empty value so local runs do not send test events to production AWS. To exercise the outbox locally, set a test URL in `.dev.vars` and replace the `AWS_ANALYTICS_SHARED_SECRET` placeholder. Test the scheduled retry handler with:

```bash
npm run dev -- --test-scheduled
curl http://localhost:8787/cdn-cgi/handler/scheduled
```

## Processing telemetry

The Processor submits one direct telemetry v1 JSON object at a time:

```text
POST /v1/jobs/:run_id/events
Authorization: Bearer $PROCESSOR_SHARED_SECRET
Content-Type: application/json
```

The required fields are `schema_version`, `event_id`, `run_id`, `attempt_id`, `sequence`, `event_type`, and `event_time`. The optional fields are `stage`, `span`, `status`, `elapsed_seconds`, `progress`, `input`, `runtime`, `resources`, `measurements`, sanitized `error` metadata, and bounded `attributes`. Bodies are limited to 64 KiB, unknown top-level fields and non-canonical stage/span names are rejected, nested metadata is bounded, and URL/path/secret-bearing keys are prohibited. The repository-wide contract is [processing-event-v1.schema.json](../schemas/processing-event-v1.schema.json).

Worker lifecycle events use sequences 1–7. A failed enqueue uses sequence 5 for the stage failure and sequence 6 for the terminal attempt failure. Successful attempts use sequence 6 for queue start and sequence 7 for its idempotent completion; Processor analysis starts at sequence 100.

Every artifact upload must include `X-Processing-Attempt-Id`, and every report body must include `attempt_id`. Stale attempts receive HTTP 409. Artifact objects are isolated at `artifacts/{run_id}/{attempt_id}/{artifact_name}`, while only the current attempt may update the public job record.

Each accepted event is first written once under `telemetry/v1/events/`, ordered by zero-padded sequence, with a separate immutable event-ID index for idempotency. If `AWS_ANALYTICS_INGEST_URL` is set, the identical event is then placed under `telemetry/v1/outbox/`. Delivery runs with `ctx.waitUntil()`, and the five-minute cron reconciles missing outbox items before retrying retained items. After a 2xx response it writes an immutable delivered receipt before deleting the outbox object, preventing reconciliation from exporting the same event forever.

The deployed RunPod template gives durable Worker mutations enough time to finish: `WHODOIRUNLIKE_TELEMETRY_DELIVERY_TIMEOUT_SECONDS=10`, `WHODOIRUNLIKE_REPORT_TIMEOUT_SECONDS=10`, and `WHODOIRUNLIKE_TELEMETRY_DRAIN_TIMEOUT_SECONDS=180`. The code carries the same defaults. Production measurements showed that serial Worker persistence could leave roughly 90 otherwise-successful events queued at the end of a five-minute processing run, so the final drain keeps the RunPod invocation alive for up to three minutes to preserve terminal telemetry. That is a bounded correctness safeguard with a billed-time cost; batching or parallel delivery should replace it. If the deadline is exhausted, the Processor writes a sanitized error containing the run and attempt IDs plus delivery counters, never event bodies, callback URLs, headers, or secrets.

AWS requests contain the exact JSON body with these headers:

- `X-WDIRL-Timestamp`: epoch seconds
- `X-WDIRL-Signature`: lowercase hex HMAC-SHA256 of `${timestamp}.${body}`

Authenticated operators can retrieve a bounded timeline with `GET /v1/jobs/:run_id/events?attempt_id=...`. Use the returned cursor when `truncated` is true.

## Telemetry retention

Apply 90-day R2 lifecycle rules to the raw operational telemetry prefixes after creating each bucket:

```bash
npx wrangler r2 bucket lifecycle add whodoirunlike-clips telemetry-events-90d telemetry/v1/events/ --expire-days 90
npx wrangler r2 bucket lifecycle add whodoirunlike-clips telemetry-event-index-90d telemetry/v1/event-ids/ --expire-days 90
npx wrangler r2 bucket lifecycle add whodoirunlike-clips telemetry-outbox-90d telemetry/v1/outbox/ --expire-days 90
npx wrangler r2 bucket lifecycle add whodoirunlike-clips telemetry-delivered-90d telemetry/v1/delivered/ --expire-days 90
```

Repeat those rules for `whodoirunlike-clips-preview`, then verify both buckets with `npx wrangler r2 bucket lifecycle list <bucket>`. The AWS adapter independently retains validated attempt/stage/span facts for 365 days and daily aggregates indefinitely.

## Deploy

Create the R2 buckets once:

```bash
npx wrangler r2 bucket create whodoirunlike-clips
npx wrangler r2 bucket create whodoirunlike-clips-preview
```

Set the processor secret:

```bash
npx wrangler secret put PROCESSOR_SHARED_SECRET --env=""
npx wrangler secret put RUNPOD_API_KEY --env=""
npx wrangler secret put AWS_ANALYTICS_SHARED_SECRET --env=""
```

Set `AWS_ANALYTICS_INGEST_URL` in `wrangler.jsonc` only after the analytics adapter is deployed. Leaving it empty disables outbox creation without affecting clip processing.

Deploy:

```bash
npm run deploy
```

Run the Worker runtime tests and deployment dry-run with `npm run check`.

The production route is `api.whodoirunlike.com`.
