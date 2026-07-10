# Processing Analytics Infrastructure

This AWS CDK stack is the downstream analytics adapter for hosted Processing Attempts. Cloudflare Worker/R2 remains the authoritative operational record and RunPod remains the Processor; this stack receives metadata events only.

All three layers use `schemas/processing-event-v1.schema.json` as the versioned wire contract.

## Deployed Shape

```text
Cloudflare Worker
  -> HMAC-authenticated API Gateway endpoint
  -> ingestion Lambda (authentication + schema validation)
  -> FIFO SQS queue (attempt ordering + event-id deduplication)
  -> consumer Lambda (validation + flattening)
  -> S3 raw and validated event zones
  -> Glue tables + Athena workgroup and named queries
  -> scheduled daily Parquet aggregate
  -> private CloudWatch operations dashboard

Private dashboard Cloudflare Worker
  -> dedicated HMAC-authenticated API Gateway endpoint
  -> allowlisted asynchronous query Lambda
  -> dedicated low-scan Athena workgroup
  -> S3 dashboard result prefix
```

The queue has a FIFO dead-letter queue and a five-attempt redrive policy. Its 210-second visibility timeout exceeds six times the 30-second consumer timeout. Lambda reports partial batch failures so one invalid event does not replay an otherwise valid batch.

## Data Zones and Retention

| Prefix | Format | Purpose | Retention |
|---|---|---|---|
| `raw/` | gzip NDJSON | Original accepted events plus ingestion metadata | 90 days |
| `validated/` facts | gzip NDJSON | Flattened attempt/stage/span boundaries | 365 days |
| `validated/` progress | gzip NDJSON | Flattened five-second progress samples, S3-tagged separately | 90 days |
| `aggregate/stage_daily/` | Snappy Parquet | Daily stage/span percentiles and failures | Indefinite |
| `athena-results/` | Athena output | Query results | 30 days |
| `dashboard-results/` | Athena output | Private dashboard query results | 7 days |

The scheduled aggregate is idempotent by UTC date. It clears an incomplete date prefix, runs an Athena `UNLOAD` into Parquet, and writes `_SUCCESS` only after Athena succeeds. A second schedule rebuilds the three- and seven-day-old partitions so events delayed by Worker outbox retries are not permanently omitted from long-term history.

Source clips, rendered videos, pose/keypoint files, prompts, filenames, URLs, and raw exception text must never enter this stack.

## Local Validation

Requirements:

- Node.js 24 or newer
- Python 3.9 or newer for unit tests; Lambda deploys on Python 3.12
- AWS credentials only for deployment

```bash
cd infra/analytics
npm install
npm test
npm run synth
```

`npm test` covers event-contract rejection, HMAC authentication, FIFO metadata, partial batch failure, gzip/partition output, and daily aggregate idempotency.

## Deploy

No AWS account credentials are stored in this repository. The production stack is `WhoDoIRunLikeAnalytics` in `us-east-1`.

### One-time deployment identity

For a single-developer account without IAM Identity Center, [deployment-role.yaml](bootstrap/deployment-role.yaml) creates a no-console caller and an assumable CDK deployment role. The caller can only assume that role; the role has administrator access because it must bootstrap CDK and let CloudFormation create the stack's scoped runtime roles. Use an existing federated administrator instead when one is available.

Deploy the access stack once from an account administrator, create one access key for `WhoDoIRunLikeDeploymentCaller`, and enter it with `aws configure --profile wdirl-analytics-caller` without printing it into shell history. Then configure the role profile:

```bash
aws cloudformation deploy \
  --stack-name WhoDoIRunLikeDeploymentAccess \
  --template-file infra/analytics/bootstrap/deployment-role.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --region us-east-1

ROLE_ARN=$(aws cloudformation describe-stacks \
  --stack-name WhoDoIRunLikeDeploymentAccess \
  --region us-east-1 \
  --query "Stacks[0].Outputs[?OutputKey=='DeploymentRoleArn'].OutputValue | [0]" \
  --output text)
aws configure set role_arn "$ROLE_ARN" --profile wdirl-analytics-deploy
aws configure set source_profile wdirl-analytics-caller --profile wdirl-analytics-deploy
aws configure set region us-east-1 --profile wdirl-analytics-deploy
```

### CDK stack

```bash
cd infra/analytics
npx cdk bootstrap aws://<account-id>/us-east-1 --profile wdirl-analytics-deploy
npx cdk deploy WhoDoIRunLikeAnalytics \
  --profile wdirl-analytics-deploy \
  --require-approval broadening
```

The deploy outputs include:

- `TelemetryIngestUrl`
- `TelemetrySecretArn`
- `EventLakeBucket`
- `AthenaDatabase`
- `AthenaWorkGroup`
- `DashboardApiUrl`
- `DashboardSecretArn`
- `DashboardAthenaWorkGroup`
- `PrivateOperationsDashboard`

Retrieve the generated HMAC secret without printing it into shell history, then set it as a Cloudflare Worker secret from the repository root:

```bash
cd worker
aws secretsmanager get-secret-value \
  --secret-id <TelemetrySecretArn> \
  --profile wdirl-analytics-deploy \
  --query SecretString \
  --output text | npx wrangler secret put AWS_ANALYTICS_SHARED_SECRET --env=""
```

Set the non-secret `AWS_ANALYTICS_INGEST_URL` Worker variable to the `TelemetryIngestUrl` output and redeploy the Worker. Do not put the HMAC secret in `wrangler.jsonc`, `.dev.vars.example`, GitHub Actions output, or RunPod environment variables.

Retrieve the separate dashboard API secret directly into the private dashboard Worker. It must not be shared with the public application Worker or delivered to browser JavaScript:

```bash
cd analytics-dashboard
aws secretsmanager get-secret-value \
  --secret-id <DashboardSecretArn> \
  --profile wdirl-analytics-deploy \
  --query SecretString \
  --output text | npx wrangler secret put AWS_DASHBOARD_SHARED_SECRET
```

Store `DashboardApiUrl` as the dashboard Worker's `AWS_DASHBOARD_API_URL` secret or server-only variable. The Worker applies operator authentication, signs upstream requests, and proxies the JSON response. The browser must never receive the AWS endpoint or shared secret.

## Authentication Contract

The Worker sends the exact JSON event bytes with:

```text
X-WDIRL-Timestamp: <unix epoch seconds>
X-WDIRL-Signature: <lowercase hex HMAC-SHA256>
```

The signed message is:

```text
<timestamp>.<raw JSON body>
```

Ingestion rejects requests more than five minutes from AWS time, invalid signatures, unknown fields, invalid stage/span names, non-finite numbers, missing error classification on failure events, and payloads above 64 KiB. SQS uses `attempt_id` as the FIFO message group and `event_id` as the deduplication ID.

### Dashboard query API

The dashboard uses an independent secret and headers:

```text
X-WDIRL-Dashboard-Timestamp: <unix epoch seconds>
X-WDIRL-Dashboard-Signature: <lowercase hex HMAC-SHA256>
```

The exact signed bytes are:

```text
<timestamp>\n<METHOD>\n<CANONICAL_PATH>\n<exact request body bytes>
```

`METHOD` is uppercase. `CANONICAL_PATH` excludes the API Gateway stage and is exactly `/queries` or `/queries/<query_execution_id>`. `POST` signs the exact compact JSON bytes sent upstream; `GET` signs a zero-byte body. Requests outside the five-minute clock-skew window are rejected. Do not parse and re-serialize the body between signing and sending.

The POST signature is also folded into Athena's idempotency token. Replaying the same signed request inside the clock-skew window returns the original query execution instead of starting another scan.

The two routes are:

```http
POST /v1/queries
Content-Type: application/json

{"query":"overview","filters":{"range_days":30}}
```

Successful submission returns HTTP 202:

```json
{
  "query": "overview",
  "query_execution_id": "2f47c767-1b08-4a72-95a1-bd5da365fe60",
  "state": "QUEUED",
  "poll_after_ms": 1500
}
```

Poll `GET /v1/queries/<query_execution_id>`. A queued or running response includes `state`, cost/timing `statistics`, and `poll_after_ms`. A successful response adds `rows` and `row_count`. Failed or cancelled responses include only a sanitized error category/type; Athena SQL, S3 paths, and state-change text are never returned.

The query IDs and accepted filters are fixed:

| Query | Purpose | Additional filters |
|---|---|---|
| `overview` | Attempt count, Result Ready p50/p90/p95, failure rate, bottleneck | common filters |
| `stage_latency` | Stage p50/p90/p95, samples, failures, confidence | `stage` |
| `span_latency` | Per-attempt nested-span p50/p90/p95 | `stage`, `span` |
| `attempts` | Recent attempt list and workload/latency values | `status`, `limit` |
| `attempt_detail` | Ordered stage/span waterfall for one attempt | `attempt_id` |
| `failures` | Sanitized error groups and affected attempts | `stage` |
| `stalls` | Non-terminal processing attempts past a threshold | `stale_minutes` |
| `freshness` | Latest event/ingest times and 24-hour counts | none |

Common filters are `range_days`, `environment`, `backend`, `gpu_type`, `processor_version`, `duration_bucket`, `resolution_bucket`, and `cold_start`. At attempt/cohort level, `backend` explicitly means the `runner_mask` backend; pose, tracking, and DensePose backends remain event-level attributes and are not collapsed into that selector. Unknown query names, fields, values, and raw SQL are rejected before Athena is called. Range, row-count, stage/span, UUID, and string values are bounded by the Lambda contract.

## Athena

The `whodoirunlike-processing` workgroup enforces encrypted S3 results and rejects a query after 1 GiB scanned. Included named queries cover:

- stage p50/p90/p95 and average milliseconds per frame
- nested span p50/p90/p95
- Result Ready latency split by cold start and GPU
- upload, enqueue, RunPod queue, and model-load latency
- one-row-per-attempt stage breakdowns for waterfall charts
- Result Ready versus Analysis Complete timing
- attempts with no terminal event and no telemetry for ten minutes
- failures by stage, span, class, and code
- daily aggregate source and indefinite Parquet history

Event IDs are deduplicated in queries because SQS/Lambda and S3 delivery are intentionally at-least-once.

The private UI does not execute the named-query workgroup directly. It uses the separate `whodoirunlike-dashboard` workgroup, which enforces Athena engine v3, a 256 MiB scan cutoff, SSE-S3 results, one-minute result reuse, and the `dashboard-results/` prefix. The Lambda chooses one of eight compiled query contracts; clients cannot submit SQL. There is no QuickSight dependency.

## Private Operations Dashboard

The stack always creates a private CloudWatch operations dashboard for accepted/rejected ingestion, event-lake write failures, queue depth, oldest-message age, and dead letters. These log-derived metrics cover handled Lambda failures that would otherwise leave the standard Lambda `Errors` metric at zero.

Business-performance analysis remains Athena-first. Use the included named queries for Result Ready p50/p90/p95, stage and span waterfalls, cold-versus-warm latency, milliseconds per frame, failure rate, and GPU/backend comparisons.

## Security and Failure Behavior

- API Gateway has no data tracing, so event bodies are not written to access logs.
- The HMAC secret is generated and retained in Secrets Manager.
- Dashboard queries use a second retained HMAC secret; compromise does not grant telemetry-ingest access.
- Lambda roles are scoped to their queue, secret, workgroup, and required S3 prefixes.
- The dashboard query role can read only the validated event prefix and its own result prefix. It cannot read raw events, write validated events, or delete event-lake objects.
- Buckets and queues require TLS and use service-managed encryption.
- S3 data and the HMAC secret are retained if the CDK stack is deleted.
- Retained resources use `RetainExceptOnCreate`, so a failed initial stack creation does not leave orphaned data resources.
- Analytics delivery is downstream and must never change the outcome of clip processing.
