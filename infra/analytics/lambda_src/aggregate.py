from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any


LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)
_athena_client: Any = None
_s3_client: Any = None


def _client(service: str) -> Any:
    import boto3

    return boto3.client(service)


def _athena() -> Any:
    global _athena_client
    if _athena_client is None:
        _athena_client = _client("athena")
    return _athena_client


def _s3() -> Any:
    global _s3_client
    if _s3_client is None:
        _s3_client = _client("s3")
    return _s3_client


def _target_date(event: dict[str, Any]) -> str:
    supplied = event.get("date")
    if supplied:
        return datetime.strptime(str(supplied), "%Y-%m-%d").date().isoformat()
    days_ago = int(event.get("days_ago") or 1)
    if not 1 <= days_ago <= 30:
        raise ValueError("days_ago must be between 1 and 30")
    return (datetime.now(timezone.utc).date() - timedelta(days=days_ago)).isoformat()


def _clear_prefix(bucket: str, prefix: str) -> None:
    continuation: str | None = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if continuation:
            kwargs["ContinuationToken"] = continuation
        response = _s3().list_objects_v2(**kwargs)
        objects = [{"Key": row["Key"]} for row in response.get("Contents") or []]
        if objects:
            _s3().delete_objects(Bucket=bucket, Delete={"Objects": objects, "Quiet": True})
        if not response.get("IsTruncated"):
            return
        continuation = response.get("NextContinuationToken")


def _query(target_date: str, bucket: str, prefix: str) -> str:
    database = os.environ["ATHENA_DATABASE"]
    table = os.environ["ATHENA_SOURCE_TABLE"]
    return f"""
UNLOAD (
  WITH ranked AS (
    SELECT *, row_number() OVER (PARTITION BY event_id ORDER BY ingested_at DESC) AS event_rank
    FROM {database}.{table}
    WHERE event_date = '{target_date}'
  ), events AS (
    SELECT * FROM ranked WHERE event_rank = 1
  )
  SELECT event_type, stage, span, backend, gpu_type, cold_start,
         duration_bucket, resolution_bucket,
         count(*) AS samples,
         approx_percentile(elapsed_seconds, 0.50) AS p50_seconds,
         approx_percentile(elapsed_seconds, 0.90) AS p90_seconds,
         approx_percentile(elapsed_seconds, 0.95) AS p95_seconds,
         avg(elapsed_seconds) AS average_seconds,
         sum(CASE WHEN event_type IN ('span_failed', 'stage_failed', 'attempt_failed') THEN 1 ELSE 0 END) AS failures
  FROM events
  WHERE event_type IN (
    'stage_completed', 'stage_failed', 'span_completed', 'span_failed',
    'result_ready', 'analysis_completed', 'attempt_completed', 'attempt_failed'
  )
  GROUP BY event_type, stage, span, backend, gpu_type, cold_start,
           duration_bucket, resolution_bucket
)
TO 's3://{bucket}/{prefix}'
WITH (format = 'PARQUET', compression = 'SNAPPY')
""".strip()


def _wait_for_query(query_id: str, *, timeout_seconds: int = 100) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        execution = _athena().get_query_execution(QueryExecutionId=query_id)["QueryExecution"]
        state = execution["Status"]["State"]
        if state == "SUCCEEDED":
            return
        if state in {"FAILED", "CANCELLED"}:
            reason = execution["Status"].get("StateChangeReason") or state
            raise RuntimeError(f"Athena aggregate query {state.lower()}: {reason}")
        time.sleep(2)
    _athena().stop_query_execution(QueryExecutionId=query_id)
    raise TimeoutError("Athena aggregate query timed out")


def handler(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    target_date = _target_date(event)
    bucket = os.environ["EVENT_BUCKET"]
    prefix = f"aggregate/stage_daily/event_date={target_date}/"
    marker = f"{prefix}_SUCCESS"
    force = bool(event.get("force"))
    if not force:
        try:
            _s3().head_object(Bucket=bucket, Key=marker)
            return {"status": "already_complete", "event_date": target_date}
        except Exception as exc:
            error_code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if error_code not in {"404", "NoSuchKey", "NotFound"}:
                raise

    _clear_prefix(bucket, prefix)
    response = _athena().start_query_execution(
        QueryString=_query(target_date, bucket, prefix),
        QueryExecutionContext={"Database": os.environ["ATHENA_DATABASE"]},
        WorkGroup=os.environ["ATHENA_WORKGROUP"],
    )
    query_id = response["QueryExecutionId"]
    _wait_for_query(query_id)
    _s3().put_object(
        Bucket=bucket,
        Key=marker,
        Body=b"",
        ContentType="application/octet-stream",
        ServerSideEncryption="AES256",
    )
    LOGGER.info(
        json.dumps(
            {
                "message": "daily processing aggregate complete",
                "event_date": target_date,
                "query_execution_id": query_id,
            }
        )
    )
    return {"status": "complete", "event_date": target_date, "query_execution_id": query_id}
