from __future__ import annotations

import gzip
import json
import logging
import os
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from event_contract import ContractError, event_partition, flatten_event, validate_event


LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)
_s3_client: Any = None


def _client(service: str) -> Any:
    import boto3

    return boto3.client(service)


def _s3() -> Any:
    global _s3_client
    if _s3_client is None:
        _s3_client = _client("s3")
    return _s3_client


def _gzip_jsonl(rows: list[dict[str, Any]]) -> bytes:
    payload = "".join(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n" for row in rows)
    return gzip.compress(payload.encode("utf-8"), compresslevel=6, mtime=0)


def _put_rows(
    *,
    zone: str,
    event_date: str,
    event_hour: str,
    rows: list[dict[str, Any]],
    request_id: str,
    event_class: str | None = None,
) -> None:
    key = (
        f"{zone}/event_date={event_date}/event_hour={event_hour}/"
        f"{request_id}-{uuid.uuid4().hex}.jsonl.gz"
    )
    request: dict[str, Any] = {
        "Bucket": os.environ["EVENT_BUCKET"],
        "Key": key,
        "Body": _gzip_jsonl(rows),
        "ContentType": "application/x-ndjson",
        "ContentEncoding": "gzip",
        "ServerSideEncryption": "AES256",
    }
    if event_class:
        request["Tagging"] = f"event-class={event_class}"
    _s3().put_object(
        **request,
    )


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    ingested_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    grouped: dict[tuple[str, str], list[tuple[str, dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    failed_ids: set[str] = set()

    for record in event.get("Records") or []:
        message_id = str(record.get("messageId") or "")
        try:
            payload = validate_event(json.loads(record["body"]))
            event_date, event_hour = event_partition(payload["event_time"])
            raw = {**payload, "ingested_at": ingested_at, "sqs_message_id": message_id}
            grouped[(event_date, event_hour)].append(
                (message_id, raw, flatten_event(payload, ingested_at=ingested_at))
            )
        except (KeyError, TypeError, ContractError, json.JSONDecodeError) as exc:
            failed_ids.add(message_id)
            LOGGER.warning(
                json.dumps(
                    {"message": "telemetry consumer validation failed", "message_id": message_id, "reason": str(exc)}
                )
            )

    request_id = str(getattr(context, "aws_request_id", "local"))
    for (event_date, event_hour), group in grouped.items():
        message_ids = {item[0] for item in group}
        try:
            _put_rows(
                zone="raw",
                event_date=event_date,
                event_hour=event_hour,
                rows=[item[1] for item in group],
                request_id=request_id,
            )
            facts = [item[2] for item in group if item[1]["event_type"] != "progress_sampled"]
            progress = [item[2] for item in group if item[1]["event_type"] == "progress_sampled"]
            if facts:
                _put_rows(
                    zone="validated",
                    event_date=event_date,
                    event_hour=event_hour,
                    rows=facts,
                    request_id=request_id,
                    event_class="fact",
                )
            if progress:
                _put_rows(
                    zone="validated",
                    event_date=event_date,
                    event_hour=event_hour,
                    rows=progress,
                    request_id=request_id,
                    event_class="progress",
                )
        except Exception:
            failed_ids.update(message_ids)
            LOGGER.exception(
                json.dumps(
                    {
                        "message": "telemetry event-lake write failed",
                        "event_date": event_date,
                        "event_hour": event_hour,
                        "message_count": len(group),
                    }
                )
            )

    grouped_message_ids = {item[0] for group in grouped.values() for item in group if item[0]}
    stored = len(grouped_message_ids - failed_ids)
    LOGGER.info(
        json.dumps(
            {
                "message": "telemetry consumer batch complete",
                "stored_messages": stored,
                "failed_messages": len(failed_ids),
            }
        )
    )
    return {
        "batchItemFailures": [
            {"itemIdentifier": message_id} for message_id in sorted(failed_ids) if message_id
        ]
    }
