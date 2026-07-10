from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import time
from typing import Any

from event_contract import ContractError, validate_event


LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)
SIGNATURE_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_secret_cache: str | None = None
_secrets_client: Any = None
_sqs_client: Any = None


def _client(service: str) -> Any:
    import boto3

    return boto3.client(service)


def _secret() -> str:
    global _secret_cache, _secrets_client
    if _secret_cache is not None:
        return _secret_cache
    if _secrets_client is None:
        _secrets_client = _client("secretsmanager")
    response = _secrets_client.get_secret_value(SecretId=os.environ["INGEST_SECRET_ARN"])
    secret = response.get("SecretString")
    if not isinstance(secret, str) or not secret:
        raise RuntimeError("ingest secret is unavailable")
    _secret_cache = secret
    return secret


def _queue() -> Any:
    global _sqs_client
    if _sqs_client is None:
        _sqs_client = _client("sqs")
    return _sqs_client


def _headers(event: dict[str, Any]) -> dict[str, str]:
    return {str(key).lower(): str(value) for key, value in (event.get("headers") or {}).items()}


def _raw_body(event: dict[str, Any]) -> bytes:
    body = event.get("body")
    if not isinstance(body, str):
        raise ContractError("request body is required")
    return base64.b64decode(body) if event.get("isBase64Encoded") else body.encode("utf-8")


def _verify_request(headers: dict[str, str], body: bytes, *, now: float | None = None) -> None:
    timestamp = headers.get("x-wdirl-timestamp", "")
    signature = headers.get("x-wdirl-signature", "")
    if not timestamp.isdigit() or not SIGNATURE_PATTERN.fullmatch(signature):
        raise PermissionError("missing or malformed telemetry signature")
    current = time.time() if now is None else now
    max_skew = int(os.getenv("MAX_CLOCK_SKEW_SECONDS", "300"))
    if abs(current - int(timestamp)) > max_skew:
        raise PermissionError("telemetry signature timestamp is stale")
    signed = timestamp.encode("ascii") + b"." + body
    expected = hmac.new(_secret().encode("utf-8"), signed, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise PermissionError("invalid telemetry signature")


def _response(status: int, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json; charset=utf-8"},
        "body": json.dumps(payload, separators=(",", ":")),
    }


def handler(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    try:
        body = _raw_body(event)
        if len(body) > int(os.getenv("MAX_EVENT_BYTES", "65536")):
            return _response(413, {"error": "event is too large"})
        _verify_request(_headers(event), body)
        payload = validate_event(json.loads(body))
        _queue().send_message(
            QueueUrl=os.environ["TELEMETRY_QUEUE_URL"],
            MessageBody=json.dumps(payload, allow_nan=False, separators=(",", ":")),
            MessageGroupId=payload["attempt_id"],
            MessageDeduplicationId=payload["event_id"],
            MessageAttributes={
                "schema_version": {"DataType": "Number", "StringValue": str(payload["schema_version"])},
                "event_type": {"DataType": "String", "StringValue": payload["event_type"]},
            },
        )
        LOGGER.info(
            json.dumps(
                {
                    "message": "telemetry event accepted",
                    "event_id": payload["event_id"],
                    "run_id": payload["run_id"],
                    "attempt_id": payload["attempt_id"],
                    "event_type": payload["event_type"],
                }
            )
        )
        return _response(202, {"accepted": True, "event_id": payload["event_id"]})
    except PermissionError as exc:
        LOGGER.warning(json.dumps({"message": "telemetry authentication rejected", "reason": str(exc)}))
        return _response(401, {"error": "unauthorized"})
    except (ContractError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        LOGGER.warning(json.dumps({"message": "telemetry event rejected", "reason": str(exc)}))
        return _response(400, {"error": str(exc)})
    except Exception:
        LOGGER.exception(json.dumps({"message": "telemetry ingestion failed"}))
        return _response(500, {"error": "ingestion unavailable"})
