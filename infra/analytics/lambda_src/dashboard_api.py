from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import logging
import os
import re
import time
from decimal import Decimal, InvalidOperation
from typing import Any

from dashboard_queries import (
    QueryContractError,
    build_query,
    max_rows_for,
    query_id_from_sql,
)


LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)
QUERY_EXECUTION_ID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
SIGNATURE_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MAX_REQUEST_BYTES = 8_192
TERMINAL_STATES = frozenset({"SUCCEEDED", "FAILED", "CANCELLED"})
_athena_client: Any = None
_secrets_client: Any = None
_secret_cache: str | None = None


def _client(service: str) -> Any:
    import boto3

    return boto3.client(service)


def _athena() -> Any:
    global _athena_client
    if _athena_client is None:
        _athena_client = _client("athena")
    return _athena_client


def _secret() -> str:
    global _secrets_client, _secret_cache
    if _secret_cache is not None:
        return _secret_cache
    if _secrets_client is None:
        _secrets_client = _client("secretsmanager")
    response = _secrets_client.get_secret_value(SecretId=os.environ["DASHBOARD_SECRET_ARN"])
    secret = response.get("SecretString")
    if not isinstance(secret, str) or not secret:
        raise RuntimeError("dashboard API secret is unavailable")
    _secret_cache = secret
    return secret


def _response(status: int, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "statusCode": status,
        "headers": {
            "content-type": "application/json; charset=utf-8",
            "cache-control": "no-store",
            "x-content-type-options": "nosniff",
        },
        "body": json.dumps(payload, allow_nan=False, separators=(",", ":"), default=str),
    }


def _raw_body(event: dict[str, Any]) -> bytes:
    body = event.get("body")
    if body is None:
        return b""
    if not isinstance(body, str):
        raise QueryContractError("request body is required")
    try:
        raw = base64.b64decode(body, validate=True) if event.get("isBase64Encoded") else body.encode("utf-8")
    except (binascii.Error, UnicodeEncodeError, ValueError) as exc:
        raise QueryContractError("request body encoding is invalid") from exc
    if len(raw) > MAX_REQUEST_BYTES:
        raise QueryContractError("request body is too large")
    return raw


def _headers(event: dict[str, Any]) -> dict[str, str]:
    return {str(key).lower(): str(value) for key, value in (event.get("headers") or {}).items()}


def _canonical_path(event: dict[str, Any]) -> str:
    method = event.get("httpMethod")
    path = event.get("path")
    if method == "POST" and path == "/queries":
        return path
    if method == "GET" and isinstance(path, str):
        execution_id = (event.get("pathParameters") or {}).get("queryExecutionId")
        if isinstance(execution_id, str) and path == f"/queries/{execution_id}" and QUERY_EXECUTION_ID_PATTERN.fullmatch(execution_id):
            return path
    raise QueryContractError("request path is invalid")


def _verify_request(
    event: dict[str, Any],
    raw_body: bytes,
    *,
    now: float | None = None,
) -> None:
    headers = _headers(event)
    timestamp = headers.get("x-wdirl-dashboard-timestamp", "")
    signature = headers.get("x-wdirl-dashboard-signature", "")
    if not timestamp.isdigit() or not SIGNATURE_PATTERN.fullmatch(signature):
        raise PermissionError("missing or malformed dashboard signature")
    current = time.time() if now is None else now
    max_skew = int(os.getenv("MAX_CLOCK_SKEW_SECONDS", "300"))
    if abs(current - int(timestamp)) > max_skew:
        raise PermissionError("dashboard signature timestamp is stale")
    method = str(event.get("httpMethod") or "").upper()
    path = _canonical_path(event)
    canonical = f"{timestamp}\n{method}\n{path}\n".encode("utf-8") + raw_body
    expected = hmac.new(_secret().encode("utf-8"), canonical, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise PermissionError("invalid dashboard signature")


def _body(raw: bytes) -> dict[str, Any]:
    if not raw:
        raise QueryContractError("request body is required")
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise QueryContractError("request body must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise QueryContractError("request body must be an object")
    unknown = sorted(set(payload) - {"query", "filters"})
    if unknown:
        raise QueryContractError(f"unsupported request fields: {', '.join(unknown)}")
    return payload


def _request_token(event: dict[str, Any], query_id: str, sql: str) -> str:
    signature = _headers(event).get("x-wdirl-dashboard-signature", "")
    material = f"{signature}\0{query_id}\0{hashlib.sha256(sql.encode('utf-8')).hexdigest()}".encode(
        "utf-8"
    )
    return hashlib.sha256(material).hexdigest()


def _submit(event: dict[str, Any], raw_body: bytes) -> dict[str, Any]:
    payload = _body(raw_body)
    query = build_query(payload.get("query"), payload.get("filters"))
    reuse_minutes = int(os.getenv("ATHENA_RESULT_REUSE_MINUTES", "1"))
    if not 1 <= reuse_minutes <= 60:
        raise RuntimeError("ATHENA_RESULT_REUSE_MINUTES must be between 1 and 60")
    response = _athena().start_query_execution(
        QueryString=query.sql,
        ClientRequestToken=_request_token(event, query.query_id, query.sql),
        QueryExecutionContext={"Database": os.environ["ATHENA_DATABASE"]},
        WorkGroup=os.environ["ATHENA_WORKGROUP"],
        ResultReuseConfiguration={
            "ResultReuseByAgeConfiguration": {
                "Enabled": True,
                "MaxAgeInMinutes": reuse_minutes,
            }
        },
    )
    execution_id = response["QueryExecutionId"]
    LOGGER.info(
        json.dumps(
            {
                "message": "dashboard query submitted",
                "query": query.query_id,
                "query_execution_id": execution_id,
            }
        )
    )
    return _response(
        202,
        {
            "query": query.query_id,
            "query_execution_id": execution_id,
            "state": "QUEUED",
            "poll_after_ms": 1_500,
        },
    )


def _execution_id(event: dict[str, Any]) -> str:
    value = (event.get("pathParameters") or {}).get("queryExecutionId")
    if not isinstance(value, str) or not QUERY_EXECUTION_ID_PATTERN.fullmatch(value):
        raise QueryContractError("query execution id is invalid")
    return value


def _scalar(value: str | None, athena_type: str) -> Any:
    if value is None:
        return None
    normalized = athena_type.lower()
    if normalized in {"tinyint", "smallint", "integer", "bigint"}:
        try:
            return int(value)
        except ValueError:
            return value
    if normalized in {"float", "real", "double"} or normalized.startswith("decimal"):
        try:
            parsed = Decimal(value)
            return float(parsed) if parsed.is_finite() else None
        except InvalidOperation:
            return value
    if normalized == "boolean":
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
    return value


def _rows(result: dict[str, Any], *, max_rows: int) -> list[dict[str, Any]]:
    metadata = result.get("ResultSet", {}).get("ResultSetMetadata", {})
    columns = metadata.get("ColumnInfo") or []
    names = [str(column.get("Name") or "") for column in columns]
    types = [str(column.get("Type") or "varchar") for column in columns]
    if not names or any(not name for name in names):
        return []

    raw_rows = result.get("ResultSet", {}).get("Rows") or []
    parsed: list[dict[str, Any]] = []
    for index, row in enumerate(raw_rows):
        cells = row.get("Data") or []
        raw_values = [cell.get("VarCharValue") for cell in cells]
        if index == 0 and raw_values == names:
            continue
        parsed.append(
            {
                name: _scalar(raw_values[position] if position < len(raw_values) else None, types[position])
                for position, name in enumerate(names)
            }
        )
        if len(parsed) >= max_rows:
            break
    return parsed


def _safe_failure(execution: dict[str, Any]) -> dict[str, Any]:
    status = execution.get("Status") or {}
    error = status.get("AthenaError") or {}
    LOGGER.warning(
        json.dumps(
            {
                "message": "dashboard Athena query did not succeed",
                "query_execution_id": execution.get("QueryExecutionId"),
                "state": status.get("State"),
                "reason": status.get("StateChangeReason"),
                "error_category": error.get("ErrorCategory"),
                "error_type": error.get("ErrorType"),
            }
        )
    )
    payload: dict[str, Any] = {"message": "query did not complete"}
    if error:
        payload["category"] = error.get("ErrorCategory")
        payload["type"] = error.get("ErrorType")
        payload["retryable"] = bool(error.get("Retryable"))
    return payload


def _poll(event: dict[str, Any]) -> dict[str, Any]:
    execution_id = _execution_id(event)
    execution = _athena().get_query_execution(QueryExecutionId=execution_id)["QueryExecution"]
    if execution.get("WorkGroup") != os.environ["ATHENA_WORKGROUP"]:
        return _response(404, {"error": "query not found"})
    query_id = query_id_from_sql(execution.get("Query"))
    if query_id is None:
        return _response(404, {"error": "query not found"})

    state = str((execution.get("Status") or {}).get("State") or "UNKNOWN")
    statistics = execution.get("Statistics") or {}
    base: dict[str, Any] = {
        "query": query_id,
        "query_execution_id": execution_id,
        "state": state,
        "statistics": {
            "engine_execution_ms": statistics.get("EngineExecutionTimeInMillis"),
            "queue_ms": statistics.get("QueryQueueTimeInMillis"),
            "data_scanned_bytes": statistics.get("DataScannedInBytes"),
            "reused_previous_result": bool(
                (statistics.get("ResultReuseInformation") or {}).get("ReusedPreviousResult")
            ),
        },
    }
    if state not in TERMINAL_STATES:
        base["poll_after_ms"] = 1_500
        return _response(200, base)
    if state != "SUCCEEDED":
        base["error"] = _safe_failure(execution)
        return _response(200, base)

    maximum = max_rows_for(query_id)
    result = _athena().get_query_results(
        QueryExecutionId=execution_id,
        MaxResults=min(maximum + 1, 1_000),
    )
    base["rows"] = _rows(result, max_rows=maximum)
    base["row_count"] = len(base["rows"])
    return _response(200, base)


def handler(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    try:
        raw_body = _raw_body(event)
        _verify_request(event, raw_body)
        method = event.get("httpMethod")
        if method == "POST":
            return _submit(event, raw_body)
        if method == "GET":
            if raw_body:
                raise QueryContractError("GET request body must be empty")
            return _poll(event)
        return _response(405, {"error": "method not allowed"})
    except PermissionError as exc:
        LOGGER.warning(
            json.dumps({"message": "dashboard authentication rejected", "reason": str(exc)})
        )
        return _response(401, {"error": "unauthorized"})
    except QueryContractError as exc:
        return _response(400, {"error": str(exc)})
    except Exception:
        LOGGER.exception(json.dumps({"message": "dashboard query API failed"}))
        return _response(500, {"error": "analytics query unavailable"})
