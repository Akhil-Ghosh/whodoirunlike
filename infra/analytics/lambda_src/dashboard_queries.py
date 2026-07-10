from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping


QUERY_MARKER = "-- dashboard-query:"
UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
SAFE_VALUE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._:+/@-]{0,63}$")
IDENTIFIER_PATTERN = re.compile(r"^[a-z_][a-z0-9_]{0,127}$")
RANGE_DAYS = frozenset({1, 7, 14, 30, 90})
ROW_LIMITS = frozenset({25, 50, 100})
STALE_MINUTES = frozenset({5, 10, 15, 30, 60})

COMMON_FILTERS = frozenset(
    {
        "environment",
        "backend",
        "gpu_type",
        "processor_version",
        "duration_bucket",
        "resolution_bucket",
        "cold_start",
    }
)
STAGES = frozenset(
    {
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
    }
)
SPANS = frozenset(
    {"model_load", "decode", "preprocess", "inference", "postprocess", "render", "encode", "write", "publish"}
)


class QueryContractError(ValueError):
    pass


@dataclass(frozen=True)
class DashboardQuery:
    query_id: str
    sql: str
    max_rows: int


def _identifier(value: str, field: str) -> str:
    if not IDENTIFIER_PATTERN.fullmatch(value):
        raise RuntimeError(f"{field} is not a valid Athena identifier")
    return value


def _source() -> str:
    database = _identifier(os.environ["ATHENA_DATABASE"], "ATHENA_DATABASE")
    table = _identifier(os.environ["ATHENA_TABLE"], "ATHENA_TABLE")
    return f"{database}.{table}"


def _integer(value: Any, field: str, allowed: frozenset[int], default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value not in allowed:
        choices = ", ".join(str(item) for item in sorted(allowed))
        raise QueryContractError(f"{field} must be one of: {choices}")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SAFE_VALUE_PATTERN.fullmatch(value):
        raise QueryContractError(f"{field} has an invalid value")
    return value


def _enum(value: Any, field: str, allowed: frozenset[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise QueryContractError(f"{field} has an invalid value")
    return value


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _validate_parameters(parameters: Any, allowed: set[str]) -> dict[str, Any]:
    if parameters is None:
        return {}
    if not isinstance(parameters, dict):
        raise QueryContractError("parameters must be an object")
    unknown = sorted(set(parameters) - allowed)
    if unknown:
        raise QueryContractError(f"unsupported parameters: {', '.join(unknown)}")
    return dict(parameters)


def _days(parameters: Mapping[str, Any], default: int = 30) -> int:
    return _integer(parameters.get("range_days"), "range_days", RANGE_DAYS, default)


def _deduplicated_ctes(days: int) -> str:
    return f"""
ranked AS (
  SELECT *, row_number() OVER (PARTITION BY event_id ORDER BY ingested_at DESC) AS event_rank
  FROM {_source()}
  WHERE event_date BETWEEN date_format(current_date - interval '{days}' day, '%Y-%m-%d')
                       AND date_format(current_date, '%Y-%m-%d')
    AND from_iso8601_timestamp(event_time) >= current_timestamp - interval '{days}' day
),
events AS (
  SELECT * FROM ranked WHERE event_rank = 1
)""".strip()


def _attempt_dimensions_cte() -> str:
    return """
attempt_dimensions AS (
  SELECT run_id, attempt_id,
         max_by(environment, sequence) FILTER (WHERE environment IS NOT NULL) AS environment,
         max_by(backend, sequence) FILTER (
           WHERE stage = 'runner_mask' AND backend IS NOT NULL
         ) AS backend,
         max_by(gpu_type, sequence) FILTER (WHERE gpu_type IS NOT NULL) AS gpu_type,
         max_by(processor_version, sequence) FILTER (WHERE processor_version IS NOT NULL) AS processor_version,
         max_by(duration_bucket, sequence) FILTER (WHERE duration_bucket IS NOT NULL) AS duration_bucket,
         max_by(resolution_bucket, sequence) FILTER (WHERE resolution_bucket IS NOT NULL) AS resolution_bucket,
         max_by(cold_start, sequence) FILTER (WHERE cold_start IS NOT NULL) AS cold_start,
         max(clip_duration_seconds) AS clip_duration_seconds,
         max(clip_frame_count) AS clip_frame_count,
         max(clip_width) AS clip_width,
         max(clip_height) AS clip_height,
         min(from_iso8601_timestamp(event_time)) AS first_event_at,
         max(from_iso8601_timestamp(event_time)) AS last_event_at,
         max(CASE WHEN event_type = 'result_ready' THEN elapsed_seconds END) AS result_ready_seconds,
         max(CASE WHEN event_type = 'analysis_completed' THEN elapsed_seconds END) AS analysis_complete_seconds,
         max(CASE WHEN event_type = 'attempt_completed' THEN elapsed_seconds END) AS attempt_complete_seconds,
         max(CASE WHEN event_type = 'attempt_failed' THEN elapsed_seconds END) AS attempt_failed_seconds,
         max(CASE WHEN event_type = 'attempt_failed' THEN 1 ELSE 0 END) AS failed,
         max(CASE WHEN event_type = 'attempt_completed' THEN 1 ELSE 0 END) AS completed,
         max(CASE WHEN stage IN ('processor_enqueue', 'processor_queue') THEN 1 ELSE 0 END) AS processing_was_requested,
         sum(CASE WHEN event_type IN ('stage_completed', 'stage_failed') THEN elapsed_seconds ELSE 0 END) AS observed_stage_seconds,
         max_by(stage, elapsed_seconds) FILTER (
           WHERE event_type IN ('stage_completed', 'stage_failed')
         ) AS bottleneck_stage,
         max(elapsed_seconds) FILTER (
           WHERE event_type IN ('stage_completed', 'stage_failed')
         ) AS bottleneck_seconds,
         max_by(stage, sequence) FILTER (WHERE stage IS NOT NULL) AS last_stage,
         max_by(span, sequence) FILTER (WHERE span IS NOT NULL) AS last_span,
         max_by(event_type, sequence) AS last_event_type
  FROM events
  GROUP BY run_id, attempt_id
)""".strip()


def _filter_predicates(parameters: Mapping[str, Any], alias: str = "a") -> list[str]:
    predicates: list[str] = []
    for field in sorted(COMMON_FILTERS - {"cold_start"}):
        if field in parameters:
            predicates.append(f"{alias}.{field} = {_literal(_string(parameters[field], field))}")
    if "cold_start" in parameters:
        value = parameters["cold_start"]
        if not isinstance(value, bool):
            raise QueryContractError("cold_start must be a boolean")
        predicates.append(f"{alias}.cold_start = {'true' if value else 'false'}")
    return predicates


def _where(predicates: list[str]) -> str:
    return " AND ".join(predicates) if predicates else "true"


def _marked(query_id: str, sql: str, max_rows: int) -> DashboardQuery:
    return DashboardQuery(query_id, f"{QUERY_MARKER}{query_id}\n{sql.strip()}", max_rows)


def _overview(raw: Any) -> DashboardQuery:
    parameters = _validate_parameters(raw, set(COMMON_FILTERS | {"range_days"}))
    days = _days(parameters)
    attempt_filter = _where(_filter_predicates(parameters))
    sql = f"""
WITH {_deduplicated_ctes(days)},
{_attempt_dimensions_cte()},
filtered_attempts AS (
  SELECT * FROM attempt_dimensions a WHERE {attempt_filter}
),
headline AS (
  SELECT count(*) AS attempts,
         count_if(result_ready_seconds IS NOT NULL) AS result_ready_attempts,
         count_if(completed = 1) AS completed_attempts,
         count_if(failed = 1) AS failed_attempts,
         count_if(completed = 1 OR failed = 1) AS terminal_attempts,
         approx_percentile(result_ready_seconds, 0.50) AS p50_result_ready_seconds,
         approx_percentile(result_ready_seconds, 0.90) AS p90_result_ready_seconds,
         approx_percentile(result_ready_seconds, 0.95) AS p95_result_ready_seconds,
         avg(result_ready_seconds) AS average_result_ready_seconds
  FROM filtered_attempts
),
stage_latency AS (
  SELECT e.stage,
         count(*) AS samples,
         approx_percentile(e.elapsed_seconds, 0.95) AS p95_seconds
  FROM events e
  INNER JOIN filtered_attempts a USING (run_id, attempt_id)
  WHERE e.event_type = 'stage_completed'
  GROUP BY e.stage
  ORDER BY p95_seconds DESC NULLS LAST
  LIMIT 1
)
SELECT h.attempts, h.result_ready_attempts, h.completed_attempts, h.failed_attempts, h.terminal_attempts,
       CASE WHEN h.terminal_attempts = 0 THEN CAST(NULL AS double)
            ELSE CAST(h.completed_attempts AS double) / h.terminal_attempts END AS success_rate,
       CASE WHEN h.terminal_attempts = 0 THEN CAST(NULL AS double)
            ELSE CAST(h.failed_attempts AS double) / h.terminal_attempts END AS failure_rate,
       h.p50_result_ready_seconds, h.p90_result_ready_seconds, h.p95_result_ready_seconds,
       h.average_result_ready_seconds,
       s.stage AS bottleneck_stage, s.samples AS bottleneck_samples, s.p95_seconds AS bottleneck_p95_seconds
FROM headline h
LEFT JOIN stage_latency s ON true
LIMIT 1
"""
    return _marked("overview", sql, 1)


def _stage_latency(raw: Any) -> DashboardQuery:
    parameters = _validate_parameters(raw, set(COMMON_FILTERS | {"range_days", "stage"}))
    days = _days(parameters)
    predicates = _filter_predicates(parameters, "a")
    if "stage" in parameters:
        predicates.append(f"e.stage = {_literal(_enum(parameters['stage'], 'stage', STAGES))}")
    sql = f"""
WITH {_deduplicated_ctes(days)},
{_attempt_dimensions_cte()}
SELECT e.stage,
       count_if(e.event_type = 'stage_completed') AS samples,
       count_if(e.event_type = 'stage_failed') AS failures,
       approx_percentile(CASE WHEN e.event_type = 'stage_completed' THEN e.elapsed_seconds END, 0.50) AS p50_seconds,
       approx_percentile(CASE WHEN e.event_type = 'stage_completed' THEN e.elapsed_seconds END, 0.90) AS p90_seconds,
       approx_percentile(CASE WHEN e.event_type = 'stage_completed' THEN e.elapsed_seconds END, 0.95) AS p95_seconds,
       avg(CASE WHEN e.event_type = 'stage_completed' AND a.clip_frame_count > 0
                THEN e.elapsed_seconds * 1000 / a.clip_frame_count END) AS average_ms_per_frame,
       CASE WHEN count_if(e.event_type = 'stage_completed') < 20 THEN 'low' ELSE 'stable' END AS confidence
FROM events e
INNER JOIN attempt_dimensions a USING (run_id, attempt_id)
WHERE e.event_type IN ('stage_completed', 'stage_failed')
  AND {_where(predicates)}
GROUP BY e.stage
ORDER BY p95_seconds DESC NULLS LAST
LIMIT 100
"""
    return _marked("stage_latency", sql, 100)


def _span_latency(raw: Any) -> DashboardQuery:
    parameters = _validate_parameters(raw, set(COMMON_FILTERS | {"range_days", "stage", "span"}))
    days = _days(parameters)
    predicates = _filter_predicates(parameters, "a")
    if "stage" in parameters:
        predicates.append(f"e.stage = {_literal(_enum(parameters['stage'], 'stage', STAGES))}")
    if "span" in parameters:
        predicates.append(f"e.span = {_literal(_enum(parameters['span'], 'span', SPANS))}")
    sql = f"""
WITH {_deduplicated_ctes(days)},
{_attempt_dimensions_cte()},
span_attempt_totals AS (
  SELECT e.run_id, e.attempt_id, e.stage, e.span,
         sum(CASE WHEN e.event_type = 'span_completed' THEN e.elapsed_seconds ELSE 0 END) AS elapsed_seconds,
         count_if(e.event_type = 'span_completed') AS occurrences,
         count_if(e.event_type = 'span_failed') AS failures
  FROM events e
  INNER JOIN attempt_dimensions a
    ON e.run_id = a.run_id AND e.attempt_id = a.attempt_id
  WHERE e.event_type IN ('span_completed', 'span_failed')
    AND {_where(predicates)}
  GROUP BY e.run_id, e.attempt_id, e.stage, e.span
)
SELECT stage, span,
       count_if(failures = 0 AND occurrences > 0) AS samples,
       sum(occurrences) AS occurrences,
       sum(failures) AS failures,
       approx_percentile(CASE WHEN failures = 0 AND occurrences > 0 THEN elapsed_seconds END, 0.50) AS p50_seconds,
       approx_percentile(CASE WHEN failures = 0 AND occurrences > 0 THEN elapsed_seconds END, 0.90) AS p90_seconds,
       approx_percentile(CASE WHEN failures = 0 AND occurrences > 0 THEN elapsed_seconds END, 0.95) AS p95_seconds,
       CASE WHEN count_if(failures = 0 AND occurrences > 0) < 20 THEN 'low' ELSE 'stable' END AS confidence
FROM span_attempt_totals
GROUP BY stage, span
ORDER BY p95_seconds DESC NULLS LAST
LIMIT 250
"""
    return _marked("span_latency", sql, 250)


def _attempts(raw: Any) -> DashboardQuery:
    parameters = _validate_parameters(raw, set(COMMON_FILTERS | {"range_days", "status", "limit"}))
    days = _days(parameters)
    limit = _integer(parameters.get("limit"), "limit", ROW_LIMITS, 50)
    predicates = _filter_predicates(parameters)
    if "status" in parameters:
        status = _enum(parameters["status"], "status", frozenset({"complete", "failed", "running"}))
        predicates.append(
            {
                "complete": "a.completed = 1",
                "failed": "a.failed = 1",
                "running": "a.completed = 0 AND a.failed = 0",
            }[status]
        )
    sql = f"""
WITH {_deduplicated_ctes(days)},
{_attempt_dimensions_cte()}
SELECT a.run_id, a.attempt_id, a.first_event_at, a.last_event_at,
       CASE WHEN a.failed = 1 THEN 'failed' WHEN a.completed = 1 THEN 'complete' ELSE 'running' END AS status,
       a.environment, a.backend, a.gpu_type, a.processor_version, a.cold_start,
       a.duration_bucket, a.resolution_bucket, a.clip_duration_seconds, a.clip_frame_count,
       a.clip_width, a.clip_height,
       a.result_ready_seconds, a.analysis_complete_seconds, a.attempt_complete_seconds,
       a.observed_stage_seconds,
       greatest(
         coalesce(a.attempt_complete_seconds, a.result_ready_seconds, a.attempt_failed_seconds, 0) - a.observed_stage_seconds,
         0
       ) AS unattributed_seconds,
       a.bottleneck_stage, a.bottleneck_seconds,
       a.last_stage, a.last_span, a.last_event_type
FROM attempt_dimensions a
WHERE {_where(predicates)}
ORDER BY a.last_event_at DESC
LIMIT {limit}
"""
    return _marked("attempts", sql, limit)


def _attempt_detail(raw: Any) -> DashboardQuery:
    parameters = _validate_parameters(raw, {"attempt_id", "range_days"})
    attempt_id = parameters.get("attempt_id")
    if not isinstance(attempt_id, str) or not UUID_PATTERN.fullmatch(attempt_id):
        raise QueryContractError("attempt_id must be a UUID")
    days = _days(parameters, 90)
    sql = f"""
WITH {_deduplicated_ctes(days)},
attempt_events AS (
  SELECT *, from_iso8601_timestamp(event_time) AS event_at
  FROM events
  WHERE attempt_id = {_literal(attempt_id)}
    AND event_type <> 'progress_sampled'
),
origin AS (
  SELECT min(event_at) AS first_event_at FROM attempt_events
)
SELECT e.run_id, e.attempt_id, e.sequence, e.event_type, e.stage, e.span, e.status,
       e.event_time, e.elapsed_seconds,
       greatest(
         date_diff('millisecond', o.first_event_at, e.event_at) / 1000.0 -
           CASE WHEN e.event_type IN ('stage_completed', 'stage_failed', 'span_completed', 'span_failed')
                THEN coalesce(e.elapsed_seconds, 0) ELSE 0 END,
         0
       ) AS start_offset_seconds,
       date_diff('millisecond', o.first_event_at, e.event_at) / 1000.0 AS end_offset_seconds,
       e.timing_basis, e.artifact_type, e.artifact_size_bytes, e.processed_frames, e.total_frames,
       e.backend, e.gpu_type, e.processor_version, e.cold_start,
       e.error_class, e.error_code
FROM attempt_events e
CROSS JOIN origin o
ORDER BY e.sequence ASC
LIMIT 500
"""
    return _marked("attempt_detail", sql, 500)


def _failures(raw: Any) -> DashboardQuery:
    parameters = _validate_parameters(raw, set(COMMON_FILTERS | {"range_days", "stage"}))
    days = _days(parameters)
    predicates = _filter_predicates(parameters, "a")
    if "stage" in parameters:
        predicates.append(f"e.stage = {_literal(_enum(parameters['stage'], 'stage', STAGES))}")
    sql = f"""
WITH {_deduplicated_ctes(days)},
{_attempt_dimensions_cte()}
SELECT e.stage, e.span, e.error_class, e.error_code,
       count(*) AS failures,
       count(DISTINCT e.attempt_id) AS affected_attempts,
       approx_percentile(e.elapsed_seconds, 0.50) AS p50_time_to_failure_seconds,
       approx_percentile(e.elapsed_seconds, 0.95) AS p95_time_to_failure_seconds,
       max(from_iso8601_timestamp(e.event_time)) AS most_recent_at
FROM events e
  INNER JOIN attempt_dimensions a
    ON e.run_id = a.run_id AND e.attempt_id = a.attempt_id
WHERE e.event_type IN ('span_failed', 'stage_failed', 'attempt_failed')
  AND {_where(predicates)}
GROUP BY e.stage, e.span, e.error_class, e.error_code
ORDER BY failures DESC, most_recent_at DESC
LIMIT 100
"""
    return _marked("failures", sql, 100)


def _stalls(raw: Any) -> DashboardQuery:
    parameters = _validate_parameters(raw, set(COMMON_FILTERS | {"range_days", "stale_minutes"}))
    days = _days(parameters)
    stale_minutes = _integer(parameters.get("stale_minutes"), "stale_minutes", STALE_MINUTES, 10)
    predicates = _filter_predicates(parameters)
    predicates.extend(
        [
            "a.completed = 0",
            "a.failed = 0",
            "a.processing_was_requested = 1",
            f"a.last_event_at < current_timestamp - interval '{stale_minutes}' minute",
        ]
    )
    sql = f"""
WITH {_deduplicated_ctes(days)},
{_attempt_dimensions_cte()}
SELECT a.run_id, a.attempt_id, a.first_event_at, a.last_event_at,
       date_diff('second', a.last_event_at, current_timestamp) AS stale_seconds,
       a.last_stage, a.last_span, a.last_event_type,
       a.environment, a.backend, a.gpu_type, a.processor_version,
       a.duration_bucket, a.resolution_bucket
FROM attempt_dimensions a
WHERE {_where(predicates)}
ORDER BY a.last_event_at ASC
LIMIT 100
"""
    return _marked("stalls", sql, 100)


def _freshness(raw: Any) -> DashboardQuery:
    parameters = _validate_parameters(raw, {"range_days"})
    days = _days(parameters, 7)
    sql = f"""
WITH {_deduplicated_ctes(days)}
SELECT max(from_iso8601_timestamp(event_time)) AS latest_event_at,
       max(from_iso8601_timestamp(ingested_at)) AS latest_ingested_at,
       date_diff('second', max(from_iso8601_timestamp(event_time)), current_timestamp) AS event_age_seconds,
       date_diff('second', max(from_iso8601_timestamp(event_time)), max(from_iso8601_timestamp(ingested_at))) AS latest_ingestion_lag_seconds,
       count_if(from_iso8601_timestamp(event_time) >= current_timestamp - interval '24' hour) AS events_last_24_hours,
       count(DISTINCT CASE WHEN from_iso8601_timestamp(event_time) >= current_timestamp - interval '24' hour
                           THEN attempt_id END) AS attempts_last_24_hours
FROM events
LIMIT 1
"""
    return _marked("freshness", sql, 1)


BUILDERS: dict[str, Callable[[Any], DashboardQuery]] = {
    "overview": _overview,
    "stage_latency": _stage_latency,
    "span_latency": _span_latency,
    "attempts": _attempts,
    "attempt_detail": _attempt_detail,
    "failures": _failures,
    "stalls": _stalls,
    "freshness": _freshness,
}


def build_query(query_id: Any, parameters: Any = None) -> DashboardQuery:
    if not isinstance(query_id, str) or query_id not in BUILDERS:
        raise QueryContractError("query is not allowlisted")
    return BUILDERS[query_id](parameters)


def query_id_from_sql(sql: Any) -> str | None:
    if not isinstance(sql, str) or not sql.startswith(QUERY_MARKER):
        return None
    first_line = sql.splitlines()[0]
    query_id = first_line[len(QUERY_MARKER) :]
    return query_id if query_id in BUILDERS else None


def max_rows_for(query_id: str) -> int:
    if query_id == "attempts":
        return 100
    return {
        "overview": 1,
        "stage_latency": 100,
        "span_latency": 250,
        "attempt_detail": 500,
        "failures": 100,
        "stalls": 100,
        "freshness": 1,
    }[query_id]
