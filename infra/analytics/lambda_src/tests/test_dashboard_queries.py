from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path


LAMBDA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAMBDA_ROOT))

import dashboard_queries  # noqa: E402


ATTEMPT_ID = "123e4567-e89b-42d3-a456-426614174000"


class DashboardQueryContractTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ.update(
            {
                "ATHENA_DATABASE": "analytics",
                "ATHENA_TABLE": "processing_events",
            }
        )

    def test_exposes_only_the_eight_fixed_contracts(self) -> None:
        self.assertEqual(
            set(dashboard_queries.BUILDERS),
            {
                "overview",
                "stage_latency",
                "span_latency",
                "attempts",
                "attempt_detail",
                "failures",
                "stalls",
                "freshness",
            },
        )
        with self.assertRaisesRegex(dashboard_queries.QueryContractError, "allowlisted"):
            dashboard_queries.build_query("SELECT * FROM processing_events")

    def test_rejects_unknown_and_injection_shaped_parameters(self) -> None:
        with self.assertRaisesRegex(dashboard_queries.QueryContractError, "unsupported parameters"):
            dashboard_queries.build_query("overview", {"sql": "SELECT 1"})
        with self.assertRaisesRegex(dashboard_queries.QueryContractError, "invalid value"):
            dashboard_queries.build_query("stage_latency", {"gpu_type": "A10'; DROP TABLE x;--"})
        with self.assertRaisesRegex(dashboard_queries.QueryContractError, "range_days"):
            dashboard_queries.build_query("overview", {"range_days": 365})

    def test_stage_query_is_partition_bounded_and_includes_confidence(self) -> None:
        query = dashboard_queries.build_query(
            "stage_latency",
            {"range_days": 14, "cold_start": True, "backend": "runpod"},
        )
        self.assertTrue(query.sql.startswith("-- dashboard-query:stage_latency"))
        self.assertIn("interval '14' day", query.sql)
        self.assertIn(
            "from_iso8601_timestamp(event_time) >= current_timestamp - interval '14' day",
            query.sql,
        )
        self.assertIn("a.cold_start = true", query.sql)
        self.assertIn("a.backend = 'runpod'", query.sql)
        self.assertIn("THEN 'low' ELSE 'stable' END AS confidence", query.sql)
        self.assertIn("LIMIT 100", query.sql)

    def test_overview_rates_use_only_terminal_attempts(self) -> None:
        query = dashboard_queries.build_query("overview")
        self.assertIn("count_if(completed = 1) AS completed_attempts", query.sql)
        self.assertIn(
            "count_if(completed = 1 OR failed = 1) AS terminal_attempts",
            query.sql,
        )
        self.assertIn("h.terminal_attempts = 0", query.sql)
        self.assertIn("AS success_rate", query.sql)

    def test_attempt_backend_dimension_is_the_runner_mask_backend(self) -> None:
        query = dashboard_queries.build_query("attempts")
        self.assertIn(
            "WHERE stage = 'runner_mask' AND backend IS NOT NULL",
            query.sql,
        )

    def test_failed_stage_time_is_attributed_to_the_attempt(self) -> None:
        query = dashboard_queries.build_query("attempts")
        self.assertIn(
            "event_type IN ('stage_completed', 'stage_failed') THEN elapsed_seconds",
            query.sql,
        )
        self.assertIn(
            "WHERE event_type IN ('stage_completed', 'stage_failed')",
            query.sql,
        )

    def test_attempt_detail_requires_uuid_and_never_selects_progress(self) -> None:
        with self.assertRaisesRegex(dashboard_queries.QueryContractError, "UUID"):
            dashboard_queries.build_query("attempt_detail", {"attempt_id": "../clip.mp4"})
        query = dashboard_queries.build_query("attempt_detail", {"attempt_id": ATTEMPT_ID})
        self.assertIn(f"attempt_id = '{ATTEMPT_ID}'", query.sql)
        self.assertIn("event_type <> 'progress_sampled'", query.sql)
        self.assertIn("start_offset_seconds", query.sql)
        self.assertNotIn("measurements_json", query.sql)
        self.assertNotIn("attributes_json", query.sql)

    def test_attempt_list_and_stalls_are_strictly_bounded(self) -> None:
        attempts = dashboard_queries.build_query("attempts", {"limit": 25, "status": "failed"})
        self.assertEqual(attempts.max_rows, 25)
        self.assertIn("a.failed = 1", attempts.sql)
        self.assertIn("LIMIT 25", attempts.sql)

        stalls = dashboard_queries.build_query("stalls", {"stale_minutes": 30})
        self.assertIn("interval '30' minute", stalls.sql)
        self.assertIn("a.completed = 0", stalls.sql)
        self.assertIn("a.failed = 0", stalls.sql)
        self.assertIn("a.processing_was_requested = 1", stalls.sql)
        self.assertIn("LIMIT 100", stalls.sql)

    def test_span_and_failure_queries_keep_event_join_columns_qualified(self) -> None:
        for query_id in ("span_latency", "failures"):
            query = dashboard_queries.build_query(query_id)
            self.assertIn(
                "ON e.run_id = a.run_id AND e.attempt_id = a.attempt_id",
                query.sql,
            )
            self.assertNotIn("USING (run_id, attempt_id)", query.sql)

    def test_query_marker_can_only_recover_allowlisted_contracts(self) -> None:
        query = dashboard_queries.build_query("freshness")
        self.assertEqual(dashboard_queries.query_id_from_sql(query.sql), "freshness")
        self.assertIsNone(dashboard_queries.query_id_from_sql("SELECT 1"))
        self.assertIsNone(dashboard_queries.query_id_from_sql("-- dashboard-query:admin\nSELECT 1"))


if __name__ == "__main__":
    unittest.main()
