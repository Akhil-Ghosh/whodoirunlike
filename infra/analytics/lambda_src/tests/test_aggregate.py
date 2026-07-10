from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


LAMBDA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAMBDA_ROOT))

import aggregate  # noqa: E402


class MissingObjectError(Exception):
    response = {"Error": {"Code": "404"}}


class AggregateTests(unittest.TestCase):
    def setUp(self) -> None:
        aggregate._s3_client = Mock()
        aggregate._athena_client = Mock()
        os.environ.update(
            {
                "EVENT_BUCKET": "event-lake",
                "ATHENA_DATABASE": "analytics",
                "ATHENA_SOURCE_TABLE": "processing_events",
                "ATHENA_WORKGROUP": "processing",
            }
        )

    @patch("aggregate.time.sleep")
    def test_runs_idempotent_daily_unload(self, _sleep: Mock) -> None:
        aggregate._s3_client.head_object.side_effect = MissingObjectError()
        aggregate._s3_client.list_objects_v2.return_value = {"IsTruncated": False}
        aggregate._athena_client.start_query_execution.return_value = {
            "QueryExecutionId": "query-1"
        }
        aggregate._athena_client.get_query_execution.return_value = {
            "QueryExecution": {"Status": {"State": "SUCCEEDED"}}
        }
        result = aggregate.handler({"date": "2026-07-08"}, None)
        self.assertEqual(result["status"], "complete")
        query = aggregate._athena_client.start_query_execution.call_args.kwargs["QueryString"]
        self.assertIn("event_date = '2026-07-08'", query)
        self.assertIn("format = 'PARQUET'", query)
        self.assertIn("'result_ready'", query)
        self.assertIn("duration_bucket", query)
        aggregate._s3_client.put_object.assert_called_once()

    def test_skips_date_with_success_marker(self) -> None:
        aggregate._s3_client.head_object.return_value = {}
        result = aggregate.handler({"date": "2026-07-08"}, None)
        self.assertEqual(result, {"status": "already_complete", "event_date": "2026-07-08"})
        aggregate._athena_client.start_query_execution.assert_not_called()

    @patch("aggregate.time.sleep")
    def test_force_rebuild_ignores_success_marker_for_late_events(self, _sleep: Mock) -> None:
        aggregate._s3_client.list_objects_v2.return_value = {"IsTruncated": False}
        aggregate._athena_client.start_query_execution.return_value = {
            "QueryExecutionId": "query-rebuild"
        }
        aggregate._athena_client.get_query_execution.return_value = {
            "QueryExecution": {"Status": {"State": "SUCCEEDED"}}
        }

        result = aggregate.handler(
            {"date": "2026-07-06", "force": True},
            None,
        )

        self.assertEqual(result["status"], "complete")
        aggregate._s3_client.head_object.assert_not_called()
        aggregate._athena_client.start_query_execution.assert_called_once()


if __name__ == "__main__":
    unittest.main()
