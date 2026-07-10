from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import Mock


LAMBDA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAMBDA_ROOT))

import dashboard_api  # noqa: E402


QUERY_ID = "2f47c767-1b08-4a72-95a1-bd5da365fe60"


class DashboardApiTests(unittest.TestCase):
    def setUp(self) -> None:
        dashboard_api._athena_client = Mock()
        dashboard_api._secret_cache = "dashboard-secret"
        dashboard_api._secrets_client = Mock()
        self.timestamp = str(int(time.time()))
        os.environ.update(
            {
                "ATHENA_DATABASE": "analytics",
                "ATHENA_TABLE": "processing_events",
                "ATHENA_WORKGROUP": "dashboard-workgroup",
                "ATHENA_RESULT_REUSE_MINUTES": "1",
            }
        )

    def signed_event(
        self,
        method: str,
        path: str,
        body: bytes,
        *,
        path_parameters: dict[str, str] | None = None,
    ) -> dict[str, object]:
        timestamp = self.timestamp
        canonical = f"{timestamp}\n{method}\n{path}\n".encode() + body
        signature = hmac.new(b"dashboard-secret", canonical, hashlib.sha256).hexdigest()
        return {
            "httpMethod": method,
            "path": path,
            "pathParameters": path_parameters,
            "headers": {
                "X-WDIRL-Dashboard-Timestamp": timestamp,
                "X-WDIRL-Dashboard-Signature": signature,
            },
            "body": body.decode(),
            "isBase64Encoded": False,
            "requestContext": {"requestId": "request-123"},
        }

    def post_event(self, payload: object) -> dict[str, object]:
        body = json.dumps(payload, separators=(",", ":")).encode()
        return self.signed_event("POST", "/queries", body)

    def get_event(self, execution_id: str = QUERY_ID) -> dict[str, object]:
        return self.signed_event(
            "GET",
            f"/queries/{execution_id}",
            b"",
            path_parameters={"queryExecutionId": execution_id},
        )

    def unsigned_post_event(self, payload: object) -> dict[str, object]:
        return {
            "httpMethod": "POST",
            "path": "/queries",
            "body": json.dumps(payload),
            "isBase64Encoded": False,
            "requestContext": {"requestId": "request-123"},
        }

    def response_body(self, response: dict[str, object]) -> dict[str, object]:
        return json.loads(str(response["body"]))

    def test_submits_allowlisted_query_asynchronously_with_result_reuse(self) -> None:
        dashboard_api._athena_client.start_query_execution.return_value = {
            "QueryExecutionId": QUERY_ID
        }
        response = dashboard_api.handler(
            self.post_event({"query": "overview", "filters": {"range_days": 7}}),
            None,
        )
        self.assertEqual(response["statusCode"], 202)
        body = self.response_body(response)
        self.assertEqual(body["query_execution_id"], QUERY_ID)
        call = dashboard_api._athena_client.start_query_execution.call_args.kwargs
        self.assertEqual(call["WorkGroup"], "dashboard-workgroup")
        self.assertTrue(call["QueryString"].startswith("-- dashboard-query:overview"))
        self.assertEqual(
            call["ResultReuseConfiguration"]["ResultReuseByAgeConfiguration"],
            {"Enabled": True, "MaxAgeInMinutes": 1},
        )
        self.assertEqual(len(call["ClientRequestToken"]), 64)

        first_token = call["ClientRequestToken"]
        dashboard_api.handler(
            self.post_event({"query": "overview", "filters": {"range_days": 7}}),
            None,
        )
        second_token = dashboard_api._athena_client.start_query_execution.call_args.kwargs[
            "ClientRequestToken"
        ]
        self.assertEqual(first_token, second_token, "an exact signed replay must be idempotent")

    def test_rejects_arbitrary_sql_before_calling_athena(self) -> None:
        response = dashboard_api.handler(self.post_event({"query": "SELECT * FROM secrets"}), None)
        self.assertEqual(response["statusCode"], 400)
        dashboard_api._athena_client.start_query_execution.assert_not_called()

    def test_rejects_missing_or_body_mismatched_hmac(self) -> None:
        response = dashboard_api.handler(self.unsigned_post_event({"query": "overview"}), None)
        self.assertEqual(response["statusCode"], 401)

        event = self.post_event({"query": "overview"})
        event["body"] = json.dumps({"query": "failures"})
        response = dashboard_api.handler(event, None)
        self.assertEqual(response["statusCode"], 401)
        dashboard_api._athena_client.start_query_execution.assert_not_called()

    def test_rejects_stale_signature_and_noncanonical_path(self) -> None:
        event = self.post_event({"query": "overview"})
        with self.assertRaisesRegex(PermissionError, "stale"):
            dashboard_api._verify_request(
                event,
                str(event["body"]).encode(),
                now=int(self.timestamp) + 301,
            )

        event["path"] = "/v1/queries"
        response = dashboard_api.handler(event, None)
        self.assertEqual(response["statusCode"], 400)
        dashboard_api._athena_client.start_query_execution.assert_not_called()

    def test_reports_running_query_without_fetching_results(self) -> None:
        dashboard_api._athena_client.get_query_execution.return_value = {
            "QueryExecution": {
                "QueryExecutionId": QUERY_ID,
                "Query": "-- dashboard-query:overview\nSELECT 1",
                "WorkGroup": "dashboard-workgroup",
                "Status": {"State": "RUNNING"},
                "Statistics": {"QueryQueueTimeInMillis": 7},
            }
        }
        response = dashboard_api.handler(self.get_event(), None)
        body = self.response_body(response)
        self.assertEqual(response["statusCode"], 200)
        self.assertEqual(body["state"], "RUNNING")
        self.assertEqual(body["poll_after_ms"], 1_500)
        dashboard_api._athena_client.get_query_results.assert_not_called()

    def test_returns_typed_rows_only_for_the_dedicated_workgroup(self) -> None:
        dashboard_api._athena_client.get_query_execution.return_value = {
            "QueryExecution": {
                "QueryExecutionId": QUERY_ID,
                "Query": "-- dashboard-query:overview\nSELECT 1",
                "WorkGroup": "dashboard-workgroup",
                "Status": {"State": "SUCCEEDED"},
                "Statistics": {
                    "DataScannedInBytes": 1024,
                    "ResultReuseInformation": {"ReusedPreviousResult": True},
                },
            }
        }
        dashboard_api._athena_client.get_query_results.return_value = {
            "ResultSet": {
                "ResultSetMetadata": {
                    "ColumnInfo": [
                        {"Name": "attempts", "Type": "bigint"},
                        {"Name": "failure_rate", "Type": "double"},
                        {"Name": "cold_start", "Type": "boolean"},
                    ]
                },
                "Rows": [
                    {
                        "Data": [
                            {"VarCharValue": "attempts"},
                            {"VarCharValue": "failure_rate"},
                            {"VarCharValue": "cold_start"},
                        ]
                    },
                    {
                        "Data": [
                            {"VarCharValue": "12"},
                            {"VarCharValue": "0.25"},
                            {"VarCharValue": "false"},
                        ]
                    },
                ],
            }
        }
        response = dashboard_api.handler(self.get_event(), None)
        body = self.response_body(response)
        self.assertEqual(body["rows"], [{"attempts": 12, "failure_rate": 0.25, "cold_start": False}])
        self.assertTrue(body["statistics"]["reused_previous_result"])

        dashboard_api._athena_client.get_query_execution.return_value["QueryExecution"][
            "WorkGroup"
        ] = "some-other-workgroup"
        response = dashboard_api.handler(self.get_event(), None)
        self.assertEqual(response["statusCode"], 404)

    def test_sanitizes_athena_failure_reason(self) -> None:
        dashboard_api._athena_client.get_query_execution.return_value = {
            "QueryExecution": {
                "QueryExecutionId": QUERY_ID,
                "Query": "-- dashboard-query:failures\nSELECT 1",
                "WorkGroup": "dashboard-workgroup",
                "Status": {
                    "State": "FAILED",
                    "StateChangeReason": "s3://private-bucket/path and full SQL",
                    "AthenaError": {"ErrorCategory": 2, "ErrorType": 1001, "Retryable": False},
                },
            }
        }
        response = dashboard_api.handler(self.get_event(), None)
        body = self.response_body(response)
        self.assertEqual(body["state"], "FAILED")
        self.assertNotIn("private-bucket", str(body))
        self.assertEqual(body["error"]["category"], 2)


if __name__ == "__main__":
    unittest.main()
