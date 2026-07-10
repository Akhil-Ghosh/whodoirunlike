from __future__ import annotations

import gzip
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


LAMBDA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAMBDA_ROOT))

import consumer  # noqa: E402
from test_event_contract import event_payload  # noqa: E402


class ConsumerTests(unittest.TestCase):
    def setUp(self) -> None:
        consumer._s3_client = Mock()
        os.environ["EVENT_BUCKET"] = "event-lake"

    def test_writes_raw_and_flattened_gzip_batches(self) -> None:
        response = consumer.handler(
            {"Records": [{"messageId": "m1", "body": json.dumps(event_payload())}]},
            SimpleNamespace(aws_request_id="request-1"),
        )
        self.assertEqual(response, {"batchItemFailures": []})
        self.assertEqual(consumer._s3_client.put_object.call_count, 2)
        calls = consumer._s3_client.put_object.call_args_list
        self.assertTrue(calls[0].kwargs["Key"].startswith("raw/event_date=2026-07-09/event_hour=21/"))
        self.assertTrue(calls[1].kwargs["Key"].startswith("validated/event_date=2026-07-09/event_hour=21/"))
        self.assertEqual(calls[1].kwargs["Tagging"], "event-class=fact")
        validated = json.loads(gzip.decompress(calls[1].kwargs["Body"]).decode())
        self.assertEqual(validated["stage"], "runner_mask")
        self.assertEqual(validated["gpu_type"], "A40")

    def test_progress_rows_are_tagged_for_shorter_retention(self) -> None:
        progress = event_payload(
            event_type="progress_sampled",
            span="inference",
            status="running",
        )
        response = consumer.handler(
            {"Records": [{"messageId": "m-progress", "body": json.dumps(progress)}]},
            SimpleNamespace(aws_request_id="request-progress"),
        )

        self.assertEqual(response, {"batchItemFailures": []})
        self.assertEqual(consumer._s3_client.put_object.call_count, 2)
        validated_call = consumer._s3_client.put_object.call_args_list[1]
        self.assertEqual(validated_call.kwargs["Tagging"], "event-class=progress")

    def test_returns_partial_batch_failure_for_invalid_event(self) -> None:
        invalid = event_payload(schema_version=99)
        response = consumer.handler(
            {"Records": [{"messageId": "bad", "body": json.dumps(invalid)}]},
            SimpleNamespace(aws_request_id="request-2"),
        )
        self.assertEqual(response, {"batchItemFailures": [{"itemIdentifier": "bad"}]})
        consumer._s3_client.put_object.assert_not_called()

    def test_returns_partial_failure_when_s3_write_fails(self) -> None:
        consumer._s3_client.put_object.side_effect = RuntimeError("S3 unavailable")
        response = consumer.handler(
            {"Records": [{"messageId": "m1", "body": json.dumps(event_payload())}]},
            SimpleNamespace(aws_request_id="request-3"),
        )
        self.assertEqual(response, {"batchItemFailures": [{"itemIdentifier": "m1"}]})


if __name__ == "__main__":
    unittest.main()
