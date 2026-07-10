from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


LAMBDA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAMBDA_ROOT))

import ingest  # noqa: E402
from test_event_contract import event_payload  # noqa: E402


class IngestTests(unittest.TestCase):
    def setUp(self) -> None:
        ingest._secret_cache = "test-secret"
        ingest._sqs_client = Mock()
        os.environ["TELEMETRY_QUEUE_URL"] = "https://sqs.example/queue.fifo"

    def api_event(self, body: bytes, *, timestamp: int = 1_750_000_000) -> dict[str, object]:
        signature = hmac.new(
            b"test-secret", str(timestamp).encode() + b"." + body, hashlib.sha256
        ).hexdigest()
        return {
            "headers": {
                "X-WDIRL-Timestamp": str(timestamp),
                "X-WDIRL-Signature": signature,
            },
            "body": body.decode(),
            "isBase64Encoded": False,
        }

    @patch("ingest.time.time", return_value=1_750_000_010)
    def test_accepts_signed_event_and_enqueues_fifo_message(self, _now: Mock) -> None:
        body = json.dumps(event_payload(), separators=(",", ":")).encode()
        response = ingest.handler(self.api_event(body), None)
        self.assertEqual(response["statusCode"], 202)
        kwargs = ingest._sqs_client.send_message.call_args.kwargs
        self.assertEqual(kwargs["MessageGroupId"], event_payload()["attempt_id"])
        self.assertEqual(kwargs["MessageDeduplicationId"], event_payload()["event_id"])

    @patch("ingest.time.time", return_value=1_750_000_010)
    def test_rejects_bad_signature_without_enqueuing(self, _now: Mock) -> None:
        body = json.dumps(event_payload()).encode()
        request = self.api_event(body)
        request["headers"]["X-WDIRL-Signature"] = "0" * 64  # type: ignore[index]
        response = ingest.handler(request, None)
        self.assertEqual(response["statusCode"], 401)
        ingest._sqs_client.send_message.assert_not_called()

    @patch("ingest.time.time", return_value=1_750_001_000)
    def test_rejects_stale_signature(self, _now: Mock) -> None:
        body = json.dumps(event_payload()).encode()
        response = ingest.handler(self.api_event(body), None)
        self.assertEqual(response["statusCode"], 401)


if __name__ == "__main__":
    unittest.main()
