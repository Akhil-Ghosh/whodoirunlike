from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
LAMBDA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAMBDA_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import event_contract  # noqa: E402
from whodoirunlike import processing_telemetry  # noqa: E402


class SharedSchemaTests(unittest.TestCase):
    def test_schema_enums_match_both_python_contracts(self) -> None:
        schema = json.loads(
            (REPO_ROOT / "schemas" / "processing-event-v1.schema.json").read_text(
                encoding="utf-8"
            )
        )

        schema_event_types = set(schema["properties"]["event_type"]["enum"])
        schema_stages = set(schema["$defs"]["stage"]["enum"])
        schema_spans = set(schema["$defs"]["span"]["enum"])

        self.assertEqual(schema_event_types, set(event_contract.EVENT_TYPES))
        self.assertEqual(schema_event_types, set(processing_telemetry.EVENT_TYPES))
        self.assertEqual(schema_stages, set(event_contract.PIPELINE_STAGES))
        self.assertEqual(schema_stages, set(processing_telemetry.PIPELINE_STAGES))
        self.assertEqual(schema_spans, set(event_contract.PROCESSING_SPANS))
        self.assertEqual(schema_spans, set(processing_telemetry.PROCESSING_SPANS))
        self.assertFalse(schema["additionalProperties"])


if __name__ == "__main__":
    unittest.main()
