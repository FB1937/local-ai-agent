import json
import unittest
from pathlib import Path


class M34HoldoutDatasetTests(unittest.TestCase):
    def test_holdout_is_separate_balanced_and_independently_named(self):
        root = Path(__file__).resolve().parents[1]
        rows = [json.loads(line) for line in (root / "evaluation" / "multi_evidence_holdout.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        development = [json.loads(line) for line in (root / "evaluation" / "multi_evidence_cases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(rows), 24)
        expected = {"supported": 6, "contradicted": 6, "conflict": 6, "insufficient": 6}
        self.assertEqual({status: sum(row["expected_status"] == status for row in rows) for status in expected}, expected)
        self.assertTrue(all(row["case_id"].startswith("M34-H-") for row in rows))
        self.assertFalse({row["case_id"] for row in rows} & {row["case_id"] for row in development})
        allowed = {"none", "direct", "scope_mismatch", "temporal_mismatch", "condition_mismatch", "duplicate_not_independent", "freshness_uncertain"}
        for row in rows:
            self.assertIn(row["expected_conflict_type"], allowed)
            self.assertTrue(row["evidence"])


if __name__ == "__main__":
    unittest.main()
