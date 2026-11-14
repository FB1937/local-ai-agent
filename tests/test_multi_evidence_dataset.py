import json
import unittest
from collections import Counter
from pathlib import Path


DATASET = Path(__file__).resolve().parents[1] / "evaluation" / "multi_evidence_cases.jsonl"


class MultiEvidenceDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_dataset_has_unique_ids_and_balanced_aggregate_labels(self):
        ids = [row["case_id"] for row in self.rows]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(self.rows), 32)
        self.assertEqual(Counter(row["expected_status"] for row in self.rows), {
            "supported": 8, "contradicted": 8, "conflict": 8, "insufficient": 8
        })

    def test_every_case_has_multiple_provenance_preserving_evidence_items(self):
        for row in self.rows:
            with self.subTest(case=row["case_id"]):
                self.assertTrue(row["claim"])
                self.assertGreaterEqual(len(row["evidence"]), 2)
                evidence_ids = [item["evidence_id"] for item in row["evidence"]]
                self.assertEqual(len(evidence_ids), len(set(evidence_ids)))
                for item in row["evidence"]:
                    self.assertTrue(item["source_id"])
                    self.assertTrue(item["source_group"])
                    self.assertIn(item["relation"], {"supports", "contradicts", "insufficient", "irrelevant"})
                    self.assertTrue(item["text"])

    def test_conflict_labels_have_a_diagnostic_type(self):
        for row in self.rows:
            if row["expected_status"] == "conflict":
                self.assertEqual(row["expected_conflict_type"], "direct")
            if row["expected_status"] == "insufficient":
                self.assertNotEqual(row["expected_conflict_type"], "direct")


if __name__ == "__main__":
    unittest.main()
