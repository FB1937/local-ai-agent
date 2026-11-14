import unittest
from pathlib import Path

from evaluation.run_claim_synthesis_evaluation import evaluate_dataset


class ClaimSynthesisEvaluationTests(unittest.TestCase):
    def test_deterministic_baseline_exposes_joint_entailment_gap(self):
        report = evaluate_dataset(Path(__file__).resolve().parents[1] / "evaluation" / "multi_evidence_cases.jsonl")
        self.assertEqual(report["total"], 32)
        self.assertEqual([item["case_id"] for item in report["mismatches"]], ["ME-S08"])
        self.assertEqual(report["correct"], 31)
        self.assertEqual(report["accuracy"], 31 / 32)


if __name__ == "__main__":
    unittest.main()
