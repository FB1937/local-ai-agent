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

    def test_conflict_type_mismatches_are_reported_separately(self):
        report = evaluate_dataset(Path(__file__).resolve().parents[1] / "evaluation" / "multi_evidence_cases.jsonl")
        self.assertEqual(report["conflict_type_correct"], 31)
        self.assertEqual(report["conflict_type_accuracy"], 31 / 32)
        self.assertEqual(
            report["conflict_type_mismatches"],
            [{
                "case_id": "ME-I04",
                "expected": "duplicate_not_independent",
                "actual": "none",
                "status": "insufficient",
                "reason": "직접 지지 또는 반박 근거가 없어 정보 부족으로 남깁니다.",
            }],
        )
        self.assertEqual(report["mismatches"][0]["case_id"], "ME-S08")


if __name__ == "__main__":
    unittest.main()
