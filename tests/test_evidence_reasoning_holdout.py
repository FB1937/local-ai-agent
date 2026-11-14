import unittest
from pathlib import Path

from evaluation.evidence_reasoning import LABELS, load_dataset


class EvidenceReasoningHoldoutTests(unittest.TestCase):
    def test_holdout_is_balanced_and_disjoint_from_development_sets(self):
        root = Path(__file__).parents[1] / "evaluation"
        holdout = load_dataset(root / "evidence_reasoning_holdout.jsonl")
        development = load_dataset(root / "evidence_reasoning_cases.jsonl")
        boundary = load_dataset(root / "evidence_reasoning_boundary_cases.jsonl")

        self.assertEqual(len(holdout), 24)
        counts = {label: sum(case["expected_relation"] == label for case in holdout)
                  for label in LABELS}
        self.assertEqual(counts, {
            "supports": 6, "contradicts": 6,
            "insufficient": 6, "irrelevant": 6,
        })
        all_prior = development + boundary
        self.assertFalse({case["id"] for case in holdout} &
                         {case["id"] for case in all_prior})
        self.assertFalse({case["claim"] for case in holdout} &
                         {case["claim"] for case in all_prior})
        self.assertFalse({case["evidence"] for case in holdout} &
                         {case["evidence"] for case in all_prior})
        self.assertEqual(len({case["id"] for case in holdout}), 24)


if __name__ == "__main__":
    unittest.main()
