import unittest
from evaluation.cases import build_cases
from evaluation.run_evaluation import run_evaluation


class M2EvaluationDatasetTests(unittest.TestCase):
    def test_dataset_has_fifty_cases_across_five_categories(self):
        cases = build_cases()
        self.assertEqual(len(cases), 50)
        self.assertEqual({case["category"] for case in cases},
                         {"direct", "search", "hybrid", "followup", "edge"})

    def test_all_deterministic_cases_match_expected_results(self):
        result = run_evaluation()
        self.assertEqual(result["total"], 50)
        self.assertEqual(result["failed"], 0, result["cases"])
        self.assertEqual(result["pass_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
