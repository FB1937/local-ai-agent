import unittest

from evaluation.run_consistency_audit_evaluation import CASES, EVIDENCE


class ConsistencyAuditEvaluationFixtureTests(unittest.TestCase):
    def test_fixture_contains_both_mismatch_directions_and_controls(self):
        self.assertEqual(len(CASES), 5)
        self.assertEqual(
            sum(not case["expected_consistent"] for case in CASES), 3
        )
        self.assertEqual(
            sum(case["expected_consistent"] for case in CASES), 2
        )
        self.assertEqual(
            {case["relation"] for case in CASES},
            {"supports", "contradicts"},
        )

    def test_c03_style_case_has_contradictory_reason_with_supports_enum(self):
        case = next(
            item for item in CASES
            if item["case_id"] == "AUDIT-MISMATCH-SUPPORTS"
        )
        self.assertFalse(case["expected_consistent"])
        self.assertEqual(case["relation"], "supports")
        self.assertIn("반박한다", case["analyst_reason"])
        self.assertEqual(len(EVIDENCE), 2)


if __name__ == "__main__":
    unittest.main()
