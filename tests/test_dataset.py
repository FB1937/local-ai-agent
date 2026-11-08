import json
import unittest
from collections import Counter
from pathlib import Path

from router import VALID_ROUTES

DATASET = Path(__file__).resolve().parents[1] / "test_cases" / "routing_cases.json"


class RoutingDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads(DATASET.read_text(encoding="utf-8"))

    def test_case_ids_are_unique(self):
        ids = [case["id"] for case in self.cases]
        self.assertEqual(len(ids), len(set(ids)))

    def test_all_expected_routes_are_valid(self):
        self.assertTrue(all(case["expected_route"] in VALID_ROUTES for case in self.cases))

    def test_dev_and_holdout_are_balanced(self):
        self.assertEqual(len(self.cases), 60)
        for split, expected_size in (("dev", 15), ("holdout", 45)):
            subset = [case for case in self.cases if case["split"] == split]
            self.assertEqual(len(subset), expected_size)
            self.assertEqual(Counter(case["expected_route"] for case in subset), {
                "DIRECT": expected_size // 3,
                "SEARCH": expected_size // 3,
                "HYBRID": expected_size // 3,
            })


if __name__ == "__main__":
    unittest.main()
