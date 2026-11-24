import unittest
from unittest.mock import patch

from evaluation.evidence_reasoning import LABELS, load_dataset, score_predictions, build_analysis, evaluate


class EvidenceReasoningEvaluationTests(unittest.TestCase):
    def test_dataset_has_balanced_four_class_labels(self):
        cases = load_dataset()
        self.assertEqual(len(cases), 32)
        counts = {label: sum(case["expected_relation"] == label for case in cases)
                  for label in LABELS}
        self.assertEqual(counts, {
            "supports": 8, "contradicts": 8,
            "insufficient": 8, "irrelevant": 8,
        })

    def test_dataset_ids_are_unique_and_text_is_present(self):
        cases = load_dataset()
        self.assertEqual(len({case["id"] for case in cases}), len(cases))
        self.assertTrue(all(case["claim"].strip() and case["evidence"].strip()
                            for case in cases))

    def test_build_analysis_keeps_one_claim_passage_link_per_case(self):
        cases = load_dataset()
        analysis = build_analysis(cases)
        self.assertEqual(len(analysis.claims), 32)
        self.assertEqual(len(analysis.passages), 32)
        self.assertEqual(len(analysis.links), 32)
        self.assertEqual(
            {link.claim_id for link in analysis.links},
            {claim.claim_id for claim in analysis.claims},
        )

    def test_perfect_predictions_score_one(self):
        cases = load_dataset()
        predictions = {case["id"]: case["expected_relation"] for case in cases}
        result = score_predictions(cases, predictions)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["macro_f1"], 1.0)
        self.assertEqual(result["unassessed"], 0)

    def test_unassessed_predictions_count_as_errors(self):
        cases = load_dataset()
        predictions = {case["id"]: case["expected_relation"] for case in cases}
        predictions[cases[0]["id"]] = None
        result = score_predictions(cases, predictions)
        self.assertAlmostEqual(result["accuracy"], 31 / 32, places=4)
        self.assertEqual(result["unassessed"], 1)
        self.assertEqual(result["confusion_matrix"]["supports"]["unassessed"], 1)

    def test_wrong_class_updates_confusion_matrix_and_macro_f1(self):
        cases = load_dataset()
        predictions = {case["id"]: case["expected_relation"] for case in cases}
        predictions["ER-S01"] = "contradicts"
        result = score_predictions(cases, predictions)
        self.assertAlmostEqual(result["accuracy"], 31 / 32, places=4)
        self.assertEqual(result["confusion_matrix"]["supports"]["contradicts"], 1)
        self.assertLess(result["macro_f1"], 1.0)


    def test_live_runner_uses_existing_reasoner_with_mocked_transport(self):
        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"message": {"content": '{"relation":"supports","reason":"direct"}'}}

        cases = load_dataset()[:3]
        with patch("evaluation.evidence_reasoning.requests.post", return_value=FakeResponse()) as post:
            result = evaluate(cases, model="test-model", ollama_url="http://localhost:11434")
        self.assertEqual(post.call_count, 3)
        self.assertEqual(result["summary"]["total"], 3)
        self.assertEqual(result["summary"]["latency_ms"]["count"], 3)
        self.assertEqual(result["model"], "test-model")

    def test_duplicate_claim_text_keeps_per_case_latency_records(self):
        original = load_dataset()[0]
        cases = [dict(original), dict(original, id="ER-DUPLICATE")]

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"message": {"content": '{"relation":"supports","reason":"direct"}'}}

        with patch("evaluation.evidence_reasoning.requests.post", return_value=FakeResponse()) as post:
            result = evaluate(cases, model="test-model", ollama_url="http://localhost:11434")

        self.assertEqual(post.call_count, 2)
        self.assertEqual(result["summary"]["latency_ms"]["count"], 2)
        self.assertTrue(all(case["latency_ms"] is not None for case in result["cases"]))

    def test_latency_statistics_ignore_unmeasured_cases(self):
        cases = load_dataset()
        predictions = {case["id"]: case["expected_relation"] for case in cases}
        result = score_predictions(cases, predictions, {"ER-S01": 10.0, "ER-S02": 30.0})
        self.assertEqual(result["latency_ms"]["count"], 2)
        self.assertEqual(result["latency_ms"]["mean"], 20.0)
        self.assertEqual(result["latency_ms"]["median"], 20.0)
        self.assertEqual(result["latency_ms"]["p95"], 30.0)


if __name__ == "__main__":
    unittest.main()
