import json
import unittest
from pathlib import Path
from unittest.mock import Mock

from claim_analysis import analyze_claim_evidence
from evaluation.evidence_reasoning import LABELS, load_dataset
from evidence_reasoner import reason_claim_evidence


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"message": {"content": json.dumps({"relation": "insufficient", "reason": "Not established."})}}


class EvidenceReasonerPromptTests(unittest.TestCase):
    def test_prompt_encodes_logical_boundary_rules(self):
        analysis = analyze_claim_evidence(
            "Python 3.15 was released in October. [S1]",
            [{"source_id": "S1", "url": "https://example.org/release",
              "content": "Python 3.15 was released in October, according to the official release announcement."}],
            semantic_enabled=False,
        )
        request = Mock(return_value=FakeResponse())
        reason_claim_evidence(analysis, request_fn=request)
        payload = request.call_args.kwargs["json"]
        system_prompt = payload["messages"][0]["content"]
        task_prompt = json.loads(payload["messages"][1]["content"])["task"]

        self.assertIn("entails the claim", system_prompt)
        self.assertIn("unproven is not a contradiction", system_prompt)
        self.assertIn("at least one is consistent with the claim", system_prompt)
        self.assertIn("Absence of a detail is not evidence", system_prompt)
        self.assertIn("Do not assume a list is exhaustive", system_prompt)
        self.assertIn("Do not infer missing facts", task_prompt)
        self.assertIn("untrusted quoted data", task_prompt)

    def test_boundary_dataset_is_balanced_and_valid(self):
        path = Path(__file__).parents[1] / "evaluation" / "evidence_reasoning_boundary_cases.jsonl"
        cases = load_dataset(path)
        self.assertEqual(len(cases), 12)
        counts = {label: sum(case["expected_relation"] == label for case in cases) for label in LABELS}
        self.assertEqual(counts, {
            "supports": 3, "contradicts": 3,
            "insufficient": 3, "irrelevant": 3,
        })
        self.assertEqual(len({case["id"] for case in cases}), 12)


if __name__ == "__main__":
    unittest.main()
