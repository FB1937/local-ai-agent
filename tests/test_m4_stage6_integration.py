"""Regression tests for the M4 Stage-6 M3 integration."""
import os
import unittest
from unittest.mock import patch

from response_generator import generate_response
from claim_synthesis import ClaimEvidenceSynthesis, ClaimSynthesisReport
from evidence_reasoner import EvidenceReasoningResult
from semantic_claim_synthesis import SemanticClaimSynthesisReport
from m3_response_grounding import evaluate_response_claims
from claim_analysis import Claim, ClaimAnalysisResult


class M4Stage6IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.source = {
            "source_id": "S1",
            "title": "Test source",
            "url": "https://example.com/source",
            "content": "The system launched in 2025.",
            "extracted_text": "The system launched in 2025.",
        }
        self.generation = {
            "ok": True,
            "incomplete": False,
            "answer": "시스템은 2025년에 출시되었습니다. [S1]",
        }
        self.env = patch.dict(os.environ, {"ENABLE_M3_CLAIM_GROUNDING": "1",
                                           "ENABLE_SEMANTIC_GROUNDING": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_search_route_runs_m3_and_suppresses_legacy_semantic_verifier(self):
        review = {
            "enabled": True, "status": "assessed", "claims": [],
            "claim_count": 1, "blocking_claim_count": 0,
        }
        with patch("response_generator.evaluate_response_claims", return_value=review) as m3, \
             patch("response_generator.verify_cited_claims") as legacy:
            result = generate_response(
                "언제 출시됐어?", {"results": [self.source], "assessment": {}},
                "SEARCH", generation_fn=lambda *a, **k: self.generation,
            )
        m3.assert_called_once()
        legacy.assert_not_called()
        self.assertEqual(result["claim_evidence_review"]["status"], "assessed")
        self.assertFalse(result["semantic_verification"]["enabled"])
        self.assertIn("실행하지 않습니다", result["semantic_verification"]["note"])
        self.assertEqual(result["answer"], self.generation["answer"])

    def test_direct_route_skips_m3(self):
        direct_generation = {**self.generation, "answer": "안녕하세요."}
        with patch("response_generator.evaluate_response_claims") as m3:
            result = generate_response(
                "안녕", {"results": [], "assessment": {}}, "DIRECT",
                generation_fn=lambda *a, **k: direct_generation,
            )
        m3.assert_not_called()
        self.assertEqual(result["claim_evidence_review"]["status"], "skipped_direct")
        self.assertEqual(result["answer"], direct_generation["answer"])

    def test_contradicted_claim_blocks_entire_generated_answer(self):
        review = {
            "enabled": True, "status": "blocked",
            "claims": [{
                "claim_id": "C1", "status": "contradicted",
                "reason": "출처는 다른 연도를 명시합니다.",
            }],
        }
        with patch("response_generator.evaluate_response_claims", return_value=review):
            result = generate_response(
                "언제 출시됐어?", {"results": [self.source], "assessment": {}},
                "SEARCH", generation_fn=lambda *a, **k: self.generation,
            )
        self.assertEqual(result["answer"], "")
        self.assertIn("[답변 보류]", result["display_answer"])

    def test_m3_error_fails_closed(self):
        review = {"enabled": True, "status": "error", "claims": [],
                  "error": "ConnectionError: unavailable"}
        with patch("response_generator.evaluate_response_claims", return_value=review):
            result = generate_response(
                "언제 출시됐어?", {"results": [self.source], "assessment": {}},
                "HYBRID", generation_fn=lambda *a, **k: self.generation,
            )
        self.assertEqual(result["answer"], "")
        self.assertIn("M3 검토 오류", result["display_answer"])

    def test_incomplete_generation_does_not_call_m3(self):
        incomplete = {**self.generation, "incomplete": True}
        with patch("response_generator.evaluate_response_claims") as m3:
            result = generate_response(
                "언제 출시됐어?", {"results": [self.source], "assessment": {}},
                "SEARCH", generation_fn=lambda *a, **k: incomplete,
            )
        m3.assert_not_called()
        self.assertEqual(
            result["claim_evidence_review"]["status"],
            "skipped_generation_incomplete",
        )

    def test_adapter_preserves_all_stage_counters(self):
        baseline = ClaimSynthesisReport(claims=[
            ClaimEvidenceSynthesis(
                claim_id="C1", status="supported",
                supporting_evidence_ids=["E1"], reason="direct support",
            )
        ])
        reasoning = EvidenceReasoningResult(
            links=[], assessed_count=1, unassessed_count=2, skipped_count=3,
        )
        semantic = SemanticClaimSynthesisReport(
            claims=[], assessed_count=1, unassessed_count=2, skipped_count=3,
        )
        with patch("m3_response_grounding.analyze_claim_evidence", return_value=object()), \
             patch("m3_response_grounding.reason_claim_evidence", return_value=reasoning), \
             patch("m3_response_grounding.synthesize_claim_evidence", return_value=baseline), \
             patch("m3_response_grounding.synthesize_claim_evidence_semantically",
                   return_value=(baseline, semantic)):
            result = evaluate_response_claims("Claim [S1]", [self.source])
        self.assertEqual(result["status"], "assessed")
        self.assertEqual(result["pairwise"]["unassessed_count"], 2)
        self.assertEqual(result["pairwise"]["skipped_count"], 3)
        self.assertEqual(result["semantic"]["unassessed_count"], 2)
        self.assertEqual(result["semantic"]["skipped_count"], 3)

    def test_adapter_reports_claim_citations_passages_and_source_text_lengths(self):
        answer = "The system launched in 2025 after a long public testing period. [S1]"
        source = {
            **self.source,
            "content": "The system launched in 2025 after a long public testing period.",
            "extracted_text": "The system launched in 2025 after a long public testing period.",
            "extraction_status": "success",
        }
        baseline = ClaimSynthesisReport(claims=[
            ClaimEvidenceSynthesis(
                claim_id="C1", status="insufficient", reason="not classified",
            )
        ])
        with patch(
            "m3_response_grounding.reason_claim_evidence",
            return_value=EvidenceReasoningResult(links=[]),
        ), patch(
            "m3_response_grounding.synthesize_claim_evidence",
            return_value=baseline,
        ), patch(
            "m3_response_grounding.synthesize_claim_evidence_semantically",
            return_value=(baseline, SemanticClaimSynthesisReport(claims=[])),
        ):
            result = evaluate_response_claims(answer, [source])

        retrieval = result["retrieval"]
        self.assertEqual(retrieval["claim_count"], 1)
        self.assertEqual(retrieval["claims_with_citations"], 1)
        self.assertEqual(retrieval["passage_count"], 1)
        self.assertEqual(retrieval["claim_evidence_link_count"], 1)
        self.assertEqual(retrieval["claims_without_evidence"], [])
        self.assertEqual(retrieval["claims"][0]["cited_source_ids"], ["S1"])
        self.assertEqual(retrieval["claims"][0]["passage_count"], 1)
        self.assertEqual(retrieval["sources"][0]["extraction_status"], "success")
        self.assertGreater(retrieval["sources"][0]["extracted_text_chars"], 40)

    def test_adapter_exception_returns_error_not_success(self):
        with patch("m3_response_grounding.analyze_claim_evidence",
                   side_effect=RuntimeError("broken")):
            result = evaluate_response_claims("Claim [S1]", [self.source])
        self.assertEqual(result["status"], "error")
        self.assertIn("RuntimeError", result["error"])


if __name__ == "__main__":
    unittest.main()
