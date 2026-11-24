import unittest

from claim_analysis import Claim, ClaimAnalysisResult, ClaimEvidenceLink, EvidencePassage
from claim_normalization import normalize_claim_analysis


class ClaimNormalizationTests(unittest.TestCase):
    def _analysis(self):
        claims = [
            Claim(claim_id="C1", text="이 답변에서는 주요 내용을 정리합니다. [S1]"),
            Claim(claim_id="C2", text="## 주요 변경 사항"),
            Claim(claim_id="C3", text="Python 3.15.2 was released on 2026-10-09. [S2]"),
            Claim(claim_id="C4", text="가상환경은 프로젝트별 패키지를 분리합니다. [S3]"),
        ]
        passages = [
            EvidencePassage(
                evidence_id=f"E{i}", source_id=f"S{i}", claim_id=f"C{i}",
                text=f"Evidence passage {i}", location="sentence:1",
            )
            for i in range(1, 5)
        ]
        links = [
            ClaimEvidenceLink(claim_id=f"C{i}", evidence_id=f"E{i}")
            for i in range(1, 5)
        ]
        return ClaimAnalysisResult(
            claims=claims, passages=passages, links=links,
        )

    def test_excludes_obvious_headings_and_answer_scaffolding(self):
        normalized, selected, skipped, excluded = normalize_claim_analysis(
            self._analysis(), max_links=20,
        )
        self.assertEqual([claim.claim_id for claim in normalized.claims], ["C3", "C4"])
        self.assertEqual(
            {item["reason"] for item in excluded},
            {"markdown_heading", "answer_scaffolding"},
        )
        self.assertEqual([link.claim_id for link in selected], ["C3", "C4"])
        self.assertEqual(skipped, [])

    def test_marks_time_sensitive_numeric_claims_high_priority(self):
        normalized, _, _, _ = normalize_claim_analysis(self._analysis())
        priorities = {claim.claim_id: claim.importance for claim in normalized.claims}
        self.assertEqual(priorities, {"C3": "high", "C4": "normal"})

    def test_budget_prioritizes_high_importance_and_remains_fair(self):
        claims = [
            Claim(claim_id="C1", text="General conceptual statement."),
            Claim(claim_id="C2", text="The service costs 20 dollars in 2026."),
            Claim(claim_id="C3", text="Another conceptual statement."),
        ]
        passages = []
        links = []
        for claim_index, claim in enumerate(claims, 1):
            for offset in range(2):
                evidence_id = f"E{(claim_index - 1) * 2 + offset + 1}"
                passages.append(EvidencePassage(
                    evidence_id=evidence_id, source_id=f"S{claim_index}",
                    claim_id=claim.claim_id, text=f"Evidence {evidence_id}",
                    location="sentence:1",
                ))
                links.append(ClaimEvidenceLink(
                    claim_id=claim.claim_id, evidence_id=evidence_id,
                ))
        analysis = ClaimAnalysisResult(claims=claims, passages=passages, links=links)
        normalized, selected, skipped, _ = normalize_claim_analysis(analysis, max_links=3)
        self.assertEqual(
            [(link.claim_id, link.evidence_id) for link in selected],
            [("C2", "E3"), ("C1", "E1"), ("C3", "E5")],
        )
        self.assertEqual(len(skipped), 3)
        self.assertEqual(len(normalized.links), 6)

    def test_keeps_unselected_links_in_graph_for_unassessed_synthesis(self):
        normalized, selected, skipped, _ = normalize_claim_analysis(
            self._analysis(), max_links=1,
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(len(skipped), 1)
        self.assertEqual(len(normalized.links), 2)
        self.assertEqual(
            {(link.claim_id, link.evidence_id) for link in normalized.links},
            {("C3", "E3"), ("C4", "E4")},
        )

    def test_rejects_non_positive_budget(self):
        with self.assertRaises(ValueError):
            normalize_claim_analysis(self._analysis(), max_links=0)


if __name__ == "__main__":
    unittest.main()
