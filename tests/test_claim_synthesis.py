import unittest

from claim_analysis import Claim, ClaimAnalysisResult, ClaimEvidenceLink, EvidencePassage
from claim_synthesis import synthesize_claim_evidence


def make_analysis(relations, *, scopes=None, times=None, groups=None):
    claim = Claim(claim_id="C1", text="Claim under review")
    passages = []
    links = []
    for index, relation in enumerate(relations, 1):
        evidence_id = f"E{index}"
        passages.append(EvidencePassage(
            evidence_id=evidence_id,
            source_id=f"S{index}",
            source_group=(groups or {}).get(index),
            text=f"Evidence text {index}",
            location=f"sentence:{index}",
            scope=(scopes or {}).get(index),
            valid_time_start=(times or {}).get(index, (None, None))[0],
            valid_time_end=(times or {}).get(index, (None, None))[1],
        ))
        links.append(ClaimEvidenceLink(claim_id="C1", evidence_id=evidence_id, relation=relation))
    return ClaimAnalysisResult(claims=[claim], passages=passages, links=links)


class ClaimSynthesisTests(unittest.TestCase):
    def test_single_direct_support_produces_supported(self):
        result = synthesize_claim_evidence(make_analysis(["supports"])).claims[0]
        self.assertEqual(result.status, "supported")
        self.assertEqual(result.supporting_evidence_ids, ["E1"])

    def test_single_direct_contradiction_produces_contradicted(self):
        result = synthesize_claim_evidence(make_analysis(["contradicts"])).claims[0]
        self.assertEqual(result.status, "contradicted")

    def test_same_scope_support_and_contradiction_remain_conflict(self):
        result = synthesize_claim_evidence(make_analysis(["supports", "contradicts"])).claims[0]
        self.assertEqual(result.status, "conflict")
        self.assertEqual(result.conflict_type, "direct")
        self.assertEqual(result.supporting_evidence_ids, ["E1"])
        self.assertEqual(result.contradicting_evidence_ids, ["E2"])

    def test_explicit_scope_mismatch_is_not_direct_conflict(self):
        result = synthesize_claim_evidence(make_analysis(
            ["supports", "contradicts"], scopes={1: "device body", 2: "battery"}
        )).claims[0]
        self.assertEqual(result.status, "insufficient")
        self.assertEqual(result.conflict_type, "scope_mismatch")

    def test_non_overlapping_valid_times_are_not_direct_conflict(self):
        result = synthesize_claim_evidence(make_analysis(
            ["supports", "contradicts"],
            times={1: ("2024-01-01", "2024-12-31"), 2: ("2025-01-01", "2025-12-31")}
        )).claims[0]
        self.assertEqual(result.status, "insufficient")
        self.assertEqual(result.conflict_type, "temporal_mismatch")

    def test_overlapping_valid_times_remain_conflict(self):
        result = synthesize_claim_evidence(make_analysis(
            ["supports", "contradicts"],
            times={1: ("2024-01-01", "2025-12-31"), 2: ("2025-01-01", "2026-12-31")}
        )).claims[0]
        self.assertEqual(result.status, "conflict")

    def test_same_source_duplicate_does_not_inflate_independence(self):
        analysis = make_analysis(["supports", "supports"], groups={1: "same-origin", 2: "same-origin"})
        analysis.passages[1].text = analysis.passages[0].text
        result = synthesize_claim_evidence(analysis).claims[0]
        self.assertEqual(result.status, "supported")
        self.assertEqual(result.independent_support_group_count, 1)
        self.assertEqual(len(result.duplicate_groups), 1)

    def test_duplicate_passages_with_inconsistent_labels_fail_closed(self):
        analysis = make_analysis(["supports", "contradicts"], groups={1: "same-origin", 2: "same-origin"})
        analysis.passages[1].text = analysis.passages[0].text
        result = synthesize_claim_evidence(analysis).claims[0]
        self.assertEqual(result.status, "unassessed")
        self.assertEqual(set(result.unassessed_evidence_ids), {"E1", "E2"})

    def test_any_unassessed_link_fails_closed(self):
        result = synthesize_claim_evidence(make_analysis(["supports", None])).claims[0]
        self.assertEqual(result.status, "unassessed")
        self.assertEqual(result.unassessed_evidence_ids, ["E2"])

    def test_only_weak_or_irrelevant_evidence_is_insufficient(self):
        result = synthesize_claim_evidence(make_analysis(["insufficient", "irrelevant"])).claims[0]
        self.assertEqual(result.status, "insufficient")

    def test_no_evidence_is_insufficient_not_unassessed(self):
        analysis = ClaimAnalysisResult(claims=[Claim(claim_id="C1", text="No evidence")])
        result = synthesize_claim_evidence(analysis).claims[0]
        self.assertEqual(result.status, "insufficient")

    def test_missing_passage_for_existing_link_is_unassessed(self):
        analysis = ClaimAnalysisResult(
            claims=[Claim(claim_id="C1", text="Claim")],
            links=[ClaimEvidenceLink(claim_id="C1", evidence_id="E9", relation="supports")],
        )
        result = synthesize_claim_evidence(analysis).claims[0]
        self.assertEqual(result.status, "unassessed")
        self.assertEqual(result.unassessed_evidence_ids, ["E9"])

    def test_pairwise_insufficient_is_not_promoted_by_majority_or_count(self):
        result = synthesize_claim_evidence(make_analysis(["insufficient", "insufficient", "insufficient"])).claims[0]
        self.assertEqual(result.status, "insufficient")

    def test_independent_source_groups_are_counted_separately(self):
        result = synthesize_claim_evidence(make_analysis(
            ["supports", "supports"], groups={1: "origin-A", 2: "origin-B"}
        )).claims[0]
        self.assertEqual(result.independent_support_group_count, 2)

    def test_input_analysis_is_not_mutated(self):
        analysis = make_analysis(["supports"])
        before = analysis.model_dump()
        synthesize_claim_evidence(analysis)
        self.assertEqual(analysis.model_dump(), before)


if __name__ == "__main__":
    unittest.main()
