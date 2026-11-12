import unittest

from claim_analysis import (
    analyze_claim_evidence,
    build_evidence_passages,
    extract_cited_claims,
    map_claims_to_cited_passages,
)


class ClaimAnalysisTests(unittest.TestCase):
    def test_extracts_sentence_claims_and_deduplicates_citations(self):
        claims = extract_cited_claims(
            "첫 번째 사실입니다. [S1] [S1] 두 번째 사실입니다. [S2]\n"
            "마지막 의견입니다."
        )
        self.assertEqual(len(claims), 3)
        self.assertEqual(claims[0].cited_source_ids, ["S1"])
        self.assertEqual(claims[1].cited_source_ids, ["S2"])
        self.assertEqual(claims[2].cited_source_ids, [])

    def test_builds_passages_with_source_provenance(self):
        passages = build_evidence_passages([{
            "source_id": "S1",
            "url": "https://example.org/source",
            "published": "2026-01-01",
            "content": "첫 번째 근거 문장입니다. 두 번째 근거 문장입니다.",
        }])
        self.assertEqual(len(passages), 2)
        self.assertEqual(passages[0].source_id, "S1")
        self.assertEqual(passages[0].location, "sentence:1")
        self.assertEqual(passages[0].published_at, "2026-01-01")
        self.assertEqual(passages[1].location, "sentence:2")

    def test_prefers_extracted_text_over_search_snippet(self):
        passages = build_evidence_passages([{
            "source_id": "S1",
            "url": "https://example.org/source",
            "content": "검색 결과 요약 문장입니다.",
            "extracted_text": "원문에서 추출한 근거 문장입니다.",
        }])
        self.assertEqual(len(passages), 1)
        self.assertIn("원문", passages[0].text)

    def test_maps_only_to_cited_sources_and_does_not_claim_support(self):
        claims = extract_cited_claims("주장입니다. [S1]")
        passages = build_evidence_passages([
            {"source_id": "S1", "url": "https://example.org/1",
             "content": "관련 근거일 수도 있는 문장입니다."},
            {"source_id": "S2", "url": "https://example.org/2",
             "content": "인용되지 않은 다른 출처입니다."},
        ])
        links, unmapped = map_claims_to_cited_passages(claims, passages)
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0].relation, None)
        self.assertEqual(unmapped, [])
        linked_passage_ids = {link.evidence_id for link in links}
        self.assertEqual(
            {p.source_id for p in passages if p.evidence_id in linked_passage_ids},
            {"S1"},
        )

    def test_unknown_citation_is_reported_as_unmapped(self):
        claims = extract_cited_claims("주장입니다. [S8]")
        links, unmapped = map_claims_to_cited_passages(claims, [])
        self.assertEqual(links, [])
        self.assertEqual(unmapped, ["S8"])

    def test_passage_limits_are_enforced(self):
        passages = build_evidence_passages([{
            "source_id": "S1",
            "url": "https://example.org/source",
            "content": "가. " * 100,
        }], max_passages_per_source=2, max_chars_per_passage=5)
        self.assertLessEqual(len(passages), 2)
        self.assertTrue(all(len(p.text) <= 5 for p in passages))

    def test_non_positive_limits_are_rejected(self):
        with self.assertRaises(ValueError):
            build_evidence_passages([], max_passages_per_source=0)


if __name__ == "__main__":
    unittest.main()
