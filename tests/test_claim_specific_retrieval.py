import unittest

from claim_analysis import analyze_claim_evidence, extract_cited_claims, retrieve_claim_evidence_passages


def source(source_id, url, content):
    return {"source_id": source_id, "url": url, "content": content}


class ClaimSpecificRetrievalTests(unittest.TestCase):
    def test_retrieves_relevant_passages_per_claim_and_preserves_provenance(self):
        text = (
            "Python 3.15 improves parser diagnostics and reports clearer syntax errors. "
            "The release schedule lists the planned publication date as October 2026. "
            "The standard library also receives several maintenance updates."
        )
        answer = (
            "Python 3.15 improves parser diagnostics. [S1]\n"
            "Python 3.15 is planned for release in October 2026. [S1]"
        )
        result = analyze_claim_evidence(answer, [source("S1", "https://example.org/python", text)],
                                        semantic_enabled=False)
        by_claim = {cid: [p for p in result.passages if p.claim_id == cid] for cid in ("C1", "C2")}
        self.assertTrue(any("parser diagnostics" in p.text for p in by_claim["C1"]))
        self.assertTrue(any("October 2026" in p.text for p in by_claim["C2"]))
        self.assertTrue(all(p.source_url == "https://example.org/python" for p in result.passages))
        self.assertTrue(all(p.retrieval_method == "lexical_fallback" for p in result.passages))

    def test_uncited_sources_never_enter_claim_retrieval(self):
        claims = extract_cited_claims("Python improves parser diagnostics. [S1]")
        passages, unmapped = retrieve_claim_evidence_passages(
            claims,
            [
                source("S1", "https://example.org/one",
                       "Python improves parser diagnostics and reports clearer syntax errors."),
                source("S2", "https://example.org/two",
                       "Python improves parser diagnostics and reports clearer syntax errors."),
            ],
            semantic_enabled=False,
        )
        self.assertTrue(passages)
        self.assertEqual({p.source_id for p in passages}, {"S1"})
        self.assertEqual(unmapped, [])

    def test_unmapped_citation_and_no_relevant_passage_are_explicit(self):
        result = analyze_claim_evidence(
            "A claim about an unrelated subject with enough words to be a claim. [S9]",
            [source("S1", "https://example.org/one",
                    "Python improves parser diagnostics and reports clearer syntax errors.")],
            semantic_enabled=False,
        )
        self.assertEqual(result.unmapped_source_ids, ["S9"])
        self.assertEqual(result.claims_without_evidence, ["C1"])
        self.assertEqual(result.links, [])

    def test_uncited_claim_does_not_create_evidence_links(self):
        result = analyze_claim_evidence(
            "An uncited statement with enough words to form a candidate claim.",
            [source("S1", "https://example.org/one",
                    "Python improves parser diagnostics and reports clearer syntax errors.")],
            semantic_enabled=False,
        )
        self.assertEqual(result.passages, [])
        self.assertEqual(result.links, [])
        self.assertEqual(result.claims_without_evidence, ["C1"])

    def test_per_source_passage_and_character_budgets_are_applied(self):
        content = " ".join(
            f"Python 3.15 feature detail number {i} improves diagnostics and error reporting."
            for i in range(12)
        )
        result = analyze_claim_evidence(
            "Python 3.15 improves diagnostics and error reporting. [S1]",
            [source("S1", "https://example.org/one", content)],
            max_passages_per_source=2, max_chars_per_source=180, semantic_enabled=False,
        )
        self.assertLessEqual(len(result.passages), 2)
        self.assertLessEqual(sum(len(p.text) for p in result.passages), 180)

    def test_semantic_failure_falls_back_to_lexical(self):
        def failing_embedder(_):
            raise TimeoutError("embedding timeout")
        result = analyze_claim_evidence(
            "Python improves parser diagnostics and reports clearer syntax errors. [S1]",
            [source("S1", "https://example.org/one",
                    "Python improves parser diagnostics and reports clearer syntax errors.")],
            semantic_enabled=True, embedder=failing_embedder,
        )
        self.assertTrue(result.passages)
        self.assertEqual(result.passages[0].retrieval_method, "lexical_fallback")


if __name__ == "__main__":
    unittest.main()
