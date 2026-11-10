import unittest
from unittest.mock import patch
from semantic_verifier import verify_cited_claims


class FakeResponse:
    def raise_for_status(self):
        pass
    def json(self):
        return {"message": {"content": '{"assessment":"supported","reason":"원문에 같은 내용이 있습니다."}'}}


class SemanticVerifierPrototypeTests(unittest.TestCase):
    def test_only_cited_source_is_sent_and_result_stays_needs_review(self):
        sources = [
            {"source_id": "S1", "url": "https://example.org/1", "title": "근거",
             "content": "서비스는 2026년에 출시되었습니다."},
            {"source_id": "S2", "url": "https://example.org/2", "title": "무관",
             "content": "다른 내용"},
        ]
        with patch("semantic_verifier.requests.post", return_value=FakeResponse()) as post:
            result = verify_cited_claims("서비스는 2026년에 출시되었습니다. [S1]", sources)
        self.assertEqual(result["checked_claim_count"], 1)
        self.assertEqual(result["checks"][0]["model_assessment"], "supported")
        self.assertEqual(result["checks"][0]["status"], "needs_review")
        self.assertFalse(result["checks"][0]["automatically_verified"])
        self.assertIn('"source_id": "S1"', post.call_args.kwargs["json"]["messages"][1]["content"])
        self.assertNotIn('"source_id": "S2"', post.call_args.kwargs["json"]["messages"][1]["content"])

    def test_unknown_citation_does_not_call_model(self):
        result = verify_cited_claims("주장입니다. [S8]", [])
        self.assertEqual(result["checks"][0]["model_assessment"], "insufficient")
        self.assertEqual(result["checks"][0]["status"], "needs_review")
        self.assertEqual(result["automatically_verified_count"], 0)

    def test_invalid_model_json_fails_closed(self):
        class BadResponse(FakeResponse):
            def json(self):
                return {"message": {"content": "not-json"}}
        with patch("semantic_verifier.requests.post", return_value=BadResponse()):
            result = verify_cited_claims("주장입니다. [S1]",
                [{"source_id": "S1", "url": "https://example.org", "content": "내용"}])
        self.assertEqual(result["checks"][0]["model_assessment"], "needs_review")


if __name__ == "__main__":
    unittest.main()
