import unittest
from unittest.mock import patch
import os

import requests

from evidence_passages import _ollama_embed, select_relevant_passages


QUESTION = "Python 3.15 주요 변경 사항"
PASSAGES = (
    "Python 3.15 introduces improved parser diagnostics and clearer error messages.",
    "Python 3.15 adds new syntax features and improvements to the standard library.",
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class SemanticPassageRetrievalTests(unittest.TestCase):
    def test_empty_text_returns_no_passages_without_calling_embedder(self):
        def unexpected_embedder(_):
            self.fail("empty text must not invoke the embedder")

        result = select_relevant_passages(
            QUESTION, "", semantic_enabled=True, embedder=unexpected_embedder
        )
        self.assertEqual(result, [])

    def test_semantic_retrieval_selects_only_candidates_above_threshold(self):
        vectors = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]
        with patch.dict(os.environ, {"OLLAMA_EMBEDDING_MIN_SCORE": "0.8"}):
            result = select_relevant_passages(
                QUESTION,
                " ".join(PASSAGES),
                semantic_enabled=True,
                embedder=lambda texts: vectors,
            )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["retrieval_method"], "semantic")
        self.assertEqual(result[0]["text"], PASSAGES[0])
        self.assertGreaterEqual(result[0]["score"], 0.8)
        self.assertNotIn("fallback_reason", result[0])

    def test_no_semantic_match_falls_back_to_lexical_with_reason(self):
        vectors = [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
        with patch.dict(os.environ, {"OLLAMA_EMBEDDING_MIN_SCORE": "0.9"}):
            result = select_relevant_passages(
                QUESTION,
                " ".join(PASSAGES),
                semantic_enabled=True,
                embedder=lambda texts: vectors,
            )

        self.assertTrue(result)
        self.assertTrue(all(p["retrieval_method"] == "lexical_fallback" for p in result))
        self.assertTrue(all(p.get("fallback_reason") for p in result))
        self.assertTrue(any("Python" in p["text"] for p in result))

    def test_embedding_request_error_falls_back_to_lexical(self):
        def failing_embedder(_):
            raise requests.Timeout("embedding timed out")

        result = select_relevant_passages(
            QUESTION,
            " ".join(PASSAGES),
            semantic_enabled=True,
            embedder=failing_embedder,
        )

        self.assertTrue(result)
        self.assertEqual(result[0]["retrieval_method"], "lexical_fallback")
        self.assertIn("Timeout", result[0]["fallback_reason"])

    def test_wrong_embedding_count_falls_back_to_lexical(self):
        result = select_relevant_passages(
            QUESTION,
            " ".join(PASSAGES),
            semantic_enabled=True,
            embedder=lambda texts: [[1.0, 0.0]],
        )

        self.assertTrue(result)
        self.assertEqual(result[0]["retrieval_method"], "lexical_fallback")
        self.assertIn("ValueError", result[0]["fallback_reason"])

    def test_dimension_mismatch_falls_back_to_lexical(self):
        result = select_relevant_passages(
            QUESTION,
            " ".join(PASSAGES),
            semantic_enabled=True,
            embedder=lambda texts: [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0, 0.0]],
        )

        self.assertTrue(result)
        self.assertEqual(result[0]["retrieval_method"], "lexical_fallback")
        self.assertIn("차원", result[0]["fallback_reason"])

    def test_invalid_numeric_embedding_falls_back_to_lexical(self):
        result = select_relevant_passages(
            QUESTION,
            " ".join(PASSAGES),
            semantic_enabled=True,
            embedder=lambda texts: [[1.0, 0.0], [float("nan"), 1.0], [0.0, 1.0]],
        )

        self.assertTrue(result)
        self.assertEqual(result[0]["retrieval_method"], "lexical_fallback")
        self.assertIn("유한하지 않은", result[0]["fallback_reason"])

    def test_passage_count_and_character_budget_are_enforced(self):
        passages = [
            "Python 3.15 change detail number one with enough words for selection.",
            "Python 3.15 change detail number two with enough words for selection.",
            "Python 3.15 change detail number three with enough words for selection.",
        ]
        vectors = [[1.0, 0.0], [0.99, 0.1], [0.98, 0.2], [0.97, 0.3]]
        with patch.dict(os.environ, {"OLLAMA_EMBEDDING_MIN_SCORE": "0.5"}):
            result = select_relevant_passages(
                QUESTION,
                " ".join(passages),
                max_passages=2,
                max_chars=130,
                semantic_enabled=True,
                embedder=lambda texts: vectors,
            )

        self.assertLessEqual(len(result), 2)
        self.assertLessEqual(sum(len(item["text"]) for item in result), 130)

    def test_disabled_semantic_retrieval_uses_lexical_without_embedding(self):
        def unexpected_embedder(_):
            self.fail("disabled semantic retrieval must not invoke the embedder")

        result = select_relevant_passages(
            QUESTION,
            " ".join(PASSAGES),
            semantic_enabled=False,
            embedder=unexpected_embedder,
        )

        self.assertTrue(result)
        self.assertEqual(result[0]["retrieval_method"], "lexical_fallback")
        self.assertNotIn("fallback_reason", result[0])

    @patch("evidence_passages.requests.post")
    def test_ollama_embed_rejects_non_object_json_payload(self, post):
        post.return_value = FakeResponse(["not", "an", "object"])
        with self.assertRaisesRegex(ValueError, "JSON 객체"):
            _ollama_embed(["question"])

    @patch("evidence_passages.requests.post")
    def test_ollama_embed_rejects_non_finite_vector_values(self, post):
        post.return_value = FakeResponse({"embeddings": [[1.0, float("inf")]]})
        with self.assertRaisesRegex(ValueError, "유한하지 않은"):
            _ollama_embed(["question"])


if __name__ == "__main__":
    unittest.main()
