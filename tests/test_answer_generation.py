import unittest
from unittest.mock import patch

from llm import generate_answer_result
from answer_validator import validate_answer


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self.body


def ollama_body(content, done=True, done_reason="stop", eval_count=12, thinking=""):
    return {
        "message": {"content": content, "thinking": thinking},
        "done": done,
        "done_reason": done_reason,
        "eval_count": eval_count,
        "eval_duration": 1_000_000_000,
        "prompt_eval_count": 50,
        "total_duration": 2_000_000_000,
    }


class AnswerGenerationStabilityTests(unittest.TestCase):
    @patch("llm.requests.post")
    def test_empty_first_response_is_retried_and_recovers(self, post):
        post.side_effect = [
            FakeResponse(ollama_body("")),
            FakeResponse(ollama_body("정상 답변")),
        ]
        result = generate_answer_result("질문", route="DIRECT")
        self.assertTrue(result["ok"])
        self.assertEqual(result["answer"], "정상 답변")
        self.assertEqual(result["retry_count"], 1)
        self.assertEqual(post.call_count, 2)

    @patch("llm.requests.post")
    def test_empty_response_after_retry_is_failure(self, post):
        post.side_effect = [
            FakeResponse(ollama_body("")),
            FakeResponse(ollama_body("")),
        ]
        result = generate_answer_result("질문", route="DIRECT")
        self.assertFalse(result["ok"])
        self.assertEqual(result["answer"], "")
        self.assertEqual(result["retry_count"], 1)
        self.assertTrue(result["error"])

    @patch("llm.requests.post")
    def test_length_limited_response_retries_with_more_tokens(self, post):
        post.side_effect = [
            FakeResponse(ollama_body("잘린 답변", done=True, done_reason="length")),
            FakeResponse(ollama_body("완결된 답변", done=True, done_reason="stop")),
        ]
        result = generate_answer_result("질문", route="DIRECT")
        self.assertTrue(result["ok"])
        self.assertEqual(result["answer"], "완결된 답변")
        first_payload = post.call_args_list[0].kwargs["json"]
        second_payload = post.call_args_list[1].kwargs["json"]
        self.assertGreater(
            second_payload["options"]["num_predict"],
            first_payload["options"]["num_predict"],
        )

    @patch("llm.requests.post")
    def test_length_limited_response_is_flagged_after_retry(self, post):
        post.side_effect = [
            FakeResponse(ollama_body("부분 답변", done_reason="length")),
            FakeResponse(ollama_body("여전히 부분 답변", done_reason="length")),
        ]
        result = generate_answer_result("질문", route="DIRECT")
        self.assertTrue(result["ok"])
        self.assertTrue(result["incomplete"])
        self.assertTrue(result["error"])

    @patch("llm.requests.post")
    def test_retry_disables_thinking_and_records_separate_lengths(self, post):
        post.side_effect = [
            FakeResponse(ollama_body("", done_reason="length", eval_count=2048,
                                     thinking="긴 추론 과정")),
            FakeResponse(ollama_body("최종 답변", done_reason="stop", eval_count=40)),
        ]
        result = generate_answer_result("질문", route="HYBRID")
        self.assertEqual(result["answer"], "최종 답변")
        self.assertGreater(result["attempts"][0]["thinking_characters"], 0)
        self.assertEqual(result["attempts"][0]["content_characters"], 0)
        self.assertEqual(result["attempts"][1]["content_characters"], 5)
        second_payload = post.call_args_list[1].kwargs["json"]
        self.assertIs(second_payload["think"], False)

    def test_extract_response_keeps_thinking_separate_from_answer(self):
        from llm import _extract_response
        result = _extract_response(
            ollama_body("본문 답변", thinking="별도 추론", eval_count=30), 1.5
        )
        self.assertEqual(result["answer"], "본문 답변")
        self.assertEqual(result["content_characters"], 5)
        self.assertEqual(result["thinking_characters"], 5)

    @patch("llm.requests.post")
    def test_http_error_returns_failed_generation(self, post):
        import requests
        post.side_effect = requests.Timeout("timeout")
        result = generate_answer_result("질문", route="DIRECT")
        self.assertFalse(result["ok"])
        self.assertIn("Timeout", result["error"])

    def test_empty_answer_fails_structural_validation(self):
        result = validate_answer("", [])
        self.assertFalse(result["valid"])
        self.assertTrue(result["empty_answer"])

    def test_generation_failure_fails_validation_even_with_text(self):
        result = validate_answer("오류 안내", [], generation_ok=False)
        self.assertFalse(result["valid"])


if __name__ == "__main__":
    unittest.main()
