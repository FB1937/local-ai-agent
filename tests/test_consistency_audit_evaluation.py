import json
import unittest

from evaluation.run_consistency_audit_evaluation import (
    CASES,
    EVIDENCE,
    evaluate,
    select_cases,
)


class _FakeResponse:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"message": {"content": self.content}}


class ConsistencyAuditEvaluationFixtureTests(unittest.TestCase):
    def test_fixture_contains_both_mismatch_directions_and_controls(self):
        self.assertEqual(len(CASES), 5)
        self.assertEqual(
            sum(not case["expected_consistent"] for case in CASES), 3
        )
        self.assertEqual(
            sum(case["expected_consistent"] for case in CASES), 2
        )
        self.assertEqual(
            {case["relation"] for case in CASES},
            {"supports", "contradicts"},
        )

    def test_c03_style_case_has_contradictory_reason_with_supports_enum(self):
        case = next(
            item for item in CASES
            if item["case_id"] == "AUDIT-MISMATCH-SUPPORTS"
        )
        self.assertFalse(case["expected_consistent"])
        self.assertEqual(case["relation"], "supports")
        self.assertIn("반박한다", case["analyst_reason"])
        self.assertEqual(len(EVIDENCE), 2)

    def test_select_cases_preserves_fixture_order(self):
        selected = select_cases([
            "AUDIT-CONTROL-SUPPORTS",
            "AUDIT-CONTROL-CONTRADICTS",
        ])
        self.assertEqual(
            [item["case_id"] for item in selected],
            ["AUDIT-CONTROL-CONTRADICTS", "AUDIT-CONTROL-SUPPORTS"],
        )

    def test_select_cases_rejects_unknown_id(self):
        with self.assertRaisesRegex(ValueError, "unknown case_id"):
            select_cases(["NOT-A-CASE"])

    def test_repeat_and_single_case_are_reported(self):
        calls = []

        def fake_request(*args, **kwargs):
            calls.append(1)
            return _FakeResponse(json.dumps({
                "consistent": True,
                "reason": "구조화 관계와 설명이 일치합니다.",
            }, ensure_ascii=False))

        report = evaluate(
            model="test-model",
            ollama_url="http://localhost:11434",
            timeout=1,
            case_ids=["AUDIT-CONTROL-SUPPORTS"],
            repeat=2,
            request_fn=fake_request,
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(report["unique_case_count"], 1)
        self.assertEqual(report["case_count"], 2)
        self.assertEqual(
            [item["repeat_index"] for item in report["results"]],
            [1, 2],
        )
        self.assertEqual(report["accuracy_including_errors_as_incorrect"], 1.0)

    def test_json_parse_error_captures_bounded_raw_output(self):
        raw = '{"consistent": true,\n "reason": "unterminated'
        def fake_request(*args, **kwargs):
            return _FakeResponse(raw)

        report = evaluate(
            model="test-model",
            ollama_url="http://localhost:11434",
            timeout=1,
            case_ids=["AUDIT-CONTROL-CONTRADICTS"],
            request_fn=fake_request,
        )
        result = report["results"][0]
        self.assertEqual(result["decision"], "error")
        diagnostics = result["error_diagnostics"]
        self.assertEqual(diagnostics["error_type"], "JSONDecodeError")
        self.assertEqual(diagnostics["raw_output_excerpt"], raw)
        self.assertGreaterEqual(diagnostics["json_line"], 1)

    def test_repeat_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "repeat must be at least 1"):
            evaluate(
                model="test-model",
                ollama_url="http://localhost:11434",
                timeout=1,
                repeat=0,
                request_fn=lambda *args, **kwargs: None,
            )


if __name__ == "__main__":
    unittest.main()
