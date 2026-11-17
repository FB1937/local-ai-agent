"""Targeted live evaluation of the semantic relation/reason consistency auditor.

This isolates the second-pass auditor from the first-pass analyst so that
C03-style structured-output mismatches can be tested deterministically.
It does not modify or reuse either M3.4 aggregate evaluation dataset.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import requests

from semantic_claim_synthesis import _verify_relation_reason_consistency


CASES = [
    {
        "case_id": "AUDIT-MISMATCH-SUPPORTS",
        "expected_consistent": False,
        "claim": "상품의 최종 결제 금액은 80,000원이다.",
        "relation": "supports",
        "analyst_reason": "상품 52,000원에 배송비 4,000원을 더하고 쿠폰 8,000원을 빼면 48,000원이다. 이는 주장된 80,000원과 다르므로 근거는 주장을 반박한다.",
        "purpose": "C03 유형: 설명은 반박하지만 구조화 관계는 supports",
    },
    {
        "case_id": "AUDIT-MISMATCH-CONTRADICTS",
        "expected_consistent": False,
        "claim": "상품의 최종 결제 금액은 48,000원이다.",
        "relation": "contradicts",
        "analyst_reason": "상품 52,000원에 배송비 4,000원을 더하고 쿠폰 8,000원을 빼면 정확히 48,000원이므로 주장을 지지한다.",
        "purpose": "설명은 지지하지만 구조화 관계는 contradicts",
    },
    {
        "case_id": "AUDIT-AMBIGUOUS-REASON",
        "expected_consistent": False,
        "claim": "상품의 최종 결제 금액은 80,000원이다.",
        "relation": "supports",
        "analyst_reason": "관련된 금액 정보가 여러 개 있으며 전체적으로 연관성이 있다.",
        "purpose": "관계 판정을 정당화하지 못하는 모호한 설명",
    },
    {
        "case_id": "AUDIT-CONTROL-CONTRADICTS",
        "expected_consistent": True,
        "claim": "상품의 최종 결제 금액은 80,000원이다.",
        "relation": "contradicts",
        "analyst_reason": "상품 52,000원에 배송비 4,000원을 더하고 쿠폰 8,000원을 빼면 48,000원이다. 따라서 80,000원이라는 주장은 반박된다.",
        "purpose": "정상적인 반박 판정 대조군",
    },
    {
        "case_id": "AUDIT-CONTROL-SUPPORTS",
        "expected_consistent": True,
        "claim": "상품의 최종 결제 금액은 48,000원이다.",
        "relation": "supports",
        "analyst_reason": "상품 52,000원에 배송비 4,000원을 더하고 쿠폰 8,000원을 빼면 48,000원으로 주장의 금액과 일치한다.",
        "purpose": "정상적인 지지 판정 대조군",
    },
]

EVIDENCE = [
    {
        "evidence_id": "E1",
        "text": "상품 가격은 52,000원이고 배송비는 4,000원이다.",
        "scope": "동일 상품 주문",
        "condition": "동일 주문의 최종 결제액 계산",
        "valid_time_start": None,
        "valid_time_end": None,
    },
    {
        "evidence_id": "E2",
        "text": "해당 주문에 8,000원 쿠폰 할인이 적용된다.",
        "scope": "동일 상품 주문",
        "condition": "동일 주문의 최종 결제액 계산",
        "valid_time_start": None,
        "valid_time_end": None,
    },
]

_RAW_OUTPUT_LIMIT = 800


def select_cases(case_ids: list[str] | None = None) -> list[dict]:
    """Return requested fixtures, preserving canonical fixture order."""
    if not case_ids:
        return list(CASES)
    requested = set(case_ids)
    known = {case["case_id"] for case in CASES}
    unknown = requested - known
    if unknown:
        raise ValueError(f"unknown case_id(s): {', '.join(sorted(unknown))}")
    return [case for case in CASES if case["case_id"] in requested]


def _error_diagnostics(exc: Exception, captured: dict) -> dict:
    details = {
        "error_type": type(exc).__name__,
        "error_message": str(exc)[:500],
    }
    if isinstance(exc, json.JSONDecodeError):
        details.update({
            "json_line": exc.lineno,
            "json_column": exc.colno,
            "json_position": exc.pos,
        })
    raw = captured.get("message_content")
    if isinstance(raw, str):
        details["raw_output_length"] = len(raw)
        details["raw_output_excerpt"] = raw[:_RAW_OUTPUT_LIMIT]
        if len(raw) > _RAW_OUTPUT_LIMIT:
            details["raw_output_truncated"] = True
    elif captured.get("response_text"):
        response_text = captured["response_text"]
        details["raw_response_excerpt"] = response_text[:_RAW_OUTPUT_LIMIT]
        details["raw_response_length"] = len(response_text)
    if captured.get("capture_error"):
        details["capture_error"] = captured["capture_error"][:300]
    return details


def evaluate(
    *,
    model: str,
    ollama_url: str,
    timeout: int,
    case_ids: list[str] | None = None,
    repeat: int = 1,
    request_fn=None,
    num_predict: int = 32,
) -> dict:
    if repeat < 1:
        raise ValueError("repeat must be at least 1")
    selected_cases = select_cases(case_ids)
    requester = request_fn or requests.post
    results = []

    for repeat_index in range(1, repeat + 1):
        for case in selected_cases:
            captured = {}

            def capture_request(*args, **kwargs):
                response = requester(*args, **kwargs)
                try:
                    body = response.json()
                    message = body.get("message") or {}
                    content = message.get("content")
                    if isinstance(content, str):
                        captured["message_content"] = content
                    for key in ("done_reason", "eval_count", "prompt_eval_count", "total_duration", "eval_duration"):
                        if key in body:
                            captured[key] = body[key]
                except Exception as capture_exc:
                    captured["capture_error"] = (
                        f"{type(capture_exc).__name__}: {capture_exc}"
                    )
                    response_text = getattr(response, "text", "")
                    if isinstance(response_text, str):
                        captured["response_text"] = response_text
                return response

            started = time.perf_counter()
            try:
                consistent, reason = _verify_relation_reason_consistency(
                    capture_request,
                    base_url=ollama_url,
                    model_name=model,
                    timeout=timeout,
                    claim_id="C1",
                    claim_text=case["claim"],
                    evidence=EVIDENCE,
                    relation=case["relation"],
                    analyst_reason=case["analyst_reason"],
                    num_predict=num_predict,
                )
                decision = "consistent" if consistent else "inconsistent"
                error = None
                diagnostics = None
            except Exception as exc:
                decision = "error"
                reason = ""
                error = f"{type(exc).__name__}: {exc}"
                diagnostics = _error_diagnostics(exc, captured)

            elapsed_ms = (time.perf_counter() - started) * 1000
            result = {
                "case_id": case["case_id"],
                "repeat_index": repeat_index,
                "purpose": case["purpose"],
                "expected_consistent": case["expected_consistent"],
                "decision": decision,
                "correct": (
                    decision == ("consistent" if case["expected_consistent"] else "inconsistent")
                ),
                "latency_ms": round(elapsed_ms, 2),
                "verifier_reason": reason,
                "error": error,
                "ollama": {
                    key: captured[key]
                    for key in ("done_reason", "eval_count", "prompt_eval_count", "total_duration", "eval_duration")
                    if key in captured
                },
            }
            if diagnostics:
                result["error_diagnostics"] = diagnostics
            results.append(result)

    completed = [item for item in results if item["decision"] != "error"]
    latencies = [item["latency_ms"] for item in results]
    false_accepts = sum(
        item["decision"] == "consistent" and not item["expected_consistent"]
        for item in results
    )
    false_rejections = sum(
        item["decision"] == "inconsistent" and item["expected_consistent"]
        for item in results
    )
    errors = sum(item["decision"] == "error" for item in results)
    return {
        "model": model,
        "num_predict": num_predict,
        "unique_case_count": len(selected_cases),
        "repeat_count": repeat,
        "case_count": len(results),
        "completed_count": len(completed),
        "correct_count": sum(item["correct"] for item in results),
        "accuracy_including_errors_as_incorrect": (
            sum(item["correct"] for item in results) / len(results)
        ),
        "false_accept_count": false_accepts,
        "false_rejection_count": false_rejections,
        "error_count": errors,
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 2) if latencies else None,
            "median": round(statistics.median(latencies), 2) if latencies else None,
            "max": round(max(latencies), 2) if latencies else None,
        },
        "results": results,
        "limitations": [
            "Five manually authored Korean diagnostic cases; not a statistical benchmark.",
            "The same model family may share correlated errors with the first-pass analyst.",
            "The evidence arithmetic is intentionally simple and does not test broad reasoning quality.",
            "Errors are fail-closed and counted as incorrect for aggregate accuracy.",
            "Raw model output is truncated to 800 characters in error diagnostics.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--ollama-url", default=None)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument(
        "--num-predict", type=int, default=32,
        help="Maximum generated tokens for the compact JSON auditor response (default: 32).",
    )
    parser.add_argument(
        "--case-id", action="append", dest="case_ids",
        help="Run only this case ID; may be supplied more than once.",
    )
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="Repeat the selected case set N times (default: 1).",
    )
    parser.add_argument("--output", default="evaluation/consistency_audit_result.json")
    args = parser.parse_args()

    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if args.num_predict < 1:
        parser.error("--num-predict must be at least 1")
    try:
        selected_cases = select_cases(args.case_ids)
    except ValueError as exc:
        parser.error(str(exc))

    import os

    model = args.model or os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
    ollama_url = (args.ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
    report = evaluate(
        model=model,
        ollama_url=ollama_url,
        timeout=args.timeout,
        case_ids=[case["case_id"] for case in selected_cases],
        repeat=args.repeat,
        num_predict=args.num_predict,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
