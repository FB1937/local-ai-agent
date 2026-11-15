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


def evaluate(*, model: str, ollama_url: str, timeout: int) -> dict:
    results = []
    for case in CASES:
        started = time.perf_counter()
        try:
            consistent, reason = _verify_relation_reason_consistency(
                __import__("requests").post,
                base_url=ollama_url,
                model_name=model,
                timeout=timeout,
                claim_id="C1",
                claim_text=case["claim"],
                evidence=EVIDENCE,
                relation=case["relation"],
                analyst_reason=case["analyst_reason"],
            )
            decision = "consistent" if consistent else "inconsistent"
            error = None
        except Exception as exc:
            decision = "error"
            reason = ""
            error = f"{type(exc).__name__}: {exc}"
        elapsed_ms = (time.perf_counter() - started) * 1000
        results.append({
            "case_id": case["case_id"],
            "purpose": case["purpose"],
            "expected_consistent": case["expected_consistent"],
            "decision": decision,
            "correct": (
                decision == ("consistent" if case["expected_consistent"] else "inconsistent")
            ),
            "latency_ms": round(elapsed_ms, 2),
            "verifier_reason": reason,
            "error": error,
        })

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
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--ollama-url", default=None)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--output", default="evaluation/consistency_audit_result.json")
    args = parser.parse_args()

    import os

    model = args.model or os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
    ollama_url = (args.ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
    report = evaluate(model=model, ollama_url=ollama_url, timeout=args.timeout)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
