"""Stage-6 adapter for bounded M3 claim/evidence grounding.

This adapter orchestrates existing M3 components without changing their contracts.
Its output is a review signal, not a truth certificate.
"""
from __future__ import annotations

from time import perf_counter

from claim_analysis import analyze_claim_evidence
from evidence_reasoner import reason_claim_evidence
from claim_synthesis import synthesize_claim_evidence
from semantic_claim_synthesis import synthesize_claim_evidence_semantically


_BLOCKING_STATUSES = {"contradicted", "conflict", "unassessed"}


def evaluate_response_claims(
    answer: str,
    sources: list[dict],
    *,
    max_links: int = 20,
    max_semantic_claims: int = 5,
    max_evidence_per_claim: int = 8,
    timeout: int = 90,
    request_fn=None,
) -> dict:
    """Run the M3 pipeline and return claim statuses, provenance and diagnostics.

    Any pipeline exception is represented as a fail-closed unassessed report.
    No result from this function is an automatic factual certification.
    """
    started = perf_counter()
    timings: dict[str, float] = {}
    try:
        step = perf_counter()
        analysis = analyze_claim_evidence(answer, sources)
        timings["claim_analysis"] = perf_counter() - step

        step = perf_counter()
        reasoning = reason_claim_evidence(
            analysis, max_links=max_links, timeout=timeout, request_fn=request_fn,
        )
        timings["pairwise_reasoning"] = perf_counter() - step

        step = perf_counter()
        baseline = synthesize_claim_evidence(analysis, reasoning.links)
        timings["deterministic_synthesis"] = perf_counter() - step

        step = perf_counter()
        final_report, semantic = synthesize_claim_evidence_semantically(
            analysis,
            baseline_report=baseline,
            max_claims=max_semantic_claims,
            max_evidence_per_claim=max_evidence_per_claim,
            timeout=timeout,
            request_fn=request_fn,
        )
        timings["semantic_synthesis"] = perf_counter() - step

        claims = [item.model_dump(mode="json") for item in final_report.claims]
        statuses = [item["status"] for item in claims]
        blocking = [item for item in claims if item["status"] in _BLOCKING_STATUSES]
        insufficient = [item for item in claims if item["status"] == "insufficient"]
        unassessed = sum(item["status"] == "unassessed" for item in claims)
        return {
            "enabled": True,
            "status": "blocked" if blocking else "review" if insufficient else "assessed",
            "claims": claims,
            "claim_count": len(claims),
            "blocking_claim_count": len(blocking),
            "insufficient_claim_count": len(insufficient),
            "unassessed_claim_count": unassessed,
            "pairwise": {
                "assessed_count": reasoning.assessed_count,
                "unassessed_count": reasoning.unassessed_count,
                "skipped_count": reasoning.skipped_count,
                "call_budget": max_links,
            },
            "semantic": {
                "assessed_count": semantic.assessed_count,
                "unassessed_count": semantic.unassessed_count,
                "skipped_count": semantic.skipped_count,
                "claim_budget": max_semantic_claims,
                "evidence_per_claim_limit": max_evidence_per_claim,
            },
            "timings_seconds": timings,
            "total_seconds": perf_counter() - started,
            "note": (
                "M3 결과는 인용 근거와 Claim 사이의 보조 검토입니다. "
                "supported는 사실 인증을 의미하지 않습니다."
            ),
        }
    except Exception as exc:
        return {
            "enabled": True,
            "status": "error",
            "claims": [],
            "claim_count": 0,
            "blocking_claim_count": 0,
            "insufficient_claim_count": 0,
            "unassessed_claim_count": 0,
            "pairwise": {
                "assessed_count": 0, "unassessed_count": 0,
                "skipped_count": 0, "call_budget": max_links,
            },
            "semantic": {
                "assessed_count": 0, "unassessed_count": 0,
                "skipped_count": 0, "claim_budget": max_semantic_claims,
                "evidence_per_claim_limit": max_evidence_per_claim,
            },
            "timings_seconds": timings,
            "total_seconds": perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
            "note": "M3 검토에 실패하여 결과를 미평가로 처리했습니다.",
        }
