"""Stage-6 adapter for bounded M3 claim/evidence grounding.

This adapter orchestrates existing M3 components without changing their contracts.
Its output is a review signal, not a truth certificate.
"""
from __future__ import annotations

from time import perf_counter

from claim_analysis import analyze_claim_evidence
from claim_normalization import normalize_claim_analysis
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
        raw_analysis = analyze_claim_evidence(answer, sources)
        analysis, selected_links, budget_skipped_links, excluded_claims = normalize_claim_analysis(
            raw_analysis, max_links=max_links,
        )
        timings["claim_analysis"] = perf_counter() - step

        # Preserve retrieval provenance so an "insufficient" result can be
        # diagnosed as missing citations, missing source text, or no matching
        # passages instead of exposing only the final status.
        links_by_claim: dict[str, list] = {}
        for link in analysis.links:
            links_by_claim.setdefault(link.claim_id, []).append(link)
        passages_by_claim: dict[str, list] = {}
        for passage in analysis.passages:
            if passage.claim_id:
                passages_by_claim.setdefault(passage.claim_id, []).append(passage)
        claim_retrieval = []
        for claim in analysis.claims:
            claim_links = links_by_claim.get(claim.claim_id, [])
            claim_passages = passages_by_claim.get(claim.claim_id, [])
            claim_retrieval.append({
                "claim_id": claim.claim_id,
                "text": claim.text,
                "importance": claim.importance,
                "cited_source_ids": list(claim.cited_source_ids),
                "passage_count": len(claim_passages),
                "linked_passage_count": len(claim_links),
                "retrieval_methods": sorted({
                    item.retrieval_method or "unknown" for item in claim_passages
                }),
            })
        source_retrieval = []
        for raw_source in sources or []:
            source = raw_source
            source_id = source.get("source_id", "") if isinstance(source, dict) else source.source_id
            extracted = source.get("extracted_text") if isinstance(source, dict) else source.extracted_text
            content = source.get("content", "") if isinstance(source, dict) else source.content
            body = extracted or content or ""
            source_retrieval.append({
                "source_id": source_id,
                "extraction_status": (
                    source.get("extraction_status", "unknown")
                    if isinstance(source, dict) else source.extraction_status
                ),
                "extracted_text_chars": len(extracted or ""),
                "content_chars": len(content or ""),
                "body_source": "extracted_text" if extracted else "content" if content else "empty",
                "body_chars_used": len(body),
            })
        retrieval_diagnostics = {
            "claim_count": len(analysis.claims),
            "excluded_claim_count": len(excluded_claims),
            "excluded_claims": excluded_claims,
            "claims_with_citations": sum(bool(item.cited_source_ids) for item in analysis.claims),
            "claims_without_citations": [
                item.claim_id for item in analysis.claims if not item.cited_source_ids
            ],
            "passage_count": len(analysis.passages),
            "claim_evidence_link_count": len(analysis.links),
            "claims_without_evidence": list(analysis.claims_without_evidence),
            "unmapped_source_ids": list(analysis.unmapped_source_ids),
            "claims": claim_retrieval,
            "sources": source_retrieval,
        }

        step = perf_counter()
        selected_analysis = analysis.model_copy(update={"links": selected_links})
        reasoning = reason_claim_evidence(
            selected_analysis, max_links=max_links, timeout=timeout, request_fn=request_fn,
        )
        reasoning_by_pair = {
            (link.claim_id, link.evidence_id): link for link in reasoning.links
        }
        # Preserve every retrieved link for deterministic synthesis. Links not
        # selected by the M4.2 budget planner remain explicitly unassessed.
        combined_reasoning_links = [
            reasoning_by_pair.get(
                (link.claim_id, link.evidence_id),
                link.model_copy(update={
                    "relation": None,
                    "reason": "M4.2 검증 예산 초과로 모델 판정을 수행하지 않았습니다.",
                }),
            )
            for link in analysis.links
        ]
        timings["pairwise_reasoning"] = perf_counter() - step

        # Expose the exact passage and pairwise label behind each claim status.
        # This is diagnostic only; it does not change synthesis or blocking policy.
        passages_by_id = {item.evidence_id: item for item in analysis.passages}
        analysis_links_by_claim: dict[str, list] = {}
        for link in analysis.links:
            analysis_links_by_claim.setdefault(link.claim_id, []).append(link)
        for item in retrieval_diagnostics["claims"]:
            details = []
            # Retrieval provenance and model assessment are separate facts:
            # keep every selected Claim-Evidence link visible even when the
            # pairwise call was skipped, failed, or returned no assessment.
            for analysis_link in analysis_links_by_claim.get(item["claim_id"], []):
                passage = passages_by_id.get(analysis_link.evidence_id)
                if passage is None:
                    continue
                assessed_link = reasoning_by_pair.get(
                    (analysis_link.claim_id, analysis_link.evidence_id)
                )
                details.append({
                    "evidence_id": passage.evidence_id,
                    "source_id": passage.source_id,
                    "relation": assessed_link.relation if assessed_link else None,
                    "reason": assessed_link.reason if assessed_link else None,
                    "text": passage.text,
                    "retrieval_method": passage.retrieval_method,
                    "retrieval_score": passage.retrieval_score,
                    "location": passage.location,
                })
            item["evidence"] = details

        step = perf_counter()
        baseline = synthesize_claim_evidence(analysis, combined_reasoning_links)
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
            "retrieval": retrieval_diagnostics,
            "claim_count": len(claims),
            "blocking_claim_count": len(blocking),
            "insufficient_claim_count": len(insufficient),
            "unassessed_claim_count": unassessed,
            "pairwise": {
                "assessed_count": reasoning.assessed_count,
                "unassessed_count": reasoning.unassessed_count,
                "skipped_count": reasoning.skipped_count + len(budget_skipped_links),
                "budget_skipped_count": len(budget_skipped_links),
                "selected_link_count": len(selected_links),
                "available_link_count": len(analysis.links),
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
            "retrieval": {
                "claim_count": 0, "claims_with_citations": 0,
                "claims_without_citations": [], "passage_count": 0,
                "claim_evidence_link_count": 0, "claims_without_evidence": [],
                "excluded_claim_count": 0, "excluded_claims": [],
                "unmapped_source_ids": [], "claims": [], "sources": [],
            },
            "claim_count": 0,
            "blocking_claim_count": 0,
            "insufficient_claim_count": 0,
            "unassessed_claim_count": 0,
            "pairwise": {
                "assessed_count": 0, "unassessed_count": 0,
                "skipped_count": 0, "budget_skipped_count": 0,
                "selected_link_count": 0, "available_link_count": 0,
                "call_budget": max_links,
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
