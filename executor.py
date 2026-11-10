"""Stage 5: execute the plan using explicitly registered capabilities."""
from search import search_web
from search_validator import search_with_validation, prepare_results, assess_results
from page_extractor import extract_official_pages


def execute_plan(question: str, plan: dict, search_fn=None,
                 page_extractor_fn=None) -> dict:
    route = plan.get("route", "SEARCH")
    if route == "DIRECT":
        return {
            "status": "skipped",
            "tool_calls": [],
            "results": [],
            "assessment": {
                "sufficient": True,
                "reason": "DIRECT 경로이므로 외부 검색을 실행하지 않았습니다.",
            },
            "attempts": [],
            "page_extraction": {
                "attempted_count": 0, "extracted_count": 0, "errors": [],
            },
        }

    search_fn = search_fn or search_web
    page_extractor_fn = page_extractor_fn or extract_official_pages
    proposed_query = (plan.get("search") or {}).get("query") or question
    try:
        outcome = search_with_validation(question, proposed_query, search_fn)
        search_plan = outcome.get("plan") or {}
        raw_results = outcome.get("results", [])

        # Only first-party official sources are fetched; the extractor applies
        # URL/redirect checks and a strict page-count/size budget.
        extraction = page_extractor_fn(raw_results, max_pages=3)
        results = extraction.get("results", raw_results)

        # Re-evaluate using extracted page text. Search retry has already
        # completed, so this pass only refreshes evidence signals and scores.
        if search_plan:
            results = prepare_results(
                search_plan.get("query", proposed_query), results, max_results=8
            )
            assessment = assess_results(
                results,
                freshness_required=search_plan.get("freshness_required", False),
                requires_official=search_plan.get("requires_official", False),
                requires_company_official=search_plan.get("requires_company_official", False),
                version_query=search_plan.get("version_query", False),
            )
        else:
            assessment = outcome.get("assessment", {})

        tool_calls = ["searxng"]
        if extraction.get("attempted_count", 0):
            tool_calls.append("trafilatura")

        return {
            "status": "completed" if results else "no_results",
            "tool_calls": tool_calls,
            "results": results,
            "assessment": assessment,
            "attempts": outcome.get("attempts", []),
            "search_plan": search_plan or outcome.get("plan"),
            "page_extraction": extraction,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "tool_calls": ["searxng"],
            "results": [],
            "assessment": {
                "sufficient": False,
                "reason": f"도구 실행 중 오류가 발생했습니다: {type(exc).__name__}: {exc}",
            },
            "attempts": [],
            "error": f"{type(exc).__name__}: {exc}",
            "page_extraction": {
                "attempted_count": 0, "extracted_count": 0,
                "errors": [f"{type(exc).__name__}: {exc}"],
            },
        }
