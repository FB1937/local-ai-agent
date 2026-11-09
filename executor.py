"""Stage 5: execute the plan using explicitly registered capabilities."""
from search import search_web
from search_validator import search_with_validation


def execute_plan(question: str, plan: dict, search_fn=None) -> dict:
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
        }

    search_fn = search_fn or search_web
    proposed_query = (plan.get("search") or {}).get("query") or question
    try:
        outcome = search_with_validation(question, proposed_query, search_fn)
        assessment = outcome.get("assessment", {})
        return {
            "status": "completed" if outcome.get("results") else "no_results",
            "tool_calls": ["searxng"],
            "results": outcome.get("results", []),
            "assessment": assessment,
            "attempts": outcome.get("attempts", []),
            "search_plan": outcome.get("plan"),
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
        }
