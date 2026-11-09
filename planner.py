"""Stage 4: combine task analysis, route selection, and search planning."""
from search_planner import plan_search


def create_plan(question: str, task: dict, route_decision: dict) -> dict:
    route = str(route_decision.get("route", "SEARCH")).upper()
    if route not in {"DIRECT", "SEARCH", "HYBRID"}:
        route = "SEARCH"

    search_plan = None
    steps = []
    if route == "DIRECT":
        steps.append({"action": "generate_from_model", "tool": None})
    else:
        search_plan = plan_search(
            question, route_decision.get("search_query", "")
        )
        steps.extend([
            {"action": "search", "tool": "searxng", "query": search_plan["query"]},
            {"action": "validate_evidence", "tool": "search_validator"},
            {"action": "generate_answer", "tool": "ollama"},
        ])

    return {
        "route": route,
        "reason": route_decision.get("reason", ""),
        "task": task,
        "search": search_plan,
        "steps": steps,
        "requires_external_execution": route in {"SEARCH", "HYBRID"},
        "retry_policy": {
            "max_search_attempts": 2 if route in {"SEARCH", "HYBRID"} else 0,
            "max_replans": 0,
            "note": "현재 버전은 검색 검증기의 1회 재검색만 지원하며, 일반 목적 재계획 루프는 후속 단계입니다.",
        },
    }
