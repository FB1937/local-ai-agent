"""Question routing for the Local AI Agent v1.1.

Routes:
- DIRECT: answer with the local LLM only.
- SEARCH: retrieve external information, then answer from the results.
- HYBRID: combine stable background explanation with current/external evidence.

The LLM proposes a route. High-precision safety rules can promote a risky
DIRECT decision to SEARCH, and uncertain/failed classification falls back to
SEARCH rather than silently answering a time-sensitive question from memory.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
ROUTER_TIMEOUT = int(os.getenv("ROUTER_TIMEOUT", "120"))

try:
    CONFIDENCE_THRESHOLD = float(os.getenv("ROUTER_CONFIDENCE_THRESHOLD", "0.55"))
except ValueError:
    CONFIDENCE_THRESHOLD = 0.55

CONFIDENCE_THRESHOLD = min(max(CONFIDENCE_THRESHOLD, 0.0), 1.0)
VALID_ROUTES = {"DIRECT", "SEARCH", "HYBRID"}

# Conservative fast path: only obvious conceptual/explanatory requests are
# routed locally. Freshness, retrieval, company/customer cases, and other
# time-sensitive topics continue through the model router and safety checks.
FAST_DIRECT_REQUEST = re.compile(
    r"(설명|개념|원리|차이|어떻게\s*(?:돼|되)|왜|무엇|무슨)",
    re.I,
)
FAST_DIRECT_EXCLUSION = re.compile(
    r"(최신|최근|현재|현시점|실시간|오늘|올해|이번\s*(?:주|달|해)|"
    r"검색|조사|찾아|공식|출처|사례|도입|고객|기업|회사|시장|"
    r"채용|가격|요금|정책|뉴스|일정|순위|버전|출시|실적|통계)",
    re.I,
)


def _safe_fast_direct_decision(question: str) -> dict[str, Any] | None:
    text = (question or "").strip()
    if not text or len(text) > 300:
        return None
    if not FAST_DIRECT_REQUEST.search(text) or FAST_DIRECT_EXCLUSION.search(text):
        return None
    return {
        "model_route": "RULE_DIRECT",
        "route": "DIRECT",
        "reason": "명확한 개념·원리 설명 요청이며 최신 외부 사실이나 검색 요구가 없어 규칙 기반 DIRECT 경로를 선택했습니다.",
        "search_query": "",
        "confidence": 0.95,
        "overridden": False,
        "fallback": False,
        "fast_path": True,
    }

ROUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "route": {"type": "string", "enum": ["DIRECT", "SEARCH", "HYBRID"]},
        "reason": {"type": "string"},
        "search_query": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["route", "reason", "search_query", "confidence"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """
너는 로컬 AI 에이전트의 질문 라우터다. 질문에 답하지 말고, 질문을 처리할 경로만 결정한다.
반드시 지정된 JSON 구조로만 출력한다.

가능한 경로는 세 가지다.

1. DIRECT
- 일반적인 개념 설명, 기초 프로그래밍, 수학·논리 문제
- 사용자가 제공한 텍스트의 번역, 요약, 교정, 재작성
- 최신 외부 사실이나 외부 출처가 없어도 충분히 처리할 수 있는 요청

2. SEARCH
- 답변의 핵심이 현재 시점의 사실 또는 외부에서 확인해야 하는 정보인 경우
- 최신 뉴스·가격·환율·주가·날씨·일정·채용공고·법률·정책·버전·제품 정보
- 사용자가 웹 검색, 최신 자료, 공식 출처, 링크 등을 명시적으로 요청한 경우
- 현재 사실의 정확한 확인이 중요하고 일반 지식만으로 답하면 위험한 경우

3. HYBRID
- 안정적인 배경 개념 설명과 최신 외부 정보 조사가 모두 필요한 경우
- 개념이나 원리를 먼저 설명한 뒤 현재 시장·기업·제품·정책·채용 사례와 연결해야 하는 경우

판단 원칙:
- 입력에 [이전 대화 문맥]과 [현재 질문]이 함께 있으면 반드시 [현재 질문]이 이번 요청이다.
- 이전 대화는 생략된 대상을 지칭하는 표현을 해석할 때만 참고한다. 이전 질문의 검색 요청이나 최신성 요구를 현재 질문에 상속하지 않는다.
- 이번 경로는 현재 질문의 의도와 필요한 사실을 기준으로 결정한다.
- 단순히 질문이 길거나 '비교', '분석'이라는 단어가 있다는 이유만으로 검색하지 않는다.
- 답변이 현재 사실에 의존하면 SEARCH 또는 HYBRID를 선택한다.
- 사용자가 제공한 자료만 변환·요약하라는 요청은 보통 DIRECT다.
- SEARCH와 HYBRID의 search_query에는 실제 검색에 적합한 짧고 구체적인 검색어를 작성한다.
- DIRECT의 search_query는 빈 문자열로 작성한다.
- confidence는 네 경로 선택에 대한 자체 확신도(0~1)다. 통계적으로 보정된 확률이 아님을 전제로 한다.
- reason은 경로를 선택한 핵심 이유를 한 문장으로 설명한다.
""".strip()

EXPLICIT_SEARCH_PATTERNS = [
    re.compile(r"(?:웹|인터넷)(?:에서)?\s*(?:검색|찾아|조사)"),
    re.compile(r"(?:검색해\s*줘|검색해서|검색해봐|검색해|찾아봐|찾아줘|조사해\s*줘)"),
    re.compile(r"(?:공식\s*(?:문서|자료|출처)|출처|참고\s*자료).{0,16}(?:포함|제시|표시|달아|링크|찾아)"),
]

FRESHNESS_MARKER = re.compile(
    r"(?:최신|최근|현재|현시점|실시간|이번\s*주|이번\s*달|이번\s*해|올해|\b20(?:2[4-9]|[3-9]\d)\b|as\s*of|latest|current)",
    re.IGNORECASE,
)

TIME_SENSITIVE_TOPIC = re.compile(
    r"(?:날씨|환율|주가|시세|가격|요금|채용|공고|정책|법률|법령|규정|뉴스|일정|버전|출시|실적|통계|순위|운영\s*시간|영업\s*시간|시간표|항공편|예약|선거|금리)",
    re.IGNORECASE,
)


def _parse_json_object(content: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating accidental Markdown fences."""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("라우터 응답에서 JSON 객체를 찾지 못했습니다.")
        parsed = json.loads(content[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("라우터 응답의 최상위 값이 JSON 객체가 아닙니다.")
    return parsed


def _call_router_model(question: str) -> dict[str, Any]:
    if not OLLAMA_URL or not OLLAMA_MODEL:
        raise RuntimeError("OLLAMA_URL 또는 OLLAMA_MODEL 환경변수가 설정되지 않았습니다.")

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"다음 질문의 처리 경로를 분류해 줘. JSON 이외의 문장은 출력하지 마.\n\n질문:\n{question}",
            },
        ],
        "format": ROUTE_SCHEMA,
        "stream": False,
        "options": {"temperature": 0},
    }

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json=payload,
        timeout=ROUTER_TIMEOUT,
    )
    response.raise_for_status()
    body = response.json()
    content = body.get("message", {}).get("content", "")
    if not content:
        raise ValueError("Ollama 라우터 응답 내용이 비어 있습니다.")
    return _parse_json_object(content)


def _normalise_model_decision(data: dict[str, Any]) -> dict[str, Any]:
    route = str(data.get("route", "")).strip().upper()
    if route not in VALID_ROUTES:
        raise ValueError(f"허용되지 않은 라우팅 결과입니다: {route!r}")

    reason = str(data.get("reason", "모델이 분류 이유를 제공하지 않았습니다.")).strip()
    search_query = str(data.get("search_query", "")).strip()

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(max(confidence, 0.0), 1.0)

    return {
        "model_route": route,
        "route": route,
        "reason": reason,
        "search_query": search_query,
        "confidence": confidence,
        "overridden": False,
        "fallback": False,
    }


def apply_safety_rules(question: str, decision: dict[str, Any]) -> dict[str, Any]:
    """Apply conservative rules to an otherwise valid model decision.

    The rules only promote uncertain or clearly time-sensitive DIRECT decisions
    to SEARCH. An existing HYBRID decision is preserved for mixed requests.
    """
    result = dict(decision)
    route = result.get("route", "SEARCH")
    confidence = float(result.get("confidence", 0.0))
    reasons: list[str] = []

    explicit_search = any(pattern.search(question) for pattern in EXPLICIT_SEARCH_PATTERNS)
    has_freshness_marker = bool(FRESHNESS_MARKER.search(question))
    has_time_sensitive_topic = bool(TIME_SENSITIVE_TOPIC.search(question))
    today_context = bool(re.search(r"오늘", question)) and has_time_sensitive_topic
    current_information_needed = has_freshness_marker or today_context or (
        has_time_sensitive_topic
        and bool(re.search(r"(?:얼마|어떻게\s*바뀌|현황|추이|비교|기준|확인|정보|자료|동향)", question))
    )

    if route == "DIRECT" and explicit_search:
        route = "SEARCH"
        reasons.append("사용자가 외부 검색 또는 출처 확인을 명시해 SEARCH로 보정했습니다.")

    if route == "DIRECT" and current_information_needed:
        route = "SEARCH"
        reasons.append("최신성 또는 현재 사실 확인이 필요해 SEARCH로 보정했습니다.")

    if confidence < CONFIDENCE_THRESHOLD and route != "SEARCH":
        route = "SEARCH"
        reasons.append(
            f"분류 확신도({confidence:.2f})가 기준({CONFIDENCE_THRESHOLD:.2f})보다 낮아 SEARCH로 보정했습니다."
        )

    result["route"] = route
    result["overridden"] = route != result.get("model_route")
    if reasons:
        result["reason"] = f"{result.get('reason', '').strip()} {' '.join(reasons)}".strip()

    if route == "DIRECT":
        result["search_query"] = ""
    elif not result.get("search_query"):
        result["search_query"] = question.strip()

    return result


def classify_question(question: str, current_question: str | None = None) -> dict[str, Any]:
    """Classify contextual wording while applying safety rules to the current turn.

    The first argument may contain resolved conversational context for pronouns.
    The optional current_question is the latest user turn and is the only text
    used to detect explicit search/freshness requirements.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("분류할 질문이 비어 있습니다.")

    cleaned_question = question.strip()
    safety_question = (current_question or cleaned_question).strip()

    fast_decision = _safe_fast_direct_decision(safety_question)
    if fast_decision is not None:
        return fast_decision

    try:
        model_data = _call_router_model(cleaned_question)
        decision = _normalise_model_decision(model_data)
    except (requests.RequestException, ValueError, KeyError, TypeError, RuntimeError) as exc:
        # If classification fails, prefer retrieving evidence to silently
        # treating potentially current information as stable model knowledge.
        decision = {
            "model_route": "ERROR",
            "route": "SEARCH",
            "reason": f"질문 분류에 실패해 안전하게 웹 검색 경로를 선택했습니다: {exc}",
            "search_query": cleaned_question,
            "confidence": 0.0,
            "overridden": True,
            "fallback": True,
        }
        return decision

    return apply_safety_rules(safety_question, decision)
