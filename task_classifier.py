"""Stage 3: lightweight intent/task classification.

This is a deterministic first-pass classifier, not a calibrated ML model.
Routing remains the responsibility of router.py.
"""
import re


PATTERNS = {
    "content_transformation": r"(번역|요약|교정|수정|다듬|재작성|변환)",
    "creation": r"(작성|만들어|생성|구현|개발|설계|코드 짜|초안)",
    "troubleshooting": r"(오류|에러|문제|실패|안 되|작동하지|디버깅|원인)",
    "comparison": r"(비교|차이|장단점|어느 .* 낫|추천)",
    "learning": r"(설명|원리|개념|공부|학습|가르쳐|이해|어떻게\s*(?:돼|되)|무슨\s*(?:일|차이)|무엇이\s*(?:달라|다르))",
    "information_seeking": r"(찾아|검색|조사|최신|최근|현황|동향|자료를\s*확인)",
    "execution": r"(실행|저장해|업로드|다운로드|예약|전송|등록|삭제)",
}
FRESHNESS_RE = re.compile(
    r"(최신|최근|현재|오늘|실시간|올해|이번\s*(주|달|해)|20\d{2}년|latest|current)",
    re.I,
)
DOMAIN_PATTERNS = {
    "software_development": r"(python|sql|코드|프로그래밍|개발|api|github|데이터베이스)",
    "data_engineering": r"(데이터\s*엔지니어|파이프라인|etl|데이터\s*분석|warehouse|데이터베이스)",
    "artificial_intelligence": r"(ai|llm|rag|에이전트|머신러닝|딥러닝|인공지능)",
    "career": r"(취업|채용|이력서|면접|자격증|커리어)",
    "general": r".*",
}


def classify_task(question: str) -> dict:
    text = (question or "").strip()
    if not text:
        raise ValueError("분류할 질문이 비어 있습니다.")

    matched = [
        name for name, pattern in PATTERNS.items()
        if re.search(pattern, text, re.I)
    ]
    if not matched:
        matched = ["information_seeking"]

    if "troubleshooting" in matched:
        intent = "troubleshooting"
    elif "execution" in matched:
        intent = "action_request"
    elif "creation" in matched:
        intent = "creation"
    elif "content_transformation" in matched:
        intent = "transformation"
    elif "comparison" in matched:
        intent = "decision_support"
    elif "learning" in matched:
        intent = "learning"
    else:
        intent = "information_seeking"

    domain = next(
        (name for name, pattern in DOMAIN_PATTERNS.items()
         if name != "general" and re.search(pattern, text, re.I)),
        "general",
    )
    freshness_required = bool(FRESHNESS_RE.search(text))

    task_types = []
    if "information_seeking" in matched:
        task_types.append("retrieval")
    if "comparison" in matched:
        task_types.append("comparison")
    if "learning" in matched:
        task_types.append("explanation")
    if "creation" in matched:
        task_types.append("generation")
    if "content_transformation" in matched:
        task_types.append("transformation")
    if "troubleshooting" in matched:
        task_types.append("diagnosis")
    if "execution" in matched:
        task_types.append("tool_execution")
    if not task_types:
        task_types.append("analysis")

    tool_candidates = []
    explicit_retrieval = bool(re.search(r"(찾아|검색|조사|자료를\\s*확인)", text, re.I))
    if freshness_required or explicit_retrieval:
        tool_candidates.append("web_search")
    if re.search(r"(계산|합계|평균|통계|분석|csv|엑셀)", text, re.I):
        tool_candidates.append("python")
    if re.search(r"(파일|pdf|문서|업로드한)", text, re.I):
        tool_candidates.append("file_reader")
    if re.search(r"(데이터베이스|sqlite|mysql|sql)", text, re.I):
        tool_candidates.append("database")

    output_format = "unspecified"
    if re.search(r"(표로|테이블)", text):
        output_format = "table"
    elif re.search(r"(목록|리스트|체크리스트)", text):
        output_format = "list"
    elif re.search(r"(코드|구현해|프로그램)", text, re.I):
        output_format = "code_or_implementation"
    elif re.search(r"(보고서|문서|노션)", text):
        output_format = "document"

    return {
        "intent": intent,
        "matched_intents": matched,
        "task_types": task_types,
        "domain": domain,
        "freshness_required": freshness_required,
        "output_format": output_format,
        "tool_candidates": tool_candidates,
        "method": "deterministic_heuristics",
        "limitations": "규칙 기반 1차 분류이며 의미 이해나 확률 보정을 보장하지 않습니다.",
    }
