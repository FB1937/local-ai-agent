"""Deterministic search planning helpers for v1.2."""
from __future__ import annotations
from datetime import datetime
import re

FRESH_RE = re.compile(r"(최신|최근|현재|현시점|실시간|오늘|이번\s*주|이번\s*달|올해|latest|current|today)", re.I)
HISTORICAL_RE = re.compile(r"(처음\s*출시|출시\s*당시|역사|과거|옛날|역대|historical|originally released)", re.I)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
VERSION_TOPIC_RE = re.compile(r"(버전|version|release|릴리스|배포판|stable|안정\s*버전)", re.I)
LATEST_VERSION_RE = re.compile(r"(최신|현재|latest|current|stable|안정)", re.I)
COMPANY_CASE_RE = re.compile(
    r"(기업|회사|고객|도입|활용\s*사례|적용\s*사례|실제\s*사례|case\s*study|customer\s*story|enterprise|deployment)",
    re.I,
)


def plan_search(question: str, proposed_query: str = "", now: datetime | None = None) -> dict:
    now = now or datetime.now().astimezone()
    question = (question or "").strip()
    query = (proposed_query or "").strip() or question
    historical_query = bool(HISTORICAL_RE.search(question))
    freshness_required = bool(FRESH_RE.search(question)) and not historical_query
    explicit_years = sorted({int(value) for value in YEAR_RE.findall(question)})

    if freshness_required:
        if len(explicit_years) == 1:
            requested_year = str(explicit_years[0])
            query = YEAR_RE.sub(requested_year, query)
            if not YEAR_RE.search(query):
                query = f"{query} {requested_year}".strip()
        elif not explicit_years:
            query = YEAR_RE.sub(str(now.year), query)
            if not YEAR_RE.search(query):
                query = f"{query} {now.year}".strip()

    version_query = bool(VERSION_TOPIC_RE.search(question) and LATEST_VERSION_RE.search(question))
    requires_official = version_query or bool(
        re.search(r"(공식\s*(배포|문서|발표|정책)|official\s*(release|documentation))", question, re.I)
    )
    requires_company_official = bool(COMPANY_CASE_RE.search(question))

    lowered = question.lower()
    time_range = None
    if re.search(r"(최근\s*24시간|지난\s*24시간|오늘|실시간|today|last\s*24\s*hours)", lowered):
        time_range = "day"
    elif re.search(r"(최근\s*한\s*달|최근\s*1개월|지난\s*30일|이번\s*달|last\s*month)", lowered):
        time_range = "month"
    elif re.search(r"(최근\s*일\s*주일|최근\s*7일|지난\s*일주일|이번\s*주|this\s*week)", lowered):
        time_range = "week"

    if requires_company_official:
        retry_query = f'{query} "실제 고객사" "도입" "구축" "운영" "고객 사례" "customer story" "deployed at"
    elif requires_official:
        retry_query = f"{query} 공식 릴리스 목록 official release"
    else:
        retry_query = ""

    return {
        "query": query,
        "retry_query": retry_query,
        "time_range": time_range,
        "freshness_required": freshness_required,
        "requires_official": requires_official,
        "requires_company_official": requires_company_official,
        "version_query": version_query,
        "as_of": now.date().isoformat(),
        "current_year": now.year,
        "historical_query": historical_query,
    }


def build_retry_query(query: str, current_year: int) -> str:
    cleaned = (query or "").strip()
    suffix = f"{current_year} 공식 문서 최신 자료"
    if suffix.lower() not in cleaned.lower():
        return f"{cleaned} {suffix}".strip()
    return f"{cleaned} 원문 1차 출처"
