"""Search evidence classification, ranking, and bounded retry for v1.2."""
from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
from datetime import datetime, timezone
import re

from search_planner import plan_search, build_retry_query

TRACKING_KEYS = {"fbclid", "gclid", "dclid", "mc_cid", "mc_eid"}

# First-party technical/product documentation and release sources.
CORE_OFFICIAL_DOMAINS = {"python.org", "postgresql.org", "pypi.org"}
COMPANY_OFFICIAL_DOMAINS = {
    "openai.com", "microsoft.com", "ibm.com", "samsungsds.com",
    "google.com", "amazon.com", "anthropic.com", "salesforce.com",
    "oracle.com", "nvidia.com", "meta.com", "cohere.com",
    "databricks.com", "snowflake.com", "elastic.co", "pinecone.io",
    "weaviate.io", "langchain.com", "langchain.dev", "mistral.ai",
    "huggingface.co",
}
OFFICIAL_DOMAINS = CORE_OFFICIAL_DOMAINS | COMPANY_OFFICIAL_DOMAINS
NEWS_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
    "yna.co.kr", "bloomberg.com", "techcrunch.com",
}
ACADEMIC_DOMAINS = {
    "arxiv.org", "acm.org", "ieee.org", "doi.org", "springer.com",
    "nature.com", "sciencedirect.com", "dl.acm.org",
}
COMMUNITY_DOMAINS = {"github.com", "gitlab.com", "stackoverflow.com"}

VERSION_RE = re.compile(r"(?<![A-Za-z0-9])(?:v)?(\d+\.\d+(?:\.\d+){0,2})(?![A-Za-z0-9])")


def _host_matches(host: str, domain: str) -> bool:
    return host == domain or host.endswith("." + domain)


def normalize_url(url: str) -> str:
    try:
        parsed = urlsplit((url or "").strip())
        scheme = parsed.scheme.lower()
        if scheme not in {"http", "https"} or not parsed.hostname:
            return ""
        host = parsed.hostname.lower()
        port = parsed.port
        if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
            host = f"{host}:{port}"
        path = parsed.path or "/"
        if path != "/":
            path = path.rstrip("/")
        query = [
            (key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if not (key.lower().startswith("utm_") or key.lower() in TRACKING_KEYS)
        ]
        return urlunsplit((scheme, host, path, urlencode(sorted(query)), ""))
    except (TypeError, ValueError):
        return ""


def classify_source(url: str) -> str:
    """Classify ownership/type heuristically; this is not a trust guarantee."""
    host = (urlsplit(url).hostname or "").lower()
    if host.endswith(".gov") or host.endswith(".gov.uk") or host.endswith(".go.kr"):
        return "official"
    if host.endswith(".edu") or host.endswith(".ac.kr"):
        return "academic"
    if any(_host_matches(host, domain) for domain in ACADEMIC_DOMAINS):
        return "academic"
    if any(_host_matches(host, domain) for domain in NEWS_DOMAINS):
        return "news"
    if any(_host_matches(host, domain) for domain in COMMUNITY_DOMAINS):
        return "technical_community"
    if any(_host_matches(host, domain) for domain in CORE_OFFICIAL_DOMAINS):
        return "official"
    if any(_host_matches(host, domain) for domain in COMPANY_OFFICIAL_DOMAINS):
        return "company_official"
    return "other"


def is_official_release_index(result: dict) -> bool:
    """Heuristic for official version/download overview pages, not individual release pages."""
    url = result.get("url", "")
    host = (urlsplit(url).hostname or "").lower()
    path = (urlsplit(url).path or "/").lower().rstrip("/") or "/"
    if not any(_host_matches(host, domain) for domain in OFFICIAL_DOMAINS):
        return False
    exact_index_paths = {"/downloads", "/download/releases", "/doc/versions", "/releases"}
    index_subtrees = {"/downloads/source", "/downloads/windows"}
    return (
        path in exact_index_paths
        or any(path == candidate or path.startswith(candidate + "/") for candidate in index_subtrees)
    )


def _tokens(text: str) -> set[str]:
    stopwords = {"현재", "최신", "최근", "알려", "찾아", "조사", "공식", "자료"}
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9가-힣+#.]{2,}", text or "")
        if token.lower() not in stopwords
    }


def relevance_score(query: str, result: dict) -> float:
    query_tokens = _tokens(query)
    document_tokens = _tokens(" ".join([result.get("title", ""), result.get("content", "")]))
    if not query_tokens or not document_tokens:
        return 0.0
    overlap = len(query_tokens & document_tokens) / len(query_tokens)
    title_tokens = _tokens(result.get("title", ""))
    return min(1.0, overlap + (0.12 if query_tokens & title_tokens else 0.0))


def _published_timestamp(value):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except (TypeError, ValueError, OverflowError):
        return None


def _source_rank(row: dict, now: datetime | None = None) -> tuple:
    source_type = row.get("source_type")
    trust_bonus = {
        "official": 0.24,
        "company_official": 0.20,
        "academic": 0.12,
        "news": 0.08,
        "technical_community": 0.03,
        "other": 0.0,
    }.get(source_type, 0.0)
    published = _published_timestamp(row.get("published"))
    freshness_bonus = 0.0
    if published is not None:
        reference = (now or datetime.now(timezone.utc)).timestamp()
        age_days = max(0.0, (reference - published) / 86400)
        freshness_bonus = 0.08 * max(0.0, 1.0 - age_days / 365)
    row["ranking_score"] = round(row.get("relevance_score", 0.0) + trust_bonus + freshness_bonus, 3)
    return (row["ranking_score"], row.get("relevance_score", 0.0))


def prepare_results(query: str, raw_results: list, max_results: int = 8,
                    now: datetime | None = None) -> list:
    seen, prepared = set(), []
    for item in raw_results or []:
        if not isinstance(item, dict):
            continue
        normalized_url = normalize_url(item.get("url", ""))
        if not normalized_url or normalized_url in seen:
            continue
        seen.add(normalized_url)
        row = dict(item)
        row["url"] = normalized_url
        row["source_type"] = classify_source(normalized_url)
        row["relevance_score"] = round(relevance_score(query, row), 3)
        row["official_release_index"] = is_official_release_index(row)
        row["version_mentions"] = sorted(set(VERSION_RE.findall(
            " ".join([row.get("title", ""), row.get("content", "")])
        )))
        prepared.append(row)
    for row in prepared:
        _source_rank(row, now)
    prepared.sort(key=lambda row: (row["ranking_score"], row["relevance_score"]), reverse=True)
    for index, row in enumerate(prepared[:max_results], 1):
        row["source_id"] = f"S{index}"
    return prepared[:max_results]


def assess_results(results: list, freshness_required: bool = False,
                   requires_official: bool = False,
                   requires_company_official: bool = False,
                   version_query: bool = False) -> dict:
    if not results:
        return {
            "sufficient": False, "reason": "검색 결과가 없습니다.", "count": 0,
            "usable_count": 0, "dated_count": 0, "official_count": 0,
            "company_official_count": 0, "official_release_index_count": 0,
            "version_mentions": [], "version_conflict": False,
        }

    usable = [row for row in results if row.get("relevance_score", 0) >= 0.08]
    dated = [row for row in usable if _published_timestamp(row.get("published")) is not None]
    official = [row for row in usable if row.get("source_type") == "official"]
    company_official = [row for row in usable if row.get("source_type") == "company_official"]
    release_indexes = [row for row in usable if row.get("official_release_index")]
    versions = sorted({
        version for row in usable for version in row.get("version_mentions", [])
    })
    version_conflict = version_query and len(versions) > 1
    index_versions = sorted({
        version for row in release_indexes for version in row.get("version_mentions", [])
    })
    index_text = " ".join(
        " ".join([row.get("title", ""), row.get("content", "")])
        for row in release_indexes
    ).lower()
    index_explicitly_identifies_latest = bool(
        index_versions and re.search(r"(latest|최신|stable|안정|current|현재)", index_text)
    )
    unresolved_version_conflict = version_conflict and not index_explicitly_identifies_latest

    sufficient = len(usable) >= 2
    reasons = []
    if not sufficient:
        reasons.append("질문과 관련된 결과가 2개 미만입니다.")
    if freshness_required and not dated and not (official or company_official):
        sufficient = False
        reasons.append("게시일 또는 1차 출처가 없어 최신성을 확인하기 어렵습니다.")
    if requires_official and not official and not release_indexes:
        sufficient = False
        reasons.append("공식 문서 또는 공식 릴리스 목록을 확보하지 못했습니다.")
    if requires_company_official and not company_official:
        sufficient = False
        reasons.append("기업 공식 사이트의 발표·고객 사례를 확보하지 못했습니다.")
    if unresolved_version_conflict:
        sufficient = False
        reasons.append("서로 다른 버전 주장이 발견됐지만 최신 버전을 명시한 공식 릴리스 목록으로 충돌을 해소하지 못했습니다.")

    if not reasons:
        reason = "요구된 출처 유형과 관련 검색 결과를 확보했습니다."
    else:
        reason = " ".join(reasons)

    return {
        "sufficient": sufficient,
        "reason": reason,
        "count": len(results),
        "usable_count": len(usable),
        "dated_count": len(dated),
        "official_count": len(official),
        "company_official_count": len(company_official),
        "official_release_index_count": len(release_indexes),
        "requires_official": requires_official,
        "requires_company_official": requires_company_official,
        "version_mentions": versions,
        "version_conflict": version_conflict,
        "unresolved_version_conflict": unresolved_version_conflict,
        "official_index_versions": index_versions,
    }


def _call_search(search_fn, query, max_results, time_range):
    try:
        return search_fn(query, max_results=max_results, time_range=time_range), None
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


def search_with_validation(question: str, proposed_query: str, search_fn,
                           now=None, max_results=8) -> dict:
    plan = plan_search(question, proposed_query, now=now)
    first, first_error = _call_search(search_fn, plan["query"], max_results, plan["time_range"])
    results = prepare_results(plan["query"], first, max_results, now=now)
    assessment = assess_results(
        results,
        freshness_required=plan["freshness_required"],
        requires_official=plan["requires_official"],
        requires_company_official=plan["requires_company_official"],
        version_query=plan["version_query"],
    )
    attempts = [{
        "query": plan["query"], "count": len(results),
        "assessment": assessment, "error": first_error,
    }]

    if not assessment["sufficient"]:
        retry_query = plan.get("retry_query") or build_retry_query(plan["query"], plan["current_year"])
        if retry_query != plan["query"]:
            second, second_error = _call_search(search_fn, retry_query, max_results, plan["time_range"])
            results = prepare_results(retry_query, results + second, max_results, now=now)
            assessment = assess_results(
                results,
                freshness_required=plan["freshness_required"],
                requires_official=plan["requires_official"],
                requires_company_official=plan["requires_company_official"],
                version_query=plan["version_query"],
            )
            attempts.append({
                "query": retry_query, "count": len(second),
                "assessment": assessment, "error": second_error,
            })

    return {"results": results, "plan": plan, "assessment": assessment, "attempts": attempts}
