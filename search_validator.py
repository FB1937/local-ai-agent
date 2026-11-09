"""Deterministic search-result normalization and bounded retry."""
from __future__ import annotations
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
import re
from search_planner import plan_search, build_retry_query

TRACKING_KEYS = {"fbclid", "gclid", "dclid", "mc_cid", "mc_eid"}
OFFICIAL_DOMAINS = {"python.org", "postgresql.org", "openai.com", "microsoft.com", "pypi.org"}
NEWS_DOMAINS = {"reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "yna.co.kr", "bloomberg.com"}


def _host_matches(host, domain):
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
        query = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
                 if not (k.lower().startswith("utm_") or k.lower() in TRACKING_KEYS)]
        return urlunsplit((scheme, host, path, urlencode(sorted(query)), ""))
    except (TypeError, ValueError):
        return ""


def classify_source(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    if host.endswith(".gov") or host.endswith(".gov.uk") or host.endswith(".go.kr"):
        return "official"
    if host.endswith(".edu") or host.endswith(".ac.kr"):
        return "academic"
    if any(_host_matches(host, domain) for domain in OFFICIAL_DOMAINS):
        return "official"
    if any(_host_matches(host, domain) for domain in NEWS_DOMAINS):
        return "news"
    if any(_host_matches(host, domain) for domain in {"arxiv.org", "acm.org", "ieee.org", "doi.org", "springer.com"}):
        return "academic"
    if any(_host_matches(host, domain) for domain in {"github.com", "gitlab.com", "stackoverflow.com"}):
        return "technical_community"
    return "other"


def _tokens(text: str) -> set[str]:
    stopwords = {"현재", "최신", "최근", "알려", "찾아", "조사", "공식", "자료"}
    return {token.lower() for token in re.findall(r"[A-Za-z0-9가-힣+#.]{2,}", text or "")
            if token.lower() not in stopwords}


def relevance_score(query: str, result: dict) -> float:
    query_tokens = _tokens(query)
    document_tokens = _tokens(" ".join([result.get("title", ""), result.get("content", "")]))
    if not query_tokens or not document_tokens:
        return 0.0
    overlap = len(query_tokens & document_tokens) / len(query_tokens)
    title_tokens = _tokens(result.get("title", ""))
    return min(1.0, overlap + (0.12 if query_tokens & title_tokens else 0.0))


def prepare_results(query: str, raw_results: list, max_results: int = 8) -> list:
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
        prepared.append(row)
    prepared.sort(key=lambda row: (row["relevance_score"], row["source_type"] == "official",
                                   bool(row.get("published"))), reverse=True)
    for index, row in enumerate(prepared[:max_results], 1):
        row["source_id"] = f"S{index}"
    return prepared[:max_results]


def assess_results(results: list, freshness_required: bool = False) -> dict:
    if not results:
        return {"sufficient": False, "reason": "검색 결과가 없습니다.", "count": 0}
    usable = [row for row in results if row.get("relevance_score", 0) >= 0.08]
    dated = [row for row in usable if row.get("published")]
    official = [row for row in usable if row.get("source_type") == "official"]
    sufficient = len(usable) >= 2
    reason = "관련 검색 결과를 확보했습니다." if sufficient else "관련 결과가 부족합니다."
    if freshness_required and not dated and not official:
        sufficient = False
        reason = "최신성을 확인할 날짜나 공식 출처가 충분하지 않습니다."
    return {"sufficient": sufficient, "reason": reason, "count": len(results),
            "usable_count": len(usable), "dated_count": len(dated), "official_count": len(official)}


def _call_search(search_fn, query, max_results, time_range):
    try:
        return search_fn(query, max_results=max_results, time_range=time_range), None
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


def search_with_validation(question: str, proposed_query: str, search_fn, now=None, max_results=8) -> dict:
    plan = plan_search(question, proposed_query, now=now)
    first, first_error = _call_search(search_fn, plan["query"], max_results, plan["time_range"])
    results = prepare_results(plan["query"], first, max_results)
    assessment = assess_results(results, plan["freshness_required"])
    attempts = [{"query": plan["query"], "count": len(results), "assessment": assessment, "error": first_error}]
    if not assessment["sufficient"]:
        retry_query = build_retry_query(plan["query"], plan["current_year"])
        if retry_query != plan["query"]:
            second, second_error = _call_search(search_fn, retry_query, max_results, plan["time_range"])
            results = prepare_results(retry_query, results + second, max_results)
            assessment = assess_results(results, plan["freshness_required"])
            attempts.append({"query": retry_query, "count": len(second), "assessment": assessment, "error": second_error})
    return {"results": results, "plan": plan, "assessment": assessment, "attempts": attempts}
