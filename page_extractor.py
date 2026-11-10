"""Selective, bounded extraction of official source pages with Trafilatura.

Search results are untrusted input. Only sources classified as first-party official
are eligible, redirects are re-validated, and private/local network destinations
are rejected to reduce SSRF risk.
"""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urljoin, urlsplit
from datetime import datetime, timezone

import requests

from search_validator import classify_source

MAX_PAGE_BYTES = 1_500_000
MAX_EXTRACTED_CHARS = 30_000
REDIRECT_STATUSES = {301, 302, 303, 307, 308}


def _is_public_http_url(url: str) -> bool:
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return False
        if parsed.username or parsed.password:
            return False
        host = parsed.hostname.lower()
        if host in {"localhost"} or host.endswith((".localhost", ".local", ".internal")):
            return False
        try:
            addresses = [ipaddress.ip_address(host)]
        except ValueError:
            addresses = [
                ipaddress.ip_address(item[4][0])
                for item in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
            ]
        return bool(addresses) and all(address.is_global for address in addresses)
    except (OSError, ValueError, TypeError):
        return False


def _download_html(url: str, timeout: int = 12) -> tuple[str, str]:
    current_url = url
    headers = {"User-Agent": "LocalAIAgentEvidenceFetcher/1.0"}
    for _ in range(4):
        if not _is_public_http_url(current_url):
            raise ValueError("차단된 URL 또는 공인 네트워크 주소가 아닙니다.")
        with requests.get(
            current_url,
            timeout=timeout,
            headers=headers,
            allow_redirects=False,
            stream=True,
        ) as response:
            if response.status_code in REDIRECT_STATUSES:
                location = response.headers.get("Location")
                if not location:
                    raise ValueError("리디렉션 응답에 Location이 없습니다.")
                current_url = urljoin(current_url, location)
                continue
            response.raise_for_status()
            declared_size = response.headers.get("Content-Length")
            if declared_size and int(declared_size) > MAX_PAGE_BYTES:
                raise ValueError("페이지 크기가 허용 한도를 초과했습니다.")
            chunks = []
            total = 0
            for chunk in response.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_PAGE_BYTES:
                    raise ValueError("페이지 크기가 허용 한도를 초과했습니다.")
                chunks.append(chunk)
            encoding = response.encoding or "utf-8"
            html = b"".join(chunks).decode(encoding, errors="replace")
            return html, current_url
    raise ValueError("리디렉션 횟수가 허용 한도를 초과했습니다.")


def _trafilatura_extract(html: str, url: str) -> tuple[str | None, str | None]:
    import trafilatura

    metadata = trafilatura.extract_metadata(html)
    title = getattr(metadata, "title", None) if metadata else None
    text = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=True,
    )
    if text:
        text = text.strip()[:MAX_EXTRACTED_CHARS]
    return (title.strip() if isinstance(title, str) and title.strip() else None, text or None)


def extract_official_pages(
    results: list[dict],
    max_pages: int = 3,
    timeout: int = 12,
    fetcher=None,
    extractor=None,
) -> dict:
    """Enrich at most max_pages first-party results; failures never fail the search."""
    fetcher = fetcher or _download_html
    extractor = extractor or _trafilatura_extract
    enriched = []
    extracted_count = 0
    errors = []

    try:
        if extractor is _trafilatura_extract:
            import trafilatura  # noqa: F401
    except ImportError:
        return {
            "results": [
                {**row, "extraction_status": "unavailable"}
                for row in (results or [])
            ],
            "extracted_count": 0,
            "attempted_count": 0,
            "errors": ["Trafilatura가 설치되지 않았습니다. requirements.txt를 설치하세요."],
        }

    attempted = 0
    for row in results or []:
        item = dict(row)
        item["retrieved_at"] = datetime.now(timezone.utc).isoformat()
        source_type = item.get("source_type") or classify_source(item.get("url", ""))
        if source_type not in {"official", "company_official"} or attempted >= max(0, max_pages):
            item.setdefault("extraction_status", "skipped_non_official")
            enriched.append(item)
            continue

        attempted += 1
        try:
            html, final_url = fetcher(item.get("url", ""), timeout=timeout)
            if classify_source(final_url) not in {"official", "company_official"}:
                raise ValueError("공식 출처가 아닌 주소로 리디렉션되어 추출을 중단했습니다.")
            title, text = extractor(html, final_url)
            item["final_url"] = final_url
            item["extracted_title"] = title
            item["extracted_text"] = text
            item["extraction_status"] = "extracted" if text else "empty"
            if text:
                extracted_count += 1
        except Exception as exc:
            item["extraction_status"] = "failed"
            item["extraction_error"] = f"{type(exc).__name__}: {exc}"
            errors.append(f"{item.get('source_id', '?')}: {item['extraction_error']}")
        enriched.append(item)

    return {
        "results": enriched,
        "extracted_count": extracted_count,
        "attempted_count": attempted,
        "errors": errors,
    }
