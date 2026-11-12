"""Selective, bounded extraction of official HTML and PDF sources.

Search results are untrusted input. Only first-party official sources are eligible,
redirects are re-validated, and private/local network destinations are rejected.
"""
from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import ipaddress
import socket
from urllib.parse import urljoin, urlsplit

import re
import requests

from search_validator import classify_source, is_official_release_index
from content_quality import assess_extracted_content

MAX_PAGE_BYTES = 1_500_000
MAX_EXTRACTED_CHARS = 30_000
MAX_PDF_PAGES = 40
REDIRECT_STATUSES = {301, 302, 303, 307, 308}
PDF_MIME_TYPES = {"application/pdf", "application/x-pdf"}


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


def _download_resource(url: str, timeout: int = 12) -> tuple[bytes, str, str]:
    """Download a bounded public HTTP resource without automatic redirects."""
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
                raise ValueError("문서 크기가 허용 한도를 초과했습니다.")
            chunks = []
            total = 0
            for chunk in response.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_PAGE_BYTES:
                    raise ValueError("문서 크기가 허용 한도를 초과했습니다.")
                chunks.append(chunk)
            return (
                b"".join(chunks),
                current_url,
                response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower(),
            )
    raise ValueError("리디렉션 횟수가 허용 한도를 초과했습니다.")


def _download_html(url: str, timeout: int = 12) -> tuple[str, str]:
    data, final_url, content_type = _download_resource(url, timeout=timeout)
    if content_type and content_type not in {"text/html", "application/xhtml+xml"}:
        raise ValueError(f"HTML이 아닌 콘텐츠 유형입니다: {content_type}")
    return data.decode("utf-8", errors="replace"), final_url


def _download_pdf(url: str, timeout: int = 12) -> tuple[bytes, str, str]:
    data, final_url, content_type = _download_resource(url, timeout=timeout)
    if content_type not in PDF_MIME_TYPES and not data.startswith(b"%PDF-"):
        raise ValueError(f"PDF 문서가 아닙니다: {content_type or '알 수 없는 콘텐츠 유형'}")
    return data, final_url, content_type or "application/pdf"


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


def _pypdf_extract(data: bytes, url: str) -> tuple[str | None, str | None, int]:
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(data), strict=False)
    if reader.is_encrypted:
        raise ValueError("암호화된 PDF는 현재 추출할 수 없습니다.")
    page_count = len(reader.pages)
    if page_count == 0:
        raise ValueError("PDF에 페이지가 없습니다.")

    chunks = []
    total_chars = 0
    for index, page in enumerate(reader.pages[:MAX_PDF_PAGES]):
        remaining = MAX_EXTRACTED_CHARS - total_chars
        if remaining <= 0:
            break
        page_text = (page.extract_text() or "").strip()
        if page_text:
            page_text = page_text[:remaining]
            chunks.append(f"[PDF 페이지 {index + 1}]\n{page_text}")
            total_chars += len(page_text)

    metadata = reader.metadata
    raw_title = getattr(metadata, "title", None) if metadata else None
    title = raw_title.strip() if isinstance(raw_title, str) and raw_title.strip() else None
    text = "\n\n".join(chunks).strip()
    return title, text or None, page_count


def _is_pdf_result(row: dict) -> bool:
    url = row.get("url", "")
    path = (urlsplit(url).path or "").lower()
    content_type = str(row.get("content_type", "")).split(";", 1)[0].strip().lower()
    return path.endswith(".pdf") or content_type in PDF_MIME_TYPES



_RELEASE_VERSION_RE = re.compile(r"\b\d+\.\d+(?:\.\d+)?\b")
_PRERELEASE_RE = re.compile(
    r"(?:alpha|beta|pre[- ]?release|release[- ]?candidate|\d+(?:\.\d+){1,2}(?:a|b|rc)\d*|사전\s*릴리스|프리릴리스)",
    re.I,
)
_RELEASE_NOTES_RE = re.compile(
    r"(?:what.?s new|release notes?|changelog|변경 사항|릴리스 노트)",
    re.I,
)


def _extraction_priority(row: dict, original_index: int) -> tuple:
    """Choose high-value official pages before consuming the bounded extraction budget."""
    source_type = row.get("source_type") or classify_source(row.get("url", ""))
    if source_type not in {"official", "company_official"}:
        return (9, original_index)
    if is_official_release_index(row):
        return (0, -float(row.get("relevance_score", 0.0)), original_index)

    descriptor = " ".join(str(row.get(key) or "") for key in ("title", "url", "content"))
    if _PRERELEASE_RE.search(descriptor):
        category = 4
    elif _RELEASE_NOTES_RE.search(descriptor):
        category = 1
    elif _RELEASE_VERSION_RE.search(descriptor):
        category = 2
    else:
        category = 3
    return (category, -float(row.get("relevance_score", 0.0)), original_index)


def extract_official_pages(
    results: list[dict],
    max_pages: int = 3,
    timeout: int = 12,
    fetcher=None,
    extractor=None,
    pdf_fetcher=None,
    pdf_extractor=None,
) -> dict:
    """Enrich bounded official HTML/PDF results; extraction failures never fail search."""
    fetcher = fetcher or _download_html
    extractor = extractor or _trafilatura_extract
    pdf_fetcher = pdf_fetcher or _download_pdf
    pdf_extractor = pdf_extractor or _pypdf_extract
    enriched = []
    extracted_count = 0
    html_extracted_count = 0
    pdf_extracted_count = 0
    html_attempted_count = 0
    pdf_attempted_count = 0
    attempted = 0
    errors = []

    html_available = True
    pdf_available = True
    try:
        if extractor is _trafilatura_extract:
            import trafilatura  # noqa: F401
    except ImportError:
        html_available = False
    try:
        if pdf_extractor is _pypdf_extract:
            import pypdf  # noqa: F401
    except ImportError:
        pdf_available = False

    ordered_results = sorted(
        enumerate(results or []),
        key=lambda pair: _extraction_priority(pair[1], pair[0]),
    )
    for original_index, row in ordered_results:
        item = dict(row)
        item["retrieved_at"] = datetime.now(timezone.utc).isoformat()
        source_type = item.get("source_type") or classify_source(item.get("url", ""))
        if source_type not in {"official", "company_official"}:
            item.setdefault("extraction_status", "skipped_non_official")
            item.setdefault("extracted_char_count", 0)
            enriched.append((original_index, item))
            continue
        if attempted >= max(0, max_pages):
            item.setdefault("extraction_status", "skipped_page_limit")
            item.setdefault("extracted_char_count", 0)
            enriched.append((original_index, item))
            continue

        is_pdf = _is_pdf_result(item)
        if is_pdf and not pdf_available:
            item["extraction_status"] = "unavailable"
            item["extraction_error"] = "pypdf가 설치되지 않았습니다. requirements.txt를 설치하세요."
            item["extracted_char_count"] = 0
            errors.append(f"{item.get('source_id', '?')}: {item['extraction_error']}")
            enriched.append((original_index, item))
            continue
        if not is_pdf and not html_available:
            item["extraction_status"] = "unavailable"
            item["extraction_error"] = "Trafilatura가 설치되지 않았습니다. requirements.txt를 설치하세요."
            item["extracted_char_count"] = 0
            errors.append(f"{item.get('source_id', '?')}: {item['extraction_error']}")
            enriched.append((original_index, item))
            continue

        attempted += 1
        if is_pdf:
            pdf_attempted_count += 1
        else:
            html_attempted_count += 1
        try:
            if is_pdf:
                data, final_url, content_type = pdf_fetcher(item.get("url", ""), timeout=timeout)
                if classify_source(final_url) not in {"official", "company_official"}:
                    raise ValueError("공식 출처가 아닌 주소로 리디렉션되어 추출을 중단했습니다.")
                title, text, page_count = pdf_extractor(data, final_url)
                item["document_type"] = "pdf"
                item["content_type"] = content_type
                item["extracted_page_count"] = min(page_count, MAX_PDF_PAGES)
                item["pdf_page_limit_reached"] = page_count > MAX_PDF_PAGES
                item["extraction_engine"] = "pypdf"
                if text:
                    pdf_extracted_count += 1
            else:
                html, final_url = fetcher(item.get("url", ""), timeout=timeout)
                if classify_source(final_url) not in {"official", "company_official"}:
                    raise ValueError("공식 출처가 아닌 주소로 리디렉션되어 추출을 중단했습니다.")
                title, text = extractor(html, final_url)
                item["document_type"] = "html"
                item["extraction_engine"] = "trafilatura"
                if text:
                    html_extracted_count += 1

            item["final_url"] = final_url
            item["extracted_title"] = title
            item["extracted_text"] = text
            item["extracted_char_count"] = len(text or "")
            item["extraction_status"] = "extracted" if text else "empty"
            if not text:
                item["extraction_error"] = "본문 텍스트를 추출하지 못했습니다."
            if text:
                extracted_count += 1
        except Exception as exc:
            item["extraction_status"] = "failed"
            item["extracted_char_count"] = 0
            item["extraction_error"] = f"{type(exc).__name__}: {exc}"
            errors.append(f"{item.get('source_id', '?')}: {item['extraction_error']}")
        if item.get("extraction_status") in {"extracted", "empty"}:
            item["content_quality"] = assess_extracted_content(
                item.get("extracted_text"),
                document_type=item.get("document_type", "html"),
            )
        else:
            item["content_quality"] = {
                "status": "not_assessed",
                "reason": item.get("extraction_error", "본문 추출이 수행되지 않았습니다."),
                "is_factual_verification": False,
            }
        enriched.append((original_index, item))

    # Extraction order is an internal scheduling detail; preserve search order downstream.
    enriched = [item for _, item in sorted(enriched, key=lambda pair: pair[0])]

    status_counts = {}
    for item in enriched:
        status = item.get("extraction_status", "not_attempted")
        status_counts[status] = status_counts.get(status, 0) + 1

    return {
        "results": enriched,
        "status_counts": status_counts,
        "extracted_count": extracted_count,
        "html_extracted_count": html_extracted_count,
        "pdf_extracted_count": pdf_extracted_count,
        "html_attempted_count": html_attempted_count,
        "pdf_attempted_count": pdf_attempted_count,
        "attempted_count": attempted,
        "errors": errors,
    }
