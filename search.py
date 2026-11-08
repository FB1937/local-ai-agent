"""SearXNG search client."""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8080").rstrip("/")


def search_web(query, max_results=5):
    """Search SearXNG and return a normalised list of results."""
    if not SEARXNG_URL:
        print("[검색 오류] SEARXNG_URL 환경변수가 설정되지 않았습니다.")
        return []

    if not query or not str(query).strip():
        print("[검색 오류] 검색어가 비어 있습니다.")
        return []

    url = f"{SEARXNG_URL}/search"
    params = {
        "q": str(query).strip(),
        "format": "json",
        "language": "auto",
        "safesearch": 0,
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if not isinstance(data, dict):
            raise ValueError("검색 응답의 JSON 최상위 값이 객체가 아닙니다.")

        results = []
        for item in data.get("results", [])[:max_results]:
            if not isinstance(item, dict):
                continue
            results.append(
                {
                    "title": item.get("title") or "제목 없음",
                    "url": item.get("url") or "",
                    "content": item.get("content") or "",
                }
            )
        return results

    except requests.RequestException as exc:
        print(f"[검색 오류] {exc}")
        return []
    except ValueError as exc:
        print(f"[JSON 처리 오류] {exc}")
        return []
