"""SearXNG client with normalized result metadata."""
from __future__ import annotations
import os
from urllib.parse import urlparse
import requests
from dotenv import load_dotenv

load_dotenv()
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8080").rstrip("/")

def search_web(query, max_results=8, time_range=None):
    """Search SearXNG. Returns normalized results; never fabricates results."""
    if not SEARXNG_URL or not query or not str(query).strip():
        return []
    params = {"q": str(query).strip(), "format": "json", "language": "auto", "safesearch": 0}
    if time_range in {"day", "week", "month", "year"}:
        params["time_range"] = time_range
    try:
        response = requests.get(f"{SEARXNG_URL}/search", params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            return []
        results = []
        for item in data.get("results", [])[:max_results]:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            results.append({
                "title": str(item.get("title") or "제목 없음"),
                "url": url,
                "content": str(item.get("content") or ""),
                "published": item.get("publishedDate") or item.get("published_date"),
                "engine": item.get("engine"),
                "score": item.get("score"),
                "source_id": None,
            })
        return results
    except (requests.RequestException, ValueError, TypeError) as exc:
        print(f"[검색 오류] {exc}")
        return []
