"""Deterministic selection of query-relevant passages from extracted source text."""
from __future__ import annotations

import math
import os
import re

import requests


TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣+#.]{2,}")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。])\s+|\n+")
STOPWORDS = {
    "현재", "최신", "최근", "알려", "찾아", "조사", "공식", "자료",
    "무엇", "무엇인가", "어떻게", "대해서", "관련", "설명",
}


def _tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in TOKEN_RE.findall(text or "")
        if token.casefold() not in STOPWORDS
    }


def _split_segments(text: str) -> list[str]:
    raw_segments = [
        segment.strip()
        for segment in SENTENCE_SPLIT_RE.split(text)
        if segment.strip()
    ]
    segments = []
    pending_page_marker = ""
    for segment in raw_segments:
        if re.fullmatch(r"\\[PDF 페이지 \\d+\\]", segment):
            pending_page_marker = segment
            continue
        if pending_page_marker:
            segment = f"{pending_page_marker} {segment}"
            pending_page_marker = ""
        segments.append(segment)
    return segments


def _select_lexical_passages(
    question: str,
    segments: list[str],
    max_passages: int,
    max_chars: int,
    fallback_reason: str | None = None,
) -> list[dict]:
    query_tokens = _tokens(question)
    if not query_tokens:
        return []

    candidates = []
    for index, segment in enumerate(segments):
        segment_tokens = _tokens(segment)
        overlap = query_tokens & segment_tokens
        if not overlap:
            continue
        score = len(overlap) / len(query_tokens)
        candidates.append({
            "index": index,
            "text": segment,
            "score": round(score, 3),
            "matched_terms": sorted(overlap),
        })

    candidates.sort(key=lambda item: (-item["score"], item["index"]))
    selected = []
    used_chars = 0
    for item in candidates:
        remaining = max_chars - used_chars
        if remaining <= 0 or len(selected) >= max(0, max_passages):
            break
        excerpt = item["text"][:remaining].strip()
        if len(excerpt) < 40:
            continue
        selected.append({
            "text": excerpt,
            "score": item["score"],
            "matched_terms": item["matched_terms"],
            "segment_index": item["index"],
            "retrieval_method": "lexical_fallback",
            **({"fallback_reason": fallback_reason} if fallback_reason else {}),
        })
        used_chars += len(excerpt)

    selected.sort(key=lambda item: item["segment_index"])
    return selected


def _ollama_embed(texts: list[str]) -> list[list[float]]:
    base_url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3").strip()
    timeout = max(5, min(int(os.getenv("OLLAMA_EMBEDDING_TIMEOUT", "60")), 120))
    response = requests.post(
        f"{base_url}/api/embed",
        json={"model": model, "input": texts, "truncate": False},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    vectors = payload.get("embeddings")
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        raise ValueError("Ollama 임베딩 응답의 벡터 개수가 입력과 일치하지 않습니다.")
    validated = []
    for vector in vectors:
        if not isinstance(vector, list) or not vector:
            raise ValueError("Ollama가 비어 있거나 잘못된 임베딩 벡터를 반환했습니다.")
        values = [float(value) for value in vector]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("임베딩 벡터에 유한하지 않은 값이 포함되어 있습니다.")
        validated.append(values)
    return validated


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("임베딩 벡터 차원이 서로 다릅니다.")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _select_semantic_passages(
    question: str,
    segments: list[str],
    max_passages: int,
    max_chars: int,
    embedder=None,
) -> list[dict]:
    if not segments:
        return []
    # Bound the embedding request so long pages cannot create unbounded batches.
    indexed_segments = [
        (index, segment[:1200])
        for index, segment in enumerate(segments[:80])
        if len(segment.strip()) >= 40
    ]
    if not indexed_segments:
        return []
    embedder = embedder or _ollama_embed
    vectors = embedder([question[:2000]] + [item[1] for item in indexed_segments])
    if len(vectors) != len(indexed_segments) + 1:
        raise ValueError("임베딩 결과 개수가 예상과 다릅니다.")

    query_vector = vectors[0]
    threshold = float(os.getenv("OLLAMA_EMBEDDING_MIN_SCORE", "0.35"))
    candidates = []
    for (index, segment), vector in zip(indexed_segments, vectors[1:]):
        score = _cosine_similarity(query_vector, vector)
        if score >= threshold:
            candidates.append({
                "index": index,
                "text": segment,
                "score": round(score, 4),
                "matched_terms": sorted(_tokens(question) & _tokens(segment)),
            })

    candidates.sort(key=lambda item: (-item["score"], item["index"]))
    selected = []
    used_chars = 0
    for item in candidates:
        remaining = max_chars - used_chars
        if remaining <= 0 or len(selected) >= max(0, max_passages):
            break
        excerpt = item["text"][:remaining].strip()
        if len(excerpt) < 40:
            continue
        selected.append({
            "text": excerpt,
            "score": item["score"],
            "matched_terms": item["matched_terms"],
            "segment_index": item["index"],
            "retrieval_method": "semantic",
        })
        used_chars += len(excerpt)

    selected.sort(key=lambda item: item["segment_index"])
    return selected


def select_relevant_passages(
    question: str,
    text: str | None,
    max_passages: int = 4,
    max_chars: int = 3200,
    semantic_enabled: bool | None = None,
    embedder=None,
) -> list[dict]:
    """Select relevant passages using embeddings when enabled, else lexical matching.

    Semantic scores are similarity heuristics, not factual entailment or verification.
    If the embedding model is unavailable or returns invalid data, lexical retrieval
    remains available as a deterministic fallback.
    """
    if not text:
        return []
    segments = _split_segments(text)
    if semantic_enabled is None:
        semantic_enabled = os.getenv(
            "ENABLE_SEMANTIC_RETRIEVAL", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}

    if semantic_enabled:
        try:
            return _select_semantic_passages(
                question, segments, max_passages, max_chars, embedder=embedder
            )
        except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
            return _select_lexical_passages(
                question, segments, max_passages, max_chars,
                fallback_reason=f"{type(exc).__name__}: {exc}",
            )

    return _select_lexical_passages(
        question, segments, max_passages, max_chars
    )
