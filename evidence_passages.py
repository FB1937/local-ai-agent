"""Deterministic selection of query-relevant passages from extracted source text."""
from __future__ import annotations

import re


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


def select_relevant_passages(
    question: str,
    text: str | None,
    max_passages: int = 4,
    max_chars: int = 3200,
) -> list[dict]:
    """Select high-overlap sentences; return no body excerpt when lexical match is absent.

    This is a lexical retrieval heuristic, not semantic entailment or fact verification.
    """
    query_tokens = _tokens(question)
    if not query_tokens or not text:
        return []

    segments = [segment.strip() for segment in SENTENCE_SPLIT_RE.split(text) if segment.strip()]
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
        })
        used_chars += len(excerpt)

    selected.sort(key=lambda item: item["segment_index"])
    return selected
