"""Structured claim/evidence contracts and claim-specific passage retrieval."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from evidence_models import EvidenceSource
from evidence_passages import select_relevant_passages

Importance = Literal["high", "normal", "low"]
EvidenceRelation = Literal["supports", "contradicts", "insufficient", "irrelevant"]
_CITATION_RE = re.compile(r"\[(S\d+)\]")


class InformationNeed(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=4000)
    required_aspects: list[str] = Field(default_factory=list)
    freshness_required: bool = False
    scope: str | None = None
    condition: str | None = None


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_id: str = Field(pattern=r"^C\d+$")
    text: str = Field(min_length=1, max_length=4000)
    importance: Importance = "normal"
    qualifiers: list[str] = Field(default_factory=list)
    cited_source_ids: list[str] = Field(default_factory=list)


class EvidencePassage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_id: str = Field(pattern=r"^E\d+$")
    source_id: str = Field(pattern=r"^S\d+$")
    text: str = Field(min_length=1, max_length=2000)
    location: str
    source_url: str | None = None
    claim_id: str | None = None
    retrieval_score: float | None = None
    retrieval_method: str | None = None
    published_at: str | None = None
    retrieved_at: datetime | None = None
    source_group: str | None = None
    canonical_source_id: str | None = None
    scope: str | None = None
    condition: str | None = None
    valid_time_start: str | None = None
    valid_time_end: str | None = None


class ClaimEvidenceLink(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_id: str = Field(pattern=r"^C\d+$")
    evidence_id: str = Field(pattern=r"^E\d+$")
    relation: EvidenceRelation | None = None
    reason: str = "판정 대기: 관련 구절을 검색했지만 의미적 지지는 아직 평가하지 않았습니다."


class ClaimAnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claims: list[Claim] = Field(default_factory=list)
    passages: list[EvidencePassage] = Field(default_factory=list)
    links: list[ClaimEvidenceLink] = Field(default_factory=list)
    unmapped_source_ids: list[str] = Field(default_factory=list)
    claims_without_evidence: list[str] = Field(default_factory=list)


def _split_answer_sentences(answer: str) -> list[str]:
    text = answer or ""
    sentences: list[str] = []
    start = index = 0
    while index < len(text):
        char = text[index]
        if char in ".!?。！？":
            if char == "." and index > 0 and index + 1 < len(text):
                if text[index - 1].isdigit() and text[index + 1].isdigit():
                    index += 1
                    continue
            cursor = index + 1
            boundary_space = False
            while cursor < len(text) and text[cursor].isspace() and text[cursor] != "\n":
                boundary_space = True
                cursor += 1
            while cursor < len(text):
                match = _CITATION_RE.match(text, cursor)
                if not match:
                    break
                cursor = match.end()
                while cursor < len(text) and text[cursor].isspace() and text[cursor] != "\n":
                    boundary_space = True
                    cursor += 1
            if cursor == len(text) or boundary_space:
                sentence = text[start:cursor].strip()
                if sentence:
                    sentences.append(sentence)
                start = cursor
                index = cursor
                continue
        if char == "\n":
            sentence = text[start:index].strip()
            if sentence:
                sentences.append(sentence)
            while index < len(text) and text[index] == "\n":
                index += 1
            start = index
            continue
        index += 1
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def extract_cited_claims(answer: str) -> list[Claim]:
    claims: list[Claim] = []
    for sentence in _split_answer_sentences(answer):
        claims.append(Claim(
            claim_id=f"C{len(claims) + 1}",
            text=sentence,
            cited_source_ids=list(dict.fromkeys(_CITATION_RE.findall(sentence))),
        ))
    return claims


def build_evidence_passages(
    sources: list[EvidenceSource | dict],
    *,
    max_passages_per_source: int = 40,
    max_chars_per_passage: int = 1200,
) -> list[EvidencePassage]:
    """Build bounded source passages without making a relevance judgment."""
    if max_passages_per_source < 1 or max_chars_per_passage < 1:
        raise ValueError("passage limits must be positive")
    passages: list[EvidencePassage] = []
    for raw_source in sources or []:
        source = raw_source if isinstance(raw_source, EvidenceSource) else EvidenceSource.model_validate(raw_source)
        body = source.extracted_text or source.content or ""
        segments = [s.strip() for s in re.split(r"(?<=[.!?。！？])\s+|\n+", body) if s.strip()]
        for index, segment in enumerate(segments[:max_passages_per_source], 1):
            passages.append(EvidencePassage(
                evidence_id=f"E{len(passages) + 1}",
                source_id=source.source_id,
                text=segment[:max_chars_per_passage],
                location=f"sentence:{index}",
                source_url=source.url,
                published_at=source.published,
                retrieved_at=source.retrieved_at,
                source_group=source.source_group,
                canonical_source_id=source.canonical_source_id,
                scope=source.scope,
                condition=source.condition,
                valid_time_start=source.valid_time_start,
                valid_time_end=source.valid_time_end,
            ))
    return passages


def map_claims_to_cited_passages(
    claims: list[Claim],
    passages: list[EvidencePassage],
) -> tuple[list[ClaimEvidenceLink], list[str]]:
    """Never map a claim to an uncited source or another claim's selected passage."""
    by_source: dict[str, list[EvidencePassage]] = {}
    for passage in passages:
        by_source.setdefault(passage.source_id, []).append(passage)
    links: list[ClaimEvidenceLink] = []
    unmapped: list[str] = []
    for claim in claims:
        for source_id in claim.cited_source_ids:
            source_passages = by_source.get(source_id, [])
            if not source_passages:
                if source_id not in unmapped:
                    unmapped.append(source_id)
                continue
            candidates = [
                p for p in source_passages
                if p.claim_id is None or p.claim_id == claim.claim_id
            ]
            # A known source with no passage for this claim is a claim-level evidence gap.
            if not candidates:
                continue
            links.extend(
                ClaimEvidenceLink(claim_id=claim.claim_id, evidence_id=p.evidence_id)
                for p in candidates
            )
    return links, unmapped


def retrieve_claim_evidence_passages(
    claims: list[Claim],
    sources: list[EvidenceSource | dict],
    *,
    max_passages_per_source: int = 4,
    max_chars_per_source: int = 4800,
    semantic_enabled: bool | None = None,
    embedder=None,
) -> tuple[list[EvidencePassage], list[str]]:
    """Search each claim only within the exact source IDs cited by that claim."""
    if max_passages_per_source < 1 or max_chars_per_source < 1:
        raise ValueError("retrieval limits must be positive")
    source_map: dict[str, EvidenceSource] = {}
    for raw_source in sources or []:
        source = raw_source if isinstance(raw_source, EvidenceSource) else EvidenceSource.model_validate(raw_source)
        source_map[source.source_id] = source

    passages: list[EvidencePassage] = []
    unmapped: list[str] = []
    for claim in claims:
        for source_id in claim.cited_source_ids:
            source = source_map.get(source_id)
            if source is None:
                if source_id not in unmapped:
                    unmapped.append(source_id)
                continue
            body = source.extracted_text or source.content or ""
            selected = select_relevant_passages(
                claim.text, body,
                max_passages=max_passages_per_source,
                max_chars=max_chars_per_source,
                semantic_enabled=semantic_enabled,
                embedder=embedder,
            )
            for item in selected:
                passages.append(EvidencePassage(
                    evidence_id=f"E{len(passages) + 1}",
                    source_id=source.source_id,
                    text=item["text"][:2000],
                    location=f"sentence:{item.get('segment_index', 0) + 1}",
                    source_url=source.url,
                    claim_id=claim.claim_id,
                    retrieval_score=float(item["score"]),
                    retrieval_method=item.get("retrieval_method"),
                    published_at=source.published,
                    retrieved_at=source.retrieved_at,
                    source_group=source.source_group,
                    canonical_source_id=source.canonical_source_id,
                    scope=source.scope,
                    condition=source.condition,
                    valid_time_start=source.valid_time_start,
                    valid_time_end=source.valid_time_end,
                ))
    return passages, unmapped


def analyze_claim_evidence(
    answer: str,
    sources: list[EvidenceSource | dict],
    *,
    max_passages_per_source: int = 4,
    max_chars_per_source: int = 4800,
    semantic_enabled: bool | None = None,
    embedder=None,
) -> ClaimAnalysisResult:
    """Build a claim-specific, provenance-preserving evidence candidate graph."""
    claims = extract_cited_claims(answer)
    passages, unmapped = retrieve_claim_evidence_passages(
        claims, sources,
        max_passages_per_source=max_passages_per_source,
        max_chars_per_source=max_chars_per_source,
        semantic_enabled=semantic_enabled,
        embedder=embedder,
    )
    links, _ = map_claims_to_cited_passages(claims, passages)
    claims_without_evidence = [
        claim.claim_id for claim in claims
        if not any(link.claim_id == claim.claim_id for link in links)
    ]
    return ClaimAnalysisResult(
        claims=claims,
        passages=passages,
        links=links,
        unmapped_source_ids=unmapped,
        claims_without_evidence=claims_without_evidence,
    )
