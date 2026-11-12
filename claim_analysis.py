"""Structured claim/evidence contracts and deterministic citation mapping.

This module does not decide whether evidence is true or entails a claim. It
creates traceable candidates for the later evidence-reasoning stage.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from evidence_models import EvidenceSource


Importance = Literal["high", "normal", "low"]
EvidenceRelation = Literal["supports", "contradicts", "insufficient", "irrelevant"]

_CITATION_RE = re.compile(r"\[(S\d+)\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")


class InformationNeed(BaseModel):
    """A normalized statement of what the user needs to learn or verify."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4000)
    required_aspects: list[str] = Field(default_factory=list)
    freshness_required: bool = False
    scope: str | None = None


class Claim(BaseModel):
    """One answer statement that can be checked against evidence."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(pattern=r"^C\d+$")
    text: str = Field(min_length=1, max_length=4000)
    importance: Importance = "normal"
    qualifiers: list[str] = Field(default_factory=list)
    cited_source_ids: list[str] = Field(default_factory=list)


class EvidencePassage(BaseModel):
    """A bounded text passage retaining its source and location provenance."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(pattern=r"^E\d+$")
    source_id: str = Field(pattern=r"^S\d+$")
    text: str = Field(min_length=1, max_length=2000)
    location: str
    published_at: str | None = None
    retrieved_at: datetime | None = None


class ClaimEvidenceLink(BaseModel):
    """A traceable claim-to-passage relationship.

    relation=None means the candidate is mapped but has not been assessed.
    Mapping a citation is not proof that the cited passage supports the claim.
    """

    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(pattern=r"^C\d+$")
    evidence_id: str = Field(pattern=r"^E\d+$")
    relation: EvidenceRelation | None = None
    reason: str = "판정 대기: 인용 출처와 구절을 연결했지만 의미적 지지는 아직 평가하지 않았습니다."


class ClaimAnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[Claim] = Field(default_factory=list)
    passages: list[EvidencePassage] = Field(default_factory=list)
    links: list[ClaimEvidenceLink] = Field(default_factory=list)
    unmapped_source_ids: list[str] = Field(default_factory=list)


def _split_answer_sentences(answer: str) -> list[str]:
    """Split sentences while keeping citations attached to the preceding claim."""
    text = answer or ""
    sentences: list[str] = []
    start = 0
    index = 0

    while index < len(text):
        char = text[index]
        if char in ".!?。！？":
            # A dot between digits is a decimal point, not a sentence boundary.
            if char == "." and index > 0 and index + 1 < len(text):
                if text[index - 1].isdigit() and text[index + 1].isdigit():
                    index += 1
                    continue

            cursor = index + 1
            had_boundary_space = False
            while cursor < len(text) and text[cursor].isspace() and text[cursor] != "\n":
                had_boundary_space = True
                cursor += 1

            # Citations commonly follow punctuation: "Claim. [S1]".
            while cursor < len(text):
                match = _CITATION_RE.match(text, cursor)
                if not match:
                    break
                cursor = match.end()
                while cursor < len(text) and text[cursor].isspace() and text[cursor] != "\n":
                    had_boundary_space = True
                    cursor += 1

            if cursor == len(text) or had_boundary_space:
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
    """Split an answer into sentence-level claims and retain citation IDs.

    This is intentionally a deterministic first pass, not a complete semantic
    claim decomposition. Uncited sentences are retained with no source IDs.
    """
    claims: list[Claim] = []
    for sentence in _split_answer_sentences(answer):
        source_ids = list(dict.fromkeys(_CITATION_RE.findall(sentence)))
        claims.append(Claim(
            claim_id=f"C{len(claims) + 1}",
            text=sentence,
            cited_source_ids=source_ids,
        ))
    return claims


def build_evidence_passages(
    sources: list[EvidenceSource | dict],
    *,
    max_passages_per_source: int = 40,
    max_chars_per_passage: int = 1200,
) -> list[EvidencePassage]:
    """Create bounded sentence passages while preserving source provenance."""
    if max_passages_per_source < 1 or max_chars_per_passage < 1:
        raise ValueError("passage limits must be positive")

    passages: list[EvidencePassage] = []
    for raw_source in sources or []:
        source = (
            raw_source if isinstance(raw_source, EvidenceSource)
            else EvidenceSource.model_validate(raw_source)
        )
        body = source.extracted_text or source.content or ""
        segments = [
            segment.strip()
            for segment in _SENTENCE_SPLIT_RE.split(body)
            if segment.strip()
        ]
        for index, segment in enumerate(segments[:max_passages_per_source], 1):
            passages.append(EvidencePassage(
                evidence_id=f"E{len(passages) + 1}",
                source_id=source.source_id,
                text=segment[:max_chars_per_passage],
                location=f"sentence:{index}",
                published_at=source.published,
                retrieved_at=source.retrieved_at,
            ))
    return passages


def map_claims_to_cited_passages(
    claims: list[Claim],
    passages: list[EvidencePassage],
) -> tuple[list[ClaimEvidenceLink], list[str]]:
    """Map each claim to passages from its cited sources, without judging entailment."""
    passages_by_source: dict[str, list[EvidencePassage]] = {}
    for passage in passages:
        passages_by_source.setdefault(passage.source_id, []).append(passage)

    links: list[ClaimEvidenceLink] = []
    unmapped: list[str] = []
    for claim in claims:
        for source_id in claim.cited_source_ids:
            candidates = passages_by_source.get(source_id, [])
            if not candidates:
                if source_id not in unmapped:
                    unmapped.append(source_id)
                continue
            for passage in candidates:
                links.append(ClaimEvidenceLink(
                    claim_id=claim.claim_id,
                    evidence_id=passage.evidence_id,
                ))
    return links, unmapped


def analyze_claim_evidence(
    answer: str,
    sources: list[EvidenceSource | dict],
) -> ClaimAnalysisResult:
    """Build an auditable claim/evidence candidate graph from an answer."""
    claims = extract_cited_claims(answer)
    passages = build_evidence_passages(sources)
    links, unmapped = map_claims_to_cited_passages(claims, passages)
    return ClaimAnalysisResult(
        claims=claims,
        passages=passages,
        links=links,
        unmapped_source_ids=unmapped,
    )
