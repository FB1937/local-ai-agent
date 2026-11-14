"""Deterministic claim-level aggregation of pairwise evidence judgments."""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from claim_analysis import ClaimAnalysisResult, ClaimEvidenceLink, EvidencePassage

SynthesisStatus = Literal["supported", "contradicted", "conflict", "insufficient", "unassessed"]
ConflictType = Literal[
    "none", "direct", "scope_mismatch", "temporal_mismatch",
    "condition_mismatch", "duplicate_not_independent", "freshness_uncertain",
]


class ClaimEvidenceSynthesis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_id: str = Field(pattern=r"^C\d+$")
    status: SynthesisStatus
    conflict_type: ConflictType = "none"
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    insufficient_evidence_ids: list[str] = Field(default_factory=list)
    irrelevant_evidence_ids: list[str] = Field(default_factory=list)
    unassessed_evidence_ids: list[str] = Field(default_factory=list)
    duplicate_groups: list[list[str]] = Field(default_factory=list)
    independent_support_group_count: int = 0
    independent_contradiction_group_count: int = 0
    distinct_source_group_count: int = 0
    source_lineage_status: Literal["not_applicable", "single_source_group", "multiple_source_groups", "unknown"] = "unknown"
    joint_supporting_evidence_ids: list[str] = Field(default_factory=list)
    joint_contradicting_evidence_ids: list[str] = Field(default_factory=list)
    semantic_reason: str | None = None
    reason: str


class ClaimSynthesisReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claims: list[ClaimEvidenceSynthesis] = Field(default_factory=list)
    note: str = (
        "결정적 종합 결과다. pairwise insufficient 구절의 공동 함의, 의미적 범위 비교, "
        "출처 신뢰도 판단은 별도 의미 추론 단계가 필요하다."
    )


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _source_group(passage: EvidencePassage) -> str:
    return passage.canonical_source_id or passage.source_group or passage.source_id


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except (TypeError, ValueError):
        return None


def _time_ranges_disjoint(left: EvidencePassage, right: EvidencePassage) -> bool:
    left_start, left_end = _parse_date(left.valid_time_start), _parse_date(left.valid_time_end)
    right_start, right_end = _parse_date(right.valid_time_start), _parse_date(right.valid_time_end)
    if not all((left_start, left_end, right_start, right_end)):
        return False
    return left_end < right_start or right_end < left_start


def _scope_differs(left: EvidencePassage, right: EvidencePassage) -> bool:
    return bool(left.scope and right.scope and left.scope.strip().casefold() != right.scope.strip().casefold())


def _condition_differs(left: EvidencePassage, right: EvidencePassage) -> bool:
    return bool(left.condition and right.condition and left.condition.strip().casefold() != right.condition.strip().casefold())


def _duplicate_groups(passages: list[EvidencePassage]) -> list[list[str]]:
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for passage in passages:
        grouped[(_source_group(passage), _normalize_text(passage.text))].append(passage.evidence_id)
    return [ids for ids in grouped.values() if len(ids) > 1]


def _independent_group_count(ids: list[str], passages: dict[str, EvidencePassage]) -> int:
    return len({_source_group(passages[evidence_id]) for evidence_id in ids if evidence_id in passages})


def _aggregate_one_claim(
    claim_id: str,
    links: list[ClaimEvidenceLink],
    passages: dict[str, EvidencePassage],
) -> ClaimEvidenceSynthesis:
    if not links:
        return ClaimEvidenceSynthesis(
            claim_id=claim_id, status="insufficient",
            distinct_source_group_count=0, source_lineage_status="not_applicable",
            reason="연결된 근거 구절이 없습니다.",
        )

    missing = [link.evidence_id for link in links if link.evidence_id not in passages]
    unassessed = [link.evidence_id for link in links if link.relation is None]
    if missing or unassessed:
        unresolved = list(dict.fromkeys(missing + unassessed))
        return ClaimEvidenceSynthesis(
            claim_id=claim_id, status="unassessed",
            unassessed_evidence_ids=unresolved,
            reason="일부 근거 연결이 누락되었거나 관계 판정이 끝나지 않아 Claim 단위 결론을 보류합니다.",
        )

    claim_passages = [passages[link.evidence_id] for link in links]
    by_id = {p.evidence_id: p for p in claim_passages}
    duplicates = _duplicate_groups(claim_passages)
    duplicate_members = {item for group in duplicates for item in group}

    # Identical passages from the same underlying source must not receive
    # conflicting pairwise labels and thereby manufacture a claim-level conflict.
    relation_by_evidence = {link.evidence_id: link.relation for link in links}
    inconsistent_duplicate_ids: list[str] = []
    for group in duplicates:
        if len({relation_by_evidence.get(evidence_id) for evidence_id in group}) > 1:
            inconsistent_duplicate_ids.extend(group)
    if inconsistent_duplicate_ids:
        return ClaimEvidenceSynthesis(
            claim_id=claim_id, status="unassessed",
            unassessed_evidence_ids=sorted(set(inconsistent_duplicate_ids)),
            duplicate_groups=duplicates,
            reason="동일 원문 구절에 서로 다른 관계 라벨이 부여되어 판정을 보류합니다.",
        )

    supporting = [link.evidence_id for link in links if link.relation == "supports"]
    contradicting = [link.evidence_id for link in links if link.relation == "contradicts"]
    insufficient = [link.evidence_id for link in links if link.relation == "insufficient"]
    irrelevant = [link.evidence_id for link in links if link.relation == "irrelevant"]

    conflict_type: ConflictType = "none"
    status: SynthesisStatus
    reason: str

    if supporting and contradicting:
        comparisons = [
            (by_id[s_id], by_id[c_id])
            for s_id in supporting for c_id in contradicting
        ]
        scope_mismatches = [_scope_differs(s, c) for s, c in comparisons]
        condition_mismatches = [_condition_differs(s, c) for s, c in comparisons]
        time_mismatches = [_time_ranges_disjoint(s, c) for s, c in comparisons]
        if all(scope_mismatches):
            status, conflict_type = "insufficient", "scope_mismatch"
            reason = "지지와 반박 구절의 명시적 적용 범위가 달라 직접 충돌로 합치지 않았습니다."
        elif all(condition_mismatches):
            status, conflict_type = "insufficient", "condition_mismatch"
            reason = "지지와 반박 구절의 명시적 전제 조건이 달라 직접 충돌로 합치지 않았습니다."
        elif all(time_mismatches):
            status, conflict_type = "insufficient", "temporal_mismatch"
            reason = "지지와 반박 구절의 명시적 유효 기간이 겹치지 않아 시간적 충돌로 진단했습니다."
        else:
            status, conflict_type = "conflict", "direct"
            reason = "동일하거나 구분되지 않은 범위·시점에서 지지와 반박 근거가 함께 남아 충돌 상태입니다."
    elif supporting:
        status = "supported"
        reason = "하나 이상의 근거 구절이 주장을 직접 지지하며 직접 반박 근거는 없습니다."
    elif contradicting:
        status = "contradicted"
        reason = "하나 이상의 근거 구절이 주장을 직접 반박하며 직접 지지 근거는 없습니다."
    else:
        status = "insufficient"
        reason = "직접 지지 또는 반박 근거가 없어 정보 부족으로 남깁니다."

    return ClaimEvidenceSynthesis(
        claim_id=claim_id,
        status=status,
        conflict_type=conflict_type,
        supporting_evidence_ids=supporting,
        contradicting_evidence_ids=contradicting,
        insufficient_evidence_ids=insufficient,
        irrelevant_evidence_ids=irrelevant,
        duplicate_groups=duplicates,
        independent_support_group_count=_independent_group_count(supporting, by_id),
        independent_contradiction_group_count=_independent_group_count(contradicting, by_id),
        distinct_source_group_count=len({_source_group(p) for p in claim_passages}),
        source_lineage_status=(
            "multiple_source_groups" if len({_source_group(p) for p in claim_passages}) > 1
            else "single_source_group" if claim_passages else "not_applicable"
        ),
        reason=reason,
    )


def synthesize_claim_evidence(
    analysis: ClaimAnalysisResult | dict,
    reasoning_links: list[ClaimEvidenceLink] | list[dict] | None = None,
) -> ClaimSynthesisReport:
    """Aggregate pairwise labels without majority voting or loss of provenance.

    Missing judgments fail closed. Exact duplicate text from one canonical source
    is grouped for independence reporting, but a single valid supporting passage
    remains support; duplication only prevents inflated corroboration counts.
    """
    parsed = analysis if isinstance(analysis, ClaimAnalysisResult) else ClaimAnalysisResult.model_validate(analysis)
    links = reasoning_links if reasoning_links is not None else parsed.links
    normalized_links = [
        item if isinstance(item, ClaimEvidenceLink) else ClaimEvidenceLink.model_validate(item)
        for item in links
    ]
    passages = {item.evidence_id: item for item in parsed.passages}
    grouped: dict[str, list[ClaimEvidenceLink]] = defaultdict(list)
    for link in normalized_links:
        grouped[link.claim_id].append(link)
    return ClaimSynthesisReport(
        claims=[
            _aggregate_one_claim(claim.claim_id, grouped.get(claim.claim_id, []), passages)
            for claim in parsed.claims
        ]
    )
