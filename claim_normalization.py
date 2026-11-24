"""M4.2 deterministic claim normalization and pairwise budget planning.

This module only removes unmistakable answer scaffolding and prioritizes review.
It does not split, rewrite, or certify factual claims.
"""
from __future__ import annotations

import re

from claim_analysis import Claim, ClaimAnalysisResult, ClaimEvidenceLink

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*+]\s+|\d{1,2}[.)]\s+)")
_META_PATTERNS = (
    re.compile(r"^(?:이하|아래)(?:에서는|에서|와 같이|의 내용은)"),
    re.compile(r"^(?:이번|본|이)\s*(?:답변|분석|정리)(?:에서는|은|는|을|를)"),
    re.compile(r"^(?:검색|제공된)\s*(?:결과|자료|정보)(?:를|에|만을)\s*(?:바탕으로|기준으로|통해)"),
    re.compile(r"^(?:먼저|다음으로|마지막으로)\s*(?:살펴보겠습니다|확인하겠습니다|정리하겠습니다)"),
)
_SECTION_LABEL_RE = re.compile(
    r"(?:주요\s*변경\s*사항|주요\s*내용|개요|결론|요약|정리|참고|"
    r"권장\s*방법|사용법|장점|단점|원인|이유|"
    r"overview|summary|conclusion|key points|major changes|references|results)",
    re.IGNORECASE,
)
_IMPORTANT_RE = re.compile(
    r"(?:\d|%|최신|현재|기준|출시|발표|버전|날짜|기간|가격|요금|한도|"
    r"최소|최대|필수|요구|지원|보안|취약|정책|법률|규정|권장|공식)",
    re.IGNORECASE,
)


def _clean_prefix(text: str) -> tuple[str, bool]:
    value = (text or "").strip()
    heading = bool(_HEADING_RE.match(value))
    value = _HEADING_RE.sub("", value).strip()
    value = _LIST_PREFIX_RE.sub("", value).strip()
    return value, heading


def _is_structural_or_meta(text: str) -> tuple[bool, str | None]:
    value, markdown_heading = _clean_prefix(text)
    if not value:
        return True, "empty"
    if markdown_heading:
        return True, "markdown_heading"
    # Short, non-sentential labels are headings; retain complete declarative
    # sentences even when they are short (e.g. "Python 3.15 is stable.").
    if len(value) <= 100 and not re.search(r"[.!?。！？][\s\]]*$", value):
        if value.endswith(":") or _SECTION_LABEL_RE.search(value):
            return True, "section_label"
    if any(pattern.search(value) for pattern in _META_PATTERNS):
        return True, "answer_scaffolding"
    return False, None


def _importance(text: str) -> str:
    return "high" if _IMPORTANT_RE.search(text or "") else "normal"


def _priority_round_robin(
    claims: list[Claim], links: list[ClaimEvidenceLink], budget: int,
) -> tuple[list[ClaimEvidenceLink], list[ClaimEvidenceLink]]:
    by_claim: dict[str, list[ClaimEvidenceLink]] = {}
    for link in links:
        by_claim.setdefault(link.claim_id, []).append(link)
    claim_order = sorted(
        by_claim,
        key=lambda claim_id: (
            0 if next(c for c in claims if c.claim_id == claim_id).importance == "high" else 1,
            next(i for i, c in enumerate(claims) if c.claim_id == claim_id),
        ),
    )
    selected: list[ClaimEvidenceLink] = []
    depth = 0
    while len(selected) < budget:
        added = False
        for claim_id in claim_order:
            items = by_claim[claim_id]
            if depth < len(items) and len(selected) < budget:
                selected.append(items[depth])
                added = True
        if not added:
            break
        depth += 1
    selected_pairs = {(item.claim_id, item.evidence_id) for item in selected}
    skipped = [
        item for item in links
        if (item.claim_id, item.evidence_id) not in selected_pairs
    ]
    return selected, skipped


def normalize_claim_analysis(
    analysis: ClaimAnalysisResult, *, max_links: int = 20,
) -> tuple[ClaimAnalysisResult, list[ClaimEvidenceLink], list[ClaimEvidenceLink], list[dict]]:
    """Filter obvious scaffolding, prioritize material claims, and select calls.

    Returns the normalized graph, links selected for model assessment, and an
    audit list of excluded structural/meta segments. Unselected links remain in
    the normalized graph and must be represented as unassessed during synthesis.
    """
    if max_links < 1:
        raise ValueError("max_links must be positive")

    kept: list[Claim] = []
    excluded: list[dict] = []
    kept_ids: set[str] = set()
    for claim in analysis.claims:
        exclude, reason = _is_structural_or_meta(claim.text)
        if exclude:
            excluded.append({
                "claim_id": claim.claim_id,
                "text": claim.text,
                "reason": reason,
            })
            continue
        normalized = claim.model_copy(update={"importance": _importance(claim.text)})
        kept.append(normalized)
        kept_ids.add(claim.claim_id)

    kept_links = [link.model_copy(deep=True) for link in analysis.links if link.claim_id in kept_ids]
    selected, skipped = _priority_round_robin(kept, kept_links, max_links)
    kept_passage_ids = {link.evidence_id for link in kept_links}
    normalized_analysis = ClaimAnalysisResult(
        claims=kept,
        passages=[item.model_copy(deep=True) for item in analysis.passages
                  if item.evidence_id in kept_passage_ids],
        links=kept_links,
        unmapped_source_ids=list(analysis.unmapped_source_ids),
        claims_without_evidence=[
            claim_id for claim_id in analysis.claims_without_evidence
            if claim_id in kept_ids
        ],
    )
    # The skipped links are returned separately for explicit unassessed accounting.
    return normalized_analysis, selected, skipped, excluded
