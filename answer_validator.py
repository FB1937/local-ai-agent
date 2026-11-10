"""Structural citation and lightweight, deterministic claim-evidence checks.

This module checks selected high-risk factual claims (versions, percentages,
and customer-deployment assertions). It does not perform general semantic
entailment verification.
"""
from __future__ import annotations

import re

from evidence_models import ClaimEvidenceCheck, ValidationReport

CITATION_RE = re.compile(r"\[(S\d+)\]")
VERSION_RE = re.compile(r"(?<![A-Za-z0-9])(\d+\.\d+(?:\.\d+){0,2})(?![A-Za-z0-9])")
LATEST_RE = re.compile(r"(최신|최신\s*안정|현재\s*안정|latest\s*(?:stable)?|current\s*stable)", re.I)
PERCENT_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(%|퍼센트|percent)", re.I)
GARTNER_RE = re.compile(r"\bGartner\b", re.I)
DEPLOYMENT_RE = re.compile(
    r"(실제\s*)?(?:고객(?:사)?\s*)?(?:도입(?:했|한|하고|하여|해|중|사례)|"
    r"구축(?:했|한|하고|하여|중)|운영(?:하고|중|한다)|배포(?:했|한|되어)|"
    r"deployed\s+at|implemented\s+at|customer\s+(?:success\s+)?story)",
    re.I,
)
INDUSTRY_ADOPTION_RE = re.compile(
    r"(업계\s*표준|기업\s*전반|대부분의\s*기업|기업들이\s*(?:널리\s*)?(?:도입|사용)|"
    r"industry\s+standard|widely\s+adopted)",
    re.I,
)
NEGATED_RE = re.compile(
    r"(확인되지\s*않|확인할\s*수\s*없|찾지\s*못|없습니다|아닙니다|미확인|"
    r"not\s+confirmed|not\s+found|no\s+evidence)",
    re.I,
)


def _sentences(text: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?。])\s+|\n+", text or "")
        if part.strip()
    ]


def _source_text(source: dict) -> str:
    return " ".join(
        str(source.get(key) or "")
        for key in ("title", "extracted_title", "content", "extracted_text")
    )


def _percent_key(value: str) -> str:
    try:
        return str(float(value.replace(",", "."))).rstrip("0").rstrip(".")
    except ValueError:
        return value


def _check_claims(answer: str, sources: list[dict], assessment: dict,
                  route: str) -> list[ClaimEvidenceCheck]:
    if (route or "").upper() == "DIRECT":
        return []

    source_text = " ".join(_source_text(source) for source in sources)
    source_by_id = {
        str(source.get("source_id")): source
        for source in sources if source.get("source_id")
    }
    checks: list[ClaimEvidenceCheck] = []

    # Version statements are checked against the deterministic result from
    # official release-index candidates, not against an LLM's free-form choice.
    if assessment.get("version_query"):
        expected = assessment.get("latest_stable_claims") or assessment.get("official_index_versions") or []
        for sentence in _sentences(answer):
            if not LATEST_RE.search(sentence):
                continue
            mentioned = VERSION_RE.findall(sentence)
            citation_ids = CITATION_RE.findall(sentence)
            if len(expected) != 1:
                checks.append(ClaimEvidenceCheck(
                    claim_type="latest_version",
                    claim=sentence,
                    status="needs_review",
                    source_ids=citation_ids,
                    reason="공식 최신 릴리스 인덱스에서 단일 최신 안정 버전을 확정하지 못했습니다.",
                ))
            elif expected[0] not in mentioned:
                checks.append(ClaimEvidenceCheck(
                    claim_type="latest_version",
                    claim=sentence,
                    status="unsupported",
                    source_ids=citation_ids,
                    reason=f"공식 인덱스의 최신 안정 버전({expected[0]})과 답변의 버전 주장이 일치하지 않습니다.",
                ))
            else:
                matching_sources = [
                    source for source in sources
                    if expected[0] in (source.get("latest_stable_claims") or [])
                    and source.get("official_release_index")
                ]
                matching_cited_sources = [
                    source for source in matching_sources
                    if str(source.get("source_id")) in citation_ids
                ]
                if not matching_cited_sources:
                    checks.append(ClaimEvidenceCheck(
                        claim_type="latest_version",
                        claim=sentence,
                        status="unsupported",
                        source_ids=citation_ids,
                        reason="버전 값은 일치하지만 해당 공식 릴리스 인덱스가 이 문장에 인용되지 않았습니다.",
                    ))
                else:
                    checks.append(ClaimEvidenceCheck(
                        claim_type="latest_version",
                        claim=sentence,
                        status="supported",
                        source_ids=[source["source_id"] for source in matching_cited_sources],
                        reason="인용된 공식 릴리스 인덱스에서 추출한 버전과 답변의 버전이 일치합니다. 문장 전체의 의미적 검증은 아닙니다.",
                    ))

    # Percentages are exact-match checked against retrieved source text.
    source_percentages = {
        _percent_key(match.group(1))
        for match in PERCENT_RE.finditer(source_text)
    }
    for sentence in _sentences(answer):
        percentages = PERCENT_RE.findall(sentence)
        for value, _unit in percentages:
            citation_ids = CITATION_RE.findall(sentence)
            if _percent_key(value) not in source_percentages:
                checks.append(ClaimEvidenceCheck(
                    claim_type="statistic",
                    claim=sentence,
                    status="unsupported",
                    source_ids=citation_ids,
                    reason=f"답변의 수치 {value}%를 검색 원문에서 확인하지 못했습니다.",
                ))
            elif not citation_ids:
                checks.append(ClaimEvidenceCheck(
                    claim_type="statistic",
                    claim=sentence,
                    status="unsupported",
                    source_ids=[],
                    reason="검색 자료에서 수치를 찾았지만 해당 주장에 출처 인용이 연결되지 않았습니다.",
                ))
            else:
                matching_citations = [
                    citation_id for citation_id in citation_ids
                    if citation_id in source_by_id
                    and _percent_key(value) in {
                        _percent_key(match.group(1))
                        for match in PERCENT_RE.finditer(_source_text(source_by_id[citation_id]))
                    }
                ]
                if not matching_citations:
                    checks.append(ClaimEvidenceCheck(
                        claim_type="statistic",
                        claim=sentence,
                        status="unsupported",
                        source_ids=citation_ids,
                        reason="인용한 출처 중 해당 수치를 직접 포함하는 자료가 없습니다.",
                    ))
                else:
                    checks.append(ClaimEvidenceCheck(
                        claim_type="statistic",
                        claim=sentence,
                        status="supported",
                        source_ids=matching_citations,
                        reason="인용한 출처 본문 중 하나 이상에 동일한 수치가 있습니다. 수치가 설명하는 대상·맥락의 의미적 검증은 아닙니다.",
                    ))

        if GARTNER_RE.search(sentence):
            citation_ids = CITATION_RE.findall(sentence)
            cited_text = " ".join(
                _source_text(source_by_id[citation_id])
                for citation_id in citation_ids if citation_id in source_by_id
            )
            if not GARTNER_RE.search(cited_text):
                checks.append(ClaimEvidenceCheck(
                    claim_type="named_source",
                    claim=sentence,
                    status="unsupported",
                    source_ids=citation_ids,
                    reason="답변은 Gartner를 언급하지만 해당 문장에 인용된 출처에서 Gartner를 확인하지 못했습니다.",
                ))

    # Deployment language is treated conservatively. A search snippet signal is
    # only a candidate; extracted first-party text still requires human review.
    if assessment.get("requires_company_official"):
        case_sources = [
            source for source in sources
            if source.get("source_type") == "company_official"
            and source.get("company_case_signal")
        ]
        for sentence in _sentences(answer):
            if NEGATED_RE.search(sentence):
                continue
            if DEPLOYMENT_RE.search(sentence):
                citation_ids = CITATION_RE.findall(sentence)
                if not case_sources:
                    checks.append(ClaimEvidenceCheck(
                        claim_type="customer_deployment",
                        claim=sentence,
                        status="unsupported",
                        source_ids=citation_ids,
                        reason="실제 고객 도입을 뒷받침하는 기업 공식 사례 후보를 확보하지 못했습니다.",
                    ))
                elif not citation_ids or not any(
                    citation_id in {str(source.get("source_id")) for source in case_sources}
                    for citation_id in citation_ids
                ):
                    checks.append(ClaimEvidenceCheck(
                        claim_type="customer_deployment",
                        claim=sentence,
                        status="unsupported",
                        source_ids=citation_ids,
                        reason="고객 사례 후보가 있지만 해당 주장에 후보 출처의 인용이 연결되지 않았습니다.",
                    ))
                else:
                    checks.append(ClaimEvidenceCheck(
                        claim_type="customer_deployment",
                        claim=sentence,
                        status="needs_review",
                        source_ids=citation_ids,
                        reason="고객명·도입 신호가 있는 공식 출처 후보를 찾았습니다. 실제 적용 제품·업무와 도입 사실은 원문 수동 확인이 필요합니다.",
                    ))

    if assessment.get("sufficient") is False:
        for sentence in _sentences(answer):
            if INDUSTRY_ADOPTION_RE.search(sentence) and not NEGATED_RE.search(sentence):
                checks.append(ClaimEvidenceCheck(
                    claim_type="industry_adoption",
                    claim=sentence,
                    status="unsupported",
                    source_ids=CITATION_RE.findall(sentence),
                    reason="검색 근거가 불충분한 상태에서 업계 전반의 도입 수준을 단정했습니다.",
                ))

    return checks


def validate_answer(
    answer: str,
    sources: list | None,
    generation_ok: bool = True,
    search_assessment: dict | None = None,
    route: str = "SEARCH",
    evidence_contract_errors: list[str] | None = None,
) -> dict:
    """Validate citation structure and selected high-risk claims, not all semantics."""
    sources = sources or []
    answer_text = answer if isinstance(answer, str) else ""
    empty_answer = not answer_text.strip()
    valid_ids = {
        str(source.get("source_id"))
        for source in sources
        if source.get("source_id")
    }
    found = CITATION_RE.findall(answer_text)
    unknown = sorted(set(found) - valid_ids)
    missing_url = sorted(
        source_id for source_id in set(found)
        if source_id in valid_ids and not next(
            (
                source.get("url")
                for source in sources
                if source.get("source_id") == source_id and source.get("url")
            ),
            None,
        )
    )
    assessment = search_assessment or {}
    evidence_warnings = []
    normalized_route = (route or "SEARCH").upper()
    if normalized_route in {"SEARCH", "HYBRID"} and assessment.get("sufficient") is False:
        evidence_warnings.append(
            "검색 근거가 충분하지 않습니다. 검색 결과를 최신 동향이나 업계 전반의 사실로 일반화하지 마세요."
        )
    if assessment.get("requires_company_official") and not assessment.get("company_official_count", 0):
        evidence_warnings.append(
            "기업 공식 출처가 확보되지 않았으므로 특정 기업의 공식 입장이나 도입 사실을 확인할 수 없습니다."
        )
    if assessment.get("requires_company_official") and not assessment.get("company_case_signal_count", 0):
        evidence_warnings.append(
            "실제 고객 도입 사례는 확인되지 않았습니다. 일반 활용 시나리오와 실제 도입 사실을 구분해야 합니다."
        )

    contract_errors = list(evidence_contract_errors or [])
    if contract_errors:
        evidence_warnings.append("근거 데이터 계약을 통과하지 못한 출처가 있어 해당 출처는 답변 근거에서 제외했습니다.")

    claim_checks = _check_claims(answer_text, sources, assessment, normalized_route)
    unsupported_checks = [check for check in claim_checks if check.status == "unsupported"]
    report = ValidationReport(
        valid=(
            not empty_answer
            and generation_ok
            and not unknown
            and not missing_url
            and not contract_errors
            and not unsupported_checks
        ),
        generation_ok=bool(generation_ok),
        found_ids=found,
        unknown_ids=unknown,
        missing_url_ids=missing_url,
        evidence_warnings=evidence_warnings,
        claim_checks=claim_checks,
        evidence_contract_errors=contract_errors,
        semantic_grounding_checked=False,
        note=(
            "인용 ID·URL 구조와 버전·백분율·기업 도입 주장 일부를 결정적 규칙으로 검사했습니다. "
            "모든 문장의 의미적 근거성(entailment)은 검증하지 않습니다."
        ),
    )
    return report.model_dump()


def append_validation_notice(answer: str, validation: dict) -> str:
    notices = []
    if validation.get("empty_answer"):
        notices.append("[검증 경고] 모델이 빈 답변을 반환했습니다.")
    if not validation.get("generation_ok", True):
        notices.append("[검증 경고] 답변 생성 단계가 정상 완료되지 않았습니다.")
    if validation.get("unknown_ids"):
        notices.append(
            "[검증 경고] 존재하지 않는 출처 ID: "
            + ", ".join(validation["unknown_ids"])
        )
    if validation.get("missing_url_ids"):
        notices.append(
            "[검증 경고] URL이 없는 출처 ID: "
            + ", ".join(validation["missing_url_ids"])
        )
    if validation.get("evidence_warnings"):
        notices.extend(
            "[근거 범위 경고] " + warning
            for warning in validation["evidence_warnings"]
        )
    for check in validation.get("claim_checks", []):
        if check.get("status") == "unsupported":
            notices.append(
                "[주장 검증 경고] "
                + check.get("claim_type", "claim")
                + ": "
                + check.get("reason", "근거를 확인하지 못했습니다.")
            )
        elif check.get("status") == "needs_review":
            notices.append(
                "[추가 확인 필요] "
                + check.get("claim_type", "claim")
                + ": "
                + check.get("reason", "")
            )
    if not notices:
        return answer
    return (answer or "").rstrip() + ("\n\n" if answer else "") + "\n".join(notices)
