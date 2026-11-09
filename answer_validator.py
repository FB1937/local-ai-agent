"""Structural citation checks; semantic entailment is not inferred."""
import re

CITATION_RE = re.compile(r"\[(S\d+)\]")


def validate_answer(answer: str, sources: list | None, generation_ok: bool = True,
                    search_assessment: dict | None = None, route: str = "SEARCH") -> dict:
    """Validate non-empty output and citation structure, not semantic support."""
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

    return {
        "valid": (
            not empty_answer
            and generation_ok
            and not unknown
            and not missing_url
        ),
        "empty_answer": empty_answer,
        "generation_ok": bool(generation_ok),
        "found_ids": found,
        "unknown_ids": unknown,
        "missing_url_ids": missing_url,
        "semantic_grounding_checked": False,
        "evidence_warnings": evidence_warnings,
        "note": "답변 존재 여부, 인용 ID와 URL의 구조 및 검색 평가의 근거 한계 경고를 검사했습니다. 개별 문장의 의미적 근거성은 검증하지 않습니다.",
    }


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
    if not notices:
        return answer
    return (answer or "").rstrip() + ("\n\n" if answer else "") + "\n".join(notices)
