"""Structural citation checks; semantic entailment is not inferred."""
import re

CITATION_RE = re.compile(r"\[(S\d+)\]")


def validate_answer(answer: str, sources: list | None, generation_ok: bool = True) -> dict:
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
        "note": "답변 존재 여부, 인용 ID와 URL의 구조만 검사했습니다. 의미적 근거성은 별도 검토가 필요합니다.",
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
    if not notices:
        return answer
    return (answer or "").rstrip() + ("\n\n" if answer else "") + "\n".join(notices)
