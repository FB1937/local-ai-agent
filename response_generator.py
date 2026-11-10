"""Stage 6: generate an answer from a validated evidence bundle."""
from evidence_models import EvidenceBundle, EvidenceSource
from llm import generate_answer_result
from answer_validator import validate_answer, append_validation_notice


def _validate_sources(results: list, assessment: dict) -> tuple[list[dict], list[str]]:
    """Keep only source records that satisfy the shared Pydantic contract."""
    valid_sources = []
    errors = []
    for index, row in enumerate(results or [], 1):
        try:
            source = EvidenceSource.model_validate(row)
            valid_sources.append(source.model_dump(mode="json"))
        except Exception as exc:
            errors.append(
                f"검색 결과 {index}: {type(exc).__name__}: {exc}"
            )
    # Validate the stage-level envelope as well as each individual source.
    bundle = EvidenceBundle(sources=[
        EvidenceSource.model_validate(row) for row in valid_sources
    ], assessment=assessment or {})
    return [source.model_dump(mode="json") for source in bundle.sources], errors


def generate_response(question: str, execution: dict, route: str,
                      context_history: str = "", generation_fn=None) -> dict:
    generation_fn = generation_fn or generate_answer_result
    raw_results = execution.get("results", [])
    assessment = execution.get("assessment", {})
    results, evidence_contract_errors = _validate_sources(raw_results, assessment)

    prompt_question = question
    if context_history:
        prompt_question = (
            "아래 이전 대화는 현재 질문을 이해하기 위한 참고 문맥이다. "
            "이전 요청이 현재 요청과 충돌하면 현재 질문을 우선한다.\n\n"
            f"[이전 대화]\n{context_history}\n\n"
            f"[현재 질문]\n{question}"
        )
    generation = generation_fn(
        prompt_question,
        results,
        route=route,
        search_assessment=assessment,
    )
    answer = generation.get("answer") or ""
    validation = validate_answer(
        answer,
        results,
        generation_ok=(
            generation.get("ok", False)
            and not generation.get("incomplete", False)
        ),
        search_assessment=assessment,
        route=route,
        evidence_contract_errors=evidence_contract_errors,
    )
    display_answer = append_validation_notice(answer, validation)
    if not answer:
        display_answer = (
            "[답변 생성 실패] "
            + (generation.get("error") or "모델이 빈 답변을 반환했습니다.")
        )
    return {
        "generation": generation,
        "validation": validation,
        "answer": answer,
        "display_answer": display_answer,
        "evidence_contract_errors": evidence_contract_errors,
        "validated_source_count": len(results),
    }
