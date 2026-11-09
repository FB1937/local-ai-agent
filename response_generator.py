"""Stage 6: generate and structurally validate the final response."""
from llm import generate_answer_result
from answer_validator import validate_answer, append_validation_notice


def generate_response(question: str, execution: dict, route: str,
                      context_history: str = "", generation_fn=None) -> dict:
    generation_fn = generation_fn or generate_answer_result
    results = execution.get("results", [])
    assessment = execution.get("assessment", {})
    generation = generation_fn(
        question,
        results,
        route=route,
        search_assessment=assessment,
        context_history=context_history,
    )
    answer = generation.get("answer") or ""
    validation = validate_answer(
        answer,
        results,
        generation_ok=(
            generation.get("ok", False)
            and not generation.get("incomplete", False)
        ),
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
    }
