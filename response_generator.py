"""Stage 6: generate and structurally validate the final response."""
from llm import generate_answer_result
from answer_validator import validate_answer, append_validation_notice


def generate_response(question: str, execution: dict, route: str,
                      context_history: str = "", generation_fn=None) -> dict:
    generation_fn = generation_fn or generate_answer_result
    results = execution.get("results", [])
    assessment = execution.get("assessment", {})
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
