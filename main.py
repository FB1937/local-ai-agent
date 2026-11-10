"""CLI entry point for the six-stage Local AI Agent."""
from agent import LocalAIAgent


def main():
    agent = LocalAIAgent()
    print("=" * 64)
    print("                 LOCAL AI AGENT")
    print("       Six-stage pipeline | Qwen 3.5 9B + SearXNG")
    print("=" * 64)
    while True:
        try:
            raw_question = input("\n질문을 입력하세요 (종료: exit): ")
        except (EOFError, KeyboardInterrupt):
            print("\nAI Agent를 종료합니다.")
            break
        if raw_question.strip().lower() == "exit":
            print("AI Agent를 종료합니다.")
            break
        if not raw_question.strip():
            continue
        try:
            result = agent.run(raw_question)
        except (TypeError, ValueError) as exc:
            print(f"[입력 처리 오류] {exc}")
            continue
        task, decision = result["task"], result["route_decision"]
        execution, response = result["execution"], result["response"]
        generation, validation = response["generation"], response["validation"]
        print("\n[① Input Processing]\n입력 정규화 완료")
        print("\n[② Context Understanding]")
        print("이전 대화 문맥: " + ("사용" if result["stages"]["context_understanding"]["prior_turns"] else "없음"))
        print("\n[③ Intent & Task Classification]")
        print(f"Intent: {task['intent']} | Domain: {task['domain']}")
        print(f"Tasks: {', '.join(task['task_types'])}")
        print(f"최신 정보 필요: {'예' if task['freshness_required'] else '아니오'}")
        print(f"예상 도구: {', '.join(task['tool_candidates']) or '없음'}")
        print("\n[④ Planning & Routing]")
        print(f"선택 경로: {decision['route']} | 판단 이유: {decision.get('reason', '')}")
        if decision.get("fast_path"):
            print("분류 방식: 규칙 기반 빠른 경로 (확률 점수가 아닙니다)")
        else:
            print(f"분류 확신도(모델 자체 평가): {decision.get('confidence', 0):.2f}")
        if decision.get("fallback"):
            print("[안내] 라우터 오류로 안전 검색 경로를 사용합니다.")
        elif decision.get("overridden"):
            print("[안내] 안전 규칙이 모델의 초기 판단을 보정했습니다.")
        print(f"실행 단계: {len(result['plan']['steps'])}개")
        print("\n[⑤ Execution]")
        print(f"실행 상태: {execution['status']}")
        for index, attempt in enumerate(execution.get("attempts", []), 1):
            print(f"검색 시도 {index}: {attempt.get('count', 0)}개 / {attempt.get('assessment', {}).get('reason', '')}")
        assessment, extraction = execution.get("assessment", {}), execution.get("page_extraction", {})
        print(f"[원문 추출] 시도 {extraction.get('attempted_count', 0)}개 / 성공 {extraction.get('extracted_count', 0)}개")
        for error in extraction.get("errors", []):
            print(f"  추출 오류: {error}")
        print("[근거 요약] " + f"관련 {assessment.get('usable_count', 0)}개 / 공식 {assessment.get('official_count', 0)}개 / 기업 공식 {assessment.get('company_official_count', 0)}개 / 고객 사례 후보 {assessment.get('company_case_signal_count', 0)}개")
        if not assessment.get("sufficient", False) and decision["route"] != "DIRECT":
            print("[주의] 검색 근거가 충분하지 않을 수 있습니다: " + assessment.get("reason", "사유 미상"))
        print("\n[⑥ Response Generation]\n" + "=" * 64 + "\nAI 답변\n" + "=" * 64)
        print(response["display_answer"])
        for item in execution.get("results", []):
            print(f"\n[{item.get('source_id', '')}] [{item.get('source_type', 'other')}] {item.get('title', '')}")
            print(f"   {item.get('url', '')}")
        print("\n[실행 시간]")
        for stage, seconds in result["timings"].items():
            print(f"- {stage}: {seconds:.2f}초")
        print(f"- 생성 재시도: {generation.get('retry_count', 0)}회")
        for attempt in generation.get("attempts", []):
            print(f"  시도 {attempt.get('attempt')}: content {attempt.get('content_characters', 0)}자 / thinking {attempt.get('thinking_characters', 0)}자 / 출력 토큰 {attempt.get('eval_count')} / num_predict {attempt.get('num_predict')} / 종료 {attempt.get('done_reason')}")
        if generation.get("eval_count") and generation.get("eval_duration_seconds"):
            print(f"- 생성 속도: {generation['eval_count'] / generation['eval_duration_seconds']:.2f} tokens/s")
        print(f"- 인용/규칙 기반 핵심 주장 검사: {'통과' if validation['valid'] else '경고'}")
        for check in validation.get("claim_checks", []):
            print(f"  - {check.get('claim_type')}: {check.get('status')} / {check.get('reason')}")
        semantic = response.get("semantic_verification", {})
        print(f"- 의미 검토 프로토타입: {'활성' if semantic.get('enabled') else '비활성'}")
        for check in semantic.get("checks", []):
            print(f"  - {check.get('model_assessment')} → {check.get('status')}: {check.get('claim')}")
        print("- 의미 검토 결과는 자동 사실 인증이 아닙니다.")


if __name__ == "__main__":
    main()
