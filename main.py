from llm import generate_answer
from router import classify_question
from search import search_web


def main():
    print("=" * 56)
    print("              LOCAL AI AGENT")
    print("        Qwen 3.5 9B + SearXNG | v1.1")
    print("=" * 56)

    while True:
        try:
            question = input("\n질문을 입력하세요 (종료: exit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAI Agent를 종료합니다.")
            break

        if question.lower() == "exit":
            print("AI Agent를 종료합니다.")
            break
        if not question:
            continue

        # 1. 질문 분석 및 처리 경로 결정
        print("\n[1] 질문 분석 중...")
        decision = classify_question(question)
        route = decision["route"]

        print(f"선택 경로: {route}")
        print(f"판단 이유: {decision['reason']}")
        print(f"분류 확신도(모델 자체 평가): {decision['confidence']:.2f}")
        if decision.get("fallback"):
            print("[안내] 라우터 오류로 안전 검색 경로를 사용합니다.")
        elif decision.get("overridden"):
            print("[안내] 안전 규칙이 모델의 초기 판단을 보정했습니다.")

        # 2. SEARCH / HYBRID 경로에서만 웹 검색
        results = []
        if route in {"SEARCH", "HYBRID"}:
            search_query = decision.get("search_query") or question
            print(f"\n[2] 웹 검색 중... ({search_query})")
            results = search_web(search_query)
            if results:
                print(f"검색 결과 {len(results)}개 수집 완료")
            else:
                print("검색 결과가 없거나 검색 서비스에 연결하지 못했습니다.")
        else:
            print("\n[2] 검색 생략: 로컬 모델로 바로 답변합니다.")

        # 3. 경로에 맞는 프롬프트로 답변 생성
        print("\n[3] Qwen 답변 생성 중...")
        answer = generate_answer(question, results, route=route)

        print("\n" + "=" * 56)
        print("AI 답변")
        print("=" * 56)
        print(answer)

        # 4. 검색을 수행한 경우 출처 목록 출력
        if results:
            print("\n[참고 자료]")
            for index, result in enumerate(results, 1):
                print(f"{index}. {result['title']}")
                print(f"   {result['url']}")


if __name__ == "__main__":
    main()
