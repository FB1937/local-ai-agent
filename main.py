"""Local AI Agent v1.2 entry point."""
from time import perf_counter
from router import classify_question
from search import search_web
from search_validator import search_with_validation
from llm import generate_answer
from answer_validator import validate_answer, append_validation_notice

def main():
    print("=" * 56)
    print("              LOCAL AI AGENT")
    print("        Qwen 3.5 9B + SearXNG | v1.2")
    print("=" * 56)
    while True:
        try:
            question=input("\\n질문을 입력하세요 (종료: exit): ").strip()
        except (EOFError,KeyboardInterrupt):
            print("\\nAI Agent를 종료합니다."); break
        if question.lower()=="exit":
            print("AI Agent를 종료합니다."); break
        if not question: continue
        timings={}
        print("\\n[1] 질문 분석 중...")
        t=perf_counter(); decision=classify_question(question); timings["router"]=perf_counter()-t
        route=decision["route"]
        print(f"선택 경로: {route}")
        print(f"판단 이유: {decision['reason']}")
        print(f"분류 확신도(모델 자체 평가): {decision['confidence']:.2f}")
        if decision.get("fallback"): print("[안내] 라우터 오류로 안전 검색 경로를 사용합니다.")
        elif decision.get("overridden"): print("[안내] 안전 규칙이 모델의 초기 판단을 보정했습니다.")
        results=[]; assessment={"sufficient":False,"reason":"DIRECT 경로에서는 검색하지 않았습니다."}
        if route in {"SEARCH","HYBRID"}:
            query=decision.get("search_query") or question
            print(f"\\n[2] 검색 계획 및 검색 검증 중... ({query})")
            t=perf_counter()
            outcome=search_with_validation(question,query,search_web)
            timings["search_total"]=perf_counter()-t
            results=outcome["results"]; assessment=outcome["assessment"]
            for i,attempt in enumerate(outcome["attempts"],1):
                print(f"  검색 시도 {i}: {attempt['count']}개 / {attempt['assessment']['reason']}")
            print(f"최종 검색 결과: {len(results)}개")
            if not assessment["sufficient"]:
                print("[주의] 검색 근거가 충분하지 않을 수 있습니다. 답변에서 한계를 표시합니다.")
        else:
            print("\\n[2] 검색 생략: 로컬 모델로 바로 답변합니다.")
        print("\\n[3] Qwen 답변 생성 중...")
        t=perf_counter()
        answer=generate_answer(question,results,route=route,search_assessment=assessment)
        timings["answer_generation"]=perf_counter()-t
        validation=validate_answer(answer,results)
        answer=append_validation_notice(answer,validation)
        print("\\n"+"="*56+"\\nAI 답변\\n"+"="*56)
        print(answer)
        if results:
            print("\\n[참고 자료]")
            for result in results:
                print(f"{result['source_id']}. [{result.get('source_type','other')}] {result['title']}")
                print(f"   {result['url']}")
                if result.get("published"): print(f"   게시일: {result['published']}")
        print("\\n[실행 시간]")
        for name,seconds in timings.items(): print(f"- {name}: {seconds:.2f}초")
        print(f"- 인용 구조 검사: {'통과' if validation['valid'] else '경고'}")
        print("- 의미적 근거 검증: 미수행 (검색 요약 기반)")

if __name__=="__main__":
    main()
