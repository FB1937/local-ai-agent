"""Small deterministic 50-case regression set for the M2 prototype."""
def build_cases() -> list[dict]:
    cases = []
    direct = [
        "변수는 값을 참조하기 위한 이름입니다.",
        "리스트는 변경 가능한 순서형 자료구조입니다.",
        "튜플은 요소 참조를 재할당할 수 없는 순서형 자료구조입니다.",
        "함수는 재사용 가능한 작업 단위입니다.",
        "반복문은 조건이나 횟수에 따라 작업을 반복합니다.",
        "기본키는 테이블의 각 행을 식별합니다.",
        "외래키는 다른 테이블의 키를 참조합니다.",
        "HTTP 요청은 클라이언트가 서버에 자원을 요청하는 메시지입니다.",
        "정규화는 데이터 중복과 갱신 이상을 줄이기 위한 설계 과정입니다.",
        "예외 처리는 실행 중 발생한 오류 상황을 다루는 구조입니다.",
    ]
    for i, answer in enumerate(direct, 1):
        cases.append({"id": f"DIRECT-{i:02d}", "category": "direct",
                      "answer": answer, "sources": [], "route": "DIRECT",
                      "assessment": {}, "expected_valid": True})
    for i in range(1, 11):
        fact = f"검증용 사실 {i}은 문서에 명시되어 있습니다."
        cases.append({"id": f"SEARCH-{i:02d}", "category": "search",
                      "answer": f"{fact} [S1]",
                      "sources": [{"source_id": "S1", "url": f"https://example.org/report/{i}",
                                   "title": f"검증 문서 {i}", "content": fact,
                                   "source_type": "official"}],
                      "route": "SEARCH", "assessment": {"sufficient": True},
                      "expected_valid": True})
    for i in range(1, 11):
        fact = f"공식 문서 {i}은 해당 기능의 지원 여부를 명시합니다."
        cases.append({"id": f"HYBRID-{i:02d}", "category": "hybrid",
                      "answer": f"개념적으로 이 기능은 특정 목적에 사용됩니다. {fact} [S1]",
                      "sources": [{"source_id": "S1", "url": f"https://vendor.example/docs/{i}",
                                   "title": f"공식 문서 {i}", "content": fact,
                                   "source_type": "official"}],
                      "route": "HYBRID", "assessment": {"sufficient": True},
                      "expected_valid": True})
    followups = [
        "앞서 설명한 리스트는 append로 요소를 추가할 수 있습니다.",
        "앞서 설명한 튜플은 생성 후 슬롯을 재할당할 수 없습니다.",
        "이 경우 기본키는 중복될 수 없습니다.",
        "해당 함수는 인자를 받아 결과를 반환할 수 있습니다.",
        "그 반복문은 조건이 거짓이 되면 종료됩니다.",
        "앞서 말한 외래키는 참조 무결성을 유지하는 데 사용됩니다.",
        "이 요청은 HTTP 메서드에 따라 의미가 달라집니다.",
        "해당 테이블은 정규화 수준에 따라 분해할 수 있습니다.",
        "이 예외는 구체적인 예외 유형으로 처리할 수 있습니다.",
        "그 변수는 새로운 값을 가리키도록 재할당할 수 있습니다.",
    ]
    for i, answer in enumerate(followups, 1):
        cases.append({"id": f"FOLLOWUP-{i:02d}", "category": "followup",
                      "answer": answer, "sources": [], "route": "DIRECT",
                      "assessment": {}, "expected_valid": True})
    for i in range(1, 6):
        cases.append({"id": f"EDGE-UNKNOWN-{i:02d}", "category": "edge",
                      "answer": f"근거를 확인했습니다. [S9] {i}",
                      "sources": [{"source_id": "S1", "url": "https://example.org/source",
                                   "title": "원문", "content": "근거 내용", "source_type": "other"}],
                      "route": "SEARCH", "assessment": {"sufficient": True},
                      "expected_valid": False})
    for i in range(1, 3):
        cases.append({"id": f"EDGE-URL-{i:02d}", "category": "edge",
                      "answer": f"자료가 확인되었습니다. [S1] {i}",
                      "sources": [{"source_id": "S1", "url": "", "title": "URL 누락",
                                   "content": "자료", "source_type": "other"}],
                      "route": "SEARCH", "assessment": {"sufficient": True},
                      "expected_valid": False})
    for i in range(1, 3):
        cases.append({"id": f"EDGE-PERCENT-{i:02d}", "category": "edge",
                      "answer": f"응답자의 {70+i}%가 이 기능을 사용합니다. [S1]",
                      "sources": [{"source_id": "S1", "url": f"https://example.org/no-percent/{i}",
                                   "title": "수치가 없는 자료", "content": "기능의 개요만 설명합니다.",
                                   "source_type": "other"}],
                      "route": "SEARCH", "assessment": {"sufficient": True},
                      "expected_valid": False})
    cases.append({"id": "EDGE-EMPTY-01", "category": "edge",
                  "answer": "", "sources": [], "route": "SEARCH",
                  "assessment": {}, "expected_valid": False})
    return cases
