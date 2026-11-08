# Local AI Agent v1.2 — Search Quality & Evidence

## 목표
v1.1의 DIRECT / SEARCH / HYBRID 라우팅은 유지하면서 검색 계획, 검색 결과 정리, 제한적 재검색, 인용 구조 검사를 추가합니다.

## 변경 파일
- `search_planner.py`: 최신성 요구와 사용자가 지정한 연도를 반영한 검색어 계획. 질문에 명시된 단일 연도가 라우터가 생성한 오래된 연도보다 우선합니다.
- `search.py`: SearXNG 기간 필터(`time_range`) 및 게시일·엔진·점수 메타데이터 수집.
- `search_validator.py`: URL 정규화, 추적 파라미터 제거, 중복 URL 제거, 출처 유형 분류, 키워드 겹침 기반 관련성 휴리스틱, 검색 실패 처리 및 최대 1회 재검색.
- `llm.py`: `[S1]` 형태의 출처 ID 전달, SEARCH/HYBRID 근거 중심 프롬프트, HYBRID 답변 구획화.
- `answer_validator.py`: 존재하지 않는 출처 ID와 URL 누락 검사.
- `main.py`: 새 검색 파이프라인 통합 및 단계별 실행 시간 출력.
- `tests/test_search_quality.py`: 네트워크 없이 실행하는 검색 계획·URL·재검색·인용 구조 단위 테스트.

## 적용
기존 `router.py`, `.env`, `run.bat`, `evaluate_router.py`, `test_cases/routing_cases.json`, `tests/test_router_rules.py`는 유지합니다. 이 브랜치에서 변경된 파일을 프로젝트에 반영한 뒤 CMD에서 실행합니다.

```cmd
.venv\Scripts\activate.bat
python -m unittest discover -s tests -v
python evaluate_router.py --split dev
python main.py
```

기존 환경에서 이미 `requests`와 `python-dotenv`가 설치되어 있다면 추가 패키지는 필요하지 않습니다. `.env`와 `run.bat`은 변경하지 않습니다.

## 테스트 및 한계
- v1.2 신규 검색 품질 단위 테스트는 15개이며, 네트워크/Ollama 호출 없이 실행됩니다.
- 기존 v1.1 라우터 테스트도 동일한 명령으로 함께 실행되어야 합니다. 이 브랜치의 신규 테스트만 통과했다고 전체 통합 테스트까지 통과한 것은 아닙니다.
- 키워드 겹침 점수는 투명한 휴리스틱이며 임베딩/의미 검색 랭커가 아닙니다.
- SearXNG의 제목과 요약문을 사용하며 페이지 본문을 직접 수집하지 않습니다. 문장별 의미적 근거성은 자동 검증하지 않습니다.
- 출처 유형은 도메인 기반 휴리스틱입니다. 공식 도메인이라는 사실만으로 개별 주장이 입증되는 것은 아닙니다.
- `publishedDate`는 검색 엔진에 따라 누락될 수 있습니다. 날짜가 없으면 최신이라고 단정하지 않습니다.
- 검색 재시도는 최초 검색 후 최대 1회입니다.
- 검색 결과 안의 지시문은 신뢰하지 않도록 답변 프롬프트에 명시합니다.
- v1.1 라우팅 정확도는 검색 품질이나 최종 답변의 사실성을 측정하지 않습니다.
