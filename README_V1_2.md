# Local AI Agent v1.2 — Search Quality & Evidence

이 패키지는 v1.1 저장소에 적용하는 v1.2 변경 파일 묶음입니다. `router.py`, `.env`, `run.bat`, `evaluate_router.py`, 기존 `test_cases/routing_cases.json`, 기존 `tests/test_router_rules.py`는 유지합니다.

## 변경 사항
- `search_planner.py`: 질문의 최신성 요구와 사용자 지정 연도를 반영한 검색어 계획. 현재 날짜 기준 연도 보정.
- `search.py`: SearXNG 기간 필터와 게시일/엔진/점수 메타데이터 수집.
- `search_validator.py`: URL 정규화, 추적 파라미터 제거, 중복 제거, 출처 유형 분류, 키워드 기반 관련성 휴리스틱, 검색 실패 처리 및 최대 1회 재검색.
- `llm.py`: [S1] 출처 ID 전달, SEARCH/HYBRID 근거 중심 프롬프트, HYBRID 출력 구획화.
- `answer_validator.py`: 존재하지 않는 출처 ID와 URL 누락 검사.
- `main.py`: 통합 실행 흐름 및 단계별 실행 시간 표시.
- `tests/test_search_quality.py`: 검색 계획/정리/재시도/인용 구조 단위 테스트.

## 적용
1. 기존 프로젝트를 백업합니다.
2. 이 패키지의 파일을 프로젝트 루트에 같은 경로로 복사합니다.
3. 기존 `.env`, `.venv`, `run.bat`, `router.py`, `evaluate_router.py` 및 v1.1 테스트 데이터는 유지합니다.
4. CMD에서 프로젝트 폴더로 이동하고 venv를 활성화한 뒤 실행합니다.

```cmd
.venv\Scripts\activate.bat
python -m unittest discover -s tests -v
python evaluate_router.py --split dev
python main.py
```

## 테스트 및 한계
- 단위 테스트는 외부 Ollama/SearXNG 연결 없이 검색 계획, URL 중복 제거, 출처 분류, 재검색 제한, 인용 ID 검사를 검증합니다.
- 검색 관련성 점수는 키워드 겹침 휴리스틱이며 임베딩/의미 검색 랭커가 아닙니다.
- SearXNG의 제목과 요약문을 사용하며 페이지 본문을 수집하지 않습니다. 문장별 의미적 근거성은 자동 검증하지 않습니다.
- 도메인 분류는 휴리스틱입니다. 공식 사이트라는 이유만으로 개별 주장이 입증되는 것은 아닙니다.
- `publishedDate`는 검색 엔진에 따라 누락될 수 있습니다. 날짜가 없으면 최신이라고 단정하지 않습니다.
- 재검색은 최초 검색 후 최대 1회입니다.
- v1.1 라우팅 평가 점수는 답변 사실성이나 검색 품질을 나타내지 않습니다.
