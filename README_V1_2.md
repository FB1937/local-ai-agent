# Local AI Agent v1.2 — Search Quality & Evidence

## 목표
v1.1의 DIRECT / SEARCH / HYBRID 라우팅을 유지하고 검색 계획, 결과 정리, 제한적 재검색, 인용 구조 검사를 추가합니다.

## 변경 파일
- `search_planner.py`: 현재 날짜 및 시간 민감도 기반 검색 계획. 사용자가 연도를 명시하지 않은 최신 질문에 모델이 오래된 연도를 넣으면 현재 연도로 보정합니다.
- `search.py`: SearXNG time_range와 게시일/엔진/점수 메타데이터.
- `search_validator.py`: URL 정규화, 중복 제거, 도메인 기반 출처 유형, 키워드 관련성 휴리스틱, 최대 1회 재검색.
- `llm.py`: [S1] 출처 ID 및 HYBRID 전용 답변 구조.
- `answer_validator.py`: 존재하지 않는 인용 ID와 URL 누락 검사.
- `main.py`: 파이프라인 통합 및 단계별 시간 출력.
- `tests/test_search_quality.py`: 네트워크 없는 단위 테스트.

## 적용
기존 `router.py`, `.env`, `run.bat`, `evaluate_router.py`, 라우팅 테스트 데이터는 유지합니다. 기존 venv에서 실행:
```cmd
python -m unittest discover -s tests -v
python evaluate_router.py --split dev
python main.py
```

## 한계
- 키워드 겹침 점수는 의미적 검색 랭킹 모델이 아닙니다.
- SearXNG 게시일은 엔진에 따라 비어 있을 수 있습니다. 날짜가 없으면 최신으로 간주하지 않습니다.
- 현재 검색기는 제목과 요약문만 사용합니다. 페이지 본문을 직접 가져오지 않으므로 문장별 근거성은 검증하지 않습니다.
- 도메인 분류는 휴리스틱이며, 공식 사이트도 개별 주장을 자동 입증하지 않습니다.
- 재검색은 최초 검색 후 최대 1회입니다.
- 추가 외부 API나 라이브러리는 필요하지 않습니다.
