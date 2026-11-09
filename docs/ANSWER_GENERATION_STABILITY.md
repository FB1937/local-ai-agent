# Answer Generation Stability (v1.2)

## What changed

- Added `generate_answer_result()`, which returns the answer plus Ollama completion metadata.
- Kept `generate_answer()` as a backward-compatible string-returning wrapper.
- Empty or explicitly incomplete responses are retried at most once.
- A retry doubles `num_predict` relative to the configured baseline.
- A non-empty response stopped by the output limit is retained but explicitly flagged as potentially incomplete.
- The CLI reports retry count, Ollama stop reason, output token count, generation speed (when available), and failure warnings.
- Empty answers and failed/incomplete generations fail structural answer validation.

## Optional environment settings

These are optional and do not replace the existing settings:

```env
OLLAMA_NUM_PREDICT=2048
OLLAMA_TIMEOUT=300
```

The retry uses twice the configured `OLLAMA_NUM_PREDICT`. Increase this only when the local model has enough memory and the task needs longer output. A larger output limit can increase latency and memory use.

## Thinking과 최종 답변 분리 진단

Ollama Chat API는 모델이 지원하는 경우 `message.thinking`과 `message.content`를 별도 필드로 반환합니다. 이 프로젝트는 추론 내용을 사용자에게 노출하지 않고 각 필드의 문자 수만 진단 정보로 기록합니다.

- 첫 번째 생성은 모델의 기본 thinking 설정을 유지합니다.
- 재시도에서는 `think: false`를 요청해 추론 과정 대신 최종 답변 생성에 출력 예산을 사용하도록 합니다.
- 시도별 `content_characters`, `thinking_characters`, `eval_count`, `num_predict`, `done_reason`을 기록합니다.
- 이 방식은 Ollama 및 해당 모델이 thinking 제어를 지원할 때만 작동합니다. 지원 여부는 Ollama `/api/show`의 thinking 메타데이터로 확인할 수 있습니다.
- thinking 문자 수는 추론 토큰 수와 같지 않습니다. `eval_count`도 thinking과 최종 답변의 토큰 수를 각각 분리하지 않으므로, 이 로그만으로 토큰 소비 비율을 단정하지 않습니다.

## Tests

Run from the project root:

```cmd
python -m unittest discover -s tests -v
```

New tests mock the Ollama HTTP response and cover empty-response retry, repeated empty output, token-limit retry, persistent truncation, HTTP timeout, and answer-validator failure states.

## Limitations

- This is generation-state validation, not semantic fact checking.
- A non-empty answer may still contain unsupported claims.
- Ollama versions/models may differ in whether they populate `done_reason`; absent metadata cannot prove that an answer is complete.
- The code changes are committed on a feature branch. They must be tested locally with the user's Ollama/SearXNG setup before merging into `main`.
