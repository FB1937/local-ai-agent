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
