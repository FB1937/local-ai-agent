# Six-Stage Local AI Agent Architecture

## Purpose

The six-stage model is the project's conceptual learning and implementation framework. It is not a claim about the internal execution order of any hosted LLM.

1. Input Processing
2. Context Understanding
3. Intent & Task Classification
4. Planning & Routing
5. Execution
6. Response Generation

## Runtime mapping

| Stage | Module | Responsibility |
|---|---|---|
| 1 | `input_processor.py` | Validate and normalize text input; attach receipt metadata |
| 2 | `context_manager.py` | Supply a bounded, session-local history of completed turns |
| 3 | `task_classifier.py` | Extract heuristic intent, task types, domain, freshness and output hints |
| 4 | `planner.py` + `router.py` + `search_planner.py` | Select DIRECT/SEARCH/HYBRID and construct a route-specific plan |
| 5 | `executor.py` + `search.py` + `search_validator.py` | Execute the currently supported SearXNG search capability and validate evidence |
| 6 | `response_generator.py` + `llm.py` + `answer_validator.py` | Generate the answer, check citation structure, and format validation notices |
| Orchestration | `agent.py` | Run the stages in order, collect timings and update session context |

## Data flow

```text
Raw input
  -> normalized input envelope
  -> prior conversation context
  -> task metadata
  -> route + execution plan
  -> tool execution + evidence assessment
  -> answer generation + structural validation
  -> completed turn added to session context
```

## Important implementation boundaries

- Tokenization is performed by the configured Ollama model/tokenizer. Stage 1 currently normalizes and validates text; it does not implement a separate tokenizer.
- Stage 2 context is in-memory only, bounded by turn count and character count, and disappears when the process exits. Obvious follow-up phrases (for example, “그 버전은?”) cause recent turns to be attached before task classification and routing. No persistent user profile or long-term memory is implemented.
- Stage 3 is deterministic rule-based classification. Its labels are hints, not a calibrated prediction or a substitute for the LLM router.
- Stage 4 keeps the existing DIRECT / SEARCH / HYBRID router. Search planning remains a separate deterministic helper.
- Stage 5 currently registers only SearXNG search. Arbitrary shell commands, database writes, file mutations, and external side effects are not enabled.
- Search validation currently allows at most one retry (two search attempts total). A general multi-tool autonomous re-planning loop is not implemented yet.
- Stage 6 validates answer presence and citation ID/URL structure. It does not prove that a source semantically entails a claim.
- Retrieved web content is untrusted input and must never be treated as executable instructions.

## Extension path

1. Add explicit typed tool interfaces and per-tool input/output schemas.
2. Add permission gates for tools that mutate files, databases, or external services.
3. Add a task planner that can decompose multi-step goals and track dependencies.
4. Add bounded observe -> evaluate -> re-plan loops with a maximum step budget.
5. Fetch source pages and add claim-level evidence checks.
6. Add persistent memory only with explicit scope, retention, and user controls.

## Test command

```bash
python -m unittest discover -s tests -v
```

The unit tests use mocks for external services. Passing them does not verify the live Ollama/SearXNG runtime or semantic correctness of generated answers.
