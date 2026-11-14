# Independent Holdout: Evidence Relationship Classification

## Purpose and separation

This is a 24-case independent holdout set for the M3 claim-to-evidence relation classifier. It is separate from both the 32-case development benchmark and the 12-case boundary diagnostic set.

The holdout cases were authored after the development prompt and boundary set were established. They use different claim/evidence wording and topic areas; they are not copied from the earlier datasets. Keep this file unchanged while evaluating a given prompt/model candidate. Do not use its individual errors to tune the same candidate and then report the tuned result as an independent holdout score.

## Composition

Balanced across four labels (6 cases each):
- supports: transaction rollback, API pagination, password hashing, retry counts, audit-log retention, configuration precedence.
- contradicts: network port, mobile OS compatibility, cryptographic algorithm, maintenance schedule, timestamp format, API idempotency.
- insufficient: browser compatibility, plan limits, configurable timeout, at-rest encryption, CPU architecture, data residency.
- irrelevant: six unrelated-topic controls.

The cases deliberately include numeric limits, scope, default-versus-configurable settings, explicit exclusions, and distinctions between adjacent concepts (for example, transport encryption versus encryption at rest).

## Run

From the repository root, with Ollama running:

    python -m evaluation.evidence_reasoning --dataset evaluation/evidence_reasoning_holdout.jsonl --model qwen3.5:9b --output evidence_reasoning_holdout_result.json

The output contains per-case predictions, model explanations, and latency. Preserve the JSON report as the audit record.

## Interpretation

1. Run this holdout only after freezing the prompt and model tag for the candidate.
2. Review every incorrect or unassessed case and its explanation.
3. Do not revise the prompt based on these cases and continue calling the same set an independent holdout. If changes are made, create a new untouched holdout version for the next candidate.
4. Report per-class precision/recall/F1, confusion matrix, unassessed count, and latency, not accuracy alone.
5. This is still a small manually labelled synthetic set. It provides a more independent diagnostic than the development/boundary sets, but it is not a statistically representative sample of production evidence or proof of real-world reliability.
