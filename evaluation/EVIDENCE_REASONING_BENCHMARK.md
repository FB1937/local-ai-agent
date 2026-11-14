# M3 Evidence Reasoning Benchmark

## Purpose

This is a small, manually labelled Korean diagnostic set for measuring whether the configured Ollama model distinguishes four claim-to-passage relationships:

- \`supports\`: the passage directly supports the claim.
- \`contradicts\`: the passage directly conflicts with the claim.
- \`insufficient\`: the passage is related but does not establish the claim at the requested specificity.
- \`irrelevant\`: the passage does not bear on the claim.

The benchmark contains 32 examples, balanced at 8 per label. It includes numeric, temporal, scope, definition, code-behavior, negation, deployment, and unrelated-topic cases.

## Run

From the repository root, with Ollama running and the configured model already pulled:

\`\`\`bash
python -m evaluation.evidence_reasoning
\`\`\`

Optional arguments:

\`\`\`bash
python -m evaluation.evidence_reasoning --model qwen3.5:9b --timeout 120 --output evidence_reasoning_result.json
python -m evaluation.evidence_reasoning --ollama-url http://localhost:11434 --max-links 32
\`\`\`

The command uses \`OLLAMA_URL\` and \`OLLAMA_MODEL\` when options are omitted. It calls the same \`reason_claim_evidence()\` function used by the prototype; it does not implement a separate prompt or classifier.

## Metrics

- Accuracy: overall agreement with the manually assigned labels.
- Macro precision / recall / F1: gives each of the four labels equal weight.
- Per-class precision / recall / F1 and support.
- Confusion matrix: shows which labels are confused with one another.
- Unassessed count: failed, invalid, or skipped classifications.
- Mean, median, and p95 request latency for model calls that were actually attempted.

An unassessed result counts as incorrect for accuracy and remains visible as its own confusion-matrix column. Latency excludes cases for which no request was made.

## Interpretation and limitations

- This is a diagnostic set, not a statistically representative sample of Korean web evidence.
- The examples are synthetic and manually labelled; labels should be reviewed by a human before using results to tune prompts.
- A high score does not establish factual truth, source credibility, temporal validity, or production reliability.
- Run at least three times if sampling or nondeterminism is suspected, and compare model/prompt changes on the exact same dataset.
- Do not tune the prompt against every individual case. Add new cases for observed failure patterns and retain a held-out set before making broader quality claims.
- Record model tag, Ollama version, prompt revision, date, and hardware alongside any report used for a release decision.
- The evaluation uses sequential requests through the existing reasoner and therefore reflects its current serial execution behavior.
- CI validates dataset integrity and metric calculations only. It does not call Ollama.

## Suggested acceptance process

1. Run the benchmark on the pinned local model and save the JSON report.
2. Review every error and unassessed case; classify it as data ambiguity, prompt failure, model failure, or transport/format failure.
3. Correct ambiguous gold labels only with documented rationale; do not silently change labels to match model output.
4. Add a separate holdout set before adopting a prompt change.
5. Compare macro-F1, contradiction recall, unassessed rate, and p95 latency. Avoid a single accuracy threshold as the only release gate.
