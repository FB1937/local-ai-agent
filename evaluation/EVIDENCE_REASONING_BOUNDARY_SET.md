# Evidence Reasoning Boundary Evaluation

## Purpose

This 12-case diagnostic set is separate from the original 32-case development benchmark. It concentrates on the boundary between direct support, direct contradiction, and insufficient information, including the three error patterns observed in the first Qwen 3.5 9B run.

The cases are manually labelled. They are a diagnostic aid, not an independent estimate of production accuracy. Review labels and model explanations before drawing conclusions.

## Coverage

- Explicit versus incomplete language lists
- Customer-selected regions versus explicitly unavailable regions
- Expandable storage versus an explicitly unbounded or explicitly capped plan
- Unrelated evidence as a control group

The set is balanced: 3 supports, 3 contradicts, 3 insufficient, and 3 irrelevant cases.

## Run

Run the boundary set separately so its results remain distinguishable from the original development benchmark:

    python -m evaluation.evidence_reasoning --dataset evaluation/evidence_reasoning_boundary_cases.jsonl --model qwen3.5:9b --output evidence_reasoning_boundary_result.json

Run the original 32-case regression benchmark separately:

    python -m evaluation.evidence_reasoning --dataset evaluation/evidence_reasoning_cases.jsonl --model qwen3.5:9b --output evidence_reasoning_regression_result.json

## Interpretation

1. Compare the original 32-case benchmark before and after the prompt change.
2. Inspect all 12 boundary cases, not only aggregate accuracy.
3. Confirm that the previous three failure patterns now receive insufficient on their matching boundary examples.
4. Check that explicit support and explicit contradiction remain correctly classified.
5. Do not tune the prompt against this small set repeatedly; create a separate holdout set before making broader reliability claims.
