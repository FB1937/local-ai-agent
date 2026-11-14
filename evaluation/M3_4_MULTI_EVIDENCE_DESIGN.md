# M3.4 — Claim-level multi-evidence synthesis and conflict diagnosis

## Status and scope

Design, labelled diagnostic dataset, deterministic aggregation baseline, and a bounded LLM-assisted joint-entailment layer are implemented on the existing M3 PR branch. Agent-pipeline integration remains pending. `main` is not changed and PR #14 remains Draft.

## Problem

M3.3 classifies one claim–passage pair at a time. A claim may have several passages, including repeated passages from one source, independent corroboration, complementary facts, and apparently inconsistent statements. A set of pairwise labels is not yet a claim-level verdict.

## Output contract

Keep every original `ClaimEvidenceLink` relation and reason. Add a separate claim-level synthesis result; never overwrite passage-level judgments.

Proposed aggregate status:

- `supported`: at least one valid, scope-aligned passage directly entails the claim, and no unresolved same-scope contradiction remains.
- `contradicted`: at least one valid, scope-aligned passage directly entails a negation/incompatible alternative, and no unresolved same-scope support remains.
- `conflict`: both support and contradiction remain after checking subject, scope, conditions, version, and valid time. Preserve both sides and require review; do not vote.
- `insufficient`: evidence is relevant but does not establish either side, or apparent conflicts are explained by scope/time/condition mismatch, or only weak/duplicated evidence is available.
- `unassessed`: required pairwise judgments are missing, failed, or skipped. Never silently coerce this to insufficient or a normal verdict.

The exact enum and model names are provisional until tests and API consumers are reviewed.

## Aggregation rules

1. **No majority vote.** One direct contradiction cannot be outvoted by several weak support passages. Conversely, duplicated copies of one source do not count as independent corroboration.
2. **Separate relation from independence.** A passage can support a claim even if it is not an independent source. Independence changes corroboration strength, not the logical relation label.
3. **Deduplicate before counting.** Use normalized exact text and source identity as a conservative first pass. Near-duplicate detection must be explicit and auditable; do not silently merge merely similar passages.
4. **Compare like with like.** Diagnose subject, product/plan, API version, environment, geography, condition, unit, measurement method, and valid-time overlap. Different scopes are not contradictions.
5. **Treat freshness as metadata, not truth.** Publication date alone must not invalidate a passage. For time-sensitive claims, compare the claim's reference time with the passage's stated valid time; if unknown, preserve uncertainty.
6. **Complementary evidence may jointly entail.** Do not require every passage independently to state the full claim. Joint inference is allowed only when the required premises are explicit and compatible; preserve which passages jointly support it.
7. **Absence is not negation.** Omitted coverage, incomplete lists, and missing details do not contradict a claim.
8. **Provenance is mandatory.** Each aggregate reason must point back to passage IDs, source IDs, source groups, scope/time metadata, and any deduplication decision.
9. **Unassessed is fail-closed.** If a relevant link needed for the decision was not evaluated, the aggregate must expose that gap rather than claim a clean verdict.
10. **No source-authority shortcut.** Official/primary sources may be ranked as more authoritative in later policy, but authority alone does not erase a direct unresolved conflict.

## Conflict taxonomy

- `direct`: mutually incompatible statements under the same subject, conditions, scope, version, and overlapping valid time.
- `scope_mismatch`: different products, plans, roles, environments, regions, or units.
- `temporal_mismatch`: statements apply to different non-overlapping periods.
- `condition_mismatch`: statements depend on different explicit preconditions.
- `duplicate_not_independent`: repeated or syndicated content from the same underlying source.
- `freshness_uncertain`: evidence is stale or its valid period is unknown for a current-state claim.
- `none`: no conflict diagnosed.

A scope/time mismatch normally removes the apparent contradiction for this claim; it must not be represented as a direct conflict.

## Evaluation design

Dataset: `evaluation/multi_evidence_cases.jsonl` — 32 manually labelled Korean synthetic cases, balanced across four aggregate outcomes (8 each). The cases cover independent corroboration, same-source repetition, complementary premises, direct conflict, explicit exhaustive lists, scope/time/condition mismatches, duplicate sources, weak evidence, and freshness uncertainty.

Each evidence item records a pairwise relation label plus source identity/group and optional scope/valid-time/publication metadata. The expected aggregate label and conflict type are manually assigned. This is a small diagnostic set, not a production benchmark or statistical estimate.

Keep the M3.3 24-case holdout untouched. Do not tune M3.4 prompts or rules on that holdout. A separate M3.4 holdout now exists at `evaluation/multi_evidence_holdout.jsonl`: 24 newly authored Korean synthetic cases, balanced 6 per aggregate status. It is a frozen diagnostic set and must not be used for prompt/rule tuning; run it once after the current development decisions are frozen.

## Implementation sequence

- [x] Freeze and review the data contract and 32 labelled cases.
- [x] Add Pydantic aggregate result contracts with evidence provenance.
- [x] Implement deterministic prechecks for missing judgments, duplicate identity, and explicit scope/time/condition metadata.
- [x] Add unit tests for direct conflict, scope/condition/time mismatch, duplicates, missing judgments, no evidence, and no-majority behavior.
- [x] Use an LLM only where joint entailment is needed; request structured explanations with evidence IDs.
- [x] Keep deterministic policy responsible for the baseline verdict; semantic upgrades require valid IDs from the supplied set and at least two distinct passages.
- [x] Run local Qwen 3.5 9B semantic evaluation on the 32-case development set and inspect status and conflict-type errors separately.
- [x] Add a separate 24-case M3.4 holdout; keep it untouched until the one-time independent evaluation.
- [ ] Run the frozen M3.4 holdout once with deterministic and semantic modes; do not tune against its results.
- [ ] Integrate into the Agent response pipeline only after standalone behavior is stable.

## Current deterministic baseline

`claim_synthesis.py` adds `synthesize_claim_evidence()` and separate `ClaimEvidenceSynthesis` / `ClaimSynthesisReport` contracts. It preserves pairwise links, returns `unassessed` when a link is missing or unjudged, and never uses majority voting. A direct support/contradiction pair remains `conflict` unless all cross-pairs have explicit scope, condition, or valid-time mismatches. Duplicate passages are grouped by canonical source/source group plus normalized exact text; duplicates do not inflate independent-source counts. A single valid support remains support even when repeated passages are duplicates.

This baseline deliberately does not infer joint entailment from multiple pairwise `insufficient` labels. Dataset case ME-S08 is a labelled example for the semantic synthesis layer. `semantic_claim_synthesis.py` now makes a bounded Ollama call only for deterministic-`insufficient` claims with at least two non-duplicate insufficient passages. It accepts structured `supports`, `contradicts`, or `insufficient` results; support/contradiction upgrades require at least two evidence IDs from the supplied set and at least two distinct canonical source groups. Different passage IDs or source IDs alone do not establish source independence; passages sharing a canonical source group cannot jointly corroborate a claim. Evidence selection prioritizes one passage per distinct source group. Original pairwise links remain unchanged, joint evidence IDs are recorded separately, and request/schema/provenance failures leave the deterministic verdict intact. The evaluator exposes this as an opt-in `--semantic` mode; the default remains deterministic. This layer is not integrated into the Agent response pipeline. The system still does not infer near-duplicate content, source authority, or freshness from publication date alone.

## Non-goals

- No majority voting or opaque confidence score.
- No automatic selection of a winning source when two same-scope primary sources conflict.
- No claim that source authority, recency, or passage count alone proves truth.
- No M3.5 pipeline integration in this increment.


## M3.4 evaluator diagnostics and holdout status

The evaluator now reports claim-status mismatches and conflict-taxonomy mismatches independently. `mismatches` tracks aggregate status; `conflict_type_mismatches` tracks taxonomy labels even when the aggregate status is correct. The 32-case semantic development run previously reported 32/32 status accuracy and macro-F1 1.0, but conflict-type accuracy was 31/32. The initial diagnostic exposed two apparent subtype mismatches. ME-I03 was a development-data metadata error: its expected `condition_mismatch` label was paired with two different `scope` values and no explicit `condition`; the case is corrected to use the same scope and distinct condition fields. After that correction, the remaining subtype mismatch is ME-I04: expected `duplicate_not_independent`, actual `none`, with aggregate status correctly `insufficient`. This identifies a taxonomy behavior gap; it must not be hidden by the perfect status score.

The independent M3.4 holdout contains 24 newly authored cases balanced across supported, contradicted, conflict, and insufficient (6 each). It includes direct and joint entailment/contradiction, source lineage, explicit scope/condition/time differences, weak evidence, and stale current-state evidence. The holdout has not been run against the local model and must remain untouched by further prompt/rule tuning. The latest GitHub Actions run on the current development head passed all 187 Python tests, including evaluator diagnostics, the ME-I03 metadata regression, and holdout integrity. Its synthetic nature means even a perfect result is diagnostic evidence, not proof of real-world generalization.


## M3.4 independent holdout evaluation (2026-11-14)

The frozen 24-case holdout has now been executed in both modes. It is therefore a consumed evaluation set: retain the results as a record and do not use these cases to tune prompts or rules. Any next iteration that needs an untouched holdout must author a new independent set.

| Metric | Deterministic | Semantic-assisted |
|---|---:|---:|
| Claim status accuracy | 19/24 (79.17%) | 22/24 (91.67%) |
| Macro-F1 | 0.7931 | 0.9115 |
| Conflict taxonomy accuracy | 23/24 (95.83%) | 23/24 (95.83%) |
| Semantic assessments | 0 | 7 |
| Semantic skipped | 0 | 1 |
| Semantic unassessed | 0 | 0 |

Semantic mode corrected three deterministic status errors, including three joint-support cases, but left two status mismatches:
- M34-H-C03: expected contradicted, actual supported. The model's explanation explicitly calculates 48,000 KRW (52,000 + 4,000 - 8,000) against the claimed 80,000 KRW and says the premises contradict the claim, yet the structured relation field is supports. The current implementation validates evidence IDs and source independence but does not validate consistency between the relation enum and its explanation. This is a fail-open semantic-output consistency gap.
- M34-H-C05: expected contradicted, actual insufficient. The model did not derive the business-day contradiction from the stated Monday-to-following-Tuesday interval. This remains a conservative joint-inference miss; the case's business-day counting convention should be reviewed when designing future evaluation data, not retroactively tuned here.

The deterministic baseline missed five joint-inference cases (S03, S04, S06, C03, C05), which is consistent with its documented non-goal of inferring joint entailment. Semantic mode improved status accuracy by 12.5 percentage points on this small synthetic set. This is diagnostic evidence only, not a production-quality estimate or proof of real-world generalization.

The development set still has one taxonomy mismatch, ME-I04 (duplicate_not_independent expected, none actual). The holdout's M34-H-I04 reproduces the same taxonomy gap. Source-group lineage should be distinguished from exact/near duplicate content and from independent corroboration; do not label every pair of passages from one source as duplicated. The next design should consider exposing independence/corroboration as a separate diagnostic field instead of overloading conflict_type.


## Follow-up implementation: relation/reason consistency and source lineage (2026-11-14)

### Semantic consistency audit

The semantic joint-entailment path now performs a second bounded Ollama request after the first structured relation response. The audit receives the exact claim, selected evidence passages, proposed relation, and the first response's explanation. It returns a strict boolean consistency decision.

- Only an explicit `consistent=true` permits the semantic assessment to proceed to the existing Python evidence-ID and source-group checks.
- A false, malformed, ambiguous, timed-out, or failed audit preserves the deterministic baseline (fail-closed).
- This is a same-model critic pass, not an independent verifier, external proof, or guarantee against correlated model error. It adds one extra model request per eligible claim and therefore increases latency.
- Mocked tests cover a relation/reason mismatch and verifier failure. Live Qwen evaluation is still required to estimate false acceptance/rejection and latency impact.

### Source lineage is not duplicate content

Claim synthesis now reports `distinct_source_group_count` and `source_lineage_status` separately from `duplicate_groups`. Exact duplicate grouping still requires normalized identical text from the same underlying source. Multiple distinct passages from one source group are reported as a single source group but are not automatically classified as duplicate content. Multiple source groups are a lineage-coverage diagnostic, not proof of true editorial/statistical independence.

The earlier ME-I04 / M34-H-I04 expected `duplicate_not_independent` subtype remains a taxonomy/labeling question; aggregate status remains `insufficient`. Do not change or retune the consumed 24-case holdout. Any future quality claim requires a fresh untouched holdout or real-world labelled documents.

### CI

Head `7edf4ab51a15e952ea0060e2d07fe351dd4800ba`: both GitHub Actions unittest jobs passed, with 191 Python tests per run (2026-11-14). This verifies mocked behavior and regression contracts, not live model quality.
