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

Keep the M3.3 24-case holdout untouched. Do not tune M3.4 prompts or rules on that holdout; use the new M3.4 dataset for development and create a separate M3.4 holdout before claiming independent generalization.

## Implementation sequence

- [x] Freeze and review the data contract and 32 labelled cases.
- [x] Add Pydantic aggregate result contracts with evidence provenance.
- [x] Implement deterministic prechecks for missing judgments, duplicate identity, and explicit scope/time/condition metadata.
- [x] Add unit tests for direct conflict, scope/condition/time mismatch, duplicates, missing judgments, no evidence, and no-majority behavior.
- [x] Use an LLM only where joint entailment is needed; request structured explanations with evidence IDs.
- [x] Keep deterministic policy responsible for the baseline verdict; semantic upgrades require valid IDs from the supplied set and at least two distinct passages.
- [ ] Run local Qwen 3.5 9B semantic evaluation and inspect every error, not just the aggregate score.
- [ ] Add a separate M3.4 semantic holdout before claiming independent generalization.
- [ ] Integrate into the Agent response pipeline only after standalone behavior is stable.

## Current deterministic baseline

`claim_synthesis.py` adds `synthesize_claim_evidence()` and separate `ClaimEvidenceSynthesis` / `ClaimSynthesisReport` contracts. It preserves pairwise links, returns `unassessed` when a link is missing or unjudged, and never uses majority voting. A direct support/contradiction pair remains `conflict` unless all cross-pairs have explicit scope, condition, or valid-time mismatches. Duplicate passages are grouped by canonical source/source group plus normalized exact text; duplicates do not inflate independent-source counts. A single valid support remains support even when repeated passages are duplicates.

This baseline deliberately does not infer joint entailment from multiple pairwise `insufficient` labels. Dataset case ME-S08 is a labelled example for the semantic synthesis layer. `semantic_claim_synthesis.py` now makes a bounded Ollama call only for deterministic-`insufficient` claims with at least two non-duplicate insufficient passages. It accepts structured `supports`, `contradicts`, or `insufficient` results; support/contradiction upgrades require at least two evidence IDs from the supplied set. Original pairwise links remain unchanged, joint evidence IDs are recorded separately, and request/schema/provenance failures leave the deterministic verdict intact. The evaluator exposes this as an opt-in `--semantic` mode; the default remains deterministic. This layer is not integrated into the Agent response pipeline. The system still does not infer near-duplicate content, source authority, or freshness from publication date alone.

## Non-goals

- No majority voting or opaque confidence score.
- No automatic selection of a winning source when two same-scope primary sources conflict.
- No claim that source authority, recency, or passage count alone proves truth.
- No M3.5 pipeline integration in this increment.
