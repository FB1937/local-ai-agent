"""Evaluate deterministic M3.4 claim synthesis against the labelled dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from claim_analysis import Claim, ClaimAnalysisResult, ClaimEvidenceLink, EvidencePassage
from claim_synthesis import synthesize_claim_evidence

LABELS = ("supported", "contradicted", "conflict", "insufficient", "unassessed")
DEFAULT_DATASET = Path(__file__).with_name("multi_evidence_cases.jsonl")


def _time_bounds(value: str | None) -> tuple[str | None, str | None]:
    if not value or "/" not in value:
        return None, None
    start, end = value.split("/", 1)
    return start or None, end or None


def evaluate_dataset(dataset_path: str | Path = DEFAULT_DATASET) -> dict:
    rows = [json.loads(line) for line in Path(dataset_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    confusion = {actual: {predicted: 0 for predicted in LABELS} for actual in LABELS}
    mismatches = []
    conflict_type_correct = 0
    for row in rows:
        passages, links = [], []
        for index, item in enumerate(row["evidence"], 1):
            evidence_id = f"E{index}"
            start, end = _time_bounds(item.get("valid_time"))
            passages.append(EvidencePassage(
                evidence_id=evidence_id,
                source_id=item["source_id"],
                source_group=item.get("source_group"),
                text=item["text"],
                location=item["evidence_id"],
                scope=None if item.get("scope") in (None, "default") else item.get("scope"),
                condition=item.get("condition"),
                valid_time_start=start,
                valid_time_end=end,
            ))
            links.append(ClaimEvidenceLink(
                claim_id="C1", evidence_id=evidence_id,
                relation=item["relation"], reason="Manually labelled pairwise relation.",
            ))
        result = synthesize_claim_evidence(ClaimAnalysisResult(
            claims=[Claim(claim_id="C1", text=row["claim"])],
            passages=passages, links=links,
        )).claims[0]
        expected = row["expected_status"]
        confusion[expected][result.status] += 1
        conflict_type_correct += int(result.conflict_type == row["expected_conflict_type"])
        if result.status != expected:
            mismatches.append({
                "case_id": row["case_id"], "expected": expected, "actual": result.status,
                "expected_conflict_type": row["expected_conflict_type"],
                "actual_conflict_type": result.conflict_type, "reason": result.reason,
            })

    total = len(rows)
    correct = total - len(mismatches)
    per_class, f1_values = {}, []
    for label in LABELS[:-1]:
        tp = confusion[label][label]
        fp = sum(confusion[actual][label] for actual in LABELS if actual != label)
        fn = sum(confusion[label][predicted] for predicted in LABELS if predicted != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
        f1_values.append(f1)
    return {
        "total": total, "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "conflict_type_accuracy": conflict_type_correct / total if total else 0.0,
        "per_class": per_class, "confusion_matrix": confusion, "mismatches": mismatches,
        "note": "Synthetic diagnostic only. Joint entailment from multiple pairwise-insufficient passages is intentionally not inferred by the deterministic baseline.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()
    report = evaluate_dataset(args.dataset)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
