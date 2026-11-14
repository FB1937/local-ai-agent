"""Run a labelled Korean claim/evidence benchmark against a local Ollama model."""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path

import requests

from claim_analysis import Claim, ClaimAnalysisResult, ClaimEvidenceLink, EvidencePassage
from evidence_reasoner import reason_claim_evidence

LABELS = ("supports", "contradicts", "insufficient", "irrelevant")
DEFAULT_DATASET = Path(__file__).with_name("evidence_reasoning_cases.jsonl")


def load_dataset(path: str | Path = DEFAULT_DATASET) -> list[dict]:
    cases = []
    seen_ids = set()
    for line_number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        case = json.loads(raw)
        required = {"id", "expected_relation", "category", "difficulty", "claim", "evidence", "gold_reason"}
        missing = required - case.keys()
        if missing:
            raise ValueError(f"line {line_number}: missing fields {sorted(missing)}")
        if case["id"] in seen_ids:
            raise ValueError(f"line {line_number}: duplicate case id {case['id']}")
        if case["expected_relation"] not in LABELS:
            raise ValueError(f"line {line_number}: invalid expected_relation")
        if not case["claim"].strip() or not case["evidence"].strip():
            raise ValueError(f"line {line_number}: claim and evidence must be non-empty")
        seen_ids.add(case["id"])
        cases.append(case)
    if not cases:
        raise ValueError("evaluation dataset is empty")
    return cases


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return round(ordered[index], 2)


def score_predictions(cases: list[dict], predictions: dict[str, str | None],
                      latencies: dict[str, float] | None = None) -> dict:
    latencies = latencies or {}
    matrix = {label: {pred: 0 for pred in (*LABELS, "unassessed")} for label in LABELS}
    per_class = {}
    correct = 0
    for case in cases:
        gold = case["expected_relation"]
        predicted = predictions.get(case["id"])
        bucket = predicted if predicted in LABELS else "unassessed"
        matrix[gold][bucket] += 1
        correct += int(bucket == gold)

    for label in LABELS:
        tp = matrix[label][label]
        predicted_count = sum(matrix[gold][label] for gold in LABELS)
        support = sum(matrix[label].values())
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    latency_values = list(latencies.values())
    total = len(cases)
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "macro_precision": round(sum(x["precision"] for x in per_class.values()) / len(LABELS), 4),
        "macro_recall": round(sum(x["recall"] for x in per_class.values()) / len(LABELS), 4),
        "macro_f1": round(sum(x["f1"] for x in per_class.values()) / len(LABELS), 4),
        "unassessed": sum(1 for case in cases if predictions.get(case["id"]) not in LABELS),
        "per_class": per_class,
        "confusion_matrix": matrix,
        "latency_ms": {
            "count": len(latency_values),
            "mean": round(statistics.mean(latency_values), 2) if latency_values else None,
            "median": round(statistics.median(latency_values), 2) if latency_values else None,
            "p95": _percentile(latency_values, 0.95),
        },
    }


def build_analysis(cases: list[dict]) -> ClaimAnalysisResult:
    claims, passages, links = [], [], []
    for index, case in enumerate(cases, 1):
        claim_id, evidence_id = f"C{index}", f"E{index}"
        claims.append(Claim(
            claim_id=claim_id, text=case["claim"], cited_source_ids=["S1"]
        ))
        passages.append(EvidencePassage(
            evidence_id=evidence_id, source_id="S1", text=case["evidence"],
            location=f"benchmark:{case['id']}",
            source_url=f"https://example.invalid/evidence-reasoning/{case['id']}",
            claim_id=claim_id,
        ))
        links.append(ClaimEvidenceLink(claim_id=claim_id, evidence_id=evidence_id))
    return ClaimAnalysisResult(claims=claims, passages=passages, links=links)


def evaluate(cases: list[dict], *, model: str | None = None,
             ollama_url: str | None = None, timeout: int = 90,
             max_links: int | None = None) -> dict:
    analysis = build_analysis(cases)
    call_timings: dict[str, float] = {}
    case_by_claim = {f"C{i}": case for i, case in enumerate(cases, 1)}
    request_index = 0

    def timed_request(url, **kwargs):
        nonlocal request_index
        case = cases[request_index] if request_index < len(cases) else None
        request_index += 1
        started = time.perf_counter()
        try:
            return requests.post(url, **kwargs)
        finally:
            if case:
                call_timings[case["id"]] = (time.perf_counter() - started) * 1000

    reasoning = reason_claim_evidence(
        analysis, model=model, ollama_url=ollama_url, timeout=timeout,
        max_links=max_links or len(cases), request_fn=timed_request,
    )
    predictions = {}
    details = []
    for link in reasoning.links:
        case = case_by_claim[link.claim_id]
        predictions[case["id"]] = link.relation
        details.append({
            "id": case["id"],
            "category": case["category"],
            "difficulty": case["difficulty"],
            "claim": case["claim"],
            "evidence": case["evidence"],
            "expected": case["expected_relation"],
            "predicted": link.relation,
            "correct": link.relation == case["expected_relation"],
            "reason": link.reason,
            "latency_ms": round(call_timings[case["id"]], 2) if case["id"] in call_timings else None,
        })
    summary = score_predictions(cases, predictions, call_timings)
    return {
        "model": model or os.getenv("OLLAMA_MODEL", "qwen3.5:9b"),
        "ollama_url": (ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434")).rstrip("/"),
        "timeout_seconds": timeout,
        "summary": summary,
        "cases": details,
        "scope_note": (
            "This is a small, manually labelled Korean diagnostic benchmark. "
            "It measures agreement with its labels, not general factual truth or production reliability. "
            "Review misclassifications manually before changing prompts or thresholds."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--model", default=None, help="Defaults to OLLAMA_MODEL or qwen3.5:9b")
    parser.add_argument("--ollama-url", default=None, help="Defaults to OLLAMA_URL or localhost:11434")
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--max-links", type=int, default=None)
    parser.add_argument("--output", help="Optional path for a JSON result file")
    args = parser.parse_args()
    if args.timeout < 1 or (args.max_links is not None and args.max_links < 1):
        parser.error("--timeout and --max-links must be positive")
    cases = load_dataset(args.dataset)
    result = evaluate(
        cases, model=args.model, ollama_url=args.ollama_url,
        timeout=args.timeout, max_links=args.max_links,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    if args.output:
        print(f"Detailed results written to: {args.output}")


if __name__ == "__main__":
    main()
