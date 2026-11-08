"""Run the labelled routing test set against the local Ollama model.

Examples:
    python evaluate_router.py --split dev
    python evaluate_router.py --split holdout
    python evaluate_router.py --split all
"""

import argparse
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from router import classify_question, VALID_ROUTES

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CASE_FILE = BASE_DIR / "test_cases" / "routing_cases.json"
DEFAULT_REPORT_DIR = BASE_DIR / "reports"
ROUTE_ORDER = ["DIRECT", "SEARCH", "HYBRID"]


def load_cases(path: Path, split: str):
    with path.open("r", encoding="utf-8") as file:
        cases = json.load(file)
    if not isinstance(cases, list):
        raise ValueError("테스트 케이스 파일의 최상위 값은 배열이어야 합니다.")

    selected = [case for case in cases if split == "all" or case.get("split") == split]
    for case in selected:
        if case.get("expected_route") not in VALID_ROUTES:
            raise ValueError(f"잘못된 정답 경로: {case.get('id')} / {case.get('expected_route')}")
    if not selected:
        raise ValueError(f"선택한 split에 테스트 케이스가 없습니다: {split}")
    return selected


def calculate_metrics(cases, prediction_key="route"):
    total = len(cases)
    correct = sum(case["expected_route"] == case[prediction_key] for case in cases)
    per_route = {}

    for route in ROUTE_ORDER:
        true_positive = sum(
            case["expected_route"] == route and case[prediction_key] == route
            for case in cases
        )
        false_positive = sum(
            case["expected_route"] != route and case[prediction_key] == route
            for case in cases
        )
        false_negative = sum(
            case["expected_route"] == route and case[prediction_key] != route
            for case in cases
        )
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        per_route[route] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(case["expected_route"] == route for case in cases),
        }

    macro_f1 = sum(item["f1"] for item in per_route.values()) / len(ROUTE_ORDER)
    return {
        "accuracy": round(correct / total, 4) if total else 0.0,
        "correct": correct,
        "total": total,
        "macro_f1": round(macro_f1, 4),
        "per_route": per_route,
    }


def build_confusion_matrix(cases):
    matrix = {actual: {predicted: 0 for predicted in ROUTE_ORDER} for actual in ROUTE_ORDER}
    for case in cases:
        matrix[case["expected_route"]][case["route"]] += 1
    return matrix


def main():
    parser = argparse.ArgumentParser(description="로컬 Qwen 라우팅 정확도 평가")
    parser.add_argument("--split", choices=["dev", "holdout", "all"], default="holdout")
    parser.add_argument("--case-file", type=Path, default=DEFAULT_CASE_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()

    cases = load_cases(args.case_file, args.split)
    print(f"라우팅 평가 시작: {len(cases)}개 / split={args.split}")
    print("실제 Qwen 호출이 수행되므로 테스트마다 시간이 걸릴 수 있습니다.\n")

    evaluated = []
    for index, case in enumerate(cases, 1):
        started = time.perf_counter()
        decision = classify_question(case["question"])
        elapsed = time.perf_counter() - started

        row = {
            **case,
            "route": decision["route"],
            "model_route": decision["model_route"],
            "confidence": decision["confidence"],
            "reason": decision["reason"],
            "search_query": decision["search_query"],
            "overridden": decision["overridden"],
            "fallback": decision["fallback"],
            "elapsed_seconds": round(elapsed, 2),
        }
        evaluated.append(row)
        status = "OK" if row["expected_route"] == row["route"] else "MISS"
        print(
            f"[{index:02d}/{len(cases):02d}] {status:4} {case['id']}: "
            f"정답={case['expected_route']:<6} 예측={row['route']:<6} "
            f"(모델={row['model_route']}, 확신도={row['confidence']:.2f}, {elapsed:.1f}s)"
        )

    metrics = calculate_metrics(evaluated)
    raw_metrics = calculate_metrics(evaluated, prediction_key="model_route")
    matrix = build_confusion_matrix(evaluated)
    override_count = sum(row["overridden"] for row in evaluated)
    fallback_count = sum(row["fallback"] for row in evaluated)
    errors = [
        {
            "id": row["id"],
            "question": row["question"],
            "expected": row["expected_route"],
            "predicted": row["route"],
            "model_route": row["model_route"],
            "confidence": row["confidence"],
            "reason": row["reason"],
        }
        for row in evaluated
        if row["expected_route"] != row["route"]
    ]

    print("\n" + "=" * 68)
    print("평가 결과")
    print("=" * 68)
    print(f"모델 원판정 정확도: {raw_metrics['correct']}/{raw_metrics['total']} = {raw_metrics['accuracy'] * 100:.1f}%")
    print(f"안전 규칙 적용 후 정확도: {metrics['correct']}/{metrics['total']} = {metrics['accuracy'] * 100:.1f}%")
    print(f"최종 Macro F1: {metrics['macro_f1']:.4f}")
    print(f"안전 규칙 보정: {override_count}건 / 분류 실패 fallback: {fallback_count}건")
    print("\n경로별 지표")
    print(f"{'경로':<10}{'Precision':>12}{'Recall':>12}{'F1':>12}{'Support':>12}")
    for route in ROUTE_ORDER:
        item = metrics["per_route"][route]
        print(
            f"{route:<10}{item['precision']:>12.3f}{item['recall']:>12.3f}"
            f"{item['f1']:>12.3f}{item['support']:>12d}"
        )

    print("\n혼동 행렬 (행=정답, 열=예측)")
    print(f"{'':<10}" + "".join(f"{route:>10}" for route in ROUTE_ORDER))
    for actual in ROUTE_ORDER:
        print(f"{actual:<10}" + "".join(f"{matrix[actual][predicted]:>10d}" for predicted in ROUTE_ORDER))

    if errors:
        print("\n오분류 사례")
        for error in errors:
            print(
                f"- {error['id']}: {error['expected']} → {error['predicted']} "
                f"(모델 원판정={error['model_route']}, 확신도={error['confidence']:.2f})"
            )
            print(f"  질문: {error['question']}")
    else:
        print("\n오분류가 없습니다.")

    report = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "split": args.split,
        "raw_model_metrics": raw_metrics,
        "final_metrics": metrics,
        "confusion_matrix": matrix,
        "override_count": override_count,
        "fallback_count": fallback_count,
        "errors": errors,
        "cases": evaluated,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f"routing_eval_{args.split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n상세 결과 저장: {report_path}")


if __name__ == "__main__":
    main()
