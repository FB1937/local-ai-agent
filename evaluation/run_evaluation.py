"""Run deterministic M2 evaluation cases without external services."""
import argparse
import json
from collections import defaultdict
from answer_validator import validate_answer
from evaluation.cases import build_cases


def run_evaluation():
    cases = build_cases()
    rows = []
    groups = defaultdict(lambda: {"passed": 0, "total": 0})
    for case in cases:
        report = validate_answer(case["answer"], case["sources"],
                                 search_assessment=case["assessment"],
                                 route=case["route"])
        actual = bool(report["valid"])
        passed = actual == case["expected_valid"]
        group = case["category"]
        groups[group]["total"] += 1
        groups[group]["passed"] += int(passed)
        rows.append({"id": case["id"], "category": group,
                     "expected_valid": case["expected_valid"],
                     "actual_valid": actual, "passed": passed,
                     "unknown_ids": report.get("unknown_ids", []),
                     "missing_url_ids": report.get("missing_url_ids", [])})
    passed = sum(row["passed"] for row in rows)
    return {"total": len(rows), "passed": passed, "failed": len(rows)-passed,
            "pass_rate": passed/len(rows) if rows else 0.0,
            "by_category": {k: {**v, "pass_rate": v["passed"]/v["total"]}
                            for k, v in sorted(groups.items())},
            "cases": rows,
            "scope_note": "합성 결정적 회귀 데이터이며 의미적 정답률이나 실제 Ollama 품질을 측정하지 않습니다."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run_evaluation()
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"M2 evaluation: {result['passed']}/{result['total']} passed ({result['pass_rate']:.1%})")
        for category, metric in result["by_category"].items():
            print(f"- {category}: {metric['passed']}/{metric['total']} ({metric['pass_rate']:.1%})")
        for row in result["cases"]:
            if not row["passed"]:
                print(f"FAIL {row['id']}: expected={row['expected_valid']} actual={row['actual_valid']}")
        print(result["scope_note"])


if __name__ == "__main__":
    main()
