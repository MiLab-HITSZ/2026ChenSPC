import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from analyze_cp_vbc_bayes_path_cv import normalize_key
from analyze_hierarchical_eb_lambda_cv import exact_mcnemar_pvalue


def record_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(row.get("task") or ""),
        str(row.get("pair_id") or ""),
        str(row.get("side") or ""),
    )


def summarize(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    values = list(rows)
    baseline_correct = sum(int(row["baseline_correct"]) for row in values)
    method_correct = sum(int(row["method_correct"]) for row in values)
    repairs = sum(
        int(not row["baseline_correct"] and row["method_correct"]) for row in values
    )
    harms = sum(
        int(row["baseline_correct"] and not row["method_correct"]) for row in values
    )
    count = len(values)
    return {
        "n": count,
        "baseline_acc": baseline_correct / count if count else None,
        "method_acc": method_correct / count if count else None,
        "delta": (method_correct - baseline_correct) / count if count else None,
        "repairs": repairs,
        "harms": harms,
        "exact_mcnemar_p_two_sided": exact_mcnemar_pvalue(repairs, harms),
    }


def grouped(rows: List[Dict[str, Any]], field: str) -> Dict[str, Any]:
    buckets = defaultdict(list)
    for row in rows:
        buckets[str(row.get(field) or "Unknown")].append(row)
    return {key: summarize(values) for key, values in sorted(buckets.items())}


def analyze(results_path: Path, heb_path: Path) -> Dict[str, Any]:
    raw_rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    heb = json.loads(heb_path.read_text(encoding="utf-8"))
    cases = {record_key(case): case for case in heb["result"]["cases"]}
    rows = []
    for raw in raw_rows:
        key = record_key(raw)
        baseline_correct = normalize_key(raw["cp_vbc"].get("baseline_key")) == normalize_key(
            raw.get("gt")
        )
        case = cases.get(key)
        method_correct = bool(case["prediction_correct"]) if case else baseline_correct
        rows.append(
            {
                "task": raw.get("task"),
                "side": raw.get("side"),
                "category": raw.get("category"),
                "subcategory": raw.get("subcategory"),
                "pair_id": raw.get("pair_id"),
                "baseline_correct": baseline_correct,
                "method_correct": method_correct,
            }
        )

    by_task_side = {}
    for task in sorted({str(row["task"]) for row in rows}):
        by_task_side[task] = {}
        for side in ("counterfactual", "commonsense"):
            selected = [row for row in rows if row["task"] == task and row["side"] == side]
            by_task_side[task][side] = {
                "overall": summarize(selected),
                "by_category": grouped(selected, "category"),
                "by_subcategory": grouped(selected, "subcategory"),
            }
    return {
        "version": "spc_stratified_analysis_v1",
        "results": str(results_path),
        "heb_result": str(heb_path),
        "rows": len(rows),
        "by_task_side": by_task_side,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stratify a frozen H-EB SPC result.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--heb", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = analyze(Path(args.results), Path(args.heb))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("task\tside\tcategory\tn\tbaseline\tmethod\tdelta\trepairs\tharms")
    for task, sides in output["by_task_side"].items():
        for side, values in sides.items():
            for category, metric in values["by_category"].items():
                print(
                    "\t".join(
                        (
                            task,
                            side,
                            category,
                            str(metric["n"]),
                            f"{metric['baseline_acc']:.3f}",
                            f"{metric['method_acc']:.3f}",
                            f"{metric['delta']:+.3f}",
                            str(metric["repairs"]),
                            str(metric["harms"]),
                        )
                    )
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
