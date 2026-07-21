import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_POLICY = "configs/fair_input_policy.json"


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                yield {
                    "_audit_error": "json_decode_error",
                    "_path": str(path),
                    "_line": line_no,
                    "_message": str(exc),
                }
                continue
            row["_path"] = str(path)
            row["_line"] = line_no
            yield row


def discover_files(paths: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            out.extend(sorted(path.rglob("*.jsonl")))
            out.extend(sorted(path.rglob("summary*.json")))
        elif path.is_file():
            out.append(path)
    return out


def method_name(row: Dict[str, Any]) -> str:
    method = str(row.get("method") or "").strip()
    if method.startswith("visual_claim_proposer_caption_v"):
        return "visual_claim_proposer"
    if method:
        return method
    if "claim_bridge" in row:
        return "claim_bridge_from_mc"
    if "token_prefix_constraint" in row:
        return "token_prefix_from_mc"
    mitigation = str(row.get("effective_mitigation") or row.get("mitigation") or "").strip()
    if mitigation == "cp_vbc":
        cp = row.get("cp_vbc") or {}
        mode = str(row.get("cp_vbc_mode") or cp.get("mode") or "").strip()
        return f"cp_vbc_{mode}" if mode else "cp_vbc"
    if mitigation:
        return mitigation
    return "unknown"


def infer_status(row: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
    if row.get("_audit_error"):
        return {"status": "audit_error", "reason": row.get("_message")}

    explicit = row.get("fairness")
    if isinstance(explicit, dict) and explicit.get("status"):
        return {
            "status": str(explicit.get("status")),
            "reason": str(explicit.get("reason") or ""),
            "source": "explicit_row_metadata",
        }

    method = method_name(row)
    task = str(row.get("task") or "").strip()
    known = (policy.get("known_method_status") or {}).get(method)
    if isinstance(known, dict):
        if task == "summary":
            raw_tasks = row.get("tasks")
            if isinstance(raw_tasks, str):
                task_list = [part.strip() for part in raw_tasks.split(",") if part.strip()]
            elif isinstance(raw_tasks, list):
                task_list = [str(part).strip() for part in raw_tasks if str(part).strip()]
            else:
                task_list = []
            task_statuses = [known.get(t) for t in task_list if known.get(t)]
            if task_statuses and all(status == "fair" for status in task_statuses):
                return {"status": "fair", "reason": "all configured tasks are fair", "source": "policy_known_method_status"}
            if task_statuses:
                unique = sorted(set(str(status) for status in task_statuses))
                return {
                    "status": "mixed_task_status",
                    "reason": ",".join(unique),
                    "source": "policy_known_method_status",
                }
        value = known.get(task) or known.get("caption") or known.get("mc") or known.get("qa")
        reason = known.get("reason") or ""
        if value:
            return {"status": str(value), "reason": str(reason), "source": "policy_known_method_status"}

    if method in {"claim_bridge_from_mc", "token_prefix_from_mc"}:
        return {
            "status": "diagnostic_upper_bound_not_fair",
            "reason": "uses MC-derived candidate information for caption",
            "source": "structural_inference",
        }

    if method == "unknown":
        return {
            "status": "unmarked_unknown",
            "reason": "no fairness metadata and no recognizable method marker",
            "source": "audit_default",
        }

    return {
        "status": "unmarked_assume_task_native_until_review",
        "reason": "no explicit fairness metadata; manual review recommended before paper reporting",
        "source": "audit_default",
    }


def audit_file(path: Path, policy: Dict[str, Any]) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = []
        for row in iter_jsonl(path):
            status = infer_status(row, policy)
            rows.append({
                "path": row.get("_path", str(path)),
                "line": row.get("_line"),
                "pair_id": row.get("pair_id"),
                "task": row.get("task"),
                "side": row.get("side"),
                "method": method_name(row),
                **status,
            })
        return rows

    try:
        payload = read_json(path)
    except Exception as exc:
        return [{
            "path": str(path),
            "line": None,
            "method": "unknown",
            "status": "audit_error",
            "reason": str(exc),
            "source": "summary_json",
        }]

    if method_name(payload) == "unknown" and path.name.startswith("summary"):
        run_config_path = path.parent / "run_config.json"
        if run_config_path.exists():
            try:
                run_config = read_json(run_config_path)
                payload = {**payload, **run_config}
            except Exception:
                pass
    if path.name.startswith("summary") and not payload.get("task"):
        payload["task"] = "summary"

    status = infer_status(payload, policy)
    return [{
        "path": str(path),
        "line": None,
        "task": "summary",
        "method": method_name(payload),
        **status,
    }]


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit CDH result files against the fair input policy.")
    parser.add_argument("paths", nargs="+", help="Result JSONL files, summary JSON files, or directories to audit.")
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--out", default="")
    parser.add_argument("--show-examples", type=int, default=20)
    args = parser.parse_args()

    policy = read_json(Path(args.policy))
    files = discover_files(args.paths)
    rows: List[Dict[str, Any]] = []
    for path in files:
        rows.extend(audit_file(path, policy))

    status_counts = Counter(str(row.get("status") or "unknown") for row in rows)
    method_counts: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        method_counts[str(row.get("method") or "unknown")][str(row.get("status") or "unknown")] += 1

    needs_review = [
        row for row in rows
        if str(row.get("status") or "").startswith("unmarked")
        or "not_fair" in str(row.get("status") or "")
        or row.get("status") == "audit_error"
    ]
    report = {
        "policy": args.policy,
        "files_scanned": [str(path) for path in files],
        "rows_scanned": len(rows),
        "status_counts": dict(status_counts),
        "method_counts": {method: dict(counts) for method, counts in sorted(method_counts.items())},
        "needs_review_count": len(needs_review),
        "needs_review_examples": needs_review[: max(0, args.show_examples)],
    }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
