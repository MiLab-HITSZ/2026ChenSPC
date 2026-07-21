import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set


def load_pair_ids(path: Path, split: str = "") -> Set[str]:
    raw = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {line.strip() for line in raw.splitlines() if line.strip()}
    if isinstance(payload, list):
        return {str(value) for value in payload}
    if split:
        payload = payload["split_manifests"][split]
    return {str(value) for value in payload["pair_ids"]}


def filter_rows(rows: List[Dict[str, Any]], pair_ids: Set[str]) -> List[Dict[str, Any]]:
    return [row for row in rows if str(row.get("pair_id") or "") in pair_ids]


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter JSONL results by a pair manifest.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--pair-split", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pair_ids = load_pair_ids(Path(args.pair_manifest), args.pair_split)
    output = filter_rows(rows, pair_ids)
    seen = {str(row.get("pair_id") or "") for row in output}
    missing = pair_ids - seen
    if missing:
        raise ValueError(f"manifest pairs absent from input: {sorted(missing)[:10]}")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output),
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(output), "pairs": len(seen), "output": str(destination)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
