#!/usr/bin/env python3
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from cprc_vlm_runtime import extract_first_letter, extract_yes_no


METHOD_CONFIG_KEYS = {
    "vcd": "vcd_candidate",
    "pai": "pai_candidate",
    "mfcd": "mfcd_candidate",
    "nolan": "nolan_candidate",
    "revis": "revis_candidate",
}


def _candidate_key_from_prediction(task: str, prediction: str) -> Optional[str]:
    if task == "qa":
        return extract_yes_no(prediction)
    if task == "mc":
        return extract_first_letter(prediction)
    raise ValueError(f"unknown task: {task}")


def _prediction_from_candidate_key(task: str, key: str) -> str:
    if task == "qa":
        return f"Final answer: {str(key).lower()}"
    if task == "mc":
        return f"Final answer: {str(key).upper()}"
    raise ValueError(f"unknown task: {task}")


def _score(task: str, prediction: str, ground_truth: str) -> bool:
    key = _candidate_key_from_prediction(task, prediction)
    expected = str(ground_truth).strip().lower() if task == "qa" else str(ground_truth).strip().upper()
    return key == expected


def _vcd_candidate_score(logp_image: float, logp_degraded: float, alpha: float) -> float:
    return float((1.0 + float(alpha)) * float(logp_image) - float(alpha) * float(logp_degraded))


def _mfcd_candidate_score(
    logp_image: float,
    logp_low_freq: float,
    logp_high_freq: float,
    alpha_low: float,
    alpha_high: float,
) -> float:
    return float(
        (1.0 + float(alpha_low) + float(alpha_high)) * float(logp_image)
        - float(alpha_low) * float(logp_low_freq)
        - float(alpha_high) * float(logp_high_freq)
    )


def _pai_candidate_score(
    logp_image: float, logp_no_image: float, guidance_scale: float
) -> float:
    return float(
        float(guidance_scale) * float(logp_image)
        - (float(guidance_scale) - 1.0) * float(logp_no_image)
    )


def _candidate_plausibility_mask(
    rows: List[Dict[str, Any]], logp_key: str, beta: float
) -> Dict[str, bool]:
    if not rows:
        return {}
    beta = float(beta)
    if beta <= 0.0:
        return {str(row["key"]): True for row in rows}
    max_logp = max(float(row[logp_key]) for row in rows)
    log_beta = math.log(min(beta, 1.0))
    return {
        str(row["key"]): float(row[logp_key]) >= max_logp + log_beta
        for row in rows
    }


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _method_block(record: Dict[str, Any], method: str) -> Optional[Dict[str, Any]]:
    block = record.get(method)
    if isinstance(block, dict):
        return block
    shared_views = record.get("view_collection")
    if method in {"vcd", "mfcd"} and isinstance(shared_views, dict):
        return shared_views
    if method == "vcd":
        shared = record.get("mfcd")
        if not isinstance(shared, dict):
            return None
        candidates = []
        for source in shared.get("candidates") or []:
            if "logp_image" not in source or "logp_low_freq" not in source:
                return None
            candidate = dict(source)
            candidate["logp_degraded"] = source["logp_low_freq"]
            candidates.append(candidate)
        if not candidates:
            return None
        return {
            "baseline_key": shared.get("baseline_key"),
            "baseline_pred": shared.get("baseline_pred"),
            "candidates": candidates,
            "degrade_mode": shared.get("low_mode"),
            "score_source": "shared_mfcd_image_low_frequency_candidates",
        }
    if method not in {"pai", "nolan"}:
        return None
    shared = record.get("cp_vbc")
    if not isinstance(shared, dict):
        return None
    candidates = []
    for source in shared.get("candidates") or []:
        if "logp_image" not in source or "logp_prior" not in source:
            return None
        candidate = dict(source)
        candidate["logp_no_image"] = source["logp_prior"]
        candidates.append(candidate)
    if not candidates:
        return None
    return {
        "baseline_key": shared.get("baseline_key"),
        "baseline_pred": shared.get("baseline_pred"),
        "candidates": candidates,
        "score_source": "shared_image_no_image_candidates",
    }


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _margin(rows: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    if len(rows) < 2:
        return None
    return float(rows[0][key]) - float(rows[1][key])


def _top(rows: Sequence[Dict[str, Any]], key: str) -> Tuple[Optional[str], Optional[float]]:
    ranked = sorted(rows, key=lambda row: float(row[key]), reverse=True)
    return (str(ranked[0]["key"]), _margin(ranked, key)) if ranked else (None, None)


def _plausible_rows(rows: List[Dict[str, Any]], beta: float) -> List[Dict[str, Any]]:
    mask = _candidate_plausibility_mask(rows, "logp_image", beta)
    return [row for row in rows if mask[str(row["key"])]] or rows


def _replay_vcd(block: Dict[str, Any], params: Dict[str, Any]) -> Optional[str]:
    degrade_mode = str(params.get("degrade_mode"))
    rows = [dict(row) for row in block["candidates"]]
    if rows and isinstance(rows[0].get("views"), dict):
        if any(degrade_mode not in row["views"] for row in rows):
            return None
        for row in rows:
            row["logp_degraded"] = row["views"][degrade_mode]["logprob"]
    elif degrade_mode != str(block.get("degrade_mode")):
        return None
    for row in rows:
        row["score"] = _vcd_candidate_score(
            row["logp_image"], row["logp_degraded"], params["alpha"]
        )
    image_key, image_margin = _top(rows, "logp_image")
    degraded_key, _ = _top(rows, "logp_degraded")
    contrast_key, contrast_margin = _top(_plausible_rows(rows, params["beta"]), "score")
    enough = contrast_margin is None or contrast_margin >= float(params["contrast_margin"])
    gate = str(params["gate"])
    active = gate == "always" or (
        gate == "visual_conflict"
        and image_key != degraded_key
        and contrast_key == image_key
        and (image_margin is None or image_margin >= 0.0)
    )
    baseline_key = block.get("baseline_key")
    return str(contrast_key) if active and enough and contrast_key != baseline_key else baseline_key


def _replay_pai(block: Dict[str, Any], params: Dict[str, Any]) -> Optional[str]:
    rows = [dict(row) for row in block["candidates"]]
    for row in rows:
        row["score"] = _pai_candidate_score(
            row["logp_image"], row["logp_no_image"], params["gamma"]
        )
    contrast_key, contrast_margin = _top(_plausible_rows(rows, params["beta"]), "score")
    enough = contrast_margin is None or contrast_margin >= float(params.get("contrast_margin", 0.0))
    baseline_key = block.get("baseline_key")
    return str(contrast_key) if enough and contrast_key != baseline_key else baseline_key


def _softmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    maximum = max(float(value) for value in values)
    weights = [math.exp(float(value) - maximum) for value in values]
    normalizer = sum(weights)
    return [weight / normalizer for weight in weights]


def _symmetric_kl(first: Sequence[float], second: Sequence[float]) -> float:
    epsilon = 1e-12
    return 0.5 * sum(
        float(p) * math.log(max(float(p), epsilon) / max(float(q), epsilon))
        + float(q) * math.log(max(float(q), epsilon) / max(float(p), epsilon))
        for p, q in zip(first, second)
    )


def _replay_nolan(block: Dict[str, Any], params: Dict[str, Any]) -> Optional[str]:
    """Replay NoLan's released dynamic score rule in the native candidate space."""
    rows = [dict(row) for row in block["candidates"]]
    image_distribution = _softmax([float(row["logp_image"]) for row in rows])
    no_image_distribution = _softmax([float(row["logp_no_image"]) for row in rows])
    divergence = _symmetric_kl(image_distribution, no_image_distribution)
    inverse_divergence = 1.0 / max(divergence, 1e-12)
    alpha = float(params["beta"]) * (math.tanh(inverse_divergence) + 1.0)
    for row in rows:
        row["score"] = float(
            (1.0 + alpha) * float(row["logp_image"])
            - alpha * float(row["logp_no_image"])
        )
    contrast_key, _ = _top(rows, "score")
    return contrast_key


def _replay_mfcd(block: Dict[str, Any], params: Dict[str, Any]) -> Optional[str]:
    low_mode = str(params.get("low_mode"))
    high_mode = str(params.get("high_mode"))
    rows = [dict(row) for row in block["candidates"]]
    if rows and isinstance(rows[0].get("views"), dict):
        if any(
            low_mode not in row["views"] or high_mode not in row["views"]
            for row in rows
        ):
            return None
        for row in rows:
            row["logp_low_freq"] = row["views"][low_mode]["logprob"]
            row["logp_high_freq"] = row["views"][high_mode]["logprob"]
    else:
        if low_mode != str(block.get("low_mode")):
            return None
        if high_mode != str(block.get("high_mode")):
            return None
    for row in rows:
        row["score"] = _mfcd_candidate_score(
            row["logp_image"],
            row["logp_low_freq"],
            row["logp_high_freq"],
            params["alpha_low"],
            params["alpha_high"],
        )
    image_key, image_margin = _top(rows, "logp_image")
    low_key, _ = _top(rows, "logp_low_freq")
    high_key, high_margin = _top(rows, "logp_high_freq")
    contrast_key, contrast_margin = _top(_plausible_rows(rows, params["beta"]), "score")
    enough = contrast_margin is None or contrast_margin >= float(params["contrast_margin"])
    baseline_key = block.get("baseline_key")
    visual_conflict = image_key != low_key and contrast_key == image_key and (
        image_margin is None or image_margin >= 0.0
    )
    frequency_rescue = (
        baseline_key == image_key == low_key
        and high_key != baseline_key
        and contrast_key == high_key
        and (high_margin is None or high_margin >= 0.0)
    )
    gate = str(params["gate"])
    active = gate == "always" or (gate == "visual_conflict" and visual_conflict) or (
        gate == "frequency_conflict" and (visual_conflict or frequency_rescue)
    )
    return str(contrast_key) if active and enough and contrast_key != baseline_key else baseline_key


def _replay_revis(block: Dict[str, Any], params: Dict[str, Any]) -> Optional[str]:
    rows = [dict(row) for row in block["candidates"]]
    for row in rows:
        row["score"] = float(
            row["logp_image"]
            - float(params["lambda_prior"]) * row["logp_prior"]
            + float(params["lambda_hidden"]) * row["hidden_gain"]
        )
    image_key, image_margin = _top(rows, "logp_image")
    prior_key, prior_margin = _top(rows, "logp_prior")
    hidden_key, hidden_margin = _top(rows, "hidden_gain")
    contrast_key, contrast_margin = _top(rows, "score")
    baseline_key = block.get("baseline_key")
    high_prior = prior_margin is None or prior_margin >= 0.0
    enough_contrast = contrast_margin is None or contrast_margin >= float(params["contrast_margin"])
    enough_absmax = float(block.get("prompt_absmax_relative_delta") or 0.0) >= float(params["absmax_min"])
    visual_conflict = image_key != prior_key and contrast_key == image_key and (
        image_margin is None or image_margin >= float(block.get("visual_conflict_image_margin", 0.5))
    )
    prior_absorption = (
        baseline_key == prior_key == image_key
        and contrast_key != baseline_key
        and high_prior
        and (image_margin is None or image_margin <= float(block.get("absorption_image_margin_max", 10.0)))
    )
    hidden_rescue = hidden_key == contrast_key and hidden_key != baseline_key and high_prior and (
        hidden_margin is None or hidden_margin >= float(block.get("hidden_margin_threshold", 0.0))
    )
    active = enough_absmax and enough_contrast and (visual_conflict or prior_absorption or hidden_rescue)
    return str(contrast_key) if active and contrast_key != baseline_key else baseline_key


REPLAY = {
    "vcd": _replay_vcd,
    "pai": _replay_pai,
    "mfcd": _replay_mfcd,
    "nolan": _replay_nolan,
    "revis": _replay_revis,
}


def _grid(config: Dict[str, Any], method: str) -> List[Dict[str, Any]]:
    values = config["baselines"][METHOD_CONFIG_KEYS[method]]["grid"]
    keys = sorted(values)
    return [dict(zip(keys, combination)) for combination in itertools.product(*(values[key] for key in keys))]


def _correct(task: str, key: Optional[str], gt: str) -> bool:
    if key is None:
        return False
    return _score(task, _prediction_from_candidate_key(task, str(key)), gt)


def _evaluate(records: List[Dict[str, Any]], method: str, task: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    counts = {
        "counterfactual": {"n": 0, "baseline": 0, "method": 0},
        "commonsense": {"n": 0, "baseline": 0, "method": 0},
    }
    overrides = 0
    for record in records:
        if record.get("status") != "ok" or record.get("task") != task:
            continue
        block = _method_block(record, method)
        if block is None:
            continue
        final_key = REPLAY[method](block, params)
        if final_key is None:
            continue
        side = str(record["side"])
        baseline_key = block.get("baseline_key") or _candidate_key_from_prediction(task, block.get("baseline_pred", ""))
        counts[side]["n"] += 1
        counts[side]["baseline"] += int(_correct(task, baseline_key, str(record["gt"])))
        counts[side]["method"] += int(_correct(task, final_key, str(record["gt"])))
        overrides += int(final_key != baseline_key)
    if not counts["counterfactual"]["n"] or not counts["commonsense"]["n"]:
        return None
    cf = counts["counterfactual"]
    cs = counts["commonsense"]
    baseline_cf = cf["baseline"] / cf["n"]
    method_cf = cf["method"] / cf["n"]
    baseline_cs = cs["baseline"] / cs["n"]
    method_cs = cs["method"] / cs["n"]
    delta_cf = method_cf - baseline_cf
    delta_cs = method_cs - baseline_cs
    return {
        "params": params,
        "n_cf": cf["n"],
        "n_cs": cs["n"],
        "baseline_cf": baseline_cf,
        "method_cf": method_cf,
        "delta_cf": delta_cf,
        "baseline_cs": baseline_cs,
        "method_cs": method_cs,
        "delta_cs": delta_cs,
        "utility_0_5": delta_cf + 0.5 * delta_cs,
        "overrides": overrides,
    }


def _pareto(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    frontier = []
    for row in rows:
        dominated = any(
            other["delta_cf"] >= row["delta_cf"]
            and other["delta_cs"] >= row["delta_cs"]
            and (other["delta_cf"] > row["delta_cf"] or other["delta_cs"] > row["delta_cs"])
            for other in rows
        )
        if not dominated:
            frontier.append(row)
    return sorted(frontier, key=lambda row: (row["delta_cf"], row["delta_cs"], str(row["params"])))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--method", choices=sorted(REPLAY), required=True)
    parser.add_argument("--task", choices=("qa", "mc"), required=True)
    parser.add_argument("--protocol", default="configs/strong_baselines_frozen_v1.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-cs-drop", type=float, default=0.2)
    args = parser.parse_args()

    input_paths = list(args.input)
    records = [record for path in input_paths for record in _read_jsonl(path)]
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    results = [
        result
        for params in _grid(protocol, args.method)
        for result in [_evaluate(records, args.method, args.task, params)]
        if result is not None
    ]
    eligible = [row for row in results if row["delta_cs"] >= -float(args.max_cs_drop)]
    if not eligible:
        raise SystemExit("no parameter setting satisfies the development CS constraint")
    selected = sorted(
        eligible,
        key=lambda row: (
            -row["utility_0_5"],
            -row["delta_cf"],
            -row["delta_cs"],
            row["overrides"],
            json.dumps(row["params"], sort_keys=True),
        ),
    )[0]
    payload = {
        "version": "candidate_baseline_development_sweep_v1",
        "selection_split": "development_only",
        "method": args.method,
        "task": args.task,
        "input": input_paths[0] if len(input_paths) == 1 else None,
        "inputs": input_paths,
        "input_sha256": _sha256(input_paths[0]) if len(input_paths) == 1 else None,
        "input_sha256s": {path: _sha256(path) for path in input_paths},
        "protocol": args.protocol,
        "selection_objective": "delta_CF + 0.5 * delta_CS",
        "max_cs_drop": args.max_cs_drop,
        "evaluated_settings": len(results),
        "eligible_settings": len(eligible),
        "selected": selected,
        "pareto_frontier": _pareto(results),
        "all_results": sorted(results, key=lambda row: str(row["params"])),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "selected": selected}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
