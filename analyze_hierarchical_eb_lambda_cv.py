import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from analyze_cp_vbc_bayes_path_cv import (
    SIDES,
    fold_ids,
    metric,
    normalize_key,
    read_jsonl,
)


VERSION = "empirical_bayes_prior_residual_calibration_v2"


@dataclass(frozen=True)
class InputSpec:
    label: str
    task: str
    path: Path


def resolved_baseline_key(row: Dict[str, Any]) -> str:
    cp_vbc = row.get("cp_vbc") or {}
    identity = " ".join(
        str(value or "")
        for value in (
            row.get("model_name"),
            row.get("model"),
            cp_vbc.get("model"),
        )
    ).lower()
    baseline_pred = str(cp_vbc.get("baseline_pred") or "")
    if "thinking" in identity and "</think>" not in baseline_pred:
        return ""
    return normalize_key(cp_vbc.get("baseline_key"))


def parse_input(raw: str) -> InputSpec:
    if "=" not in raw or ":" not in raw.split("=", 1)[0]:
        raise argparse.ArgumentTypeError("--input must be LABEL:TASK=RESULTS_JSONL")
    left, path = raw.split("=", 1)
    label, task = left.rsplit(":", 1)
    label = label.strip()
    task = task.strip().lower()
    if not label or task not in {"qa", "mc"}:
        raise argparse.ArgumentTypeError("input label must be non-empty and TASK must be qa or mc")
    return InputSpec(label=label, task=task, path=Path(path).expanduser())


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - float(np.max(values))
    output = np.exp(shifted)
    return output / max(1e-12, float(output.sum()))


def normalized_entropy(probabilities: np.ndarray) -> float:
    if len(probabilities) <= 1:
        return 0.0
    value = -float(
        np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)))
    )
    return value / math.log(len(probabilities))


def top_margin(values: np.ndarray) -> float:
    if len(values) <= 1:
        return 0.0
    ordered = np.sort(values)
    return float(ordered[-1] - ordered[-2])


def candidate_arrays(row: Dict[str, Any]) -> Tuple[List[str], np.ndarray, np.ndarray]:
    candidates = row["cp_vbc"].get("candidates") or []
    keys = [normalize_key(candidate["key"]) for candidate in candidates]
    image = np.asarray([float(candidate["logp_image"]) for candidate in candidates])
    prior = np.asarray([float(candidate["logp_prior"]) for candidate in candidates])
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("candidate keys must be non-empty and unique")
    return keys, image - image.mean(), prior - prior.mean()


def context_features(
    row: Dict[str, Any], feature_mode: str = "full"
) -> np.ndarray:
    keys, image, prior = candidate_arrays(row)
    image_prob = softmax(image)
    prior_prob = softmax(prior)
    image_top = int(np.argmax(image))
    prior_top = int(np.argmax(prior))
    baseline = resolved_baseline_key(row)
    baseline_index = keys.index(baseline) if baseline in keys else image_top
    mixture = 0.5 * (image_prob + prior_prob)
    js_divergence = 0.5 * float(
        np.sum(image_prob * np.log(np.maximum(image_prob, 1e-12) / mixture))
        + np.sum(prior_prob * np.log(np.maximum(prior_prob, 1e-12) / mixture))
    )
    symmetric_kl = 0.5 * float(
        np.sum(
            image_prob
            * np.log(np.maximum(image_prob, 1e-12) / np.maximum(prior_prob, 1e-12))
        )
        + np.sum(
            prior_prob
            * np.log(np.maximum(prior_prob, 1e-12) / np.maximum(image_prob, 1e-12))
        )
    )
    image_margin = top_margin(image)
    prior_margin = top_margin(prior)
    features = np.asarray(
        [
            1.0,
            math.log(len(keys)) / math.log(4.0),
            normalized_entropy(image_prob),
            normalized_entropy(prior_prob),
            float(image_prob[image_top]),
            float(prior_prob[prior_top]),
            math.tanh(image_margin / 4.0),
            math.tanh(prior_margin / 4.0),
            js_divergence,
            float(image_top == prior_top),
            float(baseline_index == image_top),
            float(baseline_index == prior_top),
            math.tanh((prior_margin - image_margin) / 4.0),
        ],
        dtype=np.float64,
    )
    if feature_mode == "full":
        return features
    if feature_mode == "nolan_overlap":
        overlap = math.tanh(1.0 / max(symmetric_kl, 1e-12))
        return np.concatenate([features, np.asarray([overlap], dtype=np.float64)])
    if feature_mode != "image_only":
        raise ValueError(f"unknown feature mode: {feature_mode}")

    # Preserve the exact dimensionality and fitting budget of the full model while
    # removing every statistic that depends on the no-image distribution.
    return np.asarray(
        [
            1.0,
            features[1],
            features[2],
            0.0,
            features[4],
            0.0,
            features[6],
            0.0,
            0.0,
            0.0,
            features[10],
            0.0,
            -features[6],
        ],
        dtype=np.float64,
    )


def standardize_contexts(
    contexts: np.ndarray, train_ids: Sequence[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = contexts[list(train_ids)]
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    mean[0] = 0.0
    scale[0] = 1.0
    scale[scale < 1e-6] = 1.0
    return (contexts - mean) / scale, mean, scale


def conditional_loss(
    theta: torch.Tensor,
    rows: Sequence[Dict[str, Any]],
    contexts: np.ndarray,
    ids: Sequence[int],
    l2: float,
    latent_offsets: Optional[np.ndarray] = None,
) -> torch.Tensor:
    losses = []
    for index in ids:
        keys, image, prior = candidate_arrays(rows[index])
        gt = normalize_key(rows[index].get("gt"))
        if gt not in keys:
            raise ValueError(f"ground truth {gt!r} is not a native candidate")
        x = torch.as_tensor(contexts[index], dtype=torch.float64)
        image_tensor = torch.as_tensor(image, dtype=torch.float64)
        prior_tensor = torch.as_tensor(prior, dtype=torch.float64)
        offset = (
            float(latent_offsets[index]) if latent_offsets is not None else 0.0
        )
        lambda_value = torch.nn.functional.softplus(x @ theta + offset)
        scores = image_tensor - lambda_value * prior_tensor
        target = keys.index(gt)
        losses.append(torch.logsumexp(scores, dim=0) - scores[target])
    return torch.stack(losses).sum() + 0.5 * float(l2) * (theta @ theta)


def fit_lambda_posterior(
    rows: Sequence[Dict[str, Any]],
    contexts: np.ndarray,
    train_ids: Sequence[int],
    l2: float,
    latent_offsets: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    theta = torch.zeros(contexts.shape[1], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [theta], max_iter=120, tolerance_grad=1e-9, tolerance_change=1e-11
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = conditional_loss(
            theta, rows, contexts, train_ids, l2, latent_offsets=latent_offsets
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    fitted = theta.detach().clone().requires_grad_(True)
    final_loss = conditional_loss(
        fitted, rows, contexts, train_ids, l2, latent_offsets=latent_offsets
    )
    hessian = torch.autograd.functional.hessian(
        lambda value: conditional_loss(
            value,
            rows,
            contexts,
            train_ids,
            l2,
            latent_offsets=latent_offsets,
        ),
        fitted,
    )
    hessian_np = hessian.detach().numpy()
    hessian_np = 0.5 * (hessian_np + hessian_np.T)
    covariance = np.linalg.pinv(hessian_np + 1e-8 * np.eye(len(fitted)), rcond=1e-9)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(eigenvalues, 1e-8, 4.0)
    covariance = (eigenvectors * eigenvalues) @ eigenvectors.T
    return fitted.detach().numpy(), covariance, float(final_loss.detach())


def sigmoid_softplus(values: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, values)


def inverse_softplus(value: float) -> float:
    return math.log(math.expm1(float(value)))


def normal_cdf(value: float) -> float:
    if value == float("inf"):
        return 1.0
    if value == float("-inf"):
        return 0.0
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def exact_top_support(
    proposal_index: int,
    image: np.ndarray,
    prior: np.ndarray,
    lambda_cap: float,
    latent_mean: float,
    latent_std: float,
) -> float:
    lower = 0.0
    upper = float("inf")
    for index in range(len(image)):
        if index == proposal_index:
            continue
        image_delta = float(image[proposal_index] - image[index])
        prior_delta = float(prior[proposal_index] - prior[index])
        if abs(prior_delta) < 1e-12:
            if image_delta < 0.0:
                return 0.0
            continue
        boundary = image_delta / prior_delta
        if prior_delta > 0.0:
            upper = min(upper, boundary)
        else:
            lower = max(lower, boundary)

    lower = max(0.0, lower)
    cap = float(lambda_cap)
    if upper < lower - 1e-12 or upper <= 0.0 or lower > cap + 1e-12:
        return 0.0
    if latent_std < 1e-10:
        value = min(float(np.logaddexp(0.0, latent_mean)), cap)
        return float(lower - 1e-12 <= value <= upper + 1e-12)

    if lower <= 0.0:
        latent_lower = float("-inf")
    elif lower >= cap - 1e-12:
        latent_lower = inverse_softplus(cap)
    else:
        latent_lower = inverse_softplus(lower)

    if upper >= cap - 1e-12:
        latent_upper = float("inf")
    else:
        latent_upper = inverse_softplus(upper)
    standardized_lower = (latent_lower - latent_mean) / latent_std
    standardized_upper = (latent_upper - latent_mean) / latent_std
    return max(0.0, normal_cdf(standardized_upper) - normal_cdf(standardized_lower))


def policy_details_gauss_hermite(
    rows: Sequence[Dict[str, Any]],
    contexts: np.ndarray,
    theta_mean: np.ndarray,
    theta_covariance: np.ndarray,
    lambda_cap: float = 1.5,
    quadrature_points: int = 32,
    support_distances: Optional[np.ndarray] = None,
    arity_supported: Optional[np.ndarray] = None,
    latent_offsets: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    nodes, weights = np.polynomial.hermite.hermgauss(max(8, int(quadrature_points)))
    weights = weights / math.sqrt(math.pi)
    details = []
    for index, (row, context) in enumerate(zip(rows, contexts)):
        keys, image, prior = candidate_arrays(row)
        latent_mean = float(context @ theta_mean) + (
            float(latent_offsets[index]) if latent_offsets is not None else 0.0
        )
        latent_variance = float(context @ theta_covariance @ context)
        latent_std = math.sqrt(max(1e-12, latent_variance))
        latent = latent_mean + math.sqrt(2.0) * latent_std * nodes
        raw_lambdas = sigmoid_softplus(latent)
        lambdas = np.minimum(raw_lambdas, float(lambda_cap))
        scores = image[None, :] - lambdas[:, None] * prior[None, :]
        scores -= scores.max(axis=1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        mean_probability = np.sum(weights[:, None] * probabilities, axis=0)
        proposal_index = int(np.argmax(mean_probability))
        proposal = keys[proposal_index]
        baseline = resolved_baseline_key(row)
        image_top = keys[int(np.argmax(image))]
        prior_top = keys[int(np.argmax(prior))]
        support = exact_top_support(
            proposal_index,
            image,
            prior,
            lambda_cap,
            latent_mean,
            latent_std,
        )
        ranked = np.sort(mean_probability)
        posterior_margin = float(ranked[-1] - ranked[-2]) if len(ranked) > 1 else 1.0
        lambda_mean = float(np.sum(weights * lambdas))
        lambda_variance = float(np.sum(weights * (lambdas - lambda_mean) ** 2))
        prior_relief = (
            float(prior[keys.index(baseline)] - prior[proposal_index])
            if baseline in keys
            else 0.0
        )
        details.append(
            {
                "baseline": baseline,
                "proposal": proposal,
                "image_top": image_top,
                "prior_top": prior_top,
                "visual_conflict": bool(
                    image_top != prior_top and proposal == image_top
                ),
                "prior_absorption": bool(
                    baseline == image_top == prior_top and proposal != baseline
                ),
                "posterior_support": support,
                "posterior_margin": posterior_margin,
                "lambda_mean": lambda_mean,
                "lambda_std": math.sqrt(max(0.0, lambda_variance)),
                "lambda_raw_mean": float(np.sum(weights * raw_lambdas)),
                "lambda_cap": float(lambda_cap),
                "latent_mean": latent_mean,
                "latent_std": latent_std,
                "prior_relief": prior_relief,
                "support_distance": float(support_distances[index])
                if support_distances is not None
                else 0.0,
                "arity_supported": bool(arity_supported[index])
                if arity_supported is not None
                else True,
            }
        )
    return details


def gate_grid(distance_thresholds: Sequence[float] = (float("inf"),)) -> List[Dict[str, float]]:
    output = []
    for visual_support in (0.5, 0.65, 0.8, 0.9, 0.95, 1.01):
        for absorption_support in (0.5, 0.65, 0.8, 0.9, 0.95, 1.01):
            for margin in (0.0, 0.05, 0.1, 0.2, 0.3):
                for distance_threshold in distance_thresholds:
                    output.append(
                        {
                            "visual_support_min": visual_support,
                            "absorption_support_min": absorption_support,
                            "posterior_margin_min": margin,
                            "support_distance_max": float(distance_threshold),
                        }
                    )
    return output


def apply_gate(
    rows: Sequence[Dict[str, Any]],
    details: Sequence[Dict[str, Any]],
    params: Dict[str, float],
) -> List[str]:
    output = []
    for row, detail in zip(rows, details):
        accepted = bool(
            detail["proposal"] != detail["baseline"]
            and bool(detail.get("arity_supported", True))
            and float(detail.get("support_distance", 0.0))
            <= float(params.get("support_distance_max", float("inf")))
            and detail["posterior_margin"] >= params["posterior_margin_min"]
            and (
                detail["visual_conflict"]
                and detail["posterior_support"] >= params["visual_support_min"]
                or detail["prior_absorption"]
                and detail["posterior_support"] >= params["absorption_support_min"]
            )
        )
        output.append(
            detail["proposal"]
            if accepted
            else resolved_baseline_key(row)
        )
    return output


def dataset_metrics(
    rows: Sequence[Dict[str, Any]],
    predictions: Sequence[str],
    ids: Iterable[int],
) -> Dict[str, Any]:
    id_list = list(ids)
    labels = sorted({str(rows[index]["_dataset"]) for index in id_list})
    return {
        label: metric(
            rows,
            predictions,
            [index for index in id_list if rows[index]["_dataset"] == label],
        )
        for label in labels
    }


def valid_retention(
    metrics: Dict[str, Any], max_cs_drop: float, require_cf_nonnegative: bool
) -> bool:
    for result in metrics.values():
        if result["commonsense"]["delta"] < -abs(float(max_cs_drop)) - 1e-12:
            return False
        if require_cf_nonnegative and result["counterfactual"]["delta"] < -1e-12:
            return False
    return True


def selection_key(
    metrics: Dict[str, Any],
    predictions: Sequence[str],
    rows: Sequence[Dict[str, Any]],
    ids: Sequence[int],
    selection_objective: str = "cf_first",
    cs_cost: float = 1.0,
) -> Tuple[float, ...]:
    cf_correct = sum(int(value["counterfactual"]["correct"]) for value in metrics.values())
    improved = sum(value["counterfactual"]["delta"] > 0 for value in metrics.values())
    cs_correct = sum(int(value["commonsense"]["correct"]) for value in metrics.values())
    overrides = sum(
        normalize_key(predictions[index])
        != resolved_baseline_key(rows[index])
        for index in ids
    )
    if selection_objective == "net_utility":
        cf_delta = sum(float(value["counterfactual"]["delta"]) for value in metrics.values())
        cs_delta = sum(float(value["commonsense"]["delta"]) for value in metrics.values())
        return (
            cf_delta + float(cs_cost) * cs_delta,
            cf_delta,
            cs_delta,
            float(improved),
            -float(overrides),
        )
    return cf_correct, improved, cs_correct, -overrides


def bootstrap_delta(
    rows: Sequence[Dict[str, Any]],
    predictions: Sequence[str],
    ids: Sequence[int],
    side: str,
    samples: int,
    seed: int,
) -> List[float]:
    values = [
        int(normalize_key(predictions[index]) == normalize_key(rows[index].get("gt")))
        - int(
            resolved_baseline_key(rows[index])
            == normalize_key(rows[index].get("gt"))
        )
        for index in ids
        if rows[index].get("side") == side
    ]
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    draws = sorted(
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(max(1, int(samples)))
    )
    return [
        float(draws[int(0.025 * (len(draws) - 1))]),
        float(draws[int(0.975 * (len(draws) - 1))]),
    ]


def exact_mcnemar_pvalue(repairs: int, harms: int) -> float:
    discordant = int(repairs) + int(harms)
    if discordant <= 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(int(repairs), int(harms)) + 1)
    ) / float(2**discordant)
    return min(1.0, 2.0 * tail)


def joint_folds(rows: Sequence[Dict[str, Any]], count: int) -> List[int]:
    output = [-1] * len(rows)
    labels = sorted({str(row["_dataset"]) for row in rows})
    for label in labels:
        ids = [index for index, row in enumerate(rows) if row["_dataset"] == label]
        local_rows = [rows[index] for index in ids]
        local_folds = fold_ids(local_rows, requested_folds=count)
        for index, fold in zip(ids, local_folds):
            output[index] = fold
    return output


def evaluation_splits(
    rows: Sequence[Dict[str, Any]],
    mode: str,
    folds: int,
    train_labels: Sequence[str] = (),
) -> Tuple[List[Tuple[str, List[int], List[int]]], List[str]]:
    labels = sorted({str(row["_dataset"]) for row in rows})
    if mode == "joint_oof":
        fold_values = joint_folds(rows, folds)
        return (
            [
                (
                    f"fold-{fold}",
                    [index for index, value in enumerate(fold_values) if value != fold],
                    [index for index, value in enumerate(fold_values) if value == fold],
                )
                for fold in range(folds)
            ],
            labels,
        )
    if mode == "lodo":
        return (
            [
                (
                    f"holdout-{label}",
                    [index for index, row in enumerate(rows) if row["_dataset"] != label],
                    [index for index, row in enumerate(rows) if row["_dataset"] == label],
                )
                for label in labels
            ],
            labels,
        )
    selected_train = set(train_labels)
    unknown = selected_train - set(labels)
    if unknown:
        raise ValueError(f"unknown frozen-transfer train labels: {sorted(unknown)}")
    if not selected_train:
        raise ValueError("frozen_transfer requires at least one --train-label")
    evaluation_labels = [label for label in labels if label not in selected_train]
    if not evaluation_labels:
        raise ValueError("frozen_transfer requires at least one non-training dataset")
    train_ids = [
        index for index, row in enumerate(rows) if row["_dataset"] in selected_train
    ]
    test_ids = [
        index for index, row in enumerate(rows) if row["_dataset"] not in selected_train
    ]
    return [("frozen-transfer", train_ids, test_ids)], evaluation_labels


def fit_and_select(
    rows: Sequence[Dict[str, Any]],
    raw_contexts: np.ndarray,
    train_ids: Sequence[int],
    max_cs_drop: float,
    require_cf_nonnegative: bool,
    quadrature_points: int,
    selection_objective: str = "cf_first",
    cs_cost: float = 1.0,
) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]:
    contexts, mean, scale = standardize_contexts(raw_contexts, train_ids)
    density_contexts = contexts[:, 1:]
    train_density = density_contexts[list(train_ids)]
    density_covariance = np.cov(train_density, rowvar=False)
    density_precision = np.linalg.pinv(
        np.atleast_2d(density_covariance)
        + 0.1 * np.eye(density_contexts.shape[1], dtype=np.float64),
        rcond=1e-8,
    )
    support_distances = np.sqrt(
        np.maximum(
            0.0,
            np.einsum(
                "ni,ij,nj->n", density_contexts, density_precision, density_contexts
            ),
        )
    )
    train_distances = support_distances[list(train_ids)]
    distance_thresholds = sorted(
        {
            float(np.quantile(train_distances, quantile))
            for quantile in (0.9, 0.95, 0.99)
        }
    )
    arity_min = float(np.min(raw_contexts[list(train_ids), 1]))
    arity_max = float(np.max(raw_contexts[list(train_ids), 1]))
    arity_supported = np.logical_and(
        raw_contexts[:, 1] >= arity_min - 1e-12,
        raw_contexts[:, 1] <= arity_max + 1e-12,
    )
    baseline = [resolved_baseline_key(row) for row in rows]
    best: Optional[
        Tuple[Tuple[float, ...], List[str], List[Dict[str, Any]], Dict[str, Any]]
    ] = None
    for l2 in (0.01, 0.1, 1.0, 10.0):
        theta, covariance, loss = fit_lambda_posterior(rows, contexts, train_ids, l2)
        for lambda_cap in (0.8, 1.0, 1.2, 1.5):
            details = policy_details_gauss_hermite(
                rows,
                contexts,
                theta,
                covariance,
                lambda_cap=lambda_cap,
                quadrature_points=quadrature_points,
                support_distances=support_distances,
                arity_supported=arity_supported,
            )
            for gate in gate_grid(distance_thresholds):
                predictions = apply_gate(rows, details, gate)
                metrics = dataset_metrics(rows, predictions, train_ids)
                if not valid_retention(metrics, max_cs_drop, require_cf_nonnegative):
                    continue
                key = selection_key(
                    metrics,
                    predictions,
                    rows,
                    train_ids,
                    selection_objective=selection_objective,
                    cs_cost=cs_cost,
                )
                params = {
                    "l2": l2,
                    "lambda_cap": lambda_cap,
                    "gate": gate,
                    "theta": theta.tolist(),
                    "covariance_diagonal": np.diag(covariance).tolist(),
                    "context_mean": mean.tolist(),
                    "context_scale": scale.tolist(),
                    "train_loss": loss,
                    "arity_support": [arity_min, arity_max],
                    "support_distance_candidates": distance_thresholds,
                    "train_metrics": metrics,
                }
                candidate = (key, predictions, details, params)
                if best is None or candidate[0] > best[0]:
                    best = candidate
    if best is None:
        details = [
            {
                "baseline": value,
                "proposal": value,
                "visual_conflict": False,
                "prior_absorption": False,
                "posterior_support": 0.0,
                "posterior_margin": 0.0,
                "lambda_mean": 0.0,
                "lambda_std": 0.0,
                "lambda_raw_mean": 0.0,
                "lambda_cap": 0.0,
                "prior_relief": 0.0,
                "support_distance": 0.0,
                "arity_supported": True,
            }
            for value in baseline
        ]
        return baseline, details, {"abstain": True, "reason": "no safe policy"}
    return best[1], best[2], best[3]


def run_cv(
    rows: Sequence[Dict[str, Any]],
    mode: str,
    folds: int,
    max_cs_drop: float,
    require_cf_nonnegative: bool,
    quadrature_points: int,
    bootstrap: int,
    seed: int,
    selection_objective: str = "cf_first",
    cs_cost: float = 1.0,
    train_labels: Sequence[str] = (),
    feature_mode: str = "full",
) -> Dict[str, Any]:
    raw_contexts = np.stack(
        [context_features(row, feature_mode=feature_mode) for row in rows]
    )
    baseline = [resolved_baseline_key(row) for row in rows]
    oof = list(baseline)
    oof_details: List[Optional[Dict[str, Any]]] = [None] * len(rows)
    selections = []
    splits, evaluation_labels = evaluation_splits(
        rows, mode, folds, train_labels=train_labels
    )

    for name, train_ids, test_ids in splits:
        predictions, details, params = fit_and_select(
            rows,
            raw_contexts,
            train_ids,
            max_cs_drop,
            require_cf_nonnegative,
            quadrature_points,
            selection_objective,
            cs_cost,
        )
        for index in test_ids:
            oof[index] = predictions[index]
            oof_details[index] = details[index]
        selections.append(
            {
                "split": name,
                "params": params,
                "test_metrics": dataset_metrics(rows, predictions, test_ids),
            }
        )

    all_ids = list(range(len(rows)))
    by_dataset = dataset_metrics(rows, oof, all_ids)
    if mode == "frozen_transfer":
        full_predictions = list(oof)
        full_params = selections[0]["params"]
        deployment_note = (
            "Frozen development fit applied to non-training labels; no target labels "
            "were used for fitting or selection."
        )
    else:
        full_predictions, _, full_params = fit_and_select(
            rows,
            raw_contexts,
            all_ids,
            max_cs_drop,
            require_cf_nonnegative,
            quadrature_points,
            selection_objective,
            cs_cost,
        )
        deployment_note = (
            "deployment-fit development replay; OOF metrics remain the unbiased estimate"
        )
    intervals = {
        label: {
            side: bootstrap_delta(
                rows,
                oof,
                [index for index, row in enumerate(rows) if row["_dataset"] == label],
                side,
                bootstrap,
                seed + label_index * 10 + side_index,
            )
            for side_index, side in enumerate(SIDES)
        }
        for label_index, label in enumerate(evaluation_labels)
    }
    cases = []
    for index, row in enumerate(rows):
        if oof[index] == baseline[index]:
            continue
        detail = oof_details[index] or {}
        cases.append(
            {
                "dataset": row["_dataset"],
                "task": row["_task"],
                "pair_id": row.get("pair_id"),
                "cv_group": row.get("cv_group"),
                "subcategory": row.get("subcategory"),
                "side": row.get("side"),
                "gt": normalize_key(row.get("gt")),
                "baseline": baseline[index],
                "prediction": oof[index],
                "baseline_correct": baseline[index] == normalize_key(row.get("gt")),
                "prediction_correct": oof[index] == normalize_key(row.get("gt")),
                "detail": detail,
            }
        )
    case_summary: Dict[str, Dict[str, Any]] = {}
    for label in sorted(by_dataset):
        selected = [case for case in cases if case["dataset"] == label]
        by_side = {}
        for side in SIDES:
            side_cases = [case for case in selected if case["side"] == side]
            by_side[side] = {
                "overrides": len(side_cases),
                "repairs": sum(
                    not case["baseline_correct"] and case["prediction_correct"]
                    for case in side_cases
                ),
                "harms": sum(
                    case["baseline_correct"] and not case["prediction_correct"]
                    for case in side_cases
                ),
                "neutral": sum(
                    case["baseline_correct"] == case["prediction_correct"]
                    for case in side_cases
                ),
                "visual_conflict": sum(
                    bool(case["detail"].get("visual_conflict")) for case in side_cases
                ),
                "prior_absorption": sum(
                    bool(case["detail"].get("prior_absorption")) for case in side_cases
                ),
            }
        case_summary[label] = by_side
    paired_tests = {
        label: {
            side: {
                "repairs": int(case_summary[label][side]["repairs"]),
                "harms": int(case_summary[label][side]["harms"]),
                "discordant": int(case_summary[label][side]["repairs"])
                + int(case_summary[label][side]["harms"]),
                "exact_mcnemar_p_two_sided": exact_mcnemar_pvalue(
                    int(case_summary[label][side]["repairs"]),
                    int(case_summary[label][side]["harms"]),
                ),
            }
            for side in SIDES
        }
        for label in evaluation_labels
    }
    return {
        "mode": mode,
        "feature_mode": feature_mode,
        "train_labels": sorted(set(train_labels)),
        "evaluation_labels": evaluation_labels,
        "by_dataset": by_dataset,
        "evaluation_by_dataset": {
            label: by_dataset[label] for label in evaluation_labels
        },
        "delta_ci95": intervals,
        "selections": selections,
        "full_data_selected": {
            "params": full_params,
            "by_dataset": dataset_metrics(rows, full_predictions, all_ids),
            "note": deployment_note,
        },
        "case_summary": case_summary,
        "paired_tests": paired_tests,
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hierarchical empirical-Bayes context-conditioned prior coefficient."
    )
    parser.add_argument("--input", action="append", type=parse_input, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        choices=("joint_oof", "lodo", "frozen_transfer"),
        default="joint_oof",
    )
    parser.add_argument(
        "--train-label",
        action="append",
        default=[],
        help="Dataset label used for fitting in frozen_transfer mode; repeat as needed.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-cs-drop", type=float, default=0.05)
    parser.add_argument(
        "--selection-objective",
        choices=("cf_first", "net_utility"),
        default="cf_first",
        help=(
            "cf_first preserves the legacy lexicographic selector; net_utility "
            "maximizes summed delta_CF + cs_cost * delta_CS across training datasets."
        ),
    )
    parser.add_argument(
        "--cs-cost",
        type=float,
        default=1.0,
        help="Cost assigned to one point of CS change under net_utility.",
    )
    parser.add_argument("--allow-cf-harm", action="store_true")
    parser.add_argument(
        "--posterior-draws",
        type=int,
        default=0,
        help="Deprecated alias for --quadrature-points.",
    )
    parser.add_argument(
        "--quadrature-points",
        type=int,
        default=64,
        help="One-dimensional Gauss-Hermite points for the Laplace posterior predictive integral.",
    )
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument(
        "--feature-mode",
        choices=("full", "image_only"),
        default="full",
        help="Use all score-distribution features or the matched image-only control.",
    )
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    sources = []
    for spec in args.input:
        loaded = read_jsonl(spec.path, spec.task)
        for row in loaded:
            row = dict(row)
            row["_dataset"] = spec.label
            row["_task"] = spec.task
            rows.append(row)
        sources.append(
            {"label": spec.label, "task": spec.task, "path": str(spec.path), "n": len(loaded)}
        )
    if not rows:
        raise SystemExit("no usable CP-VBC rows")

    quadrature_points = max(8, int(args.posterior_draws or args.quadrature_points))
    result = run_cv(
        rows,
        args.mode,
        max(2, int(args.folds)),
        float(args.max_cs_drop),
        not bool(args.allow_cf_harm),
        quadrature_points,
        max(1, int(args.bootstrap)),
        int(args.seed),
        args.selection_objective,
        max(0.0, float(args.cs_cost)),
        args.train_label,
        args.feature_mode,
    )
    output = {
        "version": VERSION,
        "sources": sources,
        "method": {
            "lambda": "softplus(theta^T x), with x formed only from native image/no-image candidate-distribution statistics",
            "posterior": "Gaussian hyperprior, MAP fit, Laplace covariance, one-dimensional Gauss-Hermite posterior predictive marginalization, and analytic top-candidate support",
            "gate": "REVIS-style sparse visual-conflict/prior-absorption gate calibrated under per-dataset CS constraints",
        },
        "fairness": (
            "dataset, task name, subcategory, side, pair identity, cv_group, GT, and semantic candidate text "
            "are excluded from inference features. Pair/cv_group and dataset labels are evaluation-only."
        ),
        "max_cs_drop": float(args.max_cs_drop),
        "selection_objective": args.selection_objective,
        "cs_cost": max(0.0, float(args.cs_cost)),
        "quadrature_points": quadrature_points,
        "require_train_cf_nonnegative_per_dataset": not bool(args.allow_cf_harm),
        "train_labels": sorted(set(args.train_label)),
        "result": result,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("dataset\tCF_Acc\tdCF\tCS_Acc\tdCS\tCF_CI95\tCS_CI95")
    def display(value: Any) -> str:
        return "NA" if value is None else f"{float(value):.3f}"

    for label, values in result["evaluation_by_dataset"].items():
        print(
            "\t".join(
                (
                    label,
                    display(values["counterfactual"]["acc"]),
                    display(values["counterfactual"]["delta"]),
                    display(values["commonsense"]["acc"]),
                    display(values["commonsense"]["delta"]),
                    str(result["delta_ci95"][label]["counterfactual"]),
                    str(result["delta_ci95"][label]["commonsense"]),
                )
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
