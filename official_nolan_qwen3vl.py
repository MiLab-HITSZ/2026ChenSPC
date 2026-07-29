#!/usr/bin/env python3
"""Token-level NoLan decoding for Hugging Face Qwen3-VL and LLaVA-NeXT models.

The released NoLan sampler patches an older Transformers generation loop. This
module ports the published decoding rule while keeping two synchronized
generation streams: the normal image-conditioned stream and a text-only stream.
Both streams receive every generated token, and their KV caches are updated
independently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


OFFICIAL_NOLAN_REPOSITORY = "https://github.com/lingfengren/NoLan"
OFFICIAL_NOLAN_COMMIT = "8b6d92589d8e2608f49eb7c4bca211a8c1495723"
OFFICIAL_NOLAN_BETA = 0.8


@dataclass(frozen=True)
class NoLanGeneration:
    token_ids: List[int]
    text: str
    diagnostics: Dict[str, Any]


@dataclass(frozen=True)
class NoLanCandidateChoice:
    key: str
    native_key: str
    prior_key: str
    scores: Dict[str, float]
    native_scores: Dict[str, float]
    prior_scores: Dict[str, float]
    diagnostics: Dict[str, Any]


def symmetric_kl_from_logits(logits_a: Any, logits_b: Any) -> Any:
    """Return per-row symmetric KL over the complete vocabulary."""
    import torch

    a = logits_a.float()
    b = logits_b.float()
    log_a = torch.log_softmax(a, dim=-1)
    log_b = torch.log_softmax(b, dim=-1)
    prob_a = log_a.exp()
    prob_b = log_b.exp()
    kl_a_b = torch.sum(prob_a * (log_a - log_b), dim=-1)
    kl_b_a = torch.sum(prob_b * (log_b - log_a), dim=-1)
    return 0.5 * (kl_a_b + kl_b_a)


def nolan_alpha(symmetric_kl: Any, beta: float = OFFICIAL_NOLAN_BETA) -> Any:
    """Published NoLan-Plus coefficient beta * (tanh(1 / gamma) + 1)."""
    import torch

    if float(beta) < 0:
        raise ValueError("beta must be nonnegative")
    gamma = torch.clamp(symmetric_kl.float(), min=torch.finfo(torch.float32).tiny)
    return float(beta) * (torch.tanh(gamma.reciprocal()) + 1.0)


def nolan_adjust_logits(
    multimodal_logits: Any,
    text_only_logits: Any,
    beta: float = OFFICIAL_NOLAN_BETA,
) -> Tuple[Any, Any, Any]:
    """Apply the official dynamic prior-suppression rule to next-token logits."""
    gamma = symmetric_kl_from_logits(multimodal_logits, text_only_logits)
    alpha = nolan_alpha(gamma, beta=beta)
    adjusted = (
        (1.0 + alpha.unsqueeze(-1)) * multimodal_logits.float()
        - alpha.unsqueeze(-1) * text_only_logits.float()
    )
    return adjusted, gamma, alpha


def _branch_state(inputs: Mapping[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    import torch

    if "input_ids" not in inputs:
        raise ValueError("generation inputs must contain input_ids")
    input_ids = inputs["input_ids"]
    kwargs = {key: value for key, value in inputs.items() if key != "input_ids"}
    kwargs["use_cache"] = True
    kwargs["cache_position"] = torch.arange(
        int(input_ids.shape[-1]), dtype=torch.long, device=input_ids.device
    )
    return input_ids, kwargs


def _forward_branch(
    model: Any,
    input_ids: Any,
    model_kwargs: Dict[str, Any],
    first_iteration: bool,
) -> Tuple[Any, Dict[str, Any]]:
    model_inputs = model.prepare_inputs_for_generation(
        input_ids,
        **model_kwargs,
        is_first_iteration=bool(first_iteration),
    )
    if model_inputs.get("logits_to_keep") is None:
        model_inputs["logits_to_keep"] = 1
    outputs = model(**model_inputs, return_dict=True)
    next_logits = outputs.logits[:, -1, :]
    updated = model._update_model_kwargs_for_generation(
        outputs,
        model_kwargs,
        is_encoder_decoder=False,
    )
    return next_logits, updated


def _eos_token_ids(model: Any, tokenizer: Any) -> set[int]:
    values: List[int] = []
    for source in (
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
    ):
        if source is None:
            continue
        if isinstance(source, (list, tuple, set)):
            values.extend(int(value) for value in source if value is not None)
        else:
            values.append(int(source))
    return set(values)


def _sample_token(
    logits: Any,
    do_sample: bool,
    temperature: float,
    top_p: float,
    generator: Optional[Any],
) -> Any:
    import torch

    if not do_sample:
        return torch.argmax(logits, dim=-1)
    if float(temperature) <= 0:
        raise ValueError("temperature must be positive when sampling")
    scores = logits / float(temperature)
    if 0.0 < float(top_p) < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        cumulative = torch.softmax(sorted_scores, dim=-1).cumsum(dim=-1)
        remove = cumulative > float(top_p)
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
        filtered = torch.full_like(scores, float("-inf"))
        filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_scores)
        scores = filtered
    probabilities = torch.softmax(scores, dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator).squeeze(-1)


def choose_candidate_with_nolan(
    model: Any,
    multimodal_inputs: Mapping[str, Any],
    text_only_inputs: Mapping[str, Any],
    candidate_token_ids: Mapping[str, Sequence[int]],
    *,
    beta: float = OFFICIAL_NOLAN_BETA,
) -> NoLanCandidateChoice:
    """Apply the official rule at a shared candidate-prefix decision point.

    CDH candidates have equal token lengths and share every token except the
    final answer token. Common-prefix likelihood is therefore identical across
    candidates, so the exact constrained decision is obtained from the
    official adjusted full-vocabulary logits at the final position.
    """
    import torch

    candidates = {
        str(key): tuple(int(token) for token in values)
        for key, values in candidate_token_ids.items()
    }
    if len(candidates) < 2 or any(not values for values in candidates.values()):
        raise ValueError("at least two nonempty candidate token sequences are required")
    lengths = {len(values) for values in candidates.values()}
    prefixes = {values[:-1] for values in candidates.values()}
    if len(lengths) != 1 or len(prefixes) != 1:
        raise ValueError(
            "constrained NoLan requires equal-length candidates with a shared prefix"
        )

    image_ids, image_kwargs = _branch_state(multimodal_inputs)
    prior_ids, prior_kwargs = _branch_state(text_only_inputs)
    if int(image_ids.shape[0]) != 1 or int(prior_ids.shape[0]) != 1:
        raise ValueError("the constrained NoLan port currently supports batch size 1")

    prefix = next(iter(prefixes))
    prefix_trace: List[Dict[str, float]] = []
    with torch.inference_mode():
        for step, forced_token in enumerate(prefix):
            image_logits, image_kwargs = _forward_branch(
                model, image_ids, image_kwargs, first_iteration=step == 0
            )
            prior_logits, prior_kwargs = _forward_branch(
                model, prior_ids, prior_kwargs, first_iteration=step == 0
            )
            _, gamma, alpha = nolan_adjust_logits(
                image_logits, prior_logits, beta=beta
            )
            prefix_trace.append(
                {
                    "step": float(step),
                    "token_id": float(forced_token),
                    "symmetric_kl": float(gamma[0].detach().cpu().item()),
                    "alpha": float(alpha[0].detach().cpu().item()),
                }
            )
            token_column = torch.tensor(
                [[forced_token]], dtype=image_ids.dtype, device=image_ids.device
            )
            image_ids = torch.cat([image_ids, token_column], dim=-1)
            prior_ids = torch.cat(
                [prior_ids, token_column.to(prior_ids.device)], dim=-1
            )

        final_step = len(prefix)
        image_logits, image_kwargs = _forward_branch(
            model, image_ids, image_kwargs, first_iteration=final_step == 0
        )
        prior_logits, prior_kwargs = _forward_branch(
            model, prior_ids, prior_kwargs, first_iteration=final_step == 0
        )
        adjusted, gamma, alpha = nolan_adjust_logits(
            image_logits, prior_logits, beta=beta
        )

    final_tokens = {key: values[-1] for key, values in candidates.items()}

    def select_scores(logits: Any) -> Dict[str, float]:
        return {
            key: float(logits[0, token].detach().cpu().item())
            for key, token in final_tokens.items()
        }

    scores = select_scores(adjusted)
    native_scores = select_scores(image_logits)
    prior_scores = select_scores(prior_logits)
    winner = max(scores, key=scores.get)
    native_winner = max(native_scores, key=native_scores.get)
    prior_winner = max(prior_scores, key=prior_scores.get)
    return NoLanCandidateChoice(
        key=winner,
        native_key=native_winner,
        prior_key=prior_winner,
        scores=scores,
        native_scores=native_scores,
        prior_scores=prior_scores,
        diagnostics={
            "method": "official_tokenwise_constrained_candidate_nolan",
            "official_repository": OFFICIAL_NOLAN_REPOSITORY,
            "official_commit": OFFICIAL_NOLAN_COMMIT,
            "beta": float(beta),
            "distribution": "full_vocabulary",
            "candidate_constraint": list(candidates),
            "shared_candidate_prefix": list(prefix),
            "final_candidate_token_ids": final_tokens,
            "prefix_trace": prefix_trace,
            "final_symmetric_kl": float(gamma[0].detach().cpu().item()),
            "final_alpha": float(alpha[0].detach().cpu().item()),
        },
    )


def generate_with_nolan(
    model: Any,
    processor: Any,
    multimodal_inputs: Mapping[str, Any],
    text_only_inputs: Mapping[str, Any],
    *,
    beta: float = OFFICIAL_NOLAN_BETA,
    max_new_tokens: int = 16,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    seed: Optional[int] = None,
    trace_steps: int = 16,
    stop_token_ids: Optional[Sequence[int]] = None,
) -> NoLanGeneration:
    """Generate with official token-level NoLan over synchronized VLM streams."""
    import torch

    if int(max_new_tokens) <= 0:
        raise ValueError("max_new_tokens must be positive")
    if float(beta) < 0:
        raise ValueError("beta must be nonnegative")
    tokenizer = getattr(processor, "tokenizer", processor)
    image_ids, image_kwargs = _branch_state(multimodal_inputs)
    prior_ids, prior_kwargs = _branch_state(text_only_inputs)
    if int(image_ids.shape[0]) != 1 or int(prior_ids.shape[0]) != 1:
        raise ValueError("the NoLan port currently supports batch size 1")

    generator = None
    if seed is not None:
        generator = torch.Generator(device=image_ids.device)
        generator.manual_seed(int(seed))

    eos_ids = _eos_token_ids(model, tokenizer)
    answer_stop_ids = {
        int(token) for token in (stop_token_ids or ()) if token is not None
    }
    generated: List[int] = []
    traces: List[Dict[str, Any]] = []
    gammas: List[float] = []
    alphas: List[float] = []
    stopped_on_answer = False

    with torch.inference_mode():
        for step in range(int(max_new_tokens)):
            image_logits, image_kwargs = _forward_branch(
                model, image_ids, image_kwargs, first_iteration=step == 0
            )
            prior_logits, prior_kwargs = _forward_branch(
                model, prior_ids, prior_kwargs, first_iteration=step == 0
            )
            adjusted, gamma, alpha = nolan_adjust_logits(
                image_logits, prior_logits, beta=beta
            )
            next_tokens = _sample_token(
                adjusted,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                generator=generator,
            )
            next_id = int(next_tokens[0].detach().cpu().item())
            native_id = int(torch.argmax(image_logits[0]).detach().cpu().item())
            prior_id = int(torch.argmax(prior_logits[0]).detach().cpu().item())
            gamma_value = float(gamma[0].detach().cpu().item())
            alpha_value = float(alpha[0].detach().cpu().item())
            gammas.append(gamma_value)
            alphas.append(alpha_value)
            generated.append(next_id)

            if step < max(0, int(trace_steps)):
                traces.append(
                    {
                        "step": step,
                        "symmetric_kl": gamma_value,
                        "alpha": alpha_value,
                        "native_token_id": native_id,
                        "native_token": tokenizer.decode([native_id]),
                        "prior_token_id": prior_id,
                        "prior_token": tokenizer.decode([prior_id]),
                        "nolan_token_id": next_id,
                        "nolan_token": tokenizer.decode([next_id]),
                        "changed_from_native": next_id != native_id,
                    }
                )

            token_column = next_tokens[:, None]
            image_ids = torch.cat([image_ids, token_column.to(image_ids.device)], dim=-1)
            prior_ids = torch.cat([prior_ids, token_column.to(prior_ids.device)], dim=-1)
            if next_id in answer_stop_ids:
                stopped_on_answer = True
                break
            if next_id in eos_ids:
                break

    text = tokenizer.decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    changed_steps = sum(bool(row["changed_from_native"]) for row in traces)
    diagnostics = {
        "method": "official_token_level_nolan_port",
        "official_repository": OFFICIAL_NOLAN_REPOSITORY,
        "official_commit": OFFICIAL_NOLAN_COMMIT,
        "beta": float(beta),
        "distribution": "full_vocabulary",
        "prior_condition": "text_only",
        "shared_generated_prefix": True,
        "decoding": "sampling" if do_sample else "greedy",
        "temperature": float(temperature),
        "top_p": float(top_p),
        "generated_token_count": len(generated),
        "answer_stop_token_ids": sorted(answer_stop_ids),
        "stopped_on_answer": stopped_on_answer,
        "traced_changed_steps": changed_steps,
        "symmetric_kl": {
            "minimum": min(gammas) if gammas else None,
            "mean": sum(gammas) / len(gammas) if gammas else None,
            "maximum": max(gammas) if gammas else None,
        },
        "alpha": {
            "minimum": min(alphas) if alphas else None,
            "mean": sum(alphas) / len(alphas) if alphas else None,
            "maximum": max(alphas) if alphas else None,
        },
        "steps": traces,
    }
    return NoLanGeneration(token_ids=generated, text=text, diagnostics=diagnostics)
