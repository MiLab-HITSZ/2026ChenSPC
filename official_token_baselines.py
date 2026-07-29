#!/usr/bin/env python3
"""Equation-preserving token-level ports of released hallucination baselines.

The upstream implementations patch model-specific generation loops.  This
module keeps their image transforms, full-vocabulary logit rules, and published
parameters intact while adapting only the Hugging Face model I/O needed by
Qwen3-VL and LLaVA-NeXT.
"""

from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from official_nolan_qwen3vl import (
    _branch_state,
    _eos_token_ids,
    _forward_branch,
)


VCD_REPOSITORY = "https://github.com/DAMO-NLP-SG/VCD"
VCD_COMMIT = "d6568ff81b8fd306a49e630df44f2db5c2300191"
VCD_ALPHA = 1.0
VCD_BETA = 0.1
VCD_NOISE_STEP = 500

MFCD_REPOSITORY = "https://github.com/liubq-dev/mfcd"
MFCD_COMMIT = "a5cd8de499aa9056171a8c8fc208eda2d3261e13"
MFCD_HIGH_ALPHA = 1.0
MFCD_LOW_ALPHA = 1.0
MFCD_BETA = 0.3
MFCD_HIGH_PASS_CUTOFF = 0.1
MFCD_LOW_PASS_CUTOFF = 0.9

PAI_REPOSITORY = "https://github.com/LALBJ/PAI"
PAI_COMMIT = "9bbd8bd57a0b0923f996197e4bd3e02cc10b8d58"
PAI_ATTENTION_ALPHA = 0.2
PAI_GUIDANCE_SCALE = 1.1
PAI_PLAUSIBILITY_BETA = 0.1
PAI_START_LAYER = 2


@dataclass(frozen=True)
class TokenBaselineChoice:
    key: str
    native_key: str
    scores: Dict[str, float]
    native_scores: Dict[str, float]
    branch_scores: Dict[str, Dict[str, float]]
    diagnostics: Dict[str, Any]


@dataclass(frozen=True)
class TokenBaselineGeneration:
    token_ids: List[int]
    text: str
    diagnostics: Dict[str, Any]


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import official source file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _clone_inputs(inputs: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value.clone() if hasattr(value, "clone") else value
        for key, value in inputs.items()
    }


def build_vcd_inputs(
    multimodal_inputs: Mapping[str, Any],
    *,
    repo_root: Path = Path("downloads/hallucination_refs/VCD"),
    noise_step: int = VCD_NOISE_STEP,
    seed: int | None = 42,
) -> Dict[str, Any]:
    """Construct the distorted branch with the released VCD noise function."""
    import torch

    distorted = _clone_inputs(multimodal_inputs)
    pixels = distorted.get("pixel_values")
    if pixels is None:
        raise ValueError("VCD requires pixel_values in the multimodal inputs")
    source = repo_root / "vcd_utils" / "vcd_add_noise.py"
    module = _load_module("_cdh_official_vcd_add_noise", source)
    if seed is None:
        distorted["pixel_values"] = module.add_diffusion_noise(
            pixels, int(noise_step)
        )
    else:
        devices = (
            [pixels.device.index]
            if pixels.is_cuda and pixels.device.index is not None
            else []
        )
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(seed))
            distorted["pixel_values"] = module.add_diffusion_noise(
                pixels, int(noise_step)
            )
    return distorted


def build_mfcd_images(
    image: Any,
    *,
    repo_root: Path = Path("downloads/hallucination_refs/MFCD"),
    high_pass_cutoff: float = MFCD_HIGH_PASS_CUTOFF,
    low_pass_cutoff: float = MFCD_LOW_PASS_CUTOFF,
) -> Tuple[Any, Any]:
    """Construct high- and low-frequency views with the released MFCD filters."""
    source = repo_root / "utils" / "image_utils.py"
    module = _load_module("_cdh_official_mfcd_image_utils", source)
    high = module.gaussian_high_pass_filter(
        image=image, cutoff=float(high_pass_cutoff)
    )
    low = module.gaussian_low_pass_filter(
        image=image, cutoff=float(low_pass_cutoff)
    )
    return high, low


def vcd_adjust_logits(
    native_logits: Any,
    distorted_logits: Any,
    *,
    alpha: float = VCD_ALPHA,
    beta: float = VCD_BETA,
) -> Any:
    """Released VCD full-vocabulary rule and plausibility constraint."""
    import torch

    native = native_logits.float()
    distorted = distorted_logits.float()
    cutoff = math.log(float(beta)) + native.max(dim=-1, keepdim=True).values
    adjusted = (1.0 + float(alpha)) * native - float(alpha) * distorted
    return adjusted.masked_fill(native < cutoff, -torch.inf)


def _mfcd_jsd(logits_a: Any, logits_b: Any) -> Any:
    import torch
    import torch.nn.functional as F

    p_a = F.softmax(logits_a, dim=-1)
    p_b = F.softmax(logits_b, dim=-1)
    midpoint = 0.5 * (p_a + p_b)
    kl_a = F.kl_div(torch.log(p_a), midpoint, reduction="batchmean")
    kl_b = F.kl_div(torch.log(p_b), midpoint, reduction="batchmean")
    return torch.sqrt(0.5 * (kl_a + kl_b))


def mfcd_adjust_logits(
    native_logits: Any,
    high_pass_logits: Any,
    low_pass_logits: Any,
    *,
    high_alpha: float = MFCD_HIGH_ALPHA,
    low_alpha: float = MFCD_LOW_ALPHA,
    beta: float = MFCD_BETA,
    use_jsd: bool = False,
    use_entropy: bool = False,
) -> Any:
    """Released MFCD full-vocabulary rule."""
    import torch
    import torch.nn.functional as F

    native = native_logits.float()
    high = high_pass_logits.float()
    low = low_pass_logits.float()
    native_prob = F.softmax(native, dim=-1)
    maximum = native_prob.max(dim=-1, keepdim=True).values
    effective_beta = torch.tensor(
        float(beta), dtype=native_prob.dtype, device=native_prob.device
    )
    if use_entropy:
        effective_beta = effective_beta * (
            1.0
            - torch.exp(
                torch.sum(
                    native_prob * F.log_softmax(native, dim=-1),
                    dim=-1,
                )
            )
        )
    mask = native_prob.lt(effective_beta.unsqueeze(-1) * maximum)
    effective_high = torch.tensor(
        float(high_alpha), dtype=native.dtype, device=native.device
    )
    effective_low = torch.tensor(
        float(low_alpha), dtype=native.dtype, device=native.device
    )
    if use_jsd:
        effective_high = effective_high * torch.exp(-_mfcd_jsd(native, high))
        effective_low = effective_low * torch.exp(-_mfcd_jsd(native, low))
    adjusted = (
        (1.0 + effective_high + effective_low) * native
        - effective_low * low
        - effective_high * high
    )
    return adjusted.masked_fill(mask, torch.finfo(adjusted.dtype).min)


def pai_adjust_logits(
    multimodal_logits: Any,
    text_only_logits: Any,
    *,
    guidance_scale: float = PAI_GUIDANCE_SCALE,
    plausibility_beta: float = PAI_PLAUSIBILITY_BETA,
) -> Any:
    """Released PAI classifier-free guidance rule."""
    import torch
    import torch.nn.functional as F

    multimodal = F.log_softmax(multimodal_logits.float(), dim=-1)
    text_only = F.log_softmax(text_only_logits.float(), dim=-1)
    cutoff = (
        math.log(float(plausibility_beta))
        + multimodal.max(dim=-1, keepdim=True).values
    )
    adjusted = (
        float(guidance_scale) * (multimodal - text_only) + text_only
    )
    return adjusted.masked_fill(multimodal < cutoff, -torch.inf)


def _pai_attention_forward(
    module: Any,
    query: Any,
    key: Any,
    value: Any,
    attention_mask: Any,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
) -> Tuple[Any, Any]:
    """Current-Transformers attention interface with PAI's released edit.

    PAI changes only the final query row.  SDPA computes the unchanged rows,
    while that final row is recomputed explicitly with the released equation.
    This is algebraically equivalent to the upstream eager implementation and
    avoids materializing a full sequence-by-sequence attention matrix.
    """
    import torch
    import torch.nn.functional as F
    from transformers.models.llama.modeling_llama import repeat_kv

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    active = bool(getattr(module, "_pai_use_attention", False)) and not bool(
        getattr(module, "_pai_cfg_branch", False)
    )
    causal = bool(
        query.shape[2] > 1
        and attention_mask is None
        and getattr(module, "is_causal", True)
    )
    output = F.scaled_dot_product_attention(
        query,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=causal,
    )

    if active:
        last_query = query[:, :, -1:, :]
        last_weights = (
            torch.matmul(last_query, key_states.transpose(2, 3)) * scaling
        )
        if attention_mask is not None:
            last_weights = last_weights + attention_mask[..., -1:, :]
        start = int(module._pai_image_start)
        end = getattr(module, "_pai_image_end", None)
        if end is None:
            original_length = int(module._pai_original_input_length)
            end = start + int(key_states.shape[-2]) - original_length + 1
            module._pai_image_end = int(end)
        end = min(int(end), int(key_states.shape[-2]))
        if end > start:
            image_weights = last_weights[:, :, :, start:end]
            last_weights[:, :, :, start:end] = (
                image_weights.abs() * float(module._pai_alpha) + image_weights
            )
        last_weights = F.softmax(
            last_weights, dim=-1, dtype=torch.float32
        ).to(query.dtype)
        last_weights = F.dropout(
            last_weights, p=dropout, training=module.training
        )
        output[:, :, -1:, :] = torch.matmul(last_weights, value_states)
    return output.transpose(1, 2).contiguous(), None


def _decoder_layers(model: Any) -> Sequence[Any]:
    language_model = getattr(getattr(model, "model", None), "language_model", None)
    layers = getattr(language_model, "layers", None)
    if layers is None:
        raise ValueError("cannot locate the model's language decoder layers")
    return layers


def configure_pai_attention(
    model: Any,
    multimodal_inputs: Mapping[str, Any],
    family: str,
    *,
    alpha: float = PAI_ATTENTION_ALPHA,
    start_layer: int = PAI_START_LAYER,
    end_layer: int | None = None,
) -> List[Any]:
    """Install PAI's attention edit on the target model's decoder."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS.register("cdh_pai_eager", _pai_attention_forward)
    layers = list(_decoder_layers(model))
    stop = len(layers) if end_layer is None else min(int(end_layer), len(layers))
    start = max(0, int(start_layer))
    if start >= stop:
        raise ValueError(f"empty PAI layer range: {start}:{stop}")

    input_ids = multimodal_inputs["input_ids"][0]
    if family == "qwen3_vl":
        image_token = int(model.config.image_token_id)
        positions = (input_ids == image_token).nonzero(as_tuple=False).flatten()
        if positions.numel() == 0:
            raise ValueError("Qwen3-VL input contains no image tokens")
        image_start = int(positions.min().item())
        image_end: int | None = int(positions.max().item()) + 1
    elif family == "llava_next":
        image_token = int(model.config.image_token_index)
        positions = (input_ids == image_token).nonzero(as_tuple=False).flatten()
        if positions.numel() == 0:
            raise ValueError(
                "LLaVA-NeXT input contains no image placeholder or expanded image tokens"
            )
        image_start = int(positions.min().item())
        image_end = (
            int(positions.max().item()) + 1 if positions.numel() > 1 else None
        )
    else:
        raise ValueError(f"unsupported PAI model family: {family}")

    patched: List[Any] = []
    for index, layer in enumerate(layers):
        attention = layer.self_attn
        attention.config._attn_implementation = "cdh_pai_eager"
        attention._pai_use_attention = start <= index < stop
        attention._pai_cfg_branch = False
        attention._pai_alpha = float(alpha)
        attention._pai_image_start = image_start
        attention._pai_image_end = image_end
        attention._pai_original_input_length = int(input_ids.shape[-1])
        patched.append(attention)
    return patched


def set_pai_cfg_branch(attention_modules: Sequence[Any], enabled: bool) -> None:
    for module in attention_modules:
        module._pai_cfg_branch = bool(enabled)


def _candidate_layout(
    candidate_token_ids: Mapping[str, Sequence[int]],
) -> Tuple[Dict[str, Tuple[int, ...]], Tuple[int, ...]]:
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
            "constrained token decoding requires equal-length candidates "
            "with a shared prefix"
        )
    return candidates, next(iter(prefixes))


def _choose_with_branches(
    model: Any,
    branch_inputs: Mapping[str, Mapping[str, Any]],
    candidate_token_ids: Mapping[str, Sequence[int]],
    adjust: Callable[[Mapping[str, Any]], Any],
    *,
    before_branch: Callable[[str], None] | None = None,
) -> Tuple[
    str,
    str,
    Dict[str, float],
    Dict[str, float],
    Dict[str, Dict[str, float]],
    Tuple[int, ...],
]:
    import torch

    candidates, prefix = _candidate_layout(candidate_token_ids)
    prepared: Dict[str, Dict[str, Any]] = {}
    for name, inputs in branch_inputs.items():
        values = _clone_inputs(inputs)
        if prefix:
            prefix_ids = torch.tensor(
                [list(prefix)],
                dtype=values["input_ids"].dtype,
                device=values["input_ids"].device,
            )
            values["input_ids"] = torch.cat(
                [values["input_ids"], prefix_ids], dim=-1
            )
            if values.get("attention_mask") is not None:
                extension = torch.ones(
                    (values["attention_mask"].shape[0], len(prefix)),
                    dtype=values["attention_mask"].dtype,
                    device=values["attention_mask"].device,
                )
                values["attention_mask"] = torch.cat(
                    [values["attention_mask"], extension], dim=-1
                )
        prepared[name] = values
    states = {name: _branch_state(inputs) for name, inputs in prepared.items()}
    with torch.inference_mode():
        logits: Dict[str, Any] = {}
        for name, (ids, kwargs) in states.items():
            if before_branch is not None:
                before_branch(name)
            branch_logits, _ = _forward_branch(
                model, ids, kwargs, first_iteration=True
            )
            logits[name] = branch_logits
        adjusted = adjust(logits)

    final_tokens = {key: values[-1] for key, values in candidates.items()}

    def select(source: Any) -> Dict[str, float]:
        return {
            key: float(source[0, token].detach().cpu().item())
            for key, token in final_tokens.items()
        }

    scores = select(adjusted)
    native_scores = select(logits["native"])
    branch_scores = {name: select(values) for name, values in logits.items()}
    return (
        max(scores, key=scores.get),
        max(native_scores, key=native_scores.get),
        scores,
        native_scores,
        branch_scores,
        prefix,
    )


def choose_candidate_with_vcd(
    model: Any,
    multimodal_inputs: Mapping[str, Any],
    candidate_token_ids: Mapping[str, Sequence[int]],
    *,
    repo_root: Path = Path("downloads/hallucination_refs/VCD"),
    alpha: float = VCD_ALPHA,
    beta: float = VCD_BETA,
    noise_step: int = VCD_NOISE_STEP,
    seed: int = 42,
) -> TokenBaselineChoice:
    distorted = build_vcd_inputs(
        multimodal_inputs,
        repo_root=repo_root,
        noise_step=noise_step,
        seed=seed,
    )
    result = _choose_with_branches(
        model,
        {"native": multimodal_inputs, "distorted": distorted},
        candidate_token_ids,
        lambda logits: vcd_adjust_logits(
            logits["native"], logits["distorted"], alpha=alpha, beta=beta
        ),
    )
    key, native_key, scores, native_scores, branches, prefix = result
    return TokenBaselineChoice(
        key=key,
        native_key=native_key,
        scores=scores,
        native_scores=native_scores,
        branch_scores=branches,
        diagnostics={
            "method": "vcd",
            "repository": VCD_REPOSITORY,
            "commit": VCD_COMMIT,
            "execution": "full_vocabulary_token_level",
            "alpha": float(alpha),
            "beta": float(beta),
            "noise_step": int(noise_step),
            "seed": int(seed),
            "shared_candidate_prefix": list(prefix),
        },
    )


def choose_candidate_with_mfcd(
    model: Any,
    multimodal_inputs: Mapping[str, Any],
    high_pass_inputs: Mapping[str, Any],
    low_pass_inputs: Mapping[str, Any],
    candidate_token_ids: Mapping[str, Sequence[int]],
    *,
    high_alpha: float = MFCD_HIGH_ALPHA,
    low_alpha: float = MFCD_LOW_ALPHA,
    beta: float = MFCD_BETA,
    high_pass_cutoff: float = MFCD_HIGH_PASS_CUTOFF,
    low_pass_cutoff: float = MFCD_LOW_PASS_CUTOFF,
) -> TokenBaselineChoice:
    result = _choose_with_branches(
        model,
        {
            "native": multimodal_inputs,
            "high_pass": high_pass_inputs,
            "low_pass": low_pass_inputs,
        },
        candidate_token_ids,
        lambda logits: mfcd_adjust_logits(
            logits["native"],
            logits["high_pass"],
            logits["low_pass"],
            high_alpha=high_alpha,
            low_alpha=low_alpha,
            beta=beta,
        ),
    )
    key, native_key, scores, native_scores, branches, prefix = result
    return TokenBaselineChoice(
        key=key,
        native_key=native_key,
        scores=scores,
        native_scores=native_scores,
        branch_scores=branches,
        diagnostics={
            "method": "mfcd",
            "repository": MFCD_REPOSITORY,
            "commit": MFCD_COMMIT,
            "execution": "full_vocabulary_token_level",
            "high_alpha": float(high_alpha),
            "low_alpha": float(low_alpha),
            "beta": float(beta),
            "high_pass_cutoff": float(high_pass_cutoff),
            "low_pass_cutoff": float(low_pass_cutoff),
            "filter": "gaussian",
            "jsd": False,
            "entropy": False,
            "shared_candidate_prefix": list(prefix),
        },
    )


def choose_candidate_with_pai(
    model: Any,
    multimodal_inputs: Mapping[str, Any],
    text_only_inputs: Mapping[str, Any],
    candidate_token_ids: Mapping[str, Sequence[int]],
    family: str,
    *,
    attention_alpha: float = PAI_ATTENTION_ALPHA,
    guidance_scale: float = PAI_GUIDANCE_SCALE,
    start_layer: int = PAI_START_LAYER,
) -> TokenBaselineChoice:
    modules = configure_pai_attention(
        model,
        multimodal_inputs,
        family,
        alpha=attention_alpha,
        start_layer=start_layer,
        end_layer=None,
    )

    def before_branch(name: str) -> None:
        set_pai_cfg_branch(modules, name == "text_only")

    try:
        result = _choose_with_branches(
            model,
            {"native": multimodal_inputs, "text_only": text_only_inputs},
            candidate_token_ids,
            lambda logits: pai_adjust_logits(
                logits["native"],
                logits["text_only"],
                guidance_scale=guidance_scale,
            ),
            before_branch=before_branch,
        )
    finally:
        set_pai_cfg_branch(modules, False)
        for module in modules:
            module._pai_use_attention = False
    key, native_key, scores, native_scores, branches, prefix = result
    return TokenBaselineChoice(
        key=key,
        native_key=native_key,
        scores=scores,
        native_scores=native_scores,
        branch_scores=branches,
        diagnostics={
            "method": "pai",
            "repository": PAI_REPOSITORY,
            "commit": PAI_COMMIT,
            "execution": "full_vocabulary_token_level",
            "attention_alpha": float(attention_alpha),
            "guidance_scale": float(guidance_scale),
            "plausibility_beta": PAI_PLAUSIBILITY_BETA,
            "start_layer": int(start_layer),
            "end_layer": len(modules),
            "shared_candidate_prefix": list(prefix),
        },
    )


def _generate_with_branches(
    model: Any,
    processor: Any,
    branch_inputs: Mapping[str, Mapping[str, Any]],
    adjust: Callable[[Mapping[str, Any]], Any],
    *,
    method: str,
    max_new_tokens: int,
    stop_token_ids: Sequence[int] | None = None,
    before_branch: Callable[[str], None] | None = None,
    trace_steps: int = 8,
) -> TokenBaselineGeneration:
    """Run a released token rule over synchronized full-vocabulary streams."""
    import torch

    if int(max_new_tokens) <= 0:
        raise ValueError("max_new_tokens must be positive")
    tokenizer = getattr(processor, "tokenizer", processor)
    states = {
        name: _branch_state(_clone_inputs(inputs))
        for name, inputs in branch_inputs.items()
    }
    if any(int(ids.shape[0]) != 1 for ids, _ in states.values()):
        raise ValueError("token baseline generation currently supports batch size 1")

    eos_ids = _eos_token_ids(model, tokenizer)
    answer_ids = {
        int(token) for token in (stop_token_ids or ()) if token is not None
    }
    generated: List[int] = []
    traces: List[Dict[str, Any]] = []
    stopped_on_answer = False

    with torch.inference_mode():
        for step in range(int(max_new_tokens)):
            logits: Dict[str, Any] = {}
            updated: Dict[str, Dict[str, Any]] = {}
            for name, (ids, kwargs) in states.items():
                if before_branch is not None:
                    before_branch(name)
                branch_logits, branch_kwargs = _forward_branch(
                    model,
                    ids,
                    kwargs,
                    first_iteration=step == 0,
                )
                logits[name] = branch_logits
                updated[name] = branch_kwargs

            adjusted = adjust(logits)
            next_tokens = torch.argmax(adjusted, dim=-1)
            next_id = int(next_tokens[0].detach().cpu().item())
            native_id = int(
                torch.argmax(logits["native"][0]).detach().cpu().item()
            )
            generated.append(next_id)
            if step < max(0, int(trace_steps)):
                traces.append(
                    {
                        "step": step,
                        "native_token_id": native_id,
                        "native_token": tokenizer.decode([native_id]),
                        "adjusted_token_id": next_id,
                        "adjusted_token": tokenizer.decode([next_id]),
                        "changed_from_native": next_id != native_id,
                    }
                )

            for name, (ids, _) in states.items():
                token_column = next_tokens[:, None].to(ids.device)
                states[name] = (
                    torch.cat([ids, token_column], dim=-1),
                    updated[name],
                )
            if step == 0 and next_id in answer_ids:
                stopped_on_answer = True
                break
            if next_id in eos_ids:
                break

    text = tokenizer.decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    return TokenBaselineGeneration(
        token_ids=generated,
        text=text,
        diagnostics={
            "method": method,
            "execution": "full_vocabulary_token_level_generation",
            "decoding": "greedy",
            "shared_generated_prefix": True,
            "generated_token_count": len(generated),
            "answer_stop_token_ids": sorted(answer_ids),
            "stopped_on_answer": stopped_on_answer,
            "steps": traces,
        },
    )


def generate_native(
    model: Any,
    processor: Any,
    multimodal_inputs: Mapping[str, Any],
    *,
    max_new_tokens: int = 64,
    stop_token_ids: Sequence[int] | None = None,
    trace_steps: int = 8,
) -> TokenBaselineGeneration:
    """Run the matched native greedy decoder used as the baseline."""
    return _generate_with_branches(
        model,
        processor,
        {"native": multimodal_inputs},
        lambda logits: logits["native"],
        method="native",
        max_new_tokens=max_new_tokens,
        stop_token_ids=stop_token_ids,
        trace_steps=trace_steps,
    )


def generate_with_vcd(
    model: Any,
    processor: Any,
    multimodal_inputs: Mapping[str, Any],
    *,
    alpha: float = VCD_ALPHA,
    beta: float = VCD_BETA,
    noise_step: int = VCD_NOISE_STEP,
    max_new_tokens: int = 64,
    stop_token_ids: Sequence[int] | None = None,
    trace_steps: int = 8,
) -> TokenBaselineGeneration:
    """Generate with the released VCD transform and token equation."""
    distorted = build_vcd_inputs(
        multimodal_inputs,
        noise_step=noise_step,
        seed=None,
    )
    generation = _generate_with_branches(
        model,
        processor,
        {"native": multimodal_inputs, "distorted": distorted},
        lambda logits: vcd_adjust_logits(
            logits["native"],
            logits["distorted"],
            alpha=alpha,
            beta=beta,
        ),
        method="vcd",
        max_new_tokens=max_new_tokens,
        stop_token_ids=stop_token_ids,
        trace_steps=trace_steps,
    )
    generation.diagnostics.update(
        {
            "repository": VCD_REPOSITORY,
            "commit": VCD_COMMIT,
            "alpha": float(alpha),
            "beta": float(beta),
            "noise_step": int(noise_step),
            "seed": 42,
        }
    )
    return generation


def generate_with_mfcd(
    model: Any,
    processor: Any,
    multimodal_inputs: Mapping[str, Any],
    high_pass_inputs: Mapping[str, Any],
    low_pass_inputs: Mapping[str, Any],
    *,
    high_alpha: float = MFCD_HIGH_ALPHA,
    low_alpha: float = MFCD_LOW_ALPHA,
    beta: float = MFCD_BETA,
    max_new_tokens: int = 64,
    stop_token_ids: Sequence[int] | None = None,
    trace_steps: int = 8,
) -> TokenBaselineGeneration:
    """Generate with the released MFCD views and token equation."""
    generation = _generate_with_branches(
        model,
        processor,
        {
            "native": multimodal_inputs,
            "high_pass": high_pass_inputs,
            "low_pass": low_pass_inputs,
        },
        lambda logits: mfcd_adjust_logits(
            logits["native"],
            logits["high_pass"],
            logits["low_pass"],
            high_alpha=high_alpha,
            low_alpha=low_alpha,
            beta=beta,
        ),
        method="mfcd",
        max_new_tokens=max_new_tokens,
        stop_token_ids=stop_token_ids,
        trace_steps=trace_steps,
    )
    generation.diagnostics.update(
        {
            "repository": MFCD_REPOSITORY,
            "commit": MFCD_COMMIT,
            "high_alpha": float(high_alpha),
            "low_alpha": float(low_alpha),
            "beta": float(beta),
            "high_pass_cutoff": MFCD_HIGH_PASS_CUTOFF,
            "low_pass_cutoff": MFCD_LOW_PASS_CUTOFF,
            "filter": "gaussian",
            "jsd": False,
            "entropy": False,
        }
    )
    return generation


def generate_with_pai(
    model: Any,
    processor: Any,
    multimodal_inputs: Mapping[str, Any],
    text_only_inputs: Mapping[str, Any],
    family: str,
    *,
    attention_alpha: float = PAI_ATTENTION_ALPHA,
    guidance_scale: float = PAI_GUIDANCE_SCALE,
    start_layer: int = PAI_START_LAYER,
    max_new_tokens: int = 64,
    stop_token_ids: Sequence[int] | None = None,
    trace_steps: int = 8,
) -> TokenBaselineGeneration:
    """Generate with PAI's released attention edit and CFG token equation."""
    modules = configure_pai_attention(
        model,
        multimodal_inputs,
        family,
        alpha=attention_alpha,
        start_layer=start_layer,
        end_layer=None,
    )

    def before_branch(name: str) -> None:
        set_pai_cfg_branch(modules, name == "text_only")

    try:
        generation = _generate_with_branches(
            model,
            processor,
            {"native": multimodal_inputs, "text_only": text_only_inputs},
            lambda logits: pai_adjust_logits(
                logits["native"],
                logits["text_only"],
                guidance_scale=guidance_scale,
            ),
            method="pai",
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            before_branch=before_branch,
            trace_steps=trace_steps,
        )
    finally:
        set_pai_cfg_branch(modules, False)
        for module in modules:
            module._pai_use_attention = False
    generation.diagnostics.update(
        {
            "repository": PAI_REPOSITORY,
            "commit": PAI_COMMIT,
            "attention_alpha": float(attention_alpha),
            "guidance_scale": float(guidance_scale),
            "plausibility_beta": PAI_PLAUSIBILITY_BETA,
            "start_layer": int(start_layer),
            "end_layer": len(modules),
        }
    )
    return generation
