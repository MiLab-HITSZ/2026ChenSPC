#!/usr/bin/env python3
"""Lean Qwen3-VL/LLaVA runtime for frozen external CPRC evaluations."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ModelSpec:
    name: str
    backend: str
    model: str
    temperature: float
    max_tokens: int
    models_root: str
    attn_implementation: str = ""
    quantization: str = "auto"
    dtype: str = "auto"


def load_model_specs(models_arg: str) -> List[ModelSpec]:
    path = Path(models_arg)
    raw = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else json.loads(models_arg)
    if not isinstance(raw, list):
        raise ValueError("--models must contain a JSON list")
    specs = []
    for entry in raw:
        backend = str(entry.get("backend") or "").lower().strip()
        if backend in {"hf", "transformer", "local_transformers"}:
            backend = "transformers"
        if backend != "transformers":
            raise ValueError("the frozen external runtime supports transformers models only")
        quantization = str(entry.get("quantization") or "auto").lower()
        dtype = str(entry.get("dtype") or "auto").lower()
        if quantization not in {"auto", "none", "4bit"}:
            raise ValueError(f"unknown quantization: {quantization}")
        if dtype not in {"auto", "float16", "bfloat16", "float32"}:
            raise ValueError(f"unknown dtype: {dtype}")
        model = str(entry.get("model") or "")
        if not model:
            raise ValueError("model is required")
        specs.append(
            ModelSpec(
                name=str(entry.get("name") or model),
                backend=backend,
                model=model,
                temperature=float(entry.get("temperature", 0.0)),
                max_tokens=int(entry.get("max_tokens", 64)),
                models_root=str(
                    entry.get("models_root")
                    or os.environ.get("CDH_MODELS_ROOT")
                    or "models"
                ),
                attn_implementation=str(
                    entry.get("attn_implementation")
                    or entry.get("attention_implementation")
                    or ""
                ),
                quantization=quantization,
                dtype=dtype,
            )
        )
    return specs


def strip_thinking(text: str) -> str:
    value = str(text or "")
    return value.split("</think>")[-1].strip() if "</think>" in value else value.strip()


def normalize_text(text: str) -> str:
    value = strip_thinking(text).lower()
    value = "".join(character for character in value if character.isalnum() or character.isspace())
    return " ".join(value.split())


def extract_yes_no(text: str) -> Optional[str]:
    value = normalize_text(text)
    if not value:
        return None
    words = value.split()
    if words[0] in {"yes", "y", "true", "是", "对"}:
        return "yes"
    if words[0] in {"no", "n", "false", "否", "不", "不是", "错"}:
        return "no"
    if "yes" in words or "是" in value or "对" in value:
        return "yes"
    if "no" in words or "否" in value or "不" in value:
        return "no"
    return None


def extract_first_letter(text: str) -> Optional[str]:
    value = strip_thinking(text).upper()
    patterns = [
        r"^([A-D])(?:\.|\)|$|\s)",
        r"(?:FINAL\s+ANSWER|FINAL|最终答案|答案)(?:\s*(?:IS|是|为|:|-))?\s*([A-D])",
        r"(?:ANSWER|答案)(?:IS|是|为)?\s*([A-D])",
        r"\s([A-D])(?:\.|\)|$|\s)",
        r"\b([A-D])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            return match.group(1)
    return None


def candidate_key(spec: ModelSpec, task: str, prediction: str) -> Optional[str]:
    if "thinking" in f"{spec.name} {spec.model}".lower() and "</think>" not in str(prediction):
        return None
    if task == "qa":
        return extract_yes_no(prediction)
    if task == "mc":
        return extract_first_letter(prediction)
    raise ValueError(f"unknown task: {task}")


def score_answer(task: str, prediction: str, ground_truth: str) -> bool:
    if task == "qa":
        return extract_yes_no(prediction) == str(ground_truth).strip().lower()
    if task == "mc":
        return extract_first_letter(prediction) == str(ground_truth).strip().upper()
    return str(prediction).strip() == str(ground_truth).strip()


def build_user_text(question: str) -> str:
    return f"{question}\nAnswer with yes or no."


def build_candidate_prompt(question: str) -> str:
    return f"{question}\nAnswer with yes or no.\nFinal answer:"


_MODEL_CACHE: Dict[str, Tuple[Any, Any, str]] = {}
_REVIS_MODEL_CACHE: Dict[str, Tuple[Any, Any, str]] = {}
_REVIS_ASSET_CACHE: Dict[str, Tuple[Dict[str, Any], Any]] = {}


def qwen3_vl_model_class(model_type: str, transformers_module: Any) -> Any:
    """Resolve the official dense or sparse Qwen3-VL class from config metadata."""
    classes = {
        "qwen3_vl": getattr(
            transformers_module, "Qwen3VLForConditionalGeneration", None
        ),
        "qwen3_vl_moe": getattr(
            transformers_module, "Qwen3VLMoeForConditionalGeneration", None
        ),
    }
    model_class = classes.get(str(model_type))
    if model_class is None:
        raise ValueError(f"unsupported Qwen3-VL model_type: {model_type!r}")
    return model_class


def local_candidate_bundle(spec: ModelSpec) -> Tuple[Any, Any, str]:
    cache_key = (
        f"{spec.models_root}::{spec.model}::attn={spec.attn_implementation}"
        f"::quant={spec.quantization}::dtype={spec.dtype}"
    )
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    model_path = Path(spec.models_root) / spec.model
    load_id = str(model_path) if model_path.exists() else spec.model
    if "llava" in f"{spec.name} {spec.model}".lower():
        from official_revis_adapter import load_model_bundle

        bundle = load_model_bundle(load_id)
        _MODEL_CACHE[cache_key] = bundle
        return bundle

    import importlib.util
    import torch
    import transformers
    from transformers import AutoConfig, AutoProcessor

    quantization_config = None
    use_4bit = spec.quantization == "4bit" or (
        spec.quantization == "auto" and ("32B" in spec.model or "235B" in spec.model)
    )
    if use_4bit and importlib.util.find_spec("bitsandbytes") is not None:
        from transformers import BitsAndBytesConfig

        compute_dtype = torch.float16
        if spec.dtype == "bfloat16":
            compute_dtype = torch.bfloat16
        elif spec.dtype == "float32":
            compute_dtype = torch.float32
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model_dtype: Any = "auto" if spec.dtype == "auto" else getattr(torch, spec.dtype)
    load_kwargs: Dict[str, Any] = {
        "dtype": model_dtype,
        "device_map": "auto",
        "quantization_config": quantization_config,
    }
    if spec.attn_implementation:
        load_kwargs["attn_implementation"] = spec.attn_implementation
    model_type = str(AutoConfig.from_pretrained(load_id).model_type)
    model_class = qwen3_vl_model_class(model_type, transformers)
    model = model_class.from_pretrained(load_id, **load_kwargs)
    processor = AutoProcessor.from_pretrained(load_id)
    bundle = (model, processor, "qwen3_vl")
    _MODEL_CACHE[cache_key] = bundle
    return bundle


def candidate_vlm_inputs(
    model: Any,
    processor: Any,
    family: str,
    prompt: str,
    image: Optional[Any],
) -> Dict[str, Any]:
    if family == "llava_next":
        from official_revis_adapter import build_inputs

        return build_inputs(model, processor, family, prompt, image)
    if family != "qwen3_vl":
        raise ValueError(f"unsupported frozen external runtime family: {family}")
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return {
        key: (value.to(model.device) if hasattr(value, "to") else value)
        for key, value in inputs.items()
    }


def candidate_logprob(
    model: Any,
    processor: Any,
    prompt: str,
    image: Optional[Any],
    candidate_text: str,
    family: str = "qwen3_vl",
) -> Dict[str, Any]:
    """Teacher-force one candidate continuation after a shared VLM prompt."""
    import torch

    tokenizer = getattr(processor, "tokenizer", processor)
    prompt_inputs = candidate_vlm_inputs(model, processor, family, prompt, image)
    candidate_ids = tokenizer(
        candidate_text,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(prompt_inputs["input_ids"].device)
    if candidate_ids.shape[-1] == 0:
        raise ValueError("empty candidate tokenization")

    prompt_ids = prompt_inputs["input_ids"]
    full_inputs: Dict[str, Any] = {}
    for key, value in prompt_inputs.items():
        if key == "input_ids":
            full_inputs[key] = torch.cat([prompt_ids, candidate_ids], dim=-1)
        elif key == "attention_mask":
            extension = torch.ones(
                (value.shape[0], candidate_ids.shape[-1]),
                dtype=value.dtype,
                device=value.device,
            )
            full_inputs[key] = torch.cat([value, extension], dim=-1)
        else:
            full_inputs[key] = value

    with torch.inference_mode():
        outputs = model(**full_inputs, use_cache=False)
        log_probs = torch.log_softmax(outputs.logits.float(), dim=-1)
    prompt_length = int(prompt_ids.shape[-1])
    token_logprobs = [
        float(
            log_probs[0, prompt_length + index - 1, int(token_id)]
            .detach()
            .cpu()
            .item()
        )
        for index, token_id in enumerate(candidate_ids[0])
    ]
    total = float(sum(token_logprobs))
    return {
        "text": candidate_text,
        "token_ids": [int(value) for value in candidate_ids[0].detach().cpu().tolist()],
        "token_logprobs": token_logprobs,
        "logprob": total,
        "avg_logprob": total / len(token_logprobs),
        "token_count": len(token_logprobs),
    }


def candidate_logprobs_shared_prefix(
    model: Any,
    processor: Any,
    prompt: str,
    image: Optional[Any],
    candidate_texts: Sequence[str],
    family: str = "qwen3_vl",
) -> List[Dict[str, Any]]:
    """Score candidate branches in one forward when they share all but one token."""
    import torch

    tokenizer = getattr(processor, "tokenizer", processor)
    token_ids = [
        tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        for text in candidate_texts
    ]
    if not token_ids or any(ids.numel() == 0 for ids in token_ids):
        raise ValueError("empty candidate tokenization")

    common_length = 0
    shortest = min(int(ids.numel()) for ids in token_ids)
    while common_length < shortest:
        value = int(token_ids[0][common_length])
        if any(int(ids[common_length]) != value for ids in token_ids[1:]):
            break
        common_length += 1

    if any(int(ids.numel()) != common_length + 1 for ids in token_ids):
        return [
            candidate_logprob(
                model,
                processor,
                prompt,
                image,
                text,
                family=family,
            )
            for text in candidate_texts
        ]

    prompt_inputs = candidate_vlm_inputs(model, processor, family, prompt, image)
    prompt_ids = prompt_inputs["input_ids"]
    device = prompt_ids.device
    prefix_ids = token_ids[0][:common_length].unsqueeze(0).to(device)
    full_inputs: Dict[str, Any] = {}
    for key, value in prompt_inputs.items():
        if key == "input_ids":
            full_inputs[key] = torch.cat([prompt_ids, prefix_ids], dim=-1)
        elif key == "attention_mask" and common_length:
            extension = torch.ones(
                (value.shape[0], common_length),
                dtype=value.dtype,
                device=value.device,
            )
            full_inputs[key] = torch.cat([value, extension], dim=-1)
        else:
            full_inputs[key] = value

    with torch.inference_mode():
        outputs = model(**full_inputs, use_cache=False)
        log_probs = torch.log_softmax(outputs.logits.float(), dim=-1)

    prompt_length = int(prompt_ids.shape[-1])
    shared_logprobs = [
        float(
            log_probs[0, prompt_length + index - 1, int(prefix_ids[0, index])]
            .detach()
            .cpu()
            .item()
        )
        for index in range(common_length)
    ]
    branch_position = prompt_length + common_length - 1
    scores: List[Dict[str, Any]] = []
    for text, ids in zip(candidate_texts, token_ids):
        branch_id = int(ids[common_length])
        branch_logprob = float(
            log_probs[0, branch_position, branch_id].detach().cpu().item()
        )
        values = [*shared_logprobs, branch_logprob]
        total = float(sum(values))
        scores.append(
            {
                "text": text,
                "token_ids": [int(value) for value in ids.tolist()],
                "token_logprobs": values,
                "logprob": total,
                "avg_logprob": total / len(values),
                "token_count": len(values),
            }
        )
    return scores


def generate_qa(
    spec: ModelSpec,
    question: str,
    image_path: str,
    retry: int = 2,
) -> Tuple[str, str, Dict[str, Any], int]:
    from PIL import Image

    started = time.time()
    last_error = ""
    for _ in range(max(1, retry + 1)):
        try:
            import torch

            model, processor, family = local_candidate_bundle(spec)
            image = Image.open(image_path).convert("RGB")
            image = image.resize((512, 512), Image.Resampling.LANCZOS)
            inputs = candidate_vlm_inputs(
                model, processor, family, build_user_text(question), image
            )
            generation_kwargs: Dict[str, Any] = {
                "max_new_tokens": int(spec.max_tokens),
                "do_sample": False,
            }
            with torch.inference_mode():
                generated = model.generate(**inputs, **generation_kwargs)
            width = int(inputs["input_ids"].shape[1])
            text = processor.decode(generated[0, width:], skip_special_tokens=True).strip()
            return (
                "ok",
                text,
                {
                    "backend": "local_transformers_qwen3_vl",
                    "model": spec.model,
                    "image_preprocessing": "fixed_512",
                },
                int((time.time() - started) * 1000),
            )
        except Exception as error:
            last_error = str(error)
    return "error", last_error, {}, int((time.time() - started) * 1000)


def generate_official_revis(
    spec: ModelSpec,
    question: str,
    image_path: str,
    vector_file: str,
    calibration_file: str,
    repo_root: str,
    steering: bool,
    alpha_visual: float,
    risk_gamma: float = 1.0,
) -> Tuple[str, Dict[str, Any]]:
    import torch
    from PIL import Image

    from official_revis_adapter import (
        attach_official_hook,
        build_inputs,
        load_model_bundle,
        resolve_revis_paper_alpha,
    )

    model_path = Path(spec.models_root) / spec.model
    load_id = str(model_path) if model_path.exists() else spec.model
    model_key = str(Path(load_id).resolve()) if Path(load_id).exists() else load_id
    bundle = _REVIS_MODEL_CACHE.get(model_key)
    if bundle is None:
        bundle = load_model_bundle(load_id)
        _REVIS_MODEL_CACHE[model_key] = bundle
    model, processor, family = bundle
    resolved_alpha = (
        float(alpha_visual)
        if float(alpha_visual) > 0
        else resolve_revis_paper_alpha(family)
    )

    asset_key = f"{Path(vector_file).resolve()}::{Path(calibration_file).resolve()}"
    assets = _REVIS_ASSET_CACHE.get(asset_key)
    if assets is None:
        calibration = json.loads(Path(calibration_file).read_text(encoding="utf-8"))
        vectors = torch.load(vector_file, map_location="cpu", weights_only=True)
        assets = (calibration, vectors)
        _REVIS_ASSET_CACHE[asset_key] = assets
    calibration, vectors = assets
    selected = calibration.get("selected") or calibration
    if not isinstance(selected, dict) or selected.get("layer") is None or selected.get("tau") is None:
        raise ValueError(f"invalid official REVIS calibration: {calibration_file}")
    layer = int(selected["layer"])
    tau = float(selected["tau"])

    image = Image.open(image_path).convert("RGB")
    inputs = build_inputs(
        model,
        processor,
        family,
        build_user_text(question),
        image,
    )
    hook = None
    if steering:
        hook = attach_official_hook(
            model,
            vectors,
            layer,
            tau,
            Path(repo_root),
            alpha_visual=resolved_alpha,
            risk_gamma=float(risk_gamma),
        )
    try:
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(spec.max_tokens),
            "do_sample": bool(float(spec.temperature) > 0),
        }
        if float(spec.temperature) > 0:
            generation_kwargs["temperature"] = float(spec.temperature)
        with torch.inference_mode():
            output = model.generate(**inputs, **generation_kwargs)
    finally:
        if hook is not None:
            hook.detach()
    width = int(inputs["input_ids"].shape[1])
    text = processor.decode(output[0, width:], skip_special_tokens=True).strip()
    observation = dict(getattr(hook, "observation", {})) if hook is not None else {}
    if observation.get("positions"):
        observation["trigger_rate"] = (
            observation["triggered_positions"] / observation["positions"]
        )
        observation["risk_mean"] = observation["risk_sum"] / observation["positions"]
    return text, {
        "backend": "official_revis_compat",
        "model": spec.model,
        "family": family,
        "image_preprocessing": "model_native",
        "steering": bool(steering),
        "vector_file": vector_file,
        "calibration_file": calibration_file,
        "official_parameters": {
            "layer": layer,
            "vector_index": layer + 1,
            "tau_low": tau,
            "alpha_visual": resolved_alpha,
            "risk_gamma": float(risk_gamma),
        },
        "mechanism": "official token-level risk-gated visual-vector injection",
        "gate_observation": observation,
    }
