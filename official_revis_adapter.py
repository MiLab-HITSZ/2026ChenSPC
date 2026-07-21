#!/usr/bin/env python3
"""Auditable compatibility layer for the pinned official REVIS implementation.

The released Qwen3 path is incomplete. This module keeps the released vector,
risk, calibration, and intervention equations while supplying model-specific
Hugging Face input construction for Qwen3-VL and LLaVA-NeXT.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


OFFICIAL_REVIS_COMMIT = "75f4d29d402d4c1c4dbb2c69e88e3aa50cd0c9cc"
REVIS_PAPER_QWEN_ALPHA_VISUAL = 1.6
REVIS_PAPER_LLAVA_ALPHA_VISUAL = 1.1
REVIS_REPOSITORY_DEFAULT_ALPHA_VISUAL = 2.5
# Backward-compatible name for artifacts produced before paper/repository
# parameter profiles were separated.
OFFICIAL_ALPHA_VISUAL = REVIS_REPOSITORY_DEFAULT_ALPHA_VISUAL
OFFICIAL_RISK_GAMMA = 1.0
OFFICIAL_CALIBRATION_PERCENTILE = 85.0
OFFICIAL_CALIBRATION_MAX_SAMPLES = 300
OFFICIAL_MIN_LAYER = 10
UNKNOWN_RESPONSES = (
    "I cannot answer this question because there is no image provided.",
    "Without an image, I cannot describe the scene.",
    "Please provide an image so I can describe it.",
    "I don't see any image here.",
    "The visual information is missing.",
)


@dataclass(frozen=True)
class OfficialRevisProtocol:
    repo_commit: str = OFFICIAL_REVIS_COMMIT
    alpha_visual: float = OFFICIAL_ALPHA_VISUAL
    risk_gamma: float = OFFICIAL_RISK_GAMMA
    calibration_percentile: float = OFFICIAL_CALIBRATION_PERCENTILE
    calibration_max_samples: int = OFFICIAL_CALIBRATION_MAX_SAMPLES
    min_layer: int = OFFICIAL_MIN_LAYER
    extraction_samples: int = 100
    parameter_source: str = "official repository defaults and layer_select.sh"


def resolve_revis_paper_alpha(family: str) -> float:
    if family == "llava_next":
        return REVIS_PAPER_LLAVA_ALPHA_VISUAL
    if family == "qwen3_vl":
        return REVIS_PAPER_QWEN_ALPHA_VISUAL
    raise ValueError(f"no official REVIS paper alpha profile for family: {family}")


def verify_official_snapshot(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if commit != OFFICIAL_REVIS_COMMIT:
        raise RuntimeError(f"REVIS snapshot mismatch: expected {OFFICIAL_REVIS_COMMIT}, got {commit}")
    return commit


def official_risk(hidden: Any, visual_vector: Any, gamma: float = OFFICIAL_RISK_GAMMA) -> Any:
    import torch.nn.functional as functional

    hidden_unit = functional.normalize(hidden, dim=-1)
    vector_unit = functional.normalize(
        visual_vector.to(device=hidden.device, dtype=hidden.dtype),
        dim=-1,
    )
    return -(float(gamma) * (hidden_unit @ vector_unit))


def select_official_layer(
    records_by_layer: Mapping[int, Iterable[Mapping[str, Any]]],
    percentile: float = OFFICIAL_CALIBRATION_PERCENTILE,
    min_layer: int = OFFICIAL_MIN_LAYER,
) -> dict[str, Any]:
    import numpy as np

    diagnostics: list[dict[str, Any]] = []
    for layer in sorted((int(value) for value in records_by_layer), reverse=True):
        if layer < int(min_layer):
            continue
        rows = list(records_by_layer[layer])
        factual = [float(row["risk"]) for row in rows if row["category"] in {"TP", "TN"}]
        hallucinated = [float(row["risk"]) for row in rows if row["category"] == "FP"]
        if not factual or not hallucinated:
            diagnostics.append({"layer": layer, "separability": None, "tau": None})
            continue
        separability = float(np.mean(hallucinated) - np.mean(factual))
        tau = float(np.percentile(factual, float(percentile)))
        row = {
            "layer": layer,
            "vector_index": layer + 1,
            "separability": separability,
            "tau": tau,
            "n_factual": len(factual),
            "n_hallucinated": len(hallucinated),
        }
        diagnostics.append(row)
        if separability > 0:
            return {"selected": row, "diagnostics": diagnostics}
    return {"selected": None, "diagnostics": diagnostics}


def _official_modules(repo_root: Path) -> tuple[Any, Any, Any]:
    verify_official_snapshot(repo_root)
    root = str(repo_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from steering_utils.llm_layers import MADRSteeringHook_visual_only, get_llm_layers_module
    from steering_utils.vector_calculator import get_pure_direction_with_pca

    return MADRSteeringHook_visual_only, get_llm_layers_module, get_pure_direction_with_pca


def load_model_bundle(model_path: str) -> tuple[Any, Any, str]:
    import torch
    from transformers import (
        AutoProcessor,
        LlavaNextForConditionalGeneration,
        Qwen3VLForConditionalGeneration,
    )

    lowered = model_path.lower()
    if "llava" in lowered and ("1.6" in lowered or "v1.6" in lowered or "next" in lowered):
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        family = "llava_next"
    elif "qwen3" in lowered:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        family = "qwen3_vl"
    else:
        raise ValueError(f"unsupported official REVIS adapter model: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path)
    model.eval()
    return model, processor, family


def _input_device(model: Any) -> Any:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def build_inputs(
    model: Any,
    processor: Any,
    family: str,
    question: str,
    image: Optional[Any],
    answer: Optional[str] = None,
) -> dict[str, Any]:
    has_image = image is not None
    content: list[dict[str, Any]] = []
    if has_image:
        content.append({"type": "image", "image": image} if family == "qwen3_vl" else {"type": "image"})
    content.append({"type": "text", "text": question})
    messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})

    if family == "qwen3_vl":
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=answer is None,
            return_dict=True,
            return_tensors="pt",
        )
    else:
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=answer is None,
        )
        inputs = processor(
            text=[text],
            images=[image] if has_image else None,
            padding=True,
            return_tensors="pt",
        )
    device = _input_device(model)
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def build_retrospective_inputs(
    model: Any,
    processor: Any,
    family: str,
    question: str,
    answer: str,
    image: Optional[Any],
) -> dict[str, Any]:
    """Build extraction inputs while preserving REVIS's released LLaVA template."""
    if family != "llava_next":
        return build_inputs(model, processor, family, question, image, answer=answer)

    content: list[dict[str, Any]] = []
    if image is not None:
        content.append({"type": "image"})
    content.append({"type": "text", "text": question})
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    # Official REVIS appends the retrospective answer after the generation
    # prefix, without rendering an assistant end-of-turn token.
    text = f"{prompt} {answer}"
    inputs = processor(
        text=[text],
        images=[image] if image is not None else None,
        padding=True,
        return_tensors="pt",
    )
    device = _input_device(model)
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def retrospective_states(
    model: Any,
    processor: Any,
    family: str,
    question: str,
    answer: str,
    image: Optional[Any],
) -> Any:
    import torch

    inputs = build_retrospective_inputs(model, processor, family, question, answer, image)
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    return torch.stack(
        [state[0, -1, :].detach().cpu().float() for state in outputs.hidden_states]
    )


def compute_official_vectors(
    model: Any,
    processor: Any,
    family: str,
    annotation_file: Path,
    image_dir: Path,
    repo_root: Path,
    seed: int = 0,
    max_samples: int = 0,
) -> dict[int, dict[str, Any]]:
    import torch
    import torch.nn.functional as functional
    from PIL import Image

    _, _, official_pca = _official_modules(repo_root)
    payload = json.loads(annotation_file.read_text(encoding="utf-8"))
    samples = payload["annotations"] if isinstance(payload, dict) else payload
    rng = random.Random(seed)
    collected: dict[str, list[list[Any]]] | None = None
    used = 0

    for sample in samples[:max_samples or None]:
        image_path = image_dir / f"{sample['image_id']}.jpg"
        if not image_path.is_file():
            continue
        image = Image.open(image_path).convert("RGB")
        question = str(sample.get("question") or "Describe the image in detail.")
        factual = str(sample["caption"])
        hallucinated = str(sample["h_caption"])
        unknown = rng.choice(UNKNOWN_RESPONSES)
        states = {
            "image_factual": retrospective_states(model, processor, family, question, factual, image),
            "image_hallucinated": retrospective_states(model, processor, family, question, hallucinated, image),
            "prior_factual": retrospective_states(model, processor, family, question, factual, None),
            "prior_hallucinated": retrospective_states(model, processor, family, question, hallucinated, None),
            "prior_unknown": retrospective_states(model, processor, family, question, unknown, None),
        }
        if collected is None:
            layer_count = states["image_factual"].shape[0]
            collected = {
                key: [[] for _ in range(layer_count)]
                for key in ("truth", "language", "prior", "visual_raw")
            }
        for layer in range(states["image_factual"].shape[0]):
            collected["truth"][layer].append(states["image_factual"][layer] - states["image_hallucinated"][layer])
            collected["language"][layer].append(states["prior_unknown"][layer] - states["prior_hallucinated"][layer])
            collected["prior"][layer].append(states["prior_hallucinated"][layer] - states["prior_unknown"][layer])
            collected["visual_raw"][layer].append(states["image_factual"][layer] - states["prior_factual"][layer])
        used += 1

    if collected is None or used < 2:
        raise RuntimeError(f"official REVIS vector extraction needs at least two images, got {used}")

    vectors: dict[int, dict[str, Any]] = {}
    for layer in range(len(collected["truth"])):
        truth = torch.stack(collected["truth"][layer])
        language = torch.stack(collected["language"][layer])
        prior = torch.stack(collected["prior"][layer])
        visual_raw = torch.stack(collected["visual_raw"][layer])
        prior_direction = official_pca(prior)
        prior_unit = functional.normalize(prior_direction, dim=-1)
        projections = (visual_raw @ prior_unit.unsqueeze(-1)) * prior_unit.unsqueeze(0)
        vectors[layer] = {
            "truth": official_pca(truth),
            "language": official_pca(language),
            "visual": official_pca(visual_raw - projections),
        }
    return vectors


def _parse_yes_no(text: str) -> str:
    cleaned = text.lower().strip()
    yes = cleaned.find("yes")
    no = cleaned.find("no")
    if yes >= 0 and (no < 0 or yes < no):
        return "yes"
    if no >= 0:
        return "no"
    return "unknown"


def calibrate_official_protocol(
    model: Any,
    processor: Any,
    family: str,
    vectors: Mapping[int, Mapping[str, Any]],
    question_file: Path,
    image_dir: Path,
    repo_root: Path,
    max_samples: int = OFFICIAL_CALIBRATION_MAX_SAMPLES,
    percentile: float = OFFICIAL_CALIBRATION_PERCENTILE,
    min_layer: int = OFFICIAL_MIN_LAYER,
) -> dict[str, Any]:
    import torch
    from PIL import Image

    _, get_layers, _ = _official_modules(repo_root)
    layers = get_layers(model)
    records: dict[int, list[dict[str, Any]]] = {index: [] for index in range(len(layers))}
    questions = []
    with question_file.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= max_samples:
                break
            questions.append(json.loads(line))

    for item in questions:
        captures: dict[int, float] = {}
        handles = []
        for layer_index, layer in enumerate(layers):
            vector_index = layer_index + 1
            if vector_index not in vectors:
                continue

            def hook(_module: Any, _inputs: Any, output: Any, idx: int = layer_index, vidx: int = vector_index) -> Any:
                if idx not in captures:
                    hidden = output[0] if isinstance(output, tuple) else output
                    captures[idx] = float(official_risk(hidden[:, -1, :], vectors[vidx]["visual"]).item())
                return output

            handles.append(layer.register_forward_hook(hook))
        try:
            image = Image.open(image_dir / str(item["image"])).convert("RGB")
            prompt = f"{item['text']} Please answer yes or no."
            inputs = build_inputs(model, processor, family, prompt, image)
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=5, do_sample=False)
            generated = output[0, inputs["input_ids"].shape[1] :]
            prediction = _parse_yes_no(processor.decode(generated, skip_special_tokens=True))
        finally:
            for handle in handles:
                handle.remove()
        label = str(item["label"]).lower()
        category = "Other"
        if label == "yes" and prediction == "yes":
            category = "TP"
        elif label == "no" and prediction == "yes":
            category = "FP"
        elif label == "no" and prediction == "no":
            category = "TN"
        for layer_index, risk in captures.items():
            records[layer_index].append({"risk": risk, "category": category})

    selected = select_official_layer(records, percentile=percentile, min_layer=min_layer)
    selected["protocol"] = asdict(OfficialRevisProtocol())
    selected["calibration_rows"] = len(questions)
    return selected


def attach_official_hook(
    model: Any,
    vectors: Mapping[int, Mapping[str, Any]],
    layer: int,
    tau: float,
    repo_root: Path,
    alpha_visual: float = OFFICIAL_ALPHA_VISUAL,
    risk_gamma: float = OFFICIAL_RISK_GAMMA,
) -> Any:
    hook_type, get_layers, _ = _official_modules(repo_root)
    vector_index = int(layer) + 1
    if vector_index not in vectors:
        raise KeyError(f"missing official REVIS vector index {vector_index}")
    hook = hook_type(
        vectors=vectors[vector_index],
        layer_idx=int(layer),
        alpha_vis=float(alpha_visual),
        tau_low=float(tau),
        risk_gamma=float(risk_gamma),
    )
    observation = {
        "forward_calls": 0,
        "positions": 0,
        "triggered_positions": 0,
        "risk_sum": 0.0,
        "risk_min": None,
        "risk_max": None,
    }
    original_compute_risk = hook._compute_risk

    def observed_compute_risk(hidden: Any) -> Any:
        risk = original_compute_risk(hidden)
        detached = risk.detach().float()
        count = int(detached.numel())
        observation["forward_calls"] += 1
        observation["positions"] += count
        observation["triggered_positions"] += int((detached > float(tau)).sum().item())
        observation["risk_sum"] += float(detached.sum().item())
        current_min = float(detached.min().item())
        current_max = float(detached.max().item())
        observation["risk_min"] = current_min if observation["risk_min"] is None else min(observation["risk_min"], current_min)
        observation["risk_max"] = current_max if observation["risk_max"] is None else max(observation["risk_max"], current_max)
        return risk

    hook._compute_risk = observed_compute_risk
    hook.observation = observation
    hook.attach(get_layers(model)[int(layer)])
    return hook


def _save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revis-root", default="downloads/hallucination_refs/REVIS")
    subparsers = parser.add_subparsers(dest="command", required=True)

    protocol = subparsers.add_parser("protocol")
    protocol.add_argument("--output", default="")

    compute = subparsers.add_parser("compute-vectors")
    compute.add_argument("--model-path", required=True)
    compute.add_argument("--output", required=True)
    compute.add_argument("--seed", type=int, default=0)
    compute.add_argument("--max-samples", type=int, default=0)

    calibrate = subparsers.add_parser("calibrate")
    calibrate.add_argument("--model-path", required=True)
    calibrate.add_argument("--vector-file", required=True)
    calibrate.add_argument("--output", required=True)
    calibrate.add_argument("--max-samples", type=int, default=OFFICIAL_CALIBRATION_MAX_SAMPLES)
    calibrate.add_argument("--percentile", type=float, default=OFFICIAL_CALIBRATION_PERCENTILE)
    calibrate.add_argument("--min-layer", type=int, default=OFFICIAL_MIN_LAYER)
    args = parser.parse_args()

    repo_root = Path(args.revis_root)
    verify_official_snapshot(repo_root)
    if args.command == "protocol":
        payload = asdict(OfficialRevisProtocol())
        if args.output:
            _save_json(Path(args.output), payload)
        print(json.dumps(payload, indent=2))
        return 0

    import torch

    model, processor, family = load_model_bundle(args.model_path)
    if args.command == "compute-vectors":
        vectors = compute_official_vectors(
            model,
            processor,
            family,
            repo_root / "data/OH_sampled_data_100.json",
            repo_root / "data/OH_sampled_images_100",
            repo_root,
            seed=args.seed,
            max_samples=args.max_samples,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(vectors, output)
        _save_json(output.with_suffix(output.suffix + ".meta.json"), {
            "protocol": asdict(OfficialRevisProtocol()),
            "model_path": args.model_path,
            "family": family,
            "seed": args.seed,
            "extraction_samples": args.max_samples or OfficialRevisProtocol().extraction_samples,
            "vector_indices": sorted(vectors),
        })
        return 0

    vectors = torch.load(args.vector_file, map_location="cpu", weights_only=True)
    result = calibrate_official_protocol(
        model,
        processor,
        family,
        vectors,
        repo_root / "data/coco_pope_calibration_merged.jsonl",
        repo_root / "data/coco_pope_train_risk/mini_train2014",
        repo_root,
        max_samples=args.max_samples,
        percentile=args.percentile,
        min_layer=args.min_layer,
    )
    _save_json(Path(args.output), result)
    print(json.dumps(result["selected"], indent=2))
    return 0 if result["selected"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
