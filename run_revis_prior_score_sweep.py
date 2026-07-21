#!/usr/bin/env python3
"""Run a small REVIS prior-score sweep while reusing the local model cache."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import evaluate_cdh_bench


def _lambda_slug(value: float) -> str:
    text = f"{value:g}".replace("-", "m").replace(".", "p")
    return f"lam{text}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="CDH-Bench.cdh_gen.claims.jsonl")
    ap.add_argument("--images-root", default="images")
    ap.add_argument("--models", default="configs/qwen_32b_instruct_transformers_eager_maxtok24.json")
    ap.add_argument("--output-root", default="result/cdh_revis_prior_score_sweep_pair1_32b_instruct")
    ap.add_argument("--subcategories", default="Body Parts")
    ap.add_argument("--limit-per-subcategory", type=int, default=1)
    ap.add_argument("--lambdas", default="0.5,1.0,1.5,2.0,3.0")
    ap.add_argument(
        "--prior-score",
        choices=("logprob", "centered_logit", "zscore_logit", "positive_zscore_logit"),
        default="zscore_logit",
    )
    ap.add_argument("--prior-source", choices=("no_image", "degraded_image"), default="no_image")
    ap.add_argument("--prior-degrade-mode", default="blur_downsample")
    ap.add_argument("--gate", choices=("dynamic", "always"), default="dynamic")
    ap.add_argument("--prior-inertia-gate", choices=("none", "agreement"), default="agreement")
    ap.add_argument("--prior-margin", type=float, default=0.0)
    ap.add_argument("--prior-inertia-prob-min", type=float, default=0.5)
    ap.add_argument("--prior-inertia-logprob-margin", type=float, default=0.25)
    ap.add_argument("--prior-subspace-alphas", default="")
    ap.add_argument("--prior-subspace-top-k", type=int, default=8)
    ap.add_argument("--layer-prior-subspace-alphas", default="")
    ap.add_argument("--layer-prior-subspace-top-k", type=int, default=1)
    ap.add_argument("--layer-prior-subspace-index", type=int, default=-1)
    ap.add_argument("--layer-prior-subspace-fraction", type=float, default=0.5)
    ap.add_argument("--attention-prior-alphas", default="")
    ap.add_argument("--attention-prior-layer-index", type=int, default=-1)
    ap.add_argument("--attention-prior-layer-fraction", type=float, default=0.9)
    ap.add_argument("--attention-prior-head-top-k", type=int, default=0)
    ap.add_argument("--attention-visual-alphas", default="")
    ap.add_argument("--attention-visual-layer-index", type=int, default=-1)
    ap.add_argument("--attention-visual-layer-fraction", type=float, default=0.9)
    ap.add_argument("--attention-visual-head-top-k", type=int, default=4)
    ap.add_argument("--image-attention-alphas", default="")
    ap.add_argument("--image-attention-layer-index", type=int, default=-1)
    ap.add_argument("--image-attention-layer-fraction", type=float, default=0.9)
    ap.add_argument("--image-attention-head-top-k", type=int, default=4)
    ap.add_argument(
        "--image-attention-head-select",
        choices=("low_visual", "high_visual", "high_text", "all"),
        default="low_visual",
    )
    ap.add_argument("--image-attention-text-alphas", default="")
    ap.add_argument("--image-attention-text-top-k", type=int, default=16)
    ap.add_argument("--jspace-alphas", default="")
    ap.add_argument("--jspace-gammas", default="")
    ap.add_argument("--jspace-top-k", type=int, default=4)
    ap.add_argument("--jspace-layer-index", type=int, default=-1)
    ap.add_argument("--jspace-layer-fraction", type=float, default=0.5)
    ap.add_argument("--jspace-probe", choices=("none", "summary"), default="summary")
    ap.add_argument("--jspace-lens", choices=("logit_lens", "local_jacobian", "fitted_jacobian"), default="logit_lens")
    ap.add_argument("--jspace-lens-path", default="")
    ap.add_argument("--jspace-swap-alpha", type=float, default=0.0)
    ap.add_argument("--absorption-image-margin-max", type=float, default=10.0)
    ap.add_argument("--contrast-margin", type=float, default=0.0)
    ap.add_argument("--token-top-k", type=int, default=128)
    ap.add_argument("--trace-tokens-limit", type=int, default=12)
    ap.add_argument("--timeout-s", type=int, default=600)
    args = ap.parse_args()

    lambdas = [float(x.strip()) for x in args.lambdas.split(",") if x.strip()]
    subspace_alphas = [float(x.strip()) for x in args.prior_subspace_alphas.split(",") if x.strip()]
    if not subspace_alphas:
        subspace_alphas = [0.0]
    layer_subspace_alphas = [float(x.strip()) for x in args.layer_prior_subspace_alphas.split(",") if x.strip()]
    if not layer_subspace_alphas:
        layer_subspace_alphas = [0.0]
    attention_prior_alphas = [float(x.strip()) for x in args.attention_prior_alphas.split(",") if x.strip()]
    if not attention_prior_alphas:
        attention_prior_alphas = [0.0]
    attention_visual_alphas = [float(x.strip()) for x in args.attention_visual_alphas.split(",") if x.strip()]
    if not attention_visual_alphas:
        attention_visual_alphas = [0.0]
    image_attention_alphas = [float(x.strip()) for x in args.image_attention_alphas.split(",") if x.strip()]
    if not image_attention_alphas:
        image_attention_alphas = [0.0]
    image_attention_text_alphas = [
        float(x.strip()) for x in args.image_attention_text_alphas.split(",") if x.strip()
    ]
    if not image_attention_text_alphas:
        image_attention_text_alphas = [0.0]
    jspace_alphas = [float(x.strip()) for x in args.jspace_alphas.split(",") if x.strip()]
    if not jspace_alphas:
        jspace_alphas = [0.0]
    jspace_gammas = [float(x.strip()) for x in args.jspace_gammas.split(",") if x.strip()]
    if not jspace_gammas:
        jspace_gammas = [0.0]
    original_argv = list(sys.argv)
    try:
        for (
            jspace_gamma,
            jspace_alpha,
            image_attention_text_alpha,
            image_attention_alpha,
            visual_alpha,
            attention_alpha,
            layer_alpha,
            alpha,
            lam,
        ) in itertools.product(
            jspace_gammas,
            jspace_alphas,
            image_attention_text_alphas,
            image_attention_alphas,
            attention_visual_alphas,
            attention_prior_alphas,
            layer_subspace_alphas,
            subspace_alphas,
            lambdas,
        ):
            alpha_slug = f"subspace{_lambda_slug(alpha)}" if alpha else "subspace0"
            layer_alpha_slug = f"layersubspace{_lambda_slug(layer_alpha)}" if layer_alpha else "layersubspace0"
            attention_alpha_slug = f"attnprior{_lambda_slug(attention_alpha)}" if attention_alpha else "attnprior0"
            visual_alpha_slug = f"attnvis{_lambda_slug(visual_alpha)}" if visual_alpha else "attnvis0"
            image_attention_alpha_slug = (
                f"imgattn{_lambda_slug(image_attention_alpha)}" if image_attention_alpha else "imgattn0"
            )
            text_attention_alpha_slug = (
                f"textattn{_lambda_slug(image_attention_text_alpha)}"
                if image_attention_text_alpha
                else "textattn0"
            )
            jspace_alpha_slug = f"jspacea{_lambda_slug(jspace_alpha)}" if jspace_alpha else "jspacea0"
            jspace_gamma_slug = f"jspaceg{_lambda_slug(jspace_gamma)}" if jspace_gamma else "jspaceg0"
            jspace_swap_slug = f"jspaces{_lambda_slug(args.jspace_swap_alpha)}" if args.jspace_swap_alpha else "jspaces0"
            output_dir = (
                Path(args.output_root)
                / (
                    f"{args.prior_source}_{args.prior_score}_{args.jspace_lens}_{alpha_slug}_{layer_alpha_slug}_"
                    f"{attention_alpha_slug}_{visual_alpha_slug}_{image_attention_alpha_slug}_"
                    f"{text_attention_alpha_slug}_{jspace_alpha_slug}_{jspace_gamma_slug}_{jspace_swap_slug}_{_lambda_slug(lam)}"
                )
            )
            sys.argv = [
                "evaluate_cdh_bench.py",
                "--jsonl",
                args.jsonl,
                "--images-root",
                args.images_root,
                "--models",
                args.models,
                "--tasks",
                "caption",
                "--subcategories",
                args.subcategories,
                "--limit-per-subcategory",
                str(args.limit_per_subcategory),
                "--mitigation",
                "revis",
                "--revis-mode",
                "auto",
                "--revis-gate",
                args.gate,
                "--revis-lambda-prior",
                str(lam),
                "--revis-lambda-hidden",
                "0.0",
                "--revis-latent-gamma",
                "0.0",
                "--revis-layer-gamma",
                "0.0",
                "--revis-prior-source",
                args.prior_source,
                "--revis-prior-degrade-mode",
                args.prior_degrade_mode,
                "--revis-prior-score",
                args.prior_score,
                "--revis-prior-inertia-gate",
                args.prior_inertia_gate,
                "--revis-prior-margin",
                str(args.prior_margin),
                "--revis-prior-inertia-prob-min",
                str(args.prior_inertia_prob_min),
                "--revis-prior-inertia-logprob-margin",
                str(args.prior_inertia_logprob_margin),
                "--revis-prior-subspace-alpha",
                str(alpha),
                "--revis-prior-subspace-top-k",
                str(args.prior_subspace_top_k),
                "--revis-layer-prior-subspace-alpha",
                str(layer_alpha),
                "--revis-layer-prior-subspace-top-k",
                str(args.layer_prior_subspace_top_k),
                "--revis-layer-prior-subspace-index",
                str(args.layer_prior_subspace_index),
                "--revis-layer-prior-subspace-fraction",
                str(args.layer_prior_subspace_fraction),
                "--revis-attention-prior-alpha",
                str(attention_alpha),
                "--revis-attention-prior-layer-index",
                str(args.attention_prior_layer_index),
                "--revis-attention-prior-layer-fraction",
                str(args.attention_prior_layer_fraction),
                "--revis-attention-prior-head-top-k",
                str(args.attention_prior_head_top_k),
                "--revis-attention-visual-alpha",
                str(visual_alpha),
                "--revis-attention-visual-layer-index",
                str(args.attention_visual_layer_index),
                "--revis-attention-visual-layer-fraction",
                str(args.attention_visual_layer_fraction),
                "--revis-attention-visual-head-top-k",
                str(args.attention_visual_head_top_k),
                "--revis-image-attention-alpha",
                str(image_attention_alpha),
                "--revis-image-attention-layer-index",
                str(args.image_attention_layer_index),
                "--revis-image-attention-layer-fraction",
                str(args.image_attention_layer_fraction),
                "--revis-image-attention-head-top-k",
                str(args.image_attention_head_top_k),
                "--revis-image-attention-head-select",
                args.image_attention_head_select,
                "--revis-image-attention-text-alpha",
                str(image_attention_text_alpha),
                "--revis-image-attention-text-top-k",
                str(args.image_attention_text_top_k),
                "--revis-jspace-alpha",
                str(jspace_alpha),
                "--revis-jspace-gamma",
                str(jspace_gamma),
                "--revis-jspace-top-k",
                str(args.jspace_top_k),
                "--revis-jspace-layer-index",
                str(args.jspace_layer_index),
                "--revis-jspace-layer-fraction",
                str(args.jspace_layer_fraction),
                "--revis-jspace-probe",
                args.jspace_probe,
                "--revis-jspace-lens",
                args.jspace_lens,
                "--revis-jspace-lens-path",
                args.jspace_lens_path,
                "--revis-jspace-swap-alpha",
                str(args.jspace_swap_alpha),
                "--revis-absorption-image-margin-max",
                str(args.absorption_image_margin_max),
                "--revis-contrast-margin",
                str(args.contrast_margin),
                "--revis-token-top-k",
                str(args.token_top_k),
                "--trace-tokens-limit",
                str(args.trace_tokens_limit),
                "--output-dir",
                str(output_dir),
                "--timeout-s",
                str(args.timeout_s),
                "--retry",
                "1",
            ]
            print(
                (
                    f"\n=== REVIS prior-score sweep: {args.prior_score} lambda={lam:g} "
                    f"subspace_alpha={alpha:g} layer_subspace_alpha={layer_alpha:g} "
                    f"attention_prior_alpha={attention_alpha:g} "
                    f"attention_visual_alpha={visual_alpha:g} "
                    f"image_attention_alpha={image_attention_alpha:g} "
                    f"text_attention_alpha={image_attention_text_alpha:g} "
                    f"jspace_alpha={jspace_alpha:g} "
                    f"jspace_gamma={jspace_gamma:g} ==="
                ),
                flush=True,
            )
            rc = evaluate_cdh_bench.main()
            if rc != 0:
                return int(rc)
    finally:
        sys.argv = original_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
