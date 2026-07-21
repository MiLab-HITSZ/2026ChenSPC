# Reproducing the SPC robustness experiments

All commands are run from the repository root. The reported environment is
listed in `requirements-paper.txt`.

## Obtain the data (required once)

Benchmark images, paired annotations, the split ledger, and the frozen
candidate-score files consumed by every command below are **not** tracked in
this repository. Download the dataset release from Hugging Face at
**https://huggingface.co/datasets/cks19999/CDH-Bench** and unpack it at the repository root so that `images/`,
`data/`, and `result/` are populated. The frozen scores under `result/` make
every table below reproducible without a GPU; they can also be regenerated
from the images with the `evaluate_*_candidate_scores.py` scripts (GPU and
model weights required).

Model directories are expected at
`models/Qwen3-VL-32B-Instruct`, `models/llava-v1.6-34b-hf`, and
`models/Qwen3-VL-30B-A3B-Instruct`. The lean runtime reads `model_type` from
`AutoConfig` and dispatches the 30B checkpoint to the official
`Qwen3VLMoeForConditionalGeneration` class; regression tests cover dense/MoE
dispatch and reject unknown Qwen3-VL architectures.

Some machine-readable version and family keys retain the pre-paper identifier
`bprc_full`. They are frozen for artifact compatibility and refer to SPC.

## SPC+NoLan composition (optional extension)

The optional composition gives the learned SPC correction priority and invokes
the released NoLan analytic proposal only when that route retains the native
answer. Both proposals reuse the same image-conditioned and prior-estimation
candidate scores, so the composition adds no model forward pass.

```bash
python analyze_nolan_backfill.py \
  --config configs/spc_nolan_backfill_v1.json \
  --spc-artifact result/cprc_robustness/gate_pareto_qwen_llava_v1.json \
  --output result/cprc_robustness/nolan_support_backfill_qwen_llava_v1.json
```

The paper's SPC+NoLan row uses `nolan_raw_backfill`, the fixed released
analytic proposal on learned-route abstentions. The artifact also records a
separately selected support-gated diagnostic, but that diagnostic is not the
primary method.

## Install the analysis environment

```bash
python -m pip install -r requirements-paper.txt
python -m pytest -q \
  tests/test_spc_shift_matched_controls.py \
  tests/test_spc_mc_option_permutation.py \
  tests/test_visual_counterfact_frozen.py \
  tests/test_visual_counterfact_spc.py \
  tests/test_spc_vlm_runtime.py \
  tests/test_visual_counterfact_candidate_scores.py \
  tests/test_qa_template_analysis.py \
  tests/test_qa_template_stress.py \
  tests/test_spc_paired_calibration.py \
  tests/test_qa_semantic_equivariance.py \
  tests/test_candidate_score_normalization.py \
  tests/test_qwen3_vl_model_loading.py \
  tests/test_cdh_mc_candidate_scores.py \
  tests/test_official_contrastive_baselines.py \
  tests/test_third_model_mc.py
```

## NoLan score-rule comparison

The official NoLan release applies dynamic suppression to full-vocabulary
next-token distributions. For finite MC/QA, we preserve its released equation
and coefficient while normalizing over the task-native complete answers. For
example, the Qwen MC development record and frozen transfer are:

```bash
python analyze_candidate_baseline_sweep.py \
  --input result/cprc_bf16_dev70/Qwen3-VL-32B-Instruct-BF16/results.jsonl \
  --method nolan --task mc --max-cs-drop 0.04 \
  --output result/baseline_sweeps/nolan_dev70_mc_qwen32b.json

python analyze_candidate_baseline_frozen_transfer.py \
  --development-selection result/baseline_sweeps/nolan_dev70_mc_qwen32b.json \
  --test result/cprc_bf16_cpr_unseen/Qwen3-VL-32B-Instruct-BF16/results.jsonl \
  --output result/baseline_frozen_transfer/nolan_cpr_unseen_mc_qwen32b.json
```

Repeat for QA and replace the score files by the LLaVA development/unseen files
listed in `configs/strong_baselines_frozen_v1.json`. NoLan keeps the released
dynamic coefficient; the development step records provenance and reports
whether the released setting meets the shared CS constraint rather than tuning
that coefficient.

## Paired-calibration necessity

```bash
python analyze_spc_paired_calibration.py \
  --config configs/spc_paired_calibration_v1.json \
  --output result/cprc_robustness/paired_calibration_qwen_llava_v1.json
```

This reuses the same frozen candidate scores and Point-MAP grid for three
regimes: paired fit and selection, paired fit with a CF-only objective, and
CF-only fit and selection. CS test outcomes are used only for the final audit
in the latter two regimes.

## Sample-level semantic equivariance

```bash
python analyze_qa_semantic_equivariance.py \
  --config configs/qa_template_stress_v2.json \
  --output result/qa_template_stress_v2/semantic_equivariance_conditioned_qwen_llava_v1.json
```

The partition is computed from prior-estimation predictions only. It never inspects
image-conditioned correctness, CF/CS labels, or target outcomes. The frozen
template policy is replayed without refitting, with pair-cluster bootstrapping.

## Candidate-score normalization audit

```bash
python analyze_candidate_score_normalization.py \
  --config configs/candidate_score_normalization_v1.json \
  --output result/cprc_robustness/candidate_score_normalization_qwen_llava_v1.json
```

The analyzer verifies token counts and stored sum/mean identities, compares
eight fixed residual coefficients, and independently refits sum- and
mean-normalized Point-MAP policies under the same four-point CS budget.

## Third-model core MC replication

Generate the development and unseen candidate scores on disjoint four-GPU
groups:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate_cdh_mc_candidate_scores.py \
  --dataset CDH-Bench.revised.strict.jsonl --images-root images \
  --output result/cprc_qwen30b_a3b_dev70/Qwen3-VL-30B-A3B-Instruct-BF16/results.jsonl \
  --models configs/qwen_30b_a3b_instruct_bf16_answer1.json \
  --pair-manifest runtime/paper/cdh_exposure_ledger.json \
  --pair-split cpr_development --retry 2

CUDA_VISIBLE_DEVICES=4,5,6,7 python evaluate_cdh_mc_candidate_scores.py \
  --dataset CDH-Bench.revised.strict.jsonl --images-root images \
  --output result/cprc_qwen30b_a3b_cpr_unseen/Qwen3-VL-30B-A3B-Instruct-BF16/results.jsonl \
  --models configs/qwen_30b_a3b_instruct_bf16_answer1.json \
  --pair-manifest runtime/paper/cdh_exposure_ledger.json \
  --pair-split cpr_unseen --retry 2

python analyze_third_model_mc.py \
  --config configs/spc_third_model_mc_v1.json \
  --output result/cprc_robustness/third_model_qwen30b_a3b_mc_v1.json
```

The analysis reports native-error/prior-estimate alignment, the frozen MC operating
point, and three leave-one-family-out fits. No QA or test-family label is used
to select a policy.

## Matched-capacity development shifts

```bash
python analyze_spc_shift_matched_controls.py \
  --config configs/spc_shift_matched_controls_v1.json \
  --output result/cprc_robustness/shift_matched_controls_qwen_llava_v1.json
```

This analysis uses only the frozen candidate-score files named in the config.
It refits every family within each semantic leave-out or 14-pair calibration
trial and never selects on the 230-pair target.

## MC option-order stress test

```bash
python build_mc_option_permutation.py \
  --input CDH-Bench.revised.strict.jsonl \
  --output-dir data/mc_option_permutation_v1
```

For each model and each of `rotate_1`, `rotate_2`, and `reverse`, use the same
frozen primary evaluator. For example, the Qwen `rotate_1` call is:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate_cdh_bench.py \
  --jsonl data/mc_option_permutation_v1/rotate_1.jsonl \
  --images-root images \
  --output-dir result/mc_option_permutation_v1/qwen_rotate_1 \
  --models configs/qwen_32b_instruct_bf16_answer1.json \
  --tasks mc \
  --pair-manifest runtime/paper/cdh_exposure_ledger.json \
  --pair-split cpr_unseen \
  --mitigation cp_vbc --cp-vbc-tasks mc --cp-vbc-mode candidate \
  --cp-vbc-lambda 0.6 --cp-vbc-absorption-image-margin-max 10 \
  --cp-vbc-contrast-margin 0.75 --parallel 1 --retry 2
```

Repeat with the LLaVA model config and the output paths listed in
`configs/spc_mc_option_permutation_v1.json`. Then run:

```bash
python analyze_spc_mc_option_permutation.py \
  --config configs/spc_mc_option_permutation_v1.json \
  --output result/cprc_robustness/mc_option_permutation_qwen_llava_v1.json
```

The analyzer selects once from the original 70-pair development scores and
replays that policy on all three option orders.

## Frozen Visual CounterFact transfer

Download the official `mgolov/Visual-Counterfact` color and size parquet files at
revision `f8cd0b2335356b0f01a164df0cac7403f56a6c83` into
`downloads/visual_counterfact/data`, then run:

```bash
python prepare_visual_counterfact_frozen.py \
  --source-dir downloads/visual_counterfact/data \
  --output-dir data/visual_counterfact_frozen_v1

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python evaluate_visual_counterfact_candidate_scores.py \
  --dataset data/visual_counterfact_frozen_v1/visual_counterfact_mc.jsonl \
  --models configs/qwen_32b_instruct_bf16_answer1.json \
  --output result/visual_counterfact_frozen_v1/qwen/results.jsonl

CUDA_VISIBLE_DEVICES=4,5,6,7 \
python evaluate_visual_counterfact_candidate_scores.py \
  --dataset data/visual_counterfact_frozen_v1/visual_counterfact_mc.jsonl \
  --models configs/llava16_34b_instruct_transformers_answer1.json \
  --output result/visual_counterfact_frozen_v1/llava/results.jsonl

python analyze_visual_counterfact_spc.py \
  --config configs/visual_counterfact_analysis_v1.json \
  --output result/cprc_external/visual_counterfact_qwen_llava_v1.json
```

The checked-in manifest contains source and output hashes. The protocol uses all
1,220 official color/size pairs, performs no response-based filtering, and fits
only on the original CDH development split.

## POPEv2 transfer evaluation

Build the 70-pair QA transfer set from the official POPE repository checkout
(`downloads/hallucination_refs/POPE`, revision pinned in
`data/popev2_qa_transfer/manifest.json`), then run the frozen candidate-score
evaluator and the grouped Bayes-path analysis:

```bash
python prepare_popev2_qa_transfer.py \
  --repo downloads/hallucination_refs/POPE \
  --output-root data/popev2_qa_transfer

CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate_cdh_bench.py \
  --jsonl data/popev2_qa_transfer/popev2_qa.jsonl \
  --images-root data/popev2_qa_transfer/images \
  --output-dir result/popev2_qa_transfer_32b_instruct \
  --models configs/qwen_32b_instruct_transformers_maxtok48.json \
  --tasks qa \
  --mitigation cp_vbc --cp-vbc-tasks qa --cp-vbc-mode candidate \
  --cp-vbc-lambda 0.6 --cp-vbc-absorption-image-margin-max 8.0 \
  --cp-vbc-contrast-margin 0.0 --parallel 1 --retry 2

python analyze_cp_vbc_bayes_path_cv.py \
  --input popev2=result/popev2_qa_transfer_32b_instruct/Qwen3-VL-32B-Instruct-transformers-maxtok48/results.jsonl \
  --task qa \
  --json-out result/popev2_qa_transfer_32b_instruct/bayes_path_cv.json
```

The released 140 modified images under `data/popev2_qa_transfer/images` are
derived from COCO; the manifest records the official source URLs and the
selected image ids.

## Proposal decomposition, MC-only ablation, and stability radius

```bash
python analyze_proposal_decomposition.py --task all
```

Writes `result/paper_revision_stats/proposal_decomposition.json`,
`mc_only_ablation.json`, and `stability_radius.json`. Every task first replays
the frozen operating point row by row and verifies it against
`result/anchored_cprc/main_test.json`; the verification summary is stored in
each output. The frozen-replay helpers are self-contained in the script and
reuse `analyze_hierarchical_eb_lambda_cv.py` and `analyze_spc_gate_pareto.py`.

## Lambda stable intervals

```bash
python analyze_lambda_stable_intervals.py
```

Writes `result/paper_revision_stats/lambda_intervals.json` and
`paper/aaai27/figures/lambda_stable_intervals.pdf`, and cross-checks the
dev-fitted constant coefficient against `result/paper_revision_stats/stats.json`.

## Benchmark spectrum

```bash
python analyze_benchmark_spectrum.py
```

Writes `result/paper_revision_stats/benchmark_spectrum.json` and
`benchmark_spectrum_table.tex`: the six-benchmark scope table (CDH-Bench,
VisualCounterfact, HallusionBench, ConflictVIS, POPE, POPEv2) with the frozen
SPC, SPC+NoLan, and native streams. External-benchmark replays are verified
against `result/nolan_external_transfer/*.json` and the frozen SPC artifacts
under `result/cprc_external/`.

## Paper artifacts

```bash
python paper/aaai27/make_paper_figures.py
python paper/aaai27/make_evidence_chain_v2.py
python paper/aaai27/make_rq1_matrix.py
python paper/aaai27/make_rq1_matrix_wide.py
```

The complete mapping from tables to machine-readable artifacts is in
`paper/aaai27/ARTIFACT_MAP.md`; the typeset supplement supplies protocol and
analysis details without repeating long repository paths in tables.
