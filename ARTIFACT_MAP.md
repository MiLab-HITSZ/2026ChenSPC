# SPC Artifact Map

All paths are relative to the repository root. Reproduction commands are in
`REPRODUCE.md`.

## Core Protocols and Scores

- Learned-route evaluation protocol: `configs/spc_aaai_frozen_v1.json`
- SPC+NoLan composition protocol: `configs/spc_nolan_backfill_v1.json`
- SPC+NoLan composition analyzer: `analyze_nolan_backfill.py`
- SPC+NoLan composition results: `result/cprc_robustness/nolan_support_backfill_qwen_llava_v1.json`
- Split provenance: `runtime/paper/cdh_exposure_ledger.json`
- Qwen analysis: `result/hierarchical_eb_lambda_32b/cprc_instruct_bf16_cpr_unseen_frozen_v1.json`
- LLaVA analysis: `result/hierarchical_eb_lambda_32b/cprc_llava16_34b_cpr_unseen_frozen_v1.json`
- Official REVIS protocol: `configs/official_revis_frozen_v1.json`
- Strong baseline protocol: `configs/strong_baselines_frozen_v1.json`
- NoLan Qwen MC/QA transfer: `result/baseline_frozen_transfer/nolan_cpr_unseen_{mc,qa}_qwen32b.json`
- NoLan LLaVA MC/QA transfer: `result/baseline_frozen_transfer/nolan_cpr_unseen_{mc,qa}_llava34b.json`
- Complete LLaVA baseline matrix: `result/baseline_frozen_transfer/llava16_34b_complete_baseline_matrix_v1.json`
- Original QA polarity protocol: `configs/qa_polarity_stress_v1.json`
- Qwen polarity analysis: `result/qa_polarity_stress/qwen_qa_polarity_stress_analysis_v1.json`
- LLaVA polarity analysis: `result/qa_polarity_stress/llava_qa_polarity_stress_analysis_v1.json`
- QA template protocol: `configs/qa_template_stress_v2.json`
- QA template construction: `data/qa_template_stress_v2/manifest.json`
- QA template analysis: `result/qa_template_stress_v2/qwen_llava_analysis_v2.json`
- Semantic-equivariance conditioning: `result/qa_template_stress_v2/semantic_equivariance_conditioned_qwen_llava_v1.json`
- Paired-calibration protocol: `configs/spc_paired_calibration_v1.json`
- Paired-calibration ablation: `result/cprc_robustness/paired_calibration_qwen_llava_v1.json`
- Score-normalization protocol: `configs/candidate_score_normalization_v1.json`
- Score-normalization analysis: `result/cprc_robustness/candidate_score_normalization_qwen_llava_v1.json`
- Third-model inference config: `configs/qwen_30b_a3b_instruct_bf16_answer1.json`
- Third-model score collector: `evaluate_cdh_mc_candidate_scores.py`
- Third-model analysis protocol: `configs/spc_third_model_mc_v1.json`
- Third-model development scores: `result/cprc_qwen30b_a3b_dev70/Qwen3-VL-30B-A3B-Instruct-BF16/results.jsonl`
- Third-model unseen scores: `result/cprc_qwen30b_a3b_cpr_unseen/Qwen3-VL-30B-A3B-Instruct-BF16/results.jsonl`
- Third-model core MC analysis: `result/cprc_robustness/third_model_qwen30b_a3b_mc_v1.json`

## Transfer, Controls, and Robustness

- External protocol: `configs/spc_external_v1.json`
- HallusionBench transfer: `result/cprc_external/hallusionbench_full_cdh_frozen_bf16_native_v1.json`
- HallusionBench REVIS: `result/cprc_external/hallusionbench_full_revis_alpha16_bf16_v1.json`
- POPE native protocols: `result/cprc_external/pope_native_full_cdh_frozen_bf16_v1.json`
- ConflictVIS transfer: `result/cprc_external/conflictvis_full_frozen_bf16_v1.json`
- Runtime analysis: `result/runtime_efficiency/cprc_official_revis_qwen_llava_v1.json`
- Candidate attribution: `result/cprc_attribution/candidate_attribution_qwen_llava_v1.json`
- Prior controls: `result/cprc_robustness/prior_controls_qwen32b_v1.json`
- Repeated splits: `result/cprc_robustness/repeated_splits_qwen_llava_v1.json`
- Matched gate/Pareto protocol: `configs/spc_gate_pareto_v1.json`
- Matched gate/Pareto analysis: `result/cprc_robustness/gate_pareto_qwen_llava_v1.json`
- Matched distribution-shift protocol: `configs/spc_shift_matched_controls_v1.json`
- Matched distribution-shift analysis: `result/cprc_robustness/shift_matched_controls_qwen_llava_v1.json`
- MC option-order protocol: `configs/spc_mc_option_permutation_v1.json`
- MC option-order construction: `data/mc_option_permutation_v1/manifest.json`
- MC option-order analysis: `result/cprc_robustness/mc_option_permutation_qwen_llava_v1.json`
- Visual CounterFact protocol: `configs/visual_counterfact_analysis_v1.json`
- Visual CounterFact frozen manifest: `data/visual_counterfact_frozen_v1/manifest.json`
- Visual CounterFact analysis: `result/cprc_external/visual_counterfact_qwen_llava_v1.json`
- Lambda stable-interval analysis: `result/paper_revision_stats/lambda_intervals.json` (driver: `analyze_lambda_stable_intervals.py`)
- Proposal-state decomposition: `result/paper_revision_stats/proposal_decomposition.json` (driver: `analyze_proposal_decomposition.py`)
- MC-only calibration ablation: `result/paper_revision_stats/mc_only_ablation.json`
- Stability-radius analysis: `result/paper_revision_stats/stability_radius.json`
- Benchmark spectrum: `result/paper_revision_stats/benchmark_spectrum.json` (driver: `analyze_benchmark_spectrum.py`)
- Calibration-size protocol: `configs/spc_calibration_size_v1.json`
- Calibration-size curve: `result/cprc_robustness/calibration_size_qwen_llava_v1.json`
- Qwen unseen families: `result/hierarchical_eb_lambda_32b/cprc_instruct_leave_one_supercategory_out_v1.json`
- Qwen unseen subcategories: `result/hierarchical_eb_lambda_32b/cprc_instruct_leave_one_subcategory_out_v1.json`
- LLaVA unseen families: `result/hierarchical_eb_lambda_32b/cprc_llava16_34b_leave_one_supercategory_out_v1.json`
