import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze_cprc_gate_pareto import (
    _gate_passes,
    corrected_proposals,
    json_safe,
    threshold_grid,
)


def test_gate_family_components_are_independently_removed():
    detail = {
        "baseline": "yes",
        "proposal": "no",
        "visual_conflict": True,
        "prior_absorption": False,
        "arity_supported": True,
        "support_distance": 9.0,
        "posterior_margin": 0.01,
        "posterior_support": 0.55,
    }
    gate = {
        "visual_support_min": 0.8,
        "absorption_support_min": 0.8,
        "posterior_margin_min": 0.1,
        "support_distance_max": 3.0,
    }
    assert not _gate_passes(detail, gate, "bprc_full")
    assert not _gate_passes(detail, gate, "bprc_no_density_gate")
    assert not _gate_passes(detail, gate, "bprc_no_posterior_gate")
    assert _gate_passes(detail, gate, "bprc_conflict_only")
    assert _gate_passes(detail, gate, "bprc_proposal_only")


def test_no_conflict_gate_accepts_supported_non_conflict_proposal():
    detail = {
        "baseline": "yes",
        "proposal": "no",
        "visual_conflict": False,
        "prior_absorption": False,
        "arity_supported": True,
        "support_distance": 1.0,
        "posterior_margin": 0.2,
        "posterior_support": 0.9,
    }
    gate = {
        "visual_support_min": 0.8,
        "absorption_support_min": 0.8,
        "posterior_margin_min": 0.1,
        "support_distance_max": 3.0,
    }
    assert not _gate_passes(detail, gate, "bprc_full")
    assert _gate_passes(detail, gate, "bprc_no_conflict_gate")


def test_threshold_grid_contains_always_and_never_operating_points():
    values = np.asarray([0.2, 0.4, 0.8])
    thresholds = threshold_grid(values, [0, 1, 2])
    assert thresholds[0] == 0.0
    assert thresholds[-1] > 1.0
    assert any(np.isclose(value, 0.3) for value in thresholds)
    assert any(np.isclose(value, 0.6) for value in thresholds)


def test_corrected_proposal_subtracts_candidate_prior():
    row = {
        "cp_vbc": {
            "baseline_key": "yes",
            "candidates": [
                {"key": "yes", "logp_image": 2.0, "logp_prior": 3.0},
                {"key": "no", "logp_image": 1.8, "logp_prior": 0.0},
            ],
        }
    }
    assert corrected_proposals([row], coefficient=0.0) == ["yes"]
    assert corrected_proposals([row], coefficient=0.5) == ["no"]


def test_nonfinite_gate_thresholds_are_strict_json_values():
    assert json_safe({"threshold": float("inf")}) == {"threshold": "inf"}
