"""Lightweight checks for validation evidence-bucket metrics and pairing."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluate_graph_validation_buckets import (
    BRANCH_GATE_CONTRASTS,
    BRANCH_GATE_VARIANTS,
    GRAPH_2X2_CONTRASTS,
    GRAPH_2X2_VARIANTS,
    RESIDUAL_CONTRASTS,
    RESIDUAL_VARIANTS,
    _bucket_masks,
    _metrics,
    _paired_contrasts,
)
from tools.summarize_validation_runs import _residual_parameters


def main() -> None:
    support = np.asarray([0, 1, 2, 3, 8], dtype=float)
    masks = _bucket_masks(support)
    assert masks["all"].tolist() == [True] * 5
    assert masks["n=0"].tolist() == [True, False, False, False, False]
    assert masks["n<3"].tolist() == [True, True, True, False, False]

    metrics = _metrics(
        np.asarray([0.0, 1.0]),
        np.asarray([0.25, 0.75]),
    )
    assert metrics["auc"] == 1.0
    assert math.isclose(metrics["rmse"], 0.25)
    assert math.isclose(metrics["bce_loss"], -math.log(0.75))

    rows = []
    offsets = {
        "full": 0.04,
        "no_evidence_propagation": 0.03,
        "no_message_passing": 0.02,
        "no_graph_calibration": 0.01,
    }
    for seed in (42, 43):
        for variant in GRAPH_2X2_VARIANTS:
            for bucket in ("all", "n=0", "n<3"):
                offset = offsets[variant]
                rows.append(
                    {
                        "run_dir": f"{variant}-{seed}",
                        "architecture": "graph_irt_v10",
                        "train_evidence_mode": "excluded",
                        "dataset": "demo",
                        "seed": seed,
                        "bucket": bucket,
                        "model_variant": variant,
                        "auc": 0.7 + offset,
                        "bce_loss": 0.6 - offset,
                        "rmse": 0.5 - offset,
                    }
                )
    contrasts = _paired_contrasts(
        pd.DataFrame(rows),
        variants=GRAPH_2X2_VARIANTS,
        contrasts=GRAPH_2X2_CONTRASTS,
    )
    selected = contrasts[
        (contrasts["bucket"] == "n=0")
        & (contrasts["metric"] == "auc")
        & (contrasts["contrast"] == "propagation|state_on")
    ].iloc[0]
    assert selected["n_paired"] == 2
    assert math.isclose(selected["mean"], 0.01)

    # Undefined single-class AUC may reduce AUC pairing but must not discard
    # the independently valid BCE/RMSE contrasts.
    rows[0]["auc"] = float("nan")
    contrasts = _paired_contrasts(
        pd.DataFrame(rows),
        variants=GRAPH_2X2_VARIANTS,
        contrasts=GRAPH_2X2_CONTRASTS,
    )
    selected = contrasts[
        (contrasts["bucket"] == "all")
        & (contrasts["metric"] == "auc")
        & (contrasts["contrast"] == "propagation|state_on")
    ].iloc[0]
    assert selected["n_paired"] == 1

    residual_rows = []
    for variant, auc in (("full", 0.70), ("gec_residual", 0.705)):
        residual_rows.append(
            {
                "run_dir": variant,
                "architecture": "graph_irt_v10",
                "train_evidence_mode": "excluded",
                "dataset": "demo",
                "seed": 42,
                "bucket": "all",
                "model_variant": variant,
                "auc": auc,
                "bce_loss": 0.6,
                "rmse": 0.5,
            }
        )
    residual = _paired_contrasts(
        pd.DataFrame(residual_rows),
        variants=RESIDUAL_VARIANTS,
        contrasts=RESIDUAL_CONTRASTS,
    )
    selected = residual[
        (residual["metric"] == "auc")
        & (residual["contrast"] == "residual-full")
    ].iloc[0]
    assert math.isclose(selected["mean"], 0.005)

    branch_rows = []
    for variant, auc in (
        ("full", 0.700),
        ("gec_branch_gate_candidate", 0.706),
        ("gec_branch_gate_selected", 0.705),
    ):
        branch_rows.append(
            {
                "run_dir": variant,
                "architecture": "graph_irt_v10",
                "train_evidence_mode": "excluded",
                "dataset": "demo",
                "seed": 42,
                "bucket": "all",
                "model_variant": variant,
                "auc": auc,
                "bce_loss": 0.6,
                "rmse": 0.5,
            }
        )
    branch = _paired_contrasts(
        pd.DataFrame(branch_rows),
        variants=BRANCH_GATE_VARIANTS,
        contrasts=BRANCH_GATE_CONTRASTS,
    )
    candidate = branch[
        (branch["metric"] == "auc")
        & (branch["contrast"] == "branch-candidate-full")
    ].iloc[0]
    selected = branch[
        (branch["metric"] == "auc")
        & (branch["contrast"] == "branch-selected-full")
    ].iloc[0]
    assert math.isclose(candidate["mean"], 0.006)
    assert math.isclose(selected["mean"], 0.005)

    parameters = _residual_parameters(
        {
            "model_state_dict": {
                "evidence_residual.state_route_raw": torch.tensor(0.0),
                "evidence_residual.propagation_route_raw": torch.tensor(
                    [0.0, math.atanh(0.5)]
                ),
            }
        }
    )
    assert parameters["kind"] == "branch_gate_v5"
    assert parameters["state_route"] == 0.0
    assert parameters["state_branch_scale"] == 1.0
    propagation_route = json.loads(parameters["propagation_route_json"])
    propagation_scale = json.loads(
        parameters["propagation_branch_scale_json"]
    )
    assert math.isclose(propagation_route[0], 0.0)
    assert math.isclose(propagation_route[1], 0.5, abs_tol=1e-7)
    assert math.isclose(propagation_scale[0], 1.0)
    assert math.isclose(propagation_scale[1], 1.5, abs_tol=1e-7)


if __name__ == "__main__":
    main()
