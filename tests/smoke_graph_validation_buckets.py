"""Lightweight checks for validation evidence-bucket metrics and pairing."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluate_graph_validation_buckets import (
    VARIANTS,
    _bucket_masks,
    _metrics,
    _paired_contrasts,
)


def main() -> None:
    minimum_support = np.asarray([0, 0, 2, 3, 8], dtype=float)
    maximum_support = np.asarray([0, 2, 2, 4, 8], dtype=float)
    masks = _bucket_masks(minimum_support, maximum_support)
    assert masks["all"].tolist() == [True] * 5
    assert masks["n=0"].tolist() == [True, True, False, False, False]
    assert masks["n=1"].tolist() == [False, False, False, False, False]
    assert masks["n=2"].tolist() == [False, False, True, False, False]
    assert masks["0<n<3"].tolist() == [False, False, True, False, False]
    assert masks["n<3"].tolist() == [True, True, True, False, False]
    assert masks["all_q_zero"].tolist() == [
        True,
        False,
        False,
        False,
        False,
    ]

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
        for variant in VARIANTS:
            for bucket in masks:
                offset = offsets[variant]
                rows.append(
                    {
                        "run_dir": f"{variant}-{seed}",
                        "dataset": "demo",
                        "seed": seed,
                        "bucket": bucket,
                        "model_variant": variant,
                        "rows": 100,
                        "positives": 50,
                        "negatives": 50,
                        "auc": 0.7 + offset,
                        "bce_loss": 0.6 - offset,
                        "rmse": 0.5 - offset,
                    }
                )
    contrasts = _paired_contrasts(pd.DataFrame(rows))
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
    contrasts = _paired_contrasts(pd.DataFrame(rows))
    selected = contrasts[
        (contrasts["bucket"] == "all")
        & (contrasts["metric"] == "auc")
        & (contrasts["contrast"] == "propagation|state_on")
    ].iloc[0]
    assert selected["n_paired"] == 1

    pair_rows = []
    for variant, offset in (
        ("no_graph_calibration", 0.00),
        ("mec", 0.02),
    ):
        for bucket in masks:
            pair_rows.append(
                {
                    "run_dir": variant,
                    "dataset": "demo",
                    "seed": 42,
                    "bucket": bucket,
                    "model_variant": variant,
                    "rows": 100,
                    "positives": 50,
                    "negatives": 50,
                    "auc": 0.7 + offset,
                    "bce_loss": 0.6 - offset,
                    "rmse": 0.5 - offset,
                }
            )
    pair_contrasts = _paired_contrasts(
        pd.DataFrame(pair_rows),
        variants=("no_graph_calibration", "mec"),
        contrasts=(
            (
                "mec-minus-no_graph_calibration",
                {"mec": 1.0, "no_graph_calibration": -1.0},
            ),
        ),
    )
    pair_auc = pair_contrasts[
        (pair_contrasts["bucket"] == "n=0")
        & (pair_contrasts["metric"] == "auc")
    ].iloc[0]
    assert pair_auc["n_paired"] == 1
    assert math.isclose(pair_auc["mean"], 0.02)


if __name__ == "__main__":
    main()
