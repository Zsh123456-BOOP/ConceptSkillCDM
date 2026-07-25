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
        for variant in VARIANTS:
            for bucket in ("all", "n=0", "n<3"):
                offset = offsets[variant]
                rows.append(
                    {
                        "run_dir": f"{variant}-{seed}",
                        "dataset": "demo",
                        "seed": seed,
                        "bucket": bucket,
                        "model_variant": variant,
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


if __name__ == "__main__":
    main()
