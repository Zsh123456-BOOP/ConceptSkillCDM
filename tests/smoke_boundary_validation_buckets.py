"""Lightweight checks for E/N/I validation-bucket pairing and sample counts."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluate_boundary_validation_buckets import MODES, _paired_contrasts


def _rows() -> list[dict[str, object]]:
    offsets = {
        "excluded": 0.00,
        "neutralized": 0.01,
        "self_included": 0.03,
    }
    rows = []
    for seed in (42, 43):
        for mode in MODES:
            for bucket, count in (("all", 100), ("n=0", 7), ("n<3", 20)):
                offset = offsets[mode]
                rows.append(
                    {
                        "run_dir": f"{mode}-{seed}",
                        "dataset": "demo",
                        "seed": seed,
                        "bucket": bucket,
                        "model_variant": "full",
                        "train_evidence_mode": mode,
                        "rows": count,
                        "positives": count // 2,
                        "negatives": count - count // 2,
                        "auc": 0.70 + offset,
                        "bce_loss": 0.60 - offset,
                        "rmse": 0.50 - offset,
                    }
                )
    return rows


def main() -> None:
    rows = _rows()
    contrasts = _paired_contrasts(pd.DataFrame(rows))
    selected = contrasts[
        (contrasts["bucket"] == "n=0")
        & (contrasts["metric"] == "auc")
        & (contrasts["contrast"] == "I-N")
    ].iloc[0]
    assert selected["n_paired"] == 2
    assert selected["rows_min"] == selected["rows_max"] == 7
    assert math.isclose(selected["mean"], 0.02)

    # A single undefined AUC cell only removes that seed from AUC pairing;
    # independently valid BCE/RMSE pairs and their sample counts remain.
    rows[0]["auc"] = float("nan")
    contrasts = _paired_contrasts(pd.DataFrame(rows))
    selected = contrasts[
        (contrasts["bucket"] == "all")
        & (contrasts["metric"] == "auc")
        & (contrasts["contrast"] == "N-E")
    ].iloc[0]
    assert selected["n_paired"] == 1
    assert selected["rows_min"] == selected["rows_max"] == 100

    rows = _rows()
    rows[0]["rows"] = 99
    try:
        _paired_contrasts(pd.DataFrame(rows))
    except ValueError as exc:
        assert "mode-dependent rows counts" in str(exc)
    else:
        raise AssertionError("mode-dependent sample sizes must be rejected")


if __name__ == "__main__":
    main()
