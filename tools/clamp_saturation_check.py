#!/usr/bin/env python
"""Check how often the rate-evidence clamp (+-4) binds (W2 diagnostic).

The direct rate channel is (posterior_logit - concept_logit) * reliability
clamped to [-4, 4]. If a substantial share of train/valid rows sits at the
clamp boundary on high-count datasets, the hand-set bound is throwing away
signal and should be relaxed.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

EPS = 1e-4


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def check(data_dir: str) -> dict:
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    rows = train[["stu_id", "cpt_seq", "label"]].copy()
    rows["cpt"] = rows["cpt_seq"].astype(str).str.split(",")
    rows = rows.explode("cpt")
    rows["cpt"] = rows["cpt"].str.strip().astype(np.int64)

    g = float(train["label"].mean())
    concept = rows.groupby("cpt")["label"].agg(["sum", "count"])
    concept_rate = (concept["sum"] + g) / (concept["count"] + 1.0)

    sc = rows.groupby(["stu_id", "cpt"])["label"].agg(["sum", "count"]).reset_index()
    prior = sc["cpt"].map(concept_rate).to_numpy(dtype=np.float64)
    posterior = (sc["sum"].to_numpy(dtype=np.float64) + prior) / (
        sc["count"].to_numpy(dtype=np.float64) + 1.0
    )
    reliability = sc["count"].to_numpy(dtype=np.float64) / (
        sc["count"].to_numpy(dtype=np.float64) + 1.0
    )
    raw = (_logit(posterior) - _logit(prior)) * reliability
    return {
        "pairs": len(raw),
        "mean_count": float(sc["count"].mean()),
        "abs_ge_4": float(np.mean(np.abs(raw) >= 4.0)),
        "abs_ge_3": float(np.mean(np.abs(raw) >= 3.0)),
        "p99_abs": float(np.quantile(np.abs(raw), 0.99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--data_root", default="./data")
    args = parser.parse_args()
    print(f"{'dataset':<12} {'pairs':>9} {'mean_n':>7} {'|e|>=4':>8} {'|e|>=3':>8} {'p99|e|':>8}")
    for name in [t.strip() for t in args.datasets.split(",") if t.strip()]:
        r = check(os.path.join(args.data_root, name))
        print(
            f"{name:<12} {r['pairs']:>9} {r['mean_count']:>7.2f} "
            f"{100 * r['abs_ge_4']:>7.2f}% {100 * r['abs_ge_3']:>7.2f}% {r['p99_abs']:>8.3f}"
        )


if __name__ == "__main__":
    main()
