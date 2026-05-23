"""Summarize coverage-conditioned prediction with bootstrap confidence intervals.

This script reads sample-level predictions produced by
tools/run_main_problem_experiments.py and exports a compact CSV for the paper
draft. It does not train models or modify checkpoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


DEFAULT_DATASETS = ("assist_09", "junyi", "assist_17")
DEFAULT_SUBGROUPS = (
    "direct_unseen_bridgeable",
    "weak_direct_evidence",
    "high_route_mass",
)
DEFAULT_COMPARISONS = (
    ("no_CRG", "prob_no_CRG"),
    ("no_LCRF", "prob_no_LCRF"),
)


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _bce(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_prob = np.clip(y_prob.astype(np.float64), 1e-7, 1 - 1e-7)
    y_true = y_true.astype(np.float64)
    return float(-(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)).mean())


def _ci(values: np.ndarray) -> tuple[float, float]:
    if len(values) == 0 or np.all(~np.isfinite(values)):
        return float("nan"), float("nan")
    low, high = np.nanpercentile(values, [2.5, 97.5])
    return float(low), float(high)


def _bootstrap_delta(
    y_true: np.ndarray,
    full_prob: np.ndarray,
    variant_prob: np.ndarray,
    iters: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    n = len(y_true)
    auc_delta = np.full(iters, np.nan, dtype=np.float64)
    bce_delta = np.full(iters, np.nan, dtype=np.float64)
    if n == 0:
        return {
            "delta_auc_ci_low": float("nan"),
            "delta_auc_ci_high": float("nan"),
            "delta_bce_ci_low": float("nan"),
            "delta_bce_ci_high": float("nan"),
        }
    for i in range(iters):
        idx = rng.integers(0, n, size=n)
        yy = y_true[idx]
        ff = full_prob[idx]
        vv = variant_prob[idx]
        if len(np.unique(yy)) >= 2:
            auc_delta[i] = _safe_auc(yy, ff) - _safe_auc(yy, vv)
        bce_delta[i] = _bce(yy, vv) - _bce(yy, ff)
    auc_low, auc_high = _ci(auc_delta)
    bce_low, bce_high = _ci(bce_delta)
    return {
        "delta_auc_ci_low": auc_low,
        "delta_auc_ci_high": auc_high,
        "delta_bce_ci_low": bce_low,
        "delta_bce_ci_high": bce_high,
    }


def _subgroup_mask(df: pd.DataFrame, subgroup: str) -> pd.Series:
    if subgroup == "all":
        return pd.Series(True, index=df.index)
    if subgroup not in df.columns:
        return pd.Series(False, index=df.index)
    return df[subgroup].astype(bool)


def summarize_dataset(
    dataset: str,
    input_root: Path,
    subgroups: Iterable[str],
    bootstrap_iters: int,
    seed: int,
) -> list[dict[str, object]]:
    pred_path = input_root / dataset / "main_problem_exp2_coverage_conditioned_predictions.csv"
    if not pred_path.exists():
        return [
            {
                "dataset": dataset,
                "subgroup": "",
                "comparison": "",
                "missing_reason": f"missing prediction CSV: {pred_path}",
            }
        ]

    df = pd.read_csv(pred_path)
    label_col = "label_eval" if "label_eval" in df.columns else "label"
    if "prob_full" not in df.columns:
        return [
            {
                "dataset": dataset,
                "subgroup": "",
                "comparison": "",
                "missing_reason": "missing prob_full column",
            }
        ]

    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed + abs(hash(dataset)) % 100_000)
    for subgroup in subgroups:
        mask = _subgroup_mask(df, subgroup)
        sub = df.loc[mask].copy()
        if sub.empty:
            continue
        y = sub[label_col].to_numpy(dtype=np.float64)
        full = sub["prob_full"].to_numpy(dtype=np.float64)
        full_auc = _safe_auc(y, full)
        full_bce = _bce(y, full)
        for comparison, prob_col in DEFAULT_COMPARISONS:
            if prob_col not in sub.columns:
                continue
            variant = sub[prob_col].to_numpy(dtype=np.float64)
            variant_auc = _safe_auc(y, variant)
            variant_bce = _bce(y, variant)
            boot = _bootstrap_delta(y, full, variant, bootstrap_iters, rng)
            rows.append(
                {
                    "dataset": dataset,
                    "subgroup": subgroup,
                    "comparison": f"full_vs_{comparison}",
                    "n_eval": int(len(sub)),
                    "full_auc": full_auc,
                    "variant_auc": variant_auc,
                    "delta_auc": full_auc - variant_auc,
                    **{k: boot[k] for k in ("delta_auc_ci_low", "delta_auc_ci_high")},
                    "full_bce": full_bce,
                    "variant_bce": variant_bce,
                    "delta_bce": variant_bce - full_bce,
                    **{k: boot[k] for k in ("delta_bce_ci_low", "delta_bce_ci_high")},
                    "missing_reason": "",
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("results/main_problem_experiments_20260523"))
    parser.add_argument("--output", type=Path, default=Path("results/main_problem_experiments_20260523/coverage_conditioned_prediction_table.csv"))
    parser.add_argument("--datasets", nargs="*", default=list(DEFAULT_DATASETS))
    parser.add_argument("--subgroups", nargs="*", default=list(DEFAULT_SUBGROUPS))
    parser.add_argument("--bootstrap-iters", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260524)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for dataset in args.datasets:
        rows.extend(summarize_dataset(dataset, args.input_root, args.subgroups, args.bootstrap_iters, args.seed))
    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(out)} rows)")


if __name__ == "__main__":
    main()
