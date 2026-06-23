#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""E2 concept-evidence-gap calibration curves from fixed checkpoints.

This script is inference-only. It buckets each test query by the minimum number
of times its target concepts appeared in that student's train history:
``0``, ``1-2``, ``3-5``, ``>5``.  It then reports BCE/AUC for full, no_CRG and
degree-matched random support with bootstrap confidence intervals.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_main_problem_experiments import (  # noqa: E402
    DATASETS,
    _apply_control_variant,
    _bce,
    _event_concepts,
    _load_assets,
    _load_model_for_variant,
    _make_eval_frame,
    _predict_model,
    _read_main_table,
    _restore_masks,
    _safe_auc,
    _student_history_stats,
)


BUCKET_ORDER = ("0", "1-2", "3-5", ">5")


def _parse_csv(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _bucket_count(value: int) -> str:
    v = int(value)
    if v <= 0:
        return "0"
    if v <= 2:
        return "1-2"
    if v <= 5:
        return "3-5"
    return ">5"


def _annotate_target_history_counts(frame: pd.DataFrame, assets: Any) -> pd.DataFrame:
    history = _student_history_stats(assets.train_df, assets.info["cpt_id_map"])
    min_counts: List[int] = []
    mean_counts: List[float] = []
    query_concepts: List[str] = []
    for row in frame.itertuples(index=False):
        sid = getattr(row, "stu_id")
        concepts = _event_concepts(row, assets.info["cpt_id_map"])
        counts = history["counts"].get(sid, {})
        vals = [int(counts.get(c, 0)) for c in concepts]
        min_counts.append(int(min(vals, default=0)))
        mean_counts.append(float(np.mean(vals)) if vals else 0.0)
        query_concepts.append(",".join(str(c) for c in concepts))
    out = frame.copy()
    out["query_concepts_mapped"] = query_concepts
    out["target_history_count_min"] = min_counts
    out["target_history_count_mean"] = mean_counts
    out["target_history_bucket"] = [_bucket_count(v) for v in min_counts]
    return out


def _bootstrap_metric_ci(
    labels: np.ndarray,
    probs: np.ndarray,
    *,
    metric: str,
    bootstrap: int,
    seed: int,
) -> Tuple[Optional[float], Optional[float]]:
    if len(labels) == 0 or bootstrap <= 0:
        return None, None
    rng = np.random.default_rng(seed)
    vals: List[float] = []
    n = int(len(labels))
    for _ in range(int(bootstrap)):
        idx = rng.integers(0, n, size=n)
        if metric == "auc":
            v = _safe_auc(labels[idx], probs[idx])
            if v is None:
                continue
        elif metric == "bce":
            v = _bce(labels[idx], probs[idx])
        else:
            raise ValueError(f"unknown metric: {metric}")
        vals.append(float(v))
    if not vals:
        return None, None
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def _bootstrap_delta_ci(
    labels: np.ndarray,
    full_probs: np.ndarray,
    variant_probs: np.ndarray,
    *,
    metric: str,
    bootstrap: int,
    seed: int,
) -> Tuple[Optional[float], Optional[float]]:
    if len(labels) == 0 or bootstrap <= 0:
        return None, None
    rng = np.random.default_rng(seed)
    vals: List[float] = []
    n = int(len(labels))
    for _ in range(int(bootstrap)):
        idx = rng.integers(0, n, size=n)
        if metric == "auc":
            full = _safe_auc(labels[idx], full_probs[idx])
            var = _safe_auc(labels[idx], variant_probs[idx])
            if full is None or var is None:
                continue
            vals.append(float(full - var))
        elif metric == "bce":
            vals.append(float(_bce(labels[idx], variant_probs[idx]) - _bce(labels[idx], full_probs[idx])))
        else:
            raise ValueError(f"unknown metric: {metric}")
    if not vals:
        return None, None
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def _metric_row(
    dataset: str,
    bucket: str,
    variant: str,
    labels: np.ndarray,
    probs: np.ndarray,
    full_probs: Optional[np.ndarray],
    bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    auc = _safe_auc(labels, probs)
    bce = _bce(labels, probs)
    auc_lo, auc_hi = _bootstrap_metric_ci(labels, probs, metric="auc", bootstrap=bootstrap, seed=seed + 17)
    bce_lo, bce_hi = _bootstrap_metric_ci(labels, probs, metric="bce", bootstrap=bootstrap, seed=seed + 29)
    row: Dict[str, Any] = {
        "dataset": dataset,
        "target_history_bucket": bucket,
        "variant": variant,
        "n_eval": int(len(labels)),
        "auc": auc,
        "auc_ci_low": auc_lo,
        "auc_ci_high": auc_hi,
        "bce": bce,
        "bce_ci_low": bce_lo,
        "bce_ci_high": bce_hi,
    }
    if full_probs is not None:
        full_auc = _safe_auc(labels, full_probs)
        full_bce = _bce(labels, full_probs)
        row["delta_auc_full_minus_variant"] = None if auc is None or full_auc is None else float(full_auc - auc)
        row["delta_bce_variant_minus_full"] = float(bce - full_bce)
        da_lo, da_hi = _bootstrap_delta_ci(
            labels,
            full_probs,
            probs,
            metric="auc",
            bootstrap=bootstrap,
            seed=seed + 41,
        )
        db_lo, db_hi = _bootstrap_delta_ci(
            labels,
            full_probs,
            probs,
            metric="bce",
            bootstrap=bootstrap,
            seed=seed + 53,
        )
        row["delta_auc_ci_low"] = da_lo
        row["delta_auc_ci_high"] = da_hi
        row["delta_bce_ci_low"] = db_lo
        row["delta_bce_ci_high"] = db_hi
    return row


def run_dataset(
    dataset: str,
    main_table: pd.DataFrame,
    out_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
    bootstrap: int,
    missing: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    assets = _load_assets(dataset, main_table, device, missing)
    if assets is None:
        return pd.DataFrame(), pd.DataFrame()
    frame = _annotate_target_history_counts(_make_eval_frame(assets), assets)
    predictions = frame[
        [
            "event_id",
            "stu_id",
            "exer_id",
            "label",
            "query_concepts_mapped",
            "target_history_count_min",
            "target_history_count_mean",
            "target_history_bucket",
        ]
    ].copy()
    variant_probs: Dict[str, np.ndarray] = {}

    for variant in ("full", "no_CRG"):
        model = _load_model_for_variant(assets, variant, device, missing)
        if model is None:
            continue
        pred = _predict_model(model, frame, assets.info, device, batch_size, num_workers)
        predictions[f"prob_{variant}"] = pred["prob"].to_numpy(dtype=np.float32)
        predictions["label_eval"] = pred["label_eval"].to_numpy(dtype=np.float32)
        variant_probs[variant] = pred["prob"].to_numpy(dtype=np.float32)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    full_model = _load_model_for_variant(assets, "full", device, missing)
    if full_model is not None:
        original = _apply_control_variant(full_model, assets.info, "degree_random_support", seed)
        pred = _predict_model(full_model, frame, assets.info, device, batch_size, num_workers)
        predictions["prob_degree_random_support"] = pred["prob"].to_numpy(dtype=np.float32)
        variant_probs["degree_random_support"] = pred["prob"].to_numpy(dtype=np.float32)
        if original is not None:
            _restore_masks(full_model, original)
        del full_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    labels = predictions.get("label_eval", predictions["label"]).to_numpy(dtype=np.float32)
    rows: List[Dict[str, Any]] = []
    for bucket in BUCKET_ORDER:
        mask = predictions["target_history_bucket"].eq(bucket).to_numpy(dtype=bool)
        if int(mask.sum()) == 0:
            continue
        full_probs = variant_probs.get("full")
        full_bucket = None if full_probs is None else full_probs[mask]
        for idx, (variant, probs) in enumerate(variant_probs.items()):
            rows.append(
                _metric_row(
                    dataset,
                    bucket,
                    variant,
                    labels[mask],
                    probs[mask],
                    full_bucket,
                    bootstrap,
                    seed + idx * 101 + BUCKET_ORDER.index(bucket) * 1009,
                )
            )

    pred_path = out_dir / f"{dataset}_gap_calibration_predictions.csv"
    metrics_path = out_dir / f"{dataset}_gap_calibration_metrics.csv"
    predictions.to_csv(pred_path, index=False)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(metrics_path, index=False)
    return predictions, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument(
        "--main-table",
        default=str(ROOT / "results" / "crg_lcrf_core3_final_20260520" / "main_table" / "table_main_ablation_core3.csv"),
    )
    parser.add_argument("--output-root", default=str(ROOT / "results" / "mainline_e2_gap_calibration"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()

    out_dir = Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    main_table = _read_main_table(Path(args.main_table))
    missing: List[str] = []
    all_metrics: List[pd.DataFrame] = []

    for dataset in _parse_csv(args.datasets):
        _, metrics = run_dataset(
            dataset,
            main_table,
            out_dir,
            device,
            int(args.batch_size),
            int(args.num_workers),
            int(args.seed),
            int(args.bootstrap),
            missing,
        )
        if not metrics.empty:
            all_metrics.append(metrics)

    if all_metrics:
        pd.concat(all_metrics, ignore_index=True).to_csv(out_dir / "gap_calibration_metrics_all.csv", index=False)
    if missing:
        (out_dir / "missing_or_skipped.txt").write_text("\n".join(missing) + "\n", encoding="utf-8")
    print(f"[ok] E2 gap calibration written to {out_dir}")


if __name__ == "__main__":
    main()
