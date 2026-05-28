#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Inference-only diagnostics for the concept-evidence-gap main text.

The script reuses existing CRG/LCRF checkpoints and split files. It does not
train, change model structure, add datasets, or tune parameters.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_main_problem_experiments import (  # noqa: E402
    BASE_VARIANTS,
    CONTROL_VARIANTS,
    DatasetAssets,
    _acc,
    _annotate_coverage,
    _apply_control_variant,
    _bce,
    _ensure_dir,
    _event_concepts,
    _group_masks,
    _load_assets,
    _load_model_for_variant,
    _make_eval_frame,
    _predict_model,
    _prior_variants,
    _read_main_table,
    _restore_masks,
    _rmse,
    _route_scores_from_history,
    _safe_auc,
    _student_history_stats,
)


DATASETS = ("assist_09", "junyi", "assist_17")
DATASET_LABEL = {"assist_09": "ASSIST09", "junyi": "Junyi", "assist_17": "ASSIST17"}
REQUIRED_SUBGROUPS = (
    "direct_seen",
    "direct_unseen_bridgeable",
    "direct_unseen_unbridgeable",
    "weak_direct_high_route",
    "weak_direct_low_route",
    "high_route",
    "low_route",
)


def _bootstrap_bce_ci(
    labels: np.ndarray,
    probs: np.ndarray,
    seed: int,
    n_boot: int = 200,
) -> Tuple[float, float]:
    if len(labels) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals: List[float] = []
    n = len(labels)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals.append(_bce(labels[idx], probs[idx]))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def _metric_record(
    dataset: str,
    variant: str,
    group_name: str,
    labels: np.ndarray,
    probs: np.ndarray,
    seed: int,
    full_auc: Optional[float] = None,
    full_bce: Optional[float] = None,
) -> Dict[str, Any]:
    auc = _safe_auc(labels, probs) if len(labels) >= 50 else None
    bce = _bce(labels, probs)
    ci_low, ci_high = _bootstrap_bce_ci(labels, probs, seed)
    row: Dict[str, Any] = {
        "dataset": dataset,
        "model_or_variant": variant,
        "query_count_bin": group_name,
        "n_eval": int(len(labels)),
        "positive_rate": float(np.mean(labels)) if len(labels) else float("nan"),
        "auc": auc,
        "bce": bce,
        "rmse": _rmse(labels, probs),
        "acc": _acc(labels, probs),
        "bootstrap_unit": "event",
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "claim_status": "reported" if len(labels) >= 50 else "small_n",
    }
    if full_auc is not None:
        row["delta_auc_vs_full"] = None if auc is None or full_auc is None else float(full_auc - auc)
    if full_bce is not None:
        row["delta_bce_vs_full"] = float(bce - full_bce)
    return row


def _query_min_count(frame: pd.DataFrame, assets: DatasetAssets) -> pd.DataFrame:
    hist = _student_history_stats(assets.train_df, assets.info["cpt_id_map"])
    counts: List[int] = []
    for row in frame.itertuples(index=False):
        sid = getattr(row, "stu_id")
        hcount = hist["counts"].get(sid, {})
        concepts = _event_concepts(row, assets.info["cpt_id_map"])
        counts.append(int(min((hcount.get(c, 0) for c in concepts), default=0)))
    out = frame.copy()
    out["query_history_count"] = counts
    bins: List[str] = []
    for value in counts:
        if value <= 0:
            bins.append("0")
        elif value == 1:
            bins.append("1")
        elif value <= 3:
            bins.append("2-3")
        else:
            bins.append(">=4")
    out["query_count_bin"] = bins
    return out


def _predict_variants(
    assets: DatasetAssets,
    frame: pd.DataFrame,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
    missing: List[str],
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], np.ndarray]:
    predictions = frame[
        [
            "event_id",
            "stu_id",
            "exer_id",
            "label",
            "query_concepts_mapped",
            "query_history_count",
            "query_count_bin",
            "direct_seen",
            "direct_unseen",
            "direct_unseen_bridgeable",
            "direct_unseen_unbridgeable",
            "weak_direct_evidence",
            "high_route_mass",
            "low_route_mass",
            "route_mass_to_query",
        ]
    ].copy()
    probs_by_variant: Dict[str, np.ndarray] = {}
    labels: Optional[np.ndarray] = None

    for variant in BASE_VARIANTS:
        model = _load_model_for_variant(assets, variant, device, missing)
        if model is None:
            continue
        pred = _predict_model(model, frame, assets.info, device, batch_size, num_workers)
        predictions[f"prob_{variant}"] = pred["prob"].to_numpy()
        probs_by_variant[variant] = pred["prob"].to_numpy()
        labels = pred["label_eval"].to_numpy(dtype=np.float32)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    full_model = _load_model_for_variant(assets, "full", device, missing)
    if full_model is not None:
        for control in CONTROL_VARIANTS:
            original = _apply_control_variant(full_model, assets.info, control, seed)
            pred = _predict_model(full_model, frame, assets.info, device, batch_size, num_workers)
            predictions[f"prob_{control}"] = pred["prob"].to_numpy()
            probs_by_variant[control] = pred["prob"].to_numpy()
            if original is not None:
                _restore_masks(full_model, original)
        del full_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    missing.append(
        f"{assets.dataset}: global_only skipped because no exact existing checkpoint or inference hook is available"
    )
    if labels is None:
        labels = predictions["label"].to_numpy(dtype=np.float32)
    predictions["label_eval"] = labels
    return predictions, probs_by_variant, labels


def _coverage_masks(frame: pd.DataFrame) -> Dict[str, np.ndarray]:
    return {
        "direct_seen": frame["direct_seen"].to_numpy(dtype=bool),
        "direct_unseen_bridgeable": frame["direct_unseen_bridgeable"].to_numpy(dtype=bool),
        "direct_unseen_unbridgeable": frame["direct_unseen_unbridgeable"].to_numpy(dtype=bool),
        "weak_direct_high_route": (
            frame["weak_direct_evidence"].to_numpy(dtype=bool)
            & frame["high_route_mass"].to_numpy(dtype=bool)
        ),
        "weak_direct_low_route": (
            frame["weak_direct_evidence"].to_numpy(dtype=bool)
            & frame["low_route_mass"].to_numpy(dtype=bool)
        ),
        "high_route": frame["high_route_mass"].to_numpy(dtype=bool),
        "low_route": frame["low_route_mass"].to_numpy(dtype=bool),
    }


def _load_reused_prediction_frame(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    required = {"label_eval", "query_count_bin", "prob_full"}
    if not required.issubset(frame.columns):
        return None
    return frame


def _metrics_from_prediction_frame(
    dataset: str,
    pred: pd.DataFrame,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    labels = pred["label_eval"].to_numpy(dtype=np.float32)
    variants = [c.replace("prob_", "") for c in pred.columns if c.startswith("prob_")]
    curve_rows: List[Dict[str, Any]] = []
    for bin_name in ("0", "1", "2-3", ">=4"):
        mask = pred["query_count_bin"].astype(str).eq(bin_name).to_numpy(dtype=bool)
        if int(mask.sum()) == 0:
            continue
        for variant in ("full", "no_CRG", "no_LCRF"):
            if variant not in variants:
                continue
            curve_rows.append(
                _metric_record(
                    dataset,
                    variant,
                    bin_name,
                    labels[mask],
                    pred[f"prob_{variant}"].to_numpy(dtype=np.float32)[mask],
                    seed + len(curve_rows) * 13,
                )
            )
    cov_rows: List[Dict[str, Any]] = []
    masks = {
        "direct_seen": pred["direct_seen"].to_numpy(dtype=bool),
        "direct_unseen_bridgeable": pred["direct_unseen_bridgeable"].to_numpy(dtype=bool),
        "direct_unseen_unbridgeable": pred["direct_unseen_unbridgeable"].to_numpy(dtype=bool),
        "weak_direct_high_route": pred["weak_direct_evidence"].to_numpy(dtype=bool)
        & pred["high_route_mass"].to_numpy(dtype=bool),
        "weak_direct_low_route": pred["weak_direct_evidence"].to_numpy(dtype=bool)
        & pred["low_route_mass"].to_numpy(dtype=bool),
        "high_route": pred["high_route_mass"].to_numpy(dtype=bool),
        "low_route": pred["low_route_mass"].to_numpy(dtype=bool),
    }
    for subgroup, mask in masks.items():
        if int(mask.sum()) == 0 or "full" not in variants:
            continue
        full_probs = pred["prob_full"].to_numpy(dtype=np.float32)
        full_auc = _safe_auc(labels[mask], full_probs[mask])
        full_bce = _bce(labels[mask], full_probs[mask])
        for variant in variants:
            probs = pred[f"prob_{variant}"].to_numpy(dtype=np.float32)
            row = _metric_record(
                dataset,
                variant,
                subgroup,
                labels[mask],
                probs[mask],
                seed + len(cov_rows) * 17,
                full_auc=full_auc,
                full_bce=full_bce,
            )
            row["subgroup"] = row.pop("query_count_bin")
            row["variant"] = row.pop("model_or_variant")
            if row["variant"] == "full":
                row["delta_auc_vs_full"] = 0.0
                row["delta_bce_vs_full"] = 0.0
            row["claim_status"] = _coverage_claim_status(subgroup, row)
            cov_rows.append(row)
    return pd.DataFrame(curve_rows), pd.DataFrame(cov_rows)


def run_prediction_diagnostics(
    assets: DatasetAssets,
    out_root: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
    missing: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reused = _load_reused_prediction_frame(out_root / f"{assets.dataset}_sample_level_predictions.csv")
    if reused is not None:
        curve, coverage = _metrics_from_prediction_frame(assets.dataset, reused, seed)
        return reused, curve, coverage

    frame = _annotate_coverage(_make_eval_frame(assets), assets)
    frame = _query_min_count(frame, assets)
    pred, probs_by_variant, labels = _predict_variants(
        assets, frame, device, batch_size, num_workers, seed, missing
    )
    pred.to_csv(out_root / f"{assets.dataset}_sample_level_predictions.csv", index=False)

    curve_rows: List[Dict[str, Any]] = []
    for bin_name in ("0", "1", "2-3", ">=4"):
        mask = frame["query_count_bin"].eq(bin_name).to_numpy(dtype=bool)
        if int(mask.sum()) == 0:
            continue
        for variant in ("full", "no_CRG", "no_LCRF"):
            if variant not in probs_by_variant:
                continue
            curve_rows.append(
                _metric_record(
                    assets.dataset,
                    variant,
                    bin_name,
                    labels[mask],
                    probs_by_variant[variant][mask],
                    seed + len(curve_rows) * 13,
                )
            )
    curve = pd.DataFrame(curve_rows)

    cov_rows: List[Dict[str, Any]] = []
    for subgroup, mask in _coverage_masks(frame).items():
        if int(mask.sum()) == 0 or "full" not in probs_by_variant:
            continue
        full_auc = _safe_auc(labels[mask], probs_by_variant["full"][mask])
        full_bce = _bce(labels[mask], probs_by_variant["full"][mask])
        for variant, probs in probs_by_variant.items():
            row = _metric_record(
                assets.dataset,
                variant,
                subgroup,
                labels[mask],
                probs[mask],
                seed + len(cov_rows) * 17,
                full_auc=full_auc,
                full_bce=full_bce,
            )
            row["subgroup"] = row.pop("query_count_bin")
            row["variant"] = row.pop("model_or_variant")
            if row["variant"] == "full":
                row["delta_auc_vs_full"] = 0.0
                row["delta_bce_vs_full"] = 0.0
            row["claim_status"] = _coverage_claim_status(subgroup, row)
            cov_rows.append(row)
    coverage = pd.DataFrame(cov_rows)
    return pred, curve, coverage


def _coverage_claim_status(subgroup: str, row: Mapping[str, Any]) -> str:
    if int(row.get("n_eval", 0)) < 50:
        return "small_n"
    if row.get("variant") == "full":
        return "reference"
    delta_bce = float(row.get("delta_bce_vs_full", 0.0) or 0.0)
    delta_auc = row.get("delta_auc_vs_full", None)
    delta_auc_v = 0.0 if delta_auc is None or (isinstance(delta_auc, float) and not np.isfinite(delta_auc)) else float(delta_auc)
    if subgroup in {"direct_unseen_bridgeable", "weak_direct_high_route", "high_route"} and (
        delta_bce >= 0.005 or delta_auc_v >= 0.01
    ):
        return "target_subgroup_gain"
    if subgroup in {"direct_unseen_unbridgeable", "weak_direct_low_route", "low_route"} and (
        delta_bce >= 0.005 or delta_auc_v >= 0.01
    ):
        return "negative_control_mixed"
    return "weak_or_mixed"


def run_crg_source_decomposition(out_root: Path) -> pd.DataFrame:
    src = ROOT / "results" / "main_problem_experiments_20260523"
    core = ROOT / "results" / "crg_lcrf_core3_final_20260520"
    story = pd.read_csv(core / "data_story" / "dataset_story_cards_core3.csv")
    dens = {
        str(r["dataset"]): {
            "item_edge_density": r.get("item_edge_density", np.nan),
            "seq_edge_density": r.get("seq_edge_density", np.nan),
        }
        for _, r in story.iterrows()
    }
    rows: List[Dict[str, Any]] = []
    for dataset in DATASETS:
        path = src / dataset / "main_problem_exp1_history_to_query_route_summary.csv"
        if path.exists():
            hq = pd.read_csv(path)
            for _, r in hq[hq["group"].eq("direct_unseen_bridgeable")].iterrows():
                rows.append(
                    {
                        "dataset": dataset,
                        "task": "history_to_query",
                        "method": r.get("method", r.get("variant", "")),
                        "n_eval": int(r["n_eval"]),
                        "hit5": r["hit5"],
                        "hit10": r["hit10"],
                        "ndcg10": r["ndcg10"],
                        "mrr": r["mrr"],
                        "item_edge_density": dens.get(dataset, {}).get("item_edge_density", np.nan),
                        "seq_edge_density": dens.get(dataset, {}).get("seq_edge_density", np.nan),
                    }
                )
    held = pd.read_csv(core / "crg_retrieval" / "crg_retrieval_full_core3.csv")
    for _, r in held.iterrows():
        dataset = str(r["dataset"])
        if dataset not in DATASETS:
            continue
        rows.append(
            {
                "dataset": dataset,
                "task": "heldout_transition",
                "method": r.get("method", r.get("variant", "")),
                "n_eval": int(r.get("n_eval", r.get("pairs", 0))),
                "hit5": r.get("hit5", r.get("hit@5", np.nan)),
                "hit10": r.get("hit10", r.get("hit@10", np.nan)),
                "ndcg10": r.get("ndcg10", r.get("ndcg@10", np.nan)),
                "mrr": r.get("mrr", np.nan),
                "item_edge_density": dens.get(dataset, {}).get("item_edge_density", np.nan),
                "seq_edge_density": dens.get(dataset, {}).get("seq_edge_density", np.nan),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    random_ref = (
        df[df["method"].isin(["random", "degree-random"])]
        .groupby(["dataset", "task"], as_index=False)["hit10"]
        .max()
        .rename(columns={"hit10": "random_ref_hit10"})
    )
    df = df.merge(random_ref, on=["dataset", "task"], how="left")
    df["best_minus_random"] = df["hit10"] - df["random_ref_hit10"]
    df["claim_status"] = np.where(
        (df["method"].isin(["seq-only", "fused CRG", "Best CRG", "best_crg"])) & (df["best_minus_random"] >= 0.05),
        "route_signal",
        "reported",
    )
    return df


def run_lcrf_alignment(out_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    src = ROOT / "results" / "crg_lcrf_core3_final_20260520" / "lcrf_same_query" / "lcrf_same_query_annotated_core3.csv"
    if not src.exists():
        return pd.DataFrame(), pd.DataFrame()
    df = pd.read_csv(src)
    fields = [
        "query_mastery",
        "query_recent_mastery",
        "support_mastery",
        "support_recent_mastery",
        "support_count",
        "prediction_shift_full_minus_global",
        "mean_pairwise_l1",
    ]
    rows: List[Dict[str, Any]] = []
    for dataset, part in df.groupby("dataset"):
        if dataset not in DATASETS:
            continue
        for feature in fields:
            if feature not in part.columns:
                continue
            valid = part[[feature, "posterior_minus_global"]].dropna()
            if len(valid) < 3:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "feature": feature,
                    "n_eval": int(len(valid)),
                    "pearson": float(valid[feature].corr(valid["posterior_minus_global"], method="pearson")),
                    "spearman": float(valid[feature].corr(valid["posterior_minus_global"], method="spearman")),
                    "claim_status": "case_trend" if len(valid) >= 10 else "case_small_n",
                }
            )
    align = pd.DataFrame(rows)
    summaries: List[Dict[str, Any]] = []
    if not df.empty:
        for dataset, part in df[df["dataset"].isin(DATASETS)].groupby("dataset"):
            case = part.sort_values("mean_pairwise_l1", ascending=False).head(1)
            if case.empty:
                continue
            summaries.append(
                {
                    "dataset": dataset,
                    "case_id": case.iloc[0].get("case_id"),
                    "mean_pairwise_l1": case.iloc[0].get("mean_pairwise_l1"),
                    "mean_pairwise_js": case.iloc[0].get("mean_pairwise_js"),
                    "paper_wording": "posterior shifts are aligned with learner-state variables in selected cases",
                }
            )
    return align, pd.DataFrame(summaries)


def run_direct_removal_boundary(out_root: Path) -> pd.DataFrame:
    src_root = ROOT / "results" / "main_problem_experiments_20260523"
    rows: List[Dict[str, Any]] = []
    for dataset in DATASETS:
        path = src_root / dataset / "main_problem_exp3_direct_evidence_removal_summary.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        paired = df[df.get("subgroup", pd.Series([], dtype=str)).eq("paired_bce_increase")].copy()
        if paired.empty:
            continue
        for _, r in paired.iterrows():
            target = r.get("target_bce_increase", np.nan)
            random = r.get("random_bce_increase", np.nan)
            rows.append(
                {
                    "dataset": dataset,
                    "variant": r.get("variant"),
                    "mask_type": "target_concept_mask",
                    "mask_scope": "buffer_level_state_mask",
                    "n_eval": r.get("paired_n"),
                    "auc": np.nan,
                    "bce": np.nan,
                    "rmse": np.nan,
                    "paired_bce_increase": target,
                    "target_mask_minus_random_mask": target - random if pd.notna(target) and pd.notna(random) else np.nan,
                    "claim_status": "boundary_only",
                    "paper_wording": (
                        "Direct target-concept history remains a strong diagnostic signal; "
                        "CRG/LCRF should be interpreted as route supplementation and learner-conditioned filtering when direct coverage is limited."
                    ),
                }
            )
    return pd.DataFrame(rows)


def write_manifest(out_root: Path, generated: Sequence[Path], missing: Sequence[str]) -> None:
    rows = []
    for path in generated:
        rows.append(
            {
                "artifact": str(path.relative_to(ROOT) if path.is_absolute() and path.exists() else path),
                "exists": bool(path.exists()),
                "mode": "inference_only_or_aggregation_only",
                "notes": "",
            }
        )
    for item in sorted(set(missing)):
        rows.append({"artifact": "missing_report", "exists": False, "mode": "not_run", "notes": item})
    pd.DataFrame(rows).to_csv(out_root / "diagnostic_extension_manifest.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--main-table", default=str(ROOT / "results" / "crg_lcrf_core3_final_20260520" / "main_table" / "table_main_ablation_core3.csv"))
    parser.add_argument("--output-root", default=str(ROOT / "results" / "main_problem_experiments_20260523" / "main_text"))
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260523)
    args = parser.parse_args()

    out_root = _ensure_dir(Path(args.output_root))
    missing: List[str] = []
    main_table = _read_main_table(Path(args.main_table))
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    all_curve: List[pd.DataFrame] = []
    all_cov: List[pd.DataFrame] = []
    for dataset in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        assets = _load_assets(dataset, main_table, device, missing)
        if assets is None:
            continue
        _, curve, coverage = run_prediction_diagnostics(
            assets, out_root, device, args.batch_size, args.num_workers, args.seed, missing
        )
        all_curve.append(curve)
        all_cov.append(coverage)

    curve_out = pd.concat(all_curve, ignore_index=True) if all_curve else pd.DataFrame()
    cov_out = pd.concat(all_cov, ignore_index=True) if all_cov else pd.DataFrame()
    crg_out = run_crg_source_decomposition(out_root)
    align_out, case_summary = run_lcrf_alignment(out_root)
    direct_out = run_direct_removal_boundary(out_root)

    outputs = {
        "evidence_gap_impact_curve.csv": curve_out,
        "coverage_conditioned_prediction.csv": cov_out,
        "crg_evidence_source_decomposition.csv": crg_out,
        "lcrf_posterior_state_alignment.csv": align_out,
        "caption_ready_lcrf_case_summary.csv": case_summary,
        "direct_evidence_removal_boundary_summary.csv": direct_out,
    }
    generated: List[Path] = []
    for name, frame in outputs.items():
        path = out_root / name
        frame.to_csv(path, index=False)
        generated.append(path)
    if missing:
        (ROOT / "missing_report.md").write_text("\n".join(f"- {m}" for m in sorted(set(missing))) + "\n", encoding="utf-8")
        (out_root / "missing_report.md").write_text("\n".join(f"- {m}" for m in sorted(set(missing))) + "\n", encoding="utf-8")
        generated.extend([ROOT / "missing_report.md", out_root / "missing_report.md"])
    write_manifest(out_root, generated, missing)


if __name__ == "__main__":
    main()
