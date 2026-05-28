#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Coverage-conditioned LCRF counterfactuals and gap masking stress tests.

This script is inference-only. It reuses existing checkpoints and split files,
does not retrain, and does not edit manuscript files.
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.export_ae_case_studies import _apply_personal_counterfactual  # noqa: E402
from tools.run_main_problem_experiments import (  # noqa: E402
    CONTROL_VARIANTS,
    DatasetAssets,
    _acc,
    _apply_control_variant,
    _bce,
    _ensure_dir,
    _event_concepts,
    _load_assets,
    _load_model_for_variant,
    _make_eval_frame,
    _metric_row,
    _predict_model,
    _read_main_table,
    _restore_masks,
    _rmse,
    _route_scores_from_history,
    _safe_auc,
    _student_history_stats,
)
from tools.analyze_a_support_evidence import _row_normalize  # noqa: E402


DATASETS = ("assist_09", "junyi", "assist_17")
LCRF_VARIANTS = ("full", "no_filter", "mean_state", "shuffle_state", "no_LCRF")
STRESS_VARIANTS = ("full", "no_CRG", "no_LCRF", "self_only", "degree_random_support")
MASK_RATIOS = (0.0, 0.25, 0.50, 0.75, 1.0)
GROUP_ORDER = (
    "all",
    "direct_cov_0",
    "direct_cov_1",
    "direct_cov_2_3",
    "direct_cov_ge4",
    "direct_unseen_bridgeable",
    "weak_direct_high_route",
)


def _device(name: Optional[str]) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _bootstrap_metric_ci(
    labels: np.ndarray,
    probs: np.ndarray,
    *,
    metric: str,
    seed: int,
    n_boot: int,
) -> Tuple[Optional[float], Optional[float]]:
    if n_boot <= 0 or labels.size < 2:
        return None, None
    rng = np.random.default_rng(seed)
    vals: List[float] = []
    n = int(labels.size)
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        y = labels[idx]
        p = probs[idx]
        if metric == "auc":
            val = _safe_auc(y, p)
            if val is None:
                continue
        elif metric == "bce":
            val = _bce(y, p)
        else:
            raise ValueError(metric)
        vals.append(float(val))
    if not vals:
        return None, None
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def _metric_record_ci(
    dataset: str,
    subgroup: str,
    variant: str,
    labels: np.ndarray,
    probs: np.ndarray,
    *,
    seed: int,
    n_boot: int,
    full_ref: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    row = _metric_row(dataset, subgroup, variant, labels, probs, full_ref)
    auc_low, auc_high = _bootstrap_metric_ci(labels, probs, metric="auc", seed=seed, n_boot=n_boot)
    bce_low, bce_high = _bootstrap_metric_ci(labels, probs, metric="bce", seed=seed + 17, n_boot=n_boot)
    row.update(
        {
            "auc_ci_low": auc_low,
            "auc_ci_high": auc_high,
            "bce_ci_low": bce_low,
            "bce_ci_high": bce_high,
            "bootstrap_unit": "event",
            "bootstrap_n": int(n_boot),
        }
    )
    return row


def _annotate_query_coverage(frame: pd.DataFrame, assets: DatasetAssets) -> pd.DataFrame:
    history = _student_history_stats(assets.train_df, assets.info["cpt_id_map"])
    fused = _row_normalize(
        assets.info["item_prior_matrix"].detach().cpu()
        + assets.info["sequence_prior_matrix"].detach().cpu()
    ).numpy()
    rows: List[Dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        sid = getattr(row, "stu_id")
        hset = set(history["sets"].get(sid, set()))
        hcount = history["counts"].get(sid, {})
        concepts = _event_concepts(row, assets.info["cpt_id_map"])
        counts = [int(hcount.get(c, 0)) for c in concepts]
        direct_cov_cnt = int(min(counts, default=0))
        scores: List[float] = []
        for c in concepts:
            noisy, _ = _route_scores_from_history(fused, hset)
            scores.append(float(noisy[c]) if c < noisy.size else 0.0)
        route_mass = float(max(scores)) if scores else 0.0
        rows.append(
            {
                "query_concepts_mapped": ",".join(str(c) for c in concepts),
                "direct_cov_cnt": direct_cov_cnt,
                "direct_cov_bin": (
                    "0"
                    if direct_cov_cnt <= 0
                    else "1"
                    if direct_cov_cnt == 1
                    else "2-3"
                    if direct_cov_cnt <= 3
                    else ">=4"
                ),
                "direct_seen_all_targets": bool(concepts and all(c in hset for c in concepts)),
                "direct_unseen_any_target": bool(any(c not in hset for c in concepts)),
                "direct_unseen_bridgeable": bool(direct_cov_cnt <= 0 and route_mass > 0.0),
                "weak_direct_evidence": bool(direct_cov_cnt <= 1),
                "route_mass_to_query": route_mass,
            }
        )
    out = pd.concat([frame.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    if len(out) and np.any(out["route_mass_to_query"].to_numpy(dtype=np.float64) > 0.0):
        high = float(np.quantile(out["route_mass_to_query"].to_numpy(dtype=np.float64), 0.70))
    else:
        high = float("inf")
    out["high_route_mass"] = out["route_mass_to_query"] >= high
    out["weak_direct_high_route"] = out["weak_direct_evidence"] & out["high_route_mass"]
    return out


def _group_masks(frame: pd.DataFrame) -> Dict[str, np.ndarray]:
    return {
        "all": np.ones(len(frame), dtype=bool),
        "direct_cov_0": frame["direct_cov_bin"].astype(str).eq("0").to_numpy(dtype=bool),
        "direct_cov_1": frame["direct_cov_bin"].astype(str).eq("1").to_numpy(dtype=bool),
        "direct_cov_2_3": frame["direct_cov_bin"].astype(str).eq("2-3").to_numpy(dtype=bool),
        "direct_cov_ge4": frame["direct_cov_bin"].astype(str).eq(">=4").to_numpy(dtype=bool),
        "direct_unseen_bridgeable": frame["direct_unseen_bridgeable"].to_numpy(dtype=bool),
        "weak_direct_high_route": frame["weak_direct_high_route"].to_numpy(dtype=bool),
    }


def _set_personal_enabled(model: torch.nn.Module, enabled: bool) -> Dict[str, Any]:
    original: Dict[str, Any] = {}
    original["model.use_personal_graph"] = getattr(model, "use_personal_graph", None)
    if hasattr(model, "use_personal_graph"):
        model.use_personal_graph = bool(enabled)
    structure = getattr(model, "structure_module", None)
    original["structure.use_personal_graph"] = getattr(structure, "use_personal_graph", None)
    if structure is not None and hasattr(structure, "use_personal_graph"):
        structure.use_personal_graph = bool(enabled)
    return original


def _restore_personal_enabled(model: torch.nn.Module, original: Mapping[str, Any]) -> None:
    if original.get("model.use_personal_graph") is not None and hasattr(model, "use_personal_graph"):
        model.use_personal_graph = bool(original["model.use_personal_graph"])
    structure = getattr(model, "structure_module", None)
    if (
        structure is not None
        and original.get("structure.use_personal_graph") is not None
        and hasattr(structure, "use_personal_graph")
    ):
        structure.use_personal_graph = bool(original["structure.use_personal_graph"])


def _predict_lcrf_variant(
    assets: DatasetAssets,
    variant: str,
    frame: pd.DataFrame,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
    missing: List[str],
) -> Optional[pd.DataFrame]:
    base_variant = "no_LCRF" if variant == "no_LCRF" else "full"
    model = _load_model_for_variant(assets, base_variant, device, missing)
    if model is None:
        return None
    personal_original: Optional[Dict[str, Any]] = None
    try:
        if variant == "no_filter":
            personal_original = _set_personal_enabled(model, False)
        elif variant == "mean_state":
            _apply_personal_counterfactual(model, "mean", seed=seed)
        elif variant == "shuffle_state":
            _apply_personal_counterfactual(model, "shuffle", seed=seed)
        pred = _predict_model(model, frame, assets.info, device, batch_size, num_workers)
        pred = pred.rename(columns={"prob": f"prob_{variant}"})
        return pred[["event_id", "stu_id", "exer_id", "label_eval", f"prob_{variant}"]]
    finally:
        if personal_original is not None:
            _restore_personal_enabled(model, personal_original)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_lcrf_strata(
    assets: DatasetAssets,
    out_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
    n_boot: int,
    missing: List[str],
    max_rows: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frame = _annotate_query_coverage(_make_eval_frame(assets), assets)
    if max_rows > 0 and len(frame) > max_rows:
        frame = frame.sample(n=max_rows, random_state=seed).sort_values("event_id").reset_index(drop=True)
    predictions = frame[
        [
            "event_id",
            "stu_id",
            "exer_id",
            "label",
            "query_concepts_mapped",
            "direct_cov_cnt",
            "direct_cov_bin",
            "direct_unseen_bridgeable",
            "weak_direct_high_route",
            "route_mass_to_query",
        ]
    ].copy()
    labels: Optional[np.ndarray] = None
    probs_by_variant: Dict[str, np.ndarray] = {}
    for variant in LCRF_VARIANTS:
        print(f"[{assets.dataset}] LCRF variant: {variant}", flush=True)
        pred = _predict_lcrf_variant(assets, variant, frame, device, batch_size, num_workers, seed, missing)
        if pred is None:
            continue
        if labels is None:
            labels = pred["label_eval"].to_numpy(dtype=np.float32)
            predictions["label_eval"] = labels
        else:
            aligned = predictions[["event_id", "stu_id", "exer_id"]].reset_index(drop=True)
            check = pred[["event_id", "stu_id", "exer_id"]].reset_index(drop=True)
            if not aligned.equals(check):
                raise ValueError(f"{assets.dataset} {variant}: sample alignment mismatch")
        predictions[f"prob_{variant}"] = pred[f"prob_{variant}"].to_numpy(dtype=np.float32)
        probs_by_variant[variant] = predictions[f"prob_{variant}"].to_numpy(dtype=np.float32)

    if labels is None:
        labels = predictions["label"].to_numpy(dtype=np.float32)

    rows: List[Dict[str, Any]] = []
    masks = _group_masks(frame)
    for group_name in GROUP_ORDER:
        mask = masks[group_name]
        if int(mask.sum()) == 0 or "full" not in probs_by_variant:
            continue
        full_ref = {
            "auc": _safe_auc(labels[mask], probs_by_variant["full"][mask]),
            "bce": _bce(labels[mask], probs_by_variant["full"][mask]),
        }
        for variant, probs in probs_by_variant.items():
            rows.append(
                _metric_record_ci(
                    assets.dataset,
                    group_name,
                    variant,
                    labels[mask],
                    probs[mask],
                    seed=seed + len(rows) * 19,
                    n_boot=n_boot,
                    full_ref=full_ref,
                )
            )
    metrics = pd.DataFrame(rows)
    predictions.to_csv(out_dir / "lcrf_strata_predictions.csv", index=False)
    metrics.to_csv(out_dir / "lcrf_strata_metrics.csv", index=False)
    return predictions, metrics


MASK_ATTRS = (
    "ae_student_concept_prior_logit",
    "ae_student_concept_recent_logit",
    "ae_student_concept_count_feature",
    "ae_student_concept_observed_count",
)


def _mask_student_concepts_ratio(
    model: torch.nn.Module,
    mapped_student: int,
    concepts: Sequence[int],
    ratio: float,
) -> Dict[str, torch.Tensor]:
    ratio = min(max(float(ratio), 0.0), 1.0)
    saved: Dict[str, torch.Tensor] = {}
    if ratio <= 0.0 or not concepts:
        return saved
    idx = sorted({int(c) for c in concepts})
    with torch.no_grad():
        for name in MASK_ATTRS:
            tensor = getattr(model, name, None)
            if tensor is None or not torch.is_tensor(tensor):
                continue
            saved[name] = tensor[int(mapped_student), idx].detach().clone()
            tensor[int(mapped_student), idx] = tensor[int(mapped_student), idx] * float(1.0 - ratio)
    return saved


def _restore_student_concepts_ratio(
    model: torch.nn.Module,
    mapped_student: int,
    concepts: Sequence[int],
    saved: Mapping[str, torch.Tensor],
) -> None:
    if not saved:
        return
    idx = sorted({int(c) for c in concepts})
    with torch.no_grad():
        for name, value in saved.items():
            tensor = getattr(model, name, None)
            if tensor is not None and torch.is_tensor(tensor):
                tensor[int(mapped_student), idx] = value.to(device=tensor.device, dtype=tensor.dtype)


def _single_predict(model: torch.nn.Module, row: pd.Series, info: Mapping[str, Any], device: torch.device) -> float:
    sid = torch.tensor([int(info["stu_id_map"][row["stu_id"]])], dtype=torch.long, device=device)
    eid = torch.tensor([int(info["exer_id_map"][row["exer_id"]])], dtype=torch.long, device=device)
    with torch.no_grad():
        logit = model(sid, eid, return_details=False, return_logits=True).reshape(-1)
        return float(torch.sigmoid(logit)[0].detach().cpu().item())


def _predict_mask_ratio_samples(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    assets: DatasetAssets,
    device: torch.device,
    ratio: float,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    model.eval()
    for row in frame.itertuples(index=False):
        row_s = pd.Series(row._asdict())
        concepts = _event_concepts(row, assets.info["cpt_id_map"])
        mapped_student = int(assets.info["stu_id_map"][getattr(row, "stu_id")])
        saved = _mask_student_concepts_ratio(model, mapped_student, concepts, ratio)
        prob = _single_predict(model, row_s, assets.info, device)
        _restore_student_concepts_ratio(model, mapped_student, concepts, saved)
        label = float(getattr(row, "label"))
        p = min(max(prob, 1e-8), 1.0 - 1e-8)
        bce = -(label * math.log(p) + (1.0 - label) * math.log(1.0 - p))
        rows.append(
            {
                "event_id": int(getattr(row, "event_id")),
                "stu_id": getattr(row, "stu_id"),
                "exer_id": getattr(row, "exer_id"),
                "label": label,
                "pred_prob": prob,
                "bce": bce,
                "query_concepts_mapped": ",".join(str(c) for c in concepts),
                "mask_ratio": float(ratio),
                "mask_scope": "student_concept_state_buffer_ratio_mask",
            }
        )
    return pd.DataFrame(rows)


def _select_stress_frame(frame: pd.DataFrame, max_samples: int, seed: int) -> pd.DataFrame:
    direct = frame[frame["direct_cov_cnt"] > 0].copy().reset_index(drop=True)
    if max_samples > 0 and len(direct) > max_samples:
        direct = direct.sample(n=max_samples, random_state=seed).sort_values("event_id").reset_index(drop=True)
    return direct


def _load_stress_model(
    assets: DatasetAssets,
    variant: str,
    device: torch.device,
    seed: int,
    missing: List[str],
) -> Tuple[Optional[torch.nn.Module], Optional[Mapping[str, torch.Tensor]]]:
    base_variant = "full" if variant in CONTROL_VARIANTS else variant
    model = _load_model_for_variant(assets, base_variant, device, missing)
    if model is None:
        return None, None
    original_masks = None
    if variant in CONTROL_VARIANTS:
        original_masks = _apply_control_variant(model, assets.info, variant, seed)
    return model, original_masks


def run_direct_mask_stress(
    assets: DatasetAssets,
    out_dir: Path,
    device: torch.device,
    seed: int,
    n_boot: int,
    missing: List[str],
    max_samples: int,
    max_rows: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frame = _annotate_query_coverage(_make_eval_frame(assets), assets)
    stress = _select_stress_frame(frame, max_samples=max_samples, seed=seed)
    if max_rows > 0 and len(stress) > max_rows:
        stress = stress.sample(n=max_rows, random_state=seed + 7).sort_values("event_id").reset_index(drop=True)
    sample_parts: List[pd.DataFrame] = []
    metric_rows: List[Dict[str, Any]] = []
    for variant in STRESS_VARIANTS:
        print(f"[{assets.dataset}] stress variant: {variant}", flush=True)
        model, original_masks = _load_stress_model(assets, variant, device, seed, missing)
        if model is None:
            continue
        try:
            for ratio in MASK_RATIOS:
                print(f"[{assets.dataset}] stress {variant} mask_ratio={ratio:.2f} n={len(stress)}", flush=True)
                pred = _predict_mask_ratio_samples(model, stress, assets, device, ratio)
                pred.insert(0, "variant", variant)
                pred.insert(0, "dataset", assets.dataset)
                sample_parts.append(pred)
                labels = pred["label"].to_numpy(dtype=np.float32)
                probs = pred["pred_prob"].to_numpy(dtype=np.float32)
                row = _metric_record_ci(
                    assets.dataset,
                    "direct_cov_available_sample",
                    variant,
                    labels,
                    probs,
                    seed=seed + len(metric_rows) * 23,
                    n_boot=n_boot,
                    full_ref=None,
                )
                row["mask_ratio"] = float(ratio)
                row["mask_scope"] = "student_concept_state_buffer_ratio_mask"
                metric_rows.append(row)
        finally:
            if original_masks is not None:
                _restore_masks(model, original_masks)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    sample = pd.concat(sample_parts, ignore_index=True) if sample_parts else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows)
    if not metrics.empty:
        baseline = metrics[metrics["mask_ratio"].eq(0.0)][["variant", "auc", "bce"]].rename(
            columns={"auc": "auc_at_0", "bce": "bce_at_0"}
        )
        metrics = metrics.merge(baseline, on="variant", how="left")
        metrics["auc_retention"] = metrics["auc"].astype(float) / metrics["auc_at_0"].astype(float)
        metrics["bce_increase_from_0"] = metrics["bce"].astype(float) - metrics["bce_at_0"].astype(float)
    sample.to_csv(out_dir / "direct_mask_stress_samples.csv", index=False)
    metrics.to_csv(out_dir / "direct_mask_stress_metrics.csv", index=False)
    missing.append(
        f"{assets.dataset}: stress test uses proportional student-concept state-buffer masking, not raw history recomputation"
    )
    return sample, metrics


def _read_csvs(paths: Iterable[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        try:
            frames.append(pd.read_csv(path))
        except (FileNotFoundError, pd.errors.EmptyDataError):
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _md_table(frame: pd.DataFrame, *, max_rows: int = 80) -> str:
    if frame.empty:
        return "No rows."
    show = frame.head(max_rows).copy()
    cols = list(show.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in show.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append("" if not np.isfinite(value) else f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    if len(frame) > max_rows:
        lines.append(f"| ... | truncated to {max_rows} rows | |")
    return "\n".join(lines)


def _plot_heatmap(ax: Any, table: pd.DataFrame, dataset: str) -> None:
    sub = table[(table["dataset"].eq(dataset)) & (table["variant"].isin(["no_filter", "mean_state", "shuffle_state", "no_LCRF"]))]
    groups = [g for g in GROUP_ORDER if g in set(sub["subgroup"])]
    variants = [v for v in ["no_filter", "mean_state", "shuffle_state", "no_LCRF"] if v in set(sub["variant"])]
    mat = np.full((len(variants), len(groups)), np.nan, dtype=np.float64)
    for i, variant in enumerate(variants):
        for j, group in enumerate(groups):
            hit = sub[sub["variant"].eq(variant) & sub["subgroup"].eq(group)]
            if not hit.empty:
                mat[i, j] = float(hit.iloc[0].get("auc_gap_full_minus_variant", np.nan))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0.0)
    ax.set_title(dataset)
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(groups, rotation=35, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(variants)))
    ax.set_yticklabels(variants, fontsize=8)
    for i in range(len(variants)):
        for j in range(len(groups)):
            value = mat[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=6)
    return im


def plot_outputs(out_root: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (out_root / "plot_missing_report.md").write_text(f"matplotlib unavailable: {exc}\n", encoding="utf-8")
        return
    fig_dir = _ensure_dir(out_root / "figures")
    lcrf = _read_csvs(out_root.glob("*/lcrf_strata_metrics.csv"))
    stress = _read_csvs(out_root.glob("*/direct_mask_stress_metrics.csv"))

    if not lcrf.empty:
        datasets = [d for d in DATASETS if d in set(lcrf["dataset"])]
        fig, axes = plt.subplots(1, len(datasets), figsize=(4.4 * len(datasets), 3.4), squeeze=False)
        last_im = None
        for ax, dataset in zip(axes[0], datasets):
            last_im = _plot_heatmap(ax, lcrf, dataset)
        if last_im is not None:
            fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.8, label="AUC drop from Full")
        fig.suptitle("Coverage-conditioned LCRF counterfactual")
        fig.tight_layout()
        fig.savefig(fig_dir / "fig_lcrf_coverage_counterfactual_heatmap.png", dpi=220)
        fig.savefig(fig_dir / "fig_lcrf_coverage_counterfactual_heatmap.pdf")
        plt.close(fig)

    if not stress.empty:
        for metric, ylabel, outfile in (
            ("auc_retention", "AUC retention vs mask=0", "fig_direct_mask_auc_retention"),
            ("bce_increase_from_0", "BCE increase vs mask=0", "fig_direct_mask_bce_increase"),
        ):
            fig, axes = plt.subplots(1, len(DATASETS), figsize=(4.2 * len(DATASETS), 3.2), squeeze=False)
            for ax, dataset in zip(axes[0], DATASETS):
                sub = stress[stress["dataset"].eq(dataset)]
                for variant in STRESS_VARIANTS:
                    one = sub[sub["variant"].eq(variant)].sort_values("mask_ratio")
                    if one.empty or metric not in one:
                        continue
                    ax.plot(one["mask_ratio"], one[metric], marker="o", linewidth=1.5, label=variant)
                ax.set_title(dataset)
                ax.set_xlabel("Mask ratio")
                ax.set_ylabel(ylabel)
                ax.grid(alpha=0.25)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), fontsize=8)
            fig.tight_layout(rect=(0, 0.11, 1, 1))
            fig.savefig(fig_dir / f"{outfile}.png", dpi=220)
            fig.savefig(fig_dir / f"{outfile}.pdf")
            plt.close(fig)


def _choose_target_group(lcrf: pd.DataFrame, dataset: str) -> Optional[str]:
    sub = lcrf[lcrf["dataset"].eq(dataset)]
    for group in ("direct_unseen_bridgeable", "weak_direct_high_route"):
        hit = sub[sub["subgroup"].eq(group)]
        if not hit.empty and int(pd.to_numeric(hit["n_eval"], errors="coerce").max()) >= 500:
            return group
    return None


def _lcrf_dataset_gate(lcrf: pd.DataFrame, dataset: str) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    target_group = _choose_target_group(lcrf, dataset)
    if target_group is None:
        return False, ["no target LCRF subgroup with n_eval >= 500"]
    sub = lcrf[lcrf["dataset"].eq(dataset)]
    overall = sub[sub["subgroup"].eq("all")]
    target = sub[sub["subgroup"].eq(target_group)]
    passed = True
    for variant in ("mean_state", "shuffle_state"):
        row = target[target["variant"].eq(variant)]
        all_row = overall[overall["variant"].eq(variant)]
        if row.empty or all_row.empty:
            passed = False
            reasons.append(f"{variant}: missing target or overall row")
            continue
        auc_drop = _safe_float(row.iloc[0].get("auc_gap_full_minus_variant")) or 0.0
        bce_inc = _safe_float(row.iloc[0].get("bce_gap_variant_minus_full")) or 0.0
        all_auc_drop = _safe_float(all_row.iloc[0].get("auc_gap_full_minus_variant")) or 0.0
        if not (auc_drop > 0.0 and bce_inc > 0.0):
            passed = False
            reasons.append(f"{variant}: target AUC/BCE direction failed ({target_group})")
        if not (auc_drop > all_auc_drop):
            passed = False
            reasons.append(f"{variant}: target AUC drop not larger than overall")
    for variant in ("no_filter", "no_LCRF"):
        row = target[target["variant"].eq(variant)]
        if row.empty:
            passed = False
            reasons.append(f"{variant}: missing target row")
            continue
        bce_inc = _safe_float(row.iloc[0].get("bce_gap_variant_minus_full")) or 0.0
        if bce_inc < 0.0:
            passed = False
            reasons.append(f"{variant}: BCE better than full in {target_group}")
    if passed:
        reasons.append(f"passed on {target_group}")
    return passed, reasons


def _stress_dataset_gate(stress: pd.DataFrame, dataset: str) -> Tuple[bool, List[str]]:
    sub = stress[stress["dataset"].eq(dataset)]
    reasons: List[str] = []
    if sub.empty:
        return False, ["missing stress metrics"]
    if int(pd.to_numeric(sub["n_eval"], errors="coerce").max()) < 500:
        return False, ["stress n_eval < 500"]
    final = sub[sub["mask_ratio"].astype(float).eq(1.0)]
    zero = sub[sub["mask_ratio"].astype(float).eq(0.0)]
    required = {"full", "no_CRG", "no_LCRF"}
    if not required.issubset(set(final["variant"])) or not required.issubset(set(zero["variant"])):
        return False, ["missing full/no_CRG/no_LCRF stress rows"]
    full_ret = float(final[final["variant"].eq("full")].iloc[0]["auc_retention"])
    full_bce_inc = float(final[final["variant"].eq("full")].iloc[0]["bce_increase_from_0"])
    passed = True
    for variant in ("no_CRG", "no_LCRF"):
        row = final[final["variant"].eq(variant)].iloc[0]
        ret = float(row["auc_retention"])
        bce_inc = float(row["bce_increase_from_0"])
        if not (full_ret > ret):
            passed = False
            reasons.append(f"full AUC retention <= {variant} at mask=1.0")
        if not (full_bce_inc <= bce_inc):
            passed = False
            reasons.append(f"full BCE increase > {variant} at mask=1.0")
    if passed:
        reasons.append("passed at mask=1.0")
    return passed, reasons


def write_report(out_root: Path) -> pd.DataFrame:
    lcrf = _read_csvs(out_root.glob("*/lcrf_strata_metrics.csv"))
    stress = _read_csvs(out_root.glob("*/direct_mask_stress_metrics.csv"))
    rows: List[Dict[str, Any]] = []
    for dataset in DATASETS:
        l_ok, l_reasons = _lcrf_dataset_gate(lcrf, dataset)
        s_ok, s_reasons = _stress_dataset_gate(stress, dataset)
        rows.append(
            {
                "dataset": dataset,
                "lcrf_pass": bool(l_ok),
                "stress_pass": bool(s_ok),
                "overall_pass": bool(l_ok and s_ok),
                "lcrf_notes": "; ".join(l_reasons),
                "stress_notes": "; ".join(s_reasons),
            }
        )
    gate = pd.DataFrame(rows)
    gate.to_csv(out_root / "success_gate_summary.csv", index=False)
    overall = bool(gate["overall_pass"].all()) if not gate.empty else False
    lines = [
        "# Gap Stress and LCRF Strata Experiment Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Overall status: {'PASS' if overall else 'FAIL'}",
        "",
        "Important scope note: direct-evidence masking is implemented as proportional masking of train-derived "
        "student-concept state buffers, not raw chronological history recomputation.",
        "",
        "## Success Gate Summary",
        "",
        _md_table(gate),
        "",
    ]
    if not lcrf.empty:
        focus = lcrf[
            lcrf["subgroup"].isin(["all", "direct_unseen_bridgeable", "weak_direct_high_route"])
            & lcrf["variant"].isin(["mean_state", "shuffle_state", "no_filter", "no_LCRF"])
        ][
            [
                "dataset",
                "subgroup",
                "variant",
                "n_eval",
                "auc",
                "bce",
                "auc_gap_full_minus_variant",
                "bce_gap_variant_minus_full",
            ]
        ]
        lines.extend(["## LCRF Focus Metrics", "", _md_table(focus), ""])
    if not stress.empty:
        focus = stress[
            stress["variant"].isin(["full", "no_CRG", "no_LCRF"])
            & stress["mask_ratio"].astype(float).isin([0.0, 0.5, 1.0])
        ][
            [
                "dataset",
                "variant",
                "mask_ratio",
                "n_eval",
                "auc",
                "bce",
                "auc_retention",
                "bce_increase_from_0",
            ]
        ]
        lines.extend(["## Direct-Mask Focus Metrics", "", _md_table(focus), ""])
    lines.extend(
        [
            "## Outputs",
            "",
            "- `lcrf_strata_metrics.csv` per dataset",
            "- `direct_mask_stress_metrics.csv` per dataset",
            "- `figures/fig_lcrf_coverage_counterfactual_heatmap.*`",
            "- `figures/fig_direct_mask_auc_retention.*`",
            "- `figures/fig_direct_mask_bce_increase.*`",
            "",
        ]
    )
    (out_root / "experiment_report.md").write_text("\n".join(lines), encoding="utf-8")
    return gate


def _check_only(args: argparse.Namespace, main_table: pd.DataFrame) -> None:
    rows: List[Dict[str, Any]] = []
    for dataset in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        hit = main_table[main_table["dataset"].astype(str).eq(dataset)]
        rows.append(
            {
                "dataset": dataset,
                "has_full": bool((hit["variant"].astype(str) == "full").any()),
                "has_no_CRG": bool((hit["variant"].astype(str) == "no_CRG").any()),
                "has_no_LCRF": bool((hit["variant"].astype(str) == "no_LCRF").any()),
            }
        )
    out_root = _ensure_dir(Path(args.output_root))
    pd.DataFrame(rows).to_csv(out_root / "checkpoint_table_check.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference-only LCRF strata and direct evidence masking stress tests.")
    parser.add_argument("--datasets", default="assist_09,junyi,assist_17")
    parser.add_argument(
        "--main-table",
        default="results/crg_lcrf_core3_final_20260520/main_table/table_main_ablation_core3.csv",
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--stress-max-samples", type=int, default=5000)
    parser.add_argument("--max-rows", type=int, default=0, help="Debug cap applied after dataset-specific filtering.")
    parser.add_argument("--skip-lcrf", action="store_true")
    parser.add_argument("--skip-stress", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(
        args.output_root
        or f"results/gap_stress_lcrf_strata_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    _ensure_dir(out_root)
    if args.plot_only:
        plot_outputs(out_root)
        write_report(out_root)
        return

    main_table = _read_main_table(Path(args.main_table))
    if args.check_only:
        _check_only(args, main_table)
        return

    device = _device(args.device)
    missing: List[str] = []
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for dataset in datasets:
        ds_out = _ensure_dir(out_root / dataset)
        assets = _load_assets(dataset, main_table, device, missing)
        if assets is None:
            continue
        if not args.skip_lcrf:
            run_lcrf_strata(
                assets,
                ds_out,
                device,
                args.batch_size,
                args.num_workers,
                args.seed,
                args.bootstrap,
                missing,
                args.max_rows,
            )
        if not args.skip_stress:
            run_direct_mask_stress(
                assets,
                ds_out,
                device,
                args.seed,
                args.bootstrap,
                missing,
                args.stress_max_samples,
                args.max_rows,
            )
    (out_root / "missing_report.md").write_text("\n".join(f"- {item}" for item in missing) + "\n", encoding="utf-8")
    plot_outputs(out_root)
    gate = write_report(out_root)
    print(f"Wrote {out_root}")
    if not gate.empty:
        print(gate.to_string(index=False))


if __name__ == "__main__":
    main()
