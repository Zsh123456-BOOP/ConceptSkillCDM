#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Three-dataset CRG/LCRF mechanism suite.

Inference-only runner for:
1. CRG support identity corruption.
2. Coverage-conditioned support knockout.
3. LCRF posterior-only learner conditioning.
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CognitiveDiagnosisDataset  # noqa: E402
from tools.analyze_a_sufficiency_controls import (  # noqa: E402
    _annotate_a_relevance,
    _eval_frame_from_checkpoint,
    _load_checkpoint_model,
    _metrics,
    _predict_model,
)
from tools.analyze_crg_support_corruption_controls import (  # noqa: E402
    CORRUPTION_TYPES,
    _attach_gap_and_claims,
    _build_control_masks,
    _clone_masks,
    _group_masks,
    _restore_masks,
    _set_masks,
    _story_audit_columns,
    _support_size_mean,
    _train_only_candidate_check,
)
from tools.export_ae_case_studies import _apply_personal_counterfactual  # noqa: E402
from tools.run_main_problem_experiments import _ensure_dir, _read_main_table  # noqa: E402


DATASETS = ("assist_09", "junyi", "assist_17")
POSTERIOR_MODES = ("actual", "mean_state", "shuffle_state")
TARGET_GROUPS = ("direct_unseen_bridgeable", "high_crg_mass", "high_support_mass", "graph_hits_history")


def _checkpoint_path(main_table: pd.DataFrame, dataset: str, variant: str = "full") -> Path:
    rows = main_table[
        main_table["dataset"].astype(str).eq(dataset)
        & main_table["variant"].astype(str).eq(variant)
    ]
    if rows.empty:
        raise ValueError(f"Missing {dataset}/{variant} in main table")
    raw = str(rows.iloc[0]["checkpoint_path"])
    path = Path(raw)
    return path.parent if path.name == "best_model.pth" else path


def _safe_auc(labels: np.ndarray, probs: np.ndarray) -> Optional[float]:
    return _metrics(labels, probs).get("auc")


def _bce(labels: np.ndarray, probs: np.ndarray) -> float:
    return float(_metrics(labels, probs)["bce"])


def _rmse(labels: np.ndarray, probs: np.ndarray) -> float:
    return float(np.sqrt(np.mean((labels.astype(np.float64) - probs.astype(np.float64)) ** 2))) if len(labels) else float("nan")


def _acc(labels: np.ndarray, probs: np.ndarray) -> float:
    return float(np.mean((probs >= 0.5).astype(np.float32) == labels.astype(np.float32))) if len(labels) else float("nan")


def _metric_row(
    dataset: str,
    subgroup: str,
    variant: str,
    labels: np.ndarray,
    probs: np.ndarray,
    clean: Mapping[str, Any],
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    auc = _safe_auc(labels, probs)
    bce = _bce(labels, probs)
    row = {
        "dataset": dataset,
        "subgroup": subgroup,
        "variant": variant,
        "n_eval": int(len(labels)),
        "auc": auc,
        "bce": bce,
        "rmse": _rmse(labels, probs),
        "acc": _acc(labels, probs),
        "clean_auc": clean.get("auc"),
        "clean_bce": clean.get("bce"),
        "auc_drop": None if auc is None or clean.get("auc") is None else float(clean["auc"] - auc),
        "bce_increase": float(bce - clean["bce"]),
    }
    if extra:
        row.update(extra)
    return row


def _prepare_annotated(
    model: torch.nn.Module,
    checkpoint: Mapping[str, Any],
    dataset: str,
    data_dir: Optional[str],
    split: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
    info = checkpoint["info_dict"]
    eval_frame = _eval_frame_from_checkpoint(checkpoint, dataset, data_dir, split)
    clean_pred = _predict_model(model, eval_frame, info, batch_size, num_workers, device)
    annotated = _annotate_a_relevance(
        clean_pred.rename(columns={"prob": "prob_clean"}),
        info,
        dataset,
        data_dir,
    )
    annotated = _story_audit_columns(annotated, info, dataset, data_dir)
    labels = annotated["label_eval"].to_numpy(dtype=np.float32)
    clean_probs = annotated["prob_clean"].to_numpy(dtype=np.float32)
    masks = _group_masks(annotated)
    clean_metrics = {
        group: _metrics(labels[mask], clean_probs[mask])
        for group, mask in masks.items()
        if bool(mask.any())
    }
    return annotated, masks, clean_metrics


def run_support_corruption(
    *,
    dataset: str,
    checkpoint_dir: Path,
    output_dir: Path,
    data_dir: Optional[str],
    split: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    ratios: Sequence[float],
    trials: int,
    seed: int,
    max_rows: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    checkpoint, model = _load_checkpoint_model(checkpoint_dir, device)
    info = checkpoint["info_dict"]
    eval_frame = _eval_frame_from_checkpoint(checkpoint, dataset, data_dir, split)
    if max_rows > 0 and len(eval_frame) > max_rows:
        eval_frame = eval_frame.sample(n=max_rows, random_state=seed).sort_index().reset_index(drop=True)
    clean_pred = _predict_model(model, eval_frame, info, batch_size, num_workers, device)
    annotated = _annotate_a_relevance(clean_pred.rename(columns={"prob": "prob_clean"}), info, dataset, data_dir)
    annotated = _story_audit_columns(annotated, info, dataset, data_dir)
    annotated.to_csv(output_dir / "support_corruption_sample_audit.csv", index=False)
    labels = annotated["label_eval"].to_numpy(dtype=np.float32)
    clean_probs = annotated["prob_clean"].to_numpy(dtype=np.float32)
    groups = _group_masks(annotated)
    clean_metrics = {
        group: _metrics(labels[mask], clean_probs[mask])
        for group, mask in groups.items()
        if bool(mask.any())
    }
    original = _clone_masks(model)
    check_passed = _train_only_candidate_check(original, info)
    rows: List[Dict[str, Any]] = []
    seeds = [int(seed) + i for i in range(max(1, int(trials)))]
    for corruption_type in CORRUPTION_TYPES:
        print(f"[{dataset}] support corruption: {corruption_type}", flush=True)
        for ratio in [float(x) for x in ratios]:
            effective_seeds = [int(seed)] if ratio <= 0.0 else seeds
            for trial_seed in effective_seeds:
                item_mask, seq_mask = _build_control_masks(original, info, corruption_type, ratio, trial_seed)
                _set_masks(model, item_mask, seq_mask)
                pred = _predict_model(model, eval_frame, info, batch_size, num_workers, device)
                probs = pred["prob"].to_numpy(dtype=np.float32)
                mean_support = _support_size_mean(item_mask, seq_mask, allow_self=True)
                for group, mask in groups.items():
                    if not bool(mask.any()):
                        continue
                    rows.append(
                        _metric_row(
                            dataset,
                            group,
                            corruption_type,
                            labels[mask],
                            probs[mask],
                            clean_metrics[group],
                            {
                                "corruption_type": corruption_type,
                                "corruption_ratio": ratio,
                                "seed": int(trial_seed),
                                "mean_support_size": mean_support,
                                "train_only_candidate_check_passed": bool(check_passed),
                            },
                        )
                    )
                _restore_masks(model, original)
    result = pd.DataFrame(rows)
    result, gap = _attach_gap_and_claims(
        result.rename(columns={"subgroup": "group", "auc_drop": "auc_drop", "bce_increase": "bce_increase"})
    )
    result = result.rename(columns={"group": "subgroup"})
    result.to_csv(output_dir / "support_corruption_metrics.csv", index=False)
    gap.to_csv(output_dir / "support_corruption_gap_claims.csv", index=False)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, annotated


def _tensor_or_none(details: Mapping[str, Any], key: str) -> Optional[torch.Tensor]:
    value = details.get(key)
    return value if torch.is_tensor(value) else None


def _per_sample_support_metrics(details: Mapping[str, Any], batch_size: int) -> Dict[str, np.ndarray]:
    posterior = _tensor_or_none(details, "posterior_prob")
    global_prob = _tensor_or_none(details, "global_support_prob")
    valid = _tensor_or_none(details, "support_valid_mask")
    query_mask = _tensor_or_none(details, "query_row_active_mask")
    out = {
        "posterior_delta_abs": np.zeros(batch_size, dtype=np.float32),
        "posterior_js": np.zeros(batch_size, dtype=np.float32),
        "top1_changed": np.zeros(batch_size, dtype=np.float32),
        "valid_support_rows": np.zeros(batch_size, dtype=np.float32),
    }
    if posterior is None or global_prob is None:
        return out
    p = posterior.detach()
    g = global_prob.detach().to(device=p.device, dtype=p.dtype)
    if p.dim() < 2 or p.size(0) != batch_size:
        return out
    mask = torch.ones_like(p, dtype=torch.bool)
    if valid is not None and valid.shape == p.shape:
        mask = mask & valid.detach().to(device=p.device).bool()
    if query_mask is not None:
        qm = query_mask.detach().to(device=p.device).bool()
        while qm.dim() < p.dim():
            qm = qm.unsqueeze(-1)
        try:
            mask = mask & qm.expand_as(mask)
        except RuntimeError:
            pass
    masked_float = mask.to(dtype=p.dtype)
    denom = masked_float.sum(dim=tuple(range(1, p.dim()))).clamp(min=1.0)
    delta = ((p - g).abs() * masked_float).sum(dim=tuple(range(1, p.dim()))) / denom
    m = 0.5 * (p + g)
    kl_pm = p.clamp(min=1e-8) * (p.clamp(min=1e-8).log() - m.clamp(min=1e-8).log())
    kl_gm = g.clamp(min=1e-8) * (g.clamp(min=1e-8).log() - m.clamp(min=1e-8).log())
    js = (0.5 * (kl_pm + kl_gm) * masked_float).sum(dim=tuple(range(1, p.dim()))) / denom
    masked_p = torch.where(mask, p, torch.full_like(p, -1.0))
    masked_g = torch.where(mask, g, torch.full_like(g, -1.0))
    p_top = masked_p.argmax(dim=-1)
    g_top = masked_g.argmax(dim=-1)
    row_valid = mask.any(dim=-1)
    changed = ((p_top != g_top) & row_valid).to(dtype=p.dtype)
    changed_denom = row_valid.to(dtype=p.dtype).sum(dim=tuple(range(1, row_valid.dim()))).clamp(min=1.0)
    changed_mean = changed.sum(dim=tuple(range(1, changed.dim()))) / changed_denom
    row_count = row_valid.to(dtype=p.dtype).sum(dim=tuple(range(1, row_valid.dim())))
    out["posterior_delta_abs"] = delta.detach().cpu().numpy().astype(np.float32)
    out["posterior_js"] = js.detach().cpu().numpy().astype(np.float32)
    out["top1_changed"] = changed_mean.detach().cpu().numpy().astype(np.float32)
    out["valid_support_rows"] = row_count.detach().cpu().numpy().astype(np.float32)
    return out


def run_posterior_conditioning(
    *,
    dataset: str,
    checkpoint_dir: Path,
    output_dir: Path,
    data_dir: Optional[str],
    split: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
    max_rows: int,
) -> pd.DataFrame:
    checkpoint, _ = _load_checkpoint_model(checkpoint_dir, device)
    info = checkpoint["info_dict"]
    eval_frame = _eval_frame_from_checkpoint(checkpoint, dataset, data_dir, split)
    if max_rows > 0 and len(eval_frame) > max_rows:
        eval_frame = eval_frame.sample(n=max_rows, random_state=seed).sort_index().reset_index(drop=True)
    rows: List[pd.DataFrame] = []
    for mode in POSTERIOR_MODES:
        print(f"[{dataset}] posterior mode: {mode}", flush=True)
        _, model = _load_checkpoint_model(checkpoint_dir, device)
        if mode == "mean_state":
            _apply_personal_counterfactual(model, "mean", seed=seed)
        elif mode == "shuffle_state":
            _apply_personal_counterfactual(model, "shuffle", seed=seed)
        ds = CognitiveDiagnosisDataset(eval_frame, info["stu_id_map"], info["exer_id_map"], info["cpt_id_map"])
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        model.eval()
        offset = 0
        parts: List[pd.DataFrame] = []
        with torch.no_grad():
            for student_ids, exercise_ids, labels in loader:
                take = int(labels.numel())
                student_ids = student_ids.to(device)
                exercise_ids = exercise_ids.to(device)
                logits, details = model(student_ids, exercise_ids, return_details=True, return_logits=True)
                probs = torch.sigmoid(logits.reshape(-1)).detach().cpu().numpy().astype(np.float32)
                metrics = _per_sample_support_metrics(details, take)
                frame = eval_frame.iloc[offset : offset + take][["stu_id", "exer_id", "cpt_seq", "label"]].copy()
                frame["dataset"] = dataset
                frame["mode"] = mode
                frame["mapped_student_id"] = student_ids.detach().cpu().numpy()
                frame["mapped_exercise_id"] = exercise_ids.detach().cpu().numpy()
                frame["prob"] = probs
                frame["label_eval"] = labels.detach().cpu().numpy().astype(np.float32)
                for key, value in metrics.items():
                    frame[key] = value
                parts.append(frame)
                offset += take
        mode_frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        rows.append(mode_frame)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    samples = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    samples.to_csv(output_dir / "posterior_conditioning_samples.csv", index=False)
    summary_rows: List[Dict[str, Any]] = []
    for mode, frame in samples.groupby("mode"):
        grouped = frame.groupby("exer_id")["posterior_js"].var().dropna()
        summary_rows.append(
            {
                "dataset": dataset,
                "mode": mode,
                "n_eval": int(len(frame)),
                "posterior_delta_abs_mean": float(frame["posterior_delta_abs"].mean()),
                "posterior_js_mean": float(frame["posterior_js"].mean()),
                "top1_change_rate": float(frame["top1_changed"].mean()),
                "same_query_posterior_js_var": float(grouped.mean()) if len(grouped) else 0.0,
                "valid_support_rows_mean": float(frame["valid_support_rows"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "posterior_conditioning_metrics.csv", index=False)
    return summary


def _md_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in frame.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append("" if not np.isfinite(val) else f"{val:.6g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _read_csvs(paths: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        try:
            frames.append(pd.read_csv(path))
        except (FileNotFoundError, pd.errors.EmptyDataError):
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_success_report(out_root: Path) -> pd.DataFrame:
    support = _read_csvs(out_root.glob("*/support_corruption_metrics.csv"))
    posterior = _read_csvs(out_root.glob("*/posterior_conditioning_metrics.csv"))
    rows: List[Dict[str, Any]] = []
    for dataset in DATASETS:
        ds_support = support[support["dataset"].eq(dataset)] if not support.empty else pd.DataFrame()
        ds_posterior = posterior[posterior["dataset"].eq(dataset)] if not posterior.empty else pd.DataFrame()
        strong = ds_support[
            ds_support["corruption_type"].isin(["self_only_fallback", "evidence_support_corruption"])
            & ds_support["corruption_ratio"].astype(float).eq(1.0)
            & ds_support["subgroup"].eq("all")
            & (ds_support["n_eval"].astype(float) >= 500)
            & (ds_support["bce_increase"].astype(float) > 0.0)
        ]
        target_rows = ds_support[
            ds_support["corruption_type"].isin(["self_only_fallback", "evidence_support_corruption"])
            & ds_support["corruption_ratio"].astype(float).eq(1.0)
            & ds_support["subgroup"].isin(TARGET_GROUPS)
            & (ds_support["n_eval"].astype(float) >= 500)
        ]
        overall_rows = ds_support[
            ds_support["corruption_type"].isin(["self_only_fallback", "evidence_support_corruption"])
            & ds_support["corruption_ratio"].astype(float).eq(1.0)
            & ds_support["subgroup"].eq("all")
        ]
        overall_best = float(overall_rows["bce_increase"].astype(float).max()) if not overall_rows.empty else float("nan")
        target_best = float(target_rows["bce_increase"].astype(float).max()) if not target_rows.empty else float("nan")
        support_pass = not strong.empty
        coverage_pass = bool(np.isfinite(target_best) and np.isfinite(overall_best) and target_best >= overall_best)
        actual = ds_posterior[ds_posterior["mode"].eq("actual")]
        controls = ds_posterior[ds_posterior["mode"].isin(["mean_state", "shuffle_state"])]
        if not actual.empty and not controls.empty:
            actual_js = float(actual.iloc[0]["posterior_js_mean"])
            actual_var = float(actual.iloc[0]["same_query_posterior_js_var"])
            control_var = float(controls["same_query_posterior_js_var"].astype(float).max())
            posterior_pass = actual_js > 0.0 and actual_var >= control_var
        else:
            actual_js = actual_var = control_var = float("nan")
            posterior_pass = False
        rows.append(
            {
                "dataset": dataset,
                "support_pass": support_pass,
                "coverage_knockout_pass": coverage_pass,
                "posterior_pass": posterior_pass,
                "overall_pass": bool(support_pass and coverage_pass and posterior_pass),
                "overall_bce_increase_best": overall_best,
                "target_bce_increase_best": target_best,
                "actual_posterior_js": actual_js,
                "actual_same_query_var": actual_var,
                "control_same_query_var_max": control_var,
            }
        )
    gate = pd.DataFrame(rows)
    gate.to_csv(out_root / "success_gate_summary.csv", index=False)
    lines = [
        "# Three Dataset Mechanism Suite Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Overall status: {'PASS' if bool(gate['overall_pass'].all()) else 'FAIL'}",
        "",
        "## Success Gates",
        "",
        _md_table(gate),
        "",
    ]
    if not support.empty:
        focus = support[
            support["corruption_ratio"].astype(float).eq(1.0)
            & support["corruption_type"].isin(["evidence_support_corruption", "self_only_fallback"])
            & support["subgroup"].isin(("all", *TARGET_GROUPS))
        ][["dataset", "corruption_type", "subgroup", "n_eval", "auc_drop", "bce_increase"]]
        lines.extend(["## Support Focus Metrics", "", _md_table(focus.head(80)), ""])
    if not posterior.empty:
        lines.extend(["## Posterior Metrics", "", _md_table(posterior), ""])
    (out_root / "experiment_report.md").write_text("\n".join(lines), encoding="utf-8")
    return gate


def plot_outputs(out_root: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (out_root / "plot_missing_report.md").write_text(f"matplotlib unavailable: {exc}\n", encoding="utf-8")
        return
    fig_dir = _ensure_dir(out_root / "figures")
    support = _read_csvs(out_root.glob("*/support_corruption_metrics.csv"))
    posterior = _read_csvs(out_root.glob("*/posterior_conditioning_metrics.csv"))
    if not support.empty:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.3), squeeze=False)
        for ax, dataset in zip(axes[0], DATASETS):
            sub = support[support["dataset"].eq(dataset) & support["subgroup"].eq("all")]
            for typ in ("evidence_support_corruption", "degree_matched_random_support", "sequence_shuffled_support", "self_only_fallback"):
                one = sub[sub["corruption_type"].eq(typ)].groupby("corruption_ratio", as_index=False)["bce_increase"].mean()
                if not one.empty:
                    ax.plot(one["corruption_ratio"], one["bce_increase"], marker="o", label=typ)
            ax.set_title(dataset)
            ax.set_xlabel("corruption ratio")
            ax.set_ylabel("BCE increase")
            ax.grid(alpha=0.25)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=7)
        fig.tight_layout(rect=(0, 0.18, 1, 1))
        fig.savefig(fig_dir / "fig_support_knockout_curve.png", dpi=220)
        fig.savefig(fig_dir / "fig_support_knockout_curve.pdf")
        plt.close(fig)
    if not posterior.empty:
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        labels = []
        values = []
        for dataset in DATASETS:
            for mode in POSTERIOR_MODES:
                hit = posterior[posterior["dataset"].eq(dataset) & posterior["mode"].eq(mode)]
                if hit.empty:
                    continue
                labels.append(f"{dataset}\n{mode}")
                values.append(float(hit.iloc[0]["posterior_js_mean"]))
        ax.bar(np.arange(len(values)), values, color="#4C78A8")
        ax.set_xticks(np.arange(len(values)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Posterior JS from global support")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig_posterior_movement_bar.png", dpi=220)
        fig.savefig(fig_dir / "fig_posterior_movement_bar.pdf")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="assist_09,junyi,assist_17")
    parser.add_argument("--main-table", default="results/crg_lcrf_core3_final_20260520/main_table/table_main_ablation_core3.csv")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--corruption-ratios", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--skip-support", action="store_true")
    parser.add_argument("--skip-posterior", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root or f"results/three_dataset_mechanism_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    _ensure_dir(out_root)
    if args.plot_only:
        plot_outputs(out_root)
        write_success_report(out_root)
        return
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    main_table = _read_main_table(Path(args.main_table))
    for dataset in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        ds_out = _ensure_dir(out_root / dataset)
        ckpt_dir = _checkpoint_path(main_table, dataset, "full")
        if not args.skip_support:
            run_support_corruption(
                dataset=dataset,
                checkpoint_dir=ckpt_dir,
                output_dir=ds_out,
                data_dir=args.data_dir,
                split=args.split,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                ratios=args.corruption_ratios,
                trials=args.trials,
                seed=args.seed,
                max_rows=args.max_rows,
            )
        if not args.skip_posterior:
            run_posterior_conditioning(
                dataset=dataset,
                checkpoint_dir=ckpt_dir,
                output_dir=ds_out,
                data_dir=args.data_dir,
                split=args.split,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                seed=args.seed,
                max_rows=args.max_rows,
            )
    plot_outputs(out_root)
    gate = write_success_report(out_root)
    print(f"Wrote {out_root}")
    print(gate.to_string(index=False))


if __name__ == "__main__":
    main()
