#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Gap existence and recovery experiments for the CRG/LCRF main problem.

This runner is inference-only. It reuses existing checkpoints and evaluates the
paper's query-level problem definition:

1. Whether direct target-concept evidence gaps exist in each dataset.
2. Whether Full recovers BCE in bridgeable/weak-direct gap regimes compared
   with route-absent or route-control variants.
3. Whether support corruption increases BCE in the same target regime.
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

from tools.run_main_problem_experiments import (  # noqa: E402
    BASE_VARIANTS,
    CONTROL_VARIANTS,
    DATASETS,
    DatasetAssets,
    _acc,
    _annotate_coverage,
    _bce,
    _ensure_dir,
    _load_assets,
    _make_eval_frame,
    _read_main_table,
    _rmse,
    _safe_auc,
    run_exp2,
)
from tools.run_three_dataset_mechanism_suite import (  # noqa: E402
    _checkpoint_path,
    run_support_corruption,
)


RECOVERY_VARIANTS = (*BASE_VARIANTS, *CONTROL_VARIANTS)
PROFILE_GROUPS = (
    "all",
    "direct_seen",
    "direct_unseen",
    "direct_unseen_bridgeable",
    "direct_unseen_unbridgeable",
    "weak_direct_evidence",
    "high_route_mass",
    "weak_direct_high_route",
)
TARGET_GROUPS = ("direct_unseen_bridgeable", "weak_direct_high_route")
SUPPORT_STRONG_TYPES = ("evidence_support_corruption", "self_only_fallback")


def _subset_assets(assets: DatasetAssets, max_rows: int) -> DatasetAssets:
    if max_rows <= 0 or len(assets.test_df) <= max_rows:
        return assets
    limited = assets.test_df.head(max_rows).copy().reset_index(drop=True)
    return DatasetAssets(
        dataset=assets.dataset,
        data_dir=assets.data_dir,
        train_df=assets.train_df,
        test_df=limited,
        info=assets.info,
        checkpoint_paths=assets.checkpoint_paths,
    )


def _labels(predictions: pd.DataFrame) -> np.ndarray:
    col = "label_eval" if "label_eval" in predictions.columns else "label"
    return predictions[col].to_numpy(dtype=np.float64)


def _prob_col(variant: str) -> str:
    return f"prob_{variant}"


def _mask(predictions: pd.DataFrame, subgroup: str) -> np.ndarray:
    if subgroup == "all":
        return np.ones(len(predictions), dtype=bool)
    if subgroup == "weak_direct_high_route":
        return (
            predictions["weak_direct_evidence"].to_numpy(dtype=bool)
            & predictions["high_route_mass"].to_numpy(dtype=bool)
        )
    if subgroup not in predictions.columns:
        return np.zeros(len(predictions), dtype=bool)
    return predictions[subgroup].to_numpy(dtype=bool)


def _history_stats(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {
            "history_len_mean": float("nan"),
            "history_len_median": float("nan"),
            "history_len_q25": float("nan"),
            "history_len_q75": float("nan"),
        }
    return {
        "history_len_mean": float(np.mean(values)),
        "history_len_median": float(np.median(values)),
        "history_len_q25": float(np.quantile(values, 0.25)),
        "history_len_q75": float(np.quantile(values, 0.75)),
    }


def build_gap_profile(predictions: pd.DataFrame, annotated: pd.DataFrame, dataset: str) -> pd.DataFrame:
    frame = predictions.merge(
        annotated[["event_id", "student_train_history_len"]],
        on="event_id",
        how="left",
        validate="one_to_one",
    )
    n_total = max(1, len(frame))
    rows: List[Dict[str, Any]] = []
    for subgroup in PROFILE_GROUPS:
        mask = _mask(frame, subgroup)
        history_values = frame.loc[mask, "student_train_history_len"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "dataset": dataset,
                "subgroup": subgroup,
                "n_eval": int(mask.sum()),
                "fraction": float(mask.sum() / n_total),
                **_history_stats(history_values),
            }
        )
    return pd.DataFrame(rows)


def metric_values(labels: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
    return {
        "auc": _safe_auc(labels, probs),
        "bce": _bce(labels, probs),
        "rmse": _rmse(labels, probs),
        "acc": _acc(labels, probs),
    }


def _ci(values: np.ndarray) -> Tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def bootstrap_delta(
    labels: np.ndarray,
    full_probs: np.ndarray,
    variant_probs: np.ndarray,
    iters: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    if iters <= 0 or labels.size == 0:
        return {
            "delta_auc_ci_low": float("nan"),
            "delta_auc_ci_high": float("nan"),
            "delta_bce_ci_low": float("nan"),
            "delta_bce_ci_high": float("nan"),
        }
    auc_delta = np.full(iters, np.nan, dtype=np.float64)
    bce_delta = np.full(iters, np.nan, dtype=np.float64)
    n = labels.size
    for i in range(iters):
        idx = rng.integers(0, n, size=n)
        y = labels[idx]
        f = full_probs[idx]
        v = variant_probs[idx]
        if np.unique(y).size >= 2:
            full_auc = _safe_auc(y, f)
            var_auc = _safe_auc(y, v)
            auc_delta[i] = float(full_auc - var_auc) if full_auc is not None and var_auc is not None else float("nan")
        bce_delta[i] = _bce(y, v) - _bce(y, f)
    auc_low, auc_high = _ci(auc_delta)
    bce_low, bce_high = _ci(bce_delta)
    return {
        "delta_auc_ci_low": auc_low,
        "delta_auc_ci_high": auc_high,
        "delta_bce_ci_low": bce_low,
        "delta_bce_ci_high": bce_high,
    }


def build_recovery_metrics(
    predictions: pd.DataFrame,
    dataset: str,
    bootstrap: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    labels = _labels(predictions)
    if _prob_col("full") not in predictions.columns:
        raise ValueError(f"{dataset}: missing prob_full in coverage-conditioned predictions")
    full_probs = predictions[_prob_col("full")].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed + abs(hash(dataset)) % 100_000)
    metric_rows: List[Dict[str, Any]] = []
    boot_rows: List[Dict[str, Any]] = []
    for subgroup in PROFILE_GROUPS:
        mask = _mask(predictions, subgroup)
        if not bool(mask.any()):
            continue
        y = labels[mask]
        full = full_probs[mask]
        full_metrics = metric_values(y, full)
        for variant in RECOVERY_VARIANTS:
            col = _prob_col(variant)
            if col not in predictions.columns:
                continue
            probs = predictions[col].to_numpy(dtype=np.float64)[mask]
            metrics = metric_values(y, probs)
            row = {
                "dataset": dataset,
                "subgroup": subgroup,
                "variant": variant,
                "n_eval": int(mask.sum()),
                **metrics,
                "full_auc": full_metrics["auc"],
                "full_bce": full_metrics["bce"],
                "full_rmse": full_metrics["rmse"],
                "full_acc": full_metrics["acc"],
                "delta_auc_full_minus_variant": (
                    None
                    if metrics["auc"] is None or full_metrics["auc"] is None
                    else float(full_metrics["auc"] - metrics["auc"])
                ),
                "delta_bce_variant_minus_full": float(metrics["bce"] - full_metrics["bce"]),
                "delta_rmse_variant_minus_full": float(metrics["rmse"] - full_metrics["rmse"]),
                "delta_acc_full_minus_variant": float(full_metrics["acc"] - metrics["acc"]),
            }
            if variant == "full":
                boot = {
                    "delta_auc_ci_low": float("nan"),
                    "delta_auc_ci_high": float("nan"),
                    "delta_bce_ci_low": float("nan"),
                    "delta_bce_ci_high": float("nan"),
                }
            else:
                boot = bootstrap_delta(y, full, probs, bootstrap, rng)
            row.update(boot)
            metric_rows.append(row)
            if variant != "full":
                boot_rows.append(
                    {
                        "dataset": dataset,
                        "subgroup": subgroup,
                        "comparison": f"full_vs_{variant}",
                        "n_eval": int(mask.sum()),
                        "delta_auc": row["delta_auc_full_minus_variant"],
                        "delta_bce": row["delta_bce_variant_minus_full"],
                        **boot,
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(boot_rows)


def read_csvs(paths: Iterable[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            continue
        try:
            frames.append(pd.read_csv(path))
        except pd.errors.EmptyDataError:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def support_focus(support: pd.DataFrame) -> pd.DataFrame:
    if support.empty:
        return pd.DataFrame()
    focus = support[
        support["corruption_ratio"].astype(float).eq(1.0)
        & support["corruption_type"].isin(SUPPORT_STRONG_TYPES)
        & support["subgroup"].isin(("direct_unseen_bridgeable", "high_crg_mass", "high_support_mass", "all"))
    ].copy()
    if focus.empty:
        return focus
    focus["bce_increase"] = focus["bce_increase"].astype(float)
    focus["auc_drop"] = pd.to_numeric(focus.get("auc_drop"), errors="coerce")
    return focus


def build_success_gate(
    profile: pd.DataFrame,
    recovery: pd.DataFrame,
    support: pd.DataFrame,
    min_eval: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if profile.empty or "dataset" not in profile.columns:
        return pd.DataFrame(
            columns=[
                "dataset",
                "target_groups_with_n_ge_min",
                "min_eval",
                "sample_pass",
                "no_crg_bce_recovery_best",
                "no_crg_bce_pass",
                "self_only_bce_recovery_best",
                "self_only_bce_pass",
                "support_corruption_bce_increase_best",
                "support_corruption_pass",
                "overall_pass",
            ]
        )
    support_strong = support_focus(support)
    for dataset in sorted(profile["dataset"].dropna().unique()):
        prof = profile[profile["dataset"].eq(dataset)]
        rec = recovery[recovery["dataset"].eq(dataset)]
        target_prof = prof[prof["subgroup"].isin(TARGET_GROUPS) & (prof["n_eval"].astype(int) >= min_eval)]
        target_groups = target_prof["subgroup"].astype(str).tolist()
        sample_pass = bool(target_groups)
        no_crg = rec[
            rec["subgroup"].isin(target_groups)
            & rec["variant"].eq("no_CRG")
            & (rec["n_eval"].astype(int) >= min_eval)
        ].copy()
        self_only = rec[
            rec["subgroup"].isin(target_groups)
            & rec["variant"].eq("self_only")
            & (rec["n_eval"].astype(int) >= min_eval)
        ].copy()
        no_crg_best = float(no_crg["delta_bce_variant_minus_full"].astype(float).max()) if not no_crg.empty else float("nan")
        self_best = float(self_only["delta_bce_variant_minus_full"].astype(float).max()) if not self_only.empty else float("nan")
        support_ds = support_strong[
            support_strong["dataset"].eq(dataset)
            & support_strong["subgroup"].isin(("direct_unseen_bridgeable", "high_crg_mass", "high_support_mass"))
            & (support_strong["n_eval"].astype(int) >= min_eval)
        ].copy()
        support_best = float(support_ds["bce_increase"].max()) if not support_ds.empty else float("nan")
        rows.append(
            {
                "dataset": dataset,
                "target_groups_with_n_ge_min": ",".join(target_groups),
                "min_eval": int(min_eval),
                "sample_pass": sample_pass,
                "no_crg_bce_recovery_best": no_crg_best,
                "no_crg_bce_pass": bool(np.isfinite(no_crg_best) and no_crg_best > 0.0),
                "self_only_bce_recovery_best": self_best,
                "self_only_bce_pass": bool(np.isfinite(self_best) and self_best > 0.0),
                "support_corruption_bce_increase_best": support_best,
                "support_corruption_pass": bool(np.isfinite(support_best) and support_best > 0.0),
            }
        )
    gate = pd.DataFrame(rows)
    if not gate.empty:
        gate["overall_pass"] = (
            gate["sample_pass"]
            & gate["no_crg_bce_pass"]
            & gate["self_only_bce_pass"]
            & gate["support_corruption_pass"]
        )
    return gate


def _format_float(value: Any) -> str:
    try:
        val = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(val):
        return ""
    return f"{val:.6f}"


def md_table(frame: pd.DataFrame, max_rows: int = 80) -> str:
    if frame.empty:
        return "No rows."
    show = frame.head(max_rows).copy()
    cols = list(show.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in show.itertuples(index=False):
        vals = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                vals.append(_format_float(value))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def plot_outputs(out_root: Path, recovery: pd.DataFrame, support: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (out_root / "plot_missing_report.md").write_text(f"matplotlib unavailable: {exc}\n", encoding="utf-8")
        return
    fig_dir = _ensure_dir(out_root / "figures")
    if recovery.empty or "variant" not in recovery.columns or "subgroup" not in recovery.columns:
        focus = support_focus(support)
        if not focus.empty:
            focus.to_csv(fig_dir / "support_corruption_focus_metrics.csv", index=False)
        return

    heat = recovery[
        recovery["variant"].eq("no_CRG")
        & recovery["subgroup"].isin(("direct_unseen_bridgeable", "weak_direct_high_route"))
    ].copy()
    if not heat.empty:
        datasets = [d for d in DATASETS if d in set(heat["dataset"])]
        subgroups = ["direct_unseen_bridgeable", "weak_direct_high_route"]
        mat = np.full((len(datasets), len(subgroups)), np.nan, dtype=np.float64)
        labels = [["" for _ in subgroups] for _ in datasets]
        for i, dataset in enumerate(datasets):
            for j, subgroup in enumerate(subgroups):
                hit = heat[heat["dataset"].eq(dataset) & heat["subgroup"].eq(subgroup)]
                if hit.empty:
                    continue
                row = hit.iloc[0]
                mat[i, j] = float(row["delta_bce_variant_minus_full"])
                labels[i][j] = f"{mat[i, j]:.4f}\nn={int(row['n_eval'])}"
        fig, ax = plt.subplots(figsize=(6.2, 3.0))
        vmax = max(0.001, float(np.nanmax(np.abs(mat))) if np.isfinite(mat).any() else 0.001)
        im = ax.imshow(mat, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(np.arange(len(subgroups)))
        ax.set_xticklabels(["direct-unseen\nbridgeable", "weak-direct\nhigh-route"])
        ax.set_yticks(np.arange(len(datasets)))
        ax.set_yticklabels(datasets)
        for i in range(len(datasets)):
            for j in range(len(subgroups)):
                ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=8)
        ax.set_title("BCE recovery: w/o CRG - Full")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig7_gap_recovery_heatmap.png", dpi=240)
        fig.savefig(fig_dir / "fig7_gap_recovery_heatmap.pdf")
        plt.close(fig)

    sol = recovery[
        recovery["subgroup"].eq("direct_unseen_bridgeable")
        & recovery["variant"].isin(("no_CRG", "no_LCRF", "self_only", "degree_random_support"))
    ].copy()
    if not sol.empty:
        datasets = [d for d in DATASETS if d in set(sol["dataset"])]
        variants = ["no_CRG", "no_LCRF", "self_only", "degree_random_support"]
        x = np.arange(len(datasets))
        width = 0.18
        fig, ax = plt.subplots(figsize=(7.4, 3.3))
        for idx, variant in enumerate(variants):
            vals = []
            for dataset in datasets:
                hit = sol[sol["dataset"].eq(dataset) & sol["variant"].eq(variant)]
                vals.append(float(hit["delta_bce_variant_minus_full"].iloc[0]) if not hit.empty else np.nan)
            ax.bar(x + (idx - 1.5) * width, vals, width=width, label=variant)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_ylabel("BCE recovery over variant")
        ax.set_title("Bridgeable evidence-gap solution test")
        ax.legend(fontsize=7, frameon=False, ncol=2)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig_final_bridgeable_solution.png", dpi=240)
        fig.savefig(fig_dir / "fig_final_bridgeable_solution.pdf")
        plt.close(fig)

    focus = support_focus(support)
    if not focus.empty:
        focus.to_csv(fig_dir / "support_corruption_focus_metrics.csv", index=False)


def write_report(
    out_root: Path,
    profile: pd.DataFrame,
    recovery: pd.DataFrame,
    bootstrap: pd.DataFrame,
    support: pd.DataFrame,
    gate: pd.DataFrame,
    missing: Sequence[str],
) -> None:
    overall = bool(not gate.empty and gate["overall_pass"].all())
    lines = [
        "# Gap Existence + Gap Recovery Experiment Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Overall status: {'PASS' if overall else 'FAIL'}",
        "",
        "## Success Gates",
        "",
        md_table(gate),
        "",
        "## Gap Existence/Profile",
        "",
        md_table(profile[profile["subgroup"].isin(PROFILE_GROUPS)] if "subgroup" in profile.columns else pd.DataFrame()),
        "",
        "## Gap Recovery Focus",
        "",
    ]
    if recovery.empty or "subgroup" not in recovery.columns:
        focus = pd.DataFrame()
    else:
        focus = recovery[
            recovery["subgroup"].isin(("direct_unseen_bridgeable", "weak_direct_high_route"))
            & recovery["variant"].isin(("no_CRG", "no_LCRF", "self_only", "degree_random_support"))
        ][
            [
                "dataset",
                "subgroup",
                "variant",
                "n_eval",
                "auc",
                "bce",
                "delta_auc_full_minus_variant",
                "delta_bce_variant_minus_full",
                "delta_bce_ci_low",
                "delta_bce_ci_high",
            ]
        ].copy()
    lines.extend([md_table(focus, max_rows=120), ""])
    sfocus = support_focus(support)
    lines.extend(["## Support-Corruption Mechanism Supplement", "", md_table(sfocus, max_rows=120), ""])
    if missing:
        lines.extend(["## Missing/Warnings", ""])
        for item in sorted(set(missing)):
            lines.append(f"- {item}")
        lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    if overall:
        lines.append(
            "The three-dataset main-problem gate passed under the BCE-first criterion. "
            "Use the gap-recovery heatmap as the Fig.7 replacement candidate and the "
            "bridgeable evidence-gap solution plot as the final main experiment candidate."
        )
    else:
        lines.append(
            "The three-dataset main-problem gate did not pass. Do not replace paper figures "
            "until the failed gates are inspected."
        )
    text = "\n".join(lines) + "\n"
    (out_root / "experiment_report.md").write_text(text, encoding="utf-8")
    if missing:
        (out_root / "missing_report.md").write_text("\n".join(f"- {m}" for m in sorted(set(missing))) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="assist_09,junyi,assist_17")
    parser.add_argument("--main-table", default="results/crg_lcrf_core3_final_20260520/main_table/table_main_ablation_core3.csv")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--min-eval", type=int, default=500)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--skip-support", action="store_true")
    parser.add_argument("--support-trials", type=int, default=1)
    parser.add_argument("--support-ratios", type=float, nargs="+", default=[1.0])
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = _ensure_dir(Path(args.output_root or f"results/gap_recovery_main_problem_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    if args.plot_only:
        profile = pd.read_csv(out_root / "gap_profile.csv")
        recovery = pd.read_csv(out_root / "gap_recovery_metrics.csv")
        bootstrap = pd.read_csv(out_root / "gap_recovery_bootstrap.csv")
        support = read_csvs(out_root.glob("*/support_corruption_metrics.csv"))
        gate = pd.read_csv(out_root / "success_gate_summary.csv")
        plot_outputs(out_root, recovery, support)
        write_report(out_root, profile, recovery, bootstrap, support, gate, [])
        return

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    main_table = _read_main_table(Path(args.main_table))
    missing: List[str] = []
    all_profile: List[pd.DataFrame] = []
    all_recovery: List[pd.DataFrame] = []
    all_bootstrap: List[pd.DataFrame] = []

    for dataset in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        print(f"[{dataset}] running gap recovery", flush=True)
        ds_out = _ensure_dir(out_root / dataset)
        assets = _load_assets(dataset, main_table, device, missing)
        if assets is None:
            continue
        assets = _subset_assets(assets, args.max_rows)
        predictions, _ = run_exp2(
            assets=assets,
            out_dir=ds_out,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            missing=missing,
        )
        annotated = _annotate_coverage(_make_eval_frame(assets), assets)
        profile = build_gap_profile(predictions, annotated, dataset)
        recovery, bootstrap = build_recovery_metrics(predictions, dataset, args.bootstrap, args.seed)
        profile.to_csv(ds_out / "gap_profile.csv", index=False)
        recovery.to_csv(ds_out / "gap_recovery_metrics.csv", index=False)
        bootstrap.to_csv(ds_out / "gap_recovery_bootstrap.csv", index=False)
        all_profile.append(profile)
        all_recovery.append(recovery)
        all_bootstrap.append(bootstrap)

        if not args.skip_support:
            print(f"[{dataset}] running support-corruption supplement", flush=True)
            ckpt_dir = _checkpoint_path(main_table, dataset, "full")
            run_support_corruption(
                dataset=dataset,
                checkpoint_dir=ckpt_dir,
                output_dir=ds_out,
                data_dir=None,
                split="test",
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                ratios=args.support_ratios,
                trials=args.support_trials,
                seed=args.seed,
                max_rows=args.max_rows,
            )

    profile_all = pd.concat(all_profile, ignore_index=True) if all_profile else pd.DataFrame()
    recovery_all = pd.concat(all_recovery, ignore_index=True) if all_recovery else pd.DataFrame()
    bootstrap_all = pd.concat(all_bootstrap, ignore_index=True) if all_bootstrap else pd.DataFrame()
    support_all = read_csvs(out_root.glob("*/support_corruption_metrics.csv"))
    profile_all.to_csv(out_root / "gap_profile.csv", index=False)
    recovery_all.to_csv(out_root / "gap_recovery_metrics.csv", index=False)
    bootstrap_all.to_csv(out_root / "gap_recovery_bootstrap.csv", index=False)
    gate = build_success_gate(profile_all, recovery_all, support_all, args.min_eval)
    gate.to_csv(out_root / "success_gate_summary.csv", index=False)
    plot_outputs(out_root, recovery_all, support_all)
    write_report(out_root, profile_all, recovery_all, bootstrap_all, support_all, gate, missing)
    print(f"Wrote {out_root}", flush=True)
    if not gate.empty:
        print(gate.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
