#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""E4 decoupled-support controls for CRG/LCRF.

The checkpoint must be trained with ``--decouple_support true``.  This script
keeps the graph backbone A frozen and mutates only
``decoupled_*_support_mask`` used by LCRF's S_support path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analyze_a_sufficiency_controls import (  # noqa: E402
    _annotate_a_relevance,
    _eval_frame_from_checkpoint,
    _load_checkpoint_model,
    _metrics,
    _predict_model,
)
from tools.analyze_crg_support_corruption_controls import _story_audit_columns  # noqa: E402


DATASETS = ("assist_09", "junyi", "assist_17")
GROUP_ORDER = ("all", "direct_seen", "direct_unseen", "direct_unseen_bridgeable")
VARIANTS = ("clean", "s_support_random_same_total", "s_support_degree_matched_random")


def _parse_csv(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _relation_module(model: torch.nn.Module) -> torch.nn.Module:
    return model.structure_module.relation_learning


def _clone_a_masks(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    rel = _relation_module(model)
    return {
        "item_support_mask": rel.item_support_mask.detach().clone(),
        "sequence_support_mask": rel.sequence_support_mask.detach().clone(),
    }


def _clone_s_masks(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    rel = _relation_module(model)
    return {
        "decoupled_item_support_mask": rel.decoupled_item_support_mask.detach().clone(),
        "decoupled_sequence_support_mask": rel.decoupled_sequence_support_mask.detach().clone(),
    }


def _restore_s_masks(model: torch.nn.Module, masks: Mapping[str, torch.Tensor]) -> None:
    rel = _relation_module(model)
    rel.decoupled_item_support_mask.copy_(masks["decoupled_item_support_mask"])
    rel.decoupled_sequence_support_mask.copy_(masks["decoupled_sequence_support_mask"])


def _set_s_masks(model: torch.nn.Module, item_mask: torch.Tensor, sequence_mask: Optional[torch.Tensor] = None) -> None:
    rel = _relation_module(model)
    seq = torch.zeros_like(item_mask) if sequence_mask is None else sequence_mask
    rel.decoupled_item_support_mask.copy_(item_mask.to(device=rel.decoupled_item_support_mask.device).bool())
    rel.decoupled_sequence_support_mask.copy_(seq.to(device=rel.decoupled_sequence_support_mask.device).bool())


def _a_masks_unchanged(model: torch.nn.Module, original: Mapping[str, torch.Tensor]) -> bool:
    rel = _relation_module(model)
    return bool(
        torch.equal(rel.item_support_mask.detach().cpu(), original["item_support_mask"].detach().cpu())
        and torch.equal(
            rel.sequence_support_mask.detach().cpu(),
            original["sequence_support_mask"].detach().cpu(),
        )
    )


def _offdiag(c: int) -> torch.Tensor:
    return ~torch.eye(c, dtype=torch.bool)


def _random_same_total(mask: torch.Tensor, seed: int) -> torch.Tensor:
    src = mask.detach().cpu().bool()
    c = int(src.size(0))
    off = _offdiag(c)
    total = int((src & off).sum().item())
    out = torch.zeros_like(src)
    if total <= 0 or c <= 1:
        return out
    candidates = torch.nonzero(off.reshape(-1), as_tuple=False).view(-1)
    gen = torch.Generator().manual_seed(int(seed))
    take = min(total, int(candidates.numel()))
    picked = candidates[torch.randperm(int(candidates.numel()), generator=gen)[:take]]
    out.view(-1)[picked] = True
    return out


def _random_same_row_degree(mask: torch.Tensor, seed: int) -> torch.Tensor:
    src = mask.detach().cpu().bool()
    c = int(src.size(0))
    off = _offdiag(c)
    out = torch.zeros_like(src)
    gen = torch.Generator().manual_seed(int(seed))
    for row in range(c):
        degree = int((src[row] & off[row]).sum().item())
        if degree <= 0:
            continue
        candidates = torch.nonzero(off[row], as_tuple=False).view(-1)
        picked = candidates[torch.randperm(int(candidates.numel()), generator=gen)[: min(degree, int(candidates.numel()))]]
        out[row, picked] = True
    return out


def _safe_metric(labels: np.ndarray, probs: np.ndarray, metric: str) -> Optional[float]:
    if metric == "auc":
        return _metrics(labels, probs)["auc"]
    if metric == "bce":
        return _metrics(labels, probs)["bce"]
    raise ValueError(metric)


def _bootstrap_delta_ci(
    labels: np.ndarray,
    clean_probs: np.ndarray,
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
        clean = _safe_metric(labels[idx], clean_probs[idx], metric)
        var = _safe_metric(labels[idx], variant_probs[idx], metric)
        if clean is None or var is None:
            continue
        vals.append(float(clean - var) if metric == "auc" else float(var - clean))
    if not vals:
        return None, None
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def _group_masks(frame: pd.DataFrame) -> Dict[str, np.ndarray]:
    masks: Dict[str, np.ndarray] = {"all": np.ones(len(frame), dtype=bool)}
    for group in GROUP_ORDER:
        if group == "all":
            continue
        if group in frame.columns:
            masks[group] = frame[group].to_numpy(dtype=bool)
    return masks


def _metric_rows(
    dataset: str,
    annotated: pd.DataFrame,
    probs_by_variant: Mapping[str, np.ndarray],
    *,
    bootstrap: int,
    seed: int,
    decouple_support: bool,
    a_masks_unchanged: bool,
) -> pd.DataFrame:
    labels = annotated["label_eval"].to_numpy(dtype=np.float32)
    rows: List[Dict[str, Any]] = []
    clean_probs = probs_by_variant["clean"]
    for group, mask in _group_masks(annotated).items():
        if int(mask.sum()) == 0:
            continue
        clean_m = _metrics(labels[mask], clean_probs[mask])
        for idx, variant in enumerate(VARIANTS):
            probs = probs_by_variant[variant]
            metric = _metrics(labels[mask], probs[mask])
            row: Dict[str, Any] = {
                "dataset": dataset,
                "subgroup": group,
                "variant": variant,
                "n_eval": int(mask.sum()),
                "decouple_support_runtime": bool(decouple_support),
                "a_masks_unchanged": bool(a_masks_unchanged),
                "auc": metric["auc"],
                "bce": metric["bce"],
                "auc_drop_from_clean": None
                if metric["auc"] is None or clean_m["auc"] is None
                else float(clean_m["auc"] - metric["auc"]),
                "bce_increase_from_clean": float(metric["bce"] - clean_m["bce"]),
            }
            auc_lo, auc_hi = _bootstrap_delta_ci(
                labels[mask],
                clean_probs[mask],
                probs[mask],
                metric="auc",
                bootstrap=bootstrap,
                seed=seed + idx * 101 + len(group) * 17,
            )
            bce_lo, bce_hi = _bootstrap_delta_ci(
                labels[mask],
                clean_probs[mask],
                probs[mask],
                metric="bce",
                bootstrap=bootstrap,
                seed=seed + idx * 131 + len(group) * 19,
            )
            row["auc_drop_ci_low"] = auc_lo
            row["auc_drop_ci_high"] = auc_hi
            row["bce_increase_ci_low"] = bce_lo
            row["bce_increase_ci_high"] = bce_hi
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        degree = out[out["variant"] == "s_support_degree_matched_random"][
            ["dataset", "subgroup", "auc_drop_from_clean", "bce_increase_from_clean"]
        ].rename(
            columns={
                "auc_drop_from_clean": "degree_auc_drop",
                "bce_increase_from_clean": "degree_bce_increase",
            }
        )
        out = out.merge(degree, on=["dataset", "subgroup"], how="left")
        out["minus_degree_random_auc_drop"] = out["auc_drop_from_clean"] - out["degree_auc_drop"]
        out["minus_degree_random_bce_increase"] = out["bce_increase_from_clean"] - out["degree_bce_increase"]
    return out


def _checkpoint_dir(checkpoint_root: Path, dataset: str, seed: int) -> Path:
    candidates = [
        checkpoint_root / dataset / f"seed{seed}" / "best_full",
        checkpoint_root / dataset / str(seed) / "best_full",
        checkpoint_root / dataset / "seed42" / "best_full",
    ]
    for path in candidates:
        if (path / "best_model.pth").exists():
            return path
    raise FileNotFoundError(f"Cannot find decoupled best_full checkpoint for {dataset} under {checkpoint_root}")


def run_dataset(
    dataset: str,
    checkpoint_root: Path,
    output_root: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
    bootstrap: int,
    split: str,
) -> pd.DataFrame:
    save_dir = _checkpoint_dir(checkpoint_root, dataset, seed)
    checkpoint, model = _load_checkpoint_model(save_dir, device)
    info = checkpoint["info_dict"]
    args = dict(checkpoint.get("args", {}))
    decouple_support = bool(args.get("decouple_support", False))
    eval_frame = _eval_frame_from_checkpoint(checkpoint, dataset, None, split)

    a_original = _clone_a_masks(model)
    s_original = _clone_s_masks(model)
    union = s_original["decoupled_item_support_mask"].detach().cpu().bool() | s_original[
        "decoupled_sequence_support_mask"
    ].detach().cpu().bool()

    baseline = _predict_model(model, eval_frame, info, batch_size, num_workers, device)
    annotated = _annotate_a_relevance(
        baseline.rename(columns={"prob": "prob_clean"}),
        info,
        dataset,
        None,
    )
    annotated = _story_audit_columns(annotated, info, dataset, None)
    probs_by_variant: Dict[str, np.ndarray] = {
        "clean": annotated["prob_clean"].to_numpy(dtype=np.float32)
    }

    random_mask = _random_same_total(union, seed + 10007)
    _set_s_masks(model, random_mask)
    probs_by_variant["s_support_random_same_total"] = _predict_model(
        model,
        eval_frame,
        info,
        batch_size,
        num_workers,
        device,
    )["prob"].to_numpy(dtype=np.float32)
    _restore_s_masks(model, s_original)

    degree_mask = _random_same_row_degree(union, seed + 20011)
    _set_s_masks(model, degree_mask)
    probs_by_variant["s_support_degree_matched_random"] = _predict_model(
        model,
        eval_frame,
        info,
        batch_size,
        num_workers,
        device,
    )["prob"].to_numpy(dtype=np.float32)
    _restore_s_masks(model, s_original)

    a_ok = _a_masks_unchanged(model, a_original)
    pred_out = annotated[
        [
            "stu_id",
            "exer_id",
            "cpt_seq",
            "label_eval",
            "direct_seen",
            "direct_unseen",
            "direct_unseen_bridgeable",
            "bridgeable_at_model_k",
            "crg_mass_to_history",
            "seq_mass_to_history",
        ]
    ].copy()
    for variant, probs in probs_by_variant.items():
        pred_out[f"prob_{variant}"] = probs
    output_root.mkdir(parents=True, exist_ok=True)
    pred_out.to_csv(output_root / f"{dataset}_decoupled_support_predictions.csv", index=False)

    metrics = _metric_rows(
        dataset,
        annotated,
        probs_by_variant,
        bootstrap=bootstrap,
        seed=seed,
        decouple_support=decouple_support,
        a_masks_unchanged=a_ok,
    )
    metrics.to_csv(output_root / f"{dataset}_decoupled_support_metrics.csv", index=False)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output-root", default=str(ROOT / "results" / "mainline_e4_decoupled_support"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--split", default="test", choices=("valid", "test"))
    args = parser.parse_args()

    output_root = Path(args.output_root)
    device = torch.device(args.device)
    all_metrics: List[pd.DataFrame] = []
    for dataset in _parse_csv(args.datasets):
        metrics = run_dataset(
            dataset,
            Path(args.checkpoint_root),
            output_root,
            device,
            int(args.batch_size),
            int(args.num_workers),
            int(args.seed),
            int(args.bootstrap),
            str(args.split),
        )
        if not metrics.empty:
            all_metrics.append(metrics)
    if all_metrics:
        pd.concat(all_metrics, ignore_index=True).to_csv(output_root / "decoupled_support_metrics_all.csv", index=False)
    print(f"[ok] E4 decoupled support controls written to {output_root}")


if __name__ == "__main__":
    main()
