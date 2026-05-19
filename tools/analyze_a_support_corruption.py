#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Inference-only CRG-support corruption counterfactual.

The experiment keeps the trained CRG/A_fused checkpoint fixed and progressively
replaces evidence support edges with row-degree-matched random non-evidence
edges.  It tests whether the selected support of the global concept map is
specific and necessary, rather than merely adding arbitrary graph neighbors.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

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
    _quantile_bins,
)


def _clone_original_masks(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    relation = model.structure_module.relation_learning
    return {
        "item_support_mask": relation.item_support_mask.detach().clone(),
        "sequence_support_mask": relation.sequence_support_mask.detach().clone(),
    }


def _restore_masks(model: torch.nn.Module, masks: Mapping[str, torch.Tensor]) -> None:
    relation = model.structure_module.relation_learning
    relation.item_support_mask.copy_(masks["item_support_mask"])
    relation.sequence_support_mask.copy_(masks["sequence_support_mask"])


def _apply_masks(model: torch.nn.Module, item_mask: torch.Tensor, seq_mask: torch.Tensor) -> None:
    relation = model.structure_module.relation_learning
    relation.item_support_mask.copy_(item_mask.to(device=relation.item_support_mask.device))
    relation.sequence_support_mask.copy_(seq_mask.to(device=relation.sequence_support_mask.device))


def _corrupt_one_source(
    source_mask: torch.Tensor,
    original_union: torch.Tensor,
    fraction: float,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, int, int]:
    """Replace a fraction of each row's source support with random non-evidence edges."""
    source = source_mask.detach().cpu().bool()
    union = original_union.detach().cpu().bool()
    C = int(source.size(0))
    out = source.clone()
    fraction = max(0.0, min(1.0, float(fraction)))
    removed_total = 0
    added_total = 0
    if fraction <= 0.0 or C <= 1:
        return out, removed_total, added_total
    eye = torch.eye(C, dtype=torch.bool)
    for row in range(C):
        cols = torch.nonzero(source[row] & (~eye[row]), as_tuple=False).view(-1)
        degree = int(cols.numel())
        if degree <= 0:
            continue
        take = min(degree, max(1, int(math.ceil(degree * fraction))))
        remove = cols[torch.randperm(degree, generator=generator)[:take]]
        kept = out[row].clone()
        kept[remove] = False

        # Prefer random edges outside the original evidence union.  This makes the
        # counterfactual a real support-identity corruption instead of a source swap.
        candidates = torch.nonzero((~union[row]) & (~eye[row]) & (~kept), as_tuple=False).view(-1)
        if int(candidates.numel()) < take:
            candidates = torch.nonzero((~eye[row]) & (~kept), as_tuple=False).view(-1)
        if int(candidates.numel()) <= 0:
            continue
        add_take = min(take, int(candidates.numel()))
        add = candidates[torch.randperm(int(candidates.numel()), generator=generator)[:add_take]]
        out[row, remove[:add_take]] = False
        out[row, add] = True
        removed_total += int(add_take)
        added_total += int(add_take)
    return out, removed_total, added_total


def make_corrupted_support(
    original: Mapping[str, torch.Tensor],
    fraction: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    item = original["item_support_mask"].detach().cpu().bool()
    seq = original["sequence_support_mask"].detach().cpu().bool()
    union = item | seq
    gen = torch.Generator().manual_seed(int(seed))
    item_new, item_removed, item_added = _corrupt_one_source(item, union, fraction, gen)
    seq_new, seq_removed, seq_added = _corrupt_one_source(seq, union, fraction, gen)
    new_union = item_new | seq_new
    orig_edges = int(union.sum().item())
    preserved = int((union & new_union).sum().item())
    meta = {
        "fraction": float(fraction),
        "seed": int(seed),
        "item_removed": item_removed,
        "item_added": item_added,
        "sequence_removed": seq_removed,
        "sequence_added": seq_added,
        "union_edges_original": orig_edges,
        "union_edges_new": int(new_union.sum().item()),
        "union_edges_preserved": preserved,
        "union_preservation_rate": None if orig_edges <= 0 else float(preserved / orig_edges),
    }
    return item_new, seq_new, meta


def _group_masks(annotated: pd.DataFrame) -> Dict[str, np.ndarray]:
    qseq = _quantile_bins(annotated["a_query_seq_top5_mass_mean"])
    return {
        "all": np.ones(len(annotated), dtype=bool),
        "graph_hits_history": annotated["group_graph_hits_history"].to_numpy(dtype=bool),
        "high_support_mass": annotated["group_high_support_mass"].to_numpy(dtype=bool),
        "query_seq_top5_q4_high": (qseq == "q4_high").to_numpy(dtype=bool),
        "multi_concept": annotated["group_multi_concept"].to_numpy(dtype=bool),
    }


def run_support_corruption(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint, model = _load_checkpoint_model(Path(args.full_save_dir), device)
    info = checkpoint["info_dict"]
    eval_frame = _eval_frame_from_checkpoint(checkpoint, args.dataset_name, args.data_dir, args.split)

    baseline_pred = _predict_model(model, eval_frame, info, args.batch_size, args.num_workers, device)
    annotated = _annotate_a_relevance(
        baseline_pred.rename(columns={"prob": "prob_A_fused"}),
        info,
        args.dataset_name,
        args.data_dir,
    )
    if args.save_baseline_samples:
        annotated.to_csv(out_dir / "a_support_corruption_baseline_samples.csv", index=False)
    labels = annotated["label_eval"].to_numpy(dtype=np.float32)
    baseline = annotated["prob_A_fused"].to_numpy(dtype=np.float32)
    groups = _group_masks(annotated)
    baseline_metrics = {
        name: _metrics(labels[mask], baseline[mask])
        for name, mask in groups.items()
        if bool(mask.any())
    }

    original = _clone_original_masks(model)
    rows: List[Dict[str, Any]] = []
    meta_rows: List[Dict[str, Any]] = []
    for frac in args.corruption_fracs:
        trials = 1 if float(frac) <= 0.0 else int(args.random_trials)
        for trial in range(trials):
            seed = int(args.seed) + 104729 * trial + int(round(float(frac) * 1000.0))
            item_mask, seq_mask, meta = make_corrupted_support(original, float(frac), seed)
            meta["trial"] = int(trial)
            meta_rows.append(meta)
            _apply_masks(model, item_mask, seq_mask)
            pred = _predict_model(model, eval_frame, info, args.batch_size, args.num_workers, device)
            probs = pred["prob"].to_numpy(dtype=np.float32)
            for group, mask in groups.items():
                if not mask.any():
                    continue
                m = _metrics(labels[mask], probs[mask])
                base = baseline_metrics[group]
                rows.append(
                    {
                        "fraction": float(frac),
                        "trial": int(trial),
                        "group": group,
                        "n": int(mask.sum()),
                        "auc": m["auc"],
                        "auc_drop_vs_original": None
                        if m["auc"] is None or base["auc"] is None
                        else float(base["auc"] - m["auc"]),
                        "bce": m["bce"],
                        "bce_increase_vs_original": float(m["bce"] - base["bce"]),
                        "union_preservation_rate": meta["union_preservation_rate"],
                    }
                )
            _restore_masks(model, original)

    summary = pd.DataFrame(rows)
    aggregate = (
        summary.groupby(["fraction", "group"], as_index=False)
        .agg(
            n=("n", "first"),
            auc_drop_mean=("auc_drop_vs_original", "mean"),
            auc_drop_std=("auc_drop_vs_original", "std"),
            bce_increase_mean=("bce_increase_vs_original", "mean"),
            bce_increase_std=("bce_increase_vs_original", "std"),
            union_preservation_mean=("union_preservation_rate", "mean"),
        )
        .sort_values(["group", "fraction"])
    )
    meta = pd.DataFrame(meta_rows)
    summary.to_csv(out_dir / "a_support_corruption_summary.csv", index=False)
    aggregate.to_csv(out_dir / "a_support_corruption_aggregate.csv", index=False)
    meta.to_csv(out_dir / "a_support_corruption_meta.csv", index=False)
    (out_dir / "a_support_corruption_config.json").write_text(
        json.dumps(
            {
                "dataset_name": args.dataset_name,
                "split": args.split,
                "full_save_dir": args.full_save_dir,
                "corruption_fracs": [float(x) for x in args.corruption_fracs],
                "random_trials": int(args.random_trials),
                "seed": int(args.seed),
                "batch_size": int(args.batch_size),
                "num_workers": int(args.num_workers),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "rows": {"summary": len(summary), "aggregate": len(aggregate), "meta": len(meta)},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return summary, aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", default="assist_09")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--full_save_dir", required=True)
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corruption_fracs", type=float, nargs="+", default=[0.0, 0.25, 0.50, 0.75, 1.0])
    parser.add_argument("--random_trials", type=int, default=5)
    parser.add_argument("--save_baseline_samples", action="store_true")
    args = parser.parse_args()
    run_support_corruption(args)


if __name__ == "__main__":
    main()
