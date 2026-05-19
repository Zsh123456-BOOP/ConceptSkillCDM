#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Inference-only CRG support corruption controls.

This script keeps a trained full CRG+LCRF checkpoint fixed and mutates only the
global CRG support masks at inference time.  It does not retrain.  The controls
separate four claims:

1. evidence_support_corruption: remove the strongest train-evidence support.
2. degree_matched_random_support: replace support with same-row-degree random
   neighbours.
3. sequence_shuffled_support: keep item evidence but shuffle sequence evidence.
4. self_only_fallback: remove all non-self support as a fallback control.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

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


CORRUPTION_TYPES = (
    "evidence_support_corruption",
    "degree_matched_random_support",
    "sequence_shuffled_support",
    "self_only_fallback",
)


def _relation_module(model: torch.nn.Module) -> torch.nn.Module:
    return model.structure_module.relation_learning


def _clone_masks(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    relation = _relation_module(model)
    return {
        "item_support_mask": relation.item_support_mask.detach().clone(),
        "sequence_support_mask": relation.sequence_support_mask.detach().clone(),
    }


def _restore_masks(model: torch.nn.Module, masks: Mapping[str, torch.Tensor]) -> None:
    relation = _relation_module(model)
    relation.item_support_mask.copy_(masks["item_support_mask"])
    relation.sequence_support_mask.copy_(masks["sequence_support_mask"])


def _set_masks(model: torch.nn.Module, item_mask: torch.Tensor, sequence_mask: torch.Tensor) -> None:
    relation = _relation_module(model)
    relation.item_support_mask.copy_(item_mask.to(device=relation.item_support_mask.device).bool())
    relation.sequence_support_mask.copy_(sequence_mask.to(device=relation.sequence_support_mask.device).bool())


def _offdiag_eye(c: int) -> torch.Tensor:
    return ~torch.eye(c, dtype=torch.bool)


def _row_degree(mask: torch.Tensor) -> torch.Tensor:
    c = int(mask.size(0))
    return (mask.detach().cpu().bool() & _offdiag_eye(c)).sum(dim=1)


def _support_size_mean(item_mask: torch.Tensor, sequence_mask: torch.Tensor, allow_self: bool = True) -> float:
    c = int(item_mask.size(0))
    union = item_mask.detach().cpu().bool() | sequence_mask.detach().cpu().bool()
    if allow_self:
        union = union | torch.eye(c, dtype=torch.bool)
    return float(union.sum(dim=1).float().mean().item())


def _train_only_candidate_check(original: Mapping[str, torch.Tensor], info: Mapping[str, Any]) -> bool:
    item_mask = original["item_support_mask"].detach().cpu().bool()
    seq_mask = original["sequence_support_mask"].detach().cpu().bool()
    item_prior = info["item_prior_matrix"].detach().cpu().float() > 0
    seq_prior = info["sequence_prior_matrix"].detach().cpu().float() > 0
    c = int(item_mask.size(0))
    offdiag = _offdiag_eye(c)
    item_ok = bool(((item_mask & offdiag) & (~item_prior)).sum().item() == 0)
    seq_ok = bool(((seq_mask & offdiag) & (~seq_prior)).sum().item() == 0)
    return item_ok and seq_ok


def _remove_top_evidence(
    source_mask: torch.Tensor,
    evidence_score: torch.Tensor,
    ratio: float,
) -> torch.Tensor:
    source = source_mask.detach().cpu().bool().clone()
    score = evidence_score.detach().cpu().float()
    c = int(source.size(0))
    if ratio <= 0:
        return source
    offdiag = _offdiag_eye(c)
    out = source.clone()
    for row in range(c):
        cols = torch.nonzero(source[row] & offdiag[row], as_tuple=False).view(-1)
        degree = int(cols.numel())
        if degree <= 0:
            continue
        take = min(degree, max(1, int(math.ceil(float(ratio) * degree))))
        ranked = cols[torch.argsort(score[row, cols], descending=True)[:take]]
        out[row, ranked] = False
    return out


def _replace_with_random_same_degree(
    source_mask: torch.Tensor,
    original_union: torch.Tensor,
    ratio: float,
    seed: int,
) -> torch.Tensor:
    source = source_mask.detach().cpu().bool()
    union = original_union.detach().cpu().bool()
    c = int(source.size(0))
    out = source.clone()
    if ratio <= 0 or c <= 1:
        return out
    offdiag = _offdiag_eye(c)
    gen = torch.Generator().manual_seed(int(seed))
    for row in range(c):
        cols = torch.nonzero(source[row] & offdiag[row], as_tuple=False).view(-1)
        degree = int(cols.numel())
        if degree <= 0:
            continue
        take = min(degree, max(1, int(math.ceil(float(ratio) * degree))))
        remove = cols[torch.randperm(degree, generator=gen)[:take]]
        kept = out[row].clone()
        kept[remove] = False
        candidates = torch.nonzero((~union[row]) & offdiag[row] & (~kept), as_tuple=False).view(-1)
        if int(candidates.numel()) < take:
            candidates = torch.nonzero(offdiag[row] & (~kept), as_tuple=False).view(-1)
        if int(candidates.numel()) <= 0:
            continue
        add = candidates[torch.randperm(int(candidates.numel()), generator=gen)[: min(take, int(candidates.numel()))]]
        out[row, remove[: int(add.numel())]] = False
        out[row, add] = True
    return out


def _shuffle_sequence_support(
    sequence_mask: torch.Tensor,
    ratio: float,
    seed: int,
) -> torch.Tensor:
    source = sequence_mask.detach().cpu().bool()
    c = int(source.size(0))
    out = source.clone()
    if ratio <= 0 or c <= 1:
        return out
    offdiag = _offdiag_eye(c)
    gen = torch.Generator().manual_seed(int(seed))
    col_perm = torch.randperm(c, generator=gen)
    for row in range(c):
        cols = torch.nonzero(source[row] & offdiag[row], as_tuple=False).view(-1)
        degree = int(cols.numel())
        if degree <= 0:
            continue
        take = min(degree, max(1, int(math.ceil(float(ratio) * degree))))
        remove = cols[torch.randperm(degree, generator=gen)[:take]]
        proposed = col_perm[remove]
        out[row, remove] = False
        for col in proposed.tolist():
            if col == row or bool(out[row, col]):
                candidates = torch.nonzero(offdiag[row] & (~out[row]), as_tuple=False).view(-1)
                if int(candidates.numel()) == 0:
                    continue
                col = int(candidates[torch.randint(int(candidates.numel()), (1,), generator=gen)].item())
            out[row, int(col)] = True
    return out


def _build_control_masks(
    original: Mapping[str, torch.Tensor],
    info: Mapping[str, Any],
    corruption_type: str,
    ratio: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    item = original["item_support_mask"].detach().cpu().bool()
    seq = original["sequence_support_mask"].detach().cpu().bool()
    union = item | seq
    ratio = max(0.0, min(1.0, float(ratio)))
    if corruption_type == "evidence_support_corruption":
        item_score = info["item_prior_matrix"].detach().cpu().float()
        seq_score = info["sequence_prior_matrix"].detach().cpu().float()
        return _remove_top_evidence(item, item_score, ratio), _remove_top_evidence(seq, seq_score, ratio)
    if corruption_type == "degree_matched_random_support":
        return (
            _replace_with_random_same_degree(item, union, ratio, seed + 17),
            _replace_with_random_same_degree(seq, union, ratio, seed + 31),
        )
    if corruption_type == "sequence_shuffled_support":
        return item.clone(), _shuffle_sequence_support(seq, ratio, seed + 47)
    if corruption_type == "self_only_fallback":
        if ratio <= 0:
            return item.clone(), seq.clone()
        return torch.zeros_like(item), torch.zeros_like(seq)
    raise ValueError(f"Unsupported corruption_type={corruption_type!r}")


def _group_masks(annotated: pd.DataFrame) -> Dict[str, np.ndarray]:
    qseq = _quantile_bins(annotated["a_query_seq_top5_mass_mean"])
    return {
        "all": np.ones(len(annotated), dtype=bool),
        "graph_hits_history": annotated["group_graph_hits_history"].to_numpy(dtype=bool),
        "high_support_mass": annotated["group_high_support_mass"].to_numpy(dtype=bool),
        "query_seq_top5_q4_high": (qseq == "q4_high").to_numpy(dtype=bool),
        "multi_concept": annotated["group_multi_concept"].to_numpy(dtype=bool),
    }


def run(args: argparse.Namespace) -> pd.DataFrame:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint, model = _load_checkpoint_model(Path(args.full_save_dir), device)
    info = checkpoint["info_dict"]
    eval_frame = _eval_frame_from_checkpoint(checkpoint, args.dataset_name, args.data_dir, args.split)

    baseline_pred = _predict_model(model, eval_frame, info, args.batch_size, args.num_workers, device)
    annotated = _annotate_a_relevance(
        baseline_pred.rename(columns={"prob": "prob_clean"}),
        info,
        args.dataset_name,
        args.data_dir,
    )
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
    ratios = [float(x) for x in args.corruption_ratios]
    seeds = [int(args.seed) + i for i in range(max(1, int(args.trials)))]

    for corruption_type in args.corruption_types:
        if corruption_type not in CORRUPTION_TYPES:
            raise ValueError(f"Unknown corruption type {corruption_type!r}; expected one of {CORRUPTION_TYPES}")
        for ratio in ratios:
            effective_seeds = [int(args.seed)] if ratio <= 0.0 else seeds
            if corruption_type == "self_only_fallback" and ratio not in (0.0, 1.0):
                effective_ratio = ratio
            else:
                effective_ratio = ratio
            for seed in effective_seeds:
                item_mask, seq_mask = _build_control_masks(original, info, corruption_type, effective_ratio, seed)
                _set_masks(model, item_mask, seq_mask)
                pred = _predict_model(model, eval_frame, info, args.batch_size, args.num_workers, device)
                probs = pred["prob"].to_numpy(dtype=np.float32)
                mean_support = _support_size_mean(item_mask, seq_mask, allow_self=True)
                for group, mask in groups.items():
                    if not bool(mask.any()):
                        continue
                    m = _metrics(labels[mask], probs[mask])
                    clean = clean_metrics[group]
                    rows.append(
                        {
                            "dataset": args.dataset_name,
                            "corruption_type": corruption_type,
                            "corruption_ratio": float(ratio),
                            "seed": int(seed),
                            "group": group,
                            "n_eval": int(mask.sum()),
                            "auc": m["auc"],
                            "auc_drop": None
                            if m["auc"] is None or clean["auc"] is None
                            else float(clean["auc"] - m["auc"]),
                            "bce": m["bce"],
                            "bce_increase": float(m["bce"] - clean["bce"]),
                            "clean_auc": clean["auc"],
                            "clean_bce": clean["bce"],
                            "mean_support_size": mean_support,
                            "train_only_candidate_check_passed": bool(check_passed),
                        }
                    )
                _restore_masks(model, original)

    result = pd.DataFrame(rows)
    out_csv = out_dir / "crg_support_corruption_control.csv"
    result.to_csv(out_csv, index=False)
    print(f"Wrote {len(result)} rows to {out_csv}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--full_save_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--corruption_ratios", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--corruption_types", nargs="+", default=list(CORRUPTION_TYPES))
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
