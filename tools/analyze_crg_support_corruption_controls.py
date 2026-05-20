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
    _resolve_files,
    _row_normalize,
    _student_history,
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


def _story_audit_columns(
    annotated: pd.DataFrame,
    info: Mapping[str, Any],
    dataset_name: str,
    data_dir: Optional[str],
) -> pd.DataFrame:
    """Add reachability subgroup columns used by the CRG/LCRF story audit.

    These columns are derived only from train split histories and the checkpoint
    CRG priors. They intentionally avoid correctness labels beyond train-set
    history summaries already used by the model.
    """

    _, train_file, _, _ = _resolve_files(dataset_name, data_dir)
    train_df = pd.read_csv(train_file)
    train_df = train_df[train_df["stu_id"].isin(info["stu_id_map"].keys())].reset_index(drop=True)
    stu_counts, stu_history, _ = _student_history(train_df, info["cpt_id_map"])

    q_matrix = info["q_matrix"].detach().cpu()
    item_prior = info["item_prior_matrix"].detach().cpu().float()
    sequence_prior = info["sequence_prior_matrix"].detach().cpu().float()
    prior = item_prior + sequence_prior
    support_mask = prior > 0
    crg_prob = _row_normalize(prior).numpy()
    seq_prob = _row_normalize(sequence_prior).numpy()

    direct_seen: List[bool] = []
    bridge5: List[bool] = []
    bridge10: List[bool] = []
    bridge_model: List[bool] = []
    crg_mass: List[float] = []
    seq_mass: List[float] = []
    anon_map: Dict[Any, str] = {}

    for row in annotated.itertuples(index=False):
        sid = getattr(row, "stu_id")
        if sid not in anon_map:
            anon_map[sid] = f"S{len(anon_map) + 1}"
        hist = set(stu_history.get(sid, set()))
        mapped_ex = int(getattr(row, "mapped_exercise_id"))
        concepts = torch.nonzero(q_matrix[mapped_ex] > 0, as_tuple=False).view(-1).cpu().numpy().astype(int).tolist()
        seen_values: List[bool] = []
        bridge5_values: List[bool] = []
        bridge10_values: List[bool] = []
        bridge_model_values: List[bool] = []
        crg_values: List[float] = []
        seq_values: List[float] = []
        for concept in concepts:
            seen_values.append(concept in hist)
            if hist:
                top5 = np.argsort(crg_prob[concept])[-min(5, crg_prob.shape[1]):]
                top10 = np.argsort(crg_prob[concept])[-min(10, crg_prob.shape[1]):]
                bridge5_values.append(bool(set(top5.tolist()) & hist))
                bridge10_values.append(bool(set(top10.tolist()) & hist))
                bridge_model_values.append(bool(support_mask[concept, list(hist)].any().item()))
                crg_values.append(float(crg_prob[concept, list(hist)].sum()))
                seq_values.append(float(seq_prob[concept, list(hist)].sum()))
            else:
                bridge5_values.append(False)
                bridge10_values.append(False)
                bridge_model_values.append(False)
                crg_values.append(0.0)
                seq_values.append(0.0)
        direct_seen.append(bool(any(seen_values)))
        bridge5.append(bool(any(bridge5_values)))
        bridge10.append(bool(any(bridge10_values)))
        bridge_model.append(bool(any(bridge_model_values)))
        crg_mass.append(float(np.mean(crg_values)) if crg_values else 0.0)
        seq_mass.append(float(np.mean(seq_values)) if seq_values else 0.0)

    out = annotated.copy()
    out["student_id_anonymized"] = [anon_map[sid] for sid in out["stu_id"].tolist()]
    out["direct_seen"] = direct_seen
    out["direct_unseen"] = ~out["direct_seen"]
    out["bridgeable_at_5"] = bridge5
    out["bridgeable_at_10"] = bridge10
    out["bridgeable_at_model_k"] = bridge_model
    out["direct_unseen_bridgeable"] = out["direct_unseen"] & out["bridgeable_at_model_k"]
    out["unbridgeable"] = ~out["bridgeable_at_model_k"]
    out["crg_mass_to_history"] = crg_mass
    out["seq_mass_to_history"] = seq_mass

    seq_bins = _quantile_bins(out["a_query_seq_top5_mass_mean"])
    crg_positive = out["crg_mass_to_history"] > 0.0
    crg_cut = float(np.quantile(out.loc[crg_positive, "crg_mass_to_history"], 0.75)) if crg_positive.any() else float("inf")
    history_cut = float(np.median(out["student_train_count"].astype(float))) if len(out) else 0.0
    out["high_seq_support"] = seq_bins.astype(str).eq("q4_high")
    out["low_seq_support"] = seq_bins.astype(str).isin(["zero", "q1_low"])
    out["high_crg_mass"] = out["crg_mass_to_history"] >= crg_cut
    out["low_crg_mass"] = out["crg_mass_to_history"] <= float(np.quantile(out["crg_mass_to_history"], 0.25)) if len(out) else False
    out["short_history"] = out["student_train_count"].astype(float) <= history_cut
    out["long_history"] = out["student_train_count"].astype(float) > history_cut
    return out


def _group_masks(annotated: pd.DataFrame) -> Dict[str, np.ndarray]:
    qseq = _quantile_bins(annotated["a_query_seq_top5_mass_mean"])
    groups = {
        "all": np.ones(len(annotated), dtype=bool),
        "graph_hits_history": annotated["group_graph_hits_history"].to_numpy(dtype=bool),
        "high_support_mass": annotated["group_high_support_mass"].to_numpy(dtype=bool),
        "query_seq_top5_q4_high": (qseq == "q4_high").to_numpy(dtype=bool),
        "multi_concept": annotated["group_multi_concept"].to_numpy(dtype=bool),
    }
    for col in (
        "direct_seen",
        "direct_unseen",
        "direct_unseen_bridgeable",
        "high_seq_support",
        "high_crg_mass",
        "short_history",
        "long_history",
    ):
        if col in annotated:
            groups[col] = annotated[col].to_numpy(dtype=bool)
    return groups


def _attach_gap_and_claims(result: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if result.empty:
        result["evidence_minus_degree_random_auc_drop"] = np.nan
        result["evidence_minus_degree_random_bce_increase"] = np.nan
        result["bootstrap_ci_low"] = np.nan
        result["bootstrap_ci_high"] = np.nan
        result["claim_status"] = "weak"
        return result, pd.DataFrame()

    key_cols = ["dataset", "group", "corruption_ratio", "seed"]
    degree = result[result["corruption_type"] == "degree_matched_random_support"][
        key_cols + ["auc_drop", "bce_increase"]
    ].rename(columns={"auc_drop": "degree_auc_drop", "bce_increase": "degree_bce_increase"})
    out = result.merge(degree, on=key_cols, how="left")
    out["evidence_minus_degree_random_auc_drop"] = out["auc_drop"] - out["degree_auc_drop"]
    out["evidence_minus_degree_random_bce_increase"] = out["bce_increase"] - out["degree_bce_increase"]

    ci_rows: List[Dict[str, Any]] = []
    evidence = out[
        (out["corruption_type"] == "evidence_support_corruption")
        & (out["corruption_ratio"].astype(float) == 1.0)
    ].copy()
    for (dataset, group), frame in evidence.groupby(["dataset", "group"]):
        gaps = frame["evidence_minus_degree_random_auc_drop"].dropna().to_numpy(dtype=float)
        if gaps.size:
            low, high = np.quantile(gaps, [0.025, 0.975])
            mean_gap = float(np.mean(gaps))
        else:
            low = high = mean_gap = np.nan
        bce_gaps = frame["evidence_minus_degree_random_bce_increase"].dropna().to_numpy(dtype=float)
        if bce_gaps.size:
            bce_low, bce_high = np.quantile(bce_gaps, [0.025, 0.975])
            bce_mean = float(np.mean(bce_gaps))
        else:
            bce_low = bce_high = bce_mean = np.nan
        auc_drop = float(frame["auc_drop"].dropna().mean()) if frame["auc_drop"].notna().any() else np.nan
        bce_inc = float(frame["bce_increase"].dropna().mean()) if frame["bce_increase"].notna().any() else np.nan
        strong = (np.isfinite(low) and low > 0.0) or (np.isfinite(bce_low) and bce_low > 0.0)
        support_only = (np.isfinite(auc_drop) and auc_drop >= 0.005) or (np.isfinite(bce_inc) and bce_inc >= 0.005)
        status = "strong_evidence_gap" if strong else ("support_dependence_only" if support_only else "weak")
        ci_rows.append(
            {
                "dataset": dataset,
                "subgroup": group,
                "corruption_ratio": 1.0,
                "evidence_minus_degree_random_auc_drop": mean_gap,
                "bootstrap_ci_low": float(low) if np.isfinite(low) else np.nan,
                "bootstrap_ci_high": float(high) if np.isfinite(high) else np.nan,
                "evidence_minus_degree_random_bce_increase": bce_mean,
                "bce_bootstrap_ci_low": float(bce_low) if np.isfinite(bce_low) else np.nan,
                "bce_bootstrap_ci_high": float(bce_high) if np.isfinite(bce_high) else np.nan,
                "claim_status": status,
            }
        )
    ci = pd.DataFrame(ci_rows)
    out = out.merge(
        ci[["dataset", "subgroup", "bootstrap_ci_low", "bootstrap_ci_high", "claim_status"]].rename(columns={"subgroup": "group"}),
        on=["dataset", "group"],
        how="left",
    )
    out["claim_status"] = out["claim_status"].fillna("weak")
    return out.drop(columns=["degree_auc_drop", "degree_bce_increase"], errors="ignore"), ci


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
    annotated = _story_audit_columns(annotated, info, args.dataset_name, args.data_dir)
    audit_cols = [
        "student_id_anonymized",
        "exer_id",
        "cpt_seq",
        "label_eval",
        "prob_clean",
        "direct_seen",
        "direct_unseen",
        "bridgeable_at_5",
        "bridgeable_at_10",
        "bridgeable_at_model_k",
        "direct_unseen_bridgeable",
        "unbridgeable",
        "high_seq_support",
        "low_seq_support",
        "high_crg_mass",
        "low_crg_mass",
        "short_history",
        "long_history",
        "student_train_count",
        "crg_mass_to_history",
        "seq_mass_to_history",
    ]
    annotated[[c for c in audit_cols if c in annotated.columns]].to_csv(out_dir / "sample_level_crg_audit.csv", index=False)
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
    result, ci = _attach_gap_and_claims(result)
    out_csv = out_dir / "crg_support_corruption_control.csv"
    result.to_csv(out_csv, index=False)
    result.rename(
        columns={
            "group": "subgroup",
            "auc_drop": "auc_drop_from_clean",
            "bce_increase": "bce_increase_from_clean",
        }
    ).to_csv(out_dir / "subgroup_crg_necessity_metrics.csv", index=False)
    ci.to_csv(out_dir / "crg_necessity_bootstrap_gap.csv", index=False)
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
