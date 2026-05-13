#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A-module sufficiency/necessity controls on a fixed checkpoint.

The script does not retrain.  It uses:

1. Edge deletion inference controls on an A_fused checkpoint.
2. A-relevant subgroup monotonicity using A_fused vs no_A predictions.
3. Held-out concept-transition retrieval with bootstrap CIs and random repeats.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CognitiveDiagnosisDataset, _parse_concept_seq, make_degree_random_prior  # noqa: E402
from src.experiment_utils import compute_metrics  # noqa: E402
from tools.analyze_ae_errors import _build_model, _resolve_data_dir  # noqa: E402
from tools.analyze_a_support_evidence import (  # noqa: E402
    _build_prior_variants,
    _load_info,
    _resolve_files,
    _row_normalize,
    _student_history,
    _transition_pairs,
)


def _safe_auc(labels: np.ndarray, probs: np.ndarray) -> Optional[float]:
    labels = labels.astype(np.float32)
    probs = probs.astype(np.float32)
    if len(np.unique(labels)) < 2:
        return None
    return float(compute_metrics(labels, (probs > 0.5).astype(np.float32), probs)["auc"])


def _bce(labels: np.ndarray, probs: np.ndarray) -> float:
    y = labels.astype(np.float64)
    p = probs.astype(np.float64).clip(1e-8, 1.0 - 1e-8)
    return float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))


def _metrics(labels: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
    return {"auc": _safe_auc(labels, probs), "bce": _bce(labels, probs)}


def _load_checkpoint_model(save_dir: Path, device: torch.device) -> Tuple[Mapping[str, Any], torch.nn.Module]:
    checkpoint = torch.load(save_dir / "best_model.pth", map_location=device, weights_only=False)
    model = _build_model(checkpoint, device)
    return checkpoint, model


def _eval_frame_from_checkpoint(
    checkpoint: Mapping[str, Any],
    dataset_name: str,
    data_dir: Optional[str],
    split: str,
) -> pd.DataFrame:
    info = checkpoint["info_dict"]
    raw_dir = _resolve_data_dir(dataset_name, data_dir)
    raw = pd.read_csv(os.path.join(raw_dir, f"{split}.csv"))
    return raw[
        raw["stu_id"].isin(set(info["stu_id_map"].keys()))
        & raw["exer_id"].isin(set(info["exer_id_map"].keys()))
    ].reset_index(drop=True)


def _predict_model(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    info: Mapping[str, Any],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> pd.DataFrame:
    dataset = CognitiveDiagnosisDataset(frame, info["stu_id_map"], info["exer_id_map"], info["cpt_id_map"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    probs: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    mapped_students: List[np.ndarray] = []
    mapped_exercises: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for student_ids, exercise_ids, y in loader:
            student_ids = student_ids.to(device)
            exercise_ids = exercise_ids.to(device)
            logits, _ = model(student_ids, exercise_ids, return_details=True, return_logits=True)
            probs.append(torch.sigmoid(logits.reshape(-1)).detach().cpu().numpy())
            labels.append(y.reshape(-1).detach().cpu().numpy())
            mapped_students.append(student_ids.detach().cpu().numpy())
            mapped_exercises.append(exercise_ids.detach().cpu().numpy())
    out = frame[["stu_id", "exer_id", "cpt_seq", "label"]].copy()
    out["label_eval"] = np.concatenate(labels)
    out["mapped_student_id"] = np.concatenate(mapped_students)
    out["mapped_exercise_id"] = np.concatenate(mapped_exercises)
    out["prob"] = np.concatenate(probs)
    return out


def _annotate_a_relevance(base: pd.DataFrame, info: Mapping[str, Any], dataset_name: str, data_dir: Optional[str]) -> pd.DataFrame:
    _, train_file, _, _ = _resolve_files(dataset_name, data_dir)
    train_df = pd.read_csv(train_file)
    train_df = train_df[train_df["stu_id"].isin(info["stu_id_map"].keys())].reset_index(drop=True)
    stu_counts, stu_history, concept_freq = _student_history(train_df, info["cpt_id_map"])
    q_matrix = info["q_matrix"].detach().cpu()
    item_prior = info["item_prior_matrix"].detach().cpu()
    sequence_prior = info["sequence_prior_matrix"].detach().cpu()
    support_prior = _row_normalize(item_prior + sequence_prior).numpy()

    masses: List[float] = []
    min_freqs: List[float] = []
    concept_counts: List[int] = []
    history_counts: List[int] = []
    for row in base.itertuples(index=False):
        mapped_ex = int(getattr(row, "mapped_exercise_id"))
        mapped_concepts = torch.nonzero(q_matrix[mapped_ex] > 0, as_tuple=False).view(-1).cpu().numpy().astype(int).tolist()
        hist = stu_history.get(getattr(row, "stu_id"), set())
        row_masses = [float(support_prior[c, list(hist)].sum()) if hist else 0.0 for c in mapped_concepts]
        masses.append(float(np.mean(row_masses)) if row_masses else 0.0)
        min_freqs.append(float(min([concept_freq[c] for c in mapped_concepts], default=0.0)))
        concept_counts.append(len(mapped_concepts))
        history_counts.append(int(stu_counts.get(getattr(row, "stu_id"), 0)))

    out = base.copy()
    out["a_support_mass_to_history"] = masses
    out["a_min_concept_train_freq"] = min_freqs
    out["concept_count"] = concept_counts
    out["student_train_count"] = history_counts
    positive = out["a_support_mass_to_history"].to_numpy() > 0
    high_cut = float(np.quantile(out.loc[positive, "a_support_mass_to_history"], 0.75)) if positive.any() else float("inf")
    out["group_all"] = True
    out["group_graph_hits_history"] = out["a_support_mass_to_history"] > 0.0
    out["group_high_support_mass"] = out["a_support_mass_to_history"] >= high_cut
    out["group_multi_concept"] = out["concept_count"] >= 2
    return out


def _support_mass_bins(frame: pd.DataFrame) -> pd.Series:
    masses = frame["a_support_mass_to_history"].astype(float)
    labels = pd.Series("zero", index=frame.index, dtype="object")
    positive = masses > 0.0
    if positive.sum() < 4:
        labels.loc[positive] = "positive"
        return labels
    try:
        qbins = pd.qcut(masses.loc[positive], q=4, labels=["q1_low", "q2_midlow", "q3_midhigh", "q4_high"], duplicates="drop")
        labels.loc[positive] = qbins.astype(str)
    except ValueError:
        labels.loc[positive] = "positive"
    return labels


def subgroup_monotonicity(
    args: argparse.Namespace,
    out_dir: Path,
    checkpoint_full: Mapping[str, Any],
    full_model: torch.nn.Module,
    device: torch.device,
) -> pd.DataFrame:
    info = checkpoint_full["info_dict"]
    eval_frame = _eval_frame_from_checkpoint(checkpoint_full, args.dataset_name, args.data_dir, args.split)
    full_pred = _predict_model(full_model, eval_frame, info, args.batch_size, args.num_workers, device)
    no_a_ckpt, no_a_model = _load_checkpoint_model(Path(args.no_a_save_dir), device)
    no_a_pred = _predict_model(no_a_model, eval_frame, no_a_ckpt["info_dict"], args.batch_size, args.num_workers, device)

    base = full_pred.rename(columns={"prob": "prob_A_fused"})
    base["prob_no_A"] = no_a_pred["prob"].to_numpy()
    base = _annotate_a_relevance(base, info, args.dataset_name, args.data_dir)
    base["support_mass_bin"] = _support_mass_bins(base)
    base["bce_gain_A_over_no_A"] = (
        -(
            base["label_eval"] * np.log(base["prob_no_A"].clip(1e-8, 1.0 - 1e-8))
            + (1.0 - base["label_eval"]) * np.log((1.0 - base["prob_no_A"]).clip(1e-8, 1.0 - 1e-8))
        )
        + (
            base["label_eval"] * np.log(base["prob_A_fused"].clip(1e-8, 1.0 - 1e-8))
            + (1.0 - base["label_eval"]) * np.log((1.0 - base["prob_A_fused"]).clip(1e-8, 1.0 - 1e-8))
        )
    )
    base.to_csv(out_dir / "a_relevant_samples.csv", index=False)

    rows: List[Dict[str, Any]] = []
    group_specs = [("support_mass_bin", None)]
    for col in ("group_all", "group_graph_hits_history", "group_high_support_mass", "group_multi_concept"):
        group_specs.append((col, True))
    for col, value in group_specs:
        if value is None:
            iterator = base.groupby(col, dropna=False, sort=False)
        else:
            iterator = [(col.replace("group_", ""), base[base[col] == value])]
        for key, grp in iterator:
            if grp.empty:
                continue
            labels = grp["label_eval"].to_numpy(dtype=np.float32)
            full_probs = grp["prob_A_fused"].to_numpy(dtype=np.float32)
            no_a_probs = grp["prob_no_A"].to_numpy(dtype=np.float32)
            full_m = _metrics(labels, full_probs)
            no_a_m = _metrics(labels, no_a_probs)
            rows.append(
                {
                    "group_type": col,
                    "group": str(key),
                    "n": int(len(grp)),
                    "support_mass_mean": float(grp["a_support_mass_to_history"].mean()),
                    "A_fused_auc": full_m["auc"],
                    "no_A_auc": no_a_m["auc"],
                    "A_fused_minus_no_A_auc": None
                    if full_m["auc"] is None or no_a_m["auc"] is None
                    else float(full_m["auc"] - no_a_m["auc"]),
                    "A_fused_bce": full_m["bce"],
                    "no_A_bce": no_a_m["bce"],
                    "no_A_minus_A_fused_bce": float(no_a_m["bce"] - full_m["bce"]),
                    "sample_bce_gain_mean": float(grp["bce_gain_A_over_no_A"].mean()),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "a_relevant_subgroup_monotonicity.csv", index=False)

    bins = result[result["group_type"] == "support_mass_bin"].copy()
    if len(bins) >= 3:
        result.attrs["support_gain_spearman"] = float(
            bins["support_mass_mean"].rank().corr(bins["no_A_minus_A_fused_bce"].rank())
        )
    return result


def _current_support_mask(model: torch.nn.Module) -> torch.Tensor:
    relation = model.structure_module.relation_learning
    return (relation.item_support_mask | relation.sequence_support_mask).detach().cpu().bool()


def _make_delete_mask(
    evidence: torch.Tensor,
    support: torch.Tensor,
    fraction: float,
    *,
    mode: str,
    seed: int,
) -> torch.Tensor:
    C = int(evidence.size(0))
    fraction = max(0.0, min(1.0, float(fraction)))
    delete = torch.zeros(C, C, dtype=torch.bool)
    gen = torch.Generator().manual_seed(int(seed))
    for row in range(C):
        cols = torch.nonzero(support[row], as_tuple=False).view(-1)
        cols = cols[cols != row]
        degree = int(cols.numel())
        if degree <= 0:
            continue
        take = max(1, int(math.ceil(degree * fraction)))
        take = min(take, degree)
        if mode == "top":
            scores = evidence[row, cols]
            chosen = cols[torch.argsort(scores, descending=True)[:take]]
        elif mode == "random":
            chosen = cols[torch.randperm(degree, generator=gen)[:take]]
        else:
            raise ValueError(f"unknown delete mode {mode!r}")
        delete[row, chosen] = True
    return delete


def _apply_delete_mask(model: torch.nn.Module, delete_mask: torch.Tensor, original: Mapping[str, torch.Tensor]) -> None:
    relation = model.structure_module.relation_learning
    mask = delete_mask.to(device=relation.item_support_mask.device)
    relation.item_support_mask.copy_(original["item_support_mask"] & (~mask))
    relation.sequence_support_mask.copy_(original["sequence_support_mask"] & (~mask))


def _restore_support_masks(model: torch.nn.Module, original: Mapping[str, torch.Tensor]) -> None:
    relation = model.structure_module.relation_learning
    relation.item_support_mask.copy_(original["item_support_mask"])
    relation.sequence_support_mask.copy_(original["sequence_support_mask"])


def edge_deletion_experiment(
    args: argparse.Namespace,
    out_dir: Path,
    checkpoint_full: Mapping[str, Any],
    model: torch.nn.Module,
    annotated: pd.DataFrame,
    device: torch.device,
) -> pd.DataFrame:
    info = checkpoint_full["info_dict"]
    eval_frame = _eval_frame_from_checkpoint(checkpoint_full, args.dataset_name, args.data_dir, args.split)
    labels = annotated["label_eval"].to_numpy(dtype=np.float32)
    baseline = annotated["prob_A_fused"].to_numpy(dtype=np.float32)
    relation = model.structure_module.relation_learning
    original = {
        "item_support_mask": relation.item_support_mask.detach().clone(),
        "sequence_support_mask": relation.sequence_support_mask.detach().clone(),
    }
    support = _current_support_mask(model)
    item_prior = info["item_prior_matrix"].detach().cpu()
    sequence_prior = info["sequence_prior_matrix"].detach().cpu()
    evidence = _row_normalize(item_prior + sequence_prior).cpu()
    groups = {
        "all": np.ones(len(annotated), dtype=bool),
        "graph_hits_history": annotated["group_graph_hits_history"].to_numpy(dtype=bool),
        "high_support_mass": annotated["group_high_support_mass"].to_numpy(dtype=bool),
        "multi_concept": annotated["group_multi_concept"].to_numpy(dtype=bool),
    }
    baseline_metrics = {
        name: _metrics(labels[mask], baseline[mask])
        for name, mask in groups.items()
        if bool(mask.any())
    }

    rows: List[Dict[str, Any]] = []
    for frac in args.edge_delete_fracs:
        for mode in ("top", "random"):
            trials = 1 if mode == "top" else int(args.edge_random_trials)
            for trial in range(trials):
                delete = _make_delete_mask(
                    evidence,
                    support,
                    frac,
                    mode=mode,
                    seed=int(args.seed) + 1009 * trial + int(float(frac) * 1000),
                )
                _apply_delete_mask(model, delete, original)
                pred = _predict_model(model, eval_frame, info, args.batch_size, args.num_workers, device)
                probs = pred["prob"].to_numpy(dtype=np.float32)
                for group, mask in groups.items():
                    if not mask.any():
                        continue
                    m = _metrics(labels[mask], probs[mask])
                    base = baseline_metrics[group]
                    rows.append(
                        {
                            "delete_mode": mode,
                            "fraction": float(frac),
                            "trial": int(trial),
                            "group": group,
                            "n": int(mask.sum()),
                            "deleted_edges": int(delete.sum().item()),
                            "auc": m["auc"],
                            "auc_drop_vs_baseline": None
                            if m["auc"] is None or base["auc"] is None
                            else float(base["auc"] - m["auc"]),
                            "bce": m["bce"],
                            "bce_increase_vs_baseline": float(m["bce"] - base["bce"]),
                        }
                    )
                _restore_support_masks(model, original)
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "a_edge_deletion_summary.csv", index=False)
    agg = (
        result.groupby(["delete_mode", "fraction", "group"], as_index=False)
        .agg(
            n=("n", "first"),
            auc_drop_mean=("auc_drop_vs_baseline", "mean"),
            auc_drop_std=("auc_drop_vs_baseline", "std"),
            bce_increase_mean=("bce_increase_vs_baseline", "mean"),
            bce_increase_std=("bce_increase_vs_baseline", "std"),
        )
    )
    agg.to_csv(out_dir / "a_edge_deletion_aggregate.csv", index=False)
    return result


def _rank_values(prior: torch.Tensor, pairs: pd.DataFrame, ks: Sequence[int]) -> Dict[str, np.ndarray]:
    scores = prior.detach().cpu().float().numpy()
    C = scores.shape[0]
    order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(order)
    ranks[np.arange(C)[:, None], order] = np.arange(1, C + 1)[None, :]
    q = pairs["query_concept"].to_numpy(dtype=np.int64)
    target = pairs["support_concept"].to_numpy(dtype=np.int64)
    r = ranks[q, target].astype(np.float64)
    out = {"mrr": 1.0 / r}
    for k in ks:
        kk = int(k)
        hit = (r <= kk).astype(np.float64)
        out[f"hit@{kk}"] = hit
        out[f"ndcg@{kk}"] = np.where(hit > 0, 1.0 / np.log2(r + 1.0), 0.0)
    return out


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / max(1e-12, float(np.sum(weights))))


def transition_retrieval_with_ci(args: argparse.Namespace, info: Mapping[str, Any], out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _, _, valid_file, test_file = _resolve_files(args.dataset_name, args.data_dir)
    eval_df = pd.concat([pd.read_csv(valid_file), pd.read_csv(test_file)], ignore_index=True)
    eval_df = eval_df[eval_df["stu_id"].isin(info["stu_id_map"].keys())].reset_index(drop=True)
    pairs = _transition_pairs(eval_df, info["cpt_id_map"], max_hops=args.transition_hops, decay=args.transition_decay)
    pairs.to_csv(out_dir / "a_heldout_transition_pairs.csv", index=False)
    weights = pairs["weight"].to_numpy(dtype=np.float64)
    n = len(pairs)
    rng = np.random.default_rng(int(args.seed))
    variants = _build_prior_variants(info)

    rows: List[Dict[str, Any]] = []
    ci_rows: List[Dict[str, Any]] = []
    metric = args.bootstrap_metric
    for name, prior in variants.items():
        values = _rank_values(prior, pairs, args.ks)
        row: Dict[str, Any] = {"variant": name, "pairs": int(n), "weight_sum": float(weights.sum())}
        for key, vals in values.items():
            row[key] = _weighted_mean(vals, weights)
        rows.append(row)
        if metric in values and n > 0 and args.bootstrap_iters > 0:
            boot = np.empty(int(args.bootstrap_iters), dtype=np.float64)
            vals = values[metric]
            for i in range(int(args.bootstrap_iters)):
                idx = rng.integers(0, n, size=n)
                boot[i] = _weighted_mean(vals[idx], weights[idx])
            ci_rows.append(
                {
                    "variant": name,
                    "metric": metric,
                    "mean": float(boot.mean()),
                    "ci_low": float(np.quantile(boot, 0.025)),
                    "ci_high": float(np.quantile(boot, 0.975)),
                }
            )

    random_rows: List[Dict[str, Any]] = []
    item = info["item_prior_matrix"].detach().cpu().float()
    seq = info["sequence_prior_matrix"].detach().cpu().float()
    for i in range(int(args.random_repeats)):
        prior, meta = make_degree_random_prior(item, seq, seed=int(args.seed) + 7919 * (i + 1))
        values = _rank_values(prior, pairs, args.ks)
        row: Dict[str, Any] = {"variant": "A_degree_random_repeat", "repeat": i, "seed": int(meta["degree_random_seed"])}
        for key, vals in values.items():
            row[key] = _weighted_mean(vals, weights)
        random_rows.append(row)

    retrieval = pd.DataFrame(rows)
    ci = pd.DataFrame(ci_rows)
    random_df = pd.DataFrame(random_rows)
    retrieval.to_csv(out_dir / "a_transition_retrieval.csv", index=False)
    ci.to_csv(out_dir / "a_transition_retrieval_bootstrap_ci.csv", index=False)
    random_df.to_csv(out_dir / "a_transition_degree_random_repeats.csv", index=False)
    return retrieval, ci, random_df


def write_summary(out_dir: Path, subgroup: pd.DataFrame, edge: pd.DataFrame, retrieval: pd.DataFrame, ci: pd.DataFrame, random_df: pd.DataFrame) -> None:
    support_bins = subgroup[subgroup["group_type"] == "support_mass_bin"].copy()
    edge_focus = edge[(edge["group"] == "high_support_mass") & (edge["fraction"] == edge["fraction"].min())].copy()
    summary = {
        "subgroup_bins": support_bins.to_dict("records"),
        "edge_deletion_focus": edge_focus.to_dict("records"),
        "transition_retrieval": retrieval.to_dict("records"),
        "transition_ci": ci.to_dict("records"),
        "degree_random_repeats": {
            "count": int(len(random_df)),
            "hit10_mean": None if "hit@10" not in random_df else float(random_df["hit@10"].mean()),
            "hit10_low": None if "hit@10" not in random_df else float(np.quantile(random_df["hit@10"], 0.025)),
            "hit10_high": None if "hit@10" not in random_df else float(np.quantile(random_df["hit@10"], 0.975)),
        },
    }
    (out_dir / "a_sufficiency_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# A Sufficiency Controls", ""]

    def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
        if frame.empty:
            return ""
        rows = frame.loc[:, list(columns)].copy()
        header = "| " + " | ".join(columns) + " |"
        sep = "| " + " | ".join(["---"] * len(columns)) + " |"
        body = []
        for item in rows.itertuples(index=False):
            vals = []
            for value in item:
                if isinstance(value, float):
                    vals.append(f"{value:.6f}")
                else:
                    vals.append(str(value))
            body.append("| " + " | ".join(vals) + " |")
        return "\n".join([header, sep, *body])

    lines.append("## Prediction Necessity")
    lines.append("Use `A_fused_neutralE` vs `no_A_fair` from `../mechanism_results.csv` for the main prediction necessity result.")
    lines.append("")
    lines.append("## A-Relevant Monotonicity")
    if not support_bins.empty:
        lines.append(_markdown_table(support_bins, ["group", "n", "support_mass_mean", "A_fused_minus_no_A_auc", "no_A_minus_A_fused_bce"]))
    lines.append("")
    lines.append("## Edge Deletion")
    if not edge_focus.empty:
        lines.append(_markdown_table(edge_focus, ["delete_mode", "fraction", "trial", "group", "auc_drop_vs_baseline", "bce_increase_vs_baseline"]))
    lines.append("")
    lines.append("## Held-Out Transition Retrieval")
    lines.append(_markdown_table(retrieval, ["variant", "hit@10", "mrr"]))
    (out_dir / "a_sufficiency_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", default="assist_09")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--full_save_dir", required=True)
    parser.add_argument("--no_a_save_dir", required=True)
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--edge_delete_fracs", type=float, nargs="+", default=[0.10, 0.25, 0.50])
    parser.add_argument("--edge_random_trials", type=int, default=5)
    parser.add_argument("--transition_hops", type=int, default=3)
    parser.add_argument("--transition_decay", type=float, default=0.70)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--bootstrap_iters", type=int, default=300)
    parser.add_argument("--bootstrap_metric", default="hit@10")
    parser.add_argument("--random_repeats", type=int, default=30)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_full, full_model = _load_checkpoint_model(Path(args.full_save_dir), device)
    info = checkpoint_full["info_dict"]

    subgroup = subgroup_monotonicity(args, out_dir, checkpoint_full, full_model, device)
    annotated = pd.read_csv(out_dir / "a_relevant_samples.csv")
    edge = edge_deletion_experiment(args, out_dir, checkpoint_full, full_model, annotated, device)
    retrieval, ci, random_df = transition_retrieval_with_ci(args, info, out_dir)
    write_summary(out_dir, subgroup, edge, retrieval, ci, random_df)
    print(json.dumps({"output_dir": str(out_dir), "rows": { "subgroup": len(subgroup), "edge": len(edge), "retrieval": len(retrieval)}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
