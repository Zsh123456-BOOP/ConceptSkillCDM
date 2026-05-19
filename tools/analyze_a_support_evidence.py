#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Analyze whether CRG evidence support behaves like a useful reachability map."""

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

from src.dataset import (  # noqa: E402
    CognitiveDiagnosisDataset,
    _parse_concept_seq,
    create_dataloaders,
    make_degree_random_prior,
    make_support_uniform_prior,
)
from src.experiment_utils import compute_metrics  # noqa: E402
from tools.analyze_ae_errors import _build_model, _resolve_data_dir  # noqa: E402


def _row_normalize(matrix: torch.Tensor) -> torch.Tensor:
    matrix = matrix.detach().float().clamp(min=0.0)
    row_sum = matrix.sum(dim=-1, keepdim=True)
    return torch.where(row_sum > 0, matrix / row_sum.clamp(min=1e-12), torch.zeros_like(matrix))


def _uniform_offdiag(num_concepts: int) -> torch.Tensor:
    if num_concepts <= 1:
        return torch.ones(num_concepts, num_concepts, dtype=torch.float32)
    prior = torch.ones(num_concepts, num_concepts, dtype=torch.float32) - torch.eye(num_concepts, dtype=torch.float32)
    return prior / prior.sum(dim=-1, keepdim=True).clamp(min=1e-12)


def _self_only(num_concepts: int) -> torch.Tensor:
    return torch.eye(num_concepts, dtype=torch.float32)


def _resolve_files(dataset_name: str, data_dir: Optional[str]) -> Tuple[str, str, str, str]:
    data = _resolve_data_dir(dataset_name, data_dir)
    return data, os.path.join(data, "train.csv"), os.path.join(data, "valid.csv"), os.path.join(data, "test.csv")


def _load_info(dataset_name: str, data_dir: Optional[str], batch_size: int, num_workers: int) -> Dict[str, Any]:
    _, train_file, valid_file, test_file = _resolve_files(dataset_name, data_dir)
    _, _, _, info = create_dataloaders(
        train_file=train_file,
        val_file=valid_file,
        test_file=test_file,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle_train=False,
        min_stu_interactions=15,
        min_exer_interactions=0,
        min_poison_count=0,
        dataset_name=dataset_name,
    )
    return info


def _build_prior_variants(info: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    item = info["item_prior_matrix"].detach().float()
    seq = info["sequence_prior_matrix"].detach().float()
    support_uniform, _ = make_support_uniform_prior(item, seq)
    degree_random, _ = make_degree_random_prior(item, seq)
    C = int(item.size(0))
    return {
        "A_fused_prior": _row_normalize(item + seq),
        "A_item_only": _row_normalize(item),
        "A_seq_only": _row_normalize(seq),
        "A_support_uniform": support_uniform,
        "A_degree_random": degree_random,
        "A_uniform_offdiag": _uniform_offdiag(C),
        "A_self_only": _self_only(C),
    }


def _ordered_student_groups(df: pd.DataFrame) -> Iterable[pd.DataFrame]:
    ordered = df.copy()
    ordered["_source_order"] = np.arange(len(ordered))
    order_cols = [col for col in ("timestamp", "time", "order_id", "original_row_id") if col in ordered.columns]
    ordered = ordered.sort_values(["stu_id", *order_cols, "_source_order"], kind="mergesort")
    for _, group in ordered.groupby("stu_id", sort=False):
        yield group


def _transition_pairs(df: pd.DataFrame, cpt_id_map: Mapping[int, int], max_hops: int = 3, decay: float = 0.70) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    max_hops = max(1, int(max_hops))
    decay = max(0.0, min(1.0, float(decay)))
    for stu_df in _ordered_student_groups(df):
        concept_sets: List[List[int]] = []
        for seq in stu_df["cpt_seq"].values:
            mapped = sorted({cpt_id_map[c] for c in _parse_concept_seq(seq) if c in cpt_id_map})
            if mapped:
                concept_sets.append(mapped)
        for idx, src_concepts in enumerate(concept_sets):
            for hop in range(1, max_hops + 1):
                j = idx + hop
                if j >= len(concept_sets):
                    break
                weight = decay ** (hop - 1)
                for src in src_concepts:
                    for dst in concept_sets[j]:
                        if src == dst:
                            continue
                        rows.append({"query_concept": int(dst), "support_concept": int(src), "hop": hop, "weight": weight})
    return pd.DataFrame(rows)


def _rank_metrics(prior: torch.Tensor, pairs: pd.DataFrame, ks: Sequence[int]) -> Dict[str, Any]:
    if pairs.empty:
        return {"pairs": 0}
    scores = prior.detach().cpu().float().numpy()
    max_k = max(int(k) for k in ks)
    hits = {int(k): [] for k in ks}
    ndcgs = {int(k): [] for k in ks}
    rr: List[float] = []
    weights: List[float] = []
    for row in pairs.itertuples(index=False):
        q = int(row.query_concept)
        target = int(row.support_concept)
        weight = float(row.weight)
        ranking = np.argsort(-scores[q])
        found = np.where(ranking == target)[0]
        rank = int(found[0]) + 1 if len(found) else scores.shape[1] + 1
        weights.append(weight)
        rr.append((1.0 / rank) * weight)
        for k in ks:
            hit = 1.0 if rank <= int(k) else 0.0
            hits[int(k)].append(hit * weight)
            ndcgs[int(k)].append((hit / math.log2(rank + 1)) * weight if hit else 0.0)
    total_w = max(1e-12, float(np.sum(weights)))
    out: Dict[str, Any] = {"pairs": int(len(pairs)), "weight_sum": total_w, "mrr": float(np.sum(rr) / total_w)}
    for k in ks:
        kk = int(k)
        out[f"hit@{kk}"] = float(np.sum(hits[kk]) / total_w)
        out[f"ndcg@{kk}"] = float(np.sum(ndcgs[kk]) / total_w)
    return out


def transition_retrieval(args: argparse.Namespace, info: Mapping[str, Any], out_dir: Path) -> pd.DataFrame:
    data_dir, _, valid_file, test_file = _resolve_files(args.dataset_name, args.data_dir)
    _ = data_dir
    eval_df = pd.concat([pd.read_csv(valid_file), pd.read_csv(test_file)], ignore_index=True)
    eval_df = eval_df[eval_df["stu_id"].isin(info["stu_id_map"].keys())].reset_index(drop=True)
    pairs = _transition_pairs(eval_df, info["cpt_id_map"], max_hops=args.transition_hops, decay=args.transition_decay)
    pairs.to_csv(out_dir / "a_heldout_transition_pairs.csv", index=False)
    rows = []
    for name, prior in _build_prior_variants(info).items():
        item = {"variant": name}
        item.update(_rank_metrics(prior, pairs, args.ks))
        rows.append(item)
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "a_transition_retrieval.csv", index=False)
    return result


def _parse_variant_dir(items: Optional[Sequence[str]]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for item in items or ():
        if "=" not in item:
            raise ValueError(f"--variant_dir expects NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        out[name.strip()] = Path(path.strip())
    return out


def _predict_checkpoint(save_dir: Path, dataset_name: str, data_dir: Optional[str], split: str, batch_size: int, num_workers: int, device: torch.device) -> pd.DataFrame:
    checkpoint = torch.load(save_dir / "best_model.pth", map_location=device, weights_only=False)
    info = checkpoint["info_dict"]
    model = _build_model(checkpoint, device)
    raw_dir = _resolve_data_dir(dataset_name, data_dir)
    raw = pd.read_csv(os.path.join(raw_dir, f"{split}.csv"))
    frame = raw[
        raw["stu_id"].isin(set(info["stu_id_map"].keys()))
        & raw["exer_id"].isin(set(info["exer_id_map"].keys()))
    ].reset_index(drop=True)
    dataset = CognitiveDiagnosisDataset(frame, info["stu_id_map"], info["exer_id_map"], info["cpt_id_map"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    probs: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    mapped_students: List[np.ndarray] = []
    mapped_exercises: List[np.ndarray] = []
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


def _safe_auc(labels: np.ndarray, probs: np.ndarray) -> Optional[float]:
    labels = labels.astype(np.float32)
    probs = probs.astype(np.float32)
    if len(np.unique(labels)) < 2:
        return None
    return float(compute_metrics(labels, (probs > 0.5).astype(np.float32), probs)["auc"])


def _student_history(train_df: pd.DataFrame, cpt_id_map: Mapping[int, int]) -> Tuple[Dict[Any, int], Dict[Any, set], np.ndarray]:
    counts = train_df.groupby("stu_id").size().to_dict()
    history: Dict[Any, set] = {}
    concept_freq = np.zeros(len(cpt_id_map), dtype=np.float64)
    for row in train_df.itertuples(index=False):
        mapped = {cpt_id_map[c] for c in _parse_concept_seq(getattr(row, "cpt_seq")) if c in cpt_id_map}
        history.setdefault(getattr(row, "stu_id"), set()).update(mapped)
        for c in mapped:
            concept_freq[c] += 1.0
    return counts, history, concept_freq


def subgroup_analysis(args: argparse.Namespace, info: Mapping[str, Any], out_dir: Path) -> Optional[pd.DataFrame]:
    variant_dirs = _parse_variant_dir(args.variant_dir)
    required = {"A_fused", "no_A"}
    if not required.issubset(set(variant_dirs)):
        return None
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    predictions: Dict[str, pd.DataFrame] = {}
    for name, save_dir in variant_dirs.items():
        predictions[name] = _predict_checkpoint(
            save_dir,
            args.dataset_name,
            args.data_dir,
            args.split,
            args.batch_size,
            args.num_workers,
            device,
        )
        predictions[name].rename(columns={"prob": f"prob_{name}"}, inplace=True)
    base = predictions["A_fused"]
    for name, frame in predictions.items():
        if name == "A_fused":
            continue
        base[f"prob_{name}"] = frame[f"prob_{name}"].to_numpy()

    _, train_file, _, _ = _resolve_files(args.dataset_name, args.data_dir)
    train_df = pd.read_csv(train_file)
    train_df = train_df[train_df["stu_id"].isin(info["stu_id_map"].keys())].reset_index(drop=True)
    stu_counts, stu_history, concept_freq = _student_history(train_df, info["cpt_id_map"])
    q_matrix = info["q_matrix"].detach().cpu()
    support_prior = _row_normalize(info["item_prior_matrix"] + info["sequence_prior_matrix"]).numpy()
    low_freq_cut = float(np.quantile(concept_freq[concept_freq > 0], 0.25)) if np.any(concept_freq > 0) else 0.0
    short_cut = float(np.quantile(list(stu_counts.values()), 0.25)) if stu_counts else 0.0

    support_masses = []
    min_concept_freqs = []
    concept_counts = []
    history_counts = []
    for row in base.itertuples(index=False):
        mapped_ex = int(getattr(row, "mapped_exercise_id"))
        mapped_concepts = torch.nonzero(q_matrix[mapped_ex] > 0, as_tuple=False).view(-1).cpu().numpy().astype(int).tolist()
        hist = stu_history.get(getattr(row, "stu_id"), set())
        masses = [float(support_prior[c, list(hist)].sum()) if hist else 0.0 for c in mapped_concepts]
        support_masses.append(float(np.mean(masses)) if masses else 0.0)
        min_concept_freqs.append(float(min([concept_freq[c] for c in mapped_concepts], default=0.0)))
        concept_counts.append(len(mapped_concepts))
        history_counts.append(int(stu_counts.get(getattr(row, "stu_id"), 0)))
    base["a_support_mass_to_history"] = support_masses
    base["a_min_concept_train_freq"] = min_concept_freqs
    base["concept_count"] = concept_counts
    base["student_train_count"] = history_counts
    high_support_cut = float(np.quantile(base["a_support_mass_to_history"], 0.75))
    base["group_all"] = True
    base["group_short_history"] = base["student_train_count"] <= short_cut
    base["group_low_freq_concept"] = base["a_min_concept_train_freq"] <= low_freq_cut
    base["group_graph_hits_history"] = base["a_support_mass_to_history"] > 0.0
    base["group_high_support_mass"] = base["a_support_mass_to_history"] >= high_support_cut
    base["group_multi_concept"] = base["concept_count"] >= 2

    rows: List[Dict[str, Any]] = []
    groups = [
        "group_all",
        "group_short_history",
        "group_low_freq_concept",
        "group_graph_hits_history",
        "group_high_support_mass",
        "group_multi_concept",
    ]
    labels = base["label"].to_numpy(dtype=np.float32)
    for group in groups:
        mask = base[group].to_numpy(dtype=bool)
        if not mask.any():
            continue
        row: Dict[str, Any] = {"group": group.replace("group_", ""), "n": int(mask.sum())}
        for name in variant_dirs:
            probs = base.loc[mask, f"prob_{name}"].to_numpy(dtype=np.float64).clip(1e-8, 1.0 - 1e-8)
            y = labels[mask].astype(np.float64)
            row[f"{name}_auc"] = _safe_auc(y.astype(np.float32), probs.astype(np.float32))
            row[f"{name}_bce"] = float(np.mean(-(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs))))
        if "no_A" in variant_dirs:
            for name in variant_dirs:
                if name != "no_A":
                    if row.get(f"{name}_auc") is not None and row.get("no_A_auc") is not None:
                        row[f"{name}_minus_no_A_auc"] = float(row[f"{name}_auc"] - row["no_A_auc"])
        rows.append(row)
    base.to_csv(out_dir / "a_subgroup_samples.csv", index=False)
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "a_subgroup_auc.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", default="assist_09")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--transition_hops", type=int, default=3)
    parser.add_argument("--transition_decay", type=float, default=0.70)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--variant_dir", action="append", default=[])
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = _load_info(args.dataset_name, args.data_dir, args.batch_size, args.num_workers)
    transition = transition_retrieval(args, info, out_dir)
    subgroup = subgroup_analysis(args, info, out_dir)
    best_hit_col = f"hit@{max(int(k) for k in args.ks)}"
    summary = {
        "dataset": args.dataset_name,
        f"transition_best_{best_hit_col}": transition.sort_values(best_hit_col, ascending=False).head(1).to_dict("records"),
        "has_subgroup": subgroup is not None,
    }
    (out_dir / "a_support_evidence_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
