#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Profile whether datasets can actually support interpretable CRG/LCRF mechanisms.

This is a fail-fast diagnostic.  It does not train the model.  It reuses the
same train-only data loader and prior construction path as training, then
summarizes whether the Concept Reachability Graph (CRG) and the
Learner-Conditioned Reachability Filter (LCRF) have usable signal on each
dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATASET_DEFAULTS  # noqa: E402
from src.dataset import create_dataloaders  # noqa: E402


DEFAULT_DATASETS = (
    "assist_09",
    "junyi",
    "assist_17",
    "frcsub",
    "nips34",
    "nips34_l3",
    "ednet_kt1",
    "assist_12",
    "assist_12_clean15_item50",
)


class _SilentLogger:
    def info(self, _msg: str) -> None:
        return


def _parse_csv_tokens(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _split_paths(data_dir: Path) -> Tuple[Path, Path, Path]:
    return data_dir / "train.csv", data_dir / "valid.csv", data_dir / "test.csv"


def _q_count_stats(q_matrix: torch.Tensor) -> Dict[str, float]:
    counts = q_matrix.detach().float().sum(dim=1).cpu().numpy()
    if counts.size == 0:
        return {
            "items": 0,
            "concepts_per_item_mean": 0.0,
            "concepts_per_item_max": 0.0,
            "multi_concept_item_rate": 0.0,
        }
    return {
        "items": int(counts.size),
        "concepts_per_item_mean": float(counts.mean()),
        "concepts_per_item_max": float(counts.max()),
        "multi_concept_item_rate": float((counts >= 2).mean()),
    }


def _row_normalize_offdiag(mat: torch.Tensor) -> torch.Tensor:
    mat = mat.detach().float().clone()
    if mat.dim() != 2 or mat.size(0) != mat.size(1):
        return torch.zeros_like(mat)
    if mat.size(0) <= 1:
        return torch.ones_like(mat)
    eye = torch.eye(mat.size(0), dtype=mat.dtype)
    mat = mat.clamp(min=0.0) * (1.0 - eye)
    row_sum = mat.sum(dim=1, keepdim=True)
    return torch.where(row_sum > 0, mat / row_sum.clamp(min=1e-12), torch.zeros_like(mat))


def _prior_l1_to_uniform(prior: torch.Tensor) -> float:
    prior = _row_normalize_offdiag(prior)
    c = int(prior.size(0))
    if c <= 1:
        return 0.0
    uniform = (1.0 - torch.eye(c, dtype=prior.dtype)) / float(c - 1)
    observed = prior.sum(dim=1, keepdim=True) > 0
    if not bool(observed.any()):
        return 0.0
    diff = (prior - uniform).abs().sum(dim=1) / 2.0
    return float(diff[observed.squeeze(1)].mean().item())


def _prior_top1_margin(prior: torch.Tensor) -> float:
    prior = _row_normalize_offdiag(prior)
    c = int(prior.size(0))
    if c <= 2:
        return 0.0
    row_has = prior.sum(dim=1) > 0
    if not bool(row_has.any()):
        return 0.0
    vals = torch.topk(prior[row_has], k=2, dim=1).values
    return float((vals[:, 0] - vals[:, 1]).mean().item())


def _build_student_concept_counts(train_dataset, q_matrix: torch.Tensor, num_students: int) -> torch.Tensor:
    q_matrix = q_matrix.detach().float().cpu()
    counts = torch.zeros(num_students, q_matrix.size(1), dtype=torch.float32)
    student_ids = train_dataset.student_ids.detach().long().cpu()
    exercise_ids = train_dataset.exercise_ids.detach().long().cpu()
    chunk = 65536
    for start in range(0, int(student_ids.numel()), chunk):
        end = min(start + chunk, int(student_ids.numel()))
        counts.index_add_(0, student_ids[start:end], q_matrix[exercise_ids[start:end]])
    return counts


def _student_history_stats(train_dataset, num_students: int) -> Dict[str, float]:
    if len(train_dataset) == 0 or num_students <= 0:
        return {
            "student_train_count_mean": 0.0,
            "student_train_count_p25": 0.0,
            "student_train_count_median": 0.0,
            "student_train_count_p75": 0.0,
            "student_train_count_max": 0.0,
        }
    student_ids = train_dataset.student_ids.detach().long().cpu().numpy()
    counts = np.bincount(student_ids, minlength=int(num_students)).astype(np.float64)
    observed = counts[counts > 0]
    if observed.size == 0:
        observed = counts
    return {
        "student_train_count_mean": float(observed.mean()),
        "student_train_count_p25": float(np.quantile(observed, 0.25)),
        "student_train_count_median": float(np.quantile(observed, 0.50)),
        "student_train_count_p75": float(np.quantile(observed, 0.75)),
        "student_train_count_max": float(observed.max()),
    }


def _coverage_for_split(dataset, q_matrix: torch.Tensor, counts: torch.Tensor, neighbor_prior: torch.Tensor) -> Dict[str, float]:
    if len(dataset) == 0:
        return {
            "rows": 0,
            "e_exact_query_coverage": 0.0,
            "e_neighbor_query_coverage": 0.0,
            "query_count_mean": 0.0,
            "neighbor_count_mean": 0.0,
        }
    student_ids = dataset.student_ids.detach().long().cpu()
    exercise_ids = dataset.exercise_ids.detach().long().cpu()
    q = q_matrix.detach().float().cpu()[exercise_ids]
    query_weight = q / q.sum(dim=1, keepdim=True).clamp(min=1.0)
    row_counts = counts[student_ids]
    query_count = (query_weight * row_counts).sum(dim=1)

    graph_weight = query_weight.matmul(neighbor_prior.detach().float().cpu())
    neighbor_count = (graph_weight * row_counts).sum(dim=1)
    direct_seen = query_count > 0
    neighbor_seen = neighbor_count > 0
    bridge_only = (~direct_seen) & neighbor_seen
    return {
        "rows": int(len(dataset)),
        "e_exact_query_coverage": float(direct_seen.float().mean().item()),
        "e_neighbor_query_coverage": float(neighbor_seen.float().mean().item()),
        "e_direct_unseen_rate": float((~direct_seen).float().mean().item()),
        "e_bridge_only_rate": float(bridge_only.float().mean().item()),
        "query_count_mean": float(query_count.float().mean().item()),
        "neighbor_count_mean": float(neighbor_count.float().mean().item()),
    }


def _load_dataset_profile(dataset: str, data_root: Path) -> Dict[str, Any]:
    cfg = dict(DATASET_DEFAULTS.get(dataset, {}))
    data_dir = Path(cfg.get("data_dir") or data_root / dataset)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    train, valid, test = _split_paths(data_dir)
    row: Dict[str, Any] = {"dataset": dataset, "data_dir": str(data_dir)}
    missing = [str(p.name) for p in (train, valid, test) if not p.exists()]
    if missing:
        row.update({"status": "missing_split", "missing": ",".join(missing)})
        return row

    train_loader, valid_loader, test_loader, info = create_dataloaders(
        str(train),
        str(valid),
        str(test),
        batch_size=int(cfg.get("batch_size", 512)),
        num_workers=0,
        shuffle_train=False,
        min_stu_interactions=int(cfg.get("min_stu_interactions", 15)),
        min_exer_interactions=int(cfg.get("min_exer_interactions", 0)),
        min_poison_count=int(cfg.get("min_poison_count", 0)),
        logger=_SilentLogger(),
        dataset_name=dataset,
    )

    q_matrix = info["q_matrix"].detach().float().cpu()
    item_prior = info["item_prior_matrix"].detach().float().cpu()
    seq_prior = info["sequence_prior_matrix"].detach().float().cpu()
    fused_prior = _row_normalize_offdiag(item_prior + seq_prior)
    counts = _build_student_concept_counts(train_loader.dataset, q_matrix, int(info["num_students"]))
    history_stats = _student_history_stats(train_loader.dataset, int(info["num_students"]))
    q_stats = _q_count_stats(q_matrix)
    prior_stats = dict(info.get("graph_prior_stats", {}))
    valid_cov = _coverage_for_split(valid_loader.dataset, q_matrix, counts, fused_prior)
    test_cov = _coverage_for_split(test_loader.dataset, q_matrix, counts, fused_prior)

    row.update(
        {
            "status": "ok",
            "train_rows": int(info["train_size"]),
            "valid_rows": int(info["val_size"]),
            "test_rows": int(info["test_size"]),
            "students": int(info["num_students"]),
            "items": int(info["num_exercises"]),
            "concepts": int(info["num_concepts"]),
            "valid_seen_coverage": float(info.get("val_seen_coverage", 0.0)),
            "test_seen_coverage": float(info.get("test_seen_coverage", 0.0)),
            **q_stats,
            "item_edges": float(prior_stats.get("item_observed_edge_count", 0.0)),
            "item_density": float(prior_stats.get("item_prior_density", 0.0)),
            "item_entropy": float(prior_stats.get("item_prior_entropy", 0.0)),
            "seq_edges": float(prior_stats.get("sequence_observed_edge_count", 0.0)),
            "seq_density": float(prior_stats.get("sequence_prior_density", 0.0)),
            "seq_entropy": float(prior_stats.get("sequence_prior_entropy", 0.0)),
            "seq_weighted_mass": float(prior_stats.get("seq_student_weighted_mass", 0.0)),
            **history_stats,
            "a_fused_l1_to_uniform": _prior_l1_to_uniform(fused_prior),
            "a_fused_top1_margin": _prior_top1_margin(fused_prior),
            "valid_e_exact_coverage": valid_cov["e_exact_query_coverage"],
            "valid_e_neighbor_coverage": valid_cov["e_neighbor_query_coverage"],
            "valid_e_direct_unseen_rate": valid_cov["e_direct_unseen_rate"],
            "valid_e_bridge_only_rate": valid_cov["e_bridge_only_rate"],
            "valid_query_count_mean": valid_cov["query_count_mean"],
            "valid_neighbor_count_mean": valid_cov["neighbor_count_mean"],
            "test_e_exact_coverage": test_cov["e_exact_query_coverage"],
            "test_e_neighbor_coverage": test_cov["e_neighbor_query_coverage"],
            "test_e_direct_unseen_rate": test_cov["e_direct_unseen_rate"],
            "test_e_bridge_only_rate": test_cov["e_bridge_only_rate"],
            "test_query_count_mean": test_cov["query_count_mean"],
            "test_neighbor_count_mean": test_cov["neighbor_count_mean"],
        }
    )
    return row


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _print_compact(rows: List[Dict[str, Any]]) -> None:
    headers = [
        "dataset",
        "status",
        "train_rows",
        "concepts",
        "multi_concept_item_rate",
        "item_density",
        "seq_density",
        "student_train_count_median",
        "a_fused_l1_to_uniform",
        "test_e_exact_coverage",
        "test_e_neighbor_coverage",
        "test_e_bridge_only_rate",
    ]
    print(",".join(headers))
    for row in rows:
        values = []
        for key in headers:
            value = row.get(key, "")
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        print(",".join(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--data_root", default=str(ROOT / "data"))
    parser.add_argument("--out_csv", default=str(ROOT / "results" / "ae_data_readiness.csv"))
    parser.add_argument("--out_json", default=str(ROOT / "results" / "ae_data_readiness.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, Any]] = []
    for dataset in _parse_csv_tokens(args.datasets):
        try:
            rows.append(_load_dataset_profile(dataset, Path(args.data_root)))
        except Exception as exc:  # diagnostics should not hide other datasets
            rows.append({"dataset": dataset, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
    _write_csv(Path(args.out_csv), rows)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_compact(rows)
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
