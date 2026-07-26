#!/usr/bin/env python
"""Probe cross-fitted residual concept relations on train/validation only.

The partial relation is the candidate.  The unadjusted relation is reported as
a diagnostic and is never used to select the candidate.  This tool does not
import or modify the trainable model stack and never opens ``test.csv``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.residual_relation import (  # noqa: E402
    RelationEstimate,
    QueryScores,
    add_student_excluded_item_residual,
    assign_student_folds,
    build_evidence_state,
    build_exercise_concepts,
    estimate_relations_from_residuals,
    parse_concepts,
    score_queries,
    sorted_id_map,
    student_excluded_item_expectation,
    topk_relation_estimate,
)
from src.config import DATASET_DEFAULTS  # noqa: E402


DEFAULT_DATASETS = "assist_17,junyi,nips34,ednet_kt1,moocradar,xes3g5m"


def _auc(labels: np.ndarray, logits: np.ndarray) -> float:
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, logits))


def _nll(labels: np.ndarray, logits: np.ndarray) -> float:
    if labels.size == 0:
        return float("nan")
    probability = np.empty_like(logits, dtype=np.float64)
    positive = logits >= 0.0
    probability[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    probability[~positive] = exp_logits / (1.0 + exp_logits)
    probability = np.clip(probability, 1e-8, 1.0 - 1e-8)
    return float(
        -np.mean(
            labels * np.log(probability)
            + (1.0 - labels) * np.log(1.0 - probability)
        )
    )


def _score_metrics(scores: QueryScores, prefix: str) -> Dict[str, float]:
    result: Dict[str, float] = {}
    buckets = {
        "all": np.ones(len(scores.labels), dtype=bool),
        "n0": scores.target_count == 0.0,
        "nlt3": scores.target_count < 3.0,
    }
    for bucket, mask in buckets.items():
        labels = scores.labels[mask]
        base = scores.base_logit[mask]
        raw = scores.raw_logit[mask]
        partial = scores.partial_logit[mask]
        base_auc = _auc(labels, base)
        raw_auc = _auc(labels, raw)
        partial_auc = _auc(labels, partial)
        base_nll = _nll(labels, base)
        partial_nll = _nll(labels, partial)
        stem = f"{prefix}_{bucket}"
        result.update(
            {
                f"{stem}_rows": int(mask.sum()),
                f"{stem}_base_auc": base_auc,
                f"{stem}_raw_auc_diag": raw_auc,
                f"{stem}_partial_auc": partial_auc,
                f"{stem}_partial_auc_delta": partial_auc - base_auc,
                f"{stem}_base_nll": base_nll,
                f"{stem}_partial_nll": partial_nll,
                f"{stem}_partial_nll_delta": partial_nll - base_nll,
            }
        )
    return result


def _relative_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.linalg.norm(left - right)
        / max(float(np.linalg.norm(right)), 1e-12)
    )


def _relation_metrics(
    full: RelationEstimate,
    folds: Dict[int, RelationEstimate],
    min_pair_students: int,
) -> Dict[str, float]:
    concepts = int(full.partial.shape[0])
    possible_edges = max(1, concepts * (concepts - 1))
    raw_edges = int(np.count_nonzero(np.abs(full.raw) > 1e-12))
    partial_edges = int(np.count_nonzero(np.abs(full.partial) > 1e-12))
    partial_positive_edges = int(np.count_nonzero(full.partial > 1e-12))
    partial_negative_edges = int(np.count_nonzero(full.partial < -1e-12))
    offdiag = ~np.eye(concepts, dtype=bool)
    raw_supported = int(
        np.count_nonzero((full.raw_students >= min_pair_students) & offdiag)
    )
    partial_supported = int(
        np.count_nonzero((full.partial_students >= min_pair_students) & offdiag)
    )
    raw_fold_distance = [
        _relative_distance(relation.raw, full.raw) for relation in folds.values()
    ]
    partial_fold_distance = [
        _relative_distance(relation.partial, full.partial)
        for relation in folds.values()
    ]
    result = {
        "concepts": concepts,
        "full_raw_edges": raw_edges,
        "full_raw_density": raw_edges / possible_edges,
        "full_raw_pair_supported_edges": raw_supported,
        "full_raw_pair_supported_density": raw_supported / possible_edges,
        "full_partial_edges": partial_edges,
        "full_partial_density": partial_edges / possible_edges,
        "full_partial_positive_edges": partial_positive_edges,
        "full_partial_negative_edges": partial_negative_edges,
        "full_partial_pair_supported_edges": partial_supported,
        "full_partial_pair_supported_density": partial_supported / possible_edges,
        "full_raw_partial_relative_distance": _relative_distance(
            full.raw,
            full.partial,
        ),
        "fold_full_raw_distance_mean": float(np.mean(raw_fold_distance)),
        "fold_full_raw_distance_max": float(np.max(raw_fold_distance)),
        "fold_full_partial_distance_mean": float(np.mean(partial_fold_distance)),
        "fold_full_partial_distance_max": float(np.max(partial_fold_distance)),
    }
    full_edge_mask = np.abs(full.partial) > 1e-12
    if len(folds) == 5 and partial_edges:
        full_sign = np.sign(full.partial)
        same_sign_count = sum(
            (np.sign(folds[fold].partial) == full_sign) & full_edge_mask
            for fold in sorted(folds)
        )
        for threshold, suffix in ((3, "ge3of5"), (4, "ge4of5"), (5, "5of5")):
            stable_edges = int(np.count_nonzero(same_sign_count >= threshold))
            result[f"full_partial_sign_stable_{suffix}_edges"] = stable_edges
            result[f"full_partial_sign_stable_{suffix}_rate"] = (
                stable_edges / partial_edges
            )
    else:
        for suffix in ("ge3of5", "ge4of5", "5of5"):
            result[f"full_partial_sign_stable_{suffix}_edges"] = 0
            result[f"full_partial_sign_stable_{suffix}_rate"] = float("nan")
    return result


def _topk_relation_metrics(
    full: RelationEstimate,
) -> Dict[str, float]:
    concepts = int(full.partial.shape[0])
    possible_edges = max(1, concepts * (concepts - 1))
    raw_edges = int(np.count_nonzero(np.abs(full.raw) > 1e-12))
    partial_edges = int(np.count_nonzero(np.abs(full.partial) > 1e-12))
    return {
        "full_topk_raw_edges": raw_edges,
        "full_topk_raw_density": raw_edges / possible_edges,
        "full_topk_partial_edges": partial_edges,
        "full_topk_partial_density": partial_edges / possible_edges,
    }


def probe_dataset(
    data_dir: Path,
    *,
    folds: int = 5,
    seed: int = 42,
    min_pair_students: int = 20,
    debug_max_train_rows: int = 0,
    debug_max_valid_rows: int = 0,
) -> Dict[str, float]:
    """Run the OOF-train and full-relation validation probe for one dataset."""

    started = time.perf_counter()
    train = pd.read_csv(data_dir / "train.csv")
    valid = pd.read_csv(data_dir / "valid.csv")
    if debug_max_train_rows > 0:
        train = train.iloc[:debug_max_train_rows].copy()
    if debug_max_valid_rows > 0:
        valid = valid.iloc[:debug_max_valid_rows].copy()

    q_by_item = build_exercise_concepts(train)
    try:
        relation_topk = int(DATASET_DEFAULTS[data_dir.name]["graph_topk"])
    except KeyError as error:
        raise KeyError(
            f"{data_dir.name}: graph_topk is missing from DATASET_DEFAULTS"
        ) from error
    all_concepts = set()
    for concepts in q_by_item.values():
        all_concepts.update(concepts)
    concept_map = sorted_id_map(all_concepts)
    assignment = assign_student_folds(
        train["stu_id"].unique(),
        folds=folds,
        seed=seed,
    )

    print(
        f"[{data_dir.name}] residuals: rows={len(train)} "
        f"students={len(assignment)} concepts={len(concept_map)}",
        flush=True,
    )
    full_residuals = add_student_excluded_item_residual(
        train,
        exercise_concepts=q_by_item,
    )
    full = estimate_relations_from_residuals(
        full_residuals,
        concept_map,
        min_pair_students=min_pair_students,
    )

    fold_relations: Dict[int, RelationEstimate] = {}
    for fold in range(folds):
        fold_started = time.perf_counter()
        complement_students = {
            student for student, value in assignment.items() if value != fold
        }
        fold_train = train[train["stu_id"].isin(complement_students)]
        fold_residuals = add_student_excluded_item_residual(
            fold_train,
            exercise_concepts=q_by_item,
        )
        fold_relations[fold] = estimate_relations_from_residuals(
            fold_residuals,
            concept_map,
            min_pair_students=min_pair_students,
        )
        print(
            f"[{data_dir.name}] fold {fold + 1}/{folds}: "
            f"fit_students={len(complement_students)} "
            f"elapsed={time.perf_counter() - fold_started:.1f}s",
            flush=True,
        )

    topk_full = topk_relation_estimate(full, relation_topk)
    topk_fold_relations = {
        fold: topk_relation_estimate(relation, relation_topk)
        for fold, relation in fold_relations.items()
    }
    evidence = build_evidence_state(full_residuals, concept_map)
    train_scores = score_queries(
        full_residuals,
        evidence,
        concept_map,
        fold_relations,
        assignment,
        leave_one_out=True,
    )
    train_topk_scores = score_queries(
        full_residuals,
        evidence,
        concept_map,
        topk_fold_relations,
        assignment,
        leave_one_out=True,
    )

    seen_students = set(evidence.student_map)
    valid = valid[
        valid["stu_id"].isin(seen_students)
        & valid["exer_id"].isin(q_by_item)
    ].copy()
    valid["concepts"] = valid["exer_id"].map(q_by_item)
    valid = valid[
        valid["concepts"].map(lambda value: isinstance(value, tuple) and bool(value))
    ].reset_index(drop=True)
    if valid.empty:
        raise ValueError(f"{data_dir.name}: no train-seen validation row")
    valid["item_expectation"] = student_excluded_item_expectation(train, valid)
    valid_scores = score_queries(
        valid,
        evidence,
        concept_map,
        {0: full},
        {student: 0 for student in seen_students},
        leave_one_out=False,
    )
    valid_topk_scores = score_queries(
        valid,
        evidence,
        concept_map,
        {0: topk_full},
        {student: 0 for student in seen_students},
        leave_one_out=False,
    )

    result: Dict[str, float] = {
        "dataset": data_dir.name,
        "seed": seed,
        "folds": folds,
        "min_pair_students": min_pair_students,
        "relation_topk": relation_topk,
        "primary_base_definition": "item_logit+rate_evidence+residual_evidence",
        "train_rows": int(len(train)),
        "valid_seen_rows": int(len(valid)),
        "debug_max_train_rows": int(debug_max_train_rows),
        "debug_max_valid_rows": int(debug_max_valid_rows),
    }
    result.update(_score_metrics(train_scores, "train_oof"))
    result.update(_score_metrics(valid_scores, "valid"))
    result.update(_score_metrics(train_topk_scores, "train_oof_topk"))
    result.update(_score_metrics(valid_topk_scores, "valid_topk"))
    result.update(
        _relation_metrics(full, fold_relations, min_pair_students)
    )
    result.update(_topk_relation_metrics(topk_full))
    result["elapsed_seconds"] = float(time.perf_counter() - started)
    print(
        f"[{data_dir.name}] valid partial ΔAUC="
        f"{result['valid_all_partial_auc_delta']:+.6f} "
        f"ΔNLL={result['valid_all_partial_nll_delta']:+.6f} "
        f"elapsed={result['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return result


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=DEFAULT_DATASETS)
    parser.add_argument("--data_root", default="data")
    parser.add_argument(
        "--output_csv",
        default="results/residual_relation_probe_seed42.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--min_pair_students",
        type=int,
        default=20,
        help="Fixed minimum students supporting a source-target slope.",
    )
    parser.add_argument(
        "--debug_max_train_rows",
        type=int,
        default=0,
        help="First-N train rows for a local smoke only; zero uses the full train split.",
    )
    parser.add_argument(
        "--debug_max_valid_rows",
        type=int,
        default=0,
        help="First-N validation rows for a local smoke only; zero uses all validation rows.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    if args.min_pair_students < 2:
        raise ValueError("--min_pair_students must be at least 2")
    names = [name.strip() for name in args.datasets.split(",") if name.strip()]
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:
        data_dir = Path(args.data_root) / name
        if not (data_dir / "train.csv").is_file() or not (
            data_dir / "valid.csv"
        ).is_file():
            print(f"[{name}] missing train.csv or valid.csv; skipped", flush=True)
            continue
        rows.append(
            probe_dataset(
                data_dir,
                folds=args.folds,
                seed=args.seed,
                min_pair_students=args.min_pair_students,
                debug_max_train_rows=args.debug_max_train_rows,
                debug_max_valid_rows=args.debug_max_valid_rows,
            )
        )
        pd.DataFrame(rows).to_csv(output, index=False)
        print(f"[{name}] wrote {output}", flush=True)
    if not rows:
        raise RuntimeError("no dataset was evaluated")


if __name__ == "__main__":
    main()
