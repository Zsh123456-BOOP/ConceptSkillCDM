#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Train-only statistical probe for A/E signal usefulness.

This script does not train the CD model.  It builds the same train-only data
statistics used by the interpretable roadmap/tutor modules and fits small
logistic probes to answer one question before any architecture change:

    Do A-route and E-local features contain incremental predictive signal?

If the probe cannot show incremental signal, adding more A/E code is unlikely to
help.  If the probe does show signal while the neural model does not, the issue
is in how the model consumes these signals.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import create_dataloaders
from src.trainer import _compute_train_stat_logits


def _to_numpy(x) -> np.ndarray:
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    matrix = np.maximum(matrix, 0.0)
    row_sum = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, np.maximum(row_sum, 1e-12), out=np.zeros_like(matrix), where=row_sum > 1e-12)


def _route_matrix(item_prior: np.ndarray, seq_prior: np.ndarray) -> np.ndarray:
    """A simple evidence route used only for diagnosis.

    It uses the same evidence families as A but avoids learned parameters:
    item cooccurrence + sequence transition + self retention.
    """
    c = item_prior.shape[0]
    fused = np.maximum(item_prior, 0.0) + np.maximum(seq_prior, 0.0) + np.eye(c, dtype=np.float32)
    return _row_normalize(fused)


def _query_weights(q_matrix: np.ndarray, exercise_ids: np.ndarray) -> np.ndarray:
    q = q_matrix[exercise_ids].astype(np.float32)
    denom = np.maximum(q.sum(axis=1, keepdims=True), 1.0)
    return q / denom


def _reliability(counts: np.ndarray, smoothing: float) -> np.ndarray:
    return counts / np.maximum(counts + float(smoothing), 1e-6)


def _build_split_features(
    *,
    split_name: str,
    dataset,
    q_matrix: np.ndarray,
    route: np.ndarray,
    student_logits: np.ndarray,
    exercise_logits: np.ndarray,
    concept_logits: np.ndarray,
    student_concept_logits: np.ndarray,
    recent_student_concept_logits: np.ndarray,
    student_concept_counts: np.ndarray,
    smoothing: float,
) -> Tuple[pd.DataFrame, np.ndarray]:
    sid = _to_numpy(dataset.student_ids).astype(np.int64)
    eid = _to_numpy(dataset.exercise_ids).astype(np.int64)
    y = _to_numpy(dataset.labels).astype(np.float32)

    qw = _query_weights(q_matrix, eid)
    rw = qw @ route
    route_delta = rw - qw
    off_mass = (rw * (1.0 - (qw > 0).astype(np.float32))).sum(axis=1)
    support_shift = 0.5 * np.abs(route_delta).sum(axis=1)

    sc = student_concept_logits[sid]
    recent = recent_student_concept_logits[sid]
    counts = student_concept_counts[sid]
    query_counts = (qw * counts).sum(axis=1)
    route_counts = (rw * counts).sum(axis=1)
    query_rel = _reliability(query_counts, smoothing)
    route_rel = _reliability(route_counts, smoothing)

    query_concept = qw @ concept_logits
    route_concept = rw @ concept_logits
    query_sc = (qw * sc).sum(axis=1)
    route_sc = (rw * sc).sum(axis=1)
    query_recent = (qw * recent).sum(axis=1)
    route_recent = (rw * recent).sum(axis=1)

    frame = pd.DataFrame(
        {
            "split": split_name,
            "student_prior": student_logits[sid],
            "exercise_prior": exercise_logits[eid],
            "query_concept_prior": query_concept,
            "A_route_concept_delta": route_concept - query_concept,
            "A_off_query_mass": off_mass,
            "A_support_shift": support_shift,
            "E_query_student_concept": query_sc,
            "E_query_recent": query_recent,
            "E_query_reliability": query_rel,
            "E_route_student_concept_delta": route_sc - query_sc,
            "E_route_recent_delta": route_recent - query_recent,
            "E_route_reliability_delta": route_rel - query_rel,
            "label": y,
        }
    )
    return frame, y


FEATURE_SETS: Dict[str, List[str]] = {
    "base_no_student": ["exercise_prior", "query_concept_prior"],
    "base_no_student_plus_E": [
        "exercise_prior",
        "query_concept_prior",
        "E_query_student_concept",
        "E_query_recent",
        "E_query_reliability",
    ],
    "base_no_student_plus_AE": [
        "exercise_prior",
        "query_concept_prior",
        "A_route_concept_delta",
        "A_off_query_mass",
        "A_support_shift",
        "E_query_student_concept",
        "E_query_recent",
        "E_query_reliability",
        "E_route_student_concept_delta",
        "E_route_recent_delta",
        "E_route_reliability_delta",
    ],
    "base_stat": ["student_prior", "exercise_prior", "query_concept_prior"],
    "base_plus_A": [
        "student_prior",
        "exercise_prior",
        "query_concept_prior",
        "A_route_concept_delta",
        "A_off_query_mass",
        "A_support_shift",
    ],
    "base_plus_E_current": [
        "student_prior",
        "exercise_prior",
        "query_concept_prior",
        "E_query_student_concept",
        "E_query_recent",
        "E_query_reliability",
    ],
    "base_plus_AE_route": [
        "student_prior",
        "exercise_prior",
        "query_concept_prior",
        "A_route_concept_delta",
        "A_off_query_mass",
        "A_support_shift",
        "E_query_student_concept",
        "E_query_recent",
        "E_query_reliability",
        "E_route_student_concept_delta",
        "E_route_recent_delta",
        "E_route_reliability_delta",
    ],
}


def _evaluate_feature_sets(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y_train = train_df["label"].to_numpy(dtype=np.float32)
    y_eval = eval_df["label"].to_numpy(dtype=np.float32)
    for name, cols in FEATURE_SETS.items():
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, solver="lbfgs"),
        )
        model.fit(train_df[cols].to_numpy(dtype=np.float32), y_train)
        prob = model.predict_proba(eval_df[cols].to_numpy(dtype=np.float32))[:, 1]
        rows.append(
            {
                "feature_set": name,
                "auc": float(roc_auc_score(y_eval, prob)),
                "bce": float(log_loss(y_eval, prob, labels=[0, 1])),
                "n_features": len(cols),
            }
        )
    out = pd.DataFrame(rows)
    base_auc = float(out.loc[out["feature_set"] == "base_stat", "auc"].iloc[0])
    base_bce = float(out.loc[out["feature_set"] == "base_stat", "bce"].iloc[0])
    out["auc_gain_over_base"] = out["auc"] - base_auc
    out["bce_gain_over_base"] = base_bce - out["bce"]
    return out


def _univariate_auc(frame: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    y = frame["label"].to_numpy(dtype=np.float32)
    rows = []
    for col in features:
        x = frame[col].to_numpy(dtype=np.float32)
        if np.nanstd(x) < 1e-8:
            auc = np.nan
            auc_abs = np.nan
        else:
            auc = float(roc_auc_score(y, x))
            auc_abs = max(auc, 1.0 - auc)
        rows.append({"feature": col, "auc_signed": auc, "auc_best_direction": auc_abs, "mean": float(np.nanmean(x)), "std": float(np.nanstd(x))})
    return pd.DataFrame(rows).sort_values("auc_best_direction", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="assist_09")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--min_stu_interactions", type=int, default=15)
    parser.add_argument("--min_exer_interactions", type=int, default=0)
    parser.add_argument("--min_poison_count", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--output_dir", default="results/local_diagnostics/ae_signal_probe")
    args = parser.parse_args()

    root = Path(args.data_root) / args.dataset
    out_dir = Path(args.output_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, info = create_dataloaders(
        str(root / "train.csv"),
        str(root / "valid.csv"),
        str(root / "test.csv"),
        batch_size=args.batch_size,
        num_workers=0,
        shuffle_train=False,
        min_stu_interactions=args.min_stu_interactions,
        min_exer_interactions=args.min_exer_interactions,
        min_poison_count=args.min_poison_count,
        dataset_name=args.dataset,
        graph_prior_mode="evidence",
    )
    (
        student_logits,
        exercise_logits,
        concept_logits,
        student_concept_logits,
        recent_student_concept_logits,
        _global_rate,
        count_features,
    ) = _compute_train_stat_logits(
        train_loader,
        info["q_matrix"],
        num_students=info["num_students"],
        num_exercises=info["num_exercises"],
        num_concepts=info["num_concepts"],
    )

    q_matrix = _to_numpy(info["q_matrix"])
    route = _route_matrix(_to_numpy(info["item_prior_matrix"]), _to_numpy(info["sequence_prior_matrix"]))
    common = dict(
        q_matrix=q_matrix,
        route=route,
        student_logits=_to_numpy(student_logits),
        exercise_logits=_to_numpy(exercise_logits),
        concept_logits=_to_numpy(concept_logits),
        student_concept_logits=_to_numpy(student_concept_logits),
        recent_student_concept_logits=_to_numpy(recent_student_concept_logits),
        student_concept_counts=_to_numpy(count_features["student_concept_observed"]),
        smoothing=8.0,
    )
    train_df, _ = _build_split_features(split_name="train", dataset=train_loader.dataset, **common)
    val_df, _ = _build_split_features(split_name="valid", dataset=val_loader.dataset, **common)
    test_df, _ = _build_split_features(split_name="test", dataset=test_loader.dataset, **common)

    val_results = _evaluate_feature_sets(train_df, val_df)
    test_results = _evaluate_feature_sets(train_df, test_df)
    val_results.insert(0, "eval_split", "valid")
    test_results.insert(0, "eval_split", "test")
    all_results = pd.concat([val_results, test_results], ignore_index=True)
    all_results.to_csv(out_dir / "feature_set_auc.csv", index=False)

    feature_cols = sorted({c for cols in FEATURE_SETS.values() for c in cols})
    uni_valid = _univariate_auc(val_df, feature_cols)
    uni_test = _univariate_auc(test_df, feature_cols)
    uni_valid.insert(0, "eval_split", "valid")
    uni_test.insert(0, "eval_split", "test")
    uni = pd.concat([uni_valid, uni_test], ignore_index=True)
    uni.to_csv(out_dir / "univariate_signal_auc.csv", index=False)

    summary = {
        "dataset": args.dataset,
        "num_students": int(info["num_students"]),
        "num_exercises": int(info["num_exercises"]),
        "num_concepts": int(info["num_concepts"]),
        "train_size": int(info["train_size"]),
        "val_size": int(info["val_size"]),
        "test_size": int(info["test_size"]),
        "graph_prior_stats": info.get("graph_prior_stats", {}),
        "result_csv": str(out_dir / "feature_set_auc.csv"),
        "univariate_csv": str(out_dir / "univariate_signal_auc.csv"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(all_results.to_string(index=False))
    print("\nTop univariate test signals:")
    print(uni_test.head(12).to_string(index=False))
    print(f"\nWrote: {out_dir}")


if __name__ == "__main__":
    main()
