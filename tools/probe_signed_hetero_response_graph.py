#!/usr/bin/env python
"""Validation-only frozen probe for a query-excluded signed response graph.

This script deliberately does not modify or train the production model.  It
freezes one selected Graph-IRT checkpoint, then asks whether a typed
student-item-concept graph carries validation signal beyond the frozen logit.

For every training query, all rows sharing its ``(student, item)`` pair are
held out together in one of five label-independent folds.  Right and wrong
student-item edges are built only from the other folds; item-concept edges come
from the checkpoint Q matrix.  Validation queries use the complete training
graph and never contribute response edges.

Each graph produces the normalized three-hop student-to-item score

    B @ (B.T @ B + Q @ Q.T),

where B and Q are the response and item-concept blocks after symmetric degree
normalization in the combined tripartite graph.  The score contains indirect
student-item-student-item and student-item-concept-item paths.  Because the
queried pair is absent from its fold graph, it cannot copy its own response.

Only two non-negative coefficients are fitted on out-of-fold training rows:

    frozen_logit + beta_right * right_score - beta_wrong * wrong_score.

Unsigned and within-student sign-shuffle controls use the same folds and
fitting protocol.  There is intentionally no test-split option.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CognitiveDiagnosisDataset  # noqa: E402
from src.trainer import (  # noqa: E402
    _build_model,
    _require_graph_irt_checkpoint,
    _strip_module_prefix,
)


FOLDS = 5
L2_WEIGHT = 1e-4
OPTIMIZER_MAX_ITER = 200
EPS = 1e-12


def _resolve_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(labels: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    labels_f32 = np.asarray(labels, dtype=np.float32)
    logits_f32 = np.asarray(logits, dtype=np.float32)
    probabilities = torch.sigmoid(torch.from_numpy(logits_f32)).numpy()
    auc = (
        float(roc_auc_score(labels_f32, probabilities))
        if np.unique(labels_f32).size >= 2
        else float("nan")
    )
    labels_f64 = labels_f32.astype(np.float64)
    logits_f64 = logits_f32.astype(np.float64)
    bce = float(
        np.mean(np.logaddexp(0.0, logits_f64) - labels_f64 * logits_f64)
    )
    rmse = float(np.sqrt(np.mean(np.square(labels_f32 - probabilities))))
    return {"auc": auc, "bce_loss": bce, "rmse": rmse}


def _filter_and_map(
    frame: pd.DataFrame,
    student_map: Mapping[object, int],
    exercise_map: Mapping[object, int],
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    filtered = frame[
        frame["stu_id"].isin(student_map)
        & frame["exer_id"].isin(exercise_map)
    ].copy()
    filtered = filtered.reset_index(drop=True)
    students = filtered["stu_id"].map(student_map).to_numpy(dtype=np.int64)
    exercises = filtered["exer_id"].map(exercise_map).to_numpy(dtype=np.int64)
    labels = filtered["label"].to_numpy(dtype=np.float64)
    if len(filtered) == 0:
        raise ValueError("no rows remain after applying checkpoint ID mappings")
    if not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError("response labels must be binary")
    return filtered, students, exercises, labels


def _pair_folds(
    students: np.ndarray,
    exercises: np.ndarray,
    num_exercises: int,
    *,
    folds: int,
    seed: int,
) -> Tuple[np.ndarray, int]:
    if folds < 2:
        raise ValueError("at least two folds are required")
    pair_keys = students * int(num_exercises) + exercises
    unique_keys, inverse = np.unique(pair_keys, return_inverse=True)
    if unique_keys.size < folds:
        raise ValueError(
            f"cannot assign {unique_keys.size} unique pairs to {folds} folds"
        )
    rng = np.random.RandomState(seed)
    permutation = rng.permutation(unique_keys.size)
    unique_fold = np.empty(unique_keys.size, dtype=np.int16)
    unique_fold[permutation] = (
        np.arange(unique_keys.size, dtype=np.int64) % int(folds)
    )
    row_fold = unique_fold[inverse]
    for fold in range(folds):
        held_out = pair_keys[row_fold == fold]
        included = pair_keys[row_fold != fold]
        if np.intersect1d(
            np.unique(held_out),
            np.unique(included),
            assume_unique=True,
        ).size:
            raise RuntimeError("a student-item pair was split across folds")
    return row_fold, int(unique_keys.size)


def _shuffle_labels_within_students(
    students: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Shuffle only labels present in the supplied graph-building subset."""
    shuffled = np.asarray(labels, dtype=np.float64).copy()
    order = np.argsort(students, kind="stable")
    sorted_students = students[order]
    boundaries = np.flatnonzero(
        np.r_[True, sorted_students[1:] != sorted_students[:-1], True]
    )
    rng = np.random.RandomState(seed)
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        indices = order[start:stop]
        shuffled[indices] = labels[indices][rng.permutation(len(indices))]
        if not math.isclose(
            float(shuffled[indices].sum()),
            float(labels[indices].sum()),
            abs_tol=1e-12,
        ):
            raise RuntimeError("within-student shuffle changed a label total")
    return shuffled


def _pair_response_matrices(
    students: np.ndarray,
    exercises: np.ndarray,
    labels: np.ndarray,
    shape: Tuple[int, int],
) -> Tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    ones = np.ones(len(students), dtype=np.float64)
    attempts = sp.coo_matrix(
        (ones, (students, exercises)),
        shape=shape,
        dtype=np.float64,
    ).tocsr()
    correct = sp.coo_matrix(
        (labels, (students, exercises)),
        shape=shape,
        dtype=np.float64,
    ).tocsr()
    reciprocal = attempts.copy()
    reciprocal.data = 1.0 / reciprocal.data
    right = correct.multiply(reciprocal).tocsr()
    wrong = (attempts - correct).multiply(reciprocal).tocsr()
    unsigned = attempts.copy()
    unsigned.data.fill(1.0)
    for matrix in (right, wrong, unsigned):
        matrix.eliminate_zeros()
        if matrix.nnz and (
            float(matrix.data.min()) < -EPS
            or float(matrix.data.max()) > 1.0 + EPS
        ):
            raise RuntimeError("pair response weights left the interval [0, 1]")
    return right, wrong, unsigned


def _inverse_sqrt(values: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float64)
    positive = values > 0.0
    result[positive] = np.power(values[positive], -0.5)
    return result


def _three_hop_scores(
    response: sp.csr_matrix,
    q_matrix: sp.csr_matrix,
    query_students: np.ndarray,
    query_exercises: np.ndarray,
) -> np.ndarray:
    """Return the normalized A^3 student-item entries for requested pairs."""
    if response.shape[1] != q_matrix.shape[0]:
        raise ValueError("response item count and Q-matrix item count disagree")
    student_degree = np.asarray(response.sum(axis=1)).reshape(-1)
    item_degree = (
        np.asarray(response.sum(axis=0)).reshape(-1)
        + np.asarray(q_matrix.sum(axis=1)).reshape(-1)
    )
    concept_degree = np.asarray(q_matrix.sum(axis=0)).reshape(-1)

    response_norm = sp.diags(_inverse_sqrt(student_degree)).dot(response)
    response_norm = response_norm.dot(
        sp.diags(_inverse_sqrt(item_degree))
    ).tocsr()
    q_norm = sp.diags(_inverse_sqrt(item_degree)).dot(q_matrix)
    q_norm = q_norm.dot(sp.diags(_inverse_sqrt(concept_degree))).tocsr()

    item_paths = (
        response_norm.T.dot(response_norm) + q_norm.dot(q_norm.T)
    ).tocsr()
    three_hop = response_norm.dot(item_paths).tocsr()
    scores = np.asarray(
        three_hop[query_students, query_exercises]
    ).reshape(-1)
    if not np.isfinite(scores).all():
        raise RuntimeError("three-hop graph scores contain non-finite values")
    return scores.astype(np.float64, copy=False)


def _signed_scores(
    students: np.ndarray,
    exercises: np.ndarray,
    labels: np.ndarray,
    *,
    num_students: int,
    num_exercises: int,
    q_matrix: sp.csr_matrix,
    query_students: np.ndarray,
    query_exercises: np.ndarray,
    include_unsigned: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    right, wrong, unsigned = _pair_response_matrices(
        students,
        exercises,
        labels,
        (num_students, num_exercises),
    )
    right_scores = _three_hop_scores(
        right,
        q_matrix,
        query_students,
        query_exercises,
    )
    wrong_scores = _three_hop_scores(
        wrong,
        q_matrix,
        query_students,
        query_exercises,
    )
    unsigned_scores = (
        _three_hop_scores(
            unsigned,
            q_matrix,
            query_students,
            query_exercises,
        )
        if include_unsigned
        else np.empty(0, dtype=np.float64)
    )
    return right_scores, wrong_scores, unsigned_scores


def _cross_fitted_graph_features(
    students: np.ndarray,
    exercises: np.ndarray,
    labels: np.ndarray,
    row_fold: np.ndarray,
    *,
    num_students: int,
    num_exercises: int,
    q_matrix: sp.csr_matrix,
    folds: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    features = {
        "right": np.zeros(len(labels), dtype=np.float64),
        "wrong": np.zeros(len(labels), dtype=np.float64),
        "unsigned": np.zeros(len(labels), dtype=np.float64),
        "shuffle_right": np.zeros(len(labels), dtype=np.float64),
        "shuffle_wrong": np.zeros(len(labels), dtype=np.float64),
    }
    pair_keys = students * int(num_exercises) + exercises
    for fold in range(folds):
        fit_mask = row_fold != fold
        query_mask = ~fit_mask
        fit_students = students[fit_mask]
        fit_exercises = exercises[fit_mask]
        fit_labels = labels[fit_mask]
        query_students = students[query_mask]
        query_exercises = exercises[query_mask]

        fit_pairs = np.unique(
            fit_students * int(num_exercises) + fit_exercises
        )
        query_pairs = np.unique(pair_keys[query_mask])
        if np.intersect1d(
            fit_pairs,
            query_pairs,
            assume_unique=True,
        ).size:
            raise RuntimeError(
                f"fold {fold} graph contains a held-out student-item pair"
            )

        right, wrong, unsigned = _signed_scores(
            fit_students,
            fit_exercises,
            fit_labels,
            num_students=num_students,
            num_exercises=num_exercises,
            q_matrix=q_matrix,
            query_students=query_students,
            query_exercises=query_exercises,
        )
        shuffled_labels = _shuffle_labels_within_students(
            fit_students,
            fit_labels,
            seed=seed + 1009 * (fold + 1),
        )
        shuffle_right, shuffle_wrong, _ = _signed_scores(
            fit_students,
            fit_exercises,
            shuffled_labels,
            num_students=num_students,
            num_exercises=num_exercises,
            q_matrix=q_matrix,
            query_students=query_students,
            query_exercises=query_exercises,
            include_unsigned=False,
        )
        features["right"][query_mask] = right
        features["wrong"][query_mask] = wrong
        features["unsigned"][query_mask] = unsigned
        features["shuffle_right"][query_mask] = shuffle_right
        features["shuffle_wrong"][query_mask] = shuffle_wrong
        print(
            f"graph cross-fit fold {fold + 1}/{folds}: "
            f"fit_rows={int(fit_mask.sum())} query_rows={int(query_mask.sum())}",
            flush=True,
        )
    return features


def _validation_graph_features(
    students: np.ndarray,
    exercises: np.ndarray,
    labels: np.ndarray,
    *,
    num_students: int,
    num_exercises: int,
    q_matrix: sp.csr_matrix,
    query_students: np.ndarray,
    query_exercises: np.ndarray,
    seed: int,
) -> Dict[str, np.ndarray]:
    right, wrong, unsigned = _signed_scores(
        students,
        exercises,
        labels,
        num_students=num_students,
        num_exercises=num_exercises,
        q_matrix=q_matrix,
        query_students=query_students,
        query_exercises=query_exercises,
    )
    shuffled_labels = _shuffle_labels_within_students(
        students,
        labels,
        seed=seed + 7919,
    )
    shuffle_right, shuffle_wrong, _ = _signed_scores(
        students,
        exercises,
        shuffled_labels,
        num_students=num_students,
        num_exercises=num_exercises,
        q_matrix=q_matrix,
        query_students=query_students,
        query_exercises=query_exercises,
        include_unsigned=False,
    )
    return {
        "right": right,
        "wrong": wrong,
        "unsigned": unsigned,
        "shuffle_right": shuffle_right,
        "shuffle_wrong": shuffle_wrong,
    }


def _rms_scale(train_feature: np.ndarray) -> float:
    scale = float(np.sqrt(np.mean(np.square(train_feature))))
    return max(scale, EPS)


def _fit_nonnegative_offset(
    base_logits: np.ndarray,
    labels: np.ndarray,
    feature_columns: Sequence[np.ndarray],
    signs: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    if len(feature_columns) != len(signs) or not feature_columns:
        raise ValueError("features and signs must have the same non-zero length")
    scales = np.asarray(
        [_rms_scale(column) for column in feature_columns],
        dtype=np.float64,
    )
    design = np.column_stack(
        [
            float(sign) * np.asarray(column, dtype=np.float64) / scale
            for column, sign, scale in zip(feature_columns, signs, scales)
        ]
    )
    base = np.asarray(base_logits, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)

    def objective(beta: np.ndarray) -> Tuple[float, np.ndarray]:
        logits = base + design.dot(beta)
        loss = np.mean(np.logaddexp(0.0, logits) - targets * logits)
        loss += L2_WEIGHT * float(np.square(beta).sum())
        residual = expit(logits) - targets
        gradient = design.T.dot(residual) / len(targets)
        gradient += 2.0 * L2_WEIGHT * beta
        return float(loss), gradient

    result = minimize(
        objective,
        np.zeros(design.shape[1], dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, None)] * design.shape[1],
        options={"maxiter": OPTIMIZER_MAX_ITER, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"non-negative probe fit failed: {result.message}")
    return result.x.astype(np.float64), scales


def _compose_offset_logits(
    base_logits: np.ndarray,
    feature_columns: Sequence[np.ndarray],
    signs: Sequence[float],
    beta: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    logits = np.asarray(base_logits, dtype=np.float64).copy()
    for column, sign, weight, scale in zip(
        feature_columns,
        signs,
        beta,
        scales,
    ):
        logits += (
            float(sign)
            * float(weight)
            * np.asarray(column, dtype=np.float64)
            / float(scale)
        )
    return logits


def _collect_frozen_logits(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    student_map: Mapping[object, int],
    exercise_map: Mapping[object, int],
    concept_map: Mapping[object, int],
    *,
    batch_size: int,
    device: torch.device,
    exclude_training_labels: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    dataset = CognitiveDiagnosisDataset(
        frame,
        student_map,
        exercise_map,
        concept_map,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    labels_parts = []
    logits_parts = []
    with torch.no_grad():
        for student_ids, exercise_ids, labels in loader:
            student_ids = student_ids.to(device)
            exercise_ids = exercise_ids.to(device)
            labels_device = labels.to(device)
            logits = model(
                student_ids,
                exercise_ids,
                outcome_to_exclude=(
                    labels_device if exclude_training_labels else None
                ),
                return_logits=True,
            )
            labels_parts.append(labels.reshape(-1).cpu())
            logits_parts.append(logits.reshape(-1).cpu())
    return (
        torch.cat(labels_parts).numpy().astype(np.float32),
        torch.cat(logits_parts).numpy().astype(np.float32),
    )


def _evaluate_candidate(
    name: str,
    base_train: np.ndarray,
    labels_train: np.ndarray,
    base_valid: np.ndarray,
    labels_valid: np.ndarray,
    train_columns: Sequence[np.ndarray],
    valid_columns: Sequence[np.ndarray],
    signs: Sequence[float],
) -> Dict[str, object]:
    beta, scales = _fit_nonnegative_offset(
        base_train,
        labels_train,
        train_columns,
        signs,
    )
    valid_logits = _compose_offset_logits(
        base_valid,
        valid_columns,
        signs,
        beta,
        scales,
    )
    return {
        "name": name,
        "beta": [float(value) for value in beta],
        "rms_scale": [float(value) for value in scales],
        "metrics": _metrics(labels_valid, valid_logits),
    }


def run_probe(
    checkpoint_dir: Path,
    output_json: Path,
    *,
    seed: int,
    device: torch.device,
) -> Dict[str, object]:
    checkpoint_path = checkpoint_dir / "best_model.pth"
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    _require_graph_irt_checkpoint(checkpoint, str(checkpoint_path))
    loaded_args = checkpoint["args"]
    info_dict = checkpoint["info_dict"]
    if str(loaded_args.get("model_variant", "")) != "full":
        raise ValueError("the signed graph probe requires a full checkpoint")

    model = _build_model(loaded_args, info_dict, device)
    model.load_state_dict(
        _strip_module_prefix(checkpoint["model_state_dict"]),
        strict=True,
    )
    model.eval()

    data_dir = _resolve_path(str(loaded_args["data_dir"]))
    train_path = data_dir / "train.csv"
    valid_path = data_dir / "valid.csv"
    train_source = pd.read_csv(train_path)
    valid_source = pd.read_csv(valid_path)
    student_map = info_dict["stu_id_map"]
    exercise_map = info_dict["exer_id_map"]
    concept_map = info_dict["cpt_id_map"]
    train, train_students, train_exercises, train_labels = _filter_and_map(
        train_source,
        student_map,
        exercise_map,
    )
    valid, valid_students, valid_exercises, valid_labels = _filter_and_map(
        valid_source,
        student_map,
        exercise_map,
    )
    num_students = len(student_map)
    num_exercises = len(exercise_map)
    q_tensor = info_dict["q_matrix"].detach().cpu().float()
    expected_q = (num_exercises, len(concept_map))
    if tuple(q_tensor.shape) != expected_q:
        raise ValueError(
            f"checkpoint Q shape must be {expected_q}, got {tuple(q_tensor.shape)}"
        )
    q_matrix = sp.csr_matrix(q_tensor.numpy().astype(np.float64))

    row_fold, unique_pairs = _pair_folds(
        train_students,
        train_exercises,
        num_exercises,
        folds=FOLDS,
        seed=seed,
    )
    print(
        f"{loaded_args.get('dataset_name', data_dir.name)}: "
        f"train_rows={len(train)} valid_rows={len(valid)} "
        f"unique_pairs={unique_pairs}",
        flush=True,
    )
    train_features = _cross_fitted_graph_features(
        train_students,
        train_exercises,
        train_labels,
        row_fold,
        num_students=num_students,
        num_exercises=num_exercises,
        q_matrix=q_matrix,
        folds=FOLDS,
        seed=seed,
    )
    print("cross-fitted graph features: complete", flush=True)
    valid_features = _validation_graph_features(
        train_students,
        train_exercises,
        train_labels,
        num_students=num_students,
        num_exercises=num_exercises,
        q_matrix=q_matrix,
        query_students=valid_students,
        query_exercises=valid_exercises,
        seed=seed,
    )
    print("validation train-only graph features: complete", flush=True)

    batch_size = int(loaded_args["batch_size"])
    frozen_train_labels, frozen_train_logits = _collect_frozen_logits(
        model,
        train,
        student_map,
        exercise_map,
        concept_map,
        batch_size=batch_size,
        device=device,
        exclude_training_labels=True,
    )
    print("frozen leave-one-query-out train logits: complete", flush=True)
    frozen_valid_labels, frozen_valid_logits = _collect_frozen_logits(
        model,
        valid,
        student_map,
        exercise_map,
        concept_map,
        batch_size=batch_size,
        device=device,
        exclude_training_labels=False,
    )
    print("frozen validation logits: complete", flush=True)
    if not np.array_equal(frozen_train_labels, train_labels.astype(np.float32)):
        raise RuntimeError("training label order changed during frozen inference")
    if not np.array_equal(frozen_valid_labels, valid_labels.astype(np.float32)):
        raise RuntimeError("validation label order changed during frozen inference")

    baseline = _metrics(frozen_valid_labels, frozen_valid_logits)
    stored_auc = float(checkpoint.get("val_auc", float("nan")))
    if not math.isfinite(stored_auc) or abs(baseline["auc"] - stored_auc) > 1e-7:
        raise RuntimeError(
            "frozen checkpoint reproduction failed: "
            f"stored={stored_auc:.12f}, reproduced={baseline['auc']:.12f}"
        )

    signed = _evaluate_candidate(
        "signed",
        frozen_train_logits,
        frozen_train_labels,
        frozen_valid_logits,
        frozen_valid_labels,
        [train_features["right"], train_features["wrong"]],
        [valid_features["right"], valid_features["wrong"]],
        [1.0, -1.0],
    )
    unsigned = _evaluate_candidate(
        "unsigned",
        frozen_train_logits,
        frozen_train_labels,
        frozen_valid_logits,
        frozen_valid_labels,
        [train_features["unsigned"]],
        [valid_features["unsigned"]],
        [1.0],
    )
    sign_shuffle = _evaluate_candidate(
        "within_student_sign_shuffle",
        frozen_train_logits,
        frozen_train_labels,
        frozen_valid_logits,
        frozen_valid_labels,
        [
            train_features["shuffle_right"],
            train_features["shuffle_wrong"],
        ],
        [
            valid_features["shuffle_right"],
            valid_features["shuffle_wrong"],
        ],
        [1.0, -1.0],
    )

    candidates = {
        "signed": signed,
        "unsigned": unsigned,
        "within_student_sign_shuffle": sign_shuffle,
    }
    for candidate in candidates.values():
        candidate_metrics = candidate["metrics"]
        candidate["delta"] = {
            key: float(candidate_metrics[key] - baseline[key])
            for key in ("auc", "bce_loss", "rmse")
        }

    result: Dict[str, object] = {
        "schema": "signed_hetero_frozen_probe_v1",
        "dataset": str(loaded_args.get("dataset_name", data_dir.name)),
        "seed": int(seed),
        "folds": FOLDS,
        "l2_weight": L2_WEIGHT,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "train_csv_sha256": _sha256(train_path),
        "valid_csv_sha256": _sha256(valid_path),
        "rows": {
            "train": int(len(train)),
            "valid": int(len(valid)),
            "unique_student_item_pairs": int(unique_pairs),
        },
        "baseline": baseline,
        "candidates": candidates,
        "audit": {
            "fold_unit": "student_item_pair",
            "pair_overlap_across_folds": 0,
            "training_query_edges_excluded": True,
            "validation_edges_from_train_only": True,
            "test_evaluated": False,
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return result


def _self_test() -> None:
    students = np.asarray([0, 0, 0, 1, 1, 1, 2, 2], dtype=np.int64)
    exercises = np.asarray([0, 0, 1, 0, 1, 2, 1, 2], dtype=np.int64)
    labels = np.asarray([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float64)
    q_matrix = sp.csr_matrix(
        np.asarray(
            [
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        )
    )
    row_fold, unique_pairs = _pair_folds(
        students,
        exercises,
        3,
        folds=2,
        seed=42,
    )
    assert unique_pairs == 7
    pair_keys = students * 3 + exercises
    for pair in np.unique(pair_keys):
        assert np.unique(row_fold[pair_keys == pair]).size == 1
    features = _cross_fitted_graph_features(
        students,
        exercises,
        labels,
        row_fold,
        num_students=3,
        num_exercises=3,
        q_matrix=q_matrix,
        folds=2,
        seed=42,
    )
    for values in features.values():
        assert values.shape == labels.shape
        assert np.isfinite(values).all()
    right, wrong, _ = _pair_response_matrices(
        students,
        exercises,
        labels,
        (3, 3),
    )
    assert math.isclose(float(right[0, 0]), 0.5)
    assert math.isclose(float(wrong[0, 0]), 0.5)
    shuffled = _shuffle_labels_within_students(students, labels, seed=7)
    for student in np.unique(students):
        mask = students == student
        assert math.isclose(
            float(shuffled[mask].sum()),
            float(labels[mask].sum()),
        )
    base = np.zeros(len(labels), dtype=np.float64)
    beta, scales = _fit_nonnegative_offset(
        base,
        labels,
        [features["right"], features["wrong"]],
        [1.0, -1.0],
    )
    assert beta.shape == (2,)
    assert scales.shape == (2,)
    assert (beta >= 0.0).all()
    print("signed hetero response graph probe self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir")
    parser.add_argument("--output_json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return
    if not args.checkpoint_dir or not args.output_json:
        parser.error("--checkpoint_dir and --output_json are required")

    checkpoint_dir = _resolve_path(args.checkpoint_dir)
    output_json = _resolve_path(args.output_json)
    device = torch.device(args.device)
    result = run_probe(
        checkpoint_dir,
        output_json,
        seed=args.seed,
        device=device,
    )
    signed = result["candidates"]["signed"]
    print(
        f"{result['dataset']}: baseline={result['baseline']['auc']:.9f} "
        f"signed={signed['metrics']['auc']:.9f} "
        f"delta={signed['delta']['auc']:+.9f}"
    )
    print(f"saved: {output_json}")


if __name__ == "__main__":
    main()
