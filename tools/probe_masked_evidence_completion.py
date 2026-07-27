#!/usr/bin/env python
"""Validation-only qualification probe for masked response-evidence completion.

The production model is never modified or trained here.  This probe asks a
smaller prerequisite question: after removing a whole student-item group and
excluding that item's Q concepts from the student's profile, can the remaining
train-only evidence predict the removed responses better than simple backoff
controls?

Student-item groups are assigned to five label-independent folds within each
student.  Each qualification rotation uses three folds only as response
context, one fold as probe supervision, and one fold for evaluation.  Both the
supervision and evaluation responses are therefore absent from their input
statistics.  Repeated responses stay together, and a multi-concept response
cannot survive through another concept in the same Q row.  The validation
model is fitted on five-fold out-of-fold features, never leave-one-out target
statistics.

The main probe is deliberately tiny: fixed pooled evidence features feed a
19->16->8->1 MLP with fewer than 500 trainable parameters.  It has no student
or concept embeddings, attention, graph, low-rank interaction, prediction
branch on the frozen CD model, or validation-tuned hyperparameters.  Controls
include item-only rate, student evidence mean, a linear backoff model, and the
same MLP after profile-to-query alignment is shuffled within item-rate strata.

Only train.csv and valid.csv are accepted through a checkpoint's train-only
metadata.  There is intentionally no test-split option.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import build_student_concept_response_stats  # noqa: E402
from src.trainer import _require_graph_irt_checkpoint  # noqa: E402


FOLDS = 5
HIDDEN_DIMS = (16, 8)
DEFAULT_EPOCHS = 12
TRAIN_BATCH_SIZE = 8192
FEATURE_BATCH_SIZE = 2048
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-4
DELTA_LIMIT = 2.0
EPS = 1e-8

FEATURE_NAMES = (
    "rate_evidence_mean",
    "rate_evidence_std",
    "rate_evidence_positive_mass",
    "rate_evidence_negative_mass",
    "rate_evidence_max",
    "rate_evidence_min",
    "residual_evidence_mean",
    "residual_evidence_std",
    "residual_evidence_positive_mass",
    "residual_evidence_negative_mass",
    "log_source_response_count",
    "log_source_concept_count",
    "source_rate_logit",
    "source_count_concentration",
    "item_rate_logit",
    "target_concept_prior_logit",
    "global_rate_logit",
    "log_q_size",
    "source_available",
)
PROFILE_FEATURES = tuple(range(14))
BACKOFF_FEATURES = (12, 14, 15, 16, 17, 18)
CANDIDATES = (
    "mec",
    "backoff_linear",
    "profile_shuffle",
    "item_only",
    "student_mean",
)


@dataclass(frozen=True)
class EvidenceStats:
    student_concept_count: np.ndarray
    student_concept_correct: np.ndarray
    student_concept_residual_sum: np.ndarray
    concept_count: np.ndarray
    concept_correct: np.ndarray
    item_count: np.ndarray
    item_correct: np.ndarray
    global_count: float
    global_correct: float


class OffsetMLP(nn.Module):
    """Bounded evidence correction relative to a query-excluded item rate."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIMS[0]),
            nn.GELU(),
            nn.Linear(HIDDEN_DIMS[0], HIDDEN_DIMS[1]),
            nn.GELU(),
            nn.Linear(HIDDEN_DIMS[1], 1),
        )

    def forward(self, features: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        delta = DELTA_LIMIT * torch.tanh(self.net(features).reshape(-1))
        return offset.reshape(-1) + delta


class OffsetLinear(nn.Module):
    """Small learned backoff control with no distribution-shape features."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        delta = DELTA_LIMIT * torch.tanh(self.linear(features).reshape(-1))
        return offset.reshape(-1) + delta


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


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    return np.log(clipped) - np.log1p(-clipped)


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
    labels_f64 = np.asarray(labels, dtype=np.float64)
    probs_f64 = np.clip(
        np.asarray(probabilities, dtype=np.float64),
        EPS,
        1.0 - EPS,
    )
    if labels_f64.size == 0:
        return {
            "auc": float("nan"),
            "bce_loss": float("nan"),
            "rmse": float("nan"),
        }
    auc = (
        float(roc_auc_score(labels_f64, probs_f64))
        if np.unique(labels_f64).size >= 2
        else float("nan")
    )
    bce = float(
        -np.mean(
            labels_f64 * np.log(probs_f64)
            + (1.0 - labels_f64) * np.log1p(-probs_f64)
        )
    )
    rmse = float(np.sqrt(np.mean(np.square(labels_f64 - probs_f64))))
    return {"auc": auc, "bce_loss": bce, "rmse": rmse}


def _filter_and_map(
    frame: pd.DataFrame,
    student_map: Mapping[object, int],
    exercise_map: Mapping[object, int],
    q_matrix: np.ndarray,
) -> pd.DataFrame:
    filtered = frame[
        frame["stu_id"].isin(student_map)
        & frame["exer_id"].isin(exercise_map)
    ].copy()
    filtered["_student"] = filtered["stu_id"].map(student_map).astype(np.int64)
    filtered["_exercise"] = filtered["exer_id"].map(exercise_map).astype(np.int64)
    filtered["_label"] = filtered["label"].astype(np.float64)
    filtered = filtered[
        q_matrix[filtered["_exercise"].to_numpy(dtype=np.int64)].sum(axis=1) > 0
    ].reset_index(drop=True)
    labels = filtered["_label"].to_numpy(dtype=np.float64)
    if len(filtered) == 0:
        raise ValueError("no rows remain after applying checkpoint mappings")
    if not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError("response labels must be binary")
    return filtered


def _assign_pair_folds(
    frame: pd.DataFrame,
    *,
    num_exercises: int,
    folds: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if folds < 2:
        raise ValueError("at least two folds are required")
    pair_key = (
        frame["_student"].to_numpy(dtype=np.int64) * int(num_exercises)
        + frame["_exercise"].to_numpy(dtype=np.int64)
    )
    grouped = (
        frame.assign(_pair_key=pair_key)
        .groupby(["_pair_key", "_student", "_exercise"], sort=True)["_label"]
        .agg(attempts="size", correct="sum")
        .reset_index()
    )
    grouped["_group_index"] = np.arange(len(grouped), dtype=np.int64)
    fold_values = np.empty(len(grouped), dtype=np.int16)
    for student, indices in grouped.groupby("_student", sort=True).groups.items():
        indices_array = np.asarray(list(indices), dtype=np.int64)
        rng = np.random.RandomState(int(seed) + 1009 * int(student))
        permutation = rng.permutation(len(indices_array))
        assigned = np.empty(len(indices_array), dtype=np.int16)
        assigned[permutation] = (
            np.arange(len(indices_array), dtype=np.int64) % int(folds)
        )
        fold_values[indices_array] = assigned
    grouped["_fold"] = fold_values
    pair_to_group = dict(
        zip(
            grouped["_pair_key"].astype(np.int64),
            grouped["_group_index"].astype(np.int64),
        )
    )
    pair_to_fold = dict(
        zip(
            grouped["_pair_key"].astype(np.int64),
            grouped["_fold"].astype(np.int16),
        )
    )
    annotated = frame.copy()
    annotated["_pair_key"] = pair_key
    annotated["_group_index"] = annotated["_pair_key"].map(pair_to_group).astype(
        np.int64
    )
    annotated["_fold"] = annotated["_pair_key"].map(pair_to_fold).astype(np.int16)
    for pair in grouped["_pair_key"].to_numpy(dtype=np.int64):
        row_folds = annotated.loc[annotated["_pair_key"] == pair, "_fold"].unique()
        if len(row_folds) != 1:
            raise RuntimeError("a repeated student-item pair was split across folds")
    return annotated, grouped


def _build_stats(
    context: pd.DataFrame,
    *,
    student_map: Mapping[object, int],
    exercise_map: Mapping[object, int],
    q_tensor: torch.Tensor,
) -> EvidenceStats:
    tensors = build_student_concept_response_stats(
        context,
        dict(student_map),
        dict(exercise_map),
        q_tensor,
    )
    num_exercises = len(exercise_map)
    exercises = context["_exercise"].to_numpy(dtype=np.int64)
    labels = context["_label"].to_numpy(dtype=np.float64)
    item_count = np.bincount(exercises, minlength=num_exercises).astype(np.float64)
    item_correct = np.bincount(
        exercises,
        weights=labels,
        minlength=num_exercises,
    ).astype(np.float64)
    return EvidenceStats(
        student_concept_count=tensors["student_concept_count"].numpy().astype(
            np.float64
        ),
        student_concept_correct=tensors["student_concept_correct"].numpy().astype(
            np.float64
        ),
        student_concept_residual_sum=tensors[
            "student_concept_residual_sum"
        ].numpy().astype(np.float64),
        concept_count=tensors["concept_count"].numpy().astype(np.float64),
        concept_correct=tensors["concept_correct"].numpy().astype(np.float64),
        item_count=item_count,
        item_correct=item_correct,
        global_count=float(tensors["global_count"].item()),
        global_correct=float(tensors["global_correct"].item()),
    )


def _query_arrays(frame: pd.DataFrame) -> Tuple[np.ndarray, ...]:
    attempts = (
        frame["attempts"].to_numpy(dtype=np.float64)
        if "attempts" in frame
        else np.ones(len(frame), dtype=np.float64)
    )
    correct = (
        frame["correct"].to_numpy(dtype=np.float64)
        if "correct" in frame
        else frame["_label"].to_numpy(dtype=np.float64)
    )
    return (
        frame["_student"].to_numpy(dtype=np.int64),
        frame["_exercise"].to_numpy(dtype=np.int64),
        attempts,
        correct,
    )


def _weighted_mean_std(
    values: np.ndarray,
    weights: np.ndarray,
    denominator: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    mean = (values * weights).sum(axis=1) / denominator
    variance = (
        np.square(values - mean[:, None]) * weights
    ).sum(axis=1) / denominator
    return mean, np.sqrt(np.maximum(variance, 0.0))


def _build_features(
    stats: EvidenceStats,
    queries: pd.DataFrame,
    q_matrix: np.ndarray,
    *,
    exclude_query_group: bool,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    students, exercises, attempts, correct = _query_arrays(queries)
    features = np.zeros((len(queries), len(FEATURE_NAMES)), dtype=np.float64)
    min_target_count = np.zeros(len(queries), dtype=np.float64)
    max_target_count = np.zeros(len(queries), dtype=np.float64)

    for start in range(0, len(queries), FEATURE_BATCH_SIZE):
        stop = min(len(queries), start + FEATURE_BATCH_SIZE)
        student = students[start:stop]
        exercise = exercises[start:stop]
        trials = attempts[start:stop]
        successes = correct[start:stop]
        q_mask = q_matrix[exercise] > 0
        q_size = q_mask.sum(axis=1).astype(np.float64)
        if np.any(q_size <= 0):
            raise RuntimeError("a query item has no mapped concepts")

        counts = stats.student_concept_count[student]
        correct_sum = stats.student_concept_correct[student]
        residual_sum = stats.student_concept_residual_sum[student]
        source_mask = (counts > 0.0) & (~q_mask)
        source_weights = counts * source_mask
        source_count = source_weights.sum(axis=1)
        safe_source_count = np.maximum(source_count, 1.0)
        source_concepts = source_mask.sum(axis=1).astype(np.float64)

        if exclude_query_group:
            global_count = stats.global_count - trials
            global_correct = stats.global_correct - successes
        else:
            global_count = np.full(stop - start, stats.global_count)
            global_correct = np.full(stop - start, stats.global_correct)
        if np.any(global_count <= 0.0):
            raise RuntimeError("query exclusion removed all global evidence")
        global_rate = global_correct / global_count

        source_concept_rate = (
            stats.concept_correct[None, :] + global_rate[:, None]
        ) / (stats.concept_count[None, :] + 1.0)
        posterior = (correct_sum + source_concept_rate) / (counts + 1.0)
        reliability = counts / (counts + 1.0)
        rate_evidence = (
            (_logit(posterior) - _logit(source_concept_rate)) * reliability
        )
        rate_evidence = np.clip(rate_evidence, -4.0, 4.0)
        residual_evidence = (
            residual_sum / np.maximum(counts, 1.0) * reliability
        )
        residual_evidence = np.clip(residual_evidence, -1.0, 1.0)

        rate_mean, rate_std = _weighted_mean_std(
            rate_evidence,
            source_weights,
            safe_source_count,
        )
        residual_mean, residual_std = _weighted_mean_std(
            residual_evidence,
            source_weights,
            safe_source_count,
        )
        rate_positive = (
            np.maximum(rate_evidence, 0.0) * source_weights
        ).sum(axis=1) / safe_source_count
        rate_negative = (
            np.maximum(-rate_evidence, 0.0) * source_weights
        ).sum(axis=1) / safe_source_count
        residual_positive = (
            np.maximum(residual_evidence, 0.0) * source_weights
        ).sum(axis=1) / safe_source_count
        residual_negative = (
            np.maximum(-residual_evidence, 0.0) * source_weights
        ).sum(axis=1) / safe_source_count
        rate_max = np.where(
            source_mask.any(axis=1),
            np.where(source_mask, rate_evidence, -np.inf).max(axis=1),
            0.0,
        )
        rate_min = np.where(
            source_mask.any(axis=1),
            np.where(source_mask, rate_evidence, np.inf).min(axis=1),
            0.0,
        )
        source_correct = (correct_sum * source_mask).sum(axis=1)
        source_rate = (source_correct + global_rate) / (source_count + 1.0)
        source_concentration = (
            np.where(source_mask, counts, 0.0).max(axis=1)
            / safe_source_count
        )

        item_count = stats.item_count[exercise].copy()
        item_correct = stats.item_correct[exercise].copy()
        if exclude_query_group:
            item_count -= trials
            item_correct -= successes
        item_rate = (item_correct + global_rate) / (
            np.maximum(item_count, 0.0) + 1.0
        )

        target_count = np.broadcast_to(
            stats.concept_count,
            q_mask.shape,
        ).astype(np.float64, copy=True)
        target_correct = np.broadcast_to(
            stats.concept_correct,
            q_mask.shape,
        ).astype(np.float64, copy=True)
        student_target_count = counts.copy()
        if exclude_query_group:
            target_count -= trials[:, None] * q_mask
            target_correct -= successes[:, None] * q_mask
            student_target_count -= trials[:, None] * q_mask
        target_count = np.maximum(target_count, 0.0)
        target_correct = np.maximum(target_correct, 0.0)
        student_target_count = np.maximum(student_target_count, 0.0)
        target_rate = (target_correct + global_rate[:, None]) / (
            target_count + 1.0
        )
        target_prior = (target_rate * q_mask).sum(axis=1) / q_size

        masked_target_count = np.where(q_mask, student_target_count, np.nan)
        min_target_count[start:stop] = np.nanmin(masked_target_count, axis=1)
        max_target_count[start:stop] = np.nanmax(masked_target_count, axis=1)

        batch_features = np.column_stack(
            (
                rate_mean,
                rate_std,
                rate_positive,
                rate_negative,
                rate_max,
                rate_min,
                residual_mean,
                residual_std,
                residual_positive,
                residual_negative,
                np.log1p(source_count),
                np.log1p(source_concepts),
                _logit(source_rate),
                source_concentration,
                _logit(item_rate),
                _logit(target_prior),
                _logit(global_rate),
                np.log(q_size),
                (source_count > 0.0).astype(np.float64),
            )
        )
        if not np.isfinite(batch_features).all():
            raise RuntimeError("masked evidence features contain non-finite values")
        features[start:stop] = batch_features

    return features.astype(np.float32), {
        "min_target_count": min_target_count.astype(np.float32),
        "max_target_count": max_target_count.astype(np.float32),
    }


def _shuffle_profile(
    features: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    shuffled = np.asarray(features, dtype=np.float32).copy()
    item_logits = shuffled[:, FEATURE_NAMES.index("item_rate_logit")]
    if len(shuffled) < 2:
        return shuffled
    quantiles = np.quantile(item_logits, np.linspace(0.0, 1.0, 11))
    bins = np.searchsorted(quantiles[1:-1], item_logits, side="right")
    rng = np.random.RandomState(seed)
    for bin_id in range(10):
        rows = np.flatnonzero(bins == bin_id)
        if len(rows) <= 1:
            continue
        source_rows = rows[rng.permutation(len(rows))]
        shuffled[np.ix_(rows, PROFILE_FEATURES)] = features[
            np.ix_(source_rows, PROFILE_FEATURES)
        ]
    return shuffled


def _standardize(
    train_features: np.ndarray,
    eval_features: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    mean = train_features.mean(axis=0, dtype=np.float64)
    std = train_features.std(axis=0, dtype=np.float64)
    std = np.where(std > 1e-6, std, 1.0)
    train = (train_features.astype(np.float64) - mean) / std
    evaluation = (eval_features.astype(np.float64) - mean) / std
    return train.astype(np.float32), evaluation.astype(np.float32)


def _fit_predict(
    train_features: np.ndarray,
    train_successes: np.ndarray,
    train_attempts: np.ndarray,
    eval_features: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    epochs: int,
    linear: bool,
) -> Tuple[np.ndarray, Dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if linear:
        selected = np.asarray(BACKOFF_FEATURES, dtype=np.int64)
        train_features = train_features[:, selected]
        eval_features = eval_features[:, selected]
    train_x, eval_x = _standardize(train_features, eval_features)
    train_offset = train_features[
        :, BACKOFF_FEATURES.index(14) if linear else 14
    ]
    eval_offset = eval_features[
        :, BACKOFF_FEATURES.index(14) if linear else 14
    ]

    model: nn.Module = (
        OffsetLinear(train_x.shape[1])
        if linear
        else OffsetMLP(train_x.shape[1])
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    x_tensor = torch.from_numpy(train_x).to(device)
    offset_tensor = torch.from_numpy(train_offset.astype(np.float32)).to(device)
    targets = torch.from_numpy(
        (train_successes / np.maximum(train_attempts, 1.0)).astype(np.float32)
    ).to(device)
    weights = torch.from_numpy(train_attempts.astype(np.float32)).to(device)

    model.train()
    final_loss = float("nan")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 17)
    for _ in range(int(epochs)):
        permutation = torch.randperm(
            len(x_tensor),
            generator=generator,
            device=device,
        )
        total_loss = 0.0
        total_weight = 0.0
        for start in range(0, len(permutation), TRAIN_BATCH_SIZE):
            indices = permutation[start : start + TRAIN_BATCH_SIZE]
            logits = model(x_tensor[indices], offset_tensor[indices])
            losses = F.binary_cross_entropy_with_logits(
                logits,
                targets[indices],
                reduction="none",
            )
            batch_weights = weights[indices]
            loss = (losses * batch_weights).sum() / batch_weights.sum().clamp(
                min=1.0
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float((losses * batch_weights).sum().detach().cpu())
            total_weight += float(batch_weights.sum().detach().cpu())
        final_loss = total_loss / max(total_weight, 1.0)

    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(eval_x), TRAIN_BATCH_SIZE):
            stop = min(len(eval_x), start + TRAIN_BATCH_SIZE)
            features_tensor = torch.from_numpy(eval_x[start:stop]).to(device)
            offset_eval_tensor = torch.from_numpy(
                eval_offset[start:stop].astype(np.float32)
            ).to(device)
            logits = model(features_tensor, offset_eval_tensor)
            parts.append(torch.sigmoid(logits).cpu())
    probabilities = torch.cat(parts).numpy().astype(np.float64)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    return probabilities, {
        "final_weighted_bce": float(final_loss),
        "parameter_count": parameter_count,
    }


def _bucket_masks(meta: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    minimum = np.asarray(meta["min_target_count"], dtype=np.float64)
    maximum = np.asarray(meta["max_target_count"], dtype=np.float64)
    return {
        "all": np.ones(len(minimum), dtype=bool),
        "min_n0": minimum == 0.0,
        "min_n1_2": (minimum >= 1.0) & (minimum < 3.0),
        "min_n_lt3": minimum < 3.0,
        "min_n_ge3": minimum >= 3.0,
        "all_q_zero": maximum == 0.0,
    }


def _evaluate_by_bucket(
    labels: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    meta: Mapping[str, np.ndarray],
) -> Dict[str, object]:
    output: Dict[str, object] = {}
    for bucket, mask in _bucket_masks(meta).items():
        bucket_labels = labels[mask]
        output[bucket] = {
            "rows": int(mask.sum()),
            "positives": int(bucket_labels.sum()),
            "negatives": int(len(bucket_labels) - bucket_labels.sum()),
            "candidates": {
                name: _metrics(bucket_labels, probabilities[mask])
                for name, probabilities in predictions.items()
            },
        }
    return output


def _run_cross_fitted_probe(
    train: pd.DataFrame,
    groups: pd.DataFrame,
    *,
    student_map: Mapping[object, int],
    exercise_map: Mapping[object, int],
    q_tensor: torch.Tensor,
    q_matrix: np.ndarray,
    device: torch.device,
    seed: int,
    epochs: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    predictions = {
        name: np.zeros(len(groups), dtype=np.float64) for name in CANDIDATES
    }
    fold_audit: Dict[str, object] = {}
    for evaluation_fold in range(FOLDS):
        supervision_fold = (evaluation_fold + 1) % FOLDS
        context_folds = tuple(
            fold
            for fold in range(FOLDS)
            if fold not in {evaluation_fold, supervision_fold}
        )
        context = train[train["_fold"].isin(context_folds)].reset_index(drop=True)
        fit_groups = groups[
            groups["_fold"] == supervision_fold
        ].reset_index(drop=True)
        held_groups = groups[
            groups["_fold"] == evaluation_fold
        ].reset_index(drop=True)
        if len(context) == 0 or len(fit_groups) == 0 or len(held_groups) == 0:
            raise RuntimeError(
                f"rotation {evaluation_fold} has an empty context, "
                "supervision fold, or evaluation fold"
            )
        stats = _build_stats(
            context,
            student_map=student_map,
            exercise_map=exercise_map,
            q_tensor=q_tensor,
        )
        fit_features, _ = _build_features(
            stats,
            fit_groups,
            q_matrix,
            exclude_query_group=False,
        )
        held_features, _ = _build_features(
            stats,
            held_groups,
            q_matrix,
            exclude_query_group=False,
        )
        fit_successes = fit_groups["correct"].to_numpy(dtype=np.float64)
        fit_attempts = fit_groups["attempts"].to_numpy(dtype=np.float64)
        held_indices = held_groups["_group_index"].to_numpy(dtype=np.int64)

        mec, mec_audit = _fit_predict(
            fit_features,
            fit_successes,
            fit_attempts,
            held_features,
            device=device,
            seed=seed + 101 * evaluation_fold,
            epochs=epochs,
            linear=False,
        )
        backoff, backoff_audit = _fit_predict(
            fit_features,
            fit_successes,
            fit_attempts,
            held_features,
            device=device,
            seed=seed + 101 * evaluation_fold + 1,
            epochs=epochs,
            linear=True,
        )
        shuffled_fit = _shuffle_profile(
            fit_features,
            seed=seed + 1009 * (evaluation_fold + 1),
        )
        shuffled_held = _shuffle_profile(
            held_features,
            seed=seed + 2003 * (evaluation_fold + 1),
        )
        shuffled, shuffle_audit = _fit_predict(
            shuffled_fit,
            fit_successes,
            fit_attempts,
            shuffled_held,
            device=device,
            seed=seed + 101 * evaluation_fold + 2,
            epochs=epochs,
            linear=False,
        )
        predictions["mec"][held_indices] = mec
        predictions["backoff_linear"][held_indices] = backoff
        predictions["profile_shuffle"][held_indices] = shuffled
        predictions["item_only"][held_indices] = torch.sigmoid(
            torch.from_numpy(held_features[:, 14])
        ).numpy()
        predictions["student_mean"][held_indices] = torch.sigmoid(
            torch.from_numpy(held_features[:, 12])
        ).numpy()
        fold_audit[str(evaluation_fold)] = {
            "context_folds": list(context_folds),
            "context_rows": int(len(context)),
            "supervision_fold": int(supervision_fold),
            "supervision_pairs": int(len(fit_groups)),
            "evaluation_fold": int(evaluation_fold),
            "evaluation_pairs": int(len(held_groups)),
            "mec": mec_audit,
            "backoff_linear": backoff_audit,
            "profile_shuffle": shuffle_audit,
        }
        print(
            f"masked rotation {evaluation_fold + 1}/{FOLDS}: "
            f"context_rows={len(context)} "
            f"supervision_pairs={len(fit_groups)} "
            f"evaluation_pairs={len(held_groups)}",
            flush=True,
        )
    return predictions, fold_audit


def _fit_full_and_predict_valid(
    train: pd.DataFrame,
    groups: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    student_map: Mapping[object, int],
    exercise_map: Mapping[object, int],
    q_tensor: torch.Tensor,
    q_matrix: np.ndarray,
    device: torch.device,
    seed: int,
    epochs: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, object]]:
    train_features = np.zeros(
        (len(groups), len(FEATURE_NAMES)),
        dtype=np.float32,
    )
    feature_audit: Dict[str, object] = {}
    for fold in range(FOLDS):
        context = train[train["_fold"] != fold].reset_index(drop=True)
        fold_groups = groups[groups["_fold"] == fold].reset_index(drop=True)
        stats = _build_stats(
            context,
            student_map=student_map,
            exercise_map=exercise_map,
            q_tensor=q_tensor,
        )
        fold_features, _ = _build_features(
            stats,
            fold_groups,
            q_matrix,
            exclude_query_group=False,
        )
        fold_indices = fold_groups["_group_index"].to_numpy(dtype=np.int64)
        train_features[fold_indices] = fold_features
        feature_audit[str(fold)] = {
            "context_rows": int(len(context)),
            "supervision_pairs": int(len(fold_groups)),
        }

    full_stats = _build_stats(
        train,
        student_map=student_map,
        exercise_map=exercise_map,
        q_tensor=q_tensor,
    )
    valid_features, valid_meta = _build_features(
        full_stats,
        valid,
        q_matrix,
        exclude_query_group=False,
    )
    successes = groups["correct"].to_numpy(dtype=np.float64)
    attempts = groups["attempts"].to_numpy(dtype=np.float64)
    predictions: Dict[str, np.ndarray] = {}
    predictions["mec"], mec_audit = _fit_predict(
        train_features,
        successes,
        attempts,
        valid_features,
        device=device,
        seed=seed + 5001,
        epochs=epochs,
        linear=False,
    )
    predictions["backoff_linear"], backoff_audit = _fit_predict(
        train_features,
        successes,
        attempts,
        valid_features,
        device=device,
        seed=seed + 5002,
        epochs=epochs,
        linear=True,
    )
    shuffled_train = _shuffle_profile(train_features, seed=seed + 7001)
    shuffled_valid = _shuffle_profile(valid_features, seed=seed + 7002)
    predictions["profile_shuffle"], shuffle_audit = _fit_predict(
        shuffled_train,
        successes,
        attempts,
        shuffled_valid,
        device=device,
        seed=seed + 5003,
        epochs=epochs,
        linear=False,
    )
    predictions["item_only"] = torch.sigmoid(
        torch.from_numpy(valid_features[:, 14])
    ).numpy()
    predictions["student_mean"] = torch.sigmoid(
        torch.from_numpy(valid_features[:, 12])
    ).numpy()
    return predictions, valid_meta, {
        "cross_fitted_feature_construction": feature_audit,
        "mec": mec_audit,
        "backoff_linear": backoff_audit,
        "profile_shuffle": shuffle_audit,
    }


def run_probe(
    checkpoint_dir: Path,
    output_json: Path,
    *,
    seed: int,
    device: torch.device,
    epochs: int,
) -> Dict[str, object]:
    checkpoint_path = checkpoint_dir / "best_model.pth"
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    _require_graph_irt_checkpoint(checkpoint, str(checkpoint_path))
    loaded_args = checkpoint["args"]
    info = checkpoint["info_dict"]
    data_dir = _resolve_path(str(loaded_args["data_dir"]))
    train_path = data_dir / "train.csv"
    valid_path = data_dir / "valid.csv"
    train_source = pd.read_csv(train_path)
    valid_source = pd.read_csv(valid_path)
    student_map = info["stu_id_map"]
    exercise_map = info["exer_id_map"]
    q_tensor = info["q_matrix"].detach().cpu().float()
    q_matrix = q_tensor.numpy().astype(np.float64)
    train = _filter_and_map(
        train_source,
        student_map,
        exercise_map,
        q_matrix,
    )
    valid = _filter_and_map(
        valid_source,
        student_map,
        exercise_map,
        q_matrix,
    )
    train, groups = _assign_pair_folds(
        train,
        num_exercises=len(exercise_map),
        folds=FOLDS,
        seed=seed,
    )
    print(
        f"{loaded_args.get('dataset_name', data_dir.name)}: "
        f"train_rows={len(train)} pairs={len(groups)} valid_rows={len(valid)}",
        flush=True,
    )

    pair_predictions, fold_audit = _run_cross_fitted_probe(
        train,
        groups,
        student_map=student_map,
        exercise_map=exercise_map,
        q_tensor=q_tensor,
        q_matrix=q_matrix,
        device=device,
        seed=seed,
        epochs=epochs,
    )
    group_index = train["_group_index"].to_numpy(dtype=np.int64)
    train_labels = train["_label"].to_numpy(dtype=np.float64)
    oof_metrics = {
        name: _metrics(train_labels, probabilities[group_index])
        for name, probabilities in pair_predictions.items()
    }
    valid_predictions, valid_meta, full_fit_audit = _fit_full_and_predict_valid(
        train,
        groups,
        valid,
        student_map=student_map,
        exercise_map=exercise_map,
        q_tensor=q_tensor,
        q_matrix=q_matrix,
        device=device,
        seed=seed,
        epochs=epochs,
    )
    valid_labels = valid["_label"].to_numpy(dtype=np.float64)
    validation = _evaluate_by_bucket(
        valid_labels,
        valid_predictions,
        valid_meta,
    )
    best_control_bce = min(
        oof_metrics[name]["bce_loss"] for name in CANDIDATES if name != "mec"
    )
    relative_bce_reduction = (
        best_control_bce - oof_metrics["mec"]["bce_loss"]
    ) / max(best_control_bce, EPS)

    result: Dict[str, object] = {
        "schema": "masked_evidence_completion_qualification_v2",
        "dataset": str(loaded_args.get("dataset_name", data_dir.name)),
        "seed": int(seed),
        "folds": FOLDS,
        "epochs": int(epochs),
        "feature_names": list(FEATURE_NAMES),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "train_csv_sha256": _sha256(train_path),
        "valid_csv_sha256": _sha256(valid_path),
        "rows": {
            "train": int(len(train)),
            "train_student_item_pairs": int(len(groups)),
            "valid": int(len(valid)),
        },
        "oof": {
            "metrics": oof_metrics,
            "mec_relative_bce_reduction_vs_best_control": float(
                relative_bce_reduction
            ),
        },
        "validation": validation,
        "fit_audit": {
            "cross_fitted": fold_audit,
            "full_train": full_fit_audit,
        },
        "audit": {
            "fold_unit": "student_item_pair_within_student",
            "repeated_pair_split_across_folds": False,
            "evaluation_fold_absent_from_rotation_statistics": True,
            "supervision_fold_absent_from_rotation_statistics": True,
            "validation_fit_uses_cross_fitted_training_features": True,
            "all_q_concepts_excluded_from_student_profile": True,
            "student_or_concept_embeddings_used": False,
            "graph_used": False,
            "frozen_cdm_logits_used": False,
            "validation_used_for_tuning": False,
            "test_evaluated": False,
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return result


def _self_test() -> None:
    rows = []
    for student in range(3):
        for exercise, concepts in (
            (0, "0,1"),
            (1, "2"),
            (2, "0"),
            (3, "1"),
            (4, "2"),
        ):
            rows.append(
                {
                    "stu_id": student,
                    "exer_id": exercise,
                    "cpt_seq": concepts,
                    "label": float((student + exercise) % 2),
                }
            )
    rows.append(
        {"stu_id": 0, "exer_id": 0, "cpt_seq": "0,1", "label": 1.0}
    )
    frame = pd.DataFrame(rows)
    student_map = {value: value for value in range(3)}
    exercise_map = {value: value for value in range(5)}
    q_matrix = np.asarray(
        [
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    q_tensor = torch.from_numpy(q_matrix)
    mapped = _filter_and_map(frame, student_map, exercise_map, q_matrix)
    annotated, groups = _assign_pair_folds(
        mapped,
        num_exercises=5,
        folds=FOLDS,
        seed=42,
    )
    duplicate_folds = annotated.loc[
        (annotated["_student"] == 0) & (annotated["_exercise"] == 0),
        "_fold",
    ].unique()
    assert len(duplicate_folds) == 1

    stats = _build_stats(
        annotated,
        student_map=student_map,
        exercise_map=exercise_map,
        q_tensor=q_tensor,
    )
    target = groups[
        (groups["_student"] == 0) & (groups["_exercise"] == 0)
    ].reset_index(drop=True)
    features_before, meta = _build_features(
        stats,
        target,
        q_matrix,
        exclude_query_group=True,
    )
    flipped = annotated.copy()
    mask = (flipped["_student"] == 0) & (flipped["_exercise"] == 0)
    flipped.loc[mask, "label"] = 1.0 - flipped.loc[mask, "label"]
    flipped.loc[mask, "_label"] = 1.0 - flipped.loc[mask, "_label"]
    flipped_groups = target.copy()
    flipped_groups.loc[0, "correct"] = float(
        flipped.loc[mask, "_label"].sum()
    )
    flipped_stats = _build_stats(
        flipped,
        student_map=student_map,
        exercise_map=exercise_map,
        q_tensor=q_tensor,
    )
    features_after, _ = _build_features(
        flipped_stats,
        flipped_groups,
        q_matrix,
        exclude_query_group=True,
    )
    assert np.allclose(features_before, features_after, atol=1e-6, rtol=1e-6)
    assert math.isclose(
        float(features_before[0, FEATURE_NAMES.index("log_q_size")]),
        math.log(2.0),
        abs_tol=1e-6,
    )
    assert np.isfinite(features_before).all()
    assert meta["min_target_count"].shape == (1,)

    random_features = np.random.RandomState(42).normal(size=(32, len(FEATURE_NAMES))).astype(
        np.float32
    )
    successes = np.asarray([index % 2 for index in range(32)], dtype=np.float64)
    attempts = np.ones(32, dtype=np.float64)
    probabilities, audit = _fit_predict(
        random_features[:24],
        successes[:24],
        attempts[:24],
        random_features[24:],
        device=torch.device("cpu"),
        seed=42,
        epochs=2,
        linear=False,
    )
    assert probabilities.shape == (8,)
    assert np.isfinite(probabilities).all()
    assert audit["parameter_count"] < 500
    print("masked evidence completion probe self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir")
    parser.add_argument("--output_json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    if not args.checkpoint_dir or not args.output_json:
        parser.error("--checkpoint_dir and --output_json are required")
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    checkpoint_dir = _resolve_path(args.checkpoint_dir)
    output_json = _resolve_path(args.output_json)
    result = run_probe(
        checkpoint_dir,
        output_json,
        seed=args.seed,
        device=torch.device(args.device),
        epochs=args.epochs,
    )
    validation_all = result["validation"]["all"]["candidates"]
    print(
        f"{result['dataset']}: "
        f"OOF MEC BCE={result['oof']['metrics']['mec']['bce_loss']:.6f}; "
        f"valid MEC AUC={validation_all['mec']['auc']:.9f}; "
        f"backoff AUC={validation_all['backoff_linear']['auc']:.9f}",
        flush=True,
    )
    print(f"saved: {output_json}", flush=True)


if __name__ == "__main__":
    main()
