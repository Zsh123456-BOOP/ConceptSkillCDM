#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Offline probe for the RELM/PLRC redesign.

This script does not train the CD model.  It tests whether recent-paper-inspired
evidence has incremental signal before we wire it into the architecture:

- RELM/A: item + sequence + right/wrong response transition route evidence.
- PLRC/E: exact/recent student concept evidence plus ability-group fallback and
  route-neighbor transfer, gated by evidence counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import create_dataloaders  # noqa: E402
from src.experiment_utils import compute_metrics  # noqa: E402
from src.reliable_evidence import (  # noqa: E402
    assign_student_quantile_groups,
    build_group_concept_logits,
    build_response_transition_priors,
)
from src.trainer import _compute_train_stat_logits  # noqa: E402


BASE_FEATURES = ("student_prior", "exercise_prior", "concept_prior")
A_FEATURES = (
    "a_item_delta",
    "a_seq_delta",
    "a_right_delta",
    "a_wrong_delta",
    "a_route_delta",
    "a_route_reliability",
    "a_off_query_mass",
)
E_FEATURES = (
    "e_exact",
    "e_recent",
    "e_group",
    "e_route_exact_delta",
    "e_route_group_delta",
    "e_reliability",
    "e_route_reliability",
)


def _as_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().astype(np.float32)


def _row_normalize(mat: torch.Tensor) -> torch.Tensor:
    mat = mat.detach().float().clone()
    if mat.dim() != 2 or mat.size(0) != mat.size(1):
        raise ValueError(f"matrix must be square, got {tuple(mat.shape)}")
    if mat.size(0) == 1:
        return torch.ones_like(mat)
    eye = torch.eye(mat.size(0), dtype=mat.dtype, device=mat.device)
    mat = mat.clamp(min=0.0) * (1.0 - eye)
    row_sum = mat.sum(dim=1, keepdim=True)
    return torch.where(row_sum > 0, mat / row_sum.clamp(min=1e-12), torch.zeros_like(mat))


def _query_weight(q_matrix: torch.Tensor, exercise_ids: torch.Tensor) -> torch.Tensor:
    q = q_matrix[exercise_ids.long()].float()
    return q / q.sum(dim=1, keepdim=True).clamp(min=1.0)


def _route(prior: torch.Tensor, query_weight: torch.Tensor) -> torch.Tensor:
    prior = _row_normalize(prior).to(dtype=query_weight.dtype)
    return query_weight.matmul(prior)


def _safe_fused_route(
    item_prior: torch.Tensor,
    seq_prior: torch.Tensor,
    right_prior: torch.Tensor,
    wrong_prior: torch.Tensor,
    *,
    wrong_weight: float,
) -> torch.Tensor:
    fused = (
        item_prior.detach().float().clamp(min=0.0)
        + seq_prior.detach().float().clamp(min=0.0)
        + right_prior.detach().float().clamp(min=0.0)
        + max(0.0, float(wrong_weight)) * wrong_prior.detach().float().clamp(min=0.0)
    )
    return _row_normalize(fused)


def _split_features(
    *,
    loader,
    info_dict: Dict[str, object],
    stat_tensors,
    response_priors: Dict[str, object],
    fused_prior: torch.Tensor,
    group_concept_logits: torch.Tensor,
    reliability_lambda: float,
    shuffle_student_evidence: bool,
    seed: int,
) -> Tuple[pd.DataFrame, np.ndarray]:
    dataset = loader.dataset
    student_ids = dataset.student_ids.detach().long().cpu()
    exercise_ids = dataset.exercise_ids.detach().long().cpu()
    labels = _as_numpy(dataset.labels.detach().float().cpu())
    q_matrix = info_dict["q_matrix"].detach().float().cpu()
    qw = _query_weight(q_matrix, exercise_ids)

    (
        student_logits,
        exercise_logits,
        concept_logits,
        student_concept_logits,
        recent_student_concept_logits,
        _global_rate,
        count_features,
    ) = stat_tensors
    student_logits = student_logits.detach().float().cpu()
    exercise_logits = exercise_logits.detach().float().cpu()
    concept_logits = concept_logits.detach().float().cpu()
    student_concept_logits = student_concept_logits.detach().float().cpu()
    recent_student_concept_logits = recent_student_concept_logits.detach().float().cpu()
    observed_counts = count_features["student_concept_observed"].detach().float().cpu()
    group_concept_logits = group_concept_logits.detach().float().cpu()

    if shuffle_student_evidence and student_concept_logits.size(0) > 0:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed) + 13331)
        perm = torch.randperm(student_concept_logits.size(0), generator=gen)
        student_logits = student_logits[perm]
        student_concept_logits = student_concept_logits[perm]
        recent_student_concept_logits = recent_student_concept_logits[perm]
        observed_counts = observed_counts[perm]

    item_route = _route(info_dict["item_prior_matrix"].detach().float().cpu(), qw)
    seq_route = _route(info_dict["sequence_prior_matrix"].detach().float().cpu(), qw)
    right_route = _route(response_priors["right_prior"].detach().float().cpu(), qw)
    wrong_route = _route(response_priors["wrong_prior"].detach().float().cpu(), qw)
    fused_route = _route(fused_prior.detach().float().cpu(), qw)

    concept = concept_logits
    concept_query = qw.matmul(concept)
    item_concept = item_route.matmul(concept)
    seq_concept = seq_route.matmul(concept)
    right_concept = right_route.matmul(concept)
    wrong_concept = wrong_route.matmul(concept)
    fused_concept = fused_route.matmul(concept)

    sc = student_concept_logits[student_ids]
    recent = recent_student_concept_logits[student_ids]
    counts = observed_counts[student_ids]
    # group fallback is indexed by ability group per student.
    student_group_ids = assign_student_quantile_groups(student_logits, num_groups=group_concept_logits.size(0))
    group_logits = group_concept_logits[student_group_ids[student_ids]]

    query_count = (qw * counts).sum(dim=1)
    route_count = (fused_route * counts).sum(dim=1)
    lam = max(0.0, float(reliability_lambda))
    if lam > 0.0:
        query_rel = query_count / (query_count + lam).clamp(min=1e-6)
        route_rel = route_count / (route_count + lam).clamp(min=1e-6)
    else:
        query_rel = (query_count > 0).float()
        route_rel = (route_count > 0).float()

    exact = query_rel * (qw * sc).sum(dim=1)
    recent_feat = query_rel * (qw * recent).sum(dim=1)
    group_feat = qw.mul(group_logits).sum(dim=1)
    route_exact = route_rel * (fused_route * sc).sum(dim=1)
    route_group = fused_route.mul(group_logits).sum(dim=1)
    off_query_mass = (fused_route * (1.0 - (qw > 0).float())).sum(dim=1)
    route_shift = 0.5 * (fused_route - qw).abs().sum(dim=1)
    route_reliability = torch.maximum(off_query_mass, route_shift).clamp(min=0.0, max=1.0)

    frame = pd.DataFrame(
        {
            "student_prior": _as_numpy(student_logits[student_ids]),
            "exercise_prior": _as_numpy(exercise_logits[exercise_ids]),
            "concept_prior": _as_numpy(concept_query),
            "a_item_delta": _as_numpy(item_concept - concept_query),
            "a_seq_delta": _as_numpy(seq_concept - concept_query),
            "a_right_delta": _as_numpy(right_concept - concept_query),
            "a_wrong_delta": _as_numpy(wrong_concept - concept_query),
            "a_route_delta": _as_numpy(fused_concept - concept_query),
            "a_route_reliability": _as_numpy(route_reliability),
            "a_off_query_mass": _as_numpy(off_query_mass),
            "e_exact": _as_numpy(exact),
            "e_recent": _as_numpy(recent_feat),
            "e_group": _as_numpy(group_feat),
            "e_route_exact_delta": _as_numpy(route_exact - exact),
            "e_route_group_delta": _as_numpy(route_group - group_feat),
            "e_reliability": _as_numpy(query_rel),
            "e_route_reliability": _as_numpy(route_rel),
            "query_count": _as_numpy(query_count),
            "route_count": _as_numpy(route_count),
            "q_count": _as_numpy((qw > 0).float().sum(dim=1)),
        }
    )
    return frame, labels


def _sample_rows(x: np.ndarray, y: np.ndarray, max_rows: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if max_rows <= 0 or x.shape[0] <= max_rows:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=max_rows, replace=False)
    idx.sort()
    return x[idx], y[idx]


def _fit_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    steps: int,
    lr: float,
    nonnegative: bool,
    l2: float,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    x = torch.tensor(x_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32)
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False).clamp(min=1e-6)
    xs = (x - mean) / std
    raw_w = torch.zeros(xs.size(1), dtype=torch.float32, requires_grad=True)
    bias = torch.tensor(float(torch.logit(y.mean().clamp(1e-4, 1.0 - 1e-4)).item()), requires_grad=True)
    opt = torch.optim.Adam([raw_w, bias], lr=float(lr))
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        weight = F.softplus(raw_w) if nonnegative else raw_w
        loss = F.binary_cross_entropy_with_logits(xs.matmul(weight) + bias, y)
        if l2 > 0.0:
            loss = loss + float(l2) * weight.pow(2).mean()
        loss.backward()
        opt.step()
    weight = F.softplus(raw_w) if nonnegative else raw_w
    return (
        weight.detach().numpy().astype(np.float32),
        float(bias.detach().item()),
        mean.detach().numpy().astype(np.float32),
        std.detach().numpy().astype(np.float32),
    )


def _predict(x: np.ndarray, weight: np.ndarray, bias: float, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    xs = (x.astype(np.float32) - mean) / np.maximum(std, 1e-6)
    logits = xs.dot(weight.astype(np.float32)) + float(bias)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def _metrics(y: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    return compute_metrics(y, (prob >= 0.5).astype(np.float32), prob)


def _run_dataset(args: argparse.Namespace, dataset_name: str) -> Dict[str, object]:
    data_dir = Path(args.data_root) / dataset_name
    train_path = data_dir / "train.csv"
    valid_path = data_dir / "valid.csv"
    test_path = data_dir / "test.csv"
    train_loader, valid_loader, test_loader, info = create_dataloaders(
        str(train_path),
        str(valid_path),
        str(test_path),
        batch_size=args.batch_size,
        num_workers=0,
        shuffle_train=False,
        dataset_name=dataset_name,
        graph_prior_mode="evidence",
    )
    stat_tensors = _compute_train_stat_logits(
        train_loader,
        info["q_matrix"],
        num_students=info["num_students"],
        num_exercises=info["num_exercises"],
        num_concepts=info["num_concepts"],
    )
    (
        student_logits,
        _exercise_logits,
        concept_logits,
        _student_concept_logits,
        _recent_logits,
        _global_rate,
        _count_features,
    ) = stat_tensors
    train_df = train_loader.dataset.data.copy()
    response = build_response_transition_priors(
        train_df,
        info["cpt_id_map"],
        max_hops=args.max_hops,
        decay=args.decay,
        student_reliability_lambda=args.response_reliability_lambda,
    )
    fused_prior = _safe_fused_route(
        info["item_prior_matrix"],
        info["sequence_prior_matrix"],
        response["right_prior"],
        response["wrong_prior"],
        wrong_weight=args.wrong_weight,
    )
    student_groups = assign_student_quantile_groups(student_logits, num_groups=args.num_groups)
    group_logits = build_group_concept_logits(
        student_ids=train_loader.dataset.student_ids,
        exercise_ids=train_loader.dataset.exercise_ids,
        labels=train_loader.dataset.labels,
        q_matrix=info["q_matrix"],
        student_group_ids=student_groups,
        num_groups=args.num_groups,
        concept_rate=torch.sigmoid(concept_logits + torch.logit(torch.tensor(float(_global_rate)).clamp(1e-4, 1.0 - 1e-4))),
        smoothing=args.group_smoothing,
    )

    train_frame, train_y = _split_features(
        loader=train_loader,
        info_dict=info,
        stat_tensors=stat_tensors,
        response_priors=response,
        fused_prior=fused_prior,
        group_concept_logits=group_logits,
        reliability_lambda=args.e_reliability_lambda,
        shuffle_student_evidence=False,
        seed=args.seed,
    )
    valid_frame, valid_y = _split_features(
        loader=valid_loader,
        info_dict=info,
        stat_tensors=stat_tensors,
        response_priors=response,
        fused_prior=fused_prior,
        group_concept_logits=group_logits,
        reliability_lambda=args.e_reliability_lambda,
        shuffle_student_evidence=False,
        seed=args.seed,
    )
    test_frame, test_y = _split_features(
        loader=test_loader,
        info_dict=info,
        stat_tensors=stat_tensors,
        response_priors=response,
        fused_prior=fused_prior,
        group_concept_logits=group_logits,
        reliability_lambda=args.e_reliability_lambda,
        shuffle_student_evidence=False,
        seed=args.seed,
    )
    test_shuffle_frame, _ = _split_features(
        loader=test_loader,
        info_dict=info,
        stat_tensors=stat_tensors,
        response_priors=response,
        fused_prior=fused_prior,
        group_concept_logits=group_logits,
        reliability_lambda=args.e_reliability_lambda,
        shuffle_student_evidence=True,
        seed=args.seed,
    )

    feature_sets: Dict[str, Iterable[str]] = {
        "base": BASE_FEATURES,
        "base_plus_A": (*BASE_FEATURES, *A_FEATURES),
        "base_plus_E": (*BASE_FEATURES, *E_FEATURES),
        "base_plus_AE": (*BASE_FEATURES, *A_FEATURES, *E_FEATURES),
    }
    result: Dict[str, object] = {
        "dataset": dataset_name,
        "num_students": int(info["num_students"]),
        "num_exercises": int(info["num_exercises"]),
        "num_concepts": int(info["num_concepts"]),
        "train_size": int(info["train_size"]),
        "valid_size": int(info["val_size"]),
        "test_size": int(info["test_size"]),
        "graph_prior_stats": info.get("graph_prior_stats", {}),
        "response_stats": response["stats"],
        "feature_abs_mean_test": {
            name: float(test_frame[name].abs().mean())
            for name in (*A_FEATURES, *E_FEATURES, "query_count", "route_count")
        },
        "models": {},
    }

    for model_name, columns_iter in feature_sets.items():
        columns = list(columns_iter)
        x_train = train_frame[columns].to_numpy(dtype=np.float32)
        x_valid = valid_frame[columns].to_numpy(dtype=np.float32)
        x_test = test_frame[columns].to_numpy(dtype=np.float32)
        x_fit, y_fit = _sample_rows(x_train, train_y, args.max_train_rows, args.seed)
        weight, bias, mean, std = _fit_logistic(
            x_fit,
            y_fit,
            seed=args.seed,
            steps=args.steps,
            lr=args.lr,
            nonnegative=args.nonnegative,
            l2=args.l2,
        )
        valid_prob = _predict(x_valid, weight, bias, mean, std)
        test_prob = _predict(x_test, weight, bias, mean, std)
        model_result = {
            "features": columns,
            "weights": {col: float(w) for col, w in zip(columns, weight)},
            "bias": bias,
            "valid": _metrics(valid_y, valid_prob),
            "test": _metrics(test_y, test_prob),
        }
        if model_name in {"base_plus_E", "base_plus_AE"}:
            x_shuffle = test_shuffle_frame[columns].to_numpy(dtype=np.float32)
            shuffle_prob = _predict(x_shuffle, weight, bias, mean, std)
            model_result["test_shuffle_E"] = _metrics(test_y, shuffle_prob)
            model_result["test_drop_shuffle_E"] = float(model_result["test"]["auc"]) - float(
                model_result["test_shuffle_E"]["auc"]
            )
        result["models"][model_name] = model_result

    base_auc = float(result["models"]["base"]["test"]["auc"])
    result["auc_gains"] = {
        name: float(model["test"]["auc"]) - base_auc
        for name, model in result["models"].items()
        if name != "base"
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe RELM/PLRC train-only evidence before model wiring.")
    parser.add_argument("--datasets", nargs="+", default=["assist_09", "junyi", "assist_17", "nips34"])
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--max_train_rows", type=int, default=80000)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_hops", type=int, default=3)
    parser.add_argument("--decay", type=float, default=0.70)
    parser.add_argument("--response_reliability_lambda", type=float, default=5.0)
    parser.add_argument("--e_reliability_lambda", type=float, default=8.0)
    parser.add_argument("--wrong_weight", type=float, default=0.75)
    parser.add_argument("--num_groups", type=int, default=5)
    parser.add_argument("--group_smoothing", type=float, default=4.0)
    parser.add_argument("--nonnegative", action="store_true", help="Constrain logistic weights to be non-negative.")
    parser.add_argument("--print_full_json", action="store_true", help="Print full JSON to stdout instead of compact summary.")
    parser.add_argument("--output", default="results/reliable_map_probe/reliable_map_probe.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [_run_dataset(args, dataset) for dataset in args.datasets]
    text = json.dumps(rows, indent=2, ensure_ascii=False)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    if args.print_full_json:
        print(text)
    else:
        for row in rows:
            models = row["models"]
            parts = [f"{row['dataset']} base={models['base']['test']['auc']:.6f}"]
            for name in ("base_plus_A", "base_plus_E", "base_plus_AE"):
                parts.append(
                    f"{name}={models[name]['test']['auc']:.6f} "
                    f"(gain={row['auc_gains'][name]:+.6f})"
                )
            if "test_drop_shuffle_E" in models["base_plus_AE"]:
                parts.append(f"AE_shuffle_drop={models['base_plus_AE']['test_drop_shuffle_E']:+.6f}")
            print(" | ".join(parts))


if __name__ == "__main__":
    main()
