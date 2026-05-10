"""Probe an interpretable E calibration redesign before wiring it into training.

The probe is intentionally simple: it builds train-only student/concept
statistics, converts them into signed calibration features, then fits a
non-negative linear logistic calibrator.  It is a diagnostic tool, not the
production model path.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset import create_dataloaders  # noqa: E402
from src.experiment_utils import compute_metrics  # noqa: E402
from src.trainer import _compute_train_stat_logits  # noqa: E402


BASE_FEATURES = ("exercise_prior", "concept_prior")
E_FEATURES = (
    "student_global",
    "exact_mastery",
    "recent_mastery",
    "a_neighbor_mastery",
    "mastery_gap",
)


def _as_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().astype(np.float32)


def _row_normalize(mat: torch.Tensor) -> torch.Tensor:
    mat = mat.detach().float().clone()
    if mat.dim() != 2 or mat.size(0) != mat.size(1):
        return torch.zeros_like(mat)
    eye = torch.eye(mat.size(0), dtype=mat.dtype)
    mat = mat * (1.0 - eye)
    row_sum = mat.sum(dim=1, keepdim=True)
    return torch.where(row_sum > 0, mat / row_sum.clamp(min=1e-12), torch.zeros_like(mat))


def _build_fused_neighbor_prior(info_dict: Dict[str, object]) -> torch.Tensor:
    item = info_dict.get("item_prior_matrix")
    seq = info_dict.get("sequence_prior_matrix")
    pieces: List[torch.Tensor] = []
    if isinstance(item, torch.Tensor):
        pieces.append(item.detach().float())
    if isinstance(seq, torch.Tensor):
        pieces.append(seq.detach().float())
    if not pieces:
        c = int(info_dict["num_concepts"])
        return torch.zeros(c, c, dtype=torch.float32)
    return _row_normalize(sum(pieces))


def _make_query_weight(q_matrix: torch.Tensor, exercise_ids: torch.Tensor) -> torch.Tensor:
    q = q_matrix[exercise_ids.long()].float()
    return q / q.sum(dim=1, keepdim=True).clamp(min=1.0)


def _gather_features(
    *,
    loader,
    info_dict: Dict[str, object],
    stat_tensors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, Dict[str, torch.Tensor]],
    neighbor_prior: torch.Tensor,
    reliability_lambda: float,
    shuffle_student_evidence: bool,
    seed: int,
) -> Tuple[pd.DataFrame, np.ndarray]:
    dataset = loader.dataset
    student_ids = dataset.student_ids.detach().long().cpu()
    exercise_ids = dataset.exercise_ids.detach().long().cpu()
    labels = _as_numpy(dataset.labels.detach().float().cpu())
    q_matrix = info_dict["q_matrix"].detach().float().cpu()
    query_weight = _make_query_weight(q_matrix, exercise_ids)

    (
        student_logits,
        exercise_logits,
        concept_logits,
        student_concept_logits,
        student_concept_recent_logits,
        _global_rate,
        count_features,
    ) = stat_tensors
    student_logits = student_logits.detach().float().cpu()
    exercise_logits = exercise_logits.detach().float().cpu()
    concept_logits = concept_logits.detach().float().cpu()
    student_concept_logits = student_concept_logits.detach().float().cpu()
    student_concept_recent_logits = student_concept_recent_logits.detach().float().cpu()
    observed_counts = count_features["student_concept_observed"].detach().float().cpu()

    if shuffle_student_evidence and student_concept_logits.size(0) > 0:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + 7919)
        perm = torch.randperm(student_concept_logits.size(0), generator=generator)
        student_logits = student_logits[perm]
        student_concept_logits = student_concept_logits[perm]
        student_concept_recent_logits = student_concept_recent_logits[perm]
        observed_counts = observed_counts[perm]

    sc = student_concept_logits[student_ids]
    recent = student_concept_recent_logits[student_ids]
    counts = observed_counts[student_ids]
    graph_query_weight = query_weight.matmul(neighbor_prior.to(dtype=query_weight.dtype))

    query_count = (query_weight * counts).sum(dim=1)
    graph_count = (graph_query_weight * counts).sum(dim=1)
    rel = query_count / (query_count + float(reliability_lambda))
    graph_rel = graph_count / (graph_count + float(reliability_lambda))

    exact = rel * (query_weight * sc).sum(dim=1)
    recent_feat = rel * (query_weight * recent).sum(dim=1)
    a_neighbor = graph_rel * (graph_query_weight * sc).sum(dim=1)
    gap = rel * ((query_weight * sc).sum(dim=1) - (graph_query_weight * sc).sum(dim=1))
    student_global = student_logits[student_ids]
    exercise_prior = exercise_logits[exercise_ids]
    concept_prior = query_weight.matmul(concept_logits)

    frame = pd.DataFrame(
        {
            "student_global": _as_numpy(student_global),
            "exercise_prior": _as_numpy(exercise_prior),
            "concept_prior": _as_numpy(concept_prior),
            "exact_mastery": _as_numpy(exact),
            "recent_mastery": _as_numpy(recent_feat),
            "a_neighbor_mastery": _as_numpy(a_neighbor),
            "mastery_gap": _as_numpy(gap),
            "query_count": _as_numpy(query_count),
            "graph_count": _as_numpy(graph_count),
            "query_reliability": _as_numpy(rel),
            "graph_reliability": _as_numpy(graph_rel),
            "q_count": _as_numpy(query_weight.gt(0).sum(dim=1)).astype(np.int32),
        }
    )
    return frame, labels


def _select_rows(x: np.ndarray, y: np.ndarray, max_rows: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if max_rows <= 0 or x.shape[0] <= max_rows:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=max_rows, replace=False)
    idx.sort()
    return x[idx], y[idx]


def _fit_nonnegative_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    steps: int,
    lr: float,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    x = torch.tensor(x_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32)
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False).clamp(min=1e-6)
    xs = (x - mean) / std
    raw_w = torch.zeros(xs.size(1), dtype=torch.float32, requires_grad=True)
    bias = torch.tensor(float(torch.logit(y.mean().clamp(1e-4, 1.0 - 1e-4)).item()), requires_grad=True)
    opt = torch.optim.Adam([raw_w, bias], lr=lr)
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        weight = F.softplus(raw_w)
        loss = F.binary_cross_entropy_with_logits(xs.matmul(weight) + bias, y)
        loss.backward()
        opt.step()
    return (
        F.softplus(raw_w).detach().numpy().astype(np.float32),
        float(bias.detach().item()),
        mean.detach().numpy().astype(np.float32),
        std.detach().numpy().astype(np.float32),
    )


def _predict(x: np.ndarray, weight: np.ndarray, bias: float, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    xs = (x.astype(np.float32) - mean) / np.maximum(std, 1e-6)
    logits = xs.dot(weight.astype(np.float32)) + float(bias)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def _evaluate(labels: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    return compute_metrics(labels, (probs >= 0.5).astype(np.float32), probs)


def _run_probe(args: argparse.Namespace, dataset_name: str) -> Dict[str, object]:
    data_dir = Path(args.data_root) / dataset_name
    train_loader, valid_loader, test_loader, info_dict = create_dataloaders(
        str(data_dir / "train.csv"),
        str(data_dir / "valid.csv"),
        str(data_dir / "test.csv"),
        batch_size=args.batch_size,
        num_workers=0,
        shuffle_train=False,
        dataset_name=dataset_name,
        graph_prior_mode=args.graph_prior_mode,
    )
    stat_tensors = _compute_train_stat_logits(
        train_loader,
        info_dict["q_matrix"],
        num_students=info_dict["num_students"],
        num_exercises=info_dict["num_exercises"],
        num_concepts=info_dict["num_concepts"],
    )
    neighbor_prior = _build_fused_neighbor_prior(info_dict)

    train_frame, train_y = _gather_features(
        loader=train_loader,
        info_dict=info_dict,
        stat_tensors=stat_tensors,
        neighbor_prior=neighbor_prior,
        reliability_lambda=args.reliability_lambda,
        shuffle_student_evidence=False,
        seed=args.seed,
    )
    valid_frame, valid_y = _gather_features(
        loader=valid_loader,
        info_dict=info_dict,
        stat_tensors=stat_tensors,
        neighbor_prior=neighbor_prior,
        reliability_lambda=args.reliability_lambda,
        shuffle_student_evidence=False,
        seed=args.seed,
    )
    test_frame, test_y = _gather_features(
        loader=test_loader,
        info_dict=info_dict,
        stat_tensors=stat_tensors,
        neighbor_prior=neighbor_prior,
        reliability_lambda=args.reliability_lambda,
        shuffle_student_evidence=False,
        seed=args.seed,
    )
    test_shuffle_frame, _ = _gather_features(
        loader=test_loader,
        info_dict=info_dict,
        stat_tensors=stat_tensors,
        neighbor_prior=neighbor_prior,
        reliability_lambda=args.reliability_lambda,
        shuffle_student_evidence=True,
        seed=args.seed,
    )

    feature_sets: Dict[str, Iterable[str]] = {
        "base_item_concept": BASE_FEATURES,
        "base_plus_E": (*BASE_FEATURES, *E_FEATURES),
    }
    results: Dict[str, object] = {
        "dataset": dataset_name,
        "num_students": int(info_dict["num_students"]),
        "num_exercises": int(info_dict["num_exercises"]),
        "num_concepts": int(info_dict["num_concepts"]),
        "train_size": int(info_dict["train_size"]),
        "valid_size": int(info_dict["val_size"]),
        "test_size": int(info_dict["test_size"]),
        "feature_abs_mean_test": {
            name: float(test_frame[name].abs().mean())
            for name in (*BASE_FEATURES, *E_FEATURES, "query_count", "graph_count")
        },
        "feature_nonzero_rate_test": {
            name: float((test_frame[name].abs() > 1e-8).mean())
            for name in (*E_FEATURES, "query_count", "graph_count")
        },
        "models": {},
    }

    for model_name, cols_iter in feature_sets.items():
        cols = list(cols_iter)
        x_train = train_frame[cols].to_numpy(dtype=np.float32)
        x_valid = valid_frame[cols].to_numpy(dtype=np.float32)
        x_test = test_frame[cols].to_numpy(dtype=np.float32)
        x_fit, y_fit = _select_rows(x_train, train_y, args.max_train_rows, args.seed)
        weight, bias, mean, std = _fit_nonnegative_logistic(
            x_fit,
            y_fit,
            seed=args.seed,
            steps=args.steps,
            lr=args.lr,
        )
        valid_probs = _predict(x_valid, weight, bias, mean, std)
        test_probs = _predict(x_test, weight, bias, mean, std)
        model_result = {
            "features": cols,
            "weights": {col: float(w) for col, w in zip(cols, weight)},
            "bias": bias,
            "valid": _evaluate(valid_y, valid_probs),
            "test": _evaluate(test_y, test_probs),
        }
        if model_name == "base_plus_E":
            x_shuffle = test_shuffle_frame[cols].to_numpy(dtype=np.float32)
            shuffle_probs = _predict(x_shuffle, weight, bias, mean, std)
            model_result["test_shuffle_E"] = _evaluate(test_y, shuffle_probs)
            model_result["test_drop_shuffle_E"] = (
                float(model_result["test"]["auc"]) - float(model_result["test_shuffle_E"]["auc"])
            )
        results["models"][model_name] = model_result

    base_auc = float(results["models"]["base_item_concept"]["test"]["auc"])
    e_auc = float(results["models"]["base_plus_E"]["test"]["auc"])
    results["test_auc_gain_base_plus_E"] = e_auc - base_auc
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe interpretable student evidence calibration features.")
    parser.add_argument("--datasets", nargs="+", default=["assist_09", "junyi"])
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--graph_prior_mode", default="evidence")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--reliability_lambda", type=float, default=8.0)
    parser.add_argument("--max_train_rows", type=int, default=60000)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [_run_probe(args, dataset) for dataset in args.datasets]
    text = json.dumps(rows, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
