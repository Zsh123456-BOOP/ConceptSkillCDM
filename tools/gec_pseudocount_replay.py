#!/usr/bin/env python
"""Validation-only replay of support-aware graph pseudo-count fusion.

The production model is not modified.  A frozen Full checkpoint is replayed
with a train-only student co-exposure graph, a degree-matched random graph, or
no graph.  Old propagated anchors are either retained or removed.  This tool
opens ``valid.csv`` only; it never opens a test split.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from src.dataset import CognitiveDiagnosisDataset  # noqa: E402
from src.trainer import (  # noqa: E402
    STRICT_CHECKPOINT_LOADING,
    _build_model,
    _require_graph_irt_checkpoint,
    _strip_module_prefix,
    _validate_checkpoint_data_identity,
)
from tools.evaluate_graph_validation_buckets import _bucket_masks, _metrics  # noqa: E402
from tools.summarize_validation_runs import _candidate_dirs, _tokens  # noqa: E402


CONTEXTS = ("retain_old", "replace_old")
GRAPH_NAMES = ("none", "exposure", "degree_random")
EPS = 1e-4
REPLAY_TOLERANCE = 1e-6


def _row_topk_graph(prior: torch.Tensor, topk: int) -> torch.Tensor:
    graph = torch.as_tensor(prior).detach().float().cpu().clone()
    if graph.dim() != 2 or graph.size(0) != graph.size(1):
        raise ValueError(f"exposure prior must be square, got {tuple(graph.shape)}")
    count = int(graph.size(0))
    if count <= 1:
        return torch.zeros_like(graph)
    graph.clamp_(min=0.0)
    graph.fill_diagonal_(0.0)
    k = min(max(1, int(topk)), count - 1)
    if k < count - 1:
        values, indices = torch.topk(graph, k=k, dim=-1)
        kept = torch.zeros_like(graph)
        kept.scatter_(1, indices, values)
        graph = kept
    row_sum = graph.sum(dim=-1, keepdim=True)
    if bool((row_sum <= 0.0).any()):
        empty = torch.nonzero(row_sum.squeeze(-1) <= 0.0).reshape(-1)
        raise ValueError(f"exposure graph has empty rows: {empty[:10].tolist()}")
    return graph / row_sum


def _degree_matched_random_graph(
    graph: torch.Tensor,
    seed: int,
) -> Tuple[torch.Tensor, int]:
    """Break semantic endpoints while preserving in/out degree and row weights."""
    graph = graph.detach().float().cpu()
    support = graph > 0.0
    edges = [tuple(pair) for pair in torch.nonzero(support).tolist()]
    edge_set = set(edges)
    rng = random.Random(int(seed))
    target_swaps = max(10 * len(edges), 100)
    max_attempts = 30 * target_swaps
    successful = 0

    for _ in range(max_attempts):
        if successful >= target_swaps or len(edges) < 2:
            break
        first = rng.randrange(len(edges))
        second = rng.randrange(len(edges) - 1)
        if second >= first:
            second += 1
        source_a, target_a = edges[first]
        source_b, target_b = edges[second]
        new_a = (source_a, target_b)
        new_b = (source_b, target_a)
        if (
            source_a == source_b
            or target_a == target_b
            or source_a == target_b
            or source_b == target_a
            or new_a in edge_set
            or new_b in edge_set
        ):
            continue
        edge_set.remove(edges[first])
        edge_set.remove(edges[second])
        edge_set.add(new_a)
        edge_set.add(new_b)
        edges[first], edges[second] = new_a, new_b
        successful += 1

    random_support = torch.zeros_like(support)
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long)
        random_support[edge_index[:, 0], edge_index[:, 1]] = True
    randomized = torch.zeros_like(graph)
    for row in range(int(graph.size(0))):
        weights = graph[row, support[row]].tolist()
        columns = torch.nonzero(random_support[row]).reshape(-1).tolist()
        rng.shuffle(weights)
        randomized[row, columns] = torch.tensor(weights, dtype=graph.dtype)
    return randomized, successful


def build_fixed_graphs(
    exposure_prior: torch.Tensor,
    *,
    topk: int,
    random_seed: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    exposure = _row_topk_graph(exposure_prior, topk)
    randomized, swaps = _degree_matched_random_graph(exposure, random_seed)
    zero = torch.zeros_like(exposure)
    real_support = exposure > 0.0
    random_support = randomized > 0.0
    if not torch.equal(real_support.sum(1), random_support.sum(1)):
        raise RuntimeError("random graph does not preserve row degree")
    if not torch.equal(real_support.sum(0), random_support.sum(0)):
        raise RuntimeError("random graph does not preserve column degree")
    if bool(torch.diagonal(randomized).any()):
        raise RuntimeError("random graph contains self-loops")
    overlap = float((real_support & random_support).sum().item()) / float(
        real_support.sum().item()
    )
    if overlap >= 0.75:
        raise RuntimeError(f"random graph changed too few endpoints: overlap={overlap:.3f}")
    return {
        "none": zero,
        "exposure": exposure,
        "degree_random": randomized,
    }, {
        "graph_edges": float(real_support.sum().item()),
        "random_edge_overlap": overlap,
        "random_successful_swaps": float(swaps),
    }


def fuse_pseudocount(
    *,
    rate_evidence: torch.Tensor,
    correct: torch.Tensor,
    count: torch.Tensor,
    concept_rate: torch.Tensor,
    graph: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add at most one graph-supported empirical-Bayes pseudo-response."""
    graph = graph.to(device=count.device, dtype=count.dtype)
    has_edges = graph._nnz() > 0 if graph.is_sparse else bool((graph > 0.0).any())
    if not has_edges:
        return rate_evidence, torch.zeros_like(count)

    def project(values: torch.Tensor) -> torch.Tensor:
        if graph.is_sparse:
            return torch.sparse.mm(graph, values.transpose(0, 1)).transpose(0, 1)
        return values.matmul(graph.transpose(0, 1))

    reliability = count / (count + 1.0)
    pseudo_support = project(reliability)
    graph_numerator = project(rate_evidence)
    if float(pseudo_support.max().item()) > 1.0 + 1e-5:
        raise RuntimeError("graph pseudo-support exceeds one observation")

    graph_deviation = torch.where(
        pseudo_support > 0.0,
        graph_numerator / pseudo_support.clamp(min=1e-12),
        torch.zeros_like(graph_numerator),
    ).clamp(min=-4.0, max=4.0)
    prior = concept_rate.to(device=count.device, dtype=count.dtype).clamp(
        min=EPS,
        max=1.0 - EPS,
    )
    pseudo_rate = torch.sigmoid(torch.logit(prior) + graph_deviation)
    effective_count = count + pseudo_support
    posterior = (correct + prior + pseudo_support * pseudo_rate) / (
        effective_count + 1.0
    )
    fused = (
        (torch.logit(posterior.clamp(min=EPS, max=1.0 - EPS)) - torch.logit(prior))
        * effective_count
        / (effective_count + 1.0)
    ).clamp(min=-4.0, max=4.0)
    fused = torch.where(pseudo_support > 0.0, fused, rate_evidence)
    return fused, pseudo_support


def load_validation_context(
    checkpoint_dir: Path,
    *,
    batch_size: int,
    device: torch.device,
    random_seed: int,
) -> Dict[str, object]:
    checkpoint_path = checkpoint_dir / "best_model.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _require_graph_irt_checkpoint(checkpoint, str(checkpoint_path))
    loaded_args = checkpoint.get("args", {})
    info_dict = checkpoint.get("info_dict")
    if not isinstance(info_dict, dict):
        raise RuntimeError(f"checkpoint is missing info_dict: {checkpoint_path}")
    required_contract = {
        "model_variant": "full",
        "prediction_head": "irt2pl",
        "evidence_anchor_mode": "full",
    }
    for key, expected in required_contract.items():
        actual = str(loaded_args.get(key, expected))
        if actual != expected:
            raise RuntimeError(f"{key} must be {expected!r}, got {actual!r}")
    if bool(loaded_args.get("evidence_state_injection", True)):
        raise RuntimeError("replay requires evidence_state_injection=False")

    data_dir, _ = _validate_checkpoint_data_identity(
        SimpleNamespace(explicit_arg_dests=[]),
        loaded_args,
        info_dict,
    )
    valid = pd.read_csv(Path(data_dir) / "valid.csv")
    required_columns = {"stu_id", "exer_id", "label"}
    missing = sorted(required_columns - set(valid.columns))
    if missing:
        raise ValueError(f"validation split is missing columns {missing}")
    valid = valid[
        valid["stu_id"].isin(info_dict["stu_id_map"])
        & valid["exer_id"].isin(info_dict["exer_id_map"])
    ].reset_index(drop=True)
    if valid.empty:
        raise ValueError("validation split has no train-seen rows")
    dataset = CognitiveDiagnosisDataset(
        valid,
        info_dict["stu_id_map"],
        info_dict["exer_id_map"],
        info_dict["cpt_id_map"],
    )
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=0,
    )

    model = _build_model(loaded_args, info_dict, device)
    incompatible = model.load_state_dict(
        _strip_module_prefix(checkpoint["model_state_dict"]),
        strict=STRICT_CHECKPOINT_LOADING,
    )
    if not STRICT_CHECKPOINT_LOADING and (
        incompatible.missing_keys or incompatible.unexpected_keys
    ):
        raise RuntimeError(f"checkpoint mismatch: {incompatible}")
    if hasattr(model, "set_epoch"):
        model.set_epoch(int(checkpoint.get("epoch", 1)))
    model.eval()

    topk = int(loaded_args.get("graph_topk") or 0)
    if topk <= 0:
        raise RuntimeError("replay requires a positive checkpoint graph_topk")
    exposure_prior = info_dict.get("exposure_prior_matrix")
    if exposure_prior is None:
        raise RuntimeError("checkpoint is missing the train-only exposure prior")
    graphs, graph_stats = build_fixed_graphs(
        exposure_prior,
        topk=topk,
        random_seed=random_seed,
    )
    return {
        "checkpoint": checkpoint,
        "checkpoint_dir": checkpoint_dir,
        "loaded_args": loaded_args,
        "model": model,
        "loader": loader,
        "graphs": {
            name: (
                graph.to_sparse_coo().coalesce().to(device)
                if graph.size(0) >= 192
                else graph.to(device)
            )
            for name, graph in graphs.items()
        },
        "graph_stats": graph_stats,
    }


def replay_batch(
    model,
    student_ids: torch.Tensor,
    exercise_ids: torch.Tensor,
    graphs: Mapping[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], float]:
    base_logits, details = model(
        student_ids,
        exercise_ids,
        return_details=True,
        return_logits=True,
    )
    old_anchor = details["evidence_anchor"]
    if old_anchor.size(-1) <= 2:
        raise RuntimeError("Full checkpoint has no propagated anchor channels")
    original_replay = model.diagnosis_head(
        knowledge_state=details["knowledge_state"],
        concept_mask=details["q_vector"],
        b=details["irt_b"],
        a=details["irt_a"],
        evidence_anchor=old_anchor,
    )
    replay_error = float(
        (original_replay.reshape(-1) - base_logits.reshape(-1)).abs().max().item()
    )
    if replay_error > REPLAY_TOLERANCE:
        raise RuntimeError(f"frozen diagnosis replay error is {replay_error:.3e}")

    count = model.response_student_concept_count[student_ids]
    correct = model.response_student_concept_correct[student_ids]
    global_rate = model.response_global_correct / model.response_global_count
    concept_rate = (
        model.response_concept_correct + global_rate
    ) / (model.response_concept_count + 1.0)
    rate_evidence = details["response_evidence"][..., 0]
    gate_parameters = model.anchor_gate[0]

    outputs: Dict[str, torch.Tensor] = {}
    supports: Dict[str, torch.Tensor] = {}
    for graph_name in GRAPH_NAMES:
        fused, pseudo_support = fuse_pseudocount(
            rate_evidence=rate_evidence,
            correct=correct,
            count=count,
            concept_rate=concept_rate,
            graph=graphs[graph_name],
        )
        # Keep the frozen checkpoint gate exactly on its trained count input.
        gate_input = torch.log1p(count)
        gate = torch.sigmoid(
            gate_parameters[0] + gate_parameters[1] * gate_input
        )
        direct_anchor = fused * gate
        supports[graph_name] = pseudo_support
        for context in CONTEXTS:
            propagated = (
                old_anchor[..., 2:]
                if context == "retain_old"
                else torch.zeros_like(old_anchor[..., 2:])
            )
            candidate_anchor = torch.cat(
                (direct_anchor.unsqueeze(-1), old_anchor[..., 1:2], propagated),
                dim=-1,
            )
            key = f"{context}__{graph_name}"
            outputs[key] = model.diagnosis_head(
                knowledge_state=details["knowledge_state"],
                concept_mask=details["q_vector"],
                b=details["irt_b"],
                a=details["irt_a"],
                evidence_anchor=candidate_anchor,
            )

    baseline_error = float(
        (outputs["retain_old__none"] - base_logits.reshape(-1)).abs().max().item()
    )
    if baseline_error > REPLAY_TOLERANCE:
        raise RuntimeError(f"no-graph baseline error is {baseline_error:.3e}")
    return outputs, supports, max(replay_error, baseline_error)


def evaluate_checkpoint(context: Mapping[str, object], device: torch.device) -> List[Dict[str, object]]:
    model = context["model"]
    loader = context["loader"]
    graphs = context["graphs"]
    checkpoint = context["checkpoint"]
    loaded_args = context["loaded_args"]
    labels: List[torch.Tensor] = []
    supports: List[torch.Tensor] = []
    probabilities: Dict[str, List[torch.Tensor]] = {
        f"{context_name}__{graph_name}": []
        for context_name in CONTEXTS
        for graph_name in GRAPH_NAMES
    }
    pseudo_supports: Dict[str, List[torch.Tensor]] = {
        name: [] for name in GRAPH_NAMES
    }
    max_replay_error = 0.0

    with torch.no_grad():
        for student_ids, exercise_ids, y in loader:
            student_ids = student_ids.to(device)
            exercise_ids = exercise_ids.to(device)
            batch_outputs, batch_pseudo_supports, error = replay_batch(
                model,
                student_ids,
                exercise_ids,
                graphs,
            )
            q_mask = model.q_matrix[exercise_ids] > 0
            count = model.response_student_concept_count[student_ids]
            support = torch.where(
                q_mask,
                count,
                torch.full_like(count, float("inf")),
            ).min(dim=1).values
            labels.append(y.reshape(-1).cpu())
            supports.append(support.cpu())
            for key, logits in batch_outputs.items():
                probabilities[key].append(torch.sigmoid(logits).cpu())
            for name, values in batch_pseudo_supports.items():
                q_denom = q_mask.sum(dim=1).clamp(min=1)
                query_mean = (values * q_mask).sum(dim=1) / q_denom
                pseudo_supports[name].append(query_mean.cpu())
            max_replay_error = max(max_replay_error, error)

    label_array = torch.cat(labels).numpy().astype(np.float64)
    support_array = torch.cat(supports).numpy().astype(np.float64)
    probability_arrays = {
        key: torch.cat(chunks).numpy().astype(np.float64)
        for key, chunks in probabilities.items()
    }
    pseudo_arrays = {
        key: torch.cat(chunks).numpy().astype(np.float64)
        for key, chunks in pseudo_supports.items()
    }
    identity = {
        "run_dir": str(context["checkpoint_dir"]),
        "dataset": str(loaded_args.get("dataset_name", "")),
        "seed": int(loaded_args.get("seed", 0)),
        "best_epoch": int(checkpoint.get("epoch", 0)),
        "test_evaluated": False,
        "max_replay_error": max_replay_error,
        **context["graph_stats"],
    }
    rows: List[Dict[str, object]] = []
    for key, probs in probability_arrays.items():
        context_name, graph_name = key.split("__", maxsplit=1)
        for bucket, mask in _bucket_masks(support_array).items():
            bucket_labels = label_array[mask]
            rows.append(
                {
                    **identity,
                    "context": context_name,
                    "graph": graph_name,
                    "bucket": bucket,
                    "rows": int(mask.sum()),
                    "positives": int((bucket_labels == 1).sum()),
                    "negatives": int((bucket_labels == 0).sum()),
                    "mean_pseudo_support": float(pseudo_arrays[graph_name][mask].mean())
                    if bool(mask.any())
                    else float("nan"),
                    **_metrics(bucket_labels, probs[mask]),
                }
            )
    return rows


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_ids", default="")
    parser.add_argument("--checkpoint_dirs", default="")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    checkpoint_dirs = _candidate_dirs(
        _tokens(args.run_ids),
        _tokens(args.checkpoint_dirs),
    )
    if not checkpoint_dirs:
        raise FileNotFoundError("no matching checkpoint directories")
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    rows: List[Dict[str, object]] = []
    for index, checkpoint_dir in enumerate(checkpoint_dirs, start=1):
        context = load_validation_context(
            checkpoint_dir,
            batch_size=args.batch_size,
            device=device,
            random_seed=args.random_seed,
        )
        evaluated = evaluate_checkpoint(context, device)
        rows.extend(evaluated)
        baseline = next(
            row
            for row in evaluated
            if row["context"] == "retain_old"
            and row["graph"] == "none"
            and row["bucket"] == "all"
        )
        print(
            f"[{index}/{len(checkpoint_dirs)}] {baseline['dataset']} "
            f"seed={baseline['seed']} baseline_auc={baseline['auc']:.9f}"
        )
        del context
        if device.type == "cuda":
            torch.cuda.empty_cache()

    frame = pd.DataFrame(rows).sort_values(
        ["dataset", "seed", "context", "graph", "bucket"]
    )
    output = _resolve(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"rows={len(frame)} -> {output}")


if __name__ == "__main__":
    main()
