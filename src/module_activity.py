"""Graph-IRT runtime activity diagnostics.

The activity report intentionally covers only the production path: the global
concept graph and Q-aware IRT head. It is lightweight enough to run at
checkpoint intervals and contains no experimental side heads.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model import CognitiveDiagnosisModel, GRAPH_IRT_ARCHITECTURE


def _get_base_model(model) -> CognitiveDiagnosisModel:
    """Return the underlying model when DataParallel is used."""
    return model.module if isinstance(model, nn.DataParallel) else model


def _as_values(value: Any) -> List[float]:
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return []
    return value.detach().float().reshape(-1).cpu().tolist()


def _mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _relation_learning(base_model: CognitiveDiagnosisModel) -> Optional[nn.Module]:
    return getattr(base_model, "relation_learning", None)


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, (np.bool_, np.integer)):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, list):
        return [_to_serializable(value) for value in obj]
    if isinstance(obj, dict):
        return {key: _to_serializable(value) for key, value in obj.items()}
    return obj


def compute_module_activity(
    model,
    data_loader: DataLoader,
    device: torch.device,
    num_samples: int = 500,
) -> Dict[str, Any]:
    """Measure whether the concept graph and IRT head are active."""
    base_model = _get_base_model(model)
    was_training = model.training
    model.eval()

    irt_logits: List[float] = []
    theta_values: List[float] = []
    discrimination_values: List[float] = []
    difficulty_values: List[float] = []
    graph_state_deltas: List[float] = []
    gec_state_adjustments: List[float] = []
    gec_evidence_adjustments: List[float] = []
    relation_identity_deltas: List[float] = []
    evidence_relation_snapshot: Optional[torch.Tensor] = None

    sample_count = 0
    with torch.no_grad():
        for student_ids, exercise_ids, _ in data_loader:
            if sample_count >= num_samples:
                break
            take = min(int(student_ids.size(0)), int(num_samples) - sample_count)
            student_ids = student_ids[:take].to(device)
            exercise_ids = exercise_ids[:take].to(device)
            _, details = model(
                student_ids,
                exercise_ids,
                return_details=True,
                return_logits=True,
            )
            irt_logits.extend(_as_values(details.get("irt_logit")))
            theta_values.extend(_as_values(details.get("theta_c")))
            discrimination_values.extend(_as_values(details.get("irt_a")))
            difficulty_values.extend(_as_values(details.get("irt_b")))
            graph_state_deltas.extend(_as_values(details.get("knowledge_state_graph_delta")))
            gec_state_adjustments.extend(_as_values(details.get("gec_state_adjustment")))
            gec_evidence_adjustments.extend(
                _as_values(details.get("gec_evidence_adjustment"))
            )
            relation_identity_deltas.extend(_as_values(details.get("relation_identity_delta")))
            evidence_relations = details.get("evidence_relation_matrices")
            if isinstance(evidence_relations, torch.Tensor):
                evidence_relation_snapshot = evidence_relations.detach()
            sample_count += take

    relation_learning = _relation_learning(base_model)
    propagation_alpha = float(base_model.knowledge_encoder.propagation_alpha)
    results: Dict[str, Any] = {
        "architecture": GRAPH_IRT_ARCHITECTURE,
        "num_activity_samples": min(sample_count, int(num_samples)),
        "graph_enabled": bool(relation_learning is not None),
        "message_passing_alpha": propagation_alpha,
        "message_passing_enabled": propagation_alpha > 0.0,
        "irt_enabled": True,
        "irt_logit_abs_mean": _mean([abs(value) for value in irt_logits]),
        "irt_theta_abs_mean": _mean([abs(value) for value in theta_values]),
        "irt_discrimination_mean": _mean(discrimination_values),
        "irt_difficulty_mean": _mean(difficulty_values),
        "knowledge_state_graph_delta": _mean(graph_state_deltas),
        "gec_state_adjustment_abs_mean": _mean(
            [abs(value) for value in gec_state_adjustments]
        ),
        "gec_evidence_adjustment_abs_mean": _mean(
            [abs(value) for value in gec_evidence_adjustments]
        ),
        "relation_identity_delta": _mean(relation_identity_deltas),
    }

    if results["graph_enabled"]:
        with torch.no_grad():
            if (
                not results["message_passing_enabled"]
                and evidence_relation_snapshot is not None
                and results["gec_evidence_adjustment_abs_mean"] > 1e-8
            ):
                relation_matrices = base_model.evidence_router.off_diagonal_relations(
                    evidence_relation_snapshot
                )
            else:
                relation_matrices = relation_learning()
            matrices = relation_matrices.detach().float().cpu().numpy()
        eps = 1e-12
        row_entropies = -np.sum(matrices * np.log(matrices + eps), axis=-1)
        mean_entropy = float(row_entropies.mean())
        max_entropy = float(np.log(matrices.shape[-1])) if matrices.shape[-1] > 1 else 0.0
        entropy_ratio = mean_entropy / max_entropy if max_entropy > 0.0 else 0.0
        diagonal_mass = float(np.diagonal(matrices, axis1=-2, axis2=-1).mean())
        uniform_distance = float(
            np.sqrt(np.sum((matrices - (1.0 / matrices.shape[-1])) ** 2, axis=-1)).mean()
        )

        graph_trivial = bool(entropy_ratio > 0.98)
        graph_over_sparse = bool(diagonal_mass > 0.98)
        graph_active = bool(
            (
                results["message_passing_enabled"]
                or results["gec_evidence_adjustment_abs_mean"] > 1e-8
            )
            and not graph_trivial
            and not graph_over_sparse
            and (
                results["knowledge_state_graph_delta"] > 1e-6
                or results["gec_state_adjustment_abs_mean"] > 1e-8
                or results["gec_evidence_adjustment_abs_mean"] > 1e-8
            )
        )
        if (
            not results["message_passing_enabled"]
            and results["gec_evidence_adjustment_abs_mean"] <= 1e-8
        ):
            graph_mode = "NO_MESSAGE_PASSING"
        elif graph_trivial:
            graph_mode = "UNIFORM"
        elif graph_over_sparse:
            graph_mode = "IDENTITY"
        elif graph_active:
            graph_mode = "LIVE"
        else:
            graph_mode = "INACTIVE"

        results.update(
            {
                "graph_mean_row_entropy": mean_entropy,
                "graph_max_row_entropy": max_entropy,
                "graph_entropy_ratio": float(entropy_ratio),
                "graph_diagonal_mass": diagonal_mass,
                "graph_to_uniform_l2": uniform_distance,
                "graph_trivial": graph_trivial,
                "graph_over_sparse": graph_over_sparse,
                "graph_active": graph_active,
                "graph_mode": graph_mode,
            }
        )
    else:
        results.update(
            {
                "graph_mean_row_entropy": 0.0,
                "graph_max_row_entropy": 0.0,
                "graph_entropy_ratio": 0.0,
                "graph_diagonal_mass": 0.0,
                "graph_to_uniform_l2": 0.0,
                "graph_trivial": False,
                "graph_over_sparse": False,
                "graph_active": False,
                "graph_mode": "DISABLED",
            }
        )

    results["irt_active"] = bool(
        np.isfinite(results["irt_logit_abs_mean"])
        and np.isfinite(results["irt_discrimination_mean"])
        and results["irt_discrimination_mean"] > 0.0
    )
    if was_training:
        model.train()
    return _to_serializable(results)


def format_activity_brief(activity: Dict[str, Any]) -> str:
    """Format a compact checkpoint-time activity summary."""
    graph_mode = str(activity.get("graph_mode", "DISABLED"))
    irt_mode = "LIVE" if activity.get("irt_active") else "INACTIVE"
    return f"ConceptGraph[{graph_mode}] IRT[{irt_mode}]"


def format_activity_report(
    activity: Dict[str, Any],
    dataset_name: str = "unknown",
    seed: int = 42,
    epoch: int = 0,
) -> str:
    """Format the final Graph-IRT activity report."""
    lines = [
        "=" * 60,
        "             GRAPH-IRT ACTIVITY REPORT",
        "=" * 60,
        f"Dataset: {dataset_name} | Seed: {seed} | Epoch: {epoch}",
        "",
        "1. Concept graph:",
    ]
    if activity.get("graph_enabled"):
        lines.extend(
            [
                f"   - Mode: {activity.get('graph_mode', 'INACTIVE')}",
                f"   - Entropy ratio: {activity.get('graph_entropy_ratio', 0.0):.1%}",
                f"   - Diagonal mass: {activity.get('graph_diagonal_mass', 0.0):.4f}",
                f"   - Relation/identity delta: {activity.get('relation_identity_delta', 0.0):.6f}",
                f"   - Message-passing alpha: {activity.get('message_passing_alpha', 0.0):.4f}",
                f"   - Knowledge-state graph delta: {activity.get('knowledge_state_graph_delta', 0.0):.6f}",
            ]
        )
    else:
        lines.append("   - Mode: DISABLED")

    lines.extend(
        [
            "",
            "2. IRT head:",
            f"   - Status: {'LIVE' if activity.get('irt_active') else 'INACTIVE'}",
            f"   - Logit |mean|: {activity.get('irt_logit_abs_mean', 0.0):.4f}",
            f"   - Theta |mean|: {activity.get('irt_theta_abs_mean', 0.0):.4f}",
            f"   - Discrimination mean: {activity.get('irt_discrimination_mean', 0.0):.4f}",
            f"   - Difficulty mean: {activity.get('irt_difficulty_mean', 0.0):.4f}",
            "=" * 60,
        ]
    )
    return "\n".join(lines)
