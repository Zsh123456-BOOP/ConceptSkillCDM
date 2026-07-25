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


def _empty_tensor_stats() -> Dict[str, Any]:
    return {
        "sum": 0.0,
        "abs_sum": 0.0,
        "count": 0,
        "abs_max": 0.0,
        "finite": True,
    }


def _update_tensor_stats(stats: Dict[str, Any], value: Any) -> None:
    """Accumulate tensor diagnostics without retaining large graph tensors."""
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return
    detached = value.detach().float()
    finite = torch.isfinite(detached)
    stats["finite"] = bool(stats["finite"] and finite.all().item())
    safe = torch.nan_to_num(detached, nan=0.0, posinf=0.0, neginf=0.0)
    stats["sum"] += float(safe.sum().item())
    stats["abs_sum"] += float(safe.abs().sum().item())
    stats["count"] += int(safe.numel())
    stats["abs_max"] = max(stats["abs_max"], float(safe.abs().max().item()))


def _stats_mean(stats: Dict[str, Any], *, absolute: bool = False) -> float:
    count = max(0, int(stats.get("count", 0)))
    if count == 0:
        return 0.0
    key = "abs_sum" if absolute else "sum"
    return float(stats.get(key, 0.0)) / float(count)


def _first_tensor(details: Dict[str, Any], *keys: str) -> Optional[torch.Tensor]:
    for key in keys:
        value = details.get(key)
        if isinstance(value, torch.Tensor):
            return value
    return None


def _relation_learning(base_model: CognitiveDiagnosisModel) -> Optional[nn.Module]:
    return getattr(base_model, "relation_learning", None)


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        value = obj.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
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


def _residual_parameter_summary(
    residual_module: Optional[nn.Module],
) -> Dict[str, Any]:
    """Prefer a module-owned summary and fill only missing compatibility keys."""
    if residual_module is None:
        return {}

    summary: Dict[str, Any] = {}
    summary_fn = getattr(residual_module, "parameter_summary", None)
    if callable(summary_fn):
        try:
            reported = summary_fn()
            if isinstance(reported, dict):
                summary.update(_to_serializable(reported))
            else:
                summary["summary_error"] = (
                    "parameter_summary() did not return a dictionary"
                )
        except Exception as exc:  # diagnostics must not break checkpointing
            summary["summary_error"] = str(exc)

    parameters = tuple(residual_module.parameters())
    summary.setdefault(
        "enabled",
        bool(getattr(residual_module, "enabled", True)),
    )
    summary.setdefault(
        "num_parameters",
        int(sum(parameter.numel() for parameter in parameters)),
    )
    summary.setdefault(
        "num_trainable_parameters",
        int(
            sum(
                parameter.numel()
                for parameter in parameters
                if parameter.requires_grad
            )
        ),
    )
    summary.setdefault(
        "all_parameters_finite",
        bool(all(torch.isfinite(parameter.detach()).all().item() for parameter in parameters)),
    )
    summary.setdefault(
        "max_abs_adjustment",
        float(getattr(residual_module, "max_abs_adjustment", 0.0)),
    )

    # v3 compatibility.  v4 intentionally has no scalar rho.
    rho = getattr(residual_module, "rho", None)
    if isinstance(rho, torch.Tensor) and rho.numel() == 1:
        rho_value = float(rho.detach().item())
        summary.setdefault("rho", rho_value)
        max_adjustment = float(summary.get("max_abs_adjustment", 0.0))
        alpha = max_adjustment * float(torch.tanh(rho.detach()).item())
        summary.setdefault("effective_alpha", alpha)
        summary.setdefault("route_abs_max", abs(alpha))
    else:
        summary.setdefault("rho", 0.0)
        summary.setdefault("effective_alpha", 0.0)
        summary.setdefault("route_abs_max", 0.0)
    return summary


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
    relation_identity_deltas: List[float] = []
    residual_stats = {
        name: _empty_tensor_stats()
        for name in (
            "adjustment",
            "gate",
            "support",
            "conflict",
            "quality",
            "quality_positive",
            "quality_negative",
            "positive_message",
            "negative_message",
            "route",
        )
    }

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
            relation_identity_deltas.extend(_as_values(details.get("relation_identity_delta")))
            _update_tensor_stats(
                residual_stats["adjustment"],
                _first_tensor(
                    details,
                    "gec_residual_adjustment",
                    "theta_adjustment",
                ),
            )
            for name in (
                "gate",
                "support",
                "conflict",
                "quality",
                "quality_positive",
                "quality_negative",
                "positive_message",
                "negative_message",
                "route",
            ):
                _update_tensor_stats(
                    residual_stats[name],
                    details.get(f"gec_residual_{name}"),
                )
            sample_count += take

    relation_learning = _relation_learning(base_model)
    residual_module = getattr(base_model, "evidence_residual", None)
    residual_summary = _residual_parameter_summary(residual_module)
    residual_rho = float(residual_summary.get("rho", 0.0))
    residual_alpha = float(residual_summary.get("effective_alpha", 0.0))
    residual_route_abs_max = max(
        float(residual_summary.get("route_abs_max", 0.0)),
        float(residual_stats["route"]["abs_max"]),
    )
    residual_adjustment_abs_mean = _stats_mean(
        residual_stats["adjustment"],
        absolute=True,
    )
    residual_runtime_enabled = bool(
        residual_summary.get("enabled", residual_module is not None)
    )
    residual_live = bool(
        residual_module is not None
        and residual_runtime_enabled
        and max(
            abs(residual_alpha),
            residual_route_abs_max,
            residual_adjustment_abs_mean,
        )
        > 1e-8
    )
    residual_diagnostics_finite = bool(
        all(stats["finite"] for stats in residual_stats.values())
        and residual_summary.get("all_parameters_finite", True)
    )
    positive_quality_mean = _stats_mean(
        residual_stats["quality_positive"]
    )
    negative_quality_mean = _stats_mean(
        residual_stats["quality_negative"]
    )
    legacy_quality_mean = _stats_mean(residual_stats["quality"])
    residual_quality_mean = (
        0.5 * (positive_quality_mean + negative_quality_mean)
        if (
            residual_stats["quality_positive"]["count"] > 0
            or residual_stats["quality_negative"]["count"] > 0
        )
        else legacy_quality_mean
    )
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
        "relation_identity_delta": _mean(relation_identity_deltas),
        "gec_residual_enabled": residual_module is not None,
        "gec_residual_runtime_enabled": residual_runtime_enabled,
        "gec_residual_live": residual_live,
        "gec_residual_mode": str(getattr(base_model, "gec_mode", "v1")),
        "gec_residual_module": (
            type(residual_module).__name__ if residual_module is not None else ""
        ),
        "gec_residual_parameter_summary": residual_summary,
        "gec_residual_num_parameters": int(
            residual_summary.get("num_parameters", 0)
        ),
        "gec_residual_num_trainable_parameters": int(
            residual_summary.get("num_trainable_parameters", 0)
        ),
        "gec_residual_parameters_finite": bool(
            residual_summary.get("all_parameters_finite", True)
        ),
        "gec_residual_diagnostics_finite": residual_diagnostics_finite,
        "gec_residual_rho": residual_rho,
        "gec_residual_alpha": residual_alpha,
        "gec_residual_route_abs_mean": _stats_mean(
            residual_stats["route"],
            absolute=True,
        ),
        "gec_residual_route_abs_max": residual_route_abs_max,
        "gec_residual_adjustment_abs_mean": residual_adjustment_abs_mean,
        "gec_residual_adjustment_abs_max": float(
            residual_stats["adjustment"]["abs_max"]
        ),
        "gec_residual_gate_mean": _stats_mean(residual_stats["gate"]),
        "gec_residual_support_mean": _stats_mean(residual_stats["support"]),
        "gec_residual_conflict_mean": _stats_mean(residual_stats["conflict"]),
        "gec_residual_quality_mean": residual_quality_mean,
        "gec_residual_positive_quality_mean": positive_quality_mean,
        "gec_residual_negative_quality_mean": negative_quality_mean,
        "gec_residual_positive_message_mean": _stats_mean(
            residual_stats["positive_message"]
        ),
        "gec_residual_negative_message_mean": _stats_mean(
            residual_stats["negative_message"]
        ),
    }

    if results["graph_enabled"]:
        with torch.no_grad():
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
            results["message_passing_enabled"]
            and not graph_trivial
            and not graph_over_sparse
            and results["knowledge_state_graph_delta"] > 1e-6
        )
        if not results["message_passing_enabled"]:
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
    residual = ""
    if activity.get("gec_residual_enabled"):
        residual_live = bool(
            activity.get(
                "gec_residual_live",
                abs(float(activity.get("gec_residual_alpha", 0.0))) > 1e-8,
            )
        )
        residual_mode = "LIVE" if residual_live else "FALLBACK"
        residual = f" GECResidual[{residual_mode}]"
    return f"ConceptGraph[{graph_mode}] IRT[{irt_mode}]{residual}"


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
        ]
    )
    if activity.get("gec_residual_enabled"):
        residual_live = bool(
            activity.get(
                "gec_residual_live",
                abs(float(activity.get("gec_residual_alpha", 0.0))) > 1e-8,
            )
        )
        lines.extend(
            [
                "",
                "3. Bounded evidence residual:",
                f"   - Mode: {activity.get('gec_residual_mode', 'unknown')}",
                "   - Runtime status: "
                f"{'LIVE' if residual_live else 'FALLBACK'}",
                f"   - Rho: {activity.get('gec_residual_rho', 0.0):.6f}",
                f"   - Effective alpha: {activity.get('gec_residual_alpha', 0.0):.6f}",
                "   - Route |max|: "
                f"{activity.get('gec_residual_route_abs_max', 0.0):.6f}",
                "   - Theta adjustment |mean|: "
                f"{activity.get('gec_residual_adjustment_abs_mean', 0.0):.6f}",
                f"   - Reliability gate mean: {activity.get('gec_residual_gate_mean', 0.0):.6f}",
                f"   - Cross-concept support mean: {activity.get('gec_residual_support_mean', 0.0):.6f}",
                f"   - Conflict mean: {activity.get('gec_residual_conflict_mean', 0.0):.6f}",
                f"   - Relation quality mean: {activity.get('gec_residual_quality_mean', 0.0):.6f}",
                "   - Positive/negative quality mean: "
                f"{activity.get('gec_residual_positive_quality_mean', 0.0):.6f} / "
                f"{activity.get('gec_residual_negative_quality_mean', 0.0):.6f}",
                "   - Positive/negative message mean: "
                f"{activity.get('gec_residual_positive_message_mean', 0.0):.6f} / "
                f"{activity.get('gec_residual_negative_message_mean', 0.0):.6f}",
            ]
        )
    lines.append("=" * 60)
    return "\n".join(lines)
