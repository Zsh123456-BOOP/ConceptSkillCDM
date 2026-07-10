"""Graph-only regularization for the single-path Graph-IRT model."""

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F


def get_regularization_components(
    model,
    relation_matrices: torch.Tensor,
    details: Optional[Dict[str, torch.Tensor]] = None,
    base_loss: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Return graph structure penalties and exercise-IRT L2.

    No personal-support, posterior, target-prior, or theta-calibration terms
    exist in the production objective.
    """
    device = relation_matrices.device
    zero = relation_matrices.new_tensor(0.0)
    terms: Dict[str, torch.Tensor] = {
        "graph_entropy": zero,
        "graph_diag": zero,
        "graph_uniform": zero,
        "graph_reg_scale": relation_matrices.new_tensor(1.0),
        "prediction_l2": zero,
    }

    graph_reg_ramp = relation_matrices.new_tensor(model._get_graph_reg_ramp())
    if details is not None:
        details["graph_reg_ramp"] = graph_reg_ramp.detach()

    relation_learning = getattr(model, "relation_learning", None)
    if relation_learning is not None:
        entropy = relation_learning.get_entropy_sparsity(relation_matrices)
        node_count = max(2, int(relation_matrices.size(-1)))
        normalized_entropy = entropy / (math.log(float(node_count)) + 1e-8)
        lower = relation_matrices.new_tensor(model.graph_entropy_min)
        upper = relation_matrices.new_tensor(model.graph_entropy_max)
        entropy_penalty = F.relu(lower - normalized_entropy) + F.relu(normalized_entropy - upper)
        if model.lambda_graph_entropy > 0.0:
            terms["graph_entropy"] = model.lambda_graph_entropy * entropy_penalty * graph_reg_ramp

        diagonal_mass = torch.diagonal(relation_matrices, dim1=-2, dim2=-1).mean()
        if model.lambda_graph_diag > 0.0:
            terms["graph_diag"] = model.lambda_graph_diag * diagonal_mass * graph_reg_ramp

        uniform_value = 1.0 / float(node_count)
        uniform_distance = torch.sqrt(
            torch.clamp((relation_matrices - uniform_value).pow(2).sum(dim=-1), min=1e-12)
        ).mean()
        uniform_margin = relation_matrices.new_tensor(model.graph_uniform_margin)
        uniform_penalty = F.relu(uniform_margin - uniform_distance)
        if model.lambda_graph_uniform > 0.0:
            terms["graph_uniform"] = model.lambda_graph_uniform * uniform_penalty * graph_reg_ramp

        if details is not None:
            identity = torch.eye(
                node_count,
                device=device,
                dtype=relation_matrices.dtype,
            ).unsqueeze(0).expand_as(relation_matrices)
            identity_distance = torch.sqrt(
                torch.clamp((relation_matrices - identity).pow(2).sum(dim=-1), min=1e-12)
            ).mean()
            temperature = F.softplus(relation_learning.temperature_raw) + 1e-6
            details.update(
                {
                    "graph_entropy_raw": entropy.detach(),
                    "graph_entropy_norm": normalized_entropy.detach(),
                    "graph_entropy_pen": entropy_penalty.detach(),
                    "graph_diag_mass": diagonal_mass.detach(),
                    "graph_to_uniform_l2": uniform_distance.detach(),
                    "graph_to_identity_l2": identity_distance.detach(),
                    "graph_uniform_pen": uniform_penalty.detach(),
                    "graph_tau_mean": temperature.mean().detach(),
                    "graph_tau_std": temperature.std(unbiased=False).detach(),
                }
            )

    if base_loss is not None and model.graph_reg_cap_ratio > 0.0:
        raw_graph_regularization = (
            terms["graph_entropy"] + terms["graph_diag"] + terms["graph_uniform"]
        )
        cap = model.graph_reg_cap_ratio * base_loss.detach().abs()
        scale = torch.clamp(
            cap / (raw_graph_regularization.detach().abs() + 1e-8),
            max=1.0,
        )
        scale = torch.where(torch.isfinite(scale), scale, torch.ones_like(scale))
        terms["graph_reg_scale"] = scale.detach()
        terms["graph_entropy"] = terms["graph_entropy"] * scale
        terms["graph_diag"] = terms["graph_diag"] * scale
        terms["graph_uniform"] = terms["graph_uniform"] * scale
        if details is not None:
            details["graph_reg_raw"] = raw_graph_regularization.detach()
            details["graph_reg_cap"] = cap.detach()
            details["graph_reg_scale"] = scale.detach()

    if model.prediction_l2_lambda > 0.0:
        exercise_l2 = (
            model.exercise_encoder.b.weight.pow(2).mean()
            + model.exercise_encoder.a_raw.weight.pow(2).mean()
        )
        terms["prediction_l2"] = model.prediction_l2_lambda * exercise_l2

    terms["total"] = (
        terms["graph_entropy"]
        + terms["graph_diag"]
        + terms["graph_uniform"]
        + terms["prediction_l2"]
    )
    return terms
