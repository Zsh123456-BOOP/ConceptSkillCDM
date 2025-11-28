import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MICritic(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scorer = nn.Bilinear(dim, dim, 1, bias=False)

    def forward(self, z_know: torch.Tensor, z_skill: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_know: [batch, dim]
            z_skill: [batch, dim]
        Returns:
            scores: [batch, batch]
        """
        B = z_know.size(0)
        z_k = z_know.unsqueeze(1).expand(-1, B, -1)
        z_s = z_skill.unsqueeze(0).expand(B, -1, -1)
        scores = self.scorer(z_k, z_s).squeeze(-1)
        return scores


def bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    loss_fn = nn.BCEWithLogitsLoss()
    return loss_fn(logits, labels)


def graph_loss(
    A_list: List[torch.Tensor],
    head_types: List[str],
    config,
    graph_learner,
) -> Tuple[torch.Tensor, dict]:
    reg = graph_learner.graph_regularization(A_list, head_types)
    L_graph = (
        config.lambda_graph_sparse * reg["sparse"]
        + config.lambda_graph_sym * reg["sym"]
        + config.lambda_graph_dag * reg["dag"]
    )
    return L_graph, reg


def disentangle_loss(
    S_prop: torch.Tensor,
    z_skill: torch.Tensor,
    config,
    z_know: Optional[torch.Tensor] = None,
    critic: Optional[nn.Module] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        L_de, L_orth, L_mi
    """
    if S_prop.size(0) == 0:
        device = S_prop.device
        zero = torch.tensor(0.0, device=device)
        return zero, zero, zero
    if z_know is None:
        # Fallback: project to skill_dim by truncation or mean pooling.
        if S_prop.size(1) >= z_skill.size(1):
            z_know = S_prop[:, : z_skill.size(1)]
        else:
            z_know = F.interpolate(
                S_prop.unsqueeze(1),
                size=z_skill.size(1),
                mode="linear",
                align_corners=False,
            ).squeeze(1)

    # L_orth: encourage z_know and z_skill to be orthogonal (decorrelated)
    dot = torch.sum(z_know * z_skill, dim=1)
    L_orth = torch.mean(dot * dot)

    # Approximate mutual information minimization via an InfoNCE-style loss:
    # positive = diagonal pairs (same student), negatives = all mismatched pairs.
    if critic is None:
        scores = torch.matmul(
            F.normalize(z_know, dim=-1),
            F.normalize(z_skill, dim=-1).transpose(0, 1),
        )
        scores = scores / math.sqrt(z_know.size(-1))
    else:
        scores = critic(z_know, z_skill)

    positive = torch.diag(scores)
    log_den = torch.logsumexp(scores, dim=1)
    L_mi = torch.mean(-(positive - log_den))

    # L_de: total disentanglement loss; L_orth: orthogonality; L_mi: InfoNCE MI surrogate
    L_de = config.lambda_de_orth * L_orth + config.lambda_de_mi * L_mi
    return L_de, L_orth, L_mi


__all__ = ["bce_loss", "graph_loss", "disentangle_loss", "MICritic"]
