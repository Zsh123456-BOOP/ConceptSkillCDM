"""Target-conditioned pseudo-evidence for sparse response histories."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


MEC_SCHEMA = "target_pseudocount_v2"


class MaskedEvidenceCompletion(nn.Module):
    """Complete the direct rate channel with fixed, label-free relations.

    For each queried concept, the module pools only non-Q concepts through a
    fixed item/exposure relation.  A shared 6->4->1 network predicts the
    probability of at most one pseudo response.  The resulting rate correction
    is centered at the current posterior, so zero initialization reproduces the
    matched direct-evidence model exactly.
    """

    FEATURE_NAMES: Tuple[str, ...] = (
        "related_rate",
        "related_residual",
        "related_rate_std",
        "related_support",
        "related_agreement",
        "target_prior_gap",
    )

    def __init__(
        self,
        *,
        relation_matrix: torch.Tensor,
        hidden_dim: int = 4,
        delta_limit: float = 2.0,
    ):
        super().__init__()
        relation = torch.as_tensor(relation_matrix).detach().float()
        if relation.dim() != 2 or relation.size(0) != relation.size(1):
            raise ValueError(
                "relation_matrix must be a square target-by-source matrix"
            )
        if relation.size(0) <= 0:
            raise ValueError("relation_matrix cannot be empty")
        if not bool(torch.isfinite(relation).all()) or bool((relation < 0).any()):
            raise ValueError("relation_matrix must be finite and non-negative")
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if float(delta_limit) <= 0.0:
            raise ValueError("delta_limit must be positive")

        row_sum = relation.sum(dim=1, keepdim=True)
        relation = torch.where(
            row_sum > 0.0,
            relation / row_sum.clamp(min=1e-12),
            torch.zeros_like(relation),
        )
        self.num_concepts = int(relation.size(0))
        self.delta_limit = float(delta_limit)
        self.register_buffer("relation_matrix", relation)
        self.net = nn.Sequential(
            nn.Linear(len(self.FEATURE_NAMES), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def _validate_inputs(
        self,
        response_evidence: torch.Tensor,
        response_count: torch.Tensor,
        response_correct: torch.Tensor,
        concept_prior: torch.Tensor,
        global_rate: torch.Tensor,
        q_mask: torch.Tensor,
    ) -> None:
        batch = int(response_count.size(0))
        expected_matrix = (batch, self.num_concepts)
        if tuple(response_evidence.shape) != (*expected_matrix, 2):
            raise ValueError(
                "response_evidence must have shape "
                f"{(*expected_matrix, 2)}, got {tuple(response_evidence.shape)}"
            )
        for name, value in (
            ("response_count", response_count),
            ("response_correct", response_correct),
            ("concept_prior", concept_prior),
            ("q_mask", q_mask),
        ):
            if tuple(value.shape) != expected_matrix:
                raise ValueError(
                    f"{name} must have shape {expected_matrix}, "
                    f"got {tuple(value.shape)}"
                )
        if tuple(global_rate.shape) not in {(batch, 1), (1, 1)}:
            raise ValueError(
                f"global_rate must have shape {(batch, 1)} or (1, 1), "
                f"got {tuple(global_rate.shape)}"
            )

    def build_features(
        self,
        response_evidence: torch.Tensor,
        response_count: torch.Tensor,
        concept_prior: torch.Tensor,
        global_rate: torch.Tensor,
        q_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return target-wise features, support, and agreement.

        Matrix products keep memory at O(B*C + C*C); no B*C*C tensor is built.
        Relation rows are targets and columns are candidate source concepts.
        """
        dtype = response_evidence.dtype
        count = response_count.to(dtype=dtype).clamp(min=0.0)
        source_mask = (q_mask <= 0).to(dtype=dtype)
        source_reliability = count / (count + 1.0) * source_mask
        receiver = self.relation_matrix.to(dtype=dtype).transpose(0, 1)
        support = torch.matmul(source_reliability, receiver).clamp(0.0, 1.0)
        denominator = support.clamp(min=1e-8)

        rate = response_evidence[..., 0].clamp(min=-4.0, max=4.0)
        residual = response_evidence[..., 1].clamp(min=-1.0, max=1.0)

        def related_mean(values: torch.Tensor) -> torch.Tensor:
            weighted = source_reliability * values
            return torch.matmul(weighted, receiver) / denominator

        rate_mean = related_mean(rate)
        residual_mean = related_mean(residual)
        rate_second = related_mean(rate.square())
        rate_std = (rate_second - rate_mean.square()).clamp(min=0.0).sqrt()
        rate_abs_mean = related_mean(rate.abs())
        agreement = (
            rate_mean.abs() / rate_abs_mean.clamp(min=1e-8)
        ).clamp(0.0, 1.0)
        agreement = agreement * (support > 0.0).to(dtype=dtype)

        eps = 1e-4
        prior_logit = torch.logit(
            concept_prior.to(dtype=dtype).clamp(min=eps, max=1.0 - eps)
        )
        global_logit = torch.logit(
            global_rate.to(dtype=dtype).clamp(min=eps, max=1.0 - eps)
        )
        prior_gap = (prior_logit - global_logit).clamp(min=-4.0, max=4.0)

        features = torch.stack(
            (
                rate_mean / 4.0,
                residual_mean,
                rate_std / 4.0,
                support,
                agreement,
                prior_gap / 4.0,
            ),
            dim=-1,
        )
        return features, support, agreement

    def forward(
        self,
        response_evidence: torch.Tensor,
        response_count: torch.Tensor,
        response_correct: torch.Tensor,
        concept_prior: torch.Tensor,
        global_rate: torch.Tensor,
        q_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        self._validate_inputs(
            response_evidence,
            response_count,
            response_correct,
            concept_prior,
            global_rate,
            q_mask,
        )
        features, support, agreement = self.build_features(
            response_evidence,
            response_count,
            concept_prior,
            global_rate,
            q_mask,
        )
        dtype = response_evidence.dtype
        query = (q_mask > 0).to(dtype=dtype)
        # Conflicting related evidence is less trustworthy, but never receives
        # more than one pseudo observation.
        pseudo_count = (
            support * (0.5 + 0.5 * agreement) * query
        ).clamp(0.0, 1.0)

        count = response_count.to(dtype=dtype).clamp(min=0.0)
        correct = response_correct.to(dtype=dtype).clamp(min=0.0)
        correct = torch.minimum(correct, count)
        prior = concept_prior.to(dtype=dtype).clamp(min=1e-4, max=1.0 - 1e-4)
        base_posterior = ((correct + prior) / (count + 1.0)).clamp(
            min=1e-4,
            max=1.0 - 1e-4,
        )
        base_logit = torch.logit(base_posterior)
        delta = self.delta_limit * torch.tanh(self.net(features).squeeze(-1))
        pseudo_probability = torch.sigmoid(base_logit + delta)
        reference_probability = torch.sigmoid(
            base_logit + torch.zeros_like(delta)
        )

        completed_count = count + pseudo_count
        completed_correct = correct + pseudo_count * pseudo_probability
        reference_correct = correct + pseudo_count * reference_probability
        completed_posterior = (
            (completed_correct + prior) / (completed_count + 1.0)
        ).clamp(min=1e-4, max=1.0 - 1e-4)
        reference_posterior = (
            (reference_correct + prior) / (completed_count + 1.0)
        ).clamp(min=1e-4, max=1.0 - 1e-4)
        rate_delta = (
            (
                torch.logit(completed_posterior)
                - torch.logit(reference_posterior)
            )
            * query
        )
        completed_rate = (
            response_evidence[..., 0] + rate_delta
        ).clamp(min=-4.0, max=4.0)
        details = {
            "mec_rate_delta": rate_delta,
            "mec_pseudo_count": pseudo_count,
            "mec_pseudo_probability": pseudo_probability * query,
            "mec_related_support": support * query,
            "mec_related_agreement": agreement * query,
            "mec_completed_count": completed_count,
            "mec_completed_correct": completed_correct,
        }
        return completed_rate, details
