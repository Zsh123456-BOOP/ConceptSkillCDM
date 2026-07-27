"""Complete target-concept evidence from the student's non-target profile."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


class MaskedEvidenceCompletion(nn.Module):
    """Map pooled non-Q response evidence to one Q-masked ability anchor.

    The module has no student/concept embeddings or relation matrix.  Its only
    inputs are label-excluded LEA evidence, its support counts, and the
    label-free Q mask of the current item.
    """

    FEATURE_NAMES: Tuple[str, ...] = (
        "rate_mean",
        "rate_std",
        "rate_positive_mass",
        "rate_negative_mass",
        "rate_max",
        "rate_min",
        "residual_mean",
        "residual_std",
        "residual_positive_mass",
        "residual_negative_mass",
        "log_source_response_count",
        "log_source_concept_count",
        "source_rate_logit",
        "source_concentration",
        "source_available",
    )

    def __init__(
        self,
        *,
        num_concepts: int,
        global_response_count: float,
        hidden_dims: Tuple[int, int] = (16, 8),
        delta_limit: float = 2.0,
    ):
        super().__init__()
        if int(num_concepts) <= 0:
            raise ValueError("num_concepts must be positive")
        if float(global_response_count) <= 0.0:
            raise ValueError("global_response_count must be positive")
        if len(hidden_dims) != 2 or min(int(value) for value in hidden_dims) <= 0:
            raise ValueError("hidden_dims must contain two positive integers")
        if float(delta_limit) <= 0.0:
            raise ValueError("delta_limit must be positive")

        self.num_concepts = int(num_concepts)
        self.delta_limit = float(delta_limit)
        self.register_buffer(
            "log_response_scale",
            torch.tensor(
                max(math.log1p(float(global_response_count)), 1.0),
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "log_concept_scale",
            torch.tensor(
                max(math.log1p(float(num_concepts)), 1.0),
                dtype=torch.float32,
            ),
            persistent=False,
        )
        first, second = (int(value) for value in hidden_dims)
        self.net = nn.Sequential(
            nn.Linear(len(self.FEATURE_NAMES), first),
            nn.GELU(),
            nn.Linear(first, second),
            nn.GELU(),
            nn.Linear(second, 1),
        )
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    @staticmethod
    def _weighted_mean_std(
        values: torch.Tensor,
        weights: torch.Tensor,
        denominator: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = (values * weights).sum(dim=1) / denominator
        variance = (
            (values - mean.unsqueeze(1)).square() * weights
        ).sum(dim=1) / denominator
        return mean, variance.clamp(min=0.0).sqrt()

    def build_features(
        self,
        response_evidence: torch.Tensor,
        response_count: torch.Tensor,
        response_correct: torch.Tensor,
        q_mask: torch.Tensor,
    ) -> torch.Tensor:
        expected_evidence = (
            response_count.size(0),
            self.num_concepts,
            2,
        )
        if tuple(response_evidence.shape) != expected_evidence:
            raise ValueError(
                "response_evidence must have shape "
                f"{expected_evidence}, got {tuple(response_evidence.shape)}"
            )
        expected_matrix = (response_count.size(0), self.num_concepts)
        if tuple(response_count.shape) != expected_matrix:
            raise ValueError(
                f"response_count must have shape {expected_matrix}, "
                f"got {tuple(response_count.shape)}"
            )
        if tuple(response_correct.shape) != expected_matrix:
            raise ValueError(
                f"response_correct must have shape {expected_matrix}, "
                f"got {tuple(response_correct.shape)}"
            )
        if tuple(q_mask.shape) != expected_matrix:
            raise ValueError(
                f"q_mask must have shape {expected_matrix}, got {tuple(q_mask.shape)}"
            )

        count = response_count.to(dtype=response_evidence.dtype).clamp(min=0.0)
        query = q_mask > 0
        source = (count > 0.0) & (~query)
        weights = count * source.to(dtype=count.dtype)
        source_count = weights.sum(dim=1)
        denominator = source_count.clamp(min=1.0)
        source_concepts = source.to(dtype=count.dtype).sum(dim=1)
        source_available = (source_count > 0.0).to(dtype=count.dtype)
        correct = response_correct.to(dtype=response_evidence.dtype).clamp(min=0.0)
        correct = torch.minimum(correct, count)
        source_correct = (correct * source.to(dtype=correct.dtype)).sum(dim=1)
        source_rate = (source_correct + 0.5) / (source_count + 1.0)
        source_rate_logit = torch.logit(
            source_rate.clamp(min=1e-4, max=1.0 - 1e-4)
        ).clamp(min=-4.0, max=4.0)

        rate = response_evidence[..., 0].clamp(min=-4.0, max=4.0)
        residual = response_evidence[..., 1].clamp(min=-1.0, max=1.0)
        rate_mean, rate_std = self._weighted_mean_std(
            rate,
            weights,
            denominator,
        )
        residual_mean, residual_std = self._weighted_mean_std(
            residual,
            weights,
            denominator,
        )

        def positive_mass(values: torch.Tensor) -> torch.Tensor:
            return (values.clamp(min=0.0) * weights).sum(dim=1) / denominator

        def negative_mass(values: torch.Tensor) -> torch.Tensor:
            return ((-values).clamp(min=0.0) * weights).sum(dim=1) / denominator

        rate_max = rate.masked_fill(~source, -4.0).max(dim=1).values
        rate_min = rate.masked_fill(~source, 4.0).min(dim=1).values
        rate_max = rate_max * source_available
        rate_min = rate_min * source_available
        concentration = count.masked_fill(~source, 0.0).max(dim=1).values
        concentration = concentration / denominator

        features = torch.stack(
            (
                rate_mean / 4.0,
                rate_std / 4.0,
                positive_mass(rate) / 4.0,
                negative_mass(rate) / 4.0,
                rate_max / 4.0,
                rate_min / 4.0,
                residual_mean,
                residual_std,
                positive_mass(residual),
                negative_mass(residual),
                torch.log1p(source_count) / self.log_response_scale,
                torch.log1p(source_concepts) / self.log_concept_scale,
                source_rate_logit / 4.0,
                concentration,
                source_available,
            ),
            dim=1,
        )
        return features

    def forward(
        self,
        response_evidence: torch.Tensor,
        response_count: torch.Tensor,
        response_correct: torch.Tensor,
        q_mask: torch.Tensor,
    ) -> torch.Tensor:
        features = self.build_features(
            response_evidence,
            response_count,
            response_correct,
            q_mask,
        )
        available = features[:, self.FEATURE_NAMES.index("source_available")]
        reliability = response_count.to(dtype=response_evidence.dtype)
        reliability = reliability * (q_mask <= 0).to(dtype=reliability.dtype)
        reliability = reliability.sum(dim=1)
        reliability = reliability / (reliability + 1.0)
        scalar = (
            self.delta_limit
            * torch.tanh(self.net(features).reshape(-1))
            * reliability
            * available
        )
        return scalar.unsqueeze(1) * (q_mask > 0).to(dtype=scalar.dtype)
