"""A bounded sparse-evidence adjustment for concept-level ability.

The module is deliberately independent from the production model so it can be
attached to an already-trained v1 checkpoint without changing the checkpoint's
prediction path at initialization.  It consumes sufficient statistics whose
current training outcome has already been excluded by the caller.
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn


class SparseEvidenceThetaResidual(nn.Module):
    """Compute a zero-initialized, bounded concept-ability adjustment.

    ``relation_matrices`` follow the ``A[target, source]`` convention.  The
    relation graph is detached, averaged over heads, and stripped of self
    loops.  Importantly, the remaining off-diagonal mass is *not*
    re-normalized: a graph row that assigned almost all mass to its self loop
    must not suddenly become a strong cross-concept relation.

    ``count``, ``correct``, and ``concept_rate`` must all have shape
    ``(batch, concepts)``.  During training they must be the caller's
    leave-one-out statistics; this module never accepts the current label.
    """

    SPARSE_CONCEPT_THRESHOLD = 192

    def __init__(
        self,
        max_abs_adjustment: float = 0.20,
        max_logit_delta: float = 4.0,
        probability_eps: float = 1e-4,
    ):
        super().__init__()
        if not 0.0 < float(max_abs_adjustment) <= 1.0:
            raise ValueError("max_abs_adjustment must lie in (0, 1]")
        if float(max_logit_delta) <= 0.0:
            raise ValueError("max_logit_delta must be positive")
        if not 0.0 < float(probability_eps) < 0.5:
            raise ValueError("probability_eps must lie in (0, 0.5)")

        self.max_abs_adjustment = float(max_abs_adjustment)
        self.max_logit_delta = float(max_logit_delta)
        self.probability_eps = float(probability_eps)
        self.enabled = True
        # ReZero-style scalar.  Constant initialization consumes no RNG and
        # makes a newly attached module exactly equal to the v1 model.
        self.rho = nn.Parameter(torch.zeros(()))

    def parameter_summary(self) -> Dict[str, Union[bool, int, float]]:
        with torch.no_grad():
            alpha = self.max_abs_adjustment * torch.tanh(self.rho.detach())
            return {
                "enabled": bool(self.enabled),
                "num_parameters": 1,
                "num_trainable_parameters": int(self.rho.requires_grad),
                "all_parameters_finite": bool(torch.isfinite(self.rho).item()),
                "rho": float(self.rho.detach().item()),
                "effective_alpha": float(alpha.item()),
                "route_abs_max": float(abs(alpha.item())),
                "max_abs_adjustment": float(self.max_abs_adjustment),
            }

    @staticmethod
    def reliability_gate(
        target_count: torch.Tensor,
        support: torch.Tensor,
        conflict: torch.Tensor,
    ) -> torch.Tensor:
        """Return an analytic gate with the required monotonic directions."""
        target_count = target_count.clamp(min=0.0)
        support = support.clamp(min=0.0)
        conflict = conflict.clamp(min=0.0)
        need = 1.0 / (target_count + 1.0)
        backed = support / (support + 1.0)
        agreement = 1.0 / (conflict + 1.0)
        return (need * backed * agreement).clamp(min=0.0, max=1.0)

    @staticmethod
    def _off_diagonal_head_mean(
        relation_matrices: torch.Tensor,
    ) -> torch.Tensor:
        """Detach, average heads, and remove self loops without re-normalizing."""
        if relation_matrices.dim() != 3:
            raise ValueError(
                "relation_matrices must have shape (heads, concepts, concepts); "
                f"got {tuple(relation_matrices.shape)}"
            )
        heads, rows, columns = relation_matrices.shape
        if heads <= 0 or rows <= 0 or rows != columns:
            raise ValueError(
                "relation_matrices must contain at least one square graph; "
                f"got {tuple(relation_matrices.shape)}"
            )
        relation = relation_matrices.detach().clamp(min=0.0).mean(dim=0)
        if rows == 1:
            return torch.zeros_like(relation)
        eye = torch.eye(rows, device=relation.device, dtype=torch.bool)
        return relation.masked_fill(eye, 0.0)

    @classmethod
    def _propagate(
        cls,
        source: torch.Tensor,
        relation: torch.Tensor,
    ) -> torch.Tensor:
        """Apply ``A[target, source]`` without constructing BxCxC tensors."""
        if source.dim() != 2 or relation.dim() != 2:
            raise ValueError("source and relation must both be two-dimensional")
        if relation.size(0) != relation.size(1) or source.size(1) != relation.size(0):
            raise ValueError(
                f"incompatible source/relation shapes: {tuple(source.shape)} and "
                f"{tuple(relation.shape)}"
            )
        if int(relation.size(0)) >= cls.SPARSE_CONCEPT_THRESHOLD:
            propagated = torch.sparse.mm(
                relation.to_sparse_coo().coalesce(),
                source.transpose(0, 1),
            )
            return propagated.transpose(0, 1)
        return torch.matmul(source, relation.transpose(0, 1))

    def forward(
        self,
        relation_matrices: torch.Tensor,
        *,
        count: torch.Tensor,
        correct: torch.Tensor,
        concept_rate: torch.Tensor,
        return_details: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Return a concept-level theta adjustment and optional diagnostics."""
        relation = self._off_diagonal_head_mean(relation_matrices)
        expected = tuple(count.shape)
        if count.dim() != 2:
            raise ValueError(
                f"count must have shape (batch, concepts), got {expected}"
            )
        if expected[1] != relation.size(0):
            raise ValueError(
                f"count has {expected[1]} concepts but relation has "
                f"{relation.size(0)}"
            )
        for name, value in (("correct", correct), ("concept_rate", concept_rate)):
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"{name} must have shape {expected}, got {tuple(value.shape)}"
                )
        if not self.enabled:
            adjustment = torch.zeros(
                expected,
                device=relation.device,
                dtype=relation.dtype,
            )
            if not return_details:
                return adjustment
            return adjustment, {
                "support": adjustment,
                "conflict": adjustment,
                "gate": adjustment,
                "alpha": adjustment.new_tensor(0.0),
                "enabled": adjustment.new_tensor(0.0),
            }

        device = relation.device
        dtype = relation.dtype
        count = count.to(device=device, dtype=dtype).clamp(min=0.0)
        correct = correct.to(device=device, dtype=dtype)
        eps = self.probability_eps
        concept_rate = concept_rate.to(device=device, dtype=dtype).clamp(
            min=eps,
            max=1.0 - eps,
        )

        centered = correct - count * concept_rate
        source_residual = centered / count.clamp(min=1.0)
        second_moment_mass = count * source_residual.square()
        combined = torch.cat((count, centered, second_moment_mass), dim=0)
        propagated = self._propagate(combined, relation)
        batch_size = int(count.size(0))
        support, centered_sum, second_sum = propagated.split(batch_size, dim=0)
        support = support.clamp(min=0.0)

        has_support = support > 1e-12
        neighbor_mean = torch.where(
            has_support,
            centered_sum / support.clamp(min=1e-12),
            torch.zeros_like(centered_sum),
        )
        probability_delta = centered_sum / (support + 1.0)
        variance = torch.where(
            has_support,
            second_sum / support.clamp(min=1e-12) - neighbor_mean.square(),
            torch.zeros_like(second_sum),
        ).clamp(min=0.0)
        direct_delta = centered / (count + 1.0)
        target_reliability = count / (count + 1.0)
        conflict = (
            variance
            + target_reliability * (neighbor_mean - direct_delta).square()
        ).clamp(min=0.0)
        gate = self.reliability_gate(count, support, conflict)

        neighbor_rate = (concept_rate + probability_delta).clamp(
            min=eps,
            max=1.0 - eps,
        )
        evidence_logit_delta = (
            torch.logit(neighbor_rate) - torch.logit(concept_rate)
        ).clamp(
            min=-self.max_logit_delta,
            max=self.max_logit_delta,
        )
        evidence_logit_delta = torch.nan_to_num(
            evidence_logit_delta,
            nan=0.0,
            posinf=self.max_logit_delta,
            neginf=-self.max_logit_delta,
        )

        alpha = (
            evidence_logit_delta.new_tensor(self.max_abs_adjustment)
            * torch.tanh(self.rho.to(device=device, dtype=dtype))
        )
        theta_adjustment = alpha * torch.tanh(gate * evidence_logit_delta)
        theta_adjustment = torch.where(
            has_support,
            theta_adjustment,
            torch.zeros_like(theta_adjustment),
        )

        if not return_details:
            return theta_adjustment
        details = {
            "offdiag_relation": relation,
            "support": support.detach(),
            "neighbor_centered_sum": centered_sum.detach(),
            "neighbor_mean": neighbor_mean.detach(),
            "variance": variance.detach(),
            "conflict": conflict.detach(),
            "gate": gate.detach(),
            "probability_delta": probability_delta.detach(),
            "evidence_logit_delta": evidence_logit_delta.detach(),
            "alpha": alpha.detach(),
            "enabled": theta_adjustment.detach().new_tensor(1.0),
        }
        return theta_adjustment, details


class RelationQualitySignedResidual(nn.Module):
    """A raw-support-calibrated residual over frozen relation heads.

    relation_matrices follow the A[head, target, source] convention.  The
    frozen off-diagonal relation weight is combined with globally normalized
    raw item and exposure support by one shared 3 -> 3 -> 2 MLP.  Its two
    sigmoid outputs calibrate positive and negative difficulty-adjusted
    residual evidence independently.

    The response residual channel already contains its n/(n+1) reliability
    factor.  It is therefore propagated directly, without another reliability
    multiplier or a support denominator.  A single zero-initialized ReZero
    scalar provides an exact v1 fallback and a strict adjustment bound.
    """

    SPARSE_CONCEPT_THRESHOLD = 192
    NUM_EDGE_FEATURES = 3
    NUM_QUALITY_OUTPUTS = 2
    SUPPORT_OFFSET = 0.25

    def __init__(
        self,
        num_relation_heads: int,
        item_support_matrix: torch.Tensor,
        exposure_support_matrix: torch.Tensor,
        max_abs_adjustment: float = 0.20,
        eps: float = 1e-6,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        if int(num_relation_heads) <= 0:
            raise ValueError("num_relation_heads must be positive")
        if not 0.0 < float(max_abs_adjustment) <= 1.0:
            raise ValueError("max_abs_adjustment must lie in (0, 1]")
        if float(eps) <= 0.0:
            raise ValueError("eps must be positive")

        item_support = self._validate_support_matrix(
            item_support_matrix,
            "item_support_matrix",
        )
        exposure_support = self._validate_support_matrix(
            exposure_support_matrix,
            "exposure_support_matrix",
        )
        if item_support.shape != exposure_support.shape:
            raise ValueError(
                "item_support_matrix and exposure_support_matrix must have "
                "the same shape"
            )

        self.num_relation_heads = int(num_relation_heads)
        self.num_concepts = int(item_support.size(0))
        self.max_abs_adjustment = float(max_abs_adjustment)
        self.eps = float(eps)
        self.enabled = bool(enabled)
        self.register_buffer(
            "item_support_feature",
            self._normalize_support_feature(item_support, self.eps),
            persistent=False,
        )
        self.register_buffer(
            "exposure_support_feature",
            self._normalize_support_feature(exposure_support, self.eps),
            persistent=False,
        )

        # Shared 3 -> 3 -> 2 edge-quality MLP.  Explicit Parameters avoid
        # consuming RNG state.  At initialization hidden=activation(features)
        # but the zero output layer makes both sigmoid qualities exactly 0.5.
        self.quality_hidden_weight = nn.Parameter(
            torch.eye(self.NUM_EDGE_FEATURES)
        )
        self.quality_hidden_bias = nn.Parameter(
            torch.zeros(self.NUM_EDGE_FEATURES)
        )
        self.quality_output_weight = nn.Parameter(
            torch.zeros(
                self.NUM_QUALITY_OUTPUTS,
                self.NUM_EDGE_FEATURES,
            )
        )
        self.quality_output_bias = nn.Parameter(
            torch.zeros(self.NUM_QUALITY_OUTPUTS)
        )
        self.rho = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _validate_support_matrix(
        matrix: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        value = torch.as_tensor(matrix).detach().float()
        if value.dim() != 2 or value.size(0) <= 0 or value.size(0) != value.size(1):
            raise ValueError(f"{name} must be a non-empty square matrix")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must contain only finite values")
        if bool((value < 0.0).any()):
            raise ValueError(f"{name} must be non-negative")
        return value.clone()

    @staticmethod
    def _normalize_support_feature(
        support: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        transformed = torch.log1p(support.clamp(min=0.0))
        eye = torch.eye(
            int(transformed.size(0)),
            device=transformed.device,
            dtype=torch.bool,
        )
        transformed = transformed.masked_fill(eye, 0.0)
        return transformed / transformed.max().clamp(min=float(eps))

    def parameter_summary(self) -> Dict[str, Union[bool, int, float]]:
        """Return compact, JSON-friendly adapter diagnostics."""
        with torch.no_grad():
            parameters = tuple(self.parameters())
            alpha = (
                self.max_abs_adjustment
                * torch.tanh(self.rho.detach())
            )
            return {
                "enabled": bool(self.enabled),
                "num_relation_heads": int(self.num_relation_heads),
                "num_concepts": int(self.num_concepts),
                "num_parameters": int(
                    sum(parameter.numel() for parameter in parameters)
                ),
                "num_trainable_parameters": int(
                    sum(
                        parameter.numel()
                        for parameter in parameters
                        if parameter.requires_grad
                    )
                ),
                "all_parameters_finite": bool(
                    all(
                        torch.isfinite(parameter).all().item()
                        for parameter in parameters
                    )
                ),
                "rho": float(self.rho.detach().item()),
                "effective_alpha": float(alpha.item()),
                "route_abs_max": float(abs(alpha.item())),
                "quality_hidden_weight_l2": float(
                    self.quality_hidden_weight.detach().square().sum().sqrt().item()
                ),
                "quality_output_weight_l2": float(
                    self.quality_output_weight.detach().square().sum().sqrt().item()
                ),
                "max_abs_adjustment": float(self.max_abs_adjustment),
            }

    def _validate_inputs(
        self,
        relation_matrices: torch.Tensor,
        response_evidence: torch.Tensor,
        count: torch.Tensor,
    ) -> Tuple[int, int]:
        if relation_matrices.dim() != 3:
            raise ValueError(
                "relation_matrices must have shape (heads, concepts, concepts); "
                f"got {tuple(relation_matrices.shape)}"
            )
        heads, concepts, columns = relation_matrices.shape
        if heads != self.num_relation_heads:
            raise ValueError(
                f"expected {self.num_relation_heads} relation heads, got {heads}"
            )
        if concepts != self.num_concepts or columns != self.num_concepts:
            raise ValueError(
                "relation_matrices concept shape must match the raw support "
                f"matrices ({self.num_concepts}, {self.num_concepts})"
            )
        if response_evidence.dim() != 3:
            raise ValueError(
                "response_evidence must have shape (batch, concepts, 2); "
                f"got {tuple(response_evidence.shape)}"
            )
        batch_size = int(response_evidence.size(0))
        if tuple(response_evidence.shape) != (
            batch_size,
            concepts,
            2,
        ):
            raise ValueError(
                "response_evidence must have shape "
                f"{(batch_size, concepts, 2)}, "
                f"got {tuple(response_evidence.shape)}"
            )
        if tuple(count.shape) != (batch_size, concepts):
            raise ValueError(
                f"count must have shape {(batch_size, concepts)}, "
                f"got {tuple(count.shape)}"
            )
        if not relation_matrices.dtype.is_floating_point:
            raise ValueError("relation_matrices must use a floating-point dtype")
        return batch_size, concepts

    def _off_diagonal_relation(
        self,
        relation_matrices: torch.Tensor,
    ) -> torch.Tensor:
        relation = relation_matrices.detach()
        if not bool(torch.isfinite(relation).all()):
            raise ValueError("relation_matrices must contain only finite values")
        relation = relation.clamp(min=0.0)
        eye = torch.eye(
            self.num_concepts,
            device=relation.device,
            dtype=torch.bool,
        ).unsqueeze(0)
        return relation.masked_fill(eye, 0.0)

    def _edge_qualities(
        self,
        base_relation: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return positive/negative sigmoid qualities with shape H x C x C."""
        device = base_relation.device
        dtype = base_relation.dtype
        item_feature = self.item_support_feature.to(
            device=device,
            dtype=dtype,
        )
        exposure_feature = self.exposure_support_feature.to(
            device=device,
            dtype=dtype,
        )
        row_max = base_relation.amax(dim=-1, keepdim=True)
        relation_feature = base_relation / row_max.clamp(min=self.eps)
        relation_feature = torch.where(
            row_max > self.eps,
            relation_feature,
            torch.zeros_like(relation_feature),
        )
        heads = self.num_relation_heads
        features = torch.stack(
            (
                item_feature.unsqueeze(0).expand(heads, -1, -1),
                exposure_feature.unsqueeze(0).expand(heads, -1, -1),
                relation_feature,
            ),
            dim=-1,
        )

        hidden = torch.tanh(
            torch.nn.functional.linear(
                features,
                self.quality_hidden_weight.to(device=device, dtype=dtype),
                self.quality_hidden_bias.to(device=device, dtype=dtype),
            )
        )
        quality_logits = torch.nn.functional.linear(
            hidden,
            self.quality_output_weight.to(device=device, dtype=dtype),
            self.quality_output_bias.to(device=device, dtype=dtype),
        )
        quality = torch.sigmoid(quality_logits)
        active = base_relation > self.eps
        positive = torch.where(
            active,
            quality[..., 0],
            torch.zeros_like(quality[..., 0]),
        )
        negative = torch.where(
            active,
            quality[..., 1],
            torch.zeros_like(quality[..., 1]),
        )
        return positive, negative

    @classmethod
    def _propagate_scalar_heads(
        cls,
        source: torch.Tensor,
        relation_matrices: torch.Tensor,
    ) -> torch.Tensor:
        """Return B x H x C messages without constructing B x C x C."""
        if source.dim() != 2 or relation_matrices.dim() != 3:
            raise ValueError(
                "source must be B x C and relation_matrices must be H x C x C"
            )
        batch_size, concepts = source.shape
        heads, rows, columns = relation_matrices.shape
        if concepts != rows or rows != columns:
            raise ValueError(
                f"incompatible source/relation shapes: {tuple(source.shape)} and "
                f"{tuple(relation_matrices.shape)}"
            )

        if concepts >= cls.SPARSE_CONCEPT_THRESHOLD:
            flattened = source.transpose(0, 1)
            outputs = []
            for head in range(heads):
                sparse_relation = (
                    relation_matrices[head].to_sparse_coo().coalesce()
                )
                outputs.append(
                    torch.sparse.mm(sparse_relation, flattened).transpose(0, 1)
                )
            return torch.stack(outputs, dim=1)
        return torch.einsum("hts,bs->bht", relation_matrices, source)

    @classmethod
    def analytic_gate(
        cls,
        target_count: torch.Tensor,
        support: torch.Tensor,
    ) -> torch.Tensor:
        """Trust cross-concept evidence only when it is needed and supported."""
        if target_count.shape != support.shape:
            raise ValueError("target_count and support must have the same shape")
        target_count = target_count.clamp(min=0.0)
        support = support.clamp(min=0.0)
        need = torch.rsqrt(target_count + 1.0)
        backed = support / (support + cls.SUPPORT_OFFSET)
        return (need * backed).clamp(min=0.0, max=1.0)

    def _disabled_result(
        self,
        relation_matrices: torch.Tensor,
        response_evidence: torch.Tensor,
        *,
        return_details: bool,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        batch_size, concepts, _ = response_evidence.shape
        adjustment = relation_matrices.new_zeros((batch_size, concepts))
        if not return_details:
            return adjustment
        quality_shape = (
            self.num_relation_heads,
            self.num_concepts,
            self.num_concepts,
        )
        details = {
            "adjustment": adjustment,
            "quality_positive": adjustment.new_zeros(quality_shape),
            "quality_negative": adjustment.new_zeros(quality_shape),
            "support": adjustment.new_zeros((batch_size, concepts)),
            "gate": adjustment.new_zeros((batch_size, concepts)),
            "positive_message": adjustment.new_zeros((batch_size, concepts)),
            "negative_message": adjustment.new_zeros((batch_size, concepts)),
            "alpha": adjustment.new_tensor(0.0),
            "enabled": adjustment.new_tensor(0.0),
        }
        return adjustment, details

    def forward(
        self,
        relation_matrices: torch.Tensor,
        *,
        response_evidence: torch.Tensor,
        count: torch.Tensor,
        return_details: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Return a bounded concept-level adjustment and optional diagnostics."""
        self._validate_inputs(relation_matrices, response_evidence, count)
        if not self.enabled:
            return self._disabled_result(
                relation_matrices,
                response_evidence,
                return_details=return_details,
            )

        device = relation_matrices.device
        dtype = relation_matrices.dtype
        evidence = response_evidence.to(device=device, dtype=dtype)
        target_count = count.to(device=device, dtype=dtype)
        if not bool(torch.isfinite(evidence).all()):
            raise ValueError("response_evidence must contain only finite values")
        if not bool(torch.isfinite(target_count).all()):
            raise ValueError("count must contain only finite values")
        target_count = target_count.clamp(min=0.0)

        base_relation = self._off_diagonal_relation(relation_matrices)
        quality_positive, quality_negative = self._edge_qualities(base_relation)
        positive_relation = base_relation * (2.0 * quality_positive)
        negative_relation = base_relation * (2.0 * quality_negative)

        # Only channel 1 is used.  Channel 0 (rate evidence) remains entirely
        # outside this residual adapter.
        response_residual = evidence[..., 1]
        positive_source = torch.relu(response_residual)
        negative_source = torch.relu(-response_residual)
        positive_heads = self._propagate_scalar_heads(
            positive_source,
            positive_relation,
        )
        negative_heads = self._propagate_scalar_heads(
            negative_source,
            negative_relation,
        )
        positive_message = positive_heads.mean(dim=1)
        negative_message = negative_heads.mean(dim=1)

        reliability = target_count / (target_count + 1.0)
        support = self._propagate_scalar_heads(
            reliability,
            base_relation,
        ).mean(dim=1)
        gate = self.analytic_gate(target_count, support)
        alpha = (
            response_residual.new_tensor(self.max_abs_adjustment)
            * torch.tanh(self.rho.to(device=device, dtype=dtype))
        )
        adjustment = (
            alpha
            * gate
            * torch.tanh(positive_message - negative_message)
        )

        if not return_details:
            return adjustment
        details = {
            "adjustment": adjustment.detach(),
            "quality_positive": quality_positive.detach(),
            "quality_negative": quality_negative.detach(),
            "support": support.detach(),
            "gate": gate.detach(),
            "positive_message": positive_message.detach(),
            "negative_message": negative_message.detach(),
            "alpha": alpha.detach(),
            "enabled": adjustment.detach().new_tensor(1.0),
        }
        return adjustment, details
