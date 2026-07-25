"""Train-only concept graph construction and graph knowledge encoding."""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadRelationLearning(nn.Module):
    """Build a small set of row-stochastic concept graphs.

    Edge logits only use train-split, label-free item/co-exposure priors plus a
    handful of global calibration parameters.  There is no student-conditioned
    support graph in the production prediction path.
    """

    def __init__(
        self,
        num_concepts: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        tau_init: float = 1.0,
        topk: Optional[int] = None,
        allow_self_loop: bool = True,
        identity_residual: float = 0.0,
        prior_matrix: Optional[torch.Tensor] = None,
        exposure_prior_matrix: Optional[torch.Tensor] = None,
        prior_strength_init: float = 1.0,
        allow_empty_rows: bool = False,
    ):
        super().__init__()
        self.num_concepts = int(num_concepts)
        self.num_heads = int(num_heads)
        self.topk = topk
        self.allow_self_loop = bool(allow_self_loop)
        self.identity_residual = (
            float(max(0.0, min(1.0, identity_residual))) if self.allow_self_loop else 0.0
        )
        self.allow_empty_rows = bool(allow_empty_rows)
        self.prior_strength_init = max(1e-4, float(prior_strength_init))

        item_scores, item_support_scores = self._build_prior_buffers(prior_matrix, "prior_matrix")
        exposure_scores, exposure_support_scores = self._build_prior_buffers(
            exposure_prior_matrix,
            "exposure_prior_matrix",
        )
        self.has_item_prior = prior_matrix is not None and bool(item_support_scores.any().item())
        self.has_exposure_prior = (
            exposure_prior_matrix is not None and bool(exposure_support_scores.any().item())
        )
        self.register_buffer("prior_logit_scores", item_scores, persistent=False)
        self.register_buffer("exposure_prior_logit_scores", exposure_scores, persistent=False)
        self.register_buffer(
            "item_support_mask",
            self._build_topk_support_mask(item_support_scores),
            persistent=False,
        )
        self.register_buffer(
            "exposure_support_mask",
            self._build_topk_support_mask(exposure_support_scores),
            persistent=False,
        )
        if (
            self.topk is not None
            and self.topk > 0
            and not self.allow_self_loop
            and not self.allow_empty_rows
        ):
            evidence_support = self.item_support_mask | self.exposure_support_mask
            empty_rows = torch.nonzero(~evidence_support.any(dim=-1), as_tuple=False).reshape(-1)
            if empty_rows.numel() > 0:
                preview = empty_rows[:10].tolist()
                raise ValueError(
                    "disable_self_loop with graph_topk requires observed support in every concept row; "
                    f"empty rows include {preview}"
                )

        init_strength = self.prior_strength_init
        item_init, exposure_init = self._build_source_specialized_strength_init(init_strength)
        self.prior_strength_raw = nn.Parameter(self._inverse_softplus(item_init))
        self.exposure_prior_strength_raw = nn.Parameter(self._inverse_softplus(exposure_init))
        self.receiver_bias = nn.Parameter(torch.zeros(self.num_heads, self.num_concepts))
        self.self_loop_logit = nn.Parameter(torch.full((self.num_heads,), 0.75))
        tau_values = torch.full((self.num_heads,), max(1e-4, float(tau_init)))
        self.temperature_raw = nn.Parameter(self._inverse_softplus(tau_values))
        self.dropout = nn.Dropout(dropout)
        self._last_support_diagnostics: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
        value = value.detach().float().clamp(min=1e-4)
        return torch.log(torch.expm1(value).clamp(min=1e-12))

    def _build_source_specialized_strength_init(self, base_strength: float) -> Tuple[torch.Tensor, torch.Tensor]:
        base = max(1e-4, float(base_strength))
        item = torch.full(
            (self.num_heads,),
            base if self.has_item_prior else 1e-4,
            dtype=torch.float32,
        )
        exposure = torch.full(
            (self.num_heads,),
            base if self.has_exposure_prior else 1e-4,
            dtype=torch.float32,
        )
        if self.has_item_prior and self.has_exposure_prior and self.num_heads >= 2:
            high = max(1e-4, base * 1.45)
            low = max(1e-4, base * 0.65)
            for head in range(self.num_heads):
                if head % 2 == 0:
                    item[head], exposure[head] = high, low
                else:
                    item[head], exposure[head] = low, high
        return item, exposure

    def _build_prior_buffers(
        self,
        prior_matrix: Optional[torch.Tensor],
        name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if prior_matrix is None:
            zeros = torch.zeros(self.num_concepts, self.num_concepts, dtype=torch.float32)
            return zeros, zeros
        prior = prior_matrix.detach().float()
        expected = (self.num_concepts, self.num_concepts)
        if tuple(prior.shape) != expected:
            raise ValueError(f"{name} shape must be {expected}, got {tuple(prior.shape)}")
        if self.num_concepts == 1:
            ones = torch.ones(1, 1, dtype=torch.float32)
            return ones, ones

        eye = torch.eye(self.num_concepts, dtype=torch.float32, device=prior.device)
        support_scores = prior.clamp(min=0.0) * (1.0 - eye)
        normalized = prior.clamp(min=1e-4)
        normalized = normalized / normalized.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        scores = normalized.log()
        scores = scores - scores.mean(dim=-1, keepdim=True)
        return scores.clamp(min=-4.0, max=4.0).cpu(), support_scores.cpu()

    def _build_topk_support_mask(self, support_scores: torch.Tensor) -> torch.Tensor:
        count = self.num_concepts
        mask = torch.zeros(count, count, dtype=torch.bool)
        if self.topk is None or self.topk <= 0 or count <= 1:
            return mask
        k = min(int(self.topk), count - 1)
        eye = torch.eye(count, dtype=torch.bool)
        for row_idx in range(count):
            row = support_scores[row_idx].detach().float().clone()
            positive = (row > 0.0) & (~eye[row_idx])
            take = min(k, int(positive.sum().item()))
            if take <= 0:
                continue
            row = row.masked_fill(~positive, float("-inf"))
            mask[row_idx, torch.topk(row, k=take).indices] = True
        return mask

    def _apply_source_balanced_topk(
        self,
        scores: torch.Tensor,
        *,
        item_mask: torch.Tensor,
        exposure_mask: torch.Tensor,
        self_mask: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        count = scores.size(0)
        total_budget = min(max(1, int(k)), count)
        offdiag_budget = total_budget - 1 if self.allow_self_loop else total_budget
        selected = torch.zeros((count, count), device=scores.device, dtype=torch.bool)
        offdiag = ~self_mask
        if self.allow_self_loop:
            selected |= self_mask
        if offdiag_budget <= 0:
            return scores.masked_fill(~selected, float("-inf"))

        def selected_count(mask: torch.Tensor) -> torch.Tensor:
            return (mask & offdiag).sum(dim=-1)

        def pick(
            source_mask: torch.Tensor,
            quota: int,
            current: torch.Tensor,
            row_limit: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            take = min(max(0, int(quota)), count)
            if take <= 0:
                return current
            available = source_mask & offdiag & (~current)
            values, indices = torch.topk(
                scores.masked_fill(~available, float("-inf")),
                k=take,
                dim=-1,
            )
            keep = torch.isfinite(values)
            if row_limit is not None:
                rank = torch.arange(take, device=scores.device).view(1, take)
                keep &= rank < row_limit.clamp(min=0).view(-1, 1)
            picked = torch.zeros_like(current)
            picked.scatter_(dim=-1, index=indices, src=keep)
            return current | (picked & available)

        item_quota = max(1, int(math.ceil(offdiag_budget * 0.25)))
        exposure_quota = max(1, int(math.ceil(offdiag_budget * 0.50)))
        selected = pick(item_mask, item_quota, selected)
        selected = pick(exposure_mask, exposure_quota, selected, offdiag_budget - selected_count(selected))
        selected = pick(
            item_mask | exposure_mask,
            offdiag_budget,
            selected,
            offdiag_budget - selected_count(selected),
        )

        if not self.allow_self_loop and not self.allow_empty_rows:
            empty_rows = torch.nonzero(~selected.any(dim=-1), as_tuple=False).reshape(-1)
            if empty_rows.numel() > 0:
                raise RuntimeError(
                    "top-k support unexpectedly contains empty rows with self-loops disabled: "
                    f"{empty_rows[:10].tolist()}"
                )
        return scores.masked_fill(~selected, float("-inf"))

    def _update_support_diagnostics(self, relation_matrices: torch.Tensor) -> None:
        heads, count, _ = relation_matrices.shape
        device = relation_matrices.device
        dtype = relation_matrices.dtype
        eye = torch.eye(count, device=device, dtype=torch.bool)
        item_mask = self.item_support_mask.to(device=device)
        exposure_mask = self.exposure_support_mask.to(device=device)
        candidates = item_mask | exposure_mask
        if self.allow_self_loop:
            candidates |= eye
        final_mask = relation_matrices.detach() > 1e-12

        def survival_rate(source_mask: torch.Tensor) -> torch.Tensor:
            denominator = (source_mask.to(dtype=dtype).sum() * float(max(1, heads))).clamp(min=1.0)
            return (final_mask & source_mask.view(1, count, count)).to(dtype=dtype).sum() / denominator

        overlap_union = (item_mask | exposure_mask).to(dtype=dtype).sum().clamp(min=1.0)
        self._last_support_diagnostics = {
            "support_candidate_size_mean": candidates.to(dtype=dtype).sum(dim=-1).mean().detach(),
            "support_final_size_mean": final_mask.to(dtype=dtype).sum(dim=-1).mean().detach(),
            "support_item_survival_rate": survival_rate(item_mask).detach(),
            "support_exposure_survival_rate": survival_rate(exposure_mask).detach(),
            "support_item_exposure_overlap": (
                (item_mask & exposure_mask).to(dtype=dtype).sum() / overlap_union
            ).detach(),
            "support_self_retention_rate": survival_rate(eye).detach(),
        }

    def forward(self) -> torch.Tensor:
        """Return the row-stochastic graph heads."""
        count = self.num_concepts
        device = self.receiver_bias.device
        dtype = self.receiver_bias.dtype
        if count == 1:
            relations = torch.ones((self.num_heads, 1, 1), device=device, dtype=dtype)
            self._update_support_diagnostics(relations)
            return relations

        tau = F.softplus(self.temperature_raw) + 1e-6
        item_scores = self.prior_logit_scores.to(device=device, dtype=dtype)
        exposure_scores = self.exposure_prior_logit_scores.to(device=device, dtype=dtype)
        item_strength = F.softplus(self.prior_strength_raw).to(device=device, dtype=dtype)
        exposure_strength = F.softplus(self.exposure_prior_strength_raw).to(device=device, dtype=dtype)
        eye = torch.eye(count, device=device, dtype=torch.bool)
        relations = []

        for head in range(self.num_heads):
            scores = item_strength[head] * item_scores + exposure_strength[head] * exposure_scores
            scores = scores + self.receiver_bias[head].view(1, count)
            scores = scores / tau[head]
            scores = scores - scores.mean(dim=-1, keepdim=True)
            if self.allow_self_loop:
                scores = scores + self.self_loop_logit[head] * eye.to(dtype=dtype)
            else:
                scores = scores.masked_fill(eye, float("-inf"))

            if self.topk is not None and 0 < self.topk < count:
                item_mask = self.item_support_mask.to(device=device)
                exposure_mask = self.exposure_support_mask.to(device=device)
                support_mask = item_mask | exposure_mask
                if self.allow_self_loop:
                    support_mask |= eye
                scores = self._apply_source_balanced_topk(
                    scores.masked_fill(~support_mask, float("-inf")),
                    item_mask=item_mask,
                    exposure_mask=exposure_mask,
                    self_mask=eye,
                    k=int(self.topk),
                )

            finite_rows = torch.isfinite(scores).any(dim=-1, keepdim=True)
            safe_scores = torch.where(
                finite_rows,
                scores,
                torch.zeros_like(scores),
            )
            base_adjacency = F.softmax(safe_scores, dim=-1)
            base_adjacency = torch.where(
                finite_rows,
                base_adjacency,
                torch.zeros_like(base_adjacency),
            )
            adjacency = self.dropout(base_adjacency)
            row_sum = adjacency.sum(dim=-1, keepdim=True)
            zero_rows = row_sum.squeeze(-1) < 1e-12
            if bool(zero_rows.any()):
                adjacency = adjacency.clone()
                indices = torch.nonzero(zero_rows, as_tuple=False).squeeze(-1)
                adjacency[indices, :] = base_adjacency[indices, :]
                row_sum = adjacency.sum(dim=-1, keepdim=True)
            if self.identity_residual > 0.0:
                adjacency = (
                    (1.0 - self.identity_residual) * adjacency
                    + self.identity_residual * eye.to(dtype=dtype)
                )
                row_sum = adjacency.sum(dim=-1, keepdim=True)
            relations.append(adjacency / row_sum.clamp(min=1e-12))

        relation_matrices = torch.stack(relations, dim=0)
        self._update_support_diagnostics(relation_matrices)
        return relation_matrices

    @staticmethod
    def get_entropy_sparsity(relation_matrices: torch.Tensor) -> torch.Tensor:
        adjacency = relation_matrices.clamp(min=1e-12)
        return -(adjacency * adjacency.log()).sum(dim=-1).mean()


class ReliabilityAwareEvidenceRouter(nn.Module):
    """Route label-excluded cross-concept evidence with explicit reliability.

    The input relation convention is ``A[target, source]``.  Evidence
    transport always removes the diagonal and keeps genuinely empty rows at
    zero.  The returned ``support`` is a relation-weighted support quantity,
    not an independent sample count.
    """

    SPARSE_CONCEPT_THRESHOLD = 192

    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = int(num_heads)
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")

        self.propagation_shrink_raw = nn.Parameter(
            self._inverse_softplus(torch.tensor(1.0))
        )
        self.direct_scale_raw = nn.Parameter(
            self._inverse_softplus(torch.tensor(2.0))
        )
        self.support_scale_raw = nn.Parameter(
            self._inverse_softplus(torch.tensor(2.0))
        )
        self.conflict_scale_raw = nn.Parameter(
            self._inverse_softplus(torch.tensor(1.0))
        )
        self.head_bias = nn.Parameter(torch.zeros(self.num_heads))
        self.head_support_raw = nn.Parameter(
            self._inverse_softplus(torch.tensor(0.5))
        )
        self.head_conflict_raw = nn.Parameter(
            self._inverse_softplus(torch.tensor(0.5))
        )
        # Unnormalised graph-expert weights.  A fixed null expert has weight
        # one, so both graph corrections start small and can be exactly
        # rejected when unavailable.
        self.state_route_raw = nn.Parameter(
            self._inverse_softplus(torch.tensor(0.1))
        )
        self.evidence_route_raw = nn.Parameter(
            self._inverse_softplus(torch.tensor(0.2))
        )

    @staticmethod
    def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
        value = value.detach().float().clamp(min=1e-4)
        return torch.log(torch.expm1(value).clamp(min=1e-12))

    @staticmethod
    def off_diagonal_relations(relation_matrices: torch.Tensor) -> torch.Tensor:
        """Remove self loops and row-normalise without filling empty rows."""
        if relation_matrices.dim() != 3:
            raise ValueError(
                "relation_matrices must have shape (heads, concepts, concepts); "
                f"got {tuple(relation_matrices.shape)}"
            )
        heads, rows, columns = relation_matrices.shape
        if rows != columns:
            raise ValueError(
                f"relation matrices must be square, got {(heads, rows, columns)}"
            )
        if rows <= 1:
            return torch.zeros_like(relation_matrices)
        eye = torch.eye(
            rows,
            device=relation_matrices.device,
            dtype=relation_matrices.dtype,
        ).unsqueeze(0)
        offdiag = relation_matrices * (1.0 - eye)
        row_sum = offdiag.sum(dim=-1, keepdim=True)
        return torch.where(
            row_sum > 1e-12,
            offdiag / row_sum.clamp(min=1e-12),
            torch.zeros_like(offdiag),
        )

    @classmethod
    def _propagate(
        cls,
        source: torch.Tensor,
        relation: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one ``A[target, source]`` relation without BxCxC tensors."""
        if source.dim() != 2 or relation.dim() != 2:
            raise ValueError("source and relation must both be two-dimensional")
        if source.size(1) != relation.size(0) or relation.size(0) != relation.size(1):
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

    def evidence_gate(
        self,
        target_count: torch.Tensor,
        support: torch.Tensor,
        conflict: torch.Tensor,
    ) -> torch.Tensor:
        """Return a gate decreasing in target count/conflict and increasing in support."""
        direct_scale = F.softplus(self.direct_scale_raw) + 1e-6
        support_scale = F.softplus(self.support_scale_raw) + 1e-6
        conflict_scale = F.softplus(self.conflict_scale_raw) + 1e-6
        need = direct_scale / (target_count.clamp(min=0.0) + direct_scale)
        backed = support.clamp(min=0.0) / (
            support.clamp(min=0.0) + support_scale
        )
        agreement = torch.exp(-conflict_scale * conflict.clamp(min=0.0))
        return (need * backed * agreement).clamp(min=0.0, max=1.0)

    def graph_routes(
        self,
        evidence_gate: torch.Tensor,
        *,
        state_enabled: bool,
        evidence_enabled: bool,
    ) -> torch.Tensor:
        """Return ``[..., state, evidence]`` routes with an implicit null expert."""
        null_weight = torch.ones_like(evidence_gate)
        state_weight = (
            F.softplus(self.state_route_raw).to(dtype=evidence_gate.dtype)
            * torch.ones_like(evidence_gate)
            if state_enabled
            else torch.zeros_like(evidence_gate)
        )
        evidence_weight = (
            F.softplus(self.evidence_route_raw).to(dtype=evidence_gate.dtype)
            * evidence_gate
            if evidence_enabled
            else torch.zeros_like(evidence_gate)
        )
        denominator = null_weight + state_weight + evidence_weight
        return torch.stack(
            (
                state_weight / denominator.clamp(min=1e-12),
                evidence_weight / denominator.clamp(min=1e-12),
            ),
            dim=-1,
        )

    def forward(
        self,
        relation_matrices: torch.Tensor,
        *,
        count: torch.Tensor,
        correct: torch.Tensor,
        concept_rate: torch.Tensor,
        direct_probability_delta: torch.Tensor,
        state_enabled: bool,
        evidence_enabled: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Return propagated logit correction, gate, routes, and diagnostics."""
        if relation_matrices.size(0) != self.num_heads:
            raise ValueError(
                f"expected {self.num_heads} relation heads, got "
                f"{relation_matrices.size(0)}"
            )
        expected = tuple(count.shape)
        for name, tensor in (
            ("correct", correct),
            ("concept_rate", concept_rate),
            ("direct_probability_delta", direct_probability_delta),
        ):
            if tuple(tensor.shape) != expected:
                raise ValueError(
                    f"{name} must have shape {expected}, got {tuple(tensor.shape)}"
                )

        relations = self.off_diagonal_relations(relation_matrices)
        count = count.to(dtype=relations.dtype).clamp(min=0.0)
        correct = correct.to(dtype=relations.dtype)
        concept_rate = concept_rate.to(dtype=relations.dtype)
        centered = correct - count * concept_rate
        source_rate_delta = centered / count.clamp(min=1.0)
        second_moment_mass = count * source_rate_delta.square()

        supports = []
        mean_deltas = []
        conflicts = []
        shrink = F.softplus(self.propagation_shrink_raw) + 1e-6
        combined_source = torch.cat(
            (count, centered, second_moment_mass),
            dim=0,
        )
        batch_size = int(count.size(0))
        for head in range(self.num_heads):
            propagated = self._propagate(combined_source, relations[head])
            support, centered_sum, second_sum = propagated.split(batch_size, dim=0)
            support = support.clamp(min=0.0)
            raw_mean = torch.where(
                support > 1e-12,
                centered_sum / support.clamp(min=1e-12),
                torch.zeros_like(centered_sum),
            )
            mean_delta = raw_mean * support / (support + shrink)
            variance = torch.where(
                support > 1e-12,
                second_sum / support.clamp(min=1e-12) - raw_mean.square(),
                torch.zeros_like(second_sum),
            ).clamp(min=0.0)
            target_reliability = count / (count + 1.0)
            direct_conflict = (
                target_reliability
                * (
                    mean_delta
                    - direct_probability_delta.to(dtype=mean_delta.dtype)
                ).abs()
            )
            supports.append(support)
            mean_deltas.append(mean_delta)
            conflicts.append(variance + direct_conflict.square())

        support_by_head = torch.stack(supports, dim=1)
        delta_by_head = torch.stack(mean_deltas, dim=1)
        conflict_by_head = torch.stack(conflicts, dim=1)
        head_scores = (
            self.head_bias.view(1, -1, 1)
            + F.softplus(self.head_support_raw)
            * torch.log1p(support_by_head)
            - F.softplus(self.head_conflict_raw) * conflict_by_head
        )
        valid_heads = support_by_head > 1e-12
        safe_scores = head_scores.masked_fill(~valid_heads, -1e9)
        head_weights = F.softmax(safe_scores, dim=1) * valid_heads.to(
            dtype=safe_scores.dtype
        )
        head_weights = head_weights / head_weights.sum(
            dim=1, keepdim=True
        ).clamp(min=1e-12)

        propagated_delta = (head_weights * delta_by_head).sum(dim=1)
        support = (head_weights * support_by_head).sum(dim=1)
        conflict = (head_weights * conflict_by_head).sum(dim=1)
        eps = 1e-4
        propagated_rate = (
            concept_rate + propagated_delta
        ).clamp(min=eps, max=1.0 - eps)
        propagated_logit = (
            torch.logit(propagated_rate)
            - torch.logit(concept_rate.clamp(min=eps, max=1.0 - eps))
        ).clamp(min=-4.0, max=4.0)
        propagated_logit = torch.where(
            support > 1e-12,
            propagated_logit,
            torch.zeros_like(propagated_logit),
        )
        gate = self.evidence_gate(count, support, conflict)
        routes = self.graph_routes(
            gate,
            state_enabled=state_enabled,
            evidence_enabled=evidence_enabled,
        )
        return propagated_logit, gate, routes, {
            "offdiag_relations": relations,
            "head_weights": head_weights,
            "support": support,
            "conflict": conflict,
            "propagated_delta": propagated_delta,
        }


class ConceptGraphConv(nn.Module):
    """Multi-head graph convolution over a single global concept graph."""

    # Dense matmul is faster for the small graphs used by ASSIST/Junyi.  Above
    # this size, top-k relation matrices are overwhelmingly sparse and COO SpMM
    # avoids the quadratic compute/memory blow-up of MOOCRadar/XES3G5M.
    SPARSE_CONCEPT_THRESHOLD = 192

    def __init__(self, in_features: int, out_features: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_transforms = nn.ModuleList(
            nn.Linear(in_features, out_features, bias=False) for _ in range(self.num_heads)
        )
        self.head_attention = nn.Parameter(torch.ones(self.num_heads) / self.num_heads)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.dropout = nn.Dropout(dropout)
        for transform in self.head_transforms:
            nn.init.xavier_normal_(transform.weight)

    def forward(self, states: torch.Tensor, relation_matrices: torch.Tensor) -> torch.Tensor:
        if relation_matrices.dim() != 3:
            raise ValueError(
                "relation_matrices must have shape (heads, concepts, concepts); "
                f"got {tuple(relation_matrices.shape)}"
            )
        if relation_matrices.size(0) != self.num_heads:
            raise ValueError(
                f"expected {self.num_heads} graph heads, got {relation_matrices.size(0)}"
            )
        outputs = []
        for head, transform in enumerate(self.head_transforms):
            transformed = transform(states)
            adjacency = relation_matrices[head].to(dtype=transformed.dtype)
            adjacency = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            if int(adjacency.size(0)) >= self.SPARSE_CONCEPT_THRESHOLD:
                concept_count = int(transformed.size(1))
                batch_size = int(transformed.size(0))
                feature_count = int(transformed.size(2))
                flattened = transformed.permute(1, 0, 2).reshape(
                    concept_count,
                    batch_size * feature_count,
                )
                propagated = torch.sparse.mm(
                    adjacency.to_sparse_coo().coalesce(),
                    flattened,
                )
                propagated = propagated.reshape(
                    concept_count,
                    batch_size,
                    feature_count,
                ).permute(1, 0, 2)
                outputs.append(propagated)
            else:
                outputs.append(torch.matmul(adjacency, transformed))
        stacked = torch.stack(outputs, dim=0)
        weights = F.softmax(self.head_attention, dim=0).view(-1, 1, 1, 1)
        return self.dropout((stacked * weights).sum(dim=0) + self.bias)


class StudentKnowledgeEncoder(nn.Module):
    """Encode train-evidence-initialized concept states through the global graph."""

    def __init__(
        self,
        num_students: int,
        num_concepts: int,
        knowledge_dim: int,
        num_gnn_layers: int = 2,
        num_relation_heads: int = 4,
        dropout: float = 0.1,
        gnn_residual_weight: float = 0.5,
        propagation_alpha: float = 0.20,
        use_response_evidence: bool = False,
    ):
        super().__init__()
        self.num_students = int(num_students)
        self.num_concepts = int(num_concepts)
        self.knowledge_dim = int(knowledge_dim)
        self.gnn_residual_weight = float(gnn_residual_weight)
        self.propagation_alpha = max(0.0, min(1.0, float(propagation_alpha)))
        self.use_response_evidence = bool(use_response_evidence)
        self.student_global = nn.Embedding(self.num_students, self.knowledge_dim)
        self.concept_emb = nn.Embedding(self.num_concepts, self.knowledge_dim)
        self.response_evidence_proj = (
            nn.Linear(2, self.knowledge_dim, bias=False)
            if self.use_response_evidence
            else None
        )
        self.gnn_layers = nn.ModuleList(
            ConceptGraphConv(
                self.knowledge_dim,
                self.knowledge_dim,
                num_heads=num_relation_heads,
                dropout=dropout,
            )
            for _ in range(max(0, int(num_gnn_layers)))
        )
        self.layer_norms = nn.ModuleList(
            nn.LayerNorm(self.knowledge_dim) for _ in range(max(0, int(num_gnn_layers)))
        )
        self.hop_mix_logits = nn.Parameter(torch.zeros(len(self.gnn_layers) + 1))
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_normal_(self.student_global.weight)
        nn.init.xavier_normal_(self.concept_emb.weight)
        if self.response_evidence_proj is not None:
            nn.init.normal_(self.response_evidence_proj.weight, mean=0.0, std=0.02)

    def compose_initial_state(
        self,
        student_ids: torch.Tensor,
        response_evidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        students = self.student_global(student_ids)
        concepts = self.concept_emb.weight.unsqueeze(0).expand(student_ids.size(0), -1, -1)
        initial = concepts + students.unsqueeze(1)
        if response_evidence is not None:
            if self.response_evidence_proj is None:
                raise ValueError("response_evidence was supplied to a disabled evidence encoder")
            expected = (student_ids.size(0), self.num_concepts, 2)
            if tuple(response_evidence.shape) != expected:
                raise ValueError(
                    f"response_evidence must have shape {expected}, got {tuple(response_evidence.shape)}"
                )
            evidence = response_evidence.to(device=initial.device, dtype=initial.dtype)
            initial = initial + self.response_evidence_proj(evidence)
        return self.dropout(initial)

    def forward(
        self,
        student_ids: torch.Tensor,
        relation_matrices: torch.Tensor,
        *,
        response_evidence: Optional[torch.Tensor] = None,
        return_initial: bool = False,
    ):
        initial = self.compose_initial_state(
            student_ids,
            response_evidence=response_evidence,
        )
        if self.propagation_alpha == 0.0 or not self.gnn_layers:
            return (initial, initial) if return_initial else initial
        state = initial
        hop_states = [initial]
        for graph_conv, layer_norm in zip(self.gnn_layers, self.layer_norms):
            propagated = graph_conv(state, relation_matrices)
            candidate = F.relu(layer_norm(state + self.gnn_residual_weight * propagated))
            state = (1.0 - self.propagation_alpha) * state + self.propagation_alpha * candidate
            hop_states.append(state)
        hop_weights = F.softmax(self.hop_mix_logits[: len(hop_states)], dim=0)
        mixed = sum(weight * hop for weight, hop in zip(hop_weights, hop_states))
        if return_initial:
            return mixed, initial
        return mixed
