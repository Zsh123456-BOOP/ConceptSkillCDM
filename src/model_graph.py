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
    ):
        super().__init__()
        self.num_concepts = int(num_concepts)
        self.num_heads = int(num_heads)
        self.topk = topk
        self.allow_self_loop = bool(allow_self_loop)
        self.identity_residual = (
            float(max(0.0, min(1.0, identity_residual))) if self.allow_self_loop else 0.0
        )
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
        if self.topk is not None and self.topk > 0 and not self.allow_self_loop:
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

        if not self.allow_self_loop:
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

            base_adjacency = F.softmax(scores, dim=-1)
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
    """Encode one concept state per student through the global concept graph."""

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
    ):
        super().__init__()
        self.num_students = int(num_students)
        self.num_concepts = int(num_concepts)
        self.knowledge_dim = int(knowledge_dim)
        self.gnn_residual_weight = float(gnn_residual_weight)
        self.propagation_alpha = max(0.0, min(1.0, float(propagation_alpha)))
        self.student_global = nn.Embedding(self.num_students, self.knowledge_dim)
        self.concept_emb = nn.Embedding(self.num_concepts, self.knowledge_dim)
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

    def compose_initial_state(
        self,
        student_ids: torch.Tensor,
    ) -> torch.Tensor:
        students = self.student_global(student_ids)
        concepts = self.concept_emb.weight.unsqueeze(0).expand(student_ids.size(0), -1, -1)
        return self.dropout(concepts + students.unsqueeze(1))

    def forward(
        self,
        student_ids: torch.Tensor,
        relation_matrices: torch.Tensor,
        *,
        return_initial: bool = False,
    ):
        initial = self.compose_initial_state(student_ids)
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
