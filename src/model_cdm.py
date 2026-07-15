"""A single-path, leakage-safe response-evidence Graph-IRT model.

The production prediction path is deliberately small:

    train-only label-free concept-graph priors
        + leave-one-out train response evidence
        -> student/concept graph encoder
        -> Q-masked scalar ability
        -> scalar-difficulty 2PL-IRT logit

Response sufficient statistics are built from train only.  During training the
current row is subtracted before its features are formed; validation and test
consume only the complete training statistics.  No validation/test outcome is
registered or accepted by the model.
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from src.model_graph import MultiHeadRelationLearning, StudentKnowledgeEncoder
from src.model_regularization import get_regularization_components as _get_regularization_components
from src.prediction_head import CognitiveDiagnosisHead, ExerciseDifficultyEncoder


GRAPH_IRT_ARCHITECTURE = "graph_irt_v8"


class CognitiveDiagnosisModel(nn.Module):
    """Graph encoder followed by one Q-masked 2PL-IRT prediction head."""

    architecture = GRAPH_IRT_ARCHITECTURE

    def __init__(
        self,
        num_students: int,
        num_exercises: int,
        num_concepts: int,
        q_matrix: torch.Tensor,
        item_prior_matrix: Optional[torch.Tensor] = None,
        exposure_prior_matrix: Optional[torch.Tensor] = None,
        response_evidence_stats: Optional[Dict[str, torch.Tensor]] = None,
        use_response_evidence: bool = False,
        knowledge_dim: int = 32,
        num_relation_heads: int = 4,
        num_gnn_layers: int = 2,
        dropout: float = 0.3,
        graph_topk: Optional[int] = None,
        allow_self_loop: bool = True,
        graph_identity_residual: float = 0.0,
        graph_dropout: Optional[float] = None,
        graph_tau_init: float = 1.0,
        graph_propagation_alpha: float = 0.20,
        graph_prior_strength_init: float = 1.0,
        gnn_residual_weight: float = 0.5,
        lambda_graph_entropy: float = 0.01,
        graph_entropy_min: float = 0.15,
        graph_entropy_max: float = 0.85,
        lambda_graph_diag: float = 0.10,
        lambda_graph_uniform: float = 0.04,
        graph_uniform_margin: float = 0.10,
        graph_reg_warmup_epochs: int = 1,
        graph_reg_cap_ratio: float = 6.0,
        prediction_l2_lambda: float = 5e-5,
    ):
        super().__init__()
        self.num_students = int(num_students)
        self.num_exercises = int(num_exercises)
        self.num_concepts = int(num_concepts)
        self.knowledge_dim = int(knowledge_dim)
        self.num_relation_heads = int(num_relation_heads)
        self.use_response_evidence = bool(use_response_evidence)
        self.lambda_graph_entropy = max(0.0, float(lambda_graph_entropy))
        self.graph_entropy_min = float(graph_entropy_min)
        self.graph_entropy_max = float(graph_entropy_max)
        if self.graph_entropy_min > self.graph_entropy_max:
            self.graph_entropy_min, self.graph_entropy_max = (
                self.graph_entropy_max,
                self.graph_entropy_min,
            )
        self.lambda_graph_diag = max(0.0, float(lambda_graph_diag))
        self.lambda_graph_uniform = max(0.0, float(lambda_graph_uniform))
        self.graph_uniform_margin = max(0.0, float(graph_uniform_margin))
        self.graph_reg_warmup_epochs = max(0, int(graph_reg_warmup_epochs))
        self.graph_reg_cap_ratio = max(0.0, float(graph_reg_cap_ratio))
        self.prediction_l2_lambda = max(0.0, float(prediction_l2_lambda))
        self._current_epoch = 1

        q = q_matrix.detach().float()
        expected_q_shape = (self.num_exercises, self.num_concepts)
        if tuple(q.shape) != expected_q_shape:
            raise ValueError(f"q_matrix must have shape {expected_q_shape}, got {tuple(q.shape)}")
        self.register_buffer("q_matrix", q)
        self._register_response_evidence(response_evidence_stats)

        item_prior = (
            self._build_item_prior_from_q(q, self.num_concepts)
            if item_prior_matrix is None
            else self._validate_prior_matrix(item_prior_matrix, self.num_concepts, "item_prior_matrix")
        )
        exposure_prior = (
            None
            if exposure_prior_matrix is None
            else self._validate_prior_matrix(
                exposure_prior_matrix,
                self.num_concepts,
                "exposure_prior_matrix",
            )
        )
        self.register_buffer("item_prior_matrix", item_prior, persistent=False)
        if exposure_prior is None:
            self.exposure_prior_matrix = None
        else:
            self.register_buffer("exposure_prior_matrix", exposure_prior, persistent=False)

        identity = torch.eye(self.num_concepts, dtype=torch.float32)
        identity = identity.unsqueeze(0).repeat(self.num_relation_heads, 1, 1)
        self.register_buffer("identity_relations", identity, persistent=False)

        relation_dropout = float(dropout if graph_dropout is None else graph_dropout)
        self.relation_learning = MultiHeadRelationLearning(
            num_concepts=self.num_concepts,
            num_heads=self.num_relation_heads,
            dropout=relation_dropout,
            tau_init=float(graph_tau_init),
            topk=graph_topk,
            allow_self_loop=allow_self_loop,
            identity_residual=graph_identity_residual,
            prior_matrix=self.item_prior_matrix,
            exposure_prior_matrix=exposure_prior,
            prior_strength_init=graph_prior_strength_init,
        )

        self.knowledge_encoder = StudentKnowledgeEncoder(
            num_students=self.num_students,
            num_concepts=self.num_concepts,
            knowledge_dim=self.knowledge_dim,
            num_gnn_layers=num_gnn_layers,
            num_relation_heads=self.num_relation_heads,
            dropout=dropout,
            gnn_residual_weight=gnn_residual_weight,
            propagation_alpha=graph_propagation_alpha,
            use_response_evidence=self.use_response_evidence,
        )
        self.diagnosis_head = CognitiveDiagnosisHead(
            knowledge_dim=self.knowledge_dim,
            num_concepts=self.num_concepts,
        )
        self.exercise_encoder = ExerciseDifficultyEncoder(num_exercises=self.num_exercises)

    def _register_response_evidence(
        self,
        stats: Optional[Dict[str, torch.Tensor]],
    ) -> None:
        names = (
            "student_concept_count",
            "student_concept_correct",
            "concept_count",
            "concept_correct",
            "global_count",
            "global_correct",
        )
        if not self.use_response_evidence:
            for name in names:
                setattr(self, f"response_{name}", None)
            return
        if stats is None:
            raise ValueError("full Graph-IRT requires train-only response_evidence_stats")

        expected_shapes = {
            "student_concept_count": (self.num_students, self.num_concepts),
            "student_concept_correct": (self.num_students, self.num_concepts),
            "concept_count": (self.num_concepts,),
            "concept_correct": (self.num_concepts,),
            "global_count": (),
            "global_correct": (),
        }
        for name in names:
            if name not in stats:
                raise ValueError(f"response_evidence_stats is missing {name!r}")
            value = torch.as_tensor(stats[name]).detach().float()
            if tuple(value.shape) != expected_shapes[name]:
                raise ValueError(
                    f"response evidence {name} must have shape {expected_shapes[name]}, "
                    f"got {tuple(value.shape)}"
                )
            self.register_buffer(f"response_{name}", value)

        if bool((self.response_student_concept_count < 0).any()):
            raise ValueError("response evidence counts must be non-negative")
        if bool(
            (self.response_student_concept_correct < 0).any()
            or (
                self.response_student_concept_correct
                > self.response_student_concept_count
            ).any()
        ):
            raise ValueError("response evidence correct sums must lie within counts")
        if float(self.response_global_count.item()) <= 0.0:
            raise ValueError("response evidence global_count must be positive")

    def _build_response_evidence(
        self,
        student_ids: torch.Tensor,
        q_vector: torch.Tensor,
        outcome_to_exclude: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Return empirical student-concept evidence with exact train LOO.

        ``outcome_to_exclude`` is supplied only for training rows.  Its label
        and one count are subtracted from every concept attached to that row
        before either the student-concept posterior or its concept prior is
        computed.  Therefore the current target cannot be copied into its own
        prediction.
        """
        if not self.use_response_evidence:
            return None

        q_mask = (q_vector > 0).to(dtype=self.response_student_concept_count.dtype)
        correct = self.response_student_concept_correct[student_ids]
        count = self.response_student_concept_count[student_ids]
        concept_correct = self.response_concept_correct.unsqueeze(0)
        concept_count = self.response_concept_count.unsqueeze(0)

        if outcome_to_exclude is not None:
            labels = outcome_to_exclude.reshape(-1).to(
                device=student_ids.device,
                dtype=correct.dtype,
            )
            if labels.numel() != student_ids.numel():
                raise ValueError(
                    "outcome_to_exclude must contain one value per student id"
                )
            if bool((~torch.isfinite(labels)).any()) or bool(
                ((labels < 0.0) | (labels > 1.0)).any()
            ):
                raise ValueError("outcome_to_exclude must be finite and in [0, 1]")
            label_column = labels.unsqueeze(1)
            correct = (correct - label_column * q_mask).clamp(min=0.0)
            count = (count - q_mask).clamp(min=0.0)
            concept_correct = (concept_correct - label_column * q_mask).clamp(min=0.0)
            concept_count = (concept_count - q_mask).clamp(min=0.0)
            global_count = (self.response_global_count - 1.0).clamp(min=1.0)
            global_rate = (self.response_global_correct - labels) / global_count
            global_rate = global_rate.unsqueeze(1)
        else:
            global_rate = self.response_global_correct / self.response_global_count

        # One empirical-Bayes pseudo-observation is fixed by the architecture;
        # there is no dataset-specific smoothing knob to tune.
        concept_rate = (concept_correct + global_rate) / (concept_count + 1.0)
        posterior = (correct + concept_rate) / (count + 1.0)
        reliability = count / (count + 1.0)
        eps = 1e-4
        concept_logit = torch.logit(concept_rate.clamp(min=eps, max=1.0 - eps))
        posterior_logit = torch.logit(posterior.clamp(min=eps, max=1.0 - eps))
        return ((posterior_logit - concept_logit) * reliability).clamp(min=-4.0, max=4.0)

    @staticmethod
    def _build_item_prior_from_q(q_matrix: torch.Tensor, num_concepts: int) -> torch.Tensor:
        count = int(num_concepts)
        if count <= 0:
            raise ValueError(f"num_concepts must be positive, got {num_concepts}")
        if count == 1:
            return torch.ones(1, 1, dtype=torch.float32)
        concept_occurs = (q_matrix.detach().float() > 0).float()
        cooccurrence = concept_occurs.t().matmul(concept_occurs)
        eye = torch.eye(count, dtype=torch.float32, device=cooccurrence.device)
        offdiag = 1.0 - eye
        cooccurrence = cooccurrence * offdiag
        uniform = offdiag / float(count - 1)
        row_sum = cooccurrence.sum(dim=-1, keepdim=True)
        empirical = cooccurrence / row_sum.clamp(min=1.0)
        empirical = torch.where(row_sum > 0, empirical, uniform)
        prior = 0.90 * empirical + 0.10 * uniform
        prior = prior * offdiag
        return (prior / prior.sum(dim=-1, keepdim=True).clamp(min=1e-12)).float()

    @staticmethod
    def _validate_prior_matrix(
        prior_matrix: torch.Tensor,
        num_concepts: int,
        name: str,
    ) -> torch.Tensor:
        count = int(num_concepts)
        prior = prior_matrix.detach().float()
        expected = (count, count)
        if tuple(prior.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(prior.shape)}")
        if count == 1:
            return torch.ones(1, 1, dtype=torch.float32)
        eye = torch.eye(count, dtype=torch.float32, device=prior.device)
        prior = prior.clamp(min=0.0) * (1.0 - eye)
        row_sum = prior.sum(dim=-1, keepdim=True)
        return torch.where(row_sum > 0, prior / row_sum.clamp(min=1e-12), torch.zeros_like(prior))

    def set_epoch(self, epoch: int) -> None:
        self._current_epoch = max(1, int(epoch))

    def _get_graph_reg_ramp(self) -> float:
        if self.graph_reg_warmup_epochs <= 0:
            return 1.0
        return min(1.0, float(self._current_epoch) / float(self.graph_reg_warmup_epochs))

    def forward(
        self,
        student_ids: torch.Tensor,
        exercise_ids: torch.Tensor,
        concept_vector: Optional[torch.Tensor] = None,
        outcome_to_exclude: Optional[torch.Tensor] = None,
        return_details: bool = False,
        return_logits: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Run the sole Graph-IRT prediction path.

        The public signature intentionally matches the previous model.  No
        outcome label or label-derived lookup table is accepted here.
        """
        q_vector = (
            self.q_matrix[exercise_ids]
            if concept_vector is None
            else concept_vector.to(device=student_ids.device, dtype=self.q_matrix.dtype)
        )
        if q_vector.dim() != 2 or q_vector.size(1) != self.num_concepts:
            raise ValueError(
                f"concept_vector must have shape (batch, {self.num_concepts}), got {tuple(q_vector.shape)}"
            )

        relation_matrices = self.relation_learning()
        response_evidence = self._build_response_evidence(
            student_ids,
            q_vector,
            outcome_to_exclude,
        )
        knowledge_state, initial_state = self.knowledge_encoder(
            student_ids,
            relation_matrices,
            response_evidence=response_evidence,
            return_initial=True,
        )

        difficulty, discrimination = self.exercise_encoder(exercise_ids)
        if return_details:
            irt_logit, head_details = self.diagnosis_head(
                knowledge_state=knowledge_state,
                concept_mask=q_vector,
                b=difficulty,
                a=discrimination,
                return_details=True,
            )
        else:
            irt_logit = self.diagnosis_head(
                knowledge_state=knowledge_state,
                concept_mask=q_vector,
                b=difficulty,
                a=discrimination,
                return_details=False,
            )
            head_details = None

        # Architectural invariant: there is no residual or calibration branch.
        total_logit = irt_logit
        output = total_logit if return_logits else torch.sigmoid(total_logit)
        if not return_details:
            return output

        identity = self.identity_relations.to(
            device=relation_matrices.device,
            dtype=relation_matrices.dtype,
        )
        relation_identity_delta = (
            (relation_matrices - identity).pow(2).sum(dim=-1).clamp(min=1e-12).sqrt().mean()
        )
        graph_state_delta = (knowledge_state - initial_state).pow(2).mean().sqrt()
        details: Dict[str, torch.Tensor] = {
            "relation_matrices": relation_matrices,
            "initial_state": initial_state.detach(),
            "knowledge_state": knowledge_state.detach(),
            "knowledge_state_graph_delta": graph_state_delta.detach(),
            "relation_identity_delta": relation_identity_delta.detach(),
            "q_vector": q_vector.detach(),
            "response_evidence": (
                response_evidence.detach()
                if response_evidence is not None
                else q_vector.detach().new_zeros(q_vector.shape)
            ),
            "response_evidence_leave_one_out": q_vector.detach().new_tensor(
                float(outcome_to_exclude is not None)
            ),
            "irt_b": difficulty.detach(),
            "irt_a": discrimination.detach(),
            "irt_logit": irt_logit.detach(),
            "logits": total_logit.detach(),
        }
        if head_details is not None:
            details.update(head_details)
        return output, details

    def get_regularization_components(
        self,
        relation_matrices: torch.Tensor,
        details: Optional[Dict[str, torch.Tensor]] = None,
        base_loss: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return _get_regularization_components(
            self,
            relation_matrices=relation_matrices,
            details=details,
            base_loss=base_loss,
        )

    def get_student_diagnosis(self, student_id: int) -> Dict[str, torch.Tensor]:
        """Return graph-encoded concept mastery for one known student."""
        if student_id < 0 or student_id >= self.num_students:
            raise IndexError(
                f"student_id must be in [0, {self.num_students}), got {student_id}"
            )
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                device = next(self.parameters()).device
                student_ids = torch.tensor([student_id], device=device, dtype=torch.long)
                relation_matrices = self.relation_learning()
                response_evidence = self._build_response_evidence(
                    student_ids,
                    torch.ones(
                        (1, self.num_concepts),
                        device=device,
                        dtype=self.q_matrix.dtype,
                    ),
                    outcome_to_exclude=None,
                )
                knowledge_state = self.knowledge_encoder(
                    student_ids,
                    relation_matrices,
                    response_evidence=response_evidence,
                )
                concept_state = knowledge_state.squeeze(0)
                concept_theta = self.diagnosis_head.theta_proj(concept_state).squeeze(-1)
                return {
                    "knowledge_mastery": torch.sigmoid(concept_theta),
                    "student_repr": concept_state.mean(dim=0),
                    "relation_matrices": relation_matrices,
                }
        finally:
            self.train(was_training)
