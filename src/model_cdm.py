"""A single-path, leakage-safe response-evidence Graph-IRT model.

The production prediction path is deliberately small:

    train-only label-free concept-graph priors
        + two-channel leave-one-out train response evidence
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

from src.config import EVIDENCE_ANCHOR_DROPOUT
from src.evidence_completion import MaskedEvidenceCompletion
from src.model_graph import MultiHeadRelationLearning, StudentKnowledgeEncoder
from src.model_regularization import get_regularization_components as _get_regularization_components
from src.prediction_head import CognitiveDiagnosisHead, ExerciseDifficultyEncoder


GRAPH_IRT_ARCHITECTURE = "graph_irt_v10"

# How response evidence anchors theta before the single 2PL readout.
#   full        -> direct rate + difficulty residual + graph-propagated rate
#   direct_only -> direct rate + difficulty residual (no graph transport)
#   mec         -> direct rate + difficulty residual + non-Q evidence completion
#   off         -> evidence feeds only the initial state (v9 behaviour)
EVIDENCE_ANCHOR_MODES = ("full", "direct_only", "mec", "off")
_ANCHOR_CHANNELS = {"full": 3, "direct_only": 2, "mec": 3, "off": 0}


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
        evidence_anchor_mode: str = "full",
        evidence_state_injection: bool = True,
        anchor_multihead_prop: bool = True,
        disable_graph_module: bool = False,
        prediction_head: str = "irt2pl",
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
        anchor_mode = str(evidence_anchor_mode).strip().lower()
        if anchor_mode not in EVIDENCE_ANCHOR_MODES:
            raise ValueError(
                f"evidence_anchor_mode must be one of {EVIDENCE_ANCHOR_MODES}, got {evidence_anchor_mode!r}"
            )
        self.evidence_anchor_mode = anchor_mode if self.use_response_evidence else "off"
        # When False, response statistics skip the initial-state projection
        # and reach theta exclusively through the anchor (production default).
        self.evidence_state_injection = bool(evidence_state_injection)
        self.disable_graph_module = bool(disable_graph_module)
        if self.disable_graph_module and self.evidence_anchor_mode == "full":
            raise ValueError(
                "disable_graph_module requires a non-propagating evidence anchor"
            )
        # One propagated anchor channel per relation head.
        self.anchor_multihead_prop = bool(anchor_multihead_prop)
        # Head probe: "irt2pl" (default single scalar 2PL) or "ncd_mlp"
        # (NCDM-style positive-weight monotone MLP over per-concept 2PL terms).
        self.prediction_head = str(prediction_head)
        if self.evidence_anchor_mode == "full":
            self._anchor_channels = 2 + (
                self.num_relation_heads if self.anchor_multihead_prop else 1
            )
        else:
            self._anchor_channels = _ANCHOR_CHANNELS[self.evidence_anchor_mode]
        # Count-conditioned anchor gates: sigmoid(a + b*log1p(n)) per channel.
        # a=2 starts near fully-open (~0.88) so the initial behaviour matches an
        # ungated anchor; b learns how trust responds to observation counts. The
        # third column is retained for checkpoint compatibility and unused.
        gated_channels = (
            2 if self.evidence_anchor_mode == "mec" else self._anchor_channels
        )
        gate_init = torch.zeros(max(1, gated_channels), 3)
        gate_init[:, 0] = 2.0
        self.anchor_gate = nn.Parameter(gate_init)
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
        self.relation_learning = (
            None
            if self.disable_graph_module
            else MultiHeadRelationLearning(
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
        )

        self.knowledge_encoder = StudentKnowledgeEncoder(
            num_students=self.num_students,
            num_concepts=self.num_concepts,
            knowledge_dim=self.knowledge_dim,
            num_gnn_layers=0 if self.disable_graph_module else num_gnn_layers,
            num_relation_heads=self.num_relation_heads,
            dropout=dropout,
            gnn_residual_weight=gnn_residual_weight,
            propagation_alpha=(
                0.0 if self.disable_graph_module else graph_propagation_alpha
            ),
            use_response_evidence=(
                self.use_response_evidence and self.evidence_state_injection
            ),
        )
        self.diagnosis_head = CognitiveDiagnosisHead(
            knowledge_dim=self.knowledge_dim,
            num_concepts=self.num_concepts,
            evidence_anchor_channels=self._anchor_channels,
            prediction_head=self.prediction_head,
        )
        self.exercise_encoder = ExerciseDifficultyEncoder(num_exercises=self.num_exercises)
        self.evidence_completion = None
        if self.evidence_anchor_mode == "mec":
            # Keep matched seed-42 runs paired beyond identical initial logits:
            # initializing the treatment-only MLP must not advance the global
            # RNG used by data shuffling and row-level anchor dropout.
            with torch.random.fork_rng(devices=[], enabled=True):
                self.evidence_completion = MaskedEvidenceCompletion(
                    num_concepts=self.num_concepts,
                    global_response_count=float(
                        self.response_global_count.item()
                    ),
                )

    def _register_response_evidence(
        self,
        stats: Optional[Dict[str, torch.Tensor]],
    ) -> None:
        names = (
            "student_concept_count",
            "student_concept_correct",
            "student_concept_residual_sum",
            "student_item_keys",
            "student_item_expected_correct",
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
            "student_concept_residual_sum": (
                self.num_students,
                self.num_concepts,
            ),
            "concept_count": (self.num_concepts,),
            "concept_correct": (self.num_concepts,),
            "global_count": (),
            "global_correct": (),
        }
        fixed_names = tuple(
            name
            for name in names
            if name not in {"student_item_keys", "student_item_expected_correct"}
        )
        for name in fixed_names:
            if name not in stats:
                raise ValueError(f"response_evidence_stats is missing {name!r}")
            value = torch.as_tensor(stats[name]).detach().float()
            if tuple(value.shape) != expected_shapes[name]:
                raise ValueError(
                    f"response evidence {name} must have shape {expected_shapes[name]}, "
                    f"got {tuple(value.shape)}"
                )
            self.register_buffer(f"response_{name}", value)

        for name in ("student_item_keys", "student_item_expected_correct"):
            if name not in stats:
                raise ValueError(f"response_evidence_stats is missing {name!r}")
        pair_keys = torch.as_tensor(stats["student_item_keys"]).detach().long()
        pair_expected = (
            torch.as_tensor(stats["student_item_expected_correct"])
            .detach()
            .float()
        )
        if pair_keys.dim() != 1 or pair_expected.shape != pair_keys.shape:
            raise ValueError(
                "student-item response evidence must be aligned one-dimensional tensors"
            )
        if pair_keys.numel() == 0:
            raise ValueError("student-item response evidence cannot be empty")
        if pair_keys.numel() > 1 and bool((pair_keys[1:] <= pair_keys[:-1]).any()):
            raise ValueError("student_item_keys must be strictly increasing")
        if int(pair_keys[0].item()) < 0 or int(pair_keys[-1].item()) >= (
            self.num_students * self.num_exercises
        ):
            raise ValueError("student_item_keys contain an out-of-range pair")
        if not bool(torch.isfinite(pair_expected).all()) or bool(
            ((pair_expected < 0.0) | (pair_expected > 1.0)).any()
        ):
            raise ValueError(
                "student-item expected correctness must be finite and lie in [0, 1]"
            )
        self.register_buffer("response_student_item_keys", pair_keys)
        self.register_buffer(
            "response_student_item_expected_correct",
            pair_expected,
        )

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
        if bool(
            (
                self.response_student_concept_residual_sum.abs()
                > self.response_student_concept_count + 1e-5
            ).any()
        ):
            raise ValueError("response residual sums must lie within their counts")

    def _build_response_evidence(
        self,
        student_ids: torch.Tensor,
        exercise_ids: Optional[torch.Tensor],
        q_vector: torch.Tensor,
        outcome_to_exclude: Optional[torch.Tensor],
        outcome_to_neutralize: Optional[torch.Tensor],
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Return response evidence under an explicit training-query boundary.

        ``outcome_to_exclude`` subtracts the current label, count, and residual
        from every statistic consumed by that training row.  The optional
        ``outcome_to_neutralize`` control retains the current count but replaces
        its correct contribution by the label-independent student-item
        expectation and its residual by zero.  Passing neither control reads
        the complete train-only totals, which is the validation/test contract
        and the deliberately self-included training control.
        """
        if not self.use_response_evidence:
            return None, None, None
        if outcome_to_exclude is not None and outcome_to_neutralize is not None:
            raise ValueError(
                "a query outcome cannot be both excluded and neutralized"
            )

        q_mask = (q_vector > 0).to(dtype=self.response_student_concept_count.dtype)
        correct = self.response_student_concept_correct[student_ids]
        count = self.response_student_concept_count[student_ids]
        residual_sum = self.response_student_concept_residual_sum[student_ids]
        concept_correct = self.response_concept_correct.unsqueeze(0)
        concept_count = self.response_concept_count.unsqueeze(0)

        current_outcome = (
            outcome_to_exclude
            if outcome_to_exclude is not None
            else outcome_to_neutralize
        )
        if current_outcome is not None:
            labels = current_outcome.reshape(-1).to(
                device=student_ids.device,
                dtype=correct.dtype,
            )
            if labels.numel() != student_ids.numel():
                raise ValueError(
                    "the query outcome must contain one value per student id"
                )
            if bool((~torch.isfinite(labels)).any()) or bool(
                ((labels < 0.0) | (labels > 1.0)).any()
            ):
                raise ValueError("the query outcome must be finite and in [0, 1]")
            label_column = labels.unsqueeze(1)
            if exercise_ids is None:
                raise ValueError(
                    "exercise_ids are required when adjusting a training outcome"
                )
            if exercise_ids.numel() != student_ids.numel():
                raise ValueError(
                    "exercise_ids must contain one value per student id"
                )
            pair_query = (
                student_ids.long() * self.num_exercises + exercise_ids.long()
            )
            pair_positions = torch.searchsorted(
                self.response_student_item_keys,
                pair_query,
            )
            safe_positions = pair_positions.clamp(
                max=self.response_student_item_keys.numel() - 1
            )
            matched = self.response_student_item_keys[safe_positions] == pair_query
            if bool((~matched).any()):
                raise ValueError(
                    "training row is missing its train-only student-item expectation"
                )
            expected_correct = self.response_student_item_expected_correct[
                safe_positions
            ].unsqueeze(1)
            current_residual = label_column - expected_correct
            if outcome_to_exclude is not None:
                correct = (correct - label_column * q_mask).clamp(min=0.0)
                count = (count - q_mask).clamp(min=0.0)
                residual_sum = residual_sum - current_residual * q_mask
                concept_correct = (
                    concept_correct - label_column * q_mask
                ).clamp(min=0.0)
                concept_count = (concept_count - q_mask).clamp(min=0.0)
                global_count = (self.response_global_count - 1.0).clamp(min=1.0)
                global_rate = (
                    self.response_global_correct - labels
                ) / global_count
            else:
                neutral_shift = expected_correct - label_column
                correct = (correct + neutral_shift * q_mask).clamp(min=0.0)
                residual_sum = residual_sum - current_residual * q_mask
                concept_correct = (
                    concept_correct + neutral_shift * q_mask
                ).clamp(min=0.0)
                global_rate = (
                    self.response_global_correct + neutral_shift.reshape(-1)
                ) / self.response_global_count
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
        rate_evidence = ((posterior_logit - concept_logit) * reliability).clamp(
            min=-4.0,
            max=4.0,
        )
        residual_evidence = (
            residual_sum / count.clamp(min=1.0) * reliability
        ).clamp(min=-1.0, max=1.0)
        evidence = torch.stack((rate_evidence, residual_evidence), dim=-1)
        return evidence, count, correct

    def _compose_evidence_anchor(
        self,
        relation_matrices: torch.Tensor,
        response_evidence: Optional[torch.Tensor],
        loo_count: Optional[torch.Tensor],
        completion_anchor: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Stack the direct, residual, and optional completion/graph channels.

        The propagated channel transports each row's leave-one-out rate
        evidence over the learned row-stochastic concept graph, so concepts
        without direct observations receive the evidence of their graph
        neighbours.  Each channel is scaled by a learnable count-conditioned
        gate (monotone in log evidence count), letting the anchor trust
        evidence more where more observations back it.  All inputs are
        train-only statistics; every map is linear in them, so the
        leave-one-out contract is preserved exactly.
        """
        if response_evidence is None or self.evidence_anchor_mode == "off":
            return None
        rate_evidence = response_evidence[..., 0]
        residual_evidence = response_evidence[..., 1]
        log_count = torch.log1p(loo_count.to(dtype=rate_evidence.dtype))

        def gated(channel: torch.Tensor, index: int, counts: torch.Tensor) -> torch.Tensor:
            logit = self.anchor_gate[index, 0] + self.anchor_gate[index, 1] * counts
            return channel * torch.sigmoid(logit)

        channels = [
            gated(rate_evidence, 0, log_count),
            gated(residual_evidence, 1, log_count),
        ]
        if self.evidence_anchor_mode == "mec":
            if completion_anchor is None:
                raise ValueError("MEC mode requires a completion anchor")
            channels.append(completion_anchor.to(dtype=rate_evidence.dtype))
        elif self.evidence_anchor_mode == "full":
            if self.anchor_multihead_prop:
                for head in range(self.num_relation_heads):
                    receiver = relation_matrices[head].transpose(0, 1)
                    propagated = torch.matmul(
                        rate_evidence.to(dtype=receiver.dtype), receiver
                    )
                    propagated_count = torch.matmul(log_count, receiver)
                    channels.append(gated(propagated, 2 + head, propagated_count))
            else:
                receiver = relation_matrices.mean(dim=0).transpose(0, 1)
                propagated = torch.matmul(
                    rate_evidence.to(dtype=receiver.dtype), receiver
                )
                propagated_count = torch.matmul(log_count, receiver)
                channels.append(gated(propagated, 2, propagated_count))
        return torch.stack(channels, dim=-1)

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
        outcome_to_neutralize: Optional[torch.Tensor] = None,
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

        relation_matrices = (
            self.identity_relations
            if self.relation_learning is None
            else self.relation_learning()
        )
        response_evidence, loo_count, response_correct = (
            self._build_response_evidence(
                student_ids,
                exercise_ids,
                q_vector,
                outcome_to_exclude,
                outcome_to_neutralize,
            )
        )
        completion_anchor = (
            self.evidence_completion(
                response_evidence,
                loo_count,
                response_correct,
                q_vector,
            )
            if self.evidence_completion is not None
            else None
        )
        knowledge_state, initial_state = self.knowledge_encoder(
            student_ids,
            relation_matrices,
            response_evidence=(
                response_evidence if self.evidence_state_injection else None
            ),
            return_initial=True,
        )

        difficulty, discrimination = self.exercise_encoder(exercise_ids)
        evidence_anchor = self._compose_evidence_anchor(
            relation_matrices,
            response_evidence,
            loo_count,
            completion_anchor,
        )
        if (
            evidence_anchor is not None
            and self.training
            and EVIDENCE_ANCHOR_DROPOUT > 0.0
        ):
            # Inverted row-level evidence dropout: the state path must stay
            # predictive without the statistic shortcut, while the 1/(1-p)
            # rescale keeps the training-time expected anchor equal to the
            # full anchor used at evaluation (no train/eval calibration shift).
            keep = (
                torch.rand(
                    evidence_anchor.size(0),
                    1,
                    1,
                    device=evidence_anchor.device,
                )
                >= EVIDENCE_ANCHOR_DROPOUT
            ).to(dtype=evidence_anchor.dtype)
            evidence_anchor = evidence_anchor * keep / (1.0 - EVIDENCE_ANCHOR_DROPOUT)
        if return_details:
            irt_logit, head_details = self.diagnosis_head(
                knowledge_state=knowledge_state,
                concept_mask=q_vector,
                b=difficulty,
                a=discrimination,
                evidence_anchor=evidence_anchor,
                return_details=True,
            )
        else:
            irt_logit = self.diagnosis_head(
                knowledge_state=knowledge_state,
                concept_mask=q_vector,
                b=difficulty,
                a=discrimination,
                evidence_anchor=evidence_anchor,
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
            (relation_matrices - identity).pow(2).sum(dim=-1).sqrt().mean()
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
                else q_vector.detach().new_zeros((*q_vector.shape, 2))
            ),
            "evidence_anchor": (
                evidence_anchor.detach()
                if evidence_anchor is not None
                else q_vector.detach().new_zeros((*q_vector.shape, 3))
            ),
            "mec_anchor": (
                completion_anchor.detach()
                if completion_anchor is not None
                else q_vector.detach().new_zeros(q_vector.shape)
            ),
            "response_evidence_leave_one_out": q_vector.detach().new_tensor(
                float(outcome_to_exclude is not None)
            ),
            "response_evidence_neutralized": q_vector.detach().new_tensor(
                float(outcome_to_neutralize is not None)
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
        """Return unconditional mastery; query-conditioned MEC is zero here."""
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
                relation_matrices = (
                    self.identity_relations
                    if self.relation_learning is None
                    else self.relation_learning()
                )
                diagnosis_q = torch.ones(
                    (1, self.num_concepts),
                    device=device,
                    dtype=self.q_matrix.dtype,
                )
                response_evidence, loo_count, response_correct = (
                    self._build_response_evidence(
                        student_ids,
                        None,
                        diagnosis_q,
                        outcome_to_exclude=None,
                        outcome_to_neutralize=None,
                    )
                )
                completion_anchor = (
                    self.evidence_completion(
                        response_evidence,
                        loo_count,
                        response_correct,
                        diagnosis_q,
                    )
                    if self.evidence_completion is not None
                    else None
                )
                knowledge_state = self.knowledge_encoder(
                    student_ids,
                    relation_matrices,
                    response_evidence=(
                        response_evidence if self.evidence_state_injection else None
                    ),
                )
                concept_state = knowledge_state.squeeze(0)
                concept_theta = self.diagnosis_head.theta_proj(concept_state).squeeze(-1)
                evidence_anchor = self._compose_evidence_anchor(
                    relation_matrices,
                    response_evidence,
                    loo_count,
                    completion_anchor,
                )
                if evidence_anchor is not None:
                    anchor_weights = self.diagnosis_head.evidence_anchor_weights()
                    concept_theta = concept_theta + (
                        evidence_anchor.squeeze(0) * anchor_weights
                    ).sum(dim=-1)
                return {
                    "knowledge_mastery": torch.sigmoid(concept_theta),
                    "student_repr": concept_state.mean(dim=0),
                    "relation_matrices": relation_matrices,
                }
        finally:
            self.train(was_training)
