# src/trainer.py
import math
import json
import os
import traceback
import warnings
from typing import Tuple, Dict, Any, Optional, Union, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 过滤 DataParallel 的 gather 标量警告（这是正常行为，不影响结果）
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0")

from src.dataset import CognitiveDiagnosisDataset, create_dataloaders
from src.model import CognitiveDiagnosisModel
from src.experiment_utils import (
    compute_metrics,
    select_device,
    save_epoch_history_csv,
    append_summary_csv,
)
from src.module_activity import (
    compute_module_activity,
    format_activity_brief,
    format_activity_report,
)

# =========================
# Global toggles
# =========================
# If True: fail fast when checkpoint/model keys mismatch (recommended for ablations)
STRICT_CHECKPOINT_LOADING = True
MONITOR_NAME = "val_auc"
MONITOR_MODE = "max"
STRUCTURAL_SWITCH_KEYS: Tuple[str, ...] = (
    "graph_prior_mode",
    "disable_item_prior",
    "disable_sequence_prior",
    "share_concept_embeddings",
    "personal_alpha_temperature",
    "personal_alpha_budget",
    "personal_alpha_base_init",
    "personal_alpha_bias_scale",
    "personal_freeze_alpha_gate",
    "personal_reg_warmup_epochs",
    "personal_disable_student_global_context",
    "personal_local_hops",
    "personal_include_neighbor_rows",
    "personal_support_include_query_self",
    "personal_support_include_graph",
    "personal_support_include_neighbors",
    "personal_item_support_mass",
    "personal_mastery_prior_scale",
    "personal_recent_mastery_prior_scale",
    "personal_mastery_count_smoothing",
    "personal_query_row_budget",
    "personal_neighbor_row_budget",
    "personal_query_support_hops",
    "personal_support_only",
    "personal_query_correction_scale",
    "personal_query_correction_max_ratio",
    "personal_query_correction_min_graph_anchor",
    "personal_query_message_gain",
    "personal_value_use_global_basis",
    "personal_message_alignment_gate",
    "personal_projection_hidden_factor",
    "use_personal_graph",
    "use_concept_graph",
    "graph_identity_residual",
    "graph_propagation_alpha",
    "graph_query_readout_scale",
    "graph_query_readout_2hop_scale",
    "graph_headwise_query_gate",
    "graph_edge_bias_rank",
    "graph_prior_logit_scale",
    "ae_query_residual_scale",
    "ae_logit_residual_scale",
    "ae_logit_residual_clip",
    "ae_irt_logit_scale",
    "ae_interaction_logit_scale",
    "ae_logit_dim",
    "ae_posterior_prior_scale",
    "lambda_theta_prior_align",
    "ae_lr_mult",
    "ae_stat_prior_scale",
    "relation_theta_scale",
    "relation_theta_delta_clip",
    "graph_query_adapter_enable",
    "personal_delta_scale",
    "personal_warmup_epochs",
    "lambda_personal_kl",
    "lambda_personal_query_residual",
    "personal_query_residual_margin",
    "lambda_alpha_min",
    "alpha_min_target",
    "personal_state_lr_mult",
    "personal_id_lr_mult",
)


class NonFiniteTrainingError(RuntimeError):
    def __init__(
        self,
        *,
        reason: str,
        stage: str,
        epoch: int,
        batch_idx: int,
        payload: Dict[str, Any],
    ) -> None:
        self.reason = reason
        self.stage = stage
        self.epoch = int(epoch)
        self.batch_idx = int(batch_idx)
        self.payload = payload
        super().__init__(f"{reason} at {stage} epoch={epoch} batch={batch_idx}")

    def to_failure_dict(self) -> Dict[str, Any]:
        return {
            "reason": self.reason,
            "stage": self.stage,
            "epoch": self.epoch,
            "batch_idx": self.batch_idx,
            "payload": self.payload,
        }

# ======================================================
# Helpers
# ======================================================

def _sigmoid_torch(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable sigmoid for metrics; keep in torch for speed."""
    return torch.sigmoid(x)


def _ensure_1d(t: torch.Tensor) -> torch.Tensor:
    """Ensure tensor shape (B,) for logits/labels. Use reshape to handle non-contiguous tensors."""
    return t.reshape(-1)


def _resolve_optional_graph_dropout(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return None if val < 0 else val


def _default_monitor_config() -> Dict[str, str]:
    return {
        "scheduler_monitor": MONITOR_NAME,
        "scheduler_mode": MONITOR_MODE,
        "best_monitor": MONITOR_NAME,
        "best_mode": MONITOR_MODE,
        "early_stop_monitor": MONITOR_NAME,
        "early_stop_mode": MONITOR_MODE,
    }


def _collect_structural_switches(source: Any) -> Dict[str, Any]:
    return {key: getattr(source, key, None) if not isinstance(source, dict) else source.get(key) for key in STRUCTURAL_SWITCH_KEYS}


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().reshape(-1)[0].item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0
        return int(value.detach().reshape(-1)[0].item())
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _fmt_diag_value(value: Optional[float], fmt: str = ".4f") -> str:
    if value is None:
        return "NA"
    return format(float(value), fmt)


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 0:
            return []
        if obj.numel() == 1:
            return float(obj.detach().item())
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _write_diag_json(save_dir: str, filename: str, payload: Dict[str, Any]) -> None:
    path = os.path.join(save_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, indent=2, ensure_ascii=False)


def _tensor_stats(name: str, tensor: Optional[torch.Tensor]) -> Dict[str, Any]:
    if tensor is None:
        return {"name": name, "present": False}
    det = tensor.detach()
    safe = torch.nan_to_num(det, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "name": name,
        "present": True,
        "shape": list(det.shape),
        "nonfinite_count": int((~torch.isfinite(det)).sum().item()),
        "mean": float(safe.mean().item()),
        "std": float(safe.std(unbiased=False).item()) if det.numel() > 1 else 0.0,
        "absmax": float(safe.abs().max().item()) if det.numel() > 0 else 0.0,
    }


def _first_nonfinite_reason(
    details: Dict[str, Any],
    logits: torch.Tensor,
    bce_loss: torch.Tensor,
    reg_terms: Dict[str, torch.Tensor],
    loss: torch.Tensor,
) -> Optional[str]:
    tensor_checks = (
        ("nonfinite_alpha", details.get("alpha_effective", details.get("alpha"))),
        ("nonfinite_alpha_logit", details.get("alpha_logit")),
        ("nonfinite_knowledge_state", details.get("knowledge_state")),
        ("nonfinite_logits", logits),
        ("nonfinite_bce_loss", bce_loss),
        ("nonfinite_reg_loss", reg_terms.get("total")),
        ("nonfinite_loss", loss),
    )
    for reason, tensor in tensor_checks:
        if tensor is not None and not torch.isfinite(tensor).all():
            return reason

    scalar_count_checks = (
        ("nonfinite_personal_delta", _to_int(details.get("personal_delta_nonfinite_count"))),
        ("nonfinite_personal_logits", _to_int(details.get("personal_logits_nonfinite_count"))),
        ("nonfinite_personal_matrices", _to_int(details.get("personal_matrix_nonfinite_count"))),
    )
    for reason, count in scalar_count_checks:
        if count > 0:
            return reason
    return None


def _build_nonfinite_payload(
    *,
    stage: str,
    epoch: int,
    batch_idx: int,
    details: Dict[str, Any],
    logits: torch.Tensor,
    bce_loss: torch.Tensor,
    reg_terms: Dict[str, torch.Tensor],
    loss: torch.Tensor,
    reason: str,
) -> Dict[str, Any]:
    loss_terms = {
        "bce_loss": _to_float(bce_loss),
        "reg_loss": _to_float(reg_terms.get("total")),
        "loss": _to_float(loss),
    }
    for key in _REG_COMPONENT_KEYS:
        loss_terms[key] = _to_float(reg_terms.get(key))

    return {
        "reason": reason,
        "stage": stage,
        "epoch": int(epoch),
        "batch_idx": int(batch_idx),
        "alpha": _tensor_stats("alpha", details.get("alpha_effective", details.get("alpha"))),
        "alpha_logit": _tensor_stats("alpha_logit", details.get("alpha_logit")),
        "knowledge_state": _tensor_stats("knowledge_state", details.get("knowledge_state")),
        "logits": _tensor_stats("logits", logits),
        "loss_terms": loss_terms,
        "ae_diagnostics": {
            "alpha_state_path_absmean": _to_float(details.get("alpha_state_path_absmean")),
            "alpha_id_path_absmean": _to_float(details.get("alpha_id_path_absmean")),
            "alpha_bias_path_absmean": _to_float(details.get("alpha_bias_path_absmean")),
            "head_bias_path_absmean": _to_float(details.get("head_bias_path_absmean")),
            "alpha_id_adapter_scale": _to_float(details.get("alpha_id_adapter_scale")),
            "state_embedding_nonfinite_count": _to_int(details.get("state_embedding_nonfinite_count")),
            "context_repr_nonfinite_count": _to_int(details.get("context_repr_nonfinite_count")),
            "state_logit_nonfinite_count": _to_int(details.get("state_logit_nonfinite_count")),
            "id_logit_nonfinite_count": _to_int(details.get("id_logit_nonfinite_count")),
            "alpha_base_nonfinite_count": _to_int(details.get("alpha_base_nonfinite_count")),
            "alpha_preclamp_nonfinite_count": _to_int(details.get("alpha_preclamp_nonfinite_count")),
            "used_no_a_support_fallback_mode": _to_int(details.get("used_no_a_support_fallback_mode")),
            "personal_delta_nonfinite_count": _to_int(details.get("personal_delta_nonfinite_count")),
            "personal_delta_absmax": _to_float(details.get("personal_delta_absmax")),
            "personal_delta_pre_softmax_norm": _to_float(details.get("personal_delta_pre_softmax_norm")),
            "personal_delta_student_std": _to_float(details.get("personal_delta_student_std")),
            "personal_logits_nonfinite_count": _to_int(details.get("personal_logits_nonfinite_count")),
            "personal_logits_absmax": _to_float(details.get("personal_logits_absmax")),
            "personal_matrix_nonfinite_count": _to_int(details.get("personal_matrix_nonfinite_count")),
            "personal_matrix_delta": _to_float(details.get("personal_matrix_delta")),
            "personal_matrix_student_std": _to_float(details.get("personal_matrix_student_std")),
            "personal_bad_row_count": _to_int(details.get("personal_bad_row_count")),
            "personal_fallback_row_count": _to_int(details.get("personal_fallback_row_count")),
            "personal_bad_row_count_active": _to_int(details.get("personal_bad_row_count_active")),
            "personal_fallback_row_count_active": _to_int(details.get("personal_fallback_row_count_active")),
            "personal_bad_row_rate_active": _to_float(details.get("personal_bad_row_rate_active")),
            "personal_padded_row_count": _to_int(details.get("personal_padded_row_count")),
            "personal_state_mix": _to_float(details.get("personal_state_mix")),
            "personal_student_mix": _to_float(details.get("personal_student_mix")),
            "personal_mastery_prior_scale": _to_float(details.get("personal_mastery_prior_scale")),
            "personal_mastery_scores_absmean": _to_float(details.get("personal_mastery_scores_absmean")),
            "personal_recent_mastery_prior_scale": _to_float(details.get("personal_recent_mastery_prior_scale")),
            "personal_recent_mastery_scores_absmean": _to_float(details.get("personal_recent_mastery_scores_absmean")),
            "personal_student_adapter_scale": _to_float(details.get("personal_student_adapter_scale")),
            "personal_context_adapter_scale": _to_float(details.get("personal_context_adapter_scale")),
            "personal_logits_support_absmax": _to_float(details.get("personal_logits_support_absmax")),
            "personal_query_trust_scale_mean": _to_float(details.get("personal_query_trust_scale_mean")),
            "query_row_delta_local_rms_raw": _to_float(details.get("query_row_delta_local_rms_raw")),
            "query_row_message_projection_gain": _to_float(details.get("query_row_message_projection_gain")),
            "relation_identity_delta": _to_float(details.get("relation_identity_delta")),
            "knowledge_state_graph_delta": _to_float(details.get("knowledge_state_graph_delta")),
            "knowledge_state_personal_delta": _to_float(details.get("knowledge_state_personal_delta")),
        },
    }


def _raise_if_nonfinite(
    *,
    stage: str,
    epoch: int,
    batch_idx: int,
    details: Dict[str, Any],
    logits: torch.Tensor,
    bce_loss: torch.Tensor,
    reg_terms: Dict[str, torch.Tensor],
    loss: torch.Tensor,
) -> None:
    reason = _first_nonfinite_reason(details, logits, bce_loss, reg_terms, loss)
    if reason is None:
        return
    raise NonFiniteTrainingError(
        reason=reason,
        stage=stage,
        epoch=epoch,
        batch_idx=batch_idx,
        payload=_build_nonfinite_payload(
            stage=stage,
            epoch=epoch,
            batch_idx=batch_idx,
            details=details,
            logits=logits,
            bce_loss=bce_loss,
            reg_terms=reg_terms,
            loss=loss,
            reason=reason,
        ),
    )


def _write_alpha_failure_debug(save_dir: str, payload: Dict[str, Any]) -> None:
    if not save_dir:
        return
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "alpha_failure_debug.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _dedupe_params(params: List[torch.nn.Parameter]) -> List[torch.nn.Parameter]:
    out: List[torch.nn.Parameter] = []
    seen: set[int] = set()
    for param in params:
        if param is None or not isinstance(param, torch.nn.Parameter):
            continue
        pid = id(param)
        if pid in seen:
            continue
        seen.add(pid)
        out.append(param)
    return out


def _sanitize_nonfinite_grads(
    named_params: List[Tuple[str, torch.nn.Parameter]],
) -> Tuple[int, List[str]]:
    nonfinite_count = 0
    bad_names: List[str] = []
    for name, param in named_params:
        grad = param.grad
        if grad is None:
            continue
        finite = torch.isfinite(grad)
        if bool(finite.all().item()):
            continue
        bad = int((~finite).sum().item())
        nonfinite_count += bad
        bad_names.append(name)
        grad.copy_(torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0))
    return nonfinite_count, bad_names[:8]


def _clip_grad_group(
    named_params: List[Tuple[str, torch.nn.Parameter]],
    max_norm: float,
) -> Tuple[float, int, List[str]]:
    deduped: List[Tuple[str, torch.nn.Parameter]] = []
    seen: set[int] = set()
    for name, param in named_params:
        if param is None or not isinstance(param, torch.nn.Parameter):
            continue
        pid = id(param)
        if pid in seen:
            continue
        seen.add(pid)
        deduped.append((name, param))
    params = [p for _, p in deduped if p.requires_grad and p.grad is not None]
    if not params:
        return 0.0, 0, []
    nonfinite_count, bad_names = _sanitize_nonfinite_grads(
        [(name, p) for name, p in deduped if p.requires_grad and p.grad is not None]
    )
    norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
    norm_value = float(norm.item() if isinstance(norm, torch.Tensor) else norm)
    if not math.isfinite(norm_value):
        for param in params:
            if param.grad is not None:
                param.grad.zero_()
        nonfinite_count += sum(int(p.grad.numel()) for p in params if p.grad is not None)
        bad_names = bad_names or ["<group_total_norm>"]
        norm_value = 0.0
    return norm_value, nonfinite_count, bad_names


def _clip_stability_sensitive_grads(model: nn.Module) -> Dict[str, Any]:
    base_model = _get_base_model(model)
    structure_module = getattr(base_model, "structure_module", None)
    relation_learning = getattr(structure_module, "relation_learning", None) if structure_module is not None else None
    adaptive_gate = getattr(structure_module, "adaptive_gate", None) if structure_module is not None else None
    personal_generator = getattr(structure_module, "personal_generator", None) if structure_module is not None else None
    personal_modules = [
        adaptive_gate,
        personal_generator,
    ]

    graph_named_params = (
        [(f"structure_module.relation_learning.{name}", param) for name, param in relation_learning.named_parameters()]
        if relation_learning is not None
        else []
    )
    personal_named_params: List[Tuple[str, torch.nn.Parameter]] = []
    for module in personal_modules:
        if module is None:
            continue
        module_prefix = "adaptive_gate" if module is adaptive_gate else "personal_generator"
        personal_named_params.extend(
            [(f"structure_module.{module_prefix}.{name}", param) for name, param in module.named_parameters()]
        )

    graph_ids = {id(p) for _, p in graph_named_params}
    personal_ids = {id(p) for _, p in personal_named_params}
    other_named_params = [
        (name, param)
        for name, param in base_model.named_parameters()
        if id(param) not in graph_ids and id(param) not in personal_ids
    ]

    graph_norm, graph_bad, graph_names = _clip_grad_group(graph_named_params, max_norm=1.5)
    personal_norm, personal_bad, personal_names = _clip_grad_group(personal_named_params, max_norm=1.0)
    other_norm, other_bad, other_names = _clip_grad_group(other_named_params, max_norm=5.0)
    return {
        "graph_clip_norm": graph_norm,
        "personal_clip_norm": personal_norm,
        "other_clip_norm": other_norm,
        "nonfinite_grad_count": float(graph_bad + personal_bad + other_bad),
        "nonfinite_graph_grad_count": float(graph_bad),
        "nonfinite_personal_grad_count": float(personal_bad),
        "nonfinite_other_grad_count": float(other_bad),
        "nonfinite_grad_param_names": ",".join(graph_names + personal_names + other_names),
    }


def _build_optimizer(model: nn.Module, args) -> optim.Optimizer:
    base_model = _get_base_model(model)
    structure_module = getattr(base_model, "structure_module", None)
    adaptive_gate = getattr(structure_module, "adaptive_gate", None) if structure_module is not None else None
    personal_generator = getattr(structure_module, "personal_generator", None) if structure_module is not None else None

    state_params: List[torch.nn.Parameter] = []

    if structure_module is not None:
        if adaptive_gate is not None:
            state_params.extend(list(adaptive_gate.parameters()))

        if personal_generator is not None:
            state_params.extend(list(personal_generator.parameters()))

    state_params = _dedupe_params(state_params)
    ae_params: List[torch.nn.Parameter] = []
    for module_name in (
        "ae_student_logit_bias",
        "ae_exercise_logit_bias",
        "ae_concept_logit_bias",
    ):
        module = getattr(base_model, module_name, None)
        if module is not None:
            ae_params.extend(list(module.parameters()))
    ae_logit_feature_weight = getattr(base_model, "ae_logit_feature_weight", None)
    if isinstance(ae_logit_feature_weight, torch.nn.Parameter):
        ae_params.append(ae_logit_feature_weight)
    ae_reliability_feature_weight = getattr(base_model, "ae_reliability_feature_weight", None)
    if isinstance(ae_reliability_feature_weight, torch.nn.Parameter):
        ae_params.append(ae_reliability_feature_weight)
    ae_params = _dedupe_params(ae_params)
    grouped_ids = {id(p) for p in state_params}
    grouped_ids.update(id(p) for p in ae_params)
    backbone_params = _dedupe_params([p for p in base_model.parameters() if id(p) not in grouped_ids])

    param_groups: List[Dict[str, Any]] = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": float(args.learning_rate), "group_name": "backbone"})
    if ae_params:
        param_groups.append(
            {
                "params": ae_params,
                "lr": float(args.learning_rate) * float(getattr(args, "ae_lr_mult", 1.0)),
                "weight_decay": 0.0,
                "group_name": "ae_joint",
            }
        )
    if state_params:
        param_groups.append(
            {
                "params": state_params,
                "lr": float(args.learning_rate) * float(getattr(args, "personal_state_lr_mult", 1.0)),
                "group_name": "personal_state",
            }
        )
    return optim.Adam(param_groups, weight_decay=args.weight_decay)


def _safe_logit(rate: torch.Tensor) -> torch.Tensor:
    rate = rate.clamp(min=1e-4, max=1.0 - 1e-4)
    return torch.log(rate / (1.0 - rate))


def _standardized_log_count_feature(counts: torch.Tensor) -> torch.Tensor:
    """Train-only reliability feature from interaction counts.

    The value is a standardized log-count, so it encodes evidence reliability
    without using validation/test outcomes or adding a black-box branch.
    """
    values = torch.log1p(counts.detach().float())
    if values.numel() <= 1:
        return torch.zeros_like(values)
    centered = values - values.mean()
    scaled = centered / values.std(unbiased=False).clamp(min=1e-6)
    return torch.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=-5.0, max=5.0)


def _compute_recent_student_concept_logits(
    train_loader: DataLoader,
    q_matrix: torch.Tensor,
    concept_rate: torch.Tensor,
    *,
    num_students: int,
    num_concepts: int,
    decay: float = 0.85,
    smoothing: float = 3.0,
) -> torch.Tensor:
    """Train-only recent student-concept mastery evidence.

    This is an interpretable E prior: for each student/concept it keeps an
    exponentially decayed correctness average over that concept's train rows in
    dataset order. It never reads validation/test labels and falls back to the
    train-only concept rate when the student has no concept history.
    """
    dataset = getattr(train_loader, "dataset", None)
    student_ids = getattr(dataset, "student_ids", None)
    exercise_ids = getattr(dataset, "exercise_ids", None)
    labels = getattr(dataset, "labels", None)
    if student_ids is None or exercise_ids is None or labels is None:
        return torch.zeros(num_students, num_concepts, dtype=torch.float32)

    q_cpu = q_matrix.detach().float().cpu()
    recent_count = torch.zeros(num_students, num_concepts, dtype=torch.float32)
    recent_correct = torch.zeros(num_students, num_concepts, dtype=torch.float32)
    decay_value = max(0.0, min(0.999, float(decay)))
    for sid_t, eid_t, label_t in zip(student_ids.long().cpu(), exercise_ids.long().cpu(), labels.float().cpu()):
        sid = int(sid_t.item())
        eid = int(eid_t.item())
        if sid < 0 or sid >= num_students or eid < 0 or eid >= q_cpu.size(0):
            continue
        active = torch.nonzero(q_cpu[eid] > 0, as_tuple=False).reshape(-1)
        if active.numel() == 0:
            continue
        y = float(label_t.item())
        recent_count[sid, active] = recent_count[sid, active] * decay_value + 1.0
        recent_correct[sid, active] = recent_correct[sid, active] * decay_value + y

    concept_rate = concept_rate.detach().float().cpu().clamp(min=1e-4, max=1.0 - 1e-4)
    smooth = max(0.0, float(smoothing))
    recent_rate = (
        recent_correct + smooth * concept_rate.view(1, -1)
    ) / (recent_count + smooth).clamp(min=1e-6)
    logits = (_safe_logit(recent_rate) - _safe_logit(concept_rate).view(1, -1)).clamp(min=-2.0, max=2.0)
    return torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)


def _compute_train_stat_logits(
    train_loader: DataLoader,
    q_matrix: torch.Tensor,
    *,
    num_students: int,
    num_exercises: int,
    num_concepts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, Dict[str, torch.Tensor]]:
    q_cpu = q_matrix.detach().float().cpu()
    student_count = torch.zeros(num_students, dtype=torch.float32)
    student_correct = torch.zeros(num_students, dtype=torch.float32)
    exercise_count = torch.zeros(num_exercises, dtype=torch.float32)
    exercise_correct = torch.zeros(num_exercises, dtype=torch.float32)
    concept_count = torch.zeros(num_concepts, dtype=torch.float32)
    concept_correct = torch.zeros(num_concepts, dtype=torch.float32)
    student_concept_count = torch.zeros(num_students, num_concepts, dtype=torch.float32)
    student_concept_correct = torch.zeros(num_students, num_concepts, dtype=torch.float32)
    total_count = 0
    total_correct = 0.0

    for student_ids, exercise_ids, labels in train_loader:
        student_ids = student_ids.detach().long().cpu()
        exercise_ids = exercise_ids.detach().long().cpu()
        y = labels.detach().float().cpu().reshape(-1)
        if y.numel() == 0:
            continue
        total_count += int(y.numel())
        total_correct += float(y.sum().item())
        ones = torch.ones_like(y)
        student_count.index_add_(0, student_ids, ones)
        student_correct.index_add_(0, student_ids, y)
        exercise_count.index_add_(0, exercise_ids, ones)
        exercise_correct.index_add_(0, exercise_ids, y)

        q = q_cpu[exercise_ids]
        concept_count += q.sum(dim=0)
        concept_correct += (q * y.unsqueeze(-1)).sum(dim=0)
        student_concept_count.index_add_(0, student_ids, q)
        student_concept_correct.index_add_(0, student_ids, q * y.unsqueeze(-1))

    if total_count <= 0:
        return (
            torch.zeros(num_students, dtype=torch.float32),
            torch.zeros(num_exercises, dtype=torch.float32),
            torch.zeros(num_concepts, dtype=torch.float32),
            torch.zeros(num_students, num_concepts, dtype=torch.float32),
            torch.zeros(num_students, num_concepts, dtype=torch.float32),
            0.5,
            {
                "student": torch.zeros(num_students, dtype=torch.float32),
                "exercise": torch.zeros(num_exercises, dtype=torch.float32),
                "concept": torch.zeros(num_concepts, dtype=torch.float32),
                "student_concept": torch.zeros(num_students, num_concepts, dtype=torch.float32),
            },
        )

    global_rate = min(max(total_correct / float(total_count), 1e-4), 1.0 - 1e-4)
    global_rate_t = torch.tensor(global_rate, dtype=torch.float32)
    global_logit = _safe_logit(global_rate_t)

    student_smoothing = 8.0
    exercise_smoothing = 5.0
    concept_smoothing = 20.0
    student_rate = (
        student_correct + student_smoothing * global_rate_t
    ) / (student_count + student_smoothing).clamp(min=1.0)
    exercise_rate = (
        exercise_correct + exercise_smoothing * global_rate_t
    ) / (exercise_count + exercise_smoothing).clamp(min=1.0)
    concept_rate = (
        concept_correct + concept_smoothing * global_rate_t
    ) / (concept_count + concept_smoothing).clamp(min=1.0)
    student_concept_smoothing = 8.0
    student_concept_rate = (
        student_concept_correct + student_concept_smoothing * concept_rate.view(1, -1)
    ) / (student_concept_count + student_concept_smoothing).clamp(min=1.0)

    student_logits = (_safe_logit(student_rate) - global_logit).clamp(min=-3.0, max=3.0)
    exercise_logits = (_safe_logit(exercise_rate) - global_logit).clamp(min=-3.0, max=3.0)
    concept_logits = (_safe_logit(concept_rate) - global_logit).clamp(min=-3.0, max=3.0)
    student_concept_logits = (
        _safe_logit(student_concept_rate) - _safe_logit(concept_rate).view(1, -1)
    ).clamp(min=-2.0, max=2.0)
    student_logits = torch.where(torch.isfinite(student_logits), student_logits, torch.zeros_like(student_logits))
    exercise_logits = torch.where(torch.isfinite(exercise_logits), exercise_logits, torch.zeros_like(exercise_logits))
    concept_logits = torch.where(torch.isfinite(concept_logits), concept_logits, torch.zeros_like(concept_logits))
    student_concept_logits = torch.where(
        torch.isfinite(student_concept_logits),
        student_concept_logits,
        torch.zeros_like(student_concept_logits),
    )
    count_features = {
        "student": _standardized_log_count_feature(student_count),
        "exercise": _standardized_log_count_feature(exercise_count),
        "concept": _standardized_log_count_feature(concept_count),
        "student_concept": _standardized_log_count_feature(student_concept_count),
        "student_concept_observed": student_concept_count.detach().clone(),
    }
    student_concept_recent_logits = _compute_recent_student_concept_logits(
        train_loader,
        q_cpu,
        concept_rate,
        num_students=num_students,
        num_concepts=num_concepts,
    )
    return (
        student_logits,
        exercise_logits,
        concept_logits,
        student_concept_logits,
        student_concept_recent_logits,
        float(global_rate),
        count_features,
    )


def _initialize_ae_stat_priors(
    model: nn.Module,
    train_loader: DataLoader,
    info_dict: Dict[str, Any],
    args,
    logger,
    run_tag: str,
) -> None:
    scale = max(0.0, float(getattr(args, "ae_stat_prior_scale", 0.0)))
    if scale <= 0.0:
        return
    base_model = _get_base_model(model)
    if (
        getattr(base_model, "ae_student_prior_logit", None) is None
        or getattr(base_model, "ae_exercise_prior_logit", None) is None
        or getattr(base_model, "ae_concept_prior_logit", None) is None
        or getattr(base_model, "ae_student_concept_prior_logit", None) is None
        or not (
            getattr(base_model, "use_concept_graph", False)
            or getattr(base_model, "use_personal_graph", False)
        )
    ):
        return

    (
        student_logits,
        exercise_logits,
        concept_logits,
        student_concept_logits,
        student_concept_recent_logits,
        global_rate,
        count_features,
    ) = _compute_train_stat_logits(
        train_loader,
        info_dict["q_matrix"],
        num_students=info_dict["num_students"],
        num_exercises=info_dict["num_exercises"],
        num_concepts=info_dict["num_concepts"],
    )
    base_model.initialize_ae_logit_priors(
        student_logits=student_logits,
        exercise_logits=exercise_logits,
        concept_logits=concept_logits,
        student_concept_logits=student_concept_logits,
        student_concept_recent_logits=student_concept_recent_logits,
        scale=scale,
        student_count_features=count_features["student"],
        exercise_count_features=count_features["exercise"],
        concept_count_features=count_features["concept"],
        student_concept_count_features=count_features["student_concept"],
        student_concept_observed_counts=count_features["student_concept_observed"],
    )
    logger.info(
        "%s AE stat priors initialized: scale=%.3f, train_global_rate=%.4f, "
        "student_absmean=%.4f, exercise_absmean=%.4f, concept_absmean=%.4f, "
        "student_concept_absmean=%.4f, recent_student_concept_absmean=%.4f, reliability_absmean=%.4f",
        run_tag,
        scale,
        global_rate,
        float(student_logits.abs().mean().item()),
        float(exercise_logits.abs().mean().item()),
        float(concept_logits.abs().mean().item()),
        float(student_concept_logits.abs().mean().item()),
        float(student_concept_recent_logits.abs().mean().item()),
        float(
            torch.stack(
                [
                    count_features["student"].abs().mean(),
                    count_features["exercise"].abs().mean(),
                    count_features["concept"].abs().mean(),
                    count_features["student_concept"].abs().mean(),
                ]
            ).mean().item()
        ),
    )


def _get_base_model(model: nn.Module) -> CognitiveDiagnosisModel:
    """获取基础模型（处理 DataParallel 包装）"""
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def _collect_runtime_ablation_facts(model: nn.Module) -> Dict[str, Any]:
    """收集模型运行时的模块1开关与物理存在性，用于防止“假消融”"""
    base_model = _get_base_model(model)
    structure_module = getattr(base_model, "structure_module", None)
    relation_learning = getattr(structure_module, "relation_learning", None) if structure_module is not None else None
    adaptive_gate = getattr(structure_module, "adaptive_gate", None) if structure_module is not None else None
    personal_generator = getattr(structure_module, "personal_generator", None) if structure_module is not None else None

    has_knowledge_encoder = (
        structure_module is not None and getattr(structure_module, "knowledge_encoder", None) is not None
    )

    return {
        "enable_module1": bool(getattr(base_model, "enable_module1", False)),
        "use_concept_graph": bool(getattr(base_model, "use_concept_graph", False)),
        "use_personal_graph": bool(getattr(base_model, "use_personal_graph", False)),
        "has_knowledge_encoder": bool(has_knowledge_encoder),
        "has_relation_learning": bool(relation_learning is not None),
        "has_adaptive_gate": bool(adaptive_gate is not None),
        "has_personal_generator": bool(personal_generator is not None),
        **_collect_structural_switches(base_model),
    }


def _log_and_assert_ablation_consistency(
    *,
    model: nn.Module,
    logger,
    context: str,
    ablate_module1: bool,
    expect_use_concept_graph: Optional[bool] = None,
    expect_use_personal_graph: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    记录并校验消融一致性：
    - args.ablate_module* 与 model.enable_module* 必须一致
    - 关键模块的“物理存在性”必须符合预期
    """
    facts = _collect_runtime_ablation_facts(model)

    logger.info(
        "%s Ablation runtime check: "
        "args(ablate_module1=%s) | "
        "expect(use_concept_graph=%s,use_personal_graph=%s) | "
        "model(enable_module1=%s,use_concept_graph=%s,use_personal_graph=%s) | "
        "physical(has_knowledge_encoder=%s,has_relation_learning=%s,has_adaptive_gate=%s,has_personal_generator=%s)",
        context,
        ablate_module1,
        expect_use_concept_graph,
        expect_use_personal_graph,
        facts["enable_module1"],
        facts["use_concept_graph"],
        facts["use_personal_graph"],
        facts["has_knowledge_encoder"],
        facts["has_relation_learning"],
        facts["has_adaptive_gate"],
        facts["has_personal_generator"],
    )

    if ablate_module1 and facts["enable_module1"]:
        raise RuntimeError("Ablation mismatch: ablate_module1=True but model.enable_module1=True.")
    if ablate_module1 and facts["has_knowledge_encoder"]:
        raise RuntimeError("Ablation mismatch: ablate_module1=True but knowledge_encoder still exists.")
    if ablate_module1 and (
        facts["use_concept_graph"]
        or facts["use_personal_graph"]
        or facts["has_relation_learning"]
        or facts["has_adaptive_gate"]
        or facts["has_personal_generator"]
    ):
        raise RuntimeError("Ablation mismatch: ablate_module1=True but Module1 submodules still remain active.")

    if (not ablate_module1) and (not facts["enable_module1"] or not facts["has_knowledge_encoder"]):
        raise RuntimeError("Ablation mismatch: ablate_module1=False but Module1 is not fully available.")

    if expect_use_concept_graph is not None and facts["use_concept_graph"] != bool(expect_use_concept_graph):
        raise RuntimeError(
            f"Ablation mismatch: expected use_concept_graph={bool(expect_use_concept_graph)} "
            f"but got {facts['use_concept_graph']}."
        )
    if expect_use_concept_graph is not None and bool(expect_use_concept_graph) and not facts["has_relation_learning"]:
        raise RuntimeError("Ablation mismatch: use_concept_graph=True but relation_learning is missing.")
    if expect_use_concept_graph is not None and (not bool(expect_use_concept_graph)) and facts["has_relation_learning"]:
        raise RuntimeError("Ablation mismatch: use_concept_graph=False but relation_learning still exists.")

    if expect_use_personal_graph is not None and facts["use_personal_graph"] != bool(expect_use_personal_graph):
        raise RuntimeError(
            f"Ablation mismatch: expected use_personal_graph={bool(expect_use_personal_graph)} "
            f"but got {facts['use_personal_graph']}."
        )
    if expect_use_personal_graph is not None and bool(expect_use_personal_graph):
        if not facts["has_adaptive_gate"] or not facts["has_personal_generator"]:
            raise RuntimeError("Ablation mismatch: use_personal_graph=True but E submodules are missing.")
    if expect_use_personal_graph is not None and (not bool(expect_use_personal_graph)):
        if facts["has_adaptive_gate"] or facts["has_personal_generator"]:
            raise RuntimeError("Ablation mismatch: use_personal_graph=False but E submodules still exist.")

    return facts


def _log_graph_init_state(model: nn.Module, logger, context: str) -> None:
    """记录 A/E 图结构模块的初始化状态。"""
    base_model = _get_base_model(model)
    relation_learning = getattr(getattr(base_model, "structure_module", None), "relation_learning", None)

    graph_tau_mean = 0.0
    graph_tau_std = 0.0
    graph_dropout = 0.0
    relation_temperature_raw = (
        getattr(relation_learning, "temperature_raw", getattr(relation_learning, "tau_raw", None))
        if relation_learning is not None
        else None
    )
    if relation_temperature_raw is not None:
        tau = F.softplus(relation_temperature_raw.detach()) + 1e-6
        graph_tau_mean = float(tau.mean().item())
        graph_tau_std = float(tau.std(unbiased=False).item())
        graph_dropout = float(getattr(getattr(relation_learning, "dropout", None), "p", 0.0))
    item_beta = 0.0
    seq_beta = 0.0
    item_beta_std = 0.0
    seq_beta_std = 0.0
    self_beta = 0.0
    if relation_learning is not None:
        item_strength = getattr(relation_learning, "prior_strength_raw", None)
        seq_strength = getattr(relation_learning, "sequence_prior_strength_raw", None)
        self_loop = getattr(relation_learning, "self_loop_logit", None)
        if item_strength is not None:
            item_beta_t = F.softplus(item_strength.detach())
            item_beta = float(item_beta_t.mean().item())
            item_beta_std = float(item_beta_t.std(unbiased=False).item())
        if seq_strength is not None:
            seq_beta_t = F.softplus(seq_strength.detach())
            seq_beta = float(seq_beta_t.mean().item())
            seq_beta_std = float(seq_beta_t.std(unbiased=False).item())
        if self_loop is not None:
            self_beta = float(self_loop.detach().mean().item())

    logger.info(
        "%s [Graph Init] graph_tau_mean=%.4f, graph_tau_std=%.4f, graph_dropout=%.3f, "
        "item_beta=%.4f±%.4f, seq_beta=%.4f±%.4f, self_loop_logit=%.4f, use_personal_graph=%s",
        context,
        graph_tau_mean,
        graph_tau_std,
        graph_dropout,
        item_beta,
        item_beta_std,
        seq_beta,
        seq_beta_std,
        self_beta,
        bool(getattr(base_model, "use_personal_graph", False)),
    )


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    移除 DataParallel 保存的 state_dict 中的 'module.' 前缀。
    
    DataParallel 训练时保存的权重格式为 'module.xxx'，
    但在非 DataParallel 模型中加载时需要移除这个前缀。
    """
    has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
    if not has_module_prefix:
        return state_dict
    
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_sd[k[7:]] = v  # 移除 "module." 前缀（7个字符）
        else:
            new_sd[k] = v
    return new_sd


def _hard_ablation_effective_hparams(
    *,
    use_concept_graph: bool,
    num_gnn_layers: int,
 ) -> int:
    """
    Effective GNN depth for runtime construction.

    Note:
      - `no_A` / `use_concept_graph=False` 只应关闭全局图学习 A；
      - knowledge_encoder 与 E 仍然需要保留既定的 GNN 深度，通过 identity/global prior 继续传播；
      - 只有 `ablate_module1=True` 时，调用方才会在更高层把整条模块1物理关闭并把层数置 0。
    """
    _ = bool(use_concept_graph)
    return max(0, int(num_gnn_layers))


def _safe_mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def _safe_abs_mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.abs(np.asarray(values, dtype=np.float64))
    return float(arr.mean()), float(arr.std())


def _row_entropy_mean(A: np.ndarray) -> float:
    eps = 1e-12
    A = np.clip(A, eps, None)
    row_entropy = -(A * np.log(A)).sum(axis=-1)
    return float(row_entropy.mean())


_REG_COMPONENT_KEYS: Tuple[str, ...] = (
    "graph_entropy",
    "graph_diag",
    "graph_uniform",
    "graph_reg_scale",
    "prediction_l2",
    "personal_sparse",
    "personal_kl",
    "personal_query_residual",
    "theta_prior_align",
    "alpha_var",
    "alpha_collapse",
)


def _collect_debug_forward_stats(
    model: CognitiveDiagnosisModel,
    data_loader: DataLoader,
    device: torch.device,
    max_batches: int = 2,
) -> Dict[str, float]:
    """
    仅用于诊断日志：采样少量 batch 统计 A/E 关键信号。
    不参与训练，也不改变任何训练逻辑。
    """
    max_batches = max(1, int(max_batches))
    was_training = model.training
    model.eval()

    irt_vals: List[float] = []
    alpha_vals: List[float] = []
    alpha_bias_vals: List[float] = []
    alpha_base_vals: List[float] = []
    alpha_delta_vals: List[float] = []
    alpha_saturation_vals: List[float] = []
    personal_entropy_vals: List[float] = []
    personal_matrix_delta_vals: List[float] = []
    personal_matrix_student_std_vals: List[float] = []
    personal_delta_pre_softmax_norm_vals: List[float] = []
    personal_delta_student_std_vals: List[float] = []
    alpha_head_std_vals: List[float] = []
    alpha_state_path_vals: List[float] = []
    alpha_id_path_vals: List[float] = []
    alpha_bias_path_vals: List[float] = []
    head_bias_path_vals: List[float] = []
    relation_identity_delta_vals: List[float] = []
    knowledge_state_graph_delta_vals: List[float] = []
    knowledge_state_personal_delta_vals: List[float] = []
    personal_bad_row_vals: List[float] = []
    personal_fallback_row_vals: List[float] = []
    personal_bad_row_active_vals: List[float] = []
    personal_fallback_row_active_vals: List[float] = []
    personal_bad_row_rate_active_vals: List[float] = []
    personal_padded_row_vals: List[float] = []
    personal_student_mix_vals: List[float] = []
    personal_mastery_prior_scale_vals: List[float] = []
    personal_mastery_scores_absmean_vals: List[float] = []
    personal_mastery_reliability_vals: List[float] = []
    personal_recent_mastery_prior_scale_vals: List[float] = []
    personal_recent_mastery_scores_absmean_vals: List[float] = []
    personal_recent_mastery_reliability_vals: List[float] = []
    personal_student_adapter_vals: List[float] = []
    personal_logits_absmax_vals: List[float] = []
    personal_logits_support_absmax_vals: List[float] = []
    local_row_ratio_vals: List[float] = []
    personal_support_density_vals: List[float] = []
    readout_query_delta_vals: List[float] = []
    query_row_graph_delta_vals: List[float] = []
    query_row_global_readout_delta_vals: List[float] = []
    query_row_personal_message_delta_vals: List[float] = []
    ae_query_state_residual_delta_vals: List[float] = []
    ae_logit_residual_abs_mean_vals: List[float] = []
    query_row_posterior_delta_abs_vals: List[float] = []
    query_row_posterior_kl_vals: List[float] = []
    personal_to_graph_query_ratio_vals: List[float] = []
    personal_query_trust_scale_mean_vals: List[float] = []
    query_row_global_local_rms_vals: List[float] = []
    query_row_post_local_rms_vals: List[float] = []
    query_row_delta_local_rms_raw_vals: List[float] = []
    query_row_message_projection_gain_vals: List[float] = []
    query_row_message_alignment_vals: List[float] = []
    query_row_self_support_mass_vals: List[float] = []
    query_row_graph_support_mass_vals: List[float] = []
    query_row_global_head_var_vals: List[float] = []
    query_row_global_head_margin_vals: List[float] = []
    personal_query_writeback_delta_vals: List[float] = []
    graph_query_adapter_gain_vals: List[float] = []
    neighbor_row_personal_delta_vals: List[float] = []
    personal_row_budget_mean_vals: List[float] = []
    personal_query_row_std_vals: List[float] = []
    personal_item_support_added_rate_vals: List[float] = []
    personal_item_support_added_mass_vals: List[float] = []
    readout_query_support_mass_vals: List[float] = []
    query_row_global_support_mass_vals: List[float] = []
    query_row_personal_support_mass_vals: List[float] = []
    support_candidate_size_vals: List[float] = []
    support_final_size_vals: List[float] = []
    support_item_survival_vals: List[float] = []
    support_seq_survival_vals: List[float] = []
    support_item_seq_overlap_vals: List[float] = []
    support_self_retention_vals: List[float] = []
    relation_theta_delta_abs_vals: List[float] = []
    relation_theta_logit_abs_vals: List[float] = []
    relation_theta_neighbor_mass_vals: List[float] = []
    relation_theta_personal_mass_vals: List[float] = []
    ae_posterior_prior_logit_abs_vals: List[float] = []
    ae_posterior_prior_delta_abs_vals: List[float] = []
    ae_stat_student_prior_abs_vals: List[float] = []
    ae_stat_exercise_prior_abs_vals: List[float] = []
    ae_stat_concept_prior_abs_vals: List[float] = []
    ae_stat_query_sc_prior_abs_vals: List[float] = []
    ae_stat_graph_sc_prior_abs_vals: List[float] = []
    ae_reliability_feature_logit_abs_vals: List[float] = []
    ae_interpretable_bias_abs_vals: List[float] = []
    ae_explicit_feature_logit_abs_vals: List[float] = []
    ae_interaction_logit_abs_vals: List[float] = []
    ae_raw_logit_before_clip_abs_vals: List[float] = []
    ae_raw_logit_after_clip_abs_vals: List[float] = []

    graph_row_entropy_mean = 0.0
    graph_entropy_ratio = 0.0
    graph_diag_mass = 0.0
    graph_to_uniform_l2 = 0.0
    graph_to_identity_l2 = 0.0
    graph_ready = False
    personal_warmup_scale = 1.0

    base_model = _get_base_model(model)
    tau_mean = 0.0
    tau_std = 0.0
    share_concept_embeddings = float(getattr(base_model, "share_concept_embeddings", False))
    relation_learning = getattr(getattr(base_model, "structure_module", None), "relation_learning", None)
    relation_temperature_raw = (
        getattr(relation_learning, "temperature_raw", getattr(relation_learning, "tau_raw", None))
        if relation_learning is not None
        else None
    )
    if relation_temperature_raw is not None:
        tau = F.softplus(relation_temperature_raw.detach()) + 1e-6
        tau_mean = float(tau.mean().item())
        tau_std = float(tau.std(unbiased=False).item())

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= max_batches:
                break

            student_ids, exercise_ids, _ = batch
            student_ids = student_ids.to(device)
            exercise_ids = exercise_ids.to(device)

            _, details = model(
                student_ids,
                exercise_ids,
                return_details=True,
                return_logits=True,
            )

            irt_logit = details.get("irt_logit")
            if irt_logit is not None:
                irt_vals.extend(irt_logit.detach().reshape(-1).cpu().numpy().tolist())

            alpha = details.get("alpha_effective", details.get("alpha"))
            if alpha is not None:
                alpha_vals.extend(alpha.detach().reshape(-1).cpu().numpy().tolist())

            alpha_student_bias = details.get("alpha_student_bias")
            if alpha_student_bias is not None:
                alpha_bias_vals.extend(alpha_student_bias.detach().reshape(-1).cpu().numpy().tolist())

            for detail_key, target in (
                ("alpha_base_mean", alpha_base_vals),
                ("alpha_delta_absmean", alpha_delta_vals),
                ("alpha_saturation_ratio", alpha_saturation_vals),
            ):
                val = details.get(detail_key)
                if val is not None:
                    target.extend(val.detach().reshape(-1).cpu().numpy().tolist())

            posterior_prob = details.get("posterior_prob")
            support_valid_mask = details.get("support_valid_mask")
            personal_matrices = details.get("personal_matrices")
            if posterior_prob is not None and support_valid_mask is not None:
                probs = posterior_prob.detach()
                valid = support_valid_mask.detach().bool()
                masked = probs.clamp(min=1e-12) * valid.to(dtype=probs.dtype)
                row_entropy = -(masked * masked.clamp(min=1e-12).log()).sum(dim=-1)
                row_valid = valid.any(dim=-1)
                if bool(row_valid.any()):
                    personal_entropy_vals.append(float(row_entropy[row_valid].mean().item()))
            elif personal_matrices is not None:
                pm = personal_matrices.detach().cpu().numpy()
                personal_entropy_vals.append(_row_entropy_mean(pm))

            pm_delta = details.get("personal_matrix_delta")
            if pm_delta is not None:
                personal_matrix_delta_vals.extend(pm_delta.detach().reshape(-1).cpu().numpy().tolist())
            student_std = details.get("personal_matrix_student_std")
            if student_std is not None:
                personal_matrix_student_std_vals.append(float(student_std.detach().item()))

            for detail_key, target in (
                ("personal_delta_pre_softmax_norm", personal_delta_pre_softmax_norm_vals),
                ("personal_delta_student_std", personal_delta_student_std_vals),
                ("alpha_head_std", alpha_head_std_vals),
                ("alpha_state_path_absmean", alpha_state_path_vals),
                ("alpha_id_path_absmean", alpha_id_path_vals),
                ("alpha_bias_path_absmean", alpha_bias_path_vals),
                ("head_bias_path_absmean", head_bias_path_vals),
                ("personal_bad_row_count", personal_bad_row_vals),
                ("personal_fallback_row_count", personal_fallback_row_vals),
                ("personal_bad_row_count_active", personal_bad_row_active_vals),
                ("personal_fallback_row_count_active", personal_fallback_row_active_vals),
                ("personal_bad_row_rate_active", personal_bad_row_rate_active_vals),
                ("personal_padded_row_count", personal_padded_row_vals),
                ("personal_student_mix", personal_student_mix_vals),
                ("personal_mastery_prior_scale", personal_mastery_prior_scale_vals),
                ("personal_mastery_scores_absmean", personal_mastery_scores_absmean_vals),
                ("personal_mastery_reliability_mean", personal_mastery_reliability_vals),
                ("personal_recent_mastery_prior_scale", personal_recent_mastery_prior_scale_vals),
                ("personal_recent_mastery_scores_absmean", personal_recent_mastery_scores_absmean_vals),
                ("personal_recent_mastery_reliability_mean", personal_recent_mastery_reliability_vals),
                ("personal_student_adapter_scale", personal_student_adapter_vals),
                ("personal_logits_absmax", personal_logits_absmax_vals),
                ("personal_logits_support_absmax", personal_logits_support_absmax_vals),
                ("local_row_ratio", local_row_ratio_vals),
                ("personal_support_density", personal_support_density_vals),
                ("query_row_personal_message_delta", query_row_personal_message_delta_vals),
                ("ae_query_state_residual_delta", ae_query_state_residual_delta_vals),
                ("ae_logit_residual_abs_mean", ae_logit_residual_abs_mean_vals),
                ("ae_posterior_prior_logit_abs_mean", ae_posterior_prior_logit_abs_vals),
                ("ae_posterior_prior_delta_abs_mean", ae_posterior_prior_delta_abs_vals),
                ("ae_stat_student_prior_abs_mean", ae_stat_student_prior_abs_vals),
                ("ae_stat_exercise_prior_abs_mean", ae_stat_exercise_prior_abs_vals),
                ("ae_stat_concept_prior_abs_mean", ae_stat_concept_prior_abs_vals),
                ("ae_stat_query_student_concept_prior_abs_mean", ae_stat_query_sc_prior_abs_vals),
                ("ae_stat_graph_student_concept_prior_abs_mean", ae_stat_graph_sc_prior_abs_vals),
                ("ae_reliability_feature_logit_abs_mean", ae_reliability_feature_logit_abs_vals),
                ("ae_interpretable_bias_abs_mean", ae_interpretable_bias_abs_vals),
                ("ae_explicit_feature_logit_abs_mean", ae_explicit_feature_logit_abs_vals),
                ("ae_interaction_logit_abs_mean", ae_interaction_logit_abs_vals),
                ("ae_raw_logit_before_clip_abs_mean", ae_raw_logit_before_clip_abs_vals),
                ("ae_raw_logit_after_clip_abs_mean", ae_raw_logit_after_clip_abs_vals),
                ("query_row_posterior_delta_abs", query_row_posterior_delta_abs_vals),
                ("query_row_posterior_kl", query_row_posterior_kl_vals),
                ("personal_to_graph_query_ratio_effective", personal_to_graph_query_ratio_vals),
                ("personal_query_trust_scale_mean", personal_query_trust_scale_mean_vals),
                ("query_row_global_local_rms", query_row_global_local_rms_vals),
                ("query_row_post_local_rms", query_row_post_local_rms_vals),
                ("query_row_delta_local_rms_raw", query_row_delta_local_rms_raw_vals),
                ("query_row_message_projection_gain", query_row_message_projection_gain_vals),
                ("query_row_message_alignment", query_row_message_alignment_vals),
                ("query_row_self_support_mass", query_row_self_support_mass_vals),
                ("query_row_graph_support_mass", query_row_graph_support_mass_vals),
                ("query_row_global_head_var", query_row_global_head_var_vals),
                ("query_row_global_head_margin", query_row_global_head_margin_vals),
                ("personal_query_writeback_delta", personal_query_writeback_delta_vals),
                ("graph_query_adapter_gain", graph_query_adapter_gain_vals),
                ("neighbor_row_personal_delta", neighbor_row_personal_delta_vals),
                ("personal_row_budget_mean", personal_row_budget_mean_vals),
                ("personal_query_row_std", personal_query_row_std_vals),
                ("personal_item_support_added_rate", personal_item_support_added_rate_vals),
                ("personal_item_support_added_mass", personal_item_support_added_mass_vals),
            ):
                val = details.get(detail_key)
                if val is not None:
                    target.extend(val.detach().reshape(-1).cpu().numpy().tolist())

            for detail_key, target in (
                ("relation_identity_delta", relation_identity_delta_vals),
                ("knowledge_state_graph_delta", knowledge_state_graph_delta_vals),
                ("knowledge_state_personal_delta", knowledge_state_personal_delta_vals),
                ("readout_query_delta", readout_query_delta_vals),
                ("query_row_graph_delta", query_row_graph_delta_vals),
                ("query_row_global_readout_delta", query_row_global_readout_delta_vals),
                ("readout_query_support_mass", readout_query_support_mass_vals),
                ("query_row_global_support_mass", query_row_global_support_mass_vals),
                ("query_row_personal_support_mass", query_row_personal_support_mass_vals),
                ("support_candidate_size_mean", support_candidate_size_vals),
                ("support_final_size_mean", support_final_size_vals),
                ("support_item_survival_rate", support_item_survival_vals),
                ("support_seq_survival_rate", support_seq_survival_vals),
                ("support_item_seq_overlap", support_item_seq_overlap_vals),
                ("support_self_retention_rate", support_self_retention_vals),
                ("relation_theta_delta_abs_mean", relation_theta_delta_abs_vals),
                ("relation_theta_logit_abs_mean", relation_theta_logit_abs_vals),
                ("relation_theta_neighbor_mass", relation_theta_neighbor_mass_vals),
                ("relation_theta_personal_mass", relation_theta_personal_mass_vals),
            ):
                val = details.get(detail_key)
                if val is not None:
                    target.extend(val.detach().reshape(-1).cpu().numpy().tolist())

            if details.get("personal_warmup_scale") is not None:
                personal_warmup_scale = float(details["personal_warmup_scale"].detach().reshape(-1)[0].item())

            if not graph_ready:
                relation_matrices = details.get("relation_matrices")
                if relation_matrices is not None:
                    rm = relation_matrices.detach().cpu().numpy()
                    graph_row_entropy_mean = _row_entropy_mean(rm)
                    max_row_entropy = float(np.log(rm.shape[-1])) if rm.shape[-1] > 1 else 0.0
                    graph_entropy_ratio = (
                        graph_row_entropy_mean / max_row_entropy if max_row_entropy > 0 else 0.0
                    )
                    graph_diag_mass = float(np.diagonal(rm, axis1=-2, axis2=-1).mean())
                    uniform_val = 1.0 / float(max(1, rm.shape[-1]))
                    graph_to_uniform_l2 = float(np.sqrt(np.mean((rm - uniform_val) ** 2)))
                    identity = np.eye(rm.shape[-1], dtype=rm.dtype)
                    graph_to_identity_l2 = float(np.sqrt(np.mean((rm - identity) ** 2)))
                    graph_ready = True

    if was_training:
        model.train()

    irt_abs_mean, irt_std = _safe_abs_mean_std(irt_vals)
    alpha_mean, alpha_std = _safe_mean_std(alpha_vals)
    alpha_base_mean, _ = _safe_mean_std(alpha_base_vals)
    alpha_delta_absmean, _ = _safe_mean_std(alpha_delta_vals)
    alpha_saturation_ratio, _ = _safe_mean_std(alpha_saturation_vals)
    _, alpha_bias_std_raw = _safe_mean_std(alpha_bias_vals)
    alpha_bias_std = alpha_bias_std_raw if alpha_bias_vals else None
    personal_row_entropy, _ = _safe_mean_std(personal_entropy_vals)
    personal_matrix_delta_mean, _ = _safe_mean_std(personal_matrix_delta_vals)
    personal_matrix_student_std_mean, _ = _safe_mean_std(personal_matrix_student_std_vals)
    personal_delta_pre_softmax_norm_mean, _ = _safe_mean_std(personal_delta_pre_softmax_norm_vals)
    personal_delta_student_std_mean, _ = _safe_mean_std(personal_delta_student_std_vals)
    alpha_head_std_mean, _ = _safe_mean_std(alpha_head_std_vals)
    alpha_state_path_mean, _ = _safe_mean_std(alpha_state_path_vals)
    alpha_id_path_mean, _ = _safe_mean_std(alpha_id_path_vals)
    alpha_bias_path_mean, _ = _safe_mean_std(alpha_bias_path_vals)
    head_bias_path_mean, _ = _safe_mean_std(head_bias_path_vals)
    relation_identity_delta_mean, _ = _safe_mean_std(relation_identity_delta_vals)
    knowledge_state_graph_delta_mean, _ = _safe_mean_std(knowledge_state_graph_delta_vals)
    knowledge_state_personal_delta_mean, _ = _safe_mean_std(knowledge_state_personal_delta_vals)
    personal_bad_row_mean, _ = _safe_mean_std(personal_bad_row_vals)
    personal_fallback_row_mean, _ = _safe_mean_std(personal_fallback_row_vals)
    personal_bad_row_active_mean, _ = _safe_mean_std(personal_bad_row_active_vals)
    personal_fallback_row_active_mean, _ = _safe_mean_std(personal_fallback_row_active_vals)
    personal_bad_row_rate_active_mean, _ = _safe_mean_std(personal_bad_row_rate_active_vals)
    personal_padded_row_mean, _ = _safe_mean_std(personal_padded_row_vals)
    personal_student_mix_mean, _ = _safe_mean_std(personal_student_mix_vals)
    personal_mastery_prior_scale_mean, _ = _safe_mean_std(personal_mastery_prior_scale_vals)
    personal_mastery_scores_absmean, _ = _safe_mean_std(personal_mastery_scores_absmean_vals)
    personal_mastery_reliability_mean, _ = _safe_mean_std(personal_mastery_reliability_vals)
    personal_recent_mastery_prior_scale_mean, _ = _safe_mean_std(personal_recent_mastery_prior_scale_vals)
    personal_recent_mastery_scores_absmean, _ = _safe_mean_std(personal_recent_mastery_scores_absmean_vals)
    personal_recent_mastery_reliability_mean, _ = _safe_mean_std(personal_recent_mastery_reliability_vals)
    personal_student_adapter_mean, _ = _safe_mean_std(personal_student_adapter_vals)
    personal_logits_absmax_mean, _ = _safe_mean_std(personal_logits_absmax_vals)
    personal_logits_support_absmax_mean, _ = _safe_mean_std(personal_logits_support_absmax_vals)
    local_row_ratio_mean, _ = _safe_mean_std(local_row_ratio_vals)
    personal_support_density_mean, _ = _safe_mean_std(personal_support_density_vals)
    readout_query_delta_mean, _ = _safe_mean_std(readout_query_delta_vals)
    query_row_graph_delta_mean, _ = _safe_mean_std(query_row_graph_delta_vals)
    query_row_global_readout_delta_mean, _ = _safe_mean_std(query_row_global_readout_delta_vals)
    query_row_personal_message_delta_mean, _ = _safe_mean_std(query_row_personal_message_delta_vals)
    ae_query_state_residual_delta_mean, _ = _safe_mean_std(ae_query_state_residual_delta_vals)
    ae_logit_residual_abs_mean, _ = _safe_mean_std(ae_logit_residual_abs_mean_vals)
    query_row_posterior_delta_abs_mean, _ = _safe_mean_std(query_row_posterior_delta_abs_vals)
    query_row_posterior_kl_mean, _ = _safe_mean_std(query_row_posterior_kl_vals)
    personal_to_graph_query_ratio_mean, _ = _safe_mean_std(personal_to_graph_query_ratio_vals)
    personal_query_trust_scale_mean, _ = _safe_mean_std(personal_query_trust_scale_mean_vals)
    query_row_global_local_rms_mean, _ = _safe_mean_std(query_row_global_local_rms_vals)
    query_row_post_local_rms_mean, _ = _safe_mean_std(query_row_post_local_rms_vals)
    query_row_delta_local_rms_raw_mean, _ = _safe_mean_std(query_row_delta_local_rms_raw_vals)
    query_row_message_projection_gain_mean, _ = _safe_mean_std(query_row_message_projection_gain_vals)
    query_row_message_alignment_mean, _ = _safe_mean_std(query_row_message_alignment_vals)
    query_row_self_support_mass_mean, _ = _safe_mean_std(query_row_self_support_mass_vals)
    query_row_graph_support_mass_mean, _ = _safe_mean_std(query_row_graph_support_mass_vals)
    query_row_global_head_var_mean, _ = _safe_mean_std(query_row_global_head_var_vals)
    query_row_global_head_margin_mean, _ = _safe_mean_std(query_row_global_head_margin_vals)
    personal_query_writeback_delta_mean, _ = _safe_mean_std(personal_query_writeback_delta_vals)
    graph_query_adapter_gain_mean, _ = _safe_mean_std(graph_query_adapter_gain_vals)
    neighbor_row_personal_delta_mean, _ = _safe_mean_std(neighbor_row_personal_delta_vals)
    personal_row_budget_mean, _ = _safe_mean_std(personal_row_budget_mean_vals)
    personal_query_row_std_mean, _ = _safe_mean_std(personal_query_row_std_vals)
    personal_item_support_added_rate_mean, _ = _safe_mean_std(personal_item_support_added_rate_vals)
    personal_item_support_added_mass_mean, _ = _safe_mean_std(personal_item_support_added_mass_vals)
    readout_query_support_mass_mean, _ = _safe_mean_std(readout_query_support_mass_vals)
    query_row_global_support_mass_mean, _ = _safe_mean_std(query_row_global_support_mass_vals)
    query_row_personal_support_mass_mean, _ = _safe_mean_std(query_row_personal_support_mass_vals)
    support_candidate_size_mean, _ = _safe_mean_std(support_candidate_size_vals)
    support_final_size_mean, _ = _safe_mean_std(support_final_size_vals)
    support_item_survival_rate, _ = _safe_mean_std(support_item_survival_vals)
    support_seq_survival_rate, _ = _safe_mean_std(support_seq_survival_vals)
    support_item_seq_overlap, _ = _safe_mean_std(support_item_seq_overlap_vals)
    support_self_retention_rate, _ = _safe_mean_std(support_self_retention_vals)
    relation_theta_delta_abs_mean, _ = _safe_mean_std(relation_theta_delta_abs_vals)
    relation_theta_logit_abs_mean, _ = _safe_mean_std(relation_theta_logit_abs_vals)
    relation_theta_neighbor_mass_mean, _ = _safe_mean_std(relation_theta_neighbor_mass_vals)
    relation_theta_personal_mass_mean, _ = _safe_mean_std(relation_theta_personal_mass_vals)
    ae_posterior_prior_logit_abs_mean, _ = _safe_mean_std(ae_posterior_prior_logit_abs_vals)
    ae_posterior_prior_delta_abs_mean, _ = _safe_mean_std(ae_posterior_prior_delta_abs_vals)
    ae_stat_student_prior_abs_mean, _ = _safe_mean_std(ae_stat_student_prior_abs_vals)
    ae_stat_exercise_prior_abs_mean, _ = _safe_mean_std(ae_stat_exercise_prior_abs_vals)
    ae_stat_concept_prior_abs_mean, _ = _safe_mean_std(ae_stat_concept_prior_abs_vals)
    ae_stat_query_sc_prior_abs_mean, _ = _safe_mean_std(ae_stat_query_sc_prior_abs_vals)
    ae_stat_graph_sc_prior_abs_mean, _ = _safe_mean_std(ae_stat_graph_sc_prior_abs_vals)
    ae_reliability_feature_logit_abs_mean, _ = _safe_mean_std(ae_reliability_feature_logit_abs_vals)
    ae_interpretable_bias_abs_mean, _ = _safe_mean_std(ae_interpretable_bias_abs_vals)
    ae_explicit_feature_logit_abs_mean, _ = _safe_mean_std(ae_explicit_feature_logit_abs_vals)
    ae_interaction_logit_abs_mean, _ = _safe_mean_std(ae_interaction_logit_abs_vals)
    ae_raw_logit_before_clip_abs_mean, _ = _safe_mean_std(ae_raw_logit_before_clip_abs_vals)
    ae_raw_logit_after_clip_abs_mean, _ = _safe_mean_std(ae_raw_logit_after_clip_abs_vals)

    return {
        "irt_abs_mean": irt_abs_mean,
        "irt_std": irt_std,
        "graph_row_entropy_mean": graph_row_entropy_mean,
        "graph_entropy_ratio": graph_entropy_ratio,
        "graph_diag_mass": graph_diag_mass,
        "graph_to_uniform_l2": graph_to_uniform_l2,
        "graph_to_identity_l2": graph_to_identity_l2,
        "graph_tau_mean": tau_mean,
        "graph_tau_std": tau_std,
        "alpha_mean": alpha_mean,
        "alpha_std": alpha_std,
        "alpha_base_mean": alpha_base_mean,
        "alpha_delta_absmean": alpha_delta_absmean,
        "alpha_saturation_ratio": alpha_saturation_ratio,
        "alpha_bias_std": alpha_bias_std,
        "personal_row_entropy": personal_row_entropy,
        "personal_matrix_delta": personal_matrix_delta_mean,
        "personal_matrix_student_std": personal_matrix_student_std_mean,
        "personal_delta_pre_softmax_norm": personal_delta_pre_softmax_norm_mean,
        "personal_delta_student_std": personal_delta_student_std_mean,
        "alpha_head_std": alpha_head_std_mean,
        "alpha_state_path_absmean": alpha_state_path_mean,
        "alpha_id_path_absmean": alpha_id_path_mean,
        "alpha_bias_path_absmean": alpha_bias_path_mean,
        "head_bias_path_absmean": head_bias_path_mean,
        "relation_identity_delta": relation_identity_delta_mean,
        "knowledge_state_graph_delta": knowledge_state_graph_delta_mean,
        "knowledge_state_personal_delta": knowledge_state_personal_delta_mean,
        "personal_bad_row_count": personal_bad_row_mean,
        "personal_fallback_row_count": personal_fallback_row_mean,
        "personal_bad_row_count_active": personal_bad_row_active_mean,
        "personal_fallback_row_count_active": personal_fallback_row_active_mean,
        "personal_bad_row_rate_active": personal_bad_row_rate_active_mean,
        "personal_padded_row_count": personal_padded_row_mean,
        "personal_student_mix": personal_student_mix_mean,
        "personal_mastery_prior_scale": personal_mastery_prior_scale_mean,
        "personal_mastery_scores_absmean": personal_mastery_scores_absmean,
        "personal_mastery_reliability_mean": personal_mastery_reliability_mean,
        "personal_recent_mastery_prior_scale": personal_recent_mastery_prior_scale_mean,
        "personal_recent_mastery_scores_absmean": personal_recent_mastery_scores_absmean,
        "personal_recent_mastery_reliability_mean": personal_recent_mastery_reliability_mean,
        "personal_student_adapter_scale": personal_student_adapter_mean,
        "personal_logits_absmax": personal_logits_absmax_mean,
        "personal_logits_support_absmax": personal_logits_support_absmax_mean,
        "local_row_ratio": local_row_ratio_mean,
        "personal_support_density": personal_support_density_mean,
        "readout_query_delta": readout_query_delta_mean,
        "query_row_graph_delta": query_row_graph_delta_mean,
        "query_row_global_readout_delta": query_row_global_readout_delta_mean,
        "query_row_personal_message_delta": query_row_personal_message_delta_mean,
        "ae_query_state_residual_delta": ae_query_state_residual_delta_mean,
        "ae_logit_residual_abs_mean": ae_logit_residual_abs_mean,
        "query_row_posterior_delta_abs": query_row_posterior_delta_abs_mean,
        "query_row_posterior_kl": query_row_posterior_kl_mean,
        "personal_to_graph_query_ratio": personal_to_graph_query_ratio_mean,
        "personal_to_graph_query_ratio_effective": personal_to_graph_query_ratio_mean,
        "personal_query_trust_scale_mean": personal_query_trust_scale_mean,
        "query_row_global_local_rms": query_row_global_local_rms_mean,
        "query_row_post_local_rms": query_row_post_local_rms_mean,
        "query_row_delta_local_rms_raw": query_row_delta_local_rms_raw_mean,
        "query_row_message_projection_gain": query_row_message_projection_gain_mean,
        "query_row_message_alignment": query_row_message_alignment_mean,
        "query_row_self_support_mass": query_row_self_support_mass_mean,
        "query_row_graph_support_mass": query_row_graph_support_mass_mean,
        "query_row_global_head_var": query_row_global_head_var_mean,
        "query_row_global_head_margin": query_row_global_head_margin_mean,
        "personal_query_writeback_delta": personal_query_writeback_delta_mean,
        "graph_query_adapter_gain": graph_query_adapter_gain_mean,
        "neighbor_row_personal_delta": neighbor_row_personal_delta_mean,
        "personal_row_budget_mean": personal_row_budget_mean,
        "personal_query_row_std": personal_query_row_std_mean,
        "personal_item_support_added_rate": personal_item_support_added_rate_mean,
        "personal_item_support_added_mass": personal_item_support_added_mass_mean,
        "readout_query_support_mass": readout_query_support_mass_mean,
        "query_row_global_support_mass": query_row_global_support_mass_mean,
        "query_row_personal_support_mass": query_row_personal_support_mass_mean,
        "support_candidate_size_mean": support_candidate_size_mean,
        "support_final_size_mean": support_final_size_mean,
        "support_item_survival_rate": support_item_survival_rate,
        "support_seq_survival_rate": support_seq_survival_rate,
        "support_item_seq_overlap": support_item_seq_overlap,
        "support_self_retention_rate": support_self_retention_rate,
        "relation_theta_delta_abs_mean": relation_theta_delta_abs_mean,
        "relation_theta_logit_abs_mean": relation_theta_logit_abs_mean,
        "relation_theta_neighbor_mass": relation_theta_neighbor_mass_mean,
        "relation_theta_personal_mass": relation_theta_personal_mass_mean,
        "ae_posterior_prior_logit_abs_mean": ae_posterior_prior_logit_abs_mean,
        "ae_posterior_prior_delta_abs_mean": ae_posterior_prior_delta_abs_mean,
        "ae_stat_student_prior_abs_mean": ae_stat_student_prior_abs_mean,
        "ae_stat_exercise_prior_abs_mean": ae_stat_exercise_prior_abs_mean,
        "ae_stat_concept_prior_abs_mean": ae_stat_concept_prior_abs_mean,
        "ae_stat_query_student_concept_prior_abs_mean": ae_stat_query_sc_prior_abs_mean,
        "ae_stat_graph_student_concept_prior_abs_mean": ae_stat_graph_sc_prior_abs_mean,
        "ae_reliability_feature_logit_abs_mean": ae_reliability_feature_logit_abs_mean,
        "ae_interpretable_bias_abs_mean": ae_interpretable_bias_abs_mean,
        "ae_explicit_feature_logit_abs_mean": ae_explicit_feature_logit_abs_mean,
        "ae_interaction_logit_abs_mean": ae_interaction_logit_abs_mean,
        "ae_raw_logit_before_clip_abs_mean": ae_raw_logit_before_clip_abs_mean,
        "ae_raw_logit_after_clip_abs_mean": ae_raw_logit_after_clip_abs_mean,
        "ae_posterior_prior_scale": float(getattr(base_model, "ae_posterior_prior_scale", 0.0)),
        "personal_warmup_scale": personal_warmup_scale,
        "share_concept_embeddings": share_concept_embeddings,
    }


def _grad_norm_or_zero(param: Optional[torch.Tensor]) -> float:
    if param is None:
        return 0.0
    grad = getattr(param, "grad", None)
    if grad is None:
        return 0.0
    return float(grad.detach().norm(p=2).item())


def _collect_debug_grad_norms(model: nn.Module) -> Dict[str, float]:
    base_model = _get_base_model(model)

    structure_module = getattr(base_model, "structure_module", None)
    relation_learning = getattr(structure_module, "relation_learning", None) if structure_module is not None else None
    personal_generator = getattr(structure_module, "personal_generator", None) if structure_module is not None else None
    adaptive_gate = getattr(structure_module, "adaptive_gate", None) if structure_module is not None else None

    relation_emb = getattr(relation_learning, "concept_embeddings", None) if relation_learning is not None else None
    relation_tau = (
        getattr(relation_learning, "temperature_raw", getattr(relation_learning, "tau_raw", None))
        if relation_learning is not None
        else None
    )
    relation_prior_strength = getattr(relation_learning, "prior_strength_raw", None) if relation_learning is not None else None
    relation_sequence_strength = (
        getattr(relation_learning, "sequence_prior_strength_raw", None) if relation_learning is not None else None
    )
    relation_receiver_bias = getattr(relation_learning, "receiver_bias", None) if relation_learning is not None else None
    relation_self_loop = getattr(relation_learning, "self_loop_logit", None) if relation_learning is not None else None
    personal_gate_head_bias = getattr(adaptive_gate, "head_bias", None)
    personal_gate_query_weight = getattr(adaptive_gate, "query_norm_weight", None)
    personal_gate_context_weight = getattr(adaptive_gate, "context_norm_weight", None)
    personal_generator_state_mix = getattr(personal_generator, "state_mix_logit", None)

    return {
        "relation_emb": _grad_norm_or_zero(relation_emb),
        "relation_tau": _grad_norm_or_zero(relation_tau),
        "relation_prior_strength": _grad_norm_or_zero(relation_prior_strength),
        "relation_sequence_strength": _grad_norm_or_zero(relation_sequence_strength),
        "relation_receiver_bias": _grad_norm_or_zero(relation_receiver_bias),
        "relation_self_loop": _grad_norm_or_zero(relation_self_loop),
        "personal_gate_head_bias": _grad_norm_or_zero(personal_gate_head_bias),
        "personal_gate_query_weight": _grad_norm_or_zero(personal_gate_query_weight),
        "personal_gate_context_weight": _grad_norm_or_zero(personal_gate_context_weight),
        "personal_generator_state_mix": _grad_norm_or_zero(personal_generator_state_mix),
    }


def _collect_diag_warning_tags(diag: Dict[str, float]) -> List[str]:
    tags: List[str] = []
    if diag.get("alpha_std", 0.0) < 1e-6:
        tags.append("personal-alpha-collapse")
    if diag.get("personal_delta_student_std", 0.0) < 1e-6:
        tags.append("personal-delta-student-flat")
    if diag.get("alpha_head_std", 0.0) < 1e-6:
        tags.append("alpha-head-collapse")
    if diag.get("alpha_id_path_absmean", 0.0) > max(diag.get("alpha_state_path_absmean", 0.0), 1e-6):
        tags.append("alpha-id-dominant")
    if diag.get("alpha_saturation_ratio", 0.0) > 0.60 and diag.get("alpha_std", 0.0) < 0.02:
        tags.append("gate-saturated")
    if diag.get("personal_matrix_delta", 0.0) > 1e-5 and diag.get("personal_matrix_student_std", 0.0) < 5e-4:
        tags.append("personalization-flat")
    if diag.get("query_row_personal_message_delta", 0.0) > max(1.25 * diag.get("query_row_global_readout_delta", 0.0), 0.05):
        tags.append("E-active-but-overrides-global")
    if diag.get("query_row_personal_message_delta", 0.0) < 0.002 and diag.get("query_row_posterior_delta_abs", 0.0) < 0.002:
        tags.append("E-flat-at-query")
    if diag.get("query_row_global_readout_delta", 0.0) < 0.01 and diag.get("knowledge_state_graph_delta", 0.0) > 0.02:
        tags.append("A-readout-washed-out")
    if diag.get("query_row_posterior_kl", 0.0) > 0.25 and diag.get("query_row_personal_message_delta", 0.0) < 0.01:
        tags.append("query-posterior-too-diffuse")
    if diag.get("readout_query_delta", 0.0) > 0.10 and diag.get("query_row_personal_message_delta", 0.0) < 0.002:
        tags.append("query-readout-overamplified")
    if diag.get("personal_bad_row_rate_active", 0.0) > 0.10:
        tags.append("E-fallback-heavy-active-rows")
    if diag.get("personal_padded_row_count", 0.0) > 0 and diag.get("personal_bad_row_rate_active", 0.0) < 1e-6:
        tags.append("E-padded-row-artifact")
    if diag.get("personal_query_trust_scale_mean", 1.0) < 0.98:
        tags.append("E-query-correction-capped")
    if diag.get("query_row_self_support_mass", 0.0) <= 1e-8 and diag.get("query_row_graph_support_mass", 0.0) <= 1e-8:
        tags.append("E-support-empty")
    if diag.get("query_row_posterior_kl", 0.0) > 0.01 and diag.get("query_row_message_projection_gain", 1.0) < 0.05:
        tags.append("E-projection-collapse")
        tags.append("E-posterior-changes-but-message-cancelled")
    if diag.get("query_row_message_alignment", 0.0) < 0.0 and diag.get("personal_query_trust_scale_mean", 1.0) < 0.90:
        tags.append("E-misaligned-trust-clipped")
    if (
        diag.get("relation_wq_grad_norm", 1.0) + diag.get("relation_wk_grad_norm", 1.0)
    ) < 1e-8:
        tags.append("A-adjacency-frozen")
    return tags


# ======================================================
# Train / Validate (use BCEWithLogitsLoss)
# ======================================================

def train_epoch(
    model: CognitiveDiagnosisModel,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    logger,
    epoch: int,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_bce = 0.0
    total_reg = 0.0
    reg_component_sums: Dict[str, float] = {k: 0.0 for k in _REG_COMPONENT_KEYS}

    all_labels: List[float] = []
    all_preds: List[float] = []
    all_probs: List[float] = []
    num_batches_processed = 0

    bce_fn = nn.BCEWithLogitsLoss()
    max_batches = None if max_batches is None else max(1, int(max_batches))

    for batch_idx, batch in enumerate(train_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        student_ids, exercise_ids, labels = batch
        student_ids = student_ids.to(device)
        exercise_ids = exercise_ids.to(device)
        labels = _ensure_1d(labels.to(device).float())

        # get logits + details (for regularizers)
        logits, details = model(
            student_ids,
            exercise_ids,
            return_details=True,
            return_logits=True,
        )
        logits = _ensure_1d(logits)

        bce_loss = bce_fn(logits, labels)
        reg_terms = _get_base_model(model).get_regularization_components(
            relation_matrices=details["relation_matrices"],
            details=details,
            base_loss=bce_loss,
        )
        reg_loss = reg_terms["total"]
        loss = bce_loss + reg_loss
        _raise_if_nonfinite(
            stage="train",
            epoch=epoch,
            batch_idx=batch_idx,
            details=details,
            logits=logits,
            bce_loss=bce_loss,
            reg_terms=reg_terms,
            loss=loss,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_clip_stats = _clip_stability_sensitive_grads(model)
        nonfinite_grad_count = int(grad_clip_stats.get("nonfinite_grad_count", 0.0))
        if nonfinite_grad_count > 0:
            logger.warning(
                "[Grad Guard] sanitized %d non-finite gradient values before optimizer.step "
                "(graph=%d, personal=%d, other=%d, params=%s)",
                nonfinite_grad_count,
                int(grad_clip_stats.get("nonfinite_graph_grad_count", 0.0)),
                int(grad_clip_stats.get("nonfinite_personal_grad_count", 0.0)),
                int(grad_clip_stats.get("nonfinite_other_grad_count", 0.0)),
                grad_clip_stats.get("nonfinite_grad_param_names", ""),
            )
        optimizer.step()

        total_loss += float(loss.item())
        total_bce += float(bce_loss.item())
        total_reg += float(reg_loss.item())
        for key in _REG_COMPONENT_KEYS:
            reg_component_sums[key] += float(reg_terms[key].item())
        num_batches_processed += 1

        with torch.no_grad():
            probs = _sigmoid_torch(logits)
            preds = (probs > 0.5).float()

        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_probs.extend(probs.detach().cpu().numpy().tolist())

    denom = max(1, num_batches_processed)
    avg_loss = total_loss / denom
    avg_bce = total_bce / denom
    avg_reg = total_reg / denom
    avg_reg_components = {
        f"reg_{key}": reg_component_sums[key] / denom
        for key in _REG_COMPONENT_KEYS
    }
    reg_bce_ratio = avg_reg / (abs(avg_bce) + 1e-12)
    metrics = compute_metrics(all_labels, all_preds, all_probs)

    return {
        "loss": avg_loss,
        "bce_loss": avg_bce,
        "reg_loss": avg_reg,
        "reg_bce_ratio": reg_bce_ratio,
        **avg_reg_components,
        **metrics,
    }


def validate(
    model: CognitiveDiagnosisModel,
    val_loader: DataLoader,
    device: torch.device,
    logger,
    epoch: int,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_bce = 0.0
    total_reg = 0.0
    reg_component_sums: Dict[str, float] = {k: 0.0 for k in _REG_COMPONENT_KEYS}

    all_labels: List[float] = []
    all_preds: List[float] = []
    all_probs: List[float] = []
    num_batches_processed = 0

    bce_fn = nn.BCEWithLogitsLoss()
    max_batches = None if max_batches is None else max(1, int(max_batches))

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            student_ids, exercise_ids, labels = batch
            student_ids = student_ids.to(device)
            exercise_ids = exercise_ids.to(device)
            labels = _ensure_1d(labels.to(device).float())

            logits, details = model(
                student_ids,
                exercise_ids,
                return_details=True,
                return_logits=True,
            )
            logits = _ensure_1d(logits)

            bce_loss = bce_fn(logits, labels)
            reg_terms = _get_base_model(model).get_regularization_components(
                relation_matrices=details["relation_matrices"],
                details=details,
                base_loss=bce_loss,
            )
            reg_loss = reg_terms["total"]
            loss = bce_loss + reg_loss
            _raise_if_nonfinite(
                stage="val",
                epoch=epoch,
                batch_idx=batch_idx,
                details=details,
                logits=logits,
                bce_loss=bce_loss,
                reg_terms=reg_terms,
                loss=loss,
            )

            total_loss += float(loss.item())
            total_bce += float(bce_loss.item())
            total_reg += float(reg_loss.item())
            for key in _REG_COMPONENT_KEYS:
                reg_component_sums[key] += float(reg_terms[key].item())
            num_batches_processed += 1

            probs = _sigmoid_torch(logits)
            preds = (probs > 0.5).float()

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    denom = max(1, num_batches_processed)
    avg_loss = total_loss / denom
    avg_bce = total_bce / denom
    avg_reg = total_reg / denom
    avg_reg_components = {
        f"reg_{key}": reg_component_sums[key] / denom
        for key in _REG_COMPONENT_KEYS
    }
    reg_bce_ratio = avg_reg / (abs(avg_bce) + 1e-12)
    metrics = compute_metrics(all_labels, all_preds, all_probs)

    return {
        "loss": avg_loss,
        "bce_loss": avg_bce,
        "reg_loss": avg_reg,
        "reg_bce_ratio": reg_bce_ratio,
        **avg_reg_components,
        **metrics,
    }


# ======================================================
# Train / Inference
# ======================================================

def train_one_experiment(args, logger) -> Tuple[float, int]:
    device = select_device(args, logger)

    run_tag = (
        f"[{getattr(args, 'dataset_name', 'unknown')}"
        f"|{getattr(args, 'model_variant', 'full')}"
        f"|lr={args.learning_rate:g}"
        f"|drop={args.dropout:.2f}]"
    )

    logger.info("%s Loading datasets...", run_tag)
    logger.info(
        "%s Regularization: graph_entropy(lambda_sparse)=%.6f, graph_diag=%.6f, graph_uniform=%.6f, "
        "personal_sparse=%.6f, personal_kl=%.6f, personal_query_residual=%.6f, "
        "theta_prior_align=%.6f, alpha_penalty=%.6f, alpha_min=%.6f, prediction_l2=%.6f",
        run_tag,
        args.lambda_sparse,
        getattr(args, "lambda_graph_diag", 0.10),
        getattr(args, "lambda_graph_uniform", 0.04),
        args.lambda_sparse_personal,
        getattr(args, "lambda_personal_kl", 0.0),
        getattr(args, "lambda_personal_query_residual", 0.0),
        getattr(args, "lambda_theta_prior_align", 0.0),
        args.lambda_alpha,
        getattr(args, "lambda_alpha_min", 0.0),
        getattr(args, "prediction_l2_lambda", 5e-5),
    )
    logger.info(
        "%s Graph controls: entropy_band=[%.2f, %.2f], uniform_margin=%.2f, warmup_epochs=%d, cap_ratio=%.2f, graph_dropout=%.3f, graph_tau_init=%.3f",
        run_tag,
        getattr(args, "graph_entropy_min", 0.15),
        getattr(args, "graph_entropy_max", 0.85),
        getattr(args, "graph_uniform_margin", 0.10),
        getattr(args, "graph_reg_warmup_epochs", 1),
        getattr(args, "graph_reg_cap_ratio", 6.0),
        getattr(args, "graph_dropout", -1.0),
        getattr(args, "graph_tau_init", 1.0),
    )
    logger.info(
        "%s Personal controls: personal_max_alpha=%.3f, personal_delta_scale=%.3f, personal_warmup_epochs=%d, "
        "personal_student_dim=%s, alpha_min_target=%.4f, personal_local_hops=%s, include_neighbor_rows=%s, personal_support_only=%s, "
        "personal_alpha_temperature=%.3f, personal_alpha_budget=%.3f, personal_query_row_budget=%.3f, "
        "personal_neighbor_row_budget=%.3f, personal_query_support_hops=%s, personal_item_support_mass=%.3f, "
        "personal_mastery_prior_scale=%.3f, personal_recent_mastery_prior_scale=%.3f, "
        "personal_query_message_gain=%.3f, "
        "personal_query_correction_scale=%.3f, personal_state_lr_mult=%.3f, personal_id_lr_mult=%.3f, "
        "graph_propagation_alpha=%.3f, graph_query_readout_scale=%.3f, graph_query_readout_2hop_scale=%.3f",
        run_tag,
        float(getattr(args, "personal_max_alpha", 0.35)),
        float(getattr(args, "personal_delta_scale", 1.0)),
        int(getattr(args, "personal_warmup_epochs", 0)),
        getattr(args, "personal_student_dim", None),
        float(getattr(args, "alpha_min_target", 0.0)),
        getattr(args, "personal_local_hops", None),
        getattr(args, "personal_include_neighbor_rows", None),
        getattr(args, "personal_support_only", None),
        float(getattr(args, "personal_alpha_temperature", 2.0)),
        float(getattr(args, "personal_alpha_budget", 0.10)),
        float(getattr(args, "personal_query_row_budget", 1.0)),
        float(getattr(args, "personal_neighbor_row_budget", 0.30)),
        getattr(args, "personal_query_support_hops", None),
        float(getattr(args, "personal_item_support_mass", 0.0)),
        float(getattr(args, "personal_mastery_prior_scale", 0.0)),
        float(getattr(args, "personal_recent_mastery_prior_scale", 0.0)),
        float(getattr(args, "personal_query_message_gain", 1.0)),
        float(getattr(args, "personal_query_correction_scale", 0.15)),
        float(getattr(args, "personal_state_lr_mult", 1.0)),
        float(getattr(args, "personal_id_lr_mult", 0.5)),
        float(getattr(args, "graph_propagation_alpha", 0.20)),
        float(getattr(args, "graph_query_readout_scale", 0.35)),
        float(getattr(args, "graph_query_readout_2hop_scale", 0.15)),
    )

    data_dir = args.data_dir
    train_file = os.path.join(data_dir, "train.csv")
    val_file = os.path.join(data_dir, "valid.csv")
    test_file = os.path.join(data_dir, "test.csv")

    train_loader, val_loader, test_loader, info_dict = create_dataloaders(
        train_file=train_file,
        val_file=val_file,
        test_file=test_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle_train=True,
        min_stu_interactions=args.min_stu_interactions,
        min_exer_interactions=args.min_exer_interactions,
        min_poison_count=args.min_poison_count,
        logger=logger,
        dataset_name=args.dataset_name if hasattr(args, "dataset_name") else args.dataset,
        disable_sequence_prior=bool(getattr(args, "disable_sequence_prior", False)),
        disable_item_prior=bool(getattr(args, "disable_item_prior", False)),
        graph_prior_mode=str(getattr(args, "graph_prior_mode", "evidence")),
    )

    logger.info(
        "%s Train samples: %d, Val samples: %d",
        run_tag,
        info_dict["train_size"],
        info_dict["val_size"],
    )
    logger.info("%s Number of students: %d", run_tag, info_dict["num_students"])
    logger.info("%s Number of exercises: %d", run_tag, info_dict["num_exercises"])
    logger.info("%s Number of concepts: %d", run_tag, info_dict["num_concepts"])
    prior_stats = info_dict.get("graph_prior_stats", {})
    if prior_stats:
        logger.info(
            "%s [A Prior] item_edges=%.0f item_density=%.4f item_entropy=%.4f | "
            "seq_raw_mass=%.1f seq_weighted_mass=%.1f seq_students=%.0f "
            "seq_edges=%.0f seq_density=%.4f seq_entropy=%.4f",
            run_tag,
            float(prior_stats.get("item_observed_edge_count", 0.0)),
            float(prior_stats.get("item_prior_density", 0.0)),
            float(prior_stats.get("item_prior_entropy", 0.0)),
            float(prior_stats.get("seq_raw_transition_mass", 0.0)),
            float(prior_stats.get("seq_student_weighted_mass", 0.0)),
            float(prior_stats.get("seq_student_count", 0.0)),
            float(prior_stats.get("sequence_observed_edge_count", 0.0)),
            float(prior_stats.get("sequence_prior_density", 0.0)),
            float(prior_stats.get("sequence_prior_entropy", 0.0)),
        )

    logger.info("%s Creating model...", run_tag)

    # switches from main.py (already normalized there)
    use_concept_graph = getattr(args, "use_concept_graph", True)
    ablate_module1 = bool(getattr(args, "ablate_module1", False))
    debug_graph_diag = bool(getattr(args, "debug_graph_diag", False))
    diag_batches = max(1, int(getattr(args, "diag_batches", 2)))
    expected_enable_module1 = not ablate_module1

    eff_gnn_layers = _hard_ablation_effective_hparams(
        use_concept_graph=use_concept_graph,
        num_gnn_layers=getattr(args, "num_gnn_layers", 0),
    )

    logger.info(
        "%s Ablation switches: "
        "ablate_module1=%s | enable_module1=%s | use_concept_graph=%s | "
        "effective(num_gnn_layers=%d)",
        run_tag,
        ablate_module1,
        expected_enable_module1,
        use_concept_graph,
        eff_gnn_layers,
    )

    model = CognitiveDiagnosisModel(
        num_students=info_dict["num_students"],
        num_exercises=info_dict["num_exercises"],
        num_concepts=info_dict["num_concepts"],
        q_matrix=info_dict["q_matrix"],
        item_prior_matrix=info_dict.get("item_prior_matrix"),
        sequence_prior_matrix=info_dict.get("sequence_prior_matrix"),
        knowledge_dim=args.knowledge_dim,
        num_relation_heads=args.num_relation_heads,
        num_gnn_layers=eff_gnn_layers,
        dropout=args.dropout,
        use_concept_graph=use_concept_graph,
        graph_topk=getattr(args, "graph_topk", None),
        allow_self_loop=not getattr(args, "disable_self_loop", False),
        use_personal_graph=getattr(args, "use_personal_graph", False),
        ablate_module1=ablate_module1,
        graph_dropout=_resolve_optional_graph_dropout(getattr(args, "graph_dropout", -1.0)),
        graph_tau_init=getattr(args, "graph_tau_init", 1.0),
        graph_propagation_alpha=getattr(args, "graph_propagation_alpha", 0.20),
        graph_query_readout_scale=getattr(args, "graph_query_readout_scale", 0.35),
        graph_query_readout_2hop_scale=getattr(args, "graph_query_readout_2hop_scale", 0.15),
        personal_rank=getattr(args, "personal_rank", 4),
        lambda_sparse_personal=args.lambda_sparse_personal,
        lambda_alpha=args.lambda_alpha,
        lambda_personal_kl=getattr(args, "lambda_personal_kl", 0.0),
        lambda_personal_query_residual=getattr(args, "lambda_personal_query_residual", 0.0),
        personal_query_residual_margin=getattr(args, "personal_query_residual_margin", 0.0),
        lambda_graph_entropy=args.lambda_sparse,
        graph_entropy_min=getattr(args, "graph_entropy_min", 0.15),
        graph_entropy_max=getattr(args, "graph_entropy_max", 0.85),
        lambda_graph_diag=getattr(args, "lambda_graph_diag", 0.10),
        lambda_graph_uniform=getattr(args, "lambda_graph_uniform", 0.04),
        graph_uniform_margin=getattr(args, "graph_uniform_margin", 0.10),
        graph_reg_warmup_epochs=getattr(args, "graph_reg_warmup_epochs", 1),
        graph_reg_cap_ratio=getattr(args, "graph_reg_cap_ratio", 6.0),
        prediction_l2_lambda=getattr(args, "prediction_l2_lambda", 5e-5),
        gnn_residual_weight=getattr(args, "gnn_residual_weight", 0.5),
        graph_identity_residual=getattr(args, "graph_identity_residual", 0.0),
        personal_max_alpha=getattr(args, "personal_max_alpha", 0.35),
        personal_delta_scale=getattr(args, "personal_delta_scale", 1.0),
        personal_warmup_epochs=getattr(args, "personal_warmup_epochs", 0),
        personal_reg_warmup_epochs=getattr(args, "personal_reg_warmup_epochs", None),
        personal_student_dim=getattr(args, "personal_student_dim", args.knowledge_dim),
        lambda_alpha_min=getattr(args, "lambda_alpha_min", 0.0),
        alpha_min_target=getattr(args, "alpha_min_target", 0.0),
        personal_alpha_temperature=getattr(args, "personal_alpha_temperature", 2.0),
        personal_alpha_budget=getattr(args, "personal_alpha_budget", 0.10),
        personal_alpha_base_init=getattr(args, "personal_alpha_base_init", 0.08),
        personal_alpha_bias_scale=getattr(args, "personal_alpha_bias_scale", 1.0),
        personal_freeze_alpha_gate=getattr(args, "personal_freeze_alpha_gate", False),
        personal_disable_student_global_context=getattr(
            args, "personal_disable_student_global_context", False
        ),
        personal_local_hops=getattr(args, "personal_local_hops", 1),
        personal_include_neighbor_rows=getattr(args, "personal_include_neighbor_rows", False),
        personal_support_include_query_self=getattr(args, "personal_support_include_query_self", True),
        personal_support_include_graph=getattr(args, "personal_support_include_graph", True),
        personal_support_include_neighbors=getattr(args, "personal_support_include_neighbors", False),
        personal_item_support_mass=getattr(args, "personal_item_support_mass", 0.0),
        personal_mastery_prior_scale=getattr(args, "personal_mastery_prior_scale", 0.0),
        personal_recent_mastery_prior_scale=getattr(args, "personal_recent_mastery_prior_scale", 0.0),
        personal_mastery_count_smoothing=getattr(args, "personal_mastery_count_smoothing", 0.0),
        personal_query_row_budget=getattr(args, "personal_query_row_budget", 1.0),
        personal_neighbor_row_budget=getattr(args, "personal_neighbor_row_budget", 0.30),
        personal_query_support_hops=getattr(args, "personal_query_support_hops", 0),
        personal_support_only=getattr(args, "personal_support_only", True),
        personal_query_correction_scale=getattr(args, "personal_query_correction_scale", 0.15),
        personal_query_correction_max_ratio=getattr(args, "personal_query_correction_max_ratio", 0.20),
        personal_query_correction_min_graph_anchor=getattr(args, "personal_query_correction_min_graph_anchor", 0.01),
        personal_query_message_gain=getattr(args, "personal_query_message_gain", 1.0),
        personal_value_use_global_basis=getattr(args, "personal_value_use_global_basis", True),
        personal_message_alignment_gate=getattr(args, "personal_message_alignment_gate", True),
        personal_projection_hidden_factor=getattr(args, "personal_projection_hidden_factor", 2),
        graph_headwise_query_gate=getattr(args, "graph_headwise_query_gate", True),
        graph_edge_bias_rank=getattr(args, "graph_edge_bias_rank", 8),
        graph_prior_logit_scale=getattr(args, "graph_prior_logit_scale", 0.0),
        ae_query_residual_scale=getattr(args, "ae_query_residual_scale", 0.0),
        ae_logit_residual_scale=getattr(args, "ae_logit_residual_scale", 0.0),
        ae_logit_residual_clip=getattr(args, "ae_logit_residual_clip", 1.0),
        ae_irt_logit_scale=getattr(args, "ae_irt_logit_scale", 1.0),
        ae_interaction_logit_scale=getattr(args, "ae_interaction_logit_scale", 0.0),
        ae_logit_dim=getattr(args, "ae_logit_dim", 32),
        ae_posterior_prior_scale=getattr(args, "ae_posterior_prior_scale", 0.0),
        lambda_theta_prior_align=getattr(args, "lambda_theta_prior_align", 0.0),
        relation_theta_scale=getattr(args, "relation_theta_scale", 0.0),
        relation_theta_delta_clip=getattr(args, "relation_theta_delta_clip", 2.0),
        graph_query_adapter_enable=getattr(args, "graph_query_adapter_enable", True),
        share_concept_embeddings=getattr(args, "share_concept_embeddings", False),
    ).to(device)
    _initialize_ae_stat_priors(model, train_loader, info_dict, args, logger, run_tag)

    # 多 GPU 支持（DataParallel）
    is_multi_gpu = getattr(args, "multi_gpu", False) and torch.cuda.device_count() > 1
    if is_multi_gpu:
        gpu_ids_str = getattr(args, "gpu_ids", None)
        if gpu_ids_str:
            # 使用指定的 GPU（已通过 CUDA_VISIBLE_DEVICES 映射，这里用相对索引）
            num_visible = torch.cuda.device_count()
            device_ids = list(range(num_visible))
        else:
            device_ids = None
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        logger.info("%s Multi-GPU enabled: using %d GPUs", run_tag, torch.cuda.device_count())

    _log_and_assert_ablation_consistency(
        model=model,
        logger=logger,
        context=run_tag,
        ablate_module1=ablate_module1,
        expect_use_concept_graph=use_concept_graph,
        expect_use_personal_graph=getattr(args, "use_personal_graph", False),
    )
    if debug_graph_diag:
        logger.info("%s Debug diagnostics enabled: diag_batches=%d", run_tag, diag_batches)
    _log_graph_init_state(model, logger=logger, context=run_tag)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("%s Total parameters: %s", run_tag, f"{total_params:,}")
    logger.info("%s Trainable parameters: %s", run_tag, f"{trainable_params:,}")

    optimizer = _build_optimizer(model, args)
    param_group_desc = ", ".join(
        f"{group.get('group_name', f'group{i}')}@lr={group['lr']:.6g}"
        for i, group in enumerate(optimizer.param_groups)
    )
    logger.info("%s Optimizer param groups: %s", run_tag, param_group_desc)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=MONITOR_MODE,
        factor=0.5,
        patience=args.patience,
    )
    monitor_config = _default_monitor_config()
    logger.info(
        "%s Monitor config: scheduler=%s(%s), best_checkpoint=%s(%s), early_stop=%s(%s)",
        run_tag,
        monitor_config["scheduler_monitor"],
        monitor_config["scheduler_mode"],
        monitor_config["best_monitor"],
        monitor_config["best_mode"],
        monitor_config["early_stop_monitor"],
        monitor_config["early_stop_mode"],
    )

    best_val_auc = float("-inf")
    best_epoch = 0
    patience_counter = 0

    history: Dict[str, Any] = {
        "train": [],
        "val": [],
        "best_epoch": 0,
        "best_val_auc": float("-inf"),
        "monitor": monitor_config,
    }
    alpha_zero_streak = 0
    graph_uniform_streak = 0
    graph_low_grad_streak = 0
    last_diag: Dict[str, Any] = {}

    logger.info("%s Starting training...", run_tag)

    for epoch in range(1, args.epochs + 1):
        _get_base_model(model).set_epoch(epoch)
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            logger,
            epoch,
            max_batches=getattr(args, "max_train_batches", None),
        )
        grad_norms = _collect_debug_grad_norms(model) if debug_graph_diag else None

        val_metrics = validate(
            model,
            val_loader,
            device,
            logger,
            epoch,
            max_batches=getattr(args, "max_val_batches", None),
        )

        logger.info(
            "%s Epoch [%03d/%d] | "
            "Train: Loss=%.4f, BCE=%.4f, Reg=%.4f, AUC=%.4f, ACC=%.4f, RMSE=%.4f | "
            "Val: Loss=%.4f, BCE=%.4f, Reg=%.4f, AUC=%.4f, ACC=%.4f, RMSE=%.4f",
            run_tag,
            epoch, args.epochs,
            train_metrics["loss"], train_metrics["bce_loss"], train_metrics["reg_loss"],
            train_metrics["auc"], train_metrics["acc"], train_metrics["rmse"],
            val_metrics["loss"], val_metrics["bce_loss"], val_metrics["reg_loss"],
            val_metrics["auc"], val_metrics["acc"], val_metrics["rmse"],
        )
        logger.info(
            "%s [Reg Terms] Epoch [%03d] | "
            "Train: graph_entropy=%.6f, graph_diag=%.6f, graph_uniform=%.6f, graph_reg_scale=%.4f, "
            "prediction_l2=%.6f, personal_sparse=%.6f, theta_prior_align=%.6f, "
            "alpha_var=%.6f, alpha_collapse=%.6f, reg_bce_ratio=%.4f | "
            "Val: graph_entropy=%.6f, graph_diag=%.6f, graph_uniform=%.6f, graph_reg_scale=%.4f, "
            "prediction_l2=%.6f, personal_sparse=%.6f, theta_prior_align=%.6f, "
            "alpha_var=%.6f, alpha_collapse=%.6f, reg_bce_ratio=%.4f",
            run_tag,
            epoch,
            train_metrics.get("reg_graph_entropy", 0.0),
            train_metrics.get("reg_graph_diag", 0.0),
            train_metrics.get("reg_graph_uniform", 0.0),
            train_metrics.get("reg_graph_reg_scale", 1.0),
            train_metrics.get("reg_prediction_l2", 0.0),
            train_metrics.get("reg_personal_sparse", 0.0),
            train_metrics.get("reg_theta_prior_align", 0.0),
            train_metrics.get("reg_alpha_var", 0.0),
            train_metrics.get("reg_alpha_collapse", 0.0),
            train_metrics.get("reg_bce_ratio", 0.0),
            val_metrics.get("reg_graph_entropy", 0.0),
            val_metrics.get("reg_graph_diag", 0.0),
            val_metrics.get("reg_graph_uniform", 0.0),
            val_metrics.get("reg_graph_reg_scale", 1.0),
            val_metrics.get("reg_prediction_l2", 0.0),
            val_metrics.get("reg_personal_sparse", 0.0),
            val_metrics.get("reg_theta_prior_align", 0.0),
            val_metrics.get("reg_alpha_var", 0.0),
            val_metrics.get("reg_alpha_collapse", 0.0),
            val_metrics.get("reg_bce_ratio", 0.0),
        )

        if debug_graph_diag:
            diag = _collect_debug_forward_stats(
                model=model,
                data_loader=val_loader,
                device=device,
                max_batches=diag_batches,
            )
            if grad_norms is not None:
                diag["relation_wq_grad_norm"] = float(grad_norms.get("relation_wq", 0.0))
                diag["relation_wk_grad_norm"] = float(grad_norms.get("relation_wk", 0.0))
                diag["relation_tau_grad_norm"] = float(grad_norms.get("relation_tau", 0.0))
                diag["graph_edge_bias_grad_norm"] = float(grad_norms.get("graph_edge_bias", 0.0))
            _write_diag_json(args.save_dir, "diag_last.json", diag)
            alpha_bias_std_text = _fmt_diag_value(diag.get("alpha_bias_std"), ".4f")
            logger.info(
                "%s [Diag][AE] Epoch [%03d] | "
                "irt_abs_mean=%.4f, irt_std=%.4f, personal_warmup_scale=%.2f, "
                "relation_theta_delta=%.4f, relation_theta_logit=%.4f, "
                "relation_theta_neighbor_mass=%.4f, relation_theta_personal_mass=%.4f, "
                "ae_posterior_prior_scale=%.4f, ae_posterior_prior_logit=%.4f, "
                "ae_posterior_prior_delta=%.4f, "
                "graph_row_entropy=%.4f, graph_entropy_ratio=%.4f, "
                "alpha_mean=%.4f, alpha_std=%.4f, alpha_base_mean=%.4f, alpha_delta_absmean=%.4f, "
                "alpha_saturation_ratio=%.4f, alpha_bias_std=%s, "
                "alpha_state_path=%.4f, alpha_id_path=%.4f, alpha_bias_path=%.4f, head_bias_path=%.4f, "
                "personal_row_entropy=%.4f, personal_matrix_delta=%.4f, personal_matrix_student_std=%.4f, "
                "personal_delta_pre_softmax_norm=%.4f, personal_delta_student_std=%.4f, alpha_head_std=%.4f, "
                "query_row_global_readout_delta=%.4f, query_row_personal_message_delta=%.4f, "
                "ae_query_state_residual_delta=%.4f, ae_logit_residual_abs_mean=%.4f, "
                "query_row_posterior_delta_abs=%.4f, query_row_posterior_kl=%.4f, "
                "query_row_delta_local_rms_raw=%.4f, query_row_message_projection_gain=%.4f, "
                "personal_to_graph_query_ratio=%.4f, query_row_global_support_mass=%.4f, "
                "query_row_personal_support_mass=%.4f, neighbor_row_personal_delta=%.4f, "
                "personal_row_budget_mean=%.4f, personal_query_row_std=%.4f, "
                "personal_item_support_added_rate=%.4f, personal_item_support_added_mass=%.4f, "
                "readout_query_support_mass=%.4f, "
                "personal_student_mix=%.4f, personal_mastery_prior_scale=%.4f, "
                "personal_mastery_scores_absmean=%.4f, personal_recent_mastery_prior_scale=%.4f, "
                "personal_recent_mastery_scores_absmean=%.4f, personal_student_adapter=%.4f, "
                "local_row_ratio=%.4f, support_density=%.4f, readout_query_delta=%.4f, "
                "personal_bad_rows_active=%.2f, personal_fallback_rows_active=%.2f, "
                "personal_bad_row_rate_active=%.4f, personal_padded_rows=%.2f, "
                "personal_logits_support_absmax=%.4f, personal_query_trust_scale_mean=%.4f, "
                "personal_bad_rows_all=%.2f, personal_fallback_rows_all=%.2f, personal_logits_absmax_all=%.4f",
                run_tag,
                epoch,
                diag["irt_abs_mean"],
                diag["irt_std"],
                diag["personal_warmup_scale"],
                diag["relation_theta_delta_abs_mean"],
                diag["relation_theta_logit_abs_mean"],
                diag["relation_theta_neighbor_mass"],
                diag["relation_theta_personal_mass"],
                diag["ae_posterior_prior_scale"],
                diag["ae_posterior_prior_logit_abs_mean"],
                diag["ae_posterior_prior_delta_abs_mean"],
                diag["graph_row_entropy_mean"],
                diag["graph_entropy_ratio"],
                diag["alpha_mean"],
                diag["alpha_std"],
                diag["alpha_base_mean"],
                diag["alpha_delta_absmean"],
                diag["alpha_saturation_ratio"],
                alpha_bias_std_text,
                diag["alpha_state_path_absmean"],
                diag["alpha_id_path_absmean"],
                diag["alpha_bias_path_absmean"],
                diag["head_bias_path_absmean"],
                diag["personal_row_entropy"],
                diag["personal_matrix_delta"],
                diag["personal_matrix_student_std"],
                diag["personal_delta_pre_softmax_norm"],
                diag["personal_delta_student_std"],
                diag["alpha_head_std"],
                diag["query_row_global_readout_delta"],
                diag["query_row_personal_message_delta"],
                diag["ae_query_state_residual_delta"],
                diag["ae_logit_residual_abs_mean"],
                diag["query_row_posterior_delta_abs"],
                diag["query_row_posterior_kl"],
                diag["query_row_delta_local_rms_raw"],
                diag["query_row_message_projection_gain"],
                diag["personal_to_graph_query_ratio"],
                diag["query_row_global_support_mass"],
                diag["query_row_personal_support_mass"],
                diag["neighbor_row_personal_delta"],
                diag["personal_row_budget_mean"],
                diag["personal_query_row_std"],
                diag["personal_item_support_added_rate"],
                diag["personal_item_support_added_mass"],
                diag["readout_query_support_mass"],
                diag["personal_student_mix"],
                diag["personal_mastery_prior_scale"],
                diag["personal_mastery_scores_absmean"],
                diag["personal_recent_mastery_prior_scale"],
                diag["personal_recent_mastery_scores_absmean"],
                diag["personal_student_adapter_scale"],
                diag["local_row_ratio"],
                diag["personal_support_density"],
                diag["readout_query_delta"],
                diag["personal_bad_row_count_active"],
                diag["personal_fallback_row_count_active"],
                diag["personal_bad_row_rate_active"],
                diag["personal_padded_row_count"],
                diag["personal_logits_support_absmax"],
                diag["personal_query_trust_scale_mean"],
                diag["personal_bad_row_count"],
                diag["personal_fallback_row_count"],
                diag["personal_logits_absmax"],
            )
            logger.info(
                "%s [Diag][A] Epoch [%03d] | "
                "entropy_ratio=%.4f, diag_mass=%.4f, to_uniform_l2=%.6f, to_identity_l2=%.6f, "
                "relation_identity_delta=%.6f, knowledge_state_graph_delta=%.6f, "
                "knowledge_state_personal_delta=%.6f, share_concept_embeddings=%s, "
                "tau_mean=%.4f, tau_std=%.4f, graph_reg_scale(train/val)=%.4f/%.4f",
                run_tag,
                epoch,
                diag["graph_entropy_ratio"],
                diag["graph_diag_mass"],
                diag["graph_to_uniform_l2"],
                diag["graph_to_identity_l2"],
                diag["relation_identity_delta"],
                diag["knowledge_state_graph_delta"],
                diag["knowledge_state_personal_delta"],
                bool(diag["share_concept_embeddings"]),
                diag["graph_tau_mean"],
                diag["graph_tau_std"],
                train_metrics.get("reg_graph_reg_scale", 1.0),
                val_metrics.get("reg_graph_reg_scale", 1.0),
            )
            logger.info(
                "%s [Diag][A Support] Epoch [%03d] | "
                "candidate_size=%.2f, final_size=%.2f, item_survival=%.4f, "
                "seq_survival=%.4f, item_seq_overlap=%.4f, self_retention=%.4f",
                run_tag,
                epoch,
                diag["support_candidate_size_mean"],
                diag["support_final_size_mean"],
                diag["support_item_survival_rate"],
                diag["support_seq_survival_rate"],
                diag["support_item_seq_overlap"],
                diag["support_self_retention_rate"],
            )
            logger.info(
                "%s [Diag][AE Components] Epoch [%03d] | "
                "student_prior=%.4f, exercise_prior=%.4f, concept_prior=%.4f, "
                "query_student_concept=%.4f, graph_student_concept=%.4f, "
                "posterior_prior=%.4f, reliability=%.4f, interpretable_bias=%.4f, explicit_feature=%.4f, "
                "interaction=%.4f, raw_before_clip=%.4f, raw_after_clip=%.4f",
                run_tag,
                epoch,
                diag["ae_stat_student_prior_abs_mean"],
                diag["ae_stat_exercise_prior_abs_mean"],
                diag["ae_stat_concept_prior_abs_mean"],
                diag["ae_stat_query_student_concept_prior_abs_mean"],
                diag["ae_stat_graph_student_concept_prior_abs_mean"],
                diag["ae_posterior_prior_logit_abs_mean"],
                diag["ae_reliability_feature_logit_abs_mean"],
                diag["ae_interpretable_bias_abs_mean"],
                diag["ae_explicit_feature_logit_abs_mean"],
                diag["ae_interaction_logit_abs_mean"],
                diag["ae_raw_logit_before_clip_abs_mean"],
                diag["ae_raw_logit_after_clip_abs_mean"],
            )
            last_diag = dict(diag)

            if getattr(_get_base_model(model), "use_personal_graph", False):
                if diag["alpha_std"] < 1e-6:
                    alpha_zero_streak += 1
                else:
                    alpha_zero_streak = 0
                if alpha_zero_streak >= 3:
                    logger.warning(
                        "%s [Diag Warning] alpha_std has been near zero for %d epoch(s) "
                        "(alpha_std=%.6f, personal_row_entropy=%.4f). "
                        "This indicates personal-graph mixing may be collapsed/trivial.",
                        run_tag,
                        alpha_zero_streak,
                        diag["alpha_std"],
                        diag["personal_row_entropy"],
                    )
                warning_tags = set(_collect_diag_warning_tags(diag))
                if "personal-delta-student-flat" in warning_tags:
                    logger.warning(
                        "%s [Diag Warning][E] personal_delta_student_std is near zero: "
                        "personal_delta_student_std=%.6f, personal_delta_pre_softmax_norm=%.6f",
                        run_tag,
                        diag.get("personal_delta_student_std", 0.0),
                        diag.get("personal_delta_pre_softmax_norm", 0.0),
                    )
                if "alpha-head-collapse" in warning_tags:
                    logger.warning(
                        "%s [Diag Warning][E] alpha_head_std is near zero: "
                        "alpha_head_std=%.6f, alpha_std=%.6f",
                        run_tag,
                        diag.get("alpha_head_std", 0.0),
                        diag.get("alpha_std", 0.0),
                    )
                if "alpha-id-dominant" in warning_tags:
                    logger.warning(
                        "%s [Diag Warning][E] id path is dominating gate alpha: state_path=%.6f, id_path=%.6f",
                        run_tag,
                        diag.get("alpha_state_path_absmean", 0.0),
                        diag.get("alpha_id_path_absmean", 0.0),
                    )
                if "gate-saturated" in warning_tags:
                    logger.warning(
                        "%s [Diag Warning][E] gate-saturated: alpha_saturation_ratio=%.6f, alpha_std=%.6f, alpha_delta_absmean=%.6f",
                        run_tag,
                        diag.get("alpha_saturation_ratio", 0.0),
                        diag.get("alpha_std", 0.0),
                        diag.get("alpha_delta_absmean", 0.0),
                    )
                if "personalization-flat" in warning_tags:
                    logger.warning(
                        "%s [Diag Warning][E] personalization-flat: personal_matrix_delta=%.6f, personal_matrix_student_std=%.6f, personal_query_row_std=%.6f",
                        run_tag,
                        diag.get("personal_matrix_delta", 0.0),
                        diag.get("personal_matrix_student_std", 0.0),
                        diag.get("personal_query_row_std", 0.0),
                    )
                if "query-readout-overamplified" in warning_tags:
                    logger.warning(
                        "%s [Diag Warning][AE] query-readout-overamplified: readout_query_delta=%.6f, query_row_graph_delta=%.6f, personal_matrix_delta=%.6f",
                        run_tag,
                        diag.get("readout_query_delta", 0.0),
                        diag.get("query_row_graph_delta", 0.0),
                        diag.get("personal_matrix_delta", 0.0),
                    )
                if "E-active-but-overrides-global" in warning_tags:
                    logger.warning(
                        "%s [Diag Warning][E] E-active-but-overrides-global: personal_message=%.6f, global_readout=%.6f, ratio=%.6f",
                        run_tag,
                        diag.get("query_row_personal_message_delta", 0.0),
                        diag.get("query_row_global_readout_delta", 0.0),
                        diag.get("personal_to_graph_query_ratio", 0.0),
                    )
                if "E-flat-at-query" in warning_tags:
                    logger.warning(
                        "%s [Diag Warning][E] E-flat-at-query: personal_message=%.6f, posterior_delta=%.6f, posterior_kl=%.6f",
                        run_tag,
                        diag.get("query_row_personal_message_delta", 0.0),
                        diag.get("query_row_posterior_delta_abs", 0.0),
                        diag.get("query_row_posterior_kl", 0.0),
                    )
                if "A-readout-washed-out" in warning_tags:
                    logger.warning(
                        "%s [Diag Warning][A] A-readout-washed-out: knowledge_state_graph_delta=%.6f, query_row_global_readout_delta=%.6f",
                        run_tag,
                        diag.get("knowledge_state_graph_delta", 0.0),
                        diag.get("query_row_global_readout_delta", 0.0),
                    )
                if "query-posterior-too-diffuse" in warning_tags:
                    logger.warning(
                        "%s [Diag Warning][E] query-posterior-too-diffuse: posterior_kl=%.6f, personal_message=%.6f",
                        run_tag,
                        diag.get("query_row_posterior_kl", 0.0),
                        diag.get("query_row_personal_message_delta", 0.0),
                    )
            if diag["graph_entropy_ratio"] > 0.98:
                graph_uniform_streak += 1
            else:
                graph_uniform_streak = 0
            if graph_uniform_streak >= 3:
                logger.warning(
                    "%s [Diag Warning][Graph] graph entropy ratio has stayed too high for %d epoch(s): "
                    "entropy_ratio=%.4f, to_uniform_l2=%.6f",
                    run_tag,
                    graph_uniform_streak,
                    diag["graph_entropy_ratio"],
                    diag["graph_to_uniform_l2"],
                )

            graph_grad_total = (
                grad_norms["relation_emb"]
                + grad_norms["relation_tau"]
                + grad_norms["relation_prior_strength"]
                + grad_norms["relation_receiver_bias"]
                + grad_norms["relation_self_loop"]
                + grad_norms["personal_gate_head_bias"]
                + grad_norms["personal_gate_query_weight"]
                + grad_norms["personal_gate_context_weight"]
                + grad_norms["personal_generator_state_mix"]
            )
            if graph_grad_total < 1e-8:
                graph_low_grad_streak += 1
            else:
                graph_low_grad_streak = 0
            if graph_low_grad_streak >= 2 and getattr(_get_base_model(model), "use_concept_graph", False):
                logger.warning(
                        "%s [Diag Warning][Graph] graph-related grad norms are near zero for %d epoch(s): "
                        "relation_emb=%.6e, relation_tau=%.6e, prior_strength=%.6e, receiver_bias=%.6e, "
                        "self_loop=%.6e, gate_head_bias=%.6e, gate_query_weight=%.6e, "
                        "gate_context_weight=%.6e, generator_state_mix=%.6e",
                        run_tag,
                        graph_low_grad_streak,
                        grad_norms["relation_emb"],
                        grad_norms["relation_tau"],
                        grad_norms["relation_prior_strength"],
                        grad_norms["relation_receiver_bias"],
                        grad_norms["relation_self_loop"],
                        grad_norms["personal_gate_head_bias"],
                        grad_norms["personal_gate_query_weight"],
                        grad_norms["personal_gate_context_weight"],
                        grad_norms["personal_generator_state_mix"],
                    )
            logger.info(
                "%s [Grad Norms] Epoch [%03d] | "
                "relation_emb=%.6e, relation_tau=%.6e, prior_strength=%.6e, receiver_bias=%.6e, "
                "self_loop=%.6e, gate_head_bias=%.6e, gate_query_weight=%.6e, "
                "gate_context_weight=%.6e, generator_state_mix=%.6e",
                run_tag,
                epoch,
                grad_norms["relation_emb"],
                grad_norms["relation_tau"],
                grad_norms["relation_prior_strength"],
                grad_norms["relation_receiver_bias"],
                grad_norms["relation_self_loop"],
                grad_norms["personal_gate_head_bias"],
                grad_norms["personal_gate_query_weight"],
                grad_norms["personal_gate_context_weight"],
                grad_norms["personal_generator_state_mix"],
            )

        scheduler.step(val_metrics["auc"])

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = float(val_metrics["auc"])
            best_epoch = epoch
            patience_counter = 0

            model_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": _get_base_model(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc": best_val_auc,
                    "val_metrics": val_metrics,
                    "monitor": monitor_config,
                    "debug_diag": last_diag,
                    "args": vars(args),
                    "info_dict": info_dict,
                },
                model_path,
            )
            _write_diag_json(args.save_dir, "diag_best.json", last_diag)
            logger.info("%s -> New best AUC=%.4f at epoch %d", run_tag, best_val_auc, epoch)
        else:
            patience_counter += 1
            logger.info(
                "%s -> No improvement %d epoch(s) (best AUC=%.4f @ %d)",
                run_tag, patience_counter, best_val_auc, best_epoch
            )

        if patience_counter >= args.early_stop_patience:
            logger.info(
                "%s Early stopping at epoch %d (best AUC=%.4f @ %d)",
                run_tag, epoch, best_val_auc, best_epoch
            )
            break

        if epoch % args.save_interval == 0:
            checkpoint_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": _get_base_model(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "train_metrics": train_metrics,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            logger.info("%s Checkpoint saved: %s", run_tag, checkpoint_path)

            # 模块活跃度检测（每 save_interval 输出简报）
            try:
                activity = compute_module_activity(model, val_loader, device, num_samples=300)
                brief = format_activity_brief(activity)
                logger.info("%s [Module Activity] Epoch %d: %s", run_tag, epoch, brief)
            except Exception as e:
                logger.warning("%s [Module Activity] Failed: %s", run_tag, str(e))

    history["best_epoch"] = best_epoch
    history["best_val_auc"] = best_val_auc
    history["last_debug_diag"] = last_diag

    history_path = os.path.join(args.save_dir, "training_history.json")
    save_epoch_history_csv(history, args.save_dir, logger)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    logger.info("%s Training completed! Best Val AUC=%.4f @ epoch %d", run_tag, best_val_auc, best_epoch)

    # ========== 训练结束：输出完整的模块活跃度报告 ==========
    try:
        activity = compute_module_activity(model, val_loader, device, num_samples=500)
        report = format_activity_report(
            activity,
            dataset_name=getattr(args, 'dataset_name', 'unknown'),
            seed=getattr(args, 'seed', 42),
            epoch=epoch,
        )
        logger.info("\n%s", report)

        # 保存活跃度数据到 JSON
        activity_path = os.path.join(args.save_dir, "module_activity.json")
        with open(activity_path, "w") as f:
            json.dump(activity, f, indent=4)
        logger.info("%s Module activity saved to %s", run_tag, activity_path)
    except Exception as e:
        logger.warning("%s [Module Activity Report] Failed: %s", run_tag, str(e))

    return best_val_auc, best_epoch


def run_inference(args, logger) -> Tuple[Dict[str, float], Dict[str, Any]]:
    device = select_device(args, logger)

    data_dir = args.data_dir
    test_file = os.path.join(data_dir, "test.csv")

    model_path = os.path.join(args.save_dir, "best_model.pth")
    logger.info(f"Loading model from {model_path}...")

    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return {}, {}

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    loaded_args: Dict[str, Any] = checkpoint.get("args", {})
    info_dict = checkpoint.get("info_dict", None)

    if info_dict is None:
        logger.error("info_dict not found in checkpoint. Please retrain with current code.")
        return {}, {}

    stu_id_map = info_dict["stu_id_map"]
    exer_id_map = info_dict["exer_id_map"]
    cpt_id_map = info_dict["cpt_id_map"]
    q_matrix = info_dict["q_matrix"]

    raw_test_df = pd.read_csv(test_file)

    valid_stu_ids = set(stu_id_map.keys())
    valid_exer_ids = set(exer_id_map.keys())
    before_rows = len(raw_test_df)
    filtered_test_df = raw_test_df[
        raw_test_df["stu_id"].isin(valid_stu_ids) & raw_test_df["exer_id"].isin(valid_exer_ids)
    ].reset_index(drop=True)
    after_rows = len(filtered_test_df)
    dropped = before_rows - after_rows
    coverage = (after_rows / before_rows) if before_rows > 0 else 1.0

    logger.info(
        f"[Inference Filter] before={before_rows}, after={after_rows}, dropped={dropped}, "
        f"coverage={coverage:.2%} (train-only seen student/item support)"
    )

    test_dataset = CognitiveDiagnosisDataset(
        csv_file=filtered_test_df,
        stu_id_map=stu_id_map,
        exer_id_map=exer_id_map,
        cpt_id_map=cpt_id_map,
    )

    pin_memory = bool(getattr(device, "type", "cpu") == "cuda")
    test_loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
    }
    if int(args.num_workers) > 0:
        test_loader_kwargs["persistent_workers"] = True
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **test_loader_kwargs,
    )

    # Build model from loaded args (fallback to current args)
    use_concept_graph = loaded_args.get("use_concept_graph", getattr(args, "use_concept_graph", True))
    ablate_module1 = bool(loaded_args.get("ablate_module1", getattr(args, "ablate_module1", False)))
    expected_enable_module1 = not ablate_module1

    # hard-ablation safety at inference too
    eff_gnn_layers = _hard_ablation_effective_hparams(
        use_concept_graph=use_concept_graph,
        num_gnn_layers=int(loaded_args.get("num_gnn_layers", getattr(args, "num_gnn_layers", 0))),
    )

    logger.info(
        "Inference switches: "
        "ablate_module1=%s | enable_module1=%s | use_concept_graph=%s | "
        "effective(num_gnn_layers=%d)",
        ablate_module1,
        expected_enable_module1,
        use_concept_graph,
        eff_gnn_layers,
    )

    model = CognitiveDiagnosisModel(
        num_students=info_dict["num_students"],
        num_exercises=info_dict["num_exercises"],
        num_concepts=info_dict["num_concepts"],
        q_matrix=q_matrix,
        item_prior_matrix=info_dict.get("item_prior_matrix"),
        sequence_prior_matrix=info_dict.get("sequence_prior_matrix"),
        knowledge_dim=loaded_args.get("knowledge_dim", args.knowledge_dim),
        num_relation_heads=loaded_args.get("num_relation_heads", args.num_relation_heads),
        num_gnn_layers=eff_gnn_layers,
        dropout=loaded_args.get("dropout", args.dropout),
        use_concept_graph=use_concept_graph,
        graph_topk=loaded_args.get("graph_topk", getattr(args, "graph_topk", None)),
        allow_self_loop=not loaded_args.get("disable_self_loop", getattr(args, "disable_self_loop", False)),
        use_personal_graph=loaded_args.get("use_personal_graph", getattr(args, "use_personal_graph", False)),
        ablate_module1=ablate_module1,
        graph_dropout=_resolve_optional_graph_dropout(
            loaded_args.get("graph_dropout", getattr(args, "graph_dropout", -1.0))
        ),
        graph_tau_init=loaded_args.get("graph_tau_init", getattr(args, "graph_tau_init", 1.0)),
        personal_rank=loaded_args.get("personal_rank", getattr(args, "personal_rank", 4)),
        lambda_sparse_personal=loaded_args.get("lambda_sparse_personal", args.lambda_sparse_personal),
        lambda_alpha=loaded_args.get("lambda_alpha", args.lambda_alpha),
        lambda_graph_entropy=loaded_args.get("lambda_sparse", args.lambda_sparse),
        graph_entropy_min=loaded_args.get("graph_entropy_min", getattr(args, "graph_entropy_min", 0.15)),
        graph_entropy_max=loaded_args.get("graph_entropy_max", getattr(args, "graph_entropy_max", 0.85)),
        lambda_graph_diag=loaded_args.get("lambda_graph_diag", getattr(args, "lambda_graph_diag", 0.10)),
        lambda_graph_uniform=loaded_args.get(
            "lambda_graph_uniform", getattr(args, "lambda_graph_uniform", 0.04)
        ),
        graph_uniform_margin=loaded_args.get(
            "graph_uniform_margin", getattr(args, "graph_uniform_margin", 0.10)
        ),
        graph_reg_warmup_epochs=loaded_args.get(
            "graph_reg_warmup_epochs", getattr(args, "graph_reg_warmup_epochs", 1)
        ),
        graph_reg_cap_ratio=loaded_args.get("graph_reg_cap_ratio", getattr(args, "graph_reg_cap_ratio", 6.0)),
        graph_propagation_alpha=loaded_args.get(
            "graph_propagation_alpha", getattr(args, "graph_propagation_alpha", 0.20)
        ),
        graph_query_readout_scale=loaded_args.get(
            "graph_query_readout_scale", getattr(args, "graph_query_readout_scale", 0.35)
        ),
        graph_query_readout_2hop_scale=loaded_args.get(
            "graph_query_readout_2hop_scale", getattr(args, "graph_query_readout_2hop_scale", 0.15)
        ),
        prediction_l2_lambda=loaded_args.get(
            "prediction_l2_lambda",
            getattr(args, "prediction_l2_lambda", 5e-5),
        ),
        gnn_residual_weight=loaded_args.get("gnn_residual_weight", getattr(args, "gnn_residual_weight", 0.5)),
        graph_identity_residual=loaded_args.get(
            "graph_identity_residual", getattr(args, "graph_identity_residual", 0.0)
        ),
        personal_max_alpha=loaded_args.get("personal_max_alpha", getattr(args, "personal_max_alpha", 0.35)),
        personal_delta_scale=loaded_args.get(
            "personal_delta_scale", getattr(args, "personal_delta_scale", 1.0)
        ),
        personal_warmup_epochs=loaded_args.get(
            "personal_warmup_epochs", getattr(args, "personal_warmup_epochs", 0)
        ),
        personal_reg_warmup_epochs=loaded_args.get(
            "personal_reg_warmup_epochs", getattr(args, "personal_reg_warmup_epochs", None)
        ),
        personal_student_dim=loaded_args.get(
            "personal_student_dim", getattr(args, "personal_student_dim", args.knowledge_dim)
        ),
        lambda_alpha_min=loaded_args.get("lambda_alpha_min", getattr(args, "lambda_alpha_min", 0.0)),
        alpha_min_target=loaded_args.get("alpha_min_target", getattr(args, "alpha_min_target", 0.0)),
        personal_alpha_temperature=loaded_args.get(
            "personal_alpha_temperature", getattr(args, "personal_alpha_temperature", 2.0)
        ),
        personal_alpha_budget=loaded_args.get(
            "personal_alpha_budget", getattr(args, "personal_alpha_budget", 0.10)
        ),
        personal_alpha_base_init=loaded_args.get(
            "personal_alpha_base_init", getattr(args, "personal_alpha_base_init", 0.08)
        ),
        personal_alpha_bias_scale=loaded_args.get(
            "personal_alpha_bias_scale", getattr(args, "personal_alpha_bias_scale", 1.0)
        ),
        personal_freeze_alpha_gate=loaded_args.get(
            "personal_freeze_alpha_gate", getattr(args, "personal_freeze_alpha_gate", False)
        ),
        personal_disable_student_global_context=loaded_args.get(
            "personal_disable_student_global_context",
            getattr(args, "personal_disable_student_global_context", False),
        ),
        personal_local_hops=loaded_args.get(
            "personal_local_hops", getattr(args, "personal_local_hops", 1)
        ),
        personal_include_neighbor_rows=loaded_args.get(
            "personal_include_neighbor_rows", getattr(args, "personal_include_neighbor_rows", False)
        ),
        personal_support_include_query_self=loaded_args.get(
            "personal_support_include_query_self", getattr(args, "personal_support_include_query_self", True)
        ),
        personal_support_include_graph=loaded_args.get(
            "personal_support_include_graph", getattr(args, "personal_support_include_graph", True)
        ),
        personal_support_include_neighbors=loaded_args.get(
            "personal_support_include_neighbors", getattr(args, "personal_support_include_neighbors", False)
        ),
        personal_item_support_mass=loaded_args.get(
            "personal_item_support_mass", getattr(args, "personal_item_support_mass", 0.0)
        ),
        personal_mastery_prior_scale=loaded_args.get(
            "personal_mastery_prior_scale", getattr(args, "personal_mastery_prior_scale", 0.0)
        ),
        personal_recent_mastery_prior_scale=loaded_args.get(
            "personal_recent_mastery_prior_scale",
            getattr(args, "personal_recent_mastery_prior_scale", 0.0),
        ),
        personal_mastery_count_smoothing=loaded_args.get(
            "personal_mastery_count_smoothing",
            getattr(args, "personal_mastery_count_smoothing", 0.0),
        ),
        personal_query_row_budget=loaded_args.get(
            "personal_query_row_budget", getattr(args, "personal_query_row_budget", 1.0)
        ),
        personal_neighbor_row_budget=loaded_args.get(
            "personal_neighbor_row_budget", getattr(args, "personal_neighbor_row_budget", 0.30)
        ),
        personal_query_support_hops=loaded_args.get(
            "personal_query_support_hops", getattr(args, "personal_query_support_hops", 0)
        ),
        personal_support_only=loaded_args.get(
            "personal_support_only", getattr(args, "personal_support_only", True)
        ),
        personal_query_correction_scale=loaded_args.get(
            "personal_query_correction_scale", getattr(args, "personal_query_correction_scale", 0.15)
        ),
        personal_query_correction_max_ratio=loaded_args.get(
            "personal_query_correction_max_ratio",
            getattr(args, "personal_query_correction_max_ratio", 0.20),
        ),
        personal_query_correction_min_graph_anchor=loaded_args.get(
            "personal_query_correction_min_graph_anchor",
            getattr(args, "personal_query_correction_min_graph_anchor", 0.01),
        ),
        personal_query_message_gain=loaded_args.get(
            "personal_query_message_gain", getattr(args, "personal_query_message_gain", 1.0)
        ),
        personal_value_use_global_basis=loaded_args.get(
            "personal_value_use_global_basis", getattr(args, "personal_value_use_global_basis", True)
        ),
        personal_message_alignment_gate=loaded_args.get(
            "personal_message_alignment_gate", getattr(args, "personal_message_alignment_gate", True)
        ),
        personal_projection_hidden_factor=loaded_args.get(
            "personal_projection_hidden_factor", getattr(args, "personal_projection_hidden_factor", 2)
        ),
        lambda_personal_kl=loaded_args.get(
            "lambda_personal_kl", getattr(args, "lambda_personal_kl", 0.0)
        ),
        lambda_personal_query_residual=loaded_args.get(
            "lambda_personal_query_residual", getattr(args, "lambda_personal_query_residual", 0.0)
        ),
        personal_query_residual_margin=loaded_args.get(
            "personal_query_residual_margin", getattr(args, "personal_query_residual_margin", 0.0)
        ),
        graph_headwise_query_gate=loaded_args.get(
            "graph_headwise_query_gate", getattr(args, "graph_headwise_query_gate", True)
        ),
        graph_edge_bias_rank=loaded_args.get(
            "graph_edge_bias_rank", getattr(args, "graph_edge_bias_rank", 8)
        ),
        graph_prior_logit_scale=loaded_args.get(
            "graph_prior_logit_scale", getattr(args, "graph_prior_logit_scale", 0.0)
        ),
        ae_query_residual_scale=loaded_args.get(
            "ae_query_residual_scale", getattr(args, "ae_query_residual_scale", 0.0)
        ),
        ae_logit_residual_scale=loaded_args.get(
            "ae_logit_residual_scale", getattr(args, "ae_logit_residual_scale", 0.0)
        ),
        ae_logit_residual_clip=loaded_args.get(
            "ae_logit_residual_clip", getattr(args, "ae_logit_residual_clip", 1.0)
        ),
        ae_irt_logit_scale=loaded_args.get(
            "ae_irt_logit_scale", getattr(args, "ae_irt_logit_scale", 1.0)
        ),
        ae_interaction_logit_scale=loaded_args.get(
            "ae_interaction_logit_scale", getattr(args, "ae_interaction_logit_scale", 0.0)
        ),
        ae_logit_dim=loaded_args.get(
            "ae_logit_dim", getattr(args, "ae_logit_dim", 32)
        ),
        ae_posterior_prior_scale=loaded_args.get(
            "ae_posterior_prior_scale", getattr(args, "ae_posterior_prior_scale", 0.0)
        ),
        lambda_theta_prior_align=loaded_args.get(
            "lambda_theta_prior_align", getattr(args, "lambda_theta_prior_align", 0.0)
        ),
        relation_theta_scale=loaded_args.get(
            "relation_theta_scale", getattr(args, "relation_theta_scale", 0.0)
        ),
        relation_theta_delta_clip=loaded_args.get(
            "relation_theta_delta_clip", getattr(args, "relation_theta_delta_clip", 2.0)
        ),
        graph_query_adapter_enable=loaded_args.get(
            "graph_query_adapter_enable", getattr(args, "graph_query_adapter_enable", True)
        ),
        share_concept_embeddings=loaded_args.get(
            "share_concept_embeddings", getattr(args, "share_concept_embeddings", False)
        ),
    ).to(device)

    runtime_facts = _log_and_assert_ablation_consistency(
        model=model,
        logger=logger,
        context="[Inference]",
        ablate_module1=ablate_module1,
        expect_use_concept_graph=use_concept_graph,
        expect_use_personal_graph=loaded_args.get(
            "use_personal_graph", getattr(args, "use_personal_graph", False)
        ),
    )
    _log_graph_init_state(model, logger=logger, context="[Inference]")

    state_dict = checkpoint["model_state_dict"]
    state_dict = _strip_module_prefix(state_dict)  # 处理 DataParallel 的 module. 前缀

    incompatible = model.load_state_dict(state_dict, strict=False)
    _get_base_model(model).set_epoch(int(checkpoint.get("epoch", loaded_args.get("epochs", 1))))
    missing_keys = list(getattr(incompatible, "missing_keys", []))
    unexpected_keys = list(getattr(incompatible, "unexpected_keys", []))

    if missing_keys or unexpected_keys:
        logger.warning("State dict mismatch detected.")
        logger.warning("  Missing keys (%d): %s", len(missing_keys), missing_keys[:50])
        logger.warning("  Unexpected keys (%d): %s", len(unexpected_keys), unexpected_keys[:50])

        if STRICT_CHECKPOINT_LOADING:
            raise RuntimeError(
                f"Checkpoint/model architecture mismatch. missing={len(missing_keys)}, unexpected={len(unexpected_keys)}"
            )

    logger.info(f"Model loaded from epoch {checkpoint['epoch']}. Start testing...")

    model.eval()
    all_labels: List[float] = []
    all_preds: List[float] = []
    all_probs: List[float] = []
    max_test_batches = getattr(args, "max_test_batches", None)
    max_test_batches = None if max_test_batches is None else max(1, int(max_test_batches))

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if max_test_batches is not None and batch_idx >= max_test_batches:
                break
            student_ids, exercise_ids, labels = batch
            student_ids = student_ids.to(device)
            exercise_ids = exercise_ids.to(device)
            labels = _ensure_1d(labels.to(device).float())

            logits = model(
                student_ids,
                exercise_ids,
                return_details=False,
                return_logits=True,
            )
            logits = _ensure_1d(logits)
            probs = _sigmoid_torch(logits)
            preds = (probs > 0.5).float()

            all_labels.extend(labels.detach().cpu().numpy().tolist())
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_probs.extend(probs.detach().cpu().numpy().tolist())

            if (batch_idx + 1) % 200 == 0:
                logger.info(f"[Test] {batch_idx + 1}/{len(test_loader)} batches done, samples={len(all_labels)}")

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    logger.info("\n" + "=" * 50)
    logger.info("Test Results:")
    logger.info(f"AUC: {metrics['auc']:.4f}")
    logger.info(f"ACC: {metrics['acc']:.4f}")
    logger.info(f"RMSE: {metrics['rmse']:.4f}")
    logger.info("=" * 50)

    results = {
        "metrics": metrics,
        "num_samples": len(all_labels),
        "model_epoch": int(checkpoint["epoch"]),
        "best_val_auc": float(checkpoint.get("val_auc", 0.0)),
        "test_total_rows": int(before_rows),
        "test_seen_rows": int(after_rows),
        "test_seen_coverage": float(coverage),
        "effective_batch_size": int(getattr(args, "batch_size", 0)),
        "train_only_split_hygiene": bool(info_dict.get("train_only_split_hygiene", False)),
        "runtime_facts": runtime_facts,
        "monitor": checkpoint.get("monitor", _default_monitor_config()),
        "config_switches": _collect_structural_switches(loaded_args),
        "ae_diagnostics": checkpoint.get("debug_diag", {}),
        "failure_reason": None,
    }

    result_path = os.path.join(args.save_dir, "test_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=4)

    append_summary_csv(
        args,
        metrics=metrics,
        best_val_auc=results["best_val_auc"],
        model_epoch=results["model_epoch"],
        final_model_facts=runtime_facts,
        logger=logger,
    )

    # diagnosis
    if getattr(args, "generate_diagnosis", False):
        logger.info("\nGenerating student diagnosis reports...")
        num_students_to_diagnose = min(5, info_dict["num_students"])
        diagnosis_results = []

        for stu_id in range(num_students_to_diagnose):
            diagnosis = model.get_student_diagnosis(stu_id)
            diagnosis_results.append(
                {
                    "student_id": int(stu_id),
                    "original_student_id": int(info_dict["stu_id_reverse_map"].get(stu_id, stu_id)),
                    "knowledge_mastery": [float(x) for x in diagnosis["knowledge_mastery"].cpu().numpy().tolist()],
                    "student_repr": [float(x) for x in diagnosis["student_repr"].cpu().numpy().tolist()],
                }
            )

        diagnosis_path = os.path.join(args.save_dir, "student_diagnosis.json")
        with open(diagnosis_path, "w") as f:
            json.dump(diagnosis_results, f, indent=4)

        logger.info(f"Student diagnosis reports saved to {diagnosis_path}")

    return metrics, results


def _materialize_personal_dense_preview(details: Dict[str, Any]) -> Optional[np.ndarray]:
    personal = details.get("personal_matrices")
    if personal is not None:
        return personal.detach().cpu().numpy()

    relation_used = details.get("relation_used")
    if not isinstance(relation_used, dict):
        return None

    global_A = relation_used["global_matrices"].detach()
    active_row_index = relation_used["active_row_index"].detach()
    active_row_valid_mask = relation_used["active_row_valid_mask"].detach().bool()
    support_col_index = relation_used["support_col_index"].detach()
    support_valid_mask = relation_used["support_valid_mask"].detach().bool()
    posterior_prob = relation_used["posterior_prob"].detach()

    B = active_row_index.size(0)
    H, C, _ = global_A.shape
    dense = global_A.unsqueeze(0).expand(B, -1, -1, -1).clone()
    for b in range(B):
        for r in range(active_row_index.size(1)):
            if not bool(active_row_valid_mask[b, r]):
                continue
            row = int(active_row_index[b, r].item())
            for h in range(H):
                dense[b, h, row] = 0.0
                valid = support_valid_mask[b, h, r]
                cols = support_col_index[b, h, r][valid]
                probs = posterior_prob[b, h, r][valid]
                dense[b, h, row, cols] = probs
    return dense.cpu().numpy()


def _append_sample_slices(
    storage: Dict[str, List[np.ndarray]],
    key: str,
    tensor: Optional[torch.Tensor],
    take: int,
) -> None:
    if tensor is None or take <= 0:
        return
    array = tensor[:take].detach().cpu().numpy()
    for sample in array:
        storage.setdefault(key, []).append(np.array(sample, copy=True))


def _finalize_sample_list(
    storage: Dict[str, Any],
    key: str,
    *,
    limit: int,
    ragged: bool,
) -> None:
    values = storage.get(key)
    if not values:
        return
    clipped = values[:limit]
    if ragged:
        obj = np.empty((len(clipped),), dtype=object)
        for idx, value in enumerate(clipped):
            obj[idx] = value
        storage[key] = obj
    else:
        storage[key] = np.stack(clipped, axis=0)


def _append_scalar_or_vector_samples(
    collector: List[np.ndarray],
    value: torch.Tensor,
    take: int,
) -> None:
    if value is None or take <= 0:
        return
    det = value.detach().cpu()
    if det.ndim == 0:
        collector.append(np.repeat(np.asarray([float(det.item())], dtype=np.float32), take))
        return
    arr = det.numpy()
    if arr.shape[0] >= take:
        collector.append(arr[:take].copy())
    else:
        collector.append(np.repeat(arr.reshape(1, -1), take, axis=0).reshape(take, -1).squeeze(-1))


def save_component_analysis_data(
    model: CognitiveDiagnosisModel,
    train_loader: DataLoader,
    device: torch.device,
    save_dir: str,
    logger,
    num_samples: int = 100,
    materialize_dense_preview: bool = False,
) -> Dict[str, Any]:
    """
    保存组件可视化分析所需的数据：
    1. 全局概念图 relation_matrices
    2. 真实推理路径上的 sparse personal spec 采样
    3. 与题目相关的 q_vector / local_row_mask / exercise_ids
    4. 仅在显式 materialize_dense_preview=True 时才额外物化 dense preview
    """
    base_model = _get_base_model(model)
    model.eval()
    analysis_data = {}
    
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        # ========== 1) 全局概念图分析 ==========
        if base_model.structure_module.relation_learning is not None:
            relation_matrices, _ = base_model.structure_module.relation_learning()
            analysis_data["global_relation_matrices"] = relation_matrices.detach().cpu().numpy()
            logger.info(f"[Component Analysis] Global graph: {relation_matrices.shape}")
        
        # ========== 2) 真实推理路径上的个性化图分析 ==========
        if base_model.use_personal_graph and base_model.structure_module.personal_generator is not None:
            gate_alphas = []
            personal_graphs = []
            relation_used_samples = []
            local_row_masks = []
            q_vectors = []
            exercise_id_samples = []
            query_row_global_readout_deltas = []
            query_row_personal_message_deltas = []
            query_row_posterior_delta_abs_vals = []
            query_row_posterior_kl_vals = []
            sample_count = 0
            
            for batch in train_loader:
                if sample_count >= num_samples:
                    break
                student_ids, exercise_ids, _ = batch
                student_ids = student_ids.to(device)
                exercise_ids = exercise_ids.to(device)

                _, details = model(
                    student_ids,
                    exercise_ids,
                    return_details=True,
                    return_logits=True,
                )

                remaining = max(0, int(num_samples) - sample_count)
                if remaining <= 0:
                    break
                take = min(int(student_ids.size(0)), remaining)

                alpha = details.get("alpha_effective", details.get("alpha"))
                if alpha is not None:
                    gate_alphas.append(alpha[:take].detach().cpu().numpy())
                if materialize_dense_preview:
                    dense_preview = _materialize_personal_dense_preview(details)
                    if dense_preview is not None:
                        personal_graphs.append(dense_preview[:take])
                if details.get("relation_used") is not None and materialize_dense_preview:
                    relation_used = details["relation_used"]
                    if isinstance(relation_used, dict):
                        dense_relation = _materialize_personal_dense_preview(details)
                        if dense_relation is not None:
                            relation_used_samples.append(dense_relation[:take])
                    else:
                        relation_used_samples.append(relation_used[:take].detach().cpu().numpy())
                if details.get("local_row_mask") is not None:
                    local_row_masks.append(details["local_row_mask"][:take].detach().cpu().numpy())
                if details.get("q_vector") is not None:
                    q_vectors.append(details["q_vector"][:take].detach().cpu().numpy())
                exercise_id_samples.append(exercise_ids[:take].detach().cpu().numpy())
                for detail_key, collector in (
                    ("query_row_global_readout_delta", query_row_global_readout_deltas),
                    ("query_row_personal_message_delta", query_row_personal_message_deltas),
                    ("query_row_posterior_delta_abs", query_row_posterior_delta_abs_vals),
                    ("query_row_posterior_kl", query_row_posterior_kl_vals),
                ):
                    value = details.get(detail_key)
                    if value is not None:
                        _append_scalar_or_vector_samples(collector, value, take)
                if isinstance(details.get("relation_used"), dict):
                    spec = details["relation_used"]
                    _append_sample_slices(analysis_data, "active_row_index_samples", spec.get("active_row_index"), take)
                    _append_sample_slices(analysis_data, "active_row_valid_mask_samples", spec.get("active_row_valid_mask"), take)
                    _append_sample_slices(analysis_data, "support_col_index_samples", spec.get("support_col_index"), take)
                    _append_sample_slices(analysis_data, "support_valid_mask_samples", spec.get("support_valid_mask"), take)
                    _append_sample_slices(analysis_data, "posterior_prob_samples", spec.get("posterior_prob"), take)
                    _append_sample_slices(analysis_data, "gate_alpha_samples", spec.get("gate_alpha"), take)

                sample_count += take
            
            if gate_alphas:
                analysis_data["gate_alpha"] = np.concatenate([g.flatten() for g in gate_alphas])[:num_samples]
                logger.info(f"[Component Analysis] Gate alpha samples: {len(analysis_data['gate_alpha'])}")
            
            if materialize_dense_preview and personal_graphs:
                personal_arr = np.concatenate(personal_graphs, axis=0)[:min(10, num_samples)]
                analysis_data["personal_matrices_samples"] = personal_arr
                logger.info(f"[Component Analysis] Personal graph samples: {personal_arr.shape}")
            if materialize_dense_preview and relation_used_samples:
                relation_arr = np.concatenate(relation_used_samples, axis=0)[:min(10, num_samples)]
                analysis_data["relation_used_samples"] = relation_arr
                logger.info(f"[Component Analysis] relation_used samples: {relation_arr.shape}")
            if local_row_masks:
                local_mask_arr = np.concatenate(local_row_masks, axis=0)[:num_samples]
                analysis_data["local_row_mask_samples"] = local_mask_arr
            if q_vectors:
                q_arr = np.concatenate(q_vectors, axis=0)[:num_samples]
                analysis_data["q_vector_samples"] = q_arr
            if exercise_id_samples:
                exercise_arr = np.concatenate(exercise_id_samples, axis=0)[:num_samples]
                analysis_data["exercise_ids_samples"] = exercise_arr
            if query_row_global_readout_deltas:
                analysis_data["query_row_global_readout_delta_samples"] = np.concatenate(
                    query_row_global_readout_deltas, axis=0
                )[:num_samples]
            if query_row_personal_message_deltas:
                analysis_data["query_row_personal_message_delta_samples"] = np.concatenate(
                    query_row_personal_message_deltas, axis=0
                )[:num_samples]
            if query_row_posterior_delta_abs_vals:
                analysis_data["query_row_posterior_delta_abs_samples"] = np.concatenate(
                    query_row_posterior_delta_abs_vals, axis=0
                )[:num_samples]
            if query_row_posterior_kl_vals:
                analysis_data["query_row_posterior_kl_samples"] = np.concatenate(
                    query_row_posterior_kl_vals, axis=0
                )[:num_samples]
            for key in (
                "active_row_index_samples",
                "active_row_valid_mask_samples",
                "support_col_index_samples",
                "support_valid_mask_samples",
                "posterior_prob_samples",
                "gate_alpha_samples",
            ):
                _finalize_sample_list(
                    analysis_data,
                    key,
                    limit=min(10, num_samples),
                    ragged=(key != "gate_alpha_samples"),
                )

    # ========== 保存数据 ==========
    analysis_path = os.path.join(save_dir, "component_analysis_data.npz")
    np.savez_compressed(analysis_path, **analysis_data)
    logger.info(f"[Component Analysis] Data saved to {analysis_path}")
    
    return analysis_data
