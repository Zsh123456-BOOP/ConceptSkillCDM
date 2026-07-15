"""Training and inference for the production Graph-IRT model."""

import hashlib
import json
import math
import os
import warnings
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config import PAIRWISE_AUC_WEIGHT
from src.dataset import CognitiveDiagnosisDataset, create_dataloaders
from src.experiment_utils import (
    _config_hash,
    append_summary_csv,
    compute_metrics,
    save_epoch_history_csv,
    select_device,
)
from src.model import CognitiveDiagnosisModel, GRAPH_IRT_ARCHITECTURE
from src.module_activity import (
    compute_module_activity,
    format_activity_brief,
    format_activity_report,
)


warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0")

ARCHITECTURE_NAME = GRAPH_IRT_ARCHITECTURE
STRICT_CHECKPOINT_LOADING = True
MONITOR_NAME = "val_auc"
MONITOR_MODE = "max"

STRUCTURAL_SWITCH_KEYS: Tuple[str, ...] = (
    "architecture",
    "graph_prior_mode",
    "graph_topk",
    "disable_self_loop",
    "allow_self_loop",
    "graph_identity_residual",
    "graph_propagation_alpha",
    "graph_prior_strength_init",
    "pairwise_auc_weight",
    "num_gnn_layers",
)

_REG_COMPONENT_KEYS: Tuple[str, ...] = (
    "graph_entropy",
    "graph_diag",
    "graph_uniform",
    "graph_reg_scale",
    "prediction_l2",
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


def _get_base_model(model: nn.Module) -> CognitiveDiagnosisModel:
    return model.module if isinstance(model, nn.DataParallel) else model


def _ensure_1d(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(-1)


def _source_get(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_data_identity(args: Any, train_file: str, val_file: str) -> Dict[str, Any]:
    """Bind a checkpoint to its train/validation corpus without opening test."""
    return {
        "schema": "graph_irt_data_v1",
        "dataset_name": str(getattr(args, "dataset_name", "unknown")),
        "data_dir": os.path.realpath(str(args.data_dir)),
        "train_sha256": _sha256_file(train_file),
        "valid_sha256": _sha256_file(val_file),
    }


def _resolve_optional_graph_dropout(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if parsed < 0.0 else parsed


def _default_monitor_config() -> Dict[str, str]:
    return {
        "scheduler_monitor": MONITOR_NAME,
        "scheduler_mode": MONITOR_MODE,
        "best_monitor": MONITOR_NAME,
        "best_mode": MONITOR_MODE,
        "early_stop_monitor": MONITOR_NAME,
        "early_stop_mode": MONITOR_MODE,
    }


def _resolve_pairwise_auc_weight(source: Any) -> float:
    """Return the fixed objective weight implied by the named ablation."""
    variant = str(_source_get(source, "model_variant", "full"))
    expected = 0.0 if variant == "no_pairwise_loss" else PAIRWISE_AUC_WEIGHT
    supplied = float(_source_get(source, "pairwise_auc_weight", expected))
    if supplied != expected:
        raise ValueError(
            "pairwise_auc_weight is fixed by model_variant: "
            f"variant={variant!r} requires {expected}, got {supplied}"
        )
    return expected


def _collect_structural_switches(source: Any) -> Dict[str, Any]:
    switches = {key: _source_get(source, key) for key in STRUCTURAL_SWITCH_KEYS}
    switches["pairwise_auc_weight"] = _resolve_pairwise_auc_weight(source)
    switches["architecture"] = ARCHITECTURE_NAME
    return switches


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        key[len("module.") :] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def _resolve_allow_self_loop(source: Any) -> bool:
    explicit = _source_get(source, "allow_self_loop", None)
    if explicit is not None:
        return bool(explicit)
    return not bool(_source_get(source, "disable_self_loop", False))


def _model_kwargs(source: Any, info_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "num_students": int(info_dict["num_students"]),
        "num_exercises": int(info_dict["num_exercises"]),
        "num_concepts": int(info_dict["num_concepts"]),
        "q_matrix": info_dict["q_matrix"],
        "item_prior_matrix": info_dict.get("item_prior_matrix"),
        "exposure_prior_matrix": info_dict.get("exposure_prior_matrix"),
        "knowledge_dim": int(_source_get(source, "knowledge_dim", 32)),
        "num_relation_heads": int(_source_get(source, "num_relation_heads", 4)),
        "num_gnn_layers": max(0, int(_source_get(source, "num_gnn_layers", 0))),
        "dropout": float(_source_get(source, "dropout", 0.3)),
        "graph_topk": _source_get(source, "graph_topk", None),
        "allow_self_loop": _resolve_allow_self_loop(source),
        "graph_identity_residual": float(_source_get(source, "graph_identity_residual", 0.0)),
        "lambda_graph_entropy": float(_source_get(source, "lambda_graph_entropy", 0.01)),
        "graph_entropy_min": float(_source_get(source, "graph_entropy_min", 0.15)),
        "graph_entropy_max": float(_source_get(source, "graph_entropy_max", 0.85)),
        "lambda_graph_diag": float(_source_get(source, "lambda_graph_diag", 0.10)),
        "lambda_graph_uniform": float(_source_get(source, "lambda_graph_uniform", 0.04)),
        "graph_uniform_margin": float(_source_get(source, "graph_uniform_margin", 0.10)),
        "graph_reg_warmup_epochs": int(_source_get(source, "graph_reg_warmup_epochs", 1)),
        "graph_reg_cap_ratio": float(_source_get(source, "graph_reg_cap_ratio", 6.0)),
        "graph_dropout": _resolve_optional_graph_dropout(_source_get(source, "graph_dropout", -1.0)),
        "graph_tau_init": float(_source_get(source, "graph_tau_init", 1.0)),
        "graph_propagation_alpha": float(_source_get(source, "graph_propagation_alpha", 0.20)),
        "prediction_l2_lambda": float(_source_get(source, "prediction_l2_lambda", 5e-5)),
        "gnn_residual_weight": float(_source_get(source, "gnn_residual_weight", 0.5)),
        "graph_prior_strength_init": float(_source_get(source, "graph_prior_strength_init", 1.0)),
    }


def _build_model(source: Any, info_dict: Dict[str, Any], device: torch.device) -> CognitiveDiagnosisModel:
    return CognitiveDiagnosisModel(**_model_kwargs(source, info_dict)).to(device)


def _runtime_facts(model: nn.Module) -> Dict[str, Any]:
    base_model = _get_base_model(model)
    relation_learning = getattr(base_model, "relation_learning", None)
    return {
        "architecture": ARCHITECTURE_NAME,
        "graph_enabled": bool(relation_learning is not None),
        "num_parameters": int(sum(parameter.numel() for parameter in base_model.parameters())),
        "num_trainable_parameters": int(
            sum(parameter.numel() for parameter in base_model.parameters() if parameter.requires_grad)
        ),
    }


def _log_runtime_facts(model: nn.Module, logger, context: str) -> Dict[str, Any]:
    facts = _runtime_facts(model)
    if not facts["graph_enabled"]:
        raise RuntimeError("Graph-IRT requires relation_learning to be present.")
    logger.info(
        "%s Architecture=%s | concept_graph=%s | params=%s trainable=%s",
        context,
        facts["architecture"],
        facts["graph_enabled"],
        f"{facts['num_parameters']:,}",
        f"{facts['num_trainable_parameters']:,}",
    )
    return facts


def _checkpoint_args(args: Any) -> Dict[str, Any]:
    """Store only general run metadata and Graph-IRT configuration."""
    checkpoint_keys = (
        "dataset",
        "dataset_name",
        "data_dir",
        "model_variant",
        "seed",
        "batch_size",
        "epochs",
        "learning_rate",
        "weight_decay",
        "optimizer",
        "patience",
        "early_stop_patience",
        "min_epochs",
        "early_stop_min_delta",
        "dropout",
        "knowledge_dim",
        "num_relation_heads",
        "num_gnn_layers",
        "graph_topk",
        "disable_self_loop",
        "allow_self_loop",
        "graph_identity_residual",
        "lambda_graph_entropy",
        "graph_entropy_min",
        "graph_entropy_max",
        "lambda_graph_diag",
        "lambda_graph_uniform",
        "graph_uniform_margin",
        "graph_reg_warmup_epochs",
        "graph_reg_cap_ratio",
        "graph_dropout",
        "graph_tau_init",
        "graph_propagation_alpha",
        "prediction_l2_lambda",
        "gnn_residual_weight",
        "graph_prior_strength_init",
        "pairwise_auc_weight",
        "graph_prior_mode",
        "min_stu_interactions",
        "min_exer_interactions",
    )
    clean = {
        key: getattr(args, key)
        for key in checkpoint_keys
        if hasattr(args, key)
    }
    if "data_dir" in clean:
        clean["data_dir"] = os.path.realpath(str(clean["data_dir"]))
    clean["architecture"] = ARCHITECTURE_NAME
    clean["allow_self_loop"] = _resolve_allow_self_loop(args)
    clean["pairwise_auc_weight"] = _resolve_pairwise_auc_weight(args)
    return clean


def _sanitize_nonfinite_grads(
    named_params: List[Tuple[str, torch.nn.Parameter]],
) -> Tuple[int, List[str]]:
    count = 0
    affected_params: List[str] = []
    for name, parameter in named_params:
        grad = parameter.grad
        if grad is None:
            continue
        bad = ~torch.isfinite(grad)
        bad_count = int(bad.sum().item())
        if bad_count:
            grad.masked_fill_(bad, 0.0)
            count += bad_count
            affected_params.append(name)
    return count, sorted(set(affected_params))


def _clip_stability_sensitive_grads(model: nn.Module) -> Dict[str, Any]:
    base_model = _get_base_model(model)
    relation_learning = getattr(base_model, "relation_learning", None)

    graph_ids = {id(parameter) for parameter in relation_learning.parameters()} if relation_learning is not None else set()
    graph_named = [
        (name, parameter)
        for name, parameter in base_model.named_parameters()
        if id(parameter) in graph_ids and parameter.requires_grad and parameter.grad is not None
    ]
    other_named = [
        (name, parameter)
        for name, parameter in base_model.named_parameters()
        if id(parameter) not in graph_ids and parameter.requires_grad and parameter.grad is not None
    ]
    nonfinite_graph, graph_names = _sanitize_nonfinite_grads(graph_named)
    nonfinite_other, other_names = _sanitize_nonfinite_grads(other_named)

    graph_params = [parameter for _, parameter in graph_named]
    other_params = [parameter for _, parameter in other_named]
    graph_norm = torch.nn.utils.clip_grad_norm_(graph_params, 1.5) if graph_params else 0.0
    other_norm = torch.nn.utils.clip_grad_norm_(other_params, 5.0) if other_params else 0.0
    return {
        "graph_clip_norm": float(graph_norm.item() if isinstance(graph_norm, torch.Tensor) else graph_norm),
        "other_clip_norm": float(other_norm.item() if isinstance(other_norm, torch.Tensor) else other_norm),
        "nonfinite_grad_count": float(nonfinite_graph + nonfinite_other),
        "nonfinite_graph_grad_count": float(nonfinite_graph),
        "nonfinite_other_grad_count": float(nonfinite_other),
        "nonfinite_grad_param_names": ",".join(sorted(set(graph_names + other_names))),
    }


def _build_optimizer(model: nn.Module, args: Any) -> optim.Optimizer:
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer_name = str(_source_get(args, "optimizer", "adam")).strip().lower()
    optimizer_class = {"adam": optim.Adam, "adamw": optim.AdamW}.get(optimizer_name)
    if optimizer_class is None:
        raise ValueError(f"optimizer must be 'adam' or 'adamw', got {optimizer_name!r}")
    return optimizer_class(
        [{"params": params, "lr": float(args.learning_rate), "group_name": "graph_irt"}],
        weight_decay=float(args.weight_decay),
    )


def _is_validation_improvement(current: float, best: float, min_delta: float) -> bool:
    return float(current) > float(best) + max(0.0, float(min_delta))


def _should_early_stop(epoch: int, patience_counter: int, args: Any) -> bool:
    min_epochs = max(0, int(_source_get(args, "min_epochs", 0)))
    patience = max(1, int(_source_get(args, "early_stop_patience", 1)))
    return int(epoch) >= min_epochs and int(patience_counter) >= patience


def _prediction_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pairwise_auc_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Return the fixed BCE/pairwise objective and its auditable components."""
    logits = _ensure_1d(logits)
    labels = _ensure_1d(labels).to(dtype=logits.dtype)
    if tuple(logits.shape) != tuple(labels.shape):
        raise ValueError(
            f"logits and labels must have the same shape, got {tuple(logits.shape)} "
            f"and {tuple(labels.shape)}"
        )
    weight = float(pairwise_auc_weight)
    if weight not in (0.0, PAIRWISE_AUC_WEIGHT):
        raise ValueError(
            "pairwise_auc_weight is fixed and must be either "
            f"0.0 or {PAIRWISE_AUC_WEIGHT}, got {weight}"
        )

    bce_loss = F.binary_cross_entropy_with_logits(logits, labels)
    positive_logits = logits[labels > 0.5]
    negative_logits = logits[labels <= 0.5]
    has_both_classes = bool(
        positive_logits.numel() > 0 and negative_logits.numel() > 0
    )
    if weight > 0.0 and has_both_classes:
        logit_gaps = positive_logits.unsqueeze(1) - negative_logits.unsqueeze(0)
        pairwise_auc_loss = F.softplus(-logit_gaps).mean()
    else:
        # Exact fallback also makes the no-pairwise ablation skip O(B^2) work.
        pairwise_auc_loss = bce_loss

    prediction_loss = (1.0 - weight) * bce_loss + weight * pairwise_auc_loss
    return prediction_loss, bce_loss, pairwise_auc_loss, has_both_classes


def _tensor_summary(value: Any) -> Dict[str, Any]:
    if not isinstance(value, torch.Tensor):
        return {"present": False}
    detached = value.detach()
    finite = torch.isfinite(detached)
    finite_values = detached[finite]
    return {
        "present": True,
        "shape": list(detached.shape),
        "nonfinite_count": int((~finite).sum().item()),
        "absmax": float(finite_values.abs().max().item()) if finite_values.numel() else None,
    }


def _raise_if_nonfinite(
    *,
    stage: str,
    epoch: int,
    batch_idx: int,
    details: Dict[str, Any],
    logits: torch.Tensor,
    prediction_loss: torch.Tensor,
    bce_loss: torch.Tensor,
    pairwise_auc_loss: torch.Tensor,
    reg_terms: Dict[str, torch.Tensor],
    loss: torch.Tensor,
) -> None:
    tensors = {
        "logits": logits,
        "prediction_loss": prediction_loss,
        "bce_loss": bce_loss,
        "pairwise_auc_loss": pairwise_auc_loss,
        "reg_loss": reg_terms.get("total"),
        "loss": loss,
        "relation_matrices": details.get("relation_matrices"),
    }
    bad_name = next(
        (
            name
            for name, value in tensors.items()
            if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all())
        ),
        None,
    )
    if bad_name is None:
        return
    payload = {
        "architecture": ARCHITECTURE_NAME,
        "bad_tensor": bad_name,
        "tensors": {name: _tensor_summary(value) for name, value in tensors.items()},
    }
    raise NonFiniteTrainingError(
        reason=f"nonfinite_{bad_name}",
        stage=stage,
        epoch=epoch,
        batch_idx=batch_idx,
        payload=payload,
    )


def _regularization_terms(
    model: nn.Module,
    details: Dict[str, Any],
    bce_loss: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    base_model = _get_base_model(model)
    terms = base_model.get_regularization_components(
        relation_matrices=details["relation_matrices"],
        details=details,
        # Keep the regularization cap on the original BCE scale so the
        # pairwise-loss ablation changes only the prediction objective.
        base_loss=bce_loss,
    )
    zero = bce_loss.new_tensor(0.0)
    normalized = {key: terms.get(key, zero) for key in _REG_COMPONENT_KEYS}
    normalized["total"] = terms.get(
        "total",
        sum((normalized[key] for key in _REG_COMPONENT_KEYS if key != "graph_reg_scale"), zero),
    )
    return normalized


def _run_epoch(
    *,
    model: CognitiveDiagnosisModel,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
    stage: str,
    optimizer: Optional[optim.Optimizer] = None,
    logger=None,
    max_batches: Optional[int] = None,
    pairwise_auc_weight: float = 0.0,
) -> Dict[str, float]:
    is_train = optimizer is not None
    effective_pairwise_weight = float(pairwise_auc_weight) if is_train else 0.0
    if effective_pairwise_weight not in (0.0, PAIRWISE_AUC_WEIGHT):
        raise ValueError(
            "training pairwise_auc_weight must be fixed at either "
            f"0.0 or {PAIRWISE_AUC_WEIGHT}"
        )
    model.train(is_train)
    max_batches = None if max_batches is None else max(1, int(max_batches))

    total_loss = 0.0
    total_prediction = 0.0
    total_bce = 0.0
    total_pairwise = 0.0
    total_reg = 0.0
    pairwise_active_batches = 0
    reg_sums = {key: 0.0 for key in _REG_COMPONENT_KEYS}
    all_labels: List[float] = []
    all_preds: List[float] = []
    all_probs: List[float] = []
    processed = 0

    grad_context = torch.enable_grad() if is_train else torch.no_grad()
    with grad_context:
        for batch_idx, (student_ids, exercise_ids, labels) in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
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
            prediction_loss, bce_loss, pairwise_auc_loss, has_both_classes = (
                _prediction_loss(
                    logits,
                    labels,
                    effective_pairwise_weight,
                )
            )
            if is_train:
                reg_terms = _regularization_terms(model, details, bce_loss)
                reg_loss = reg_terms["total"]
            else:
                zero = prediction_loss.new_tensor(0.0)
                reg_terms = {key: zero for key in _REG_COMPONENT_KEYS}
                reg_terms["graph_reg_scale"] = prediction_loss.new_tensor(1.0)
                reg_terms["total"] = zero
                reg_loss = zero
            loss = prediction_loss + reg_loss
            _raise_if_nonfinite(
                stage=stage,
                epoch=epoch,
                batch_idx=batch_idx,
                details=details,
                logits=logits,
                prediction_loss=prediction_loss,
                bce_loss=bce_loss,
                pairwise_auc_loss=pairwise_auc_loss,
                reg_terms=reg_terms,
                loss=loss,
            )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_stats = _clip_stability_sensitive_grads(model)
                if logger is not None and grad_stats["nonfinite_grad_count"] > 0:
                    logger.warning(
                        "[Grad Guard] sanitized %d non-finite gradients (graph=%d, other=%d, params=%s)",
                        int(grad_stats["nonfinite_grad_count"]),
                        int(grad_stats["nonfinite_graph_grad_count"]),
                        int(grad_stats["nonfinite_other_grad_count"]),
                        grad_stats["nonfinite_grad_param_names"],
                    )
                optimizer.step()

            total_loss += float(loss.detach().item())
            total_prediction += float(prediction_loss.detach().item())
            total_bce += float(bce_loss.detach().item())
            total_pairwise += float(pairwise_auc_loss.detach().item())
            total_reg += float(reg_loss.detach().item())
            pairwise_active_batches += int(
                is_train and effective_pairwise_weight > 0.0 and has_both_classes
            )
            for key in _REG_COMPONENT_KEYS:
                reg_sums[key] += float(reg_terms[key].detach().item())
            processed += 1

            probs = torch.sigmoid(logits.detach())
            preds = (probs > 0.5).float()
            all_labels.extend(labels.detach().cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    denominator = max(1, processed)
    avg_prediction = total_prediction / denominator
    avg_bce = total_bce / denominator
    avg_reg = total_reg / denominator
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    if stage != "train" and not math.isfinite(metrics["auc"]):
        raise ValueError(
            f"{stage} AUC is undefined because the evaluated rows contain only one outcome class"
        )
    return {
        "loss": total_loss / denominator,
        "prediction_loss": avg_prediction,
        "bce_loss": avg_bce,
        "pairwise_auc_loss": total_pairwise / denominator,
        "pairwise_auc_weight": effective_pairwise_weight,
        "pairwise_active_batch_rate": pairwise_active_batches / denominator,
        "reg_loss": avg_reg,
        "reg_prediction_ratio": avg_reg / (abs(avg_prediction) + 1e-12),
        "reg_bce_ratio": avg_reg / (abs(avg_bce) + 1e-12),
        **{f"reg_{key}": reg_sums[key] / denominator for key in _REG_COMPONENT_KEYS},
        **metrics,
    }


def train_epoch(
    model: CognitiveDiagnosisModel,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    logger,
    epoch: int,
    max_batches: Optional[int] = None,
    pairwise_auc_weight: float = PAIRWISE_AUC_WEIGHT,
) -> Dict[str, float]:
    return _run_epoch(
        model=model,
        data_loader=train_loader,
        device=device,
        epoch=epoch,
        stage="train",
        optimizer=optimizer,
        logger=logger,
        max_batches=max_batches,
        pairwise_auc_weight=pairwise_auc_weight,
    )


def validate(
    model: CognitiveDiagnosisModel,
    val_loader: DataLoader,
    device: torch.device,
    logger,
    epoch: int,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    return _run_epoch(
        model=model,
        data_loader=val_loader,
        device=device,
        epoch=epoch,
        stage="val",
        logger=logger,
        max_batches=max_batches,
    )


def _relation_learning(model: nn.Module) -> Optional[nn.Module]:
    base_model = _get_base_model(model)
    return getattr(base_model, "relation_learning", None)


def _mean_detail(details: Dict[str, Any], key: str, *, absolute: bool = False) -> float:
    value = details.get(key)
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return 0.0
    value = value.detach().float()
    if absolute:
        value = value.abs()
    return float(value.mean().item())


def _collect_debug_forward_stats(
    model: CognitiveDiagnosisModel,
    data_loader: DataLoader,
    device: torch.device,
    max_batches: int = 2,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    accumulators: Dict[str, List[float]] = {
        "relation_identity_delta": [],
        "knowledge_state_graph_delta": [],
        "irt_logit_abs_mean": [],
        "theta_abs_mean": [],
        "irt_discrimination_mean": [],
        "irt_difficulty_mean": [],
    }
    with torch.no_grad():
        for batch_idx, (student_ids, exercise_ids, _) in enumerate(data_loader):
            if batch_idx >= max(1, int(max_batches)):
                break
            _, details = model(
                student_ids.to(device),
                exercise_ids.to(device),
                return_details=True,
                return_logits=True,
            )
            accumulators["relation_identity_delta"].append(_mean_detail(details, "relation_identity_delta"))
            accumulators["knowledge_state_graph_delta"].append(
                _mean_detail(details, "knowledge_state_graph_delta")
            )
            accumulators["irt_logit_abs_mean"].append(_mean_detail(details, "irt_logit", absolute=True))
            accumulators["theta_abs_mean"].append(_mean_detail(details, "theta_c", absolute=True))
            accumulators["irt_discrimination_mean"].append(_mean_detail(details, "irt_a"))
            accumulators["irt_difficulty_mean"].append(_mean_detail(details, "irt_b"))

    relation = _relation_learning(model)
    graph_diag = {
        "graph_entropy_ratio": 0.0,
        "graph_diag_mass": 0.0,
        "graph_to_uniform_l2": 0.0,
        "graph_to_identity_l2": 0.0,
        "graph_tau_mean": 0.0,
    }
    if relation is not None:
        with torch.no_grad():
            matrices = relation()
            matrices = matrices.detach().float()
            count = max(1, int(matrices.size(-1)))
            entropy = -(matrices.clamp(min=1e-12) * matrices.clamp(min=1e-12).log()).sum(dim=-1).mean()
            max_entropy = math.log(float(count)) if count > 1 else 1.0
            identity = torch.eye(count, device=matrices.device, dtype=matrices.dtype).unsqueeze(0)
            graph_diag = {
                "graph_entropy_ratio": float((entropy / max_entropy).item()) if count > 1 else 0.0,
                "graph_diag_mass": float(torch.diagonal(matrices, dim1=-2, dim2=-1).mean().item()),
                "graph_to_uniform_l2": float(
                    (matrices - (1.0 / count)).pow(2).sum(dim=-1).sqrt().mean().item()
                ),
                "graph_to_identity_l2": float(
                    (matrices - identity).pow(2).sum(dim=-1).sqrt().mean().item()
                ),
                "graph_tau_mean": 0.0,
            }
            tau_raw = getattr(relation, "temperature_raw", getattr(relation, "tau_raw", None))
            if isinstance(tau_raw, torch.Tensor):
                graph_diag["graph_tau_mean"] = float((torch.nn.functional.softplus(tau_raw) + 1e-6).mean().item())

    if was_training:
        model.train()
    return {
        "architecture": ARCHITECTURE_NAME,
        **{
            key: float(np.mean(values)) if values else 0.0
            for key, values in accumulators.items()
        },
        **graph_diag,
    }


def _grad_norm_or_zero(parameter: Optional[torch.Tensor]) -> float:
    if parameter is None or getattr(parameter, "grad", None) is None:
        return 0.0
    return float(parameter.grad.detach().norm().item())


def _collect_debug_grad_norms(model: nn.Module) -> Dict[str, float]:
    relation = _relation_learning(model)
    if relation is None:
        return {
            "relation_tau": 0.0,
            "relation_prior_strength": 0.0,
            "relation_exposure_prior_strength": 0.0,
            "relation_receiver_bias": 0.0,
            "relation_self_loop": 0.0,
        }
    return {
        "relation_tau": _grad_norm_or_zero(
            getattr(relation, "temperature_raw", getattr(relation, "tau_raw", None))
        ),
        "relation_prior_strength": _grad_norm_or_zero(getattr(relation, "prior_strength_raw", None)),
        "relation_exposure_prior_strength": _grad_norm_or_zero(
            getattr(relation, "exposure_prior_strength_raw", None)
        ),
        "relation_receiver_bias": _grad_norm_or_zero(getattr(relation, "receiver_bias", None)),
        "relation_self_loop": _grad_norm_or_zero(getattr(relation, "self_loop_logit", None)),
    }


def _create_loaders(args: Any, logger, *, shuffle_train: bool = True):
    data_dir = args.data_dir
    train_file = os.path.join(data_dir, "train.csv")
    val_file = os.path.join(data_dir, "valid.csv")
    train_loader, val_loader, test_loader, info_dict = create_dataloaders(
        train_file=train_file,
        val_file=val_file,
        test_file=None,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        shuffle_train=shuffle_train,
        min_stu_interactions=int(getattr(args, "min_stu_interactions", 0)),
        min_exer_interactions=int(getattr(args, "min_exer_interactions", 0)),
        logger=logger,
        dataset_name=getattr(args, "dataset_name", getattr(args, "dataset", None)),
        graph_prior_mode=str(getattr(args, "graph_prior_mode", "evidence")),
        seed=int(getattr(args, "seed", 42)),
        load_test=False,
    )
    if test_loader is not None:
        raise RuntimeError("training loader unexpectedly opened the sealed test split")
    info_dict["data_identity"] = _build_data_identity(args, train_file, val_file)
    return train_loader, val_loader, None, info_dict


def _save_checkpoint(
    *,
    path: str,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    args: Any,
    info_dict: Dict[str, Any],
    monitor: Dict[str, str],
    val_metrics: Dict[str, float],
    train_metrics: Optional[Dict[str, float]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
    best_val_auc: Optional[float] = None,
) -> None:
    payload: Dict[str, Any] = {
        "architecture": ARCHITECTURE_NAME,
        "epoch": int(epoch),
        "model_state_dict": _get_base_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_metrics": val_metrics,
        "monitor": monitor,
        "graph_irt_diagnostics": diagnostics or {},
        "args": _checkpoint_args(args),
        "info_dict": info_dict,
    }
    if train_metrics is not None:
        payload["train_metrics"] = train_metrics
    if best_val_auc is not None:
        payload["val_auc"] = float(best_val_auc)
    torch.save(payload, path)


def train_one_experiment(args, logger) -> Tuple[float, int]:
    device = select_device(args, logger)
    run_tag = (
        f"[{getattr(args, 'dataset_name', 'unknown')}"
        f"|{getattr(args, 'model_variant', 'full')}"
        f"|lr={args.learning_rate:g}|drop={args.dropout:.2f}]"
    )
    os.makedirs(args.save_dir, exist_ok=True)
    sealed_results_path = os.path.join(args.save_dir, "test_results.json")
    test_seal_path = os.path.join(args.save_dir, "test_seal.json")
    sealed_path = next(
        (path for path in (sealed_results_path, test_seal_path) if os.path.exists(path)),
        None,
    )
    if sealed_path is not None:
        raise RuntimeError(
            "Refusing to train in a test-sealed run directory; choose a new --save_dir: "
            f"{sealed_path}"
        )
    logger.info("%s Loading datasets...", run_tag)
    train_loader, val_loader, _, info_dict = _create_loaders(args, logger, shuffle_train=True)
    logger.info(
        "%s Samples train=%d val=%d | students=%d exercises=%d concepts=%d",
        run_tag,
        info_dict["train_size"],
        info_dict["val_size"],
        info_dict["num_students"],
        info_dict["num_exercises"],
        info_dict["num_concepts"],
    )

    model = _build_model(args, info_dict, device)
    if bool(getattr(args, "multi_gpu", False)) and torch.cuda.device_count() > 1:
        raw_ids = getattr(args, "gpu_ids", None)
        device_ids = list(range(torch.cuda.device_count())) if raw_ids else None
        model = nn.DataParallel(model, device_ids=device_ids)
        logger.info("%s DataParallel enabled on %d GPUs", run_tag, torch.cuda.device_count())
    _log_runtime_facts(model, logger, run_tag)

    pairwise_auc_weight = _resolve_pairwise_auc_weight(args)
    optimizer = _build_optimizer(model, args)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=MONITOR_MODE,
        factor=0.5,
        patience=int(args.patience),
    )
    monitor = _default_monitor_config()
    logger.info(
        "%s Graph regularization: entropy=%.6f diag=%.6f uniform=%.6f l2=%.6f",
        run_tag,
        float(getattr(args, "lambda_graph_entropy", 0.01)),
        float(getattr(args, "lambda_graph_diag", 0.10)),
        float(getattr(args, "lambda_graph_uniform", 0.04)),
        float(getattr(args, "prediction_l2_lambda", 5e-5)),
    )
    logger.info(
        "%s Training protocol: optimizer=%s plateau_patience=%d early_stop_patience=%d "
        "min_epochs=%d min_delta=%.2g pairwise_auc_weight=%.1f",
        run_tag,
        str(getattr(args, "optimizer", "adam")),
        int(args.patience),
        int(args.early_stop_patience),
        int(getattr(args, "min_epochs", 0)),
        float(getattr(args, "early_stop_min_delta", 0.0)),
        pairwise_auc_weight,
    )
    if int(args.early_stop_patience) <= int(args.patience) + 1:
        logger.warning(
            "%s Early stopping may pre-empt ReduceLROnPlateau: early_stop_patience=%d, "
            "plateau_patience=%d. Use a larger early-stop window for tuning.",
            run_tag,
            int(args.early_stop_patience),
            int(args.patience),
        )

    best_val_auc = float("-inf")
    best_epoch = 0
    patience_counter = 0
    history: Dict[str, Any] = {
        "architecture": ARCHITECTURE_NAME,
        "train": [],
        "val": [],
        "best_epoch": 0,
        "best_val_auc": float("-inf"),
        "monitor": monitor,
    }
    debug_enabled = bool(getattr(args, "debug_graph_diag", False))
    diag_batches = max(1, int(getattr(args, "diag_batches", 2)))
    last_diag: Dict[str, Any] = {}
    graph_uniform_streak = 0
    graph_low_grad_streak = 0

    last_epoch = 0
    for epoch in range(1, int(args.epochs) + 1):
        last_epoch = epoch
        base_model = _get_base_model(model)
        if hasattr(base_model, "set_epoch"):
            base_model.set_epoch(epoch)
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            logger,
            epoch,
            max_batches=getattr(args, "max_train_batches", None),
            pairwise_auc_weight=pairwise_auc_weight,
        )
        grad_norms = _collect_debug_grad_norms(model) if debug_enabled else {}
        val_metrics = validate(
            model,
            val_loader,
            device,
            logger,
            epoch,
            max_batches=getattr(args, "max_val_batches", None),
        )
        scheduler.step(val_metrics["auc"])
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        logger.info(
            "%s Epoch [%03d/%d] | Train loss=%.4f pred=%.4f BCE=%.4f pair=%.4f "
            "pair_w=%.1f pair_batches=%.1f%% reg=%.4f AUC=%.4f ACC=%.4f RMSE=%.4f | "
            "Val loss(BCE)=%.4f AUC=%.4f ACC=%.4f RMSE=%.4f",
            run_tag,
            epoch,
            int(args.epochs),
            train_metrics["loss"],
            train_metrics["prediction_loss"],
            train_metrics["bce_loss"],
            train_metrics["pairwise_auc_loss"],
            train_metrics["pairwise_auc_weight"],
            100.0 * train_metrics["pairwise_active_batch_rate"],
            train_metrics["reg_loss"],
            train_metrics["auc"],
            train_metrics["acc"],
            train_metrics["rmse"],
            val_metrics["loss"],
            val_metrics["auc"],
            val_metrics["acc"],
            val_metrics["rmse"],
        )
        logger.info(
            "%s [Reg] graph_entropy=%.6f graph_diag=%.6f graph_uniform=%.6f prediction_l2=%.6f cap_scale=%.4f",
            run_tag,
            train_metrics["reg_graph_entropy"],
            train_metrics["reg_graph_diag"],
            train_metrics["reg_graph_uniform"],
            train_metrics["reg_prediction_l2"],
            train_metrics["reg_graph_reg_scale"],
        )

        if debug_enabled:
            last_diag = _collect_debug_forward_stats(model, val_loader, device, max_batches=diag_batches)
            last_diag["grad_norms"] = grad_norms
            logger.info(
                "%s [Graph-IRT Diag] entropy_ratio=%.4f diag_mass=%.4f relation_delta=%.6f "
                "concept_state_delta=%.6f irt_logit=%.4f theta=%.4f a=%.4f b=%.4f",
                run_tag,
                last_diag["graph_entropy_ratio"],
                last_diag["graph_diag_mass"],
                last_diag["relation_identity_delta"],
                last_diag["knowledge_state_graph_delta"],
                last_diag["irt_logit_abs_mean"],
                last_diag["theta_abs_mean"],
                last_diag["irt_discrimination_mean"],
                last_diag["irt_difficulty_mean"],
            )
            graph_uniform_streak = graph_uniform_streak + 1 if last_diag["graph_entropy_ratio"] > 0.98 else 0
            graph_grad_total = sum(grad_norms.values())
            graph_low_grad_streak = graph_low_grad_streak + 1 if graph_grad_total < 1e-8 else 0
            if graph_uniform_streak >= 3:
                logger.warning("%s Graph has remained near-uniform for %d epochs.", run_tag, graph_uniform_streak)
            if graph_low_grad_streak >= 2 and float(args.graph_propagation_alpha) > 0.0:
                logger.warning("%s Graph gradients have remained near zero for %d epochs.", run_tag, graph_low_grad_streak)

        if _is_validation_improvement(
            val_metrics["auc"],
            best_val_auc,
            getattr(args, "early_stop_min_delta", 0.0),
        ):
            best_val_auc = float(val_metrics["auc"])
            best_epoch = epoch
            patience_counter = 0
            _save_checkpoint(
                path=os.path.join(args.save_dir, "best_model.pth"),
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                args=args,
                info_dict=info_dict,
                monitor=monitor,
                val_metrics=val_metrics,
                train_metrics=train_metrics,
                diagnostics=last_diag,
                best_val_auc=best_val_auc,
            )
            with open(os.path.join(args.save_dir, "diag_best.json"), "w", encoding="utf-8") as handle:
                json.dump(last_diag, handle, indent=2)
            logger.info("%s -> New best AUC=%.4f at epoch %d", run_tag, best_val_auc, epoch)
        else:
            patience_counter += 1

        if epoch % int(args.save_interval) == 0:
            checkpoint_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch}.pth")
            _save_checkpoint(
                path=checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                args=args,
                info_dict=info_dict,
                monitor=monitor,
                val_metrics=val_metrics,
                train_metrics=train_metrics,
                diagnostics=last_diag,
            )
            logger.info("%s Checkpoint saved: %s", run_tag, checkpoint_path)
            try:
                activity = compute_module_activity(model, val_loader, device, num_samples=300)
                logger.info("%s [Module Activity] %s", run_tag, format_activity_brief(activity))
            except Exception as exc:
                logger.warning("%s Module activity failed: %s", run_tag, exc)

        if _should_early_stop(epoch, patience_counter, args):
            logger.info("%s Early stopping at epoch %d (best AUC=%.4f @ %d)", run_tag, epoch, best_val_auc, best_epoch)
            break

    history["best_epoch"] = best_epoch
    history["best_val_auc"] = best_val_auc
    history["last_graph_irt_diagnostics"] = last_diag
    save_epoch_history_csv(history, args.save_dir, logger)
    with open(os.path.join(args.save_dir, "training_history.json"), "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=4)

    best_model_path = os.path.join(args.save_dir, "best_model.pth")
    if best_epoch <= 0 or not os.path.exists(best_model_path):
        raise RuntimeError("training finished without a validation-selected best checkpoint")
    best_checkpoint = torch.load(best_model_path, map_location="cpu", weights_only=False)
    _require_graph_irt_checkpoint(best_checkpoint, best_model_path)
    _get_base_model(model).load_state_dict(
        _strip_module_prefix(best_checkpoint["model_state_dict"]),
        strict=STRICT_CHECKPOINT_LOADING,
    )
    if hasattr(_get_base_model(model), "set_epoch"):
        _get_base_model(model).set_epoch(best_epoch)
    logger.info("%s Reloaded best checkpoint for final diagnostics (epoch=%d).", run_tag, best_epoch)

    try:
        activity = compute_module_activity(model, val_loader, device, num_samples=500)
        report = format_activity_report(
            activity,
            dataset_name=getattr(args, "dataset_name", "unknown"),
            seed=int(getattr(args, "seed", 42)),
            epoch=best_epoch,
        )
        logger.info("\n%s", report)
        with open(os.path.join(args.save_dir, "module_activity.json"), "w", encoding="utf-8") as handle:
            json.dump(activity, handle, indent=4)
    except Exception as exc:
        logger.warning("%s Module activity report failed: %s", run_tag, exc)

    if debug_enabled:
        try:
            save_component_analysis_data(
                model,
                train_loader,
                device,
                args.save_dir,
                logger,
                num_samples=100,
            )
        except Exception as exc:
            logger.warning("%s Component analysis export failed: %s", run_tag, exc)

    return best_val_auc, best_epoch


def _require_graph_irt_checkpoint(checkpoint: Dict[str, Any], model_path: str) -> None:
    architecture = checkpoint.get("architecture")
    if architecture != ARCHITECTURE_NAME:
        raise RuntimeError(
            f"Refusing checkpoint {model_path}: expected architecture={ARCHITECTURE_NAME!r}, "
            f"found {architecture!r}. Legacy checkpoints must be evaluated with the legacy code."
        )


def _write_json_atomic(path: str, payload: Dict[str, Any]) -> None:
    temp_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=4, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _claim_test_seal(path: str, payload: Dict[str, Any]) -> None:
    """Atomically claim the one allowed test read for a checkpoint directory."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Refusing concurrent or repeated test evaluation; seal already exists: {path}"
        ) from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=4, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        # Keep the exclusive claim in place even after an interrupted write:
        # once a process reaches this point, another process must not open test.
        raise


def _validate_checkpoint_data_identity(
    args: Any,
    loaded_args: Dict[str, Any],
    info_dict: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    identity = info_dict.get("data_identity")
    if not isinstance(identity, dict) or identity.get("schema") != "graph_irt_data_v1":
        raise RuntimeError("Graph-IRT checkpoint is missing a valid train/validation data identity.")

    checkpoint_dataset = str(loaded_args.get("dataset_name", ""))
    if not checkpoint_dataset or checkpoint_dataset != str(identity.get("dataset_name", "")):
        raise RuntimeError("Checkpoint args and data identity disagree on dataset_name.")

    loaded_data_dir = loaded_args.get("data_dir")
    identified_data_dir = identity.get("data_dir")
    if not loaded_data_dir or not identified_data_dir:
        raise RuntimeError("Checkpoint data identity is missing data_dir.")
    checkpoint_data_dir = os.path.realpath(str(loaded_data_dir))
    identity_data_dir = os.path.realpath(str(identified_data_dir))
    if checkpoint_data_dir != identity_data_dir:
        raise RuntimeError("Checkpoint args and data identity disagree on data_dir.")

    explicit = set(getattr(args, "explicit_arg_dests", []) or [])
    if "dataset_name" in explicit and str(args.dataset_name) != checkpoint_dataset:
        raise RuntimeError(
            f"Explicit --dataset_name={args.dataset_name!r} conflicts with checkpoint "
            f"dataset={checkpoint_dataset!r}."
        )
    if "data_dir" in explicit and os.path.realpath(str(args.data_dir)) != checkpoint_data_dir:
        raise RuntimeError(
            f"Explicit --data_dir={args.data_dir!r} conflicts with checkpoint data_dir."
        )

    for split_name, filename, hash_key in (
        ("train", "train.csv", "train_sha256"),
        ("valid", "valid.csv", "valid_sha256"),
    ):
        path = os.path.join(checkpoint_data_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint-bound {split_name} split is missing: {path}")
        expected_hash = str(identity.get(hash_key, ""))
        actual_hash = _sha256_file(path)
        if not expected_hash or actual_hash != expected_hash:
            raise RuntimeError(
                f"Checkpoint-bound {split_name} split fingerprint mismatch; refusing test evaluation."
            )
    return checkpoint_data_dir, identity


def run_inference(args, logger) -> Tuple[Dict[str, float], Dict[str, Any]]:
    model_path = os.path.join(args.save_dir, "best_model.pth")
    test_results_path = os.path.join(args.save_dir, "test_results.json")
    test_seal_path = os.path.join(args.save_dir, "test_seal.json")
    if os.path.exists(test_results_path) or os.path.exists(test_seal_path):
        sealed_path = test_results_path if os.path.exists(test_results_path) else test_seal_path
        checkpoint_note = ""
        if os.path.exists(model_path):
            try:
                with open(sealed_path, "r", encoding="utf-8") as handle:
                    sealed_payload = json.load(handle)
                sealed_hash = sealed_payload.get("checkpoint_sha256")
                current_hash = _sha256_file(model_path)
                if sealed_hash and sealed_hash != current_hash:
                    checkpoint_note = " The sealed result belongs to a different checkpoint file."
            except (OSError, ValueError, TypeError):
                checkpoint_note = " The existing seal is unreadable."
        raise RuntimeError(
            "Refusing to evaluate test twice for the same sealed checkpoint: "
            f"{sealed_path}.{checkpoint_note}"
        )
    logger.info("Loading model from %s...", model_path)
    if not os.path.exists(model_path):
        logger.error("Model file not found: %s", model_path)
        return {}, {}

    device = select_device(args, logger)
    # Keep the unused optimizer state and train-only graph metadata on CPU;
    # model construction moves only the buffers and weights needed for inference.
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    _require_graph_irt_checkpoint(checkpoint, model_path)
    loaded_args: Dict[str, Any] = checkpoint.get("args", {})
    info_dict = checkpoint.get("info_dict")
    if info_dict is None:
        raise RuntimeError("Graph-IRT checkpoint is missing info_dict.")

    checkpoint_data_dir, data_identity = _validate_checkpoint_data_identity(
        args,
        loaded_args,
        info_dict,
    )
    checkpoint_sha256 = _sha256_file(model_path)
    test_file = os.path.join(checkpoint_data_dir, "test.csv")
    if not os.path.isfile(test_file):
        raise FileNotFoundError(f"Checkpoint-bound test split is missing: {test_file}")

    # Validate the complete checkpoint/model contract before unsealing test.
    model = _build_model(loaded_args, info_dict, device)
    state_dict = _strip_module_prefix(checkpoint["model_state_dict"])
    incompatible = model.load_state_dict(state_dict, strict=STRICT_CHECKPOINT_LOADING)
    if not STRICT_CHECKPOINT_LOADING:
        missing = list(getattr(incompatible, "missing_keys", []))
        unexpected = list(getattr(incompatible, "unexpected_keys", []))
        if missing or unexpected:
            raise RuntimeError(f"Graph-IRT checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    if hasattr(model, "set_epoch"):
        model.set_epoch(int(checkpoint.get("epoch", 1)))
    runtime_facts = _log_runtime_facts(model, logger, "[Inference]")

    _claim_test_seal(
        test_seal_path,
        {
            "status": "started",
            "architecture": ARCHITECTURE_NAME,
            "checkpoint_sha256": checkpoint_sha256,
            "dataset_name": loaded_args.get("dataset_name"),
            "model_variant": loaded_args.get("model_variant"),
            "seed": loaded_args.get("seed"),
        },
    )

    test_sha256 = _sha256_file(test_file)
    raw_test = pd.read_csv(test_file)
    required_columns = {"stu_id", "exer_id", "cpt_seq", "label"}
    missing_columns = sorted(required_columns - set(raw_test.columns))
    if missing_columns:
        raise ValueError(f"test split is missing required columns: {missing_columns}")
    raw_labels = pd.to_numeric(raw_test["label"], errors="coerce")
    if raw_labels.isna().any() or not raw_labels.between(0.0, 1.0).all():
        raise ValueError("test labels must be finite values in [0, 1]")
    raw_test["label"] = raw_labels.astype(float)
    student_map = info_dict["stu_id_map"]
    exercise_map = info_dict["exer_id_map"]
    concept_map = info_dict["cpt_id_map"]
    filtered_test = raw_test[
        raw_test["stu_id"].isin(student_map) & raw_test["exer_id"].isin(exercise_map)
    ].reset_index(drop=True)
    before_rows = len(raw_test)
    after_rows = len(filtered_test)
    coverage = after_rows / before_rows if before_rows else 1.0
    logger.info(
        "[Inference Filter] before=%d after=%d dropped=%d coverage=%.2f%%",
        before_rows,
        after_rows,
        before_rows - after_rows,
        coverage * 100.0,
    )
    if after_rows == 0:
        raise ValueError("test split has no train-seen student-item rows")
    if filtered_test["label"].nunique(dropna=True) < 2:
        raise ValueError("filtered test split must contain both outcome classes")

    dataset = CognitiveDiagnosisDataset(
        csv_file=filtered_test,
        stu_id_map=student_map,
        exer_id_map=exercise_map,
        cpt_id_map=concept_map,
    )
    loader_kwargs: Dict[str, Any] = {
        "num_workers": int(args.num_workers),
        "pin_memory": device.type == "cuda",
    }
    if int(args.num_workers) > 0:
        loader_kwargs["persistent_workers"] = True
    test_loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        **loader_kwargs,
    )

    model.eval()
    all_labels: List[float] = []
    all_preds: List[float] = []
    all_probs: List[float] = []
    with torch.no_grad():
        for student_ids, exercise_ids, labels in test_loader:
            labels = _ensure_1d(labels.to(device).float())
            logits = model(
                student_ids.to(device),
                exercise_ids.to(device),
                return_details=False,
                return_logits=True,
            )
            probs = torch.sigmoid(_ensure_1d(logits))
            preds = (probs > 0.5).float()
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    if not math.isfinite(metrics["auc"]):
        raise ValueError("test AUC is undefined because the evaluated rows contain only one outcome class")
    logger.info("Test Results: AUC=%.4f ACC=%.4f RMSE=%.4f", metrics["auc"], metrics["acc"], metrics["rmse"])
    results = {
        "architecture": ARCHITECTURE_NAME,
        "dataset_name": loaded_args.get("dataset_name"),
        "model_variant": loaded_args.get("model_variant"),
        "seed": int(loaded_args.get("seed", 0)),
        "config_hash": _config_hash(SimpleNamespace(**loaded_args)),
        "checkpoint_sha256": checkpoint_sha256,
        "test_file_sha256": test_sha256,
        "data_identity": data_identity,
        "metrics": metrics,
        "num_samples": len(all_labels),
        "model_epoch": int(checkpoint["epoch"]),
        "best_val_auc": float(checkpoint.get("val_auc", 0.0)),
        "test_total_rows": int(before_rows),
        "test_seen_rows": int(after_rows),
        "test_seen_coverage": float(coverage),
        "effective_batch_size": int(getattr(args, "batch_size", 0)),
        "train_only_split_hygiene": bool(info_dict.get("train_only_split_hygiene", False)),
        "test_sealed_during_training": bool(
            info_dict.get("test_sealed_during_training", False)
        ),
        "runtime_facts": runtime_facts,
        "monitor": checkpoint.get("monitor", _default_monitor_config()),
        "config_switches": _collect_structural_switches(loaded_args),
        "graph_irt_diagnostics": checkpoint.get("graph_irt_diagnostics", {}),
        "failure_reason": None,
    }
    _write_json_atomic(test_results_path, results)
    _write_json_atomic(
        test_seal_path,
        {
            "status": "complete",
            "architecture": ARCHITECTURE_NAME,
            "checkpoint_sha256": checkpoint_sha256,
            "test_file_sha256": test_sha256,
            "test_results_sha256": _sha256_file(test_results_path),
            "dataset_name": loaded_args.get("dataset_name"),
            "model_variant": loaded_args.get("model_variant"),
            "seed": loaded_args.get("seed"),
        },
    )

    result_args = SimpleNamespace(**loaded_args)
    result_args.save_dir = args.save_dir
    result_args.log_dir = getattr(args, "log_dir", "")
    result_args.run_mode = "test"
    append_summary_csv(
        result_args,
        metrics=metrics,
        best_val_auc=results["best_val_auc"],
        model_epoch=results["model_epoch"],
        final_model_facts=runtime_facts,
        logger=logger,
    )

    if getattr(args, "generate_diagnosis", False) and hasattr(model, "get_student_diagnosis"):
        diagnosis_rows = []
        for student_id in range(min(5, int(info_dict["num_students"]))):
            diagnosis = model.get_student_diagnosis(student_id)
            diagnosis_rows.append(
                {
                    "student_id": student_id,
                    "original_student_id": int(info_dict["stu_id_reverse_map"].get(student_id, student_id)),
                    "knowledge_mastery": diagnosis["knowledge_mastery"].detach().cpu().tolist(),
                    "student_repr": diagnosis["student_repr"].detach().cpu().tolist(),
                }
            )
        with open(os.path.join(args.save_dir, "student_diagnosis.json"), "w", encoding="utf-8") as handle:
            json.dump(diagnosis_rows, handle, indent=4)

    return metrics, results


def _append_batch_samples(
    collector: Dict[str, List[np.ndarray]],
    key: str,
    value: Any,
    take: int,
) -> None:
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return
    array = value.detach().cpu().numpy()
    if array.ndim == 0:
        array = np.repeat(array.reshape(1), take, axis=0)
    elif array.shape[0] != take and array.shape[0] != 1:
        array = np.repeat(array.reshape(1, *array.shape), take, axis=0)
    collector.setdefault(key, []).append(array[:take].copy())


def save_component_analysis_data(
    model: CognitiveDiagnosisModel,
    train_loader: DataLoader,
    device: torch.device,
    save_dir: str,
    logger,
    num_samples: int = 100,
    materialize_dense_preview: bool = False,
) -> Dict[str, Any]:
    """Save compact graph and IRT samples to ``component_analysis_data.npz``."""
    del materialize_dense_preview
    os.makedirs(save_dir, exist_ok=True)
    base_model = _get_base_model(model)
    was_training = model.training
    model.eval()
    analysis_data: Dict[str, Any] = {"architecture": np.asarray(ARCHITECTURE_NAME)}

    relation = _relation_learning(model)
    if relation is not None:
        with torch.no_grad():
            matrices = relation()
        analysis_data["global_relation_matrices"] = matrices.detach().cpu().numpy()
        logger.info("[Component Analysis] Global graph: %s", tuple(matrices.shape))

    collectors: Dict[str, List[np.ndarray]] = {}
    sample_count = 0
    with torch.no_grad():
        for student_ids, exercise_ids, _ in train_loader:
            if sample_count >= int(num_samples):
                break
            remaining = int(num_samples) - sample_count
            take = min(int(student_ids.size(0)), remaining)
            _, details = model(
                student_ids.to(device),
                exercise_ids.to(device),
                return_details=True,
                return_logits=True,
            )
            _append_batch_samples(collectors, "exercise_ids_samples", exercise_ids, take)
            for detail_key, output_key in (
                ("q_vector", "q_vector_samples"),
                ("irt_logit", "irt_logit_samples"),
                ("theta_c", "theta_samples"),
                ("irt_a", "irt_discrimination_samples"),
                ("irt_b", "irt_difficulty_samples"),
                ("knowledge_state_graph_delta", "knowledge_state_graph_delta_samples"),
                ("relation_identity_delta", "relation_identity_delta_samples"),
            ):
                _append_batch_samples(collectors, output_key, details.get(detail_key), take)
            sample_count += take

    for key, parts in collectors.items():
        if not parts:
            continue
        try:
            analysis_data[key] = np.concatenate(parts, axis=0)[: int(num_samples)]
        except ValueError:
            analysis_data[key] = np.asarray(parts, dtype=object)

    analysis_path = os.path.join(save_dir, "component_analysis_data.npz")
    np.savez_compressed(analysis_path, **analysis_data)
    logger.info("[Component Analysis] Data saved to %s", analysis_path)
    if was_training:
        model.train()
    return analysis_data
