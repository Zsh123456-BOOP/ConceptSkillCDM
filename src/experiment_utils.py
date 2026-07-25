"""Shared runtime utilities for Graph-IRT experiments."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import subprocess
import tempfile
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score

from gpu_utils import get_best_gpu, parse_gpu_ids
from src.model import GRAPH_IRT_ARCHITECTURE


def setup_logging(log_dir: str, name: Optional[str] = None) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger_name = name or f"graph_irt.{hashlib.sha1(os.path.abspath(log_dir).encode()).hexdigest()[:10]}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(os.path.join(log_dir, f"train_{timestamp}.log"))
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def compute_metrics(labels, preds, probs) -> Dict[str, float]:
    labels_array = np.asarray(labels, dtype=np.float64)
    preds_array = np.asarray(preds, dtype=np.float64)
    probs_array = np.asarray(probs, dtype=np.float64)
    if labels_array.size == 0:
        raise ValueError("cannot compute metrics for an empty split")
    auc = (
        float(roc_auc_score(labels_array, probs_array))
        if np.unique(labels_array).size >= 2
        else float("nan")
    )
    return {
        "auc": auc,
        "acc": float(accuracy_score(labels_array, preds_array)),
        "rmse": float(np.sqrt(mean_squared_error(labels_array, probs_array))),
    }


def select_device(args, logger) -> torch.device:
    if not torch.cuda.is_available() or bool(getattr(args, "no_cuda", False)):
        logger.info("Using device: cpu")
        return torch.device("cpu")

    if bool(getattr(args, "multi_gpu", False)) and torch.cuda.device_count() > 1:
        torch.cuda.set_device(0)
        logger.info("Using device: cuda:0 (DataParallel primary device)")
        return torch.device("cuda:0")

    candidates = None
    raw_candidates = getattr(args, "gpu_candidates", None)
    if raw_candidates:
        parsed = parse_gpu_ids(str(raw_candidates))
        candidates = parsed or None

    visible = None
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        parsed = parse_gpu_ids(os.environ["CUDA_VISIBLE_DEVICES"])
        visible = parsed or None
    if visible:
        candidates = visible if candidates is None else [gpu for gpu in candidates if gpu in visible] or visible

    physical_gpu = get_best_gpu(candidates=candidates, memory_threshold=2000)
    if physical_gpu is None:
        physical_gpu = candidates[0] if candidates else 0
    local_gpu = visible.index(physical_gpu) if visible and physical_gpu in visible else 0 if visible else physical_gpu
    torch.cuda.set_device(local_gpu)
    logger.info("Using device: cuda:%d (physical gpu %d)", local_gpu, physical_gpu)
    return torch.device(f"cuda:{local_gpu}")


def save_epoch_history_csv(history: Dict[str, Any], save_dir: str, logger) -> None:
    rows = []
    for epoch, (train_metrics, val_metrics) in enumerate(
        zip(history.get("train", []), history.get("val", [])), start=1
    ):
        row: Dict[str, Any] = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        rows.append(row)
    path = os.path.join(save_dir, "metrics_history.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info("Epoch-wise metrics saved to %s", path)


def _git_sha(project_root: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip() or "unknown"
    except Exception:
        return "unknown"


def _log_path(logger) -> str:
    for handler in getattr(logger, "handlers", []):
        if isinstance(handler, logging.FileHandler):
            return os.path.abspath(getattr(handler, "baseFilename", ""))
    return ""


CONFIG_HASH_KEYS = (
    "dataset_name",
    "seed",
    "model_variant",
    "gec_mode",
    "warm_start_checkpoint_sha256",
    "train_evidence_mode",
    "knowledge_dim",
    "num_relation_heads",
    "num_gnn_layers",
    "dropout",
    "graph_topk",
    "disable_self_loop",
    "gnn_residual_weight",
    "graph_identity_residual",
    "graph_propagation_alpha",
    "graph_prior_strength_init",
    "pairwise_auc_weight",
    "ema_decay",
    "graph_prior_mode",
    "graph_tau_init",
    "graph_dropout",
    "learning_rate",
    "weight_decay",
    "optimizer",
    "batch_size",
    "epochs",
    "patience",
    "early_stop_patience",
    "min_epochs",
    "early_stop_min_delta",
    "min_stu_interactions",
    "min_exer_interactions",
    "lambda_graph_entropy",
    "graph_reg_warmup_epochs",
    "graph_reg_cap_ratio",
    "graph_entropy_min",
    "graph_entropy_max",
    "lambda_graph_diag",
    "lambda_graph_uniform",
    "graph_uniform_margin",
    "prediction_l2_lambda",
)


def _config_hash(args) -> str:
    payload = {
        key: getattr(args, key)
        for key in CONFIG_HASH_KEYS
        if hasattr(args, key)
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _append_dataframe_csv_atomic(new_row: pd.DataFrame, summary_path: str) -> None:
    """Append one row without losing concurrent experiment results."""
    lock_path = f"{summary_path}.lock"
    with open(lock_path, "a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if os.path.exists(summary_path):
            try:
                existing = pd.read_csv(summary_path)
            except Exception as exc:
                raise RuntimeError(
                    f"Refusing to overwrite unreadable experiment summary: {summary_path}"
                ) from exc
            output = pd.concat([existing, new_row], ignore_index=True)
        else:
            output = new_row

        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=os.path.dirname(summary_path),
                prefix=".experiment_results.",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                temp_path = temp_file.name
                output.to_csv(temp_file, index=False)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temp_path, summary_path)
            temp_path = ""
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)


def append_summary_csv(
    args,
    metrics: Dict[str, float],
    best_val_auc: float,
    model_epoch: int,
    logger,
    final_model_facts: Optional[Dict[str, Any]] = None,
) -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, "experiment_results.csv")

    row: Dict[str, Any] = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "architecture": GRAPH_IRT_ARCHITECTURE,
        "dataset": getattr(args, "dataset_name", "unknown"),
        "model_variant": getattr(args, "model_variant", "full"),
        "train_evidence_mode": getattr(
            args, "train_evidence_mode", "excluded"
        ),
        "seed": int(getattr(args, "seed", 0)),
        "git_sha": _git_sha(project_root),
        "run_dir": os.path.abspath(getattr(args, "save_dir", "")),
        "log_path": _log_path(logger),
        "config_hash": _config_hash(args),
        "test_auc": float(metrics["auc"]),
        "test_acc": float(metrics["acc"]),
        "test_rmse": float(metrics["rmse"]),
        "best_val_auc": float(best_val_auc),
        "model_epoch": int(model_epoch),
        "effective_batch_size": int(getattr(args, "batch_size", 0)),
    }
    for key, value in (final_model_facts or {}).items():
        if isinstance(value, (str, int, float, bool)):
            row[f"runtime_{key}"] = value

    skipped = {"data_dir", "save_dir", "log_dir", "gpu_candidates", "generate_diagnosis"}
    for key, value in sorted(vars(args).items()):
        if key in row or key in skipped:
            continue
        if isinstance(value, np.integer):
            value = int(value)
        elif isinstance(value, np.floating):
            value = float(value)
        elif isinstance(value, (list, dict)):
            value = json.dumps(value, ensure_ascii=False)
        row[key] = value

    _append_dataframe_csv_atomic(pd.DataFrame([row]), summary_path)
    logger.info("Experiment summary appended to %s", summary_path)
