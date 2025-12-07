# src/experiment_utils.py
import os
import json
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from gpu_utils import get_best_gpu


def setup_logging(log_dir: str, name: str | None = None) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{name or "train"}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logger = logging.getLogger(name or __name__)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # 防止重复 handler

    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        sh = logging.StreamHandler()
        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(fmt)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)

    return logger


def compute_metrics(labels, preds, probs) -> dict:
    labels = np.array(labels)
    preds = np.array(preds)
    probs = np.array(probs)

    auc = roc_auc_score(labels, probs)
    acc = accuracy_score(labels, preds)
    rmse = np.sqrt(mean_squared_error(labels, probs))
    return {"auc": float(auc), "acc": float(acc), "rmse": float(rmse)}


def select_device(args, logger) -> torch.device:
    if not torch.cuda.is_available() or getattr(args, "no_cuda", False):
        device = torch.device("cpu")
        logger.info("Using device: cpu")
        return device

    candidates = None
    raw_candidates = getattr(args, "gpu_candidates", None)
    if raw_candidates is not None:
        try:
            candidates = [
                int(x.strip()) for x in str(raw_candidates).split(",") if x.strip() != ""
            ]
            if not candidates:
                candidates = None
        except Exception:
            candidates = None

    visible_env = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    visible_list = None
    if visible_env:
        try:
            visible_list = [
                int(x.strip()) for x in visible_env.split(",") if x.strip() != ""
            ]
            if not visible_list:
                visible_list = None
        except Exception:
            visible_list = None

    candidate_list = candidates
    if visible_list:
        if candidate_list is None:
            candidate_list = visible_list
        else:
            overlap = [gid for gid in candidate_list if gid in visible_list]
            candidate_list = overlap if overlap else candidate_list

    physical_gpu = get_best_gpu(candidates=candidate_list, memory_threshold=2000)
    if physical_gpu is None:
        physical_gpu = candidates[0] if candidates else 0

    local_gpu = physical_gpu
    if visible_list:
        if physical_gpu in visible_list:
            local_gpu = visible_list.index(physical_gpu)
        else:
            local_gpu = 0
            physical_gpu = visible_list[0]

    device = torch.device(f"cuda:{local_gpu}")
    torch.cuda.set_device(local_gpu)
    if visible_list and physical_gpu != local_gpu:
        logger.info(f"Using device: cuda:{local_gpu} (physical gpu {physical_gpu})")
    else:
        logger.info(f"Using device: cuda:{local_gpu}")
    return device


def save_epoch_history_csv(history: dict, save_dir: str, logger):
    rows = []
    for ep, (tr, va) in enumerate(zip(history["train"], history["val"]), start=1):
        row = {"epoch": ep}
        for k, v in tr.items():
            row[f"train_{k}"] = v
        for k, v in va.items():
            row[f"val_{k}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    path = os.path.join(save_dir, "metrics_history.csv")
    df.to_csv(path, index=False)
    logger.info(f"Epoch-wise metrics saved to {path}")


def append_summary_csv(args, metrics: dict, best_val_auc: float, model_epoch: int, logger):
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "all_results.csv")
    dataset_name = getattr(args, "dataset_name", None) or os.path.basename(
        os.path.normpath(args.data_dir)
    )

    ablation_flags = {
        "ablate_soft_prototype": getattr(args, "ablate_soft_prototype", False),
        "ablate_skill_encoder": getattr(args, "ablate_skill_encoder", False),
        "ablate_exercise_graph": getattr(args, "ablate_exercise_graph", False),
        "ablate_concept_fusion": getattr(args, "ablate_concept_fusion", False),
    }
    ablation_str = ";".join([f"{k}={v}" for k, v in ablation_flags.items()])

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": dataset_name,
        "model_variant": getattr(args, "model_variant", "full"),
        "ablation_flags": ablation_str,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "knowledge_dim": args.knowledge_dim,
        "skill_dim": args.skill_dim,
        "exercise_dim": args.exercise_dim,
        "num_relation_heads": args.num_relation_heads,
        "num_gnn_layers": args.num_gnn_layers,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "lambda_sparse": args.lambda_sparse,
        "lambda_independence": args.lambda_independence,
        "lambda_proto_div": args.lambda_proto_div,
        "lambda_proto_usage": args.lambda_proto_usage,
        "use_soft_prototype": getattr(args, "use_soft_prototype", True),
        "use_skill_encoder": getattr(args, "use_skill_encoder", True),
        "use_exercise_graph": getattr(args, "use_exercise_graph", True),
        "use_concept_fusion": getattr(args, "use_concept_fusion", True),
        "num_prototypes": args.num_prototypes,
        "proto_tau": args.proto_tau,
        "proto_lambda": args.proto_lambda,
        "test_auc": metrics["auc"],
        "test_acc": metrics["acc"],
        "test_rmse": metrics["rmse"],
        "best_val_auc": best_val_auc,
        "model_epoch": model_epoch,
    }

    df = pd.DataFrame([row])
    file_exists = os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=not file_exists, index=False)
    logger.info(f"Experiment summary appended to {csv_path}")
