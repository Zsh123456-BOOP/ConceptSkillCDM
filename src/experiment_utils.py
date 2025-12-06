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


def setup_logging(log_dir, name=None):
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


def compute_metrics(labels, preds, probs):
    labels = np.array(labels)
    preds = np.array(preds)
    probs = np.array(probs)

    auc = roc_auc_score(labels, probs)
    acc = accuracy_score(labels, preds)
    rmse = np.sqrt(mean_squared_error(labels, probs))
    return {"auc": float(auc), "acc": float(acc), "rmse": float(rmse)}


def select_device(args, logger):
    if not torch.cuda.is_available() or args.no_cuda:
        device = torch.device("cpu")
        logger.info("Using device: cpu")
        return device

    candidates = None
    if getattr(args, "gpu_candidates", None) is not None:
        try:
            candidates = [
                int(x.strip()) for x in str(args.gpu_candidates).split(",")
                if x.strip() != ""
            ]
            if not candidates:
                candidates = None
        except Exception:
            candidates = None

    best_gpu = get_best_gpu(candidates=candidates, memory_threshold=2000)
    if best_gpu is None:
        best_gpu = 0

    device = torch.device(f"cuda:{best_gpu}")
    torch.cuda.set_device(best_gpu)
    logger.info(f"Using device: cuda:{best_gpu}")
    return device


def save_epoch_history_csv(history, save_dir, logger):
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


def append_summary_csv(args, metrics, best_val_auc, model_epoch, logger):
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "all_results.csv")
    dataset_name = os.path.basename(os.path.normpath(args.data_dir))

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": dataset_name,
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
        "use_soft_prototype": not args.disable_soft_prototype,
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
