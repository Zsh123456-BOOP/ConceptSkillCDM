# src/experiment_utils.py
import os
import json
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, Optional
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from gpu_utils import get_best_gpu, parse_gpu_ids


def setup_logging(log_dir: str, name: Optional[str] = None) -> logging.Logger:
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

    # DataParallel 要求主模型在 device_ids[0]，这里固定多卡主卡为可见卡 0。
    if getattr(args, "multi_gpu", False) and torch.cuda.device_count() > 1:
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        logger.info("Using device: cuda:0 (DataParallel primary device)")
        return device

    candidates = None
    raw_candidates = getattr(args, "gpu_candidates", None)
    if raw_candidates is not None:
        try:
            candidates = parse_gpu_ids(str(raw_candidates))
            if not candidates:
                candidates = None
        except Exception:
            candidates = None

    visible_env = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    visible_list = None
    if visible_env:
        try:
            visible_list = parse_gpu_ids(visible_env)
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

# src/experiment_utils.py 里，替换原来的 append_summary_csv

import os
import json
import hashlib
import subprocess
import logging
from datetime import datetime

import pandas as pd


def _get_git_sha(project_root: str) -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return sha or "unknown"
    except Exception:
        return "unknown"


def _get_log_path_from_logger(logger) -> str:
    try:
        for handler in getattr(logger, "handlers", []):
            if isinstance(handler, logging.FileHandler):
                return os.path.abspath(getattr(handler, "baseFilename", ""))
    except Exception:
        pass
    return ""


def _build_config_hash(args) -> str:
    # Keep hash stable and comparable across runs: exclude path/timestamp-only fields.
    keys = [
        "dataset_name",
        "seed",
        "model_variant",
        "ablate_module1",
        "learning_rate",
        "dropout",
        "batch_size",
        "lambda_sparse",
        "lambda_sparse_personal",
        "lambda_alpha",
        "prediction_l2_lambda",
        "graph_reg_warmup_epochs",
        "graph_reg_cap_ratio",
        "graph_propagation_alpha",
        "graph_query_readout_scale",
        "graph_query_readout_2hop_scale",
        "use_concept_graph",
        "use_personal_graph",
        "share_concept_embeddings",
        "graph_identity_residual",
        "personal_local_hops",
        "personal_include_neighbor_rows",
        "personal_support_include_query_self",
        "personal_support_include_graph",
        "personal_support_include_neighbors",
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
        "personal_alpha_temperature",
        "personal_alpha_budget",
        "personal_alpha_base_init",
        "personal_alpha_bias_scale",
        "personal_reg_warmup_epochs",
        "personal_disable_student_global_context",
        "personal_delta_scale",
        "personal_warmup_epochs",
        "graph_headwise_query_gate",
        "graph_edge_bias_rank",
        "graph_prior_logit_scale",
        "ae_query_residual_scale",
        "ae_logit_residual_scale",
        "ae_logit_residual_clip",
        "graph_query_adapter_enable",
        "lambda_personal_kl",
        "lambda_personal_query_residual",
        "personal_query_residual_margin",
        "lambda_alpha_min",
        "alpha_min_target",
        "personal_state_lr_mult",
        "personal_id_lr_mult",
    ]
    payload = {}
    for k in keys:
        if hasattr(args, k):
            payload[k] = getattr(args, k)
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def append_summary_csv(
    args,
    metrics,
    best_val_auc: float,
    model_epoch: int,
    logger,
    final_model_facts: Optional[Dict[str, Any]] = None,
):
    """
    将一次实验的结果追加到统一的 summary CSV 中：
    <project_root>/results/experiment_results.csv

    - 所有数据集、所有消融实验/网格搜索共用一个文件；
    - 指标列放前，参数列放后；
    - 通过 pandas 自动对齐列，避免字段不匹配报错。
    """

    # === 1. 确定 results 目录和文件路径 ===
    # 当前文件在 src/experiment_utils.py，所以项目根目录是上一级
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    summary_path = os.path.join(results_dir, "experiment_results.csv")

    # === 2. 构造当前这一行的内容 ===
    row = {}

    # 基本信息
    row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row["dataset"] = getattr(args, "dataset_name", "unknown")
    row["model_variant"] = getattr(args, "model_variant", "full")
    row["git_sha"] = _get_git_sha(project_root)
    row["run_dir"] = os.path.abspath(getattr(args, "save_dir", ""))
    row["log_path"] = _get_log_path_from_logger(logger)
    row["config_hash"] = _build_config_hash(args)

    # 消融标记，方便后续筛选
    ablation_flags = []
    for flag in [
        "ablate_concept_graph",
        "ablate_module1",
    ]:
        if hasattr(args, flag):
            ablation_flags.append(f"{flag}={getattr(args, flag)}")
    row["ablation_flags"] = ";".join(ablation_flags)

    row["seed"] = getattr(args, "seed", 0)

    # 指标（放在最前面的一批）
    row["test_auc"] = float(metrics["auc"])
    row["test_acc"] = float(metrics["acc"])
    row["test_rmse"] = float(metrics["rmse"])
    row["best_val_auc"] = float(best_val_auc)
    row["model_epoch"] = int(model_epoch)
    row["effective_batch_size"] = int(getattr(args, "batch_size", 0))

    # 运行时事实：防止“CSV 标记消融，但模型实际未生效”
    runtime_keys = (
        ("enable_module1", "final_enable_module1"),
        ("use_concept_graph", "final_use_concept_graph"),
        ("use_personal_graph", "final_use_personal_graph"),
    )
    for key, out_key in runtime_keys:
        if final_model_facts is not None and key in final_model_facts:
            row[out_key] = bool(final_model_facts[key])
        elif hasattr(args, key):
            row[out_key] = bool(getattr(args, key))

    # === 3. 把 args 里的超参数摊平成后面的列 ===
    skip_keys = {
        "log_dir",
        "save_dir",
        "data_dir",
        "gpu_candidates",
        "generate_diagnosis",
    }

    for k, v in sorted(vars(args).items()):
        if k in row or k in skip_keys:
            continue

        # 处理 numpy 类型，避免写 CSV 时不兼容
        try:
            import numpy as np

            if isinstance(v, (np.floating, np.integer)):
                v = float(v) if isinstance(v, np.floating) else int(v)
        except Exception:
            pass

        # list/dict 转成 JSON 字符串
        if isinstance(v, (list, dict)):
            v = json.dumps(v, ensure_ascii=False)

        row[k] = v

    # === 4. 用 pandas 读旧文件 + 追加新行，自动对齐列 ===
    if os.path.exists(summary_path):
        try:
            df = pd.read_csv(summary_path)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        except Exception as e:
            logger.warning(
                f"Failed to read existing summary csv ({summary_path}), "
                f"recreating a new one. Error: {e}"
            )
            df = pd.DataFrame([row])
    else:
        df = pd.DataFrame([row])

    if "ablate_exercise_graph" in df.columns:
        if "ablate_concept_graph" not in df.columns:
            df["ablate_concept_graph"] = df["ablate_exercise_graph"]
        df = df.drop(columns=["ablate_exercise_graph"])
    if "ablation_flags" in df.columns:
        df["ablation_flags"] = df["ablation_flags"].astype(str).str.replace(
            "ablate_exercise_graph=", "ablate_concept_graph=", regex=False
        )

    # === 5. 调整列顺序：指标和关键信息放前面 ===
    front_cols = [
        "timestamp",
        "dataset",
        "model_variant",
        "git_sha",
        "run_dir",
        "log_path",
        "config_hash",
        "ablation_flags",
        "seed",
        "test_auc",
        "test_acc",
        "test_rmse",
        "best_val_auc",
        "model_epoch",
        "effective_batch_size",
        "final_enable_module1",
        "final_use_concept_graph",
        "final_use_personal_graph",
    ]
    front_cols = [c for c in front_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in front_cols]
    df = df[front_cols + other_cols]

    df.to_csv(summary_path, index=False)
    logger.info(f"Summary appended to {summary_path}")
