# src/experiment_utils.py
import os
import json
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any
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

# src/experiment_utils.py 里，替换原来的 append_summary_csv

import os
import json
from datetime import datetime

import pandas as pd

def append_summary_csv(
    args,
    metrics,
    best_val_auc: float,
    model_epoch: int,
    logger,
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

    # 消融标记，方便后续筛选
    ablation_flags = []
    for flag in [
        "ablate_soft_prototype",
        "ablate_skill_encoder",
        "ablate_concept_graph",
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
        "ablation_flags",
        "seed",
        "test_auc",
        "test_acc",
        "test_rmse",
        "best_val_auc",
        "model_epoch",
    ]
    front_cols = [c for c in front_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in front_cols]
    df = df[front_cols + other_cols]

    df.to_csv(summary_path, index=False)
    logger.info(f"Summary appended to {summary_path}")
