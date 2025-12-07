#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_ablation.py

在「已找到较优超参」的前提下，对各模块做结构消融：
- soft prototype
- skill encoder
- exercise graph
- concept fusion

一次命令同时跑 assist_09 和 junyi，
同一数据集的所有 ablation 先跑完，再进入下一个数据集。
"""

import argparse
import os
import subprocess
import time
from typing import Dict, List, Tuple

# =======================
# 1. 每个数据集的“固定超参”
#    —— 根据你当前 grid search 的最优结果手工写死
# =======================
DATASET_BASE_PARAMS: Dict[str, Dict] = {
    "assist_09": {
        # 基本训练超参（固定不动）
        "batch_size": 512,
        "learning_rate": 3e-4,      # 来自当前最优行
        "dropout": 0.1,
        "lambda_sparse": 0.1,
        "lambda_independence": 0.05,
        "lambda_proto_div": 0.0,
        "lambda_proto_usage": 0.0,
        "lambda_sparse_personal": 0.0,
        "lambda_alpha": 0.0,
        # 其它交给 main.py 默认 / DATASET_DEFAULTS
    },
    "junyi": {
        "batch_size": 512,
        "learning_rate": 1e-3,     # 来自当前最优行
        "dropout": 0.1,
        "lambda_sparse": 0.1,
        "lambda_independence": 0.1,
        "lambda_proto_div": 0.0,
        "lambda_proto_usage": 0.0,
        "lambda_sparse_personal": 0.0,
        "lambda_alpha": 0.0,
    },
}

# =======================
# 2. 定义要跑的消融配置
# =======================
ABLATIONS: List[Dict] = [
    {
        "name": "full",           # 对应 model_variant
        "flags": {
            # 全开：不设置任何 ablate_xxx，就用 main.py 里的 use_* 开关默认值
        },
    },
    {
        "name": "no_soft_proto",
        "flags": {
            "ablate_soft_prototype": True,
        },
    },
    {
        "name": "no_skill",
        "flags": {
            "ablate_skill_encoder": True,
        },
    },
    {
        "name": "no_exgraph",
        "flags": {
            "ablate_exercise_graph": True,
        },
    },
    {
        "name": "no_cfuse",
        "flags": {
            "ablate_concept_fusion": True,
        },
    },
]


def build_run_tag(dataset: str, ablation_name: str, overrides: Dict) -> str:
    """
    生成一个比较可读的 tag，用来命名 checkpoints / logs 目录.
    形如: assist_09_no_soft_proto_lr0p0003_ls0p1_li0p05_dp0p1
    """
    parts = [dataset, ablation_name]
    # 只挑一些关键的超参放进 tag，避免太长
    key_short = {
        "learning_rate": "lr",
        "dropout": "dp",
        "lambda_sparse": "ls",
        "lambda_independence": "li",
    }
    for k, short_name in key_short.items():
        if k in overrides:
            v = overrides[k]
            if isinstance(v, float):
                v_str = str(v).replace(".", "p")
            else:
                v_str = str(v)
            parts.append(f"{short_name}{v_str}")
    return "_".join(parts)


def launch_experiment(
    gpu_id: int,
    dataset_name: str,
    ablation_name: str,
    overrides: Dict,
) -> subprocess.Popen:
    """
    启动单个 experiment（调用 main.py）
    """
    tag = build_run_tag(dataset_name, ablation_name, overrides)
    save_dir = os.path.join("checkpoints", tag)
    log_dir = os.path.join("logs", tag)

    cmd = ["python", "main.py", "--dataset_name", dataset_name]

    # model_variant 用来在日志 / 汇总表里标注当前是哪个消融
    cmd += ["--model_variant", ablation_name]

    # 把超参 & ablation flag 展成命令行参数
    for k in sorted(overrides.keys()):
        v = overrides[k]
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd.extend([f"--{k}", str(v)])

    cmd += ["--save_dir", save_dir, "--log_dir", log_dir]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[LAUNCH] GPU {gpu_id} | dataset={dataset_name} | ablation={ablation_name}")
    print("         CMD:", " ".join(cmd))
    return subprocess.Popen(cmd, env=env)


def run_for_dataset(
    dataset_name: str,
    gpus: List[int],
    max_concurrent: int,
):
    """
    对单个数据集，顺序跑完所有 ablation 版本。
    同一时刻最多 max_concurrent 个实验并发，且不会和其它数据集混在一起。
    """
    if dataset_name not in DATASET_BASE_PARAMS:
        raise ValueError(f"Dataset '{dataset_name}' not in DATASET_BASE_PARAMS")

    base = DATASET_BASE_PARAMS[dataset_name]
    print(f"\n========== DATASET: {dataset_name} ==========")
    print(f"Base params: {base}")
    print(f"Ablations: {[a['name'] for a in ABLATIONS]}")
    print(f"GPUs: {gpus}, max_concurrent={max_concurrent}")

    running: List[Tuple[subprocess.Popen, int]] = []
    ablation_idx = 0

    while ablation_idx < len(ABLATIONS) or running:
        # 先清理已经结束的任务
        active: List[Tuple[subprocess.Popen, int]] = []
        busy_gpus = set()
        for proc, gpu in running:
            if proc.poll() is None:
                active.append((proc, gpu))
                busy_gpus.add(gpu)
        running = active

        # 若还有 ablation 没提交，且没达到并发上限，则继续提交
        if ablation_idx < len(ABLATIONS) and len(running) < max_concurrent:
            free_gpus = [g for g in gpus if g not in busy_gpus]
            if not free_gpus:
                time.sleep(5)
                continue

            gpu_id = free_gpus[0]
            ab_cfg = ABLATIONS[ablation_idx]
            ab_name = ab_cfg["name"]
            flags = ab_cfg.get("flags", {})

            overrides = {**base, **flags}
            proc = launch_experiment(gpu_id, dataset_name, ab_name, overrides)
            running.append((proc, gpu_id))
            ablation_idx += 1
        else:
            # 要么没有空位，要么所有 ablation 已提交但还在跑，稍等一下
            if running:
                time.sleep(5)
            else:
                # 所有任务完成
                break

    # 等最后一批彻底结束（理论上上面的循环已经处理完，这里多一层保险）
    for proc, gpu in running:
        proc.wait()
        print(f"[DONE] dataset={dataset_name} | GPU {gpu} finished with code {proc.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies for CognitiveDiagnosisModel.")
    parser.add_argument(
        "--datasets",
        type=str,
        default="assist_09,junyi",
        help='Comma-separated dataset names, e.g. "assist_09,junyi"',
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1",
        help='Comma-separated GPU ids, e.g. "0,1"',
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=2,
        help="Maximum concurrent experiments per dataset.",
    )
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip() != ""]
    gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]

    print("Datasets to run:", datasets)

    for ds in datasets:
        run_for_dataset(ds, gpus, args.max_concurrent)


if __name__ == "__main__":
    main()
