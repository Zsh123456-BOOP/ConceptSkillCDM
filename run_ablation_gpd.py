#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
专门跑 G-PDS / 正则项消融的脚本。
- 固定为上一轮 grid search 的“较优”超参
- 在 independence / personal graph 上做小规模消融
"""

import argparse
import os
import subprocess
import time


# ===== 基础配置：来自你上一轮 grid search 的“较优点” =====
BASE_CFG = {
    "assist_09": {
        "learning_rate": 3e-4,
        "dropout": 0.1,
        "lambda_sparse": 0.10,
        "lambda_independence": 0.05,
    },
    "junyi": {
        "learning_rate": 1e-3,
        "dropout": 0.1,
        "lambda_sparse": 0.10,
        "lambda_independence": 0.10,
    },
}

# ===== 消融组合：每个数据集跑这 4 组 =====
ABLATIONS = {
    # name: overrides
    "base": {
        # 保留 BASE_CFG 的 lambda_independence
        "use_personal_graph": False,
        "lambda_sparse_personal": 0.0,
        "lambda_alpha": 0.0,
    },
    "no_indep": {
        "lambda_independence": 0.0,
        "use_personal_graph": False,
        "lambda_sparse_personal": 0.0,
        "lambda_alpha": 0.0,
    },
    "gpd_light": {
        "use_personal_graph": True,
        "lambda_sparse_personal": 1e-4,
        "lambda_alpha": 0.01,
    },
    "gpd_strong": {
        "use_personal_graph": True,
        "lambda_sparse_personal": 1e-3,
        "lambda_alpha": 0.05,
    },
}


def format_val(v):
    if isinstance(v, float):
        return str(v).replace(".", "p")
    return str(v)


def launch_experiment(dataset_name, base_cfg, ablation_name, overrides, gpu_id):
    """拼 main.py 命令并启动子进程"""
    tag = f"{dataset_name}_gpd_{ablation_name}"
    save_dir = os.path.join("checkpoints", tag)
    log_dir = os.path.join("logs", tag)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    cmd = [
        "python",
        "main.py",
        "--dataset_name",
        dataset_name,
        "--model_variant",
        f"gpd_{ablation_name}",
        "--save_dir",
        save_dir,
        "--log_dir",
        log_dir,
    ]

    # 1) 固定基础超参
    for k, v in base_cfg.items():
        cmd.append(f"--{k}")
        cmd.append(str(v))

    # 2) 覆盖消融特定参数
    for k, v in overrides.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")  # bool 参数用 flag 形式
        else:
            cmd.append(f"--{k}")
            cmd.append(str(v))

    # 3) 统一 seed，方便对比
    cmd += ["--seed", "42"]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[LAUNCH] dataset={dataset_name}, ablation={ablation_name}, gpu={gpu_id}")
    print("         CMD:", " ".join(cmd))
    return subprocess.Popen(cmd, env=env)


def main():
    parser = argparse.ArgumentParser(description="Run G-PDS / regularization ablations.")
    parser.add_argument(
        "--datasets",
        type=str,
        default="assist_09,junyi",
        help="Comma-separated dataset names, e.g. 'assist_09,junyi'",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1",
        help="Comma-separated GPU ids to use, e.g. '0,2'",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=2,
        help="Maximum concurrent experiments.",
    )
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    max_concurrent = args.max_concurrent

    print(f"Datasets: {datasets}")
    print(f"GPUs: {gpus}, max_concurrent={max_concurrent}")
    print(f"Ablations: {list(ABLATIONS.keys())}")

    # 任务队列：[(dataset, ablation_name)]
    jobs = []
    for dataset in datasets:
        if dataset not in BASE_CFG:
            raise ValueError(f"Dataset '{dataset}' not in BASE_CFG.")
        for ablation_name in ABLATIONS.keys():
            jobs.append((dataset, ablation_name))

    print(f"Total experiments: {len(jobs)}")

    running = []
    job_idx = 0
    gpu_rr = 0

    while job_idx < len(jobs) or running:
        # 清理已结束进程
        new_running = []
        for proc, gpu, desc in running:
            ret = proc.poll()
            if ret is None:
                new_running.append((proc, gpu, desc))
            else:
                print(f"[DONE] {desc} on gpu {gpu} exited with code {ret}")
        running = new_running

        # 提交新任务
        while job_idx < len(jobs) and len(running) < max_concurrent:
            dataset, ablation_name = jobs[job_idx]
            base_cfg = BASE_CFG[dataset]
            overrides = ABLATIONS[ablation_name]

            gpu_id = gpus[gpu_rr % len(gpus)]
            gpu_rr += 1

            desc = f"{dataset}|{ablation_name}"
            proc = launch_experiment(dataset, base_cfg, ablation_name, overrides, gpu_id)
            running.append((proc, gpu_id, desc))
            job_idx += 1

        if running:
            time.sleep(10)


if __name__ == "__main__":
    main()
