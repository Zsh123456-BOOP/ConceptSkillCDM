#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
批量跑 CognitiveDiagnosisModel 的消融实验。

- 固定随机种子 seed=42
- 基于当前 experiment_results.csv 中的最优配置作为 base config
- 单因素消融：
    1) full               : 全部模块开启
    2) no_soft_proto      : 关闭 soft prototype 模块
    3) no_skill           : 关闭应试技巧编码器
    4) no_exercise_graph  : 关闭习题侧图传播

使用示例：
    # 默认：两个数据集 + 两张 GPU (0,1)，最多并行 2 个实验
    python run_ablation.py

    # 只跑 assist_09，在 0,1,2,3 四张卡上并行
    python run_ablation.py --datasets assist_09 --gpus 0,1,2,3 --max_concurrent 4
"""

import argparse
import os
import subprocess
import time

# ===== 1. 当前 best 配置（来自你刚才打印的两行） =====
BEST_CFG = {
    "junyi": {
        # 来自 CSV 最优行：2025-12-09 14:06:18
        "seed": 42,
        "batch_size": 512,
        "disable_soft_prototype": False,
        "dropout": 0.15,          # CSV: 0.15
        "early_stop_patience": 5,
        "epochs": 100,
        "exercise_dim": 128,
        "knowledge_dim": 128,
        "lambda_alpha": 0.0,
        "lambda_proto_div": 0.0,
        "lambda_proto_usage": 0.0,
        "lambda_sparse": 0.05,    # CSV: 0.05
        "lambda_sparse_personal": 0.0,
        "learning_rate": 1.5e-3,  # CSV: 0.0015
        "min_exer_interactions": 0,
        "min_poison_count": 0,
        "min_stu_interactions": 15,
        "no_cuda": False,
        "num_gnn_layers": 2,
        "num_prototypes": 3,
        "num_relation_heads": 4,
        "num_workers": 4,
        "patience": 5,
        "proto_lambda": 0.1,      # CSV: 0.1
        "proto_tau": 1.0,
        "save_interval": 10,
        "skill_dim": 2,
        "weight_decay": 1e-5,
        "ablate_exercise_graph": False,
        "ablate_skill_encoder": False,
        "ablate_soft_prototype": False,
        "use_personal_graph": False,  # CSV: 0
        "model_variant": "gpd_base",
        "exercise_l2_lambda": 5e-5,   # CSV: 0.00005
    },

    "assist_09": {
        # 来自 CSV 最优行：2025-12-09 12:44:09
        "seed": 42,
        "batch_size": 128,
        "disable_soft_prototype": False,
        "dropout": 0.20,          # CSV: 0.20
        "early_stop_patience": 5,
        "epochs": 100,
        "exercise_dim": 64,       # CSV: 64
        "knowledge_dim": 64,      # CSV: 64
        "lambda_alpha": 0.0,
        "lambda_proto_div": 0.0,
        "lambda_proto_usage": 0.0,
        "lambda_sparse": 0.10,    # CSV: 0.10
        "lambda_sparse_personal": 0.0,
        "learning_rate": 3e-4,    # CSV: 0.0003
        "min_exer_interactions": 0,
        "min_poison_count": 0,
        "min_stu_interactions": 15,
        "no_cuda": False,
        "num_gnn_layers": 1,      # CSV: 1
        "num_prototypes": 3,
        "num_relation_heads": 2,  # CSV: 2
        "num_workers": 4,
        "patience": 5,
        "proto_lambda": 0.1,      # CSV: 0.1
        "proto_tau": 1.0,
        "save_interval": 10,
        "skill_dim": 2,
        "weight_decay": 1e-5,
        "ablate_exercise_graph": False,
        "ablate_skill_encoder": False,
        "ablate_soft_prototype": False,
        "use_personal_graph": False,  # CSV: 0
        "model_variant": "gpd_base",
        "exercise_l2_lambda": 5e-5,   # CSV: 0.00005
    },
}

# ===== 2. 消融组合设置 =====
ABLATIONS = [
    {
        "name": "full",
        "flags": {},  # 不关任何模块
    },
    {
        "name": "no_soft_proto",
        "flags": {"ablate_soft_prototype": True},
    },
    {
        "name": "no_skill",
        "flags": {"ablate_skill_encoder": True},
    },
    {
        "name": "no_exercise_graph",
        "flags": {"ablate_exercise_graph": True},
    },
]

# 固定种子：只看结构影响
SEEDS = [42]


def launch_experiment(dataset_name, base_cfg, ablation, seed, gpu_id):
    """
    按照 BEST_CFG + 消融组合构造 main.py 命令并启动子进程。
    """
    abl_name = ablation["name"]
    tag = f"{dataset_name}_ablation_{abl_name}_seed{seed}"

    save_dir = os.path.join("checkpoints", tag)
    log_dir = os.path.join("logs", tag)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # model_variant 帮你在 experiment_results 里区分不同实验
    model_variant = f"gpd_base_{abl_name}"

    cmd = [
        "python",
        "main.py",
        "--dataset_name",
        dataset_name,
        "--model_variant",
        model_variant,
        "--save_dir",
        save_dir,
        "--log_dir",
        log_dir,
        "--seed",
        str(seed),
    ]

    # 1) 先把 base 配置转成命令行
    for k, v in base_cfg.items():
        if k in ["model_variant", "seed"]:
            continue  # 已单独处理

        # bool 参数：True 才加 flag，False 不写（保持默认）
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd.append(f"--{k}")
            cmd.append(str(v))

    # 2) 再叠加消融 flag（会覆盖 base 配置的布尔开关）
    for flag_name, flag_value in ablation["flags"].items():
        if flag_value:
            cmd.append(f"--{flag_name}")

    # 环境变量控制用哪块物理 GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[LAUNCH] dataset={dataset_name}, ablation={abl_name}, seed={seed}, gpu={gpu_id}")
    print("         CMD:", " ".join(cmd))
    return subprocess.Popen(cmd, env=env)


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments for CognitiveDiagnosisModel.")
    parser.add_argument(
        "--datasets",
        type=str,
        default="assist_09,junyi",
        help="逗号分隔的数据集名称，例如 'assist_09,junyi' 或 'assist_09'",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1,2,3",
        help="逗号分隔的 GPU 编号，例如 '0,1' 或 '0,1,2,3'",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=4,
        help="最多并行的实验数（不超过 GPU 数更稳）",
    )
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    max_concurrent = max(1, args.max_concurrent)

    print(f"Datasets: {datasets}")
    print(f"GPUs: {gpus}, max_concurrent={max_concurrent}")

    # 构建任务队列：dataset × ablation × seed
    jobs = []
    for dataset in datasets:
        if dataset not in BEST_CFG:
            raise ValueError(f"Dataset '{dataset}' not in BEST_CFG.")
        for ablation in ABLATIONS:
            for seed in SEEDS:
                jobs.append((dataset, ablation, seed))

    print(f"Total experiments: {len(jobs)}")

    running = []
    job_idx = 0
    gpu_rr = 0  # 简单 round-robin 分配 GPU

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
            dataset, ablation, seed = jobs[job_idx]
            base_cfg = BEST_CFG[dataset]

            gpu_id = gpus[gpu_rr % len(gpus)]
            gpu_rr += 1

            desc = f"{dataset}|{ablation['name']}|seed{seed}"
            proc = launch_experiment(dataset, base_cfg, ablation, seed, gpu_id)
            running.append((proc, gpu_id, desc))
            job_idx += 1

        if running:
            time.sleep(10)


if __name__ == "__main__":
    main()
