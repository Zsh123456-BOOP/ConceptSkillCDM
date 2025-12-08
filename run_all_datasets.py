#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用当前最优超参，同时跑 assist_09 和 junyi。

默认：
- 数据集：assist_09, junyi
- GPU：0,1
- 每个数据集只跑一组（gpd_base 的 best 配置）

用法示例：
    python run_all_datasets.py
    python run_all_datasets.py --gpus 0,2
    python run_all_datasets.py --datasets assist_09 --gpus 3
"""

import argparse
import os
import subprocess
import time

# ===== 根据你贴出来的 CSV 行手工整理的“最佳配置” =====
# 列名：
# timestamp, dataset, model_variant, ablation_flags, seed, test_auc, test_acc,
# test_rmse, best_val_auc, model_epoch, ablate_exercise_graph, ablate_skill_encoder,
# ablate_soft_prototype, batch_size, dataset_name, disable_soft_prototype,
# dropout, early_stop_patience, epochs, exercise_dim, knowledge_dim,
# lambda_alpha, lambda_proto_div, lambda_proto_usage, lambda_sparse,
# lambda_sparse_personal, learning_rate, min_exer_interactions, min_poison_count,
# min_stu_interactions, no_cuda, num_gnn_layers, num_prototypes,
# num_relation_heads, num_workers, patience, proto_lambda, proto_tau,
# save_interval, skill_dim, use_exercise_graph, use_personal_graph,
# use_skill_encoder, use_soft_prototype, weight_decay

BEST_CFG = {
    "junyi": {
        # 来自 CSV
        "seed": 42,
        "batch_size": 1024,
        "disable_soft_prototype": False,
        "dropout": 0.1,
        "early_stop_patience": 5,
        "epochs": 100,
        "exercise_dim": 128,
        "knowledge_dim": 128,
        "lambda_alpha": 0.0,
        "lambda_proto_div": 0.0,
        "lambda_proto_usage": 0.0,
        "lambda_sparse": 0.1,
        "lambda_sparse_personal": 0.0,
        "learning_rate": 1e-3,
        "min_exer_interactions": 0,
        "min_poison_count": 0,
        "min_stu_interactions": 15,
        "no_cuda": False,
        "num_gnn_layers": 2,
        "num_prototypes": 3,
        "num_relation_heads": 4,
        "num_workers": 4,
        "patience": 5,
        "proto_lambda": 0.5,
        "proto_tau": 1.0,
        "save_interval": 10,
        "skill_dim": 2,
        "weight_decay": 1e-5,
        # 消融/模块开关相关
        "ablate_exercise_graph": False,
        "ablate_skill_encoder": False,
        "ablate_soft_prototype": False,
        "use_personal_graph": False,  # CSV 里是 0
        # 其他
        "model_variant": "gpd_base",
    },
    "assist_09": {
        "seed": 42,
        "batch_size": 128,
        "disable_soft_prototype": False,
        "dropout": 0.2,
        "early_stop_patience": 5,
        "epochs": 100,
        "exercise_dim": 128,
        "knowledge_dim": 128,
        "lambda_alpha": 0.0,
        "lambda_proto_div": 0.0,
        "lambda_proto_usage": 0.0,
        "lambda_sparse": 0.05,
        "lambda_sparse_personal": 0.0,
        "learning_rate": 3e-4,  # assist_09 的最佳行是 0.0003
        "min_exer_interactions": 0,
        "min_poison_count": 0,
        "min_stu_interactions": 15,
        "no_cuda": False,
        "num_gnn_layers": 2,
        "num_prototypes": 3,
        "num_relation_heads": 4,
        "num_workers": 4,
        "patience": 5,
        "proto_lambda": 0.3,
        "proto_tau": 1.0,
        "save_interval": 10,
        "skill_dim": 2,
        "weight_decay": 1e-5,
        "ablate_exercise_graph": False,
        "ablate_skill_encoder": False,
        "ablate_soft_prototype": False,
        "use_personal_graph": False,  # CSV 里是 0
        "model_variant": "gpd_base",
    },
}


def launch_experiment(dataset_name, cfg, gpu_id):
    """
    按照 BEST_CFG 构造 main.py 命令并启动子进程
    """
    tag = f"{dataset_name}_best_gpd_base"
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
        cfg.get("model_variant", "gpd_base"),
        "--save_dir",
        save_dir,
        "--log_dir",
        log_dir,
    ]

    # 把 cfg 里和 argparse 对应的参数都转成命令行
    for k, v in cfg.items():
        if k == "model_variant":
            continue  # 上面已经单独处理过

        # bool 参数：True 才加 flag，False 不写（保持默认）
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd.append(f"--{k}")
            cmd.append(str(v))

    env = os.environ.copy()
    # 直接用物理 GPU id；如果你想用逻辑映射，可以自己改成 "0" / "1" 等
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[LAUNCH] dataset={dataset_name}, gpu={gpu_id}")
    print("         CMD:", " ".join(cmd))
    return subprocess.Popen(cmd, env=env)


def main():
    parser = argparse.ArgumentParser(description="Run best configs for assist_09 & junyi.")
    parser.add_argument(
        "--datasets",
        type=str,
        default="assist_09,junyi",
        help="逗号分隔的数据集名称，例如 'assist_09,junyi' 或 'assist_09'",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1",
        help="逗号分隔的 GPU 编号，例如 '0,1' 或 '2,3'",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=2,
        help="最多并行的实验数（默认 2，刚好两个数据集一起跑）",
    )
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    max_concurrent = max(1, args.max_concurrent)

    print(f"Datasets: {datasets}")
    print(f"GPUs: {gpus}, max_concurrent={max_concurrent}")

    # 构建任务队列：每个 dataset 跑一条 best 配置
    jobs = []
    for dataset in datasets:
        if dataset not in BEST_CFG:
            raise ValueError(f"Dataset '{dataset}' not in BEST_CFG.")
        jobs.append(dataset)

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
            dataset = jobs[job_idx]
            cfg = BEST_CFG[dataset]

            gpu_id = gpus[gpu_rr % len(gpus)]
            gpu_rr += 1

            desc = f"{dataset}|best_gpd_base"
            proc = launch_experiment(dataset, cfg, gpu_id)
            running.append((proc, gpu_id, desc))
            job_idx += 1

        if running:
            time.sleep(10)


if __name__ == "__main__":
    main()
