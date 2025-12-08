#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
双模块消融脚本（软原型 / 技巧编码器）
- Soft Prototype（软原型）        -> ablate_soft_prototype
- Skill Encoder（技巧编码器）     -> ablate_skill_encoder

对每个数据集（assist_09, junyi），基于 BEST_CFG 超参跑 4 组组合：
    abl_p1s1  : soft proto ON,  skill ON   （基线）
    abl_p0s1  : soft proto OFF, skill ON
    abl_p1s0  : soft proto ON,  skill OFF
    abl_p0s0  : soft proto OFF, skill OFF

用法示例：
    python run_ablation_final.py --datasets assist_09,junyi --gpus 0,1 --max_concurrent 2
"""

import argparse
import os
import subprocess
import time

# ===== 当前最优配置（不含个性化关系图） =====
BEST_CFG = {
    "junyi": {
        "seed": 42,
        "batch_size": 1024,
        "disable_soft_prototype": False,
        "dropout": 0.1,
        "early_stop_patience": 5,
        "epochs": 100,
        "exercise_dim": 128,
        "knowledge_dim": 128,
        "lambda_proto_div": 0.0,
        "lambda_proto_usage": 0.0,
        "lambda_sparse": 0.1,
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
        "ablate_skill_encoder": False,
        "ablate_soft_prototype": False,
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
        "lambda_proto_div": 0.0,
        "lambda_proto_usage": 0.0,
        "lambda_sparse": 0.05,
        "learning_rate": 3e-4,
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
        "ablate_skill_encoder": False,
        "ablate_soft_prototype": False,
        "model_variant": "gpd_base",
    },
}


def build_variants():
    """
    构造 4 组软原型/技巧编码器 on/off 组合。
    返回 dict: variant_name -> flag dict
    """
    variants = {}
    for p in [0, 1]:
        for s in [0, 1]:
            name = f"abl_p{p}s{s}"
            variants[name] = {
                "ablate_soft_prototype": (p == 0),
                "ablate_skill_encoder": (s == 0),
            }
    return variants


def launch_experiment(dataset_name, best_cfg, variant_name, variant_flags, gpu_id):
    """
    拼 main.py 命令并启动子进程：
    - 用 BEST_CFG 作为基线
    - variant_flags 覆盖当前组合的 ablation 开关
    """
    tag = f"{dataset_name}_{variant_name}"
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
        variant_name,
        "--save_dir",
        save_dir,
        "--log_dir",
        log_dir,
    ]

    skip_keys = {
        "ablate_soft_prototype",
        "ablate_skill_encoder",
        "model_variant",
    }

    for k, v in best_cfg.items():
        if k in skip_keys:
            continue

        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(v)])

    for k, v in variant_flags.items():
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(v)])

    cmd.extend(["--seed", str(best_cfg.get("seed", 42))])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[LAUNCH] dataset={dataset_name}, variant={variant_name}, gpu={gpu_id}")
    print("         CMD:", " ".join(cmd))
    return subprocess.Popen(cmd, env=env)


def main():
    parser = argparse.ArgumentParser(description="Run soft-proto / skill ablations.")
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
        help="最多并行的实验数（默认 2）",
    )
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    max_concurrent = max(1, args.max_concurrent)

    print(f"Datasets: {datasets}")
    print(f"GPUs: {gpus}, max_concurrent={max_concurrent}")

    variants = build_variants()
    jobs = []
    for dataset in datasets:
        if dataset not in BEST_CFG:
            raise ValueError(f"Dataset '{dataset}' not in BEST_CFG.")
        for name, flags in variants.items():
            jobs.append((dataset, name, flags))

    print(f"Total experiments: {len(jobs)}")

    running = []
    job_idx = 0
    gpu_rr = 0

    while job_idx < len(jobs) or running:
        new_running = []
        for proc, gpu, desc in running:
            ret = proc.poll()
            if ret is None:
                new_running.append((proc, gpu, desc))
            else:
                print(f"[DONE] {desc} on gpu {gpu} exited with code {ret}")
        running = new_running

        while job_idx < len(jobs) and len(running) < max_concurrent:
            dataset, variant_name, flags = jobs[job_idx]
            best_cfg = BEST_CFG[dataset]

            gpu_id = gpus[gpu_rr % len(gpus)]
            gpu_rr += 1

            desc = f"{dataset}|{variant_name}"
            proc = launch_experiment(dataset, best_cfg, variant_name, flags, gpu_id)
            running.append((proc, gpu_id, desc))
            job_idx += 1

        if running:
            time.sleep(10)


if __name__ == "__main__":
    main()
