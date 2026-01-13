#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run best configs for assist_09 and junyi (as provided by your CSV).

IMPORTANT:
- This file is intentionally kept consistent with your BEST_CFG.
- We do NOT change any hyper-parameter values here.
"""

import argparse
import os
import subprocess
import time

BEST_CFG = {
    "junyi": {
        "seed": 42,
        "batch_size": 512,
        "disable_soft_prototype": False,
        "dropout": 0.4,
        "early_stop_patience": 3,
        "epochs": 100,
        "exercise_dim": 128,
        "knowledge_dim": 128,
        "lambda_alpha": 0.0,
        "lambda_proto_div": 0.0,
        "lambda_proto_usage": 0.0,
        "lambda_sparse": 1,
        "lambda_sparse_personal": 0.0,
        "learning_rate": 0.003,
        "min_exer_interactions": 0,
        "min_poison_count": 0,
        "min_stu_interactions": 15,
        "no_cuda": False,
        "num_gnn_layers": 2,
        "num_prototypes": 3,
        "num_relation_heads": 4,
        "num_workers": 4,
        "patience": 1,
        "proto_lambda": 5,
        "proto_tau": 1.0,
        "save_interval": 10,
        "skill_dim": 64,
        "weight_decay": 0.001,
        "ablate_skill_encoder": False,
        "ablate_soft_prototype": False,
        "use_personal_graph": False,
        "model_variant": "gpd_base",
        "exercise_l2_lambda": 5e-5,
    },

    "assist_09": {
        "seed": 42,
        "batch_size": 128,
        "disable_soft_prototype": False,
        "dropout": 0.20,
        "early_stop_patience": 3,
        "epochs": 100,
        "exercise_dim": 64,
        "knowledge_dim": 64,
        "lambda_alpha": 0.0,
        "lambda_proto_div": 0.0,
        "lambda_proto_usage": 0.0,
        "lambda_sparse": 1,
        "lambda_sparse_personal": 0.0,
        "learning_rate": 3e-4,
        "min_exer_interactions": 0,
        "min_poison_count": 0,
        "min_stu_interactions": 15,
        "no_cuda": False,
        "num_gnn_layers": 1,
        "num_prototypes": 3,
        "num_relation_heads": 2,
        "num_workers": 4,
        "patience": 5,
        "proto_lambda": 5,
        "proto_tau": 1.0,
        "save_interval": 10,
        "skill_dim": 64,
        "weight_decay": 1e-5,
        "ablate_skill_encoder": False,
        "ablate_soft_prototype": False,
        "use_personal_graph": False,
        "model_variant": "gpd_base",
        "exercise_l2_lambda": 5e-5,
    },
}


def launch_experiment(dataset_name, cfg, gpu_id):
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

    # Convert cfg to CLI args
    for k, v in cfg.items():
        if k == "model_variant":
            continue
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd.append(f"--{k}")
            cmd.append(str(v))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[LAUNCH] dataset={dataset_name}, gpu={gpu_id}")
    print("         CMD:", " ".join(cmd))
    return subprocess.Popen(cmd, env=env)


def main():
    parser = argparse.ArgumentParser(description="Run best configs for assist_09 & junyi.")
    parser.add_argument("--datasets", type=str, default="assist_09,junyi")
    parser.add_argument("--gpus", type=str, default="0,1")
    parser.add_argument("--max_concurrent", type=int, default=2)
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    max_concurrent = max(1, args.max_concurrent)

    print(f"Datasets: {datasets}")
    print(f"GPUs: {gpus}, max_concurrent={max_concurrent}")

    jobs = []
    for dataset in datasets:
        if dataset not in BEST_CFG:
            raise ValueError(f"Dataset '{dataset}' not in BEST_CFG.")
        jobs.append(dataset)

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
