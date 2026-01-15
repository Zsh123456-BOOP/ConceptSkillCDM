#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run best configs for assist_09 and junyi.

配置从 best_configs.py 导入，修改配置请编辑该文件。
"""

import argparse
import os
import subprocess
import time

from best_configs import BEST_CFG



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
