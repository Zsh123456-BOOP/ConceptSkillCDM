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
from typing import Dict, List, Optional, Tuple

from best_configs import BEST_CFG
from gpu_utils import get_best_gpus


def _pick_gpus_for_job(
    required: int,
    all_gpus: List[int],
    gpu_load: Dict[int, int],
    max_per_gpu: int,
) -> Optional[List[int]]:
    """为一个任务挑选满足槽位限制的 GPU 列表。"""
    available = [gid for gid in all_gpus if gpu_load.get(gid, 0) < max_per_gpu]
    if len(available) < required:
        return None

    selected = get_best_gpus(n=required, candidates=available)
    if len(selected) < required:
        selected = available[:required]
    return selected


def launch_experiment(dataset_name, cfg, selected_gpus):
    """启动单个实验，支持多 GPU"""
    tag = f"{dataset_name}_best_gpd_base"
    save_dir = os.path.join("checkpoints", tag)
    log_dir = os.path.join("logs", tag)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 获取需要的 GPU 数量
    num_gpus = cfg.get("num_gpus", 1)
    if len(selected_gpus) < num_gpus:
        raise ValueError(
            f"Insufficient GPU assignment for dataset={dataset_name}: "
            f"required={num_gpus}, got={selected_gpus}"
        )

    gpu_str = ",".join(str(g) for g in selected_gpus)

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
        if k in ("model_variant", "num_gpus"):  # 跳过这些参数
            continue
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd.append(f"--{k}")
            cmd.append(str(v))

    # 多 GPU 时添加参数
    if num_gpus > 1:
        cmd.append("--multi_gpu")
        cmd.append("--gpu_ids")
        cmd.append(gpu_str)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_str

    print(f"[LAUNCH] dataset={dataset_name}, gpus={gpu_str} (n={num_gpus})")
    print("         CMD:", " ".join(cmd))
    return subprocess.Popen(cmd, env=env), selected_gpus


def main():
    parser = argparse.ArgumentParser(description="Run best configs for assist_09 & junyi.")
    parser.add_argument("--datasets", type=str, default="assist_09,junyi")
    parser.add_argument("--gpus", type=str, default="1,2")
    parser.add_argument("--max_concurrent", type=int, default=1)
    parser.add_argument("--max_per_gpu", type=int, default=1)
    parser.add_argument("--poll_interval", type=int, default=10)
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    max_concurrent = max(1, args.max_concurrent)
    max_per_gpu = max(1, args.max_per_gpu)
    poll_interval = max(1, args.poll_interval)
    if not gpus:
        raise ValueError("No GPUs provided. Use --gpus 0 or set properly.")
    effective_max_concurrent = min(max_concurrent, len(gpus) * max_per_gpu)

    print(f"Datasets: {datasets}")
    print(
        f"GPUs: {gpus}, max_concurrent={max_concurrent}, "
        f"max_per_gpu={max_per_gpu}, effective_max={effective_max_concurrent}"
    )

    jobs = []
    for dataset in datasets:
        if dataset not in BEST_CFG:
            raise ValueError(f"Dataset '{dataset}' not in BEST_CFG.")
        required_gpus = max(1, int(BEST_CFG[dataset].get("num_gpus", 1)))
        if required_gpus > len(gpus):
            raise ValueError(
                f"Dataset '{dataset}' requires num_gpus={required_gpus}, "
                f"but only {len(gpus)} GPU(s) were provided: {gpus}"
            )
        jobs.append(dataset)

    print(f"Total experiments: {len(jobs)}")

    running: List[Tuple[subprocess.Popen, List[int], str]] = []
    gpu_load: Dict[int, int] = {gid: 0 for gid in gpus}
    job_idx = 0

    while job_idx < len(jobs) or running:
        new_running = []
        for proc, used_gpus, desc in running:
            ret = proc.poll()
            if ret is None:
                new_running.append((proc, used_gpus, desc))
            else:
                gpu_desc = ",".join(str(g) for g in used_gpus)
                print(f"[DONE] {desc} on gpu {gpu_desc} exited with code {ret}")
                for gid in used_gpus:
                    gpu_load[gid] = max(0, gpu_load.get(gid, 0) - 1)
        running = new_running

        while job_idx < len(jobs) and len(running) < effective_max_concurrent:
            dataset = jobs[job_idx]
            cfg = BEST_CFG[dataset]
            required_gpus = max(1, int(cfg.get("num_gpus", 1)))
            selected_gpus = _pick_gpus_for_job(required_gpus, gpus, gpu_load, max_per_gpu)
            if selected_gpus is None:
                break

            desc = f"{dataset}|best_gpd_base"
            proc, selected_gpus = launch_experiment(dataset, cfg, selected_gpus)
            running.append((proc, selected_gpus, desc))
            for gid in selected_gpus:
                gpu_load[gid] += 1
            job_idx += 1

        if running:
            time.sleep(poll_interval)


if __name__ == "__main__":
    main()
