#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_ablation.py
批量消融脚本（基于 run_all_dataset.py 的 BEST_CFG）。

默认（重要）：
- 只跑“model-level”消融（不跑 A~E 子模块消融）
  1) full
  2) no_module1   : --ablate_module1
  3) no_module2   : --ablate_module2
  4) no_module3   : --ablate_module3

可选：
- 若你想跑子模块消融（A~E 对应的旧开关），显式传：
  --ablation_set sub
- 若想 model + sub 全部都跑：
  --ablation_set all

注意：
- 禁止同时 ablate_module2 与 ablate_module3（无预测路径）；main.py 会报错。
"""

import argparse
import os
import subprocess
import time
import sys
from typing import Dict, Any, List, Tuple, Optional

from best_configs import BEST_CFG, DEFAULT_SEEDS


# ========== 3) 消融集合：分成 model-level 与 submodule-level ==========
MODEL_ABLATIONS: List[Dict[str, Any]] = [
    {"name": "full", "flags": {}, "overrides": {}},
    {"name": "no_module1", "flags": {"ablate_module1": True}, "overrides": {}},
    # {"name": "no_module2", "flags": {"ablate_module2": True}, "overrides": {}},
    {"name": "no_module3", "flags": {"ablate_module3": True}, "overrides": {}},
]

# 这些是你说的 “A~E 子模块消融（旧开关）”，默认不跑
SUBMODULE_ABLATIONS: List[Dict[str, Any]] = [
    {"name": "no_soft_proto",
     "flags": {"ablate_soft_prototype": True},
     "overrides": {"num_prototypes": 0}},
    {"name": "no_skill",
     "flags": {"ablate_skill_encoder": True},
     "overrides": {}},
    {"name": "no_concept_graph",
     "flags": {"ablate_concept_graph": True},
     "overrides": {"num_gnn_layers": 0}},
    {"name": "no_personal_graph",
     "flags": {"use_personal_graph": False},  # 注意：这里不是 ablate_*, 而是直接控制 toggle
     "overrides": {}},
]


def _append_arg(cmd: List[str], k: str, v: Any) -> None:
    """bool True -> --k；bool False 不追加；其他 -> --k v"""
    if isinstance(v, bool):
        if v:
            cmd.append(f"--{k}")
        return
    cmd.extend([f"--{k}", str(v)])


def _parse_csv_list(x: str) -> List[str]:
    return [t.strip() for t in x.split(",") if t.strip()]


def _parse_int_list(x: str) -> List[int]:
    out = []
    for t in x.split(","):
        t = t.strip()
        if not t:
            continue
        out.append(int(t))
    return out


def _get_ablation_pool(ablation_set: str) -> List[Dict[str, Any]]:
    if ablation_set == "model":
        return list(MODEL_ABLATIONS)
    if ablation_set == "sub":
        return list(SUBMODULE_ABLATIONS)
    if ablation_set == "all":
        return list(MODEL_ABLATIONS) + list(SUBMODULE_ABLATIONS)
    raise ValueError(f"Unknown ablation_set='{ablation_set}'. Choose from: model, sub, all")


def launch_experiment(
    dataset_name: str,
    base_cfg: Dict[str, Any],
    ablation: Dict[str, Any],
    seed: int,
    gpu_id: int,
    generate_diagnosis: bool,
    dry_run: bool,
) -> Optional[subprocess.Popen]:
    abl_name = ablation["name"]
    tag = f"{dataset_name}_ablation_{abl_name}_seed{seed}"

    save_dir = os.path.join("checkpoints", tag)
    log_dir = os.path.join("logs", tag)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    model_variant = f"gpd_base_{abl_name}"

    cmd = [
        sys.executable, "main.py",
        "--dataset_name", dataset_name,
        "--model_variant", model_variant,
        "--save_dir", save_dir,
        "--log_dir", log_dir,
        "--seed", str(seed),
        "--generate_diagnosis", "True" if generate_diagnosis else "False",
    ]

    # 合并 config：base + overrides + flags
    final_cfg = base_cfg.copy()
    
    # 0. 如果 flag 在 flags 列表里，说明要由 flags 控制（如 use_personal_graph=False），
    #    则先从 config 里删掉它，避免被 _append_arg 当作普通参数加进去了。
    for k in ablation.get("flags", {}).keys():
        if k in final_cfg:
            del final_cfg[k]

    # 1. 应用 overrides (参数值覆盖)
    if "overrides" in ablation:
        for k, v in ablation["overrides"].items():
            final_cfg[k] = v
            
    # 2. 生成此合并后 config 的参数
    for k, v in final_cfg.items():
        if k in ["model_variant", "seed"]:
            continue
        # 跳过消融相关的控制参数（这些不由 config 字典控制，而是由 flags 控制）
        if k.startswith("ablate_") or k.startswith("disable_") or k.startswith("enable_"):
            continue
            
        _append_arg(cmd, k, v)

    # 3. 应用 flags (单纯的 toggle，如 --ablate_module3, --use_personal_graph)
    for k, v in ablation.get("flags", {}).items():
        # 对于 ablate_*, 只有 True 才添加；
        # 对于普通 boolean 参数 (如 use_personal_graph)，True -> --use_personal_graph, False -> 不添加
        if v is True:
            cmd.append(f"--{k}")
        elif isinstance(v, bool) and v is False:
             pass # False flag 不添加
        else:
             # 非 bool 类型的 flag? 本应该去 overrides，以防万一
             _append_arg(cmd, k, v)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    desc = f"{dataset_name}|{abl_name}|seed{seed}|gpu{gpu_id}"
    print(f"[LAUNCH] {desc}")
    print("         CMD:", " ".join(cmd))

    if dry_run:
        return None
    return subprocess.Popen(cmd, env=env)


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments for CognitiveDiagnosisModel.")

    parser.add_argument("--datasets", type=str, default="assist_09,junyi", help="Comma-separated datasets.")
    parser.add_argument("--gpus", type=str, default="0,1,2,3", help="Comma-separated GPU ids.")
    parser.add_argument("--max_concurrent", type=int, default=4, help="Max concurrent experiments.")

    parser.add_argument("--generate_diagnosis", action="store_true", help="Enable diagnosis generation.")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds, e.g., 42,43")

    # 核心：控制跑哪一类消融（默认只跑 model-level）
    parser.add_argument("--ablation_set", type=str, default="model",
                        choices=["model", "sub", "all"],
                        help="Which ablation pool to use. Default=model (no submodule ablations).")

    # 可选：在 ablation_set 的基础上再筛选
    parser.add_argument("--ablations", type=str, default=None,
                        help="Comma-separated ablation names to run (filter within the selected pool).")

    parser.add_argument("--dry_run", action="store_true", help="Print commands only, do not run.")
    parser.add_argument("--poll_interval", type=int, default=10, help="Seconds between polling processes.")

    args = parser.parse_args()

    datasets = _parse_csv_list(args.datasets)
    gpus = _parse_int_list(args.gpus)
    max_concurrent = max(1, args.max_concurrent)
    if not gpus:
        raise ValueError("No GPUs provided. Use --gpus 0 or set properly.")

    seeds = list(DEFAULT_SEEDS) if args.seeds is None else _parse_int_list(args.seeds)

    # 先选定 ablation pool（默认 model-only）
    ablation_pool = _get_ablation_pool(args.ablation_set)

    # 再做 names 过滤
    if args.ablations is None:
        ablations = list(ablation_pool)
    else:
        selected = set(_parse_csv_list(args.ablations))
        ablations = [a for a in ablation_pool if a["name"] in selected]
        pool_names = set(a["name"] for a in ablation_pool)
        missing = selected - pool_names
        if missing:
            raise ValueError(f"Unknown ablation(s) in current pool: {sorted(missing)}. Pool={sorted(pool_names)}")
        if not ablations:
            raise ValueError("No ablations selected after filtering.")

    print(f"Datasets: {datasets}")
    print(f"GPUs: {gpus}, max_concurrent={max_concurrent}")
    print(f"Seeds: {seeds}")
    print(f"Ablation set: {args.ablation_set}")
    print(f"Ablations: {[a['name'] for a in ablations]}")
    print(f"Generate diagnosis: {args.generate_diagnosis}")
    print(f"Dry run: {args.dry_run}")

    jobs: List[Tuple[str, Dict[str, Any], Dict[str, Any], int]] = []
    for dataset in datasets:
        if dataset not in BEST_CFG:
            raise ValueError(f"Dataset '{dataset}' not in BEST_CFG.")
        for ablation in ablations:
            for seed in seeds:
                jobs.append((dataset, BEST_CFG[dataset], ablation, seed))

    print(f"Total experiments: {len(jobs)}")

    running: List[Tuple[subprocess.Popen, int, str]] = []
    job_idx = 0
    gpu_rr = 0

    while job_idx < len(jobs) or running:
        still_running = []
        for proc, gpu, desc in running:
            ret = proc.poll()
            if ret is None:
                still_running.append((proc, gpu, desc))
            else:
                print(f"[DONE] {desc} exited with code {ret}")
        running = still_running

        while job_idx < len(jobs) and len(running) < max_concurrent:
            dataset, base_cfg, ablation, seed = jobs[job_idx]
            gpu_id = gpus[gpu_rr % len(gpus)]
            gpu_rr += 1

            desc = f"{dataset}|{ablation['name']}|seed{seed}|gpu{gpu_id}"

            proc = launch_experiment(
                dataset_name=dataset,
                base_cfg=base_cfg,
                ablation=ablation,
                seed=seed,
                gpu_id=gpu_id,
                generate_diagnosis=args.generate_diagnosis,
                dry_run=args.dry_run,
            )

            if not args.dry_run and proc is not None:
                running.append((proc, gpu_id, desc))

            job_idx += 1

        if running:
            time.sleep(max(1, args.poll_interval))
        else:
            if args.dry_run:
                break


if __name__ == "__main__":
    main()
