#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_ablation.py
批量消融脚本（基于 run_all_dataset.py 的 BEST_CFG）。

默认（重要）：
- 只跑“model-level”消融
  1) full
  2) no_module1   : --ablate_module1

可选：
- 若你想跑子模块消融（A/B/E 对应的旧开关），显式传：
  --ablation_set sub
- 若想 model + sub 全部都跑：
  --ablation_set all

注意：
- B 已删除，D 固定启用，因此不再支持 no_B / no_D / no_module2 / no_module3。
"""

import argparse
import os
import subprocess
import time
import sys
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from best_configs import BEST_CFG, DEFAULT_SEEDS
from gpu_utils import (
    calc_effective_max_concurrent,
    parse_int_csv,
    parse_gpu_ids,
    pick_gpu_with_slot_round_robin,
)


# ========== 3) 消融集合：分成 model-level 与 submodule-level ==========
MODEL_ABLATIONS: List[Dict[str, Any]] = [
    {"name": "full", "flags": {}, "overrides": {}},
    {"name": "no_module1", "flags": {"ablate_module1": True}, "overrides": {}},
]

# 这些是 A/E 子模块消融，默认不跑
SUBMODULE_ABLATIONS: List[Dict[str, Any]] = [
    {"name": "no_concept_graph",
     "flags": {"ablate_concept_graph": True},
     "overrides": {}},
    {"name": "no_personal_graph",
     "flags": {"use_personal_graph": False},  # 注意：这里不是 ablate_*, 而是直接控制 toggle
     "overrides": {}},
]

ABLATION_NAME_ALIASES: Dict[str, str] = {
    "no_A": "no_concept_graph",
    "no_E": "no_personal_graph",
    "no_concept_graph": "no_concept_graph",
    "no_personal_graph": "no_personal_graph",
    "no_module1": "no_module1",
    "full": "full",
}


def _append_arg(cmd: List[str], k: str, v: Any) -> None:
    """bool True -> --k；bool False 不追加；其他 -> --k v"""
    if isinstance(v, bool):
        if v:
            cmd.append(f"--{k}")
        return
    cmd.extend([f"--{k}", str(v)])


def _parse_csv_list(x: str) -> List[str]:
    return [t.strip() for t in x.split(",") if t.strip()]


def _get_ablation_pool(ablation_set: str) -> List[Dict[str, Any]]:
    if ablation_set == "model":
        return list(MODEL_ABLATIONS)
    if ablation_set == "sub":
        return list(SUBMODULE_ABLATIONS)
    if ablation_set == "all":
        return list(MODEL_ABLATIONS) + list(SUBMODULE_ABLATIONS)
    raise ValueError(f"Unknown ablation_set='{ablation_set}'. Choose from: model, sub, all")


def resolve_ablation_names(names: List[str], ablation_set: str) -> List[str]:
    pool_names = {a["name"] for a in _get_ablation_pool(ablation_set)}
    resolved: List[str] = []
    for name in names:
        canonical = ABLATION_NAME_ALIASES.get(name, name)
        if canonical not in pool_names:
            raise ValueError(
                f"Unknown ablation '{name}' (resolved='{canonical}') for pool '{ablation_set}'. "
                f"Available={sorted(pool_names)}"
            )
        resolved.append(canonical)
    return resolved


def launch_experiment(
    dataset_name: str,
    base_cfg: Dict[str, Any],
    ablation: Dict[str, Any],
    seed: int,
    gpu_id: int,
    generate_diagnosis: bool,
    dry_run: bool,
    run_session: str,
) -> Optional[subprocess.Popen]:
    abl_name = ablation["name"]
    tag = f"{dataset_name}_ablation_{abl_name}_seed{seed}_{run_session}"

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

    # 3. 应用 flags (单纯的 toggle，如 --ablate_module1, --use_personal_graph)
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
    parser.add_argument("--gpus", type=str, default="2", help="Comma-separated GPU ids.")
    parser.add_argument("--max_concurrent", type=int, default=1, help="Max concurrent experiments.")
    parser.add_argument("--max_per_gpu", type=int, default=1,
                        help="Max experiments per GPU at the same time.")

    parser.add_argument("--generate_diagnosis", action="store_true", help="Enable diagnosis generation.")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds, e.g., 42,43")

    # 核心：控制跑哪一类消融（默认只跑 model-level）
    parser.add_argument("--ablation_set", type=str, default="model",
                        choices=["model", "sub", "all"],
                        help="Which ablation pool to use. Default=model (no submodule ablations).")

    # 可选：在 ablation_set 的基础上再筛选
    parser.add_argument("--ablations", type=str, default=None,
                        help="Comma-separated ablation names to run (filter within the selected pool). "
                             "Supports aliases: no_A/no_E.")

    parser.add_argument("--dry_run", action="store_true", help="Print commands only, do not run.")
    parser.add_argument("--poll_interval", type=int, default=10, help="Seconds between polling processes.")
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Optional run identifier appended to output directories. Default uses current timestamp.",
    )

    args = parser.parse_args()

    datasets = _parse_csv_list(args.datasets)
    gpus = parse_gpu_ids(args.gpus)
    max_concurrent = max(1, args.max_concurrent)
    max_per_gpu = max(1, args.max_per_gpu)
    if not gpus:
        raise ValueError("No GPUs provided. Use --gpus 0 or set properly.")
    effective_max_concurrent = calc_effective_max_concurrent(max_concurrent, gpus, max_per_gpu)

    seeds = list(DEFAULT_SEEDS) if args.seeds is None else parse_int_csv(args.seeds)

    # 先选定 ablation pool（默认 model-only）
    ablation_pool = _get_ablation_pool(args.ablation_set)

    # 再做 names 过滤
    if args.ablations is None:
        ablations = list(ablation_pool)
    else:
        selected = set(resolve_ablation_names(_parse_csv_list(args.ablations), args.ablation_set))
        ablations = [a for a in ablation_pool if a["name"] in selected]
        pool_names = set(a["name"] for a in ablation_pool)
        missing = selected - pool_names
        if missing:
            raise ValueError(f"Unknown ablation(s) in current pool: {sorted(missing)}. Pool={sorted(pool_names)}")
        if not ablations:
            raise ValueError("No ablations selected after filtering.")

    print(f"Datasets: {datasets}")
    print(
        f"GPUs: {gpus}, max_concurrent={max_concurrent}, "
        f"max_per_gpu={max_per_gpu}, effective_max={effective_max_concurrent}"
    )
    print(f"Seeds: {seeds}")
    print(f"Ablation set: {args.ablation_set}")
    print(f"Ablations: {[a['name'] for a in ablations]}")
    print(f"Generate diagnosis: {args.generate_diagnosis}")
    print(f"Dry run: {args.dry_run}")
    run_session = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Run session: {run_session}")

    jobs: List[Tuple[str, Dict[str, Any], Dict[str, Any], int]] = []
    for dataset in datasets:
        if dataset not in BEST_CFG:
            raise ValueError(f"Dataset '{dataset}' not in BEST_CFG.")
        for ablation in ablations:
            for seed in seeds:
                jobs.append((dataset, BEST_CFG[dataset], ablation, seed))

    print(f"Total experiments: {len(jobs)}")

    running: List[Tuple[subprocess.Popen, int, str]] = []
    gpu_load: Dict[int, int] = {gid: 0 for gid in gpus}
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
                gpu_load[gpu] = max(0, gpu_load.get(gpu, 0) - 1)
        running = still_running

        while job_idx < len(jobs) and len(running) < effective_max_concurrent:
            dataset, base_cfg, ablation, seed = jobs[job_idx]
            gpu_id, gpu_rr = pick_gpu_with_slot_round_robin(gpus, gpu_load, max_per_gpu, gpu_rr)
            if gpu_id is None:
                break

            desc = f"{dataset}|{ablation['name']}|seed{seed}|gpu{gpu_id}"

            proc = launch_experiment(
                dataset_name=dataset,
                base_cfg=base_cfg,
                ablation=ablation,
                seed=seed,
                gpu_id=gpu_id,
                generate_diagnosis=args.generate_diagnosis,
                dry_run=args.dry_run,
                run_session=run_session,
            )

            if not args.dry_run and proc is not None:
                running.append((proc, gpu_id, desc))
                gpu_load[gpu_id] += 1

            job_idx += 1

        if running:
            time.sleep(max(1, args.poll_interval))
        else:
            if args.dry_run:
                break


if __name__ == "__main__":
    main()
