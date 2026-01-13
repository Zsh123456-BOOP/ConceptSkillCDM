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


# ========== 1) Best configs（保持不变：来自你的 BEST_CFG） ==========
BEST_CFG: Dict[str, Dict[str, Any]] = {
    "junyi": {
        "seed": 42,
        "batch_size": 256,
        "disable_soft_prototype": False,
        "dropout": 0.4,
        "early_stop_patience": 3,
        "epochs": 100,
        "exercise_dim": 128,
        "knowledge_dim": 128,
        "lambda_alpha": 0.0,
        "lambda_proto_div": 0.0,
        "lambda_proto_usage": 0.0,
        "lambda_sparse": 2,
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
        "lambda_sparse": 2,
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

# ========== 2) 默认 seeds（你也可以 CLI 传 --seeds 覆盖） ==========
DEFAULT_SEEDS = [42]


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

    # base cfg -> CLI args（避免把 seed/model_variant/ablate_* 透传进去）
    for k, v in base_cfg.items():
        if k in ["model_variant", "seed"]:
            continue
        if k.startswith("ablate_") or k.startswith("disable_") or k.startswith("enable_"):
            continue
        _append_arg(cmd, k, v)

    # overrides
    for k, v in ablation.get("overrides", {}).items():
        _append_arg(cmd, k, v)

    # flags (store_true)
    for k, v in ablation.get("flags", {}).items():
        if v:
            cmd.append(f"--{k}")

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
