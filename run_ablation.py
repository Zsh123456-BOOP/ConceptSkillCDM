#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Batch ablation runs for CognitiveDiagnosisModel (based on run_all_dataset.py best configs).

Default:
- datasets: assist_09,junyi
- seeds: only 42 (edit SEEDS list in code)
- 4 ablations (single-factor):
  1) full
  2) no_soft_proto        : --ablate_soft_prototype + --num_prototypes 0
  3) no_skill             : --ablate_skill_encoder
  4) no_concept_graph     : --ablate_concept_graph + --num_gnn_layers 0  (hard ablation)

Diagnosis default OFF (enable via --generate_diagnosis).
"""

import argparse
import os
import subprocess
import time


# ========== 1) Best configs (from your run_all_dataset.py; keep values unchanged) ==========
BEST_CFG = {
    "junyi": {
        "seed": 42,
        "batch_size": 512,
        "dropout": 0.4,
        "early_stop_patience": 3,
        "epochs": 100,
        "exercise_dim": 128,
        "knowledge_dim": 128,
        "lambda_alpha": 0.0,
        "lambda_proto_div": 0.0,
        "lambda_proto_usage": 0.0,
        "lambda_sparse": 0.05,
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
        "proto_lambda": 0.5,
        "proto_tau": 1.0,
        "save_interval": 10,
        "skill_dim": 2,
        "weight_decay": 0.001,
        "use_personal_graph": False,
        "model_variant": "gpd_base",
        "exercise_l2_lambda": 5e-5,
    },

    "assist_09": {
        "seed": 42,
        "batch_size": 128,
        "dropout": 0.20,
        "early_stop_patience": 3,
        "epochs": 100,
        "exercise_dim": 64,
        "knowledge_dim": 64,
        "lambda_alpha": 0.0,
        "lambda_proto_div": 0.0,
        "lambda_proto_usage": 0.0,
        "lambda_sparse": 0.10,
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
        "proto_lambda": 0.1,
        "proto_tau": 1.0,
        "save_interval": 10,
        "skill_dim": 64,
        "weight_decay": 1e-5,
        "use_personal_graph": False,
        "model_variant": "gpd_base",
        "exercise_l2_lambda": 5e-5,
    },
}

# ========== 2) Default seeds (edit here if you want more) ==========
SEEDS = [42]

# ========== 3) Ablations ==========
# flags: store_true flags passed to main.py
# overrides: key-value args passed as --k v (hard override)
ABLATIONS = [
    {"name": "full", "flags": {}, "overrides": {}},

    # hard prototype removal: disable path + remove prototype params
    {"name": "no_soft_proto",
     "flags": {"ablate_soft_prototype": True},
     "overrides": {"num_prototypes": 0}},

    # MF branch off
    {"name": "no_skill",
     "flags": {"ablate_skill_encoder": True},
     "overrides": {}},

    # hard graph removal: disable graph + remove GNN layers to avoid any per-concept transform
    {"name": "no_concept_graph",
     "flags": {"ablate_concept_graph": True},
     "overrides": {"num_gnn_layers": 0}},
]


def _append_arg(cmd, k, v):
    """Append a CLI arg in a safe way for our current argparse contract."""
    if isinstance(v, bool):
        if v:
            cmd.append(f"--{k}")
        return
    cmd.extend([f"--{k}", str(v)])


def launch_experiment(dataset_name, base_cfg, ablation, seed, gpu_id, generate_diagnosis):
    abl_name = ablation["name"]
    tag = f"{dataset_name}_ablation_{abl_name}_seed{seed}"

    save_dir = os.path.join("checkpoints", tag)
    log_dir = os.path.join("logs", tag)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    model_variant = f"gpd_base_{abl_name}"

    cmd = [
        "python", "main.py",
        "--dataset_name", dataset_name,
        "--model_variant", model_variant,
        "--save_dir", save_dir,
        "--log_dir", log_dir,
        "--seed", str(seed),
        "--generate_diagnosis", "True" if generate_diagnosis else "False",
    ]

    # 1) base cfg -> CLI args (skip seed/model_variant; ablate flags handled only by ablation)
    for k, v in base_cfg.items():
        if k in ["model_variant", "seed"]:
            continue
        if k.startswith("ablate_") or k == "disable_soft_prototype":
            continue  # avoid any accidental base ablation
        _append_arg(cmd, k, v)

    # 2) ablation overrides first (non-bool)
    for k, v in ablation.get("overrides", {}).items():
        _append_arg(cmd, k, v)

    # 3) ablation flags (store_true)
    for k, v in ablation.get("flags", {}).items():
        if v:
            cmd.append(f"--{k}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[LAUNCH] dataset={dataset_name}, ablation={abl_name}, seed={seed}, gpu={gpu_id}")
    print("         CMD:", " ".join(cmd))
    return subprocess.Popen(cmd, env=env)


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments for CognitiveDiagnosisModel.")
    parser.add_argument("--datasets", type=str, default="assist_09,junyi")
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--max_concurrent", type=int, default=4)
    parser.add_argument("--generate_diagnosis", action="store_true")
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    max_concurrent = max(1, args.max_concurrent)

    print(f"Datasets: {datasets}")
    print(f"GPUs: {gpus}, max_concurrent={max_concurrent}")
    print(f"Seeds (edit in code): {SEEDS}")
    print(f"Ablations: {[a['name'] for a in ABLATIONS]}")
    print(f"Generate diagnosis: {args.generate_diagnosis}")

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
    gpu_rr = 0

    while job_idx < len(jobs) or running:
        still_running = []
        for proc, gpu, desc in running:
            ret = proc.poll()
            if ret is None:
                still_running.append((proc, gpu, desc))
            else:
                print(f"[DONE] {desc} on gpu {gpu} exited with code {ret}")
        running = still_running

        while job_idx < len(jobs) and len(running) < max_concurrent:
            dataset, ablation, seed = jobs[job_idx]
            base_cfg = BEST_CFG[dataset]

            gpu_id = gpus[gpu_rr % len(gpus)]
            gpu_rr += 1

            desc = f"{dataset}|{ablation['name']}|seed{seed}"
            proc = launch_experiment(dataset, base_cfg, ablation, seed, gpu_id, args.generate_diagnosis)
            running.append((proc, gpu_id, desc))
            job_idx += 1

        if running:
            time.sleep(10)


if __name__ == "__main__":
    main()
