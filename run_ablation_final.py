#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
三模块全排列消融脚本（最终版）：
- Soft Prototype（软原型）        -> ablate_soft_prototype
- Skill Encoder（技巧编码器）     -> ablate_skill_encoder
- G-PDS 个性化关系图             -> use_personal_graph + λ_sparse_personal + λ_alpha

设计：
- 对每个数据集（assist_09, junyi），在当前精简后的最终模型上，
  用 BEST_CFG 作为基线超参，做这三个模块的 2^3 = 8 种 on/off 组合。
- 组合命名：abl_p{0/1}s{0/1}g{0/1}，例如：
    abl_p1s1g0 ：soft proto ON, skill ON, G-PDS OFF（基线）
    abl_p0s1g0 ：soft proto OFF, skill ON, G-PDS OFF
    abl_p1s0g1 ：soft proto ON, skill OFF, G-PDS ON
    abl_p0s0g0 ：三个全关（最简模型）
    ...

使用方式示例：
    python run_ablation_final.py \
        --datasets assist_09,junyi \
        --gpus 0,1 \
        --max_concurrent 2
"""

import argparse
import os
import subprocess
import time


# ===== 你给的当前最优配置（直接拿来用做基线） =====
BEST_CFG = {
    "junyi": {
        # 核心训练超参
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
        # 消融/模块开关相关（基线全开）
        "ablate_skill_encoder": False,
        "ablate_soft_prototype": False,
        "use_personal_graph": False,  # G-PDS 关
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
        "learning_rate": 3e-4,  # assist_09 的最佳行为 0.0003
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
        "use_personal_graph": False,  # G-PDS 关
        "model_variant": "gpd_base",
    },
}

# ===== G-PDS 默认强度（可以之后根据实验结果微调） =====
# ON 时使用的 λ_sparse_personal / λ_alpha；OFF 时统一置 0.
GPD_REG_CFG = {
    "assist_09": {
        "lambda_sparse_personal": 1e-4,
        "lambda_alpha": 0.01,
    },
    "junyi": {
        "lambda_sparse_personal": 1e-4,
        "lambda_alpha": 0.01,
    },
}


def build_variants(dataset_name):
    """
    构造 8 组排列组合：
    p: soft prototype (1=ON, 0=OFF)
    s: skill encoder   (1=ON, 0=OFF)
    g: G-PDS           (1=ON, 0=OFF)
    """
    gpd_on = GPD_REG_CFG[dataset_name]
    gpd_off = {"lambda_sparse_personal": 0.0, "lambda_alpha": 0.0}

    variants = {}

    for p in [0, 1]:
        for s in [0, 1]:
            for g in [0, 1]:
                name = f"abl_p{p}s{s}g{g}"

                # soft proto on/off
                # main.py 中：use_soft_prototype = not disable_soft_prototype and not ablate_soft_prototype
                # 这里统一通过 ablate_soft_prototype 控制（disable_soft_prototype 保持 False）
                ablate_soft_proto = (p == 0)

                # skill encoder on/off
                ablate_skill = (s == 0)

                # G-PDS on/off
                if g == 1:
                    gpd_flags = {
                        "use_personal_graph": True,
                        "lambda_sparse_personal": gpd_on["lambda_sparse_personal"],
                        "lambda_alpha": gpd_on["lambda_alpha"],
                    }
                else:
                    gpd_flags = {
                        "use_personal_graph": False,
                        "lambda_sparse_personal": gpd_off["lambda_sparse_personal"],
                        "lambda_alpha": gpd_off["lambda_alpha"],
                    }

                variants[name] = {
                    "ablate_soft_prototype": ablate_soft_proto,
                    "ablate_skill_encoder": ablate_skill,
                    **gpd_flags,
                }

    return variants


def launch_experiment(dataset_name, best_cfg, variant_name, variant_flags, gpu_id):
    """
    拼 main.py 命令并启动子进程：
    - 用 BEST_CFG 作为基线
    - variant_flags 覆盖 ablation 相关开关
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

    # 1) 固定 BEST_CFG 里的训练超参（跳过我们要自己控制的几个 key）
    skip_keys = {
        "ablate_soft_prototype",
        "ablate_skill_encoder",
        "use_personal_graph",
        "lambda_sparse_personal",
        "lambda_alpha",
        "model_variant",  # 已单独设置
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

    # 2) 覆盖当前消融变体的模块开关 & G-PDS 正则
    for k, v in variant_flags.items():
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(v)])

    # 3) 统一 seed（虽然 BEST_CFG 已经带了，这里再强制一次也没坏处）
    cmd.extend(["--seed", str(best_cfg.get("seed", 42))])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[LAUNCH] dataset={dataset_name}, variant={variant_name}, gpu={gpu_id}")
    print("         CMD:", " ".join(cmd))
    return subprocess.Popen(cmd, env=env)


def main():
    parser = argparse.ArgumentParser(description="Run full 3-module ablation (soft proto / skill / G-PDS).")
    parser.add_argument(
        "--datasets",
        type=str,
        default="assist_09,junyi",
        help="Comma-separated dataset names, e.g. 'assist_09,junyi'",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1",
        help="Comma-separated GPU ids to use, e.g. '0,2'",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=2,
        help="Maximum concurrent experiments.",
    )
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    max_concurrent = args.max_concurrent

    print(f"Datasets: {datasets}")
    print(f"GPUs: {gpus}, max_concurrent={max_concurrent}")

    # 构造任务列表：[(dataset, variant_name)]
    jobs = []
    for dataset in datasets:
        if dataset not in BEST_CFG:
            raise ValueError(f"Dataset '{dataset}' not in BEST_CFG.")
        variants = build_variants(dataset)
        print(f"[{dataset}] variants: {list(variants.keys())}")
        for vname in variants.keys():
            jobs.append((dataset, vname))

    print(f"Total experiments: {len(jobs)}")

    running = []
    job_idx = 0
    gpu_rr = 0

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
            dataset, variant_name = jobs[job_idx]
            base_cfg = BEST_CFG[dataset]
            variants = build_variants(dataset)
            v_flags = variants[variant_name]

            gpu_id = gpus[gpu_rr % len(gpus)]
            gpu_rr += 1

            desc = f"{dataset}|{variant_name}"
            proc = launch_experiment(dataset, base_cfg, variant_name, v_flags, gpu_id)
            running.append((proc, gpu_id, desc))
            job_idx += 1

        if running:
            time.sleep(10)


if __name__ == "__main__":
    main()
