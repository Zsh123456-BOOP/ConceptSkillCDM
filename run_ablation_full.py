#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
全模块消融脚本：
- 以当前 BEST_CFG 为基线
- 对 soft prototype / skill encoder / exercise graph / student-factor 做系统消融
- 便于判断哪些模块可以从模型中删掉

用法示例：
  python run_ablation_full.py --datasets assist_09,junyi --gpus 0,1 --max_concurrent 2
"""

import argparse
import os
import subprocess
import time


# ===== 当前最优配置（你给的 BEST_CFG） =====
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
        "ablate_exercise_graph": False,
        "ablate_skill_encoder": False,
        "ablate_soft_prototype": False,
        "use_personal_graph": False,  # CSV 里是 0
        # 其他
        "model_variant": "gpd_base",
        # 学生侧低秩因子（如果在 main/config 里用到）
        "student_factor_rank": 4,
        "disable_student_factor": False,
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
        "ablate_exercise_graph": False,
        "ablate_skill_encoder": False,
        "ablate_soft_prototype": False,
        "use_personal_graph": False,  # CSV 里是 0
        "model_variant": "gpd_base",
        "student_factor_rank": 4,
        "disable_student_factor": False,
    },
}


# ===== 模块消融组合 =====
# 尽量控制数量：每个数据集 6 组
ABLATIONS = {
    # 基线：所有模块按 BEST_CFG 全开
    "full": {
        # 不额外加 flag，完全使用 BEST_CFG 中的配置
    },

    # 只关 soft prototype（看原型有没有贡献）
    "no_proto": {
        "ablate_soft_prototype": True,
    },

    # 只关 skill encoder（猜测/失误这块是否有用）
    "no_skill": {
        "ablate_skill_encoder": True,
    },

    # 只关 exercise graph（题目侧 GNN 是否有贡献）
    "no_exgraph": {
        "ablate_exercise_graph": True,
    },

    # 只关学生侧低秩因子（student-factor）
    "no_stufactor": {
        "disable_student_factor": True,
        # student_factor_rank 可以保留原值，但为了明确，也可以显式置 0
        "student_factor_rank": 0,
    },

    # 极简模型：把上面几个全部关掉，只保留“全局概念图 + 学生知识状态 + IRT 头”
    "minimal": {
        "ablate_soft_prototype": True,
        "ablate_skill_encoder": True,
        "ablate_exercise_graph": True,
        "disable_student_factor": True,
        "student_factor_rank": 0,
        "use_personal_graph": False,  # 保险起见，强制关掉个性化图
    },
}


def launch_experiment(dataset_name, base_cfg, ablation_name, overrides, gpu_id):
    """
    构造 main.py 命令并启动一个子进程。
    - dataset_name: "assist_09" / "junyi"
    - base_cfg: BEST_CFG[dataset_name]
    - ablation_name: "full" / "no_proto" / ...
    - overrides: ABLATIONS[ablation_name]
    """
    tag = f"{dataset_name}_ablation_{ablation_name}"
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
        f"ablation_{ablation_name}",
        "--save_dir",
        save_dir,
        "--log_dir",
        log_dir,
    ]

    # 1) 先灌入该数据集的 BEST_CFG（数值/标量参数）
    for k, v in base_cfg.items():
        if k == "model_variant":
            # model_variant 用我们自己的 ablation_xx，不用 BEST_CFG 里的
            continue

        # bool flag：只在 True 时加上（例如 no_cuda / use_personal_graph）
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd.append(f"--{k}")
            cmd.append(str(v))

    # 2) 再覆盖本次消融的特定参数（会覆盖 BEST_CFG）
    for k, v in overrides.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")  # bool 参数使用 flag 形式
            # 如果是 False，保持默认/基线，不额外传参
        else:
            cmd.append(f"--{k}")
            cmd.append(str(v))

    # 3) 统一 seed，方便对比（也可以让 BEST_CFG 决定，这里就强制一下）
    cmd += ["--seed", str(base_cfg.get("seed", 42))]

    # 4) 绑定到指定 GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[LAUNCH] dataset={dataset_name}, ablation={ablation_name}, gpu={gpu_id}")
    print("         CMD:", " ".join(cmd))
    return subprocess.Popen(cmd, env=env)


def main():
    parser = argparse.ArgumentParser(description="Run full-module ablations for CognitiveDiagnosisModel.")
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
        help="Comma-separated GPU ids to use, e.g. '0,1'",
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
    print(f"Ablations: {list(ABLATIONS.keys())}")

    # 任务队列：[(dataset, ablation_name)]
    jobs = []
    for dataset in datasets:
        if dataset not in BEST_CFG:
            raise ValueError(f"Dataset '{dataset}' not in BEST_CFG (当前只配置了: {list(BEST_CFG.keys())})")
        for ablation_name in ABLATIONS.keys():
            jobs.append((dataset, ablation_name))

    print(f"Total experiments: {len(jobs)}")

    running = []
    job_idx = 0
    gpu_rr = 0

    while job_idx < len(jobs) or running:
        # 1) 清理已结束的进程
        new_running = []
        for proc, gpu, desc in running:
            ret = proc.poll()
            if ret is None:
                new_running.append((proc, gpu, desc))
            else:
                print(f"[DONE] {desc} on gpu {gpu} exited with code {ret}")
        running = new_running

        # 2) 若有空余 slot，就继续提交新任务
        while job_idx < len(jobs) and len(running) < max_concurrent:
            dataset, ablation_name = jobs[job_idx]
            base_cfg = BEST_CFG[dataset]
            overrides = ABLATIONS[ablation_name]

            gpu_id = gpus[gpu_rr % len(gpus)]
            gpu_rr += 1

            desc = f"{dataset}|{ablation_name}"
            proc = launch_experiment(dataset, base_cfg, ablation_name, overrides, gpu_id)
            running.append((proc, gpu_id, desc))
            job_idx += 1

        if running:
            time.sleep(10)


if __name__ == "__main__":
    main()
