#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
消融实验运行脚本 (Ablation Study Runner)

基于当前最优超参 (BEST_CFG)，对模型各个模块进行消融测试。
默认消融项：
1. Full (完整模型)
2. w/o Soft Proto (去除软原型)
3. w/o Skill (去除技巧编码)
4. w/o Exer Graph (去除习题图传播)

用法示例：
    python run_ablation_experiments.py
    python run_ablation_experiments.py --gpus 0,1
    python run_ablation_experiments.py --datasets junyi --max_concurrent 4
"""

import argparse
import os
import subprocess
import time
import copy

# ===== 最佳配置 (与 run_all_datasets.py 保持一致) =====
BEST_CFG = {
    "junyi": {
        "seed": 42,
        "batch_size": 512,
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
        "use_personal_graph": False,
        "model_variant": "gpd_base",
    },
    "assist_09": {
        "seed": 42,
        "batch_size": 128,
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
        "proto_lambda": 0.5,
        "proto_tau": 1.0,
        "save_interval": 10,
        "skill_dim": 2,
        "weight_decay": 1e-5,
        "use_personal_graph": False,
        "model_variant": "gpd_base",
    },
}

# ===== 定义消融变体 =====
# 键名会作为文件夹后缀，值是需要覆盖或添加到命令行的参数
ABLATION_VARIANTS = {
    "Full": {
        # 完整模型，不添加任何额外的 ablate 参数
        "ablate_soft_prototype": False,
        "ablate_skill_encoder": False,
        "ablate_exercise_graph": False,
    },
    "No_SoftProto": {
        "ablate_soft_prototype": True,  # 去除软原型
        "ablate_skill_encoder": False,
        "ablate_exercise_graph": False,
    },
    "No_Skill": {
        "ablate_soft_prototype": False,
        "ablate_skill_encoder": True,   # 去除技巧
        "ablate_exercise_graph": False,
    },
    "No_ExerGraph": {
        "ablate_soft_prototype": False,
        "ablate_skill_encoder": False,
        "ablate_exercise_graph": True,  # 去除习题图
    },
}


def launch_experiment(dataset_name, base_cfg, variant_name, variant_args, gpu_id):
    """
    启动单个消融实验子进程
    """
    # 构造保存路径：checkpoints/{dataset}_ablation_{variant_name}
    tag = f"{dataset_name}_ablation_{variant_name}"
    save_dir = os.path.join("checkpoints", tag)
    log_dir = os.path.join("logs", tag)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 融合配置：先拷贝 base，再用 variant_args 覆盖
    # 注意：main.py 里 ablate_* 参数是 store_true，所以 True 时传 flag，False 时不传
    final_cfg = copy.deepcopy(base_cfg)
    final_cfg.update(variant_args)

    cmd = [
        "python",
        "main.py",
        "--dataset_name",
        dataset_name,
        "--model_variant",
        f"ablation_{variant_name}", # 在日志里标记变体名
        "--save_dir",
        save_dir,
        "--log_dir",
        log_dir,
    ]

    # 将配置字典转换为命令行参数
    for k, v in final_cfg.items():
        if k == "model_variant":
            continue  # 已经在上面处理过了

        # 对于 bool 类型参数的处理逻辑
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
            # if False: 不添加任何 flag，即保持默认 (main.py 中默认通常是 False)
        else:
            cmd.append(f"--{k}")
            cmd.append(str(v))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[LAUNCH] Dataset={dataset_name} | Variant={variant_name} | GPU={gpu_id}")
    # print("        CMD:", " ".join(cmd)) # 调试时可打开
    return subprocess.Popen(cmd, env=env)


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies for assist_09 & junyi.")
    parser.add_argument(
        "--datasets",
        type=str,
        default="assist_09,junyi",
        help="逗号分隔的数据集名称",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1",
        help="逗号分隔的 GPU 编号",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=2,
        help="最大并行实验数",
    )
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    max_concurrent = max(1, args.max_concurrent)

    print(f"Datasets: {datasets}")
    print(f"Ablations: {list(ABLATION_VARIANTS.keys())}")
    print(f"GPUs: {gpus}, Max Concurrent: {max_concurrent}")

    # 构建任务队列：(dataset_name, variant_name, variant_args)
    jobs = []
    for dataset in datasets:
        if dataset not in BEST_CFG:
            print(f"Warning: {dataset} not found in BEST_CFG, skipping.")
            continue
        
        for var_name, var_args in ABLATION_VARIANTS.items():
            jobs.append({
                "dataset": dataset,
                "variant_name": var_name,
                "variant_args": var_args
            })

    print(f"Total ablation experiments: {len(jobs)}")

    running = []
    job_idx = 0
    gpu_rr = 0  # Round-Robin 指针

    # 主循环
    while job_idx < len(jobs) or running:
        # 1. 检查并清理已完成的进程
        new_running = []
        for proc, gpu, desc in running:
            ret = proc.poll()
            if ret is None:
                new_running.append((proc, gpu, desc))
            else:
                print(f"[DONE] {desc} on gpu {gpu} finished (code {ret})")
        running = new_running

        # 2. 提交新任务 (只要 GPU 有空闲且未达最大并发)
        while job_idx < len(jobs) and len(running) < max_concurrent:
            job = jobs[job_idx]
            dataset = job["dataset"]
            var_name = job["variant_name"]
            var_args = job["variant_args"]

            # 获取基础配置
            base_cfg = BEST_CFG[dataset]
            
            # 分配 GPU
            gpu_id = gpus[gpu_rr % len(gpus)]
            gpu_rr += 1

            desc = f"{dataset}|{var_name}"
            
            # 启动
            try:
                proc = launch_experiment(dataset, base_cfg, var_name, var_args, gpu_id)
                running.append((proc, gpu_id, desc))
                job_idx += 1
            except Exception as e:
                print(f"[ERROR] Failed to launch {desc}: {e}")
                job_idx += 1 # 跳过错误的，继续下一个

        # 3. 避免空转，稍作等待
        if running:
            time.sleep(5)

    print("All ablation experiments completed.")

if __name__ == "__main__":
    main()