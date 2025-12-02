import subprocess
import time
import os
import sys

# ========================
# 每个数据集的默认配置（只包含当前 config.py 中的参数）
# ========================
DATASET_CONFIGS = {
    "assist_09": {
        "batch_size": 512,
        "epochs": 100,
        "patience": 10,

        "dim_emb": 64,
        "dim_skill": 4,

        "lr": 0.001,
        "weight_decay": 1e-5,

        "lambda_dag": 0.5,
        "lambda_sparse": 0.01,
        "lambda_hsic": 0.1,

        "min_stu_interactions": 10,
        "min_exer_interactions": 10,
        "min_poison_count": 10,

        "low_quantile": 0.33,
        "high_quantile": 0.67,
    },
    "assist_17": {
        "batch_size": 128,
        "epochs": 100,
        "patience": 10,

        "dim_emb": 64,
        "dim_skill": 4,

        "lr": 0.001,
        "weight_decay": 1e-5,

        "lambda_dag": 0.5,
        "lambda_sparse": 0.01,
        "lambda_hsic": 0.1,

        "min_stu_interactions": 10,
        "min_exer_interactions": 10,
        "min_poison_count": 10,

        "low_quantile": 0.33,
        "high_quantile": 0.67,
    },
    "junyi": {
        "batch_size": 2048,
        "epochs": 100,
        "patience": 10,

        # Junyi 可以设大一点 embedding
        "dim_emb": 128,
        "dim_skill": 4,

        "lr": 0.001,
        "weight_decay": 1e-5,

        "lambda_dag": 0.5,
        "lambda_sparse": 0.01,
        "lambda_hsic": 0.1,

        "min_stu_interactions": 10,
        "min_exer_interactions": 10,
        "min_poison_count": 10,

        "low_quantile": 0.33,
        "high_quantile": 0.67,
    },
}

# 可用 GPU & 并发控制
ALLOWED_GPUS = [0, 1, 2, 3]
MAX_CONCURRENT_JOBS = min(3, len(ALLOWED_GPUS))
COOLDOWN_SECONDS = 30


def get_balanced_gpu(running_procs, candidates, memory_threshold=2000):
    """
    简单的负载均衡策略：
    - 尽量让 job 数量平衡
    - 在满足显存阈值的前提下，优先选择“当前任务数少 + 剩余显存多”的 GPU
    """
    from gpu_utils import get_gpu_memory_map
    gpu_map = get_gpu_memory_map()
    if not gpu_map:
        # 如果无法获取 GPU 信息，就退化成用 0 号卡
        return 0

    gpu_load = {gpu_id: 0 for gpu_id in candidates}
    for _, _, gpu_id in running_procs:
        if gpu_id in gpu_load:
            gpu_load[gpu_id] += 1

    best_gpu = None
    best_score = (999, -1)  # (jobs, -free_mem) 越小越好

    print(f"🔍 Load Balancing Status:")
    for gpu_id in candidates:
        if gpu_id not in gpu_map:
            continue
        free_mem = gpu_map[gpu_id]
        current_jobs = gpu_load[gpu_id]
        print(f"   - GPU {gpu_id}: Jobs={current_jobs}, Free={free_mem}MiB")

        if free_mem < memory_threshold:
            continue

        score = (current_jobs, -free_mem)
        if score < best_score:
            best_score = score
            best_gpu = gpu_id

    return best_gpu


def run():
    print(f"🔥 Starting multi-dataset run with datasets={list(DATASET_CONFIGS.keys())}")
    print(f"   Allowed GPUs: {ALLOWED_GPUS}, Max concurrent jobs: {MAX_CONCURRENT_JOBS}\n")

    running_procs = []
    todo_datasets = list(DATASET_CONFIGS.keys())

    while len(todo_datasets) > 0 or len(running_procs) > 0:

        # 1. 检查已经结束的进程
        for p_info in running_procs[:]:
            proc, ds_name, gpu_id = p_info
            if proc.poll() is not None:
                print(f"✅ Finished: {ds_name} on GPU {gpu_id}")
                running_procs.remove(p_info)

        # 2. 尝试启动新的数据集任务
        while len(running_procs) < MAX_CONCURRENT_JOBS and len(todo_datasets) > 0:
            current_ds = todo_datasets[0]

            # Junyi 需要更多显存
            req_mem = 10000 if current_ds == "junyi" else 2000

            best_gpu = get_balanced_gpu(running_procs, ALLOWED_GPUS, memory_threshold=req_mem)
            if best_gpu is None:
                print(f"⏳ No sufficient GPU memory for {current_ds}. Waiting...")
                time.sleep(20)
                break

            # 从待执行队列中弹出
            todo_datasets.pop(0)
            config = DATASET_CONFIGS[current_ds]

            print(f"🚀 Launching {current_ds} on GPU {best_gpu} ...")

            cmd = [
                sys.executable, "main.py",
                "--dataset", current_ds,
                "--seed", "888",

                "--batch_size", str(config["batch_size"]),
                "--epochs", str(config["epochs"]),
                "--patience", str(config["patience"]),

                "--dim_emb", str(config["dim_emb"]),
                "--dim_skill", str(config["dim_skill"]),

                "--lr", str(config["lr"]),
                "--weight_decay", str(config["weight_decay"]),

                "--lambda_dag", str(config["lambda_dag"]),
                "--lambda_sparse", str(config["lambda_sparse"]),
                "--lambda_hsic", str(config["lambda_hsic"]),

                "--min_stu_interactions", str(config["min_stu_interactions"]),
                "--min_exer_interactions", str(config["min_exer_interactions"]),
                "--min_poison_count", str(config["min_poison_count"]),

                "--low_quantile", str(config["low_quantile"]),
                "--high_quantile", str(config["high_quantile"]),

                # 用 tag 标记这是 DisentangledCDM full run，方便在 CSV 里区分
                "--tag", f"{current_ds}_disentangled_full",
            ]

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(best_gpu)

            print(f"    Command: {' '.join(cmd)}")
            print(f"    CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")

            try:
                proc = subprocess.Popen(cmd, env=env)
                running_procs.append((proc, current_ds, best_gpu))
                # 冷却一下，避免瞬间起太多进程
                time.sleep(COOLDOWN_SECONDS)
            except Exception as e:
                print(f"❌ Failed to launch {current_ds}: {e}")

        # 如果还有任务在跑，就休眠一会儿再检查
        if len(running_procs) > 0:
            time.sleep(10)

    print("\n🎉 All tasks finished! Check results/all_datasets_results.csv")


if __name__ == "__main__":
    run()
