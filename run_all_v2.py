import argparse
import os
import subprocess
import sys
import time
from gpu_utils import get_gpu_memory_map

# ================================
# 全局默认参数（兜底）
# ================================
GLOBAL_DEFAULTS = {
    "batch_size": 1024,
    "epochs": 80,           # 统一略降，用 early stop 截断
    "patience": 15,
    "lr": 1e-3,
    "weight_decay": 1e-4,

    "dim_student": 64,
    "dim_item": 64,
    "dim_concept": 64,
    "dim_hidden": 64,
    "dim_skill": 16,

    "graph_dropout": 0.2,
    "graph_topk": 8,
    "dropout": 0.3,
    "softmin_beta": 5.0,

    "lambda_dag": 0.01,
    "lambda_sym": 0.10,
    "lambda_ent_dag": 0.02,
    "lambda_ent_sym": 0.02,
    "lambda_orth": 0.01,

    "model_name": "HybridKTCDM"
}

# ================================
# 针对 V3 模型按数据集微调
# ================================
DATASET_CONFIGS = {
    "sample": {
        "batch_size": 1024,
        "epochs": 60,
        "patience": 10,

        "dim_student": 32,
        "dim_item": 32,
        "dim_concept": 32,
        "dim_hidden": 32,
        "dim_skill": 8,

        "dropout": 0.5,
        "graph_dropout": 0.3,
        "graph_topk": 4,

        # 图正则整体降一个量级
        "lambda_ent_dag": 0.01,
        "lambda_ent_sym": 0.01,
        "lambda_dag":     0.005,
        "lambda_sym":     0.02,
    },

    "assist_09": {
        "batch_size": 512,
        "epochs": 100,
        "patience": 10,

        "dim_student": 32,
        "dim_item": 32,
        "dim_concept": 32,
        "dim_hidden": 32,
        "dim_skill": 8,

        "dropout": 0.5,
        "graph_dropout": 0.5,
        "graph_topk": 8,
        "weight_decay": 1e-3,

        # 原来是 ent_dag=0.05, ent_sym=0.02, dag=0.01, sym=0.10
        # 现在整体调低，让目标熵约束起到“轻微形状引导”，而不是死压成 one-hot
        "lambda_ent_dag": 0.01,
        "lambda_ent_sym": 0.005,
        "lambda_dag":     0.005,
        "lambda_sym":     0.02,
    },

    "assist_17": {
        "batch_size": 1024,
        "epochs": 100,
        "patience": 10,

        "dim_student": 128,
        "dim_item": 64,
        "dim_concept": 64,
        "dim_hidden": 128,
        "dim_skill": 16,

        "dropout": 0.3,
        "graph_dropout": 0.3,
        "graph_topk": 8,
        "lr": 5e-4,

        # 原来 ent_dag=0.02, ent_sym=0.01, dag=0.01, sym=0.10
        # 单概念数据本来图就次要，再降一点强度
        "lambda_ent_dag": 0.005,
        "lambda_ent_sym": 0.002,
        "lambda_dag":     0.005,
        "lambda_sym":     0.02,
    }
}


DATASETS = ["sample", "assist_09", "assist_17"]


def get_balanced_gpu(running_procs, candidates, memory_threshold=2000):
    gpu_map = get_gpu_memory_map()
    # [修复] 如果获取不到显存信息，返回 None 而不是 0
    if not gpu_map: 
        return None
        
    gpu_load = {gpu_id: 0 for gpu_id in candidates}
    for _, _, gpu_id in running_procs:
        if gpu_id in gpu_load: gpu_load[gpu_id] += 1
    best_gpu = None
    best_score = (999, -1)
    for gpu_id in candidates:
        if gpu_id not in gpu_map: continue
        free_mem = gpu_map[gpu_id]
        if free_mem < memory_threshold: continue
        score = (gpu_load[gpu_id], -free_mem)
        if score < best_score:
            best_score = score
            best_gpu = gpu_id
    return best_gpu

def launch_job(dataset_name, gpu_id):
    # 调用新的 test_full_model.py
    cmd = [sys.executable, "test_full_model.py", "--dataset", dataset_name]
    
    cfg = dict(GLOBAL_DEFAULTS)
    cfg.update(DATASET_CONFIGS.get(dataset_name, {}))
    
    for k, v in cfg.items():
        cmd.extend([f"--{k}", str(v)])
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print(f"\n🚀 Launching {dataset_name} on GPU {gpu_id} ...")
    proc = subprocess.Popen(cmd, env=env)
    return proc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("--max_jobs", type=int, default=3)
    parser.add_argument("--memory_threshold", type=int, default=2000)
    args = parser.parse_args()
    
    if args.gpus is None:
        import torch
        allowed_gpus = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [0]
    else:
        allowed_gpus = [int(x) for x in args.gpus.split(",")]
        
    max_concurrent = min(args.max_jobs, len(allowed_gpus))
    running_procs = [] 
    todo = list(DATASETS)
    
    print(f"🔥 Starting Scheduler. Queue: {todo}")
    
    while todo or running_procs:
        for p_info in running_procs[:]:
            proc, ds, gpu = p_info
            if proc.poll() is not None:
                print(f"✅ Finished: {ds} on GPU {gpu}")
                running_procs.remove(p_info)
        
        while len(running_procs) < max_concurrent and todo:
            current_ds = todo[0]
            req_mem = args.memory_threshold * 2 if current_ds == "assist_17" else args.memory_threshold
            best_gpu = get_balanced_gpu(running_procs, allowed_gpus, memory_threshold=req_mem)
            
            if best_gpu is None:
                print("⏳ Waiting for GPU memory (or nvidia-smi failed)...")
                time.sleep(10)
                break
            
            todo.pop(0)
            proc = launch_job(current_ds, best_gpu)
            running_procs.append((proc, current_ds, best_gpu))
            time.sleep(2)
            
        time.sleep(5)

if __name__ == "__main__":
    main()