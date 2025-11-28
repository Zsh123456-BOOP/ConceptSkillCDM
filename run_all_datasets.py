import argparse
import os
import subprocess
import sys
import time

from gpu_utils import get_gpu_memory_map

# 固定的数据集列表
DATASETS = ["assist_09", "assist_17", "junyi", "sample"]

# 可选的按数据集追加的命令行参数（当前为空）
DATASET_EXTRA_ARGS = {
    # "junyi": ["--batch_size", "4096", "--epochs", "80"],
}


def get_balanced_gpu(running_procs, candidates, memory_threshold=2000):
    """
    根据当前空闲显存和正在运行的任务数选择一块 GPU。
    返回最佳 gpu_id；若都不满足阈值则返回 None。
    """
    gpu_map = get_gpu_memory_map()
    if not gpu_map:
        return 0

    gpu_load = {gpu_id: 0 for gpu_id in candidates}
    for _, _, gpu_id in running_procs:
        if gpu_id in gpu_load:
            gpu_load[gpu_id] += 1

    best_gpu = None
    best_score = (999, -1)  # (任务数, -空闲显存)

    print("🔍 Load Balancing Status:")
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


def launch_dataset(dataset_name, gpu_id, base_python=sys.executable, seed=888):
    """
    Launch main.py for a given dataset on a specific GPU via CUDA_VISIBLE_DEVICES.
    """
    cmd = [
        base_python,
        "main.py",
        "--dataset", dataset_name,
        "--seed", str(seed),
    ]
    extra = DATASET_EXTRA_ARGS.get(dataset_name, [])
    cmd.extend(extra)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"🚀 Launching {dataset_name} on GPU {gpu_id}...")
    print(f"    Command: {' '.join(cmd)}")

    proc = subprocess.Popen(cmd, env=env)
    return proc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpus", type=str, default=None,
        help='Comma-separated GPU ids to use, e.g., "0,1,2,3". If not set, use all visible GPUs.'
    )
    parser.add_argument(
        "--max_jobs", type=int, default=3,
        help="Max concurrent training jobs (capped by number of allowed GPUs)."
    )
    parser.add_argument(
        "--memory_threshold", type=int, default=2000,
        help="Minimum free memory (MiB) required to schedule a new job on a GPU."
    )
    parser.add_argument(
        "--seed", type=int, default=888,
        help="Seed passed to each main.py run."
    )
    args = parser.parse_args()

    # 确定允许使用的 GPU 列表
    if args.gpus is None:
        try:
            import torch
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                allowed_gpus = list(range(num_gpus))
            else:
                allowed_gpus = [0]
        except ImportError:
            allowed_gpus = [0]
    else:
        allowed_gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]

    if not allowed_gpus:
        print("❌ No GPUs specified or detected. You can still run main.py directly with --device cpu.")
        return

    max_concurrent = min(args.max_jobs, len(allowed_gpus))
    print(f"🔥 Starting multi-dataset run with datasets={DATASETS}")
    print(f"   Allowed GPUs: {allowed_gpus}, Max concurrent jobs: {max_concurrent}\n")

    running_procs = []
    todo = list(DATASETS)
    COOLDOWN_SECONDS = 10

    while todo or running_procs:
        # 清理已完成的子进程
        for p_info in running_procs[:]:
            proc, ds_name, gpu_id = p_info
            if proc.poll() is not None:
                print(f"✅ Finished: {ds_name} on GPU {gpu_id}")
                running_procs.remove(p_info)

        # 若有空闲槽位则启动新任务
        while len(running_procs) < max_concurrent and todo:
            current_ds = todo[0]
            req_mem = args.memory_threshold * 4 if current_ds in ("junyi", "sample") else args.memory_threshold
            best_gpu = get_balanced_gpu(running_procs, allowed_gpus, memory_threshold=req_mem)
            if best_gpu is None:
                print("⏳ No sufficient GPU memory right now. Waiting...")
                time.sleep(10)
                break

            todo.pop(0)
            proc = launch_dataset(current_ds, best_gpu, seed=args.seed)
            running_procs.append((proc, current_ds, best_gpu))
            time.sleep(COOLDOWN_SECONDS)

        if running_procs:
            time.sleep(5)

    print("\n🎉 All datasets finished!")


if __name__ == "__main__":
    main()
