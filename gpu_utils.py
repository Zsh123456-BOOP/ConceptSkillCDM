import os
import subprocess
from typing import Dict, List, Optional, Tuple


def parse_int_csv(text: str) -> List[int]:
    """
    Parse comma-separated integer list.
    Example: "0,1,2" -> [0, 1, 2]
    """
    out: List[int] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return out


def parse_gpu_ids(text: str) -> List[int]:
    """Alias for GPU id parsing."""
    return parse_int_csv(text)


def calc_effective_max_concurrent(max_concurrent: int, gpus: List[int], max_per_gpu: int) -> int:
    """
    Effective concurrency is bounded by both global limit and per-GPU slot count.
    """
    max_concurrent = max(1, int(max_concurrent))
    max_per_gpu = max(1, int(max_per_gpu))
    return min(max_concurrent, len(gpus) * max_per_gpu)


def get_gpu_memory_map() -> Dict[int, int]:
    """
    获取当前所有 GPU 的剩余显存 (单位: MiB)
    返回: {gpu_id: free_memory}
    """
    try:
        # 调用 nvidia-smi 查询 ID 和 剩余显存
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,nounits,noheader"],
            encoding="utf-8",
        )
        
        gpu_memory = {}
        for line in result.strip().split("\n"):
            if not line:
                continue
            parts = line.split(",")
            idx = int(parts[0].strip())
            free_mem = int(parts[1].strip())
            gpu_memory[idx] = free_mem
            
        return gpu_memory
    except FileNotFoundError:
        print("[GPU] nvidia-smi not found. Assuming CPU only or manual assignment.")
        return {0: 0}  # Fallback
    except Exception as e:
        print(f"[GPU] Error reading GPU stats: {e}")
        return {}


def get_best_gpu(candidates: Optional[List[int]] = None, memory_threshold: int = 2000) -> Optional[int]:
    """
    从候选列表中选择显存剩余最多的 GPU
    :param candidates: 允许使用的 GPU ID 列表，如 [0, 1]。如果为 None，则检查所有 GPU。
    :param memory_threshold: 最小需要的显存 (MiB)，默认 2GB。
    :return: best_gpu_id (int)
    """
    gpu_map = get_gpu_memory_map()
    
    if not gpu_map:
        return 0  # Default fallback
    
    best_gpu = None
    max_free = -1
    
    # 筛选候选 GPU
    target_gpus = candidates if candidates is not None else list(gpu_map.keys())
    
    # 打印当前状态供调试
    status_msg = " | ".join([f"GPU {k}: {v}MiB free" for k, v in gpu_map.items() if k in target_gpus])
    if status_msg:
        print(f"[GPU] Status: [{status_msg}]")
    
    for gpu_id in target_gpus:
        if gpu_id not in gpu_map:
            continue
            
        free_mem = gpu_map[gpu_id]
        
        # 找剩余显存最大的那个
        if free_mem > max_free:
            max_free = free_mem
            best_gpu = gpu_id
            
    if max_free < memory_threshold:
        print(f"[GPU] Warning: all candidate GPUs are busy (max free: {max_free} MiB).")
        # 这里可以选择让外部循环继续等待，或者返回最不忙的那个
    
    return best_gpu


def get_best_gpus(
    n: int = 2,
    candidates: Optional[List[int]] = None,
    memory_threshold: int = 2000,
) -> List[int]:
    """
    从候选列表中选择 N 张显存剩余最多的 GPU
    :param n: 需要的 GPU 数量
    :param candidates: 允许使用的 GPU ID 列表
    :param memory_threshold: 最小需要的显存 (MiB)
    :return: list of gpu_ids (按显存从大到小排列)
    """
    gpu_map = get_gpu_memory_map()
    
    if not gpu_map:
        return [0]
    
    target_gpus = candidates if candidates is not None else list(gpu_map.keys())
    
    # 按显存从大到小排序
    sorted_gpus = sorted(
        [(gpu_id, gpu_map.get(gpu_id, 0)) for gpu_id in target_gpus],
        key=lambda x: x[1],
        reverse=True
    )
    
    # 取前 N 个
    selected = [gpu_id for gpu_id, mem in sorted_gpus[:n] if mem >= memory_threshold]
    
    if len(selected) < n:
        print(f"[GPU] Warning: only {len(selected)} GPUs meet memory threshold, requested {n}.")
        # 如果不够，补充其他 GPU
        for gpu_id, mem in sorted_gpus:
            if gpu_id not in selected:
                selected.append(gpu_id)
            if len(selected) >= n:
                break
    
    return selected


def pick_gpus_for_job(
    required: int,
    all_gpus: List[int],
    gpu_load: Dict[int, int],
    max_per_gpu: int,
    memory_threshold: int = 2000,
) -> Optional[List[int]]:
    """
    Pick multiple GPUs for one job:
    - only choose GPUs that still have free slots
    - prefer GPUs with higher free memory
    """
    required = max(1, int(required))
    available = [gid for gid in all_gpus if gpu_load.get(gid, 0) < max_per_gpu]
    if len(available) < required:
        return None

    selected = get_best_gpus(n=required, candidates=available, memory_threshold=memory_threshold)
    if len(selected) < required:
        selected = available[:required]
    return selected


def pick_gpu_with_slot_round_robin(
    gpus: List[int],
    gpu_load: Dict[int, int],
    max_per_gpu: int,
    start_idx: int,
) -> Tuple[Optional[int], int]:
    """
    Pick one GPU with free slot using round-robin order.
    """
    if not gpus:
        return None, start_idx

    n = len(gpus)
    for offset in range(n):
        idx = (start_idx + offset) % n
        gid = gpus[idx]
        if gpu_load.get(gid, 0) < max_per_gpu:
            return gid, (idx + 1) % n

    return None, start_idx


def configure_main_process_gpus(
    *,
    gpus: List[int],
    num_gpus: int,
    memory_threshold: int = 2000,
) -> List[int]:
    """
    Select GPUs for a single main.py process and set CUDA_VISIBLE_DEVICES.
    Returns selected physical GPU ids.
    """
    if not gpus:
        raise ValueError("No GPUs provided.")

    num_gpus = max(1, int(num_gpus))
    if num_gpus > len(gpus):
        raise ValueError(f"Requested num_gpus={num_gpus} but only {len(gpus)} candidate GPUs provided: {gpus}")

    selected = get_best_gpus(n=num_gpus, candidates=gpus, memory_threshold=memory_threshold)
    if len(selected) < num_gpus:
        selected = gpus[:num_gpus]

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in selected)
    return selected
