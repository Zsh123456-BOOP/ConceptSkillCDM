import subprocess
import sys
import re
import time

def get_gpu_memory_map():
    """
    获取当前所有 GPU 的剩余显存 (单位: MiB)
    返回: {gpu_id: free_memory}
    """
    try:
        # 调用 nvidia-smi 查询 ID 和 剩余显存
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        
        gpu_memory = {}
        for line in result.strip().split('\n'):
            if not line: continue
            parts = line.split(',')
            idx = int(parts[0].strip())
            free_mem = int(parts[1].strip())
            gpu_memory[idx] = free_mem
            
        return gpu_memory
    except FileNotFoundError:
        print("❌ Error: nvidia-smi not found. Assuming CPU only or manual assignment.")
        return {0: 0} # Fallback
    except Exception as e:
        print(f"❌ Error reading GPU stats: {e}")
        return {}

def get_best_gpu(candidates=None, memory_threshold=2000):
    """
    从候选列表中选择显存剩余最多的 GPU
    :param candidates: 允许使用的 GPU ID 列表，如 [0, 1]。如果为 None，则检查所有 GPU。
    :param memory_threshold: 最小需要的显存 (MiB)，默认 2GB。
    :return: best_gpu_id (int)
    """
    gpu_map = get_gpu_memory_map()
    
    if not gpu_map:
        return 0 # Default fallback
    
    best_gpu = None
    max_free = -1
    
    # 筛选候选 GPU
    target_gpus = candidates if candidates is not None else list(gpu_map.keys())
    
    # 打印当前状态供调试
    status_msg = " | ".join([f"GPU {k}: {v}MiB free" for k, v in gpu_map.items() if k in target_gpus])
    print(f"🔍 GPU Status: [{status_msg}]")
    
    for gpu_id in target_gpus:
        if gpu_id not in gpu_map:
            continue
            
        free_mem = gpu_map[gpu_id]
        
        # 找剩余显存最大的那个
        if free_mem > max_free:
            max_free = free_mem
            best_gpu = gpu_id
            
    if max_free < memory_threshold:
        print(f"⚠️ Warning: All candidate GPUs are busy (Max free: {max_free} MiB). Waiting might be needed.")
        # 这里可以选择让外部循环继续等待，或者返回最不忙的那个
    
    return best_gpu