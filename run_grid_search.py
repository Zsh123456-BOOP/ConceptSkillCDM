import argparse
import itertools
import os
import subprocess
import time

from src.config import GRID_SEARCH_SPACE


def build_param_grid(search_space: dict):
    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def format_tag_value(value):
    """将参数值格式化为路径友好的字符串。"""
    if isinstance(value, float):
        return f"{value}".replace(".", "p")
    return str(value)


def build_variant_from_flags(params: dict) -> str:
    parts = []
    if params.get("ablate_soft_prototype"):
        parts.append("proto")
    if params.get("ablate_skill_encoder"):
        parts.append("skill")
    if params.get("ablate_exercise_graph"):
        parts.append("graph")
    if params.get("ablate_concept_fusion"):
        parts.append("fusion")
    if not parts:
        return "full"
    return "no_" + "_".join(parts)


def launch_experiment(gpu_id: int, overrides: dict, dataset_name: str):
    tag_parts = [dataset_name]
    for k in sorted(overrides.keys()):
        v = overrides[k]
        if isinstance(v, bool):
            if v:
                tag_parts.append(k)
        else:
            tag_parts.append(f"{k}{format_tag_value(v)}")
    tag = "_".join(tag_parts)
    save_dir = os.path.join("checkpoints", tag)
    log_dir = os.path.join("logs", tag)

    cmd = ["python", "main.py", "--dataset_name", dataset_name]
    variant = overrides.get("model_variant", build_variant_from_flags(overrides))
    cmd += ["--model_variant", variant]

    for k in sorted(overrides.keys()):
        v = overrides[k]
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd.extend([f"--{k}", str(v)])

    cmd += ["--save_dir", save_dir, "--log_dir", log_dir]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[LAUNCH] GPU {gpu_id} -> tag={tag}")
    print("         CMD:", " ".join(cmd))
    return subprocess.Popen(cmd, env=env)


def main():
    parser = argparse.ArgumentParser(description="Parallel grid search over hyperparameters and ablations.")
    parser.add_argument("--dataset_name", type=str, default="assist_09")
    parser.add_argument("--gpus", type=str, default="0,1,2", help="Comma-separated GPU ids to use.")
    parser.add_argument("--max_concurrent", type=int, default=2, help="Maximum concurrent experiments.")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    max_concurrent = args.max_concurrent

    config_entry = GRID_SEARCH_SPACE.get(dataset_name)
    if config_entry is None:
        raise ValueError(f"Dataset '{dataset_name}' not found in GRID_SEARCH_SPACE.")

    base_params = config_entry.get("base", {}) if isinstance(config_entry, dict) else {}
    search_space = config_entry.get("search", config_entry)

    param_grid = []
    for combo in build_param_grid(search_space):
        merged = {**base_params, **combo}
        param_grid.append(merged)
    print(
        f"Dataset: {dataset_name}, total experiments: {len(param_grid)}, GPUs: {gpus}, max_concurrent: {max_concurrent}"
    )

    running = []
    gpu_rr_idx = 0
    for overrides in param_grid:
        submitted = False
        while not submitted:
            # 清理已结束的进程
            active = []
            busy_gpus = set()
            for proc, gpu in running:
                if proc.poll() is None:
                    active.append((proc, gpu))
                    busy_gpus.add(gpu)
            running = active

            # 全局并发限制
            if len(running) >= max_concurrent:
                time.sleep(5)
                continue

            # 找出空闲 GPU
            free_gpus = [g for g in gpus if g not in busy_gpus]
            if not free_gpus:
                time.sleep(5)
                continue

            gpu_id = free_gpus[gpu_rr_idx % len(free_gpus)]
            gpu_rr_idx += 1

            proc = launch_experiment(gpu_id, overrides, dataset_name)
            running.append((proc, gpu_id))
            submitted = True

    # 等待所有任务结束
    for proc, gpu in running:
        proc.wait()
        print(f"[DONE] GPU {gpu} experiment finished with code {proc.returncode}")


if __name__ == "__main__":
    main()
