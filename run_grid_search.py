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

    print(f"[LAUNCH] GPU {gpu_id} -> {tag}")
    print("         CMD:", " ".join(cmd))
    return subprocess.Popen(cmd, env=env)


def main():
    parser = argparse.ArgumentParser(description="Parallel grid search over hyperparameters and ablations.")

    # 新增：支持多个数据集一次性跑完，按顺序执行
    parser.add_argument(
        "--datasets",
        type=str,
        default="assist_09,junyi",
        help="Comma-separated dataset names to run, e.g. 'assist_09,junyi'. "
             "They must exist in GRID_SEARCH_SPACE.",
    )
    # 兼容老参数（如果只指定一个）
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="(Deprecated) Single dataset name. If --datasets is given, this will be ignored.",
    )

    parser.add_argument("--gpus", type=str, default="0,1,2", help="Comma-separated GPU ids to use.")
    parser.add_argument("--max_concurrent", type=int, default=2, help="Maximum concurrent experiments.")
    args = parser.parse_args()

    # 解析数据集列表：优先使用 --datasets
    if args.datasets is not None:
        dataset_names = [x.strip() for x in args.datasets.split(",") if x.strip() != ""]
    elif args.dataset_name is not None:
        dataset_names = [args.dataset_name]
    else:
        dataset_names = ["assist_09"]

    gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    max_concurrent = args.max_concurrent

    print(f"Datasets to run (sequentially): {dataset_names}")
    print(f"GPUs: {gpus}, max_concurrent per dataset: {max_concurrent}")

    # ******** 关键：外层按数据集顺序循环，内层再跑该数据集所有组合 ********
    for dataset_name in dataset_names:
        config_entry = GRID_SEARCH_SPACE.get(dataset_name)
        if config_entry is None:
            raise ValueError(f"Dataset '{dataset_name}' not found in GRID_SEARCH_SPACE.")

        base_params = config_entry.get("base", {}) if isinstance(config_entry, dict) else {}
        search_space = config_entry.get("search", config_entry)

        param_grid = []
        for combo in build_param_grid(search_space):
            merged = {**base_params, **combo}
            param_grid.append(merged)

        print("\n" + "=" * 80)
        print(f"[DATASET] {dataset_name} | total experiments: {len(param_grid)}")
        print("=" * 80)

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

                # 全局并发限制（当前数据集内部）
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

        # 等待当前数据集所有任务结束，再进入下一个数据集
        for proc, gpu in running:
            proc.wait()
            print(f"[DONE] Dataset {dataset_name} | GPU {gpu} experiment finished with code {proc.returncode}")

    print("\nAll datasets finished.")


if __name__ == "__main__":
    main()
