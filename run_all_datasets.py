import copy
import torch
import gpu_utils

from main import run_experiment, resolve_device
from src.config import get_config


def main():
    base_config = get_config()
    device = resolve_device(base_config.device)
    datasets = ["assist_09", "assist_17", "junyi", "sample"]
    results = {}
    for ds in datasets:
        cfg = copy.deepcopy(base_config)
        cfg.dataset = ds
        cfg.device = str(device)
        print(f"\n===== Running dataset: {ds} =====")
        results[ds] = run_experiment(cfg)

    print("\nSummary (best test metrics):")
    for ds in datasets:
        test_metrics = results[ds]["test"]
        print(
            f"{ds}: AUC={test_metrics['AUC']:.4f} "
            f"ACC={test_metrics['ACC']:.4f} RMSE={test_metrics['RMSE']:.4f}"
        )


if __name__ == "__main__":
    main()
