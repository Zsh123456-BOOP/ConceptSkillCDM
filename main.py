import os

import torch
import gpu_utils

from src.config import get_config
from src.dataset import get_dataloaders
from src.model import ConceptSkillCDM
from src.trainer import Trainer
from src.utils import set_seed, get_logger, ensure_dir, count_parameters


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            best_gpu_id = gpu_utils.get_best_gpu()
            print(f"[Device] Auto-selected GPU: cuda:{best_gpu_id}")
            return torch.device(f"cuda:{best_gpu_id}")
        print("[Device] CUDA not available, using CPU.")
        return torch.device("cpu")
    print(f"[Device] Using manually specified device: {device_str}")
    return torch.device(device_str)


def run_experiment(config):
    set_seed(config.seed)
    ensure_dir(config.log_dir)
    logger = get_logger(config.log_dir, filename=f"{config.dataset}.log")
    device = resolve_device(config.device)
    config.device = str(device)

    logger.info(f"Running ConceptSkillCDM on {config.dataset} with device {device}")

    train_loader, valid_loader, test_loader, dataset_info = get_dataloaders(config)

    model = ConceptSkillCDM(
        config=config,
        num_students=dataset_info["num_students"],
        num_items=dataset_info["num_items"],
        num_concepts=dataset_info["num_concepts"],
    ).to(device)
    logger.info(f"Model parameters: {count_parameters(model):,}")

    trainer = Trainer(model, config, train_loader, valid_loader, test_loader, device, logger=logger)
    trainer.train()

    # Evaluate on all splits with best checkpoint loaded
    train_metrics = trainer.evaluate(train_loader, split_name="train")
    valid_metrics = trainer.evaluate(valid_loader, split_name="valid")
    test_metrics = trainer.evaluate(test_loader, split_name="test")

    logger.info(f"[Train] AUC={train_metrics['AUC']:.4f} ACC={train_metrics['ACC']:.4f} RMSE={train_metrics['RMSE']:.4f}")
    logger.info(f"[Valid] AUC={valid_metrics['AUC']:.4f} ACC={valid_metrics['ACC']:.4f} RMSE={valid_metrics['RMSE']:.4f}")
    logger.info(f"[Test ] AUC={test_metrics['AUC']:.4f} ACC={test_metrics['ACC']:.4f} RMSE={test_metrics['RMSE']:.4f}")

    print(
        f"Final Results ({config.dataset}): "
        f"AUC={test_metrics['AUC']:.4f} ACC={test_metrics['ACC']:.4f} RMSE={test_metrics['RMSE']:.4f}"
    )
    return {"train": train_metrics, "valid": valid_metrics, "test": test_metrics}


def main():
    config = get_config()
    run_experiment(config)


if __name__ == "__main__":
    main()
