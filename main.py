import os
import csv
from datetime import datetime

import torch

import gpu_utils
from src.config import get_config
from src.dataset import get_dataloaders
from src.model import ConceptSkillCDM
from src.trainer import Trainer
from src.utils import set_seed, get_logger, ensure_dir, count_parameters


def resolve_device(device_str: str) -> torch.device:
    if device_str != "auto":
        dev = torch.device(device_str)
        print(f"[Device] Using manually specified device: {dev}")
        return dev

    if torch.cuda.is_available():
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible not in ("", None, "-1"):
            dev = torch.device("cuda")
            print(f"[Device] Auto (respect CUDA_VISIBLE_DEVICES={visible}) -> {dev}")
            return dev

        try:
            best_gpu_id = gpu_utils.get_best_gpu()
            dev = torch.device(f"cuda:{best_gpu_id}")
            print(f"[Device] Auto-selected GPU via gpu_utils: cuda:{best_gpu_id}")
            return dev
        except Exception as e:
            print(f"[Device] Auto gpu_utils failed ({e}), fallback to plain 'cuda'")
            return torch.device("cuda")

    dev = torch.device("cpu")
    print("[Device] CUDA not available, using CPU.")
    return dev


def save_results(config, metrics, trainer: Trainer, filepath: str = None):
    """
    Append experiment results to a CSV for quick comparison across runs.
    """
    if filepath is None:
        filepath = os.path.join(config.log_dir, "results.csv")
    ensure_dir(os.path.dirname(filepath))

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": config.dataset,
        "seed": config.seed,
        "device": config.device,
        "num_workers": config.num_workers,
        "best_valid_auc": getattr(trainer, "best_valid_auc", None),
        "best_epoch": getattr(trainer, "best_epoch", None),
        "test_auc": metrics.get("AUC"),
        "test_acc": metrics.get("ACC"),
        "test_rmse": metrics.get("RMSE"),
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "patience": getattr(config, "patience", None),
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "dim_student": config.dim_student,
        "dim_item": config.dim_item,
        "dim_concept": config.dim_concept,
        "num_heads": config.num_heads,
        "head_role_assign": getattr(config, "head_role_assign", None),
        "graph_dropout": config.graph_dropout,
        "graph_topk": config.graph_topk,
        "gnn_layers": config.gnn_layers,
        "gnn_hidden_dim": config.gnn_hidden_dim,
        "skill_dim": config.skill_dim,
        "agg_type": getattr(config, "agg_type", None),
        "tau": config.tau,
        "lambda_graph_sparse": config.lambda_graph_sparse,
        "lambda_graph_sym": config.lambda_graph_sym,
        "lambda_graph_dag": config.lambda_graph_dag,
        "lambda_graph_trans": config.lambda_graph_trans,
        "lambda_de_orth": config.lambda_de_orth,
        "lambda_de_mi": config.lambda_de_mi,
    }

    fieldnames = list(row.keys())
    file_exists = os.path.exists(filepath)
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"[Result] Appended metrics to {filepath}")


def run_experiment(config):
    set_seed(config.seed)
    ensure_dir(config.log_dir)
    log_name = f"{config.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = get_logger(config.log_dir, filename=log_name)
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

    # 评估所有数据集划分（已加载最佳验证模型）
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

    save_results(config, test_metrics, trainer)
    return {"train": train_metrics, "valid": valid_metrics, "test": test_metrics}


def main():
    config = get_config()
    run_experiment(config)


if __name__ == "__main__":
    main()
