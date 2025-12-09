import argparse
import json
import os

import numpy as np
import torch

from src.config import apply_dataset_defaults
from src.experiment_utils import setup_logging
from src.trainer import train_one_experiment, run_inference


def parse_args():
    parser = argparse.ArgumentParser(description="Cognitive Diagnosis Model Training and Testing")

    # 数据参数
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="assist_09",
        choices=["assist_09", "assist_17", "junyi"],
        help="Dataset name to use; controls default data_dir and some hyperparameters.",
    )
    parser.add_argument("--data_dir", type=str, default=None)

    # 模型参数
    parser.add_argument("--knowledge_dim", type=int, default=128)
    parser.add_argument("--skill_dim", type=int, default=2)
    parser.add_argument("--exercise_dim", type=int, default=128)
    parser.add_argument("--num_relation_heads", type=int, default=4)
    parser.add_argument("--num_gnn_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    # soft prototype 相关参数
    parser.add_argument("--num_prototypes", type=int, default=3, help="Number of soft prototypes for students")
    parser.add_argument("--proto_tau", type=float, default=1.0, help="Temperature for soft prototype assignment")
    parser.add_argument("--proto_lambda", type=float, default=0.5, help="Residual weight for prototype correction on knowledge state")
    parser.add_argument("--disable_soft_prototype", action="store_true", help="Disable soft prototype module")

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lambda_sparse", type=float, default=0.01)
    parser.add_argument(
        "--lambda_proto_div",
        type=float,
        default=0.0,
        help="Weight for prototype diversity regularization",
    )
    parser.add_argument(
        "--lambda_proto_usage",
        type=float,
        default=0.0,
        help="Weight for prototype usage balance regularization",
    )
    parser.add_argument(
        "--lambda_sparse_personal",
        type=float,
        default=0.0,
        help="Weight for personalized relation sparsity (G-PDS hook).",
    )
    parser.add_argument(
        "--lambda_alpha",
        type=float,
        default=0.0,
        help="Penalty for gate alpha to favor global graph (G-PDS hook).",
    )

    # 早停和调度器参数
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--early_stop_patience", type=int, default=5)

    # 其他参数
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generate_diagnosis", default=True)
    parser.add_argument("--use_personal_graph", action="store_true", help="Enable personalized relation graph (G-PDS hook).")
    parser.add_argument(
        "--model_variant",
        type=str,
        default="full",
        help="Name tag for this experiment variant (e.g., full, no_proto, no_skill).",
    )

    parser.add_argument(
        "--gpu_candidates",
        type=str,
        default=None,
        help='Comma-separated GPU ids to consider, e.g. "0,2". If None, use all GPUs.',
    )

    parser.add_argument(
        "--exercise_l2_lambda",
        type=float,
        default=5e-5,   # 保持你现在 model 里的默认值一致
        help="L2 regularization weight for exercise parameters (embedding/difficulty/discrimination).",
    )

    # 保存参数
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--save_interval", type=int, default=10)

    # 数据清洗参数
    parser.add_argument(
        "--min_stu_interactions",
        type=int,
        default=15,
        help="Minimum interactions for students to keep (0 = disable)",
    )
    parser.add_argument(
        "--min_exer_interactions",
        type=int,
        default=0,
        help="Minimum interactions for exercises to keep (0 = disable)",
    )
    parser.add_argument(
        "--min_poison_count",
        type=int,
        default=0,
        help="Minimum count for detecting toxic items with acc=0 or 1 (0 = disable)",
    )

    # 消融开关（默认 False，不做消融）
    parser.add_argument("--ablate_soft_prototype", action="store_true", help="关闭 soft prototype 模块")
    parser.add_argument("--ablate_skill_encoder", action="store_true", help="关闭应试技巧编码器")
    parser.add_argument("--ablate_exercise_graph", action="store_true", help="关闭习题侧图传播")

    return parser


def main():
    parser = parse_args()
    args = parser.parse_args()
    args = apply_dataset_defaults(args, parser)

    # 统一计算各模块开关（保持向后兼容 disable_soft_prototype）
    args.use_soft_prototype = (
        not getattr(args, "disable_soft_prototype", False)
        and not getattr(args, "ablate_soft_prototype", False)
    )
    args.use_skill_encoder = not getattr(args, "ablate_skill_encoder", False)
    args.use_exercise_graph = not getattr(args, "ablate_exercise_graph", False)
    args.use_personal_graph = getattr(args, "use_personal_graph", False)

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    logger = setup_logging(args.log_dir)

    args_file = os.path.join(args.save_dir, "args.json")
    with open(args_file, "w") as f:
        json.dump(vars(args), f, indent=4)

    logger.info("Arguments:")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"  {arg}: {value}")

    best_val_auc, best_epoch = train_one_experiment(args, logger)
    metrics, inference_info = run_inference(args, logger)

    if not metrics:
        logger.error("Inference failed; please check previous logs.")
        return

    logger.info("Run completed.")
    logger.info(f"Best validation AUC: {best_val_auc:.4f} at epoch {best_epoch}")
    logger.info(
        f"Test metrics - AUC: {metrics['auc']:.4f}, ACC: {metrics['acc']:.4f}, RMSE: {metrics['rmse']:.4f}"
    )


if __name__ == "__main__":
    main()
