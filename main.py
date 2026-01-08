import argparse
import json
import os
import sys

import numpy as np
import torch

from src.config import apply_dataset_defaults
from src.experiment_utils import setup_logging
from src.trainer import train_one_experiment, run_inference


def _normalize_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "t"}
    return bool(value)


def parse_args():
    parser = argparse.ArgumentParser(description="Cognitive Diagnosis Model Training and Testing")

    # ======================
    # Data
    # ======================
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="assist_09",
        choices=["assist_09", "assist_17", "junyi"],
        help="Dataset name; controls data_dir and some defaults.",
    )
    parser.add_argument("--data_dir", type=str, default=None)

    # ======================
    # Model (keep old args)
    # ======================
    parser.add_argument("--knowledge_dim", type=int, default=128)
    parser.add_argument("--skill_dim", type=int, default=64)       # new model prefers >=64; run_all_datasets overrides
    parser.add_argument("--exercise_dim", type=int, default=128)
    parser.add_argument("--num_relation_heads", type=int, default=4)
    parser.add_argument("--num_gnn_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    # soft prototype
    parser.add_argument("--num_prototypes", type=int, default=3)
    parser.add_argument("--proto_tau", type=float, default=1.0)
    parser.add_argument("--proto_lambda", type=float, default=0.5)
    parser.add_argument("--disable_soft_prototype", action="store_true")

    # ======================
    # Training
    # ======================
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    # Regularization (keep old names!)
    parser.add_argument("--lambda_sparse", type=float, default=0.01,
                        help="Graph sparsity weight (mapped to graph entropy weight in the new model).")
    parser.add_argument("--lambda_proto_div", type=float, default=0.0)
    parser.add_argument("--lambda_proto_usage", type=float, default=0.0)
    parser.add_argument("--lambda_sparse_personal", type=float, default=0.0)
    parser.add_argument("--lambda_alpha", type=float, default=0.0)

    # scheduler / early stop
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--early_stop_patience", type=int, default=5)

    # misc
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # NOTE: keep behavior; if you want a proper CLI bool, change to action=store_true later
    parser.add_argument("--generate_diagnosis", default=True)

    parser.add_argument("--use_personal_graph", action="store_true")
    parser.add_argument("--model_variant", type=str, default="full")

    parser.add_argument("--gpu_candidates", type=str, default=None)

    parser.add_argument(
        "--exercise_l2_lambda",
        type=float,
        default=5e-5,
        help="L2 regularization weight (mapped to mf_l2_lambda in the new model).",
    )

    # save
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--save_interval", type=int, default=10)

    # cleaning
    parser.add_argument("--min_stu_interactions", type=int, default=15)
    parser.add_argument("--min_exer_interactions", type=int, default=0)
    parser.add_argument("--min_poison_count", type=int, default=0)

    # ablations (kept for compatibility; new model may ignore some)
    parser.add_argument("--ablate_soft_prototype", action="store_true")
    parser.add_argument("--ablate_skill_encoder", action="store_true")
    parser.add_argument("--ablate_concept_graph", action="store_true")

    # ======================
    # New optional knobs (do NOT affect run_all_datasets unless used)
    # ======================
    parser.add_argument("--graph_topk", type=int, default=None,
                        help="Hard top-k neighbors per concept row (None=disable).")
    parser.add_argument("--disable_q_conditioning", action="store_true",
                        help="Disable Q-conditioning in MF branch (not recommended).")
    parser.add_argument("--disable_self_loop", action="store_true",
                        help="Disable self-loop in learned concept graph.")
    parser.add_argument("--personal_rank", type=int, default=4,
                        help="Low-rank size for personal adjacency.")
    parser.add_argument("--gnn_residual_weight", type=float, default=0.5,
                        help="Residual weight in GNN update.")

    return parser


def main():
    parser = parse_args()
    if any(arg.startswith("--ablate_exercise_graph") for arg in sys.argv):
        raise SystemExit("error: --ablate_exercise_graph is removed. Use --ablate_concept_graph instead.")
    args = parser.parse_args()
    args = apply_dataset_defaults(args, parser)

    # unified switches (backward compatible)
    args.generate_diagnosis = _normalize_bool(args.generate_diagnosis, default=True)
    args.use_soft_prototype = (
        not getattr(args, "disable_soft_prototype", False)
        and not getattr(args, "ablate_soft_prototype", False)
    )
    # NOTE: new model always has MF student latent; ablate_skill_encoder kept for compatibility
    args.use_mf_branch = not getattr(args, "ablate_skill_encoder", False)
    args.use_skill_encoder = args.use_mf_branch
    args.use_concept_graph = not getattr(args, "ablate_concept_graph", False)
    args.use_exercise_graph = args.use_concept_graph
    args.use_personal_graph = getattr(args, "use_personal_graph", False)

    # seeds
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
    metrics, _ = run_inference(args, logger)

    if not metrics:
        logger.error("Inference failed; check logs.")
        return

    logger.info("Run completed.")
    logger.info(f"Best validation AUC: {best_val_auc:.4f} at epoch {best_epoch}")
    logger.info(f"Test metrics - AUC: {metrics['auc']:.4f}, ACC: {metrics['acc']:.4f}, RMSE: {metrics['rmse']:.4f}")


if __name__ == "__main__":
    main()
