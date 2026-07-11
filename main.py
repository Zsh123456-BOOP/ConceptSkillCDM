"""Command-line entry point for the Graph-IRT model."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback

from gpu_utils import configure_main_process_gpus, parse_gpu_ids
from src.config import DATASET_DEFAULTS, apply_dataset_defaults, collect_explicit_arg_dests


MODEL_VARIANTS = (
    "full",
    "no_message_passing",
    "item_only",
    "exposure_only",
    "degree_random",
)

GRAPH_PRIOR_MODES = (
    "evidence",
    "item_only",
    "exposure_only",
    "degree_random",
)

STUDENT_CONCEPT_INTERACTIONS = ("none", "hadamard", "low_rank")


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate the Graph-IRT cognitive diagnosis model")

    # Data
    parser.add_argument(
        "--dataset_name",
        default="assist_09",
        choices=sorted(DATASET_DEFAULTS),
        help="Dataset name; selects the corresponding defaults and data directory.",
    )
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--min_stu_interactions", type=int, default=15)
    parser.add_argument("--min_exer_interactions", type=int, default=0)
    parser.add_argument(
        "--graph_prior_mode",
        default="evidence",
        choices=GRAPH_PRIOR_MODES,
        help="Relation prior: observed evidence, one evidence source, or a row-degree-matched random control.",
    )

    # Model
    parser.add_argument("--model_variant", default="full", choices=MODEL_VARIANTS)
    parser.add_argument("--knowledge_dim", type=int, default=128)
    parser.add_argument("--num_relation_heads", type=int, default=4)
    parser.add_argument("--num_gnn_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--graph_topk", type=int, default=None)
    parser.add_argument("--disable_self_loop", action="store_true")
    parser.add_argument("--gnn_residual_weight", type=float, default=0.5)
    parser.add_argument("--graph_identity_residual", type=float, default=0.0)
    parser.add_argument("--graph_propagation_alpha", type=float, default=0.20)
    parser.add_argument("--graph_prior_strength_init", type=float, default=1.0)
    parser.add_argument(
        "--student_concept_interaction",
        choices=STUDENT_CONCEPT_INTERACTIONS,
        default="none",
        help="Optional explicit student-by-concept factorization inside the knowledge state.",
    )
    parser.add_argument(
        "--student_concept_interaction_scale",
        type=float,
        default=1.0,
        help="Interaction scale; must be positive when low_rank is active.",
    )
    parser.add_argument(
        "--student_concept_interaction_ratio_cap",
        type=float,
        default=0.0,
        help="Per-student interaction/additive RMS ratio cap; 0 disables capping.",
    )
    parser.add_argument("--student_concept_interaction_rank", type=int, default=8)
    parser.add_argument(
        "--student_concept_interaction_init_std",
        type=float,
        default=0.1,
        help="Low-rank factor initialization std; must be positive when low_rank is active.",
    )
    parser.add_argument("--graph_tau_init", type=float, default=1.0)
    parser.add_argument(
        "--graph_dropout",
        type=float,
        default=-1.0,
        help="Dropout inside relation learning; a negative value follows --dropout.",
    )

    # Training and regularization
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--optimizer", choices=("adam", "adamw"), default="adam")
    parser.add_argument("--lambda_graph_entropy", type=float, default=0.01)
    parser.add_argument("--graph_reg_warmup_epochs", type=int, default=1)
    parser.add_argument("--graph_reg_cap_ratio", type=float, default=6.0)
    parser.add_argument("--graph_entropy_min", type=float, default=0.15)
    parser.add_argument("--graph_entropy_max", type=float, default=0.85)
    parser.add_argument("--lambda_graph_diag", type=float, default=0.10)
    parser.add_argument("--lambda_graph_uniform", type=float, default=0.04)
    parser.add_argument("--graph_uniform_margin", type=float, default=0.10)
    parser.add_argument("--prediction_l2_lambda", type=float, default=5e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--min_epochs", type=int, default=0)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-5)

    # Runtime and outputs
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--max_test_batches", type=int, default=None)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generate_diagnosis", type=_parse_bool, nargs="?", const=True, default=True)
    parser.add_argument("--save_dir", default="./checkpoints")
    parser.add_argument("--log_dir", default="./logs")
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--debug_graph_diag", action="store_true")
    parser.add_argument("--diag_batches", type=int, default=2)

    # GPU selection. --gpus refers to physical ids; --gpu_ids contains the
    # relative ids visible to DataParallel after CUDA_VISIBLE_DEVICES is set.
    parser.add_argument("--gpu_candidates", default=None)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--gpu_memory_threshold", type=int, default=2000)
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI arguments; exposed for launcher/config contract tests."""
    return build_parser().parse_args(argv)


def _apply_model_variant(args: argparse.Namespace) -> None:
    """Map named, interpretable controls to the underlying graph settings."""
    if args.model_variant == "no_message_passing":
        args.graph_propagation_alpha = 0.0
    elif args.model_variant == "item_only":
        args.graph_prior_mode = "item_only"
    elif args.model_variant == "exposure_only":
        args.graph_prior_mode = "exposure_only"
    elif args.model_variant == "degree_random":
        args.graph_prior_mode = "degree_random"

    args.allow_self_loop = not bool(args.disable_self_loop)


def _configure_requested_gpus(args: argparse.Namespace) -> None:
    if not args.gpus:
        return

    candidates = parse_gpu_ids(args.gpus)
    if not candidates:
        raise SystemExit("error: --gpus is empty after parsing; expected a list such as 0,1")

    selected = configure_main_process_gpus(
        gpus=candidates,
        num_gpus=max(1, int(args.num_gpus)),
        memory_threshold=int(args.gpu_memory_threshold),
    )
    args.selected_gpus = ",".join(str(gpu_id) for gpu_id in selected)
    args.gpu_candidates = args.selected_gpus
    args.multi_gpu = len(selected) > 1
    args.gpu_ids = ",".join(str(index) for index in range(len(selected))) if args.multi_gpu else None


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "knowledge_dim": args.knowledge_dim,
        "num_relation_heads": args.num_relation_heads,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "save_interval": args.save_interval,
        "learning_rate": args.learning_rate,
        "graph_prior_strength_init": args.graph_prior_strength_init,
        "graph_tau_init": args.graph_tau_init,
        "early_stop_patience": args.early_stop_patience,
        "student_concept_interaction_rank": args.student_concept_interaction_rank,
    }
    for name, value in positive.items():
        if float(value) <= 0:
            raise SystemExit(f"error: --{name} must be positive, got {value}")
    if int(args.num_gnn_layers) < 0:
        raise SystemExit(f"error: --num_gnn_layers must be non-negative, got {args.num_gnn_layers}")
    if int(args.num_workers) < 0:
        raise SystemExit(f"error: --num_workers must be non-negative, got {args.num_workers}")
    if int(args.patience) < 0:
        raise SystemExit(f"error: --patience must be non-negative, got {args.patience}")
    if int(args.min_epochs) < 0:
        raise SystemExit(f"error: --min_epochs must be non-negative, got {args.min_epochs}")
    if int(args.min_epochs) > int(args.epochs):
        raise SystemExit(
            f"error: --min_epochs cannot exceed --epochs, got {args.min_epochs}>{args.epochs}"
        )
    for name in (
        "weight_decay",
        "lambda_graph_entropy",
        "lambda_graph_diag",
        "lambda_graph_uniform",
        "graph_uniform_margin",
        "prediction_l2_lambda",
        "early_stop_min_delta",
        "student_concept_interaction_scale",
    ):
        value = float(getattr(args, name))
        if value < 0.0:
            raise SystemExit(f"error: --{name} must be non-negative, got {value}")
    if args.graph_topk is not None and int(args.graph_topk) <= 0:
        raise SystemExit(f"error: --graph_topk must be positive when supplied, got {args.graph_topk}")
    if not 0.0 <= float(args.dropout) < 1.0:
        raise SystemExit(f"error: --dropout must be in [0, 1), got {args.dropout}")
    if float(args.graph_dropout) >= 0.0 and not 0.0 <= float(args.graph_dropout) < 1.0:
        raise SystemExit(f"error: --graph_dropout must be negative or in [0, 1), got {args.graph_dropout}")
    for name in ("graph_identity_residual", "graph_propagation_alpha"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"error: --{name} must be in [0, 1], got {value}")
    if float(args.graph_entropy_min) > float(args.graph_entropy_max):
        raise SystemExit("error: --graph_entropy_min cannot exceed --graph_entropy_max")
    interaction_scale = float(args.student_concept_interaction_scale)
    if not math.isfinite(interaction_scale) or not 0.0 <= interaction_scale <= 4.0:
        raise SystemExit(
            "error: --student_concept_interaction_scale must be finite and in [0, 4], "
            f"got {args.student_concept_interaction_scale}"
        )
    if args.student_concept_interaction == "low_rank" and interaction_scale == 0.0:
        raise SystemExit(
            "error: --student_concept_interaction_scale must be positive for low_rank "
            "to avoid a disabled interaction"
        )
    interaction_ratio_cap = float(args.student_concept_interaction_ratio_cap)
    if not math.isfinite(interaction_ratio_cap) or not 0.0 <= interaction_ratio_cap <= 4.0:
        raise SystemExit(
            "error: --student_concept_interaction_ratio_cap must be finite and in [0, 4], "
            f"got {args.student_concept_interaction_ratio_cap}"
        )
    interaction_init_std = float(args.student_concept_interaction_init_std)
    if not math.isfinite(interaction_init_std) or not 0.0 <= interaction_init_std <= 1.0:
        raise SystemExit(
            "error: --student_concept_interaction_init_std must be finite and in [0, 1], "
            f"got {args.student_concept_interaction_init_std}"
        )
    if args.student_concept_interaction == "low_rank" and interaction_init_std == 0.0:
        raise SystemExit(
            "error: --student_concept_interaction_init_std must be positive for low_rank "
            "to avoid zero-gradient factors"
        )


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main() -> None:
    parser = build_parser()
    raw_argv = sys.argv[1:]
    args = parser.parse_args(raw_argv)
    explicit_dests = collect_explicit_arg_dests(raw_argv, parser)
    args = apply_dataset_defaults(args, parser, explicit_dests=explicit_dests)

    _apply_model_variant(args)
    _validate_args(args)
    _configure_requested_gpus(args)
    _seed_everything(int(args.seed))

    from src.experiment_utils import setup_logging
    from src.trainer import run_inference, train_one_experiment

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logging(args.log_dir)

    with open(os.path.join(args.save_dir, "args.json"), "w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=4, ensure_ascii=False)

    logger.info("Arguments:")
    for name, value in sorted(vars(args).items()):
        logger.info("  %s: %s", name, value)

    failure_path = os.path.join(args.save_dir, "failure_reason.json")

    try:
        best_val_auc, best_epoch = train_one_experiment(args, logger)
        metrics, _ = run_inference(args, logger)
    except Exception as exc:
        extra = exc.to_failure_dict() if hasattr(exc, "to_failure_dict") else {}
        extra["traceback"] = traceback.format_exc()
        payload = {
            "reason": extra.get("reason", "runtime_exception"),
            "error_type": type(exc).__name__,
            "message": str(exc),
            **extra,
        }
        with open(failure_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=4, ensure_ascii=False)
        logger.error("Run failed with exception: %s", exc)
        logger.error("%s", extra["traceback"])
        raise SystemExit(1) from exc

    if not metrics:
        with open(failure_path, "w", encoding="utf-8") as file:
            json.dump({"reason": "inference_failed"}, file, indent=4)
        raise SystemExit("Inference failed; check logs.")

    if os.path.exists(failure_path):
        os.remove(failure_path)

    logger.info("Run completed. Best validation AUC: %.4f at epoch %d", best_val_auc, best_epoch)
    logger.info(
        "Test metrics - AUC: %.4f, ACC: %.4f, RMSE: %.4f",
        metrics["auc"],
        metrics["acc"],
        metrics["rmse"],
    )


if __name__ == "__main__":
    main()
