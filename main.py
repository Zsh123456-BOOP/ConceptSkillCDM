"""Command-line entry point for the Graph-IRT model."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

from gpu_utils import configure_main_process_gpus, parse_gpu_ids
from src.config import (
    DATASET_DEFAULTS,
    EMA_DECAY,
    PAIRWISE_AUC_WEIGHT,
    TRAIN_EVIDENCE_MODES,
    apply_dataset_defaults,
    collect_explicit_arg_dests,
)


MODEL_VARIANTS = (
    "full",
    "no_response_evidence",
    "no_evidence_anchor",
    "no_evidence_propagation",
    # Diagnostic: adds the historical pairwise-AUC surrogate back on top of
    # the pure-BCE production objective (it never earned its keep on test).
    "pairwise_auc",
    "ema_bce",
    "no_message_passing",
    # Module-level ablation: removes every graph-calibration pathway at once
    # (state message passing AND the anchor's propagated-evidence channel).
    "no_graph_calibration",
    "mec",
    "mec_state_graph",
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
    parser.add_argument("--graph_tau_init", type=float, default=1.0)
    parser.add_argument(
        "--graph_dropout",
        type=float,
        default=-1.0,
        help="Dropout inside relation learning; a negative value follows --dropout.",
    )

    # Training and regularization
    parser.add_argument(
        "--train_evidence_mode",
        default="excluded",
        choices=TRAIN_EVIDENCE_MODES,
        help=(
            "Training-query response-statistic boundary: subtract the current "
            "row, replace its outcome by a label-independent expectation while "
            "retaining its count, or include the current row unchanged. "
            "Validation and test always read complete train-only statistics."
        ),
    )
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
    parser.add_argument(
        "--evidence_state_injection",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=False,
        help="When False (default), response statistics reach theta only through the anchor.",
    )
    parser.add_argument(
        "--anchor_multihead_prop",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=True,
        help="One propagated anchor channel per relation head (production default).",
    )
    parser.add_argument(
        "--prediction_head",
        choices=("irt2pl", "ncd_mlp"),
        default="irt2pl",
        help="Head probe: single scalar 2PL (default) or NCDM-style monotone MLP.",
    )
    parser.add_argument(
        "--exposure_prior_pmi",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=False,
        help="PMI-normalize co-exposure counts before row normalization (per-dataset config).",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generate_diagnosis", type=_parse_bool, nargs="?", const=True, default=True)
    parser.add_argument("--save_dir", default="./checkpoints")
    parser.add_argument("--log_dir", default="./logs")
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument(
        "--warm_start_checkpoint",
        default=None,
        help=(
            "Matched direct-evidence checkpoint required only while training "
            "the MEC variants."
        ),
    )
    parser.add_argument("--debug_graph_diag", action="store_true")
    parser.add_argument("--diag_batches", type=int, default=2)
    parser.add_argument(
        "--run_mode",
        choices=("train", "test", "train_test"),
        default="train",
        help="Use train for validation-only model selection and test only after the checkpoint is sealed.",
    )

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
    args.use_response_evidence = args.model_variant != "no_response_evidence"
    args.evidence_anchor_mode = {
        "no_response_evidence": "off",
        "no_evidence_anchor": "off",
        "no_evidence_propagation": "direct_only",
        "no_graph_calibration": "direct_only",
        "mec": "mec",
        "mec_state_graph": "mec",
    }.get(args.model_variant, "full")
    args.pairwise_auc_weight = (
        PAIRWISE_AUC_WEIGHT if args.model_variant == "pairwise_auc" else 0.0
    )
    args.ema_decay = EMA_DECAY if args.model_variant == "ema_bce" else 0.0
    args.disable_graph_module = args.model_variant in {
        "no_graph_calibration",
        "mec",
    }
    if args.model_variant in {
        "no_message_passing",
        "no_graph_calibration",
        "mec",
    }:
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
    if (
        str(args.train_evidence_mode) != "excluded"
        and not bool(args.use_response_evidence)
    ):
        raise SystemExit(
            "error: non-default --train_evidence_mode requires response evidence"
        )
    is_mec = str(args.model_variant) in {"mec", "mec_state_graph"}
    trains = str(args.run_mode) in {"train", "train_test"}
    if is_mec and trains and not args.warm_start_checkpoint:
        raise SystemExit(
            "error: MEC training requires --warm_start_checkpoint from its "
            "matched direct-evidence baseline"
        )
    if not is_mec and args.warm_start_checkpoint:
        raise SystemExit(
            "error: --warm_start_checkpoint is only valid for MEC variants"
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
    # In test mode the checkpoint owns dataset/model identity.  Keep track of
    # explicit CLI choices so inference can reject a genuine conflict without
    # mistaking parser defaults for user intent.
    args.explicit_arg_dests = sorted(explicit_dests)

    _apply_model_variant(args)
    _validate_args(args)
    _configure_requested_gpus(args)
    _seed_everything(int(args.seed))

    from src.experiment_utils import setup_logging
    from src.trainer import run_inference, train_one_experiment

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logging(args.log_dir)

    if args.run_mode in {"train", "train_test"}:
        sealed_artifacts = (
            os.path.join(args.save_dir, "test_results.json"),
            os.path.join(args.save_dir, "test_seal.json"),
        )
        sealed_path = next((path for path in sealed_artifacts if os.path.exists(path)), None)
        if sealed_path is not None:
            raise SystemExit(
                "Refusing to overwrite a test-sealed run; choose a new --save_dir: "
                f"{sealed_path}"
            )

    args_name = "test_args.json" if args.run_mode == "test" else "args.json"
    with open(os.path.join(args.save_dir, args_name), "w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=4, ensure_ascii=False)

    logger.info("Arguments:")
    for name, value in sorted(vars(args).items()):
        logger.info("  %s: %s", name, value)

    failure_path = os.path.join(args.save_dir, "failure_reason.json")

    best_val_auc = None
    best_epoch = None
    metrics = {}
    try:
        if args.run_mode in {"train", "train_test"}:
            best_val_auc, best_epoch = train_one_experiment(args, logger)
        if args.run_mode in {"test", "train_test"}:
            metrics, test_results = run_inference(args, logger)
            if best_val_auc is None:
                best_val_auc = float(test_results.get("best_val_auc", 0.0))
                best_epoch = int(test_results.get("model_epoch", 0))
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

    if args.run_mode == "train":
        validation_result = {
            "dataset": args.dataset_name,
            "model_variant": args.model_variant,
            "train_evidence_mode": args.train_evidence_mode,
            "seed": int(args.seed),
            "best_val_auc": float(best_val_auc),
            "best_epoch": int(best_epoch),
            "test_evaluated": False,
        }
        with open(
            os.path.join(args.save_dir, "validation_result.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(validation_result, file, indent=4, ensure_ascii=False)
        if os.path.exists(failure_path):
            os.remove(failure_path)
        logger.info(
            "Validation-only training completed. Best validation AUC: %.4f at epoch %d; test remains sealed.",
            float(best_val_auc),
            int(best_epoch),
        )
        return

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
