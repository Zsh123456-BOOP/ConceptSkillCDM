# main.py
import argparse
import json
import os
import sys
import traceback

import numpy as np
import torch

from gpu_utils import configure_main_process_gpus, parse_gpu_ids
from src.config import apply_dataset_defaults, collect_explicit_arg_dests
from src.experiment_utils import setup_logging
from src.trainer import train_one_experiment, run_inference, save_component_analysis_data


def _normalize_bool(value, default=False):
    """
     --generate_diagnosis 
    -  True/False
    -  "True"/"False"/"1"/"0"/"yes"/"no"
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "t"}
    return bool(value)


def parse_args():
    parser = argparse.ArgumentParser(description="Cognitive Diagnosis Model Training and Testing")
    bool_action = argparse.BooleanOptionalAction

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
    # Model (/)
    # ======================
    parser.add_argument("--knowledge_dim", type=int, default=128)
    parser.add_argument("--num_relation_heads", type=int, default=4)
    parser.add_argument("--num_gnn_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    # ======================
    # Training
    # ======================
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    # ---- Regularization----
    parser.add_argument(
        "--lambda_sparse",
        type=float,
        default=0.01,
        help="Graph sparsity weight (mapped to graph entropy weight in the new model).",
    )
    parser.add_argument(
        "--graph_reg_warmup_epochs",
        type=int,
        default=1,
        help="Linear warmup epochs for graph-related regularization terms (<=0 disables warmup).",
    )
    parser.add_argument(
        "--graph_reg_cap_ratio",
        type=float,
        default=6.0,
        help="Cap ratio for graph regularizers relative to base BCE loss.",
    )
    parser.add_argument(
        "--graph_entropy_min",
        type=float,
        default=0.15,
        help="Lower bound of normalized graph entropy band.",
    )
    parser.add_argument(
        "--graph_entropy_max",
        type=float,
        default=0.85,
        help="Upper bound of normalized graph entropy band.",
    )
    parser.add_argument(
        "--lambda_graph_diag",
        type=float,
        default=0.10,
        help="Penalty weight for graph diagonal mass (near-identity suppressor).",
    )
    parser.add_argument(
        "--lambda_graph_uniform",
        type=float,
        default=0.04,
        help="Penalty weight to keep graph away from uniform adjacency.",
    )
    parser.add_argument(
        "--graph_uniform_margin",
        type=float,
        default=0.10,
        help="Minimum L2 distance from uniform adjacency before graph_uniform penalty becomes zero.",
    )
    parser.add_argument(
        "--graph_dropout",
        type=float,
        default=-1.0,
        help="Dropout used inside graph relation learning; <0 means follow global dropout.",
    )
    parser.add_argument(
        "--graph_tau_init",
        type=float,
        default=1.0,
        help="Initial temperature for graph relation learning softmax.",
    )
    parser.add_argument(
        "--graph_identity_residual",
        type=float,
        default=0.0,
        help="Blend ratio of identity residual into each learned global graph head.",
    )
    parser.add_argument(
        "--graph_propagation_alpha",
        type=float,
        default=0.20,
        help="Teleport strength for APPNP-style concept propagation.",
    )
    parser.add_argument(
        "--graph_query_readout_scale",
        type=float,
        default=0.35,
        help="Canonical 1-hop query-local global graph readout scale before the fixed diagnosis head.",
    )
    parser.add_argument(
        "--graph_query_readout_2hop_scale",
        type=float,
        default=0.15,
        help="Canonical 2-hop query-local global graph readout scale before the fixed diagnosis head.",
    )
    parser.add_argument("--lambda_sparse_personal", type=float, default=0.0)
    parser.add_argument("--lambda_alpha", type=float, default=0.0)
    parser.add_argument("--lambda_personal_kl", type=float, default=0.0,
                        help="KL penalty between personalized posterior and global support distribution on query rows.")
    parser.add_argument("--lambda_personal_query_residual", type=float, default=0.0,
                        help="Penalty for overly large personal query correction magnitude.")
    parser.add_argument("--personal_query_residual_margin", type=float, default=0.0,
                        help="Margin before personal query residual penalty activates.")

    # scheduler / early stop
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--early_stop_patience", type=int, default=5)

    # misc
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--max_test_batches", type=int, default=None)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    #  bool action=store_true
    parser.add_argument("--generate_diagnosis", default=True)

    # personal graph/
    parser.add_argument("--use_personal_graph", action=bool_action, default=None)
    parser.add_argument("--model_variant", type=str, default="full")

    #  GPU  trainer/launcher 
    parser.add_argument("--gpu_candidates", type=str, default=None)
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated physical GPU ids for auto selection, e.g. '0,1'.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="How many GPUs to use when --gpus is provided.",
    )
    parser.add_argument(
        "--gpu_memory_threshold",
        type=int,
        default=2000,
        help="Minimum free memory (MiB) preferred by auto GPU selection.",
    )

    parser.add_argument(
        "--prediction_l2_lambda",
        type=float,
        default=5e-5,
        help="L2 regularization weight for the fixed IRT prediction head.",
    )

    # save
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--save_interval", type=int, default=10)

    # cleaning
    parser.add_argument("--min_stu_interactions", type=int, default=15)
    parser.add_argument("--min_exer_interactions", type=int, default=0)
    parser.add_argument("--min_poison_count", type=int, default=0)

    # ======================
    # Ablations
    # ======================
    parser.add_argument("--ablate_concept_graph", action="store_true")   #  concept graph

    # ======================
    # Ablations
    # ======================
    parser.add_argument("--ablate_module1", action="store_true",
                        help="Fully disable Module 1 (Concept Structure Modeling: A+E+knowledge_encoder).")

    # ======================
    # New optional knobs
    # ======================
    parser.add_argument("--graph_topk", type=int, default=None,
                        help="Hard top-k neighbors per concept row (None=disable).")

    parser.add_argument("--disable_self_loop", action="store_true",
                        help="Disable self-loop in learned concept graph.")

    parser.add_argument("--personal_rank", type=int, default=4,
                        help="Low-rank size for personal adjacency.")

    parser.add_argument("--gnn_residual_weight", type=float, default=0.5,
                        help="Residual weight in GNN update.")

    parser.add_argument("--personal_max_alpha", type=float, default=0.35,
                        help="Upper bound for personal-graph mixing alpha.")
    parser.add_argument("--personal_delta_scale", type=float, default=1.0,
                        help="Scale factor on personal graph delta before softmax.")
    parser.add_argument("--personal_warmup_epochs", type=int, default=0,
                        help="Linear warmup epochs for personal graph mixing. 0 disables rescue warmup.")
    parser.add_argument("--personal_reg_warmup_epochs", type=int, default=None,
                        help="Linear warmup epochs for personal regularizers. Default follows personal_warmup_epochs.")
    parser.add_argument("--personal_student_dim", type=int, default=None,
                        help="Dedicated student embedding dim for E branch. Default follows knowledge_dim.")
    parser.add_argument("--lambda_alpha_min", type=float, default=0.0,
                        help="Penalty weight when personal alpha std falls below target.")
    parser.add_argument("--alpha_min_target", type=float, default=0.0,
                        help="Minimum desired std of personal alpha before collapse penalty becomes zero.")
    parser.add_argument("--personal_alpha_temperature", type=float, default=2.0,
                        help="Temperature used by bounded personal alpha delta.")
    parser.add_argument("--personal_alpha_budget", type=float, default=0.10,
                        help="Maximum additive alpha delta driven by state/context.")
    parser.add_argument("--personal_alpha_base_init", type=float, default=0.08,
                        help="Initial alpha base level before state-driven delta.")
    parser.add_argument("--personal_alpha_bias_scale", type=float, default=0.0,
                        help="Max magnitude scale for bounded student-specific alpha bias.")
    parser.add_argument("--personal_disable_student_global_context", action=bool_action, default=None,
                        help="Use state-primary personal context without raw student_global direct concatenation.")
    parser.add_argument("--personal_local_hops", type=int, default=1,
                        help="Number of support hops around current item concepts used for local personalization.")
    parser.add_argument("--personal_include_neighbor_rows", action=bool_action, default=None,
                        help="Whether E should personalize 1-hop local neighbor rows in addition to query rows.")
    parser.add_argument("--personal_support_include_query_self", action=bool_action, default=None,
                        help="Always keep query-self support available for E, even when graph support is sparse or A is disabled.")
    parser.add_argument("--personal_support_include_graph", action=bool_action, default=None,
                        help="Include global graph support as part of E's sparse posterior support.")
    parser.add_argument("--personal_support_include_neighbors", action=bool_action, default=None,
                        help="Allow local-neighbor support columns in E's sparse support basis.")
    parser.add_argument("--personal_query_row_budget", type=float, default=1.0,
                        help="Relative personalization budget assigned to queried concept rows.")
    parser.add_argument("--personal_neighbor_row_budget", type=float, default=0.30,
                        help="Relative personalization budget assigned to 1-hop local neighbor rows.")
    parser.add_argument("--personal_query_support_hops", type=int, default=0,
                        help="Additional global-graph hops used to widen E's query-time message basis without widening active rows.")
    parser.add_argument("--personal_support_only", action=bool_action, default=None,
                        help="Restrict personal residual edges to the support of global graph A when A is enabled.")
    parser.add_argument("--personal_query_message_gain", type=float, default=1.0,
                        help="Bounded gain applied after E's query-message projection normalization.")
    parser.add_argument("--personal_value_use_global_basis", action=bool_action, default=None,
                        help="Build E's value basis from both local state and global graph context.")
    parser.add_argument("--personal_message_alignment_gate", action=bool_action, default=None,
                        help="Enable alignment-aware gating before applying E's query writeback.")
    parser.add_argument("--personal_projection_hidden_factor", type=int, default=2,
                        help="Hidden expansion factor for E's value writer / alignment gate MLPs.")
    parser.add_argument("--personal_query_correction_scale", type=float, default=0.15,
                        help="Scale for injecting E's query-time message correction into query rows only.")
    parser.add_argument("--personal_query_correction_max_ratio", type=float, default=0.20,
                        help="Hard trust-region cap: max allowed personal query correction RMS as a ratio of graph query RMS.")
    parser.add_argument("--personal_query_correction_min_graph_anchor", type=float, default=0.01,
                        help="Minimum graph anchor used by the personal query trust-region when graph query RMS is tiny.")
    parser.add_argument("--share_concept_embeddings", action=bool_action, default=None,
                        help="Share concept embeddings between relation learning and knowledge encoder.")
    parser.add_argument("--graph_headwise_query_gate", action=bool_action, default=None,
                        help="Use query-discriminative head gating instead of averaging graph heads before query readout.")
    parser.add_argument("--graph_edge_bias_rank", type=int, default=8,
                        help="Low-rank edge bias rank added to A's adjacency logits.")
    parser.add_argument("--graph_prior_logit_scale", type=float, default=0.0,
                        help="Logit scale for the train-Q concept co-occurrence prior injected into A.")
    parser.add_argument("--ae_query_residual_scale", type=float, default=0.0,
                        help="Scale for the A/E-only nonlinear query-state residual before the fixed diagnosis head.")
    parser.add_argument("--ae_logit_residual_scale", type=float, default=0.0,
                        help="Scale for the A/E query logit predictor; no_A/no_E keep the remaining A or E prior component active.")
    parser.add_argument("--ae_logit_residual_clip", type=float, default=1.0,
                        help="Tanh clip magnitude for the A/E-only query logit residual before scaling.")
    parser.add_argument("--ae_irt_logit_scale", type=float, default=1.0,
                        help="Scale for the backbone IRT logit when the full A/E logit predictor is active; A-only/E-only ablations suppress IRT.")
    parser.add_argument("--ae_interaction_logit_scale", type=float, default=0.0,
                        help="Scale for the full-only A x E student-exercise interaction logit.")
    parser.add_argument("--ae_logit_dim", type=int, default=32,
                        help="Embedding width for the A/E joint logit interaction head.")
    parser.add_argument("--ae_lr_mult", type=float, default=1.0,
                        help="Learning-rate multiplier for the A/E joint prediction head.")
    parser.add_argument("--ae_stat_prior_scale", type=float, default=0.0,
                        help="Scale for train-set student/concept logit priors used to initialize the A/E head.")
    parser.add_argument("--relation_theta_scale", type=float, default=0.0,
                        help="Scale for interpretable A/E support theta readout inside the IRT logit.")
    parser.add_argument("--relation_theta_delta_clip", type=float, default=2.0,
                        help="Tanh clip for relation support theta contrast before scaling.")
    parser.add_argument("--graph_query_adapter_enable", action=bool_action, default=None,
                        help="Enable query adapter on top of graph readout before the fixed diagnosis head.")
    parser.add_argument("--personal_state_lr_mult", type=float, default=1.0,
                        help="Learning-rate multiplier for E state/context adapters.")
    parser.add_argument("--personal_id_lr_mult", type=float, default=0.5,
                        help="Learning-rate multiplier for E id/bias adapters.")

    #  GPU 
    parser.add_argument("--multi_gpu", action="store_true",
                        help="Enable multi-GPU training with DataParallel.")
    parser.add_argument("--gpu_ids", type=str, default=None,
                        help="Comma-separated GPU IDs for multi-GPU training (e.g., '0,1').")

    # 
    parser.add_argument(
        "--debug_graph_diag",
        action="store_true",
        help="Enable per-epoch A/E diagnostics and graph-related gradient norms.",
    )
    parser.add_argument(
        "--diag_batches",
        type=int,
        default=2,
        help="Number of validation batches sampled per epoch for debug diagnostics.",
    )

    return parser


def main():
    parser = parse_args()

    # 
    if any(arg.startswith("--ablate_exercise_graph") for arg in sys.argv):
        raise SystemExit("error: --ablate_exercise_graph is removed. Use --ablate_concept_graph instead.")

    raw_argv = sys.argv[1:]
    args = parser.parse_args(raw_argv)
    explicit_dests = collect_explicit_arg_dests(raw_argv, parser)
    args = apply_dataset_defaults(args, parser, explicit_dests=explicit_dests)

    args.graph_query_readout_scale = float(getattr(args, "graph_query_readout_scale", 0.35))
    args.graph_query_readout_2hop_scale = float(getattr(args, "graph_query_readout_2hop_scale", 0.15))

    # =========================================================
    # -1) main  launcher 
    # =========================================================
    if getattr(args, "gpus", None):
        gpu_candidates = parse_gpu_ids(args.gpus)
        if not gpu_candidates:
            raise SystemExit("error: --gpus is empty after parsing. Example: --gpus 0,1")

        selected_physical = configure_main_process_gpus(
            gpus=gpu_candidates,
            num_gpus=max(1, int(getattr(args, "num_gpus", 1))),
            memory_threshold=int(getattr(args, "gpu_memory_threshold", 2000)),
        )
        args.selected_gpus = ",".join(str(g) for g in selected_physical)
        args.gpu_candidates = args.selected_gpus

        if len(selected_physical) > 1:
            args.multi_gpu = True
            # DataParallel  trainer 
            args.gpu_ids = ",".join(str(i) for i in range(len(selected_physical)))
        else:
            args.multi_gpu = False

    #  bool 
    args.generate_diagnosis = _normalize_bool(args.generate_diagnosis, default=True)

    # =========================================================
    # 1)  enable/disable
    # =========================================================
    args.enable_module1 = not bool(getattr(args, "ablate_module1", False))

    # =========================================================
    # 2)  ablate_*  use_*
    #    
    # =========================================================
    # (A) ablate_concept_graph == concept graph no_concept_graph
    use_concept_graph = not getattr(args, "ablate_concept_graph", False)

    # (E) personal graph
    use_personal_graph = bool(getattr(args, "use_personal_graph", False))

    # ----------  ----------
    if not args.enable_module1:
        use_concept_graph = False
        use_personal_graph = False
        args.num_gnn_layers = 0
        args.lambda_sparse = 0.0
        args.lambda_sparse_personal = 0.0
        args.lambda_alpha = 0.0

    args.use_concept_graph = bool(use_concept_graph)
    args.use_personal_graph = bool(use_personal_graph)
    args.use_exercise_graph = args.use_concept_graph

    # =========================================================
    # 3)  trainer/model 
    # =========================================================
    # disable_self_loop -> allow_self_loop allow_self_loop
    args.allow_self_loop = not getattr(args, "disable_self_loop", False)

    # =========================================================
    # 4) 
    # =========================================================
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # =========================================================
    # 5) 
    # =========================================================
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    logger = setup_logging(args.log_dir)

    #  args
    args_file = os.path.join(args.save_dir, "args.json")
    with open(args_file, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)

    logger.info("Arguments:")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"  {arg}: {value}")

    # =========================================================
    # 6)  + 
    # =========================================================
    failure_path = os.path.join(args.save_dir, "failure_reason.json")

    def _write_failure(reason: str, exc: Exception | None = None, extra: dict | None = None) -> None:
        payload = {
            "reason": reason,
            "error_type": type(exc).__name__ if exc is not None else None,
            "message": str(exc) if exc is not None else None,
        }
        if extra:
            payload.update(extra)
        with open(failure_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)

    try:
        best_val_auc, best_epoch = train_one_experiment(args, logger)
        metrics, _ = run_inference(args, logger)
    except Exception as exc:
        extra = {}
        if hasattr(exc, "to_failure_dict"):
            extra = getattr(exc, "to_failure_dict")()
        if extra.get("reason") == "nonfinite_alpha" and extra.get("payload") is not None:
            alpha_debug_path = os.path.join(args.save_dir, "alpha_failure_debug.json")
            with open(alpha_debug_path, "w", encoding="utf-8") as f:
                json.dump(extra.get("payload"), f, indent=4, ensure_ascii=False)
        extra["traceback"] = traceback.format_exc()
        _write_failure(
            reason=extra.get("reason", "runtime_exception"),
            exc=exc,
            extra=extra,
        )
        logger.error("Run failed with exception: %s", exc)
        logger.error("%s", extra["traceback"])
        raise SystemExit(1) from exc

    if not metrics:
        logger.error("Inference failed; check logs.")
        _write_failure(reason="inference_failed")
        raise SystemExit(1)

    if os.path.exists(failure_path):
        os.remove(failure_path)

    logger.info("Run completed.")
    logger.info(f"Best validation AUC: {best_val_auc:.4f} at epoch {best_epoch}")
    logger.info(
        f"Test metrics - AUC: {metrics['auc']:.4f}, ACC: {metrics['acc']:.4f}, RMSE: {metrics['rmse']:.4f}"
    )
    
    # =========================================================
    # 7) 
    # =========================================================
    if getattr(args, "generate_diagnosis", True):
        try:
            # 
            from torch.utils.data import DataLoader
            from src.dataset import create_dataloaders
            
            device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
            
            # 
            model_path = os.path.join(args.save_dir, "best_model.pth")
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            info_dict = checkpoint["info_dict"]
            loaded_args = checkpoint.get("args", {})
            
            # 
            from src.model import CognitiveDiagnosisModel
            model = CognitiveDiagnosisModel(
                num_students=info_dict["num_students"],
                num_exercises=info_dict["num_exercises"],
                num_concepts=info_dict["num_concepts"],
                q_matrix=info_dict["q_matrix"],
                item_prior_matrix=info_dict.get("item_prior_matrix"),
                sequence_prior_matrix=info_dict.get("sequence_prior_matrix"),
                knowledge_dim=loaded_args.get("knowledge_dim", args.knowledge_dim),
                num_relation_heads=loaded_args.get("num_relation_heads", args.num_relation_heads),
                num_gnn_layers=loaded_args.get("num_gnn_layers", args.num_gnn_layers),
                dropout=loaded_args.get("dropout", args.dropout),
                use_concept_graph=loaded_args.get("use_concept_graph", args.use_concept_graph),
                graph_topk=loaded_args.get("graph_topk", getattr(args, "graph_topk", None)),
                allow_self_loop=not loaded_args.get("disable_self_loop", getattr(args, "disable_self_loop", False)),
                use_personal_graph=loaded_args.get("use_personal_graph", args.use_personal_graph),
                personal_rank=loaded_args.get("personal_rank", getattr(args, "personal_rank", 4)),
                ablate_module1=loaded_args.get("ablate_module1", getattr(args, "ablate_module1", False)),
                graph_dropout=None
                if float(loaded_args.get("graph_dropout", getattr(args, "graph_dropout", -1.0))) < 0
                else float(loaded_args.get("graph_dropout", getattr(args, "graph_dropout", -1.0))),
                graph_tau_init=loaded_args.get("graph_tau_init", getattr(args, "graph_tau_init", 1.0)),
                graph_identity_residual=loaded_args.get(
                    "graph_identity_residual", getattr(args, "graph_identity_residual", 0.0)
                ),
                graph_propagation_alpha=loaded_args.get(
                    "graph_propagation_alpha", getattr(args, "graph_propagation_alpha", 0.20)
                ),
                graph_query_readout_scale=loaded_args.get(
                    "graph_query_readout_scale", getattr(args, "graph_query_readout_scale", 0.35)
                ),
                graph_query_readout_2hop_scale=loaded_args.get(
                    "graph_query_readout_2hop_scale", getattr(args, "graph_query_readout_2hop_scale", 0.15)
                ),
                lambda_graph_entropy=loaded_args.get("lambda_sparse", getattr(args, "lambda_sparse", 0.01)),
                graph_entropy_min=loaded_args.get("graph_entropy_min", getattr(args, "graph_entropy_min", 0.15)),
                graph_entropy_max=loaded_args.get("graph_entropy_max", getattr(args, "graph_entropy_max", 0.85)),
                lambda_graph_diag=loaded_args.get("lambda_graph_diag", getattr(args, "lambda_graph_diag", 0.10)),
                lambda_graph_uniform=loaded_args.get(
                    "lambda_graph_uniform", getattr(args, "lambda_graph_uniform", 0.04)
                ),
                graph_uniform_margin=loaded_args.get(
                    "graph_uniform_margin", getattr(args, "graph_uniform_margin", 0.10)
                ),
                graph_reg_warmup_epochs=loaded_args.get(
                    "graph_reg_warmup_epochs", getattr(args, "graph_reg_warmup_epochs", 1)
                ),
                graph_reg_cap_ratio=loaded_args.get(
                    "graph_reg_cap_ratio", getattr(args, "graph_reg_cap_ratio", 6.0)
                ),
                lambda_sparse_personal=loaded_args.get(
                    "lambda_sparse_personal", getattr(args, "lambda_sparse_personal", 0.0)
                ),
                lambda_alpha=loaded_args.get("lambda_alpha", getattr(args, "lambda_alpha", 0.0)),
                lambda_personal_kl=loaded_args.get("lambda_personal_kl", getattr(args, "lambda_personal_kl", 0.0)),
                lambda_personal_query_residual=loaded_args.get(
                    "lambda_personal_query_residual", getattr(args, "lambda_personal_query_residual", 0.0)
                ),
                personal_query_residual_margin=loaded_args.get(
                    "personal_query_residual_margin", getattr(args, "personal_query_residual_margin", 0.0)
                ),
                prediction_l2_lambda=loaded_args.get(
                    "prediction_l2_lambda", getattr(args, "prediction_l2_lambda", 5e-5)
                ),
                gnn_residual_weight=loaded_args.get(
                    "gnn_residual_weight", getattr(args, "gnn_residual_weight", 0.5)
                ),
                personal_max_alpha=loaded_args.get(
                    "personal_max_alpha", getattr(args, "personal_max_alpha", 0.35)
                ),
                personal_delta_scale=loaded_args.get(
                    "personal_delta_scale", getattr(args, "personal_delta_scale", 1.0)
                ),
                personal_warmup_epochs=loaded_args.get(
                    "personal_warmup_epochs", getattr(args, "personal_warmup_epochs", 0)
                ),
                personal_reg_warmup_epochs=loaded_args.get(
                    "personal_reg_warmup_epochs", getattr(args, "personal_reg_warmup_epochs", None)
                ),
                personal_student_dim=loaded_args.get(
                    "personal_student_dim", getattr(args, "personal_student_dim", args.knowledge_dim)
                ),
                lambda_alpha_min=loaded_args.get(
                    "lambda_alpha_min", getattr(args, "lambda_alpha_min", 0.0)
                ),
                alpha_min_target=loaded_args.get("alpha_min_target", getattr(args, "alpha_min_target", 0.0)),
                personal_alpha_temperature=loaded_args.get(
                    "personal_alpha_temperature", getattr(args, "personal_alpha_temperature", 2.0)
                ),
                personal_alpha_budget=loaded_args.get(
                    "personal_alpha_budget", getattr(args, "personal_alpha_budget", 0.10)
                ),
                personal_alpha_base_init=loaded_args.get(
                    "personal_alpha_base_init", getattr(args, "personal_alpha_base_init", 0.08)
                ),
                personal_alpha_bias_scale=loaded_args.get(
                    "personal_alpha_bias_scale", getattr(args, "personal_alpha_bias_scale", 0.0)
                ),
                personal_disable_student_global_context=loaded_args.get(
                    "personal_disable_student_global_context",
                    getattr(args, "personal_disable_student_global_context", False),
                ),
                personal_local_hops=loaded_args.get(
                    "personal_local_hops", getattr(args, "personal_local_hops", 1)
                ),
                personal_include_neighbor_rows=loaded_args.get(
                    "personal_include_neighbor_rows", getattr(args, "personal_include_neighbor_rows", False)
                ),
                personal_query_row_budget=loaded_args.get(
                    "personal_query_row_budget", getattr(args, "personal_query_row_budget", 1.0)
                ),
                personal_neighbor_row_budget=loaded_args.get(
                    "personal_neighbor_row_budget", getattr(args, "personal_neighbor_row_budget", 0.30)
                ),
                personal_query_support_hops=loaded_args.get(
                    "personal_query_support_hops", getattr(args, "personal_query_support_hops", 0)
                ),
                personal_support_only=loaded_args.get(
                    "personal_support_only", getattr(args, "personal_support_only", True)
                ),
                personal_query_message_gain=loaded_args.get(
                    "personal_query_message_gain", getattr(args, "personal_query_message_gain", 1.0)
                ),
                personal_query_correction_scale=loaded_args.get(
                    "personal_query_correction_scale", getattr(args, "personal_query_correction_scale", 0.15)
                ),
                personal_query_correction_max_ratio=loaded_args.get(
                    "personal_query_correction_max_ratio",
                    getattr(args, "personal_query_correction_max_ratio", 0.20),
                ),
                personal_query_correction_min_graph_anchor=loaded_args.get(
                    "personal_query_correction_min_graph_anchor",
                    getattr(args, "personal_query_correction_min_graph_anchor", 0.01),
                ),
                graph_prior_logit_scale=loaded_args.get(
                    "graph_prior_logit_scale", getattr(args, "graph_prior_logit_scale", 0.0)
                ),
                ae_query_residual_scale=loaded_args.get(
                    "ae_query_residual_scale", getattr(args, "ae_query_residual_scale", 0.0)
                ),
                ae_logit_residual_scale=loaded_args.get(
                    "ae_logit_residual_scale", getattr(args, "ae_logit_residual_scale", 0.0)
                ),
                ae_logit_residual_clip=loaded_args.get(
                    "ae_logit_residual_clip", getattr(args, "ae_logit_residual_clip", 1.0)
                ),
                ae_irt_logit_scale=loaded_args.get(
                    "ae_irt_logit_scale", getattr(args, "ae_irt_logit_scale", 1.0)
                ),
                ae_interaction_logit_scale=loaded_args.get(
                    "ae_interaction_logit_scale", getattr(args, "ae_interaction_logit_scale", 0.0)
                ),
                ae_logit_dim=loaded_args.get(
                    "ae_logit_dim", getattr(args, "ae_logit_dim", 32)
                ),
                relation_theta_scale=loaded_args.get(
                    "relation_theta_scale", getattr(args, "relation_theta_scale", 0.0)
                ),
                relation_theta_delta_clip=loaded_args.get(
                    "relation_theta_delta_clip", getattr(args, "relation_theta_delta_clip", 2.0)
                ),
                share_concept_embeddings=loaded_args.get(
                    "share_concept_embeddings", getattr(args, "share_concept_embeddings", False)
                ),
            ).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.set_epoch(int(checkpoint.get("epoch", getattr(args, "epochs", 1))))
            
            # 
            train_file = os.path.join(args.data_dir, "train.csv")
            val_file = os.path.join(args.data_dir, "valid.csv")
            test_file = os.path.join(args.data_dir, "test.csv")
            train_loader, _, _, _ = create_dataloaders(
                train_file=train_file,
                val_file=val_file,
                test_file=test_file,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle_train=False,
                min_stu_interactions=args.min_stu_interactions,
                min_exer_interactions=args.min_exer_interactions,
                min_poison_count=args.min_poison_count,
                logger=logger,
            )
            
            # 
            analysis_data = save_component_analysis_data(
                model=model,
                train_loader=train_loader,
                device=device,
                save_dir=args.save_dir,
                logger=logger,
                num_samples=200,
            )
            
            # 
            try:
                from plot_component_analysis import (
                    plot_global_graph,
                    plot_personal_graph_analysis,
                )
                
                logger.info("Generating component analysis visualizations...")
                plot_global_graph(analysis_data, args.save_dir)
                plot_personal_graph_analysis(analysis_data, args.save_dir)
                logger.info(f"Visualizations saved to {args.save_dir}")
            except Exception as plot_err:
                logger.warning(f"Failed to generate visualizations: {plot_err}")
                
        except Exception as e:
            logger.warning(f"Failed to save component analysis data: {e}")


if __name__ == "__main__":
    main()
