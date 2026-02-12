# main.py
import argparse
import json
import os
import sys

import numpy as np
import torch

from gpu_utils import configure_main_process_gpus, parse_gpu_ids
from src.config import apply_dataset_defaults
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
    parser.add_argument("--skill_dim", type=int, default=64)
    parser.add_argument("--exercise_dim", type=int, default=128)
    parser.add_argument("--num_relation_heads", type=int, default=4)
    parser.add_argument("--num_gnn_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    # ---- prototype----
    parser.add_argument("--num_prototypes", type=int, default=3)
    parser.add_argument("--proto_tau", type=float, default=1.0)
    parser.add_argument("--proto_lambda", type=float, default=0.5)
    parser.add_argument(
        "--enable_soft_prototype",
        action="store_true",
        help="Enable Soft Prototype module (default is OFF to avoid negative transfer on junyi).",
    )
    parser.add_argument("--disable_soft_prototype", action="store_true")

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
        default=3,
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
        default=0.95,
        help="Upper bound of normalized graph entropy band.",
    )
    parser.add_argument(
        "--lambda_graph_diag",
        type=float,
        default=0.05,
        help="Penalty weight for graph diagonal mass (near-identity suppressor).",
    )
    parser.add_argument(
        "--lambda_graph_uniform",
        type=float,
        default=0.02,
        help="Penalty weight to keep graph away from uniform adjacency.",
    )
    parser.add_argument(
        "--graph_uniform_margin",
        type=float,
        default=0.08,
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

    #  bool action=store_true
    parser.add_argument("--generate_diagnosis", default=True)

    # personal graph/
    parser.add_argument("--use_personal_graph", action="store_true")
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

    # exercise_l2_lambda ->  mf_l2_lambda
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

    # ======================
    # Ablations
    # ======================
    parser.add_argument("--ablate_soft_prototype", action="store_true")
    parser.add_argument("--ablate_skill_encoder", action="store_true")   #  MF 
    parser.add_argument("--ablate_concept_graph", action="store_true")   #  concept graph

    # ======================
    # Ablations
    # ======================
    parser.add_argument("--ablate_module1", action="store_true",
                        help="Fully disable Module 1 (Concept Structure Modeling: A+E+knowledge_encoder).")
    parser.add_argument("--ablate_module2", action="store_true",
                        help="Fully disable Module 2 (IRT diagnosis head D; also removes IRT b/a params).")
    parser.add_argument("--ablate_module3", action="store_true",
                        help="Fully disable Module 3 (Neural residual + prototype: B+C).")

    # ======================
    # New optional knobs
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

    # Module3 conservative fusion / residual knobs
    parser.add_argument("--fusion_gate_max", type=float, default=1.0,
                        help="Maximum residual gate amplitude in conservative fusion.")
    parser.add_argument("--fusion_gate_bias_init", type=float, default=-1.1,
                        help="Initial bias for conservative fusion gate (negative => small initial gate).")
    parser.add_argument("--residual_clip_t", type=float, default=2.0,
                        help="T for residual clipping: residual = T * tanh(residual / T).")
    parser.add_argument("--residual_scale_init", type=float, default=0.1,
                        help="Initial positive scale for module3 residual branches (after softplus).")
    parser.add_argument("--disable_q_aligned_residual", action="store_true",
                        help="Compatibility flag; q-aligned residual is enabled by default.")
    parser.add_argument("--use_soft_prototype_main_path", action="store_true",
                        help="If set, prototype mix is injected into knowledge_state. Default off.")

    #  GPU 
    parser.add_argument("--multi_gpu", action="store_true",
                        help="Enable multi-GPU training with DataParallel.")
    parser.add_argument("--gpu_ids", type=str, default=None,
                        help="Comma-separated GPU IDs for multi-GPU training (e.g., '0,1').")

    # 
    parser.add_argument(
        "--debug_module3_diag",
        action="store_true",
        help="Enable per-epoch module diagnostics for module1/module3 and gradient norms.",
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

    args = parser.parse_args()
    args = apply_dataset_defaults(args, parser)

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

    # =========================================================
    # 0)  sanity check
    # =========================================================
    if getattr(args, "ablate_module2", False) and getattr(args, "ablate_module3", False):
        raise SystemExit("error: invalid ablation: both --ablate_module2 and --ablate_module3 are set (no prediction path).")

    #  bool 
    args.generate_diagnosis = _normalize_bool(args.generate_diagnosis, default=True)

    # =========================================================
    # 1)  enable/disable
    # =========================================================
    args.enable_module1 = not bool(getattr(args, "ablate_module1", False))
    args.enable_module2 = not bool(getattr(args, "ablate_module2", False))
    args.enable_module3 = not bool(getattr(args, "ablate_module3", False))

    # module2 diagnosis 
    if not args.enable_module2:
        args.generate_diagnosis = False

    # =========================================================
    # 2)  ablate_*  use_*
    #    
    # =========================================================

    # (A) Prototype --enable_soft_prototype 
    use_soft_proto = (
        bool(getattr(args, "enable_soft_prototype", False))
        and
        (not getattr(args, "disable_soft_prototype", False))
        and (not getattr(args, "ablate_soft_prototype", False))
        and (getattr(args, "num_prototypes", 0) > 0)
    )

    # (B) MF ablate_skill_encoder ==  MFno_skill
    use_mf_branch = not getattr(args, "ablate_skill_encoder", False)

    # (C) ablate_concept_graph ==  concept graphno_concept_graph
    use_concept_graph = not getattr(args, "ablate_concept_graph", False)

    # (D) personal graph
    use_personal_graph = bool(getattr(args, "use_personal_graph", False))

    # ----------  ----------
    if not args.enable_module1:
        # 1A/E/knowledge_encoder 
        use_concept_graph = False
        use_personal_graph = False
        #  trainer 
        args.num_gnn_layers = 0
        args.lambda_sparse = 0.0
        args.lambda_sparse_personal = 0.0
        args.lambda_alpha = 0.0

    if not args.enable_module3:
        # 3MF/Proto 
        use_mf_branch = False
        use_soft_proto = False
        args.num_prototypes = 0
        args.lambda_proto_div = 0.0
        args.lambda_proto_usage = 0.0

    if not args.enable_module2:
        # 2Prototype 
        use_soft_proto = False
        args.num_prototypes = 0
        args.lambda_proto_div = 0.0
        args.lambda_proto_usage = 0.0

    args.use_soft_prototype = bool(use_soft_proto)
    args.use_mf_branch = bool(use_mf_branch)
    args.use_concept_graph = bool(use_concept_graph)
    args.use_personal_graph = bool(use_personal_graph)
    args.use_q_aligned_residual = not bool(getattr(args, "disable_q_aligned_residual", False))
    args.use_soft_prototype_main_path = bool(
        getattr(args, "use_soft_prototype_main_path", False) and args.use_soft_prototype
    )

    # trainer/
    args.use_skill_encoder = args.use_mf_branch
    args.use_exercise_graph = args.use_concept_graph

    # =========================================================
    # 3)  trainer/model 
    # =========================================================
    # disable_self_loop -> allow_self_loop allow_self_loop
    args.allow_self_loop = not getattr(args, "disable_self_loop", False)

    # disable_q_conditioning -> use_q_conditioning use_q_conditioning
    args.use_q_conditioning = not getattr(args, "disable_q_conditioning", False)

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
    best_val_auc, best_epoch = train_one_experiment(args, logger)
    metrics, _ = run_inference(args, logger)

    if not metrics:
        logger.error("Inference failed; check logs.")
        return

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
            
            # 
            from src.model import CognitiveDiagnosisModel
            model = CognitiveDiagnosisModel(
                num_students=info_dict["num_students"],
                num_exercises=info_dict["num_exercises"],
                num_concepts=info_dict["num_concepts"],
                q_matrix=info_dict["q_matrix"],
                knowledge_dim=args.knowledge_dim,
                skill_dim=args.skill_dim,
                exercise_dim=args.exercise_dim,
                num_relation_heads=args.num_relation_heads,
                num_gnn_layers=args.num_gnn_layers if args.use_concept_graph else 0,
                dropout=args.dropout,
                use_mf_branch=args.use_mf_branch,
                use_concept_graph=args.use_concept_graph,
                num_prototypes=args.num_prototypes if args.use_soft_prototype else 0,
                proto_tau=args.proto_tau,
                proto_lambda=args.proto_lambda,
                use_soft_prototype=args.use_soft_prototype,
                use_soft_prototype_main_path=args.use_soft_prototype_main_path,
                use_personal_graph=args.use_personal_graph,
                ablate_module1=getattr(args, "ablate_module1", False),
                ablate_module2=getattr(args, "ablate_module2", False),
                ablate_module3=getattr(args, "ablate_module3", False),
                use_q_aligned_residual=getattr(args, "use_q_aligned_residual", True),
                fusion_gate_max=getattr(args, "fusion_gate_max", 1.0),
                fusion_gate_bias_init=getattr(args, "fusion_gate_bias_init", -1.1),
                residual_clip_t=getattr(args, "residual_clip_t", 2.0),
                residual_scale_init=getattr(args, "residual_scale_init", 0.1),
                graph_dropout=None if float(getattr(args, "graph_dropout", -1.0)) < 0 else float(getattr(args, "graph_dropout", -1.0)),
                graph_tau_init=getattr(args, "graph_tau_init", 1.0),
                personal_rank=getattr(args, "personal_rank", 4),
                lambda_graph_entropy=getattr(args, "lambda_sparse", 0.01),
                graph_entropy_min=getattr(args, "graph_entropy_min", 0.15),
                graph_entropy_max=getattr(args, "graph_entropy_max", 0.95),
                lambda_graph_diag=getattr(args, "lambda_graph_diag", 0.05),
                lambda_graph_uniform=getattr(args, "lambda_graph_uniform", 0.02),
                graph_uniform_margin=getattr(args, "graph_uniform_margin", 0.08),
                graph_reg_warmup_epochs=getattr(args, "graph_reg_warmup_epochs", 3),
                graph_reg_cap_ratio=getattr(args, "graph_reg_cap_ratio", 6.0),
                lambda_sparse_personal=getattr(args, "lambda_sparse_personal", 0.0),
                lambda_alpha=getattr(args, "lambda_alpha", 0.0),
                mf_l2_lambda=getattr(args, "exercise_l2_lambda", 5e-5),
            ).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            
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
                    plot_prototype_analysis,
                    plot_global_graph,
                    plot_personal_graph_analysis,
                )
                
                logger.info("Generating component analysis visualizations...")
                plot_prototype_analysis(analysis_data, args.save_dir)
                plot_global_graph(analysis_data, args.save_dir)
                plot_personal_graph_analysis(analysis_data, args.save_dir)
                logger.info(f"Visualizations saved to {args.save_dir}")
            except Exception as plot_err:
                logger.warning(f"Failed to generate visualizations: {plot_err}")
                
        except Exception as e:
            logger.warning(f"Failed to save component analysis data: {e}")


if __name__ == "__main__":
    main()


