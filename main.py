# main.py
import argparse
import json
import os
import sys

import numpy as np
import torch

from src.config import apply_dataset_defaults
from src.experiment_utils import setup_logging
from src.trainer import train_one_experiment, run_inference, save_component_analysis_data


def _normalize_bool(value, default=False):
    """
    兼容旧的 --generate_diagnosis 传参方式：
    - 可能是 True/False
    - 可能是 "True"/"False"/"1"/"0"/"yes"/"no"
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
    # Model (保留旧参数名，兼容历史脚本/日志)
    # ======================
    parser.add_argument("--knowledge_dim", type=int, default=128)
    parser.add_argument("--skill_dim", type=int, default=64)
    parser.add_argument("--exercise_dim", type=int, default=128)
    parser.add_argument("--num_relation_heads", type=int, default=4)
    parser.add_argument("--num_gnn_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    # ---- prototype（可消融）----
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

    # ---- Regularization（保留旧名字）----
    parser.add_argument(
        "--lambda_sparse",
        type=float,
        default=0.01,
        help="Graph sparsity weight (mapped to graph entropy weight in the new model).",
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

    # 注意：这是“兼容旧行为”的 bool（不是标准 action=store_true）
    parser.add_argument("--generate_diagnosis", default=True)

    # personal graph（可消融/可扩展）
    parser.add_argument("--use_personal_graph", action="store_true")
    parser.add_argument("--model_variant", type=str, default="full")

    # 多 GPU 选择（如果你后续要做自动选卡，可在 trainer/launcher 用）
    parser.add_argument("--gpu_candidates", type=str, default=None)

    # 旧名字：exercise_l2_lambda -> 新模型 mf_l2_lambda
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
    # Ablations（旧开关：子模块级；用于历史对齐）
    # ======================
    parser.add_argument("--ablate_soft_prototype", action="store_true")
    parser.add_argument("--ablate_skill_encoder", action="store_true")   # 关闭 MF 分支
    parser.add_argument("--ablate_concept_graph", action="store_true")   # 关闭 concept graph

    # ======================
    # Ablations（新开关：模块级“完全消融”）
    # ======================
    parser.add_argument("--ablate_module1", action="store_true",
                        help="Fully disable Module 1 (Concept Structure Modeling: A+E+knowledge_encoder).")
    parser.add_argument("--ablate_module2", action="store_true",
                        help="Fully disable Module 2 (IRT diagnosis head D; also removes IRT b/a params).")
    parser.add_argument("--ablate_module3", action="store_true",
                        help="Fully disable Module 3 (Neural residual + prototype: B+C).")

    # ======================
    # New optional knobs（不影响旧脚本，除非显式传参）
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

    # 兼容：老参数如果有人还在用，直接报错提示
    if any(arg.startswith("--ablate_exercise_graph") for arg in sys.argv):
        raise SystemExit("error: --ablate_exercise_graph is removed. Use --ablate_concept_graph instead.")

    args = parser.parse_args()
    args = apply_dataset_defaults(args, parser)

    # =========================================================
    # 0) 基础 sanity check：避免无预测路径
    # =========================================================
    if getattr(args, "ablate_module2", False) and getattr(args, "ablate_module3", False):
        raise SystemExit("error: invalid ablation: both --ablate_module2 and --ablate_module3 are set (no prediction path).")

    # 兼容旧 bool 传法
    args.generate_diagnosis = _normalize_bool(args.generate_diagnosis, default=True)

    # =========================================================
    # 1) 模块级完全消融：先定 enable/disable，再派生子开关
    # =========================================================
    args.enable_module1 = not bool(getattr(args, "ablate_module1", False))
    args.enable_module2 = not bool(getattr(args, "ablate_module2", False))
    args.enable_module3 = not bool(getattr(args, "ablate_module3", False))

    # module2 完全消融时，diagnosis 没意义：强制关掉（避免误用）
    if not args.enable_module2:
        args.generate_diagnosis = False

    # =========================================================
    # 2) 子模块开关：将旧 ablate_* 映射到 use_*
    #    并受“模块级消融”强制覆盖，保证彻底
    # =========================================================

    # (A) Prototype：disable_soft_prototype 或 ablate_soft_prototype 会关闭
    use_soft_proto = (
        (not getattr(args, "disable_soft_prototype", False))
        and (not getattr(args, "ablate_soft_prototype", False))
        and (getattr(args, "num_prototypes", 0) > 0)
    )

    # (B) MF 分支：ablate_skill_encoder == 关闭 MF（no_skill）
    use_mf_branch = not getattr(args, "ablate_skill_encoder", False)

    # (C) 概念图：ablate_concept_graph == 关闭 concept graph（no_concept_graph）
    use_concept_graph = not getattr(args, "ablate_concept_graph", False)

    # (D) personal graph：显式开关
    use_personal_graph = bool(getattr(args, "use_personal_graph", False))

    # ---------- 模块级强制覆盖（保证“完全消融不残留”） ----------
    if not args.enable_module1:
        # 模块1全关：结构模块彻底禁用（A/E/knowledge_encoder 都不应参与）
        use_concept_graph = False
        use_personal_graph = False
        # 这些参数即使留着也不生效，但为了日志清晰与避免 trainer 侧额外正则逻辑，统一归零
        args.num_gnn_layers = 0
        args.lambda_sparse = 0.0
        args.lambda_sparse_personal = 0.0
        args.lambda_alpha = 0.0

    if not args.enable_module3:
        # 模块3全关：MF/Proto 全关
        use_mf_branch = False
        use_soft_proto = False
        args.num_prototypes = 0
        args.lambda_proto_div = 0.0
        args.lambda_proto_usage = 0.0

    if not args.enable_module2:
        # 模块2全关：Prototype 在结构上无贡献且可能形成“伪路径”，强制关
        use_soft_proto = False
        args.num_prototypes = 0
        args.lambda_proto_div = 0.0
        args.lambda_proto_usage = 0.0

    args.use_soft_prototype = bool(use_soft_proto)
    args.use_mf_branch = bool(use_mf_branch)
    args.use_concept_graph = bool(use_concept_graph)
    args.use_personal_graph = bool(use_personal_graph)

    # 兼容历史字段（trainer/日志若仍使用旧名字）
    args.use_skill_encoder = args.use_mf_branch
    args.use_exercise_graph = args.use_concept_graph

    # =========================================================
    # 3) 新增派生字段：避免 trainer/model 之间参数名不一致
    # =========================================================
    # disable_self_loop -> allow_self_loop（模型里用 allow_self_loop）
    args.allow_self_loop = not getattr(args, "disable_self_loop", False)

    # disable_q_conditioning -> use_q_conditioning（模型里用 use_q_conditioning）
    args.use_q_conditioning = not getattr(args, "disable_q_conditioning", False)

    # =========================================================
    # 4) 随机种子（保证可复现）
    # =========================================================
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # =========================================================
    # 5) 目录与日志
    # =========================================================
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    logger = setup_logging(args.log_dir)

    # 保存最终 args（非常关键：消融实验一定要落盘）
    args_file = os.path.join(args.save_dir, "args.json")
    with open(args_file, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)

    logger.info("Arguments:")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"  {arg}: {value}")

    # =========================================================
    # 6) 训练 + 推理
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
    # 7) 保存组件分析数据（用于可视化验证）
    # =========================================================
    if getattr(args, "generate_diagnosis", True):
        try:
            # 需要重新加载模型和数据加载器
            from torch.utils.data import DataLoader
            from src.dataset import create_dataloaders
            
            device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
            
            # 加载最佳模型
            model_path = os.path.join(args.save_dir, "best_model.pth")
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            info_dict = checkpoint["info_dict"]
            
            # 重建模型
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
                use_personal_graph=args.use_personal_graph,
                personal_rank=getattr(args, "personal_rank", 4),
            ).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            
            # 创建数据加载器
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
            
            # 保存组件分析数据
            analysis_data = save_component_analysis_data(
                model=model,
                train_loader=train_loader,
                device=device,
                save_dir=args.save_dir,
                logger=logger,
                num_samples=200,
            )
            
            # 自动生成可视化图表
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
