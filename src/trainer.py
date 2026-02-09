# src/trainer.py
import json
import os
import warnings
from typing import Tuple, Dict, Any, Optional, Union, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 过滤 DataParallel 的 gather 标量警告（这是正常行为，不影响结果）
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0")

from src.dataset import CognitiveDiagnosisDataset, create_dataloaders
from src.model import CognitiveDiagnosisModel
from src.experiment_utils import (
    compute_metrics,
    select_device,
    save_epoch_history_csv,
    append_summary_csv,
)
from src.module_activity import (
    compute_module_activity,
    format_activity_brief,
    format_activity_report,
)

# =========================
# Global toggles
# =========================
# If True: fail fast when checkpoint/model keys mismatch (recommended for ablations)
STRICT_CHECKPOINT_LOADING = True

# ======================================================
# Helpers
# ======================================================

def _sigmoid_torch(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable sigmoid for metrics; keep in torch for speed."""
    return torch.sigmoid(x)


def _ensure_1d(t: torch.Tensor) -> torch.Tensor:
    """Ensure tensor shape (B,) for logits/labels. Use reshape to handle non-contiguous tensors."""
    return t.reshape(-1)


def _get_base_model(model: nn.Module) -> CognitiveDiagnosisModel:
    """获取基础模型（处理 DataParallel 包装）"""
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def _collect_runtime_ablation_facts(model: nn.Module) -> Dict[str, Any]:
    """收集模型运行时的模块开关与物理存在性，用于防止“假消融”"""
    base_model = _get_base_model(model)
    structure_module = getattr(base_model, "structure_module", None)

    has_knowledge_encoder = (
        structure_module is not None and getattr(structure_module, "knowledge_encoder", None) is not None
    )
    has_mf_branch = (
        getattr(base_model, "skill_encoder", None) is not None
        and getattr(base_model, "mf_head", None) is not None
    )
    has_soft_prototype = getattr(base_model, "prototype_module", None) is not None

    return {
        "enable_module1": bool(getattr(base_model, "enable_module1", False)),
        "enable_module2": bool(getattr(base_model, "enable_module2", False)),
        "enable_module3": bool(getattr(base_model, "enable_module3", False)),
        "has_knowledge_encoder": bool(has_knowledge_encoder),
        "has_mf_branch": bool(has_mf_branch),
        "has_soft_prototype": bool(has_soft_prototype),
    }


def _log_and_assert_ablation_consistency(
    *,
    model: nn.Module,
    logger,
    context: str,
    ablate_module1: bool,
    ablate_module2: bool,
    ablate_module3: bool,
) -> Dict[str, Any]:
    """
    记录并校验消融一致性：
    - args.ablate_module* 与 model.enable_module* 必须一致
    - 关键模块的“物理存在性”必须符合预期
    """
    facts = _collect_runtime_ablation_facts(model)

    logger.info(
        "%s Ablation runtime check: "
        "args(ablate_module1=%s,ablate_module2=%s,ablate_module3=%s) | "
        "model(enable_module1=%s,enable_module2=%s,enable_module3=%s) | "
        "physical(has_knowledge_encoder=%s,has_mf_branch=%s,has_soft_prototype=%s)",
        context,
        ablate_module1,
        ablate_module2,
        ablate_module3,
        facts["enable_module1"],
        facts["enable_module2"],
        facts["enable_module3"],
        facts["has_knowledge_encoder"],
        facts["has_mf_branch"],
        facts["has_soft_prototype"],
    )

    if ablate_module1 and facts["enable_module1"]:
        raise RuntimeError("Ablation mismatch: ablate_module1=True but model.enable_module1=True.")
    if ablate_module2 and facts["enable_module2"]:
        raise RuntimeError("Ablation mismatch: ablate_module2=True but model.enable_module2=True.")
    if ablate_module3 and facts["enable_module3"]:
        raise RuntimeError("Ablation mismatch: ablate_module3=True but model.enable_module3=True.")

    if ablate_module1 and facts["has_knowledge_encoder"]:
        raise RuntimeError("Ablation mismatch: ablate_module1=True but knowledge_encoder still exists.")
    if ablate_module3 and facts["has_mf_branch"]:
        raise RuntimeError("Ablation mismatch: ablate_module3=True but MF branch modules still exist.")
    if ablate_module3 and facts["has_soft_prototype"]:
        raise RuntimeError("Ablation mismatch: ablate_module3=True but prototype_module still exists.")

    return facts


def _convert_legacy_weight_norm_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Compatibility helper:
    Old torch.nn.utils.weight_norm produced:
      ...theta_proj.weight_g, ...theta_proj.weight_v
    New torch.nn.utils.parametrizations.weight_norm produces:
      ...theta_proj.parametrizations.weight.original0 (g)
      ...theta_proj.parametrizations.weight.original1 (v)
    """
    has_legacy = any(k.endswith("weight_g") or k.endswith("weight_v") for k in state_dict.keys())
    if not has_legacy:
        return state_dict

    new_sd = dict(state_dict)
    keys = list(state_dict.keys())
    for k in keys:
        if k.endswith("weight_g"):
            base = k[:-len("weight_g")]
            new_k = base + "parametrizations.weight.original0"
            new_sd[new_k] = new_sd.pop(k)
        elif k.endswith("weight_v"):
            base = k[:-len("weight_v")]
            new_k = base + "parametrizations.weight.original1"
            new_sd[new_k] = new_sd.pop(k)
    return new_sd


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    移除 DataParallel 保存的 state_dict 中的 'module.' 前缀。
    
    DataParallel 训练时保存的权重格式为 'module.xxx'，
    但在非 DataParallel 模型中加载时需要移除这个前缀。
    """
    has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
    if not has_module_prefix:
        return state_dict
    
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_sd[k[7:]] = v  # 移除 "module." 前缀（7个字符）
        else:
            new_sd[k] = v
    return new_sd


def _hard_ablation_effective_hparams(
    *,
    use_soft_prototype: bool,
    use_concept_graph: bool,
    num_gnn_layers: int,
    num_prototypes: int,
) -> Tuple[int, int]:
    """
    Hard-ablation safety:
      - If concept graph is disabled, force num_gnn_layers=0 to prevent "I-graph GNN" leakage.
      - If soft prototype is disabled, force num_prototypes=0 to prevent prototype params from existing.
    """
    eff_gnn = int(num_gnn_layers)
    if not use_concept_graph:
        eff_gnn = 0

    eff_proto = int(num_prototypes)
    if not use_soft_prototype:
        eff_proto = 0

    return eff_gnn, eff_proto


# ======================================================
# Train / Validate (use BCEWithLogitsLoss)
# ======================================================

def train_epoch(
    model: CognitiveDiagnosisModel,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    lambda_proto_div: float,
    lambda_proto_usage: float,
    logger,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_bce = 0.0
    total_reg = 0.0

    all_labels: List[float] = []
    all_preds: List[float] = []
    all_probs: List[float] = []

    bce_fn = nn.BCEWithLogitsLoss()

    for batch_idx, batch in enumerate(train_loader):
        student_ids, exercise_ids, concept_vector, labels = batch
        student_ids = student_ids.to(device)
        exercise_ids = exercise_ids.to(device)
        concept_vector = concept_vector.to(device)
        labels = _ensure_1d(labels.to(device).float())

        # get logits + details (for regularizers)
        logits, details = model(
            student_ids,
            exercise_ids,
            concept_vector=concept_vector,
            return_details=True,
            return_logits=True,
        )
        logits = _ensure_1d(logits)

        bce_loss = bce_fn(logits, labels)
        reg_loss = _get_base_model(model).get_regularization_loss(
            relation_matrices=details["relation_matrices"],
            details=details,
            lambda_proto_div=lambda_proto_div,
            lambda_proto_usage=lambda_proto_usage,
        )
        loss = bce_loss + reg_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += float(loss.item())
        total_bce += float(bce_loss.item())
        total_reg += float(reg_loss.item())

        with torch.no_grad():
            probs = _sigmoid_torch(logits)
            preds = (probs > 0.5).float()

        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_probs.extend(probs.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(1, len(train_loader))
    avg_bce = total_bce / max(1, len(train_loader))
    avg_reg = total_reg / max(1, len(train_loader))
    metrics = compute_metrics(all_labels, all_preds, all_probs)

    return {"loss": avg_loss, "bce_loss": avg_bce, "reg_loss": avg_reg, **metrics}


def validate(
    model: CognitiveDiagnosisModel,
    val_loader: DataLoader,
    device: torch.device,
    lambda_proto_div: float,
    lambda_proto_usage: float,
    logger,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_bce = 0.0
    total_reg = 0.0

    all_labels: List[float] = []
    all_preds: List[float] = []
    all_probs: List[float] = []

    bce_fn = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            student_ids, exercise_ids, concept_vector, labels = batch
            student_ids = student_ids.to(device)
            exercise_ids = exercise_ids.to(device)
            concept_vector = concept_vector.to(device)
            labels = _ensure_1d(labels.to(device).float())

            logits, details = model(
                student_ids,
                exercise_ids,
                concept_vector=concept_vector,
                return_details=True,
                return_logits=True,
            )
            logits = _ensure_1d(logits)

            bce_loss = bce_fn(logits, labels)
            reg_loss = _get_base_model(model).get_regularization_loss(
                relation_matrices=details["relation_matrices"],
                details=details,
                lambda_proto_div=lambda_proto_div,
                lambda_proto_usage=lambda_proto_usage,
            )
            loss = bce_loss + reg_loss

            total_loss += float(loss.item())
            total_bce += float(bce_loss.item())
            total_reg += float(reg_loss.item())

            probs = _sigmoid_torch(logits)
            preds = (probs > 0.5).float()

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    avg_loss = total_loss / max(1, len(val_loader))
    avg_bce = total_bce / max(1, len(val_loader))
    avg_reg = total_reg / max(1, len(val_loader))
    metrics = compute_metrics(all_labels, all_preds, all_probs)

    return {"loss": avg_loss, "bce_loss": avg_bce, "reg_loss": avg_reg, **metrics}


# ======================================================
# Train / Inference
# ======================================================

def train_one_experiment(args, logger) -> Tuple[float, int]:
    device = select_device(args, logger)

    run_tag = (
        f"[{getattr(args, 'dataset_name', 'unknown')}"
        f"|{getattr(args, 'model_variant', 'full')}"
        f"|lr={args.learning_rate:g}"
        f"|drop={args.dropout:.2f}]"
    )

    logger.info("%s Loading datasets...", run_tag)
    logger.info(
        "%s Regularization: graph_entropy(lambda_sparse)=%.6f, proto_div=%.6f, proto_usage=%.6f, "
        "personal_sparse=%.6f, alpha_penalty=%.6f, mf_l2(exercise_l2_lambda)=%.6f",
        run_tag,
        args.lambda_sparse,
        args.lambda_proto_div,
        args.lambda_proto_usage,
        args.lambda_sparse_personal,
        args.lambda_alpha,
        getattr(args, "exercise_l2_lambda", 5e-5),
    )

    data_dir = args.data_dir
    train_file = os.path.join(data_dir, "train.csv")
    val_file = os.path.join(data_dir, "valid.csv")
    test_file = os.path.join(data_dir, "test.csv")

    train_loader, val_loader, test_loader, info_dict = create_dataloaders(
        train_file=train_file,
        val_file=val_file,
        test_file=test_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle_train=True,
        min_stu_interactions=args.min_stu_interactions,
        min_exer_interactions=args.min_exer_interactions,
        min_poison_count=args.min_poison_count,
        logger=logger,
    )

    logger.info(
        "%s Train samples: %d, Val samples: %d",
        run_tag,
        info_dict["train_size"],
        info_dict["val_size"],
    )
    logger.info("%s Number of students: %d", run_tag, info_dict["num_students"])
    logger.info("%s Number of exercises: %d", run_tag, info_dict["num_exercises"])
    logger.info("%s Number of concepts: %d", run_tag, info_dict["num_concepts"])

    logger.info("%s Creating model...", run_tag)

    # switches from main.py (already normalized there)
    use_soft_prototype = getattr(args, "use_soft_prototype", True)
    use_mf_branch = getattr(args, "use_mf_branch", getattr(args, "use_skill_encoder", True))
    use_concept_graph = getattr(args, "use_concept_graph", True)
    ablate_module1 = bool(getattr(args, "ablate_module1", False))
    ablate_module2 = bool(getattr(args, "ablate_module2", False))
    ablate_module3 = bool(getattr(args, "ablate_module3", False))
    expected_enable_module1 = not ablate_module1
    expected_enable_module2 = not ablate_module2
    expected_enable_module3 = not ablate_module3

    # hard-ablation safety (prevents "half-ablation")
    eff_gnn_layers, eff_num_prototypes = _hard_ablation_effective_hparams(
        use_soft_prototype=use_soft_prototype,
        use_concept_graph=use_concept_graph,
        num_gnn_layers=getattr(args, "num_gnn_layers", 0),
        num_prototypes=getattr(args, "num_prototypes", 0),
    )

    logger.info(
        "%s Ablation switches: "
        "ablate_module1=%s, ablate_module2=%s, ablate_module3=%s | "
        "enable_module1=%s, enable_module2=%s, enable_module3=%s | "
        "use_soft_prototype=%s, use_mf_branch=%s, use_concept_graph=%s | "
        "effective(num_gnn_layers=%d, num_prototypes=%d)",
        run_tag,
        ablate_module1,
        ablate_module2,
        ablate_module3,
        expected_enable_module1,
        expected_enable_module2,
        expected_enable_module3,
        use_soft_prototype,
        use_mf_branch,
        use_concept_graph,
        eff_gnn_layers,
        eff_num_prototypes,
    )

    model = CognitiveDiagnosisModel(
        num_students=info_dict["num_students"],
        num_exercises=info_dict["num_exercises"],
        num_concepts=info_dict["num_concepts"],
        q_matrix=info_dict["q_matrix"],
        knowledge_dim=args.knowledge_dim,
        skill_dim=args.skill_dim,
        exercise_dim=args.exercise_dim,
        num_relation_heads=args.num_relation_heads,
        num_gnn_layers=eff_gnn_layers,
        dropout=args.dropout,
        use_mf_branch=use_mf_branch,
        use_concept_graph=use_concept_graph,
        graph_topk=getattr(args, "graph_topk", None),
        allow_self_loop=not getattr(args, "disable_self_loop", False),
        num_prototypes=eff_num_prototypes,
        proto_tau=args.proto_tau,
        proto_lambda=args.proto_lambda,
        use_soft_prototype=use_soft_prototype,
        use_personal_graph=getattr(args, "use_personal_graph", False),
        ablate_module1=ablate_module1,
        ablate_module2=ablate_module2,
        ablate_module3=ablate_module3,
        personal_rank=getattr(args, "personal_rank", 4),
        lambda_sparse_personal=args.lambda_sparse_personal,
        lambda_alpha=args.lambda_alpha,
        lambda_graph_entropy=args.lambda_sparse,
        mf_l2_lambda=getattr(args, "exercise_l2_lambda", 5e-5),
        gnn_residual_weight=getattr(args, "gnn_residual_weight", 0.5),
        use_q_conditioning=not getattr(args, "disable_q_conditioning", False),
    ).to(device)

    # 多 GPU 支持（DataParallel）
    is_multi_gpu = getattr(args, "multi_gpu", False) and torch.cuda.device_count() > 1
    if is_multi_gpu:
        gpu_ids_str = getattr(args, "gpu_ids", None)
        if gpu_ids_str:
            # 使用指定的 GPU（已通过 CUDA_VISIBLE_DEVICES 映射，这里用相对索引）
            num_visible = torch.cuda.device_count()
            device_ids = list(range(num_visible))
        else:
            device_ids = None
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        logger.info("%s Multi-GPU enabled: using %d GPUs", run_tag, torch.cuda.device_count())

    _log_and_assert_ablation_consistency(
        model=model,
        logger=logger,
        context=run_tag,
        ablate_module1=ablate_module1,
        ablate_module2=ablate_module2,
        ablate_module3=ablate_module3,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("%s Total parameters: %s", run_tag, f"{total_params:,}")
    logger.info("%s Trainable parameters: %s", run_tag, f"{trainable_params:,}")

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.patience,
    )

    best_val_auc = 0.0
    best_epoch = 0
    patience_counter = 0

    history: Dict[str, Any] = {"train": [], "val": [], "best_epoch": 0, "best_val_auc": 0.0}

    logger.info("%s Starting training...", run_tag)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.lambda_proto_div,
            args.lambda_proto_usage,
            logger,
        )

        val_metrics = validate(
            model,
            val_loader,
            device,
            args.lambda_proto_div,
            args.lambda_proto_usage,
            logger,
        )

        logger.info(
            "%s Epoch [%03d/%d] | "
            "Train: Loss=%.4f, BCE=%.4f, Reg=%.4f, AUC=%.4f, ACC=%.4f, RMSE=%.4f | "
            "Val: Loss=%.4f, BCE=%.4f, Reg=%.4f, AUC=%.4f, ACC=%.4f, RMSE=%.4f",
            run_tag,
            epoch, args.epochs,
            train_metrics["loss"], train_metrics["bce_loss"], train_metrics["reg_loss"],
            train_metrics["auc"], train_metrics["acc"], train_metrics["rmse"],
            val_metrics["loss"], val_metrics["bce_loss"], val_metrics["reg_loss"],
            val_metrics["auc"], val_metrics["acc"], val_metrics["rmse"],
        )

        scheduler.step(val_metrics["loss"])

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = float(val_metrics["auc"])
            best_epoch = epoch
            patience_counter = 0

            model_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": _get_base_model(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc": best_val_auc,
                    "val_metrics": val_metrics,
                    "args": vars(args),
                    "info_dict": info_dict,
                },
                model_path,
            )
            logger.info("%s -> New best AUC=%.4f at epoch %d", run_tag, best_val_auc, epoch)
        else:
            patience_counter += 1
            logger.info(
                "%s -> No improvement %d epoch(s) (best AUC=%.4f @ %d)",
                run_tag, patience_counter, best_val_auc, best_epoch
            )

        if patience_counter >= args.early_stop_patience:
            logger.info(
                "%s Early stopping at epoch %d (best AUC=%.4f @ %d)",
                run_tag, epoch, best_val_auc, best_epoch
            )
            break

        if epoch % args.save_interval == 0:
            checkpoint_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": _get_base_model(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "train_metrics": train_metrics,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            logger.info("%s Checkpoint saved: %s", run_tag, checkpoint_path)

            # 模块活跃度检测（每 save_interval 输出简报）
            try:
                activity = compute_module_activity(model, val_loader, device, num_samples=300)
                brief = format_activity_brief(activity)
                logger.info("%s [Module Activity] Epoch %d: %s", run_tag, epoch, brief)
            except Exception as e:
                logger.warning("%s [Module Activity] Failed: %s", run_tag, str(e))

    history["best_epoch"] = best_epoch
    history["best_val_auc"] = best_val_auc

    history_path = os.path.join(args.save_dir, "training_history.json")
    save_epoch_history_csv(history, args.save_dir, logger)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    logger.info("%s Training completed! Best Val AUC=%.4f @ epoch %d", run_tag, best_val_auc, best_epoch)

    # ========== 训练结束：输出完整的模块活跃度报告 ==========
    try:
        activity = compute_module_activity(model, val_loader, device, num_samples=500)
        report = format_activity_report(
            activity,
            dataset_name=getattr(args, 'dataset_name', 'unknown'),
            seed=getattr(args, 'seed', 42),
            epoch=epoch,
        )
        logger.info("\n%s", report)

        # 保存活跃度数据到 JSON
        activity_path = os.path.join(args.save_dir, "module_activity.json")
        with open(activity_path, "w") as f:
            json.dump(activity, f, indent=4)
        logger.info("%s Module activity saved to %s", run_tag, activity_path)
    except Exception as e:
        logger.warning("%s [Module Activity Report] Failed: %s", run_tag, str(e))

    return best_val_auc, best_epoch


def run_inference(args, logger) -> Tuple[Dict[str, float], Dict[str, Any]]:
    device = select_device(args, logger)

    data_dir = args.data_dir
    test_file = os.path.join(data_dir, "test.csv")

    model_path = os.path.join(args.save_dir, "best_model.pth")
    logger.info(f"Loading model from {model_path}...")

    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return {}, {}

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    loaded_args: Dict[str, Any] = checkpoint.get("args", {})
    info_dict = checkpoint.get("info_dict", None)

    if info_dict is None:
        logger.error("info_dict not found in checkpoint. Please retrain with current code.")
        return {}, {}

    stu_id_map = info_dict["stu_id_map"]
    exer_id_map = info_dict["exer_id_map"]
    cpt_id_map = info_dict["cpt_id_map"]
    q_matrix = info_dict["q_matrix"]

    raw_test_df = pd.read_csv(test_file)

    valid_stu_ids = set(stu_id_map.keys())
    valid_exer_ids = set(exer_id_map.keys())
    before_rows = len(raw_test_df)
    filtered_test_df = raw_test_df[
        raw_test_df["stu_id"].isin(valid_stu_ids) & raw_test_df["exer_id"].isin(valid_exer_ids)
    ].reset_index(drop=True)
    after_rows = len(filtered_test_df)
    dropped = before_rows - after_rows

    if dropped > 0:
        logger.info(
            f"[Inference Filter] before={before_rows}, after={after_rows}, dropped={dropped} "
            f"(students/items not seen after cleaning)"
        )

    test_dataset = CognitiveDiagnosisDataset(
        csv_file=filtered_test_df,
        stu_id_map=stu_id_map,
        exer_id_map=exer_id_map,
        cpt_id_map=cpt_id_map,
    )

    pin_memory = bool(getattr(device, "type", "cpu") == "cuda")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    # Build model from loaded args (fallback to current args)
    use_soft_prototype = loaded_args.get("use_soft_prototype", getattr(args, "use_soft_prototype", True))
    use_mf_branch = loaded_args.get(
        "use_mf_branch",
        loaded_args.get("use_skill_encoder", getattr(args, "use_mf_branch", getattr(args, "use_skill_encoder", True))),
    )
    use_concept_graph = loaded_args.get("use_concept_graph", getattr(args, "use_concept_graph", True))
    ablate_module1 = bool(loaded_args.get("ablate_module1", getattr(args, "ablate_module1", False)))
    ablate_module2 = bool(loaded_args.get("ablate_module2", getattr(args, "ablate_module2", False)))
    ablate_module3 = bool(loaded_args.get("ablate_module3", getattr(args, "ablate_module3", False)))
    expected_enable_module1 = not ablate_module1
    expected_enable_module2 = not ablate_module2
    expected_enable_module3 = not ablate_module3

    # hard-ablation safety at inference too
    eff_gnn_layers, eff_num_prototypes = _hard_ablation_effective_hparams(
        use_soft_prototype=use_soft_prototype,
        use_concept_graph=use_concept_graph,
        num_gnn_layers=int(loaded_args.get("num_gnn_layers", getattr(args, "num_gnn_layers", 0))),
        num_prototypes=int(loaded_args.get("num_prototypes", getattr(args, "num_prototypes", 0))),
    )

    logger.info(
        "Inference switches: "
        "ablate_module1=%s, ablate_module2=%s, ablate_module3=%s | "
        "enable_module1=%s, enable_module2=%s, enable_module3=%s | "
        "use_soft_prototype=%s, use_mf_branch=%s, use_concept_graph=%s | "
        "effective(num_gnn_layers=%d, num_prototypes=%d)",
        ablate_module1,
        ablate_module2,
        ablate_module3,
        expected_enable_module1,
        expected_enable_module2,
        expected_enable_module3,
        use_soft_prototype,
        use_mf_branch,
        use_concept_graph,
        eff_gnn_layers,
        eff_num_prototypes,
    )

    model = CognitiveDiagnosisModel(
        num_students=info_dict["num_students"],
        num_exercises=info_dict["num_exercises"],
        num_concepts=info_dict["num_concepts"],
        q_matrix=q_matrix,
        knowledge_dim=loaded_args.get("knowledge_dim", args.knowledge_dim),
        skill_dim=loaded_args.get("skill_dim", args.skill_dim),
        exercise_dim=loaded_args.get("exercise_dim", args.exercise_dim),
        num_relation_heads=loaded_args.get("num_relation_heads", args.num_relation_heads),
        num_gnn_layers=eff_gnn_layers,
        dropout=loaded_args.get("dropout", args.dropout),
        use_mf_branch=use_mf_branch,
        use_concept_graph=use_concept_graph,
        graph_topk=loaded_args.get("graph_topk", getattr(args, "graph_topk", None)),
        allow_self_loop=not loaded_args.get("disable_self_loop", getattr(args, "disable_self_loop", False)),
        num_prototypes=eff_num_prototypes,
        proto_tau=loaded_args.get("proto_tau", args.proto_tau),
        proto_lambda=loaded_args.get("proto_lambda", args.proto_lambda),
        use_soft_prototype=use_soft_prototype,
        use_personal_graph=loaded_args.get("use_personal_graph", getattr(args, "use_personal_graph", False)),
        ablate_module1=ablate_module1,
        ablate_module2=ablate_module2,
        ablate_module3=ablate_module3,
        personal_rank=loaded_args.get("personal_rank", getattr(args, "personal_rank", 4)),
        lambda_sparse_personal=loaded_args.get("lambda_sparse_personal", args.lambda_sparse_personal),
        lambda_alpha=loaded_args.get("lambda_alpha", args.lambda_alpha),
        lambda_graph_entropy=loaded_args.get("lambda_sparse", args.lambda_sparse),
        mf_l2_lambda=loaded_args.get("exercise_l2_lambda", getattr(args, "exercise_l2_lambda", 5e-5)),
        gnn_residual_weight=loaded_args.get("gnn_residual_weight", getattr(args, "gnn_residual_weight", 0.5)),
        use_q_conditioning=not loaded_args.get("disable_q_conditioning", getattr(args, "disable_q_conditioning", False)),
    ).to(device)

    runtime_facts = _log_and_assert_ablation_consistency(
        model=model,
        logger=logger,
        context="[Inference]",
        ablate_module1=ablate_module1,
        ablate_module2=ablate_module2,
        ablate_module3=ablate_module3,
    )

    # compatibility for legacy weight_norm checkpoints
    state_dict = checkpoint["model_state_dict"]
    state_dict = _convert_legacy_weight_norm_keys(state_dict)
    state_dict = _strip_module_prefix(state_dict)  # 处理 DataParallel 的 module. 前缀

    incompatible = model.load_state_dict(state_dict, strict=False)
    missing_keys = list(getattr(incompatible, "missing_keys", []))
    unexpected_keys = list(getattr(incompatible, "unexpected_keys", []))

    if missing_keys or unexpected_keys:
        logger.warning("State dict mismatch detected.")
        logger.warning("  Missing keys (%d): %s", len(missing_keys), missing_keys[:50])
        logger.warning("  Unexpected keys (%d): %s", len(unexpected_keys), unexpected_keys[:50])

        if STRICT_CHECKPOINT_LOADING:
            raise RuntimeError(
                f"Checkpoint/model architecture mismatch. missing={len(missing_keys)}, unexpected={len(unexpected_keys)}"
            )

    logger.info(f"Model loaded from epoch {checkpoint['epoch']}. Start testing...")

    model.eval()
    all_labels: List[float] = []
    all_preds: List[float] = []
    all_probs: List[float] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            student_ids, exercise_ids, concept_vector, labels = batch
            student_ids = student_ids.to(device)
            exercise_ids = exercise_ids.to(device)
            concept_vector = concept_vector.to(device)
            labels = _ensure_1d(labels.to(device).float())

            logits = model(
                student_ids,
                exercise_ids,
                concept_vector=concept_vector,
                return_details=False,
                return_logits=True,
            )
            logits = _ensure_1d(logits)
            probs = _sigmoid_torch(logits)
            preds = (probs > 0.5).float()

            all_labels.extend(labels.detach().cpu().numpy().tolist())
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_probs.extend(probs.detach().cpu().numpy().tolist())

            if (batch_idx + 1) % 200 == 0:
                logger.info(f"[Test] {batch_idx + 1}/{len(test_loader)} batches done, samples={len(all_labels)}")

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    logger.info("\n" + "=" * 50)
    logger.info("Test Results:")
    logger.info(f"AUC: {metrics['auc']:.4f}")
    logger.info(f"ACC: {metrics['acc']:.4f}")
    logger.info(f"RMSE: {metrics['rmse']:.4f}")
    logger.info("=" * 50)

    results = {
        "metrics": metrics,
        "num_samples": len(all_labels),
        "model_epoch": int(checkpoint["epoch"]),
        "best_val_auc": float(checkpoint.get("val_auc", 0.0)),
    }

    result_path = os.path.join(args.save_dir, "test_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=4)

    append_summary_csv(
        args,
        metrics=metrics,
        best_val_auc=results["best_val_auc"],
        model_epoch=results["model_epoch"],
        final_model_facts=runtime_facts,
        logger=logger,
    )

    # diagnosis
    if getattr(args, "generate_diagnosis", False):
        logger.info("\nGenerating student diagnosis reports...")
        num_students_to_diagnose = min(5, info_dict["num_students"])
        diagnosis_results = []

        for stu_id in range(num_students_to_diagnose):
            diagnosis = model.get_student_diagnosis(stu_id)
            diagnosis_results.append(
                {
                    "student_id": int(stu_id),
                    "original_student_id": int(info_dict["stu_id_reverse_map"].get(stu_id, stu_id)),
                    "knowledge_mastery": [float(x) for x in diagnosis["knowledge_mastery"].cpu().numpy().tolist()],
                    "skill_latent": [float(x) for x in diagnosis["skill_latent"].cpu().numpy().tolist()],
                }
            )

        diagnosis_path = os.path.join(args.save_dir, "student_diagnosis.json")
        with open(diagnosis_path, "w") as f:
            json.dump(diagnosis_results, f, indent=4)

        logger.info(f"Student diagnosis reports saved to {diagnosis_path}")

    return metrics, results


def save_component_analysis_data(
    model: CognitiveDiagnosisModel,
    train_loader: DataLoader,
    device: torch.device,
    save_dir: str,
    logger,
    num_samples: int = 100,
) -> Dict[str, Any]:
    """
    保存组件可视化分析所需的数据：
    1. Prototype 向量和相似度矩阵
    2. 全局概念图 relation_matrices
    3. Gate Alpha 分布（个性化图混合系数）
    4. 个性化图采样
    """
    model.eval()
    analysis_data = {}
    
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        # ========== 1) Prototype 分析 ==========
        if model.prototype_module is not None:
            proto_vectors = model.prototype_module.prototypes.detach().cpu().numpy()
            analysis_data["prototype_vectors"] = proto_vectors
            
            # 计算原型间相似度
            P = F.normalize(model.prototype_module.prototypes, dim=-1, eps=1e-12)
            proto_similarity = (P @ P.t()).detach().cpu().numpy()
            analysis_data["prototype_similarity"] = proto_similarity
            
            # 收集原型分配分布
            proto_assigns = []
            sample_count = 0
            for batch in train_loader:
                if sample_count >= num_samples:
                    break
                student_ids, exercise_ids, concept_vector, _ = batch
                student_ids = student_ids.to(device)
                
                # 获取学生表示
                s_out = model.structure_module(student_ids, identity_relations=model.identity_relations)
                student_repr = s_out["student_repr"]
                
                _, assign = model.prototype_module(student_repr)
                proto_assigns.append(assign.detach().cpu().numpy())
                sample_count += len(student_ids)
            
            if proto_assigns:
                analysis_data["prototype_assign"] = np.concatenate(proto_assigns, axis=0)[:num_samples]
            
            logger.info(f"[Component Analysis] Prototype: {proto_vectors.shape}, similarity: {proto_similarity.shape}")
        
        # ========== 2) 全局概念图分析 ==========
        if model.structure_module.relation_learning is not None:
            relation_matrices, _ = model.structure_module.relation_learning()
            analysis_data["global_relation_matrices"] = relation_matrices.detach().cpu().numpy()
            logger.info(f"[Component Analysis] Global graph: {relation_matrices.shape}")
        
        # ========== 3) 个性化图分析 ==========
        if model.use_personal_graph and model.structure_module.personal_generator is not None:
            gate_alphas = []
            personal_graphs = []
            sample_count = 0
            
            for batch in train_loader:
                if sample_count >= num_samples:
                    break
                student_ids, _, _, _ = batch
                student_ids = student_ids.to(device)
                
                s_out = model.structure_module(student_ids, identity_relations=model.identity_relations)
                
                if s_out["alpha"] is not None:
                    gate_alphas.append(s_out["alpha"].squeeze().detach().cpu().numpy())
                if s_out["personal_matrices"] is not None:
                    personal_graphs.append(s_out["personal_matrices"].detach().cpu().numpy())
                
                sample_count += len(student_ids)
            
            if gate_alphas:
                analysis_data["gate_alpha"] = np.concatenate([g.flatten() for g in gate_alphas])[:num_samples]
                logger.info(f"[Component Analysis] Gate alpha samples: {len(analysis_data['gate_alpha'])}")
            
            if personal_graphs:
                # 只保存少量个性化图样本（节省空间）
                personal_arr = np.concatenate(personal_graphs, axis=0)[:min(10, num_samples)]
                analysis_data["personal_matrices_samples"] = personal_arr
                logger.info(f"[Component Analysis] Personal graph samples: {personal_arr.shape}")
    
    # ========== 保存数据 ==========
    analysis_path = os.path.join(save_dir, "component_analysis_data.npz")
    np.savez_compressed(analysis_path, **analysis_data)
    logger.info(f"[Component Analysis] Data saved to {analysis_path}")
    
    return analysis_data
