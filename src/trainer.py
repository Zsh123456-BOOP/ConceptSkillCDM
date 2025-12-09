import json
import os
from typing import Tuple, Dict, Any

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset import CognitiveDiagnosisDataset, create_dataloaders
from src.model import CognitiveDiagnosisModel
from src.experiment_utils import (
    compute_metrics,
    select_device,
    save_epoch_history_csv,
    append_summary_csv,
)


def train_epoch(
    model: CognitiveDiagnosisModel,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    lambda_sparse: float,
    lambda_proto_div: float,
    lambda_proto_usage: float,
    lambda_sparse_personal: float,
    lambda_alpha: float,
    logger,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_bce_loss = 0.0
    total_reg_loss = 0.0

    all_labels = []
    all_predictions = []
    all_probs = []

    for batch_idx, batch in enumerate(train_loader):
        student_ids, exercise_ids, concept_vector, labels = batch
        student_ids = student_ids.to(device)
        exercise_ids = exercise_ids.to(device)
        concept_vector = concept_vector.to(device)
        labels = labels.to(device).float()

        pred_probs, details = model(
            student_ids,
            exercise_ids,
            concept_vector=concept_vector,
            return_details=True,
        )

        bce_loss = nn.BCELoss()(pred_probs, labels)

        proto_assign = details.get("prototype_assign", None)
        personal_matrices = details.get("personal_matrices", None)
        gate_alpha = details.get("alpha", None)
        reg_loss = model.get_regularization_loss(
            details["relation_matrices"],
            details["skill_vector"],
            details["knowledge_state"],
            prototype_assign=proto_assign,
            lambda_sparse=lambda_sparse,
            lambda_proto_div=lambda_proto_div,
            lambda_proto_usage=lambda_proto_usage,
            personal_matrices=personal_matrices,
            alpha=gate_alpha,
            lambda_sparse_personal=lambda_sparse_personal,
            lambda_alpha=lambda_alpha,
        )

        loss = bce_loss + reg_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        total_bce_loss += bce_loss.item()
        total_reg_loss += reg_loss.item()

        predictions = (pred_probs > 0.5).float()
        all_labels.extend(labels.cpu().detach().numpy())
        all_predictions.extend(predictions.cpu().detach().numpy())
        all_probs.extend(pred_probs.cpu().detach().numpy())

    avg_loss = total_loss / len(train_loader)
    avg_bce_loss = total_bce_loss / len(train_loader)
    avg_reg_loss = total_reg_loss / len(train_loader)

    metrics = compute_metrics(all_labels, all_predictions, all_probs)

    return {
        "loss": avg_loss,
        "bce_loss": avg_bce_loss,
        "reg_loss": avg_reg_loss,
        **metrics,
    }


def validate(
    model: CognitiveDiagnosisModel,
    val_loader: DataLoader,
    device: torch.device,
    lambda_sparse: float,
    lambda_proto_div: float,
    lambda_proto_usage: float,
    lambda_sparse_personal: float,
    lambda_alpha: float,
    logger,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_bce_loss = 0.0
    total_reg_loss = 0.0

    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            student_ids, exercise_ids, concept_vector, labels = batch
            student_ids = student_ids.to(device)
            exercise_ids = exercise_ids.to(device)
            concept_vector = concept_vector.to(device)
            labels = labels.to(device).float()

            pred_probs, details = model(
                student_ids,
                exercise_ids,
                concept_vector=concept_vector,
                return_details=True,
            )

            bce_loss = nn.BCELoss()(pred_probs, labels)

            proto_assign = details.get("prototype_assign", None)
            personal_matrices = details.get("personal_matrices", None)
            gate_alpha = details.get("alpha", None)
            reg_loss = model.get_regularization_loss(
                details["relation_matrices"],
                details["skill_vector"],
                details["knowledge_state"],
                prototype_assign=proto_assign,
                lambda_sparse=lambda_sparse,
                lambda_proto_div=lambda_proto_div,
                lambda_proto_usage=lambda_proto_usage,
                personal_matrices=personal_matrices,
                alpha=gate_alpha,
                lambda_sparse_personal=lambda_sparse_personal,
                lambda_alpha=lambda_alpha,
            )

            loss = bce_loss + reg_loss

            total_loss += loss.item()
            total_bce_loss += bce_loss.item()
            total_reg_loss += reg_loss.item()

            predictions = (pred_probs > 0.5).float()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probs.extend(pred_probs.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    avg_bce_loss = total_bce_loss / len(val_loader)
    avg_reg_loss = total_reg_loss / len(val_loader)

    metrics = compute_metrics(all_labels, all_predictions, all_probs)

    return {
        "loss": avg_loss,
        "bce_loss": avg_bce_loss,
        "reg_loss": avg_reg_loss,
        **metrics,
    }

def train_one_experiment(args, logger) -> Tuple[float, int]:
    device = select_device(args, logger)

    # ========= 统一前缀：把这一轮实验的身份写清楚 =========
    run_tag = (
        f"[{getattr(args, 'dataset_name', 'unknown')}"
        f"|{getattr(args, 'model_variant', 'full')}"
        f"|lr={args.learning_rate:g}"
        f"|drop={args.dropout:.2f}]"
    )

    logger.info("%s Loading datasets...", run_tag)
    logger.info(
        "%s [Ablation] model_variant=%s | soft_proto=%s, skill=%s, exercise_graph=%s",
        run_tag,
        getattr(args, "model_variant", "full"),
        str(getattr(args, "use_soft_prototype", True)),
        str(getattr(args, "use_skill_encoder", True)),
        str(getattr(args, "use_exercise_graph", True)),
    )
    logger.info(
        "%s Regularization: sparse=%.4f, proto_div=%.4f, proto_usage=%.4f, "
        "personal_sparse=%.4f, alpha_penalty=%.4f",
        run_tag,
        args.lambda_sparse,
        args.lambda_proto_div,
        args.lambda_proto_usage,
        args.lambda_sparse_personal,
        args.lambda_alpha,
    )

    logger.info(
        "[Config] dataset=%s | lambda_sparse=%.4f",
        getattr(args, "dataset_name", "N/A"),
        args.lambda_sparse,
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
    model = CognitiveDiagnosisModel(
        num_students=info_dict["num_students"],
        num_exercises=info_dict["num_exercises"],
        num_concepts=info_dict["num_concepts"],
        q_matrix=info_dict["q_matrix"],
        knowledge_dim=args.knowledge_dim,
        skill_dim=args.skill_dim,
        exercise_dim=args.exercise_dim,
        num_relation_heads=args.num_relation_heads,
        num_gnn_layers=args.num_gnn_layers,
        dropout=args.dropout,
        num_prototypes=args.num_prototypes,
        proto_tau=args.proto_tau,
        proto_lambda=args.proto_lambda,
        use_soft_prototype=getattr(args, "use_soft_prototype", True),
        use_skill_encoder=getattr(args, "use_skill_encoder", True),
        use_exercise_graph=getattr(args, "use_exercise_graph", True),
        use_personal_graph=getattr(args, "use_personal_graph", False),
        lambda_sparse_personal=args.lambda_sparse_personal,
        lambda_alpha=args.lambda_alpha,
        # ✅ 新增：从 args 传入，而不是 model 里写死
        exercise_l2_lambda=getattr(args, "exercise_l2_lambda", 5e-5),
    ).to(device)

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

    logger.info("%s Starting training...", run_tag)

    history: Dict[str, Any] = {
        "train": [],
        "val": [],
        "best_epoch": 0,
        "best_val_auc": 0.0,
    }

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.lambda_sparse,
            args.lambda_proto_div,
            args.lambda_proto_usage,
            args.lambda_sparse_personal,
            args.lambda_alpha,
            logger,
        )

        val_metrics = validate(
            model,
            val_loader,
            device,
            args.lambda_sparse,
            args.lambda_proto_div,
            args.lambda_proto_usage,
            args.lambda_sparse_personal,
            args.lambda_alpha,
            logger,
        )

        summary_line = (
            f"{run_tag} Epoch [{epoch:03d}/{args.epochs}] | "
            f"Train: Loss={train_metrics['loss']:.4f}, BCE={train_metrics['bce_loss']:.4f}, "
            f"Reg={train_metrics['reg_loss']:.4f}, AUC={train_metrics['auc']:.4f}, "
            f"ACC={train_metrics['acc']:.4f}, RMSE={train_metrics['rmse']:.4f} | "
            f"Val: Loss={val_metrics['loss']:.4f}, BCE={val_metrics['bce_loss']:.4f}, "
            f"Reg={val_metrics['reg_loss']:.4f}, AUC={val_metrics['auc']:.4f}, "
            f"ACC={val_metrics['acc']:.4f}, RMSE={val_metrics['rmse']:.4f}"
        )
        logger.info(summary_line)

        scheduler.step(val_metrics["loss"])

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_epoch = epoch
            patience_counter = 0

            model_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc": best_val_auc,
                    "val_metrics": val_metrics,
                    "args": vars(args),
                    "info_dict": info_dict,
                },
                model_path,
            )

            logger.info(
                "%s   -> New best AUC=%.4f at epoch %d (patience reset)",
                run_tag,
                best_val_auc,
                epoch,
            )
        else:
            patience_counter += 1
            logger.info(
                "%s   -> No improvement for %d epoch(s) (best AUC=%.4f @ epoch %d)",
                run_tag,
                patience_counter,
                best_val_auc,
                best_epoch,
            )

        if patience_counter >= args.early_stop_patience:
            logger.info(
                "%s Early stopping triggered at epoch %d (best AUC=%.4f @ epoch %d)",
                run_tag,
                epoch,
                best_val_auc,
                best_epoch,
            )
            break

        if epoch % args.save_interval == 0:
            checkpoint_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "train_metrics": train_metrics,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            logger.info("%s Checkpoint saved: %s", run_tag, checkpoint_path)

    history["best_epoch"] = best_epoch
    history["best_val_auc"] = best_val_auc

    history_path = os.path.join(args.save_dir, "training_history.json")
    save_epoch_history_csv(history, args.save_dir, logger)

    with open(history_path, "w") as f:
        json.dump(history, f, indent=4, default=lambda x: float(x) if torch.is_tensor(x) else x)

    logger.info("%s %s", run_tag, "=" * 50)
    logger.info("%s Training completed!", run_tag)
    logger.info("%s Best validation AUC: %.4f at epoch %d", run_tag, best_val_auc, best_epoch)
    logger.info("%s %s", run_tag, "=" * 50)

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
    loaded_args = checkpoint.get("args", {})
    info_dict = checkpoint.get("info_dict", None)

    if info_dict is None:
        logger.error("info_dict not found in checkpoint. Please retrain the model with the current code.")
        return {}, {}

    stu_id_map = info_dict["stu_id_map"]
    exer_id_map = info_dict["exer_id_map"]
    cpt_id_map = info_dict["cpt_id_map"]
    q_matrix = info_dict["q_matrix"]

    logger.info("Building test dataloader with saved ID mappings...")

    raw_test_df = pd.read_csv(test_file)

    valid_stu_ids = set(stu_id_map.keys())
    valid_exer_ids = set(exer_id_map.keys())

    before_rows = len(raw_test_df)
    filtered_test_df = raw_test_df[
        raw_test_df["stu_id"].isin(valid_stu_ids) & raw_test_df["exer_id"].isin(valid_exer_ids)
    ].reset_index(drop=True)
    after_rows = len(filtered_test_df)

    dropped_rows = before_rows - after_rows
    if dropped_rows > 0:
        logger.info(
            f"[推理阶段数据过滤] 原始测试集共有 {before_rows} 条记录；"
            f"由于学生/题目在训练阶段被清洗或未出现，本次测试集实际保留 {after_rows} 条记录，"
            f"共丢弃 {dropped_rows} 条。"
        )
    else:
        logger.info("[推理阶段数据过滤] 测试集中所有学生与题目均在训练阶段出现，无需额外过滤。")

    test_dataset = CognitiveDiagnosisDataset(
        csv_file=filtered_test_df,
        stu_id_map=stu_id_map,
        exer_id_map=exer_id_map,
        cpt_id_map=cpt_id_map,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    test_size = len(test_dataset)
    logger.info(f"Test samples: {test_size}")

    num_prototypes = loaded_args.get("num_prototypes", args.num_prototypes)
    proto_tau = loaded_args.get("proto_tau", args.proto_tau)
    proto_lambda = loaded_args.get("proto_lambda", args.proto_lambda)

    # 兼容旧 checkpoint：缺省时默认启用
    loaded_use_soft = loaded_args.get(
        "use_soft_prototype",
        not loaded_args.get("disable_soft_prototype", False),
    )
    loaded_use_skill = loaded_args.get("use_skill_encoder", True)
    loaded_use_ex_graph = loaded_args.get("use_exercise_graph", True)
    use_soft_prototype = loaded_use_soft and getattr(args, "use_soft_prototype", True)
    use_skill_encoder = loaded_use_skill and getattr(args, "use_skill_encoder", True)
    use_exercise_graph = loaded_use_ex_graph and getattr(args, "use_exercise_graph", True)
    use_personal_graph = loaded_args.get("use_personal_graph", getattr(args, "use_personal_graph", False))

    lambda_sparse_personal = loaded_args.get("lambda_sparse_personal", args.lambda_sparse_personal)
    lambda_alpha = loaded_args.get("lambda_alpha", args.lambda_alpha)

    args.use_soft_prototype = use_soft_prototype
    args.use_skill_encoder = use_skill_encoder
    args.use_exercise_graph = use_exercise_graph
    args.use_personal_graph = use_personal_graph

    model = CognitiveDiagnosisModel(
        num_students=info_dict["num_students"],
        num_exercises=info_dict["num_exercises"],
        num_concepts=info_dict["num_concepts"],
        q_matrix=q_matrix,
        knowledge_dim=loaded_args.get("knowledge_dim", args.knowledge_dim),
        skill_dim=loaded_args.get("skill_dim", args.skill_dim),
        exercise_dim=loaded_args.get("exercise_dim", args.exercise_dim),
        num_relation_heads=loaded_args.get("num_relation_heads", args.num_relation_heads),
        num_gnn_layers=loaded_args.get("num_gnn_layers", args.num_gnn_layers),
        dropout=loaded_args.get("dropout", args.dropout),
        num_prototypes=num_prototypes,
        proto_tau=proto_tau,
        proto_lambda=proto_lambda,
        use_soft_prototype=use_soft_prototype,
        use_skill_encoder=use_skill_encoder,
        use_exercise_graph=use_exercise_graph,
        use_personal_graph=use_personal_graph,
        lambda_sparse_personal=lambda_sparse_personal,
        lambda_alpha=lambda_alpha,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f'Model loaded from epoch {checkpoint["epoch"]}')

    logger.info("Starting testing...")
    model.eval()

    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            student_ids, exercise_ids, concept_vector, labels = batch
            student_ids = student_ids.to(device)
            exercise_ids = exercise_ids.to(device)
            concept_vector = concept_vector.to(device)
            labels = labels.to(device).float()

            pred_probs = model(
                student_ids,
                exercise_ids,
                concept_vector=concept_vector,
                return_details=False,
            )

            predictions = (pred_probs > 0.5).float()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probs.extend(pred_probs.cpu().numpy())

            if (batch_idx + 1) % 200 == 0:
                logger.info(
                    f"[Test] 已完成 {batch_idx + 1} / {len(test_loader)} 个 batch，"
                    f"累计样本数 {len(all_labels)}/{test_size}"
                )

    metrics = compute_metrics(all_labels, all_predictions, all_probs)

    logger.info(f'\n{"=" * 50}')
    logger.info("Test Results:")
    logger.info(f'AUC: {metrics["auc"]:.4f}')
    logger.info(f'ACC: {metrics["acc"]:.4f}')
    logger.info(f'RMSE: {metrics["rmse"]:.4f}')
    logger.info(f'{"=" * 50}')

    results = {
        "metrics": metrics,
        "num_samples": len(all_labels),
        "model_epoch": int(checkpoint["epoch"]),
        "best_val_auc": float(checkpoint.get("val_auc", 0.0)),
    }

    result_path = os.path.join(args.save_dir, "test_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"Test results saved to {result_path}")

    append_summary_csv(
        args,
        metrics=metrics,
        best_val_auc=results["best_val_auc"],
        model_epoch=results["model_epoch"],
        logger=logger,
    )

    if args.generate_diagnosis:
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
                    "skill_level": [float(x) for x in diagnosis["skill_level"].cpu().numpy().tolist()],
                }
            )

        diagnosis_path = os.path.join(args.save_dir, "student_diagnosis.json")
        with open(diagnosis_path, "w") as f:
            json.dump(diagnosis_results, f, indent=4, default=str)

        logger.info(f"Student diagnosis reports saved to {diagnosis_path}")

    return metrics, results
