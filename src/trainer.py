# src/trainer.py
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

# ======================================================
# Helpers
# ======================================================

def _sigmoid_np(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable sigmoid for metrics; keep in torch for speed."""
    return torch.sigmoid(x)


def _ensure_1d(t: torch.Tensor) -> torch.Tensor:
    """Ensure tensor shape (B,) for logits/labels."""
    return t.view(-1)


def _convert_legacy_weight_norm_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Compatibility helper:
    If you ever loaded checkpoints created with deprecated torch.nn.utils.weight_norm,
    keys are like:
      ...theta_proj.weight_g, ...theta_proj.weight_v
    New parametrizations.weight_norm uses:
      ...theta_proj.parametrizations.weight.original0 (g)
      ...theta_proj.parametrizations.weight.original1 (v)

    This converter allows strict loading for old checkpoints.
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

    all_labels = []
    all_preds = []
    all_probs = []

    bce_fn = nn.BCEWithLogitsLoss()

    for batch_idx, batch in enumerate(train_loader):
        student_ids, exercise_ids, concept_vector, labels = batch
        student_ids = student_ids.to(device)
        exercise_ids = exercise_ids.to(device)
        concept_vector = concept_vector.to(device)
        labels = labels.to(device).float()
        labels = _ensure_1d(labels)

        # ✅ get logits + details (for regularizers)
        logits, details = model(
            student_ids,
            exercise_ids,
            concept_vector=concept_vector,
            return_details=True,
            return_logits=True,
        )
        logits = _ensure_1d(logits)

        bce_loss = bce_fn(logits, labels)

        reg_loss = model.get_regularization_loss(
            relation_matrices=details["relation_matrices"],
            details=details,
            lambda_proto_div=lambda_proto_div,
            lambda_proto_usage=lambda_proto_usage,
        )

        loss = bce_loss + reg_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += float(loss.item())
        total_bce += float(bce_loss.item())
        total_reg += float(reg_loss.item())

        with torch.no_grad():
            probs = _sigmoid_np(logits)
            preds = (probs > 0.5).float()

        all_labels.extend(labels.detach().cpu().numpy())
        all_preds.extend(preds.detach().cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

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

    all_labels = []
    all_preds = []
    all_probs = []

    bce_fn = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            student_ids, exercise_ids, concept_vector, labels = batch
            student_ids = student_ids.to(device)
            exercise_ids = exercise_ids.to(device)
            concept_vector = concept_vector.to(device)
            labels = labels.to(device).float()
            labels = _ensure_1d(labels)

            logits, details = model(
                student_ids,
                exercise_ids,
                concept_vector=concept_vector,
                return_details=True,
                return_logits=True,
            )
            logits = _ensure_1d(logits)

            bce_loss = bce_fn(logits, labels)
            reg_loss = model.get_regularization_loss(
                relation_matrices=details["relation_matrices"],
                details=details,
                lambda_proto_div=lambda_proto_div,
                lambda_proto_usage=lambda_proto_usage,
            )
            loss = bce_loss + reg_loss

            total_loss += float(loss.item())
            total_bce += float(bce_loss.item())
            total_reg += float(reg_loss.item())

            probs = _sigmoid_np(logits)
            preds = (probs > 0.5).float()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

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

    use_soft_prototype = getattr(args, "use_soft_prototype", True)
    use_mf_branch = getattr(args, "use_mf_branch", getattr(args, "use_skill_encoder", True))
    use_concept_graph = getattr(args, "use_concept_graph", True)
    logger.info(
        "%s Ablation switches: use_soft_prototype=%s, use_mf_branch=%s, use_concept_graph=%s",
        run_tag,
        use_soft_prototype,
        use_mf_branch,
        use_concept_graph,
    )

    # ✅ Mapping old args semantics to new model:
    # - args.lambda_sparse -> model.lambda_graph_entropy
    # - args.exercise_l2_lambda -> model.mf_l2_lambda
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
        use_mf_branch=use_mf_branch,
        use_concept_graph=use_concept_graph,
        graph_topk=getattr(args, "graph_topk", None),
        allow_self_loop=not getattr(args, "disable_self_loop", False),
        num_prototypes=args.num_prototypes,
        proto_tau=args.proto_tau,
        proto_lambda=args.proto_lambda,
        use_soft_prototype=use_soft_prototype,
        use_personal_graph=getattr(args, "use_personal_graph", False),
        personal_rank=getattr(args, "personal_rank", 4),
        lambda_sparse_personal=args.lambda_sparse_personal,
        lambda_alpha=args.lambda_alpha,
        lambda_graph_entropy=args.lambda_sparse,
        mf_l2_lambda=getattr(args, "exercise_l2_lambda", 5e-5),
        gnn_residual_weight=getattr(args, "gnn_residual_weight", 0.5),
        use_q_conditioning=not getattr(args, "disable_q_conditioning", False),
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
            logger.info("%s -> New best AUC=%.4f at epoch %d", run_tag, best_val_auc, epoch)
        else:
            patience_counter += 1
            logger.info("%s -> No improvement %d epoch(s) (best AUC=%.4f @ %d)",
                        run_tag, patience_counter, best_val_auc, best_epoch)

        if patience_counter >= args.early_stop_patience:
            logger.info("%s Early stopping at epoch %d (best AUC=%.4f @ %d)",
                        run_tag, epoch, best_val_auc, best_epoch)
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
        json.dump(history, f, indent=4)

    logger.info("%s Training completed! Best Val AUC=%.4f @ epoch %d", run_tag, best_val_auc, best_epoch)
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Build model from loaded args (fallback to current args)
    use_soft_prototype = loaded_args.get("use_soft_prototype", getattr(args, "use_soft_prototype", True))
    use_mf_branch = loaded_args.get(
        "use_mf_branch",
        loaded_args.get("use_skill_encoder", getattr(args, "use_mf_branch", getattr(args, "use_skill_encoder", True))),
    )
    use_concept_graph = loaded_args.get("use_concept_graph", getattr(args, "use_concept_graph", True))
    logger.info(
        "Inference switches: use_soft_prototype=%s, use_mf_branch=%s, use_concept_graph=%s",
        use_soft_prototype,
        use_mf_branch,
        use_concept_graph,
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
        num_gnn_layers=loaded_args.get("num_gnn_layers", args.num_gnn_layers),
        dropout=loaded_args.get("dropout", args.dropout),
        use_mf_branch=use_mf_branch,
        use_concept_graph=use_concept_graph,
        graph_topk=loaded_args.get("graph_topk", getattr(args, "graph_topk", None)),
        allow_self_loop=not loaded_args.get("disable_self_loop", getattr(args, "disable_self_loop", False)),
        num_prototypes=loaded_args.get("num_prototypes", args.num_prototypes),
        proto_tau=loaded_args.get("proto_tau", args.proto_tau),
        proto_lambda=loaded_args.get("proto_lambda", args.proto_lambda),
        use_soft_prototype=use_soft_prototype,
        use_personal_graph=loaded_args.get("use_personal_graph", getattr(args, "use_personal_graph", False)),
        personal_rank=loaded_args.get("personal_rank", getattr(args, "personal_rank", 4)),
        lambda_sparse_personal=loaded_args.get("lambda_sparse_personal", args.lambda_sparse_personal),
        lambda_alpha=loaded_args.get("lambda_alpha", args.lambda_alpha),
        lambda_graph_entropy=loaded_args.get("lambda_sparse", args.lambda_sparse),
        mf_l2_lambda=loaded_args.get("exercise_l2_lambda", getattr(args, "exercise_l2_lambda", 5e-5)),
        gnn_residual_weight=loaded_args.get("gnn_residual_weight", getattr(args, "gnn_residual_weight", 0.5)),
        use_q_conditioning=not loaded_args.get("disable_q_conditioning", getattr(args, "disable_q_conditioning", False)),
    ).to(device)

    # ✅ compatibility for legacy weight_norm checkpoints
    state_dict = checkpoint["model_state_dict"]
    state_dict = _convert_legacy_weight_norm_keys(state_dict)
    model.load_state_dict(state_dict, strict=False)

    logger.info(f"Model loaded from epoch {checkpoint['epoch']}. Start testing...")

    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            student_ids, exercise_ids, concept_vector, labels = batch
            student_ids = student_ids.to(device)
            exercise_ids = exercise_ids.to(device)
            concept_vector = concept_vector.to(device)
            labels = labels.to(device).float()
            labels = _ensure_1d(labels)

            logits = model(
                student_ids,
                exercise_ids,
                concept_vector=concept_vector,
                return_details=False,
                return_logits=True,
            )
            logits = _ensure_1d(logits)
            probs = _sigmoid_np(logits)
            preds = (probs > 0.5).float()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

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
