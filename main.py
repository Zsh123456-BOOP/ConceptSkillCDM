import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
import json
from datetime import datetime
import logging
import pandas as pd

from src.dataset import CognitiveDiagnosisDataset, create_dataloaders
from src.model import CognitiveDiagnosisModel
from gpu_utils import get_best_gpu

from src.experiment_utils import (
    setup_logging,
    compute_metrics,
    select_device,
    save_epoch_history_csv,
    append_summary_csv,
)

from src.config import DATASET_DEFAULTS


def apply_dataset_default(args):
    from copy import deepcopy
    dname = getattr(args, "dataset_name", None)
    if dname is None or dname not in DATASET_DEFAULTS:
        return args

    defaults = DATASET_DEFAULTS[dname]
    for k, v in defaults.items():
        # 只有当命令行没有显式覆盖时才用默认
        if getattr(args, k, None) == parser.get_default(k):
            setattr(args, k, v)
    return args

def setup_logging(log_dir):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def compute_metrics(labels, predictions, probs):
    """
    计算评估指标
    """
    labels = np.array(labels)
    predictions = np.array(predictions)
    probs = np.array(probs)

    auc = roc_auc_score(labels, probs)
    acc = accuracy_score(labels, predictions)
    rmse = np.sqrt(mean_squared_error(labels, probs))

    return {
        'auc': float(auc),
        'acc': float(acc),
        'rmse': float(rmse)
    }


def train_epoch(
        model,
        train_loader,
        optimizer,
        device,
        lambda_sparse,
        lambda_independence,
        lambda_proto_div,
        lambda_proto_usage,
        logger
):
    """
    训练一个epoch
    """
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

        # 前向传播：把 concept_vector 也传给模型
        pred_probs, details = model(
            student_ids,
            exercise_ids,
            concept_vector=concept_vector,
            return_details=True
        )

        # 计算BCE损失
        bce_loss = nn.BCELoss()(pred_probs, labels)

        # 计算正则化损失（包含 soft prototype 部分）
        proto_assign = details.get('prototype_assign', None)
        reg_loss = model.get_regularization_loss(
            details['relation_matrices'],
            details['skill_vector'],
            details['knowledge_state'],
            prototype_assign=proto_assign,
            lambda_sparse=lambda_sparse,
            lambda_independence=lambda_independence,
            lambda_proto_div=lambda_proto_div,
            lambda_proto_usage=lambda_proto_usage
        )

        # 总损失
        loss = bce_loss + reg_loss

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # 统计
        total_loss += loss.item()
        total_bce_loss += bce_loss.item()
        total_reg_loss += reg_loss.item()

        # 收集预测结果
        predictions = (pred_probs > 0.5).float()
        all_labels.extend(labels.cpu().detach().numpy())
        all_predictions.extend(predictions.cpu().detach().numpy())
        all_probs.extend(pred_probs.cpu().detach().numpy())

    avg_loss = total_loss / len(train_loader)
    avg_bce_loss = total_bce_loss / len(train_loader)
    avg_reg_loss = total_reg_loss / len(train_loader)

    metrics = compute_metrics(all_labels, all_predictions, all_probs)

    return {
        'loss': avg_loss,
        'bce_loss': avg_bce_loss,
        'reg_loss': avg_reg_loss,
        **metrics
    }


def validate(
        model,
        val_loader,
        device,
        lambda_sparse,
        lambda_independence,
        lambda_proto_div,
        lambda_proto_usage,
        logger
):
    """
    验证模型
    """
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

            # 前向传播
            pred_probs, details = model(
                student_ids,
                exercise_ids,
                concept_vector=concept_vector,
                return_details=True
            )

            # BCE
            bce_loss = nn.BCELoss()(pred_probs, labels)

            # 正则
            proto_assign = details.get('prototype_assign', None)
            reg_loss = model.get_regularization_loss(
                details['relation_matrices'],
                details['skill_vector'],
                details['knowledge_state'],
                prototype_assign=proto_assign,
                lambda_sparse=lambda_sparse,
                lambda_independence=lambda_independence,
                lambda_proto_div=lambda_proto_div,
                lambda_proto_usage=lambda_proto_usage
            )

            loss = bce_loss + reg_loss

            total_loss += loss.item()
            total_bce_loss += bce_loss.item()
            total_reg_loss += reg_loss.item()

            # 预测统计
            predictions = (pred_probs > 0.5).float()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probs.extend(pred_probs.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    avg_bce_loss = total_bce_loss / len(val_loader)
    avg_reg_loss = total_reg_loss / len(val_loader)

    metrics = compute_metrics(all_labels, all_predictions, all_probs)

    return {
        'loss': avg_loss,
        'bce_loss': avg_bce_loss,
        'reg_loss': avg_reg_loss,
        **metrics
    }

def select_device(args, logger):
    """
    统一选择 device：
    - 如果 no_cuda 为 True 或者没有 GPU，就用 CPU
    - 否则在候选 GPU 列表中，用 get_best_gpu 选显存最多的那块
    """
    if not torch.cuda.is_available() or args.no_cuda:
        device = torch.device('cpu')
        logger.info('Using device: cpu')
        return device

    # 解析候选 GPU 列表
    candidates = None
    if getattr(args, "gpu_candidates", None) is not None:
        try:
            candidates = [
                int(x.strip()) for x in str(args.gpu_candidates).split(',')
                if x.strip() != ''
            ]
            if len(candidates) == 0:
                candidates = None  # 退回：所有 GPU 都可选
        except Exception:
            candidates = None

    best_gpu = get_best_gpu(candidates=candidates, memory_threshold=2000)
    if best_gpu is None:
        best_gpu = 0  # 兜底

    device = torch.device(f'cuda:{best_gpu}')
    torch.cuda.set_device(best_gpu)
    logger.info(f'Using device: cuda:{best_gpu}')
    return device


def save_epoch_history_csv(history, save_dir, logger):
    """
    把每个 epoch 的 train / val 指标写进一个 CSV，方便画图/对比。
    """
    rows = []
    for epoch_idx, (tr, va) in enumerate(zip(history['train'], history['val']), start=1):
        row = {'epoch': epoch_idx}
        for k, v in tr.items():
            row[f'train_{k}'] = v
        for k, v in va.items():
            row[f'val_{k}'] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(save_dir, 'metrics_history.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f'Epoch-wise metrics history saved to {csv_path}')


def train(args, logger):
    """
    训练模型
    """
    device = select_device(args, logger)

    logger.info('Loading datasets...')

    data_dir = args.data_dir
    train_file = os.path.join(data_dir, 'train.csv')
    val_file = os.path.join(data_dir, 'valid.csv')
    test_file = os.path.join(data_dir, 'test.csv')

    # 创建数据加载器
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
        logger=logger
    )

    logger.info(f'Train samples: {info_dict["train_size"]}, Val samples: {info_dict["val_size"]}')
    logger.info(f'Number of students: {info_dict["num_students"]}')
    logger.info(f'Number of exercises: {info_dict["num_exercises"]}')
    logger.info(f'Number of concepts: {info_dict["num_concepts"]}')

    # 创建模型
    logger.info('Creating model...')

    use_soft_prototype = not args.disable_soft_prototype

    model = CognitiveDiagnosisModel(
        num_students=info_dict['num_students'],
        num_exercises=info_dict['num_exercises'],
        num_concepts=info_dict['num_concepts'],
        q_matrix=info_dict['q_matrix'],
        knowledge_dim=args.knowledge_dim,
        skill_dim=args.skill_dim,
        exercise_dim=args.exercise_dim,
        num_relation_heads=args.num_relation_heads,
        num_gnn_layers=args.num_gnn_layers,
        dropout=args.dropout,
        num_prototypes=args.num_prototypes,
        proto_tau=args.proto_tau,
        proto_lambda=args.proto_lambda,
        use_soft_prototype=use_soft_prototype
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Total parameters: {total_params:,}')
    logger.info(f'Trainable parameters: {trainable_params:,}')

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=args.patience,
    )

    best_val_auc = 0.0
    best_epoch = 0
    patience_counter = 0

    logger.info('Starting training...')

    history = {
        'train': [],
        'val': [],
        'best_epoch': 0,
        'best_val_auc': 0.0
    }

    for epoch in range(1, args.epochs + 1):
        logger.info(f'\n{"=" * 50}')
        logger.info(f'Epoch {epoch}/{args.epochs}')
        logger.info(f'{"=" * 50}')

        # 训练
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            args.lambda_sparse, args.lambda_independence,
            args.lambda_proto_div, args.lambda_proto_usage,
            logger
        )

        logger.info(f'\nTraining Results:')
        logger.info(
            f'Loss: {train_metrics["loss"]:.4f}, '
            f'BCE: {train_metrics["bce_loss"]:.4f}, '
            f'Reg: {train_metrics["reg_loss"]:.4f}'
        )
        logger.info(
            f'AUC: {train_metrics["auc"]:.4f}, '
            f'ACC: {train_metrics["acc"]:.4f}, '
            f'RMSE: {train_metrics["rmse"]:.4f}'
        )

        # 验证
        val_metrics = validate(
            model, val_loader, device,
            args.lambda_sparse, args.lambda_independence,
            args.lambda_proto_div, args.lambda_proto_usage,
            logger
        )

        logger.info(f'\nValidation Results:')
        logger.info(
            f'Loss: {val_metrics["loss"]:.4f}, '
            f'BCE: {val_metrics["bce_loss"]:.4f}, '
            f'Reg: {val_metrics["reg_loss"]:.4f}'
        )
        logger.info(
            f'AUC: {val_metrics["auc"]:.4f}, '
            f'ACC: {val_metrics["acc"]:.4f}, '
            f'RMSE: {val_metrics["rmse"]:.4f}'
        )

        # 更新学习率
        scheduler.step(val_metrics['loss'])

        history['train'].append(train_metrics)
        history['val'].append(val_metrics)

        # 保存最佳模型
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_epoch = epoch
            patience_counter = 0

            model_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': best_val_auc,
                'val_metrics': val_metrics,
                'args': vars(args),
                'info_dict': info_dict
            }, model_path)

            logger.info(f'\n*** New best model saved with AUC: {best_val_auc:.4f} ***')
        else:
            patience_counter += 1
            logger.info(f'\nNo improvement for {patience_counter} epochs')

        if patience_counter >= args.early_stop_patience:
            logger.info(f'\nEarly stopping triggered after {epoch} epochs')
            break

        if epoch % args.save_interval == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'train_metrics': train_metrics,
                'args': vars(args)
            }, checkpoint_path)
            logger.info(f'Checkpoint saved: {checkpoint_path}')

    history['best_epoch'] = best_epoch
    history['best_val_auc'] = best_val_auc

    history_path = os.path.join(args.save_dir, 'training_history.json')
    # 额外保存一份 CSV 形式的 epoch 曲线
    save_epoch_history_csv(history, args.save_dir, logger)

    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4, default=lambda x: float(x) if torch.is_tensor(x) else x)

    logger.info(f'\n{"=" * 50}')
    logger.info(f'Training completed!')
    logger.info(f'Best validation AUC: {best_val_auc:.4f} at epoch {best_epoch}')
    logger.info(f'{"=" * 50}')


def append_summary_csv(args, metrics, best_val_auc, model_epoch, logger):
    """
    把当前实验的关键配置 + Test 指标追加写入 results/all_results.csv
    """
    os.makedirs('results', exist_ok=True)
    csv_path = os.path.join('results', 'all_results.csv')

    dataset_name = os.path.basename(os.path.normpath(args.data_dir))

    row = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset': dataset_name,
        'seed': args.seed,
        'batch_size': args.batch_size,
        'knowledge_dim': args.knowledge_dim,
        'skill_dim': args.skill_dim,
        'exercise_dim': args.exercise_dim,
        'num_relation_heads': args.num_relation_heads,
        'num_gnn_layers': args.num_gnn_layers,
        'dropout': args.dropout,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'lambda_sparse': args.lambda_sparse,
        'lambda_independence': args.lambda_independence,
        'lambda_proto_div': args.lambda_proto_div,
        'lambda_proto_usage': args.lambda_proto_usage,
        'use_soft_prototype': not args.disable_soft_prototype,
        'num_prototypes': args.num_prototypes,
        'proto_tau': args.proto_tau,
        'proto_lambda': args.proto_lambda,
        'test_auc': metrics['auc'],
        'test_acc': metrics['acc'],
        'test_rmse': metrics['rmse'],
        'best_val_auc': best_val_auc,
        'model_epoch': model_epoch,
    }

    df = pd.DataFrame([row])
    file_exists = os.path.exists(csv_path)
    df.to_csv(csv_path, mode='a', header=not file_exists, index=False)

    logger.info(f'Experiment summary appended to {csv_path}')


def inference(args, logger):
    """
    测试模型
    """
    device = select_device(args, logger)

    data_dir = args.data_dir
    test_file = os.path.join(data_dir, 'test.csv')

    # 1. 先加载模型 checkpoint（里面有 info_dict 和 q_matrix）
    model_path = os.path.join(args.save_dir, 'best_model.pth')
    logger.info(f'Loading model from {model_path}...')

    if not os.path.exists(model_path):
        logger.error(f'Model file not found: {model_path}')
        return

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    loaded_args = checkpoint.get('args', {})
    info_dict = checkpoint.get('info_dict', None)

    if info_dict is None:
        logger.error('info_dict not found in checkpoint. Please retrain the model with the current code.')
        return

    stu_id_map = info_dict['stu_id_map']
    exer_id_map = info_dict['exer_id_map']
    cpt_id_map = info_dict['cpt_id_map']
    q_matrix = info_dict['q_matrix']

    # 2. 构建测试集（使用保存的 ID 映射，避免 KeyError）
    logger.info('Building test dataloader with saved ID mappings...')

    raw_test_df = pd.read_csv(test_file)

    valid_stu_ids = set(stu_id_map.keys())
    valid_exer_ids = set(exer_id_map.keys())

    before_rows = len(raw_test_df)
    filtered_test_df = raw_test_df[
        raw_test_df['stu_id'].isin(valid_stu_ids)
        & raw_test_df['exer_id'].isin(valid_exer_ids)
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
        cpt_id_map=cpt_id_map
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    test_size = len(test_dataset)
    logger.info(f'Test samples: {test_size}')

    # 3. 创建模型（从 checkpoint 的 args 恢复配置）
    num_prototypes = loaded_args.get('num_prototypes', args.num_prototypes)
    proto_tau = loaded_args.get('proto_tau', args.proto_tau)
    proto_lambda = loaded_args.get('proto_lambda', args.proto_lambda)
    disable_soft_prototype = loaded_args.get('disable_soft_prototype', args.disable_soft_prototype)
    use_soft_prototype = not disable_soft_prototype

    model = CognitiveDiagnosisModel(
        num_students=info_dict['num_students'],
        num_exercises=info_dict['num_exercises'],
        num_concepts=info_dict['num_concepts'],
        q_matrix=q_matrix,
        knowledge_dim=loaded_args.get('knowledge_dim', args.knowledge_dim),
        skill_dim=loaded_args.get('skill_dim', args.skill_dim),
        exercise_dim=loaded_args.get('exercise_dim', args.exercise_dim),
        num_relation_heads=loaded_args.get('num_relation_heads', args.num_relation_heads),
        num_gnn_layers=loaded_args.get('num_gnn_layers', args.num_gnn_layers),
        dropout=loaded_args.get('dropout', args.dropout),
        num_prototypes=num_prototypes,
        proto_tau=proto_tau,
        proto_lambda=proto_lambda,
        use_soft_prototype=use_soft_prototype
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f'Model loaded from epoch {checkpoint["epoch"]}')

    # 4. 测试
    logger.info('Starting testing...')
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
                return_details=False
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

    # 5. 计算指标
    metrics = compute_metrics(all_labels, all_predictions, all_probs)

    logger.info(f'\n{"=" * 50}')
    logger.info('Test Results:')
    logger.info(f'AUC: {metrics["auc"]:.4f}')
    logger.info(f'ACC: {metrics["acc"]:.4f}')
    logger.info(f'RMSE: {metrics["rmse"]:.4f}')
    logger.info(f'{"=" * 50}')

    results = {
        'metrics': metrics,
        'num_samples': len(all_labels),
        'model_epoch': int(checkpoint['epoch']),
        'best_val_auc': float(checkpoint.get('val_auc', 0.0))
    }

    result_path = os.path.join(args.save_dir, 'test_results.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=4)

    logger.info(f'Test results saved to {result_path}')

    # 追加写入全局 CSV 汇总
    append_summary_csv(
        args,
        metrics=metrics,
        best_val_auc=results['best_val_auc'],
        model_epoch=results['model_epoch'],
        logger=logger
    )

    
    # 6. 学生诊断
    if args.generate_diagnosis:
        logger.info('\nGenerating student diagnosis reports...')

        num_students_to_diagnose = min(5, info_dict['num_students'])
        diagnosis_results = []

        for stu_id in range(num_students_to_diagnose):
            diagnosis = model.get_student_diagnosis(stu_id)
            diagnosis_results.append({
                'student_id': int(stu_id),
                'original_student_id': int(info_dict['stu_id_reverse_map'].get(stu_id, stu_id)),
                'knowledge_mastery': [float(x) for x in diagnosis['knowledge_mastery'].cpu().numpy().tolist()],
                'skill_level': [float(x) for x in diagnosis['skill_level'].cpu().numpy().tolist()]
            })

        diagnosis_path = os.path.join(args.save_dir, 'student_diagnosis.json')
        with open(diagnosis_path, 'w') as f:
            json.dump(diagnosis_results, f, indent=4, default=str)

        logger.info(f'Student diagnosis reports saved to {diagnosis_path}')


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Cognitive Diagnosis Model Training and Testing')

    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./data/assist_09')

    # 模型参数
    parser.add_argument('--knowledge_dim', type=int, default=128)
    parser.add_argument('--skill_dim', type=int, default=2)
    parser.add_argument('--exercise_dim', type=int, default=128)
    parser.add_argument('--num_relation_heads', type=int, default=4)
    parser.add_argument('--num_gnn_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)

    # === soft prototype 相关参数（新增） ===
    parser.add_argument('--num_prototypes', type=int, default=3,
                        help='Number of soft prototypes for students')
    parser.add_argument('--proto_tau', type=float, default=1.0,
                        help='Temperature for soft prototype assignment')
    parser.add_argument('--proto_lambda', type=float, default=0.5,
                        help='Residual weight for prototype correction on knowledge state')
    parser.add_argument('--disable_soft_prototype', action='store_true',
                        help='Disable soft prototype module')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lambda_sparse', type=float, default=0.1)
    parser.add_argument('--lambda_independence', type=float, default=0.1)

    # soft prototype 正则权重（新增）
    parser.add_argument('--lambda_proto_div', type=float, default=0.01,
                        help='Weight for prototype diversity regularization')
    parser.add_argument('--lambda_proto_usage', type=float, default=0.01,
                        help='Weight for prototype usage balance regularization')

    # 早停和调度器参数
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--early_stop_patience', type=int, default=5)

    # 其他参数
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--generate_diagnosis', default=True)

    # ✅ 新增：候选 GPU 列表，比如 "0" 或 "0,1"
    parser.add_argument(
        '--gpu_candidates',
        type=str,
        default='0',
        help='Comma-separated GPU ids to choose from, e.g. "0" or "0,1"'
    )

    # 保存参数
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--save_interval', type=int, default=10)

    # 数据清洗参数
    parser.add_argument('--min_stu_interactions', type=int, default=15,
                        help='Minimum interactions for students to keep (0 = disable)')
    parser.add_argument('--min_exer_interactions', type=int, default=0,
                        help='Minimum interactions for exercises to keep (0 = disable)')
    parser.add_argument('--min_poison_count', type=int, default=0,
                        help='Minimum count for detecting toxic items with acc=0 or 1 (0 = disable)')
    
    args = parser.parse_args()

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

    args_file = os.path.join(args.save_dir, 'args.json')
    with open(args_file, 'w') as f:
        json.dump(vars(args), f, indent=4)

    logger.info('Arguments:')
    for arg, value in sorted(vars(args).items()):
        logger.info(f'  {arg}: {value}')

    train(args, logger)
    inference(args, logger)


if __name__ == '__main__':
    main()
