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
from tqdm import tqdm
import pandas as pd

from dataset import CognitiveDiagnosisDataset, create_dataloaders, build_id_mappings, build_q_matrix
from model import CognitiveDiagnosisModel


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

    Args:
        labels: 真实标签 (numpy array)
        predictions: 预测标签 (numpy array)
        probs: 预测概率 (numpy array)

    Returns:
        metrics: 包含各项指标的字典
    """
    # 确保是numpy数组
    labels = np.array(labels)
    predictions = np.array(predictions)
    probs = np.array(probs)

    # AUC
    auc = roc_auc_score(labels, probs)

    # ACC
    acc = accuracy_score(labels, predictions)

    # RMSE
    rmse = np.sqrt(mean_squared_error(labels, probs))

    metrics = {
        'auc': float(auc),
        'acc': float(acc),
        'rmse': float(rmse)
    }

    return metrics


def train_epoch(model, train_loader, optimizer, device, lambda_sparse, lambda_independence, logger):
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

    # 原来是：pbar = tqdm(train_loader, desc="Training", leave=False)
    # for batch_idx, batch in enumerate(pbar):
    for batch_idx, batch in enumerate(train_loader):
        student_ids, exercise_ids, _, labels = batch
        student_ids = student_ids.to(device)
        exercise_ids = exercise_ids.to(device)
        labels = labels.to(device).float()

        # 前向传播
        pred_probs, details = model(student_ids, exercise_ids, return_details=True)

        # 计算BCE损失
        bce_loss = nn.BCELoss()(pred_probs, labels)

        # 计算正则化损失
        reg_loss = model.get_regularization_loss(
            details['relation_matrices'],
            details['skill_vector'],
            details['knowledge_state'],
            lambda_sparse=lambda_sparse,
            lambda_independence=lambda_independence
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

        # 如果你想，可以加一个很轻量的 batch 级别 log（可选）
        # if (batch_idx + 1) % 200 == 0:
        #     logger.info(f"  [Train] Batch {batch_idx+1}/{len(train_loader)} "
        #                 f"Loss={loss.item():.4f}, BCE={bce_loss.item():.4f}, Reg={reg_loss.item():.4f}")

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

def validate(model, val_loader, device, lambda_sparse, lambda_independence, logger):
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
            student_ids, exercise_ids, _, labels = batch
            student_ids = student_ids.to(device)
            exercise_ids = exercise_ids.to(device)
            labels = labels.to(device).float()

            # 前向传播
            pred_probs, details = model(student_ids, exercise_ids, return_details=True)

            # BCE
            bce_loss = nn.BCELoss()(pred_probs, labels)

            # 正则
            reg_loss = model.get_regularization_loss(
                details['relation_matrices'],
                details['skill_vector'],
                details['knowledge_state'],
                lambda_sparse=lambda_sparse,
                lambda_independence=lambda_independence
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

            # 可选的详细 batch 级日志（不需要可以删掉）
            # if (batch_idx + 1) % 200 == 0:
            #     logger.info(f"[Val] Batch {batch_idx+1}/{len(val_loader)} Loss={loss.item():.4f}")

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



def train(args, logger):
    """
    训练模型

    Args:
        args: 命令行参数
        logger: 日志记录器
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    logger.info(f'Using device: {device}')

    # 加载数据集
    logger.info('Loading datasets...')

    # 确定文件路径
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
        dropout=args.dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Total parameters: {total_params:,}')
    logger.info(f'Trainable parameters: {trainable_params:,}')

    # 创建优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=args.patience,
        # verbose=True
    )

    # 训练循环
    best_val_auc = 0.0
    best_epoch = 0
    patience_counter = 0

    logger.info('Starting training...')

    # 保存训练历史
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
            args.lambda_sparse, args.lambda_independence, logger
        )

        logger.info(f'\nTraining Results:')
        logger.info(f'Loss: {train_metrics["loss"]:.4f}, '
                    f'BCE: {train_metrics["bce_loss"]:.4f}, '
                    f'Reg: {train_metrics["reg_loss"]:.4f}')
        logger.info(f'AUC: {train_metrics["auc"]:.4f}, '
                    f'ACC: {train_metrics["acc"]:.4f}, '
                    f'RMSE: {train_metrics["rmse"]:.4f}')

        # 验证
        val_metrics = validate(
            model, val_loader, device,
            args.lambda_sparse, args.lambda_independence, logger
        )

        logger.info(f'\nValidation Results:')
        logger.info(f'Loss: {val_metrics["loss"]:.4f}, '
                    f'BCE: {val_metrics["bce_loss"]:.4f}, '
                    f'Reg: {val_metrics["reg_loss"]:.4f}')
        logger.info(f'AUC: {val_metrics["auc"]:.4f}, '
                    f'ACC: {val_metrics["acc"]:.4f}, '
                    f'RMSE: {val_metrics["rmse"]:.4f}')

        # 更新学习率
        scheduler.step(val_metrics['loss'])

        # 保存训练历史
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)

        # 保存最佳模型
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_epoch = epoch
            patience_counter = 0

            # 保存模型
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

        # 早停
        if patience_counter >= args.early_stop_patience:
            logger.info(f'\nEarly stopping triggered after {epoch} epochs')
            break

        # 定期保存检查点
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

    # 保存训练历史
    history['best_epoch'] = best_epoch
    history['best_val_auc'] = best_val_auc

    history_path = os.path.join(args.save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        # 转换tensor为可序列化的格式
        json.dump(history, f, indent=4, default=lambda x: float(x) if torch.is_tensor(x) else x)

    logger.info(f'\n{"=" * 50}')
    logger.info(f'Training completed!')
    logger.info(f'Best validation AUC: {best_val_auc:.4f} at epoch {best_epoch}')
    logger.info(f'{"=" * 50}')

def inference(args, logger):
    """
    测试模型

    Args:
        args: 命令行参数
        logger: 日志记录器
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    logger.info(f'Using device: {device}')

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

    # 从 checkpoint 中取出映射和 Q 矩阵（完全复用训练时的）
    stu_id_map = info_dict['stu_id_map']
    exer_id_map = info_dict['exer_id_map']
    cpt_id_map = info_dict['cpt_id_map']
    q_matrix = info_dict['q_matrix']

    # 2. 用保存好的映射构建测试数据集（不再调用 create_dataloaders，不会重复打印/构建 Q 矩阵）
    logger.info('Building test dataloader with saved ID mappings...')

    # 2.1 读取原始 test.csv
    raw_test_df = pd.read_csv(test_file)

    # 2.2 按照训练阶段的 ID 映射过滤一遍，避免出现“冷启动学生/题目”导致 KeyError
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

    # 2.3 使用过滤后的 DataFrame 构建 Dataset（不会再出现 stu_id_map 的 KeyError）
    test_dataset = CognitiveDiagnosisDataset(
        csv_file=filtered_test_df,   # 这里传入的是 DataFrame，不是路径
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


    # 3. 创建模型，使用保存的超参数和 Q 矩阵
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
        dropout=loaded_args.get('dropout', args.dropout)
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
        pbar = tqdm(test_loader, desc="Testing", leave=False)
        for batch_idx, batch in enumerate(pbar):
            student_ids, exercise_ids, _, labels = batch
            student_ids = student_ids.to(device)
            exercise_ids = exercise_ids.to(device)
            labels = labels.to(device).float()

            # 前向传播
            pred_probs = model(student_ids, exercise_ids, return_details=False)

            # 收集预测结果
            predictions = (pred_probs > 0.5).float()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probs.extend(pred_probs.cpu().numpy())

            pbar.set_postfix({'Processed': f'{len(all_labels)}/{test_size}'})

    # 5. 计算指标
    metrics = compute_metrics(all_labels, all_predictions, all_probs)

    logger.info(f'\n{"=" * 50}')
    logger.info(f'Test Results:')
    logger.info(f'AUC: {metrics["auc"]:.4f}')
    logger.info(f'ACC: {metrics["acc"]:.4f}')
    logger.info(f'RMSE: {metrics["rmse"]:.4f}')
    logger.info(f'{"=" * 50}')

    # 保存测试结果
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

    # 6. 学生诊断（仍然复用 checkpoint 的映射）
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
    parser.add_argument('--data_dir', type=str, default='./data/assist_09/process_data',)

    # 模型参数
    parser.add_argument('--knowledge_dim', type=int, default=128,
                        help='Dimension of knowledge state (default: 32)')
    parser.add_argument('--skill_dim', type=int, default=2,
                        help='Dimension of test-taking skill ')
    parser.add_argument('--exercise_dim', type=int, default=128,
                        help='Dimension of exercise embedding (default: 64)')
    parser.add_argument('--num_relation_heads', type=int, default=4,
                        help='Number of relation heads (default: 4)')
    parser.add_argument('--num_gnn_layers', type=int, default=2,
                        help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (default: 0.1)')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (default: 1e-5)')
    parser.add_argument('--lambda_sparse', type=float, default=0.1,
                        help='Sparse regularization coefficient (default: 0.01)')
    parser.add_argument('--lambda_independence', type=float, default=0.1,
                        help='Independence regularization coefficient (default: 0.01)')

    # 早停和调度器参数
    parser.add_argument('--patience', type=int, default=5,
                        help='Patience for learning rate scheduler (default: 5)')
    parser.add_argument('--early_stop_patience', type=int, default=5,
                        help='Patience for early stopping (default: 15)')

    # 其他参数
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--generate_diagnosis', default=True,
                        help='Generate student diagnosis reports after testing')

    # 保存参数
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save models (default: ./checkpoints)')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory to save logs (default: ./logs)')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')
    
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

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # 设置日志
    logger = setup_logging(args.log_dir)

    # 保存参数
    args_file = os.path.join(args.save_dir, 'args.json')
    with open(args_file, 'w') as f:
        json.dump(vars(args), f, indent=4)

    # 打印参数
    logger.info('Arguments:')
    for arg, value in sorted(vars(args).items()):
        logger.info(f'  {arg}: {value}')

    # 训练和测试
    train(args, logger)
    inference(args, logger)


if __name__ == '__main__':
    main()