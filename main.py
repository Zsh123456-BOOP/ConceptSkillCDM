import sys
import os
import torch
import pandas as pd
from datetime import datetime

from src.config import parse_args
from src.utils import setup_logger, get_device, set_seed
from src.dataset import CognitiveDataProcessor
from src.disentangled_cdm import DisentangledCDM
from src.cdm_loss import CDMLoss


def save_results(args, metrics, result_file):
    """将实验结果追加保存到 CSV，包含所有关键超参数（已改为适配 DisentangledCDM）"""
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    filepath = os.path.join(args.result_dir, result_file)

    row = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Dataset": args.dataset,
        "Seed": args.seed,
        "Tag": getattr(args, "tag", ""),

        # 核心指标
        "Test_AUC": metrics.get("Test_AUC", "N/A"),
        "Test_ACC": metrics.get("Test_ACC", "N/A"),
        "Test_RMSE": metrics.get("Test_RMSE", "N/A"),

        # 分组指标
        "High_AUC": metrics.get("High_AUC", "N/A"),
        "Medium_AUC": metrics.get("Medium_AUC", "N/A"),
        "Low_AUC": metrics.get("Low_AUC", "N/A"),
        "High_RMSE": metrics.get("High_RMSE", "N/A"),
        "Medium_RMSE": metrics.get("Medium_RMSE", "N/A"),
        "Low_RMSE": metrics.get("Low_RMSE", "N/A"),

        # 模型与优化超参数
        "DimEmb": args.dim_emb,
        "DimSkill": args.dim_skill,
        "LR": args.lr,
        "Batch": args.batch_size,
        "WeightDecay": args.weight_decay,
        "Epochs": args.epochs,
        "Patience": args.patience,

        # Loss 超参数
        "Lambda_DAG": args.lambda_dag,
        "Lambda_Sparse": args.lambda_sparse,
        "Lambda_HSIC": args.lambda_hsic,

        # 数据清洗与分组相关
        "MinStu": getattr(args, "min_stu_interactions", -1),
        "MinExer": getattr(args, "min_exer_interactions", -1),
        "MinPoison": getattr(args, "min_poison_count", -1),
        "LowQuantile": getattr(args, "low_quantile", "N/A"),
        "HighQuantile": getattr(args, "high_quantile", "N/A"),
    }

    df_new = pd.DataFrame([row])

    # 智能写入：如果列有变化，自动重新对齐
    if not os.path.exists(filepath):
        df_new.to_csv(filepath, index=False)
    else:
        try:
            df_old = pd.read_csv(filepath, nrows=0)
            if list(df_new.columns) != list(df_old.columns):
                df_old_full = pd.read_csv(filepath)
                df_combined = pd.concat([df_old_full, df_new], ignore_index=True)
                df_combined.to_csv(filepath, index=False)
            else:
                df_new.to_csv(filepath, mode='a', header=False, index=False)
        except Exception:
            df_new.to_csv(filepath, mode='a', header=False, index=False)

    print(f"✅ Results saved to {filepath}")


def build_model_save_path(args):
    """
    根据 dataset / batch_size / dim_emb / tag 构造模型保存路径，
    适配新的 DisentangledCDM。
    """
    name = f"best_{args.dataset}_bs{args.batch_size}_dim{args.dim_emb}"
    tag = getattr(args, "tag", "")
    if tag:
        safe_tag = str(tag).replace(" ", "_")
        name += f"_{safe_tag}"
    filename = name + ".pth"
    return os.path.join(args.log_dir, filename)


def main():
    # 1. 基础设置
    args = parse_args()
    logger = setup_logger(args.log_dir)

    logger.info(f"🚀 Starting Experiment on [{args.dataset}] with Seed {args.seed}")
    logger.info(f"   Tag = {getattr(args, 'tag', '')}")

    device = get_device()
    set_seed(args.seed)

    # 2. 数据处理（保持使用你原来的 CognitiveDataProcessor）
    try:
        dp = CognitiveDataProcessor(args, logger)
        loaders = dp.get_loaders()
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        return

    # 要求 CognitiveDataProcessor 提供如下属性：
    #   dp.num_students, dp.num_exercises, dp.num_concepts
    #   dp.Q_matrix（可选）
    num_students = dp.num_students
    num_exercises = dp.num_exercises
    num_concepts = dp.num_concepts
    q_matrix = getattr(dp, "Q_matrix", None)

    logger.info(
        f"📊 Data Summary: num_students={num_students}, "
        f"num_exercises={num_exercises}, num_concepts={num_concepts}"
    )

    # 3. 模型 & 损失
    model = DisentangledCDM(
        num_students=num_students,
        num_exercises=num_exercises,
        num_concepts=num_concepts,
        dim_emb=args.dim_emb,
        dim_skill=args.dim_skill,
        q_matrix=q_matrix,
    ).to(device)

    criterion = CDMLoss(
        lambda_dag=args.lambda_dag,
        lambda_sparse=args.lambda_sparse,
        lambda_hsic=args.lambda_hsic,
    )

    # 4. Trainer
    from src.trainer import Trainer  # 避免循环引用

    trainer = Trainer(model, criterion, loaders, args, logger)

    # 5. 训练循环
    best_auc = 0.0
    patience = 0

    model_save_path = build_model_save_path(args)
    logger.info(f"🧾 Model will be saved to: {model_save_path}")

    # 开始前清理旧 checkpoint，避免结构不匹配
    if os.path.exists(model_save_path):
        logger.info(f"🧹 [Auto-Clean] Found existing checkpoint: {model_save_path}, removing...")
        try:
            os.remove(model_save_path)
        except Exception as e:
            logger.warning(f"   -> Failed to remove old checkpoint: {e}")

    for epoch in range(1, args.epochs + 1):
        train_loss = trainer.train_epoch(epoch)
        auc, _, _ = trainer.evaluate(trainer.val_loader, "Valid")

        if auc > best_auc:
            best_auc = auc
            patience = 0
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"*** New Best Valid AUC: {best_auc:.4f} *** (saved to {model_save_path})")
        else:
            patience += 1
            if patience >= args.patience:
                logger.info("Early stopping triggered.")
                break

    # 6. 最终测试与结果保存
    if os.path.exists(model_save_path):
        logger.info("\n>>> Final Testing...")
        try:
            state = torch.load(model_save_path, map_location=device)
            model.load_state_dict(state)
        except RuntimeError as e:
            logger.error(f"❌ Error loading state_dict from {model_save_path}: {e}")
            return

        final_metrics = {}

        # 主测试集
        auc, acc, rmse = trainer.evaluate(trainer.test_loader, "Test")
        final_metrics.update({"Test_AUC": auc, "Test_ACC": acc, "Test_RMSE": rmse})

        # 分层测试集 (如果有)
        if trainer.high_loader:
            h_auc, h_acc, h_rmse = trainer.evaluate(trainer.high_loader, "High")
            final_metrics.update({"High_AUC": h_auc, "High_ACC": h_acc, "High_RMSE": h_rmse})

        if trainer.med_loader:
            m_auc, m_acc, m_rmse = trainer.evaluate(trainer.med_loader, "Medium")
            final_metrics.update({"Medium_AUC": m_auc, "Medium_ACC": m_acc, "Medium_RMSE": m_rmse})

        if trainer.low_loader:
            l_auc, l_acc, l_rmse = trainer.evaluate(trainer.low_loader, "Low")
            final_metrics.update({"Low_AUC": l_auc, "Low_ACC": l_acc, "Low_RMSE": l_rmse})

        # 保存到 CSV
        save_results(args, final_metrics, "all_datasets_results.csv")
    else:
        logger.error("No model saved. Training might have failed or Valid AUC never improved.")


if __name__ == '__main__':
    main()
