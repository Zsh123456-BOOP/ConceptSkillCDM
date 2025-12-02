import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Disentangled Cognitive Diagnosis with Learned Concept Graphs'
    )

    # ===== 路径相关 =====
    parser.add_argument('--data_root', type=str, default='./data/', help='Root data directory')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Log directory')
    parser.add_argument('--result_dir', type=str, default='./results', help='Results directory')

    parser.add_argument('--dataset', type=str, default='assist_09',
                        choices=['assist_09', 'assist_17', 'junyi'],
                        help='Dataset name')
    parser.add_argument('--train_file', type=str, default='train.csv')
    parser.add_argument('--valid_file', type=str, default='valid.csv')
    parser.add_argument('--test_file', type=str, default='test.csv')
    parser.add_argument('--high_test_file', type=str, default='high_performance_test.csv')
    parser.add_argument('--medium_test_file', type=str, default='medium_performance_test.csv')
    parser.add_argument('--low_test_file', type=str, default='low_performance_test.csv')

    # ===== 模型参数 (DisentangledCDM) =====
    parser.add_argument('--dim_emb', type=int, default=64,
                        help='Student / concept embedding dimension')
    parser.add_argument('--dim_skill', type=int, default=4,
                        help='Skill latent dimension for guess/slip')

    # ===== 训练参数 =====
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=888, help='Random seed')

    # ===== Loss 参数 (CDMLoss) =====
    parser.add_argument('--lambda_dag', type=float, default=0.5,
                        help='DAG constraint weight')
    parser.add_argument('--lambda_sparse', type=float, default=0.01,
                        help='Adjacency L1 sparsity weight')
    parser.add_argument('--lambda_hsic', type=float, default=0.1,
                        help='Disentanglement (HSIC) weight')

    # ===== 数据清洗相关 =====
    parser.add_argument(
        "--min_stu_interactions",
        type=int,
        default=10,
        help="学生最少交互次数阈值，< 该值的学生会被过滤；设为 0 或负数表示不做学生过滤。"
    )
    parser.add_argument(
        "--min_exer_interactions",
        type=int,
        default=10,
        help="题目最少交互次数阈值，< 该值的题目会被过滤；设为 0 或负数表示不做题目过滤。"
    )
    parser.add_argument(
        "--min_poison_count",
        type=int,
        default=10,
        help="毒题检测所需的最少作答次数（count >= 该值 且 正答率=0 或 1 时视为毒题）；设为 0 或负数表示关闭毒题过滤。"
    )

    # ===== 学生分组相关 =====
    parser.add_argument(
        "--low_quantile",
        type=float,
        default=0.33,
        help="用于划分低表现学生的分位数阈值（基于训练集历史正确率）"
    )
    parser.add_argument(
        "--high_quantile",
        type=float,
        default=0.67,
        help="用于划分高表现学生的分位数阈值（基于训练集历史正确率）"
    )

    # ===== 实验标记 =====
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="实验标签，用于 CSV / 日志区分不同配置（如 base/ablation 等）"
    )

    return parser.parse_args()
