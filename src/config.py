import argparse


def get_config(args=None):
    """
    Parse command line arguments and return a config namespace.
    """
    parser = argparse.ArgumentParser(
        description="ConceptSkillCDM: Graph-based Concept & Skill Disentangled Cognitive Diagnosis"
    )

    # 基础参数
    parser.add_argument("--dataset", type=str, default="assist_09",
                        choices=["assist_09", "assist_17", "junyi", "sample", "junyi copy"])
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Device to use: "auto" (default, picks best GPU if available), "cpu", "cuda", "cuda:0", "cuda:1", ...',
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)

    # 训练相关
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience based on validation AUC; set <=0 to disable.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    # 各嵌入维度
    parser.add_argument("--dim_student", type=int, default=64)
    parser.add_argument("--dim_item", type=int, default=64)
    parser.add_argument("--dim_concept", type=int, default=64)

    # 图学习器
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--graph_dropout", type=float, default=0.1)
    parser.add_argument("--graph_topk", type=int, default=32,
                        help="图学习阶段每行保留的 Top-K 边，<=0 表示不裁剪，依赖 softmax+稀疏正则。")

    # GNN 传播
    parser.add_argument("--gnn_layers", type=int, default=2)
    parser.add_argument("--gnn_hidden_dim", type=int, default=64)

    # 解耦相关
    parser.add_argument("--skill_dim", type=int, default=8)

    # 损失项权重
    parser.add_argument("--lambda_graph_sparse", type=float, default=1e-4)
    parser.add_argument("--lambda_graph_sym", type=float, default=1e-4)
    parser.add_argument("--lambda_graph_dag", type=float, default=1e-5)
    parser.add_argument("--lambda_graph_trans", type=float, default=1e-4,
                        help="概念前置关系的传递性/层次性软约束权重，0 表示关闭。")
    parser.add_argument("--lambda_de_orth", type=float, default=1e-3)
    parser.add_argument("--lambda_de_mi", type=float, default=1e-3)

    # 其他
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--save_dir", type=str, default="logs")
    parser.add_argument("--tau", type=float, default=0.5,
                        help="Temperature for soft-min aggregation")
    parser.add_argument("--agg_type", type=str, default="softmin",
                        choices=["softmin", "min", "mean"],
                        help="题目概念聚合方式：softmin（可调 tau）、min、mean。")

    return parser.parse_args(args=args)


__all__ = ["get_config"]
