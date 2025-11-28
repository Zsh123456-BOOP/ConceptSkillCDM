import argparse


def get_config(args=None):
    """
    Parse command line arguments and return a config namespace.
    """
    parser = argparse.ArgumentParser(
        description="ConceptSkillCDM: Graph-based Concept & Skill Disentangled Cognitive Diagnosis"
    )

    # Basic
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

    # Training
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    # Embedding dimensions
    parser.add_argument("--dim_student", type=int, default=64)
    parser.add_argument("--dim_item", type=int, default=64)
    parser.add_argument("--dim_concept", type=int, default=64)

    # Graph learner
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--graph_dropout", type=float, default=0.1)

    # GNN propagation
    parser.add_argument("--gnn_layers", type=int, default=2)
    parser.add_argument("--gnn_hidden_dim", type=int, default=64)

    # Disentanglement
    parser.add_argument("--skill_dim", type=int, default=8)

    # Loss related
    parser.add_argument("--lambda_graph_sparse", type=float, default=1e-4)
    parser.add_argument("--lambda_graph_sym", type=float, default=1e-4)
    parser.add_argument("--lambda_graph_dag", type=float, default=1e-5)
    parser.add_argument("--lambda_de_orth", type=float, default=1e-3)
    parser.add_argument("--lambda_de_mi", type=float, default=1e-3)

    # Misc
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--save_dir", type=str, default="logs")
    parser.add_argument("--tau", type=float, default=0.5,
                        help="Temperature for soft-min aggregation")

    return parser.parse_args(args=args)


__all__ = ["get_config"]
