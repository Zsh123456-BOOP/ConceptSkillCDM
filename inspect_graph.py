import os
import argparse
import torch
import numpy as np

from test_full_model import HybridKTCDM, load_data

# ===== 和调度脚本一致的 GLOBAL_DEFAULTS / DATASET_CONFIGS =====
GLOBAL_DEFAULTS = {
    "batch_size": 1024,
    "epochs": 80,
    "patience": 15,
    "lr": 1e-3,
    "weight_decay": 1e-4,

    "dim_student": 64,
    "dim_item": 64,
    "dim_concept": 64,
    "dim_hidden": 64,
    "dim_skill": 16,

    "graph_dropout": 0.2,
    "graph_topk": 8,
    "dropout": 0.3,
    "softmin_beta": 5.0,

    "lambda_dag": 0.01,
    "lambda_sym": 0.10,
    "lambda_ent_dag": 0.02,
    "lambda_ent_sym": 0.02,
    "lambda_orth": 0.01,

    "model_name": "HybridKTCDM"
}

DATASET_CONFIGS = {
    "sample": {
        "batch_size": 1024,
        "epochs": 60,
        "patience": 10,

        "dim_student": 32,
        "dim_item": 32,
        "dim_concept": 32,
        "dim_hidden": 32,
        "dim_skill": 8,

        "dropout": 0.5,
        "graph_dropout": 0.3,
        "graph_topk": 4,

        "lambda_ent_dag": 0.02,
        "lambda_ent_sym": 0.02,
    },

    "assist_09": {
        "batch_size": 512,
        "epochs": 100,
        "patience": 10,

        "dim_student": 32,
        "dim_item": 32,
        "dim_concept": 32,
        "dim_hidden": 32,
        "dim_skill": 8,

        "dropout": 0.5,
        "graph_dropout": 0.5,
        "graph_topk": 8,
        "weight_decay": 1e-3,

        "lambda_ent_dag": 0.05,
        "lambda_ent_sym": 0.02,
        "lambda_dag": 0.01,
        "lambda_sym": 0.10,
    },

    "assist_17": {
        "batch_size": 1024,
        "epochs": 100,
        "patience": 10,

        "dim_student": 128,
        "dim_item": 64,
        "dim_concept": 64,
        "dim_hidden": 128,
        "dim_skill": 16,

        "dropout": 0.3,
        "graph_dropout": 0.3,
        "graph_topk": 8,
        "lr": 5e-4,

        "lambda_ent_dag": 0.02,
        "lambda_ent_sym": 0.01,
        "lambda_dag": 0.01,
        "lambda_sym": 0.10,
    }
}

def build_model_args(dataset_name: str):
    """
    构造一个“假的 args 对象”，只包含 HybridKTCDM 初始化需要的字段。
    """
    cfg = dict(GLOBAL_DEFAULTS)
    cfg.update(DATASET_CONFIGS.get(dataset_name, {}))

    class Obj:
        pass

    args = Obj()
    for k, v in cfg.items():
        setattr(args, k, v)
    return args

def print_graph_stats(adj, name):
    import numpy as np
    row_entropy = -np.sum(adj * np.log(adj + 1e-12), axis=1)
    print(f"\n[{name}] row entropy: mean={row_entropy.mean():.4f}, "
          f"min={row_entropy.min():.4f}, max={row_entropy.max():.4f}")
    col_sum = adj.sum(axis=0)
    print(f"[{name}] in-degree (col-sum): mean={col_sum.mean():.4f}, "
          f"max={col_sum.max():.4f}")   

def inspect_graph(dataset: str, log_dir: str, ckpt_suffix: str, topk: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 载入数据以拿到 num_concepts
    _, _, _, info = load_data(dataset)
    num_students = info["num_students"]
    num_items = info["num_items"]
    num_concepts = info["num_concepts"]

    print(f"[{dataset}] num_students={num_students}, num_items={num_items}, num_concepts={num_concepts}")

    # 2. 构造 args & 模型
    args_model = build_model_args(dataset)
    model = HybridKTCDM(num_students, num_items, num_concepts, args_model).to(device)

    # 3. 加载 ckpt
    ckpt_path = os.path.join(log_dir, f"{dataset}{ckpt_suffix}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 4. 调用 get_graph()
    with torch.no_grad():
        adj = model.get_graph().detach().cpu().numpy()  # [2, N, N]

    adj_dag = adj[0]   # 前置关系头
    adj_sym = adj[1]   # 相似关系头

    print("\n=== Head 0: 前置关系（每行概念 i -> 概念 j） ===")
    for i in range(min(num_concepts, 20)):  # 初步只看前 20 个概念，防止刷屏
        row = adj_dag[i]
        top_idx = np.argsort(row)[::-1][:topk]
        print(f"Concept {i} ->", end=" ")
        for j in top_idx:
            print(f"{j}(w={row[j]:.3f})", end="; ")
        print()

    print("\n=== Head 1: 相似关系（尽量对称） ===")
    for i in range(min(num_concepts, 20)):
        row = adj_sym[i]
        top_idx = np.argsort(row)[::-1][:topk]
        print(f"[Sim] Concept {i} ~", end=" ")
        for j in top_idx:
            print(f"{j}(w={row[j]:.3f})", end="; ")
        print()

    print_graph_stats(adj_dag, "Head0-DAG")
    print_graph_stats(adj_sym, "Head1-Sym")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["sample", "assist_09", "assist_17"])
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--ckpt_suffix", type=str, default="_best_v3_fixed.pt")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    inspect_graph(args.dataset, args.log_dir, args.ckpt_suffix, args.topk)

if __name__ == "__main__":
    main()
