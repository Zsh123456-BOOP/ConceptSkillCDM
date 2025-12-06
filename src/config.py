# src/config.py

# 每个数据集的基础配置
DATASET_DEFAULTS = {
    "assist_09": {
        "data_dir": "./data/assist_09",
        "batch_size": 512,
        "knowledge_dim": 128,
        "exercise_dim": 128,
        "skill_dim": 2,
        "num_relation_heads": 4,
        "num_gnn_layers": 2,
        "dropout": 0.1,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "lambda_sparse": 0.1,
        "lambda_independence": 0.1,
        "lambda_proto_div": 0.01,
        "lambda_proto_usage": 0.01,
        "num_prototypes": 3,
        "proto_tau": 1.0,
        "proto_lambda": 0.5,
    },
    "assist_17": {
        "data_dir": "./data/assist_17",
        "batch_size": 512,
        "learning_rate": 1e-4,
        # 其他如果不写就用 argparse 默认
    },
    "junyi": {
        "data_dir": "./data/junyi",
        "batch_size": 512,
        "learning_rate": 1e-4,
    },
}

# 小范围网格搜索空间（每个数据集可以复用同一套，也可以单独定义）
GRID_SEARCH_SPACE = {
    "assist_09": {
        "learning_rate": [1e-4, 3e-4],
        "lambda_sparse": [0.05, 0.1],
        "lambda_independence": [0.05, 0.1],
        "dropout": [0.1, 0.2],
    },
    "assist_17": {
        "learning_rate": [1e-4, 3e-4],
    },
    "junyi": {
        "learning_rate": [1e-4, 3e-4],
    },
}
