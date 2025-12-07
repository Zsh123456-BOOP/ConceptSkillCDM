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
        "base": {},
        "search": {
            "learning_rate": [1e-4, 3e-4],
            "lambda_sparse": [0.05, 0.1],
            "lambda_independence": [0.05, 0.1],
            "dropout": [0.1, 0.2],
            # 消融参数预留（当前单一取值）
            "ablate_soft_prototype": [False],
            "ablate_skill_encoder": [False],
            "ablate_exercise_graph": [False],
            "ablate_concept_fusion": [False],
        },
    },
    "assist_17": {
        "base": {},
        "search": {
            "learning_rate": [1e-4, 3e-4],
            "ablate_soft_prototype": [False],
            "ablate_skill_encoder": [False],
        },
    },
    "junyi": {
        "base": {},
        "search": {
            "learning_rate": [1e-4, 3e-4],
            "ablate_soft_prototype": [False],
            "ablate_skill_encoder": [False],
        },
    },
}


def apply_dataset_defaults(args, parser=None):
    """
    根据 dataset_name 自动覆盖数据集默认配置。
    - 若提供 parser，则仅当当前值等于 parser 的默认值（即用户未手动覆盖）时才替换。
    - 若未提供 parser，则仅在属性不存在或为 None 时覆盖。
    """
    dataset_name = getattr(args, "dataset_name", None)
    if dataset_name is None or dataset_name not in DATASET_DEFAULTS:
        return args

    defaults = DATASET_DEFAULTS[dataset_name]
    for key, value in defaults.items():
        if parser is not None and parser.get_default(key) is not None:
            # 用户未覆盖且当前值等于 argparse 默认值时才替换
            if getattr(args, key, None) == parser.get_default(key):
                setattr(args, key, value)
        else:
            # 没有 parser 的情况下，仅当不存在或为 None 时覆盖
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
    return args
