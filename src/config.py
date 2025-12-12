# src/config.py
"""
Dataset defaults + small grid space.

Important:
- We keep the meaning of `lambda_sparse` at CLI level:
  it is still the "graph sparsity regularization weight".
- In the new model, it is mapped to `lambda_graph_entropy`.
"""

DATASET_DEFAULTS = {
    "assist_09": {
        "data_dir": "./data/assist_09",
        "batch_size": 512,
        "knowledge_dim": 128,
        "exercise_dim": 128,
        "skill_dim": 64,  # new model prefers MF latent >= 64 (but run_all_datasets overrides anyway)
        "num_relation_heads": 4,
        "num_gnn_layers": 2,
        "dropout": 0.1,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "lambda_sparse": 0.01,          # graph entropy weight
        "lambda_proto_div": 0.0,
        "lambda_proto_usage": 0.0,
        "lambda_sparse_personal": 0.0,
        "lambda_alpha": 0.0,
        "use_personal_graph": 0,
        "num_prototypes": 3,
        "proto_tau": 1.0,
        "proto_lambda": 0.5,
        "exercise_l2_lambda": 5e-5,     # mapped to mf_l2_lambda
        # optional new knobs (do not affect old runs if not used)
        "graph_topk": None,
        "disable_q_conditioning": False,
        "disable_self_loop": False,
        "personal_rank": 4,
        "gnn_residual_weight": 0.5,
    },
    "assist_17": {
        "data_dir": "./data/assist_17",
        "batch_size": 512,
        "learning_rate": 1e-4,
        "lambda_sparse": 0.01,
        "lambda_proto_div": 0.0,
        "lambda_proto_usage": 0.0,
        "lambda_sparse_personal": 0.0,
        "lambda_alpha": 0.0,
        "use_personal_graph": 0,
        "exercise_l2_lambda": 5e-5,
    },
    "junyi": {
        "data_dir": "./data/junyi",
        "batch_size": 512,
        "learning_rate": 1e-4,
        "lambda_sparse": 0.01,
        "lambda_proto_div": 0.0,
        "lambda_proto_usage": 0.0,
        "lambda_sparse_personal": 0.0,
        "lambda_alpha": 0.0,
        "use_personal_graph": 0,
        "exercise_l2_lambda": 5e-5,
    },
}

GRID_SEARCH_SPACE = {
    "assist_09": {
        "base": {},
        "search": {
            "learning_rate": [1e-4, 3e-4],
            "lambda_sparse": [0.05, 0.1],
            "dropout": [0.1, 0.2],
            "ablate_soft_prototype": [False],
            "ablate_skill_encoder": [False],
            "ablate_exercise_graph": [False],
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
            "learning_rate": [3e-4, 1e-3],
            "lambda_sparse": [0.05, 0.1],
            "dropout": [0.1, 0.2],
            "ablate_soft_prototype": [False],
            "ablate_skill_encoder": [False],
            "ablate_exercise_graph": [False],
        },
    },
}


def apply_dataset_defaults(args, parser=None):
    """
    Apply dataset defaults only when user did not manually override.

    If parser is provided:
      - overwrite only when current value equals argparse default.
    Else:
      - overwrite only when attribute missing or None.
    """
    dataset_name = getattr(args, "dataset_name", None)
    if dataset_name is None or dataset_name not in DATASET_DEFAULTS:
        return args

    defaults = DATASET_DEFAULTS[dataset_name]
    for key, value in defaults.items():
        if parser is not None and parser.get_default(key) is not None:
            if getattr(args, key, None) == parser.get_default(key):
                setattr(args, key, value)
        else:
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
    return args
