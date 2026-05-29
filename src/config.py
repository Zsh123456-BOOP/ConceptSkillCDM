# src/config.py
"""
Dataset defaults and optional grid-search space.

Notes:
- `lambda_sparse` is interpreted as graph sparsity regularization weight at CLI level.
- In model code it is mapped to `lambda_graph_entropy`.
"""

DATASET_DEFAULTS = {
    "assist_09": {
        "data_dir": "./data/assist_09",
        "batch_size": 512,
        "knowledge_dim": 128,
        "num_relation_heads": 4,
        "num_gnn_layers": 2,
        "dropout": 0.1,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "lambda_sparse": 0.3,
        "graph_entropy_min": 0.15,
        "graph_entropy_max": 0.80,
        "lambda_graph_diag": 0.10,
        "lambda_graph_uniform": 0.04,
        "graph_uniform_margin": 0.10,
        "graph_reg_warmup_epochs": 0,
        "graph_reg_cap_ratio": 6.0,
        "graph_dropout": 0.0,
        "graph_tau_init": 0.55,
        "graph_identity_residual": 0.05,
        "graph_propagation_alpha": 0.02,
        "graph_query_readout_scale": 0.02,
        "graph_query_readout_2hop_scale": 0.01,
        "lambda_sparse_personal": 0.0005,
        "lambda_alpha": 0.0,
        "lambda_alpha_min": 0.08,
        "alpha_min_target": 0.05,
        "lambda_personal_kl": 0.04,
        "lambda_personal_query_residual": 0.06,
        "personal_query_residual_margin": 0.14,
        "use_personal_graph": True,
        "personal_rank": 8,
        "personal_max_alpha": 0.28,
        "personal_delta_scale": 4.0,
        "personal_warmup_epochs": 8,
        "personal_reg_warmup_epochs": 8,
        "personal_student_dim": 32,
        "personal_alpha_temperature": 2.2,
        "personal_alpha_budget": 0.10,
        "personal_alpha_base_init": 0.05,
        "personal_alpha_bias_scale": 0.0,
        "personal_disable_student_global_context": True,
        "personal_local_hops": 1,
        "personal_include_neighbor_rows": False,
        "personal_support_include_query_self": True,
        "personal_support_include_graph": True,
        "personal_support_include_neighbors": False,
        "personal_item_support_mass": 0.20,
        "personal_mastery_prior_scale": 0.0,
        "personal_recent_mastery_prior_scale": 0.0,
        "personal_mastery_count_smoothing": 8.0,
        "personal_query_row_budget": 1.0,
        "personal_neighbor_row_budget": 0.35,
        "personal_query_support_hops": 1,
        "personal_support_only": True,
        "personal_query_message_gain": 1.0,
        "personal_value_use_global_basis": True,
        "personal_message_alignment_gate": True,
        "personal_projection_hidden_factor": 2,
        "personal_query_correction_scale": 0.20,
        "personal_query_correction_max_ratio": 0.05,
        "personal_query_correction_min_graph_anchor": 0.02,
        "graph_headwise_query_gate": True,
        "graph_edge_bias_rank": 8,
        "graph_prior_logit_scale": 0.55,
        "ae_query_residual_scale": 0.0,
        "ae_logit_residual_scale": 1.00,
        "roadmap_logit_residual_scale": 0.70,
        "tutor_logit_residual_scale": 1.30,
        "ae_logit_residual_clip": 6.00,
        "ae_irt_logit_scale": 0.20,
        "ae_interaction_logit_scale": 1.00,
        "ae_logit_dim": 64,
        "ae_lr_mult": 30.0,
        "ae_stat_prior_scale": 1.0,
        "lambda_theta_prior_align": 0.0,
        "relation_theta_scale": 0.0,
        "relation_theta_delta_clip": 2.0,
        "roadmap_theta_calibration_scale": 0.0,
        "tutor_theta_calibration_scale": 0.0,
        "theta_calibration_clip": 0.75,
        "ae_posterior_prior_scale": 1.0,
        "ae_posterior_theta_scale": 0.0,
        "graph_query_adapter_enable": True,
        "personal_state_lr_mult": 1.0,
        "personal_id_lr_mult": 0.5,
        "share_concept_embeddings": True,
        "prediction_l2_lambda": 1e-5,
        "debug_graph_diag": True,
        "diag_batches": 1,
        "graph_topk": 12,
        "disable_self_loop": False,
        "gnn_residual_weight": 0.5,
    },
    "assist_17": {
        "data_dir": "./data/assist_17",
        "batch_size": 512,
        "learning_rate": 1e-4,
        "lambda_sparse": 0.6,
        "graph_entropy_min": 0.15,
        "graph_entropy_max": 0.80,
        "lambda_graph_diag": 0.10,
        "lambda_graph_uniform": 0.04,
        "graph_uniform_margin": 0.10,
        "graph_reg_warmup_epochs": 0,
        "graph_reg_cap_ratio": 6.0,
        "graph_dropout": 0.0,
        "graph_tau_init": 0.55,
        "graph_identity_residual": 0.05,
        "graph_propagation_alpha": 0.20,
        "graph_query_readout_scale": 0.35,
        "graph_query_readout_2hop_scale": 0.12,
        "lambda_sparse_personal": 0.0,
        "lambda_alpha": 0.0,
        "lambda_alpha_min": 0.0,
        "alpha_min_target": 0.0,
        "lambda_personal_kl": 0.0,
        "lambda_personal_query_residual": 0.0,
        "personal_query_residual_margin": 0.08,
        "use_personal_graph": True,
        "personal_delta_scale": 2.0,
        "personal_warmup_epochs": 3,
        "personal_reg_warmup_epochs": 3,
        "personal_alpha_temperature": 2.0,
        "personal_alpha_budget": 0.08,
        "personal_alpha_base_init": 0.05,
        "personal_alpha_bias_scale": 0.0,
        "personal_disable_student_global_context": True,
        "personal_local_hops": 1,
        "personal_include_neighbor_rows": False,
        "personal_support_include_query_self": True,
        "personal_support_include_graph": True,
        "personal_support_include_neighbors": False,
        "personal_item_support_mass": 0.20,
        "personal_mastery_prior_scale": 1.0,
        "personal_recent_mastery_prior_scale": 0.5,
        "personal_mastery_count_smoothing": 8.0,
        "personal_query_row_budget": 1.0,
        "personal_neighbor_row_budget": 0.30,
        "personal_query_support_hops": 1,
        "personal_support_only": True,
        "personal_query_message_gain": 1.0,
        "personal_value_use_global_basis": True,
        "personal_message_alignment_gate": True,
        "personal_projection_hidden_factor": 2,
        "personal_query_correction_scale": 0.15,
        "personal_query_correction_max_ratio": 0.25,
        "personal_query_correction_min_graph_anchor": 0.05,
        "graph_headwise_query_gate": True,
        "graph_edge_bias_rank": 8,
        "graph_prior_logit_scale": 0.35,
        "ae_query_residual_scale": 0.0,
        "ae_logit_residual_scale": 0.35,
        "roadmap_logit_residual_scale": 0.70,
        "tutor_logit_residual_scale": 1.10,
        "ae_logit_residual_clip": 1.00,
        "ae_irt_logit_scale": 1.00,
        "ae_interaction_logit_scale": 0.00,
        "ae_logit_dim": 32,
        "ae_lr_mult": 1.0,
        "ae_stat_prior_scale": 0.0,
        "lambda_theta_prior_align": 0.0,
        "relation_theta_scale": 0.0,
        "relation_theta_delta_clip": 2.0,
        "ae_posterior_prior_scale": 1.0,
        "ae_posterior_theta_scale": 0.0,
        "graph_query_adapter_enable": True,
        "personal_state_lr_mult": 1.0,
        "personal_id_lr_mult": 0.5,
        "share_concept_embeddings": True,
        "prediction_l2_lambda": 1e-5,
        "debug_graph_diag": True,
        "diag_batches": 1,
        "graph_topk": 24,
    },
    "junyi": {
        "data_dir": "./data/junyi",
        "batch_size": 64,
        "knowledge_dim": 128,
        "num_relation_heads": 2,
        "num_gnn_layers": 2,
        "dropout": 0.40,
        "learning_rate": 0.003,
        "weight_decay": 0.001,
        "lambda_sparse": 0.6,
        "graph_entropy_min": 0.15,
        "graph_entropy_max": 0.70,
        "lambda_graph_diag": 0.12,
        "lambda_graph_uniform": 0.06,
        "graph_uniform_margin": 0.12,
        "graph_reg_warmup_epochs": 2,
        "graph_reg_cap_ratio": 6.0,
        "graph_dropout": 0.0,
        "graph_tau_init": 0.55,
        "graph_identity_residual": 0.05,
        "graph_propagation_alpha": 0.25,
        "graph_query_readout_scale": 0.45,
        "graph_query_readout_2hop_scale": 0.12,
        "lambda_sparse_personal": 0.0,
        "lambda_alpha": 0.0,
        "lambda_alpha_min": 0.0,
        "alpha_min_target": 0.0,
        "lambda_personal_kl": 0.0,
        "lambda_personal_query_residual": 0.0,
        "personal_query_residual_margin": 0.12,
        "use_personal_graph": True,
        "personal_rank": 8,
        "personal_max_alpha": 0.32,
        "personal_delta_scale": 1.5,
        "personal_warmup_epochs": 3,
        "personal_reg_warmup_epochs": 3,
        "personal_student_dim": 64,
        "personal_alpha_temperature": 1.8,
        "personal_alpha_budget": 0.09,
        "personal_alpha_base_init": 0.04,
        "personal_alpha_bias_scale": 0.0,
        "personal_disable_student_global_context": True,
        "personal_local_hops": 1,
        "personal_include_neighbor_rows": False,
        "personal_support_include_query_self": True,
        "personal_support_include_graph": True,
        "personal_support_include_neighbors": False,
        "personal_item_support_mass": 0.20,
        "personal_mastery_prior_scale": 1.0,
        "personal_recent_mastery_prior_scale": 0.5,
        "personal_mastery_count_smoothing": 8.0,
        "personal_query_row_budget": 1.3,
        "personal_neighbor_row_budget": 0.20,
        "personal_query_support_hops": 1,
        "personal_support_only": True,
        "personal_query_message_gain": 1.0,
        "personal_value_use_global_basis": True,
        "personal_message_alignment_gate": True,
        "personal_projection_hidden_factor": 2,
        "personal_query_correction_scale": 0.10,
        "personal_query_correction_max_ratio": 0.2,
        "personal_query_correction_min_graph_anchor": 0.01,
        "graph_headwise_query_gate": True,
        "graph_edge_bias_rank": 8,
        "graph_prior_logit_scale": 0.45,
        "ae_query_residual_scale": 0.0,
        "ae_logit_residual_scale": 0.50,
        "roadmap_logit_residual_scale": 0.70,
        "tutor_logit_residual_scale": 1.10,
        "ae_logit_residual_clip": 6.00,
        "ae_irt_logit_scale": 1.00,
        "ae_interaction_logit_scale": 0.25,
        "ae_logit_dim": 64,
        "ae_lr_mult": 1.0,
        "ae_stat_prior_scale": 1.0,
        "lambda_theta_prior_align": 0.0,
        "relation_theta_scale": 0.0,
        "relation_theta_delta_clip": 2.0,
        "ae_posterior_prior_scale": 1.0,
        "ae_posterior_theta_scale": 0.0,
        "graph_query_adapter_enable": True,
        "personal_state_lr_mult": 1.5,
        "personal_id_lr_mult": 0.25,
        "share_concept_embeddings": True,
        "prediction_l2_lambda": 1e-5,
        "debug_graph_diag": True,
        "diag_batches": 1,
        "graph_topk": 24,
        "disable_self_loop": False,
        "gnn_residual_weight": 0.5,
    },
}

DATASET_DEFAULTS["cdbd_lsat"] = {
    **DATASET_DEFAULTS["assist_09"],
    "data_dir": "./data/cdbd_lsat",
    "batch_size": 32,
    "knowledge_dim": 32,
    "num_relation_heads": 2,
    "num_gnn_layers": 1,
    "dropout": 0.10,
    "learning_rate": 1e-3,
    "graph_topk": 4,
    "personal_student_dim": 16,
    "personal_rank": 4,
}

DATASET_DEFAULTS["cdbd_a0910"] = {
    **DATASET_DEFAULTS["assist_09"],
    "data_dir": "./data/cdbd_a0910",
}

PUBLIC_BENCHMARK_DEFAULTS = {
    **DATASET_DEFAULTS["assist_09"],
    "min_stu_interactions": 0,
    "min_exer_interactions": 0,
    "min_poison_count": 0,
}

DATASET_DEFAULTS["frcsub"] = {
    **PUBLIC_BENCHMARK_DEFAULTS,
    "data_dir": "./data/frcsub",
    "batch_size": 128,
    "knowledge_dim": 32,
    "num_relation_heads": 2,
    "num_gnn_layers": 1,
    "dropout": 0.10,
    "learning_rate": 1e-3,
    "graph_topk": 7,
    "personal_student_dim": 16,
    "personal_rank": 4,
    "disable_sequence_prior": True,
}

DATASET_DEFAULTS["math2"] = {
    **PUBLIC_BENCHMARK_DEFAULTS,
    "data_dir": "./data/math2",
    "batch_size": 256,
    "knowledge_dim": 32,
    "num_relation_heads": 2,
    "num_gnn_layers": 1,
    "dropout": 0.10,
    "learning_rate": 1e-3,
    "graph_topk": 8,
    "personal_student_dim": 16,
    "personal_rank": 4,
    "disable_sequence_prior": True,
}

DATASET_DEFAULTS["assist_12"] = {
    **PUBLIC_BENCHMARK_DEFAULTS,
    "data_dir": "./data/assist_12",
    "batch_size": 1024,
    "graph_topk": 24,
}

DATASET_DEFAULTS["assist_15"] = {
    **PUBLIC_BENCHMARK_DEFAULTS,
    "data_dir": "./data/assist_15",
    "batch_size": 1024,
    "knowledge_dim": 64,
    "num_relation_heads": 2,
    "graph_topk": 24,
}

DATASET_DEFAULTS["nips34"] = {
    **PUBLIC_BENCHMARK_DEFAULTS,
    "data_dir": "./data/nips34",
    "batch_size": 1024,
    "graph_topk": 24,
}

DATASET_DEFAULTS["nips34_l3"] = {
    **DATASET_DEFAULTS["nips34"],
    "data_dir": "./data/nips34_l3",
}

DATASET_DEFAULTS["ednet_kt1"] = {
    **PUBLIC_BENCHMARK_DEFAULTS,
    "data_dir": "./data/ednet_kt1",
    "batch_size": 4096,
    "knowledge_dim": 64,
    "num_relation_heads": 2,
    "num_gnn_layers": 1,
    "graph_topk": 24,
    "personal_student_dim": 32,
}

DATASET_DEFAULTS["assist_12_clean15_item50"] = {
    **DATASET_DEFAULTS["assist_12"],
    "data_dir": "./data/assist_12_clean15_item50",
}

DATASET_DEFAULTS["ednet_kt1_clean15_sample5000"] = {
    **DATASET_DEFAULTS["ednet_kt1"],
    "data_dir": "./data/ednet_kt1_clean15_sample5000",
}

DATASET_DEFAULTS["ednet_kt1_gap"] = {
    **DATASET_DEFAULTS["ednet_kt1"],
    "data_dir": "./data/ednet_kt1_gap",
}

for _ednet_gap_sweep_name in (
    "ednet_kt1_gap_t70u5000",
    "ednet_kt1_gap_t60u5000",
    "ednet_kt1_gap_t50u5000",
    "ednet_kt1_gap_t65u3000_long",
    "ednet_kt1_gap_small_t70u2000",
    "ednet_kt1_gap_small_t60u2200",
    "ednet_kt1_gap_small_t50u2600",
    "ednet_kt1_gap_small_t65u1200_long",
):
    DATASET_DEFAULTS[_ednet_gap_sweep_name] = {
        **DATASET_DEFAULTS["ednet_kt1"],
        "data_dir": f"./data/{_ednet_gap_sweep_name}",
    }

GRID_SEARCH_SPACE = {
    "assist_09": {
        "base": {},
        "search": {
            "learning_rate": [1e-4, 3e-4],
            "lambda_sparse": [0.05, 0.1],
            "dropout": [0.1, 0.2],
            "ablate_concept_graph": [False],
        },
    },
    "assist_17": {
        "base": {},
        "search": {
            "learning_rate": [1e-4, 3e-4],
        },
    },
    "junyi": {
        "base": {},
        "search": {
            "learning_rate": [3e-4, 1e-3],
            "lambda_sparse": [0.05, 0.1],
            "dropout": [0.1, 0.2],
            "ablate_concept_graph": [False],
        },
    },
}

GRID_SEARCH_SPACE["cdbd_lsat"] = {
    "base": {},
    "search": {
        "learning_rate": [1e-3],
        "dropout": [0.1],
        "ablate_concept_graph": [False],
    },
}

GRID_SEARCH_SPACE["cdbd_a0910"] = {
    "base": {},
    "search": {
        "learning_rate": [1e-4, 3e-4],
        "dropout": [0.1, 0.2],
        "ablate_concept_graph": [False],
    },
}


def collect_explicit_arg_dests(argv, parser=None):
    """
    Collect argparse dest names that were explicitly provided on CLI.

    This is needed because a user may intentionally pass a value that equals
    the argparse default, for example `--lambda_alpha 0.0`. In that case we
    must still treat the field as explicitly overridden and avoid replacing it
    with dataset defaults.
    """
    if parser is None:
        return set()

    option_to_dest = {}
    for action in getattr(parser, "_actions", []):
        for opt in getattr(action, "option_strings", []):
            option_to_dest[opt] = action.dest

    explicit = set()
    for token in argv or []:
        if not isinstance(token, str):
            continue
        if token == "--":
            break
        if not token.startswith("-"):
            continue
        opt = token.split("=", 1)[0]
        dest = option_to_dest.get(opt)
        if dest:
            explicit.add(dest)
    return explicit


def apply_dataset_defaults(args, parser=None, explicit_dests=None):
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

    explicit_dests = set(explicit_dests or ())
    defaults = DATASET_DEFAULTS[dataset_name]
    for key, value in defaults.items():
        if key in explicit_dests:
            continue
        if parser is not None and parser.get_default(key) is not None:
            if getattr(args, key, None) == parser.get_default(key):
                setattr(args, key, value)
        else:
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
    return args
