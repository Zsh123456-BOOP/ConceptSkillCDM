import json
import logging
import os
import sys
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _materialize_sparse_personal_dense(details: dict) -> torch.Tensor:
    personal = details.get("personal_matrices")
    if personal is not None:
        return personal.detach()

    relation_used = details.get("relation_used")
    _assert(isinstance(relation_used, dict), "缺少 sparse relation spec，无法重建 personal graph。")

    global_A = relation_used["global_matrices"].detach()
    active_row_index = relation_used["active_row_index"].detach()
    active_row_valid_mask = relation_used["active_row_valid_mask"].detach().bool()
    support_col_index = relation_used["support_col_index"].detach()
    support_valid_mask = relation_used["support_valid_mask"].detach().bool()
    posterior_prob = relation_used["posterior_prob"].detach()

    B = active_row_index.size(0)
    H, C, _ = global_A.shape
    dense = global_A.unsqueeze(0).expand(B, -1, -1, -1).clone()
    for b in range(B):
        for r in range(active_row_index.size(1)):
            if not bool(active_row_valid_mask[b, r]):
                continue
            row = int(active_row_index[b, r].item())
            for h in range(H):
                dense[b, h, row] = 0.0
                valid = support_valid_mask[b, h, r]
                cols = support_col_index[b, h, r][valid]
                probs = posterior_prob[b, h, r][valid]
                dense[b, h, row, cols] = probs
    return dense


def _materialize_relation_used_dense(details: dict) -> torch.Tensor:
    relation_used = details.get("relation_used")
    if not isinstance(relation_used, dict):
        return relation_used.detach()

    global_A = relation_used["global_matrices"].detach()
    active_row_index = relation_used["active_row_index"].detach()
    active_row_valid_mask = relation_used["active_row_valid_mask"].detach().bool()
    support_col_index = relation_used["support_col_index"].detach()
    support_valid_mask = relation_used["support_valid_mask"].detach().bool()
    global_support_prob = relation_used["global_support_prob"].detach()
    posterior_prob = relation_used["posterior_prob"].detach()
    gate_alpha = relation_used["gate_alpha"].detach()

    B = active_row_index.size(0)
    H, C, _ = global_A.shape
    dense = global_A.unsqueeze(0).expand(B, -1, -1, -1).clone()
    for b in range(B):
        for r in range(active_row_index.size(1)):
            if not bool(active_row_valid_mask[b, r]):
                continue
            row = int(active_row_index[b, r].item())
            for h in range(H):
                mixed_row = global_A[h, row].clone()
                valid = support_valid_mask[b, h, r]
                cols = support_col_index[b, h, r][valid]
                post = posterior_prob[b, h, r][valid]
                glob = global_support_prob[b, h, r][valid]
                mixed_row[cols] = glob + gate_alpha[b, h] * (post - glob)
                dense[b, h, row] = mixed_row
    return dense


def _check_no_a_keeps_gnn_layers() -> None:
    from src.trainer import _hard_ablation_effective_hparams

    eff = _hard_ablation_effective_hparams(
        use_concept_graph=False,
        num_gnn_layers=2,
    )
    _assert(
        eff == 2,
        "no_A 只应关闭全局图学习 A，不应把 knowledge_encoder 的 GNN 层数直接清零。",
    )


def _check_best_configs_enable_e_rescue_knobs() -> None:
    from best_configs import BEST_CFG

    required_positive = (
        "personal_delta_scale",
        "personal_warmup_epochs",
        "lambda_alpha_min",
        "alpha_min_target",
        "graph_propagation_alpha",
        "graph_query_readout_scale",
        "graph_query_readout_2hop_scale",
        "personal_alpha_temperature",
        "personal_alpha_budget",
        "personal_query_row_budget",
        "personal_neighbor_row_budget",
        "personal_query_correction_scale",
        "lambda_personal_kl",
        "lambda_personal_query_residual",
        "personal_state_lr_mult",
        "personal_id_lr_mult",
    )
    for dataset in ("assist_09", "junyi"):
        cfg = BEST_CFG[dataset]
        for key in required_positive:
            _assert(key in cfg, f"{dataset} 缺少 E-rescue 配置项: {key}")
            _assert(float(cfg[key]) > 0.0, f"{dataset} 的 {key} 必须为正值，当前={cfg[key]}")
        _assert(
            float(cfg["personal_delta_scale"]) >= 2.0,
            f"{dataset} 的 personal_delta_scale 过小，无法有效放大 E 扰动。",
        )

    assist_cfg = BEST_CFG["assist_09"]
    _assert(
        float(assist_cfg["lambda_alpha"]) == 0.0,
        "assist_09 不应继续奖励 alpha 方差；当前应关闭 alpha_var 奖励，只保留 anti-collapse 约束。",
    )
    _assert(
        float(assist_cfg["lambda_alpha_min"]) > 0.0,
        "assist_09 仍需保留 alpha anti-collapse 约束，避免 E 直接塌掉。",
    )
    for dataset in ("assist_09", "junyi"):
        cfg = BEST_CFG[dataset]
        _assert(
            bool(cfg["share_concept_embeddings"]) is True,
            f"{dataset} 应开启 share_concept_embeddings，避免 A 的图学习和知识编码继续分裂成两套概念空间。",
        )
        _assert(
            "personal_disable_direct_bias" not in cfg and "personal_direct_bias_scale" not in cfg,
            f"{dataset} 不应再保留已经失效的 direct bias 配置开关；否则实验表会继续记录假结构。",
        )
        _assert(
            bool(cfg["personal_disable_student_global_context"]) is True,
            f"{dataset} 必须显式禁用 raw student_global context shortcut。",
        )
        _assert(
            "personal_reg_warmup_epochs" in cfg,
            f"{dataset} 必须显式声明 personal_reg_warmup_epochs，不能再靠隐式回退。",
        )
        _assert(
            float(cfg["personal_alpha_bias_scale"]) == 0.0,
            f"{dataset} 默认实验不应继续依赖 student alpha bias shortcut，当前={cfg['personal_alpha_bias_scale']}",
        )
        _assert(
            int(cfg["personal_local_hops"]) >= 1,
            f"{dataset} 应显式开启基于题目局部子图的 E，当前 personal_local_hops={cfg['personal_local_hops']}",
        )
        _assert(
            bool(cfg["personal_include_neighbor_rows"]) is False,
            f"{dataset} 当前默认实验应切到 query rows only，neighbor rows 只保留为可选消融。",
        )
        _assert(
            bool(cfg["personal_support_only"]) is True,
            f"{dataset} 应显式启用 support-preserving E，避免 E 再退化为 dense personal graph。",
        )
        _assert(
            "graph_query_writeback_scale" not in cfg and "graph_readout_1hop_scale" not in cfg,
            f"{dataset} 的 best config 不应继续存旧 alias 字段；内部应只保留 graph_query_readout_scale。",
        )
        _assert(
            "graph_query_writeback_2hop_scale" not in cfg and "graph_readout_2hop_scale" not in cfg,
            f"{dataset} 的 best config 不应继续存旧 2-hop alias 字段；内部应只保留 graph_query_readout_2hop_scale。",
        )


def _check_dataset_defaults_respect_explicit_zero_overrides() -> None:
    from main import parse_args as build_parser
    from src.config import apply_dataset_defaults, collect_explicit_arg_dests

    argv = [
        "--dataset_name",
        "assist_09",
        "--lambda_alpha",
        "0.0",
        "--lambda_sparse_personal",
        "0.0",
        "--graph_identity_residual",
        "0.0",
        "--graph_query_readout_scale",
        "0.0",
        "--personal_alpha_bias_scale",
        "0.0",
        "--personal_alpha_budget",
        "0.0",
        "--no-use_personal_graph",
        "--no-share_concept_embeddings",
        "--no-personal_disable_student_global_context",
        "--no-personal_support_only",
    ]
    parser = build_parser()
    args = parser.parse_args(argv)
    explicit_dests = collect_explicit_arg_dests(argv, parser)
    args = apply_dataset_defaults(args, parser, explicit_dests=explicit_dests)

    _assert(
        float(args.lambda_alpha) == 0.0,
        "显式传入的 --lambda_alpha 0.0 不应再被数据集默认值覆盖。",
    )
    _assert(
        float(args.lambda_sparse_personal) == 0.0,
        "显式传入的 --lambda_sparse_personal 0.0 不应再被数据集默认值覆盖。",
    )
    _assert(
        float(args.graph_identity_residual) == 0.0,
        "显式传入的 --graph_identity_residual 0.0 不应再被数据集默认值覆盖。",
    )
    _assert(
        float(args.graph_query_readout_scale) == 0.0,
        "显式传入的 --graph_query_readout_scale 0.0 不应再被数据集默认值覆盖。",
    )
    _assert(
        float(args.graph_query_readout_scale) == 0.0,
        "canonical graph_query_readout_scale 不应被数据集默认值覆盖。",
    )
    _assert(
        bool(args.use_personal_graph) is False,
        "显式传入的 --no-use_personal_graph 不应再被数据集默认值回写成 True。",
    )
    _assert(
        bool(args.share_concept_embeddings) is False,
        "显式传入的 --no-share_concept_embeddings 不应被数据集默认值回写成 True。",
    )
    _assert(
        float(args.personal_alpha_bias_scale) == 0.0,
        "显式传入的 personal_alpha_bias_scale=0.0 不应被数据集默认值回写成非零值。",
    )
    _assert(
        float(args.personal_alpha_budget) == 0.0,
        "显式传入的 personal_alpha_budget=0.0 不应被数据集默认值回写成非零值。",
    )
    _assert(
        bool(args.personal_disable_student_global_context) is False,
        "显式传入的 --no-personal_disable_student_global_context 不应被数据集默认值回写成 True。",
    )
    _assert(
        bool(args.personal_support_only) is False,
        "显式传入的 --no-personal_support_only 不应被数据集默认值回写成 True。",
    )
    _assert(
        float(args.lambda_sparse) == 0.3,
        "未显式覆盖的字段仍应继续继承数据集默认值，避免把默认机制整体打坏。",
    )
    _assert(
        not hasattr(args, "personal_disable_direct_bias") and not hasattr(args, "personal_direct_bias_scale"),
        "direct bias 既然已经从 E 结构里删除，parser/defaults 也不应继续暴露这些假开关。",
    )


def _check_no_a_personal_graph_is_not_identity_locked() -> None:
    from src.model import CognitiveDiagnosisModel

    torch.manual_seed(0)
    q_matrix = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    model = CognitiveDiagnosisModel(
        num_students=5,
        num_exercises=4,
        num_concepts=3,
        q_matrix=q_matrix,
        knowledge_dim=8,
        num_relation_heads=2,
        num_gnn_layers=1,
        dropout=0.0,
        use_concept_graph=False,
        use_personal_graph=True,
        personal_rank=4,
        personal_max_alpha=0.4,
        personal_delta_scale=6.0,
        personal_warmup_epochs=0,
        personal_student_dim=8,
    )
    model.eval()

    with torch.no_grad():
        _, details = model(
            student_ids=torch.tensor([0, 1], dtype=torch.long),
            exercise_ids=torch.tensor([0, 2], dtype=torch.long),
            return_details=True,
            return_logits=True,
        )

    ident = model.identity_relations.unsqueeze(0)
    personal_dense = _materialize_sparse_personal_dense(details)
    matrix_delta = float((personal_dense - ident).abs().mean())
    _assert(
        float(details["personal_delta_pre_softmax_norm"]) > 0.05,
        "测试前提失败：personal delta 本身应明显非零。",
    )
    _assert(
        matrix_delta > 0.05,
        f"no_A 下的 E 不应被 identity prior 钉死；当前 personal matrix delta 只有 {matrix_delta:.6f}。",
    )


def _check_personal_branch_is_state_primary_with_small_id_adapter() -> None:
    from src.model import CognitiveDiagnosisModel

    q_matrix = torch.eye(3, dtype=torch.float32)
    model = CognitiveDiagnosisModel(
        num_students=5,
        num_exercises=3,
        num_concepts=3,
        q_matrix=q_matrix,
        knowledge_dim=8,
        num_relation_heads=2,
        num_gnn_layers=1,
        dropout=0.0,
        use_concept_graph=True,
        use_personal_graph=True,
        personal_rank=4,
        personal_max_alpha=0.4,
        personal_delta_scale=6.0,
        personal_warmup_epochs=0,
        personal_student_dim=8,
    )
    sm = model.structure_module

    _assert(
        hasattr(sm, "personal_gate_from_state"),
        "E 分支缺少从 A 编码 student_repr 到 gate 的主路投影。",
    )
    _assert(
        hasattr(sm, "personal_generator_from_state"),
        "E 分支缺少从 A 编码 student_repr 到 generator 的主路投影。",
    )
    _assert(
        hasattr(sm, "personal_gate_id_logit"),
        "E 分支缺少 gate 的 id-adapter 缩放参数。",
    )
    _assert(
        hasattr(sm, "personal_generator_id_logit"),
        "E 分支缺少 generator 的 id-adapter 缩放参数。",
    )

    gate_id_scale = float(torch.sigmoid(sm.personal_gate_id_logit.detach()).item())
    gen_id_scale = float(torch.sigmoid(sm.personal_generator_id_logit.detach()).item())
    _assert(
        gate_id_scale < 0.25,
        f"gate 的 id-adapter 初始占比应较小，当前={gate_id_scale:.4f}",
    )
    _assert(
        gen_id_scale < 0.25,
        f"generator 的 id-adapter 初始占比应较小，当前={gen_id_scale:.4f}",
    )


def _check_adaptive_gate_is_state_primary() -> None:
    from src.model import AdaptiveGate

    torch.manual_seed(0)
    gate = AdaptiveGate(
        student_dim=8,
        context_dim=12,
        num_heads=2,
        max_alpha=0.4,
        hidden_dim=16,
    )
    gate.eval()

    state_a = torch.tensor([[0.0, 0.5, -0.5, 1.0, -1.0, 0.25, -0.25, 0.75]], dtype=torch.float32)
    state_b = torch.tensor([[1.5, -1.0, 0.75, -0.25, 0.5, -0.5, 1.0, -1.5]], dtype=torch.float32)
    context_a = torch.tensor([[0.0, 0.2, -0.1, 0.4, -0.3, 0.6, -0.5, 0.8, -0.7, 1.0, -0.9, 0.3]], dtype=torch.float32)
    context_b = torch.tensor([[1.0, -0.8, 0.6, -0.4, 0.2, -0.1, 0.9, -0.7, 0.5, -0.3, 0.4, -0.2]], dtype=torch.float32)
    id_a = torch.tensor([[0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4]], dtype=torch.float32)
    id_b = torch.tensor([[0.4, -0.4, 0.3, -0.3, 0.2, -0.2, 0.1, -0.1]], dtype=torch.float32)
    bias = torch.zeros(1, 2)

    with torch.no_grad():
        alpha_base, _, diag_base = gate(
            state_a,
            context_a,
            id_a,
            return_diagnostics=True,
        )
        alpha_state, _, diag_state = gate(
            state_b,
            context_b,
            id_a,
            return_diagnostics=True,
        )
        alpha_id, _, diag_id = gate(
            state_a,
            context_a,
            id_b,
            return_diagnostics=True,
        )

    state_delta = float((alpha_state - alpha_base).abs().mean().item())
    id_delta = float((alpha_id - alpha_base).abs().mean().item())
    _assert(
        state_delta > id_delta * 2.0,
        f"默认安全 gate 下，state/context 改动应显著强于 id 改动；当前 state_delta={state_delta:.6f}, id_delta={id_delta:.6f}",
    )
    _assert(
        float(diag_state["state_path_absmean"].item()) > float(diag_id["id_path_absmean"].item()),
        "gate 的主路径幅度应由 state/context 主导，而不是 id adapter 主导。",
    )
    _assert(
        float(diag_base["effective_id_scale"].item()) < 0.02,
        f"id adapter 的有效缩放应保持在 tiny adapter 量级，当前={float(diag_base['effective_id_scale'].item()):.6f}",
    )


def _check_gate_saturation_regression() -> None:
    from src.model import AdaptiveGate

    torch.manual_seed(0)
    gate = AdaptiveGate(
        student_dim=8,
        context_dim=12,
        num_heads=2,
        max_alpha=0.28,
        hidden_dim=16,
        alpha_temperature=2.0,
        alpha_budget=0.10,
        alpha_base_init=0.05,
    )
    gate.eval()

    huge_state = torch.full((4, 8), 25.0, dtype=torch.float32)
    huge_context = torch.full((4, 12), -18.0, dtype=torch.float32)
    student_id = torch.randn(4, 8)

    with torch.no_grad():
        _, _, diag = gate(
            huge_state,
            huge_context,
            student_id,
            warmup_scale=1.0,
            return_diagnostics=True,
        )

    _assert(
        float(diag["alpha_delta_absmean"].item()) > 1e-4,
        "大 state logit 下 alpha_delta_absmean 仍应非零，否则 gate 的 state 路径被抹平了。",
    )
    _assert(
        float(diag["alpha_saturation_ratio"].item()) < 0.95,
        f"新 gate 不应在大 state logit 下几乎全部打满边界，当前 saturation_ratio={float(diag['alpha_saturation_ratio'].item()):.6f}",
    )


def _check_assist_like_harmfulness_proxy_warning() -> None:
    from src.trainer import _collect_diag_warning_tags

    diag = {
        "alpha_std": 0.001,
        "alpha_saturation_ratio": 0.92,
        "alpha_delta_absmean": 0.05,
        "personal_matrix_delta": 0.0012,
        "personal_matrix_student_std": 0.0002,
        "personal_query_row_std": 0.0001,
        "readout_query_delta": 0.21,
        "query_row_graph_delta": 0.18,
        "alpha_head_std": 0.0005,
        "personal_delta_student_std": 0.05,
        "alpha_id_path_absmean": 0.0,
        "alpha_state_path_absmean": 1.0,
    }
    tags = set(_collect_diag_warning_tags(diag))
    for expected in ("gate-saturated", "personalization-flat", "query-readout-overamplified"):
        _assert(expected in tags, f"assist-like harmfulness proxy 应触发 {expected} warning。")


def _check_personal_graph_is_support_preserving_and_local() -> None:
    from src.model import CognitiveDiagnosisModel

    torch.manual_seed(0)
    q_matrix = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    model = CognitiveDiagnosisModel(
        num_students=4,
        num_exercises=2,
        num_concepts=4,
        q_matrix=q_matrix,
        knowledge_dim=8,
        num_relation_heads=2,
        num_gnn_layers=1,
        dropout=0.0,
        use_concept_graph=True,
        graph_topk=2,
        use_personal_graph=True,
        share_concept_embeddings=True,
        graph_identity_residual=0.0,
        personal_rank=4,
        personal_max_alpha=0.4,
        personal_delta_scale=4.0,
        personal_warmup_epochs=0,
        personal_reg_warmup_epochs=0,
        personal_student_dim=8,
        personal_disable_student_global_context=True,
        personal_local_hops=1,
        personal_support_only=True,
    )
    model.eval()

    with torch.no_grad():
        _, details = model(
            student_ids=torch.tensor([0], dtype=torch.long),
            exercise_ids=torch.tensor([0], dtype=torch.long),
            return_details=True,
            return_logits=True,
        )

    global_A = details["relation_matrices"].detach()
    personal_A = _materialize_sparse_personal_dense(details)[0]
    relation_used = _materialize_relation_used_dense(details)[0]
    local_row_mask = details["local_row_mask"].detach()[0].bool()

    support_mask = global_A > 0
    unsupported_mass = float(personal_A.masked_select(~support_mask).abs().max().item())
    _assert(
        unsupported_mass < 1e-6,
        f"E 应保持在 A 的 support 上重加权，当前 unsupported mass={unsupported_mass:.6e}",
    )

    local_rows = local_row_mask.unsqueeze(0).unsqueeze(-1).expand_as(relation_used)
    off_local_delta = float((relation_used - global_A).masked_select(~local_rows).abs().max().item())
    _assert(
        off_local_delta < 1e-6,
        f"E 不应改动题目局部子图之外的行，当前 off-local delta={off_local_delta:.6e}",
    )
    local_delta = float((relation_used - global_A).masked_select(local_rows).abs().mean().item())
    _assert(
        local_delta > 1e-5,
        "局部 posterior mixing 应在题目相关子图上产生非零扰动，而不是完全退化回全局图。",
    )


def _check_personal_graph_uses_sparse_runtime_spec() -> None:
    from src.model import CognitiveDiagnosisModel

    torch.manual_seed(0)
    q_matrix = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    model = CognitiveDiagnosisModel(
        num_students=4,
        num_exercises=2,
        num_concepts=4,
        q_matrix=q_matrix,
        knowledge_dim=8,
        num_relation_heads=2,
        num_gnn_layers=1,
        dropout=0.0,
        use_concept_graph=True,
        graph_topk=2,
        use_personal_graph=True,
        share_concept_embeddings=True,
        graph_identity_residual=0.05,
        personal_rank=4,
        personal_max_alpha=0.4,
        personal_delta_scale=4.0,
        personal_warmup_epochs=0,
        personal_reg_warmup_epochs=0,
        personal_student_dim=8,
        personal_disable_student_global_context=True,
        personal_local_hops=1,
        personal_support_only=True,
    )
    model.eval()

    with torch.no_grad():
        _, details = model(
            student_ids=torch.tensor([0], dtype=torch.long),
            exercise_ids=torch.tensor([0], dtype=torch.long),
            return_details=True,
            return_logits=True,
        )

    relation_used = details["relation_used"]
    _assert(
        isinstance(relation_used, dict),
        "E 开启时主训练路径应改用 sparse/local relation spec，而不是继续传 dense (B,H,C,C) 矩阵。",
    )
    for key in (
        "global_matrices",
        "active_row_index",
        "active_row_valid_mask",
        "support_col_index",
        "support_valid_mask",
        "global_support_prob",
        "posterior_prob",
        "gate_alpha",
    ):
        _assert(key in relation_used, f"sparse relation spec 缺少关键字段: {key}")
    _assert(
        details.get("personal_matrices") is None,
        "训练/诊断主路径不应再默认物化 dense personal_matrices；否则 E 仍会在大数据集上退回 O(B*H*C*C)。",
    )


def _check_query_readout_injects_graph_signal() -> None:
    from src.model import CognitiveDiagnosisModel

    torch.manual_seed(0)
    q_matrix = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    model_off = CognitiveDiagnosisModel(
        num_students=3,
        num_exercises=2,
        num_concepts=3,
        q_matrix=q_matrix,
        knowledge_dim=8,
        num_relation_heads=2,
        num_gnn_layers=1,
        dropout=0.0,
        use_concept_graph=True,
        use_personal_graph=False,
        graph_query_readout_scale=0.0,
        graph_query_readout_2hop_scale=0.0,
    )
    torch.manual_seed(0)
    model_on = CognitiveDiagnosisModel(
        num_students=3,
        num_exercises=2,
        num_concepts=3,
        q_matrix=q_matrix,
        knowledge_dim=8,
        num_relation_heads=2,
        num_gnn_layers=1,
        dropout=0.0,
        use_concept_graph=True,
        use_personal_graph=False,
        graph_query_readout_scale=0.5,
        graph_query_readout_2hop_scale=0.15,
    )
    model_off.eval()
    model_on.eval()

    with torch.no_grad():
        logits_off, details_off = model_off(
            student_ids=torch.tensor([0], dtype=torch.long),
            exercise_ids=torch.tensor([0], dtype=torch.long),
            return_details=True,
            return_logits=True,
        )
        logits_on, details_on = model_on(
            student_ids=torch.tensor([0], dtype=torch.long),
            exercise_ids=torch.tensor([0], dtype=torch.long),
            return_details=True,
            return_logits=True,
        )

    _assert(
        float(details_on["readout_query_delta"].item()) > 1e-5,
        "A 的 query-local readout 应对最终送入固定预测头的状态产生非零影响。",
    )
    _assert(
        float(details_on["query_row_global_readout_delta"].item()) > 1e-5,
        "A 的 global query readout 应直接改动 queried concept rows，而不是只在全局状态里稀释。",
    )
    _assert(
        float((logits_on - logits_off).abs().mean().item()) > 1e-6,
        "打开 query-row writeback 后，最终 logits 应出现可测变化。",
    )


def _check_e_query_only_active_rows() -> None:
    from src.model import CognitiveDiagnosisModel

    torch.manual_seed(0)
    q_matrix = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    model = CognitiveDiagnosisModel(
        num_students=3,
        num_exercises=1,
        num_concepts=4,
        q_matrix=q_matrix,
        knowledge_dim=8,
        num_relation_heads=2,
        num_gnn_layers=1,
        dropout=0.0,
        use_concept_graph=True,
        use_personal_graph=True,
        personal_include_neighbor_rows=False,
        personal_support_only=True,
    )
    model.eval()

    with torch.no_grad():
        _, details = model(
            student_ids=torch.tensor([0], dtype=torch.long),
            exercise_ids=torch.tensor([0], dtype=torch.long),
            return_details=True,
            return_logits=True,
        )

    active_row_index = details["active_row_index"][0]
    active_row_valid_mask = details["active_row_valid_mask"][0].bool()
    active_rows = active_row_index[active_row_valid_mask].tolist()
    _assert(active_rows == [0], f"default E 应只激活 query rows，当前 active_rows={active_rows}")


def _check_e_readout_only_does_not_change_backbone() -> None:
    from src.model import CognitiveDiagnosisModel

    torch.manual_seed(0)
    q_matrix = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    model = CognitiveDiagnosisModel(
        num_students=3,
        num_exercises=1,
        num_concepts=3,
        q_matrix=q_matrix,
        knowledge_dim=8,
        num_relation_heads=2,
        num_gnn_layers=1,
        dropout=0.0,
        use_concept_graph=True,
        use_personal_graph=True,
        personal_include_neighbor_rows=False,
    )
    model.eval()

    with torch.no_grad():
        _, details = model(
            student_ids=torch.tensor([0], dtype=torch.long),
            exercise_ids=torch.tensor([0], dtype=torch.long),
            return_details=True,
            return_logits=True,
        )

    _assert(
        float(details["knowledge_state_personal_delta"].item()) == 0.0,
        "E 改成 query-time correction 后，不应再改 backbone knowledge_state。",
    )


def _check_posterior_equal_global_gives_zero_personal_query_correction() -> None:
    from src.model import CognitiveDiagnosisModel

    torch.manual_seed(0)
    q_matrix = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    model = CognitiveDiagnosisModel(
        num_students=3,
        num_exercises=1,
        num_concepts=3,
        q_matrix=q_matrix,
        knowledge_dim=8,
        num_relation_heads=2,
        num_gnn_layers=1,
        dropout=0.0,
        use_concept_graph=True,
        use_personal_graph=True,
        personal_include_neighbor_rows=False,
    )
    model.eval()

    with torch.no_grad():
        _, details = model(
            student_ids=torch.tensor([0], dtype=torch.long),
            exercise_ids=torch.tensor([0], dtype=torch.long),
            return_details=True,
            return_logits=True,
        )
        relation_spec = details["personal_relation_spec"]
        relation_spec = dict(relation_spec)
        relation_spec["posterior_prob"] = relation_spec["global_support_prob"].clone()
        correction, msg_delta, post_abs, post_kl, _ = model._build_personal_query_correction(
            details["knowledge_state"],
            relation_spec,
            details["q_vector"],
        )

    _assert(float(correction.abs().max().item()) < 1e-8, "当 posterior==global 时，personal query correction 应为 0。")
    _assert(float(msg_delta.item()) < 1e-8, "当 posterior==global 时，query_row_personal_message_delta 应为 0。")
    _assert(float(post_abs.item()) < 1e-8 and float(post_kl.item()) < 1e-8, "posterior==global 时 raw posterior diagnostics 也应归零。")


def _check_summary_prefers_matched_no_e() -> None:
    from run_abce_ablation import write_summary

    rows = [
        {"dataset": "junyi", "seed": "42", "profile": "best", "ablation": "full", "status": "ok", "ablation_valid": "True", "test_auc": "0.8281"},
        {"dataset": "junyi", "seed": "42", "profile": "best", "ablation": "no_A", "status": "ok", "ablation_valid": "True", "test_auc": "0.8270"},
        {"dataset": "junyi", "seed": "42", "profile": "best", "ablation": "no_E", "status": "ok", "ablation_valid": "True", "test_auc": "0.8290"},
        {"dataset": "junyi", "seed": "42", "profile": "best", "ablation": "no_E_bs64", "status": "ok", "ablation_valid": "True", "test_auc": "0.8278"},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        summary_rows, _ = write_summary(Path(tmpdir) / "summary.csv", Path(tmpdir) / "mean.csv", rows)

    _assert(len(summary_rows) == 1, "matched no_E summary smoke 应生成单行 summary。")
    row = summary_rows[0]
    _assert(abs(float(row["no_E_matched_auc"]) - 0.8278) < 1e-8, "summary 应优先写入 no_E_bs64 作为 matched 对照。")
    _assert(abs(float(row["delta_E_full_minus_noE_matched"]) - (0.8281 - 0.8278)) < 1e-8, "matched delta_E 计算错误。")


def _check_module_activity_brief_uses_live_risk() -> None:
    from src.module_activity import format_activity_brief

    brief = format_activity_brief(
        {
            "graph_enabled": True,
            "graph_active": True,
            "personal_graph_enabled": True,
            "personal_graph_active": True,
            "personal_graph_risk": True,
        }
    )
    _assert("Graph[LIVE]" in brief, "module_activity 简报不应继续输出 Graph[OK]。")
    _assert("Personal[RISK]" in brief, "module_activity 简报应能区分 active 但 risky 的 E。")


def _check_split_hygiene_uses_train_only_maps_and_q_matrix() -> None:
    from src.dataset import create_dataloaders

    train_df = pd.DataFrame(
        [
            {"stu_id": 1, "exer_id": 10, "cpt_seq": "100", "label": 1},
            {"stu_id": 2, "exer_id": 11, "cpt_seq": "101", "label": 0},
        ]
    )
    val_df = pd.DataFrame(
        [
            {"stu_id": 1, "exer_id": 10, "cpt_seq": "100", "label": 1},
            {"stu_id": 999, "exer_id": 10, "cpt_seq": "100", "label": 0},
            {"stu_id": 1, "exer_id": 999, "cpt_seq": "999", "label": 0},
        ]
    )
    test_df = pd.DataFrame(
        [
            {"stu_id": 2, "exer_id": 11, "cpt_seq": "101", "label": 0},
            {"stu_id": 2, "exer_id": 12, "cpt_seq": "102", "label": 1},
        ]
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        train_path = os.path.join(tmpdir, "train.csv")
        val_path = os.path.join(tmpdir, "val.csv")
        test_path = os.path.join(tmpdir, "test.csv")
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        train_loader, val_loader, test_loader, info_dict = create_dataloaders(
            train_file=train_path,
            val_file=val_path,
            test_file=test_path,
            batch_size=2,
            num_workers=0,
            min_stu_interactions=0,
            min_exer_interactions=0,
            min_poison_count=0,
        )

    _assert(
        set(info_dict["stu_id_map"].keys()) == {1, 2},
        f"student map 应严格基于 train 构建，当前={sorted(info_dict['stu_id_map'].keys())}",
    )
    _assert(
        set(info_dict["exer_id_map"].keys()) == {10, 11},
        f"exercise map 应严格基于 train 构建，当前={sorted(info_dict['exer_id_map'].keys())}",
    )
    _assert(
        set(info_dict["cpt_id_map"].keys()) == {100, 101},
        f"concept map 应严格基于 train 构建，当前={sorted(info_dict['cpt_id_map'].keys())}",
    )
    _assert(
        tuple(info_dict["q_matrix"].shape) == (2, 2),
        f"Q 矩阵维度应只覆盖 train 中 seen item/concept，当前={tuple(info_dict['q_matrix'].shape)}",
    )
    _assert(
        len(train_loader.dataset) == 2 and len(val_loader.dataset) == 1 and len(test_loader.dataset) == 1,
        "val/test 中未在 train 出现的 student/item 应在建 loader 前被过滤。",
    )


def _check_runtime_ablation_guardrails_cover_no_a_and_no_e() -> None:
    from src.model import CognitiveDiagnosisModel
    from src.trainer import _collect_runtime_ablation_facts, _log_and_assert_ablation_consistency

    logger = logging.getLogger("smoke_ae_runtime")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())

    q_matrix = torch.eye(3, dtype=torch.float32)

    no_a_model = CognitiveDiagnosisModel(
        num_students=4,
        num_exercises=3,
        num_concepts=3,
        q_matrix=q_matrix,
        knowledge_dim=8,
        num_relation_heads=2,
        num_gnn_layers=1,
        dropout=0.0,
        use_concept_graph=False,
        use_personal_graph=True,
        personal_rank=4,
        personal_max_alpha=0.4,
        personal_delta_scale=6.0,
        personal_warmup_epochs=0,
        personal_student_dim=8,
    )
    facts_no_a = _collect_runtime_ablation_facts(no_a_model)
    _assert(facts_no_a["enable_module1"] is True, "no_A 不应顺带关闭整个模块1。")
    _assert(facts_no_a["use_concept_graph"] is False, "no_A runtime 必须显示 A 已关闭。")
    _assert(facts_no_a["has_relation_learning"] is False, "no_A 时 relation_learning 不应物理存在。")
    _assert(facts_no_a["use_personal_graph"] is True, "no_A 不应顺带关闭 E。")
    _assert(facts_no_a["has_adaptive_gate"] is True, "no_A 时 E 的 adaptive_gate 应仍然存在。")
    _assert(facts_no_a["has_personal_generator"] is True, "no_A 时 E 的 personal_generator 应仍然存在。")
    _log_and_assert_ablation_consistency(
        model=no_a_model,
        logger=logger,
        context="[smoke][no_A]",
        ablate_module1=False,
        expect_use_concept_graph=False,
        expect_use_personal_graph=True,
    )

    no_e_model = CognitiveDiagnosisModel(
        num_students=4,
        num_exercises=3,
        num_concepts=3,
        q_matrix=q_matrix,
        knowledge_dim=8,
        num_relation_heads=2,
        num_gnn_layers=1,
        dropout=0.0,
        use_concept_graph=True,
        use_personal_graph=False,
        personal_rank=4,
        personal_max_alpha=0.4,
        personal_delta_scale=6.0,
        personal_warmup_epochs=0,
        personal_student_dim=8,
    )
    facts_no_e = _collect_runtime_ablation_facts(no_e_model)
    _assert(facts_no_e["enable_module1"] is True, "no_E 不应顺带关闭整个模块1。")
    _assert(facts_no_e["use_concept_graph"] is True, "no_E 不应顺带关闭 A。")
    _assert(facts_no_e["has_relation_learning"] is True, "no_E 时 relation_learning 应仍然存在。")
    _assert(facts_no_e["use_personal_graph"] is False, "no_E runtime 必须显示 E 已关闭。")
    _assert(facts_no_e["has_adaptive_gate"] is False, "no_E 时 adaptive_gate 不应存在。")
    _assert(facts_no_e["has_personal_generator"] is False, "no_E 时 personal_generator 不应存在。")
    _log_and_assert_ablation_consistency(
        model=no_e_model,
        logger=logger,
        context="[smoke][no_E]",
        ablate_module1=False,
        expect_use_concept_graph=True,
        expect_use_personal_graph=False,
    )


def _build_tiny_ae_model(**overrides):
    from src.model import CognitiveDiagnosisModel

    q_matrix = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    kwargs = dict(
        num_students=5,
        num_exercises=4,
        num_concepts=3,
        q_matrix=q_matrix,
        knowledge_dim=8,
        num_relation_heads=2,
        num_gnn_layers=1,
        dropout=0.0,
        use_concept_graph=True,
        use_personal_graph=True,
        personal_rank=4,
        personal_max_alpha=0.4,
        personal_delta_scale=6.0,
        personal_warmup_epochs=0,
        personal_reg_warmup_epochs=0,
        personal_student_dim=8,
        personal_alpha_temperature=2.0,
        personal_alpha_budget=0.10,
        personal_alpha_base_init=0.05,
        personal_alpha_bias_scale=0.02,
        personal_disable_student_global_context=True,
        personal_local_hops=1,
        personal_query_row_budget=1.0,
        personal_neighbor_row_budget=0.30,
        personal_support_only=True,
        personal_include_neighbor_rows=False,
        graph_query_readout_scale=0.40,
        graph_query_readout_2hop_scale=0.12,
        share_concept_embeddings=True,
    )
    kwargs.update(overrides)
    return CognitiveDiagnosisModel(**kwargs)


def _check_generator_state_adapter_is_live() -> None:
    from src.model import PersonalRelationGenerator

    torch.manual_seed(0)
    generator = PersonalRelationGenerator(
        student_dim=8,
        context_dim=12,
        knowledge_dim=6,
        num_concepts=4,
        num_heads=2,
        rank=3,
        hidden_dim=10,
    )
    generator.eval()

    student_state_a = torch.tensor([[0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9]], dtype=torch.float32)
    student_state_b = torch.tensor([[-0.9, 0.8, -0.7, 0.6, -0.5, 0.4, -0.3, 0.2]], dtype=torch.float32)
    student_id_a = torch.tensor([[0.1, 0.0, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4]], dtype=torch.float32)
    student_id_b = torch.tensor([[0.4, -0.3, 0.2, -0.1, 0.0, 0.1, -0.2, 0.3]], dtype=torch.float32)
    context = torch.tensor([[0.2, -0.1, 0.3, -0.2, 0.4, -0.3, 0.5, -0.4, 0.6, -0.5, 0.7, -0.6]], dtype=torch.float32)
    knowledge_state = torch.tensor(
        [[[0.2, -0.1, 0.0, 0.3, -0.2, 0.1],
          [-0.3, 0.4, -0.2, 0.1, 0.0, -0.1],
          [0.5, -0.4, 0.3, -0.2, 0.1, 0.0],
          [-0.1, 0.2, -0.3, 0.4, -0.5, 0.6]]],
        dtype=torch.float32,
    )

    with torch.no_grad():
        base_scores = generator(student_state_a, context, knowledge_state, student_id_embedding=student_id_a)
        state_scores = generator(student_state_b, context, knowledge_state, student_id_embedding=student_id_a)
        id_scores = generator(student_state_a, context, knowledge_state, student_id_embedding=student_id_b)

    state_delta = float((state_scores - base_scores).abs().max().item())
    id_delta = float((id_scores - base_scores).abs().mean().item())
    _assert(
        state_delta > 1e-6,
        "generator 的 student_state_embedding 改变后，输出应发生变化；否则 state adapter 仍是死路径。",
    )
    _assert(
        state_delta > id_delta,
        f"state adapter 应比 tiny id adapter 更有影响；当前 state_delta={state_delta:.6f}, id_delta={id_delta:.6f}",
    )


def _check_alpha_bias_scale_is_live() -> None:
    torch.manual_seed(0)
    model_zero = _build_tiny_ae_model(personal_alpha_bias_scale=0.0)
    torch.manual_seed(0)
    model_bias = _build_tiny_ae_model(personal_alpha_bias_scale=0.05)
    model_zero.eval()
    model_bias.eval()

    with torch.no_grad():
        _, details_zero = model_zero(
            student_ids=torch.tensor([0, 1], dtype=torch.long),
            exercise_ids=torch.tensor([0, 2], dtype=torch.long),
            return_details=True,
            return_logits=True,
        )
        _, details_bias = model_bias(
            student_ids=torch.tensor([0, 1], dtype=torch.long),
            exercise_ids=torch.tensor([0, 2], dtype=torch.long),
            return_details=True,
            return_logits=True,
        )

    _assert(
        details_bias.get("alpha_student_bias") is not None,
        "启用 personal_alpha_bias_scale 后，应返回真实的 alpha_student_bias 诊断张量。",
    )
    alpha_logit_delta = float((details_bias["alpha_logit"] - details_zero["alpha_logit"]).abs().max().item())
    bias_delta = float(details_bias["alpha_student_bias"].abs().max().item())
    _assert(
        bias_delta > 1e-6 and alpha_logit_delta > 1e-6,
        "调整 personal_alpha_bias_scale 后，alpha bias 路径和 alpha_logit 都应发生变化；否则该开关仍是 no-op。",
    )


def _check_alpha_bias_diagnostics_consistency() -> None:
    from src.trainer import _collect_debug_forward_stats

    model = _build_tiny_ae_model(personal_alpha_bias_scale=0.0)
    loader = DataLoader(
        TensorDataset(
            torch.tensor([0, 1], dtype=torch.long),
            torch.tensor([0, 2], dtype=torch.long),
            torch.tensor([1.0, 0.0], dtype=torch.float32),
        ),
        batch_size=2,
        shuffle=False,
    )
    model.eval()
    with torch.no_grad():
        _, details = model(
            student_ids=torch.tensor([0, 1], dtype=torch.long),
            exercise_ids=torch.tensor([0, 2], dtype=torch.long),
            return_details=True,
            return_logits=True,
        )

    _assert(
        details.get("alpha_student_bias") is None,
        "当 personal_alpha_bias_scale=0 时，details 不应伪造 alpha_student_bias。",
    )
    diag = _collect_debug_forward_stats(model, loader, torch.device("cpu"), max_batches=1)
    _assert(
        diag.get("alpha_bias_std") is None,
        "当 alpha student bias 不存在时，trainer 不应继续输出 alpha_bias_std。",
    )


def _check_summary_classification_regression() -> None:
    from run_abce_ablation import classify_delta

    _assert(
        classify_delta(-0.0029, threshold=0.003) == "non_positive",
        "delta_E=-0.0029 不应继续被标成 neutral。",
    )


def _check_config_hash_tracks_ae_structure_switches() -> None:
    from src.experiment_utils import _build_config_hash

    base = dict(
        dataset_name="assist_09",
        seed=42,
        model_variant="assist_09_abce_best_full",
        ablate_module1=False,
        learning_rate=3e-4,
        dropout=0.2,
        batch_size=128,
        lambda_sparse=0.8,
        lambda_sparse_personal=5e-4,
        lambda_alpha=0.0,
        prediction_l2_lambda=1e-5,
        graph_reg_warmup_epochs=4,
        graph_reg_cap_ratio=6.0,
        graph_propagation_alpha=0.2,
        graph_query_readout_scale=0.6,
        graph_query_readout_2hop_scale=0.2,
        use_concept_graph=True,
        use_personal_graph=True,
        personal_local_hops=1,
        personal_include_neighbor_rows=False,
        personal_query_row_budget=1.0,
        personal_neighbor_row_budget=0.35,
        personal_support_only=True,
        personal_query_correction_scale=0.10,
        share_concept_embeddings=True,
        personal_alpha_temperature=2.2,
        personal_alpha_budget=0.10,
        personal_alpha_base_init=0.05,
        personal_alpha_bias_scale=0.03,
        personal_reg_warmup_epochs=8,
        personal_disable_student_global_context=True,
        graph_identity_residual=0.1,
        personal_delta_scale=6.0,
        personal_warmup_epochs=8,
        lambda_personal_kl=0.02,
        lambda_personal_query_residual=0.05,
        personal_query_residual_margin=0.08,
        lambda_alpha_min=0.08,
        alpha_min_target=0.05,
        personal_state_lr_mult=1.0,
        personal_id_lr_mult=0.5,
    )
    hash_base = _build_config_hash(SimpleNamespace(**base))
    hash_share = _build_config_hash(SimpleNamespace(**{**base, "share_concept_embeddings": False}))
    hash_alpha_bias = _build_config_hash(SimpleNamespace(**{**base, "personal_alpha_bias_scale": 0.0}))
    hash_reg_warmup = _build_config_hash(SimpleNamespace(**{**base, "personal_reg_warmup_epochs": 4}))
    hash_query_readout = _build_config_hash(SimpleNamespace(**{**base, "graph_query_readout_scale": 0.4}))
    hash_query_residual = _build_config_hash(SimpleNamespace(**{**base, "lambda_personal_query_residual": 0.02}))

    _assert(hash_base != hash_share, "config hash 必须区分 share_concept_embeddings 的结构差异。")
    _assert(hash_base != hash_alpha_bias, "config hash 必须区分 personal_alpha_bias_scale 的结构差异。")
    _assert(hash_base != hash_reg_warmup, "config hash 必须区分 personal_reg_warmup_epochs 的差异。")
    _assert(hash_base != hash_query_readout, "config hash 必须区分 graph_query_readout_scale 的结构差异。")
    _assert(hash_base != hash_query_residual, "config hash 必须区分 lambda_personal_query_residual 的差异。")


def _check_append_summary_csv_tracks_runtime_structure_fields() -> None:
    from src.experiment_utils import append_summary_csv

    results_path = Path(ROOT) / "results" / "experiment_results.csv"
    backup = results_path.read_bytes() if results_path.exists() else None

    args = SimpleNamespace(
        dataset_name="assist_09",
        model_variant="assist_09_abce_best_full",
        save_dir=str(Path(ROOT) / "tmp_smoke_save"),
        seed=42,
        share_concept_embeddings=True,
        personal_disable_student_global_context=True,
        personal_support_only=True,
        graph_identity_residual=0.1,
        personal_delta_scale=6.0,
        personal_warmup_epochs=8,
        personal_include_neighbor_rows=False,
        personal_query_correction_scale=0.10,
        lambda_personal_kl=0.02,
        lambda_personal_query_residual=0.05,
        personal_query_residual_margin=0.08,
        lambda_alpha_min=0.08,
        alpha_min_target=0.05,
        graph_query_readout_scale=0.60,
        graph_query_readout_2hop_scale=0.20,
        personal_query_row_budget=1.0,
        personal_neighbor_row_budget=0.35,
    )
    logger = logging.getLogger("smoke_append_summary")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())

    try:
        append_summary_csv(
            args,
            metrics={"auc": 0.75, "acc": 0.70, "rmse": 0.43},
            best_val_auc=0.74,
            model_epoch=6,
            logger=logger,
            final_model_facts={
                "enable_module1": True,
                "use_concept_graph": True,
                "use_personal_graph": True,
            },
        )
        df = pd.read_csv(results_path)
        row = df.iloc[-1].to_dict()
    finally:
        if backup is None:
            if results_path.exists():
                results_path.unlink()
        else:
            results_path.write_bytes(backup)

    for key in (
        "final_use_concept_graph",
        "final_use_personal_graph",
        "share_concept_embeddings",
        "personal_disable_student_global_context",
        "personal_support_only",
        "graph_identity_residual",
        "personal_delta_scale",
        "personal_warmup_epochs",
        "personal_include_neighbor_rows",
        "personal_query_correction_scale",
        "lambda_personal_kl",
        "lambda_personal_query_residual",
        "personal_query_residual_margin",
        "lambda_alpha_min",
        "alpha_min_target",
        "graph_query_readout_scale",
        "graph_query_readout_2hop_scale",
        "personal_query_row_budget",
        "personal_neighbor_row_budget",
    ):
        _assert(key in row, f"experiment_results.csv 必须记录 {key}，否则看不清真实结构。")
    _assert(bool(row["final_use_concept_graph"]) is True, "summary 应记录最终 runtime 的 use_concept_graph。")
    _assert(bool(row["final_use_personal_graph"]) is True, "summary 应记录最终 runtime 的 use_personal_graph。")


def _check_diagnosis_csv_schema_upgrade() -> None:
    from run_abce_ablation import append_result_row

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "diagnosis.csv"
        csv_path.write_text(
            "dataset,seed,profile,ablation,test_seen_coverage,clean_baseline,status\n"
            "assist_09,42,best,full,0.97,True,ok\n",
            encoding="utf-8",
        )
        append_result_row(
            csv_path,
            {
                "dataset": "assist_09",
                "seed": 42,
                "profile": "best",
                "ablation": "no_E",
                "model_variant": "assist_09_abce_best_no_E",
                "test_auc": 0.76,
                "test_acc": 0.73,
                "test_rmse": 0.43,
                "best_val_auc": 0.75,
                "best_epoch": 2,
                "model_epoch": 2,
                "test_total_rows": 100,
                "test_seen_rows": 97,
                "test_seen_coverage": 0.97,
                "effective_batch_size": 128,
                "clean_baseline": True,
                "effective_enable_module1": True,
                "effective_use_concept_graph": True,
                "effective_use_personal_graph": False,
                "status": "ok",
                "exit_code": 0,
                "save_dir": "tmp/save",
                "log_dir": "tmp/logs",
                "log_file": "tmp/logs/train.log",
                "params_json": "{}",
                "flags_json": "{}",
            },
        )
        df = pd.read_csv(csv_path)

    _assert(
        "effective_batch_size" in df.columns,
        "diagnosis.csv 遇到旧表头时应自动升级到新 schema。",
    )
    _assert(
        int(df.iloc[-1]["effective_batch_size"]) == 128,
        "升级后的 diagnosis.csv 应把新行的 effective_batch_size 写到正确列，而不是造成整行错位。",
    )
    _assert(
        str(df.iloc[0]["status"]) == "ok",
        "旧 schema 升级时不应破坏原有行的数据对齐。",
    )


def _check_component_analysis_uses_real_q_conditioned_path() -> None:
    from src.trainer import save_component_analysis_data

    model = _build_tiny_ae_model()
    loader = DataLoader(
        TensorDataset(
            torch.tensor([0, 1, 2], dtype=torch.long),
            torch.tensor([0, 1, 2], dtype=torch.long),
            torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32),
        ),
        batch_size=2,
        shuffle=False,
    )
    logger = logging.getLogger("smoke_component_analysis")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())

    with tempfile.TemporaryDirectory() as tmpdir:
        analysis = save_component_analysis_data(
            model=model,
            train_loader=loader,
            device=torch.device("cpu"),
            save_dir=tmpdir,
            logger=logger,
            num_samples=3,
        )

    for key in (
        "relation_used_samples",
        "local_row_mask_samples",
        "q_vector_samples",
        "exercise_ids_samples",
        "personal_matrices_samples",
    ):
        _assert(key in analysis, f"component analysis 必须保存 {key}，否则分析图不是走真实推理路径。")
    q_vectors = torch.tensor(analysis["q_vector_samples"])
    local_masks = torch.tensor(analysis["local_row_mask_samples"])
    exercise_ids = torch.tensor(analysis["exercise_ids_samples"])
    _assert(q_vectors.shape[0] == local_masks.shape[0], "analysis 中 q_vector 与 local_row_mask 样本数应一致。")
    _assert(
        int(torch.unique(exercise_ids).numel()) >= 2,
        "component analysis 至少应保留两个不同题目的样本，才能验证它走的是题目条件化路径。",
    )
    _assert(
        torch.allclose(q_vectors.float(), model.q_matrix[exercise_ids].cpu().float()),
        "component analysis 保存的 q_vector 应与真实 exercise_ids 对应的题目概念向量一致。",
    )


def _check_train_validate_use_processed_batch_count() -> None:
    from src.trainer import train_epoch, validate

    model = _build_tiny_ae_model()
    logger = logging.getLogger("smoke_batch_mean")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())

    dataset = TensorDataset(
        torch.tensor([0, 1, 2, 3], dtype=torch.long),
        torch.tensor([0, 1, 2, 3], dtype=torch.long),
        torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    student_ids, exercise_ids, labels = next(iter(loader))

    model.eval()
    with torch.no_grad():
        logits, details = model(
            student_ids,
            exercise_ids,
            return_details=True,
            return_logits=True,
        )
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        reg = model.get_regularization_components(details["relation_matrices"], details, bce)["total"]
        expected_val_loss = float((bce + reg).item())

    val_metrics = validate(
        model=model,
        val_loader=loader,
        device=torch.device("cpu"),
        logger=logger,
        epoch=1,
        max_batches=1,
    )
    _assert(
        abs(val_metrics["loss"] - expected_val_loss) < 1e-6,
        f"validate 使用 max_batches 时应按实际处理 batch 数取平均；当前 loss={val_metrics['loss']:.6f}, expected={expected_val_loss:.6f}",
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    model.train()
    logits, details = model(
        student_ids,
        exercise_ids,
        return_details=True,
        return_logits=True,
    )
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    reg = model.get_regularization_components(details["relation_matrices"], details, bce)["total"]
    expected_train_loss = float((bce + reg).item())

    train_metrics = train_epoch(
        model=model,
        train_loader=loader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        logger=logger,
        epoch=1,
        max_batches=1,
    )
    _assert(
        abs(train_metrics["loss"] - expected_train_loss) < 1e-6,
        f"train_epoch 使用 max_batches 时应按实际处理 batch 数取平均；当前 loss={train_metrics['loss']:.6f}, expected={expected_train_loss:.6f}",
    )


def _check_concept_embedding_sharing_uses_same_storage() -> None:
    from src.model import CognitiveDiagnosisModel

    q_matrix = torch.eye(3, dtype=torch.float32)
    model = CognitiveDiagnosisModel(
        num_students=4,
        num_exercises=3,
        num_concepts=3,
        q_matrix=q_matrix,
        knowledge_dim=8,
        num_relation_heads=2,
        num_gnn_layers=1,
        dropout=0.0,
        use_concept_graph=True,
        use_personal_graph=True,
        share_concept_embeddings=True,
        personal_rank=4,
        personal_max_alpha=0.4,
        personal_delta_scale=6.0,
        personal_warmup_epochs=0,
        personal_student_dim=8,
    )
    rel_weight = model.structure_module.relation_learning.concept_embeddings
    enc_weight = model.structure_module.knowledge_encoder.concept_emb.weight
    _assert(
        rel_weight.data_ptr() == enc_weight.data_ptr(),
        "开启 share_concept_embeddings 后，relation_learning 与 knowledge_encoder 应共享同一块参数存储。",
    )


def _check_personal_generator_is_state_aware_and_bounded() -> None:
    from src.model import PersonalRelationGenerator

    torch.manual_seed(0)
    generator = PersonalRelationGenerator(
        student_dim=8,
        context_dim=16,
        knowledge_dim=8,
        num_concepts=4,
        num_heads=2,
        rank=3,
    )
    generator.eval()

    student_embedding = torch.zeros(2, 8)
    context_repr = torch.zeros(2, 16)
    knowledge_state_a = torch.randn(2, 4, 8) * 50.0
    knowledge_state_b = knowledge_state_a.clone()
    knowledge_state_b[1] = knowledge_state_b[1].flip(0)

    with torch.no_grad():
        out_a = generator(student_embedding, context_repr, knowledge_state_a)
        out_b = generator(student_embedding, context_repr, knowledge_state_b)

    _assert(torch.isfinite(out_a).all(), "state-aware personal generator 的输出不应出现 NaN/Inf。")
    _assert(
        float(out_a.abs().max()) <= 1.0001,
        "state-aware personal generator 应输出有界 residual，避免大图上的 softmax logits 爆炸。",
    )
    delta = float((out_a[1] - out_b[1]).abs().mean())
    _assert(
        delta > 1e-3,
        f"personal generator 应对 knowledge_state 变化敏感，当前差异过小: {delta:.6f}",
    )


def _check_junyi_like_personal_graph_stays_finite() -> None:
    from src.model import CognitiveDiagnosisModel

    torch.manual_seed(0)
    num_concepts = 24
    q_matrix = torch.eye(num_concepts, dtype=torch.float32)
    model = CognitiveDiagnosisModel(
        num_students=6,
        num_exercises=num_concepts,
        num_concepts=num_concepts,
        q_matrix=q_matrix,
        knowledge_dim=16,
        num_relation_heads=2,
        num_gnn_layers=2,
        dropout=0.0,
        use_concept_graph=True,
        use_personal_graph=True,
        share_concept_embeddings=True,
        graph_identity_residual=0.05,
        personal_rank=4,
        personal_max_alpha=0.35,
        personal_delta_scale=4.0,
        personal_warmup_epochs=0,
        personal_reg_warmup_epochs=0,
        personal_student_dim=8,
        personal_alpha_temperature=1.8,
        personal_alpha_budget=0.09,
        personal_alpha_base_init=0.04,
        personal_alpha_bias_scale=0.0,
        personal_disable_student_global_context=True,
        personal_include_neighbor_rows=False,
        personal_query_row_budget=1.3,
        personal_neighbor_row_budget=0.20,
        graph_query_readout_scale=0.40,
        graph_query_readout_2hop_scale=0.10,
    )
    model.eval()

    with torch.no_grad():
        _, details = model(
            student_ids=torch.tensor([0, 1, 2], dtype=torch.long),
            exercise_ids=torch.tensor([0, 1, 2], dtype=torch.long),
            return_details=True,
            return_logits=True,
        )

    _assert(torch.isfinite(details["knowledge_state"]).all(), "junyi-like 单概念场景下 knowledge_state 仍必须保持 finite。")
    personal_dense = _materialize_sparse_personal_dense(details)
    _assert(torch.isfinite(personal_dense).all(), "junyi-like 单概念场景下 personal graph 不应出现 NaN/Inf。")
    _assert(
        _to_int_tensor(details["personal_bad_row_count"]) == 0,
        f"junyi-like 单概念场景下不应出现 bad row fallback，当前={_to_int_tensor(details['personal_bad_row_count'])}",
    )
    _assert(
        float(details["personal_matrix_student_std"].item()) > 1e-5,
        "junyi-like 单概念场景下 personal_matrix_student_std 不应继续接近 0。",
    )
    _assert(
        float(details["query_row_personal_delta"].item()) > 1e-4,
        "junyi-like 单概念场景下 queried concept rows 必须有非零 personalization。",
    )
    _assert(
        float(details["personal_query_row_std"].item()) > 1e-6,
        "junyi-like 单概念场景下 queried concept rows 之间应保留非零学生差异。",
    )


def _to_int_tensor(value: torch.Tensor) -> int:
    return int(value.detach().reshape(-1)[0].item())


def _check_trainer_monitors_are_aligned() -> None:
    from src.trainer import _default_monitor_config

    monitor = _default_monitor_config()
    _assert(monitor["scheduler_monitor"] == "val_auc", "scheduler monitor 默认应切到 val_auc。")
    _assert(monitor["best_monitor"] == "val_auc", "best checkpoint monitor 默认应切到 val_auc。")
    _assert(monitor["early_stop_monitor"] == "val_auc", "early stopping monitor 默认应切到 val_auc。")
    _assert(monitor["scheduler_mode"] == monitor["best_mode"] == monitor["early_stop_mode"] == "max", "monitor 方向必须统一为 max。")


def _check_invalid_ablation_rows_are_filtered_from_summary() -> None:
    from run_abce_ablation import write_summary

    rows = [
        {
            "dataset": "assist_09",
            "seed": "42",
            "profile": "best",
            "ablation": "full",
            "status": "ok",
            "ablation_valid": "False",
            "test_auc": "0.75",
        },
        {
            "dataset": "assist_09",
            "seed": "42",
            "profile": "best",
            "ablation": "no_A",
            "status": "ok",
            "ablation_valid": "True",
            "test_auc": "0.74",
        },
        {
            "dataset": "assist_09",
            "seed": "42",
            "profile": "best",
            "ablation": "no_E",
            "status": "ok",
            "ablation_valid": "True",
            "test_auc": "0.74",
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        summary_rows, mean_rows = write_summary(
            Path(tmpdir) / "summary.csv",
            Path(tmpdir) / "mean.csv",
            rows,
        )

    _assert(summary_rows == [], "invalid full run 不应进入 summary。")
    _assert(mean_rows == [], "invalid full run 不应进入 mean summary。")


def _check_failure_reason_is_collected() -> None:
    from run_abce_ablation import AblationSpec, JobSpec, collect_result

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir) / "ckpt"
        log_dir = Path(tmpdir) / "log"
        save_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / "failure_reason.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "reason": "nonfinite_personal_logits",
                    "stage": "train",
                    "message": "nonfinite_personal_logits at train epoch=1 batch=3",
                },
                f,
            )

        job = JobSpec(
            dataset="junyi",
            seed=42,
            profile="best",
            ablation=AblationSpec(name="full", flags={}, overrides={}),
            model_variant="junyi_abce_best_full",
            save_dir=save_dir,
            log_dir=log_dir,
            params={"use_personal_graph": True},
            cmd=[],
        )
        row = collect_result(job, exit_code=1)

    _assert(row["failure_reason"] == "nonfinite_personal_logits", "collector 应保留结构化 failure reason。")
    _assert(row["failure_stage"] == "train", "collector 应保留 failure stage。")


def _check_junyi_e_on_jobs_use_oom_safe_batch_size() -> None:
    from run_abce_ablation import make_jobs

    args = SimpleNamespace(
        datasets="junyi",
        seeds="42",
        component_set="single",
        ablations="full,no_A,no_E",
        profiles="best",
        rerun_existing=True,
        epochs=None,
        early_stop_patience=None,
        learning_rate=None,
        generate_diagnosis=True,
        max_train_batches=None,
        max_val_batches=None,
        max_test_batches=None,
        include_matched_no_e=False,
    )
    jobs = make_jobs(args, run_id="smoke_junyi_batch")
    jobs_by_ablation = {job.ablation.name: job for job in jobs}

    _assert(set(jobs_by_ablation.keys()) == {"full", "no_A", "no_E"}, "junyi smoke 应生成 full/no_A/no_E 三个作业。")
    _assert(
        int(jobs_by_ablation["full"].params["batch_size"]) == 64,
        f"junyi full 在 E 开启时应自动降到 OOM-safe batch_size=64，当前={jobs_by_ablation['full'].params['batch_size']}",
    )
    _assert(
        int(jobs_by_ablation["no_A"].params["batch_size"]) == 64,
        f"junyi no_A 在 E 开启时也应自动降到 OOM-safe batch_size=64，当前={jobs_by_ablation['no_A'].params['batch_size']}",
    )
    _assert(
        int(jobs_by_ablation["no_E"].params["batch_size"]) == 256,
        f"junyi no_E 不应被错误降 batch；当前={jobs_by_ablation['no_E'].params['batch_size']}",
    )


def _check_runner_env_enables_expandable_segments() -> None:
    from run_abce_ablation import _build_job_env

    env = _build_job_env({"PATH": "dummy"}, gpu_id=2)
    _assert(env["CUDA_VISIBLE_DEVICES"] == "2", "runner 应按作业设置 CUDA_VISIBLE_DEVICES。")
    _assert(
        env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True",
        "runner 应默认启用 expandable_segments 以减轻 CUDA 显存碎片化。",
    )

    env2 = _build_job_env({"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"}, gpu_id=1)
    _assert(
        env2["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:128,expandable_segments:True",
        "runner 应保留已有 CUDA alloc 配置，并追加 expandable_segments:True。",
    )


def _check_matched_no_e_job_can_be_enabled() -> None:
    from run_abce_ablation import make_jobs

    args = SimpleNamespace(
        datasets="junyi",
        seeds="42",
        component_set="single",
        ablations="full,no_A,no_E",
        profiles="best",
        rerun_existing=True,
        epochs=None,
        early_stop_patience=None,
        learning_rate=None,
        generate_diagnosis=True,
        max_train_batches=None,
        max_val_batches=None,
        max_test_batches=None,
        include_matched_no_e=True,
    )
    jobs = make_jobs(args, run_id="smoke_junyi_matched_no_e")
    jobs_by_ablation = {job.ablation.name: job for job in jobs}
    _assert("no_E_bs64" in jobs_by_ablation, "启用 include_matched_no_e 后应生成 junyi 的 no_E_bs64 对照作业。")
    _assert(
        int(jobs_by_ablation["no_E_bs64"].params["batch_size"]) == 64,
        "matched no_E 作业应固定使用 batch_size=64。",
    )


def main() -> None:
    _check_no_a_keeps_gnn_layers()
    _check_best_configs_enable_e_rescue_knobs()
    _check_dataset_defaults_respect_explicit_zero_overrides()
    _check_no_a_personal_graph_is_not_identity_locked()
    _check_personal_branch_is_state_primary_with_small_id_adapter()
    _check_adaptive_gate_is_state_primary()
    _check_gate_saturation_regression()
    _check_assist_like_harmfulness_proxy_warning()
    _check_personal_graph_is_support_preserving_and_local()
    _check_personal_graph_uses_sparse_runtime_spec()
    _check_query_readout_injects_graph_signal()
    _check_e_query_only_active_rows()
    _check_e_readout_only_does_not_change_backbone()
    _check_posterior_equal_global_gives_zero_personal_query_correction()
    _check_split_hygiene_uses_train_only_maps_and_q_matrix()
    _check_runtime_ablation_guardrails_cover_no_a_and_no_e()
    _check_generator_state_adapter_is_live()
    _check_alpha_bias_scale_is_live()
    _check_alpha_bias_diagnostics_consistency()
    _check_config_hash_tracks_ae_structure_switches()
    _check_append_summary_csv_tracks_runtime_structure_fields()
    _check_diagnosis_csv_schema_upgrade()
    _check_component_analysis_uses_real_q_conditioned_path()
    _check_train_validate_use_processed_batch_count()
    _check_concept_embedding_sharing_uses_same_storage()
    _check_personal_generator_is_state_aware_and_bounded()
    _check_junyi_like_personal_graph_stays_finite()
    _check_trainer_monitors_are_aligned()
    _check_summary_classification_regression()
    _check_summary_prefers_matched_no_e()
    _check_invalid_ablation_rows_are_filtered_from_summary()
    _check_failure_reason_is_collected()
    _check_junyi_e_on_jobs_use_oom_safe_batch_size()
    _check_runner_env_enables_expandable_segments()
    _check_matched_no_e_job_can_be_enabled()
    _check_module_activity_brief_uses_live_risk()
    print("OK: AE rescue regression smoke checks passed.")


if __name__ == "__main__":
    main()
