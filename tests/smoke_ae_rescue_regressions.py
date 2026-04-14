import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


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
            bool(cfg["personal_disable_direct_bias"]) is True,
            f"{dataset} 应禁用 E 的 direct bias shortcut，避免个性化图继续退化为 student-id 偏置。",
        )
        _assert(
            float(cfg["personal_direct_bias_scale"]) == 0.0,
            f"{dataset} 的 personal_direct_bias_scale 必须显式为 0.0，避免 direct bias 被旧默认值偷偷带回。",
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
            float(cfg["personal_alpha_bias_scale"]) <= 0.03,
            f"{dataset} 的 personal_alpha_bias_scale 应降到 tiny adapter 量级，当前={cfg['personal_alpha_bias_scale']}",
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
        "--no-use_personal_graph",
        "--no-share_concept_embeddings",
        "--no-personal_disable_direct_bias",
        "--no-personal_disable_student_global_context",
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
        bool(args.use_personal_graph) is False,
        "显式传入的 --no-use_personal_graph 不应再被数据集默认值回写成 True。",
    )
    _assert(
        bool(args.share_concept_embeddings) is False,
        "显式传入的 --no-share_concept_embeddings 不应被数据集默认值回写成 True。",
    )
    _assert(
        bool(args.personal_disable_direct_bias) is False,
        "显式传入的 --no-personal_disable_direct_bias 不应被数据集默认值回写成 True。",
    )
    _assert(
        bool(args.personal_disable_student_global_context) is False,
        "显式传入的 --no-personal_disable_student_global_context 不应被数据集默认值回写成 True。",
    )
    _assert(
        float(args.lambda_sparse) == 0.3,
        "未显式覆盖的字段仍应继续继承数据集默认值，避免把默认机制整体打坏。",
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
        model.structure_module.personal_alpha_bias.weight.fill_(10.0)
        _, details = model(
            student_ids=torch.tensor([0, 1], dtype=torch.long),
            exercise_ids=torch.tensor([0, 2], dtype=torch.long),
            return_details=True,
            return_logits=True,
        )

    ident = model.identity_relations.unsqueeze(0)
    matrix_delta = float((details["personal_matrices"] - ident).abs().mean())
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
            bias,
            return_diagnostics=True,
        )
        alpha_state, _, diag_state = gate(
            state_b,
            context_b,
            id_a,
            bias,
            return_diagnostics=True,
        )
        alpha_id, _, diag_id = gate(
            state_a,
            context_a,
            id_b,
            bias,
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


def _check_direct_bias_can_be_disabled_without_collapsing_personal_graph() -> None:
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
        personal_alpha_bias_scale=0.0,
        personal_direct_bias_scale=0.0,
        personal_disable_student_global_context=True,
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
    matrix_delta = float((details["personal_matrices"] - ident).abs().mean())
    _assert(
        matrix_delta > 0.01,
        f"禁用 direct bias 后，E 仍应能生成非 identity 个性化图，当前 delta={matrix_delta:.6f}",
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
        max_direct_scale=0.0,
        disable_direct_bias=True,
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
        personal_alpha_bias_scale=0.015,
        personal_direct_bias_scale=0.0,
        personal_disable_direct_bias=True,
        personal_disable_student_global_context=True,
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
    _assert(torch.isfinite(details["personal_matrices"]).all(), "junyi-like 单概念场景下 personal_matrices 不应出现 NaN/Inf。")
    _assert(
        _to_int_tensor(details["personal_bad_row_count"]) == 0,
        f"junyi-like 单概念场景下不应出现 bad row fallback，当前={_to_int_tensor(details['personal_bad_row_count'])}",
    )
    _assert(
        float(details["personal_matrix_delta"].mean().item()) > 1e-3,
        "禁用 direct bias 后，junyi-like 场景下 E 仍应保留非零个性化扰动。",
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


def main() -> None:
    _check_no_a_keeps_gnn_layers()
    _check_best_configs_enable_e_rescue_knobs()
    _check_dataset_defaults_respect_explicit_zero_overrides()
    _check_no_a_personal_graph_is_not_identity_locked()
    _check_personal_branch_is_state_primary_with_small_id_adapter()
    _check_adaptive_gate_is_state_primary()
    _check_split_hygiene_uses_train_only_maps_and_q_matrix()
    _check_runtime_ablation_guardrails_cover_no_a_and_no_e()
    _check_direct_bias_can_be_disabled_without_collapsing_personal_graph()
    _check_concept_embedding_sharing_uses_same_storage()
    _check_personal_generator_is_state_aware_and_bounded()
    _check_junyi_like_personal_graph_stays_finite()
    _check_trainer_monitors_are_aligned()
    _check_invalid_ablation_rows_are_filtered_from_summary()
    _check_failure_reason_is_collected()
    print("OK: AE rescue regression smoke checks passed.")


if __name__ == "__main__":
    main()
