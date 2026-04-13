import os
import sys

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


def main() -> None:
    _check_no_a_keeps_gnn_layers()
    _check_best_configs_enable_e_rescue_knobs()
    _check_no_a_personal_graph_is_not_identity_locked()
    print("OK: AE rescue regression smoke checks passed.")


if __name__ == "__main__":
    main()
