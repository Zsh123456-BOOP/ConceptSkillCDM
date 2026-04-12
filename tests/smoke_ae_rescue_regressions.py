import os
import sys


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


def main() -> None:
    _check_no_a_keeps_gnn_layers()
    _check_best_configs_enable_e_rescue_knobs()
    print("OK: AE rescue regression smoke checks passed.")


if __name__ == "__main__":
    main()
