import os
import sys

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _build_small_model():
    from src.model import CognitiveDiagnosisModel

    q_matrix = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
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
        use_concept_graph=True,
        use_personal_graph=True,
        share_concept_embeddings=True,
        graph_prior_logit_scale=0.8,
        graph_topk=2,
        ae_logit_residual_scale=1.0,
        ae_interaction_logit_scale=0.5,
        ae_query_residual_scale=0.2,
        personal_delta_scale=3.0,
        personal_mastery_prior_scale=1.0,
        personal_recent_mastery_prior_scale=0.5,
        personal_mastery_count_smoothing=1.0,
        personal_warmup_epochs=0,
        personal_reg_warmup_epochs=0,
    )
    model.initialize_ae_logit_priors(
        student_logits=torch.tensor([0.4, -0.3, 0.2, -0.1, 0.1], dtype=torch.float32),
        exercise_logits=torch.tensor([0.3, -0.2, 0.1, -0.1], dtype=torch.float32),
        concept_logits=torch.tensor([0.2, -0.1, 0.3], dtype=torch.float32),
        student_concept_logits=torch.tensor(
            [
                [0.6, -0.2, 0.1],
                [-0.1, 0.5, 0.2],
                [0.2, -0.3, 0.4],
                [0.1, 0.2, -0.5],
                [0.3, 0.1, -0.2],
            ],
            dtype=torch.float32,
        ),
        student_concept_recent_logits=torch.tensor(
            [
                [0.2, -0.1, 0.1],
                [-0.1, 0.2, 0.0],
                [0.1, -0.2, 0.3],
                [0.0, 0.1, -0.2],
                [0.2, 0.0, -0.1],
            ],
            dtype=torch.float32,
        ),
        student_concept_observed_counts=torch.full((5, 3), 4.0, dtype=torch.float32),
        scale=1.0,
    )
    model.eval()
    return model


def _check_a_has_only_interpretable_graph_parameters() -> None:
    from src.model_graph import MultiHeadRelationLearning

    graph = MultiHeadRelationLearning(
        num_concepts=4,
        concept_dim=8,
        num_heads=2,
        dropout=0.0,
        prior_logit_scale=0.5,
    )
    forbidden = (
        "Wq",
        "Wk",
        "rel_query_anchor",
        "rel_key_anchor",
        "graph_edge_bias_u",
        "graph_edge_bias_v",
    )
    for name in forbidden:
        _assert(not hasattr(graph, name), f"A 不应再保留黑箱注意力/低秩边参数: {name}")
    for name in ("prior_strength_raw", "receiver_bias", "self_loop_logit"):
        _assert(hasattr(graph, name), f"A 应暴露可解释参数: {name}")
    relation, _ = graph()
    _assert(torch.isfinite(relation).all().item(), "A relation matrix should stay finite.")
    _assert((relation >= -1e-7).all().item(), "A relation matrix should be non-negative.")
    _assert((relation.sum(dim=-1) - 1.0).abs().max().item() < 1e-5, "A rows should sum to 1.")


def _check_roadmap_and_tutor_are_not_blackbox_modules() -> None:
    model = _build_small_model()
    structure = model.structure_module

    for name in (
        "personal_gate_embedding",
        "personal_generator_embedding",
        "personal_gate_from_state",
        "personal_generator_from_state",
        "personal_alpha_bias",
    ):
        _assert(not hasattr(structure, name), f"E 不应再保留 student-id/MLP shortcut: {name}")

    for name in (
        "ae_query_state_adapter",
        "ae_logit_adapter",
        "ae_student_logit_emb",
        "ae_concept_logit_emb",
        "ae_student_factor",
        "ae_exercise_factor",
        "ae_graph_state_proj",
        "ae_personal_state_proj",
    ):
        _assert(not hasattr(model, name), f"AE logit residual 不应再保留黑箱模块: {name}")

    for name in (
        "ae_logit_feature_weight",
        "ae_reliability_feature_weight",
        "ae_student_logit_bias",
        "ae_exercise_logit_bias",
        "ae_concept_logit_bias",
    ):
        _assert(not hasattr(model, name), f"AE logit residual 不应保留旧黑箱校正参数: {name}")
    for name in (
        "roadmap_difficulty_weight_raw",
        "roadmap_reliability_weight_raw",
        "roadmap_item_weight_raw",
        "roadmap_sequence_weight_raw",
        "tutor_current_mastery_weight_raw",
        "tutor_current_recent_weight_raw",
        "tutor_route_mastery_weight_raw",
        "tutor_route_recent_weight_raw",
        "tutor_gap_penalty_weight_raw",
    ):
        param = getattr(model, name, None)
        _assert(isinstance(param, torch.nn.Parameter) and tuple(param.shape) == (), f"地图模块应暴露标量可解释权重: {name}")

    with torch.no_grad():
        logits, details = model(
            student_ids=torch.tensor([0, 1, 2, 3], dtype=torch.long),
            exercise_ids=torch.tensor([0, 1, 2, 3], dtype=torch.long),
            return_details=True,
            return_logits=True,
        )
    _assert(torch.isfinite(logits).all().item(), "Interpretable AE logits should stay finite.")
    _assert(details["ae_logit_residual_abs_mean"].item() > 0.0, "Interpretable AE residual should be active.")
    _assert(details["roadmap_macro_logit_abs_mean"].item() > 0.0, "A roadmap contribution should be active.")
    _assert(details["tutor_local_navigation_logit_abs_mean"].item() > 0.0, "E tutoring contribution should be active.")
    _assert(
        details["tutor_personal_route_delta_abs_mean"].item() > 0.0,
        "E local navigation should use the personalized posterior route, not only the global A route.",
    )
    _assert(isinstance(details["personal_relation_spec"], dict), "E should still expose sparse relation spec.")
    posterior = details["personal_relation_spec"]["posterior_prob"]
    support_valid = details["personal_relation_spec"]["support_valid_mask"].bool()
    active = details["personal_relation_spec"]["active_row_valid_mask"].bool().unsqueeze(1)
    valid = active & support_valid.any(dim=-1)
    _assert(valid.any().item(), "E should have active supported rows.")
    row_sums = posterior.sum(dim=-1)
    _assert((row_sums[valid] - 1.0).abs().max().item() < 1e-5, "E posterior rows should sum to 1.")
    _assert(details["query_row_posterior_delta_abs"].item() > 0.0, "E posterior should differ from A support.")


def _check_runner_keeps_three_patience_rounds() -> None:
    from run_abce_ablation import make_jobs
    from types import SimpleNamespace

    args = SimpleNamespace(
        datasets="junyi",
        seeds="42",
        component_set="single",
        ablations="full,no_A,no_E",
        profiles="best",
        rerun_existing=True,
        epochs=None,
        early_stop_patience=1,
        learning_rate=None,
        generate_diagnosis=True,
        max_train_batches=None,
        max_val_batches=None,
        max_test_batches=None,
        param_overrides=None,
        include_matched_no_e=True,
    )
    jobs = make_jobs(args, run_id="smoke_interpretable_patience")
    for job in jobs:
        _assert(
            int(job.params["early_stop_patience"]) >= 3,
            f"{job.ablation.name} early_stop_patience 必须至少为 3，当前={job.params['early_stop_patience']}",
        )


def main() -> None:
    _check_a_has_only_interpretable_graph_parameters()
    _check_roadmap_and_tutor_are_not_blackbox_modules()
    _check_runner_keeps_three_patience_rounds()
    print("OK: interpretable AE smoke checks passed.")


if __name__ == "__main__":
    main()
