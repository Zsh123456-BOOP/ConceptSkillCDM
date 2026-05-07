import os
import sys

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _assert_all_finite(name: str, value) -> None:
    if value is None:
        return
    if isinstance(value, torch.Tensor):
        _assert(torch.isfinite(value).all().item(), f"{name} contains non-finite values.")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_all_finite(f"{name}.{key}", child)
        return
    if isinstance(value, (list, tuple)):
        for idx, child in enumerate(value):
            _assert_all_finite(f"{name}[{idx}]", child)


def _assert_sparse_rows_normalized(name: str, spec: dict) -> None:
    posterior = spec["posterior_prob"]
    support_valid = spec["support_valid_mask"].bool()
    active_rows = spec["active_row_valid_mask"].bool().unsqueeze(1).expand(-1, support_valid.size(1), -1)
    row_has_support = support_valid.any(dim=-1)
    valid_rows = active_rows & row_has_support
    _assert(valid_rows.any().item(), f"{name} should contain active supported rows.")
    row_sums = posterior.sum(dim=-1)
    max_err = (row_sums[valid_rows] - 1.0).abs().max().item()
    _assert(max_err < 1e-5, f"{name} posterior rows should sum to 1, max_err={max_err:.3e}.")
    _assert((posterior >= -1e-7).all().item(), f"{name} posterior should be non-negative.")


def _build_model(
    *,
    ablate_module1: bool = False,
    use_concept_graph: bool = True,
    use_personal_graph: bool = True,
    relation_theta_scale: float = 0.0,
):
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
    model = CognitiveDiagnosisModel(
        num_students=5,
        num_exercises=4,
        num_concepts=3,
        q_matrix=q_matrix,
        knowledge_dim=8,
        num_relation_heads=2,
        num_gnn_layers=1,
        dropout=0.0,
        use_concept_graph=use_concept_graph,
        use_personal_graph=use_personal_graph,
        personal_rank=2,
        ablate_module1=ablate_module1,
        graph_tau_init=0.6,
        graph_topk=2,
        allow_self_loop=True,
        relation_theta_scale=relation_theta_scale,
    )
    model.eval()
    return model


def _check_module1_ae_numerics_and_signals() -> None:
    torch.manual_seed(7)
    model = _build_model(ablate_module1=False, use_concept_graph=True, use_personal_graph=True)
    model.set_epoch(3)
    if getattr(model.structure_module, "personal_alpha_bias", None) is not None:
        with torch.no_grad():
            model.structure_module.personal_alpha_bias.weight.fill_(5.0)
    model.structure_module.personal_delta_scale = 10.0

    student_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    exercise_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)

    logits, details = model(
        student_ids=student_ids,
        exercise_ids=exercise_ids,
        return_details=True,
        return_logits=True,
    )
    probs = model(
        student_ids=student_ids,
        exercise_ids=exercise_ids,
        return_details=False,
        return_logits=False,
    )

    _assert_all_finite("logits", logits)
    _assert_all_finite("probs", probs)
    _assert(logits.abs().max().item() < 100.0, "Logits should stay in a numerically usable range.")
    _assert(((probs >= 0.0) & (probs <= 1.0)).all().item(), "Probabilities should stay in [0, 1].")

    relation = details["relation_matrices"]
    _assert_all_finite("relation_matrices", relation)
    _assert(tuple(relation.shape) == (2, 3, 3), f"Unexpected relation_matrices shape: {tuple(relation.shape)}.")
    _assert((relation >= -1e-7).all().item(), "Global concept graph should be non-negative.")
    graph_row_err = (relation.sum(dim=-1) - 1.0).abs().max().item()
    _assert(graph_row_err < 1e-5, f"Global concept graph rows should sum to 1, max_err={graph_row_err:.3e}.")
    _assert(details["relation_identity_delta"].item() > 1e-5, "Module 1(A) should learn a non-identity graph signal.")
    _assert(details["knowledge_state_graph_delta"].item() > 1e-5, "Module 1(A) should affect knowledge_state.")
    _assert(details["a_diag_semantic_ok"].item() == 1.0, "Module 1(A) semantic diagnostic should be OK.")

    personal_spec = details["personal_relation_spec"]
    _assert(isinstance(personal_spec, dict), "Module 1(E) should expose a sparse personal relation spec.")
    _assert_all_finite("personal_relation_spec", personal_spec)
    _assert_sparse_rows_normalized("personal_relation_spec", personal_spec)

    personal_delta = (personal_spec["posterior_prob"] - personal_spec["global_support_prob"]).abs().max().item()
    _assert(personal_delta > 1e-5, "Module 1(E) posterior should differ from the global support.")
    _assert(details["query_row_posterior_delta_abs"].item() > 1e-5, "Module 1(E) should produce query-row posterior signal.")
    _assert(details["personal_matrix_delta"].mean().item() > 1e-5, "Module 1(E) matrix delta should be non-trivial.")
    _assert(details["personal_logits_support_absmax"].item() < 30.0, "Personal support logits should not explode.")

    for count_key in (
        "personal_delta_nonfinite_count",
        "personal_logits_nonfinite_count",
        "personal_matrix_nonfinite_count",
        "state_embedding_nonfinite_count",
        "context_repr_nonfinite_count",
        "state_logit_nonfinite_count",
        "id_logit_nonfinite_count",
        "alpha_base_nonfinite_count",
        "alpha_preclamp_nonfinite_count",
        "personal_bad_row_count_active",
        "personal_fallback_row_count_active",
    ):
        _assert(count_key in details, f"Expected numeric diagnostic key: {count_key}.")
        _assert(int(details[count_key].item()) == 0, f"{count_key} should be 0.")

    alpha = details["alpha"]
    _assert_all_finite("alpha", alpha)
    _assert((alpha >= 0.0).all().item(), "Personal alpha should be non-negative.")
    _assert((alpha <= model.personal_max_alpha + 1e-7).all().item(), "Personal alpha should respect max_alpha.")

    base_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    reg_terms = model.get_regularization_components(
        relation_matrices=relation,
        details=details,
        base_loss=base_loss,
    )
    _assert_all_finite("regularization_terms", reg_terms)
    _assert(reg_terms["total"].abs().item() < 10.0, "Regularization total should not numerically dominate smoke loss.")


def _check_graph_query_adapter_is_initially_residual() -> None:
    torch.manual_seed(1)
    model = _build_model(ablate_module1=False, use_concept_graph=True, use_personal_graph=True)
    student_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    exercise_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    with torch.no_grad():
        _, details = model(
            student_ids=student_ids,
            exercise_ids=exercise_ids,
            return_details=True,
            return_logits=True,
        )

    pre_gate_delta = details["query_row_global_readout_pre_gate_delta"].item()
    post_gate_delta = details["query_row_global_readout_delta"].item()
    adapter_gain = details["graph_query_adapter_gain"].item()
    _assert(pre_gate_delta > 1e-5, "Smoke case should exercise non-zero graph query readout.")
    _assert(
        adapter_gain <= 1.05,
        (
            "Graph query adapter should be a near-residual no-op at initialization; "
            f"got gain={adapter_gain:.4f}, pre={pre_gate_delta:.4f}, post={post_gate_delta:.4f}."
        ),
    )


def _check_per_head_personal_graph() -> None:
    model = _build_model(ablate_module1=False, use_personal_graph=True)
    if getattr(model.structure_module, "personal_alpha_bias", None) is not None:
        with torch.no_grad():
            model.structure_module.personal_alpha_bias.weight.fill_(5.0)
    model.structure_module.personal_delta_scale = 10.0
    student_ids = torch.tensor([0, 1], dtype=torch.long)
    exercise_ids = torch.tensor([0, 2], dtype=torch.long)

    with torch.no_grad():
        _, details = model(
            student_ids=student_ids,
            exercise_ids=exercise_ids,
            return_details=True,
            return_logits=True,
        )

    alpha = details.get("alpha")
    personal_spec = details.get("personal_relation_spec")
    _assert(alpha is not None, "Expected per-head alpha details.")
    _assert(personal_spec is not None, "Expected sparse per-head personal relation spec.")
    _assert(tuple(alpha.shape) == (2, 2, 1, 1), f"Expected alpha shape (2,2,1,1), got {tuple(alpha.shape)}.")
    _assert(
        tuple(personal_spec["posterior_prob"].shape) == (2, 2, 2, 2),
        f"Expected sparse posterior_prob shape (2,2,2,2), got {tuple(personal_spec['posterior_prob'].shape)}.",
    )
    _assert_all_finite("personal_relation_spec", personal_spec)
    _assert_sparse_rows_normalized("personal_relation_spec", personal_spec)
    relation_used = details.get("relation_used")
    _assert(relation_used is not None, "Expected relation_used details when personal graph is enabled.")
    _assert(
        isinstance(relation_used, dict),
        f"Expected sparse relation_used spec dict, got {type(relation_used)}.",
    )
    _assert(
        tuple(relation_used["posterior_prob"].shape) == (2, 2, 2, 2),
        f"Expected relation_used sparse posterior_prob shape (2,2,2,2), got {tuple(relation_used['posterior_prob'].shape)}.",
    )
    _assert_all_finite("relation_used", relation_used)
    _assert_sparse_rows_normalized("relation_used", relation_used)
    personal_delta = (personal_spec["posterior_prob"] - personal_spec["global_support_prob"]).abs().max().item()
    used_delta = (relation_used["posterior_prob"] - relation_used["global_support_prob"]).abs().max().item()
    _assert(personal_delta > 1e-5, "Sparse posterior should differ from global support in this smoke case.")
    _assert(abs(used_delta - personal_delta) < 1e-6, "relation_used 应复用同一份 personalized sparse posterior 规格。")
    for key in ("personal_delta_pre_softmax_norm", "personal_delta_student_std", "alpha_head_std"):
        _assert(key in details, f"Expected detail key: {key}.")


def _check_removed_residual_artifacts() -> None:
    model = _build_model(ablate_module1=False, use_personal_graph=True)
    student_ids = torch.tensor([0, 1], dtype=torch.long)
    exercise_ids = torch.tensor([0, 1], dtype=torch.long)

    with torch.no_grad():
        _, details = model(
            student_ids=student_ids,
            exercise_ids=exercise_ids,
            return_details=True,
            return_logits=True,
        )

    for removed_key in ("mf_logit", "residual_logit", "gate", "gate_raw", "delta_logit"):
        _assert(removed_key not in details, f"Removed B detail should not appear: {removed_key}")
    _assert("irt_logit" in details, "Fixed prediction head should still expose irt_logit.")


def _check_relation_theta_readout_is_interpretable_and_ablatable() -> None:
    torch.manual_seed(11)
    full = _build_model(use_concept_graph=True, use_personal_graph=True, relation_theta_scale=0.5)
    no_a = _build_model(use_concept_graph=False, use_personal_graph=True, relation_theta_scale=0.5)
    student_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    exercise_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    with torch.no_grad():
        full_logits, full_details = full(
            student_ids=student_ids,
            exercise_ids=exercise_ids,
            return_details=True,
            return_logits=True,
        )
        off_logits, off_details = no_a(
            student_ids=student_ids,
            exercise_ids=exercise_ids,
            return_details=True,
            return_logits=True,
        )

    _assert_all_finite("relation_theta_full_logits", full_logits)
    _assert(full_details["relation_theta_scale"].item() == 0.5, "Relation-theta scale should be exposed.")
    _assert(
        full_details["relation_theta_neighbor_mass"].item() > 0.0,
        "Relation-theta readout should use explicit A/E support outside query self when A is enabled.",
    )
    _assert(
        full_details["relation_theta_logit_abs_mean"].item() >= 0.0,
        "Relation-theta readout should expose a bounded logit diagnostic.",
    )
    _assert(
        off_details["relation_theta_logit_abs_mean"].item() == 0.0,
        "no_A should remove relation-theta support readout completely.",
    )


def _check_get_student_diagnosis() -> None:
    for ablate_module1 in (False, True):
        model = _build_model(ablate_module1=ablate_module1, use_personal_graph=True)
        out = model.get_student_diagnosis(0)
        mastery = out.get("knowledge_mastery")
        student_repr = out.get("student_repr")
        _assert(mastery is not None, "knowledge_mastery should exist.")
        _assert(student_repr is not None, "student_repr should exist.")
        _assert(tuple(mastery.shape) == (3,), f"Unexpected mastery shape: {tuple(mastery.shape)}.")
        _assert(tuple(student_repr.shape) == (8,), f"Unexpected student_repr shape: {tuple(student_repr.shape)}.")
        _assert("skill_latent" not in out, "skill_latent should be removed with module B.")


def _check_grad_guard_keeps_nan_from_poisoning_group() -> None:
    from src.trainer import _clip_stability_sensitive_grads

    model = _build_model(ablate_module1=False, use_concept_graph=True, use_personal_graph=True)
    model.train()
    for param in model.parameters():
        if param.requires_grad:
            param.grad = torch.full_like(param, 0.01)

    bad_param = model.exercise_encoder.b.weight
    good_param = model.structure_module.knowledge_encoder.concept_emb.weight
    bad_param.grad.view(-1)[0] = float("nan")
    good_before = good_param.grad.detach().clone()

    stats = _clip_stability_sensitive_grads(model)

    _assert(int(stats["nonfinite_grad_count"]) == 1, "Grad guard should report the injected NaN gradient.")
    _assert(torch.isfinite(good_param.grad).all().item(), "A finite knowledge-state grad must not be poisoned by another NaN grad.")
    _assert(torch.isfinite(bad_param.grad).all().item(), "Non-finite grad entries should be sanitized before optimizer.step.")
    _assert(good_param.grad.abs().max().item() <= good_before.abs().max().item() + 1e-8, "Grad clipping should remain bounded.")


def main() -> None:
    _check_module1_ae_numerics_and_signals()
    _check_graph_query_adapter_is_initially_residual()
    _check_per_head_personal_graph()
    _check_removed_residual_artifacts()
    _check_relation_theta_readout_is_interpretable_and_ablatable()
    _check_get_student_diagnosis()
    _check_grad_guard_keeps_nan_from_poisoning_group()
    print("OK: runtime regression smoke checks passed.")


if __name__ == "__main__":
    main()
