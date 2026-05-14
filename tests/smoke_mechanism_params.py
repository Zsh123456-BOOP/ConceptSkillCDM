import os
import sys
from types import SimpleNamespace


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _args(*, variants: str, phase: str = "phase1", phase1_epochs: int = 6) -> SimpleNamespace:
    return SimpleNamespace(
        phase=phase,
        datasets="assist_09",
        variants=variants,
        seed=42,
        num_workers=0,
        cpu_threads_per_job=2,
        phase1_epochs=phase1_epochs,
        phase1_max_train_batches=4,
        phase1_max_val_batches=2,
        phase1_max_test_batches=2,
        phase2_epochs=45,
        phase2_patience=8,
        limit_jobs=0,
        rerun_existing=True,
        generate_diagnosis=False,
        student_global_scale=None,
        student_global_mode=None,
    )


def _params_by_variant(variants: str, *, phase: str = "phase1"):
    from tools.run_mechanism_experiments import build_jobs

    jobs = build_jobs(_args(variants=variants, phase=phase), run_id="smoke_mechanism_params")
    return {job.variant: job.params for job in jobs}


def _check_phase1_full_activates_e_writing() -> None:
    params = _params_by_variant("full")["full"]
    _assert(bool(params["use_personal_graph"]), "full should keep E enabled.")
    _assert(float(params["personal_delta_scale"]) > 0.0, "full should let E change local posterior weights.")
    _assert(
        float(params["personal_query_correction_scale"]) > 0.0,
        "full should let E write a bounded query correction.",
    )
    _assert(
        float(params["personal_query_message_gain"]) > 0.0,
        "full should let E write a query-row personal message.",
    )
    _assert(
        int(params["personal_warmup_epochs"]) <= 2,
        "phase1 should not hide E behind a long warmup.",
    )
    _assert(
        int(params["personal_reg_warmup_epochs"]) <= 2,
        "phase1 should not hide E regularization behind a long warmup.",
    )
    _assert(
        float(params["ae_lr_mult"]) <= 5.0,
        "phase1 should cap fast AE scalar learning so E mechanism tests do not become train-prior overfit tests.",
    )


def _check_controls_keep_their_semantics() -> None:
    params_by_variant = _params_by_variant("no_E,A_fused_neutralE,E_posterior_only,E_query_only")
    _assert(not bool(params_by_variant["no_E"]["use_personal_graph"]), "no_E should disable E.")
    neutral = params_by_variant["A_fused_neutralE"]
    _assert(float(neutral["personal_delta_scale"]) == 0.0, "neutral-E A control should not change posterior.")
    _assert(float(neutral["personal_query_correction_scale"]) == 0.0, "neutral-E A control should not correct query.")
    _assert(float(neutral["personal_query_message_gain"]) == 0.0, "neutral-E A control should not write message.")

    posterior_only = params_by_variant["E_posterior_only"]
    _assert(float(posterior_only["personal_delta_scale"]) > 0.0, "posterior-only should keep E posterior active.")
    _assert(float(posterior_only["personal_query_correction_scale"]) == 0.0, "posterior-only should not use query correction.")
    _assert(float(posterior_only["personal_query_message_gain"]) == 0.0, "posterior-only should not use query message.")

    query_only = params_by_variant["E_query_only"]
    _assert(float(query_only["personal_delta_scale"]) == 0.0, "query-only should not change posterior.")
    _assert(float(query_only["personal_query_correction_scale"]) > 0.0, "query-only should keep query correction active.")
    _assert(float(query_only["personal_query_message_gain"]) > 0.0, "query-only should keep query message active.")


def _check_phase2_full_cannot_mute_posterior_route() -> None:
    params_by_variant = _params_by_variant("full,E_shuffle_student,no_E", phase="phase2")
    for variant in ("full", "E_shuffle_student"):
        params = params_by_variant[variant]
        _assert(bool(params["use_personal_graph"]), f"{variant} should keep E enabled.")
        _assert(float(params["personal_delta_scale"]) > 0.0, f"{variant} should keep posterior route movable.")
        _assert(
            float(params["ae_posterior_prior_scale"]) > 0.0 or float(params["ae_posterior_theta_scale"]) > 0.0,
            f"{variant} should keep an E posterior write path active.",
        )
    _assert(not bool(params_by_variant["no_E"]["use_personal_graph"]), "phase2 no_E should disable E.")


def main() -> None:
    _check_phase1_full_activates_e_writing()
    _check_controls_keep_their_semantics()
    _check_phase2_full_cannot_mute_posterior_route()
    print("OK: mechanism params keep E active only where intended.")


if __name__ == "__main__":
    main()
