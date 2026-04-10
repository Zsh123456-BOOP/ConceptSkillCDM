import os
import sys

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _check_qaware_model_details() -> None:
    from src.model import CognitiveDiagnosisModel

    q_matrix = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    model = CognitiveDiagnosisModel(
        num_students=5,
        num_exercises=3,
        num_concepts=4,
        q_matrix=q_matrix,
        knowledge_dim=8,
        exercise_dim=8,
        skill_dim=8,
        num_relation_heads=2,
        num_gnn_layers=1,
        dropout=0.0,
        use_personal_graph=False,
        fusion_gate_max=0.25,
        fusion_gate_bias_init=-2.0,
        residual_scale_init=0.05,
    )
    model.eval()

    student_ids = torch.tensor([0, 1, 2])
    exercise_ids = torch.tensor([0, 1, 2])
    logits, details = model(
        student_ids,
        exercise_ids,
        return_details=True,
        return_logits=True,
    )

    _assert(logits.shape == (3,), "model should return one logit per example.")
    for key in (
        "q_interaction_logit",
        "id_adapter_logit",
        "bias_logit",
        "b_q_share",
        "b_id_share",
        "student_q_norm",
        "student_id_adapter_norm",
        "item_q_norm",
        "item_id_adapter_norm",
    ):
        _assert(key in details, f"missing q-aware residual detail: {key}")

    _assert(details["q_interaction_logit"].shape == (3,), "q interaction should be per example.")
    _assert(details["id_adapter_logit"].shape == (3,), "id adapter should be per example.")
    _assert(details["bias_logit"].shape == (3,), "bias logit should be per example.")
    _assert(float(details["student_q_norm"].mean()) > 0.0, "student Q representation should be non-zero.")
    _assert(float(details["item_q_norm"].mean()) > 0.0, "item Q representation should be non-zero.")
    _assert(float(details["b_q_share"]) >= 0.0, "B q share should be non-negative.")
    _assert(float(details["b_id_share"]) >= 0.0, "B id share should be non-negative.")


def _check_runner_qaware_ablations() -> None:
    import run_abce_ablation as runner

    names = {ab.name for ab in runner.pick_base_ablations("single_plus")}
    _assert("B_q_only" in names, "runner should expose B_q_only.")
    _assert("B_no_q" in names, "runner should expose B_no_q.")

    args = runner.parse_args()
    args.datasets = "assist_09"
    args.seeds = "42"
    args.profiles = "ae_dominant"
    args.component_set = "single_plus"
    args.ablations = "B_q_only,B_no_q,no_D"
    args.dry_run = True
    args.rerun_existing = True
    args.generate_diagnosis = False

    jobs = runner.make_jobs(args, run_id="smoke_qaware")
    _assert(len(jobs) == 3, f"expected 3 q-aware jobs, got {len(jobs)}")
    by_name = {job.ablation.name: job for job in jobs}
    _assert("--disable_b_id_adapter" in by_name["B_q_only"].cmd, "B_q_only should disable id adapter.")
    _assert("--disable_b_bias" in by_name["B_q_only"].cmd, "B_q_only should disable residual bias.")
    _assert("--disable_q_conditioning" in by_name["B_no_q"].cmd, "B_no_q should disable Q path.")
    _assert("--ablate_module2" in by_name["no_D"].cmd, "no_D should disable module2.")


def main() -> None:
    _check_qaware_model_details()
    _check_runner_qaware_ablations()
    print("OK: q-aware residual smoke checks passed.")


if __name__ == "__main__":
    main()
