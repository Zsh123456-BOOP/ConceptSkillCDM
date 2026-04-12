import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _check_run_abce_supports_ae_only() -> None:
    import run_abce_ablation as runner

    ablations = runner.pick_base_ablations("single")
    names = {ab.name for ab in ablations}
    _assert(names == {"full", "no_A", "no_E"}, f"unexpected AE-only ablations: {sorted(names)}")

    args = runner.parse_args()
    args.datasets = "assist_09"
    args.seeds = "42"
    args.profiles = "best"
    args.component_set = "single"
    args.ablations = "full,no_A,no_E"
    args.dry_run = True
    args.rerun_existing = True
    args.generate_diagnosis = False

    jobs = runner.make_jobs(args, run_id="smoke_ae_only")
    _assert(len(jobs) == 3, f"expected exactly three AE-only jobs, got {len(jobs)}")

    banned_tokens = {
        "--ablate_module2",
        "--ablate_module3",
        "--ablate_skill_encoder",
        "--disable_q_conditioning",
        "--disable_b_id_adapter",
        "--disable_b_bias",
        "--fusion_gate_max",
        "--fusion_gate_bias_init",
        "--residual_scale_init",
    }
    for job in jobs:
        overlap = sorted(token for token in banned_tokens if token in job.cmd)
        _assert(not overlap, f"{job.ablation.name} should not carry removed B/D flags: {overlap}")


def _check_run_ablation_rejects_removed_aliases() -> None:
    import run_ablation

    resolved = run_ablation.resolve_ablation_names(["no_A", "no_E"], ablation_set="all")
    _assert("no_concept_graph" in resolved, "no_A alias should map to no_concept_graph.")
    _assert("no_personal_graph" in resolved, "no_E alias should map to no_personal_graph.")

    for removed in ("no_B", "no_D"):
        try:
            run_ablation.resolve_ablation_names([removed], ablation_set="all")
        except ValueError:
            continue
        raise AssertionError(f"{removed} should be rejected after removing B/D ablations.")


def main() -> None:
    _check_run_abce_supports_ae_only()
    _check_run_ablation_rejects_removed_aliases()
    print("OK: AE-only runner smoke checks passed.")


if __name__ == "__main__":
    main()
