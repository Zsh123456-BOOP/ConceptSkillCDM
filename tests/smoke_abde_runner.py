import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _check_run_abce_supports_abde() -> None:
    import run_abce_ablation as abce_runner

    ablations = abce_runner.pick_base_ablations("single_plus")
    names = {ab.name for ab in ablations}
    _assert("full" in names, "run_abce_ablation should include full.")
    _assert("no_A" in names, "run_abce_ablation should include no_A.")
    _assert("no_B" in names, "run_abce_ablation should include no_B.")
    _assert("no_D" in names, "run_abce_ablation should include no_D.")
    _assert("no_E" in names, "run_abce_ablation should include no_E.")

    args = abce_runner.parse_args()
    args.datasets = "assist_09"
    args.seeds = "42"
    args.profiles = "best"
    args.component_set = "single_plus"
    args.ablations = "no_D"
    args.dry_run = True
    args.rerun_existing = True
    args.generate_diagnosis = False

    jobs = abce_runner.make_jobs(args, run_id="smoke_abde")
    _assert(len(jobs) == 1, f"Expected exactly one no_D job, got {len(jobs)}.")
    job = jobs[0]
    _assert(job.ablation.name == "no_D", "Expected no_D ablation job.")
    _assert("--ablate_module2" in job.cmd, "no_D job should pass --ablate_module2.")
    _assert("--ablate_module3" not in job.cmd, "no_D job must not also disable module3.")


def _check_run_ablation_accepts_abde_aliases() -> None:
    import run_ablation

    resolved = run_ablation.resolve_ablation_names(["no_A", "no_B", "no_D", "no_E"], ablation_set="all")
    _assert("no_concept_graph" in resolved, "no_A alias should map to no_concept_graph.")
    _assert("no_skill" in resolved, "no_B alias should map to no_skill.")
    _assert("no_module2" in resolved, "no_D alias should map to no_module2.")
    _assert("no_personal_graph" in resolved, "no_E alias should map to no_personal_graph.")


def main() -> None:
    _check_run_abce_supports_abde()
    _check_run_ablation_accepts_abde_aliases()
    print("OK: ABDE runners smoke checks passed.")


if __name__ == "__main__":
    main()
