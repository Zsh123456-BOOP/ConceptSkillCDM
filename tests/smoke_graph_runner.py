"""Dry-run contract for the clean graph ablation launcher."""

import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import run_graph_ablation as runner


def main() -> None:
    args = runner.parse_args(
        [
            "--datasets",
            "assist_09",
            "--seeds",
            "42",
            "--ablations",
            "full,no_message_passing,item_only,exposure_only,degree_random",
            "--student_concept_interaction",
            "low_rank",
            "--student_concept_interaction_scale",
            "0.75",
            "--student_concept_interaction_ratio_cap",
            "0.3",
            "--student_concept_interaction_rank",
            "4",
            "--student_concept_interaction_init_std",
            "0.05",
            "--dry_run",
        ]
    )
    jobs = runner.make_jobs(args, run_id="smoke")
    assert len(jobs) == 5
    by_name = {job.ablation.name: job for job in jobs}
    assert by_name["no_message_passing"].params["graph_propagation_alpha"] == 0.0
    assert by_name["item_only"].params["graph_prior_mode"] == "item_only"
    assert by_name["exposure_only"].params["graph_prior_mode"] == "exposure_only"
    assert by_name["degree_random"].params["graph_prior_mode"] == "degree_random"

    banned = ("no_A", "no_E", "personal", "ae_", "roadmap", "tutor")
    for job in jobs:
        assert job.params["student_concept_interaction"] == "low_rank"
        assert job.params["student_concept_interaction_scale"] == 0.75
        assert job.params["student_concept_interaction_ratio_cap"] == 0.3
        assert job.params["student_concept_interaction_rank"] == 4
        assert job.params["student_concept_interaction_init_std"] == 0.05
        command = " ".join(job.cmd)
        assert "--student_concept_interaction low_rank" in command
        assert "--student_concept_interaction_ratio_cap 0.3" in command
        assert "--student_concept_interaction_rank 4" in command
        assert "--student_concept_interaction_init_std 0.05" in command
        assert not any(token in command for token in banned), command

    for parameter in (
        "student_concept_interaction_scale",
        "student_concept_interaction_init_std",
    ):
        zero_args = runner.parse_args(
            [
                "--datasets",
                "assist_09",
                "--seeds",
                "42",
                "--ablations",
                "full",
                "--student_concept_interaction",
                "low_rank",
                f"--{parameter}",
                "0",
            ]
        )
        try:
            runner.make_jobs(zero_args, run_id="invalid")
        except ValueError:
            pass
        else:
            raise AssertionError(f"low_rank runner must reject zero {parameter}")
    for invalid_cap in ("nan", "-0.1", "4.1"):
        invalid_cap_args = runner.parse_args(
            [
                "--datasets",
                "assist_09",
                "--seeds",
                "42",
                "--ablations",
                "full",
                "--student_concept_interaction_ratio_cap",
                invalid_cap,
            ]
        )
        try:
            runner.make_jobs(invalid_cap_args, run_id="invalid_cap")
        except ValueError:
            pass
        else:
            raise AssertionError(f"runner must reject ratio cap {invalid_cap}")
    for inactive_mode in ("none", "hadamard"):
        inactive_zero_args = runner.parse_args(
            [
                "--datasets",
                "assist_09",
                "--seeds",
                "42",
                "--ablations",
                "full",
                "--student_concept_interaction",
                inactive_mode,
                "--student_concept_interaction_scale",
                "0",
                "--student_concept_interaction_init_std",
                "0",
            ]
        )
        inactive_jobs = runner.make_jobs(inactive_zero_args, run_id="inactive_zero")
        assert inactive_jobs[0].params["student_concept_interaction_scale"] == 0.0
        assert inactive_jobs[0].params["student_concept_interaction_init_std"] == 0.0
    print("OK: Graph-IRT ablation runner contract passed.")


if __name__ == "__main__":
    main()
