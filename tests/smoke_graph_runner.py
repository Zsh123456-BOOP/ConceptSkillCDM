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
            "hadamard",
            "--student_concept_interaction_scale",
            "1.0",
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
        assert job.params["student_concept_interaction"] == "hadamard"
        assert job.params["student_concept_interaction_scale"] == 1.0
        command = " ".join(job.cmd)
        assert not any(token in command for token in banned), command
    print("OK: Graph-IRT ablation runner contract passed.")


if __name__ == "__main__":
    main()
