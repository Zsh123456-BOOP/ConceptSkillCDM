"""Dry-run contract for the clean graph ablation launcher."""

import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import run_graph_ablation as runner
import main as main_module


def main() -> None:
    args = runner.parse_args(
        [
            "--datasets",
            "assist_09",
            "--seeds",
            "42",
            "--ablations",
            "full,no_item_matching,no_message_passing,item_only,exposure_only,degree_random",
            "--dry_run",
        ]
    )
    jobs = runner.make_jobs(args, run_id="smoke")
    assert len(jobs) == 6
    by_name = {job.ablation.name: job for job in jobs}
    assert by_name["no_item_matching"].params["enable_item_matching"] is False
    no_item_command = by_name["no_item_matching"].cmd
    assert "--model_variant" in no_item_command
    assert no_item_command[no_item_command.index("--model_variant") + 1] == "no_item_matching"
    parsed = main_module.parse_args(
        no_item_command[no_item_command.index("--dataset_name") :]
    )
    main_module._apply_model_variant(parsed)
    assert parsed.enable_item_matching is False
    assert by_name["no_message_passing"].params["graph_propagation_alpha"] == 0.0
    assert by_name["item_only"].params["graph_prior_mode"] == "item_only"
    assert by_name["exposure_only"].params["graph_prior_mode"] == "exposure_only"
    assert by_name["degree_random"].params["graph_prior_mode"] == "degree_random"

    banned = ("no_A", "no_E", "personal", "ae_", "roadmap", "tutor")
    for job in jobs:
        command = " ".join(job.cmd)
        assert not any(token in command for token in banned), command
    print("OK: Graph-IRT ablation runner contract passed.")


if __name__ == "__main__":
    main()
