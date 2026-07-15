"""Dry-run contract for the clean graph ablation launcher."""

import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path


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
            "full,no_response_evidence,no_pairwise_loss,no_message_passing,item_only,exposure_only,degree_random",
            "--dry_run",
        ]
    )
    assert args.run_mode == "train"
    jobs = runner.make_jobs(args, run_id="smoke")
    assert len(jobs) == 7
    by_name = {job.ablation.name: job for job in jobs}
    no_evidence_command = by_name["no_response_evidence"].cmd
    parsed_no_evidence = main_module.parse_args(
        no_evidence_command[no_evidence_command.index("--dataset_name") :]
    )
    main_module._apply_model_variant(parsed_no_evidence)
    assert not parsed_no_evidence.use_response_evidence
    no_pairwise_command = by_name["no_pairwise_loss"].cmd
    assert "--model_variant" in no_pairwise_command
    assert no_pairwise_command[no_pairwise_command.index("--model_variant") + 1] == "no_pairwise_loss"
    parsed = main_module.parse_args(
        no_pairwise_command[no_pairwise_command.index("--dataset_name") :]
    )
    main_module._apply_model_variant(parsed)
    assert parsed.pairwise_auc_weight == 0.0
    assert by_name["no_message_passing"].params["graph_propagation_alpha"] == 0.0
    assert by_name["item_only"].params["graph_prior_mode"] == "item_only"
    assert by_name["exposure_only"].params["graph_prior_mode"] == "exposure_only"
    assert by_name["degree_random"].params["graph_prior_mode"] == "degree_random"

    banned = ("no_A", "no_E", "personal", "ae_", "roadmap", "tutor")
    for job in jobs:
        command = " ".join(job.cmd)
        assert not any(token in command for token in banned), command
        run_mode_index = job.cmd.index("--run_mode")
        assert job.cmd[run_mode_index + 1] == "train"

    test_args = runner.parse_args(
        [
            "--datasets",
            "assist_09",
            "--seeds",
            "42",
            "--ablations",
            "full,no_pairwise_loss",
            "--run_mode",
            "test",
            "--run_id",
            "smoke",
            "--dry_run",
        ]
    )
    test_jobs = runner.make_jobs(test_args)
    assert len(test_jobs) == 2
    for job in test_jobs:
        run_mode_index = job.cmd.index("--run_mode")
        assert job.cmd[run_mode_index + 1] == "test"
        parsed_test = main_module.parse_args(
            job.cmd[job.cmd.index("--dataset_name") :]
        )
        assert parsed_test.run_mode == "test"

    missing_run_id = runner.parse_args(
        ["--datasets", "assist_09", "--run_mode", "test", "--dry_run"]
    )
    try:
        runner.make_jobs(missing_run_id)
    except ValueError as exc:
        assert "requires --run_id" in str(exc)
    else:
        raise AssertionError("test phase must name an existing training run id")

    with tempfile.TemporaryDirectory() as directory:
        checkpoint_dir = Path(directory) / "checkpoint"
        checkpoint_job = replace(test_jobs[0], save_dir=checkpoint_dir)
        try:
            runner._require_existing_test_checkpoints([checkpoint_job])
        except FileNotFoundError as exc:
            assert "best_model.pth" in str(exc)
        else:
            raise AssertionError("test phase must reject a missing checkpoint")
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "best_model.pth").write_bytes(b"smoke")
        runner._require_existing_test_checkpoints([checkpoint_job])
    print("OK: Graph-IRT ablation runner contract passed.")


if __name__ == "__main__":
    main()
