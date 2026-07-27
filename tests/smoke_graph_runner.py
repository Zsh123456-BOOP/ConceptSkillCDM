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
    assert runner.EXPERIMENT_CONFIGS["junyi"]["knowledge_dim"] == 64
    assert runner.EXPERIMENT_CONFIGS["junyi"]["exposure_prior_pmi"] is True
    assert runner.EXPERIMENT_CONFIGS["nips34"]["dropout"] == 0.10
    assert runner.EXPERIMENT_CONFIGS["nips34"]["epochs"] == 120
    assert runner.EXPERIMENT_CONFIGS["nips34"]["early_stop_patience"] == 15

    args = runner.parse_args(
        [
            "--datasets",
            "assist_09",
            "--seeds",
            "42",
            "--ablations",
            "full,no_response_evidence,pairwise_auc,no_message_passing,"
            "no_graph_calibration,mec,mec_state_graph,"
            "item_only,exposure_only,degree_random",
            "--warm_start_run_id",
            "baseline_smoke",
            "--dry_run",
        ]
    )
    assert args.run_mode == "train"
    jobs = runner.make_jobs(args, run_id="smoke")
    assert len(jobs) == 10
    assert {job.train_evidence_mode for job in jobs} == {"excluded"}
    by_name = {job.ablation.name: job for job in jobs}
    no_evidence_command = by_name["no_response_evidence"].cmd
    parsed_no_evidence = main_module.parse_args(
        no_evidence_command[no_evidence_command.index("--dataset_name") :]
    )
    main_module._apply_model_variant(parsed_no_evidence)
    assert not parsed_no_evidence.use_response_evidence
    pairwise_command = by_name["pairwise_auc"].cmd
    assert "--model_variant" in pairwise_command
    assert pairwise_command[pairwise_command.index("--model_variant") + 1] == "pairwise_auc"
    parsed = main_module.parse_args(
        pairwise_command[pairwise_command.index("--dataset_name") :]
    )
    main_module._apply_model_variant(parsed)
    assert parsed.pairwise_auc_weight == 0.5
    assert by_name["no_message_passing"].params["graph_propagation_alpha"] == 0.0
    assert by_name["no_graph_calibration"].params["graph_propagation_alpha"] == 0.0
    assert by_name["mec"].params["graph_propagation_alpha"] == 0.0
    assert by_name["mec_state_graph"].params["graph_propagation_alpha"] > 0.0
    assert by_name["mec"].params["learning_rate"] == 0.003
    assert by_name["mec_state_graph"].params["optimizer"] == "adamw"
    assert "no_graph_calibration" in by_name["mec"].params[
        "warm_start_checkpoint"
    ]
    assert "no_evidence_propagation" in by_name["mec_state_graph"].params[
        "warm_start_checkpoint"
    ]
    assert by_name["item_only"].params["graph_prior_mode"] == "item_only"
    assert by_name["exposure_only"].params["graph_prior_mode"] == "exposure_only"
    assert by_name["degree_random"].params["graph_prior_mode"] == "degree_random"

    boundary_args = runner.parse_args(
        [
            "--datasets",
            "assist_09",
            "--seeds",
            "42,43",
            "--ablations",
            "full",
            "--train_evidence_modes",
            "neutralized,self_included",
            "--dry_run",
        ]
    )
    boundary_jobs = runner.make_jobs(boundary_args, run_id="boundary_smoke")
    assert len(boundary_jobs) == 4
    assert {job.train_evidence_mode for job in boundary_jobs} == {
        "neutralized",
        "self_included",
    }
    assert len({job.save_dir for job in boundary_jobs}) == len(boundary_jobs)
    for job in boundary_jobs:
        parsed_boundary = main_module.parse_args(
            job.cmd[job.cmd.index("--dataset_name") :]
        )
        assert parsed_boundary.train_evidence_mode == job.train_evidence_mode

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
            "full,pairwise_auc",
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

        fresh_job = replace(
            jobs[0],
            save_dir=Path(directory) / "fresh_checkpoint",
            log_dir=Path(directory) / "fresh_log",
        )
        runner._require_fresh_training_dirs([fresh_job])
        fresh_job.save_dir.mkdir()
        try:
            runner._require_fresh_training_dirs([fresh_job])
        except FileExistsError as exc:
            assert "choose a new --run_id" in str(exc)
        else:
            raise AssertionError("training must refuse an existing run directory")

        warm_path = Path(directory) / "warm" / "best_model.pth"
        missing_warm_job = replace(
            by_name["mec_state_graph"],
            params={
                **by_name["mec_state_graph"].params,
                "warm_start_checkpoint": str(warm_path),
            },
        )
        try:
            runner._require_mec_warm_starts([missing_warm_job])
        except FileNotFoundError as exc:
            assert "matched warm-start" in str(exc)
        else:
            raise AssertionError("MEC must reject a missing matched baseline")
        warm_path.parent.mkdir(parents=True)
        warm_path.write_bytes(b"smoke")
        runner._require_mec_warm_starts([missing_warm_job])
    print("OK: Graph-IRT ablation runner contract passed.")


if __name__ == "__main__":
    main()
