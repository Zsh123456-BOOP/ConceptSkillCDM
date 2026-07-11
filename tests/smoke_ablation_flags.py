"""CLI/config parity and legacy-flag rejection checks."""

import contextlib
import io
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import main as main_module
from experiment_configs import EXPERIMENT_CONFIGS


REMOVED_FLAGS = (
    "--use_personal_graph",
    "--use_concept_graph",
    "--ablate_module1",
    "--ablate_concept_graph",
    "--disable_item_prior",
    "--disable_sequence_prior",
    "--support_only_unseen",
    "--route_mastery_unseen",
    "--ae_stat_prior_scale",
    "--roadmap_logit_residual_scale",
    "--relation_theta_scale",
    "--graph_prior_logit_scale",
    "--lambda_sparse",
    "--student_concept_interaction",
    "--student_concept_interaction_scale",
    "--student_concept_interaction_ratio_cap",
    "--student_concept_interaction_rank",
    "--student_concept_interaction_init_std",
    "--enable_item_matching",
    "--item_matching_rank",
)


def _parse(argv):
    return main_module.parse_args(argv)


def _assert_legacy_flags_are_rejected() -> None:
    for flag in REMOVED_FLAGS:
        stderr = io.StringIO()
        try:
            with contextlib.redirect_stderr(stderr):
                _parse([flag])
        except SystemExit as exc:
            assert exc.code != 0, flag
        else:
            raise AssertionError(f"legacy flag should be rejected: {flag}")
        assert "unrecognized arguments" in stderr.getvalue(), (flag, stderr.getvalue())


def _append_arg(argv, key, value):
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            argv.append(f"--{key}")
        return
    argv.extend((f"--{key}", str(value)))


def _assert_experiment_configs_match_parser() -> None:
    for dataset, config in EXPERIMENT_CONFIGS.items():
        argv = ["--dataset_name", dataset]
        for key, value in config.items():
            if key == "num_gpus":
                continue
            _append_arg(argv, key, value)
        parsed = _parse(argv)
        assert parsed.dataset_name == dataset


def main() -> None:
    _assert_legacy_flags_are_rejected()
    _assert_experiment_configs_match_parser()
    print("OK: CLI cleanup and experiment-config parity passed.")


if __name__ == "__main__":
    main()
