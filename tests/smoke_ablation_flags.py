import os
import subprocess
import sys
import tempfile


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _deps_available() -> bool:
    for mod in ("torch", "pandas", "numpy"):
        try:
            __import__(mod)
        except ModuleNotFoundError:
            return False
    return True


def _check_argparse_rejects_legacy_flag() -> None:
    if not _deps_available():
        print("SKIP: torch/pandas/numpy not installed; argparse check skipped.")
        return

    cmd = [sys.executable, os.path.join(ROOT, "main.py"), "--ablate_exercise_graph"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        raise AssertionError("Expected non-zero exit code for --ablate_exercise_graph.")
    combined = (result.stderr or "") + (result.stdout or "")
    if "Use --ablate_concept_graph instead." not in combined:
        raise AssertionError("Expected guidance to use --ablate_concept_graph.")


def _check_csv_migration_tool() -> None:
    try:
        import pandas as pd
    except ModuleNotFoundError:
        print("SKIP: pandas not installed; CSV migration check skipped.")
        return

    sys.path.insert(0, ROOT)
    from tools.migrate_ablation_csv import migrate_csv

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "old.csv")
        out_path = os.path.join(tmpdir, "new.csv")
        df = pd.DataFrame(
            [
                {
                    "ablate_soft_prototype": False,
                    "ablate_skill_encoder": False,
                    "ablate_exercise_graph": True,
                    "ablation_flags": "ablate_soft_prototype=False;ablate_skill_encoder=False;ablate_exercise_graph=True",
                }
            ]
        )
        df.to_csv(in_path, index=False)
        migrate_csv(in_path, out_path)
        out = pd.read_csv(out_path)
        if "ablate_exercise_graph" in out.columns:
            raise AssertionError("ablate_exercise_graph column should be removed.")
        if "ablate_concept_graph" not in out.columns:
            raise AssertionError("ablate_concept_graph column should exist.")
        if "ablate_concept_graph=True" not in str(out.loc[0, "ablation_flags"]):
            raise AssertionError("ablation_flags should be normalized to ablate_concept_graph.")


def main() -> None:
    _check_argparse_rejects_legacy_flag()
    _check_csv_migration_tool()
    print("OK: ablation flag cleanup smoke checks passed.")


if __name__ == "__main__":
    main()
