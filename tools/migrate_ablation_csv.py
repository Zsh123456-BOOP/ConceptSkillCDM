import argparse
import os

import pandas as pd


def _normalize_ablation_flags(value: str) -> str:
    if not isinstance(value, str):
        return value
    parts = [p for p in value.split(";") if p]
    if not parts:
        return value

    normalized = []
    seen = {}
    for part in parts:
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        if key == "ablate_exercise_graph":
            key = "ablate_concept_graph"
        seen[key] = val

    ordered_keys = [
        "ablate_soft_prototype",
        "ablate_skill_encoder",
        "ablate_concept_graph",
    ]
    for key in ordered_keys:
        if key in seen:
            normalized.append(f"{key}={seen.pop(key)}")
    for key, val in seen.items():
        normalized.append(f"{key}={val}")
    return ";".join(normalized)


def migrate_csv(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path)

    if "ablate_exercise_graph" in df.columns:
        if "ablate_concept_graph" not in df.columns:
            df["ablate_concept_graph"] = df["ablate_exercise_graph"]
        df = df.drop(columns=["ablate_exercise_graph"])

    if "ablation_flags" in df.columns:
        df["ablation_flags"] = df["ablation_flags"].apply(_normalize_ablation_flags)

    df.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate ablation CSV to use ablate_concept_graph only.")
    parser.add_argument("--input", required=True, help="Path to existing experiment_results.csv")
    parser.add_argument("--output", default=None, help="Path to migrated CSV (default: <input>_migrated.csv)")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    if output_path is None:
        root, ext = os.path.splitext(input_path)
        output_path = f"{root}_migrated{ext or '.csv'}"

    migrate_csv(input_path, output_path)
    print(f"Migrated CSV saved to {output_path}")


if __name__ == "__main__":
    main()
