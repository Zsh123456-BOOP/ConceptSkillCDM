#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Build concept-heldout splits from existing processed benchmark CSVs.

This implements the student-conditioned split requested for the main-problem
experiment on the original project datasets, not EdNet.  It merges the existing
train/valid/test CSVs, shuffles each student's records, then assigns records by
held-out concept groups while approximating a 7:1:2 split.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_ednet_concept_heldout_from_raw import (  # noqa: E402
    PROJECT_COLUMNS,
    SPLITS,
    aggregate_student_audit,
    assign_concept_heldout,
    gap_profile,
    split_stats,
)

def read_existing_dataset(data_dir: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for split in SPLITS:
        path = data_dir / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        missing = [col for col in PROJECT_COLUMNS if col not in frame.columns]
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")
        frame = frame[PROJECT_COLUMNS].copy()
        frame["_source_split"] = split
        frame["_source_row"] = range(len(frame))
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out["stu_id"] = out["stu_id"].astype(int)
    out["exer_id"] = out["exer_id"].astype(int)
    out["label"] = out["label"].astype(int)
    out["cpt_seq"] = out["cpt_seq"].astype(str)
    return out


def student_rows_from_frame(frame: pd.DataFrame, *, shuffle: bool, rng: random.Random) -> List[Tuple[int, int, str, int]]:
    rows: List[Tuple[int, int, str, int]] = []
    ordered = frame.sort_values(["_source_split", "_source_row"], kind="stable")
    tuples = [
        (int(row.exer_id), str(row.cpt_seq), int(row.label))
        for row in ordered.itertuples(index=False)
    ]
    if shuffle:
        rng.shuffle(tuples)
    for pos, (exer_id, cpt_seq, label) in enumerate(tuples):
        rows.append((pos, exer_id, cpt_seq, label))
    return rows


def build_split(args: argparse.Namespace) -> Dict[str, Any]:
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_existing_dataset(source_dir)
    rng = random.Random(args.seed)
    split_rows: Dict[str, List[Tuple[int, int, str, int]]] = {s: [] for s in SPLITS}
    audits = []
    skipped = 0
    kept_students = 0
    for mapped_stu, (stu_id, part) in enumerate(df.groupby("stu_id", sort=True)):
        if len(part) < args.min_interactions:
            skipped += 1
            continue
        rows = student_rows_from_frame(part, shuffle=not args.no_shuffle, rng=random.Random(args.seed + int(stu_id)))
        assigned = assign_concept_heldout(
            rows,
            train_frac=args.train_frac,
            valid_frac=args.valid_frac,
            min_unique_concepts=args.min_unique_concepts,
            rng=random.Random(args.seed + 100000 + int(stu_id)),
            tolerance=args.concept_tolerance,
        )
        if assigned is None:
            skipped += 1
            continue
        assignment, audit = assigned
        for split in SPLITS:
            for idx in assignment[split]:
                _, exer_id, cpt_seq, label = rows[idx]
                split_rows[split].append((kept_students, exer_id, cpt_seq, label))
        audits.append(audit)
        kept_students += 1
        if kept_students % 1000 == 0:
            print(f"[build] kept_students={kept_students} skipped={skipped}", flush=True)

    stats = {}
    for split in SPLITS:
        out = pd.DataFrame(split_rows[split], columns=PROJECT_COLUMNS)
        out.to_csv(output_dir / f"{split}.csv", index=False)
        stats[split] = asdict(split_stats(split_rows[split]))

    manifest = {
        "dataset": args.output_dataset,
        "source_dataset": args.source_dataset,
        "source_dir": str(source_dir),
        "split_policy": (
            "merge existing train/valid/test; shuffle records within each student; "
            "assign held-out concept groups per student to approximate train/valid/test ratios"
        ),
        "has_true_temporal_order": False,
        "shuffle_within_student": not args.no_shuffle,
        "filter": {
            "min_interactions": args.min_interactions,
            "min_unique_concepts": args.min_unique_concepts,
            "concept_tolerance": args.concept_tolerance,
            "source_students": int(df["stu_id"].nunique()),
            "selected_students": kept_students,
            "skipped_students": skipped,
            "seed": args.seed,
        },
        "target_ratios": {
            "train_frac": args.train_frac,
            "valid_frac": args.valid_frac,
            "test_frac": max(0.0, 1.0 - args.train_frac - args.valid_frac),
        },
        "splits": stats,
        "gap_profile": gap_profile(split_rows),
        "student_split_audit": aggregate_student_audit(audits),
        "output_format": PROJECT_COLUMNS,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--valid-frac", type=float, default=0.10)
    parser.add_argument("--min-interactions", type=int, default=15)
    parser.add_argument("--min-unique-concepts", type=int, default=3)
    parser.add_argument("--concept-tolerance", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    build_split(parse_args())


if __name__ == "__main__":
    main()
