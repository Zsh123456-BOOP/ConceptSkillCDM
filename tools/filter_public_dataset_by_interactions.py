#!/usr/bin/env python3
"""Filter a processed public benchmark by train-only interaction frequency.

The script is intended for very large processed datasets such as EdNet-KT1.
It keeps the same CSV schema used by the project:

    stu_id,exer_id,cpt_seq,label

Filtering is determined only from the training split:

1. keep the most frequent train items, matching the reference dataset's train
   item count by default;
2. within those items, keep high-interaction students until the filtered train
   row count is close to the reference train row count;
3. apply the selected train students/items to train/valid/test.

This avoids random downsampling while removing low-evidence students/items.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_COLUMNS = ["stu_id", "exer_id", "cpt_seq", "label"]
SPLITS = ("train", "valid", "test")


@dataclass
class SplitStats:
    rows: int = 0
    students: int = 0
    items: int = 0
    concepts: int = 0
    positive_rate: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--dataset", default="ednet_kt1")
    parser.add_argument("--reference-dataset", default="junyi")
    parser.add_argument("--output-dataset", default=None)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--target-train-scale", type=float, default=1.0)
    parser.add_argument("--target-item-scale", type=float, default=1.0)
    parser.add_argument("--min-item-interactions", type=int, default=50)
    parser.add_argument("--min-selected-students", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Replace data/<dataset> CSVs. The old manifest is preserved, but large CSVs are not duplicated.",
    )
    return parser.parse_args()


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return next(csv.reader(f))


def iter_chunks(path: Path, *, usecols: list[str] | None, chunksize: int) -> Iterable[pd.DataFrame]:
    return pd.read_csv(path, usecols=usecols, chunksize=chunksize)


def count_rows(path: Path) -> int:
    with path.open("rb") as f:
        return max(0, sum(1 for _ in f) - 1)


def split_summary(dataset_dir: Path) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        path = dataset_dir / f"{split}.csv"
        header = read_header(path)
        if header != PROJECT_COLUMNS:
            raise ValueError(f"{path} columns mismatch: {header} != {PROJECT_COLUMNS}")
        df = pd.read_csv(path, usecols=["stu_id", "exer_id"])
        summary[split] = {
            "rows": int(len(df)),
            "students": int(df["stu_id"].nunique()),
            "items": int(df["exer_id"].nunique()),
        }
    return summary


def count_train_entities(train_path: Path, chunksize: int) -> tuple[Counter[int], Counter[int], int]:
    student_counts: Counter[int] = Counter()
    item_counts: Counter[int] = Counter()
    rows = 0
    for chunk in iter_chunks(train_path, usecols=["stu_id", "exer_id"], chunksize=chunksize):
        rows += int(len(chunk))
        student_counts.update({int(k): int(v) for k, v in chunk["stu_id"].value_counts().items()})
        item_counts.update({int(k): int(v) for k, v in chunk["exer_id"].value_counts().items()})
    return student_counts, item_counts, rows


def select_items(
    item_counts: Counter[int],
    *,
    target_item_count: int,
    min_item_interactions: int,
) -> set[int]:
    candidates = [
        (item, count)
        for item, count in item_counts.items()
        if count >= min_item_interactions
    ]
    candidates.sort(key=lambda pair: (-pair[1], pair[0]))
    if target_item_count > 0:
        candidates = candidates[:target_item_count]
    return {int(item) for item, _ in candidates}


def count_students_on_items(train_path: Path, kept_items: set[int], chunksize: int) -> Counter[int]:
    student_counts: Counter[int] = Counter()
    for chunk in iter_chunks(train_path, usecols=["stu_id", "exer_id"], chunksize=chunksize):
        filtered = chunk[chunk["exer_id"].isin(kept_items)]
        if filtered.empty:
            continue
        student_counts.update({int(k): int(v) for k, v in filtered["stu_id"].value_counts().items()})
    return student_counts


def select_students(
    eligible_student_counts: Counter[int],
    *,
    target_train_rows: int,
    min_selected_students: int,
) -> tuple[set[int], int, int]:
    ranked = sorted(eligible_student_counts.items(), key=lambda pair: (-pair[1], pair[0]))
    kept: set[int] = set()
    total = 0
    cutoff = 0
    for student, count in ranked:
        if total >= target_train_rows and len(kept) >= min_selected_students:
            break
        kept.add(int(student))
        total += int(count)
        cutoff = int(count)
    return kept, total, cutoff


def parse_concepts(series: pd.Series) -> set[int]:
    concepts: set[int] = set()
    for seq in series.dropna().astype(str):
        for part in seq.split(","):
            part = part.strip()
            if part:
                concepts.add(int(part))
    return concepts


def write_filtered_split(
    src_path: Path,
    dst_path: Path,
    *,
    kept_students: set[int],
    kept_items: set[int],
    chunksize: int,
) -> SplitStats:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    stats = SplitStats()
    students: set[int] = set()
    items: set[int] = set()
    concepts: set[int] = set()
    positives = 0
    wrote_header = False
    for chunk in iter_chunks(src_path, usecols=PROJECT_COLUMNS, chunksize=chunksize):
        filtered = chunk[
            chunk["stu_id"].isin(kept_students)
            & chunk["exer_id"].isin(kept_items)
        ].copy()
        if filtered.empty:
            continue
        filtered = filtered[PROJECT_COLUMNS]
        filtered.to_csv(dst_path, mode="a", index=False, header=not wrote_header)
        wrote_header = True
        stats.rows += int(len(filtered))
        positives += int(filtered["label"].sum())
        students.update(int(v) for v in filtered["stu_id"].unique())
        items.update(int(v) for v in filtered["exer_id"].unique())
        concepts.update(parse_concepts(filtered["cpt_seq"]))

    if not wrote_header:
        pd.DataFrame(columns=PROJECT_COLUMNS).to_csv(dst_path, index=False)

    stats.students = len(students)
    stats.items = len(items)
    stats.concepts = len(concepts)
    stats.positive_rate = float(positives / stats.rows) if stats.rows else 0.0
    return stats


def file_info(path: Path, *, manifest_path: Path | None = None) -> dict[str, int | str]:
    size = path.stat().st_size
    shown_path = manifest_path if manifest_path is not None else path
    return {"path": shown_path.as_posix(), "bytes": int(size)}


def write_manifest(
    out_dir: Path,
    *,
    dataset: str,
    manifest_data_dir: Path | None = None,
    reference_dataset: str,
    source_manifest: dict,
    reference_summary: dict[str, dict[str, int]],
    split_stats: dict[str, SplitStats],
    filter_info: dict,
) -> None:
    shown_dir = manifest_data_dir or out_dir
    manifest = {
        "dataset": dataset,
        "source": source_manifest.get("source", "processed public benchmark"),
        "source_urls": source_manifest.get("source_urls", []),
        "raw_files": source_manifest.get("raw_files", {}),
        "raw_csv_members": source_manifest.get("raw_csv_members", {}),
        "split_policy": source_manifest.get("split_policy"),
        "has_true_temporal_order": source_manifest.get("has_true_temporal_order"),
        "concept_source": source_manifest.get("concept_source"),
        "output_format": PROJECT_COLUMNS,
        "filter": {
            "method": "train-only high-interaction student/item filtering",
            "reference_dataset": reference_dataset,
            "reference_summary": reference_summary,
            **filter_info,
        },
        "splits": {split: asdict(stats) for split, stats in split_stats.items()},
        "processed_files": {
            split: file_info(
                out_dir / f"{split}.csv",
                manifest_path=shown_dir / f"{split}.csv",
            )
            for split in SPLITS
        },
        "notes": (
            "Filtered from the processed dataset to keep high-evidence train students/items. "
            "Selection uses train split interaction counts only; valid/test are filtered by the same IDs."
        ),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def replace_in_place(src_dir: Path, tmp_dir: Path) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    old_manifest = src_dir / "manifest.json"
    if old_manifest.exists():
        old_manifest.rename(src_dir / f"manifest.unfiltered.{timestamp}.json")
    for split in SPLITS:
        target = src_dir / f"{split}.csv"
        if target.exists():
            target.unlink()
        shutil.move(str(tmp_dir / f"{split}.csv"), str(target))
    shutil.move(str(tmp_dir / "manifest.json"), str(src_dir / "manifest.json"))
    shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> None:
    args = parse_args()
    dataset_dir = args.data_dir / args.dataset
    reference_dir = args.data_dir / args.reference_dataset
    output_dataset = args.output_dataset or (args.dataset if args.in_place else f"{args.dataset}_filtered")
    output_dir = args.data_dir / output_dataset

    if not dataset_dir.exists():
        raise FileNotFoundError(dataset_dir)
    if not reference_dir.exists():
        raise FileNotFoundError(reference_dir)
    if output_dir.exists() and output_dir != dataset_dir and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists; pass --overwrite")

    for split in SPLITS:
        if read_header(dataset_dir / f"{split}.csv") != PROJECT_COLUMNS:
            raise ValueError(f"{dataset_dir / f'{split}.csv'} does not use {PROJECT_COLUMNS}")

    reference = split_summary(reference_dir)
    target_train_rows = max(1, int(reference["train"]["rows"] * args.target_train_scale))
    target_item_count = max(1, int(reference["train"]["items"] * args.target_item_scale))

    train_path = dataset_dir / "train.csv"
    print(f"[filter] counting train entities in {train_path}")
    _, item_counts, original_train_rows = count_train_entities(train_path, args.chunksize)
    kept_items = select_items(
        item_counts,
        target_item_count=target_item_count,
        min_item_interactions=args.min_item_interactions,
    )
    print(
        f"[filter] kept_items={len(kept_items)} target_items={target_item_count} "
        f"min_item_interactions={args.min_item_interactions}"
    )

    eligible_student_counts = count_students_on_items(train_path, kept_items, args.chunksize)
    kept_students, estimated_train_rows, student_cutoff = select_students(
        eligible_student_counts,
        target_train_rows=target_train_rows,
        min_selected_students=args.min_selected_students,
    )
    print(
        f"[filter] kept_students={len(kept_students)} target_train_rows={target_train_rows} "
        f"estimated_train_rows={estimated_train_rows} student_cutoff={student_cutoff}"
    )

    tmp_dir = output_dir
    if args.in_place:
        tmp_dir = args.data_dir / f".{args.dataset}_filter_tmp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    split_stats: dict[str, SplitStats] = {}
    for split in SPLITS:
        print(f"[filter] writing {split}")
        split_stats[split] = write_filtered_split(
            dataset_dir / f"{split}.csv",
            tmp_dir / f"{split}.csv",
            kept_students=kept_students,
            kept_items=kept_items,
            chunksize=args.chunksize,
        )
        print(f"[filter] {split}: {asdict(split_stats[split])}")

    source_manifest_path = dataset_dir / "manifest.json"
    source_manifest = (
        json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if source_manifest_path.exists()
        else {}
    )
    filter_info = {
        "source_dataset": args.dataset,
        "original_train_rows": int(original_train_rows),
        "target_train_rows": int(target_train_rows),
        "target_item_count": int(target_item_count),
        "min_item_interactions": int(args.min_item_interactions),
        "selected_students": int(len(kept_students)),
        "selected_items": int(len(kept_items)),
        "student_interaction_cutoff": int(student_cutoff),
        "estimated_filtered_train_rows": int(estimated_train_rows),
    }
    write_manifest(
        tmp_dir,
        dataset=output_dataset,
        manifest_data_dir=dataset_dir if args.in_place else output_dir,
        reference_dataset=args.reference_dataset,
        source_manifest=source_manifest,
        reference_summary=reference,
        split_stats=split_stats,
        filter_info=filter_info,
    )

    if args.in_place:
        replace_in_place(dataset_dir, tmp_dir)
        print(f"[filter] replaced in place: {dataset_dir}")
    else:
        print(f"[filter] wrote: {output_dir}")


if __name__ == "__main__":
    main()
