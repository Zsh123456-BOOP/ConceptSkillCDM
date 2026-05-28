#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Build an EdNet-KT1 evidence-gap dataset directly from raw archives.

This avoids re-splitting an already filtered dataset. It reads:
  data/_public_raw/ednet_kt1/EdNet-Contents.zip
  data/_public_raw/ednet_kt1/EdNet-KT1.zip

The output keeps the project schema:
  stu_id,exer_id,cpt_seq,label
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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


class SplitWriter:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.files = {}
        self.writers = {}
        self.rows: Dict[str, List[Tuple[int, int, str, int]]] = {s: [] for s in SPLITS}

    def write(self, split: str, row: Tuple[int, int, str, int]) -> None:
        self.rows[split].append(row)

    def flush(self) -> Dict[str, SplitStats]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stats: Dict[str, SplitStats] = {}
        for split in SPLITS:
            path = self.out_dir / f"{split}.csv"
            rows = self.rows[split]
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(PROJECT_COLUMNS)
                writer.writerows(rows)
            stats[split] = split_stats(rows)
        return stats


def parse_tags(value: Any) -> str:
    if pd.isna(value):
        return ""
    parts = []
    for token in str(value).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            parts.append(str(int(float(token))))
        except ValueError:
            continue
    return ",".join(dict.fromkeys(parts))


def normalize_int(value: Any) -> Optional[int]:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(float(str(value).strip()))
    except ValueError:
        return None


def normalize_question_int(value: Any) -> Optional[int]:
    text = str(value).strip()
    if text[:1].lower() == "q":
        text = text[1:]
    return normalize_int(text)


def load_question_map(contents_zip: Path) -> Dict[str, Tuple[int, str, str]]:
    with zipfile.ZipFile(contents_zip) as zf:
        q_name = next((n for n in zf.namelist() if n.endswith("contents/questions.csv")), None)
        if q_name is None:
            raise ValueError("contents/questions.csv not found")
        q_df = pd.read_csv(zf.open(q_name))
    out: Dict[str, Tuple[int, str, str]] = {}
    for row in q_df.itertuples(index=False):
        qid_text = str(getattr(row, "question_id")).strip()
        qid = normalize_question_int(qid_text)
        tags = parse_tags(getattr(row, "tags"))
        correct = str(getattr(row, "correct_answer")).strip().lower()
        if qid is not None and tags and correct:
            out[qid_text] = (qid, tags, correct)
    return out


def user_members(kt1_zip: Path) -> List[zipfile.ZipInfo]:
    with zipfile.ZipFile(kt1_zip) as zf:
        return [
            info
            for info in zf.infolist()
            if info.filename.lower().endswith(".csv")
            and "__macosx" not in info.filename.lower()
            and "/u" in info.filename.lower()
        ]


def read_user_rows(
    zf: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    q_map: Dict[str, Tuple[int, str, str]],
) -> List[Tuple[int, int, str, int]]:
    rows: List[Tuple[int, int, str, int]] = []
    with zf.open(info) as f:
        text = io.TextIOWrapper(f, encoding="utf-8", errors="replace", newline="")
        reader = csv.DictReader(text)
        for raw in reader:
            meta = q_map.get(str(raw.get("question_id", "")).strip())
            if meta is None:
                continue
            exer_id, cpt_seq, correct = meta
            answer = str(raw.get("user_answer", "")).strip().lower()
            if not answer:
                continue
            timestamp = normalize_int(raw.get("timestamp")) or 0
            label = 1 if answer == correct else 0
            rows.append((timestamp, exer_id, cpt_seq, label))
    rows.sort(key=lambda x: x[0])
    return rows


def scan_user_lengths(
    kt1_zip: Path,
    q_map: Dict[str, Tuple[int, str, str]],
    min_interactions: int,
    max_interactions: int,
    max_users_scan: int,
) -> List[Tuple[str, int]]:
    selected: List[Tuple[str, int]] = []
    with zipfile.ZipFile(kt1_zip) as zf:
        members = user_members(kt1_zip)
        for idx, info in enumerate(members):
            if max_users_scan > 0 and idx >= max_users_scan:
                break
            rows = read_user_rows(zf, info, q_map)
            n = len(rows)
            if min_interactions <= n <= max_interactions:
                selected.append((info.filename, n))
            if (idx + 1) % 10000 == 0:
                print(f"[scan] users={idx + 1}/{len(members)} selected={len(selected)}", flush=True)
    return selected


def split_positions(n: int, train_frac: float, valid_frac: float) -> Tuple[int, int]:
    train_end = max(1, min(n - 2, int(n * train_frac)))
    valid_end = max(train_end + 1, min(n - 1, int(n * (train_frac + valid_frac))))
    return train_end, valid_end


def split_stats(rows: Iterable[Tuple[int, int, str, int]]) -> SplitStats:
    rows = list(rows)
    students = set()
    items = set()
    concepts = set()
    positives = 0
    for stu, exer, cpt_seq, label in rows:
        students.add(stu)
        items.add(exer)
        positives += int(label)
        for token in str(cpt_seq).split(","):
            if token:
                concepts.add(int(token))
    return SplitStats(
        rows=len(rows),
        students=len(students),
        items=len(items),
        concepts=len(concepts),
        positive_rate=float(positives / len(rows)) if rows else 0.0,
    )


def gap_profile(split_rows: Dict[str, List[Tuple[int, int, str, int]]]) -> Dict[str, Any]:
    history: Dict[int, set[int]] = {}
    for stu, _, cpt_seq, _ in split_rows["train"]:
        history.setdefault(stu, set()).update(int(x) for x in str(cpt_seq).split(",") if x)
    rows = 0
    direct_unseen = 0
    weak_direct = 0
    direct_seen = 0
    for stu, _, cpt_seq, _ in split_rows["test"]:
        concepts = {int(x) for x in str(cpt_seq).split(",") if x}
        if not concepts:
            continue
        rows += 1
        seen = concepts & history.get(stu, set())
        if seen:
            direct_seen += 1
        else:
            direct_unseen += 1
        if len(seen) < len(concepts):
            weak_direct += 1
    return {
        "test_rows_with_concepts": rows,
        "test_direct_seen_rows": direct_seen,
        "test_direct_unseen_rows": direct_unseen,
        "test_direct_unseen_rate": float(direct_unseen / rows) if rows else 0.0,
        "test_weak_direct_rows": weak_direct,
        "test_weak_direct_rate": float(weak_direct / rows) if rows else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/_public_raw/ednet_kt1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/ednet_kt1_gap"))
    parser.add_argument("--output-dataset", default="ednet_kt1_gap")
    parser.add_argument("--min-interactions", type=int, default=80)
    parser.add_argument("--max-interactions", type=int, default=500)
    parser.add_argument("--target-users", type=int, default=5000)
    parser.add_argument("--max-users-scan", type=int, default=0)
    parser.add_argument("--train-frac", type=float, default=0.45)
    parser.add_argument("--valid-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    contents_zip = args.raw_dir / "EdNet-Contents.zip"
    kt1_zip = args.raw_dir / "EdNet-KT1.zip"
    if not contents_zip.exists() or not kt1_zip.exists():
        raise FileNotFoundError(f"missing EdNet raw zips under {args.raw_dir}")
    if args.output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} exists; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    q_map = load_question_map(contents_zip)
    candidates = scan_user_lengths(kt1_zip, q_map, args.min_interactions, args.max_interactions, args.max_users_scan)
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    selected = candidates[: args.target_users] if args.target_users > 0 else candidates
    selected_names = {name for name, _ in selected}
    print(f"[select] candidates={len(candidates)} selected={len(selected)}", flush=True)

    writer = SplitWriter(args.output_dir)
    split_rows: Dict[str, List[Tuple[int, int, str, int]]] = {s: [] for s in SPLITS}
    with zipfile.ZipFile(kt1_zip) as zf:
        info_by_name = {info.filename: info for info in user_members(kt1_zip)}
        for mapped_stu, filename in enumerate(sorted(selected_names)):
            rows = read_user_rows(zf, info_by_name[filename], q_map)
            train_end, valid_end = split_positions(len(rows), args.train_frac, args.valid_frac)
            for idx, (_, exer_id, cpt_seq, label) in enumerate(rows):
                split = "train" if idx < train_end else "valid" if idx < valid_end else "test"
                out_row = (mapped_stu, exer_id, cpt_seq, label)
                writer.write(split, out_row)
                split_rows[split].append(out_row)
            if (mapped_stu + 1) % 500 == 0:
                print(f"[write] users={mapped_stu + 1}/{len(selected_names)}", flush=True)
    stats = writer.flush()
    profile = gap_profile(split_rows)
    manifest = {
        "dataset": args.output_dataset,
        "source": "EdNet-KT1 raw archives",
        "source_urls": [
            "https://base.ustc.edu.cn/data/EdNet/EdNet-Contents.zip",
            "https://base.ustc.edu.cn/data/EdNet/EdNet-KT1.zip",
        ],
        "raw_dir": args.raw_dir.as_posix(),
        "split_policy": (
            "selected medium-length users; per-user chronological split sorted by raw timestamp; "
            f"train_frac={args.train_frac}, valid_frac={args.valid_frac}"
        ),
        "has_true_temporal_order": True,
        "concept_source": "contents/questions.csv tags",
        "filter": {
            "min_interactions": args.min_interactions,
            "max_interactions": args.max_interactions,
            "target_users": args.target_users,
            "selected_users": len(selected),
            "candidate_users": len(candidates),
            "seed": args.seed,
        },
        "splits": {split: asdict(stat) for split, stat in stats.items()},
        "gap_profile": profile,
        "output_format": PROJECT_COLUMNS,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
