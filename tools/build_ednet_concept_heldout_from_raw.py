#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Build an EdNet-KT1 student-conditioned concept-heldout split.

For each selected student, rows are assigned by held-out concept groups instead
of by chronological cut points.  The goal is to make validation/test evaluate
query concepts that are mostly absent from the same student's training history,
while keeping each student in train/valid/test and preserving chronological
order within each split file.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd


PROJECT_COLUMNS = ["stu_id", "exer_id", "cpt_seq", "label"]
SPLITS = ("train", "valid", "test")
SOURCE_URLS = [
    "https://base.ustc.edu.cn/data/EdNet/EdNet-Contents.zip",
    "https://base.ustc.edu.cn/data/EdNet/EdNet-KT1.zip",
]


@dataclass
class SplitStats:
    rows: int = 0
    students: int = 0
    items: int = 0
    concepts: int = 0
    positive_rate: float = 0.0


@dataclass
class StudentSplitAudit:
    raw_rows: int
    train_rows: int
    valid_rows: int
    test_rows: int
    unique_concepts: int
    train_concepts: int
    valid_concepts: int
    test_concepts: int
    valid_rows_with_train_concept: int
    test_rows_with_train_concept: int
    valid_direct_unseen_rows: int
    test_direct_unseen_rows: int
    weak_valid_rows: int
    weak_test_rows: int


def parse_tags(value: Any) -> str:
    if pd.isna(value):
        return ""
    parts: List[str] = []
    for token in str(value).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            parts.append(str(int(float(token))))
        except ValueError:
            continue
    return ",".join(dict.fromkeys(parts))


def parse_concepts(cpt_seq: str) -> Set[int]:
    out: Set[int] = set()
    for token in str(cpt_seq).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.add(int(token))
        except ValueError:
            continue
    return out


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
        concepts.update(parse_concepts(cpt_seq))
    return SplitStats(
        rows=len(rows),
        students=len(students),
        items=len(items),
        concepts=len(concepts),
        positive_rate=float(positives / len(rows)) if rows else 0.0,
    )


def row_indices_for_concepts(row_concepts: Sequence[Set[int]]) -> Dict[int, Set[int]]:
    out: Dict[int, Set[int]] = defaultdict(set)
    for idx, concepts in enumerate(row_concepts):
        for concept in concepts:
            out[concept].add(idx)
    return out


def greedy_select_concepts(
    *,
    concepts: Sequence[int],
    concept_to_rows: Dict[int, Set[int]],
    unavailable_rows: Set[int],
    target_rows: int,
    rng: random.Random,
    tolerance: float,
) -> Tuple[Set[int], Set[int]]:
    chosen_concepts: Set[int] = set()
    chosen_rows: Set[int] = set()
    remaining = list(concepts)
    rng.shuffle(remaining)
    target_rows = max(1, int(target_rows))

    while remaining and len(chosen_rows) < max(1, int(target_rows * (1.0 - tolerance))):
        best_idx = -1
        best_key: Optional[Tuple[float, int, float]] = None
        for idx, concept in enumerate(remaining):
            new_rows = concept_to_rows[concept] - unavailable_rows - chosen_rows
            gain = len(new_rows)
            if gain <= 0:
                continue
            new_total = len(chosen_rows) + gain
            overshoot = max(0, new_total - target_rows)
            key = (abs(new_total - target_rows) + 2.0 * overshoot, -gain, rng.random())
            if best_key is None or key < best_key:
                best_idx = idx
                best_key = key
        if best_idx < 0:
            break
        concept = remaining.pop(best_idx)
        chosen_concepts.add(concept)
        chosen_rows.update(concept_to_rows[concept] - unavailable_rows)
        if len(chosen_rows) >= target_rows and best_key and best_key[0] <= target_rows * tolerance:
            break
    return chosen_concepts, chosen_rows


def assign_concept_heldout(
    rows: Sequence[Tuple[int, int, str, int]],
    *,
    train_frac: float,
    valid_frac: float,
    min_unique_concepts: int,
    rng: random.Random,
    tolerance: float,
) -> Optional[Tuple[Dict[str, List[int]], StudentSplitAudit]]:
    n = len(rows)
    row_concepts = [parse_concepts(row[2]) for row in rows]
    all_concepts = sorted(set().union(*row_concepts)) if row_concepts else []
    if n < 3 or len(all_concepts) < min_unique_concepts:
        return None

    concept_to_rows = row_indices_for_concepts(row_concepts)
    target_valid = max(1, int(round(n * valid_frac)))
    target_test = max(1, int(round(n * max(0.0, 1.0 - train_frac - valid_frac))))

    test_concepts, test_rows = greedy_select_concepts(
        concepts=all_concepts,
        concept_to_rows=concept_to_rows,
        unavailable_rows=set(),
        target_rows=target_test,
        rng=rng,
        tolerance=tolerance,
    )
    valid_candidates = [c for c in all_concepts if c not in test_concepts]
    valid_concepts, valid_rows = greedy_select_concepts(
        concepts=valid_candidates,
        concept_to_rows=concept_to_rows,
        unavailable_rows=test_rows,
        target_rows=target_valid,
        rng=rng,
        tolerance=tolerance,
    )
    valid_rows -= test_rows
    train_rows = set(range(n)) - valid_rows - test_rows

    if not train_rows or not valid_rows or not test_rows:
        return None

    assignment = {
        "train": sorted(train_rows, key=lambda i: rows[i][0]),
        "valid": sorted(valid_rows, key=lambda i: rows[i][0]),
        "test": sorted(test_rows, key=lambda i: rows[i][0]),
    }
    audit = audit_student_split(rows, assignment)
    return assignment, audit


def audit_student_split(
    rows: Sequence[Tuple[int, int, str, int]],
    assignment: Dict[str, List[int]],
) -> StudentSplitAudit:
    split_concepts: Dict[str, Set[int]] = {}
    for split in SPLITS:
        concepts: Set[int] = set()
        for idx in assignment[split]:
            concepts.update(parse_concepts(rows[idx][2]))
        split_concepts[split] = concepts

    train_concepts = split_concepts["train"]

    def count_rows(split: str, predicate) -> int:
        return sum(1 for idx in assignment[split] if predicate(parse_concepts(rows[idx][2])))

    valid_overlap = count_rows("valid", lambda c: bool(c & train_concepts))
    test_overlap = count_rows("test", lambda c: bool(c & train_concepts))
    valid_unseen = count_rows("valid", lambda c: bool(c) and not bool(c & train_concepts))
    test_unseen = count_rows("test", lambda c: bool(c) and not bool(c & train_concepts))
    train_counts: Counter[int] = Counter()
    for idx in assignment["train"]:
        train_counts.update(parse_concepts(rows[idx][2]))
    weak_valid = count_rows("valid", lambda c: bool(c) and min((train_counts.get(x, 0) for x in c), default=0) <= 1)
    weak_test = count_rows("test", lambda c: bool(c) and min((train_counts.get(x, 0) for x in c), default=0) <= 1)

    return StudentSplitAudit(
        raw_rows=len(rows),
        train_rows=len(assignment["train"]),
        valid_rows=len(assignment["valid"]),
        test_rows=len(assignment["test"]),
        unique_concepts=len(set().union(*(parse_concepts(r[2]) for r in rows))),
        train_concepts=len(split_concepts["train"]),
        valid_concepts=len(split_concepts["valid"]),
        test_concepts=len(split_concepts["test"]),
        valid_rows_with_train_concept=valid_overlap,
        test_rows_with_train_concept=test_overlap,
        valid_direct_unseen_rows=valid_unseen,
        test_direct_unseen_rows=test_unseen,
        weak_valid_rows=weak_valid,
        weak_test_rows=weak_test,
    )


def gap_profile(split_rows: Dict[str, List[Tuple[int, int, str, int]]]) -> Dict[str, Any]:
    history: Dict[int, Set[int]] = {}
    train_counts: Dict[int, Counter[int]] = defaultdict(Counter)
    for stu, _, cpt_seq, _ in split_rows["train"]:
        concepts = parse_concepts(cpt_seq)
        history.setdefault(stu, set()).update(concepts)
        train_counts[stu].update(concepts)

    out: Dict[str, Any] = {}
    for split in ("valid", "test"):
        rows = 0
        direct_unseen = 0
        weak_direct = 0
        overlap = 0
        for stu, _, cpt_seq, _ in split_rows[split]:
            concepts = parse_concepts(cpt_seq)
            if not concepts:
                continue
            rows += 1
            seen = concepts & history.get(stu, set())
            if not seen:
                direct_unseen += 1
            else:
                overlap += 1
            counts = train_counts.get(stu, Counter())
            if min((counts.get(c, 0) for c in concepts), default=0) <= 1:
                weak_direct += 1
        out[f"{split}_rows_with_concepts"] = rows
        out[f"{split}_direct_unseen_rows"] = direct_unseen
        out[f"{split}_direct_unseen_rate"] = float(direct_unseen / rows) if rows else 0.0
        out[f"{split}_rows_with_train_concept"] = overlap
        out[f"{split}_train_concept_overlap_rate"] = float(overlap / rows) if rows else 0.0
        out[f"{split}_weak_direct_rows"] = weak_direct
        out[f"{split}_weak_direct_rate"] = float(weak_direct / rows) if rows else 0.0
    return out


def aggregate_student_audit(audits: Sequence[StudentSplitAudit]) -> Dict[str, Any]:
    if not audits:
        return {}
    total_raw = sum(a.raw_rows for a in audits)
    total_train = sum(a.train_rows for a in audits)
    total_valid = sum(a.valid_rows for a in audits)
    total_test = sum(a.test_rows for a in audits)
    return {
        "students_with_all_splits": len(audits),
        "mean_unique_concepts_per_student": float(sum(a.unique_concepts for a in audits) / len(audits)),
        "row_ratio_train": float(total_train / total_raw) if total_raw else 0.0,
        "row_ratio_valid": float(total_valid / total_raw) if total_raw else 0.0,
        "row_ratio_test": float(total_test / total_raw) if total_raw else 0.0,
        "valid_direct_unseen_rows": int(sum(a.valid_direct_unseen_rows for a in audits)),
        "test_direct_unseen_rows": int(sum(a.test_direct_unseen_rows for a in audits)),
        "valid_rows_with_train_concept": int(sum(a.valid_rows_with_train_concept for a in audits)),
        "test_rows_with_train_concept": int(sum(a.test_rows_with_train_concept for a in audits)),
    }


def write_split_csv(out_dir: Path, split_rows: Dict[str, List[Tuple[int, int, str, int]]]) -> Dict[str, SplitStats]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: Dict[str, SplitStats] = {}
    for split in SPLITS:
        rows = split_rows[split]
        with (out_dir / f"{split}.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(PROJECT_COLUMNS)
            writer.writerows(rows)
        stats[split] = split_stats(rows)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/_public_raw/ednet_kt1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/ednet_kt1_concept_holdout"))
    parser.add_argument("--output-dataset", default="ednet_kt1_concept_holdout")
    parser.add_argument("--min-interactions", type=int, default=60)
    parser.add_argument("--max-interactions", type=int, default=600)
    parser.add_argument("--target-users", type=int, default=2200)
    parser.add_argument("--max-users-scan", type=int, default=40000)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--valid-frac", type=float, default=0.10)
    parser.add_argument("--min-unique-concepts", type=int, default=6)
    parser.add_argument("--concept-tolerance", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260529)
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
    candidates = scan_user_lengths(
        kt1_zip,
        q_map,
        args.min_interactions,
        args.max_interactions,
        args.max_users_scan,
    )
    rng = random.Random(args.seed)
    rng.shuffle(candidates)

    split_rows: Dict[str, List[Tuple[int, int, str, int]]] = {s: [] for s in SPLITS}
    audits: List[StudentSplitAudit] = []
    skipped = 0
    with zipfile.ZipFile(kt1_zip) as zf:
        info_by_name = {info.filename: info for info in user_members(kt1_zip)}
        mapped_stu = 0
        for filename, _ in candidates:
            if mapped_stu >= args.target_users:
                break
            rows = read_user_rows(zf, info_by_name[filename], q_map)
            assigned = assign_concept_heldout(
                rows,
                train_frac=args.train_frac,
                valid_frac=args.valid_frac,
                min_unique_concepts=args.min_unique_concepts,
                rng=random.Random(args.seed + mapped_stu),
                tolerance=args.concept_tolerance,
            )
            if assigned is None:
                skipped += 1
                continue
            assignment, audit = assigned
            for split in SPLITS:
                for idx in assignment[split]:
                    _, exer_id, cpt_seq, label = rows[idx]
                    split_rows[split].append((mapped_stu, exer_id, cpt_seq, label))
            audits.append(audit)
            mapped_stu += 1
            if mapped_stu % 500 == 0:
                print(f"[write] users={mapped_stu}/{args.target_users} skipped={skipped}", flush=True)

    if len(audits) < args.target_users:
        print(f"[WARN] requested {args.target_users} users but built {len(audits)}; skipped={skipped}", flush=True)

    stats = write_split_csv(args.output_dir, split_rows)
    profile = gap_profile(split_rows)
    audit_summary = aggregate_student_audit(audits)
    manifest = {
        "dataset": args.output_dataset,
        "source": "EdNet-KT1 raw archives",
        "source_urls": SOURCE_URLS,
        "raw_dir": args.raw_dir.as_posix(),
        "split_policy": (
            "student-conditioned concept-heldout split; concept groups selected per student "
            "to approximate train/valid/test ratios while reducing train concept overlap in held-out splits; "
            "rows remain chronological within each split"
        ),
        "has_true_temporal_order": True,
        "concept_source": "contents/questions.csv tags",
        "filter": {
            "min_interactions": args.min_interactions,
            "max_interactions": args.max_interactions,
            "target_users": args.target_users,
            "selected_users": len(audits),
            "candidate_users": len(candidates),
            "skipped_users": skipped,
            "min_unique_concepts": args.min_unique_concepts,
            "concept_tolerance": args.concept_tolerance,
            "seed": args.seed,
        },
        "target_ratios": {
            "train_frac": args.train_frac,
            "valid_frac": args.valid_frac,
            "test_frac": max(0.0, 1.0 - args.train_frac - args.valid_frac),
        },
        "splits": {split: asdict(stat) for split, stat in stats.items()},
        "gap_profile": profile,
        "student_split_audit": audit_summary,
        "output_format": PROJECT_COLUMNS,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
