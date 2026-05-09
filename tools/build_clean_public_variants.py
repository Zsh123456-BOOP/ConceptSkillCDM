#!/usr/bin/env python3
"""Build cleaned public benchmark variants used for follow-up full runs.

The variants are intentionally separate datasets so earlier public benchmark
results stay reproducible:

* assist_12_clean15_item50: train-only iterative stu>=15 and item>=50 filter.
* ednet_kt1_clean15_sample5000: train-count>=15 students, stratified sample 5000.
* nips34_l3: NIPS34 using only Level-3 leaf subjects as concepts.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import shutil
import zipfile
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from adapt_public_benchmarks import (
    URLS,
    StreamingSplitWriters,
    dataframe_to_rows,
    download,
    extract_csv_from_zip,
    file_size,
    find_column,
    human_size,
    log,
    normalize_binary,
    normalize_int,
    parse_concepts,
    save_manifest,
    split_dataframe_by_student,
    split_positions,
    write_csv,
)


RAW_DIR_DEFAULT = Path("data") / "_public_raw"
OUTPUT_DIR_DEFAULT = Path("data")
PROJECT_COLUMNS = ["stu_id", "exer_id", "cpt_seq", "label"]
DEFAULT_DATASETS = [
    "assist_12_clean15_item50",
    "ednet_kt1_clean15_sample5000",
    "nips34_l3",
]


def _prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _split_stats_from_frames(out_dir: Path, splits: dict[str, pd.DataFrame]) -> dict[str, dict]:
    stats = {}
    for split in ("train", "valid", "test"):
        frame = splits.get(split, pd.DataFrame(columns=PROJECT_COLUMNS))
        stats[split] = write_csv(out_dir / f"{split}.csv", dataframe_to_rows(frame[PROJECT_COLUMNS]))
    return stats


def _filter_splits_by_train_kcore(
    splits: dict[str, pd.DataFrame],
    *,
    min_student_interactions: int,
    min_item_interactions: int,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Filter IDs using only train split counts until both thresholds hold."""
    train = splits["train"][PROJECT_COLUMNS].copy()
    kept_students = set(train["stu_id"].unique())
    kept_items = set(train["exer_id"].unique())
    history = []

    for iteration in range(1, 51):
        current = train[
            train["stu_id"].isin(kept_students)
            & train["exer_id"].isin(kept_items)
        ]
        if current.empty:
            raise ValueError("train-only k-core filter removed all interactions")

        student_counts = current.groupby("stu_id").size()
        item_counts = current.groupby("exer_id").size()
        next_students = set(student_counts[student_counts >= min_student_interactions].index)
        next_items = set(item_counts[item_counts >= min_item_interactions].index)
        history.append(
            {
                "iteration": iteration,
                "rows": int(len(current)),
                "students": int(len(next_students)),
                "items": int(len(next_items)),
            }
        )
        if next_students == kept_students and next_items == kept_items:
            break
        kept_students, kept_items = next_students, next_items
    else:
        raise RuntimeError("train-only k-core filter did not converge within 50 iterations")

    filtered = {}
    for split, frame in splits.items():
        out = frame[
            frame["stu_id"].isin(kept_students)
            & frame["exer_id"].isin(kept_items)
        ].copy()
        filtered[split] = out[PROJECT_COLUMNS]

    return filtered, {
        "method": "train-only iterative k-core interaction filtering",
        "min_student_interactions": int(min_student_interactions),
        "min_item_interactions": int(min_item_interactions),
        "iterations": history,
        "selected_students": int(len(kept_students)),
        "selected_items": int(len(kept_items)),
    }


def build_assist12_clean15_item50(raw_dir: Path, output_dir: Path, overwrite: bool) -> dict:
    name = "assist_12_clean15_item50"
    out_dir = output_dir / name
    _prepare_output_dir(out_dir, overwrite)

    dataset_raw = raw_dir / "assist_12"
    zip_path = dataset_raw / "2012-2013-data-with-predictions-4-final.zip"
    download(URLS["assist2012"], zip_path, overwrite=False)
    csv_name, payload = extract_csv_from_zip(zip_path, lambda n: n.lower().endswith(".csv"))
    df = pd.read_csv(io.BytesIO(payload), low_memory=False)

    stu_col = find_column(df.columns, ["user_id", "student_id", "anon_student_id"])
    exer_col = find_column(df.columns, ["problem_id", "item_id", "question_id", "problem_log_id"])
    cpt_col = find_column(df.columns, ["skill_id", "knowledge_component_id", "kc_id"])
    label_col = find_column(df.columns, ["correct", "is_correct", "score"])
    order_col = find_column(df.columns, ["start_time", "order_id", "ms_first_response", "problem_log_id", "log_id"])
    missing = [
        key for key, value in {
            "student": stu_col,
            "exercise": exer_col,
            "concept": cpt_col,
            "label": label_col,
        }.items()
        if value is None
    ]
    if missing:
        raise ValueError(f"ASSIST2012 missing columns: {missing}; columns={list(df.columns)[:50]}")

    out = pd.DataFrame(
        {
            "stu_id": df[stu_col].map(normalize_int),
            "exer_id": df[exer_col].map(normalize_int),
            "cpt_seq": df[cpt_col].map(parse_concepts),
            "label": df[label_col].map(normalize_binary),
        }
    )
    if order_col:
        out["order_id"] = df[order_col]
    before = len(out)
    out = out.dropna(subset=["stu_id", "exer_id", "label"]).copy()
    out = out[out["cpt_seq"] != ""].copy()
    for col in ("stu_id", "exer_id", "label"):
        out[col] = out[col].astype(int)

    splits = split_dataframe_by_student(out, order_col="order_id" if "order_id" in out.columns else None)
    filtered_splits, filter_info = _filter_splits_by_train_kcore(
        splits,
        min_student_interactions=15,
        min_item_interactions=50,
    )
    split_stats = _split_stats_from_frames(out_dir, filtered_splits)
    manifest = {
        "dataset": name,
        "source": "USTC BASE ASSISTment mirror",
        "source_urls": [URLS["assist2012"]],
        "raw_files": {zip_path.name: {"bytes": file_size(zip_path), "human_size": human_size(file_size(zip_path))}},
        "raw_csv_member": csv_name,
        "source_columns": {"student": stu_col, "exercise": exer_col, "concept": cpt_col, "label": label_col, "order": order_col},
        "split_policy": f"per-student 80/10/10 sorted by {order_col or 'source order'}; train-only k-core filter",
        "has_true_temporal_order": bool(order_col),
        "concept_source": cpt_col,
        "dropped": {"invalid_or_missing_rows": int(before - len(out))},
        "filter": filter_info,
        "splits": split_stats,
    }
    save_manifest(out_dir, manifest)
    return manifest


def _extract_level3_concepts(subject_value, level3_ids: set[int]) -> str:
    ids = [normalize_int(part) for part in str(subject_value).replace("[", " ").replace("]", " ").split(",")]
    clean = []
    seen = set()
    for cid in ids:
        if cid is None or cid not in level3_ids or cid in seen:
            continue
        clean.append(str(cid))
        seen.add(cid)
    return ",".join(clean)


def build_nips34_l3(raw_dir: Path, output_dir: Path, overwrite: bool) -> dict:
    name = "nips34_l3"
    out_dir = output_dir / name
    _prepare_output_dir(out_dir, overwrite)

    dataset_raw = raw_dir / "nips34"
    zip_path = dataset_raw / "public_data.zip"
    download(URLS["nips2020"], zip_path, overwrite=False)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        train_name = next((n for n in names if n.lower().endswith("train_task_3_4.csv")), None)
        q_meta_name = next((n for n in names if n.lower().endswith("question_metadata_task_3_4.csv")), None)
        subj_meta_name = next((n for n in names if n.lower().endswith("subject_metadata.csv")), None)
        if train_name is None or q_meta_name is None or subj_meta_name is None:
            raise ValueError("NIPS34 train/question/subject metadata files not found")
        train_df = pd.read_csv(zf.open(train_name), low_memory=False)
        q_df = pd.read_csv(zf.open(q_meta_name), low_memory=False)
        subject_df = pd.read_csv(zf.open(subj_meta_name), low_memory=False)

    level3_ids = set(
        int(v)
        for v in subject_df.loc[subject_df["Level"] == 3, "SubjectId"].dropna().map(normalize_int)
        if v is not None
    )
    q_col = find_column(q_df.columns, ["QuestionId", "question_id"])
    subj_col = find_column(q_df.columns, ["SubjectId", "subject_id", "KnowledgeTag", "ConstructId"])
    if q_col is None or subj_col is None:
        raise ValueError(f"NIPS34 question metadata columns not recognized: {list(q_df.columns)}")
    concept_map = {}
    for row in q_df.itertuples(index=False):
        qid = normalize_int(getattr(row, q_col))
        cpt_seq = _extract_level3_concepts(getattr(row, subj_col), level3_ids)
        if qid is not None and cpt_seq:
            concept_map[qid] = cpt_seq

    stu_col = find_column(train_df.columns, ["UserId", "user_id", "student_id"])
    exer_col = find_column(train_df.columns, ["QuestionId", "question_id"])
    label_col = find_column(train_df.columns, ["IsCorrect", "is_correct", "correct"])
    order_col = find_column(train_df.columns, ["AnswerId", "answer_id", "timestamp"])
    if None in (stu_col, exer_col, label_col):
        raise ValueError(f"NIPS34 train columns not recognized: {list(train_df.columns)}")

    out = pd.DataFrame(
        {
            "stu_id": train_df[stu_col].map(normalize_int),
            "exer_id": train_df[exer_col].map(normalize_int),
            "label": train_df[label_col].map(normalize_binary),
        }
    )
    if order_col:
        out["order_id"] = train_df[order_col].map(normalize_int)
    out["cpt_seq"] = out["exer_id"].map(concept_map).fillna("")
    before = len(out)
    out = out.dropna(subset=["stu_id", "exer_id", "label"]).copy()
    out = out[out["cpt_seq"] != ""].copy()
    for col in ("stu_id", "exer_id", "label"):
        out[col] = out[col].astype(int)

    splits = split_dataframe_by_student(out, order_col="order_id" if "order_id" in out.columns else None)
    split_stats = _split_stats_from_frames(out_dir, splits)
    manifest = {
        "dataset": name,
        "source": "USTC BASE NIPS2020 public_data mirror",
        "source_urls": [URLS["nips2020"]],
        "raw_files": {zip_path.name: {"bytes": file_size(zip_path), "human_size": human_size(file_size(zip_path))}},
        "raw_csv_members": {"train": train_name, "question_metadata": q_meta_name, "subject_metadata": subj_meta_name},
        "source_columns": {"student": stu_col, "exercise": exer_col, "concept": subj_col, "label": label_col, "order": order_col},
        "split_policy": f"per-student 80/10/10 sorted by {order_col or 'source order'}",
        "has_true_temporal_order": bool(order_col),
        "concept_source": "question_metadata_task_3_4 SubjectId restricted to subject_metadata Level=3",
        "filter": {
            "method": "concept remap only",
            "level3_subject_count_in_metadata": int(len(level3_ids)),
            "level3_subject_count_used": int(len({int(c) for seq in concept_map.values() for c in seq.split(',')})),
        },
        "dropped": {"invalid_or_missing_rows": int(before - len(out))},
        "splits": split_stats,
    }
    save_manifest(out_dir, manifest)
    return manifest


def _iter_ednet_members(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    return sorted(
        [
            info for info in zf.infolist()
            if info.filename.lower().endswith(".csv")
            and "__macosx" not in info.filename.lower()
            and "/u" in info.filename.lower()
        ],
        key=lambda info: info.filename,
    )


def _read_ednet_question_map(contents_zip: Path) -> tuple[dict[str, tuple[int, str, str]], str]:
    download(URLS["ednet_contents"], contents_zip, overwrite=False)
    with zipfile.ZipFile(contents_zip) as zf:
        q_name = next((n for n in zf.namelist() if n.endswith("contents/questions.csv")), None)
        if q_name is None:
            raise ValueError("contents/questions.csv not found in EdNet contents zip")
        q_df = pd.read_csv(zf.open(q_name))
    q_map = {}
    for row in q_df.itertuples(index=False):
        qid_text = str(getattr(row, "question_id")).strip()
        qid = normalize_int(qid_text)
        cpt_seq = parse_concepts(getattr(row, "tags"))
        correct = str(getattr(row, "correct_answer")).strip().lower()
        if qid is not None and cpt_seq and correct:
            q_map[qid_text] = (qid, cpt_seq, correct)
    return q_map, q_name


def _count_valid_ednet_rows(
    zf: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    q_map: dict[str, tuple[int, str, str]],
) -> tuple[int, dict[str, int]]:
    dropped = {"missing_question_metadata": 0, "bad_answer": 0}
    count = 0
    with zf.open(info) as f:
        text = io.TextIOWrapper(f, encoding="utf-8", errors="replace", newline="")
        reader = csv.DictReader(text)
        for raw in reader:
            meta = q_map.get(str(raw.get("question_id", "")).strip())
            if meta is None:
                dropped["missing_question_metadata"] += 1
                continue
            user_answer = str(raw.get("user_answer", "")).strip().lower()
            if not user_answer:
                dropped["bad_answer"] += 1
                continue
            count += 1
    return count, dropped


def _sample_rank_stratified(
    candidates: list[tuple[int, str, int, int]],
    *,
    sample_size: int,
    seed: int,
    strata: int = 10,
) -> list[tuple[int, str, int, int]]:
    if len(candidates) <= sample_size:
        return list(candidates)
    rng = random.Random(seed)
    ordered = sorted(candidates, key=lambda row: (row[3], row[0]))
    chunks = []
    for idx in range(strata):
        start = int(idx * len(ordered) / strata)
        end = int((idx + 1) * len(ordered) / strata)
        chunk = ordered[start:end]
        if chunk:
            chunks.append(chunk)

    selected: list[tuple[int, str, int, int]] = []
    for idx, chunk in enumerate(chunks):
        target = sample_size // len(chunks)
        if idx < sample_size % len(chunks):
            target += 1
        selected.extend(rng.sample(chunk, min(target, len(chunk))))

    if len(selected) < sample_size:
        selected_ids = {row[0] for row in selected}
        remaining = [row for row in ordered if row[0] not in selected_ids]
        selected.extend(rng.sample(remaining, sample_size - len(selected)))
    elif len(selected) > sample_size:
        selected = rng.sample(selected, sample_size)

    return sorted(selected, key=lambda row: row[0])


def _write_selected_ednet_users(
    *,
    kt1_zip: Path,
    q_map: dict[str, tuple[int, str, str]],
    selected_students: set[int],
    out_dir: Path,
) -> tuple[dict[str, dict], dict[str, int]]:
    dropped = {
        "missing_question_metadata": 0,
        "bad_answer": 0,
        "bad_user_file": 0,
        "empty_user_history": 0,
        "unselected_user_files": 0,
    }
    writers = StreamingSplitWriters(out_dir)
    processed = 0
    try:
        with zipfile.ZipFile(kt1_zip) as zf:
            for info in _iter_ednet_members(zf):
                basename = Path(info.filename).stem
                stu_id = normalize_int(basename)
                if stu_id is None:
                    dropped["bad_user_file"] += 1
                    continue
                if stu_id not in selected_students:
                    dropped["unselected_user_files"] += 1
                    continue

                rows: list[tuple[int, int, str, int, int]] = []
                with zf.open(info) as f:
                    text = io.TextIOWrapper(f, encoding="utf-8", errors="replace", newline="")
                    reader = csv.DictReader(text)
                    for raw in reader:
                        meta = q_map.get(str(raw.get("question_id", "")).strip())
                        if meta is None:
                            dropped["missing_question_metadata"] += 1
                            continue
                        exer_id, cpt_seq, correct = meta
                        user_answer = str(raw.get("user_answer", "")).strip().lower()
                        if not user_answer:
                            dropped["bad_answer"] += 1
                            continue
                        timestamp = normalize_int(raw.get("timestamp")) or 0
                        label = 1 if user_answer == correct else 0
                        rows.append((timestamp, stu_id, exer_id, cpt_seq, label))
                if not rows:
                    dropped["empty_user_history"] += 1
                    continue
                rows.sort(key=lambda row: row[0])
                train_end, valid_end = split_positions(len(rows))
                for idx, (_, s, e, c, y) in enumerate(rows):
                    split = "train" if idx < train_end else "valid" if idx < valid_end else "test"
                    writers.write(split, (s, e, c, y))
                processed += 1
                if processed % 500 == 0:
                    log(f"[ednet_clean] wrote selected users={processed}/{len(selected_students)}")
    finally:
        writers.close()
    return writers.stats_dict(), dropped


def build_ednet_clean15_sample5000(
    raw_dir: Path,
    output_dir: Path,
    overwrite: bool,
    *,
    min_student_interactions: int,
    sample_students: int,
    seed: int,
) -> dict:
    name = "ednet_kt1_clean15_sample5000"
    out_dir = output_dir / name
    _prepare_output_dir(out_dir, overwrite)

    dataset_raw = raw_dir / "ednet_kt1"
    contents_zip = dataset_raw / "EdNet-Contents.zip"
    kt1_zip = dataset_raw / "EdNet-KT1.zip"
    q_map, q_name = _read_ednet_question_map(contents_zip)
    download(URLS["ednet_kt1"], kt1_zip, overwrite=False)

    candidates: list[tuple[int, str, int, int]] = []
    pass1_dropped = {"missing_question_metadata": 0, "bad_answer": 0, "bad_user_file": 0}
    with zipfile.ZipFile(kt1_zip) as zf:
        members = _iter_ednet_members(zf)
        log(f"[ednet_clean] user CSV files={len(members)}")
        for idx, info in enumerate(members, start=1):
            stu_id = normalize_int(Path(info.filename).stem)
            if stu_id is None:
                pass1_dropped["bad_user_file"] += 1
                continue
            valid_rows, dropped = _count_valid_ednet_rows(zf, info, q_map)
            pass1_dropped["missing_question_metadata"] += dropped["missing_question_metadata"]
            pass1_dropped["bad_answer"] += dropped["bad_answer"]
            if valid_rows <= 0:
                continue
            train_end, _ = split_positions(valid_rows)
            if train_end >= min_student_interactions:
                candidates.append((stu_id, info.filename, valid_rows, train_end))
            if idx % 50000 == 0:
                log(f"[ednet_clean] scanned users={idx}/{len(members)} eligible={len(candidates)}")

    selected = _sample_rank_stratified(
        candidates,
        sample_size=sample_students,
        seed=seed,
        strata=10,
    )
    selected_students = {row[0] for row in selected}
    split_stats, pass2_dropped = _write_selected_ednet_users(
        kt1_zip=kt1_zip,
        q_map=q_map,
        selected_students=selected_students,
        out_dir=out_dir,
    )

    selected_train_counts = [row[3] for row in selected]
    manifest = {
        "dataset": name,
        "source": "USTC BASE EdNet mirror",
        "source_urls": [URLS["ednet_contents"], URLS["ednet_kt1"]],
        "raw_files": {
            contents_zip.name: {"bytes": file_size(contents_zip), "human_size": human_size(file_size(contents_zip))},
            kt1_zip.name: {"bytes": file_size(kt1_zip), "human_size": human_size(file_size(kt1_zip))},
        },
        "raw_csv_members": {"question_metadata": q_name, "kt1_user_csv_files": len(candidates)},
        "split_policy": "per-user 80/10/10 sorted by timestamp; train-count>=15 then rank-stratified student sample",
        "has_true_temporal_order": True,
        "concept_source": "contents/questions.csv tags",
        "filter": {
            "method": "train-only minimum student interactions plus deterministic rank-stratified student sample",
            "min_student_interactions": int(min_student_interactions),
            "eligible_students": int(len(candidates)),
            "sampled_students": int(len(selected_students)),
            "sample_seed": int(seed),
            "sample_strata": 10,
            "selected_train_count_min": int(min(selected_train_counts)) if selected_train_counts else 0,
            "selected_train_count_median": float(pd.Series(selected_train_counts).median()) if selected_train_counts else 0.0,
            "selected_train_count_max": int(max(selected_train_counts)) if selected_train_counts else 0,
        },
        "dropped_pass1_all_users": pass1_dropped,
        "dropped_pass2_selected_users": pass2_dropped,
        "splits": split_stats,
        "notes": "Unlike the previous Junyi-sized EdNet subset, this variant avoids selecting only the most active students.",
    }
    save_manifest(out_dir, manifest)
    return manifest


def build_dataset(name: str, raw_dir: Path, output_dir: Path, overwrite: bool, args: argparse.Namespace) -> dict:
    if name == "assist_12_clean15_item50":
        return build_assist12_clean15_item50(raw_dir, output_dir, overwrite)
    if name == "nips34_l3":
        return build_nips34_l3(raw_dir, output_dir, overwrite)
    if name == "ednet_kt1_clean15_sample5000":
        return build_ednet_clean15_sample5000(
            raw_dir,
            output_dir,
            overwrite,
            min_student_interactions=args.ednet_min_student_interactions,
            sample_students=args.ednet_sample_students,
            seed=args.seed,
        )
    raise ValueError(f"unknown dataset variant: {name}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS), help="Comma-separated variants to build.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--ednet-min-student-interactions", type=int, default=15)
    parser.add_argument("--ednet-sample-students", type=int, default=5000)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    requested = [part.strip() for part in args.datasets.split(",") if part.strip()]
    for name in requested:
        log(f"[build] start {name}")
        manifest = build_dataset(name, args.raw_dir, args.output_dir, args.overwrite, args)
        log(f"[build] done {name}: {json.dumps(manifest.get('splits', {}), ensure_ascii=False)}")


if __name__ == "__main__":
    main()
