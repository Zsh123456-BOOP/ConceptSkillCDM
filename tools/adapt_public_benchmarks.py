#!/usr/bin/env python
"""Download and adapt public KT/CD benchmarks to the repo CSV format.

Output format:
    stu_id,exer_id,cpt_seq,label

The script keeps raw archives under data/_public_raw by default and writes
cleaned model-ready splits under data/<dataset>. It intentionally drops columns
that the current model cannot consume, but records source caveats in each
dataset manifest.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable
from urllib.request import Request, urlopen, urlretrieve

import pandas as pd


RAW_DIR_DEFAULT = Path("data") / "_public_raw"
OUTPUT_DIR_DEFAULT = Path("data")

URLS = {
    "frcsub_response": "https://huggingface.co/datasets/shenjunhao/mind-data/resolve/main/FrcSub/response.csv",
    "frcsub_q": "https://huggingface.co/datasets/shenjunhao/mind-data/resolve/main/FrcSub/q_matrix.csv",
    "math2_response": "https://huggingface.co/datasets/shenjunhao/mind-data/resolve/main/Math2/response.csv",
    "math2_q": "https://huggingface.co/datasets/shenjunhao/mind-data/resolve/main/Math2/q_matrix.csv",
    "assist2012": "https://base.ustc.edu.cn/data/ASSISTment/2012-2013-data-with-predictions-4-final.zip",
    "assist2015": "https://base.ustc.edu.cn/data/ASSISTment/2015_100_skill_builders_main_problems.zip",
    "nips2020": "https://base.ustc.edu.cn/data/NIPS2020/public_data.zip",
    "ednet_contents": "https://base.ustc.edu.cn/data/EdNet/EdNet-Contents.zip",
    "ednet_kt1": "https://base.ustc.edu.cn/data/EdNet/EdNet-KT1.zip",
}


DATASET_ALIASES = {
    "frcsub": "frcsub",
    "math2": "math2",
    "assist2012": "assist_12",
    "assist_12": "assist_12",
    "assist2015": "assist_15",
    "assist_15": "assist_15",
    "nips34": "nips34",
    "nips_task34": "nips34",
    "ednet-kt1": "ednet_kt1",
    "ednet_kt1": "ednet_kt1",
}

DEFAULT_DATASETS = ["frcsub", "math2", "assist_12", "assist_15", "nips34", "ednet_kt1"]


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{stamp} {message}", flush=True)


def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def download(url: str, target: Path, overwrite: bool = False) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0 and not overwrite:
        log(f"[download] reuse {target} ({human_size(target.stat().st_size)})")
        return
    tmp = target.with_suffix(target.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    def _hook(block_count: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = block_count * block_size
        if downloaded >= total_size or downloaded // (100 * 1024 * 1024) != (downloaded - block_size) // (100 * 1024 * 1024):
            pct = min(100.0, downloaded * 100.0 / total_size)
            log(f"[download] {target.name}: {pct:.1f}% ({human_size(min(downloaded, total_size))}/{human_size(total_size)})")

    log(f"[download] start {url}")
    urlretrieve(url, tmp, _hook)
    tmp.replace(target)
    log(f"[download] done {target} ({human_size(target.stat().st_size)})")


def read_hf_csv(url: str, raw_path: Path, overwrite: bool = False) -> pd.DataFrame:
    download(url, raw_path, overwrite=overwrite)
    return pd.read_csv(raw_path, header=None)


def normalize_int(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return int(float(text))
    except ValueError:
        digits = re.findall(r"-?\d+", text)
        return int(digits[0]) if digits else None


def normalize_binary(value) -> int | None:
    parsed = normalize_int(value)
    if parsed in (0, 1):
        return parsed
    return None


def parse_concepts(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (list, tuple, set)):
        ids = [normalize_int(v) for v in value]
    else:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null", "[]"}:
            return ""
        ids = [normalize_int(part) for part in re.findall(r"-?\d+(?:\.\d+)?", text)]
    clean = []
    seen = set()
    for cid in ids:
        if cid is None or cid in seen:
            continue
        clean.append(str(cid))
        seen.add(cid)
    return ",".join(clean)


def write_csv(path: Path, rows: Iterable[tuple[int, int, str, int]]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = SplitStats()
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stu_id", "exer_id", "cpt_seq", "label"])
        for stu_id, exer_id, cpt_seq, label in rows:
            writer.writerow([stu_id, exer_id, cpt_seq, label])
            stats.add(stu_id, exer_id, cpt_seq, label)
    return stats.to_dict()


@dataclass
class SplitStats:
    rows: int = 0
    positives: int = 0
    students: set[int] = field(default_factory=set)
    items: set[int] = field(default_factory=set)
    concepts: set[int] = field(default_factory=set)

    def add(self, stu_id: int, exer_id: int, cpt_seq: str, label: int) -> None:
        self.rows += 1
        self.positives += int(label)
        self.students.add(int(stu_id))
        self.items.add(int(exer_id))
        for part in str(cpt_seq).split(","):
            if part.strip():
                self.concepts.add(int(part))

    def merge(self, other: "SplitStats") -> None:
        self.rows += other.rows
        self.positives += other.positives
        self.students.update(other.students)
        self.items.update(other.items)
        self.concepts.update(other.concepts)

    def to_dict(self) -> dict:
        return {
            "rows": int(self.rows),
            "students": int(len(self.students)),
            "items": int(len(self.items)),
            "concepts": int(len(self.concepts)),
            "positive_rate": float(self.positives / self.rows) if self.rows else None,
        }


class StreamingSplitWriters:
    def __init__(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.paths = {
            "train": out_dir / "train.csv",
            "valid": out_dir / "valid.csv",
            "test": out_dir / "test.csv",
        }
        self.files = {
            split: path.open("w", newline="", encoding="utf-8")
            for split, path in self.paths.items()
        }
        self.writers = {split: csv.writer(f) for split, f in self.files.items()}
        self.stats = {split: SplitStats() for split in self.paths}
        for writer in self.writers.values():
            writer.writerow(["stu_id", "exer_id", "cpt_seq", "label"])

    def write(self, split: str, row: tuple[int, int, str, int]) -> None:
        stu_id, exer_id, cpt_seq, label = row
        self.writers[split].writerow([stu_id, exer_id, cpt_seq, label])
        self.stats[split].add(stu_id, exer_id, cpt_seq, label)

    def close(self) -> None:
        for f in self.files.values():
            f.close()

    def sizes(self) -> dict[str, int]:
        return {split: file_size(path) for split, path in self.paths.items()}

    def stats_dict(self) -> dict[str, dict]:
        return {split: stats.to_dict() for split, stats in self.stats.items()}


def split_positions(n: int, train_ratio: float = 0.8, valid_ratio: float = 0.1) -> tuple[int, int]:
    if n <= 1:
        return n, n
    train_end = max(1, int(n * train_ratio))
    valid_end = max(train_end, int(n * (train_ratio + valid_ratio)))
    if n >= 3:
        train_end = min(train_end, n - 2)
        valid_end = min(max(valid_end, train_end + 1), n - 1)
    else:
        valid_end = min(valid_end, n)
    return train_end, valid_end


def split_dataframe_by_student(df: pd.DataFrame, order_col: str | None = None) -> dict[str, pd.DataFrame]:
    parts = {"train": [], "valid": [], "test": []}
    sort_cols = ["stu_id"]
    if order_col and order_col in df.columns:
        sort_cols.append(order_col)
    sort_cols.append("_source_order")
    df = df.copy()
    df["_source_order"] = range(len(df))
    df = df.sort_values(sort_cols, kind="mergesort")
    for _, group in df.groupby("stu_id", sort=False):
        n = len(group)
        train_end, valid_end = split_positions(n)
        parts["train"].append(group.iloc[:train_end])
        if valid_end > train_end:
            parts["valid"].append(group.iloc[train_end:valid_end])
        if n > valid_end:
            parts["test"].append(group.iloc[valid_end:])
    out = {}
    cols = ["stu_id", "exer_id", "cpt_seq", "label"]
    for split, chunks in parts.items():
        out[split] = pd.concat(chunks, ignore_index=True)[cols] if chunks else pd.DataFrame(columns=cols)
    return out


def split_dataframe_by_student_random(df: pd.DataFrame, seed: int = 20260508) -> dict[str, pd.DataFrame]:
    """Per-student split for response-matrix data without true temporal order."""
    parts = {"train": [], "valid": [], "test": []}
    df = df.copy()
    df["_source_order"] = range(len(df))
    for stu_id, group in df.groupby("stu_id", sort=True):
        n = len(group)
        shuffled = group.sample(frac=1.0, random_state=seed + int(stu_id)).reset_index(drop=True)
        train_end, valid_end = split_positions(n)
        parts["train"].append(shuffled.iloc[:train_end])
        if valid_end > train_end:
            parts["valid"].append(shuffled.iloc[train_end:valid_end])
        if n > valid_end:
            parts["test"].append(shuffled.iloc[valid_end:])
    out = {}
    cols = ["stu_id", "exer_id", "cpt_seq", "label"]
    for split, chunks in parts.items():
        out[split] = pd.concat(chunks, ignore_index=True)[cols] if chunks else pd.DataFrame(columns=cols)
    return out


def dataframe_to_rows(df: pd.DataFrame) -> Iterable[tuple[int, int, str, int]]:
    for row in df.itertuples(index=False):
        yield int(row.stu_id), int(row.exer_id), str(row.cpt_seq), int(row.label)


def save_manifest(out_dir: Path, manifest: dict) -> None:
    manifest = dict(manifest)
    manifest["output_format"] = ["stu_id", "exer_id", "cpt_seq", "label"]
    manifest["processed_files"] = {
        split: {
            "path": str((out_dir / f"{split}.csv").as_posix()),
            "bytes": file_size(out_dir / f"{split}.csv"),
            "human_size": human_size(file_size(out_dir / f"{split}.csv")),
        }
        for split in ("train", "valid", "test")
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")


def adapt_matrix_dataset(name: str, raw_dir: Path, output_dir: Path, response_key: str, q_key: str, overwrite: bool) -> dict:
    dataset_raw = raw_dir / name
    out_dir = output_dir / name
    response_df = read_hf_csv(URLS[response_key], dataset_raw / "response.csv", overwrite=overwrite)
    q_df = read_hf_csv(URLS[q_key], dataset_raw / "q_matrix.csv", overwrite=overwrite)

    q_map = {}
    for item_idx, row in q_df.iterrows():
        cpts = [str(int(i)) for i, val in enumerate(row.tolist()) if normalize_binary(val) == 1]
        if cpts:
            q_map[int(item_idx)] = ",".join(cpts)

    rows = []
    dropped = {"missing_concepts": 0, "bad_label": 0, "bad_ids": 0}
    for r in response_df.itertuples(index=False):
        stu_id = normalize_int(r[0])
        exer_id = normalize_int(r[1])
        label = normalize_binary(r[2])
        if stu_id is None or exer_id is None:
            dropped["bad_ids"] += 1
            continue
        if label is None:
            dropped["bad_label"] += 1
            continue
        cpt_seq = q_map.get(exer_id, "")
        if not cpt_seq:
            dropped["missing_concepts"] += 1
            continue
        rows.append((stu_id, exer_id, cpt_seq, label))

    df = pd.DataFrame(rows, columns=["stu_id", "exer_id", "cpt_seq", "label"])
    splits = split_dataframe_by_student_random(df)
    split_stats = {
        split: write_csv(out_dir / f"{split}.csv", dataframe_to_rows(split_df))
        for split, split_df in splits.items()
    }
    manifest = {
        "dataset": name,
        "source": "HuggingFace shenjunhao/mind-data",
        "source_urls": [URLS[response_key], URLS[q_key]],
        "raw_files": {
            "response.csv": {"bytes": file_size(dataset_raw / "response.csv"), "human_size": human_size(file_size(dataset_raw / "response.csv"))},
            "q_matrix.csv": {"bytes": file_size(dataset_raw / "q_matrix.csv"), "human_size": human_size(file_size(dataset_raw / "q_matrix.csv"))},
        },
        "split_policy": "per-student seeded random 80/10/10 because source has no true temporal order",
        "has_true_temporal_order": False,
        "concept_source": "q_matrix.csv binary item-concept rows",
        "dropped": dropped,
        "splits": split_stats,
        "notes": (
            "The source provides response triples and a Q-matrix, but no timestamp. "
            "The output is model-ready for CD, while sequence-transition evidence "
            "should be interpreted cautiously or disabled in later training."
        ),
    }
    save_manifest(out_dir, manifest)
    return manifest


def extract_csv_from_zip(zip_path: Path, name_filter: Callable[[str], bool]) -> tuple[str, bytes]:
    with zipfile.ZipFile(zip_path) as zf:
        candidates = [n for n in zf.namelist() if name_filter(n)]
        if not candidates:
            raise ValueError(f"no matching CSV found in {zip_path}")
        candidates.sort(key=lambda n: zf.getinfo(n).file_size, reverse=True)
        name = candidates[0]
        return name, zf.read(name)


def adapt_assist2015(raw_dir: Path, output_dir: Path, overwrite: bool) -> dict:
    name = "assist_15"
    dataset_raw = raw_dir / name
    out_dir = output_dir / name
    zip_path = dataset_raw / "2015_100_skill_builders_main_problems.zip"
    download(URLS["assist2015"], zip_path, overwrite=overwrite)
    csv_name, payload = extract_csv_from_zip(zip_path, lambda n: n.lower().endswith(".csv"))
    df = pd.read_csv(io.BytesIO(payload))
    required = {"user_id", "log_id", "sequence_id", "correct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ASSIST2015 missing columns: {sorted(missing)}")
    out = pd.DataFrame(
        {
            "stu_id": df["user_id"].map(normalize_int),
            "exer_id": df["sequence_id"].map(normalize_int),
            "cpt_seq": df["sequence_id"].map(lambda v: str(normalize_int(v)) if normalize_int(v) is not None else ""),
            "label": df["correct"].map(normalize_binary),
            "order_id": df["log_id"].map(normalize_int),
        }
    )
    before = len(out)
    out = out.dropna(subset=["stu_id", "exer_id", "label"]).copy()
    out = out[out["cpt_seq"] != ""].copy()
    for col in ("stu_id", "exer_id", "label"):
        out[col] = out[col].astype(int)
    splits = split_dataframe_by_student(out, order_col="order_id")
    split_stats = {
        split: write_csv(out_dir / f"{split}.csv", dataframe_to_rows(split_df))
        for split, split_df in splits.items()
    }
    manifest = {
        "dataset": name,
        "source": "USTC BASE ASSISTment mirror",
        "source_urls": [URLS["assist2015"]],
        "raw_files": {zip_path.name: {"bytes": file_size(zip_path), "human_size": human_size(file_size(zip_path))}},
        "raw_csv_member": csv_name,
        "split_policy": "per-student 80/10/10 sorted by log_id",
        "has_true_temporal_order": True,
        "concept_source": "sequence_id used as both exercise and concept because source has no problem_id/Q-matrix",
        "dropped": {"invalid_or_missing_rows": int(before - len(out))},
        "splits": split_stats,
        "notes": (
            "ASSIST2015 is KC/sequence-level in this public file. It is usable as a "
            "KT-style benchmark, but exercise-level CD interpretation is weaker than "
            "datasets with explicit problem_id and Q-matrix."
        ),
    }
    save_manifest(out_dir, manifest)
    return manifest


def find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lower_map = {str(c).lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def adapt_assist2012(raw_dir: Path, output_dir: Path, overwrite: bool) -> dict:
    name = "assist_12"
    dataset_raw = raw_dir / name
    out_dir = output_dir / name
    zip_path = dataset_raw / "2012-2013-data-with-predictions-4-final.zip"
    download(URLS["assist2012"], zip_path, overwrite=overwrite)
    csv_name, payload = extract_csv_from_zip(zip_path, lambda n: n.lower().endswith(".csv"))
    df = pd.read_csv(io.BytesIO(payload), low_memory=False)
    stu_col = find_column(df.columns, ["user_id", "student_id", "anon_student_id"])
    exer_col = find_column(df.columns, ["problem_id", "item_id", "question_id", "problem_log_id"])
    cpt_col = find_column(df.columns, ["skill_id", "knowledge_component_id", "kc_id"])
    label_col = find_column(df.columns, ["correct", "is_correct", "score"])
    order_col = find_column(df.columns, ["start_time", "order_id", "ms_first_response", "problem_log_id", "log_id"])
    required = {"stu_id": stu_col, "exer_id": exer_col, "cpt_seq": cpt_col, "label": label_col}
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(f"ASSIST2012 could not identify columns: {missing}; columns={list(df.columns)[:50]}")
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
    split_stats = {
        split: write_csv(out_dir / f"{split}.csv", dataframe_to_rows(split_df))
        for split, split_df in splits.items()
    }
    manifest = {
        "dataset": name,
        "source": "USTC BASE ASSISTment mirror",
        "source_urls": [URLS["assist2012"]],
        "raw_files": {zip_path.name: {"bytes": file_size(zip_path), "human_size": human_size(file_size(zip_path))}},
        "raw_csv_member": csv_name,
        "source_columns": {"student": stu_col, "exercise": exer_col, "concept": cpt_col, "label": label_col, "order": order_col},
        "split_policy": f"per-student 80/10/10 sorted by {order_col or 'source order'}",
        "has_true_temporal_order": bool(order_col),
        "concept_source": cpt_col,
        "dropped": {"invalid_or_missing_rows": int(before - len(out))},
        "splits": split_stats,
    }
    save_manifest(out_dir, manifest)
    return manifest


def adapt_nips34(raw_dir: Path, output_dir: Path, overwrite: bool) -> dict:
    name = "nips34"
    dataset_raw = raw_dir / name
    out_dir = output_dir / name
    zip_path = dataset_raw / "public_data.zip"
    download(URLS["nips2020"], zip_path, overwrite=overwrite)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        train_name = next((n for n in names if n.lower().endswith("train_task_3_4.csv")), None)
        q_meta_name = next((n for n in names if n.lower().endswith("question_metadata_task_3_4.csv")), None)
        if train_name is None:
            train_name = next((n for n in names if "task_3_4" in n.lower() and "train" in n.lower() and n.lower().endswith(".csv")), None)
        if q_meta_name is None:
            q_meta_name = next((n for n in names if "question" in n.lower() and "metadata" in n.lower() and "3_4" in n.lower() and n.lower().endswith(".csv")), None)
        if train_name is None or q_meta_name is None:
            raise ValueError(f"NIPS34 files not found; train={train_name}, question_meta={q_meta_name}, names={names[:50]}")
        train_df = pd.read_csv(zf.open(train_name), low_memory=False)
        q_df = pd.read_csv(zf.open(q_meta_name), low_memory=False)

    q_col = find_column(q_df.columns, ["QuestionId", "question_id"])
    subj_col = find_column(q_df.columns, ["SubjectId", "subject_id", "KnowledgeTag", "ConstructId"])
    if q_col is None or subj_col is None:
        raise ValueError(f"NIPS34 question metadata columns not recognized: {list(q_df.columns)}")
    concept_map = {}
    for row in q_df.itertuples(index=False):
        qid = normalize_int(getattr(row, q_col))
        cpt_seq = parse_concepts(getattr(row, subj_col))
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
    split_stats = {
        split: write_csv(out_dir / f"{split}.csv", dataframe_to_rows(split_df))
        for split, split_df in splits.items()
    }
    manifest = {
        "dataset": name,
        "source": "USTC BASE NIPS2020 public_data mirror",
        "source_urls": [URLS["nips2020"]],
        "raw_files": {zip_path.name: {"bytes": file_size(zip_path), "human_size": human_size(file_size(zip_path))}},
        "raw_csv_members": {"train": train_name, "question_metadata": q_meta_name},
        "source_columns": {"student": stu_col, "exercise": exer_col, "concept": subj_col, "label": label_col, "order": order_col},
        "split_policy": f"per-student 80/10/10 sorted by {order_col or 'source order'}",
        "has_true_temporal_order": bool(order_col),
        "concept_source": "question_metadata_task_3_4 SubjectId",
        "dropped": {"invalid_or_missing_rows": int(before - len(out))},
        "splits": split_stats,
    }
    save_manifest(out_dir, manifest)
    return manifest


def adapt_ednet_kt1(raw_dir: Path, output_dir: Path, overwrite: bool) -> dict:
    name = "ednet_kt1"
    dataset_raw = raw_dir / name
    out_dir = output_dir / name
    contents_zip = dataset_raw / "EdNet-Contents.zip"
    kt1_zip = dataset_raw / "EdNet-KT1.zip"
    download(URLS["ednet_contents"], contents_zip, overwrite=overwrite)
    download(URLS["ednet_kt1"], kt1_zip, overwrite=overwrite)

    with zipfile.ZipFile(contents_zip) as zf:
        q_name = next((n for n in zf.namelist() if n.endswith("contents/questions.csv")), None)
        if q_name is None:
            raise ValueError("contents/questions.csv not found in EdNet contents zip")
        q_df = pd.read_csv(zf.open(q_name))
    q_map: dict[str, tuple[int, str, str]] = {}
    for row in q_df.itertuples(index=False):
        qid_text = str(getattr(row, "question_id")).strip()
        qid = normalize_int(qid_text)
        cpt_seq = parse_concepts(getattr(row, "tags"))
        correct = str(getattr(row, "correct_answer")).strip().lower()
        if qid is not None and cpt_seq and correct:
            q_map[qid_text] = (qid, cpt_seq, correct)

    dropped = {
        "missing_question_metadata": 0,
        "bad_answer": 0,
        "bad_user_file": 0,
        "empty_user_history": 0,
    }
    writers = StreamingSplitWriters(out_dir)
    total_user_files = 0
    processed_user_files = 0
    try:
        with zipfile.ZipFile(kt1_zip) as zf:
            members = [
                info for info in zf.infolist()
                if info.filename.lower().endswith(".csv")
                and "__macosx" not in info.filename.lower()
                and "/u" in info.filename.lower()
            ]
            total_user_files = len(members)
            log(f"[ednet_kt1] user CSV files={total_user_files}")
            for info in members:
                basename = Path(info.filename).stem
                stu_id = normalize_int(basename)
                if stu_id is None:
                    dropped["bad_user_file"] += 1
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
                rows.sort(key=lambda x: x[0])
                n = len(rows)
                train_end, valid_end = split_positions(n)
                for idx, (_, s, e, c, y) in enumerate(rows):
                    split = "train" if idx < train_end else "valid" if idx < valid_end else "test"
                    writers.write(split, (s, e, c, y))
                processed_user_files += 1
                if processed_user_files % 10000 == 0:
                    log(f"[ednet_kt1] processed users={processed_user_files}/{total_user_files}; train_rows={writers.stats['train'].rows}")
    finally:
        writers.close()

    split_stats = writers.stats_dict()
    manifest = {
        "dataset": name,
        "source": "USTC BASE EdNet mirror",
        "source_urls": [URLS["ednet_contents"], URLS["ednet_kt1"]],
        "raw_files": {
            contents_zip.name: {"bytes": file_size(contents_zip), "human_size": human_size(file_size(contents_zip))},
            kt1_zip.name: {"bytes": file_size(kt1_zip), "human_size": human_size(file_size(kt1_zip))},
        },
        "raw_csv_members": {"question_metadata": q_name, "kt1_user_csv_files": total_user_files},
        "split_policy": "per-user 80/10/10 sorted by timestamp",
        "has_true_temporal_order": True,
        "concept_source": "contents/questions.csv tags",
        "dropped": dropped,
        "processed_user_files": processed_user_files,
        "splits": split_stats,
        "notes": "EdNet-KT1 is large. Raw archive is not intended for git; processed CSVs may also be too large for normal commits.",
    }
    save_manifest(out_dir, manifest)
    return manifest


ADAPTERS = {
    "frcsub": lambda raw, out, ow: adapt_matrix_dataset("frcsub", raw, out, "frcsub_response", "frcsub_q", ow),
    "math2": lambda raw, out, ow: adapt_matrix_dataset("math2", raw, out, "math2_response", "math2_q", ow),
    "assist_12": adapt_assist2012,
    "assist_15": adapt_assist2015,
    "nips34": adapt_nips34,
    "ednet_kt1": adapt_ednet_kt1,
}


def summarize_all(output_dir: Path, datasets: list[str]) -> list[dict]:
    rows = []
    for name in datasets:
        manifest_path = output_dir / name / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_bytes = sum(v.get("bytes", 0) for v in manifest.get("raw_files", {}).values())
        processed_bytes = sum(v.get("bytes", 0) for v in manifest.get("processed_files", {}).values())
        total = SplitStats()
        for split_stats in manifest.get("splits", {}).values():
            synthetic = SplitStats()
            synthetic.rows = int(split_stats.get("rows") or 0)
            synthetic.positives = int(round((split_stats.get("positive_rate") or 0.0) * synthetic.rows))
            total.rows += synthetic.rows
            total.positives += synthetic.positives
        rows.append(
            {
                "dataset": name,
                "raw_size": human_size(raw_bytes),
                "processed_size": human_size(processed_bytes),
                "train_rows": manifest.get("splits", {}).get("train", {}).get("rows"),
                "valid_rows": manifest.get("splits", {}).get("valid", {}).get("rows"),
                "test_rows": manifest.get("splits", {}).get("test", {}).get("rows"),
                "train_students": manifest.get("splits", {}).get("train", {}).get("students"),
                "train_items": manifest.get("splits", {}).get("train", {}).get("items"),
                "train_concepts": manifest.get("splits", {}).get("train", {}).get("concepts"),
                "has_true_temporal_order": manifest.get("has_true_temporal_order"),
                "manifest": str(manifest_path.as_posix()),
            }
        )
    return rows


def write_summary(output_dir: Path, datasets: list[str]) -> Path:
    rows = summarize_all(output_dir, datasets)
    summary_path = output_dir / "public_benchmark_manifest_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "dataset",
            "raw_size",
            "processed_size",
            "train_rows",
            "valid_rows",
            "test_rows",
            "train_students",
            "train_items",
            "train_concepts",
            "has_true_temporal_order",
            "manifest",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def parse_dataset_names(values: list[str], all_flag: bool) -> list[str]:
    if all_flag:
        return DEFAULT_DATASETS
    names = []
    for value in values:
        key = value.strip().lower()
        if not key:
            continue
        if key not in DATASET_ALIASES:
            raise ValueError(f"unknown dataset {value!r}; choices={sorted(DATASET_ALIASES)}")
        names.append(DATASET_ALIASES[key])
    seen = set()
    ordered = []
    for name in names:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", default=[], help="Dataset to adapt. Can be passed multiple times.")
    parser.add_argument("--all", action="store_true", help="Adapt all supported public benchmarks.")
    parser.add_argument("--raw-dir", default=str(RAW_DIR_DEFAULT))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR_DEFAULT))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    datasets = parse_dataset_names(args.dataset, args.all)
    if not datasets:
        parser.error("pass --all or at least one --dataset")

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    manifests = []
    for name in datasets:
        log(f"[adapt] start {name}")
        manifest = ADAPTERS[name](raw_dir, output_dir, bool(args.overwrite))
        manifests.append(manifest)
        log(f"[adapt] done {name}: {json.dumps(manifest.get('splits', {}), ensure_ascii=False)}")
    summary_path = write_summary(output_dir, datasets)
    log(f"[summary] wrote {summary_path}")
    print(json.dumps(summarize_all(output_dir, datasets), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
