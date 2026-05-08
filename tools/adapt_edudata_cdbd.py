#!/usr/bin/env python
"""Download and adapt EduData CDBD datasets to this repo's CSV format.

Output format:
    stu_id,exer_id,cpt_seq,label

The script intentionally only handles data import/adaptation. It does not touch
training code, logs, results, checkpoints, or README files.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd


BASE_URL = "http://base.ustc.edu.cn/data/cdbd"

DATASETS = {
    "cdbd_lsat": {
        "base_url": f"{BASE_URL}/LSAT",
        "splits": ("train.csv", "valid.csv", "test.csv"),
        "item_file": None,
        "concept_source": "item_id_as_concept",
        "notes": (
            "EduData CDBD LSAT does not ship an item-to-concept metadata file. "
            "The adapter uses each item_id as its own cpt_seq so the dataset can "
            "enter the repo's Q-matrix pipeline; concept semantics should be "
            "reviewed before using it as a final CD benchmark."
        ),
    },
    "cdbd_a0910": {
        "base_url": f"{BASE_URL}/a0910",
        "splits": ("train.csv", "valid.csv", "test.csv"),
        "item_file": "item.csv",
        "concept_source": "item.csv:knowledge_code",
        "notes": (
            "EduData CDBD ASSISTments 2009-2010 variant. This overlaps the repo's "
            "existing assist_09 family, but has EduData's CDBD split and explicit "
            "item.csv knowledge_code metadata."
        ),
    },
}


def _download(url: str, target: Path, overwrite: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        return
    urlretrieve(url, target)


def _parse_knowledge_code(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(int(v)) for v in value)
    text = str(value).strip()
    if not text:
        return ""
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = text
    if isinstance(parsed, (list, tuple, set)):
        return ",".join(str(int(v)) for v in parsed)
    return str(int(parsed))


def _load_item_concepts(raw_dir: Path, item_file: str | None) -> dict[int, str]:
    if item_file is None:
        return {}
    item_df = pd.read_csv(raw_dir / item_file)
    required = {"item_id", "knowledge_code"}
    missing = required - set(item_df.columns)
    if missing:
        raise ValueError(f"{item_file} missing required columns: {sorted(missing)}")
    item_df = item_df.dropna(subset=["item_id", "knowledge_code"]).copy()
    item_df["item_id"] = item_df["item_id"].astype(int)
    item_df["cpt_seq"] = item_df["knowledge_code"].map(_parse_knowledge_code)
    item_df = item_df[item_df["cpt_seq"] != ""]
    return dict(zip(item_df["item_id"], item_df["cpt_seq"]))


def _convert_split(src: Path, dst: Path, concept_map: dict[int, str]) -> dict:
    raw = pd.read_csv(src)
    required = {"user_id", "item_id", "score"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{src.name} missing required columns: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "stu_id": raw["user_id"].astype(int),
            "exer_id": raw["item_id"].astype(int),
            "label": raw["score"].astype(float),
        }
    )
    if not out["label"].isin([0.0, 1.0]).all():
        bad = sorted(out.loc[~out["label"].isin([0.0, 1.0]), "label"].unique().tolist())
        raise ValueError(f"{src.name} has non-binary score values: {bad[:10]}")

    if concept_map:
        out["cpt_seq"] = out["exer_id"].map(concept_map)
        before = len(out)
        out = out.dropna(subset=["cpt_seq"]).copy()
        dropped_missing_concepts = before - len(out)
    else:
        out["cpt_seq"] = out["exer_id"].astype(str)
        dropped_missing_concepts = 0

    out["label"] = out["label"].astype(int)
    out = out[["stu_id", "exer_id", "cpt_seq", "label"]]
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)

    concept_ids = set()
    for seq in out["cpt_seq"]:
        concept_ids.update(int(part) for part in str(seq).split(",") if part)
    return {
        "rows": int(len(out)),
        "students": int(out["stu_id"].nunique()),
        "items": int(out["exer_id"].nunique()),
        "concepts": int(len(concept_ids)),
        "positive_rate": float(out["label"].mean()) if len(out) else None,
        "dropped_missing_concepts": int(dropped_missing_concepts),
    }


def adapt_dataset(name: str, output_dir: Path, raw_dir: Path, overwrite: bool) -> dict:
    if name not in DATASETS:
        raise ValueError(f"unsupported dataset {name!r}; choose from {sorted(DATASETS)}")

    spec = DATASETS[name]
    dataset_raw_dir = raw_dir / name
    dataset_out_dir = output_dir / name
    dataset_raw_dir.mkdir(parents=True, exist_ok=True)
    dataset_out_dir.mkdir(parents=True, exist_ok=True)

    for filename in spec["splits"]:
        _download(f"{spec['base_url']}/{filename}", dataset_raw_dir / filename, overwrite)
    if spec["item_file"]:
        _download(f"{spec['base_url']}/{spec['item_file']}", dataset_raw_dir / spec["item_file"], overwrite)

    concept_map = _load_item_concepts(dataset_raw_dir, spec["item_file"])
    split_stats = {}
    for filename in spec["splits"]:
        split = filename.replace(".csv", "")
        split_stats[split] = _convert_split(
            dataset_raw_dir / filename,
            dataset_out_dir / filename,
            concept_map,
        )

    manifest = {
        "dataset": name,
        "source": "EduData CDBD",
        "source_url": spec["base_url"],
        "concept_source": spec["concept_source"],
        "output_format": ["stu_id", "exer_id", "cpt_seq", "label"],
        "splits": split_stats,
        "notes": spec["notes"],
    }
    with (dataset_out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--raw_dir", default=os.path.join("data", "_edudata_raw"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = adapt_dataset(
        name=args.dataset,
        output_dir=Path(args.output_dir),
        raw_dir=Path(args.raw_dir),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
