import os
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class InteractionDataset(Dataset):
    """
    学生-题目交互数据集，同时包含对应的概念集合。
    """

    def __init__(
        self,
        df: pd.DataFrame,
        student2idx: Dict[int, int],
        item2idx: Dict[int, int],
        concept2idx: Dict[int, int],
    ):
        self.student2idx = student2idx
        self.item2idx = item2idx
        self.concept2idx = concept2idx

        records = []
        for _, row in df.iterrows():
            s_raw = int(row["student_id"])
            i_raw = int(row["item_id"])
            c_list_raw: List[int] = row["concept_list"]
            label = float(row["correct"])

            s_idx = self.student2idx[s_raw]
            i_idx = self.item2idx[i_raw]
            c_idx_list = [self.concept2idx[c] for c in c_list_raw]

            records.append(
                (
                    s_idx,
                    i_idx,
                    torch.tensor(c_idx_list, dtype=torch.long),
                    torch.tensor(label, dtype=torch.float),
                )
            )

        self.records = records
        self.num_students = len(student2idx)
        self.num_items = len(item2idx)
        self.num_concepts = len(concept2idx)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    if "student_id" not in df.columns and "stu_id" in df.columns:
        rename_map["stu_id"] = "student_id"
    if "item_id" not in df.columns and "exer_id" in df.columns:
        rename_map["exer_id"] = "item_id"
    if "correct" not in df.columns and "label" in df.columns:
        rename_map["label"] = "correct"
    if "concept_ids" not in df.columns and "cpt_seq" in df.columns:
        rename_map["cpt_seq"] = "concept_ids"
    df = df.rename(columns=rename_map)

    required = ["student_id", "item_id", "correct", "concept_ids"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def _parse_concept_list(concept_str) -> List[int]:
    if pd.isna(concept_str):
        return []
    cleaned = str(concept_str).replace('"', "").replace("'", "")
    parts = cleaned.replace(",", ";").split(";")
    return [int(p) for p in parts if str(p).strip() != ""]


def _build_mappings(dfs: List[pd.DataFrame]) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
    students, items, concepts = set(), set(), set()
    for df in dfs:
        students.update(df["student_id"].astype(int).unique().tolist())
        items.update(df["item_id"].astype(int).unique().tolist())
        for lst in df["concept_list"]:
            concepts.update(lst)
    student2idx = {sid: idx for idx, sid in enumerate(sorted(students))}
    item2idx = {iid: idx for idx, iid in enumerate(sorted(items))}
    concept2idx = {cid: idx for idx, cid in enumerate(sorted(concepts))}
    return student2idx, item2idx, concept2idx


def _collate_fn(batch):
    student_ids = torch.tensor([b[0] for b in batch], dtype=torch.long)
    item_ids = torch.tensor([b[1] for b in batch], dtype=torch.long)
    labels = torch.stack([b[3] for b in batch]).float()

    concept_ptr = [0]
    all_concepts: List[int] = []
    for _, _, c_ids, _ in batch:
        concept_ptr.append(concept_ptr[-1] + len(c_ids))
        all_concepts.extend(c_ids.tolist())

    batch_concept_ids = torch.tensor(all_concepts, dtype=torch.long)
    concept_ptr = torch.tensor(concept_ptr, dtype=torch.long)

    return {
        "student_ids": student_ids,
        "item_ids": item_ids,
        "batch_concept_ids": batch_concept_ids,
        "concept_ptr": concept_ptr,
        "labels": labels,
    }


def get_dataloaders(config):
    base_dir = os.path.join(config.data_dir, config.dataset)
    train_path = os.path.join(base_dir, "train.csv")
    valid_path = os.path.join(base_dir, "valid.csv")
    test_path = os.path.join(base_dir, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found at {train_path} (dataset={config.dataset})")
    if not os.path.exists(valid_path):
        raise FileNotFoundError(f"Valid file not found at {valid_path} (dataset={config.dataset})")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found at {test_path} (dataset={config.dataset})")

    train_df = _standardize_columns(pd.read_csv(train_path))
    valid_df = _standardize_columns(pd.read_csv(valid_path))
    test_df = _standardize_columns(pd.read_csv(test_path))

    for df in (train_df, valid_df, test_df):
        df["concept_list"] = df["concept_ids"].apply(_parse_concept_list)

    student2idx, item2idx, concept2idx = _build_mappings([train_df, valid_df, test_df])

    train_dataset = InteractionDataset(train_df, student2idx, item2idx, concept2idx)
    valid_dataset = InteractionDataset(valid_df, student2idx, item2idx, concept2idx)
    test_dataset = InteractionDataset(test_df, student2idx, item2idx, concept2idx)

    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": False,
        "collate_fn": _collate_fn,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    dataset_info = {
        "num_students": train_dataset.num_students,
        "num_items": train_dataset.num_items,
        "num_concepts": train_dataset.num_concepts,
        "student2idx": student2idx,
        "item2idx": item2idx,
        "concept2idx": concept2idx,
    }

    return train_loader, valid_loader, test_loader, dataset_info


__all__ = ["InteractionDataset", "get_dataloaders"]
