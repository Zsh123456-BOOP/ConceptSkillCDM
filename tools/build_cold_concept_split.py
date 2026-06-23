#!/usr/bin/env python3
"""Build a cold-concept-per-student split to make CRG support load-bearing.

Motivation (Path 2 of the mainline redesign): in the public splits, even
"direct-unseen" test queries usually have the target concept somewhere else in
the *same student's* training history, so the GNN backbone / student embedding
can recover the target mastery without the LCRF route support -> support and
LCRF measure as decorative (E4 null).

This builder removes a controlled subset of (student, concept) pairs from a
student's TRAIN (and VALID) interactions while keeping that student's TEST
interactions on those concepts. Key property: the held-out concept is removed
only for THAT student, so it still appears in training for other students and
CRG can still learn its support edges from the global graph. For the affected
student, the only remaining route to the target concept is the concept relation
support -> if the method's thesis is true, perturbing that support (E4) should
now hurt; if it is still decorative, the thesis is false even under the ideal
task and a method change is required.

Output (under data/<dataset><out_suffix>/):
  train.csv, valid.csv, test.csv         (test identical to source)
  cold_concept_pairs.csv                  (stu_id, cold_concept)
  test_cold_annotation.csv               (event-level is_cold_query flag)
  cold_concept_split_summary.csv         (rates + sanity checks)

This script does NOT train or change any model. Run locally:
  python tools/build_cold_concept_split.py --dataset assist_09
"""
from __future__ import annotations

import argparse
import os
import random
from typing import Dict, List, Set, Tuple

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _concepts(cpt_seq: object) -> List[str]:
    if cpt_seq is None:
        return []
    return [c for c in str(cpt_seq).split(",") if str(c).strip() != ""]


def _student_concept_sets(df: pd.DataFrame) -> Dict[int, Set[str]]:
    out: Dict[int, Set[str]] = {}
    for sid, seq in zip(df["stu_id"].tolist(), df["cpt_seq"].tolist()):
        out.setdefault(int(sid), set()).update(_concepts(seq))
    return out


def build_split(
    dataset: str,
    data_root: str,
    out_suffix: str,
    holdout_frac: float,
    min_train_rows_keep: int,
    max_cold_per_student: int,
    seed: int,
) -> Dict[str, object]:
    src = os.path.join(data_root, dataset)
    train = pd.read_csv(os.path.join(src, "train.csv"))
    valid = pd.read_csv(os.path.join(src, "valid.csv"))
    test = pd.read_csv(os.path.join(src, "test.csv"))

    train_sc = _student_concept_sets(train)
    test_sc = _student_concept_sets(test)
    global_train_concepts: Set[str] = set().union(*train_sc.values()) if train_sc else set()

    # A concept may be made cold for a student only if at least one OTHER student
    # also has it in train; otherwise removing it would delete the concept from
    # the global graph entirely and CRG could not route it.
    concept_student_count: Dict[str, int] = {}
    for _sid, cset in train_sc.items():
        for c in cset:
            concept_student_count[c] = concept_student_count.get(c, 0) + 1
    shared_concepts = {c for c, n in concept_student_count.items() if n >= 2}

    rng = random.Random(seed)
    cold_pairs: List[Tuple[int, str]] = []
    cold_by_student: Dict[int, Set[str]] = {}

    # train row count per student (to enforce min retention)
    train_rows_per_student = train.groupby("stu_id").size().to_dict()

    for sid in sorted(train_sc.keys()):
        eligible = sorted(train_sc[sid] & test_sc.get(sid, set()) & shared_concepts)
        if not eligible:
            continue
        n_hold = max(1, int(round(holdout_frac * len(eligible))))
        if max_cold_per_student > 0:
            n_hold = min(n_hold, max_cold_per_student)
        # do not hold out all of a student's concepts
        n_hold = min(n_hold, max(0, len(eligible) - 1))
        if n_hold <= 0:
            continue
        chosen = set(rng.sample(eligible, n_hold))
        # estimate train rows that would be removed for this student
        s_train = train[train["stu_id"] == sid]
        removed_mask = s_train["cpt_seq"].apply(
            lambda q: bool(set(_concepts(q)) & chosen)
        )
        kept = int((~removed_mask).sum())
        if kept < min_train_rows_keep:
            # shrink chosen until retention satisfied
            chosen_list = list(chosen)
            rng.shuffle(chosen_list)
            while chosen_list and kept < min_train_rows_keep:
                chosen_list.pop()
                chosen = set(chosen_list)
                if not chosen:
                    break
                removed_mask = s_train["cpt_seq"].apply(
                    lambda q: bool(set(_concepts(q)) & chosen)
                )
                kept = int((~removed_mask).sum())
        if not chosen:
            continue
        cold_by_student[sid] = chosen
        for c in sorted(chosen):
            cold_pairs.append((sid, c))

    def _row_is_cold(sid: int, seq: object) -> bool:
        cold = cold_by_student.get(int(sid))
        if not cold:
            return False
        return bool(set(_concepts(seq)) & cold)

    # Remove cold (student, concept) interactions from train and valid.
    train_keep = ~train.apply(lambda r: _row_is_cold(r["stu_id"], r["cpt_seq"]), axis=1)
    valid_keep = ~valid.apply(lambda r: _row_is_cold(r["stu_id"], r["cpt_seq"]), axis=1)
    new_train = train[train_keep].reset_index(drop=True)
    new_valid = valid[valid_keep].reset_index(drop=True)

    # Annotate test: a query is "cold" if any of its concepts was made cold for
    # that student (genuine zero-train-history target concept for this learner).
    test = test.copy()
    test["is_cold_query"] = test.apply(
        lambda r: _row_is_cold(r["stu_id"], r["cpt_seq"]), axis=1
    )

    out_dir = os.path.join(data_root, f"{dataset}{out_suffix}")
    os.makedirs(out_dir, exist_ok=True)
    new_train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    new_valid.to_csv(os.path.join(out_dir, "valid.csv"), index=False)
    test.drop(columns=["is_cold_query"]).to_csv(os.path.join(out_dir, "test.csv"), index=False)
    pd.DataFrame(cold_pairs, columns=["stu_id", "cold_concept"]).to_csv(
        os.path.join(out_dir, "cold_concept_pairs.csv"), index=False
    )
    test[["stu_id", "exer_id", "cpt_seq", "is_cold_query"]].to_csv(
        os.path.join(out_dir, "test_cold_annotation.csv"), index=False
    )

    # Sanity: do the held-out concepts still exist in the GLOBAL train graph?
    cold_concepts = {c for _, c in cold_pairs}
    new_train_concepts: Set[str] = set().union(*_student_concept_sets(new_train).values()) if len(new_train) else set()
    cold_present_global = sum(1 for c in cold_concepts if c in new_train_concepts)

    summary = {
        "dataset": dataset,
        "holdout_frac": holdout_frac,
        "seed": seed,
        "students_total": len(train_sc),
        "students_with_cold": len(cold_by_student),
        "cold_pairs": len(cold_pairs),
        "cold_concepts_distinct": len(cold_concepts),
        "cold_concepts_still_in_global_train": cold_present_global,
        "cold_concepts_lost_from_global_train": len(cold_concepts) - cold_present_global,
        "train_rows_before": len(train),
        "train_rows_after": len(new_train),
        "train_rows_removed": len(train) - len(new_train),
        "test_rows": len(test),
        "test_cold_queries": int(test["is_cold_query"].sum()),
        "test_cold_query_rate": round(float(test["is_cold_query"].mean()), 4),
        "out_dir": out_dir,
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(out_dir, "cold_concept_split_summary.csv"), index=False
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", nargs="+", default=["assist_09", "junyi", "assist_17"])
    ap.add_argument("--data-root", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out-suffix", default="_coldconcept")
    ap.add_argument("--holdout-frac", type=float, default=0.30,
                    help="fraction of each student's eligible (train∩test) concepts to make cold")
    ap.add_argument("--min-train-rows-keep", type=int, default=5)
    ap.add_argument("--max-cold-per-student", type=int, default=0, help="0 = no cap")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"{'dataset':12s} {'cold_rate':9s} {'cold_q':8s} {'students':9s} "
          f"{'train_rm':9s} {'concepts_kept_global'}")
    for ds in args.dataset:
        s = build_split(
            ds, args.data_root, args.out_suffix, args.holdout_frac,
            args.min_train_rows_keep, args.max_cold_per_student, args.seed,
        )
        print(f"{s['dataset']:12s} {s['test_cold_query_rate']!s:9s} "
              f"{s['test_cold_queries']!s:8s} {s['students_with_cold']!s:9s} "
              f"{s['train_rows_removed']!s:9s} "
              f"{s['cold_concepts_still_in_global_train']}/{s['cold_concepts_distinct']}")
        if s["cold_concepts_lost_from_global_train"] > 0:
            print(f"  [warn] {s['cold_concepts_lost_from_global_train']} cold concepts "
                  f"vanished from global train (CRG cannot route them); consider lowering holdout-frac")


if __name__ == "__main__":
    main()
