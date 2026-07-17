"""Aggregate multi-seed test AUC/ACC for the main and ablation tables.

Seeds 42 (final single-seed runs) plus 43/44/2025/2026/2027 (ms6_0717 queue).
Writes per-run rows and mean/std summaries used by the paper tables.
"""
import csv
import json
import os
import statistics

CK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
OUT = os.path.join(os.path.dirname(CK), "results")

SEED42_FULL = {
    "assist_17": "assist_17_mh17_mh_0717",
    "ednet_kt1": "ednet_kt1_mhed_mh_0717",
    "moocradar": "moocradar_mhmo_mh_0717",
    "xes3g5m": "xes3g5m_mhxe_mh_0717",
    "nips34": "nips34_mh_lv_0717",
    "junyi": "junyi_mh_lv_0717",
}
DATASETS = ["assist_17", "nips34", "ednet_kt1", "moocradar", "xes3g5m", "junyi"]
VARIANTS = ["full", "woA", "woB"]
MS6_SEEDS = [43, 44, 2025, 2026, 2027]


def read_metrics(ck_dir):
    path = os.path.join(CK, ck_dir, "test_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        m = json.load(f)["metrics"]
    return m["auc"], m["acc"]


def main():
    runs = []
    for ds in DATASETS:
        for var in VARIANTS:
            d42 = SEED42_FULL[ds] if var == "full" else f"{ds}_{var}_fin_0717"
            m = read_metrics(d42)
            if m:
                runs.append((ds, var, 42, *m))
            for seed in MS6_SEEDS:
                m = read_metrics(f"{ds}_{var}_s{seed}_ms6_0717")
                if m:
                    runs.append((ds, var, seed, *m))

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "multiseed_auc_runs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "variant", "seed", "auc", "acc"])
        w.writerows(runs)

    with open(os.path.join(OUT, "multiseed_auc_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "variant", "n_seeds", "auc_mean", "auc_std", "acc_mean", "acc_std"])
        for ds in DATASETS:
            for var in VARIANTS:
                aucs = [r[3] for r in runs if r[0] == ds and r[1] == var]
                accs = [r[4] for r in runs if r[0] == ds and r[1] == var]
                if not aucs:
                    continue
                w.writerow([
                    ds, var, len(aucs),
                    f"{statistics.mean(aucs):.6f}",
                    f"{statistics.stdev(aucs):.6f}" if len(aucs) > 1 else "0",
                    f"{statistics.mean(accs):.6f}",
                    f"{statistics.stdev(accs):.6f}" if len(accs) > 1 else "0",
                ])
    print(f"runs={len(runs)} -> {OUT}/multiseed_auc_runs.csv, multiseed_auc_summary.csv")


if __name__ == "__main__":
    main()
