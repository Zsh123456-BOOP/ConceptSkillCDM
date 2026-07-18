"""Extract learned count-gate parameters from the final checkpoints.

The anchor gate of channel ch is sigmoid(a_ch + b_ch * log1p(n)). This script
reads `anchor_gate` from each final full-model checkpoint and writes the
direct-channel (a, b) pair per dataset, so the figure can overlay the learned
gate curves on the Bayesian shrinkage weight n / (n + 2) of the smoothed rate
estimator itself.
"""

from __future__ import annotations

import argparse
import csv
import os

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FINAL_CHECKPOINTS = {
    "assist_17": "checkpoints/assist_17_mh17_mh_0717/best_model.pth",
    "junyi": "checkpoints/junyi_mh_lv_0717/best_model.pth",
    "nips34": "checkpoints/nips34_mh_lv_0717/best_model.pth",
    "ednet_kt1": "checkpoints/ednet_kt1_mhed_mh_0717/best_model.pth",
    "moocradar": "checkpoints/moocradar_mhmo_mh_0717/best_model.pth",
    "xes3g5m": "checkpoints/xes3g5m_mhxe_mh_0717/best_model.pth",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_csv", default="results/gate_reliability.csv")
    args = parser.parse_args()

    rows = []
    for dataset, rel_path in FINAL_CHECKPOINTS.items():
        ck = torch.load(os.path.join(ROOT, rel_path), map_location="cpu")
        state = ck.get("model_state_dict") or ck.get("state_dict") or ck
        gate = state["anchor_gate"]
        for channel in range(gate.shape[0]):
            rows.append(
                {
                    "dataset": dataset,
                    "channel": channel,
                    "a": float(gate[channel, 0]),
                    "b": float(gate[channel, 1]),
                }
            )
        print(f"{dataset}: gate shape {tuple(gate.shape)}")

    out_path = os.path.join(ROOT, args.output_csv)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "channel", "a", "b"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
