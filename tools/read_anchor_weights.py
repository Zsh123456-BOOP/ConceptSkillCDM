#!/usr/bin/env python
"""Print learned evidence-anchor channel weights across checkpoints (S4).

The full model learns one non-negative weight per evidence channel
(direct rate, difficulty residual, graph-propagated rate).  Comparing them
across datasets tests the interpretability claim: datasets with dense direct
evidence should favour the direct channel, datasets with large concept
evidence gaps should favour the propagated channel.
"""

from __future__ import annotations

import argparse
import glob
import os

import torch
import torch.nn.functional as F


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint_glob",
        required=True,
        help="Glob over checkpoint dirs, e.g. 'checkpoints/*_v10d_0715'",
    )
    args = parser.parse_args()

    print(f"{'run':<50} channel-wise mean softplus weights (direct, residual, prop-heads...)")
    for run_dir in sorted(glob.glob(args.checkpoint_glob)):
        model_path = os.path.join(run_dir, "best_model.pth")
        if not os.path.isfile(model_path):
            continue
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model_state_dict", {})
        raw = state.get("diagnosis_head.evidence_anchor_raw")
        if raw is None:
            print(f"{os.path.basename(run_dir):<50} (no anchor)")
            continue
        weights = F.softplus(raw)
        if weights.dim() == 1:
            weights = weights.unsqueeze(0)
        channel_means = weights.mean(dim=0).tolist()
        formatted = " ".join(f"{value:.3f}" for value in channel_means)
        print(f"{os.path.basename(run_dir):<50} [{formatted}]")


if __name__ == "__main__":
    main()
