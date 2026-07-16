#!/usr/bin/env python
"""Second-eigenvalue (spectral gap) of learned relation matrices (A3 note).

A small second eigenvalue of the row-stochastic relation matrix means fast
mixing: two-hop propagation is already close to the stationary distribution
and carries little individualized evidence — the one-line justification for
the single-hop propagated anchor channel.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.trainer import _build_model, _require_graph_irt_checkpoint, _strip_module_prefix  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_glob", required=True)
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"{'dataset':<12} {'concepts':>9} {'|lambda2|':>10} {'|lambda2|^2':>12}")
    for run_dir in sorted(glob.glob(args.checkpoint_glob)):
        model_path = os.path.join(run_dir, "best_model.pth")
        if not os.path.isfile(model_path):
            continue
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        _require_graph_irt_checkpoint(checkpoint, model_path)
        model = _build_model(checkpoint["args"], checkpoint["info_dict"], device)
        model.load_state_dict(_strip_module_prefix(checkpoint["model_state_dict"]), strict=True)
        model.eval()
        with torch.no_grad():
            relations = model.relation_learning().mean(dim=0).numpy()
        eigenvalues = np.sort(np.abs(np.linalg.eigvals(relations)))[::-1]
        lam2 = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
        print(
            f"{checkpoint['args'].get('dataset_name', '?'):<12} {relations.shape[0]:>9} "
            f"{lam2:>10.4f} {lam2 ** 2:>12.4f}"
        )


if __name__ == "__main__":
    main()
