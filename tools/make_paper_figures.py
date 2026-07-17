#!/usr/bin/env python
"""Generate all main paper figures from the final result tables.

Reads results/evidence_gain_curve.csv and results/anchor_contribution.csv;
other tables (ablation, leakage, noise) are embedded from the sealed test
numbers recorded in docs/final_results_v2_20260717.md. Writes PDF + PNG to
docs/paper_figures/.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "paper_figures")
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUT, exist_ok=True)

# Colorblind-safe palette (Okabe-Ito).
C = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "grey": "#999999",
}
plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

DATASETS = ["ASSIST17", "Junyi", "NIPS34", "EdNet", "MOOCRadar", "XES3G5M"]
KEYS = ["assist_17", "junyi", "nips34", "ednet_kt1", "moocradar", "xes3g5m"]


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print(f"wrote {name}.pdf / .png")


def fig_gain_curve():
    df = pd.read_csv(os.path.join(RESULTS, "evidence_gain_curve.csv"))
    pooled = df[df["dataset"] == "POOLED"].copy()
    labels = ["0", "1-2", "3-5", "6-11", "12+"]
    gain = pooled["gain"].to_numpy()[: len(labels)]
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.axhline(0, color=C["grey"], lw=0.8, ls="--")
    ax.plot(range(len(labels)), gain, "-o", color=C["blue"], lw=2, ms=6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Same-concept training observations")
    ax.set_ylabel("AUC gain from response evidence")
    ax.set_title("Pooled over six datasets", fontsize=10)
    for i, g in enumerate(gain):
        ax.annotate(f"{g:+.4f}", (i, g), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    save(fig, "fig_evidence_gain_curve")


def fig_anchor_contribution():
    df = pd.read_csv(os.path.join(RESULTS, "anchor_contribution.csv")).set_index("dataset")
    chans = ["direct", "residual", "prop_h1", "prop_h2"]
    labels = ["direct rate", "difficulty residual", "propagated (head 1)", "propagated (head 2)"]
    colors = [C["blue"], C["orange"], C["green"], C["vermillion"]]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    bottom = np.zeros(len(KEYS))
    for chan, lab, col in zip(chans, labels, colors):
        vals = np.array([df.loc[k, chan] if k in df.index else 0.0 for k in KEYS])
        ax.bar(DATASETS, vals, bottom=bottom, label=lab, color=col)
        bottom += vals
    ax.set_ylabel("Mean |Δθ| contribution")
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper right")
    ax.tick_params(axis="x", rotation=20)
    save(fig, "fig_anchor_contribution")


def fig_ablation():
    # test AUC: full, w/o A (evidence anchoring), w/o B (graph calibration)
    full = [0.7891, 0.8305, 0.7902, 0.7487, 0.9345, 0.8009]
    woA = [0.7868, 0.8304, 0.7877, 0.7480, 0.9335, 0.7969]
    woB = [0.7892, 0.8300, 0.7898, 0.7460, 0.9308, 0.7997]
    dA = np.array(full) - np.array(woA)
    dB = np.array(full) - np.array(woB)
    x = np.arange(len(DATASETS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.axhline(0, color=C["grey"], lw=0.8)
    ax.bar(x - w / 2, dA, w, label="remove evidence anchoring", color=C["blue"])
    ax.bar(x + w / 2, dB, w, label="remove graph calibration", color=C["orange"])
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, rotation=20)
    ax.set_ylabel("AUC drop when module removed")
    ax.legend(fontsize=9, frameon=False)
    save(fig, "fig_ablation")


def fig_leakage():
    # zero-parameter direct-evidence AUC under honest LOO vs corpus-leak vs self-leak
    honest = [0.6688, 0.7891, 0.7228, 0.6918, 0.8302, 0.6472]
    corpus = [0.8324, 1.0000, 0.7727, 0.7671, 0.9863, 0.9737]
    selfl = [0.8380, 1.0000, 0.7757, 0.7713, 0.9875, 0.9765]
    x = np.arange(len(DATASETS))
    w = 0.27
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(x - w, honest, w, label="leave-one-out (ours)", color=C["green"])
    ax.bar(x, selfl, w, label="self-leak", color=C["vermillion"])
    ax.bar(x + w, corpus, w, label="corpus-leak", color=C["grey"])
    ax.axhline(1.0, color=C["grey"], lw=0.6, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, rotation=20)
    ax.set_ylabel("Zero-parameter statistic AUC")
    ax.set_ylim(0.5, 1.03)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    save(fig, "fig_leakage")


def fig_noise():
    eta = [0, 5, 10, 20]
    full = {
        "ASSIST17": [0.7891, 0.7861, 0.7826, 0.7714],
        "NIPS34": [0.7902, 0.7867, 0.7836, 0.7770],
        "XES3G5M": [0.8009, 0.7914, 0.7831, 0.7611],
        "MOOCRadar": [0.9345, 0.9241, 0.9194, 0.9037],
    }
    woA = {
        "ASSIST17": [0.7868, 0.7842, 0.7812, 0.7706],
        "NIPS34": [0.7877, 0.7844, 0.7812, 0.7751],
        "XES3G5M": [0.7969, 0.7890, 0.7813, 0.7598],
        "MOOCRadar": [0.9335, 0.9241, 0.9192, 0.9036],
    }
    cols = {"ASSIST17": C["blue"], "NIPS34": C["orange"], "XES3G5M": C["green"], "MOOCRadar": C["purple"]}
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for ds in full:
        gap = np.array(full[ds]) - np.array(woA[ds])
        ax.plot(eta, gap, "-o", color=cols[ds], lw=1.8, ms=5, label=ds)
    ax.axhline(0, color=C["grey"], lw=0.8, ls="--")
    ax.set_xlabel("Train label noise (%)")
    ax.set_ylabel("Evidence AUC advantage")
    ax.legend(fontsize=8, frameon=False)
    save(fig, "fig_noise_degradation")


if __name__ == "__main__":
    fig_gain_curve()
    fig_anchor_contribution()
    fig_ablation()
    fig_leakage()
    fig_noise()
    print("all figures written to docs/paper_figures/")
