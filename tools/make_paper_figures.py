#!/usr/bin/env python
"""Generate all main paper figures from the final result tables.

Inputs: results/evidence_gain_curve_v3.csv (bootstrap CIs incl. pooled). The
leakage-probe numbers are embedded from the sealed runs recorded in
docs/analysis_propositions_20260716.md. Writes PDF + PNG to
docs/paper_figures/.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "paper_figures")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)

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
BUCKETS = ["0-1", "1-3", "3-6", "6-12", ">=12"]
GAIN_CSV = os.path.join(RESULTS, "evidence_gain_curve_v3.csv")
if not os.path.exists(GAIN_CSV):
    GAIN_CSV = os.path.join(RESULTS, "evidence_gain_curve_v2.csv")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print(f"wrote {name}.pdf / .png")


def fig_motivation():
    df = pd.read_csv(GAIN_CSV)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.5))

    shades = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c"]
    edges = [0.0, 1.0, 3.0, 6.0, 12.0]
    shares = np.zeros((len(KEYS), len(edges)))
    for i, key in enumerate(KEYS):
        sub = df[df["dataset"] == key]
        total = sub["rows"].sum()
        for j, low in enumerate(edges):
            row = sub[sub["bucket_low"] == low]
            if len(row):
                shares[i, j] = row["rows"].iloc[0] / total
    left = np.zeros(len(KEYS))
    y = np.arange(len(KEYS))[::-1]
    for j, lab in enumerate(BUCKETS):
        ax1.barh(y, shares[:, j], left=left, color=shades[j], label=lab, height=0.62)
        left += shares[:, j]
    ax1.set_yticks(y)
    ax1.set_yticklabels(DATASETS)
    ax1.set_xlim(0, 1)
    ax1.set_xlabel("share of test responses")
    ax1.legend(fontsize=8, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.16), title="same-concept train responses", title_fontsize=8)
    ax1.set_title("(a) evidence coverage", fontsize=11, pad=22)

    honest = [0.6688, 0.7891, 0.7228, 0.6918, 0.8302, 0.6472]
    selfl = [0.8380, 1.0000, 0.7757, 0.7713, 0.9875, 0.9765]
    corpus = [0.8324, 1.0000, 0.7727, 0.7671, 0.9863, 0.9737]
    x = np.arange(len(DATASETS))
    w = 0.27
    ax2.bar(x - w, honest, w, label="leave-one-out", color=C["green"])
    ax2.bar(x, selfl, w, label="self-leak", color=C["vermillion"])
    ax2.bar(x + w, corpus, w, label="corpus-leak", color=C["grey"])
    ax2.axhline(1.0, color=C["grey"], lw=0.8, ls=":")
    ax2.annotate("1.0000", xy=(1, 1.0), xytext=(1.3, 1.03), fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(DATASETS, rotation=20)
    ax2.set_ylim(0.5, 1.08)
    ax2.set_ylabel("AUC of the plain statistic")
    ax2.legend(fontsize=8, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax2.set_title("(b) leakage inflation", fontsize=11, pad=22)
    save(fig, "fig_motivation")


def _box(ax, x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec="#444444", lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.2)


def _arrow(ax, x0, y0, x1, y1, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=11, lw=1.1, color="#444444"))


def fig_framework():
    fig, ax = plt.subplots(figsize=(10.8, 3.6))
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    _box(ax, 0.15, 1.30, 1.55, 1.00, "train\nresponse logs", "#f0f0f0")
    _box(ax, 2.15, 2.30, 2.30, 0.95, "label-free graph priors\nitem co-occurrence\nstudent co-exposure", "#dbeaf7")
    _box(ax, 2.15, 0.30, 2.30, 0.95, "leave-one-out evidence\nrate gap / residual / count\n(current label subtracted)", "#dff0e8")
    _box(ax, 5.00, 2.30, 2.10, 0.95, "relation learning\nrow-stochastic\nconcept graphs", "#dbeaf7")
    _box(ax, 5.00, 1.28, 2.10, 0.72, "GNN state encoder", "#dbeaf7")
    _box(ax, 5.00, 0.30, 2.10, 0.72, "evidence anchor\ncount-gated channels", "#dff0e8")
    _box(ax, 7.65, 0.95, 1.55, 1.35, "Q-masked\nability read-out\n$\\theta_e$", "#f6e8d5")
    _box(ax, 9.55, 1.10, 1.10, 1.05, "2PL\n$a(\\theta_e-b)$", "#f6e8d5")
    _arrow(ax, 1.70, 2.05, 2.15, 2.55)
    _arrow(ax, 1.70, 1.55, 2.15, 0.95)
    _arrow(ax, 4.45, 2.77, 5.00, 2.77)
    _arrow(ax, 6.05, 2.30, 6.05, 2.00)
    _arrow(ax, 4.45, 0.77, 5.00, 0.66)
    _arrow(ax, 4.45, 1.05, 5.00, 1.55)
    _arrow(ax, 6.05, 2.30, 6.05, 2.02)
    _arrow(ax, 7.10, 1.64, 7.65, 1.64)
    _arrow(ax, 7.10, 0.66, 7.65, 1.25)
    _arrow(ax, 9.20, 1.62, 9.55, 1.62)
    ax.text(5.35, 2.18, "propagated channel", fontsize=7.8, color="#444444")
    save(fig, "fig_framework")


def fig_gain_curve():
    df = pd.read_csv(GAIN_CSV)
    pooled = df[df["dataset"] == "POOLED_COMPLETE"].sort_values("bucket_low")
    gain = pooled["gain"].to_numpy() * 1e3
    lo = pooled["gain_ci_low"].to_numpy(dtype=float) * 1e3
    hi = pooled["gain_ci_high"].to_numpy(dtype=float) * 1e3
    x = np.arange(len(gain))
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    if np.isfinite(lo).all():
        ax.fill_between(x, lo, hi, color=C["blue"], alpha=0.16, lw=0, label="95% bootstrap CI")
    ax.plot(x, gain, color=C["blue"], lw=2.4, marker="o", ms=6,
            mec="white", mew=1.2, label="pooled gain (complete coverage)")
    ax.axhline(0.0, color=C["grey"], lw=0.9, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(BUCKETS[: len(gain)])
    ax.set_xlabel("same-concept train responses of the target concept")
    ax.set_ylabel(r"AUC gain of the anchor ($\times 10^{-3}$)")
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    save(fig, "fig_gain_curve")


def fig_leakage_fan():
    df = pd.read_csv(os.path.join(RESULTS, "leakage_fan.csv"))
    rng = np.random.RandomState(11)
    x = df["n"].to_numpy() + rng.uniform(-0.28, 0.28, size=len(df))
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for label, color, name in ((1, C["blue"], "correct (y=1)"), (0, C["vermillion"], "incorrect (y=0)")):
        mask = df["label"].to_numpy() == label
        ax.scatter(x[mask], df["delta"].to_numpy()[mask], s=5, alpha=0.3,
                   color=color, linewidths=0, label=name, rasterized=True)
    n_cont = np.linspace(0, 12.4, 200)
    env = (n_cont + 1) / ((n_cont + 2) * (n_cont + 3))
    ax.plot(n_cont, env, color="black", ls="--", lw=1.3, label="theory envelope")
    ax.plot(n_cont, -env, color="black", ls="--", lw=1.3)
    ax.axhline(0.0, color=C["grey"], lw=0.8, ls=":")
    ax.set_xlim(-0.6, 12.6)
    ax.set_xlabel("same-concept train responses $n$")
    ax.set_ylabel("leakage shift of the statistic")
    leg = ax.legend(fontsize=9, frameon=False, loc="upper right")
    for handle in leg.legend_handles:
        if hasattr(handle, "set_alpha"):
            handle.set_alpha(1.0)
        if hasattr(handle, "set_sizes"):
            handle.set_sizes([28])
    save(fig, "fig_leakage_fan")


def fig_gate_reliability():
    df = pd.read_csv(os.path.join(RESULTS, "gate_reliability.csv"))
    direct = df[df["channel"] == 0].set_index("dataset")
    n = np.linspace(0, 30, 300)
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    colors = [C["blue"], C["orange"], C["green"], C["vermillion"], C["purple"], C["sky"]]
    for (key, name), color in zip(zip(KEYS, DATASETS), colors):
        if key not in direct.index or key == "junyi":
            continue
        a, b = direct.loc[key, "a"], direct.loc[key, "b"]
        gate = 1.0 / (1.0 + np.exp(-(a + b * np.log1p(n))))
        ax.plot(n, gate, color=color, lw=1.8, label=name)
    ax.plot(n, n / (n + 2), color="black", ls="--", lw=1.6,
            label="statistic data weight $n/(n+2)$")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("same-concept train responses $n$")
    ax.set_ylabel("gate value / data weight")
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="lower right")
    save(fig, "fig_gate_reliability")


def fig_neighbor_decay():
    df = pd.read_csv(os.path.join(RESULTS, "neighbor_information.csv"))
    base = df[df["tier"] == "concept_mean"].set_index("dataset")["auc_all"]
    tiers = ["1hop", "2hop", "random"]
    labels = ["1-hop", "2-hop", "random"]
    x = np.arange(len(tiers))
    colors = [C["blue"], C["orange"], C["green"], C["vermillion"], C["purple"], C["sky"]]
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    for (key, name), color in zip(zip(KEYS, DATASETS), colors):
        sub = df[df["dataset"] == key].set_index("tier")
        delta = [sub.loc[t, "auc_all"] - base[key] for t in tiers]
        ax.plot(x, delta, color=color, lw=1.8, marker="o", ms=5,
                mec="white", mew=0.8, label=name)
    ax.axhline(0.0, color=C["grey"], lw=0.9, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("source of the aggregated neighbor statistics")
    ax.set_ylabel("AUC gain over concept-mean baseline")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    save(fig, "fig_neighbor_decay")


if __name__ == "__main__":
    fig_motivation()
    fig_framework()
    fig_gain_curve()
    fig_leakage_fan()
    fig_neighbor_decay()
