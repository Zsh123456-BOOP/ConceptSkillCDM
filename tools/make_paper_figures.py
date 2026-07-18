#!/usr/bin/env python
"""Generate all main paper figures from the final result tables.

Inputs: results/evidence_gain_curve_v3.csv (bootstrap CIs incl. pooled),
results/anchor_contribution_v2.csv (usage + causal drops),
results/multiseed_auc_summary.csv (6-seed mean/std). The leakage-probe
numbers are embedded from the sealed runs recorded in
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


def fig_channel_scatter():
    df = pd.read_csv(os.path.join(RESULTS, "anchor_contribution_v2.csv"))
    order = {k: i for i, k in enumerate(KEYS)}
    df = df.sort_values(by="dataset", key=lambda s: s.map(order))
    chans = [("direct", "drop_direct", C["blue"]),
             ("residual", "drop_residual", C["green"]),
             ("prop_h1", "drop_prop_h1", C["orange"]),
             ("prop_h2", "drop_prop_h2", C["purple"])]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for use_col, drop_col, color in chans:
        ax.scatter(df[use_col], df[drop_col] * 1e3, s=52, color=color,
                   edgecolors="white", linewidths=1.0, label=use_col.replace("_", "-"), zorder=3)
    ax.axhline(0.0, color=C["grey"], lw=0.9, ls=":")
    names = dict(zip(KEYS, DATASETS))
    for _, r in df.iterrows():
        if r["drop_direct"] * 1e3 > 1.0:
            ax.annotate(names[r["dataset"]], (r["direct"], r["drop_direct"] * 1e3),
                        xytext=(4, 4), textcoords="offset points", fontsize=8)
    jy = df[df["dataset"] == "junyi"].iloc[0]
    ax.annotate("Junyi: propagated only", (jy["prop_h1"], jy["drop_prop_h1"] * 1e3),
                xytext=(8, -14), textcoords="offset points", fontsize=8,
                arrowprops=dict(arrowstyle="-", lw=0.8, color="#444444"))
    ax.set_xlabel(r"channel usage  $|\Delta\theta|$")
    ax.set_ylabel(r"AUC drop when zeroed ($\times 10^{-3}$)")
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    save(fig, "fig_channel_scatter")


def fig_ablation():
    df = pd.read_csv(os.path.join(RESULTS, "multiseed_auc_summary.csv"))
    piv_m = df.pivot(index="dataset", columns="variant", values="auc_mean").loc[KEYS]
    piv_s = df.pivot(index="dataset", columns="variant", values="auc_std").loc[KEYS]
    dA = (piv_m["full"] - piv_m["woA"]).to_numpy() * 1e3
    dB = (piv_m["full"] - piv_m["woB"]).to_numpy() * 1e3
    eA = np.sqrt(piv_s["full"] ** 2 + piv_s["woA"] ** 2).to_numpy() * 1e3
    eB = np.sqrt(piv_s["full"] ** 2 + piv_s["woB"] ** 2).to_numpy() * 1e3
    x = np.arange(len(KEYS))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.bar(x - w / 2, dA, w, yerr=eA, capsize=3, label="w/o LEA",
           color=C["blue"], error_kw={"lw": 1.0})
    ax.bar(x + w / 2, dB, w, yerr=eB, capsize=3, label="w/o GEC",
           color=C["orange"], error_kw={"lw": 1.0})
    ax.axhline(0.0, color=C["grey"], lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, rotation=20)
    ax.set_ylabel(r"AUC drop when removed ($\times 10^{-3}$)")
    ax.legend(fontsize=9, frameon=False)
    save(fig, "fig_ablation")


if __name__ == "__main__":
    fig_motivation()
    fig_framework()
    fig_gain_curve()
    fig_channel_scatter()
    fig_ablation()
