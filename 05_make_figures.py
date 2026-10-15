"""Step 5: create all figures used in the paper (results/figures/*.png)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = "results"
F = os.path.join(R, "figures")
os.makedirs(F, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})
ORDER = ["YAKE", "TF-IDF", "TextRank", "Supervised-LR", "Supervised-GBM (proposed)"]
COL = {"YAKE": "#9aa5b1", "TF-IDF": "#7fb3d5", "TextRank": "#5d9cc9", "Supervised-LR": "#f0b27a",
       "Supervised-GBM (proposed)": "#d35400"}


def fig_f1_bars():
    df = pd.read_csv(os.path.join(R, "keyphrase_results.csv"), index_col=0).loc[ORDER]
    ks = [5, 10, 15]
    w = 0.16
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for i, name in enumerate(ORDER):
        vals = [df.loc[name, f"F1@{k}"] for k in ks]
        bars = ax.bar(np.arange(3) + (i - 2) * w, vals, w, label=name, color=COL[name])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"F1@{k}" for k in ks])
    ax.set_ylabel("F1 (macro, exact stem match)")
    ax.set_ylim(0, 0.5)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.set_title("Keyphrase extraction on Inspec test (500 abstracts)")
    fig.tight_layout()
    fig.savefig(os.path.join(F, "fig1_f1_comparison.png"))
    plt.close(fig)


def fig_pr_curve():
    c = pd.read_csv(os.path.join(R, "pr_curve.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for name in ORDER:
        s = c[c.system == name]
        axes[0].plot(s.R, s.P, marker="o", ms=3, label=name, color=COL[name])
        axes[1].plot(s.k, s.F1, marker="o", ms=3, label=name, color=COL[name])
    axes[0].set_xlabel("Recall@k")
    axes[0].set_ylabel("Precision@k")
    axes[0].set_title("Precision-recall as k goes 1 to 15")
    axes[1].set_xlabel("k (number of keyphrases returned)")
    axes[1].set_ylabel("F1@k")
    axes[1].set_title("F1 versus k")
    axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(F, "fig2_pr_curve.png"))
    plt.close(fig)


def fig_ablation():
    ab = pd.read_csv(os.path.join(R, "ablation.csv"))
    ab = ab[~ab.setting.isin(["no redundancy filter"])]
    drop = ab[ab.setting.str.startswith("-") | (ab.setting == "all features")]
    only = ab[ab.setting.str.startswith("only")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    axes[0].barh(drop.setting[::-1], drop["F1@10"][::-1], color="#d35400")
    axes[0].set_title("Leave one feature group out")
    axes[1].barh(only.setting[::-1], only["F1@10"][::-1], color="#5d9cc9")
    axes[1].set_title("Single feature group only")
    for ax in axes:
        ax.set_xlabel("F1@10")
        ax.set_xlim(0.15, 0.45)
    fig.tight_layout()
    fig.savefig(os.path.join(F, "fig3_ablation.png"))
    plt.close(fig)


def fig_error():
    e = json.load(open(os.path.join(R, "error_analysis.json")))
    fn = e["false_negative_breakdown"]
    fp = e["false_positive_breakdown"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    labs = ["found in top-10", "candidate generated but ranked below 10",
            "in text but not generated as candidate (POS/length/boundary)", "absent from the abstract text"]
    short = ["Found in top-10", "Ranked below 10", "Not a candidate", "Absent from text"]
    axes[0].barh(short[::-1], [fn[l]["pct"] for l in labs][::-1], color=["#c0392b", "#e59866", "#f5cba7", "#27ae60"])
    axes[0].set_xlabel("% of gold keyphrases")
    axes[0].set_title("Fate of gold keyphrases")
    fl = ["sub-phrase of a gold keyphrase", "partial word overlap with gold", "super-phrase of a gold keyphrase",
          "generic frequent single word", "other (no lexical relation to gold)"]
    fs = ["Sub-phrase of gold", "Partial overlap", "Super-phrase of gold", "Generic word", "Unrelated"]
    axes[1].barh(fs[::-1], [fp[l]["pct"] for l in fl][::-1], color="#5d9cc9")
    axes[1].set_xlabel("% of false positives")
    axes[1].set_title("Types of false positives (top-10)")
    fig.tight_layout()
    fig.savefig(os.path.join(F, "fig6_error_breakdown.png"))
    plt.close(fig)


def fig_topics():
    sw = pd.read_csv(os.path.join(R, "topic_k_sweep.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    cols = {"fulltext": "#555555", "TextRank-10": "#5d9cc9", "Proposed-10": "#d35400", "Gold-10": "#27ae60",
            "Random-10": "#bbbbbb"}
    for rep, c in cols.items():
        s = sw[sw.representation == rep]
        axes[0].plot(s.k, s.NMI, marker="o", ms=3, label=rep, color=c)
        axes[1].plot(s.k, s.NPMI, marker="o", ms=3, label=rep, color=c)
    axes[0].set_xlabel("number of clusters k")
    axes[0].set_ylabel("NMI with thesaurus reference")
    axes[0].set_title("Agreement with reference topics")
    axes[1].set_xlabel("number of clusters k")
    axes[1].set_ylabel("NPMI coherence")
    axes[1].set_title("Topic coherence")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(F, "fig5_topic_k_sweep.png"))
    plt.close(fig)


def fig_importance():
    imp = pd.read_csv(os.path.join(R, "feature_importance.csv")).head(12)[::-1]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.barh(imp.feature, imp.importance, color="#d35400")
    ax.set_xlabel("Permutation importance (drop in average precision)")
    ax.set_title("Top features of the GBM ranker")
    fig.tight_layout()
    fig.savefig(os.path.join(F, "fig4_feature_importance.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig_f1_bars()
    fig_pr_curve()
    fig_ablation()
    fig_error()
    fig_topics()
    fig_importance()
    print("Figures written to", F)
