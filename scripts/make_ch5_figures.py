#!/usr/bin/env python3
"""Chapter 5 result figures (recognition + translation quality)."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")
os.makedirs(OUT, exist_ok=True)
TXT = "#0f172a"
C_LLM = "#2563eb"; C_BASE = "#cbd5e1"; C_ASL = "#16a34a"; C_AR = "#9333ea"


def finish(fig, ax, t, name):
    ax.set_title(t, fontsize=12, weight="bold", color=TXT, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight")
    plt.close(fig); print("wrote:", name)


# ---- Fig 5.1  chrF: LLM vs raw gloss ----------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.4))
langs = ["ASL (English)", "ArSL (Arabic)"]
llm = [0.9115, 0.5929]; base = [0.5434, 0.3442]
x = np.arange(2); w = 0.36
b1 = ax.bar(x - w / 2, llm, w, label="With LLM (Gemini)", color=C_LLM)
b2 = ax.bar(x + w / 2, base, w, label="No LLM (raw gloss)", color=C_BASE)
ax.bar_label(b1, fmt="%.2f", fontsize=9); ax.bar_label(b2, fmt="%.2f", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(langs); ax.set_ylabel("chrF"); ax.set_ylim(0, 1.0)
ax.legend(fontsize=9)
finish(fig, ax, "Figure 5.1  —  Translation quality (chrF): LLM vs. raw gloss", "fig5_01_chrf.png")


# ---- Fig 5.2  chrF precision vs recall --------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.4))
groups = ["ASL\nPrecision", "ASL\nRecall", "ArSL\nPrecision", "ArSL\nRecall"]
llm = [0.9259, 0.9100, 0.6414, 0.5868]; base = [0.7052, 0.5196, 0.5368, 0.3166]
x = np.arange(4); w = 0.36
b1 = ax.bar(x - w / 2, llm, w, label="With LLM", color=C_LLM)
b2 = ax.bar(x + w / 2, base, w, label="No LLM", color=C_BASE)
ax.bar_label(b1, fmt="%.2f", fontsize=8); ax.bar_label(b2, fmt="%.2f", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylabel("chrF component"); ax.set_ylim(0, 1.0)
ax.legend(fontsize=9)
finish(fig, ax, "Figure 5.2  —  chrF precision vs. recall (the LLM supplies grammar = recall)", "fig5_02_precision_recall.png")


# ---- Fig 5.3  BLEU-n profiles -----------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.4))
n = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]
asl_llm = [0.9453, 0.9053, 0.8553, 0.8048]
ar_llm = [0.5270, 0.4436, 0.4235, 0.4194]
ar_base = [0.2114, 0.0610, 0.0236, 0.0261]
ax.plot(n, asl_llm, "-o", color=C_ASL, lw=2, label="ASL — with LLM")
ax.plot(n, ar_llm, "-o", color=C_AR, lw=2, label="ArSL — with LLM")
ax.plot(n, ar_base, "--o", color="#94a3b8", lw=1.8, label="ArSL — raw gloss")
for xi, v in enumerate(asl_llm): ax.text(xi, v + 0.02, f"{v:.2f}", fontsize=8, ha="center", color=C_ASL)
for xi, v in enumerate(ar_llm): ax.text(xi, v + 0.02, f"{v:.2f}", fontsize=8, ha="center", color=C_AR)
ax.set_ylabel("Cumulative BLEU"); ax.set_ylim(0, 1.0); ax.legend(fontsize=9)
finish(fig, ax, "Figure 5.3  —  Cumulative BLEU-n profiles", "fig5_03_bleu.png")


# ---- Fig 5.4  Recognition accuracy (in-dist vs generalization) --------------
fig, ax = plt.subplots(figsize=(7.5, 4.4))
groups = ["ASL (250-class)", "ArSL (20-class)"]
indist = [0.80, 0.9941]; general = [0.624, 0.88]
x = np.arange(2); w = 0.36
b1 = ax.bar(x - w / 2, indist, w, label="In-distribution test", color=C_ASL)
b2 = ax.bar(x + w / 2, general, w, label="Generalization test", color="#93c5fd")
ax.bar_label(b1, labels=[f"{a*100:.1f}%" for a in indist], fontsize=10, weight="bold")
ax.bar_label(b2, labels=[f"{a*100:.1f}%" for a in general], fontsize=10, weight="bold")
ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.08)
ax.legend(fontsize=9, loc="lower center")
ax.text(0.0, 0.30, "WLASL\ncross-dataset\n(Top-1)", ha="center", fontsize=7.5, color="#475569")
ax.text(1.0, 0.30, "signer-\nindependent", ha="center", fontsize=7.5, color="#475569")
finish(fig, ax, "Figure 5.4  —  Recognition accuracy: in-distribution vs. generalization", "fig5_04_recognition.png")


# ---- Fig 5.5 / 5.6  Confusion-matrix placeholders ---------------------------
def placeholder(name, text):
    fig, ax = plt.subplots(figsize=(6, 5)); ax.axis("off")
    ax.add_patch(plt.Rectangle((0.04, 0.04), 0.92, 0.92, fill=True, facecolor="#f8fafc",
                 edgecolor="#94a3b8", lw=1.6, ls="--"))
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=12, color="#64748b", style="italic")
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight"); plt.close(fig); print("wrote:", name)

placeholder("fig5_05_arsl_confusion.png",
            "[ Placeholder ]\n\nArSL 20×20 confusion matrix\n(generate from test predictions)")
placeholder("fig5_06_asl_confusion.png",
            "[ Placeholder ]\n\nASL confusion matrix / top-confused\nsign pairs (generate from predictions)")

print("done")
