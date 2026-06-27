#!/usr/bin/env python3
"""Crisp redraws of the ASL model slides (flow, Conv1DBlock, causal padding)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")
os.makedirs(OUT, exist_ok=True)
C_IN = "#dbeafe"; C_MODEL = "#bfdbeafe" if False else "#c7ddf7"; C_NEUT = "#f1f5f9"
C_OUT = "#fde9c8"; C_HL = "#dcfce7"; EDGE = "#334155"; TXT = "#0f172a"


def box(ax, x, y, w, h, text, fc, fs=10, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                                linewidth=1.3, edgecolor=EDGE, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=TXT, weight=weight)


def arrow(ax, x1, y1, x2, y2, style="-|>", color=EDGE, lw=1.7, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=15,
                                 linewidth=lw, color=color, connectionstyle=f"arc3,rad={rad}"))


def save(fig, name):
    fig.tight_layout()
    p = os.path.join(OUT, name); fig.savefig(p, dpi=220, bbox_inches="tight"); plt.close(fig)
    print("wrote:", os.path.basename(p))


# ---------------------------------------------------- (A) overall ASL flow
fig, ax = plt.subplots(figsize=(5.2, 7)); ax.set_xlim(0, 6); ax.set_ylim(0, 9.5); ax.axis("off")
steps = [("Input\n(L, C) = (384, 708)", C_IN),
         ("Model  (1D-CNN + Transformer)\n(L, D) = (384, 384)", C_MODEL),
         ("Global Average Pooling\n(D,) = (384,)", C_NEUT),
         ("Output  (Dense + softmax)\n(num_classes,) = (250,)", C_OUT)]
y = 7.7
for t, c in steps:
    box(ax, 0.8, y, 4.4, 1.05, t, c, 10)
    if (t, c) != steps[-1]:
        arrow(ax, 3.0, y, 3.0, y - 0.55)
    y -= 1.6
arrow(ax, 3.0, 2.9, 3.0, 2.35)
ax.text(3.0, 2.0, "Categorical Cross-Entropy Loss", ha="center", fontsize=10.5,
        weight="bold", color=TXT)
ax.set_title("Figure — ASL model: input → output", fontsize=11, weight="bold", color=TXT, pad=8)
ax.text(3.0, 0.5, "training feature shapes; deployed TFLite takes raw (60, 543, 3)\n"
        "landmarks and preprocesses inside the graph",
        ha="center", fontsize=7.5, color="#64748b")
save(fig, "fig_asl_flow.png")

# ---------------------------------------------------- (B) Conv1DBlock detail
fig, ax = plt.subplots(figsize=(9.5, 6)); ax.set_xlim(0, 12); ax.set_ylim(0, 9); ax.axis("off")
# macro stack (left)
ax.text(2.4, 8.5, "192-d", ha="center", fontsize=10, color=TXT)
macro = [("Conv1DBlock", C_IN), ("Conv1DBlock", C_IN), ("Conv1DBlock", C_IN),
         ("TransformerBlock", C_OUT)]
y = 7.4
for t, c in macro:
    box(ax, 1.1, y, 2.6, 0.95, t, c, 10, "bold")
    if (t, c) != macro[-1]:
        arrow(ax, 2.4, y, 2.4, y - 0.35)
    y -= 1.3
# x2 bracket
ax.plot([0.7, 0.5, 0.5, 0.7], [7.4, 7.4, 3.55, 3.55], color=EDGE, lw=1.4)
ax.text(0.2, 5.5, "× 2", ha="center", va="center", fontsize=12, weight="bold", color=TXT, rotation=90)
# micro (right) — Conv1DBlock internals
ax.text(8.7, 8.5, "192-d", ha="center", fontsize=10, color=TXT)
micro = [("inputs", C_NEUT), ("Conv1D, ksize=1", C_NEUT),
         ("DWConv1D, ksize=17, BN", C_HL), ("ECA", C_HL), ("Conv1D, ksize=1", C_NEUT)]
labels_side = {2: "384-d"}
y = 7.6
ys = []
for i, (t, c) in enumerate(micro):
    box(ax, 6.7, y, 4.0, 0.8, t, c, 9.5)
    ys.append(y)
    if i in labels_side:
        ax.text(6.4, y + 0.4, labels_side[i], ha="right", fontsize=9, color="#475569")
    if i < len(micro) - 1:
        arrow(ax, 8.7, y, 8.7, y - 0.42)
    y -= 1.22
# residual skip from inputs -> after last conv
arrow(ax, 10.7, ys[0] + 0.4, 11.4, ys[0] + 0.4, style="-")
ax.add_patch(FancyArrowPatch((11.4, ys[0] + 0.4), (11.4, ys[-1] + 0.4), arrowstyle="-",
                             linewidth=1.5, color=EDGE))
arrow(ax, 11.4, ys[-1] + 0.4, 10.7, ys[-1] + 0.4)
ax.text(11.55, (ys[0] + ys[-1]) / 2 + 0.4, "residual", rotation=90, va="center",
        fontsize=8.5, color="#475569")
# connector macro -> micro
arrow(ax, 3.7, 6.3, 6.7, 7.2, style="-", lw=1.0)
ax.set_title("Figure — ASL Conv1DBlock (causal depthwise conv + ECA + residual)",
             fontsize=11, weight="bold", color=TXT, pad=8)
save(fig, "fig_asl_conv1dblock.png")

# ---------------------------------------------------- (C) causal vs same padding
fig, ax = plt.subplots(figsize=(10, 4.2)); ax.set_xlim(0, 14); ax.set_ylim(0, 5); ax.axis("off")


def timeline(ax, x0, y, cells, title):
    ax.text(x0 + 3.0, y + 1.15, title, ha="center", fontsize=10, weight="bold", color=TXT)
    for i, (lab, c) in enumerate(cells):
        box(ax, x0 + i * 1.0, y, 0.95, 0.7, lab, c, 8)


pad = "#fecdd3"; frm = "white"; cau = "#dbeafe"
same_in = [("pad", pad), ("t1", frm), ("t2", frm), ("t3", frm), ("PAD", pad), ("PAD", pad)]
same_out = [("", frm), ("", frm), ("PAD?", pad), ("PAD", pad), ("PAD", pad), ("", frm)]
timeline(ax, 0.5, 3.4, same_in, "padding = 'same'")
timeline(ax, 0.5, 1.4, same_out, "")
for i in range(6):
    arrow(ax, 0.5 + i + 0.47, 3.4, 0.5 + i + 0.47, 2.1, lw=1.0)
ax.text(3.5, 0.9, "output frame contaminated by neighbouring PAD", ha="center",
        fontsize=8, color="#9f1239")

caus_in = [("causal", cau), ("causal", cau), ("t1", frm), ("t2", frm), ("t3", frm), ("PAD", pad)]
caus_out = [("t1", frm), ("t2", frm), ("t3", frm), ("PAD", pad), ("PAD", pad), ("", frm)]
timeline(ax, 7.5, 3.4, caus_in, "padding = 'causal'")
timeline(ax, 7.5, 1.4, caus_out, "")
for i in range(5):
    arrow(ax, 7.5 + i + 0.47, 3.4, 7.5 + (i) + 0.47, 2.1, lw=1.0, rad=0.0)
ax.text(10.5, 0.9, "past-only context; time frames preserved", ha="center",
        fontsize=8, color="#065f46")
ax.set_title("Figure — Causal vs. 'same' padding in the Conv1D blocks",
             fontsize=11, weight="bold", color=TXT, pad=6)
save(fig, "fig_asl_causal_padding.png")

print("done")
