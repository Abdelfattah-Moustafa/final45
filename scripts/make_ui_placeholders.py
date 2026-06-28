#!/usr/bin/env python3
"""Generate UI screenshot placeholders (2 web, 2 phone, 1 overview)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")
os.makedirs(OUT, exist_ok=True)


def placeholder(name, w, h, text, device=""):
    fig, ax = plt.subplots(figsize=(w, h)); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(plt.Rectangle((0.03, 0.03), 0.94, 0.94, fill=True, facecolor="#f8fafc",
                 edgecolor="#94a3b8", lw=1.8, ls="--"))
    # a faux browser/phone chrome bar
    ax.add_patch(plt.Rectangle((0.03, 0.88), 0.94, 0.09, fill=True, facecolor="#e2e8f0",
                 edgecolor="#94a3b8", lw=1.0))
    if device == "phone":
        ax.add_patch(plt.Circle((0.5, 0.925), 0.012, color="#94a3b8"))
    else:
        for i, cx in enumerate((0.07, 0.10, 0.13)):
            ax.add_patch(plt.Circle((cx, 0.925), 0.008, color=["#ef4444", "#f59e0b", "#22c55e"][i]))
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=12.5, color="#475569",
            style="italic", wrap=True)
    ax.text(0.5, 0.10, "[ screenshot placeholder — insert real capture ]", ha="center",
            va="center", fontsize=8.5, color="#94a3b8")
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight"); plt.close(fig)
    print("wrote:", name)


placeholder("ui_web_1.png", 8, 5, "Web UI — Sign → Text / Speech dashboard", "web")
placeholder("ui_web_2.png", 8, 5, "Web UI — Live two-person meeting", "web")
placeholder("ui_phone_1.png", 4, 8, "Mobile UI — Sign → Text\n(portrait)", "phone")
placeholder("ui_phone_2.png", 4, 8, "Mobile UI — Text → Sign avatar\n(portrait)", "phone")
placeholder("ui_overview.png", 10, 5, "UI Overview — navigation, modules,\nand bilingual (EN/AR) layout", "web")
print("done")
