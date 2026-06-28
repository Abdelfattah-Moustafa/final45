"""Shared drawing helpers for thesis diagrams (matplotlib, no external deps)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Circle, Polygon, Rectangle

# palette
BLUE = "#dbeafe"; AMBER = "#fef3c7"; GREEN = "#dcfce7"; PURPLE = "#fae8ff"
GREY = "#e2e8f0"; ROSE = "#ffe4e6"; NEUT = "#f1f5f9"; WHITE = "#ffffff"
EDGE = "#334155"; TXT = "#0f172a"; MUTE = "#64748b"


def box(ax, x, y, w, h, text, fc=NEUT, fs=10, weight="normal", round=True, ec=EDGE):
    style = "round,pad=0.02,rounding_size=0.06" if round else "square,pad=0.0"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, linewidth=1.3,
                                edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=TXT, weight=weight, zorder=5)


def arrow(ax, x1, y1, x2, y2, style="-|>", color=EDGE, lw=1.6, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13,
                                 linewidth=lw, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=4))


def label_arrow(ax, x1, y1, x2, y2, text, **kw):
    arrow(ax, x1, y1, x2, y2, **kw)
    ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, text, ha="center", fontsize=8, color=MUTE)


def diamond(ax, cx, cy, w, h, text, fc=AMBER, fs=8.5):
    ax.add_patch(Polygon([(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2),
                          (cx - w / 2, cy)], closed=True, facecolor=fc, edgecolor=EDGE, lw=1.3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=TXT)


def ellipse(ax, cx, cy, w, h, text, fc=AMBER, fs=9):
    ax.add_patch(Ellipse((cx, cy), w, h, facecolor=fc, edgecolor=EDGE, lw=1.2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=TXT)


def actor(ax, x, y, label, s=1.0):
    ax.add_patch(Circle((x, y + 0.5 * s), 0.16 * s, fill=False, lw=1.6, edgecolor=EDGE))
    ax.plot([x, x], [y + 0.34 * s, y - 0.15 * s], color=EDGE, lw=1.6)
    ax.plot([x - 0.22 * s, x + 0.22 * s], [y + 0.18 * s, y + 0.18 * s], color=EDGE, lw=1.6)
    ax.plot([x, x - 0.18 * s], [y - 0.15 * s, y - 0.5 * s], color=EDGE, lw=1.6)
    ax.plot([x, x + 0.18 * s], [y - 0.15 * s, y - 0.5 * s], color=EDGE, lw=1.6)
    ax.text(x, y - 0.75 * s, label, ha="center", va="center", fontsize=9.5, weight="bold", color=TXT)


def classbox(ax, x, y, w, name, attrs, methods, fc=BLUE, fs=8.5):
    n_a, n_m = len(attrs), len(methods)
    hdr, row = 0.5, 0.34
    h = hdr + max(1, n_a) * row + max(1, n_m) * row + 0.1
    ax.add_patch(Rectangle((x, y - h), w, h, facecolor=WHITE, edgecolor=EDGE, lw=1.3, zorder=3))
    ax.add_patch(Rectangle((x, y - hdr), w, hdr, facecolor=fc, edgecolor=EDGE, lw=1.3, zorder=3))
    ax.text(x + w / 2, y - hdr / 2, name, ha="center", va="center", fontsize=fs + 1.0,
            weight="bold", color=TXT, zorder=5)
    yy = y - hdr - row / 2
    for a in attrs:
        ax.text(x + 0.12, yy, a, ha="left", va="center", fontsize=fs, color=TXT, zorder=5)
        yy -= row
    sep = y - hdr - max(1, n_a) * row
    ax.plot([x, x + w], [sep, sep], color=EDGE, lw=1.0, zorder=4)
    yy = sep - row / 2
    for m in methods:
        ax.text(x + 0.12, yy, m, ha="left", va="center", fontsize=fs, color=TXT, zorder=5)
        yy -= row
    return (x, y, w, h)


def datastore(ax, x, y, w, h, text, fc=GREY, fs=9):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=EDGE, lw=1.3))
    ax.plot([x, x + w], [y + h - 0.01, y + h - 0.01], color=EDGE, lw=1.0)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=TXT)


def lifeline(ax, x, ytop, ybot, label, fc=BLUE):
    box(ax, x - 1.1, ytop, 2.2, 0.7, label, fc, 9, "bold")
    ax.plot([x, x], [ybot, ytop], color=MUTE, lw=1.0, ls="--", zorder=1)


def msg(ax, x1, x2, y, text, dashed=False, fs=8):
    ddir = 1 if x2 > x1 else -1
    arrow(ax, x1 + 0.05 * ddir, y, x2 - 0.05 * ddir, y, lw=1.4, ls="--" if dashed else "-")
    ax.text((x1 + x2) / 2, y + 0.12, text, ha="center", fontsize=fs, color=TXT)


def start_end(ax, cx, cy, text, fc=EDGE):
    ax.add_patch(Ellipse((cx, cy), 1.4, 0.55, facecolor=fc, edgecolor=EDGE, lw=1.3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=8.5, color="white", weight="bold")


def title(ax, t):
    ax.set_title(t, fontsize=11.5, weight="bold", color=TXT, pad=10)


def newfig(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
    return fig, ax


def save(fig, out_dir, name, dpi=200):
    import os
    fig.tight_layout()
    p = os.path.join(out_dir, name)
    fig.savefig(p, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    print("wrote:", name)
