#!/usr/bin/env python3
"""Generate Chapter 1 figures for the thesis (system architecture + pipeline)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")
os.makedirs(OUT, exist_ok=True)

# palette
C_BROWSER = "#dbeafe"; C_SERVER = "#fef3c7"; C_MODEL = "#dcfce7"
C_LLM = "#fae8ff"; C_DB = "#e2e8f0"; EDGE = "#334155"; TXT = "#0f172a"


def box(ax, x, y, w, h, text, fc, fs=10, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.3, edgecolor=EDGE, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=TXT, weight=weight, wrap=True)


def arrow(ax, x1, y1, x2, y2, style="-|>", color=EDGE, lw=1.6, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                                 linewidth=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))


# ---------------------------------------------------------------- Figure 1.1
fig, ax = plt.subplots(figsize=(9, 6.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

box(ax, 0.5, 8.4, 9, 1.2,
    "Browser  —  MediaPipe Holistic (543 landmarks) · WebRTC · vanilla-JS bilingual UI",
    C_BROWSER, 10, "bold")
box(ax, 2.7, 6.3, 4.6, 1.1,
    "FastAPI + Socket.IO  server\n(routing · auth · real-time signaling)", C_SERVER, 10, "bold")

box(ax, 0.4, 3.9, 2.9, 1.6,
    "Recognition\n• ASL: TFLite (250-class)\n• ArSL: PyTorch CNN-GRU\n  (20-class)", C_MODEL, 9)
box(ax, 3.55, 3.9, 2.9, 1.6,
    "Gloss <-> Sentence\n• Topic-Comment + NMM\n• LLM: Gemini -> Ollama\n  (offline fallback)", C_LLM, 9)
box(ax, 6.7, 3.9, 2.9, 1.6,
    "Sign synthesis\n• text -> gloss\n• SBERT semantic\n  landmark lookup", C_MODEL, 9)

box(ax, 2.4, 1.6, 5.2, 1.1,
    "PostgreSQL + pgvector\n(users · tokens · sign metadata + embeddings)", C_DB, 10, "bold")

# arrows (bidirectional between layers)
arrow(ax, 4.8, 8.4, 4.8, 7.42, rad=0)      # browser -> server
arrow(ax, 5.2, 7.42, 5.2, 8.4, rad=0)      # server -> browser (back)
ax.text(6.0, 7.9, "landmarks / text / audio  ⇄  results",
        fontsize=8, color="#475569", va="center", ha="left")
# server -> three modules
for cx in (1.85, 5.0, 8.15):
    arrow(ax, 5, 6.3, cx, 5.52, rad=0.0)
# three modules -> database
for cx, tx in ((1.85, 3.4), (5.0, 5.0), (8.15, 6.6)):
    arrow(ax, cx, 3.9, tx, 2.72, style="<|-|>", rad=0.0)

ax.set_title("Figure 1.1  —  High-level architecture of the Together system",
             fontsize=11, weight="bold", color=TXT, pad=10)
plt.tight_layout()
f1 = os.path.join(OUT, "fig1_1_architecture.png")
plt.savefig(f1, dpi=200, bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- Figure 1.2
fig, ax = plt.subplots(figsize=(9.5, 4.6))
ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")

# forward: sign -> text/speech
y = 4.2
steps_fwd = [("Camera\nframes", C_BROWSER), ("MediaPipe\nlandmarks", C_BROWSER),
             ("Sign model\n(isolated)", C_MODEL), ("Vote buffer\n-> gloss", C_MODEL),
             ("LLM\nsentence", C_LLM), ("Text /\nTTS speech", C_SERVER)]
w, h, gap = 1.7, 1.0, 0.25
x = 0.3
for i, (t, c) in enumerate(steps_fwd):
    box(ax, x, y, w, h, t, c, 9)
    if i < len(steps_fwd) - 1:
        arrow(ax, x + w, y + h / 2, x + w + gap, y + h / 2)
    x += w + gap
ax.text(0.3, y + h + 0.25, "Sign  ->  Text / Speech", fontsize=10, weight="bold", color=TXT)

# reverse: text/speech -> sign
y = 1.1
steps_rev = [("Text /\nspeech (STT)", C_SERVER), ("Text -> gloss\n(+ NMM)", C_LLM),
             ("SBERT sign\nlookup", C_MODEL), ("Landmark\nsequences", C_MODEL),
             ("Avatar\nplayback", C_BROWSER)]
x = 0.3
for i, (t, c) in enumerate(steps_rev):
    box(ax, x, y, w, h, t, c, 9)
    if i < len(steps_rev) - 1:
        arrow(ax, x + w, y + h / 2, x + w + gap, y + h / 2)
    x += w + gap
ax.text(0.3, y + h + 0.25, "Text / Speech  ->  Sign", fontsize=10, weight="bold", color=TXT)

ax.set_title("Figure 1.2  —  The bidirectional translation pipelines",
             fontsize=11, weight="bold", color=TXT, pad=8)
plt.tight_layout()
f2 = os.path.join(OUT, "fig1_2_pipeline.png")
plt.savefig(f2, dpi=200, bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- Figure 1.3
fig, ax = plt.subplots(figsize=(9.5, 5.0))
ax.set_xlim(0, 12); ax.set_ylim(0, 6.4); ax.axis("off")

box(ax, 0.3, 2.5, 2.3, 1.4, "Signer\n(webcam)", C_BROWSER, 10, "bold")
box(ax, 9.4, 2.5, 2.3, 1.4, "Speaker\n(mic / speaker)", C_BROWSER, 10, "bold")
box(ax, 4.1, 2.4, 3.8, 1.6,
    "Server\nWebRTC media relay +\nSocket.IO signaling +\ntranslation", C_SERVER, 9, "bold")

# top: signer -> speaker (sign -> text/speech)
arrow(ax, 2.6, 3.5, 4.1, 3.5)
arrow(ax, 7.9, 3.5, 9.4, 3.5)
ax.text(6.0, 5.7, "Signer  ->  Speaker", fontsize=10, weight="bold", color=TXT, ha="center")
box(ax, 3.0, 4.5, 6.0, 0.85,
    "landmarks -> recognition -> gloss -> LLM -> captions / speech", C_LLM, 8.5)
arrow(ax, 3.0, 4.9, 2.0, 3.92, rad=-0.2)
arrow(ax, 9.0, 4.9, 10.0, 3.92, rad=0.2)

# bottom: speaker -> signer (speech -> sign)
arrow(ax, 9.4, 2.9, 7.9, 2.9)
arrow(ax, 4.1, 2.9, 2.6, 2.9)
ax.text(6.0, 0.55, "Speaker  ->  Signer", fontsize=10, weight="bold", color=TXT, ha="center")
box(ax, 3.0, 1.05, 6.0, 0.85,
    "speech -> STT -> gloss -> SBERT lookup -> sign avatar", C_MODEL, 8.5)
arrow(ax, 9.0, 1.5, 10.0, 2.48, rad=-0.2)
arrow(ax, 3.0, 1.5, 2.0, 2.48, rad=0.2)

ax.set_title("Figure 1.3  —  The live two-person meeting pipeline",
             fontsize=11, weight="bold", color=TXT, pad=8)
plt.tight_layout()
f3 = os.path.join(OUT, "fig1_3_meeting.png")
plt.savefig(f3, dpi=200, bbox_inches="tight"); plt.close()

print("wrote:", f1)
print("wrote:", f2)
print("wrote:", f3)
