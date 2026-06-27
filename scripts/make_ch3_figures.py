#!/usr/bin/env python3
"""Generate Chapter 3 (System Analysis & Design) figures."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Circle

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")
os.makedirs(OUT, exist_ok=True)

C_CLIENT = "#dbeafe"; C_APP = "#fef3c7"; C_ML = "#dcfce7"
C_LLM = "#fae8ff"; C_DB = "#e2e8f0"; C_ACC = "#ffe4e6"
EDGE = "#334155"; TXT = "#0f172a"


def box(ax, x, y, w, h, text, fc, fs=10, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                                linewidth=1.3, edgecolor=EDGE, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=TXT, weight=weight)


def arrow(ax, x1, y1, x2, y2, style="-|>", color=EDGE, lw=1.6, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13,
                                 linewidth=lw, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def actor(ax, x, y, label):
    ax.add_patch(Circle((x, y + 0.5), 0.16, fill=False, lw=1.6, edgecolor=EDGE))
    ax.plot([x, x], [y + 0.34, y - 0.15], color=EDGE, lw=1.6)
    ax.plot([x - 0.22, x + 0.22], [y + 0.18, y + 0.18], color=EDGE, lw=1.6)
    ax.plot([x, x - 0.18], [y - 0.15, y - 0.5], color=EDGE, lw=1.6)
    ax.plot([x, x + 0.18], [y - 0.15, y - 0.5], color=EDGE, lw=1.6)
    ax.text(x, y - 0.75, label, ha="center", va="center", fontsize=10, weight="bold", color=TXT)


def title(ax, t):
    ax.set_title(t, fontsize=11, weight="bold", color=TXT, pad=10)


def save(fig, name):
    fig.tight_layout()
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("wrote:", os.path.basename(p))


# ---------------------------------------------------------- Fig 3.1 Use cases
fig, ax = plt.subplots(figsize=(9.5, 6)); ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")
ax.add_patch(FancyBboxPatch((3.2, 0.6), 5.6, 6.8, boxstyle="round,pad=0.02",
                            linewidth=1.4, edgecolor=EDGE, facecolor="#f8fafc"))
ax.text(6.0, 7.05, "Together System", ha="center", fontsize=10, weight="bold", color=TXT)
actor(ax, 1.4, 4.0, "Signer"); actor(ax, 10.6, 4.0, "Speaker")
ucs = [("Sign → Text", 6.4), ("Sign → Speech", 5.5), ("Text → Sign", 4.6),
       ("Speech → Sign", 3.7), ("Live Meeting", 2.8), ("Sign up / Log in", 1.6)]
for label, y in ucs:
    ax.add_patch(Ellipse((6.0, y), 3.2, 0.7, fill=True, facecolor=C_APP, edgecolor=EDGE, lw=1.2))
    ax.text(6.0, y, label, ha="center", va="center", fontsize=9.5, color=TXT)
for _, y in ucs[:5] + [ucs[5]]:
    arrow(ax, 1.75, 4.0, 4.4, y, style="-", lw=1.0)
for _, y in [ucs[2], ucs[3], ucs[4], ucs[5]]:
    arrow(ax, 10.25, 4.0, 7.6, y, style="-", lw=1.0)
title(ax, "Figure 3.1  —  Use-case diagram")
save(fig, "fig3_1_usecases.png")

# ---------------------------------------------------------- Fig 3.2 Architecture
fig, ax = plt.subplots(figsize=(9.5, 7)); ax.set_xlim(0, 12); ax.set_ylim(0, 10); ax.axis("off")
box(ax, 0.5, 8.3, 11, 1.3,
    "Presentation Layer (Browser)\nMediaPipe Holistic · WebRTC · vanilla-JS bilingual UI · avatar", C_CLIENT, 9.5, "bold")
box(ax, 0.5, 6.2, 11, 1.4,
    "Application Layer — FastAPI + Socket.IO\nREST routes · WebRTC signaling · JWT auth · rate limiting", C_APP, 9.5, "bold")
box(ax, 0.4, 3.7, 3.5, 1.9, "Recognition\n• ASL TFLite (250)\n• ArSL CNN-GRU (20)\n• vote buffer", C_ML, 9)
box(ax, 4.25, 3.7, 3.5, 1.9, "Language\n• gloss <-> sentence\n• Topic-Comment + NMM\n• SBERT sign lookup", C_LLM, 9)
box(ax, 8.1, 3.7, 3.4, 1.9, "Providers (fallback)\n• LLM gemini->ollama\n• TTS gemini->pyttsx3\n• STT gemini->whisper", C_ACC, 9)
box(ax, 2.5, 1.2, 7, 1.4, "Data Layer — PostgreSQL + pgvector\nusers · refresh tokens · sign metadata + embeddings", C_DB, 9.5, "bold")
arrow(ax, 5.6, 8.3, 5.6, 7.62); arrow(ax, 6.4, 7.62, 6.4, 8.3)
for cx in (2.15, 6.0, 9.8):
    arrow(ax, 6.0, 6.2, cx, 5.62)
arrow(ax, 6.0, 3.7, 6.0, 2.62)
title(ax, "Figure 3.2  —  Layered system architecture")
save(fig, "fig3_2_architecture.png")

# ---------------------------------------------------------- Fig 3.3 Sign->Text pipeline
fig, ax = plt.subplots(figsize=(10, 4.2)); ax.set_xlim(0, 13); ax.set_ylim(0, 5); ax.axis("off")
steps = [("Webcam\nframes", C_CLIENT), ("MediaPipe\n543 landmarks", C_CLIENT),
         ("Recognition\nmodel", C_ML), ("Vote buffer\n(debounce)", C_ML),
         ("Gloss\nsequence", C_ML), ("LLM\nsentence", C_LLM), ("Text /\nTTS", C_APP)]
w, h, gap, x, y = 1.55, 1.1, 0.2, 0.2, 3.0
for i, (t, c) in enumerate(steps):
    box(ax, x, y, w, h, t, c, 9)
    if i < len(steps) - 1:
        arrow(ax, x + w, y + h / 2, x + w + gap, y + h / 2)
    x += w + gap
box(ax, 3.2, 0.5, 6.2, 1.4,
    "Pad / trim to 60 frames (NaN for missing hands)\n"
    "majority vote over last 15 · accept ≥ 0.80 / 0.65\n"
    "gloss → LLM after 5 s of no hands", "#f1f5f9", 8)
arrow(ax, 6.2, 3.0, 6.2, 1.95, ls="--", lw=1.2)
title(ax, "Figure 3.3  —  Sign → Text / Speech pipeline")
save(fig, "fig3_3_sign2text.png")

# ---------------------------------------------------------- Fig 3.4 Text->Sign pipeline
fig, ax = plt.subplots(figsize=(10, 3.4)); ax.set_xlim(0, 13); ax.set_ylim(0, 3.2); ax.axis("off")
steps = [("Text / speech\n(STT)", C_APP), ("Sentence ->\ngloss + NMM", C_LLM),
         ("SBERT semantic\nsign lookup", C_ML), ("Landmark\nsequences", C_ML),
         ("Stitch +\navatar playback", C_CLIENT)]
w, h, gap, x, y = 2.1, 1.2, 0.25, 0.3, 1.2
for i, (t, c) in enumerate(steps):
    box(ax, x, y, w, h, t, c, 9)
    if i < len(steps) - 1:
        arrow(ax, x + w, y + h / 2, x + w + gap, y + h / 2)
    x += w + gap
title(ax, "Figure 3.4  —  Text / Speech → Sign pipeline")
save(fig, "fig3_4_text2sign.png")

# ---------------------------------------------------------- Fig 3.5 Model architectures
fig, ax = plt.subplots(figsize=(9.5, 6)); ax.set_xlim(0, 12); ax.set_ylim(0, 9); ax.axis("off")
ax.text(3.0, 8.4, "(a) ASL — preprocessing + Squeezeformer (TFLite)", ha="center", fontsize=9, weight="bold", color=TXT)
asl = ["Raw 60×543×3 → select\n118 landmarks, drop Z (X,Y)", "Nose-center (#17)\n+ std normalise",
       "Motion (dx, dy, dx², dy²)\n→ 708 feats / frame", "Stem Conv → 192",
       "Conv1D blocks (causal DW\nk=17 + ECA) + transformer", "GAP → Dense 250\nsoftmax → TFLite"]
y = 7.2
for t in asl:
    box(ax, 1.2, y, 3.5, 0.8, t, C_ML, 7.5)
    if t != asl[-1]:
        arrow(ax, 2.95, y, 2.95, y - 0.5)
    y -= 1.2
ax.text(9.0, 8.4, "(b) ArSL — CNN-GRU model", ha="center", fontsize=9.5, weight="bold", color=TXT)
ar = ["Input: N × 177\nskeletal sequence", "Conv1D 177→128\n(k=3) + BN + ReLU",
      "Conv1D 128→128\n(k=3) + BN + ReLU", "Bi-GRU 128→64\n2 layers, dropout 0.3",
      "FC 128→64\nReLU + Dropout 0.5", "FC 64→20\nsoftmax"]
y = 7.2
for t in ar:
    box(ax, 7.4, y, 3.2, 0.8, t, C_LLM, 8)
    if t != ar[-1]:
        arrow(ax, 9.0, y, 9.0, y - 0.5)
    y -= 1.2
title(ax, "Figure 3.5  —  Recognition model architectures")
save(fig, "fig3_5_models.png")

# ---------------------------------------------------------- Fig 3.6 Provider fallback
fig, ax = plt.subplots(figsize=(9.5, 4.5)); ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
rows = [("LLM", "Gemini (cloud)", "Ollama (offline)", 3.7),
        ("TTS", "Gemini TTS", "pyttsx3 (offline)", 2.4),
        ("STT", "Gemini STT", "faster-whisper (offline)", 1.1)]
for name, primary, fb, y in rows:
    box(ax, 0.4, y, 1.5, 0.9, name, "#f1f5f9", 10, "bold")
    box(ax, 2.6, y, 3.4, 0.9, primary, C_LLM, 9)
    box(ax, 7.4, y, 3.8, 0.9, fb, C_ACC, 9)
    arrow(ax, 6.0, y + 0.45, 7.4, y + 0.45)
    ax.text(6.7, y + 0.72, "if unavailable", ha="center", fontsize=7.5, color="#64748b")
title(ax, "Figure 3.6  —  Provider fallback chains (cloud → offline)")
save(fig, "fig3_6_providers.png")

# ---------------------------------------------------------- Fig 3.7 Meeting sequence
fig, ax = plt.subplots(figsize=(9.5, 6.5)); ax.set_xlim(0, 12); ax.set_ylim(0, 10); ax.axis("off")
lanes = [(2.0, "Signer\n(browser)"), (6.0, "Server\n(Socket.IO)"), (10.0, "Speaker\n(browser)")]
for lx, lab in lanes:
    box(ax, lx - 1.1, 9.0, 2.2, 0.8, lab, C_CLIENT if lx != 6 else C_APP, 9, "bold")
    ax.plot([lx, lx], [0.6, 9.0], color="#94a3b8", lw=1.0, ls="--")
msgs = [(2.0, 6.0, "join_room / announce_presence", 8.3),
        (10.0, 6.0, "join_room / announce_presence", 7.6),
        (2.0, 10.0, "webrtc_offer (via server)", 6.7),
        (10.0, 2.0, "webrtc_answer", 6.0),
        (2.0, 10.0, "ice_candidate ⇄", 5.3),
        (2.0, 10.0, "P2P media (WebRTC)", 4.4),
        (2.0, 6.0, "translate_sentence (gloss→text)", 3.3),
        (6.0, 10.0, "caption / speech", 2.6),
        (10.0, 6.0, "speech→sign request", 1.8),
        (6.0, 2.0, "sign avatar", 1.1)]
for x1, x2, label, y in msgs:
    ddir = 1 if x2 > x1 else -1
    arrow(ax, x1 + 0.05 * ddir, y, x2 - 0.05 * ddir, y, lw=1.4)
    ax.text((x1 + x2) / 2, y + 0.12, label, ha="center", fontsize=8, color=TXT)
title(ax, "Figure 3.7  —  Live-meeting signaling sequence")
save(fig, "fig3_7_meeting_seq.png")

# ---------------------------------------------------------- Fig 3.8 ER diagram
fig, ax = plt.subplots(figsize=(9.5, 5.5)); ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")

def entity(ax, x, y, w, name, attrs, fc):
    h = 0.55 + 0.42 * len(attrs)
    ax.add_patch(FancyBboxPatch((x, y - h), w, h, boxstyle="square,pad=0.0",
                                linewidth=1.4, edgecolor=EDGE, facecolor="white"))
    ax.add_patch(FancyBboxPatch((x, y - 0.55), w, 0.55, boxstyle="square,pad=0.0",
                                linewidth=1.4, edgecolor=EDGE, facecolor=fc))
    ax.text(x + w / 2, y - 0.28, name, ha="center", va="center", fontsize=9.5, weight="bold", color=TXT)
    for i, a in enumerate(attrs):
        ax.text(x + 0.15, y - 0.8 - i * 0.42, a, ha="left", va="center", fontsize=8, color=TXT)
    return (x, y, w, h)

entity(ax, 0.5, 7.4, 3.3, "users",
       ["id (PK)", "email", "password_hash", "created_at"], C_DB)
entity(ax, 0.5, 3.0, 3.3, "refresh_tokens",
       ["id (PK)", "user_id (FK)", "token_hash", "expires_at"], C_DB)
entity(ax, 7.0, 7.4, 4.3, "signs",
       ["id (PK)", "language (en/ar)", "gloss / label", "landmark_ref",
        "embedding  vector(384)"], C_ML)
arrow(ax, 2.15, 4.6, 2.15, 3.45, style="-|>")
ax.text(2.35, 4.0, "1", fontsize=9, color=TXT); ax.text(2.35, 3.6, "*", fontsize=11, color=TXT)
ax.text(4.0, 4.0, "owns", fontsize=8, color="#64748b")
ax.text(9.1, 2.4, "queried by SBERT\nsemantic lookup", ha="center", fontsize=8, color="#64748b")
title(ax, "Figure 3.8  —  Database entity-relationship diagram")
save(fig, "fig3_8_er.png")

print("done")
