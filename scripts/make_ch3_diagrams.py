#!/usr/bin/env python3
"""Chapter 3 diagram suite: UML + architecture + flowcharts (13 figures)."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from diagram_helpers import *  # noqa

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")
os.makedirs(OUT, exist_ok=True)


# ---- Fig 3.1  System context diagram ----------------------------------------
fig, ax = newfig(10, 6.5, (0, 12), (0, 9))
box(ax, 4.3, 3.9, 3.4, 1.4, "Together\nSystem", AMBER, 12, "bold")
actor(ax, 1.2, 5.0, "Signer"); actor(ax, 1.2, 2.3, "Speaker")
ext = [("MediaPipe\n(browser)", 9.6, 7.2, BLUE), ("Gemini API\n(cloud LLM/TTS/STT)", 9.6, 5.3, PURPLE),
       ("Ollama / Whisper /\npyttsx3 (offline)", 9.6, 3.4, ROSE), ("PostgreSQL\n+ pgvector", 9.6, 1.4, GREY)]
for t, x, y, c in ext:
    box(ax, x - 1.4, y - 0.5, 2.8, 1.0, t, c, 8.5, "bold")
    label_arrow(ax, 7.7, 4.6, x - 1.4, y, "", style="<|-|>")
label_arrow(ax, 1.7, 5.0, 4.3, 4.9, "signs / video", style="<|-|>")
label_arrow(ax, 1.7, 2.3, 4.3, 4.3, "speech / text", style="<|-|>")
title(ax, "Figure 3.1  —  System context diagram")
save(fig, OUT, "fig3_01_context.png")


# ---- Fig 3.2  Use-case diagram ----------------------------------------------
fig, ax = newfig(10, 6.5, (0, 12), (0, 8))
ax.add_patch(FancyBboxPatch((3.2, 0.6), 5.6, 6.8, boxstyle="round,pad=0.02",
             linewidth=1.4, edgecolor=EDGE, facecolor="#f8fafc"))
ax.text(6.0, 7.05, "Together System", ha="center", fontsize=10, weight="bold", color=TXT)
actor(ax, 1.4, 4.0, "Signer"); actor(ax, 10.6, 4.0, "Speaker")
ucs = [("Sign → Text", 6.4), ("Sign → Speech", 5.5), ("Text → Sign", 4.6),
       ("Speech → Sign", 3.7), ("Live Meeting", 2.8), ("Sign up / Log in", 1.6)]
for label, y in ucs:
    ellipse(ax, 6.0, y, 3.2, 0.72, label, AMBER, 9.5)
for _, y in ucs:
    arrow(ax, 1.75, 4.0, 4.4, y, style="-", lw=1.0)
for _, y in [ucs[2], ucs[3], ucs[4], ucs[5]]:
    arrow(ax, 10.25, 4.0, 7.6, y, style="-", lw=1.0)
title(ax, "Figure 3.2  —  Use-case diagram")
save(fig, OUT, "fig3_02_usecase.png")


# ---- Fig 3.3  Component / layered architecture ------------------------------
fig, ax = newfig(10, 7, (0, 12), (0, 10))
box(ax, 0.5, 8.3, 11, 1.2, "Presentation Layer (Browser)\nMediaPipe Holistic · WebRTC · vanilla-JS UI · avatar", BLUE, 9.5, "bold")
box(ax, 0.5, 6.3, 11, 1.3, "Application Layer — FastAPI + Socket.IO\nREST routes · WebRTC signaling · JWT auth · rate limiting", AMBER, 9.5, "bold")
box(ax, 0.4, 3.8, 3.5, 1.9, "Recognition\n• ASL TFLite (250)\n• ArSL CNN-GRU (20)\n• vote buffer", GREEN, 9)
box(ax, 4.25, 3.8, 3.5, 1.9, "Language\n• gloss ↔ sentence\n• Topic-Comment + NMM\n• SBERT lookup", PURPLE, 9)
box(ax, 8.1, 3.8, 3.4, 1.9, "Providers (fallback)\n• LLM gemini→ollama\n• TTS gemini→pyttsx3\n• STT gemini→whisper", ROSE, 9)
box(ax, 2.5, 1.3, 7, 1.3, "Data Layer — PostgreSQL + pgvector\nusers · refresh tokens · sign metadata + embeddings", GREY, 9.5, "bold")
arrow(ax, 5.6, 8.3, 5.6, 7.62); arrow(ax, 6.4, 7.62, 6.4, 8.3)
for cx in (2.15, 6.0, 9.8):
    arrow(ax, 6.0, 6.3, cx, 5.72)
arrow(ax, 6.0, 3.8, 6.0, 2.62)
title(ax, "Figure 3.3  —  Component / layered architecture")
save(fig, OUT, "fig3_03_component.png")


# ---- Fig 3.4  Class diagram -------------------------------------------------
fig, ax = newfig(11.5, 7.5, (0, 14), (0, 10))
classbox(ax, 0.3, 9.6, 3.2, "ASLService",
         ["- interpreter: TFLite", "- labels: list"], ["+ predict(landmarks)", "+ warmup()"], GREEN)
classbox(ax, 0.3, 4.7, 3.2, "ArabicSignService",
         ["- model: CNN_GRU", "- labels: list"], ["+ predict(seq)"], GREEN)
classbox(ax, 4.0, 9.6, 3.4, "GlossEngine",
         ["- llm: ProviderChain", "- cache"], ["+ gloss_to_sentence()", "+ sentence_to_gloss()"], PURPLE)
classbox(ax, 4.0, 4.9, 3.4, "SignDB / ArabicSignDB",
         ["- embeddings: pgvector", "- sbert"], ["+ lookup(word)", "+ batch(words)"], PURPLE)
classbox(ax, 7.9, 9.6, 3.0, "ProviderChain",
         ["- providers[]"], ["+ generate()", "+ fallback()"], ROSE)
classbox(ax, 7.9, 5.2, 3.0, "AuthService",
         ["- secret", "- limiter"], ["+ signup() / login()", "+ refresh() / verify()"], AMBER)
classbox(ax, 11.1, 9.6, 2.6, "MeetingHub",
         ["- rooms"], ["+ join() / leave()", "+ relay()"], BLUE)
classbox(ax, 11.1, 5.4, 2.6, "Repository",
         ["- session"], ["+ get() / add()", "+ commit()"], GREY)
arrow(ax, 5.7, 7.0, 5.7, 6.3, style="-|>")          # GlossEngine -> SignDB (uses)
arrow(ax, 3.5, 8.6, 4.0, 8.6, style="-|>")           # ASLService -> GlossEngine
arrow(ax, 9.4, 7.6, 9.4, 7.0, style="-|>")           # ProviderChain used by Auth? no: GlossEngine->ProviderChain
arrow(ax, 7.4, 8.8, 7.9, 8.8, style="-|>")           # GlossEngine -> ProviderChain
arrow(ax, 9.4, 5.2, 11.1, 6.6, style="-|>")          # AuthService -> Repository
title(ax, "Figure 3.4  —  Class diagram (back-end services)")
save(fig, OUT, "fig3_04_class.png")


# ---- Fig 3.5  Deployment diagram --------------------------------------------
fig, ax = newfig(11, 6.5, (0, 13), (0, 9))
def node(ax, x, y, w, h, stereo, name, body, fc):
    box(ax, x, y, w, h, "", fc, 9)
    ax.text(x + w / 2, y + h - 0.32, "«" + stereo + "»", ha="center", fontsize=8, color=MUTE)
    ax.text(x + w / 2, y + h - 0.7, name, ha="center", fontsize=9.5, weight="bold", color=TXT)
    ax.text(x + w / 2, y + (h - 1.0) / 2, body, ha="center", va="center", fontsize=8, color=TXT)
node(ax, 0.4, 4.2, 3.4, 3.0, "device", "Client (Browser)",
     "MediaPipe.js\napp.js / auth.js\nWebRTC peer\navatar renderer", BLUE)
node(ax, 4.6, 4.2, 4.2, 3.0, "server", "Web Server (Docker)",
     "FastAPI + Socket.IO\nTFLite · PyTorch\nSBERT · gloss engine", AMBER)
node(ax, 9.4, 5.4, 3.2, 1.8, "db", "Database Server",
     "PostgreSQL\n+ pgvector", GREY)
node(ax, 9.4, 3.0, 3.2, 1.8, "service", "Local LLM",
     "Ollama\n(llama3.2)", ROSE)
node(ax, 4.6, 1.0, 4.2, 1.6, "cloud", "Cloud Provider",
     "Gemini API (LLM/TTS/STT)", PURPLE)
label_arrow(ax, 3.8, 5.7, 4.6, 5.7, "HTTPS / WS", style="<|-|>")
label_arrow(ax, 8.8, 6.0, 9.4, 6.0, "SQL", style="<|-|>")
label_arrow(ax, 8.8, 5.0, 9.4, 4.0, "HTTP", style="<|-|>")
label_arrow(ax, 6.7, 4.2, 6.7, 2.6, "HTTPS", style="<|-|>")
ax.text(2.1, 3.9, "WebRTC P2P media ⇄ peer", ha="center", fontsize=7.5, color=MUTE)
title(ax, "Figure 3.5  —  Deployment diagram")
save(fig, OUT, "fig3_05_deployment.png")


# ---- Fig 3.6  ER diagram ----------------------------------------------------
fig, ax = newfig(10, 5.5, (0, 12), (0, 8))
def entity(ax, x, y, w, name, attrs, fc):
    h = 0.55 + 0.42 * len(attrs)
    ax.add_patch(Rectangle((x, y - h), w, h, lw=1.4, edgecolor=EDGE, facecolor=WHITE))
    ax.add_patch(Rectangle((x, y - 0.55), w, 0.55, lw=1.4, edgecolor=EDGE, facecolor=fc))
    ax.text(x + w / 2, y - 0.28, name, ha="center", va="center", fontsize=9.5, weight="bold", color=TXT)
    for i, a in enumerate(attrs):
        ax.text(x + 0.15, y - 0.8 - i * 0.42, a, ha="left", va="center", fontsize=8, color=TXT)
entity(ax, 0.5, 7.4, 3.3, "users", ["id (PK)", "email", "password_hash", "created_at"], GREY)
entity(ax, 0.5, 3.0, 3.3, "refresh_tokens", ["id (PK)", "user_id (FK)", "token_hash", "expires_at"], GREY)
entity(ax, 7.0, 7.4, 4.3, "signs", ["id (PK)", "language (en/ar)", "gloss / label", "landmark_ref", "embedding vector(384)"], GREEN)
arrow(ax, 2.15, 4.6, 2.15, 3.45, style="-|>")
ax.text(2.35, 4.0, "1", fontsize=9, color=TXT); ax.text(2.35, 3.6, "∗", fontsize=11, color=TXT)
ax.text(3.7, 4.05, "owns", fontsize=8, color=MUTE)
ax.text(9.1, 2.4, "queried by SBERT\nsemantic lookup", ha="center", fontsize=8, color=MUTE)
title(ax, "Figure 3.6  —  Entity-relationship diagram")
save(fig, OUT, "fig3_06_er.png")


# ---- Fig 3.7  Data-flow diagram (level 1) -----------------------------------
fig, ax = newfig(11, 6, (0, 13), (0, 8))
box(ax, 0.4, 5.6, 1.9, 1.0, "Signer", BLUE, 9, "bold", round=False)
box(ax, 0.4, 1.2, 1.9, 1.0, "Speaker", BLUE, 9, "bold", round=False)
for cx, cy, t in [(4.2, 6.0, "1. Extract\nlandmarks"), (7.0, 6.0, "2. Recognise\n+ vote"),
                  (9.8, 6.0, "3. Gloss →\nsentence"), (4.2, 1.6, "5. Synthesise\nsign"),
                  (7.0, 1.6, "4. Sentence →\ngloss")]:
    ellipse(ax, cx, cy, 2.2, 1.1, t, GREEN, 8.5)
datastore(ax, 8.7, 0.8, 3.2, 0.8, "D1  signs + embeddings", GREY, 8.5)
label_arrow(ax, 2.3, 6.0, 3.1, 6.0, "video")
label_arrow(ax, 5.3, 6.0, 5.9, 6.0, "landmarks")
label_arrow(ax, 8.1, 6.0, 8.7, 6.0, "gloss")
label_arrow(ax, 9.8, 5.45, 1.35, 2.2, "text / speech", rad=-0.15)
label_arrow(ax, 2.3, 1.6, 5.9, 1.6, "text")
label_arrow(ax, 5.9, 1.6, 5.3, 1.6, "gloss")
label_arrow(ax, 4.2, 2.15, 2.3, 5.6, "avatar", rad=0.15)
arrow(ax, 7.0, 1.05, 8.7, 1.1, style="-|>")
arrow(ax, 8.7, 1.4, 5.3, 1.7, style="-|>", rad=0.1)
title(ax, "Figure 3.7  —  Data-flow diagram (level 1)")
save(fig, OUT, "fig3_07_dfd.png")


# ---- Fig 3.8  Activity diagram ----------------------------------------------
fig, ax = newfig(7.5, 9.5, (0, 8), (0, 13))
start_end(ax, 4, 12.3, "start")
acts = [("Capture frame & extract landmarks", 11.2),
        ("Buffer to 60 frames", 10.2),
        ("Model predicts sign", 9.2)]
for t, y in acts:
    box(ax, 1.3, y - 0.4, 5.4, 0.8, t, BLUE, 9)
arrow(ax, 4, 12.05, 4, 11.6)
arrow(ax, 4, 10.8, 4, 10.6); arrow(ax, 4, 9.8, 4, 9.6)
diamond(ax, 4, 8.2, 3.2, 1.1, "top prob ≥ 0.80 ?", AMBER, 8.5)
arrow(ax, 4, 8.8, 4, 9.0 - 0.2)
arrow(ax, 2.4, 8.2, 1.3, 8.2); ax.text(1.0, 8.45, "no", fontsize=8, color=MUTE)
arrow(ax, 1.0, 8.2, 1.0, 11.2); arrow(ax, 1.0, 11.2, 1.3, 11.2)  # loop back to capture
box(ax, 1.3, 6.7, 5.4, 0.8, "Majority vote (last 15) → commit gloss", GREEN, 9)
arrow(ax, 4, 7.65, 4, 7.5); ax.text(4.25, 7.55, "yes", fontsize=8, color=MUTE)
diamond(ax, 4, 5.3, 3.4, 1.1, "hands idle ≥ 5 s ?", AMBER, 8.5)
arrow(ax, 4, 6.7, 4, 5.85)
arrow(ax, 5.7, 5.3, 6.7, 5.3); ax.text(6.9, 5.55, "no", fontsize=8, color=MUTE)
arrow(ax, 6.7, 5.3, 6.7, 11.2); arrow(ax, 6.7, 11.2, 6.7, 11.2)
ax.add_patch(FancyArrowPatch((6.7, 5.3), (6.7, 11.6), arrowstyle="-", lw=1.2, color=EDGE))
arrow(ax, 6.7, 11.6, 6.7, 11.2)
box(ax, 1.3, 3.8, 5.4, 0.8, "LLM forms sentence (gloss → text)", PURPLE, 9)
arrow(ax, 4, 4.75, 4, 4.6); ax.text(4.25, 4.67, "yes", fontsize=8, color=MUTE)
box(ax, 1.3, 2.5, 5.4, 0.8, "Display caption + speak (TTS)", AMBER, 9)
arrow(ax, 4, 3.8, 4, 3.3)
start_end(ax, 4, 1.4, "end")
arrow(ax, 4, 2.5, 4, 1.7)
title(ax, "Figure 3.8  —  Activity diagram (sign → sentence)")
save(fig, OUT, "fig3_08_activity.png")


# ---- Fig 3.9  State machine -------------------------------------------------
fig, ax = newfig(10.5, 5.5, (0, 13), (0, 7))
ax.add_patch(Circle((0.7, 3.5), 0.18, color=EDGE))
states = [("Idle", 2.2), ("Detecting", 4.4), ("Voting", 6.6), ("Committed", 8.8), ("Composing", 11.0)]
for t, x in states:
    box(ax, x - 1.0, 3.0, 2.0, 1.0, t, BLUE, 9.5, "bold")
arrow(ax, 0.88, 3.5, 1.2, 3.5)
trans = [(3.2, 4.4, "hands in frame"), (5.4, 6.6, "≥18 frames"),
         (7.6, 8.8, "vote stable"), (9.8, 11.0, "idle 5 s")]
for x1, x2, lab in [(3.2, 3.4, ""), ]:
    pass
pairs = [(2.2, 4.4, "hands detected"), (4.4, 6.6, "buffer full"),
         (6.6, 8.8, "≥0.80 & vote"), (8.8, 11.0, "idle 5 s")]
for x1, x2, lab in pairs:
    label_arrow(ax, x1 + 1.0, 3.5, x2 - 1.0, 3.5, lab)
arrow(ax, 11.0, 4.0, 11.0, 5.2); arrow(ax, 11.0, 5.2, 2.2, 5.2); arrow(ax, 2.2, 5.2, 2.2, 4.0)
ax.text(6.6, 5.4, "sentence spoken → reset", ha="center", fontsize=8, color=MUTE)
arrow(ax, 4.4, 3.0, 4.4, 2.0); arrow(ax, 4.4, 2.0, 2.2, 2.0); arrow(ax, 2.2, 2.0, 2.2, 3.0)
ax.text(3.3, 1.8, "low confidence", ha="center", fontsize=8, color=MUTE)
title(ax, "Figure 3.9  —  Recognition state machine")
save(fig, OUT, "fig3_09_state.png")


# ---- Sequence diagram helper ------------------------------------------------
def seqfig(name, fignum_title, lanes, msgs, w=10, h=6.5, ytop=8.6, ybot=0.6):
    fig, ax = newfig(w, h, (0, 12), (0, 9.4))
    xs = {}
    for lx, lab, fc in lanes:
        lifeline(ax, lx, ytop, ybot, lab, fc)
        xs[lab.split("\n")[0]] = lx
    for a, b, y, text, dash in msgs:
        msg(ax, a, b, y, text, dashed=dash)
    title(ax, fignum_title)
    save(fig, OUT, name)


L = lambda x, lab, fc=BLUE: (x, lab, fc)
# ---- Fig 3.10  Sequence: Sign -> Text/Speech --------------------------------
seqfig("fig3_10_seq_sign2text.png", "Figure 3.10  —  Sequence: Sign → Text / Speech",
       [L(2.0, "Browser"), L(6.0, "FastAPI\nserver", AMBER), L(10.0, "Model +\nLLM", GREEN)],
       [(2.0, 6.0, 7.9, "POST /api/translate (landmarks×60)", False),
        (6.0, 10.0, 7.1, "infer sign (TFLite)", False),
        (10.0, 6.0, 6.3, "label + confidence", True),
        (6.0, 2.0, 5.5, "voted gloss", True),
        (2.0, 6.0, 4.6, "POST /api/translate/sentence (gloss)", False),
        (6.0, 10.0, 3.8, "gloss → sentence (Gemini)", False),
        (10.0, 6.0, 3.0, "sentence", True),
        (6.0, 2.0, 2.2, "sentence (+ /api/tts audio)", True)])

# ---- Fig 3.11  Sequence: Text/Speech -> Sign --------------------------------
seqfig("fig3_11_seq_text2sign.png", "Figure 3.11  —  Sequence: Text / Speech → Sign",
       [L(2.0, "Browser"), L(6.0, "FastAPI\nserver", AMBER), L(10.0, "SBERT +\nDB", PURPLE)],
       [(2.0, 6.0, 7.9, "POST /api/gloss (text)", False),
        (6.0, 6.0, 7.1, "sentence → gloss + NMM", False),
        (6.0, 2.0, 6.3, "gloss tokens", True),
        (2.0, 6.0, 5.5, "POST /api/signs/batch (tokens)", False),
        (6.0, 10.0, 4.7, "semantic lookup (pgvector)", False),
        (10.0, 6.0, 3.9, "landmark sequences", True),
        (6.0, 2.0, 3.1, "sign clips", True),
        (2.0, 2.0, 2.3, "stitch + play avatar", False)])

# ---- Fig 3.12  Sequence: Live meeting ---------------------------------------
seqfig("fig3_12_seq_meeting.png", "Figure 3.12  —  Sequence: Live meeting (WebRTC)",
       [L(2.0, "Signer"), L(6.0, "Server\n(Socket.IO)", AMBER), L(10.0, "Speaker")],
       [(2.0, 6.0, 8.0, "join_room / announce_presence", False),
        (10.0, 6.0, 7.3, "join_room / announce_presence", False),
        (2.0, 10.0, 6.5, "webrtc_offer (via server)", False),
        (10.0, 2.0, 5.8, "webrtc_answer", False),
        (2.0, 10.0, 5.1, "ice_candidate ⇄", False),
        (2.0, 10.0, 4.3, "P2P media (WebRTC)", False),
        (2.0, 6.0, 3.4, "translate_sentence (gloss→text)", False),
        (6.0, 10.0, 2.7, "caption / speech", True),
        (10.0, 6.0, 1.9, "speech → sign request", False),
        (6.0, 2.0, 1.2, "sign avatar", True)])

# ---- Fig 3.13  Sequence: Authentication -------------------------------------
seqfig("fig3_13_seq_auth.png", "Figure 3.13  —  Sequence: Authentication (JWT)",
       [L(2.0, "Browser"), L(6.0, "AuthService", AMBER), L(10.0, "Database", GREY)],
       [(2.0, 6.0, 8.0, "POST /api/auth/login (email, pw)", False),
        (6.0, 10.0, 7.2, "fetch user by email", False),
        (10.0, 6.0, 6.4, "password_hash", True),
        (6.0, 6.0, 5.6, "verify (Argon2) + rate-limit", False),
        (6.0, 10.0, 4.8, "store refresh_token hash", False),
        (6.0, 2.0, 4.0, "access token + httpOnly refresh", True),
        (2.0, 6.0, 3.1, "POST /api/auth/refresh", False),
        (6.0, 2.0, 2.3, "new access token", True)])

print("done")
