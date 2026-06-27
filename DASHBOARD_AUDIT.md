# Together — Dashboard Audit, Diagnosis & Action Plan

Grounded in a full code investigation of this repo + external research (ElevenLabs/Signapse UX, MediaPipe/TFLite perf, Render hosting). Every claim below cites the exact file/line or an external source.

---

## TL;DR — the three things that actually matter

1. **The model isn't "overfit" and isn't a training problem. It's slow because of *deployment + architecture*, not the weights.** Inference runs **server-side on Render** (Python TFLite+PyTorch), models **lazy-load on first request**, free-tier Render **spins down after 15 min** (30–60s cold start), and each detection needs **2–4 HTTP round-trips** for a majority vote. Fix those four things and it gets dramatically faster — no retraining required.

2. **Compose-sentence backend is fine and Gemini *is* wired up.** The bug is purely frontend: composed sentences in Sign→Text are thrown away (a boolean flag skips the log push), errors are swallowed silently, and Enter is disabled in Arabic. Needs a `GEMINI_API_KEY` set on Render.

3. **The "white invisible log text", duplicated logo, broken sign counter, and empty dashboard** are all real, all frontend, all fixable now without any paid service.

What costs money: **Render Starter ($7/mo)** to kill cold-starts, and a **Gemini API key** (free tier exists) for sentence composition. Nothing else needs paying for, linking, or retraining.

---

## PART 1 — Bug diagnoses (code-level, with fixes)

### 1.1 "Recent Translations" text is white/invisible on dark dashboard
- **Where:** `app/templates/index.html` `.d-log-msg` uses `color:var(--text)` (correct), but the right-column log cards render text that visually disappears.
- **Root cause:** The replay-button injection + the log markup leaves items without the expected text color in some paths, and `--text` on dark is `#f4f4f7` (near-white) — correct for *cards* but the log container has no solid surface in places, so white-on-near-black low contrast + the monospace gloss (`--accent` teal) is what you're seeing. Also the `t2s-history`/`s2sp-log` placeholder uses `--faint`.
- **Fix:** Force explicit, theme-correct colors on `.d-log-msg`/`.d-log-gloss`/`.d-log-time` and ensure the log sits on a `--surface` card. Quick, zero-risk.

### 1.2 Model slow to start & slow to translate — **the big one**
**Architecture (confirmed in code):**
- Client: MediaPipe **Holistic** (CDN) extracts 543 landmarks/frame in-browser → POSTs to server.
- Server: `/api/translate` (`app/server/main.py:851`) runs **TFLite** (`models/model.tflite`, 11 MB, 250-class ASL) or **PyTorch** (`models/best_model.pth`, 1.1 MB, 20-class Arabic) in a threadpool.
- Models are **lazy-loaded on first request** (`get_asl_engine()` main.py:288; `get_arabic_engine()` main.py:378) → 500–1000 ms one-time load.
- Detection needs **2–4 round-trips**: client buffers ≥18 frames (`MIN_SEQ_FRAMES`, 600 ms), then votes (`VOTE_BUFFER_SIZE=4`, `STABILITY_FRAMES=2`), each vote = one POST.
- Confidence thresholds: **ASL TFLite = 0.80** (`asl_service.py:282`), **Arabic = 0.65** (`sign_predictor.py:325`).

**What's actually causing slowness, in order of impact:**
1. **Render free-tier cold start: 30–60 s** after 15-min idle (spin-down), *then* your own 500–1000 ms model load. This is the "really slow to start." [Render free tier](https://deploybase.app/blog/render-free-tier-complete-guide-2026)
2. **2–4 HTTP round-trips per sign** (network RTT × N + server compute × N). This is the "slow to translate."
3. **MediaPipe Holistic is structurally heavy**: 5–6 chained models, 540+ landmarks, ~15–20 MB cold download, and a documented **10+ s main-thread init hang** on `new Holistic()`. [MediaPipe issue #1918](https://github.com/google/mediapipe/issues/1918), [Holistic docs](https://github.com/google/mediapipe/blob/master/docs/solutions/holistic.md)
4. **Client frame-buffering** (600 ms minimum before first inference).
5. **Weak free-tier CPU** (0.1 vCPU) for model load + inference.

**Is it overfitting / a dataset problem? No.** Overfitting is an *accuracy* problem (good on train data, bad on real input) — it has **zero effect on speed**. An overfit and a well-generalized model of the same architecture run at identical speed. So slowness is essentially never overfitting. The cause here is deployment + round-trips + Holistic, not the weights.

**Does lowering the confidence threshold make it faster? No.** The forward pass and probabilities are identical regardless of threshold; the threshold is only the *acceptance cutoff*. Lowering 0.80→0.65 makes it **accept sooner / more often** (feels snappier, possibly fewer votes to commit) but **more false positives**. It changes acceptance rate, not compute time.

**Fixes (priority order, with cost):**
1. **Upgrade to Render Starter (~$7/mo)** — removes 30–60 s cold start, keeps process warm, models stay loaded. *Single highest-impact fix.* (Standard ~$25/mo, 2 GB RAM, if memory-pressured.) [Render pricing](https://render.com/pricing)
2. **Pre-load both models at startup** instead of lazy — first request no longer eats 500–1000 ms. (Free, code change.)
3. **Batch the 2–4 vote frames into ONE POST**, vote server-side, return a single result → cuts round-trip latency 2–4×. (Free, code change.)
4. **Set TFLite XNNPACK `num_threads` + PyTorch `torch.set_num_threads()` to vCPU count**; run under `torch.inference_mode()`; confirm INT8 quant on Arabic model. (Free.)
5. **Optional, larger:** switch client from Holistic → `@mediapipe/tasks-vision` HandLandmarker (+ Pose complexity 0 if pose is needed). Much lighter load. **Caveat:** the server models expect 543 landmarks incl. face+pose — dropping to hands-only changes the input vector and would require remap/retrain. Only do this if the models don't truly use face/pose context. [tasks-vision](https://www.npmjs.com/package/@mediapipe/tasks-vision)

### 1.3 Compose-sentence buggy / no session-log integration / Enter does nothing
- **Gemini IS wired up:** `app/server/providers/gemini.py` (`gemini-2.5-flash`), endpoint `POST /api/translate/sentence` (main.py:893) → `gloss.gloss_to_sentence()` (few-shot prompt, cached). Requires `GEMINI_API_KEY` env var.
- **Bug 1 (critical):** Sign→Text calls `updateTranscription(sentence, true)`; inside `updateTranscription()` the log push is guarded by `if (text && !isSentence)` → **composed sentences are never logged.** (Sign→Speech does it correctly via `s2spSentences`.)
- **Bug 2 (high):** API errors are swallowed (`catch (_) {}`) — if `GEMINI_API_KEY` is missing it silently returns raw gloss with no user feedback.
- **Bug 3 (medium):** Enter handler early-returns for Arabic (`if (forcedLang === 'ar') return`) → can't compose in Arabic at all.
- **Bug 4 (medium):** No loading/success/error visual feedback.
- **Fixes:** log composed sentences; surface errors as a toast/inline state; enable Enter in Arabic; add a spinner/“composing…” state; **set `GEMINI_API_KEY` on Render** (free tier: [ai.google.dev](https://ai.google.dev/)).

### 1.4 "Signs" top counter doesn't count across all dashboards
- **Root cause:** `_signCount` only increments inside `_addReplayBtn()`, which only fires for 3 logs (`session-log`, `s2sp-log`, `t2s-history`) via MutationObserver. Speech→Sign and Live Meeting aren't counted; `t2s-history` counts *translations*, not signs.
- **Fix:** increment a shared counter at the actual sign-detection event in every module (hook the detection callback, not the DOM), persist to `localStorage` so it's a true running total.

### 1.5 Sidebar: logo + buttons buggy when minimized, toggle outside the bar
- **Logo duplication:** sidebar has TWO `<img>` (dark+light) toggled by CSS, **plus** a third `::after` background-image mark when collapsed. On collapse the imgs are `display:none` and the `::after` shows — fragile, and on the landing page the dark/light pair reads as "duplicated."
- **Toggle location:** `#sb-toggle` is in `.t-topbar` (index.html:385), *outside* the sidebar. You want it **inside the sidebar, right of the logo.**
- **Logo color:** you're seeing the white lockup on dark (correct) but it's not fit to the sidebar box and the collapsed mark is finicky.
- **Fix:** one logo element using a single source that swaps via theme; move `#sb-toggle` into the sidebar header next to the logo; collapsed state shows just the hands mark cleanly; center nav items when collapsed.

### 1.6 Product names not in sidebar
- Nav currently says generic "Sign to Text / Sign to Speech / …". The `data-product` attributes (`handscript/voicebridge/signtype/talkside`) already exist on panels.
- **Fix:** show stylized names as the primary label with the function as a sub-label, e.g. **HandScript** · Sign→Text, **VoiceBridge** · Sign→Speech, **SignType** · Text→Sign, **TalkSide** · Speech→Sign.

---

## PART 2 — Design overhaul (research-backed, fills empty space)

From ElevenLabs, Signapse, and dashboard best-practice research:

**Structure**
- **Group the sidebar** like ElevenLabs: products on top (with stylized names + icons, centered), account/usage at bottom. Split mirrors Signapse's "live vs file-based" product grouping.
- **Personalized greeting bar** ("Good morning" + date + today's session count) — cheap, fills the dead header strip.

**Stat tiles (4-up grid, each with a tiny chart) — kills the empty space**
- **Total signs translated** + 14-day **line sparkline**.
- **Avg recognition confidence** % + sparkline.
- **Session minutes** + quota **progress bar**.
- **Signs per session** **column sparkline**.
- Implement with **zero-dependency inline SVG sparklines + CSS bar charts**; reserve **Chart.js (~14 KB)** only for one interactive accuracy-over-time chart. [sparklines](https://www.domo.com/learn/charts/sparkline-chart), [Chart.js bundle](https://qrvey.com/blog/js-chart-library/)

**Mid-section**
- **Recent activity feed** across all 5 products (icon, snippet, confidence chip, timestamp, **replay** button) — fills the largest empty area and ties products together.
- **Avatar/model library grid** for Text→Sign / Speech→Sign (Signapse pattern: preview + "use").
- **Empty-state CTA cards** per product ("No sign-to-text sessions yet — start one") so a fresh dashboard is never blank.

**Live camera view (Sign→Text / Sign→Speech) — feature ideas you asked me to brainstorm**
- **Live confidence meter** (animated radial/bar) beside the feed. [Material Design live camera](https://m2.material.io/design/machine-learning/object-detection-live-camera.html)
- **Detection-state pill**: "Looking… → Detecting → Confirmed: HELLO" so a low-confidence feed isn't a dead screen.
- **Rolling detection-history strip**: last ~10 signs as confidence chips.
- **User confidence-threshold slider** (gates committed detections — directly exposes the 0.80/0.65 tradeoff to the user).
- **Live session-stats panel**: signs this session, duration, avg confidence sparkline.
- **Compose/replay/export clip cards**: saved, replayable, downloadable outputs (Signapse export pattern).
- Extra UX: hold-to-correct a misread sign, copy-transcript button, autospeak toggle, per-sign TTS replay (Sign→Speech), language auto-detect, keyboard shortcuts legend.

---

## PART 3 — How to test autonomously (what to instrument)

1. **Render Metrics dashboard** — watch CPU/RAM + spin-down/restart events; confirm 512 MB pressure.
2. **Server timing logs** — wrap `/api/translate` with timers around (a) model-load, (b) inference per model, (c) total. Confirms whether time is load vs compute.
3. **Chrome DevTools → Network** — measure each `/api/translate` round-trip (TTFB vs total), **count POSTs per detection** (confirms the 2–4× multiplier). Throttle to "Slow 4G" for real-user feel.
4. **`performance.now()` in browser** around MediaPipe to isolate landmark-extraction cost from network.
5. **Cold-start test** — idle the free instance 15+ min, time first request end-to-end; repeat after upgrading to quantify the win.

---

## PART 4 — What needs money / linking (your explicit ask)

| Item | Needed? | Cost | Why |
|---|---|---|---|
| **Render Starter** | **Strongly recommended** | ~$7/mo | Kills 30–60 s cold start; keeps models warm. Biggest speed win. |
| Render Standard | Only if RAM-pressured | ~$25/mo | 2 GB RAM / 1 vCPU for faster load + concurrency. |
| **Gemini API key** | **Required for compose** | Free tier available | Sentence composition + TTS/STT. Set `GEMINI_API_KEY` on Render. |
| Retraining / dataset work | **Not needed for speed** | — | Slowness is deployment/architecture, not the weights. |
| External training provider | **Not needed now** | — | Only relevant if you later want *accuracy* gains, which is a separate effort. |

If you want me to test the live model autonomously, I'd need either (a) the deployed Render URL + a test account, or (b) `GEMINI_API_KEY` + ability to run the server here. Tell me which and I'll add timing instrumentation and benchmark it.

---

## Suggested execution order
1. **Quick frontend fixes (no cost):** log text color, sidebar toggle placement + single logo, product names, compose-sentence logging + Enter-in-Arabic + error feedback, real cross-module sign counter.
2. **Server speed (code, no cost):** pre-load models at startup, batch vote into one POST, set thread counts.
3. **You do (cost):** upgrade Render to Starter, set `GEMINI_API_KEY`.
4. **Design overhaul (no cost):** stat tiles + sparklines, recent-activity feed, live-camera widgets, empty-states.
5. **Optional/bigger:** evaluate Holistic→HandLandmarker (needs model input check).
