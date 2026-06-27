# Chapter 2 — Review of Literature

This chapter reviews the systems that already attempt to translate between signed
and spoken language — the direct competitors to *Together* — and analyses each by
the **dataset** it is built on, the **deep-learning architecture** it employs, and
the **limitation** that follows. It first establishes the technical background
needed to make that analysis precise, then examines five representative
applications in detail, and finally compares them with *Together* to show where the
present work advances the state of practice.

## 2.1 Technical Background

Sign-language translation decomposes into two directions, and the systems reviewed
later differ chiefly in which they implement and how. The **sign-to-spoken**
direction begins with recognition. Research benchmarks such as the Word-Level ASL
dataset have shown that recognition models split into appearance-based networks —
two-dimensional CNNs with recurrent aggregation, or three-dimensional CNNs such as
I3D — and pose-based networks that operate on extracted skeleton keypoints, of
which graph convolutional networks are the strongest; on a 2,000-sign vocabulary
both families remain below ~35% top-1 accuracy, but pose-based models match
appearance models at far lower cost [10]. The skeleton-aware graph approach later
won the signer-independent recognition challenge with around 98% accuracy on a
smaller vocabulary, confirming that keypoint models generalise to unseen signers
when training data is diverse [11].

The **spoken-to-sign** direction, and full translation, was formalised by Camgöz et
al., who paired a CNN visual encoder with an attention-based recurrent
encoder–decoder and used gloss as an intermediate representation [8], later
replacing it with a transformer trained jointly for recognition and translation via
a CTC loss, which roughly doubled quality on the German PHOENIX14T corpus [9].
These models are powerful but depend on large parallel corpora of continuous
signing — a resource that does not exist for Arabic Sign Language, whose datasets
are small, recorded from few signers, and often limited to the alphabet (Table
2.1) [12], [13], [14]. Finally, all of these pipelines terminate in gloss, which is
not a grammatical sentence: sign languages use space, non-manual markers, and a
Topic–Comment order that differ from spoken syntax [6], [7], so fluent output
requires a dedicated language-generation stage. These three facts — the cost of
recognition, the corpus dependence of translation, and the grammar gap — are the
lenses through which the following systems are assessed.

**Table 2.1 — Representative public datasets for sign-language recognition.**

| Dataset | Language | Level | Size | Signers | Modality |
| --- | --- | --- | --- | --- | --- |
| WLASL [10] | ASL | Isolated (word) | ~21,000 videos, 2,000 glosses | 100+ | RGB video |
| KArSL [12] | ArSL | Isolated (word) | 502 signs, 75,300 samples | 3 | RGB, depth, skeleton |
| ArASL / ArSL2018 [13] | ArSL | Alphabet (static) | 54,049 images, 32 classes | 40 | Grayscale image |

## 2.2 Existing Applications and Systems

Five deployed systems best represent the current state of practice. Because they
are commercial products, their datasets and architectures are only partially
disclosed; where a detail is reported rather than published it is described as
such, and the absence of an open dataset is itself treated as a limitation.

### 2.2.1 SignAll

SignAll translates American Sign Language into English text at kiosks and in
educational tools, and is one of the few systems to attempt sign-to-text rather
than the easier reverse direction.

- **Dataset.** A large proprietary corpus of ASL recorded from native Deaf
  signers, built in-house to train the recogniser; it is not publicly available.
- **Architecture.** A multi-camera rig (typically three or more cameras) combined
  with colour-marked gloves for precise finger tracking; computer-vision modules
  perform hand-shape and motion recognition, and a rule-based linguistic component
  maps the recognised manual and non-manual features to English grammar.
- **Limitation.** The dependence on multiple calibrated cameras and gloves makes it
  non-portable and impossible to run in a browser; it is unidirectional (sign →
  text) and ASL-only, and its proprietary dataset prevents reproduction.

### 2.2.2 Hand Talk

Hand Talk is a widely adopted mobile application and website plug-in that converts
written and spoken language into sign through a three-dimensional avatar (Hugo),
for ASL and Brazilian Sign Language (Libras).

- **Dataset.** Rather than a recognition dataset, it relies on a curated dictionary
  of signs animated by human translators, paired with a text-processing engine; the
  translation component is reported to combine linguistic rules with machine
  learning.
- **Architecture.** A natural-language layer maps input text to a gloss sequence,
  which a 3D character-animation engine renders by playing and blending pre-built
  sign animations.
- **Limitation.** It is strictly one-directional (spoken → sign) and cannot
  interpret a user's own signing; output is bounded by the curated animation
  vocabulary, and avatar comprehensibility is a known constraint for fluent signers.

### 2.2.3 SLAIT

SLAIT demonstrated real-time American Sign Language recognition running directly in
a web browser from an ordinary webcam, and is the closest competitor in deployment
model to *Together*.

- **Dataset.** A self-collected set of recorded signs; public demonstrations
  covered only a limited vocabulary of words and phrases.
- **Architecture.** In-browser landmark/keypoint extraction (MediaPipe-style)
  followed by a deep sequence model — a recurrent or transformer network over the
  keypoint stream — producing text in real time.
- **Limitation.** Restricted vocabulary, ASL-only, and unidirectional (sign →
  text); as an early-stage start-up product its availability and robustness in the
  wild are uncertain.

### 2.2.4 KinTrans

KinTrans converts signing into text and voice for enterprise settings such as
service counters, and is notable for supporting Arabic Sign Language alongside ASL.

- **Dataset.** A large proprietary multi-signer corpus recorded by the company,
  reportedly spanning thousands of signs and including Arabic content; not public.
- **Architecture.** A three-dimensional depth camera captures skeletal joint
  positions, and a machine-learning classifier recognises signs from the resulting
  body- and hand-joint motion sequences.
- **Limitation.** It requires dedicated 3D-camera hardware and a fixed
  installation, so it is neither mobile nor browser-based; it is unidirectional
  (sign → spoken) and its data is closed.

### 2.2.5 Signapse

Signapse generates photorealistic sign-language video from text for fixed contexts
such as transport announcements and website content, in British and American Sign
Language.

- **Dataset.** A proprietary corpus of professionally recorded signer footage used
  to train its generative models.
- **Architecture.** Deep generative models (GAN-based video synthesis) produce
  continuous, photorealistic signing; a text-to-gloss stage drives which signs are
  generated and stitched.
- **Limitation.** It is one-directional (spoken → sign), oriented to pre-rendered,
  domain-specific announcements rather than open conversation, and does not
  interpret a user's signing.

## 2.3 Comparative Analysis and Positioning of *Together*

Table 2.2 places the five systems and *Together* side by side along the dimensions
that matter most for accessible, everyday communication.

**Table 2.2 — Comparison of existing systems with *Together*.**

| System | Direction | Languages | Dataset | Architecture | Key limitation |
| --- | --- | --- | --- | --- | --- |
| SignAll [17] | Sign → Spoken | ASL | Proprietary ASL corpus | Multi-camera CV + marker gloves + rule-based NLP | Needs camera rig + gloves; not portable |
| Hand Talk [15] | Spoken → Sign | ASL, Libras | Curated animation dictionary | Text→gloss + 3D avatar | One-directional; fixed vocabulary |
| SLAIT [19] | Sign → Spoken | ASL | Self-collected (small) | Webcam landmarks + deep sequence model | Limited vocabulary; one-directional |
| KinTrans [18] | Sign → Spoken | ASL, ArSL | Proprietary multi-signer corpus | 3D-camera skeleton + ML classifier | Needs 3D-camera hardware; one-directional |
| Signapse [16] | Spoken → Sign | BSL, ASL | Proprietary signer video | Deep generative (GAN) video | One-directional; domain-limited |
| **Together (this work)** | **Bidirectional** | **ASL, ArSL** | **Public landmark datasets (detailed in Ch. 3)** | **Landmarks + isolated models + LLM gloss→sentence + semantic sign synthesis** | **Isolated (not continuous) signing** |

Three differences set *Together* apart, each addressing a gap left by the systems
above.

**It is bidirectional.** Every reviewed product commits to a single direction:
SignAll, SLAIT, and KinTrans interpret signing but cannot produce it, while Hand
Talk and Signapse produce signing but cannot interpret it. *Together* implements
both directions and, in its meeting mode, runs them concurrently so a signer and a
speaker can actually converse — something none of the five supports.

**It is bilingual across a high- and a low-resource language.** Only KinTrans
also covers Arabic, and only in the sign-to-spoken direction on closed hardware.
*Together* treats Arabic Sign Language as a first-class language in *both*
directions, directly confronting the data-scarcity problem documented in §2.1
rather than avoiding it.

**It is hardware-free and reproducible.** SignAll needs a calibrated multi-camera
rig and gloves, and KinTrans a depth camera; only SLAIT shares *Together*'s
browser-and-webcam model, and only in one direction. *Together* runs entirely in an
ordinary browser and is built on **public** datasets and an LLM-based gloss-to-text
stage, making it both deployable without special equipment and, unlike the
proprietary products, reproducible.

The trade-off is honest: *Together* recognises **isolated** signs rather than
continuous signing, so it relies on a buffer-and-language-model stage to assemble
sentences. But this is precisely the design that removes the corpus dependence
which makes the academic translation models of §2.1 infeasible for Arabic — and it
is what lets a single system be bidirectional, bilingual, and browser-based at once.

## 2.4 Summary and Research Gap

The evidence is consistent. Recognition has matured around efficient pose- and
graph-based models [10], [11]; the strongest translation models are corpus-hungry
transformers available only for high-resource languages [8], [9]; Arabic Sign
Language is held back by small, low-signer datasets [12], [13], [14]; and the
deep-learning pipeline stops at gloss, leaving the grammar gap to a separate stage
[6], [7]. The deployed products that build on these foundations are, as Table 2.2
shows, uniformly single-direction, narrow in language, and frequently hardware- or
data-bound [15]–[19].

The gap is therefore clear and unfilled: **no available system is at once
bidirectional, bilingual across a high- and a low-resource sign language, and
deployable in an ordinary web browser.** *Together* targets exactly this gap with a
modular, gloss-mediated design — a lightweight landmark-based recogniser feeding a
general-purpose language model — that sidesteps the corpus dependence blocking the
academic approaches while surpassing the single-direction, hardware-bound limits of
the commercial ones. The design and implementation of this system are presented in
the chapters that follow.
