# Chapter 2 — Review of Literature

## 2.1 Technical Background

Sign-language translation decomposes into two directions, and the systems reviewed
later differ chiefly in which they implement and how. The **sign-to-spoken**
direction begins with recognition. Research benchmarks such as the Word-Level ASL
dataset have shown that recognition models split into appearance-based networks —
two-dimensional CNNs with recurrent aggregation, or three-dimensional CNNs such as
I3D — and pose-based networks that operate on extracted skeleton keypoints, of
which graph convolutional networks are the strongest; on a 2,000-sign vocabulary
both families remain below ~35% top-1 accuracy, but pose-based models match
appearance models at far lower cost [10]. Microsoft's MS-ASL provides a comparably
large American benchmark of 1,000 signs recorded from 222 signers in unconstrained
conditions [15], while the Turkish AUTSL corpus added depth and skeleton modalities
for 226 signs [16]. The skeleton-aware graph approach later won the
signer-independent recognition challenge on AUTSL with around 98% accuracy,
confirming that keypoint models generalise to unseen signers when training data is
diverse [11].

The **spoken-to-sign** direction, and full translation, was formalised by Camgöz et
al., who paired a CNN visual encoder with an attention-based recurrent
encoder–decoder and used gloss as an intermediate representation [8], later
replacing it with a transformer trained jointly for recognition and translation via
a CTC loss, which roughly doubled quality on the German PHOENIX14T corpus [9].
Comparable continuous corpora such as How2Sign offer many hours of aligned ASL
video and English text [17], but these resources exist only for high-resource
languages. They are powerful yet depend on large parallel corpora of continuous
signing — a resource that does not exist for Arabic Sign Language, whose datasets
are small, recorded from few signers, and often limited to the alphabet (Fig. 2.1)
[12], [13], [14]. Finally, all of these pipelines terminate in gloss, which is not a
grammatical sentence: sign languages use space, non-manual markers, and a
Topic–Comment order that differ from spoken syntax [6], [7], so fluent output
requires a dedicated language-generation stage. These three facts — the cost of
recognition, the corpus dependence of translation, and the grammar gap — are the
lenses through which the following systems are assessed.

| Dataset | Language | Level | Size | Signers | Modality |
| --- | --- | --- | --- | --- | --- |
| WLASL [10] | ASL | Isolated (word) | ~21,000 videos, 2,000 glosses | 100+ | RGB video |
| MS-ASL [15] | ASL | Isolated (word) | ~25,000 videos, 1,000 signs | 222 | RGB video |
| AUTSL [16] | Turkish SL | Isolated (word) | 38,336 samples, 226 signs | 43 | RGB, depth, skeleton |
| How2Sign [17] | ASL | Continuous (sentence) | 80+ hours | 11 | RGB, depth, pose, speech |
| PHOENIX14T [8] | German SL | Continuous (sentence) | 8,257 sentences, 1,066 glosses | 9 | RGB video |
| KArSL [12] | ArSL | Isolated (word) | 502 signs, 75,300 samples | 3 | RGB, depth, skeleton |
| ArASL / ArSL2018 [13] | ArSL | Alphabet (static) | 54,049 images, 32 classes | 40 | Grayscale image |

**Figure 2.1 — Comparison of different datasets.**

### 2.1.1 Deep-Learning Sequence Architectures: From CNNs to Transformers

A sign is not a single pose but a *trajectory through time*, so the recogniser must
model both spatial structure (the configuration of the hands and body in a frame)
and temporal structure (how that configuration evolves). Four families of
deep-learning architecture have been applied to this problem, each improving on the
last in how it handles temporal dependencies.

**Convolutional Neural Networks (CNNs).** A CNN applies learnable filters that slide
across the input, sharing weights so that the same local pattern is detected
anywhere. This makes CNNs extremely efficient at capturing *local* structure —
edges in an image, or short motion patterns in a one-dimensional (1-D) sequence of
landmarks. Their weakness for sign language is range: a single convolution sees only
its kernel-sized window, so capturing a dependency spanning the whole sign requires
stacking many layers or using dilation, and a plain CNN has no explicit notion of
sequence order beyond what the filters encode.

**Recurrent Neural Networks (RNNs).** An RNN processes a sequence step by step,
carrying a hidden state $h_t = f(h_{t-1}, x_t)$ that summarises everything seen so
far, which makes it a natural fit for temporal data. In practice, however, vanilla
RNNs suffer from vanishing and exploding gradients, so they struggle to learn
long-range dependencies, and because each step depends on the previous one they
cannot be parallelised across time, making them slow to train.

**LSTM and BiLSTM/BiGRU.** Long Short-Term Memory networks add a gated memory cell —
input, forget, and output gates — that lets information flow across many steps
without vanishing, capturing far longer dependencies than a plain RNN. The Gated
Recurrent Unit (GRU) is a lighter variant with only update and reset gates and fewer
parameters, which is advantageous on small datasets. Bidirectional variants
(BiLSTM, BiGRU) run two recurrences — one forward, one backward — and concatenate
their states, so each time step is informed by both past and future context, at
roughly twice the compute. These models remain, however, fundamentally *sequential*.

**Transformers.** The Transformer replaces recurrence entirely with *self-attention*.
For every position, attention compares a query against the keys of all other
positions and forms a weighted sum of their values (Eq. 4.5), so any two time steps
are connected by a single computation regardless of how far apart they are — the
path length for a long-range dependency is $O(1)$ rather than $O(n)$. Several
attention "heads" (Eq. 4.6) learn different relations in parallel, a position-wise
feed-forward network (Eq. 4.7) adds non-linear capacity, and because there is no
recurrence a positional encoding is added so the model knows the order of frames.
Crucially, all positions are processed *simultaneously*, so Transformers parallelise
well and scale to large data. Their costs are a quadratic $O(n^2)$ attention
computation and a weaker built-in sense of locality, which is why modern speech and
gesture models — including the Squeezeformer used here — interleave convolution
(for local detail) with attention (for global context). Table 2.2 summarises the
trade-offs.

**Table 2.2 — Comparison of deep-learning sequence architectures.**

| Architecture | Temporal mechanism | Long-range dependencies | Parallel over time | Key weakness |
| --- | --- | --- | --- | --- |
| CNN (1-D) | Local sliding filters | Limited (needs depth/dilation) | Yes | Fixed receptive field |
| RNN | Recurrent hidden state | Poor (vanishing gradients) | No | Slow; forgets long context |
| LSTM / GRU | Gated memory | Good | No | Sequential; still bounded |
| BiLSTM / BiGRU | Gated memory, both directions | Good (past + future) | No | ~2× compute; sequential |
| Transformer | Self-attention | Excellent ($O(1)$ path) | Yes | $O(n^2)$ cost; weak locality |

These trade-offs explain the two model choices in this project. The ASL recogniser,
trained on a large 250-class corpus, adopts a **Squeezeformer** that combines 1-D
convolution (local motion) with Transformer attention (global context), exploiting
both the abundance of data and the parallelism of attention. The ArSL recogniser,
trained on a much smaller 20-class dataset, instead uses a compact **CNN + BiGRU**:
the bidirectional recurrence captures temporal context in both directions while the
far smaller parameter count avoids the over-fitting that a data-hungry Transformer
would risk on limited Arabic data.

## 2.2 Existing Applications and Systems

### 2.2.1 SignAll

SignAll [20] translates American Sign Language into English text at kiosks and in
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

Hand Talk [18] is a widely adopted mobile application and website plug-in that converts
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

SLAIT [22] demonstrated real-time American Sign Language recognition running directly in
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

KinTrans [21] converts signing into text and voice for enterprise settings such as
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

Signapse [19] generates photorealistic sign-language video from text for fixed contexts
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

## 2.3 Comparison with Together

Three core differences set *Together* apart, directly addressing the critical gaps
left by existing systems:

- **It is fully bidirectional.** Every other product we reviewed commits to just
  one direction. Tools like SignAll, SLAIT, and KinTrans can interpret sign
  language but cannot produce it, while Hand Talk and Signapse generate signs but
  cannot interpret them. *Together* handles both. Furthermore, its live meeting mode
  runs these processes concurrently, empowering a signer and a speaker to have a
  genuine, back-and-forth conversation — a feature none of the other five systems
  offer.
- **It champions both high- and low-resource languages.** While KinTrans does
  include Arabic, it only translates from sign to spoken language and requires
  closed hardware. *Together* elevates Arabic Sign Language (ArSL) to a first-class
  language alongside ASL, seamlessly translating it in both directions. Rather than
  sidestepping the data-scarcity problem documented in Fig. 2.1, *Together* tackles
  it head-on.
- **It is hardware-free and reproducible.** SignAll demands a calibrated
  multi-camera rig and data gloves, and KinTrans requires a depth camera. While
  SLAIT shares our browser-and-webcam model, it only works in one direction.
  *Together* runs effortlessly in an ordinary web browser. Because it relies on
  public datasets and an LLM-driven gloss-to-text pipeline, it is readily deployable
  without specialized equipment and — unlike proprietary products — fully
  reproducible.

## 2.4 Comparative Analysis

| System | Direction | Languages | Dataset | Architecture | Key limitation |
| --- | --- | --- | --- | --- | --- |
| SignAll [20] | Sign → Spoken | ASL | Proprietary ASL corpus | Multi-camera CV + marker gloves + rule-based NLP | Needs camera rig + gloves; not portable |
| Hand Talk [18] | Spoken → Sign | ASL, Libras | Curated animation dictionary | Text→gloss + 3D avatar | One-directional; fixed vocabulary |
| SLAIT [22] | Sign → Spoken | ASL | Self-collected (small) | Webcam landmarks + deep sequence model | Limited vocabulary; one-directional |
| KinTrans [21] | Sign → Spoken | ASL, ArSL | Proprietary multi-signer corpus | 3D-camera skeleton + ML classifier | Needs 3D-camera hardware; one-directional |
| Signapse [19] | Spoken → Sign | BSL, ASL | Proprietary signer video | Deep generative (GAN) video | One-directional; domain-limited |
| **Together (this work)** | **Bidirectional** | **ASL, ArSL** | **Public datasets** | **Landmarks + isolated models + LLM gloss→sentence** | **Isolated (not continuous) signing** |
