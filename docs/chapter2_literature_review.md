# Chapter 2 — Review of Literature

Whereas the previous chapter introduced the problem in general terms, this
chapter reviews the work most relevant to the design of *Together*. It focuses on
the deep-learning models that have been applied to sign-language recognition and
translation, the datasets that those models are trained on, and the limitations
that follow from both. It then surveys the systems already deployed for the same
purpose and, finally, synthesises these threads into the research gap that this
project addresses.

## 2.1 Deep-Learning Models for Sign Language Recognition

Sign-language recognition has followed the same architectural trajectory as the
wider field of video understanding. Early systems combined hand-crafted features
with hidden Markov models, but these have been almost entirely superseded by deep
neural networks, which fall into three broad families.

The first family operates on **raw video**. Two-dimensional convolutional
networks extract per-frame appearance features that are then aggregated over time
by recurrent layers such as long short-term memory (LSTM) or gated recurrent
units (GRU), while three-dimensional convolutional networks such as I3D learn
spatio-temporal features directly from the video volume. On the Word-Level ASL
benchmark, such appearance-based models — including an I3D variant — form the
strongest video baselines but are computationally heavy [10].

The second family operates on **pose or skeleton** data, in which the body,
hands, and face are first reduced to a set of keypoints and only those keypoints
are modelled. Because the input is far smaller than a video frame, these models
are markedly more efficient. Graph convolutional networks (GCNs), which treat the
skeleton as a graph of joints, have proven especially effective: the skeleton-aware
multi-modal approach of Jiang et al. built a GCN over sign keypoints and combined
it with complementary streams to achieve state-of-the-art isolated recognition,
demonstrating that pose alone carries most of the discriminative signal [11].
Pose-based baselines on the same word-level benchmark confirm that keypoint models
rival appearance models at a fraction of the cost [10].

The third and most recent family applies **transformer** architectures, whose
self-attention mechanism captures long-range temporal dependencies without
recurrence and now underpins much of the state of the art in both recognition and
translation. The clear trend across all three families is a move from heavy
appearance-based recurrent models toward lighter pose-based and attention-based
ones — a trend this project follows by classifying signs from extracted landmarks
rather than from raw frames.

## 2.2 Deep-Learning Models for Sign Language Translation

Recognition produces *gloss* — a sequence of sign labels — whereas translation
must produce a grammatical sentence in the target spoken language. The task was
formalised as a neural sequence-to-sequence problem by Camgöz et al., who paired a
convolutional visual encoder with an attention-based recurrent encoder–decoder and
used gloss as an intermediate supervision signal between video and text [8]. This
established the two paradigms that still define the field: **gloss-based**
(two-stage) translation, which recognises signs into gloss and then translates the
gloss, and **end-to-end** translation, which maps signing directly to text.

Subsequent work replaced the recurrent backbone with transformers, jointly
learning recognition and translation in a single network and improving accuracy,
while confirming that gloss-level information remains a strong intermediate signal
even for powerful models [9]. Two points are decisive for the present work. First,
these are large supervised sequence-to-sequence models. Second, they are trained
on parallel corpora of continuous signing aligned to text — resources that exist
for only a handful of high-resource languages, and not for Arabic Sign Language.

## 2.3 Datasets for Sign Language Recognition

Because deep models are data-driven, the available datasets shape what is
achievable, and here the asymmetry between languages is stark.

For American Sign Language, the Word-Level ASL dataset provides on the order of
two thousand glosses performed by more than one hundred signers, collected from
web video, and has become the standard isolated-recognition benchmark [10]. Its
scale and signer diversity make signer-independent evaluation meaningful, but it
remains an *isolated*-sign resource rather than a continuous, sentence-level one.

For Arabic Sign Language the picture is far thinner. The ArASL dataset consists of
tens of thousands of static images covering only the manual alphabet, supporting
letter recognition but not word- or sentence-level translation [13]. The KArSL
database is larger in vocabulary, offering several hundred isolated signs captured
in multiple modalities, but it is recorded from only a small number of signers,
which limits signer-independent generalisation [12]. Broader surveys confirm that
Arabic sign-language research is fragmented across image- and sensor-based methods
and lacks the large, standardised benchmarks available for other languages [14].
This scarcity is the central practical obstacle that any Arabic sign-language
system must confront.

Table 2.2 summarises the principal datasets discussed above, highlighting the
gap in scale, signer diversity, and linguistic level between the resources
available for American and Arabic Sign Language.

**Table 2.2 — Representative public datasets for sign-language recognition.**

| Dataset | Language | Level | Size | Signers | Modality |
| --- | --- | --- | --- | --- | --- |
| WLASL [10] | ASL | Isolated (word) | ~21,000 videos, 2,000 glosses | 100+ | RGB video |
| KArSL [12] | ArSL | Isolated (word) | 502 signs, 75,300 samples | 3 | RGB, depth, skeleton |
| ArASL / ArSL2018 [13] | ArSL | Alphabet (static) | 54,049 images, 32 classes | 40 | Grayscale image |

Two contrasts stand out. The American resource offers an order of magnitude more
vocabulary and far greater signer diversity than either Arabic resource, which
makes signer-independent evaluation feasible for ASL but difficult for ArSL.
Moreover, the larger Arabic corpus (KArSL) is recorded from only three signers,
and the most accessible one (ArASL) covers only the static alphabet rather than
words — neither supports sentence-level translation. The datasets actually used to
train the models in this work are described in Chapter 3.

## 2.4 Limitations of Existing Deep-Learning Approaches

Taken together, the models and datasets reviewed above expose four limitations
that directly motivate the design choices in this thesis.

**Data dependence.** End-to-end and large gloss-based translation models require
substantial parallel corpora of continuous signing aligned to text [8], [9]. Such
corpora do not exist for Arabic Sign Language, making the most powerful published
architectures infeasible to reproduce for a low-resource language.

**Cost and deployment.** The most accurate appearance-based recognisers rely on
3D convolutions or deep recurrent stacks that are expensive to run, which is at
odds with real-time operation in a browser; lighter pose-based models are a more
practical foundation [10], [11].

**Generalisation to unseen signers.** Models trained on few signers tend to
overfit to their appearance and articulation, so accuracy on previously unseen
signers can drop sharply. This risk is greatest precisely where data is scarce,
as with Arabic Sign Language, and it makes the distinction between
signer-dependent and signer-independent evaluation essential.

**The grammar gap.** Recognition architectures emit gloss, but gloss is not a
grammatical sentence. Sign languages organise meaning through space, non-manual
markers, and a Topic–Comment order that differ systematically from spoken-language
syntax [6], [7]. Bridging this gap requires a dedicated language-generation stage
capable of reordering tokens and inserting the function words and morphology that
signing conveys non-manually — a capability that recognition models do not
provide.

These limitations point away from a single end-to-end network and toward a
modular design: a lightweight pose-based recogniser, followed by a general-purpose
language model that supplies the missing grammar without requiring a parallel
signing corpus for the target language.

## 2.5 Existing Systems and Commercial Solutions

Beyond academic prototypes, several commercial products and deployed applications
attempt to bridge signed and spoken communication. They fall into two broad
groups that mirror the two translation directions, and each adopts a distinct
architecture.

The first group translates **spoken language into sign** through an animated
avatar or synthesised video. Hand Talk is a widely used mobile application and
website plug-in that renders text and speech as a three-dimensional signing
avatar for American Sign Language and Brazilian Sign Language (Libras) [15].
Signapse follows a similar direction but generates photorealistic, AI-synthesised
sign-language video for fixed announcements and websites, primarily in British and
American Sign Language [16]. Both are essentially one-directional: they produce
sign output but do not interpret a user's own signing.

The second group translates **sign into spoken language**. SignAll uses a
multi-camera rig, historically combined with colour-marked gloves, to convert ASL
into English text at kiosks and desktop stations [17]. KinTrans applies
three-dimensional cameras and machine learning to convert signing into text and
speech, and is notable for supporting Arabic Sign Language alongside ASL, although
it is delivered as fixed enterprise hardware [18]. SLAIT demonstrated real-time,
webcam-based ASL-to-text recognition running directly in a browser, showing the
feasibility of hardware-free deployment, albeit again in a single direction [19].

Table 2.1 compares these systems across the dimensions most relevant to this
project: translation direction, supported languages, core technology, deployment
model, and real-time operation, with *Together* included for reference.

**Table 2.1 — Comparison of existing sign-language systems with *Together*.**

| System | Direction | Languages | Core technology / architecture | Deployment | Real-time |
| --- | --- | --- | --- | --- | --- |
| Hand Talk [15] | Spoken → Sign | ASL, Libras | 3D signing avatar | Mobile app, web plug-in | Yes |
| Signapse [16] | Spoken → Sign | BSL, ASL | AI-synthesised photorealistic sign video | Web / digital signage | Pre-rendered |
| SignAll [17] | Sign → Spoken (text) | ASL | Multi-camera vision (+ marker gloves) | Kiosk / desktop | Yes |
| KinTrans [18] | Sign → Spoken (text + voice) | ASL, ArSL | 3D camera + machine learning | Kiosk / enterprise hardware | Yes |
| SLAIT [19] | Sign → Spoken (text) | ASL | Webcam + deep learning | Browser | Yes |
| **Together (this work)** | **Bidirectional** | **ASL, ArSL** | **Landmarks + isolated models + LLM gloss→sentence + semantic sign synthesis** | **Browser (no install)** | **Yes** |

Across the surveyed products, three patterns are clear. Each system commits to a
single translation direction; coverage is concentrated on a few high-resource
languages, with Arabic supported by only one of them; and the systems offering the
richest functionality rely on dedicated hardware or controlled installations
rather than an ordinary browser. No single product combines bidirectional
translation, support for both a high- and a low-resource sign language, and
zero-install browser deployment.

## 2.6 Summary and Research Gap

The reviewed models, datasets, and systems support a consistent set of
conclusions. Recognition has matured around efficient pose- and graph-based models
[10], [11]; translation is reliably framed as a gloss-mediated sequence-to-sequence
problem, but its strongest forms depend on large parallel corpora that exist only
for high-resource languages [8], [9]; Arabic Sign Language is constrained by small,
mostly isolated or alphabet-level datasets recorded from few signers [12], [13],
[14]; and the deep-learning pipeline ends at gloss, leaving the grammar gap to be
closed by a separate stage [6], [7]. The deployed commercial systems, as Table 2.1
shows, are uniformly single-direction, narrow in language coverage, and frequently
hardware-bound [15]–[19].

From these observations, a clear gap emerges. No widely available system is
simultaneously *bidirectional*, *bilingual* across a high-resource and a
low-resource sign language, and *deployable in an ordinary web browser*. The
corpus dependence of end-to-end models makes them ill-suited to Arabic Sign
Language, which argues instead for a modular, gloss-mediated design in which a
lightweight isolated recogniser feeds a general-purpose language model that
supplies the missing grammatical structure — without requiring a large parallel
signing corpus for the target language. The remainder of this thesis develops
exactly such a system, the design and implementation of which are presented in the
chapters that follow.
