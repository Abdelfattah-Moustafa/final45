# Chapter 2 — Review of Literature

This chapter reviews the work most relevant to the design of *Together* from a
technical standpoint. For each strand it examines the **datasets** used, the
**input representations** chosen, and the **deep-learning architectures** applied,
together with the results those choices produced and the limitations they exposed.
It first treats sign-language recognition, then translation, then the specific
situation of Arabic Sign Language, before drawing out the architectural
limitations, surveying deployed commercial systems, and synthesising the research
gap that this project addresses.

## 2.1 Sign Language Recognition: Models and Architectures

Modern sign-language recognition is dominated by deep neural networks, and the
field divides cleanly by **input representation** into appearance-based models,
which consume raw video, and pose-based models, which consume extracted keypoints.

The most thorough comparison of these families is the Word-Level ASL (WLASL)
study, which benchmarked four architectures on the same data [10]. On the
appearance side it evaluated a two-dimensional CNN (VGG-16) whose per-frame
features are aggregated by a gated recurrent unit (GRU), and a three-dimensional
CNN (Inflated 3D ConvNet, I3D) that learns spatio-temporal features directly from
the RGB volume. On the pose side it evaluated a GRU over two-dimensional body
keypoints (Pose-GRU) and a pose-based temporal graph convolutional network
(Pose-TGCN) that treats the skeleton as a graph and convolves over both its
spatial structure and time. The results are instructive: on the largest,
2,000-gloss split the strongest appearance model (I3D) reached only about 32%
top-1 accuracy, while Pose-TGCN reached roughly 24% — comparable performance at a
fraction of the input size, but a steep drop from the ~65% achievable on the
easier 100-gloss split. This establishes two facts that shape the present work:
large-vocabulary isolated recognition is hard, and pose-based models are a
cost-effective alternative to heavy video models.

The pose-based direction matured further with the skeleton-aware multi-modal
framework (SAM-SLR), which won the CVPR 2021 signer-independent isolated-recognition
challenge on the Turkish AUTSL dataset [11]. Its core is a Sign Language Graph
Convolution Network (SL-GCN) operating on whole-body skeleton keypoints
(hands, body, and face), complemented by a Separable Spatial-Temporal Convolution
Network (SSTCN) over skeleton features, with RGB and depth streams ensembled for
additional global context. The framework reached 98.42% accuracy on the RGB track
and 98.53% on RGB-D — and, crucially, did so under a **signer-independent**
protocol, demonstrating that skeleton-based graph models can generalise to unseen
signers when sufficient signer diversity is available in training. This both
validates a landmark-based front-end and underscores why signer-independent
evaluation is the meaningful benchmark.

## 2.2 Sign Language Translation: Models and Architectures

Recognition yields gloss; translation must yield fluent text, and the canonical
work here is by Camgöz et al. The 2018 study introduced the first public
translation corpus, RWTH-PHOENIX-Weather 2014T — German Sign Language weather
broadcasts with aligned gloss and German text — and cast translation as neural
machine translation [8]. Architecturally it embedded each frame with a CNN
(GoogLeNet), then fed those embeddings to an attention-based recurrent
encoder–decoder, using gloss as an intermediate supervision signal. The reported
scores expose the difficulty of the end-to-end problem: translating directly from
sign video to text (Sign2Text) reached only 9.58 BLEU-4, whereas translating from
ground-truth gloss to text (Gloss2Text) reached roughly 19 BLEU-4. The gap is the
key insight — most of the translation difficulty lies in recognition, and a
reliable gloss intermediate makes the downstream language generation far easier.

The 2020 follow-up replaced the recurrent backbone with a transformer and unified
recognition and translation in one network, attaching a Connectionist Temporal
Classification (CTC) loss to the encoder so that gloss recognition and text
generation are learned jointly [9]. This roughly **doubled** translation quality
on the same corpus, from 9.58 to about 21.3 BLEU-4. Two conclusions follow for
this project. First, the gloss-mediated, two-stage formulation remains the most
reliable design. Second — and decisively — these are large supervised models
trained on a sizeable parallel corpus of *continuous* signing aligned to text, a
resource that simply does not exist for Arabic Sign Language. Table 2.1 summarises
the representative recognition and translation works by dataset, input, and
architecture.

**Table 2.1 — Representative academic works by dataset, input, and architecture.**

| Work | Task | Dataset | Input representation | DL architecture | Reported result |
| --- | --- | --- | --- | --- | --- |
| Li et al., WLASL [10] | Isolated recognition | WLASL2000 (ASL) | RGB video; 2D pose | I3D (3D-CNN); Pose-TGCN (graph conv) | ~32% / ~24% top-1 |
| Jiang et al., SAM-SLR [11] | Isolated recognition (signer-independent) | AUTSL (Turkish) | Whole-body skeleton + RGB/depth | SL-GCN + SSTCN + multimodal ensemble | 98.4% (RGB) |
| Camgöz et al., NSLT [8] | Continuous translation | PHOENIX14T (German) | CNN frame features | CNN + attention RNN seq2seq | 9.58 BLEU-4 (Sign2Text) |
| Camgöz et al., SL-Transformers [9] | Recognition + translation | PHOENIX14T (German) | CNN spatial embeddings | Transformer enc–dec + CTC | ~21.3 BLEU-4 |

## 2.3 Arabic Sign Language: Datasets and Approaches

Against this backdrop, Arabic Sign Language (ArSL) is conspicuously
under-resourced, and the constraint is fundamentally one of data. The most
substantial isolated-word corpus, KArSL, contains 502 signs captured with a
Microsoft Kinect V2 in three modalities — RGB, depth, and skeleton joints — but is
performed by only **three** signers, yielding 75,300 samples [12]. The accompanying
benchmarks report strong accuracy when the same signers appear in training and
testing, but a marked drop under signer-independent evaluation — exactly the
generalisation problem expected when signer diversity is so low. The most widely
used Arabic resource, ArSL2018 (ArASL), is larger in samples but far narrower in
scope: 54,049 grayscale images spanning only the 32 letters of the manual
alphabet, collected from 40 participants, and therefore suited to static
image-classification CNNs rather than to word- or sentence-level translation [13].
Broader surveys of the field confirm the pattern, documenting a fragmented mix of
image-based and sensor-based (instrumented-glove) approaches and the absence of
the large, standardised benchmarks available for other languages [14].

Table 2.2 contrasts these resources with the American benchmark from §2.1.

**Table 2.2 — Representative public datasets for sign-language recognition.**

| Dataset | Language | Level | Size | Signers | Modality |
| --- | --- | --- | --- | --- | --- |
| WLASL [10] | ASL | Isolated (word) | ~21,000 videos, 2,000 glosses | 100+ | RGB video |
| KArSL [12] | ArSL | Isolated (word) | 502 signs, 75,300 samples | 3 | RGB, depth, skeleton |
| ArASL / ArSL2018 [13] | ArSL | Alphabet (static) | 54,049 images, 32 classes | 40 | Grayscale image |

The contrast is stark: ASL offers an order of magnitude more vocabulary and far
greater signer diversity, making signer-independent evaluation feasible, whereas
the larger Arabic corpus is limited to three signers and the most accessible one
covers only the alphabet. Neither Arabic resource supports sentence-level
translation. The datasets actually used to train the models in this work are
described in Chapter 3.

## 2.4 Limitations of Existing Deep-Learning Approaches

Taken together, the architectures and datasets above expose four limitations that
directly motivate the design choices in this thesis.

**Corpus dependence.** State-of-the-art translation models such as the
transformer of [9] require large parallel corpora of continuous signing aligned to
text [8]. No such corpus exists for Arabic Sign Language, making these
architectures impossible to reproduce for a low-resource language and arguing for a
design that does not depend on parallel signing data.

**Computational cost.** The most accurate appearance-based recognisers rely on 3D
convolutions or deep recurrent stacks [10], which are costly to run and ill-suited
to real-time, in-browser operation. Pose- and graph-based models offer comparable
accuracy at far lower cost [10], [11], making a landmark front-end the practical
choice.

**Generalisation to unseen signers.** Models trained on few signers overfit to
their appearance and articulation, as the signer-independent drop on KArSL shows
[12]; conversely, the signer diversity behind SAM-SLR's results is what enables its
generalisation [11]. Where data is scarce, the distinction between
signer-dependent and signer-independent evaluation is therefore essential.

**The grammar gap.** Every architecture above terminates in gloss or in a
language-specific decoder trained on a particular corpus. Gloss itself is not a
grammatical sentence: sign languages encode meaning through space, non-manual
markers, and a Topic–Comment order that differs systematically from spoken-language
syntax [6], [7]. Closing this gap requires a dedicated language-generation stage
that can reorder tokens and supply the function words and morphology that signing
conveys non-manually — a capability the recognition pipeline does not provide.

These limitations point away from a single end-to-end network and toward a modular
design: a lightweight pose-based recogniser feeding a general-purpose language
model that supplies the missing grammar without requiring a parallel signing corpus
for the target language.

## 2.5 Existing Systems and Commercial Solutions

Beyond academic prototypes, several deployed products tackle the same problem, and
their architectures fall into two groups mirroring the two translation directions.

The first group translates **spoken language into sign** and is built around
generation rather than recognition. Hand Talk maps written or spoken input to a
gloss sequence and renders it with a rule-driven three-dimensional signing avatar,
covering American Sign Language and Brazilian Sign Language (Libras) [15]. Signapse
takes a data-driven route, using deep generative models trained on recorded signer
footage to synthesise photorealistic sign-language video for fixed announcements
and websites, primarily in British and American Sign Language [16]. Both are
one-directional: they produce sign output but do not interpret a user's signing.

The second group translates **sign into spoken language** and therefore centres on
a recognition model. SignAll combines a multi-camera rig with colour-marked gloves
for precise hand tracking, feeding a vision pipeline and a rule-based linguistic
module that maps recognised components to English text [17]. KinTrans captures
signing with a three-dimensional (depth) camera and classifies the resulting
skeletal motion with a machine-learning model trained on its own multi-signer data,
notably covering Arabic Sign Language alongside ASL, but delivered as fixed
enterprise hardware [18]. SLAIT is the closest in spirit to this work, performing
real-time, webcam-based ASL-to-text recognition in the browser from extracted
landmarks, demonstrating hardware-free deployment, though again in a single
direction [19].

Table 2.3 compares these systems across the dimensions most relevant to this
project, with *Together* included for reference.

**Table 2.3 — Comparison of existing sign-language systems with *Together*.**

| System | Direction | Languages | Core technology / architecture | Deployment | Real-time |
| --- | --- | --- | --- | --- | --- |
| Hand Talk [15] | Spoken → Sign | ASL, Libras | Text→gloss + rule-driven 3D avatar | Mobile app, web plug-in | Yes |
| Signapse [16] | Spoken → Sign | BSL, ASL | Deep generative (photorealistic) sign video | Web / digital signage | Pre-rendered |
| SignAll [17] | Sign → Spoken (text) | ASL | Multi-camera CV + marker gloves + rule-based NLP | Kiosk / desktop | Yes |
| KinTrans [18] | Sign → Spoken (text + voice) | ASL, ArSL | 3D-camera skeletal capture + ML classifier | Kiosk / enterprise hardware | Yes |
| SLAIT [19] | Sign → Spoken (text) | ASL | Webcam landmarks + deep sequence model | Browser | Yes |
| **Together (this work)** | **Bidirectional** | **ASL, ArSL** | **Landmarks + isolated models + LLM gloss→sentence + semantic sign synthesis** | **Browser (no install)** | **Yes** |

Across these products, three patterns hold: each commits to a single direction;
language coverage is narrow, with Arabic supported by only one of them; and the
richest functionality depends on dedicated hardware or controlled installations.
No single product combines bidirectional translation, support for both a
high- and a low-resource sign language, and zero-install browser deployment.

## 2.6 Summary and Research Gap

The reviewed models, datasets, and systems support a consistent set of
conclusions. Recognition has converged on efficient pose- and graph-based
architectures that can generalise to unseen signers given enough signer diversity
[10], [11]; translation is reliably framed as a gloss-mediated sequence-to-sequence
problem whose strongest transformer-based forms depend on large parallel corpora
available only for high-resource languages [8], [9]; Arabic Sign Language is
constrained by small, low-signer, mostly isolated or alphabet-level datasets [12],
[13], [14]; the deep-learning pipeline ends at gloss, leaving the grammar gap to a
separate stage [6], [7]; and deployed commercial systems are, as Table 2.3 shows,
uniformly single-direction, narrow in language, and frequently hardware-bound
[15]–[19].

From these observations a clear gap emerges. No widely available system is
simultaneously *bidirectional*, *bilingual* across a high-resource and a
low-resource sign language, and *deployable in an ordinary web browser*. The corpus
dependence of end-to-end models makes them unsuitable for Arabic Sign Language,
which argues instead for a modular, gloss-mediated design: a lightweight isolated
recogniser running on extracted landmarks, followed by a general-purpose language
model that supplies the missing grammatical structure without requiring a parallel
signing corpus for the target language. The remainder of this thesis develops
exactly such a system, the design and implementation of which are presented in the
chapters that follow.
