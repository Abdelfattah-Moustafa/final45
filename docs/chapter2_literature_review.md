# Chapter 2 — Review of Literature

Whereas the previous chapter introduced the problem in general terms, this
chapter reviews the body of work that relates specifically to the design of
*Together*. It begins with the linguistic foundations that explain *why* sign
translation cannot be treated as a simple word-for-word mapping, then surveys
prior work in sign-language recognition and translation, examines the state of
research on Arabic Sign Language, and finally synthesises these threads into the
research gap that this project addresses.

## 2.1 Linguistic Foundations of Sign Languages

A recurring misconception is that signing is a manual re-encoding of a spoken
language. Linguistic research has long established the opposite: sign languages
are full, autonomous natural languages with their own phonology, morphology, and
syntax, exhibiting the same structural complexity and universals found in spoken
languages [6]. They are not derived from the surrounding spoken language, and
their grammar is organised around the visual–spatial modality rather than a
linear stream of words.

Two properties of sign-language grammar are particularly relevant to automatic
translation. The first is the **use of space and non-manual markers**: facial
expressions, head tilts, eye gaze, and body posture are not paralinguistic
decoration but carry grammatical meaning, marking questions, negation,
conditionals, and topic [7]. The second is **word order**: many sign languages,
including American Sign Language, favour a Topic–Comment structure in which the
topic is established first and then commented upon, producing an ordering that
differs systematically from the subject–verb–object order of languages such as
English.

Together, these properties mean that the raw output of a sign recogniser — a
sequence of sign labels, or *gloss* — is not a grammatical sentence in the
target spoken language. A faithful translation must reorder the gloss, insert the
function words and morphology that signing conveys spatially or non-manually, and
resolve the grammatical role of each sign. This linguistic gap is the central
justification for treating gloss as an intermediate representation and for
employing a dedicated language-generation stage, as discussed in the following
sections.

## 2.2 Sign Language Recognition

Early approaches to sign-language recognition relied on sensor-based capture,
using instrumented gloves or motion sensors to record hand configuration and
movement. While accurate, such systems require specialised and often intrusive
hardware, limiting their practicality for everyday use. The field has therefore
shifted decisively toward **vision-based recognition**, which operates on
ordinary video.

### 2.2.1 Vision-based and skeleton-based recognition

Vision-based recognition is commonly divided into *isolated* recognition, which
classifies a single sign performed in a segmented clip, and *continuous*
recognition, which must additionally locate sign boundaries within an unbroken
stream. To support data-driven research, large-scale benchmarks have been
introduced; the Word-Level American Sign Language (WLASL) dataset, for example,
provides thousands of glosses performed by many signers and has become a standard
benchmark for isolated recognition, with baselines spanning both appearance-based
and pose-based methods [10].

A significant trend is the move from raw-pixel models toward **skeleton- or
landmark-based** representations, in which the body, hand, and face are reduced to
a compact set of keypoints before classification. Skeleton-aware multi-modal
approaches have achieved strong results by combining keypoint streams with
complementary modalities, demonstrating that pose information alone captures much
of the signal needed for recognition while being far more efficient than
full-frame video [11]. This finding directly motivates the landmark-based
front-end adopted in this project, which classifies signs from extracted keypoints
rather than from images.

## 2.3 Sign Language Translation

Recognition alone yields gloss, not fluent language; *translation* is the further
step of producing a grammatical sentence in the target spoken language. The task
was formalised as a neural sequence-to-sequence problem by Camgöz et al., who
introduced a benchmark for sign-language translation and an encoder–decoder model
with attention, explicitly using gloss as an intermediate supervision signal
between video and spoken-language text [8]. This established gloss as a useful
pivot representation: recognising signs into gloss and then translating gloss into
text decomposes a very hard problem into two more tractable ones.

Subsequent work replaced the recurrent architecture with transformers, jointly
learning recognition and translation in a single network and improving
performance, while confirming that gloss-level information remains a strong
intermediate signal even as models grow more powerful [9]. These results validate
a two-stage, gloss-mediated design. They also highlight a practical limitation:
such end-to-end translation models are trained on large parallel corpora of
continuous signing, which exist for a small number of high-resource languages but
not for most sign languages — a constraint that strongly shapes the present work.

## 2.4 Arabic Sign Language

Compared with American Sign Language, Arabic Sign Language (ArSL) is markedly
**under-resourced**. Much of the available research targets the manual alphabet
or isolated signs rather than continuous translation. Representative of dataset
efforts at the alphabet level, the ArASL dataset provides a large collection of
labelled images of Arabic sign-language letters, enabling static recognition of
the alphabet but not sentence-level translation [13]. Broader surveys of the
field have compared image-based and sensor-based recognition techniques for ArSL,
documenting both the diversity of approaches and the persistent scarcity of
standardised data [14].

More recent efforts have begun to close the data gap. The KArSL database, for
instance, offers a comparatively large, multi-modal Arabic sign-language corpus
recorded from multiple signers, together with recognition benchmarks [12]. Even
so, the ArSL literature remains dominated by isolated recognition, frequently
relies on controlled capture or specialised sensors, and offers very little in
the way of fluent, bidirectional, sentence-level translation. This imbalance
between the rich tooling available for ASL and the sparse resources for ArSL is
precisely the gap that motivates treating ArSL as a first-class language in this
project.

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

The reviewed literature and existing systems support five conclusions. First,
sign languages are
grammatically distinct from spoken languages, so translation requires more than
recognition; the gloss produced by a recogniser must be restructured into fluent
text [6], [7]. Second, vision- and landmark-based recognition has matured to the
point where signs can be classified efficiently from keypoints without
specialised hardware [10], [11]. Third, gloss-mediated, two-stage translation is a
well-validated design, but the strongest end-to-end models depend on large
parallel corpora that exist only for a few high-resource languages [8], [9].
Fourth, Arabic Sign Language remains under-resourced, with most work confined to
isolated or alphabet-level recognition rather than translation [12], [13], [14].
Fifth, as Table 2.1 shows, deployed commercial systems are uniformly
single-direction, narrow in language coverage, and frequently hardware-bound
[15]–[19].

From these observations, a clear gap emerges. There is, at present, no widely
available system that is simultaneously *bidirectional*, *bilingual* across a
high-resource and a low-resource sign language, and *deployable in an ordinary web
browser*. The corpus dependence of end-to-end models makes them ill-suited to
ArSL, suggesting instead a modular, gloss-mediated design in which an isolated
recogniser feeds a general-purpose language model that supplies the missing
grammatical structure — an approach that does not require a large parallel signing
corpus for the target language. The remainder of this thesis develops exactly such
a system, the design and implementation of which are presented in the chapters
that follow.
