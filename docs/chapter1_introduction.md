# Chapter 1 — Introduction

Communication is a fundamental human right, yet for millions of Deaf and
hard-of-hearing people it is obstructed daily. The World Health Organization
estimates that more than 1.5 billion people live with some degree of hearing
loss, a figure projected to rise substantially over the coming decades [1]. For
a large proportion of this community, a signed language — not a spoken one — is
the natural first language. Sign languages such as American Sign Language (ASL)
and Arabic Sign Language (ArSL) are complete, rule-governed natural languages
with their own phonology, morphology, and syntax, which differ fundamentally
from the spoken languages of the surrounding hearing community [2]. The barrier
between Deaf and hearing people is therefore not a linguistic deficiency on the
part of signers but a *translation gap*: most hearing people cannot sign, and
few tools can mediate fluently between the two modalities in real time.

Although automatic sign-language processing has advanced considerably, existing
systems remain constrained in three recurring ways [3]. First, the majority are
**unidirectional** — they convert sign to text, or animate text into sign, but
rarely support both directions within one system. Second, research and
deployment overwhelmingly target a single **high-resource** language, almost
always ASL, leaving languages such as Arabic Sign Language comparatively
under-served by data and tools. Third, many systems rely on **specialised
hardware** such as data gloves or depth cameras, or operate only under
controlled laboratory conditions, which limits their adoption in everyday use.

This thesis presents **Together**, a real-time, bidirectional, and bilingual
sign-language translation system that operates entirely within a standard web
browser and requires no installation or special equipment beyond a webcam.
*Together* supports both ASL and ArSL and translates in both directions: from
sign to text and synthesised speech, and from text or speech back into sign
through a landmark-driven avatar. It additionally provides a live two-person
meeting mode, in which a signer and a speaker hold a conversation mediated by
the system over a peer-to-peer video connection. By addressing
bidirectionality, bilingualism, and browser-based deployment together, the
project tackles the three limitations identified above within a single,
practical application.

## 1.1 Motivation

The motivation for this work is both social and technical.

Socially, accessible communication has a direct effect on the education,
employment, and independence of Deaf individuals. A tool that lowers the
barrier between signers and non-signers therefore carries tangible human value.
This need is especially pronounced in the Arabic-speaking world: Arabic Sign
Language is used by a large population, yet it remains a low-resource language
in computational terms, with few public datasets and almost no fluent,
bidirectional translation systems. Treating ArSL as a first-class language
alongside ASL — rather than as an afterthought — addresses a genuine and
under-explored need.

Technically, recent advances make a practical solution feasible for the first
time. Browser-based landmark extraction removes the dependence on specialised
cameras and allows sign articulation to be captured on ordinary consumer
devices [5]. Lightweight neural inference makes real-time recognition possible
on commodity hardware. Most importantly, the emergence of capable large
language models offers, for the first time, a robust way to transform the terse,
grammatically distinct gloss produced by sign recognition into fluent,
natural-language sentences without relying on brittle hand-crafted rules [4].
The convergence of these technologies enables a system that is simultaneously
real-time, bidirectional, bilingual, and universally deployable through a web
browser — a combination not previously demonstrated.

## 1.2 Aims and Objectives

The **aim** of this project is to design, implement, and evaluate a real-time,
bidirectional sign-language translation system that supports both a
high-resource (ASL) and a low-resource (ArSL) sign language directly in the web
browser.

To achieve this aim, the following **objectives** were defined:

1. To develop a browser-based landmark-extraction front-end capable of capturing
   sign articulation in real time without specialised hardware.
2. To train and evaluate sign-recognition models for both ASL and ArSL, and to
   assess their generalisation, including signer-independent evaluation.
3. To design a gloss-based translation stage that uses a large language model to
   convert recognised sign gloss into fluent spoken-language sentences, with a
   robust offline fallback.
4. To implement the reverse, text- and speech-to-sign direction through semantic
   retrieval of sign animations.
5. To integrate all components into a single bilingual web application,
   including a real-time two-person meeting mode.
6. To evaluate the system quantitatively in terms of recognition accuracy and
   translation quality, and to analyse the contribution of the language-model
   stage.

## 1.3 Proposed Approach

*Together* combines computer vision, machine learning, and natural-language
processing into a single end-to-end pipeline. Its high-level architecture is
shown in Fig. 1.1. The client runs entirely in the browser, where MediaPipe
Holistic extracts pose, hand, and face landmarks from the webcam feed [5] and
streams them to the server. A FastAPI back-end routes requests, performs model
inference, and handles real-time signalling, while a PostgreSQL database with a
vector extension stores user data and the sign representations used for
synthesis.

The two translation directions are illustrated in Fig. 1.2. In the
**sign-to-text/speech** direction, the landmark stream is classified into
individual signs by one of two models — a network for the ASL vocabulary and a
recurrent network for the ArSL vocabulary. Because the models recognise
*isolated* signs, the recognised outputs are accumulated in a voting buffer to
form a sequence of glosses, which is then passed to a large language model that
reconstructs a grammatical sentence and, optionally, speaks it through
text-to-speech. In the **text/speech-to-sign** direction, spoken input is
converted to text, mapped to gloss, and matched to stored sign animations
through semantic embedding search, which the avatar then plays back. Throughout,
provider chains degrade gracefully to offline components so that the system
remains functional even without cloud connectivity.

*[Figure 1.1 — High-level architecture of the Together system]*

*[Figure 1.2 — The bidirectional translation pipelines]*

## 1.4 Thesis Organization

The remainder of this thesis is organised as follows. **Chapter 2** reviews the
relevant literature, covering sign-language linguistics, prior work in sign
recognition and translation, and existing resources for Arabic Sign Language.
**Chapter 3** presents the system analysis, requirements, datasets, and overall
design. **Chapter 4** describes the implementation of the recognition models,
the gloss-to-sentence translation stage, the sign-synthesis path, and the
real-time infrastructure. **Chapter 5** reports the testing and validation
results across recognition accuracy, translation quality, and system behaviour.
**Chapter 6** concludes the thesis, summarises its achievements, and outlines
directions for future work.
