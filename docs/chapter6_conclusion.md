# Chapter 6 — Conclusion and Future Work

## 6.1 Conclusion

This thesis set out to address a persistent and practical problem: the
communication barrier between Deaf and hard-of-hearing signers and the hearing
world, and the absence of tools that bridge it in both directions, for more than one
sign language, without specialised hardware. The work delivered **Together**, a
real-time, bidirectional, bilingual sign-language translation platform that runs
entirely in a standard web browser and treats both a high-resource language
(American Sign Language) and a low-resource one (Arabic/Egyptian Sign Language) as
first-class.

The project met each of its objectives. Real-time landmark capture was achieved in
the browser with MediaPipe, streaming a compact, privacy-preserving skeleton rather
than raw video. Isolated-sign recognition was implemented for both languages with
models suited to their data — a Squeezeformer-style 1-D convolution-and-transformer
network for the 250-class ASL vocabulary, exported to TFLite, and a compact CNN-GRU
for the 20-class ArSL vocabulary in PyTorch. Recognised signs were turned into fluent
sentences by a gloss-mediated language stage, and the reverse path animated an avatar
from sign clips retrieved by semantic search. The four translation directions were
combined into a live, two-party meeting over WebRTC, and the system was finished with
the production qualities of authentication, a bilingual right-to-left interface, and
a deployable container.

The evaluation supports the central design decision. The ASL model reached **62.4%
Top-1 accuracy** over 250 classes and the ArSL model **99.41%** over 20 classes,
and — most importantly for the gloss-mediated approach — the language-model stage
improved translation quality substantially and consistently in both languages
(chrF rising from 0.54 to 0.91 for ASL and from 0.34 to 0.59 for ArSL), with the
gain concentrated in recall, confirming that the language model supplies the
grammatical structure that raw gloss lacks. *Together* therefore demonstrates that a
modular, gloss-mediated, landmark-based design can deliver a complete bidirectional,
bilingual system on commodity hardware — filling a gap left by the unidirectional,
monolingual, or hardware-bound systems surveyed in Chapter 2.

## 6.2 Contributions

The principal contributions of this work are:

1. **A bidirectional, bilingual, browser-based system** that translates between sign
   and spoken language in four directions and supports a live two-party meeting,
   without gloves, depth cameras, or installation.
2. **A working Egyptian Arabic Sign Language recogniser**, a contribution to an
   under-resourced language for which word-level resources are scarce.
3. **A gloss-mediated translation design** that uses a general-purpose language
   model to bridge sign-language gloss and spoken-language grammar, removing the
   dependence on the large parallel signing corpora that end-to-end models require.
4. **An engineering blueprint** for real-time landmark inference in the browser with
   graceful offline fallback, documented and reproducible from public datasets.

## 6.3 Limitations

The results must be read with four limitations in mind. First, the recognition
evaluations are **signer-dependent**: both models use random rather than
signer-independent splits, so — particularly for ArSL, whose 72 signers appear
across all splits — the reported accuracy may overstate real-world performance
through identity leakage. Second, the **ASL model is trained on a subset** of the
full corpus, and its 250-class accuracy, while strong for the task, leaves clear
headroom. Third, the system recognises **isolated** signs rather than continuous
signing, so the language model rather than a continuous recogniser bears the burden
of forming sentences, and natural conversational signing is not yet supported.
Fourth, the **Arabic vocabulary is small** (20 signs) and the Arabic translation
metrics rest on a limited reference set, so they should be read as indicative.

## 6.4 Future Work

These limitations map directly onto the most valuable directions for future work:

- **Signer-independent evaluation.** Re-evaluate both models under a
  leave-one-signer-out protocol — as the original ArSL dataset authors did — to
  obtain an honest measure of generalisation to unseen signers.
- **Larger and more diverse training data.** Train the ASL model on the full
  corpus, and expand the ArSL vocabulary well beyond 20 signs to make it practically
  useful.
- **Continuous sign-language recognition.** Move from isolated signs to continuous
  signing, which would allow natural, sentence-level conversation rather than
  sign-by-sign accumulation.
- **A user study with the Deaf community.** Evaluate the system with Deaf and
  hard-of-hearing participants on real communication tasks, measuring comprehension
  and usability rather than only automatic metrics.
- **Latency and deployment hardening.** Quantify and optimise end-to-end latency
  (warm models, batched inference, edge deployment) to guarantee interactive
  performance under real network conditions.
- **Broader language coverage.** Generalise the architecture to additional sign
  languages, building on the demonstration that it already spans two typologically
  different ones.

## 6.5 Closing Remarks

*Together* shows that an accessible, bidirectional, bilingual sign-language
translator is achievable today with commodity hardware and a carefully chosen,
modular design. While continuous signing and signer-independent robustness remain
open challenges, the system delivers a complete and deployable platform and, in
treating Arabic Sign Language as a first-class language, contributes a small but
meaningful step toward technology that serves under-represented Deaf communities.
