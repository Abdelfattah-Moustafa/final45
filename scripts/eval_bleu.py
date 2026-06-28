#!/usr/bin/env python3
"""Translation-quality evaluation for the gloss -> natural-sentence step.

Computes corpus BLEU and chrF (and optional per-sentence scores) between the
system's generated sentences (hypotheses) and human reference sentences.

Why these metrics:
  * BLEU  - standard MT metric; word n-gram overlap with the brevity penalty.
  * chrF  - character n-gram F-score; more reliable than BLEU for morphologically
            rich languages such as Arabic, so report it for the ArSL set.

Both come from `sacrebleu`, which gives reproducible, comparable numbers
(unlike ad-hoc NLTK BLEU). Install once:

    pip install sacrebleu

------------------------------------------------------------------------------
INPUT FORMATS (pick one)

1) Plain text, one sentence per line, lines aligned across files:
       --hyp  hyps.txt          # your system output, one sentence per line
       --ref  refs.txt          # human reference, same number of lines
   For multiple references per example, pass --ref several times:
       --ref refs1.txt --ref refs2.txt

2) JSONL, one JSON object per line (easiest to produce from your pipeline):
       --jsonl data.jsonl
   where each line looks like:
       {"gloss": "CAT MAT ON", "hyp": "the cat is on the mat",
        "refs": ["the cat is on the mat", "a cat sits on the mat"]}

------------------------------------------------------------------------------
EXAMPLES
    python scripts/eval_bleu.py --jsonl scripts/sample_eval.jsonl --per-sentence
    python scripts/eval_bleu.py --hyp hyps.txt --ref refs.txt
    python scripts/eval_bleu.py --jsonl asl.jsonl --lang en
    python scripts/eval_bleu.py --jsonl arsl.jsonl --lang ar   # uses intl tokenizer
"""
import argparse
import json
import sys


def load_jsonl(path):
    hyps, refs_per_example = [], []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "hyp" not in obj or "refs" not in obj:
                sys.exit(f"{path}:{ln}: each line needs 'hyp' and 'refs' keys")
            refs = obj["refs"]
            if isinstance(refs, str):
                refs = [refs]
            hyps.append(obj["hyp"])
            refs_per_example.append(refs)
    return hyps, refs_per_example


def load_text(hyp_path, ref_paths):
    with open(hyp_path, "r", encoding="utf-8") as f:
        hyps = [l.rstrip("\n") for l in f]
    ref_files = []
    for rp in ref_paths:
        with open(rp, "r", encoding="utf-8") as f:
            ref_files.append([l.rstrip("\n") for l in f])
    n = len(hyps)
    for rp, rf in zip(ref_paths, ref_files):
        if len(rf) != n:
            sys.exit(f"line-count mismatch: {hyp_path} has {n}, {rp} has {len(rf)}")
    # transpose to per-example list of references
    refs_per_example = [[rf[i] for rf in ref_files] for i in range(n)]
    return hyps, refs_per_example


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", help="JSONL file with 'hyp' and 'refs' per line")
    ap.add_argument("--hyp", help="hypotheses file, one sentence per line")
    ap.add_argument("--ref", action="append", default=[],
                    help="reference file (repeat for multiple references)")
    ap.add_argument("--lang", default="en",
                    help="language code; 'ar' switches BLEU to the intl tokenizer")
    ap.add_argument("--per-sentence", action="store_true",
                    help="also print sentence-level BLEU for each example")
    args = ap.parse_args()

    try:
        from sacrebleu.metrics import BLEU, CHRF
    except ImportError:
        sys.exit("sacrebleu not installed. Run:  pip install sacrebleu")

    if args.jsonl:
        hyps, refs_per_example = load_jsonl(args.jsonl)
    elif args.hyp and args.ref:
        hyps, refs_per_example = load_text(args.hyp, args.ref)
    else:
        sys.exit("provide either --jsonl, or both --hyp and --ref")

    if not hyps:
        sys.exit("no examples found")

    # sacrebleu wants references grouped BY reference-slot:
    #   refs_grouped[k][i] = k-th reference of the i-th example.
    # Pad examples that have fewer references by repeating their last one.
    max_refs = max(len(r) for r in refs_per_example)
    refs_grouped = [[] for _ in range(max_refs)]
    for r in refs_per_example:
        for k in range(max_refs):
            refs_grouped[k].append(r[k] if k < len(r) else r[-1])

    tokenize = "intl" if args.lang.lower().startswith("ar") else "13a"
    bleu = BLEU(tokenize=tokenize)
    chrf = CHRF()

    bleu_score = bleu.corpus_score(hyps, refs_grouped)
    chrf_score = chrf.corpus_score(hyps, refs_grouped)

    print(f"examples      : {len(hyps)}")
    print(f"references/ex : up to {max_refs}")
    print(f"tokenizer     : {tokenize}  (lang={args.lang})")
    print("-" * 50)
    print(f"BLEU : {bleu_score.score:.2f}")
    print(f"chrF : {chrf_score.score:.2f}")
    print("-" * 50)
    print(bleu_score)  # full signature line, paste this verbatim into the thesis

    if args.per_sentence:
        print("\nper-sentence BLEU:")
        sent_bleu = BLEU(tokenize=tokenize, effective_order=True)
        for i, h in enumerate(hyps):
            refs_i = [refs_grouped[k][i] for k in range(max_refs)]
            s = sent_bleu.sentence_score(h, refs_i)
            print(f"  [{i:>3}] BLEU={s.score:6.2f} | {h}")


if __name__ == "__main__":
    main()
