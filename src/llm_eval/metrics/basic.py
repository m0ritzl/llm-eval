from __future__ import annotations
from typing import List
import math

# Simple BLEU implementation (corpus-level) for brevity

def corpus_bleu(references: List[List[str]], hypotheses: List[List[str]], max_n: int = 4) -> float:
    import collections

    clipped_counts = [0] * max_n
    total_counts = [0] * max_n
    ref_length = 0
    hyp_length = 0

    for refs, hyp in zip(references, hypotheses):
        # we assume single reference list (tokenized) for now
        ref = refs
        ref_length += len(ref)
        hyp_length += len(hyp)

        for n in range(1, max_n + 1):
            hyp_ngrams = collections.Counter(
                [tuple(hyp[i : i + n]) for i in range(len(hyp) - n + 1)]
            )
            ref_ngrams = collections.Counter(
                [tuple(ref[i : i + n]) for i in range(len(ref) - n + 1)]
            )
            for ng, cnt in hyp_ngrams.items():
                clipped = min(cnt, ref_ngrams.get(ng, 0))
                clipped_counts[n - 1] += clipped
                total_counts[n - 1] += cnt

    precisions = [
        (clipped_counts[i] / total_counts[i]) if total_counts[i] > 0 else 0.0 for i in range(max_n)
    ]
    if min(precisions) == 0:
        return 0.0
    log_p = sum((1.0 / max_n) * math.log(p) for p in precisions)
    bp = 1.0 if hyp_length > ref_length else math.exp(1 - ref_length / hyp_length) if hyp_length > 0 else 0.0
    bleu = bp * math.exp(log_p)
    return bleu * 100.0  # scale like SacreBLEU
