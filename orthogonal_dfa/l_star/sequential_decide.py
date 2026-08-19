"""Early-stopping batched read of a suffix family.

Classifying a string by its mean membership over a suffix family only needs the
whole family when the string sits near the decision threshold.  This reads the
family a block at a time and drops each string as soon as a caller-supplied test
is confident which side it falls on -- so a string far from the threshold settles
in the first block -- while every block still batches its queries across all the
strings still undecided.

The classification semantics live with the caller: ``verdict`` turns a full-family
mean into a decision, and ``confident_side`` decides when a partial count is
already conclusive.  This module owns only the block-at-a-time draw and the
early-stop bookkeeping.
"""

import random
from typing import Callable, List, Optional

import numpy as np

#: Suffixes drawn per sequential block.
DEFAULT_BLOCK = 16


def sequential_decisions(
    bases: List[list],
    family: List[list],
    membership: Callable[[List[list]], List[int]],
    verdict: Callable[[float], Optional[bool]],
    confident_side: Callable[[int, int], Optional[bool]],
    *,
    block: int = DEFAULT_BLOCK,
) -> List[Optional[bool]]:
    """Classify each ``base`` by its accept-rate over the suffix ``family``.

    Draw the family a ``block`` at a time in a fixed shuffle, accumulating each
    base's accept count, and drop a base as soon as ``confident_side(accepts,
    drawn)`` returns a decision.  A base that never becomes confident reads the
    whole family and falls back to ``verdict(accepts / len(family))``.

    ``membership(strings)`` returns one 0/1 per string; the query for a base and a
    suffix is ``base + suffix``.  The shuffle makes each block a representative
    sample even when the family was screened in correlated batches.
    """
    n = len(family)
    order = random.Random(0).sample(range(n), n)
    results: List[Optional[bool]] = [None] * len(bases)
    accepts = [0] * len(bases)
    active = list(range(len(bases)))
    drawn = 0
    upto = min(block, n)
    while active:
        queries, spans = [], []
        for i in active:
            lo = len(queries)
            queries.extend(bases[i] + family[order[k]] for k in range(drawn, upto))
            spans.append((i, lo, len(queries)))
        answers = np.asarray(membership(queries))
        for i, lo, hi in spans:
            accepts[i] += int(answers[lo:hi].sum())
        drawn = upto
        if drawn >= n:
            for i in active:
                results[i] = verdict(accepts[i] / n)
            break
        still = []
        for i in active:
            side = confident_side(accepts[i], drawn)
            if side is None:
                still.append(i)
            else:
                results[i] = side
        active = still
        upto = min(upto * 2, n)
    return results
