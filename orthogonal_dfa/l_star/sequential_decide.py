"""Classify strings by their accept-rate over a suffix family, reading the family
only as far as a binomial test needs to decide which side of the threshold a
string falls on."""

import random
from typing import Callable, List, Optional

import numpy as np

from .statistics import binomial_side_of_boundary

DEFAULT_BLOCK = 16
DEFAULT_ALPHA = 1e-3


def sequential_decisions(
    strings: List[List[int]],
    suffix_family: List[List[int]],
    membership: Callable[[List[List[int]]], List[int]],
    *,
    accept: float,
    reject: float,
    alpha: float = DEFAULT_ALPHA,
    block: int = DEFAULT_BLOCK,
) -> List[Optional[bool]]:
    """Classify each ``string`` by its accept-rate over ``suffix_family``: ``True``
    above ``accept``, ``False`` below ``reject``, ``None`` in the band between.

    The family is drawn ``block`` at a time in a fixed shuffle; a string is settled
    as soon as a binomial test at confidence ``alpha`` clears a threshold, else it
    reads the whole family and takes the exact mean's verdict. Each block batches
    across every string still undecided. ``membership`` returns one 0/1 per query;
    a string's query for a suffix is ``string + suffix``.
    """

    def verdict(mean: float) -> Optional[bool]:
        if mean > accept:
            return True
        if mean < reject:
            return False
        return None

    def confident_side(accepts: int, drawn: int) -> Optional[bool]:
        if binomial_side_of_boundary(accepts, drawn, accept, failure_prob=alpha):
            return True
        if (
            binomial_side_of_boundary(accepts, drawn, reject, failure_prob=alpha)
            is False
        ):
            return False
        return None

    n = len(suffix_family)
    order = random.Random(0).sample(range(n), n)
    results: List[Optional[bool]] = [None] * len(strings)
    accepts = [0] * len(strings)
    active = list(range(len(strings)))
    drawn = 0
    upto = min(block, n)
    while active:
        queries, spans = [], []
        for i in active:
            lo = len(queries)
            queries.extend(
                strings[i] + suffix_family[order[k]] for k in range(drawn, upto)
            )
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
