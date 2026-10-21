"""``tripartiteSearch``, proved by ``TripartiteSearchCorrect`` in ``OrthoDFA/Utils/TripartiteSearch.lean``."""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

UNDECIDED_LO = "undecided_lo"
DISAGREE_LO = "disagree_lo"
UNDECIDED_HI = "undecided_hi"
ALL_AGREE = "all_agree"
EDGE = "edge"
PAIR = "pair"
TRIPLE = "triple"
UNDECIDED_MID = "undecided_mid"


@dataclass(frozen=True)
class Found:
    """``kind`` is one of the constants above; ``position`` is ``None`` for the four
    endpoint kinds; ``queried`` lists the positions ``agrees`` was called at, in order.
    """

    kind: str
    position: Optional[int]
    queried: Tuple[int, ...]


def tripartite_search(
    agrees: Callable[[int], Optional[bool]], lo: int, hi: int, *, recover: bool
) -> Found:
    """Where ``agrees`` turns from ``True`` to ``False`` between ``lo`` and ``hi``."""
    queried = []

    def ask(i):
        queried.append(i)
        return agrees(i)

    def found(kind, position):
        return Found(kind, position, tuple(queried))

    first = ask(lo)
    if first is None:
        return found(UNDECIDED_LO, None)
    if not first:
        return found(DISAGREE_LO, None)
    last = ask(hi)
    if last is None:
        return found(UNDECIDED_HI, None)
    if last:
        return found(ALL_AGREE, None)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        here = ask(mid)
        if here is not None:
            lo, hi = (mid, hi) if here else (lo, mid)
            continue
        if not recover:
            return found(UNDECIDED_MID, mid)
        left = ask(mid - 1)
        if left is None:
            return found(PAIR, mid - 1)
        right = ask(mid + 1)
        if right is None:
            return found(PAIR, mid)
        if left and not right:
            return found(TRIPLE, mid)
        lo, hi = (mid + 1, hi) if left else (lo, mid - 1)
    return found(EDGE, hi)


def skipping_search(agrees: Callable[[int], Optional[bool]], lo: int, hi: int) -> Found:
    """``tripartite_search`` without ``recover``, except that an undecided midpoint
    gives way to the decided position between the bounds nearest it; ``UNDECIDED_MID``
    only once every position between them is undecided.  Not covered by the proof.
    """
    queried = []

    def ask(i):
        queried.append(i)
        return agrees(i)

    def found(kind, position):
        return Found(kind, position, tuple(queried))

    first = ask(lo)
    if first is None:
        return found(UNDECIDED_LO, None)
    if not first:
        return found(DISAGREE_LO, None)
    last = ask(hi)
    if last is None:
        return found(UNDECIDED_HI, None)
    if last:
        return found(ALL_AGREE, None)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        here = ask(mid)
        for other in sorted(range(lo + 1, hi), key=lambda i: (abs(i - mid), i))[1:]:
            if here is not None:
                break
            mid, here = other, ask(other)
        if here is None:
            return found(UNDECIDED_MID, (lo + hi) // 2)
        lo, hi = (mid, hi) if here else (lo, mid)
    return found(EDGE, hi)
