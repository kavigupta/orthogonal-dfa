"""``tripartite_search`` stepping past undecided midpoints."""

from typing import Callable, Optional

from ..tripartite_search import UNDECIDED_MID, Found, tripartite_search


def skipping_search(agrees: Callable[[int], Optional[bool]], lo: int, hi: int) -> Found:
    """``tripartite_search`` without ``recover``, except that an undecided midpoint
    gives way to the decided position between the bounds nearest it; ``UNDECIDED_MID``
    only once every position between them is undecided.  Not covered by the proof.
    """
    answers = {}
    queried = []

    def ask(i):
        if i not in answers:
            queried.append(i)
            answers[i] = agrees(i)
        return answers[i]

    def bounds():
        return (
            max(i for i, a in answers.items() if a),
            min(i for i, a in answers.items() if a is False),
        )

    found = tripartite_search(ask, lo, hi, recover=False)
    while found.kind == UNDECIDED_MID:
        lo, hi = bounds()
        nearest = sorted(range(lo + 1, hi), key=lambda i: (abs(i - found.position), i))
        if all(ask(i) is None for i in nearest):
            break
        found = tripartite_search(ask, *bounds(), recover=False)
    return Found(found.kind, found.position, tuple(queried))
