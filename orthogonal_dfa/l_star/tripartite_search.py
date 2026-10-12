"""The tripartite binary search of ``OrthoDFA/TallyLoop.lean``."""

PAIR = "pair"
EDGE = "edge"
TRIPLE = "triple"


def tripartite_search(agrees, fuel, lo, hi):
    """``bracketAt`` and ``bracketSifts`` at once: ``(kind, position, sifted)``.

    ``agrees`` is ``None`` where a read is undecided; ``lo`` counts as agreeing and
    ``hi`` as disagreeing whatever it says."""

    def at(i):
        return True if i == lo else False if i == hi else agrees(i)

    sifted = []
    while fuel > 0 and lo + 1 < hi:
        fuel -= 1
        mid = (lo + hi) // 2
        sifted.append(mid)
        here = at(mid)
        if here is not None:
            lo, hi = (mid, hi) if here else (lo, mid)
            continue
        sifted.append(mid - 1)
        left = at(mid - 1)
        if left is None:
            return PAIR, mid - 1, sifted
        sifted.append(mid + 1)
        right = at(mid + 1)
        if right is None:
            return PAIR, mid, sifted
        if left and not right:
            return TRIPLE, mid, sifted
        lo, hi = (mid + 1, hi) if left else (lo, mid - 1)
    return EDGE, hi, sifted
