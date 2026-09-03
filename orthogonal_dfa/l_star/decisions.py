"""How much of what the round asked the family, the family could answer."""


class Decisions:
    """Placements asked of one round's family, pooled over the paths whose
    strings the FNR gate certified: the push-down and the ``(s, c)`` edge
    probes.

    Counterexample probes are deliberately not counted.  They sift arbitrary
    sampled strings the gate never saw, and run four to five times the
    indecision the gate allows; pooling them in would measure the sampler
    rather than the family, and no round would ever read as settled.
    """

    def __init__(self):
        self.placed = 0
        self.unplaced = 0

    def record(self, placed: bool) -> None:
        if placed:
            self.placed += 1
        else:
            self.unplaced += 1

    @property
    def total(self) -> int:
        return self.placed + self.unplaced
