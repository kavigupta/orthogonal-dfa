"""How much of what a round asked the family, the family could answer."""

from typing import Dict, List, Tuple

from .statistics import binomial_side_of_boundary

#: A node is settled when fewer than half of its pushes are indecisive.  Past
#: half the modal answer is "cannot place", so the node is reading the family's
#: noise rather than a distinction.  Deliberately not ``fnr_limit``: that is the
#: rate the gate holds a whole family to, not the point one node stops saying
#: anything, and at a node's sample size nothing near it is provable either way.
SETTLED = 0.5

_FAILURE_PROB = 0.01


def _smallest_testable() -> int:
    """Fewest decisions at which a node can come out below `SETTLED` at all."""
    n = 1
    while (
        binomial_side_of_boundary(0, n, SETTLED, failure_prob=_FAILURE_PROB)
        is not False
    ):
        n += 1
    return n


#: Below this a node has no verdict at any count, so testing it alone would keep
#: the round unsettled for good.
_TESTABLE = _smallest_testable()


class Decisions:
    """One round's push-down decisions, bucketed by the node that made them.

    Pooling every node into one rate averages a node well above `SETTLED`
    together with the many below it, which is the node another round would
    actually resolve.
    """

    def __init__(self):
        self._at: Dict[object, List[int]] = {}

    def record(self, node, placed: bool) -> None:
        counts = self._at.setdefault(node, [0, 0])
        counts[0 if placed else 1] += 1

    def _buckets(self) -> List[Tuple[int, int]]:
        """The nodes to judge: those with a verdict, plus the thin ones as one."""
        buckets, thin = [], [0, 0]
        for placed, unplaced in self._at.values():
            if placed + unplaced >= _TESTABLE:
                buckets.append((placed, unplaced))
            else:
                thin[0] += placed
                thin[1] += unplaced
        if sum(thin) >= _TESTABLE:
            buckets.append(tuple(thin))
        return buckets

    def every_node_settled(self) -> bool:
        """Whether every node is provably below `SETTLED`."""
        if not self._at:
            return False
        return all(
            binomial_side_of_boundary(
                unplaced, placed + unplaced, SETTLED, failure_prob=_FAILURE_PROB
            )
            is False
            for placed, unplaced in self._buckets()
        )
