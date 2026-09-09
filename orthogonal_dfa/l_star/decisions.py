"""How much of what a round asked the family, the family could answer."""

from typing import Dict, List

from .statistics import binomial_side_of_boundary

#: A node is settled when fewer than half of its pushes are indecisive.  Edge
#: closure only needs each decision to come out resolvable sometimes -- it
#: closes from whichever members do resolve -- so a node answering at least half
#: the time still gets counterexample synthesis to the right DFA.  Sufficient,
#: not tight: it could be lower.
SETTLED = 0.5

_FAILURE_PROB = 0.01


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

    def every_node_settled(self) -> bool:
        """Whether every node is provably below `SETTLED`.

        Only asked of a round whose every state is full, so a node too thin for
        the test to resolve cannot arise: filling a state pushes strings through
        its ancestors.
        """
        if not self._at:
            return False
        return all(
            binomial_side_of_boundary(
                unplaced, placed + unplaced, SETTLED, failure_prob=_FAILURE_PROB
            )
            is False
            for placed, unplaced in self._at.values()
        )
