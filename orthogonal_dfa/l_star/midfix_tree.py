"""The discrimination tree direct-L* sifts against.

Internal nodes are *midfixes*.  A node's midfix ``p`` classifies a string ``s``
by the family membership of ``s + p + v`` over the base suffixes ``v`` -- so ``p``
sits between the string and each suffix, hence the name.  Leaves are state ids.

The tree owns *structure* only: which midfix cuts where, and which leaf a branch
lands in.  Every classification is delegated to a caller-supplied

    decide(seq, midfix) -> True | False | None

so the oracle, the query caching and the accept/reject thresholds all stay with
the learner.  ``None`` means the family cannot place ``seq`` at that node, and a
sift that meets one reports the boundary string it died on rather than a leaf.

Splitting only ever refines a leaf, which is what lets the learner keep its
transition function: the accept side of the root stays the accept side forever,
and every id in ``range(num_states)`` is always a live leaf.
"""

from typing import Callable, Iterator, List, Optional, Tuple

from .structures import DecisionTree, DecisionTreeInternalNode, DecisionTreeLeafNode

# A leaf is an ``int`` state id; an internal node is
# ``(midfix, {True: accept_child, False: reject_child})``.
Node = object

Decide = Callable[[List[int], tuple], Optional[bool]]


def _leaves(node: Node) -> Iterator[int]:
    if isinstance(node, int):
        yield node
        return
    _, lookup = node
    for child in lookup.values():
        yield from _leaves(child)


def _replace_leaf(node: Node, state: int, new_node: Node) -> Node:
    if isinstance(node, int):
        return new_node if node == state else node
    midfix, lookup = node
    return (
        midfix,
        {k: _replace_leaf(v, state, new_node) for k, v in lookup.items()},
    )


class MidfixTree:
    """Discrimination tree over midfixes; see the module docstring."""

    def __init__(self):
        # The empty midfix splits accept (state 0) from reject (state 1).
        self._root: Node = ((), {True: 0, False: 1})
        self.num_states = 2

    # -- structure ----------------------------------------------------------

    @property
    def root(self) -> Node:
        """The raw root node, for renderers that draw the structure itself.
        Everything else should go through the methods below."""
        return self._root

    def leaves(self) -> Iterator[int]:
        return _leaves(self._root)

    def accepting_leaves(self) -> set:
        """The leaves on the accept side of the root, i.e. the accepting states.
        Sound because a split only refines a leaf, never moves it across."""
        _, lookup = self._root
        return set(_leaves(lookup[True]))

    def split(self, state: int, midfix) -> int:
        """Refine leaf ``state`` into ``{True: state, False: new}`` under
        ``midfix``, returning the new state id.  The True branch reuses the old
        id so the learner's existing references to ``state`` stay valid."""
        new_state = self.num_states
        self.num_states += 1
        self._root = _replace_leaf(
            self._root, state, (tuple(midfix), {True: state, False: new_state})
        )
        return new_state

    # -- classification -----------------------------------------------------

    def sift(self, seq, decide: Decide) -> Tuple[Optional[int], Optional[tuple]]:
        """Route ``seq`` to a leaf.  Returns ``(state, None)`` when it lands
        decisively, or ``(None, boundary)`` when some node cannot place it.

        The boundary is ``seq + midfix``, not ``seq``: the indecision is over
        ``seq + midfix + v``, so it is ``seq + midfix`` that the family failed on
        and that the caller can enrich the next family against."""
        node = self._root
        while not isinstance(node, int):
            midfix, lookup = node
            decision = decide(seq, midfix)
            if decision is None:
                return None, tuple(seq) + tuple(midfix)
            node = lookup[decision]
        return node, None

    def sift_levels(self, seqs, decide: Decide) -> Iterator[List[Tuple[tuple, tuple]]]:
        """Walk ``seqs`` down the tree in lockstep, yielding each level's
        ``(seq, midfix)`` pairs *before* that level's decisions are read.

        A caller can therefore warm a cache one batch per level and still visit
        exactly the nodes the individual sifts would -- the batching costs no
        extra queries."""
        level = [(self._root, [tuple(s) for s in seqs])]
        while level:
            pairs = [
                (s, node[0])
                for node, group in level
                if not isinstance(node, int)
                for s in group
            ]
            if not pairs:
                return
            yield pairs
            nxt = []
            for node, group in level:
                if isinstance(node, int):
                    continue
                midfix, lookup = node
                buckets: dict = {}
                for s in group:
                    decision = decide(s, midfix)
                    if decision is not None:
                        buckets.setdefault(decision, []).append(s)
                nxt.extend((lookup[d], g) for d, g in buckets.items())
            level = nxt

    def first_disagreement(self, s, sprime, decide: Decide, prefix) -> Optional[tuple]:
        """The midfix separating ``s`` and ``sprime``, or ``None``.

        Both currently sift to the same leaf, but ``s + prefix`` and
        ``sprime + prefix`` are known to reach different leaves.  Walk down the
        branch where they still agree; the first node where they disagree yields
        the separating midfix ``prefix + node midfix``.  ``None`` when a needed
        classification is indecisive, or when they agree all the way to a leaf."""
        node = self._root
        while not isinstance(node, int):
            midfix, lookup = node
            full = (*prefix, *midfix)
            d, dprime = decide(s, full), decide(sprime, full)
            if d is None or dprime is None:
                return None
            if d != dprime:
                return full
            node = lookup[d]
        return None

    # -- export -------------------------------------------------------------

    def to_decision_tree(self, predicate_for) -> DecisionTree:
        """Convert to the generic :class:`DecisionTree`, with
        ``predicate_for(midfix)`` supplying each internal node's predicate."""

        def convert(node: Node) -> DecisionTree:
            if isinstance(node, int):
                return DecisionTreeLeafNode(node)
            midfix, lookup = node
            # by_rejection is (if rejected, if accepted).
            return DecisionTreeInternalNode(
                predicate=predicate_for(midfix),
                by_rejection=(convert(lookup[False]), convert(lookup[True])),
            )

        return convert(self._root)
