"""
The discrimination tree the transition resolver builds.

Internal nodes are midfixes. A node's midfix p classifies a string s by the family
membership of s + p + v over the round's base suffixes v, so p sits between the
string and each suffix, hence the name. Leaves are DFA state ids.
"""

from typing import Callable, Iterator, List, Optional, Tuple

import numpy as np

# A leaf is an int state id; an internal node is
# (midfix, {True: accept_child, False: reject_child}).
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


def _depth(node: Node) -> int:
    if isinstance(node, int):
        return 0
    _, lookup = node
    return 1 + max(_depth(child) for child in lookup.values())


class MidfixTree:
    """
    Discrimination tree over midfixes; see the module docstring.
    """

    def __init__(self, base_family: List[List[int]]):
        self.base_family = [list(v) for v in base_family]
        # The empty midfix splits accept (state 0) from reject (state 1).
        self._root: Node = ((), {True: 0, False: 1})
        self.num_states = 2

    # -- structure ----------------------------------------------------------

    @property
    def root(self) -> Node:
        """
        The raw root node, for renderers that draw the structure itself.
        """
        return self._root

    def leaves(self) -> Iterator[int]:
        return _leaves(self._root)

    def path_of(self, state: int) -> Optional[Tuple[bool, ...]]:
        """The branches from the root to leaf ``state`` (True = accept child); a
        stable node key, unlike the node objects a split rebuilds."""

        def find(node: Node, path: Tuple[bool, ...]) -> Optional[Tuple[bool, ...]]:
            if isinstance(node, int):
                return path if node == state else None
            _, lookup = node
            for branch, child in lookup.items():
                found = find(child, path + (branch,))
                if found is not None:
                    return found
            return None

        return find(self._root, ())

    def midfix_at(self, path: Tuple[bool, ...]) -> tuple:
        """The midfix of the internal node reached by following ``path``."""
        node = self._root
        for branch in path:
            node = node[1][branch]
        return node[0]

    def accepting_leaves(self) -> set:
        """
        The leaves on the accept side of the root, i.e. the accepting states. Sound
        because a split only refines a leaf, never moves it across.
        """
        _, lookup = self._root
        return set(_leaves(lookup[True]))

    @property
    def depth(self) -> int:
        return _depth(self._root)

    def split(self, state: int, midfix) -> int:
        """
        Refine leaf state into {True: state, False: new} under midfix, returning the
        new state id. The accept branch reuses the old id so existing references to
        state stay valid.
        """
        new_state = self.num_states
        self.num_states += 1
        self._root = _replace_leaf(
            self._root, state, (tuple(midfix), {True: state, False: new_state})
        )
        return new_state

    def suffixes(self, midfix) -> List[List[int]]:
        """
        The node's distinguisher family: each base suffix behind midfix.
        """
        return [list(midfix) + v for v in self.base_family]

    # -- classification -----------------------------------------------------

    def classify(self, seq, decide: Decide) -> Optional[int]:
        """
        Route seq to a leaf, or None when a node's decide callback abstains.
        """
        node = self._root
        while not isinstance(node, int):
            midfix, lookup = node
            decision = decide(seq, midfix)
            if decision is None:
                return None
            node = lookup[decision]
        return node

    def first_disagreement(self, s, sprime, decide: Decide, prefix) -> Optional[tuple]:
        """
        The midfix separating s and sprime, or None.

        s and sprime currently sift to the same leaf, but s + prefix and
        sprime + prefix are known to reach different leaves. Walk down the branch
        where they still agree; the first node where they disagree yields the
        separating midfix prefix + node midfix. None when a needed classification
        is indecisive, or when they agree all the way to a leaf.
        """
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

    def classify_many(self, seqs, decide_level) -> List[Optional[int]]:
        """
        Like [classify(s) for s in seqs] but with the reads batched one level at a
        time: decide_level gets every (seq, midfix) at the current level and returns
        their decisions, so a batching oracle spends one call per level rather than one
        per node. A None decision drops that string (its result stays None).
        """
        out: List[Optional[int]] = [None] * len(seqs)
        active = [(self._root, i) for i in range(len(seqs))]
        while active:
            pairs: List[Tuple[list, tuple]] = []
            meta = []
            for node, i in active:
                if isinstance(node, int):
                    out[i] = node
                else:
                    midfix, lookup = node
                    pairs.append((seqs[i], midfix))
                    meta.append((lookup, i))
            if not pairs:
                break
            nxt = []
            for (lookup, i), decision in zip(meta, decide_level(pairs)):
                if decision is not None:
                    nxt.append((lookup[decision], i))
            active = nxt
        return out

    def classify_pool(self, num_prefixes: int, decide_columns) -> np.ndarray:
        """
        Classify a whole fixed population at once. decide_columns(midfix) returns the
        (accept, reject) boolean columns over all num_prefixes for that node's family,
        read straight off a cached mask matrix with no oracle. A prefix an ancestor
        abstained on stays -1.
        """

        def recurse(node: Node) -> np.ndarray:
            if isinstance(node, int):
                return np.full(num_prefixes, node)
            midfix, lookup = node
            acc, rej = decide_columns(midfix)
            results = np.full(num_prefixes, -1)
            results[rej] = recurse(lookup[False])[rej]
            results[acc] = recurse(lookup[True])[acc]
            return results

        return recurse(self._root)

    def render(self, render_midfix, indent=0) -> List[str]:
        def recurse(node: Node, indent: int) -> List[str]:
            pad = " " * indent
            if isinstance(node, int):
                return [f"{pad}State {node}"]
            midfix, lookup = node
            lines = [f"{pad}{render_midfix(midfix)}:"]
            lines += recurse(lookup[False], indent + 4)
            lines += recurse(lookup[True], indent + 4)
            return lines

        return recurse(self._root, indent)


def oracle_decider(oracle, base_family: List[List[int]], accept: float, reject: float):
    """
    A (decide, decide_level) pair that reads a midfix node against the oracle. Both
    query s + midfix + v over base_family and threshold the mean the same way
    (> accept accepts, < reject rejects, the band between abstains); decide scores one
    string, decide_level scores a whole level in one batched call for classify_many.
    """

    def verdict(mean: float) -> Optional[bool]:
        if mean > accept:
            return True
        if mean < reject:
            return False
        return None

    def decide(seq, midfix) -> Optional[bool]:
        vs = [list(seq) + list(midfix) + v for v in base_family]
        return verdict(float(np.mean(oracle.membership_queries(vs))))

    def decide_level(pairs) -> List[Optional[bool]]:
        queries, spans = [], []
        for seq, midfix in pairs:
            lo = len(queries)
            queries.extend(list(seq) + list(midfix) + v for v in base_family)
            spans.append((lo, len(queries)))
        answers = np.asarray(oracle.membership_queries(queries))
        assert len(answers) == len(queries), "oracle dropped answers"
        return [verdict(float(answers[lo:hi].mean())) for lo, hi in spans]

    return decide, decide_level
