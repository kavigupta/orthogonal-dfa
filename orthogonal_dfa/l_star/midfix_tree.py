"""
The discrimination tree the transition resolver builds.

Internal nodes are midfixes. A node's midfix p classifies a string s by the family
membership of s + p + v over the round's base suffixes v, so p sits between the
string and each suffix, hence the name. Leaves are DFA state ids.
"""

from typing import Callable, Iterator, List, Optional, Tuple

from .sequential_decide import sequential_decisions

# A leaf is an int state id; an internal node is
# (midfix, {True: accept_child, False: reject_child}).
Node = object

Decide = Callable[[bytes, bytes], Optional[bool]]


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

    def __init__(self, base_family: List[bytes]):
        self.base_family = list(base_family)
        # The empty midfix splits accept (state 0) from reject (state 1).
        self._root: Node = (b"", {True: 0, False: 1})
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

    def midfix_at(self, path: Tuple[bool, ...]) -> bytes:
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
            self._root, state, (midfix, {True: state, False: new_state})
        )
        return new_state

    # -- classification -----------------------------------------------------

    def sift(self, seq, decide: Decide) -> Tuple[Optional[int], Optional[bytes]]:
        """
        Route seq to a leaf: (state, None) when it lands decisively, or
        (None, boundary) when some node cannot place it.

        The boundary is seq + midfix, not seq: the indecision is over
        seq + midfix + v, so it is seq + midfix that the family failed on and
        that a caller can force the next family to resolve.
        """
        node = self._root
        while not isinstance(node, int):
            midfix, lookup = node
            decision = decide(seq, midfix)
            if decision is None:
                return None, seq + midfix
            node = lookup[decision]
        return node, None

    def classify(self, seq, decide: Decide) -> Optional[int]:
        """
        Route seq to a leaf, or None when a node's decide callback abstains.
        """
        return self.sift(seq, decide)[0]

    def first_disagreement(self, s, sprime, decide: Decide, prefix) -> Optional[bytes]:
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
            full = prefix + midfix
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
            pairs: List[Tuple[bytes, bytes]] = []
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


def oracle_decider(oracle, base_family: List[bytes], accept: float, reject: float):
    """
    A (decide, decide_level) pair that classifies a midfix node by the accept-rate
    of s + midfix + v over base_family (> accept accepts, < reject rejects, the band
    between abstains); decide scores one string, decide_level a whole level.

    The rate is read sequentially (see :func:`sequential_decisions`): a string far
    from the threshold -- the common case for the accuracy estimate's random samples
    -- settles in the first block rather than spending the whole family.
    """

    def decide_level(pairs) -> List[Optional[bool]]:
        strings = [seq + midfix for seq, midfix in pairs]
        return sequential_decisions(
            strings,
            base_family,
            oracle.membership_queries,
            accept=accept,
            reject=reject,
        )

    def decide(seq, midfix) -> Optional[bool]:
        # Reuse the level path for one string, so the single and batched readers
        # early-stop identically -- callers that mix them (the accuracy estimate's
        # batched s_end plus its per-y binary search) must never disagree.
        return decide_level([(seq, midfix)])[0]

    return decide, decide_level
