"""On-demand population of the discrimination tree's leaves.

Each string rests at the deepest node it has been classified to, and is pulled
further down only when a leaf below it is asked for more members than it holds --
so the pool is sifted lazily, a chunk at a time, not re-sifted from the root.

Keyed by *path*, the accept/reject branches from the root to a node.  A path is
stable across splits (a split only appends child paths below the split leaf), so
this companion never has to be told the tree changed: a former leaf becomes an
internal node whose resting strings flush through it on the next pull.
"""

from typing import Callable, Dict, List, Optional, Tuple

Path = Tuple[bool, ...]
#: Classify a batch of strings against one node's midfix, decisions aligned with
#: the input (``None`` = indecisive, dropped).
Classify = Callable[[List[bytes], bytes], List[Optional[bool]]]


class LeafPopulation:
    """Strings resting at tree nodes, pulled toward a leaf on demand.

    ``classify(strings, midfix)`` reads a node: it should batch the family queries
    for ``strings`` at ``midfix`` and return one decision per string.
    """

    def __init__(self, tree, classify: Classify, *, chunk: int = 128):
        self._tree = tree
        self._classify = classify
        self._chunk = chunk
        # path -> strings currently resting at that node.
        self._at: Dict[Path, List[bytes]] = {}
        # Where each string the population holds is resting.  A lookup per add,
        # which a scan of the node would make quadratic in the pool.
        self._held: Dict[bytes, Path] = {}

    def add(self, string, at: Path = ()) -> None:
        """Add ``string`` to the population resting at node ``at`` -- the root by
        default (pooled, leaf unknown), or a leaf the caller has already sifted.

        A string the population already holds is never held twice.  Members are
        counted as independent evidence about a state, so a second copy is not a
        second member however it arrived: seeded twice at one leaf, or added twice
        at the root and pushed down together.  A leaf-targeted add of a held
        string moves it there instead, the caller having sifted it further than
        the pull has."""
        resting = self._held.get(string)
        if resting == at:
            return
        if resting is not None:
            if not at:
                return  # a root add never drags a sifted string back up
            self._at[resting].remove(string)
        self._held[string] = at
        self._at.setdefault(at, []).append(string)

    def members(self, at: Path, count: int) -> List[bytes]:
        """Up to ``count`` strings reaching leaf ``at``, pulling from ancestors as
        needed and stopping as soon as ``count`` are in hand."""
        self._fill(at, count)
        return self._at.get(at, [])[:count]

    def representative(self, at: Path, count: int) -> Optional[bytes]:
        """The canonical member reaching leaf ``at`` -- the shortest, ties broken
        lexicographically -- or ``None`` if none do. ``count`` bounds how many
        members are pulled to choose among."""
        members = self.members(at, count)
        return min(members, key=lambda m: (len(m), m)) if members else None

    def _fill(self, at: Path, count: int) -> None:
        """Pull strings down into ``at`` until it holds ``count`` or its ancestors
        are exhausted."""
        while len(self._at.get(at, ())) < count:
            if not at:
                return  # the root has no parent to pull from
            parent = at[:-1]
            if not self._at.get(parent):
                self._fill(parent, self._chunk)  # get a chunk into the parent first
                if not self._at.get(parent):
                    return  # ancestors exhausted -- at holds all it ever will
            self._push_chunk(parent)

    def _push_chunk(self, parent: Path) -> None:
        """Classify one chunk of ``parent``'s strings and drop each into its
        child; indecisive strings fall out of the population."""
        bucket = self._at[parent]
        chunk, self._at[parent] = bucket[: self._chunk], bucket[self._chunk :]
        midfix = self._tree.midfix_at(parent)
        decisions = self._classify(chunk, midfix)
        for string, decision in zip(chunk, decisions):
            if decision is None:
                del self._held[string]
            else:
                child = parent + (decision,)
                self._held[string] = child
                self._at.setdefault(child, []).append(string)
