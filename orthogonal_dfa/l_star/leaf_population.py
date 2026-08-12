"""On-demand population of the discrimination tree's leaves.

Each string rests at the deepest tree node it has so far been classified to, and
is pulled further down only when a leaf below it is asked for more members than
it already holds.  So the prefix pool is sifted lazily and incrementally -- a
chunk at a time, toward the leaf being asked about -- rather than re-sifted from
the root on every query.

Keyed by *path*: the tuple of accept/reject branches from the root to a node.  A
path is stable across splits (a split only appends two child paths below the
split leaf), so this companion never has to be told the tree changed -- a former
leaf simply becomes an internal node whose resting strings get flushed through it
on the next pull.
"""

from typing import Callable, Dict, List, Optional, Tuple

Path = Tuple[bool, ...]
#: Classify a batch of strings against one node's midfix, decisions aligned with
#: the input (``None`` = indecisive, dropped).
Classify = Callable[[List[list], tuple], List[Optional[bool]]]


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
        self._at: Dict[Path, List[tuple]] = {}

    def add(self, string, at: Path = ()) -> None:
        """Add ``string`` to the population resting at node ``at`` -- the root by
        default (pooled, leaf unknown), or a leaf the caller has already sifted."""
        self._at.setdefault(at, []).append(tuple(string))

    def members(self, at: Path, count: int) -> List[list]:
        """Up to ``count`` strings reaching leaf ``at``, pulling from ancestors as
        needed and stopping as soon as ``count`` are in hand."""
        self._fill(at, count)
        return [list(s) for s in self._at.get(at, [])[:count]]

    def representative(self, at: Path, count: int) -> Optional[list]:
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
        decisions = self._classify([list(s) for s in chunk], midfix)
        for string, decision in zip(chunk, decisions):
            if decision is not None:
                self._at.setdefault(parent + (decision,), []).append(string)
