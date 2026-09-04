"""On-demand population of the discrimination tree's leaves.

Each string rests at the deepest node it has been classified to, and is pulled
further down only when a leaf below it is asked for more members than it holds --
so the pool is sifted lazily, a chunk at a time, not re-sifted from the root.

Keyed by *path*, the accept/reject branches from the root to a node.  A path is
stable across splits (a split only appends child paths below the split leaf), so
this companion never has to be told the tree changed: a former leaf becomes an
internal node whose resting strings flush through it on the next pull.
"""

from itertools import islice
from typing import Callable, Dict, List, Optional, Tuple

Path = Tuple[bool, ...]
#: An insertion-ordered set of strings; Python has no such builtin, and the order
#: matters because members() hands out a prefix of one.
OrderedSet = Dict[bytes, None]
#: Classify a batch of strings against one node's midfix, decisions aligned with
#: the input (``None`` = indecisive, dropped).
Classify = Callable[[List[bytes], bytes], List[Optional[bool]]]


class LeafPopulation:
    """Strings resting at tree nodes, pulled toward a leaf on demand.

    ``classify(strings, midfix)`` reads a node: it should batch the family queries
    for ``strings`` at ``midfix`` and return one decision per string.
    """

    def __init__(
        self, tree, classify: Classify, *, harvest, decisions, chunk: int = 128
    ):
        self._tree = tree
        self._classify = classify
        self._chunk = chunk
        self._harvest = harvest
        self._decisions = decisions
        # path -> strings currently resting at that node.
        self._at: Dict[Path, OrderedSet] = {}

    def add(self, string, at: Path = ()) -> None:
        """Add ``string`` to the population resting at node ``at`` -- the root by
        default (pooled, leaf unknown), or a leaf the caller has already sifted.

        A string the population already holds is never held twice.  Members are
        counted as independent evidence about a state, so a second copy is not a
        second member however it arrived: seeded twice at one leaf, or added twice
        at the root and pushed down together.  A leaf-targeted add of a held
        string moves it there instead, the caller having sifted it further than
        the pull has."""
        resting = self.resting_at(string)
        if resting == at:
            return
        if resting is not None:
            if not at:
                return  # a root add never drags a sifted string back up
            del self._at[resting][string]
        self._at.setdefault(at, {})[string] = None

    def members(self, at: Path, count: int) -> List[bytes]:
        """Up to ``count`` strings reaching leaf ``at``, pulling from ancestors as
        needed and stopping as soon as ``count`` are in hand."""
        self._fill(at, count)
        return self._held(at, count)

    def representative(self, at: Path, count: int) -> Optional[bytes]:
        """The canonical member already resting at leaf ``at`` -- the shortest,
        ties broken lexicographically -- or ``None`` if none are. ``count``
        bounds how many are read to choose among.

        Reads rather than descends: descending harvests, and a render must not
        decide what the next round samples.
        """
        resting = self._held(at, count)
        return min(resting, key=lambda m: (len(m), m)) if resting else None

    def _held(self, at: Path, count: int) -> List[bytes]:
        return list(islice(self._at.get(at, ()), count))

    def __len__(self) -> int:
        """Strings the population holds, wherever they rest."""
        return sum(len(held) for held in self._at.values())

    def resting_at(self, string) -> Optional[Path]:
        """Where ``string`` rests, or ``None`` if the population does not hold it
        -- never added, or dropped as indecisive."""
        return next((p for p, held in self._at.items() if string in held), None)

    def settle(self, string, at: Path) -> bool:
        """Push ``string`` toward ``at`` and say whether it came to rest there.

        Asked of one string rather than a leaf, because a caller aiming at a
        state wants to know about the string it aimed, not to fill the leaf."""
        while True:
            resting = self.resting_at(string)
            if resting is None or resting == at:
                return resting == at
            # Only a node ``at`` hangs below can be pushed toward it; anywhere
            # else is where the string came to rest, which is the answer.
            if resting != at[: len(resting)]:
                return False
            self._push_chunk(resting)

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
        child; indecisive strings leave the population, harvested."""
        bucket = self._at[parent]
        chunk = list(islice(bucket, self._chunk))
        for string in chunk:
            del bucket[string]
        midfix = self._tree.midfix_at(parent)
        decisions = self._classify(chunk, midfix)
        for string, decision in zip(chunk, decisions):
            self._decisions.record(parent, decision is not None)
            if decision is not None:
                self._at.setdefault(parent + (decision,), {})[string] = None
            else:
                # The indecision is over string + midfix + v, so string + midfix
                # is what failed, not string.
                self._harvest(string + midfix)
