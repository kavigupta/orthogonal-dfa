"""One string probed against a hypothesis, as in ``OrthoDFA/TallyLoop.lean``."""

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from .midfix_tree import MidfixTree
from .tripartite_search import (
    ALL_AGREE,
    DISAGREE_LO,
    EDGE,
    PAIR,
    TRIPLE,
    UNDECIDED_HI,
    UNDECIDED_LO,
    tripartite_search,
)

Path = Tuple[bool, ...]
Edge = Tuple[Path, int]
Cut = Callable[[bytes], Optional[bool]]

AGREE = "agree"
START = "start"
END = "end"
MEMBER = "member"


@dataclass(frozen=True)
class ProbeResult:
    outcome: tuple
    #: ``recordBy``: the edge crossed, the target, and the prefix there.
    record: Optional[Tuple[Tuple[Path, int, Path], bytes]]
    #: A member's unlearned edge, and the target and witness it is learned with.
    learned: Optional[Tuple[Edge, Tuple[Path, bytes]]]
    sifts: List[int]
    traversals: Counter
    undecided_by_edge: Dict[Edge, List[bytes]]
    middle: List[bytes]
    start: List[bytes]


def follow(edges, leaf: Optional[Path], letters: bytes):
    """The leaves visited, none from a ``None`` leaf, and the step whose edge is
    not learned, or ``None``."""
    if leaf is None:
        return [], None
    visited = [leaf]
    for step, letter in enumerate(letters):
        edge = edges.get((visited[-1], letter))
        if edge is None:
            return visited, step
        visited.append(edge[0])
    return visited, None


def ends(cut: Cut, tree: MidfixTree, edges, k: int, x: bytes) -> List[int]:
    """The positions ``walkCheckBy`` sifts before any search, the second only
    where the first is decided."""
    walk, open_ = follow(edges, tree.route(cut, x[:k])[1], x[k:])
    if not walk:
        return []
    if open_ is None:
        return [len(x)]
    return [k + open_ + 1, k + open_]


def probe(cut: Cut, tree: MidfixTree, edges, k: int, x: bytes) -> ProbeResult:
    routes = {}

    def route(i):
        if i not in routes:
            routes[i] = tree.route(cut, x[:i])
        return routes[i]

    def leaf(i) -> Optional[Path]:
        return route(i)[1]

    def undecided(i) -> List[bytes]:
        reads, path = route(i)
        return [reads[-1]] if path is None else []

    walk, open_ = follow(edges, leaf(k), x[k:])

    def pos_edge(i) -> Optional[Edge]:
        """``posEdgeBy``: the walk's edge out of ``i``, else out of ``i - 1``."""
        for j in (i, max(i - 1, 0)):
            steps = max(j - k, 0)
            if walk and j < len(x) and (open_ is None or steps <= open_):
                return walk[steps], x[j]
        return None

    def agrees(i):
        here = leaf(i)
        if here is None:
            return None
        at = max(i - k, 0)
        return at < len(walk) and here == walk[at]

    if open_ is None:
        hi, positions = len(x), []
    else:
        hi, positions = k + open_, [k + open_ + 1]
    if open_ is not None and leaf(hi + 1) is None:
        outcome = (END, x[: hi + 1])
    else:
        found = tripartite_search(agrees, k, hi, recover=True)
        assert found.kind != DISAGREE_LO
        positions += found.queried
        outcome = {
            UNDECIDED_LO: (START, x[:k]),
            ALL_AGREE: (AGREE,) if open_ is None else (MEMBER, x[:hi]),
            UNDECIDED_HI: (END, x[:hi]),
            EDGE: (EDGE, walk, found.position),
        }.get(found.kind, (found.kind, found.position))
    sifts = list(dict.fromkeys(i for i in positions if k < i))
    undecided_by_edge = {}
    for i in sifts:
        if pos_edge(i) is not None and undecided(i):
            undecided_by_edge.setdefault(pos_edge(i), []).extend(undecided(i))
    learned = record = None
    if outcome[0] == MEMBER:
        j = len(outcome[1])
        learned = (leaf(j), x[j]), (leaf(j + 1), outcome[1])
    if outcome[0] == EDGE:
        fd = outcome[2]
        before, target = max(fd - 1, 0), leaf(fd)
        if before < len(x) and target is not None:
            at = max(fd - 1 - k, 0)
            source = walk[at] if at < len(walk) else None
            record = (source, x[before], target), x[:before]
    return ProbeResult(
        outcome=outcome,
        record=record,
        learned=learned,
        sifts=sifts,
        traversals=Counter(
            e for e in map(pos_edge, range(k + 1, len(x) + 1)) if e is not None
        ),
        undecided_by_edge=undecided_by_edge,
        middle=undecided(outcome[1]) if outcome[0] in (PAIR, TRIPLE) else [],
        start=undecided(k),
    )
