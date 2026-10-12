"""One string probed against a hypothesis, as in ``OrthoDFA/TallyLoop.lean``."""

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

from .midfix_tree import MidfixTree
from .tripartite_search import (
    ALL_AGREE,
    DISAGREE_LO,
    EDGE,
    PAIR,
    UNDECIDED_HI,
    UNDECIDED_LO,
    tripartite_search,
)

Path = Tuple[bool, ...]
Edge = Tuple[Path, int]
Cut = Callable[[bytes], Optional[bool]]


@dataclass(frozen=True)
class Agree:
    pass


@dataclass(frozen=True)
class Start:
    undecided: bytes


@dataclass(frozen=True)
class End:
    undecided: bytes


@dataclass(frozen=True)
class Member:
    edge: Edge
    target: Path
    witness: bytes


@dataclass(frozen=True)
class EdgeFound:
    source: Path
    letter: int
    target: Path
    prefix: bytes


@dataclass(frozen=True)
class Pair:
    position: int
    undecided: bytes


@dataclass(frozen=True)
class Triple:
    position: int
    undecided: bytes


Outcome = Union[Agree, Start, End, Member, EdgeFound, Pair, Triple]


@dataclass(frozen=True)
class Located:
    outcome: Outcome
    walk: List[Path]
    #: Each position past ``k`` sifted, in order, with what it read undecided.
    sifts: Dict[int, Optional[bytes]]


@dataclass(frozen=True)
class Charges:
    """What one probe adds to the harvest tests' counts."""

    traversals: Counter
    undecided_by_edge: Dict[Edge, List[bytes]]
    start: List[bytes]
    middle: List[bytes]


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


def locate(cut: Cut, tree: MidfixTree, edges, k: int, x: bytes) -> Located:
    routes = {}

    def sift(i) -> Tuple[Optional[Path], Optional[bytes]]:
        """The leaf at ``i``, and the string read undecided there."""
        if i not in routes:
            reads, path = tree.route(cut, x[:i])
            routes[i] = path, reads[-1] if path is None else None
        return routes[i]

    def located(outcome, positions):
        return Located(outcome, walk, {i: sift(i)[1] for i in positions if k < i})

    walk, open_ = follow(edges, sift(k)[0], x[k:])
    hi = len(x) if open_ is None else k + open_
    before = [] if open_ is None else [hi + 1]
    if before and sift(hi + 1)[0] is None:
        return located(End(sift(hi + 1)[1]), before)

    def agrees(i):
        here = sift(i)[0]
        return None if here is None else here == walk[max(i - k, 0)]

    found = tripartite_search(agrees, k, hi, recover=True)
    kind, j = found.kind, found.position
    assert kind != DISAGREE_LO
    if kind == UNDECIDED_LO:
        outcome = Start(sift(k)[1])
    elif kind == UNDECIDED_HI:
        outcome = End(sift(hi)[1])
    elif kind == ALL_AGREE and open_ is None:
        outcome = Agree()
    elif kind == ALL_AGREE:
        outcome = Member((walk[-1], x[hi]), sift(hi + 1)[0], x[:hi])
    elif kind == EDGE:
        outcome = EdgeFound(walk[j - 1 - k], x[j - 1], sift(j)[0], x[: j - 1])
    else:
        outcome = (Pair if kind == PAIR else Triple)(j, sift(j)[1])
    return located(outcome, before + list(found.queried))


def charges(located: Located, k: int, x: bytes) -> Charges:
    walk, outcome = located.walk, located.outcome

    def edge_out_of(j) -> Optional[Edge]:
        steps = max(j - k, 0)
        return (walk[steps], x[j]) if j < len(x) and steps < len(walk) else None

    def charged_edge(i) -> Optional[Edge]:
        """``posEdgeBy``: the walk's edge out of ``i``, else out of ``i - 1``."""
        return edge_out_of(i) or edge_out_of(max(i - 1, 0))

    undecided_by_edge = {}
    for i, undecided in located.sifts.items():
        if undecided is not None and charged_edge(i) is not None:
            undecided_by_edge.setdefault(charged_edge(i), []).append(undecided)
    return Charges(
        traversals=Counter(
            e for e in map(charged_edge, range(k + 1, len(x) + 1)) if e is not None
        ),
        undecided_by_edge=undecided_by_edge,
        start=[outcome.undecided] if isinstance(outcome, Start) else [],
        middle=[outcome.undecided] if isinstance(outcome, (Pair, Triple)) else [],
    )
