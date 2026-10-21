"""One string probed against a hypothesis, as in ``OrthoDFA/Level5.lean``."""

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

from .hypothesis_dfa import Edge, HypothesisDFA, Path
from .tripartite_search import (
    ALL_AGREE,
    EDGE,
    UNDECIDED_HI,
    UNDECIDED_LO,
    UNDECIDED_MID,
    skipping_search,
)


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
class Middle:
    position: int
    undecided: bytes


Outcome = Union[Agree, Start, End, Member, EdgeFound, Middle]


@dataclass(frozen=True)
class Located:
    outcome: Outcome
    walk: List[Path]
    #: Each position past ``k`` sifted, in order, with what it read undecided.
    sifts: Dict[int, Optional[bytes]]
    search_queries: int


@dataclass(frozen=True)
class Charges:
    """What one probe adds to the harvest tests' counts."""

    traversals: Counter
    undecided_by_edge: Dict[Edge, List[bytes]]
    start: List[bytes]
    middle: List[bytes]
    search_queries: int


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


def ends(hypothesis: HypothesisDFA, k: int, x: bytes) -> List[int]:
    """The positions ``walkCheckBy`` sifts before any search, the second only
    where the first is decided."""
    walk, unlearned_step = follow(hypothesis.edges, hypothesis.sift(x[:k]), x[k:])
    if not walk:
        return []
    if unlearned_step is None:
        return [len(x)]
    return [k + unlearned_step + 1, k + unlearned_step]


def locate(
    hypothesis: HypothesisDFA,
    k: int,
    x: bytes,
    start: Optional[Callable[[HypothesisDFA], Optional[Path]]] = None,
) -> Located:
    """``start`` places the empty string where the cut leaves it undecided."""
    routes = {}

    def sift(i) -> Tuple[Optional[Path], Optional[bytes]]:
        """The leaf at ``i``, and the string read undecided there."""
        if i not in routes:
            reads, path = hypothesis.tree.sift_with_path(hypothesis.cut, x[:i])
            if path is None and i == 0 and start is not None:
                path = start(hypothesis)
            routes[i] = path, reads[-1] if path is None else None
        return routes[i]

    def located(outcome, positions, search_queries):
        sifts = {i: sift(i)[1] for i in positions if k < i}
        return Located(outcome, walk, sifts, search_queries)

    walk, unlearned_step = follow(hypothesis.edges, sift(k)[0], x[k:])
    hi = len(x) if unlearned_step is None else k + unlearned_step
    before = [] if unlearned_step is None else [hi + 1]
    if before and sift(hi + 1)[0] is None:
        return located(End(sift(hi + 1)[1]), before, 0)

    def agrees(i):
        here = sift(i)[0]
        return None if here is None else here == walk[max(i - k, 0)]

    found = skipping_search(agrees, k, hi)
    kind, j = found.kind, found.position
    if kind == UNDECIDED_LO:
        outcome = Start(sift(k)[1])
    elif kind == UNDECIDED_HI:
        outcome = End(sift(hi)[1])
    elif kind == ALL_AGREE and unlearned_step is None:
        outcome = Agree()
    elif kind == ALL_AGREE:
        outcome = Member((walk[-1], x[hi]), sift(hi + 1)[0], x[:hi])
    elif kind == EDGE:
        outcome = EdgeFound(walk[j - 1 - k], x[j - 1], sift(j)[0], x[: j - 1])
    else:
        assert kind == UNDECIDED_MID, kind
        outcome = Middle(j, sift(j)[1])
    return located(outcome, before + list(found.queried), len(found.queried))


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
        middle=[outcome.undecided] if isinstance(outcome, Middle) else [],
        search_queries=(
            located.search_queries if isinstance(outcome, (EdgeFound, Middle)) else 0
        ),
    )
