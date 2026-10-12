"""``tallyStep`` of ``OrthoDFA/TallyLoop.lean``, read through the round's suffix family."""

import math
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from automata.fa.dfa import DFA

from .statistics import binom_cdf, binom_sf

Path = Tuple[bool, ...]
Edge = Tuple[Path, int]
Cut = Callable[[bytes], Optional[bool]]
Warm = Callable[[List[bytes]], None]

MAX_BLOCK = 64

#: Leaves past which a round gives up.
MAX_LEAVES = 100

#: Probes before a rate test may first decide, as the consistency estimate's.
FIRST_LOOK = 30

AGREE = "agree"
START = "start"
END = "end"
PAIR = "pair"
EDGE = "edge"
TRIPLE = "triple"
MEMBER = "member"

SUCCESS = "success"
HARVEST_START = "harvest start"
HARVEST_EDGE = "harvest edge"
HARVEST_MIDDLES = "harvest middles"
TOO_BIG = "too big"
OUT_OF_PROBES = "out of probes"
HARVESTS = (HARVEST_START, HARVEST_EDGE, HARVEST_MIDDLES)


@dataclass(frozen=True)
class TallyConfig:
    """``TallyCfg``, from the values it does not derive."""

    #: Records at one target before the edge they are at is fixed.
    m: int
    #: Chance a rate test errs at one look.
    level: float
    #: Probes a round draws before giving up.
    max_probes: int
    #: Each harvest's undecided rate, as a multiple of ``acceptable_fnr``.
    start_multiple: float
    edge_multiple: float
    middle_multiple: float
    #: Agreement the hypothesis is driven to.
    acc_threshold: float
    #: The share of prefixes the family may leave undecided.
    acceptable_fnr: float

    @property
    def k(self) -> int:
        return 0

    @property
    def max_leaves(self) -> int:
        return MAX_LEAVES

    @property
    def first_look(self) -> int:
        return FIRST_LOOK

    @property
    def disagreement_rate(self) -> float:
        return 1 - self.acc_threshold

    @property
    def start_rate(self) -> float:
        return self.start_multiple * self.acceptable_fnr

    @property
    def edge_rate(self) -> float:
        return self.edge_multiple * self.acceptable_fnr

    @property
    def middle_rate(self) -> float:
        return self.middle_multiple * self.acceptable_fnr

    @property
    def edge_excess(self) -> int:
        """An edge undecided at most ln 2 times ``edge_rate`` reaches it with
        chance under ``level``."""
        return math.ceil(math.log2(1 / self.level))

    @property
    def edge_traffic(self) -> float:
        return self.acceptable_fnr

    @property
    def search_share(self) -> float:
        return self.acceptable_fnr


def tally_config(acc_threshold: float, acceptable_fnr: float) -> TallyConfig:
    return TallyConfig(
        m=5,
        level=1e-3,
        max_probes=50_000,
        start_multiple=5,
        edge_multiple=10,
        middle_multiple=20,
        acc_threshold=acc_threshold,
        acceptable_fnr=acceptable_fnr,
    )


def rate_side(theta, a, n0, n, h) -> Optional[bool]:
    """``True`` where ``P(Bin(n, θ) ≥ h) < a``, ``False`` where ``P(Bin(n, θ) ≤ h) < a``."""
    if n < n0:
        return None
    if binom_sf(h - 1, n, theta) < a:
        return True
    if binom_cdf(h, n, theta) < a:
        return False
    return None


def route(cut: Cut, root, z: bytes) -> Tuple[List[bytes], Optional[Path]]:
    """The strings sifting ``z`` reads, and its leaf, ``None`` where the last read is undecided."""
    reads, path, node = [], (), root
    while not isinstance(node, int):
        midfix, lookup = node
        reads.append(z + midfix)
        side = cut(reads[-1])
        if side is None:
            return reads, None
        path += (side,)
        node = lookup[side]
    return reads, path


def leaf_paths(root) -> List[Path]:
    if isinstance(root, int):
        return [()]
    _, lookup = root
    return [
        (side,) + path for side in (False, True) for path in leaf_paths(lookup[side])
    ]


def leaf_id(root, path: Path) -> int:
    node = root
    for side in path:
        node = node[1][side]
    return node


def lcp(a: Path, b: Path) -> Path:
    out = []
    for x, y in zip(a, b):
        if x != y:
            break
        out.append(x)
    return tuple(out)


def follow(edges, leaf: Path, letters: bytes):
    """The leaves visited, and the step whose edge is not learned, or ``None``."""
    visited = [leaf]
    for step, letter in enumerate(letters):
        edge = edges.get((visited[-1], letter))
        if edge is None:
            return visited, step
        visited.append(edge[0])
    return visited, None


def tripartite_search(agrees, fuel, lo, hi):
    """``bracketAt`` and ``bracketSifts`` at once: ``(kind, position, sifted)``.

    ``agrees`` is ``None`` where a read is undecided; ``lo`` counts as agreeing and
    ``hi`` as disagreeing whatever it says."""

    def at(i):
        return True if i == lo else False if i == hi else agrees(i)

    sifted = []
    while fuel > 0 and lo + 1 < hi:
        fuel -= 1
        mid = (lo + hi) // 2
        sifted.append(mid)
        here = at(mid)
        if here is not None:
            lo, hi = (mid, hi) if here else (lo, mid)
            continue
        sifted.append(mid - 1)
        left = at(mid - 1)
        if left is None:
            return PAIR, mid - 1, sifted
        sifted.append(mid + 1)
        right = at(mid + 1)
        if right is None:
            return PAIR, mid, sifted
        if left and not right:
            return TRIPLE, mid, sifted
        lo, hi = (mid + 1, hi) if left else (lo, mid - 1)
    return EDGE, hi, sifted


class Probe:
    def __init__(self, cut: Cut, root, edges, *, k: int, x: bytes):
        self._cut, self._root = cut, root
        self.k, self.x = k, x
        self._routes = {}
        start = self.leaf(k)
        self._walk, self._open = (
            ([], None) if start is None else follow(edges, start, x[k:])
        )
        self._resolved = None

    def route(self, i):
        if i not in self._routes:
            self._routes[i] = route(self._cut, self._root, self.x[:i])
        return self._routes[i]

    def leaf(self, i) -> Optional[Path]:
        return self.route(i)[1]

    def undecided(self, i) -> Optional[bytes]:
        reads, path = self.route(i)
        return reads[-1] if path is None else None

    def _steps(self, j) -> Optional[int]:
        """Edges ``walkToBy j`` follows, ``None`` where it is empty."""
        if not self._walk:
            return None
        steps = min(max(j - self.k, 0), max(len(self.x) - self.k, 0))
        if self._open is not None and steps > self._open:
            return None
        return steps

    def ends(self) -> List[int]:
        """The positions ``walkCheckBy`` sifts before any search, the second only
        where the first is decided."""
        if not self._walk:
            return []
        if self._open is None:
            return [len(self.x)]
        return [self.k + self._open + 1, self.k + self._open]

    def _resolve(self):
        """``probeBy`` and ``siftsBy``: the outcome, and the positions it sifts."""
        k, x = self.k, self.x
        if not self._walk:
            return (START, x[:k]), []
        if self._open is not None:
            j = k + self._open
            if self.leaf(j + 1) is None:
                return (END, x[: j + 1]), [j + 1]
            here = self.leaf(j)
            if here is None:
                return (END, x[:j]), [j + 1, j]
            if here == self._walk[-1]:
                return (MEMBER, x[:j]), [j + 1, j]
            walk, hi, before = self._walk[: self._steps(j) + 1], j, [j + 1]
        else:
            end = self.leaf(len(x))
            if end is None:
                return (END, x), [len(x)]
            if end == self._walk[-1]:
                return (AGREE,), [len(x)]
            walk, hi, before = self._walk, len(x), []

        def agrees(i):
            here = self.leaf(i)
            if here is None:
                return None
            at = max(i - k, 0)
            return at < len(walk) and here == walk[at]

        kind, j, sifted = tripartite_search(agrees, hi - k, k, hi)
        outcome = (EDGE, walk, j) if kind == EDGE else (kind, j)
        return outcome, before + [hi] + sifted

    def outcome(self):
        if self._resolved is None:
            self._resolved = self._resolve()
        return self._resolved[0]

    def sifts(self) -> List[int]:
        if self._resolved is None:
            self._resolved = self._resolve()
        return list(dict.fromkeys(i for i in self._resolved[1] if self.k < i))

    def record(self) -> Optional[Tuple[Tuple[Path, int, Path], bytes]]:
        """``recordBy``: the edge crossed, the target, and the prefix there."""
        outcome = self.outcome()
        if outcome[0] != EDGE:
            return None
        _, walk, fd = outcome
        before = max(fd - 1, 0)
        target = self.leaf(fd)
        if before >= len(self.x) or target is None:
            return None
        at = max(fd - 1 - self.k, 0)
        source = walk[at] if at < len(walk) else None
        return (source, self.x[before], target), self.x[:before]

    def pos_edge(self, i) -> Optional[Edge]:
        """``posEdgeBy``: the walk's edge out of ``i``, else out of ``i - 1``."""
        for j in (i, max(i - 1, 0)):
            steps = self._steps(j)
            if steps is not None and j < len(self.x):
                return (self._walk[steps], self.x[j])
        return None

    def traversals(self) -> Counter:
        return Counter(
            e
            for e in map(self.pos_edge, range(self.k + 1, len(self.x) + 1))
            if e is not None
        )

    def undecided_by_edge(self) -> Dict[Edge, List[bytes]]:
        out = {}
        for i in self.sifts():
            edge, undecided = self.pos_edge(i), self.undecided(i)
            if edge is not None and undecided is not None:
                out.setdefault(edge, []).append(undecided)
        return out

    def middle(self) -> List[bytes]:
        outcome = self.outcome()
        if outcome[0] in (PAIR, TRIPLE) and self.undecided(outcome[1]) is not None:
            return [self.undecided(outcome[1])]
        return []

    def start(self) -> List[bytes]:
        undecided = self.undecided(self.k)
        return [] if undecided is None else [undecided]


@dataclass
class Ending:
    kind: str
    harvest: List[bytes]
    edge: Optional[Edge]


def harvest_reader(ending: Ending) -> Callable[[Probe], List[bytes]]:
    """What a probe reads undecided where ``ending`` harvested."""
    if ending.kind == HARVEST_START:
        return Probe.start
    if ending.kind == HARVEST_EDGE:
        return lambda probe: probe.undecided_by_edge().get(ending.edge, [])
    return Probe.middle


class TallyRound:
    """``TState``, over a tree it splits in place."""

    def __init__(self, config: TallyConfig, cut: Cut, warm: Warm, tree):
        self.config, self.cut, self.warm, self.tree = config, cut, warm, tree
        self.edges: Dict[Edge, Tuple[Path, bytes]] = {}
        self.version = 0
        self.records: Dict[Edge, List[Tuple[bytes, Path]]] = {}
        self.n = 0
        self.searches = 0
        self.middles: List[bytes] = []
        self.probes = 0
        self.start_harvest: List[bytes] = []
        self.charged: Counter = Counter()
        self.edge_harvest: Dict[Edge, List[bytes]] = {}
        self.paths = set(leaf_paths(tree.root))

    def probe(self, x: bytes) -> Probe:
        return Probe(self.cut, self.tree.root, self.edges, k=self.config.k, x=x)

    def sift(self, z: bytes) -> Optional[Path]:
        return route(self.cut, self.tree.root, z)[1]

    def run(self, draw: Callable[[], bytes]) -> Ending:
        """``step`` over ``max_probes`` draws."""
        for x in self._blocks(draw):
            ending = self.step(x)
            if ending is not None:
                return ending
        return Ending(OUT_OF_PROBES, [], None)

    def _blocks(self, draw: Callable[[], bytes]):
        # Optimisation only: pre-reads, in batches, what the next probes will read.
        # Not trajectory-neutral: it draws a block ahead, and a block that sees a
        # change has read for its later probes under the old hypothesis.
        block, done = 1, 0
        while done < self.config.max_probes:
            version = self.version
            settle = next(
                (m for m in range(1, MAX_BLOCK) if self._settles_below(self.n + m)),
                MAX_BLOCK,
            )
            size = min(block, self.config.max_probes - done, settle)
            xs = [draw() for _ in range(size)]
            ends = [(x, e) for x in xs for e in [self.probe(x).ends()] if e]
            self._warm([x[: e[0]] for x, e in ends])
            decided = [(x, e) for x, e in ends if self.sift(x[: e[0]]) is not None]
            self._warm([x[: e[1]] for x, e in decided if len(e) == 2])
            yield from xs
            done += size
            block = min(2 * block, MAX_BLOCK) if self.version == version else 1

    def _warm(self, strings: List[bytes]):
        active = [(z, self.tree.root) for z in strings]
        while active:
            active = [(z, node) for z, node in active if not isinstance(node, int)]
            self.warm([z + node[0] for z, node in active])
            active = [
                (z, node[1][side])
                for z, node in active
                for side in [self.cut(z + node[0])]
                if side is not None
            ]

    def step(self, x: bytes) -> Optional[Ending]:
        probe = self.probe(x)
        self._charge(probe)
        self._act(probe)
        return self._look() or self._settle_one()

    def tally(self, edge: Edge, target: Path) -> int:
        return sum(1 for _, t in self.records.get(edge, ()) if t == target)

    def violation(self) -> Optional[Tuple[Edge, Path]]:
        """``Violates``: an edge with ``m`` records at a target it does not point at."""
        for edge, records in self.records.items():
            if edge[0] not in self.paths:
                continue
            current = self.edges.get(edge)
            for target, count in Counter(t for _, t in records).items():
                if count >= self.config.m and (current is None or current[0] != target):
                    return edge, target
        return None

    def _fresh(self):
        self.n = 0
        self.searches = 0
        self.middles = []

    def _set_edge(self, edge: Edge, value: Tuple[Path, bytes]):
        self._fresh()
        self.edges[edge] = value
        self.version += 1

    def _charge(self, probe: Probe):
        self.n += 1
        self.middles += probe.middle()
        self.probes += 1
        self.start_harvest += probe.start()
        self.charged.update(probe.traversals())
        for edge, strings in probe.undecided_by_edge().items():
            self.edge_harvest.setdefault(edge, []).extend(strings)

    def _act(self, probe: Probe):
        outcome = probe.outcome()
        kind = outcome[0]
        if kind == MEMBER:
            j = len(outcome[1])
            if j >= len(probe.x):
                return
            source, target = probe.leaf(j), probe.leaf(j + 1)
            edge = (source, probe.x[j])
            if source is not None and target is not None and edge not in self.edges:
                self._set_edge(edge, (target, outcome[1]))
        elif kind in (EDGE, PAIR, TRIPLE):
            self.searches += 1
            record = probe.record()
            if record is not None:
                (source, letter, target), prefix = record
                self.records.setdefault((source, letter), []).append((prefix, target))

    def _above(self, theta, n, h) -> bool:
        return rate_side(theta, self.config.level, self.config.first_look, n, h) is True

    def _settles_below(self, n) -> bool:
        c = self.config
        return (
            rate_side(c.disagreement_rate, c.level, c.first_look, n, self.searches)
            is False
        )

    def _look(self) -> Optional[Ending]:
        c = self.config
        if self._above(c.start_rate, self.probes, len(self.start_harvest)):
            return Ending(HARVEST_START, list(self.start_harvest), None)
        for edge, count in self.charged.items():
            harvest = self.edge_harvest.get(edge, [])
            if (
                edge[0] in self.paths
                and c.edge_traffic * self.probes <= count
                and c.edge_excess <= len(harvest) - c.edge_rate * count
            ):
                return Ending(HARVEST_EDGE, list(harvest), edge)
        if c.search_share * self.n <= self.searches and self._above(
            c.middle_rate, self.searches, len(self.middles)
        ):
            return Ending(HARVEST_MIDDLES, list(self.middles), None)
        if self._settles_below(self.n):
            return Ending(SUCCESS, [], None)
        return None

    def _settle_one(self) -> Optional[Ending]:
        found = self.violation()
        if found is None:
            return None
        self._fix(*found)
        if len(self.paths) > self.config.max_leaves:
            return Ending(TOO_BIG, [], None)
        return None

    def _fix(self, edge: Edge, target: Path):
        """``fixEdge``: split where the current target also has ``m`` records, else redirect."""
        witness = next((w for w, t in self.records[edge] if t == target), b"")
        current = self.edges.get(edge)
        if current is not None and self.tally(edge, current[0]) >= self.config.m:
            source, letter = edge
            node = self.tree.midfix_at(lcp(target, current[0]))
            self._split(source, bytes([letter]) + node)
        else:
            self._set_edge(edge, (target, witness))

    def _split(self, leaf: Path, midfix: bytes):
        self.tree.split(leaf_id(self.tree.root, leaf), midfix)
        self.paths = set(leaf_paths(self.tree.root))
        edges = {}
        for (source, letter), (target, witness) in self.edges.items():
            if source[: len(leaf)] == leaf:
                continue
            if target == leaf:
                target = self.sift(witness + bytes([letter]))
                if target is None:
                    continue
            edges[(source, letter)] = (target, witness)
        self.edges = edges
        self.records = {}
        self._fresh()
        self.version += 1

    def to_dfa(self, alphabet_size: int, initial_state: int) -> DFA:
        """An unlearned edge loops on its leaf."""
        root = self.tree.root
        ids = {path: leaf_id(root, path) for path in self.paths}
        transitions = {
            ids[path]: {
                c: ids[self.edges.get((path, c), (path,))[0]]
                for c in range(alphabet_size)
            }
            for path in self.paths
        }
        return DFA(
            states=set(ids.values()),
            input_symbols=set(range(alphabet_size)),
            transitions=transitions,
            initial_state=initial_state,
            final_states=self.tree.accepting_leaves(),
            allow_partial=False,
        )


@dataclass(frozen=True, eq=False)
class Replay:
    """Fresh probes of a round's last hypothesis, kept as ``read`` reads them."""

    round: TallyRound
    read: Callable[[Probe], List[bytes]]
    draw: Callable[[], bytes]

    def sample(self) -> List[bytes]:
        return self.read(self.round.probe(self.draw()))
