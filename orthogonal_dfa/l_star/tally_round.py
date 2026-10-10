"""One round of the tally loop: ``tallyStep`` of ``OrthoDFA/TallyLoop.lean``.

The round's suffix family is the oracle: a string reads accept, reject or undecided
(``cut``), and a tree node reads a string ``w`` at its midfix ``m`` as the read of
``w + m``.  Leaves are named by their paths from the root (``True`` the accepting
side), as in the Lean, and only the export names them by the tree's state ids.

Each probe is walked from position ``k`` along the learned edges and searched for its
first disagreement (``Probe``).  What it comes to is learned (an unlearned edge),
recorded (a clean disagreement, at the edge it crosses) or counted, its reads are
charged to the start and the edges, and then the tests may end the round; otherwise
one edge with ``m`` records at a target it does not point at is fixed.

Lean to Python:

    DTree, route, sift, splitAt, midAt     MidfixTree, route, TallyRound._split
    Edges                                  TallyRound.edges, (leaf, letter) -> (target, witness)
    follow, kWalkBy, walkToBy, walkCheckBy follow, Probe.__init__, Probe.walk_to, Probe.walk_check
    bracketAt, probeBy, recordBy           Probe.bracket, Probe.outcome, Probe.record
    bracketSifts, siftsBy                  Probe.bracket_sifts, Probe.sifts
    edgeAtBy, posEdgeBy, travBy            Probe.edge_at, Probe.pos_edge, Probe.traversals
    edgeHarvBy, ptHarvBy, startHarvBy      Probe.edge_harvest, Probe.middle, Probe.start
    TState, fresh, tally, Violates         TallyRound, _fresh, tally, violation
    setEdge, charge, fixEdge, retargetBy   _set_edge, _charge, _fix, _split
    tallyPre, tallyLook, settleOne         _act, _look, _settle_one
    tallyStep, TallyCfg, rateSide, TEnd    step, TallyConfig, rate_side, Ending

Where ``TallyConfig.count_reads`` is set, an edge's count is the strings its positions
read for the first time in the round, and its harvest the undecided ones among them:
the per-read test the README's planned replacement describes.  Unset, it is
``travBy`` and ``edgeHarvBy`` as proved.

Gaps.  ``tally_config`` is outside the proof's conditions here, at sample length
L = 40, |Σ| = 4, a 30-state target and S = 0:

* p₀ and k.  ``k = 0``, so every probe starts at the one string ``ε`` and p₀ = 1, where
  ``TallyRoundFamily``'s edge and middles tails are vacuous.  A larger k leaves the
  edges out of states only short prefixes reach unlearned.
* θ.  ``θe ≥ 4θg`` with ``θg ≥ 3(|Q| + S + 1)θ``, so θe = 0.1 holds only for read-states
  undecided at most θ = 2.7e-4 of the time to count as good.
* Threshold ratios.  ``HarvestGood`` needs ``exc > 4L(1 + 4θg Lmax) log(1/a)``, over
  1474 here, against exc = 10; and ``P(Bin(n, φpt/2) ≥ ⌈φpt n⌉) ≤ a`` from ``n₀`` on,
  0.14 at φpt = 0.01, n₀ = 30, against a = 1e-4.
* κ.  ``FakeRace`` needs ``ν ≥ ηκ`` roughly, and charges ``Wν(θe(L + 1) + 2 Lmax |Σ| φe)``
  against ``η m``: at the band's κ = 4e-3 and m = 5 its bound exceeds 1 past about
  W = 100 probes.
* Probe budget.  ``TallyRound`` needs ``T ≥ (S + |Q| + 1) subT`` with
  ``subT ≥ (2 Lmax |Σ| + 1) nEnd``, about 1e7 probes at the nEnd = 460 that success
  at εd = 0.02 needs; the cap is 5e4.  Its union bound
  ``(2^(Lmax+1) |Σ| + T + 3T²) a`` exceeds 1 for any T.
* Counting.  ``count_reads`` is on, which is not proved, and its edge gate compares the
  edge's reads, not its positions, to ``φe`` of the probes.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from automata.fa.dfa import DFA

from .statistics import binom_cdf, binom_sf

Path = Tuple[bool, ...]
Edge = Tuple[Path, int]
Cut = Callable[[bytes], Optional[bool]]

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


@dataclass(frozen=True)
class TallyConfig:
    k: int
    m: int
    max_leaves: int  # Lmax
    start_rate: float  # θs
    edge_rate: float  # θe
    middle_rate: float  # θpt
    disagreement_rate: float  # εd
    level: float  # a
    first_look: int  # n₀
    edge_excess: float  # exc, the same after any count
    edge_traffic: float  # φe
    search_share: float  # φpt
    max_probes: int
    count_reads: bool


def tally_config(acc_threshold: float) -> TallyConfig:
    """The round's settings; see the module's gaps."""
    return TallyConfig(
        k=0,
        m=5,
        max_leaves=100,
        start_rate=0.05,
        edge_rate=0.1,
        middle_rate=0.2,
        disagreement_rate=1 - acc_threshold,
        level=1e-4,
        first_look=30,
        edge_excess=10,
        edge_traffic=0.01,
        search_share=0.01,
        max_probes=50_000,
        count_reads=True,
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
    """The strings sifting ``z`` reads, root first, and its leaf's path, or ``None``
    where the last of them is undecided."""
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
    """The leaves visited from ``leaf`` along ``letters``, and the step at which an
    edge is not learned, or ``None``."""
    visited = [leaf]
    for step, letter in enumerate(letters):
        edge = edges.get((visited[-1], letter))
        if edge is None:
            return visited, step
        visited.append(edge[0])
    return visited, None


def _bracketed(agrees, lo, hi, i) -> Optional[bool]:
    """``agrees``, taking the bracket's ends as agreeing and disagreeing."""
    if i == lo:
        return True
    if i == hi:
        return False
    return agrees(i)


class Probe:
    """What one probe of ``x`` comes to against a tree and its edges."""

    def __init__(self, cut: Cut, root, edges, *, k: int, x: bytes):
        self._cut, self._root = cut, root
        self.k, self.x = k, x
        self._routes = {}
        start = self.leaf(k)
        self._walk, self._open = (
            ([], None) if start is None else follow(edges, start, x[k:])
        )

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
        """How many edges ``walk_to(j)`` follows, or ``None`` where it is empty."""
        if not self._walk:
            return None
        steps = min(max(j - self.k, 0), max(len(self.x) - self.k, 0))
        if self._open is not None and steps > self._open:
            return None
        return steps

    def walk_to(self, j) -> List[Path]:
        steps = self._steps(j)
        return [] if steps is None else self._walk[: steps + 1]

    def walk_check(self):
        """An outcome, or the walk and the length to search below."""
        k, x = self.k, self.x
        if not self._walk:
            return (START, x[:k]), None
        if self._open is not None:
            j = k + self._open
            if self.leaf(j + 1) is None:
                return (END, x[: j + 1]), None
            here = self.leaf(j)
            if here is None:
                return (END, x[:j]), None
            if here == self._walk[-1]:
                return (MEMBER, x[:j]), None
            return None, (self.walk_to(j), j)
        end = self.leaf(len(x))
        if end is None:
            return (END, x), None
        if end == self._walk[-1]:
            return (AGREE,), None
        return None, (self._walk, len(x))

    def _agrees(self, walk):
        def agrees(i):
            here = self.leaf(i)
            if here is None:
                return None
            at = max(i - self.k, 0)
            return at < len(walk) and here == walk[at]

        return agrees

    def bracket(self, walk, fuel, lo, hi):
        agrees = self._agrees(walk)
        while fuel > 0 and lo + 1 < hi:
            fuel -= 1
            mid = (lo + hi) // 2
            here = _bracketed(agrees, lo, hi, mid)
            if here is True:
                lo = mid
                continue
            if here is False:
                hi = mid
                continue
            left = _bracketed(agrees, lo, hi, mid - 1)
            if left is None:
                return (PAIR, mid - 1)
            right = _bracketed(agrees, lo, hi, mid + 1)
            if right is None:
                return (PAIR, mid)
            if left and not right:
                return (TRIPLE, mid)
            if left:
                lo = mid + 1
            else:
                hi = mid - 1
        return (EDGE, walk, hi)

    def bracket_sifts(self, walk, fuel, lo, hi) -> List[int]:
        agrees = self._agrees(walk)
        out = []
        while fuel > 0 and lo + 1 < hi:
            fuel -= 1
            mid = (lo + hi) // 2
            here = _bracketed(agrees, lo, hi, mid)
            out.append(mid)
            if here is True:
                lo = mid
                continue
            if here is False:
                hi = mid
                continue
            out.append(mid - 1)
            left = _bracketed(agrees, lo, hi, mid - 1)
            if left is None:
                return out
            out.append(mid + 1)
            right = _bracketed(agrees, lo, hi, mid + 1)
            if left is True and right is True:
                lo = mid + 1
            elif left is False and right is not None:
                hi = mid - 1
            else:
                return out
        return out

    def outcome(self):
        outcome, search = self.walk_check()
        if outcome is not None:
            return outcome
        walk, hi = search
        return self.bracket(walk, hi - self.k, self.k, hi)

    def record(self, outcome) -> Optional[Tuple[Tuple[Path, int, Path], bytes]]:
        """The edge a clean disagreement crosses, the leaf its next prefix sifts to,
        and its prefix at the edge."""
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

    def sifts(self) -> List[int]:
        """The positions past the start the probe sifts."""
        k, x = self.k, self.x

        def found(walk, hi):
            return [hi] + self.bracket_sifts(walk, hi - k, k, hi)

        if not self._walk:
            positions = []
        elif self._open is not None:
            j = k + self._open
            if self.leaf(j + 1) is None:
                positions = [j + 1]
            elif self.leaf(j) is None or self.leaf(j) == self._walk[-1]:
                positions = [j + 1, j]
            else:
                positions = [j + 1] + found(self.walk_to(j), j)
        else:
            end = self.leaf(len(x))
            if end is not None and end != self._walk[-1]:
                positions = found(self._walk, len(x))
            else:
                positions = [len(x)]
        return list(dict.fromkeys(i for i in positions if k < i))

    def edge_at(self, i) -> Optional[Edge]:
        steps = self._steps(i)
        if steps is not None and i < len(self.x):
            return (self._walk[steps], self.x[i])
        return None

    def pos_edge(self, i) -> Optional[Edge]:
        """The edge position ``i``'s reads are charged to."""
        return self.edge_at(i) or self.edge_at(max(i - 1, 0))

    def traversals(self) -> Counter:
        return Counter(
            e
            for e in map(self.pos_edge, range(self.k + 1, len(self.x) + 1))
            if e is not None
        )

    def edge_harvest(self, edge: Edge) -> List[bytes]:
        return [
            self.undecided(i)
            for i in self.sifts()
            if self.pos_edge(i) == edge and self.undecided(i) is not None
        ]

    def middle(self, outcome) -> List[bytes]:
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


class TallyRound:
    """The loop's state, ``TState``, over a tree it splits in place."""

    def __init__(self, config: TallyConfig, cut: Cut, tree):
        self.config, self.cut, self.tree = config, cut, tree
        self.edges: Dict[Edge, Tuple[Path, bytes]] = {}
        self.version = 0
        self.records: Dict[Edge, List[Tuple[bytes, Path]]] = {}
        self.n = 0
        self.searches = 0  # dis
        self.middles: List[bytes] = []  # pt
        self.probes = 0
        self.start_harvest: List[bytes] = []
        self.charged: Counter = Counter()  # trav, or the first reads
        self.edge_harvest: Dict[Edge, List[bytes]] = {}
        self.paths = set(leaf_paths(tree.root))
        self._seen = set()

    def probe(self, x: bytes) -> Probe:
        return Probe(self.cut, self.tree.root, self.edges, k=self.config.k, x=x)

    def sift(self, z: bytes) -> Optional[Path]:
        return route(self.cut, self.tree.root, z)[1]

    def run(self, draw: Callable[[], bytes]) -> Ending:
        for _ in range(self.config.max_probes):
            ending = self.step(draw())
            if ending is not None:
                return ending
        return Ending(OUT_OF_PROBES, [], None)

    def step(self, x: bytes) -> Optional[Ending]:
        probe = self.probe(x)
        outcome = probe.outcome()
        self._charge(probe, outcome)
        self._act(probe, outcome)
        return self._look() or self._settle_one()

    def tally(self, edge: Edge, target: Path) -> int:
        return sum(1 for _, t in self.records.get(edge, ()) if t == target)

    def violation(self) -> Optional[Tuple[Edge, Path]]:
        """An edge out of a leaf with ``m`` records at a target it does not point at."""
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

    def _charge(self, probe: Probe, outcome):
        self.n += 1
        self.middles += probe.middle(outcome)
        self.probes += 1
        self.start_harvest += probe.start()
        if not self.config.count_reads:
            self.charged.update(probe.traversals())
            for i in probe.sifts():
                edge, undecided = probe.pos_edge(i), probe.undecided(i)
                if edge is not None and undecided is not None:
                    self.edge_harvest.setdefault(edge, []).append(undecided)
            return
        self._first_reads(probe.route(probe.k)[0])
        for i in probe.sifts():
            reads, leaf = probe.route(i)
            new = self._first_reads(reads)
            edge = probe.pos_edge(i)
            if edge is None:
                continue
            self.charged[edge] += len(new)
            if leaf is None and reads[-1] in new:
                self.edge_harvest.setdefault(edge, []).append(reads[-1])

    def _first_reads(self, reads) -> List[bytes]:
        new = [z for z in dict.fromkeys(reads) if z not in self._seen]
        self._seen.update(new)
        return new

    def _act(self, probe: Probe, outcome):
        kind = outcome[0]
        if kind == MEMBER:
            j = len(outcome[1])
            if j >= len(probe.x):
                return
            source, target = probe.leaf(j), probe.leaf(j + 1)
            edge = (source, probe.x[j])
            if source is not None and target is not None and edge not in self.edges:
                self._set_edge(edge, (target, outcome[1]))
        elif kind == EDGE:
            self.searches += 1
            record = probe.record(outcome)
            if record is not None:
                (source, letter, target), prefix = record
                self.records.setdefault((source, letter), []).append((prefix, target))
        elif kind in (PAIR, TRIPLE):
            self.searches += 1

    def _look(self) -> Optional[Ending]:
        c = self.config
        if (
            rate_side(
                c.start_rate,
                c.level,
                c.first_look,
                self.probes,
                len(self.start_harvest),
            )
            is True
        ):
            return Ending(HARVEST_START, list(self.start_harvest), None)
        for edge, count in self.charged.items():
            harvest = self.edge_harvest.get(edge, [])
            if (
                edge[0] in self.paths
                and c.edge_traffic * self.probes <= count
                and c.edge_excess <= len(harvest) - c.edge_rate * count
            ):
                return Ending(HARVEST_EDGE, list(harvest), edge)
        if (
            c.search_share * self.n <= self.searches
            and rate_side(
                c.middle_rate, c.level, c.first_look, self.searches, len(self.middles)
            )
            is True
        ):
            return Ending(HARVEST_MIDDLES, list(self.middles), None)
        if (
            rate_side(c.disagreement_rate, c.level, c.first_look, self.n, self.searches)
            is False
        ):
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
        """Split where the current target also has ``m`` records, else redirect."""
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
                reads, target = route(
                    self.cut, self.tree.root, witness + bytes([letter])
                )
                if self.config.count_reads:
                    self._first_reads(reads)
                if target is None:
                    continue
            edges[(source, letter)] = (target, witness)
        self.edges = edges
        self.records = {}
        self._fresh()
        self.version += 1

    def to_dfa(self, alphabet_size: int, initial_state: int) -> DFA:
        """The hypothesis, an unlearned edge looping on its leaf."""
        root = self.tree.root
        ids = {path: leaf_id(root, path) for path in self.paths}
        transitions = {
            ids[path]: {
                c: (
                    ids[self.edges[(path, c)][0]]
                    if (path, c) in self.edges
                    else ids[path]
                )
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
    """A fresh probe of a round's last hypothesis, keeping the strings it reads
    undecided where the round's harvest was."""

    round: TallyRound
    ending: Ending
    draw: Callable[[], bytes]

    def sample(self) -> List[bytes]:
        probe = self.round.probe(self.draw())
        if self.ending.kind == HARVEST_START:
            return probe.start()
        if self.ending.kind == HARVEST_EDGE:
            return probe.edge_harvest(self.ending.edge)
        return probe.middle(probe.outcome())
