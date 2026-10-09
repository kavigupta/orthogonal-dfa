"""Incremental, transition-driven DFA discovery.

The tree starts as the initial family's cut, two leaves.  Each (state, symbol)
edge points where most of the state's members land under the symbol (see
EdgeResolver).  The counterexample pass reads probes from ``k`` (see
``sifting.read``) and splits a leaf where a probe exhibits two of its prefixes
that one more symbol tells apart (see SplitEvidence); the gate then reads the
exported DFA on fresh draws (see ``read_fresh``).  A split keeps the accept side
as the old id and gives the reject side a fresh one, so ids stay dense.
"""

import math
from collections import namedtuple

from automata.fa.dfa import DFA

from .edge_resolver import EdgeResolver
from .leaf_population import LeafPopulation
from .midfix_tree import MidfixTree, fmt_seq
from .partial_dfa import PartialDFA
from .preconditions import start_length
from .progress import counter, write
from .sifting import EDGE, END_UNDECIDED, PAIR, TRIPLE, UNLEARNED_EDGE, Sifter, read
from .split_evidence import _MEMBER_LIMIT, SPLIT, SplitEvidence
from .statistics import binomial_side_of_boundary
from .suffix_family import SuffixFamily

#: Most fresh draws one reading takes.
READING_DRAWS = 2000
#: Most fresh draws a refusal sample takes.
REFUSAL_DRAWS = 480
#: Chance each of a reading's tests settles on the wrong side.
READING_FAILURE_PROB = 1e-3

#: A reading of fresh draws (see ``TransitionResolver.read_fresh``): whether it
#: passed (None where its test was still unsettled at the last look), the start
#: that agrees on most draws and the share it agrees on; and once refused, what
#: its sample left per class whose test fired, those classes, the draws it
#: searched down to a live edge, and the edges they were read against.
Reading = namedtuple(
    "Reading",
    "passed start agreement transitions harvests fired disagreements learned",
    defaults=({}, None, [], None),
)

#: A refusal sample's draw: the midfixes the cut stops its start and its whole
#: below the root at, if any, and its read from k.
Row = namedtuple("Row", "w start end outcome")

TRIPLES, PAIRS, MEMBERS, OPEN_EDGES = "triple", "pair", "member", "open edge"
PAIR_TRIP = "pairs over half"


def _some(x) -> tuple:
    return () if x is None else (x,)


def _ended(kind):
    def left(r):
        o = r.outcome
        return () if o.kind != kind else o.string if kind == PAIR else (o.string,)

    return left


def _open_edge(r) -> tuple:
    o = r.outcome
    return (o.string,) if o.kind == END_UNDECIDED and o.at < len(r.w) else ()


#: Per class of what a refusal sample leaves: what a row leaves for it (a hit
#: where anything), whether only searched rows are trials, the rate a family
#: undecided at ``fnr_limit`` f could reach by chance at tree depth d over s
#: sifts a search reads, and whether a firing class's hits are held.
Class = namedtuple("Class", "name left searched rate held")
CLASSES = (
    Class("start", lambda r: _some(r.start), False, lambda d, f, s: (d - 1) * f, True),
    Class("end", lambda r: _some(r.end), False, lambda d, f, s: (d - 1) * f, True),
    Class(TRIPLES, _ended(TRIPLE), True, lambda d, f, s: f * d * s, True),
    Class(PAIRS, _ended(PAIR), True, lambda d, f, s: f * d * s, True),
    Class(PAIR_TRIP, _ended(PAIR), True, lambda d, f, s: 0.5, False),
    Class(MEMBERS, _ended(UNLEARNED_EDGE), False, lambda d, f, s: 0, True),
    Class(OPEN_EDGES, _open_edge, False, lambda d, f, s: 2 * d * f, True),
)


def _searched(r) -> bool:
    return r.outcome.kind in (PAIR, EDGE, TRIPLE)


def _edge(r) -> bool:
    return r.outcome.kind == EDGE


#: The draw counts the tests are read at, so their failure chances add over a
#: handful of looks rather than every draw.
_REFUSAL_LOOKS = {30 * 2**i for i in range(5)}
_LOOKS = {30 * 2**i for i in range(16) if 30 * 2**i < READING_DRAWS} | {READING_DRAWS}


def _side(hits, trials, rate, failure_prob):
    return binomial_side_of_boundary(hits, trials, rate, failure_prob=failure_prob)


def _fires(hits, trials, rate, final) -> bool:
    """Whether ``hits`` of ``trials`` are significantly over ``rate``, or, at the
    ``final`` look, over it at all where the test has not settled."""
    if rate <= 0:
        return hits > 0
    if rate >= 1 or not trials:
        return False
    side = _side(hits, trials, rate, READING_FAILURE_PROB)
    return final and hits > rate * trials if side is None else side


class TransitionResolver:
    def __init__(self, pst, vs, held_out):
        self.pst = pst
        #: Boundary strings the family could not place.
        self.indecisive = set()
        #: Probes since the pass's last split or undecided split test.
        self.quiet_probes = 0
        #: Gate readings this round.
        self.readings = 0
        #: Probes this round.
        self.probed = 0
        self.k = start_length(pst.sampler.length)
        self.family = SuffixFamily(pst, vs, held_out)
        self.tree = MidfixTree([pst.table.suffix(i) for i in vs])
        self.sifter = Sifter(self.tree, self.family)
        self.population = LeafPopulation(
            self.tree, self._classify, harvest=self.indecisive.add
        )
        for p in pst.table.prefixes:
            self.population.add(p)
        self.splits = SplitEvidence(
            pst,
            self.family,
            population=self.population,
            tree=self.tree,
        )
        self.dfa = PartialDFA(pst.alphabet_size, num_states=self.tree.num_states)
        self.edges = EdgeResolver(
            self.dfa, self.sifter, self.indecisive.add, population=self.population
        )

    # -- membership / population -------------------------------------------

    def _classify(self, strings, midfix):
        """Which side of ``midfix`` each string sits on; the indecisive band
        between the thresholds returns None and drops out of the population."""
        self.family.prefill([s + midfix for s in strings])
        return [self.family.is_accept(s, midfix) for s in strings]

    @property
    def num_states(self):
        """Leaf count; the tree allocates the ids as it splits."""
        return self.tree.num_states

    @property
    def access(self):
        """Canonical access string per state, for renderers -- the shortest known
        member reaching each leaf, or nothing where none is known."""
        reps = (
            (s, self.population.representative(self.tree.path_of(s), _MEMBER_LIMIT))
            for s in range(self.num_states)
        )
        return {s: rep for s, rep in reps if rep is not None}

    def _sift(self, seq):
        return self.sifter.sift_and_boundary(seq)[0]

    def _draw(self):
        return self.pst.sampler.sample(self.pst.rng, self.pst.alphabet_size)

    def _split(self, state_id, midfix):
        # The population re-sifts state_id's prefixes on the next members() call.
        new_id = self.tree.split(state_id, midfix)
        self.dfa.split_state(state_id, new_id)
        write(
            f"  split state {state_id} on {fmt_seq(midfix)}: accept {state_id}, "
            f"reject {new_id} ({self.tree.num_states} states)"
        )

    def learned(self) -> dict:
        """A copy of the edges learned so far."""
        return {s: dict(edges) for s, edges in self.dfa.transitions.items()}

    # -- reading fresh draws -------------------------------------------------

    def read_fresh(self, *, acc_threshold) -> Reading:
        """Run the exported DFA from every start over fresh draws, a start agreeing
        on a draw where it accepts it as the middle of the band at the root does,
        and test the best start's agreement against ``acc_threshold`` at each of
        ``_LOOKS`` until it settles."""
        transitions = self._totalised()[0]
        n = self.tree.num_states
        # The round's i-th reading at 2 ** -i of the chance, so a round's
        # readings sum to twice the first's, over every start.
        failure_prob = READING_FAILURE_PROB * 2.0**-self.readings / n
        self.readings += 1
        agree = [0] * n
        drawn = 0
        side = None
        while drawn < READING_DRAWS:
            w = self._draw()
            drawn += 1
            for q, end in enumerate(self._ends_at(w, transitions)):
                agree[q] += self._accepts(end, w)
            if drawn in _LOOKS:
                side = _side(max(agree), drawn, acc_threshold, failure_prob)
                if side is not None:
                    break
        start = max(range(n), key=lambda q: (agree[q], -q))
        return Reading(side, start, agree[start] / drawn, transitions)

    def _ends_at(self, w, transitions):
        """Where the exported DFA ends on ``w`` from each state."""
        ends_at = list(range(self.tree.num_states))
        for symbol in w:
            ends_at = [transitions[q][symbol] for q in ends_at]
        return ends_at

    def _accepts(self, end, w) -> bool:
        return (end in self.tree.accepting_leaves()) == self.family.middle_side(w, b"")

    def refusal_sample(self, gate) -> Reading:
        """``gate`` with what a sample of fresh draws, each read from ``k`` with its
        start and whole sifted, leaves per class whose test fires (see
        ``CLASSES``).  It stops at the first of ``_REFUSAL_LOOKS`` where a test
        fires or an edge has been met."""
        learned = self.learned()
        rates = (
            self.tree.depth,
            self.pst.fnr_limit,
            math.log2(self.pst.sampler.length - self.k) + 1,
        )
        sample = []
        while True:
            w = self._draw()
            outcome = self._read(w, learned)
            sample.append(
                Row(w, self._below_root(w[: self.k]), self._below_root(w), outcome)
            )
            final = len(sample) == REFUSAL_DRAWS
            if not final and len(sample) not in _REFUSAL_LOOKS:
                continue
            searched = sum(map(_searched, sample))
            fired = {
                c.name
                for c in CLASSES
                if _fires(
                    sum(bool(c.left(r)) for r in sample),
                    searched if c.searched else len(sample),
                    c.rate(*rates),
                    final,
                )
            }
            if final or fired or any(map(_edge, sample)):
                break
        return gate._replace(
            harvests={
                c.name: list(dict.fromkeys(x for r in sample for x in c.left(r)))
                for c in CLASSES
                if c.held and c.name in fired
            },
            fired=fired,
            disagreements=[r.w for r in sample if _edge(r)],
            learned=learned,
        )

    def _read(self, w, learned):
        return read(w, self.sifter.sift_and_boundary, learned, self.k)

    def replay(self, gate, name):
        """Read a fresh draw as the ``gate``'s refusal sample did, for what it
        leaves the class ``name``."""
        w = self._draw()
        row = Row(w, None, None, self._read(w, gate.learned))
        return list(next(c for c in CLASSES if c.name == name).left(row))

    # -- counterexamples ----------------------------------------------------

    def counterexample_pass(self, *, patience, first, budget):
        """Split in place on the disagreements probes find until ``patience``
        probes in a row go without one or the round has drawn ``budget`` probes,
        starting with the probes ``first``."""
        self.quiet_probes = 0
        first = iter(first)
        with counter(None, "Probing for counterexamples") as pbar:
            while self.probed < budget:
                self.probed += 1
                w = next(first, None)
                w = self._draw() if w is None else w
                self.quiet_probes = 0 if self._check(w) else self.quiet_probes + 1
                # A split drops edges and rewrites the state set, and any probe may
                # have read successors a re-vote counts.
                self.edges.close()
                pbar.set_postfix(
                    states=self.tree.num_states,
                    clean=f"{self.quiet_probes}/{patience}",
                    refresh=False,
                )
                pbar.update(1)
                if self.quiet_probes >= patience:
                    break

    def _below_root(self, seq):
        """The midfix the cut cannot place ``seq`` at below the root, if any."""
        boundary = self.sifter.sift_and_boundary(seq)[1]
        return None if boundary is None or boundary == seq else boundary[len(seq) :]

    def _check(self, w) -> bool:
        """Whether the probe split a leaf or asks for more of its members."""
        outcome = self._read(w, self.dfa.transitions)
        if outcome.kind == UNLEARNED_EDGE:
            self.population.add(outcome.string, at=self.tree.path_of(outcome.state))
        return outcome.kind == EDGE and self._act_on_disagreement(
            w, outcome.state, outcome.at
        )

    def _act_on_disagreement(self, w, s1, fd) -> bool:
        """Whether weighing a split of ``s1`` on the edge into ``w[:fd]`` split it
        or asks for more members."""
        c = w[fd - 1]
        witness, sprime = self.dfa.witness(s1, c), w[: fd - 1]
        distinguisher = None
        if self._sift(witness) == s1 and self._sift(sprime) == s1:
            distinguisher = self.sifter.disagreement(witness, sprime, bytes([c]))
        if (
            distinguisher is not None
            and self.splits.verdict(s1, distinguisher) == SPLIT
        ):
            self._split(s1, distinguisher)
            for p in (witness, sprime):
                st = self._sift(p)
                if st is not None:
                    self.population.add(p, at=self.tree.path_of(st))
            return True
        if distinguisher is None:
            return False
        # The leaf may hold too few members of sprime's state to split on, even
        # where they rule a split out; keeping sprime, ahead of the member limit,
        # lets the next probe through that state weigh one more.
        self.population.add_first(sprime, self.tree.path_of(s1))
        return True

    # -- edge closing -------------------------------------------------------

    def close_edges(self):
        self.edges.close()

    # -- output -------------------------------------------------------------

    def _totalised(self):
        return self.dfa.totalise(
            range(self.tree.num_states),
            lambda s, c: self.edges.decisive_target(s, c)[0],
        )

    def to_dfa_and_tree(self, initial):
        transitions, unresolved = self._totalised()
        for state, c in unresolved:
            print(
                f"  no decisive edge for (state {state}, symbol {c}); "
                "falling back to a self-loop"
            )

        dfa = DFA(
            states=set(range(self.tree.num_states)),
            input_symbols=set(range(self.pst.alphabet_size)),
            transitions=transitions,
            initial_state=initial,
            final_states=self.tree.accepting_leaves(),
            allow_partial=False,
        )
        return dfa, self.tree
