"""Incremental, transition-driven DFA discovery.

Builds the discrimination tree (states) and the transition function together.

The tree starts as the initial distinguisher family v_eps, partitioning the
prefix pool into accept / reject -- two leaves, the initial two states.  Each
(state, symbol) edge is resolved by sifting the state's members extended by the
symbol: the first one the tree places, and every later one it can place from reads
already made, vote, and the edge points where most of them land, with a member that
landed there as its witness.  A leaf every one of whose members is indecisive, or
that no prefix reaches, leaves its edge open.

States beyond the initial two are found by the counterexample pass.  A probe is
read (see ``sifting.read``): anchored where the cut sifts its first ``k``
symbols, walked along the learned edges only, and sifted whole; where the walk
and the sift part over an edge, the probe has exhibited two prefixes that reach
one leaf yet behave differently under one more symbol -- a Myhill-Nerode
counterexample -- so that leaf is split (see SplitEvidence).  A split drops the
edges it made ambiguous; both they and the new leaf's edges then read as
unresolved and are refilled on the next resolve pass.

After the pass the hypothesis is read on fresh draws (see ``read_fresh``), the
round's gate.

Each state's prefixes -- the pool prefixes that sift to its leaf -- live in a
:class:`~orthogonal_dfa.l_star.leaf_population.LeafPopulation`.  The split keeps
the accept side as s itself and gives the reject side a fresh id (see
MidfixTree.split), so state ids stay a dense range(num_states) and need no
remapping on export.
"""

import math
from collections import namedtuple

from automata.fa.dfa import DFA

from .edge_resolver import EdgeResolver
from .leaf_population import LeafPopulation
from .midfix_tree import MidfixTree, fmt_seq
from .partial_dfa import PartialDFA
from .progress import counter, write
from .sifting import (
    AGREE,
    EDGE,
    END_UNDECIDED,
    PAIR,
    START_UNDECIDED,
    TRIPLE,
    UNLEARNED_EDGE,
    Sifter,
    read,
)
from .split_evidence import _MEMBER_LIMIT, SPLIT, SplitEvidence
from .statistics import binomial_side_of_boundary
from .suffix_family import SuffixFamily

#: Most fresh draws one reading takes.
READING_DRAWS = 2000
#: Chance each of a reading's tests settles on the wrong side.
READING_FAILURE_PROB = 1e-3

#: A reading of fresh draws (see ``TransitionResolver.read_fresh``): the share
#: of them the hypothesis agrees on, the boundary strings of the triples their
#: disagreements were searched down to, how many came down to a pair, the ends
#: and midfixes the cut stopped below the root at where it did so too often, the
#: ones that disagree decidedly, and the learned edges they were read against.
Reading = namedtuple("Reading", "agreement triples pairs ends disagreements learned")

#: The outcomes a read cut short ends with, and the end each was cut short at.
_CUT_SHORT = {START_UNDECIDED: "start", END_UNDECIDED: "end"}
#: The outcomes of a search for where a decided disagreement parts.
_SEARCHED = (PAIR, EDGE, TRIPLE)


def _side(hits, trials, rate):
    return binomial_side_of_boundary(
        hits, trials, rate, failure_prob=READING_FAILURE_PROB
    )


def start_length(length: int) -> int:
    """k, where the pass anchors a probe of ``length``: a uniform draw's first k
    symbols are as many strings as `aim_at` asks a leaf to hold at full length,
    and the rest of the probe is at least as long for the walk."""
    return math.ceil(length / 2)


class TransitionResolver:
    def __init__(self, pst, vs):
        self.pst = pst
        #: Boundary strings the family could not place.
        self.indecisive = set()
        #: Probes since the pass's last split or undecided split test.
        self.quiet_probes = 0
        self.k = start_length(pst.sampler.length)
        self.family = SuffixFamily(pst, vs)
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
        """Read fresh draws against the hypothesis as it stands (see ``read``).

        A draw agrees where it reads as agreeing, or, where a read is cut short,
        where the learned edges from where the middle of the band places its
        start take it where the middle places it whole.  A read cut short below
        the root counts against ``fnr_limit`` a node below the root on the
        deepest path: where those come significantly more often, the ends and
        midfixes they were cut short at are kept.  Reading stops once both tests
        settle."""
        learned = self.learned()
        incidental = (self.tree.depth - 1) * self.pst.fnr_limit
        agreed = deep = 0
        agrees = cut = None
        ends, outcomes = {}, []
        while len(outcomes) < READING_DRAWS:
            w = self._draw()
            outcome = self._read(w, learned)
            outcomes.append((w, outcome))
            agreed += outcome.kind == AGREE or (
                outcome.kind in _CUT_SHORT and self._ends_at_middle(w, learned)
            )
            if outcome.kind in _CUT_SHORT and len(outcome.string) > outcome.at:
                deep += 1
                ends[_CUT_SHORT[outcome.kind], outcome.string[outcome.at :]] = None
            # As the gate always has, before an early run of agreements can stop it.
            if len(outcomes) >= 30:
                agrees = _side(agreed, len(outcomes), acc_threshold)
            if 0 < incidental < 1:
                cut = _side(deep, len(outcomes), incidental)
            if agrees is not None and (cut is not None or not 0 < incidental < 1):
                break
        if cut is None:
            cut = 0 < incidental < 1 and deep > incidental * len(outcomes)
        return Reading(
            agreed / len(outcomes),
            list(dict.fromkeys(o.string for _, o in outcomes if o.kind == TRIPLE)),
            sum(o.kind == PAIR for _, o in outcomes),
            list(ends) if cut else [],
            [w for w, o in outcomes if o.kind in _SEARCHED],
            learned,
        )

    def _read(self, w, learned):
        return read(w, self.sifter.sift_and_boundary, learned, self.k)

    def _ends_at_middle(self, w, learned) -> bool:
        state = self.sifter.halfway(w[: self.k])
        for symbol in w[self.k :]:
            state = learned[state].get(symbol)
            if state is None:
                return False
        return state == self.sifter.halfway(w)

    def replay(self, learned):
        """Read a fresh draw against the ``learned`` edges as ``read_fresh``
        does, for the middle of a triple."""
        outcome = self._read(self._draw(), learned)
        return [outcome.string] if outcome.kind == TRIPLE else []

    # -- counterexamples ----------------------------------------------------

    def counterexample_pass(self, *, max_probes, patience, first):
        """Split in place on the disagreements probes find until ``patience``
        probes in a row go without one, starting with the probes ``first``;
        returns how many it probed."""
        self.quiet_probes = probed = 0
        with counter(max_probes, "Probing for counterexamples") as pbar:
            for w in self._probes(first, max_probes):
                probed += 1
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
        return probed

    def _probes(self, first, count):
        yield from first[:count]
        for _ in range(count - len(first[:count])):
            yield self._draw()

    def _check(self, w) -> bool:
        """Whether the probe split a leaf or asks for more of its members."""
        outcome = self._read(w, self.dfa.transitions)
        if outcome.kind == UNLEARNED_EDGE:
            self.population.add(outcome.string, at=self.tree.path_of(outcome.state))
        return outcome.kind == EDGE and self._act_on_disagreement(
            w, outcome.state, outcome.at
        )

    def _act_on_disagreement(self, w, s1, fd) -> bool:
        c = w[fd - 1]
        witness = self.dfa.witness(s1, c)
        sprime = w[: fd - 1]
        if self._sift(witness) != s1 or self._sift(sprime) != s1:
            return False
        distinguisher = self.sifter.disagreement(witness, sprime, bytes([c]))
        if distinguisher is None:
            return False
        if self.splits.verdict(s1, distinguisher) == SPLIT:
            self._split(s1, distinguisher)
            for p in (witness, sprime):
                st = self._sift(p)
                if st is not None:
                    self.population.add(p, at=self.tree.path_of(st))
            return True
        # The leaf may hold too few members of sprime's state to split on, even
        # where they rule a split out; keeping sprime, ahead of the member limit,
        # lets the next probe through that state weigh one more.
        self.population.add_first(sprime, self.tree.path_of(s1))
        return True

    # -- edge closing -------------------------------------------------------

    def close_edges(self):
        self.edges.close()

    # -- output -------------------------------------------------------------

    def to_dfa_and_tree(self):
        n = self.tree.num_states

        transitions, unresolved = self.dfa.totalise(
            range(n), lambda s, c: self.edges.decisive_target(s, c)[0]
        )
        for state, c in unresolved:
            print(
                f"  no decisive edge for (state {state}, symbol {c}); "
                "falling back to a self-loop"
            )

        accepting = self.tree.accepting_leaves()
        # Any start serves the round; the certificate tries them all.
        initial = self.sifter.halfway(b"")

        dfa = DFA(
            states=set(range(n)),
            input_symbols=set(range(self.pst.alphabet_size)),
            transitions=transitions,
            initial_state=initial,
            final_states=accepting,
            allow_partial=False,
        )
        return dfa, self.tree
