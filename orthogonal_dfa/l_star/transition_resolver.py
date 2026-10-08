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
from collections import deque, namedtuple

from automata.fa.dfa import DFA

from .edge_resolver import EdgeResolver
from .leaf_population import LeafPopulation
from .midfix_tree import MidfixTree, fmt_seq
from .partial_dfa import PartialDFA
from .progress import counter, write
from .sifting import EDGE, PAIR, TRIPLE, UNLEARNED_EDGE, Sifter, read
from .split_evidence import _MEMBER_LIMIT, SPLIT, SplitEvidence
from .statistics import binomial_side_of_boundary
from .suffix_family import SuffixFamily

#: Most fresh draws one reading takes.
READING_DRAWS = 2000
#: Chance each of a reading's tests settles on the wrong side.
READING_FAILURE_PROB = 1e-3

#: A reading of fresh draws (see ``TransitionResolver.read_fresh``): the start
#: that agrees on most of them and the share it agrees on, the boundary strings
#: of the triples its disagreements were searched down to, how many came down to
#: a pair, the ends and midfixes the cut stopped below the root at where it did
#: so too often, the disagreements searched, and the learned and exported edges
#: they were read against.
Reading = namedtuple(
    "Reading",
    "start agreement triples pairs ends disagreements learned transitions",
)

#: The outcomes of a search for where a decided disagreement parts.
_SEARCHED = (PAIR, EDGE, TRIPLE)


#: The draw counts the gate's tests are read at, so their failure chances add
#: over a handful of looks rather than every draw.  The first, as the gate always
#: has, waits out an early run of agreements.
_LOOKS = {
    *(
        30 * 2**i
        for i in range(READING_DRAWS.bit_length())
        if 30 * 2**i < READING_DRAWS
    ),
    READING_DRAWS,
}


def _side(hits, trials, rate, tests):
    """The binomial test's side at ``READING_FAILURE_PROB`` over ``tests``."""
    return binomial_side_of_boundary(
        hits, trials, rate, failure_prob=READING_FAILURE_PROB / tests
    )


def _fires(hits, trials, rate) -> bool:
    if not 0 < rate < 1 or not trials:
        return False
    side = _side(hits, trials, rate, 1)
    return hits > rate * trials if side is None else side


def _best(agree) -> int:
    return max(range(len(agree)), key=lambda q: (agree[q], -q))


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
        #: Per probe of those, the midfix its start was cut short at below the
        #: root, if it was.
        self.window = deque()
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
        """Read fresh draws against the hypothesis as it stands.

        Each draw is sifted whole once, and the exported DFA run on it from every
        state: a start agrees on it where it accepts it as the middle of the band
        at the root does.  The best start's agreement is tested against
        ``acc_threshold``, over every start.  A draw the best start so far
        disagrees on is read from ``k`` (see ``read``).  A whole sift cut short
        below the root counts against ``fnr_limit`` a node below the root on the
        deepest path; so does the start of each of the pass's last quiet probes.
        Where either comes significantly more often, the midfixes they were cut
        short at are kept.  Reading stops at the first of ``_LOOKS`` where both
        the agreement's test and the whole sifts' settle."""
        learned, transitions = self.learned(), self._totalised()[0]
        accepting = self.tree.accepting_leaves()
        n = self.tree.num_states
        incidental = (self.tree.depth - 1) * self.pst.fnr_limit
        agree = [0] * n
        deep = drawn = 0
        agrees = cut = None
        ends, outcomes = {}, []
        while drawn < READING_DRAWS:
            w = self._draw()
            drawn += 1
            boundary = self.sifter.sift_and_boundary(w)[1]
            if boundary is not None and boundary != w:
                deep += 1
                ends["end", boundary[len(w) :]] = None
            label = self.family.middle_side(w, b"")
            ends_at = list(range(n))
            for symbol in w:
                ends_at = [transitions[q][symbol] for q in ends_at]
            if (ends_at[_best(agree)] in accepting) != label:
                outcomes.append((w, self._read(w, learned)))
            for q, end in enumerate(ends_at):
                agree[q] += (end in accepting) == label
            if drawn not in _LOOKS:
                continue
            agrees = _side(agree[_best(agree)], drawn, acc_threshold, n)
            if 0 < incidental < 1:
                cut = _side(deep, drawn, incidental, 1)
            if agrees is not None and (cut is not None or not 0 < incidental < 1):
                break
        if cut is None:
            cut = 0 < incidental < 1 and deep > incidental * drawn
        start = self.window
        if not _fires(sum(m is not None for m in start), len(start), incidental):
            start = []
        return Reading(
            _best(agree),
            agree[_best(agree)] / drawn,
            list(dict.fromkeys(o.string for _, o in outcomes if o.kind == TRIPLE)),
            sum(o.kind == PAIR for _, o in outcomes),
            [
                *(ends if cut else ()),
                *{("start", m): None for m in start if m is not None},
            ],
            [w for w, o in outcomes if o.kind in _SEARCHED],
            learned,
            transitions,
        )

    def _read(self, w, learned):
        return read(w, self.sifter.sift_and_boundary, learned, self.k)

    def replay(self, gate):
        """Read a fresh draw as the ``gate`` reading did, for the middle of a
        triple."""
        w = self._draw()
        state = gate.start
        for symbol in w:
            state = gate.transitions[state][symbol]
        accepts = state in self.tree.accepting_leaves()
        if accepts == self.family.middle_side(w, b""):
            return []
        outcome = self._read(w, gate.learned)
        return [outcome.string] if outcome.kind == TRIPLE else []

    # -- counterexamples ----------------------------------------------------

    def counterexample_pass(self, *, max_probes, patience, first):
        """Split in place on the disagreements probes find until ``patience``
        probes in a row go without one, starting with the probes ``first``;
        returns how many it probed."""
        self.quiet_probes = probed = 0
        self.window = deque(maxlen=patience)
        with counter(max_probes, "Probing for counterexamples") as pbar:
            for w in self._probes(first, max_probes):
                probed += 1
                if self._check(w):
                    self.quiet_probes = 0
                    self.window.clear()
                else:
                    self.quiet_probes += 1
                    self.window.append(self._below_root(w[: self.k]))
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

    def _totalised(self):
        return self.dfa.totalise(
            range(self.tree.num_states),
            lambda s, c: self.edges.decisive_target(s, c)[0],
        )

    def to_dfa_and_tree(self, initial):
        n = self.tree.num_states

        transitions, unresolved = self._totalised()
        for state, c in unresolved:
            print(
                f"  no decisive edge for (state {state}, symbol {c}); "
                "falling back to a self-loop"
            )

        accepting = self.tree.accepting_leaves()

        dfa = DFA(
            states=set(range(n)),
            input_symbols=set(range(self.pst.alphabet_size)),
            transitions=transitions,
            initial_state=initial,
            final_states=accepting,
            allow_partial=False,
        )
        return dfa, self.tree
