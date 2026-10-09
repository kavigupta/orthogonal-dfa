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
from collections import Counter, namedtuple

from automata.fa.dfa import DFA

from .edge_resolver import EdgeResolver
from .leaf_population import LeafPopulation
from .midfix_tree import MidfixTree, fmt_seq
from .partial_dfa import PartialDFA
from .progress import counter, write
from .sifting import EDGE, END_UNDECIDED, PAIR, TRIPLE, UNLEARNED_EDGE, Sifter, read
from .split_evidence import _MEMBER_LIMIT, SPLIT, SplitEvidence
from .statistics import binomial_side_of_boundary
from .suffix_family import SuffixFamily

#: Most fresh draws one reading takes.
READING_DRAWS = 2000
#: Most fresh draws a refused gate reads against its start.
REFUSAL_DRAWS = 480
#: The share of a refusal sample's searches that may end in a pair.
PAIR_SHARE = 0.5
#: Chance each of a reading's tests settles on the wrong side.
READING_FAILURE_PROB = 1e-3

#: A reading of fresh draws (see ``TransitionResolver.read_fresh``): the start
#: that agrees on most of them and the share it agrees on, and on a refusal
#: sample, per harvest whose test fired the strings it left, the classes whose
#: tests fired, the ends and midfixes the cut stopped
#: below the root at where it did so too often, the draws searched down to an
#: edge, and the learned and exported edges they were read against.
Reading = namedtuple(
    "Reading",
    "start agreement sample_agreement harvests fired ends disagreements learned"
    " transitions",
)

#: The populations a refusal sample's outcomes may be held as.
TRIPLES, PAIRS, MEMBERS, OPEN_EDGES = "triple", "pair", "member", "open edge"


def _harvest(w, outcome):
    """The population an outcome of reading ``w`` is held in, if any: a
    triple's, a pair's, an unlearned edge's member, or a read an unlearned edge
    left undecided."""
    if outcome.kind == END_UNDECIDED:
        return OPEN_EDGES if outcome.at < len(w) else None
    return {TRIPLE: TRIPLES, PAIR: PAIRS, UNLEARNED_EDGE: MEMBERS}.get(outcome.kind)


def _harvested(outcome):
    """The strings an outcome leaves: a pair's two reads, or its one."""
    return outcome.string if outcome.kind == PAIR else (outcome.string,)


#: The outcomes of a search for where a decided disagreement parts.
_SEARCHED = (PAIR, EDGE, TRIPLE)


#: The draw counts the gate's tests are read at, so their failure chances add
#: over a handful of looks rather than every draw.  The first, as the gate always
#: has, waits out an early run of agreements.
_REFUSAL_LOOKS = {30 * 2**i for i in range(5)}
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
    """Whether ``hits`` of ``trials`` are significantly over ``rate``, or, where
    the test has not settled, over it at all.  Any hit is over a rate of 0, and
    none is over a rate of 1."""
    if rate <= 0:
        return hits > 0
    if rate >= 1 or not trials:
        return False
    side = _side(hits, trials, rate, 1)
    return hits > rate * trials if side is None else side


def _settled(hits, trials, rate) -> bool:
    if not 0 < rate < 1:
        return True
    return bool(trials) and _side(hits, trials, rate, 1) is not None


def _refusal_tests(sample, rates):
    """Per class, ``(hits, trials, rate)`` on ``sample``: its starts and its
    wholes the cut stops below the root at, over its draws; its triples and
    pairs, over its searches; and its other harvests, over its draws."""
    harvests = [_harvest(w, o) for w, *_, o in sample]
    searched = sum(o.kind in _SEARCHED for *_, o in sample)
    return {
        "start": (sum(s[1] is not None for s in sample), len(sample), rates["start"]),
        "end": (sum(s[2] is not None for s in sample), len(sample), rates["end"]),
        **{
            name: (
                harvests.count(name),
                searched if name in (TRIPLES, PAIRS) else len(sample),
                rate,
            )
            for name, rate in rates.items()
            if name not in ("start", "end")
        },
    }


def give_up_after(pst, *, edges, test_suffixes) -> int:
    """Three times the members a split test needs on a real wrong edge: each
    decided member moves the log Bayes factor by about ``test_suffixes`` times
    KL(1/2 + s || 1/2) at the promised signal s, against the threshold
    log(2T / split_pval) over the T ``edges``."""
    gain = test_suffixes * _kl_from_half(pst.config.min_signal_strength)
    threshold = math.log(2 * edges / pst.config.split_pval)
    return 3 * math.ceil(threshold / gain)


def _kl_from_half(signal) -> float:
    p = 0.5 + signal
    return p * math.log(2 * p) + (1 - p) * math.log(2 * (1 - p))


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
        #: Per edge, the split tests on it this round that did not split.
        self.unsplit = Counter()
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

        The exported DFA is run on each draw from every state: a start agrees on
        it where it accepts it as the middle of the band at the root does.  The
        best start's agreement is tested against ``acc_threshold``, over every
        start, at each of ``_LOOKS`` until it settles.  Where it falls short, a
        sample is read against that start (see ``_refused``)."""
        transitions = self._totalised()[0]
        n = self.tree.num_states
        agree = [0] * n
        drawn = 0
        while drawn < READING_DRAWS:
            w = self._draw()
            drawn += 1
            ends_at = self._ends_at(w, transitions)
            for q, end in enumerate(ends_at):
                agree[q] += self._accepts(end, w)
            if (
                drawn in _LOOKS
                and _side(agree[_best(agree)], drawn, acc_threshold, n) is not None
            ):
                break
        start = _best(agree)
        reading = Reading(
            start, agree[start] / drawn, None, {}, None, [], [], None, transitions
        )
        if reading.agreement >= acc_threshold:
            return reading
        return self._refused(reading, transitions)

    def _ends_at(self, w, transitions):
        """Where the exported DFA ends on ``w`` from each state."""
        ends_at = list(range(self.tree.num_states))
        for symbol in w:
            ends_at = [transitions[q][symbol] for q in ends_at]
        return ends_at

    def _accepts(self, end, w) -> bool:
        """Whether a run ending at ``end`` agrees with the root's reading of
        ``w``."""
        return (end in self.tree.accepting_leaves()) == self.family.middle_side(w, b"")

    def _refused(self, gate, transitions) -> Reading:
        """``gate`` with what a refusal sample holds.

        Each draw is read from ``k`` (see ``read``), and its start and whole
        sifted.  Each class of what the sample leaves is tested against what a
        family undecided at ``fnr_limit`` could leave by chance (see
        ``_incidental``), at each of ``_REFUSAL_LOOKS`` until all settle, and
        held where it fires.  Edges given up on (see ``given_up``) are not
        rerun."""
        learned = self.learned()
        rates = self._incidental()
        sample = []
        agreed = 0
        while len(sample) < REFUSAL_DRAWS:
            w = self._draw()
            agreed += self._accepts(self._ends_at(w, transitions)[gate.start], w)
            outcome = self._read(w, learned)
            sample.append(
                (w, self._below_root(w[: self.k]), self._below_root(w), outcome)
            )
            if len(sample) in _REFUSAL_LOOKS and all(
                _settled(*test) for test in _refusal_tests(sample, rates).values()
            ):
                break
        fired = {
            name
            for name, test in _refusal_tests(sample, rates).items()
            if _fires(*test)
        }
        ends = [
            (end, m)
            for i, end in ((1, "start"), (2, "end"))
            if end in fired
            for m in dict.fromkeys(s[i] for s in sample)
            if m is not None
        ]
        return gate._replace(
            sample_agreement=agreed / len(sample),
            harvests={
                name: list(
                    dict.fromkeys(
                        string
                        for w, *_, o in sample
                        if _harvest(w, o) == name
                        for string in _harvested(o)
                    )
                )
                for name in fired - {"start", "end"}
            },
            fired=fired,
            ends=ends,
            disagreements=[
                w
                for w, *_, o in sample
                if o.kind == EDGE and not self.given_up(o.state, w[o.at - 1])
            ],
            learned=learned,
        )

    def _incidental(self):
        """Per class, the share a family undecided at ``fnr_limit`` at each node
        could leave by chance: a sift reads at most ``depth`` nodes, one below
        the root fewer, a search sifts about log2(L - k) + 1 prefixes, and an
        unlearned edge two.  An unlearned edge's member is never chance."""
        depth, limit = self.tree.depth, self.pst.fnr_limit
        search = math.log2(self.pst.sampler.length - self.k) + 1
        return {
            "start": (depth - 1) * limit,
            "end": (depth - 1) * limit,
            TRIPLES: limit * depth * search,
            PAIRS: PAIR_SHARE,
            MEMBERS: 0,
            OPEN_EDGES: 2 * depth * limit,
        }

    def given_up(self, state, c) -> bool:
        """Whether the edge ``(state, c)`` has had ``give_up_after`` attempts to
        split on it in the round end other than in a split."""
        return self.unsplit[state, c] >= give_up_after(
            self.pst,
            edges=self.num_states * self.pst.alphabet_size,
            test_suffixes=len(self.family.test_idx),
        )

    def _read(self, w, learned):
        return read(w, self.sifter.sift_and_boundary, learned, self.k)

    def replay(self, gate, name):
        """Read a fresh draw as the ``gate``'s refusal sample did, for what it
        leaves the harvest ``name``."""
        w = self._draw()
        outcome = self._read(w, gate.learned)
        return list(_harvested(outcome)) if _harvest(w, outcome) == name else []

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
        """Weigh splitting ``s1`` on the edge into ``w[:fd]``: whether it split
        or asks for more members.  Any other end counts against the edge (see
        ``given_up``)."""
        c = w[fd - 1]
        witness = self.dfa.witness(s1, c)
        sprime = w[: fd - 1]
        distinguisher = None
        if self._sift(witness) == s1 and self._sift(sprime) == s1:
            distinguisher = self.sifter.disagreement(witness, sprime, bytes([c]))
        if (
            distinguisher is not None
            and self.splits.verdict(s1, distinguisher) == SPLIT
        ):
            for edge in [e for e in self.unsplit if e[0] == s1]:
                del self.unsplit[edge]
            self._split(s1, distinguisher)
            for p in (witness, sprime):
                st = self._sift(p)
                if st is not None:
                    self.population.add(p, at=self.tree.path_of(st))
            return True
        self.unsplit[s1, c] += 1
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
