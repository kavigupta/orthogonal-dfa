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
anchored where the cut sifts its first ``k`` symbols and walked along the learned
edges only; where the walk's end and a fresh sift of the probe disagree, the
probe has exhibited two prefixes that reach one leaf yet behave differently under
one more symbol -- a Myhill-Nerode counterexample -- so that leaf is split (see
SplitEvidence).  A split drops the edges it made ambiguous; both they and the new
leaf's edges then read as unresolved and are refilled on the next resolve pass.

A probe the cut cannot anchor, whose walk meets an open edge, or whose sift the
cut cannot place, is blocked.  The round's hypothesis is read on fresh draws
before the pass and after it (see ``read_fresh``); one whose draws block more
often than the FNR limit allows leaves what they met to a population that
replays the reading.

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
from .sifting import Sifter, first_disagreeing_edge, walk
from .split_evidence import _MEMBER_LIMIT, SPLIT, SplitEvidence
from .statistics import binomial_side_of_boundary
from .suffix_family import SuffixFamily

#: Most fresh draws one reading takes.
READING_DRAWS = 2000
#: Chance each of a reading's tests settles on the wrong side.
READING_FAILURE_PROB = 1e-3

#: A reading of fresh draws (see ``TransitionResolver.read_fresh``): whether
#: they blocked the round, the share of them the hypothesis agrees on, what the
#: blocked ones left, the ones whose walk and sift disagree decidedly, every
#: draw, and the learned edges they were read against.
Reading = namedtuple("Reading", "blocks agreement found disagreements draws learned")


def start_length(length: int) -> int:
    """k, where the pass anchors a probe of ``length``: a uniform draw's first k
    symbols are as many strings as `aim_at` asks a leaf to hold at full length,
    and the rest of the probe is at least as long for the walk."""
    return math.ceil(length / 2)


class TransitionResolver:
    def __init__(self, pst, vs):
        self.pst = pst
        #: Probes since the pass's last split or undecided split test.
        self.quiet_probes = 0
        self.k = start_length(pst.sampler.length)
        #: What the pass's split attempts could not place.
        self.dropped = {}
        self.family = SuffixFamily(pst, vs)
        self.tree = MidfixTree([pst.table.suffix(i) for i in vs])
        self.sifter = Sifter(self.tree, self.family)
        self.population = LeafPopulation(self.tree, self._classify)
        for p in pst.table.prefixes:
            self.population.add(p)
        self.splits = SplitEvidence(
            pst,
            self.family,
            population=self.population,
            tree=self.tree,
        )
        self.dfa = PartialDFA(pst.alphabet_size, num_states=self.tree.num_states)
        self.edges = EdgeResolver(self.dfa, self.sifter, population=self.population)

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
        """Read fresh draws against the hypothesis as it stands, walking each
        from ``k`` and sifting it whole.

        A blocked draw counts once against the node reads it made, and the
        draws block the round where they come to more than ``fnr_limit`` of the
        reads: as many as a family undecided at the limit could leave, or where
        the agreement falls short with no decided disagreement among them.  A
        draw agrees where the learned edges, from where the middle of the band
        places its start, take it where the middle places it whole; an open edge
        on the way disagrees.  Reading stops once both tests settle."""
        learned = self.learned()
        fnr_limit = self.pst.fnr_limit
        blocked = reads = agreed = 0
        found, disagreements, draws = {}, [], []
        blocks = agrees = None
        while len(draws) < READING_DRAWS:
            w = self._draw()
            draws.append(w)
            before = self.sifter.reads
            states, block, end = walk(w, self.sifter.sift_and_boundary, learned, self.k)
            reads += self.sifter.reads - before
            if block is not None:
                blocked += 1
                found[block.found] = None
            elif end != states[-1]:
                disagreements.append(w)
            agreed += self._ends_at_middle(w, learned)
            blocks = binomial_side_of_boundary(
                blocked, reads, fnr_limit, failure_prob=READING_FAILURE_PROB
            )
            # As the gate always has, before an early run of agreements can stop it.
            if len(draws) >= 30:
                agrees = binomial_side_of_boundary(
                    agreed, len(draws), acc_threshold, failure_prob=READING_FAILURE_PROB
                )
            if blocks is not None and agrees is not None:
                break
        if blocks is None:
            blocks = blocked > fnr_limit * reads
        # A refusal with no decided disagreement for the pass to rerun.
        if agreed < acc_threshold * len(draws) and not disagreements:
            blocks = True
        return Reading(
            blocks, agreed / len(draws), list(found), disagreements, draws, learned
        )

    def _ends_at_middle(self, w, learned) -> bool:
        state = self.sifter.halfway(w[: self.k])
        for symbol in w[self.k :]:
            state = learned[state].get(symbol)
            if state is None:
                return False
        return state == self.sifter.halfway(w)

    def replay(self, learned):
        """Read a fresh draw against the ``learned`` edges as ``read_fresh``
        does, for what it leaves where it blocks."""
        _, block, _ = walk(self._draw(), self.sifter.sift_and_boundary, learned, self.k)
        return [] if block is None else [block.found]

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
        states, block, end = walk(
            w, self.sifter.sift_and_boundary, self.dfa.transitions, self.k
        )
        if block is not None:
            if not block.undecided:
                self.population.add(block.found, at=self.tree.path_of(states[-1]))
            return False
        if end == states[-1]:
            return False
        fd = first_disagreeing_edge(
            w, states, self.sifter.halfway, self.k, len(states) - 1
        )
        return self._act_on_disagreement(w, states, fd)

    def _act_on_disagreement(self, w, states, fd) -> bool:
        s1, c = states[fd - 1], w[fd - 1]
        witness = self.dfa.witness(s1, c)
        sprime = w[: fd - 1]
        if self._sift(witness) != s1:
            return False
        landed, boundary = self.sifter.sift_and_boundary(sprime)
        if landed != s1:
            self._drop(boundary)
            return False
        distinguisher, undecided = self.sifter.disagreement(witness, sprime, bytes([c]))
        if distinguisher is None:
            self._drop(undecided)
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

    def _drop(self, undecided) -> None:
        if undecided is not None:
            self.dropped[undecided] = None

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
