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
cut cannot place, is blocked.  After the pass the hypothesis is frozen and read on
the gate's draws; a round whose draws are blocked often enough leaves what they
met to a population that replays the reading (see FrozenCheck).

Each state's prefixes -- the pool prefixes that sift to its leaf -- live in a
:class:`~orthogonal_dfa.l_star.leaf_population.LeafPopulation`.  The split keeps
the accept side as s itself and gives the reject side a fresh id (see
MidfixTree.split), so state ids stay a dense range(num_states) and need no
remapping on export.
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import List

from automata.fa.dfa import DFA

from .edge_resolver import EdgeResolver
from .leaf_population import LeafPopulation
from .lstar import SequentialRate
from .midfix_tree import MidfixTree, fmt_seq
from .partial_dfa import PartialDFA
from .prefix_sources import UniformSource
from .progress import counter, write
from .provenance import Provenance, Read, ReadBlocked, WalkBlocked
from .sifting import EDGE, PROBE_BLOCK, Sifter, check_from, read_from, walk_from
from .split_evidence import _MEMBER_LIMIT, SPLIT, SplitEvidence
from .suffix_family import SuffixFamily

# Outcome of processing one probe (see TransitionResolver.counterexample_pass).
_RESOLVED = 0  # clean probe, or one no read left undecided the check passed by
_SPLIT = 1  # the leaf bifurcated decisively; a split was applied
_UNDECIDED = 2  # evidence not yet conclusive -- keep sifting to accumulate members
_UNCHECKED = 3  # counted clean, but indecision kept the probe from being checked

#: Which reading blocked a round, naming the population it leaves.
WALK, CHECK = "walk", "check"


def start_length(length: int) -> int:
    """k, where the pass anchors a probe of ``length``: a uniform draw's first k
    symbols are as many strings as `aim_at` asks a leaf to hold at full length,
    and the rest of the probe is at least as long for the walk."""
    return math.ceil(length / 2)


@dataclass
class Blocked:
    """The ``kind`` of reading that blocked a round, the distinct strings its
    blocked draws left, and the replay that draws more of them."""

    kind: str
    found: List[bytes]
    replay: Provenance


def prefill_walks(sifter, probes, transitions, k) -> None:
    """Warm the cache for the starts of ``probes`` and then for the whole of each
    whose walk from ``k`` already goes through on what the cache holds."""
    sifter.prefill([w[:k] for w in probes])

    def known(seq):
        return sifter.known_sift(seq), None

    sifter.prefill(
        [w for w in probes if walk_from(w, known, transitions, k)[1] is None]
    )


class FrozenCheck:
    """A round's hypothesis, frozen, reading fresh draws.

    Each draw is walked from ``k`` and, where ``whole``, sifted whole too.  The
    share of draws that blocks is tested against ``1 - acc_threshold`` as the
    gate tests its agreement; where ``whole``, so is the share of the rest whose
    walk ends where the sift lands, against ``acc_threshold``, and the draws whose
    walk and sift disagree are kept."""

    def __init__(self, resolver, *, acc_threshold, whole):
        self._resolver = resolver
        self._transitions = resolver.learned()
        self._whole = whole
        self.blocked = SequentialRate(1 - acc_threshold, min_draws=1)
        self.agreement = SequentialRate(acc_threshold, min_draws=30)
        self.rates = (self.agreement, self.blocked) if whole else (self.blocked,)
        self._found = {}
        self.draws = []
        self.disagreements = []

    def prefill(self, draws) -> None:
        sifter, k = self._resolver.sifter, self._resolver.k
        if self._whole:
            prefill_walks(sifter, draws, self._transitions, k)
        else:
            sifter.prefill([w[:k] for w in draws])

    def observe(self, draw) -> None:
        self.draws.append(draw)
        sift, k = self._resolver.sift_and_harvest, self._resolver.k
        if self._whole:
            _, block, disagrees = read_from(draw, sift, self._transitions, k)
        else:
            (_, block), disagrees = walk_from(draw, sift, self._transitions, k), False
        if disagrees:
            self.disagreements.append(draw)
        if self.blocked.side is None:
            self.blocked.add(block is not None)
        if self._whole and block is None and self.agreement.side is None:
            self.agreement.add(not disagrees)
        if block is not None and block.found is not None:
            self._found[block.found] = None

    @property
    def blocks(self) -> bool:
        return self.blocked.above

    def outcome(self) -> List[Blocked]:
        """The population the round leaves, if any: what the blocked draws left
        where they block the round, and, where the draws were sifted whole, what
        the pass's split attempts could not place either way."""
        found = dict(self._found) if self.blocks else {}
        if self._whole:
            found.update(self._resolver.dropped)
        if not found:
            return []
        resolver = self._resolver
        replay = ReadBlocked if self._whole else WalkBlocked
        return [
            Blocked(
                CHECK if self._whole else WALK,
                list(found),
                replay(
                    UniformSource(resolver.pst),
                    resolver.sifter,
                    self._transitions,
                    resolver.k,
                ),
            )
        ]


class TransitionResolver:
    def __init__(self, pst, vs, draws):
        """``draws[p]`` is the draw the table prefix ``p`` was; one it does not
        name was the sampler's."""
        self.pst = pst
        #: Boundary strings the family could not place, each with the read that
        #: met it.
        self.indecisive = {}
        sampler = UniformSource(pst)
        #: A probe, or a string taken from one, is read the way the pass walks it.
        self._walked = Read(sampler, None)
        #: Probes since the pass's last split, how many of them were unchecked, and
        #: the node reads they made.
        self.quiet_probes = 0
        self.unchecked_quiet_probes = 0
        self.quiet_reads = 0
        self.k = start_length(pst.sampler.length)
        #: The pass's last fresh probes, for the export to pick its start on.
        self.recent = deque()
        #: What the pass's split attempts could not place.
        self.dropped = {}
        self.family = SuffixFamily(pst, vs)
        self.tree = MidfixTree([pst.table.suffix(i) for i in vs])
        self.sifter = Sifter(self.tree, self.family)
        self.population = LeafPopulation(
            self.tree,
            self._classify,
            harvest=self._harvest,
        )
        for p in pst.table.prefixes:
            self.population.add(p, draw=draws.get(p, Read(sampler, b"")))
        self.splits = SplitEvidence(
            pst,
            self.family,
            population=self.population,
            tree=self.tree,
        )
        self.dfa = PartialDFA(pst.alphabet_size, num_states=self.tree.num_states)
        self.edges = EdgeResolver(
            self.dfa, self.sifter, self._harvest, population=self.population
        )

    # -- membership / population -------------------------------------------

    def _harvest(self, boundary, read):
        self.indecisive.setdefault(boundary, read)

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

    def sift_and_harvest(self, seq):
        """``(leaf, boundary)`` as the sifter gives them, harvesting the boundary:
        a string the tree cannot place is one the current family straddles, and
        the driver feeds these back so the next family is forced to resolve
        them."""
        leaf, boundary = self.sifter.sift_and_boundary(seq)
        if leaf is None:
            self._harvest(boundary, self._walked)
        return leaf, boundary

    def _sift(self, seq):
        return self.sift_and_harvest(seq)[0]

    def _middle(self, seq):
        return self.sifter.halfway(seq)[0]

    def _split(self, state_id, midfix):
        # The population re-sifts state_id's prefixes on the next members() call.
        new_id = self.tree.split(state_id, midfix)
        self.dfa.split_state(state_id, new_id)
        write(
            f"  split state {state_id} on {fmt_seq(midfix)}: accept {state_id}, "
            f"reject {new_id} ({self.tree.num_states} states)"
        )

    # -- counterexamples ----------------------------------------------------

    def counterexample_pass(self, *, max_probes, patience, first):
        """Split in place on the disagreements probes find until ``patience``
        probes in a row go without one, starting with the probes ``first``;
        returns how many it probed."""
        self.recent = deque(self.recent, maxlen=patience)
        self.quiet_probes = self.unchecked_quiet_probes = self.quiet_reads = 0
        probed = 0
        with counter(max_probes, "Probing for counterexamples") as pbar:
            for w in self._probes(first, max_probes):
                probed += 1
                reads = self.sifter.reads
                status = self._check(w)
                if status in (_SPLIT, _UNDECIDED):
                    self.quiet_probes = self.unchecked_quiet_probes = 0
                    self.quiet_reads = 0
                else:
                    self.quiet_probes += 1
                    self.unchecked_quiet_probes += status == _UNCHECKED
                    self.quiet_reads += self.sifter.reads - reads
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
        pending = list(first[:count])
        while count > 0:
            fresh = not pending
            if fresh:
                pending = [
                    self.pst.sampler.sample(self.pst.rng, self.pst.alphabet_size)
                    for _ in range(min(PROBE_BLOCK, count))
                ]
            block, pending = pending[:PROBE_BLOCK], pending[PROBE_BLOCK:]
            prefill_walks(self.sifter, block, self.dfa.transitions, self.k)
            if fresh:
                self.recent.extend(block)
            count -= len(block)
            yield from block

    def learned(self) -> dict:
        """A copy of the edges learned so far."""
        return {s: dict(edges) for s, edges in self.dfa.transitions.items()}

    def _check(self, w):
        states, block, fd = check_from(
            w, self.sift_and_harvest, self._middle, self.dfa.transitions, self.k
        )
        if block is not None:
            if block.kind == EDGE and not block.undecided and block.found is not None:
                self.population.add(
                    block.found,
                    at=self.tree.path_of(states[block.at - 1]),
                    draw=self._walked,
                )
            return _UNCHECKED if block.undecided else _RESOLVED
        if fd is None:
            return _RESOLVED
        return self._act_on_disagreement(w, states, fd)

    def _act_on_disagreement(self, w, states, fd):
        s1, c = states[fd - 1], w[fd - 1]
        witness = self.dfa.witness(s1, c)
        sprime = w[: fd - 1]
        if self._sift(witness) != s1:
            return _RESOLVED
        landed, boundary = self.sift_and_harvest(sprime)
        if landed != s1:
            self._drop(boundary)
            return _RESOLVED
        distinguisher, undecided = self.sifter.disagreement(witness, sprime, bytes([c]))
        if distinguisher is None:
            self._drop(undecided)
            return _RESOLVED
        if self.splits.verdict(s1, distinguisher) == SPLIT:
            self._apply_split(s1, distinguisher, witness, sprime)
            return _SPLIT
        # The leaf may hold too few members of sprime's state to split on, even
        # where they rule a split out; keeping sprime, ahead of the member limit,
        # lets the next probe through that state weigh one more.
        self.population.add_first(sprime, self.tree.path_of(s1), draw=self._walked)
        return _UNDECIDED

    def _drop(self, undecided) -> None:
        if undecided is not None:
            self.dropped[undecided] = None

    def _apply_split(self, s1, distinguisher, witness, sprime):
        self._split(s1, distinguisher)
        for p in (witness, sprime):
            st = self._sift(p)
            if st is not None:
                self.population.add(p, at=self.tree.path_of(st), draw=self._walked)

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
        initial = self._best_start(transitions, accepting)

        dfa = DFA(
            states=set(range(n)),
            input_symbols=set(range(self.pst.alphabet_size)),
            transitions=transitions,
            initial_state=initial,
            final_states=accepting,
            allow_partial=False,
        )
        return dfa, self.tree

    def _best_start(self, transitions, accepting) -> int:
        """The state from which the hypothesis accepts the pass's last probes
        most often where the middle of the band at the root does; the lowest
        such id."""
        n = self.tree.num_states
        agree = [0] * n
        for w in self.recent:
            label = self.family.middle_side(w, b"")
            ends = list(range(n))
            for symbol in w:
                ends = [transitions[q][symbol] for q in ends]
            for q, end in enumerate(ends):
                agree[q] += (end in accepting) == label
        return max(range(n), key=lambda q: (agree[q], -q))
