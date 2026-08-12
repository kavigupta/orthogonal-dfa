"""Random-walk, transition-driven DFA discovery (a "direct L*").

Grows a discrimination tree whose leaves are DFA states while maintaining a
transition function on the side.  It draws random probe strings and walks each one
through the *cached* transition function, then re-classifies the same string
directly against the tree.  Where the two disagree, the probe has on its own
exhibited two prefixes that reach the same leaf yet behave differently under one
more symbol -- a Myhill-Nerode counterexample -- and the offending leaf is split.

What the learner owns is the loop that turns probes into splits.  The pieces it
works through are their own objects:

    MidfixTree      which midfix cuts where, and which leaf a branch lands in
    PartialDFA      the edges resolved so far, their witnesses, and the worklist
    SuffixFamily    the family a round classifies against, and how to read it
    SplitEvidence   whether a proposed distinguisher really splits a leaf

Driving the learner in rounds -- resampling the family until it can place the
strings it could not -- is :mod:`orthogonal_dfa.l_star.fnr_synthesis`, which the
learner knows nothing about.

``to_dfa_and_tree`` exports the result as ``(DFA, MidfixTree)``.
"""

from typing import List, Optional, Set, Tuple

from automata.fa.dfa import DFA

from .edge_resolver import EdgeResolver
from .leaf_population import LeafPopulation
from .midfix_tree import MidfixTree, oracle_decider
from .partial_dfa import PartialDFA
from .sifting import Sifter
from .split_evidence import NO_SPLIT, SPLIT, SplitEvidence
from .suffix_family import SuffixFamily

# Outcome of processing one probe (see DirectLStarLearner.process):
_RESOLVED = 0  # clean probe, or the leaf is a single state at this distinguisher
_SPLIT = 1  # the leaf bifurcated decisively; a split was applied
_UNDECIDED = 2  # evidence not yet conclusive -- keep sifting to accumulate members

# How many probes a counterexample pass sifts per batched pass.
_PROBE_BLOCK = 16


class DirectLStarLearner:
    """Learns a DFA from random probe strings via transition/tree disagreement.

    Parameters
    ----------
    pst:
        The :class:`~orthogonal_dfa.l_star.prefix_suffix_tracker.PrefixSuffixTracker`
        providing oracle access, the prefix/suffix table, and the decision
        thresholds (``accept_thresh`` / ``reject_thresh``).
    vs:
        Row indices (into ``pst.table``) of the base suffix family -- the
        distinguishers for the root accept/reject split.  Obtain them with
        :func:`~orthogonal_dfa.l_star.cluster.sample_suffix_family`; the round
        driver in :mod:`orthogonal_dfa.l_star.fnr_synthesis` resamples them each
        round.
    """

    def __init__(
        self,
        pst,
        vs: List[int],
        *,
        split_fpr: Optional[float],
        split_miss_rate: float,
    ):
        self.pst = pst
        self.family = SuffixFamily(pst, vs)

        # The discrimination tree owns the structure -- midfixes, branches and
        # leaves -- and calls back into is_accept for every classification. Its
        # base family is the round's suffixes, so the exported tree can be re-read
        # decisively (via oracle_decider) for the accuracy estimate.
        self.tree = MidfixTree([pst.table.suffix(v) for v in vs])

        # The partial transition function, its witnesses and the queue of edges
        # still to resolve.
        self.dfa = PartialDFA(pst.alphabet_size, num_states=self.tree.num_states)

        # Classifying against the tree, and closing the transition function with
        # it.  The resolver harvests boundary strings into ``indecisive``.
        self.sifter = Sifter(self.tree, self.family)
        self.indecisive: Set[Tuple[int, ...]] = set()

        # Strings resting at tree nodes, pulled toward a leaf on demand.  Seeded
        # with the fixed prefix pool at the root (the empty string leads, pinning
        # the initial state); probe-seen members are added at the leaf they sift
        # to.  Shared by the split test and the edge resolver, and -- unlike the
        # tree it reads -- persists unchanged across splits.
        self.population = LeafPopulation(self.tree, self._classify)
        self.population.add([])
        for prefix in pst.table.prefixes:
            self.population.add(prefix)

        # The sequential population test: weighs whether a distinguisher splits a
        # leaf.  Stateless over (population, tree), so it is never rebound.
        self.splits = SplitEvidence(
            pst,
            self.family,
            population=self.population,
            tree=self.tree,
            num_states=lambda: self.tree.num_states,
            split_fpr=split_fpr,
            split_miss_rate=split_miss_rate,
        )

        # Closes the transition function against the tree, drawing members from
        # the same population and harvesting boundary strings into ``indecisive``.
        self.edges = EdgeResolver(
            self.dfa,
            self.sifter,
            self.indecisive,
            population=self.population,
            representative=self.splits.representative,
        )

    def _classify(self, strings, midfix) -> List[Optional[bool]]:
        """Read a node for the population: batch the family queries for
        ``strings`` at ``midfix``, then return each one's accept/reject/None."""
        self.family.prefill([list(s) + list(midfix) for s in strings])
        return [self.family.is_accept(list(s), midfix) for s in strings]

    @property
    def access(self) -> dict:
        """Canonical access string per state, for renderers.  Derived: the
        evidence owns the strings known to reach each leaf."""
        reps = ((s, self.splits.representative(s)) for s in range(self.num_states))
        return {s: rep for s, rep in reps if rep is not None}

    @property
    def num_states(self) -> int:
        """Leaf count; the tree allocates the ids as it splits."""
        return self.tree.num_states

    # -- membership / classification ---------------------------------------

    # -- splitting ----------------------------------------------------------

    def init_worklist(self) -> None:
        self.edges.open_all_edges()

    def run_worklist(self) -> int:
        return self.edges.close()

    def split(self, state: int, distinguisher: tuple) -> int:
        """Refine leaf ``state`` into ``{True: state, False: new_state}`` under
        ``distinguisher`` and return the new state id.

        The tree refines the leaf and the partial DFA re-opens exactly the edges
        that refinement made ambiguous (see :meth:`PartialDFA.split_state`).  The
        population and split test read the tree, so the split needs no fix-up
        there -- the moved leaf's strings flush down on the next pull.
        """
        new_state = self.tree.split(state, distinguisher)
        self.dfa.split_state(state, new_state)
        return new_state

    # -- one probe ----------------------------------------------------------

    def _first_disagreement(
        self, w: List[int], states: List[Optional[int]], lo: int, hi: int
    ) -> Optional[int]:
        """Binary-search the first index where the *followed* state ``states[i]``
        diverges from a fresh sift of ``w[:i]``.  Invariant: sift agrees at ``lo``
        and disagrees at ``hi``.  Returns ``None`` on an indecisive sift."""
        assert 0 <= lo < hi <= len(w), (lo, hi)
        if lo + 1 == hi:
            return hi
        mid = (lo + hi) // 2
        actual, boundary = self.sifter.sift_and_boundary(w[:mid])
        if actual is None:
            # The binary search homes in on the DFA-vs-tree error, which sits at a
            # boundary state -- so this indecisive midpoint is a boundary string
            # worth harvesting (a probe need not *end* at the boundary to expose
            # one).  Collect it before bailing.
            self.indecisive.add(boundary)
            return None
        if states[mid] is None:
            return None
        if actual == states[mid]:
            return self._first_disagreement(w, states, mid, hi)
        return self._first_disagreement(w, states, lo, mid)

    def process(self, w: List[int], delta) -> int:
        """Walk one probe string through ``delta`` and act on the disagreement it
        exposes.

        ``delta`` is total, so the walk itself never has to stop and sift: it
        anchors once, follows an edge per symbol, and the only oracle work left is
        the whole-probe sift the disagreement is measured against.  If the walk is
        wrong -- because an edge was a guess -- that mismatch is exactly the signal
        being hunted.

        The anchor is the shortest prefix the tree can place *decisively*, which is
        usually not the empty string: ``[]`` is the string the family classifies
        worst, indecisive on most probes.  Anchoring deeper is what lets a probe
        make progress on the rest of the automaton without the initial state ever
        being classifiable -- on a target with transient initial states, no probe
        would otherwise contribute anything.  The disagreement is only meaningful
        because both of its ends are decisive sifts, so the anchor cannot be
        inferred from ``delta`` instead.

        Returns ``_SPLIT`` when the disagreement's leaf bifurcated decisively (a
        split was applied), ``_UNDECIDED`` when the population evidence is not yet
        conclusive either way (the leaf stays open so more members accumulate), and
        ``_RESOLVED`` otherwise -- a clean probe, or the leaf accepted as a single
        state at this distinguisher.
        """
        w = list(w)
        state = None
        start = 0
        while start < len(w):
            state, boundary = self.sifter.sift_and_boundary(w[:start])
            if state is not None:
                break
            self.indecisive.add(boundary)
            start += 1
        if state is None:
            return _RESOLVED  # no prefix of this probe can be placed
        self.population.add(w[:start], at=self.tree.path_of(state))
        states: List[Optional[int]] = [None] * start + [state]
        for c in w[start:]:
            state = delta[state][c]
            states.append(state)
        return self._act_on_disagreement(w, states, start)

    def _act_on_disagreement(self, w, states, agree_point) -> int:
        """Localise the first followed-vs-sift disagreement in the walked probe
        and run the sequential population split test on the leaf it exposes.
        Returns ``_SPLIT`` / ``_UNDECIDED`` / ``_RESOLVED`` (see :meth:`process`)."""
        state = states[-1]
        actual = self.sifter.sift(w)
        if actual is None or state is None or actual == state:
            return _RESOLVED
        fd = self._first_disagreement(w, states, agree_point, len(w))
        if fd is None:
            return _RESOLVED
        s1, c, s2 = states[fd - 1], w[fd - 1], states[fd]
        if s1 is None or s2 is None:
            return _RESOLVED
        # The followed edge s1 -(c)-> s2 is usually a resolved DFA edge, but the
        # walk uses a *totalised* delta, so it can instead be a gap the totaliser
        # filled -- then the DFA does not hold that edge, there is no witness to
        # separate on, and the re-sifts need not land on s1.  Any of those means
        # this disagreement is not one we can act on, so bail to _RESOLVED.
        if self.dfa.target(s1, c) != s2:
            return _RESOLVED
        witness = self.dfa.witness(s1, c)
        if witness is None:
            return _RESOLVED
        sprime = w[: fd - 1]
        if self.sifter.sift(witness) != s1 or self.sifter.sift(sprime) != s1:
            return _RESOLVED
        distinguisher = self.sifter.disagreement(witness, sprime, [c])
        if distinguisher is None:
            return _RESOLVED
        verdict = self.splits.verdict(s1, distinguisher)
        if verdict == SPLIT:
            self._apply_split(s1, distinguisher, witness, sprime)
            return _SPLIT
        return _RESOLVED if verdict == NO_SPLIT else _UNDECIDED

    def _apply_split(self, s1, distinguisher, witness, sprime) -> None:
        """Split leaf ``s1`` on ``distinguisher`` and record the two prefixes the
        disagreement separated as members of whichever side they land on -- they
        are the first strings known to reach the new leaves."""
        self.split(s1, distinguisher)
        for p in (witness, sprime):
            st = self.sifter.sift(p)
            if st is not None:
                self.population.add(list(p), at=self.tree.path_of(st))

    # -- driver -------------------------------------------------------------

    def _probe_blocks(self, max_probes: int):
        """Yield up to ``max_probes`` sampled probes, sifting each block in one
        batched pass just before it is walked.  :meth:`process` sifts every probe
        in full, so warming a block up front costs nothing extra; only the tail of
        a block is wasted, when the caller bails or a split rewrites the tree
        part-way through.  Blocks are drawn lazily, so a bail never samples ahead."""
        drawn = 0
        while drawn < max_probes:
            block = [
                self.pst.sampler.sample(self.pst.rng, self.pst.alphabet_size)
                for _ in range(min(_PROBE_BLOCK, max_probes - drawn))
            ]
            drawn += len(block)
            self.sifter.prefill(block)
            yield from block

    def _total_delta(self):
        """A total transition function to walk.  The worklist cannot always close
        an edge -- a leaf every one of whose members is indecisive has no
        successor the family can name -- so the remainder is filled here, once,
        rather than every probe that passes through it re-sifting."""
        delta, _ = self.dfa.totalise(
            range(self.num_states), lambda s, c: self.edges.decisive_target(s, c)[0]
        )
        return delta

    def counterexample_pass(self, *, max_probes: int, patience: int) -> int:
        """Hunt counterexamples until they dry up.  Returns the split count.

        Each sampled string is walked through :meth:`process`: a walk that
        disagrees with a direct sift exposes a split, applied at the break point,
        and every ``sift -> None`` prefix it passes is collected as a boundary
        string.  Stops after ``patience`` consecutive clean probes -- see
        :func:`fnr_synthesis._default_patience` for what that buys."""
        splits = 0
        since_split = 0
        delta = self._total_delta()
        for w in self._probe_blocks(max_probes):
            status = self.process(w, delta)
            if status == _SPLIT:
                splits += 1
                since_split = 0
                self.run_worklist()
                delta = self._total_delta()  # the split rewrote the state set
            elif status == _UNDECIDED:
                since_split = 0  # a leaf is still resolving -- keep sifting it
            else:
                since_split += 1
            if since_split >= patience:
                break
        return splits

    # -- export -------------------------------------------------------------

    def to_dfa_and_tree(self) -> Tuple[DFA, MidfixTree]:
        """Export the learned automaton as ``(DFA, MidfixTree)``."""
        return export_dfa(
            self.tree,
            self.dfa,
            self.family,
            self.pst,
            lambda state, c: self.edges.decisive_target(state, c)[0],
        )


def export_dfa(tree, partial, family, pst, decisive_target) -> Tuple[DFA, MidfixTree]:
    """The learned automaton as ``(DFA, MidfixTree)``.

    ``decisive_target(state, symbol)`` fills an edge the worklist left open,
    returning ``None`` when the leaf is wholly indecisive -- only such an edge
    self-loops."""
    transitions, unresolved = partial.totalise(range(tree.num_states), decisive_target)
    for state, c in unresolved:
        print(
            f"direct_lstar: no decisive edge for (state {state}, symbol {c}); "
            "falling back to a self-loop"
        )

    # Read the tree decisively (accept==reject==boundary) for the initial state,
    # over the round's family behind each midfix -- the same distinguishers the
    # learner classified with.
    boundary = pst.decision_boundary
    base_family = [pst.table.suffix(v) for v in family.vs]
    decide, _ = oracle_decider(pst.oracle, base_family, boundary, boundary)
    initial = tree.classify([], decide)
    dfa = DFA(
        states=set(range(tree.num_states)),
        input_symbols=set(range(pst.alphabet_size)),
        transitions=transitions,
        initial_state=0 if initial is None else initial,
        final_states=tree.accepting_leaves(),
        allow_partial=False,
    )
    return dfa, tree
