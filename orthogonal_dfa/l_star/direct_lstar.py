"""Random-walk, transition-driven DFA discovery (a "direct L*").

An alternative to :mod:`orthogonal_dfa.l_star.transition_resolver`.  Both grow a
discrimination tree whose leaves are DFA states while maintaining a transition
function on the side, but they find the work differently:

  * ``TransitionResolver`` sweeps a queue of ``(state, symbol)`` pairs and, for
    each, runs a *statistical* split test over the whole prefix pool.

  * This learner draws random probe strings and walks each one through the
    *cached* transition function, then re-classifies the same string directly
    against the tree.  Where the two disagree, the probe has on its own exhibited
    two prefixes that reach the same leaf yet behave differently under one more
    symbol -- a Myhill-Nerode counterexample -- and the offending leaf is split.

What the learner owns is the loop that turns probes into splits.  The pieces it
works through are their own objects:

    MidfixTree      which midfix cuts where, and which leaf a branch lands in
    PartialDFA      the edges resolved so far, their witnesses, and the worklist
    SuffixFamily    the family a round classifies against, and how to read it
    SplitEvidence   whether a proposed distinguisher really splits a leaf

Driving the learner in rounds -- resampling the family until it can place the
strings it could not -- is :mod:`orthogonal_dfa.l_star.fnr_synthesis`, which the
learner knows nothing about.

``to_dfa_and_tree`` exports the result in the same ``(DFA, DecisionTree)`` shape
as ``resolve_dfa``, so it is a drop-in alternative.
"""

from typing import List, Optional, Set, Tuple

from automata.fa.dfa import DFA

from .edge_resolver import EdgeResolver
from .midfix_tree import MidfixTree
from .partial_dfa import PartialDFA
from .sifting import Sifter
from .split_evidence import NO_SPLIT, SPLIT, SplitEvidence
from .structures import DecisionTree, TriPredicate
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
        # leaves -- and calls back into is_accept for every classification.
        self.tree = MidfixTree()

        # The partial transition function, its witnesses, the per-state access
        # strings and the queue of edges still to resolve.
        self.dfa = PartialDFA(pst.alphabet_size, num_states=self.tree.num_states)

        # Classifying against the tree, and closing the transition function with
        # it.  The resolver harvests boundary strings into ``indecisive``.
        self.sifter = Sifter(self.tree, self.family)
        self.indecisive: Set[Tuple[int, ...]] = set()
        self.edges = EdgeResolver(pst, self.dfa, self.sifter, self.indecisive)

        # The sequential population test: it accumulates each leaf's members and
        # says whether a proposed distinguisher splits it.
        self.splits = SplitEvidence(
            pst,
            self.family,
            pool_members=self.edges.leaf_members,
            num_states=lambda: self.tree.num_states,
            split_fpr=split_fpr,
            split_miss_rate=split_miss_rate,
            members={},
        )

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
        that refinement made ambiguous (see :meth:`PartialDFA.split_state`).
        """
        new_state = self.tree.split(state, distinguisher)
        self.dfa.split_state(state, new_state)
        self.splits = self.splits.after_split(state, self.sifter.sift)
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

    def process(self, w: List[int]) -> int:
        """Walk one probe string through the cached transitions, recording the
        leaves its prefixes reach, and act on the first disagreement it exposes.

        Returns ``_SPLIT`` when the disagreement's leaf bifurcated decisively (a
        split was applied), ``_UNDECIDED`` when the population evidence is not yet
        conclusive either way (the leaf stays open so more members accumulate), and
        ``_RESOLVED`` otherwise -- a clean probe, or the leaf accepted as a single
        state at this distinguisher.

        The walk only *follows* edges; the worklist is what resolves them, and it
        has closed the hypothesis before any probing.  An edge it could not close
        is one no member of the leaf can place, so there is nothing to compare the
        walk against and the probe carries no information.
        """
        w = list(w)
        state: Optional[int] = None
        sifted_here = False
        agree_point: Optional[int] = None
        states: List[Optional[int]] = []
        for i in range(len(w)):  # pylint: disable=consider-using-enumerate
            if state is None:
                state, boundary = self.sifter.sift_and_boundary(w[:i])
                if state is None:
                    self.indecisive.add(boundary)  # boundary: seq + bail prepend
                else:
                    self.splits.record(state, w[:i])
                sifted_here = True
            states.append(state)
            if state is None:
                continue
            if agree_point is None:
                agree_point = i
            # Only an access string reached by a real sift genuinely sifts to
            # ``state``; a followed edge proves nothing about the prefix.
            if sifted_here and state not in self.dfa.access:
                self.dfa.access[state] = w[:i]
            target = self.dfa.target(state, w[i])
            if target is None:
                return _RESOLVED  # unresolved edge; nothing to compare against
            # Trust the cached edge.  If it is wrong, the mismatch against the
            # direct sift of the whole probe is exactly the signal we want.
            state = target
            sifted_here = False
        states.append(state)
        return self._act_on_disagreement(w, states, agree_point)

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
        # The disagreeing edge is necessarily a cached follow (a fresh sift could
        # not disagree with itself), so its witness is present and still valid.
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
        verdict = self.splits.verdict(s1, distinguisher, witness, sprime)
        if verdict == SPLIT:
            self._apply_split(s1, distinguisher, witness, sprime)
            return _SPLIT
        return _RESOLVED if verdict == NO_SPLIT else _UNDECIDED

    def _apply_split(self, s1, distinguisher, witness, sprime) -> None:
        """Split leaf ``s1`` on ``distinguisher`` and give each new side an access
        string from the two prefixes the disagreement separated (both reached the
        old leaf and land on opposite sides of the distinguisher)."""
        self.split(s1, distinguisher)
        for p in (witness, sprime):
            st = self.sifter.sift(p)
            if st is not None:
                self.dfa.access[st] = list(p)

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

    def counterexample_pass(self, *, max_probes: int, patience: int) -> int:
        """Hunt counterexamples until they dry up.  Returns the split count.

        Each sampled string is walked through :meth:`process`: a walk that
        disagrees with a direct sift exposes a split, applied at the break point,
        and every ``sift -> None`` prefix it passes is collected as a boundary
        string.  Stops after ``patience`` consecutive clean probes -- see
        :func:`fnr_synthesis._default_patience` for what that buys."""
        splits = 0
        since_split = 0
        for w in self._probe_blocks(max_probes):
            status = self.process(w)
            if status == _SPLIT:
                splits += 1
                since_split = 0
                self.run_worklist()
            elif status == _UNDECIDED:
                since_split = 0  # a leaf is still resolving -- keep sifting it
            else:
                since_split += 1
            if since_split >= patience:
                break
        return splits

    # -- export -------------------------------------------------------------

    def to_dfa_and_tree(self) -> Tuple[DFA, DecisionTree]:
        """Export the learned automaton as ``(DFA, DecisionTree)``, matching the
        shape returned by ``resolve_dfa``."""
        return export_dfa(
            self.tree,
            self.dfa,
            self.family,
            self.pst,
            lambda state, c: self.edges.decisive_target(state, c)[0],
        )


def export_dfa(tree, partial, family, pst, decisive_target) -> Tuple[DFA, DecisionTree]:
    """The learned automaton as ``(DFA, DecisionTree)``.

    ``decisive_target(state, symbol)`` fills an edge the worklist left open,
    returning ``None`` when the leaf is wholly indecisive -- only such an edge
    self-loops."""
    transitions, unresolved = partial.totalise(range(tree.num_states), decisive_target)
    for state, c in unresolved:
        print(
            f"direct_lstar: no decisive edge for (state {state}, symbol {c}); "
            "falling back to a self-loop"
        )

    def predicate_for(midfix) -> TriPredicate:
        return TriPredicate(
            [list(midfix) + pst.table.suffix(v) for v in family.vs],
            pst.accept_thresh,
            pst.reject_thresh,
        )

    dt = tree.to_decision_tree(predicate_for)
    boundary = pst.decision_boundary
    dt_decisive = dt.map_over_predicates(lambda p, b=boundary: TriPredicate(p.vs, b, b))
    initial = dt_decisive.classify([], pst.oracle)
    dfa = DFA(
        states=set(range(tree.num_states)),
        input_symbols=set(range(pst.alphabet_size)),
        transitions=transitions,
        initial_state=0 if initial is None else initial,
        final_states=tree.accepting_leaves(),
        allow_partial=False,
    )
    return dfa, dt
