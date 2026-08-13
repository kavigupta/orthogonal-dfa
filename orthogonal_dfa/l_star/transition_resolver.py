"""Incremental, transition-driven DFA discovery.

Builds the discrimination tree (states) and the transition function together.

The tree starts as the initial distinguisher family v_eps, partitioning the
prefix pool into accept / reject -- two leaves, the initial two states.  Each
(state, symbol) edge is then resolved by sifting a member of the state extended
by the symbol: the leaf it lands on is the target, and the member is kept as the
edge's witness (the tree is consistent, so any member resolves it the same way).

States beyond the initial two are found by the counterexample pass: random probe
strings are walked through the resolved transition function and re-sifted, and
where the walk and the sift disagree the probe has exhibited two prefixes that
reach one leaf yet behave differently under one more symbol -- a Myhill-Nerode
counterexample -- so that leaf is split (see SplitEvidence).  A split drops the
edges it made ambiguous; both they and the new leaf's edges then read as
unresolved and are refilled on the next resolve pass.

Each state's prefixes -- the pool prefixes that sift to its leaf -- live in a
:class:`~orthogonal_dfa.l_star.leaf_population.LeafPopulation`.  The split keeps
the accept side as s itself and gives the reject side a fresh id (see
MidfixTree.split), so state ids stay a dense range(num_states) and need no
remapping on export.
"""

from automata.fa.dfa import DFA

from .cluster import sample_suffix_family
from .leaf_population import LeafPopulation
from .midfix_tree import MidfixTree, oracle_decider
from .partial_dfa import PartialDFA
from .sifting import Sifter
from .split_evidence import NO_SPLIT, SPLIT, SplitEvidence
from .suffix_family import SuffixFamily

# Outcome of processing one probe (see TransitionResolver.counterexample_pass).
_RESOLVED = 0  # clean probe, or the leaf is a single state at this distinguisher
_SPLIT = 1  # the leaf bifurcated decisively; a split was applied
_UNDECIDED = 2  # evidence not yet conclusive -- keep sifting to accumulate members

#: Probes sifted per batched pass.
_PROBE_BLOCK = 16


class TransitionResolver:
    def __init__(self, pst):
        self.pst = pst
        self.tree = None
        self.family = None
        self.sifter = None
        self.splits = None
        self.population = None  # pool prefixes, per leaf
        self.dfa = None  # the partial transition function
        self.indecisive = set()  # boundary strings the family could not place

    # -- membership / population -------------------------------------------

    def _classify(self, strings, midfix):
        """Which side of ``midfix`` each string sits on; the indecisive band
        between the thresholds returns None and drops out of the population."""
        self.family.prefill([list(s) + list(midfix) for s in strings])
        return [self.family.is_accept(s, midfix) for s in strings]

    def _members(self, state_id):
        """The pool prefixes that sift to leaf ``state_id``."""
        return self.population.members(
            self.tree.path_of(state_id), self.pst.num_prefixes
        )

    def _sift(self, seq):
        """The leaf ``seq`` sifts to, or ``None`` when a node cannot place it.

        Every string the tree cannot place is harvested into ``indecisive``: it is
        a boundary string the current family straddles, and the driver feeds these
        back so the next family is forced to resolve them."""
        leaf, boundary = self.sifter.sift_and_boundary(list(seq))
        if leaf is None:
            self.indecisive.add(boundary)
        return leaf

    # -- the resolution step ------------------------------------------------

    def _resolve(self, state_id, c):
        members = self._members(state_id)
        target, witness = self._edge_target(c, members)
        self.dfa.set_edge(state_id, c, target, witness)

    def _tally(self, members, distinguisher):
        """Accept/reject counts of ``members`` classified against ``distinguisher``.
        Means are memoized, so the split-detection and edge-target descents that
        both call this share every read."""
        self.family.prefill([list(m) + distinguisher for m in members])
        votes = [self.family.is_accept(m, distinguisher) for m in members]
        return sum(v is True for v in votes), sum(v is False for v in votes)

    def _edge_target(self, c, members):
        """Where the ``(leaf, c)`` edge points and the member that witnesses it:
        the first decisive member's ``c``-successor, or the whole-population
        majority (witnessed by any member) when every member is indecisive."""
        for m in members:
            target = self._sift(list(m) + [c])
            if target is not None:
                return target, list(m)
        node = self.tree.root
        while not isinstance(node, int):
            midfix, lookup = node
            n_acc, n_rej = self._tally(members, [c] + list(midfix))
            node = lookup[True] if n_acc >= n_rej else lookup[False]
        return node, (list(members[0]) if members else [])

    def _split(self, state_id, midfix):
        # The population re-sifts state_id's prefixes on the next members() call.
        new_id = self.tree.split(state_id, midfix)
        self.dfa.split_state(state_id, new_id)

    # -- counterexamples ----------------------------------------------------

    def counterexample_pass(self, *, max_probes, patience):
        """Split in place on DFA-vs-tree disagreements until they dry up.

        Each probe is walked through the resolved delta and re-sifted; where they
        disagree, the probe has exhibited two prefixes reaching one leaf that
        behave differently under one more symbol, so the leaf is split -- the same
        counterexample the outer loop used to defer by adding a prefix and
        rebuilding. Stops after ``patience`` consecutive clean probes."""
        since_split = 0
        for w in self._probe_blocks(max_probes):
            status = self._process(w)
            if status == _SPLIT:
                since_split = 0
                self.dfa.drain(self._resolve)  # the split dropped edges; refill
            elif status == _UNDECIDED:
                since_split = 0
            else:
                since_split += 1
            if since_split >= patience:
                break

    def _probe_blocks(self, max_probes):
        drawn = 0
        while drawn < max_probes:
            block = [
                self.pst.sampler.sample(self.pst.rng, self.pst.alphabet_size)
                for _ in range(min(_PROBE_BLOCK, max_probes - drawn))
            ]
            drawn += len(block)
            self.sifter.prefill(block)
            yield from block

    def _process(self, w):
        """Anchor at the shortest prefix the tree places, follow the resolved
        delta, then act on where the walk and a fresh sift disagree."""
        w = list(w)
        state = None
        start = 0
        while start < len(w):
            state = self._sift(w[:start])
            if state is not None:
                break
            start += 1
        if state is None:
            return _RESOLVED
        # Seed the anchor leaf's population. The prefix pool is length-L, so it
        # only reaches deep leaves; short anchor prefixes are what give the shallow
        # leaves enough members for the one-state test to settle them.
        self.population.add(w[:start], at=self.tree.path_of(state))
        states = [None] * start + [state]
        for c in w[start:]:
            state = self.dfa.target(state, c)
            states.append(state)
        return self._act_on_disagreement(w, states, start)

    def _act_on_disagreement(self, w, states, agree_point):
        actual = self._sift(w)
        if actual is None or actual == states[-1]:
            return _RESOLVED
        fd = self._first_bad_edge(w, states, agree_point, len(w))
        if fd is None:
            return _RESOLVED
        # The walk follows resolved edges over a total delta, so s1, its
        # c-successor and the edge's witness are all present, and sprime reaches s1
        # (the sift agrees up to fd - 1). Only a majority-fallback witness (from an
        # empty-membered leaf) can miss s1, which the sift below screens out.
        s1, c = states[fd - 1], w[fd - 1]
        sprime = w[: fd - 1]
        witness = self.dfa.witness(s1, c)
        if self._sift(witness) != s1:
            return _RESOLVED
        distinguisher = self.sifter.disagreement(witness, sprime, [c])
        if distinguisher is None:
            return _RESOLVED
        verdict = self.splits.verdict(s1, tuple(distinguisher))
        if verdict == SPLIT:
            self._apply_split(s1, list(distinguisher), witness, sprime)
            return _SPLIT
        return _RESOLVED if verdict == NO_SPLIT else _UNDECIDED

    def _first_bad_edge(self, w, states, lo, hi):
        """Binary-search the first index where the followed state diverges from a
        fresh sift of ``w[:i]``; ``None`` on an indecisive sift.  Invariant: the
        sift agrees at ``lo`` and disagrees at ``hi``."""
        if lo + 1 == hi:
            return hi
        mid = (lo + hi) // 2
        actual = self._sift(w[:mid])
        if actual is None:
            return None
        if actual == states[mid]:
            return self._first_bad_edge(w, states, mid, hi)
        return self._first_bad_edge(w, states, lo, mid)

    def _apply_split(self, s1, distinguisher, witness, sprime):
        self._split(s1, distinguisher)
        for p in (witness, sprime):
            st = self._sift(p)
            if st is not None:
                self.population.add(list(p), at=self.tree.path_of(st))

    # -- driver -------------------------------------------------------------

    def build(self):
        pst = self.pst
        v_idx = pst.table.intern_suffix([])
        vs, boundary = sample_suffix_family(pst, v_idx)
        pst.decision_boundary = boundary
        self.family = SuffixFamily(pst, vs)
        self.tree = MidfixTree([pst.table.suffix(i) for i in vs])
        self.sifter = Sifter(self.tree, self.family)
        self.population = LeafPopulation(self.tree, self._classify)
        for p in pst.table.prefixes:
            self.population.add(list(p))
        self.splits = SplitEvidence(
            pst,
            self.family,
            population=self.population,
            tree=self.tree,
        )
        self.dfa = PartialDFA(pst.alphabet_size, num_states=self.tree.num_states)
        self.dfa.drain(self._resolve)

    # -- output -------------------------------------------------------------

    def export(self):
        pst = self.pst
        n = self.tree.num_states

        transitions = {i: dict(self.dfa.transitions[i]) for i in range(n)}

        accepting = self.tree.accepting_leaves()

        boundary = pst.decision_boundary
        decide, _ = oracle_decider(
            pst.oracle, self.tree.base_family, boundary, boundary
        )
        initial = self.tree.classify([], decide)
        if initial is None:
            initial = 0

        dfa = DFA(
            states=set(range(n)),
            input_symbols=set(range(pst.alphabet_size)),
            transitions=transitions,
            initial_state=initial,
            final_states=accepting,
            allow_partial=False,
        )
        return dfa, self.tree


def resolve_dfa(pst):
    """
    Build the (DFA, MidfixTree) for the current prefix pool via the resolver.
    """
    resolver = TransitionResolver(pst)
    resolver.build()
    return resolver.export()
