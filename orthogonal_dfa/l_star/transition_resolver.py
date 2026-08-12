"""Incremental, transition-driven DFA discovery.

Builds the discrimination tree (states) and the transition function together:

The tree starts as the initial distinguisher family v_eps which partitions the
prefix pool into two sets s_acc / s_rej.  The leaves of the tree are the states
of the DFA. We also separately maintain a transition function, which maps
(state, symbol) pairs to target states.

We work a queue of unresolved (state, symbol) pairs.  Resolving (s, c) classifies
every prefix of s extended by c. Doing so involves querying the current tree for
the prefixes of s except with every distinguisher family prepended by c.
This has two possibilities:
  1. They all land on one leaf t. In this case we record the transition (s, c) -> t and move on.
  2. They diverge: s is really more than one state. We split it in two at the
  first decision tree node (from the root) where its prefixes disagree about where c leads

This only directly affects state s, so all we need to do at this point is
to re-enqueue all (s, c') for all symbols c' in the alphabet, as well as every
edge (s', c') -> s, which needs to be reclassified into one of the newly split states.

Each state's prefixes -- the pool prefixes that sift to its leaf -- live in a
:class:`~orthogonal_dfa.l_star.leaf_population.LeafPopulation`, read through the
mask's shared MemoizedOracle so a cell the mask already holds costs no new query.

The split keeps the accept side of the divergence as s itself and gives the reject
side a fresh id (see MidfixTree.split), so state ids stay a dense range(num_states)
and never need remapping on export.
"""

from automata.fa.dfa import DFA

from .cluster import sample_suffix_family
from .leaf_population import LeafPopulation
from .midfix_tree import MidfixTree, oracle_decider
from .partial_dfa import PartialDFA
from .split_evidence import SPLIT, SplitEvidence
from .suffix_family import SuffixFamily


class TransitionResolver:
    def __init__(self, pst):
        self.pst = pst
        self.tree = None
        self.family = None
        self.splits = None
        self.population = None  # pool prefixes, per leaf
        self.dfa = None  # the partial transition function + its worklist

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

    # -- the resolution step ------------------------------------------------

    def _resolve(self, state_id, c):
        members = self._members(state_id)
        distinguisher = self._divergence(state_id, c, members)
        if distinguisher is not None:
            self._split(state_id, distinguisher)
            return
        self.dfa.set_edge(state_id, c, self._edge_target(c, members))

    def _tally(self, members, distinguisher):
        """Accept/reject counts of ``members`` classified against ``distinguisher``.
        Means are memoized, so the split-detection and edge-target descents that
        both call this share every read."""
        self.family.prefill([list(m) + distinguisher for m in members])
        votes = [self.family.is_accept(m, distinguisher) for m in members]
        return sum(v is True for v in votes), sum(v is False for v in votes)

    def _divergence(self, state_id, c, members):
        """
        The distinguisher [c] + midfix at the first node down the descent where
        SplitEvidence confirms state_id's members are really two states, or None
        if none does.

        The held-out train/test test is the sole decision -- a real second state
        reproduces across the disjoint halves where scattered noise does not -- so
        no cheap pre-filter is needed to propose candidates first.
        """
        node = self.tree.root
        while not isinstance(node, int):
            midfix, lookup = node
            distinguisher = [c] + list(midfix)
            if self.splits.verdict(state_id, tuple(distinguisher)) == SPLIT:
                return distinguisher
            n_acc, n_rej = self._tally(members, distinguisher)
            node = lookup[True] if n_acc >= n_rej else lookup[False]
        return None

    def _edge_target(self, c, members):
        """The leaf ``members`` extended by ``c`` reach, following the majority at
        each node -- the same descent ``_divergence`` just took, so its tallies are
        already memoized."""
        node = self.tree.root
        while not isinstance(node, int):
            midfix, lookup = node
            n_acc, n_rej = self._tally(members, [c] + list(midfix))
            node = lookup[True] if n_acc >= n_rej else lookup[False]
        return node

    def _split(self, state_id, midfix):
        # The population re-sifts state_id's prefixes on the next members() call.
        new_id = self.tree.split(state_id, midfix)
        self.dfa.split_state(state_id, new_id)

    # -- driver -------------------------------------------------------------

    def build(self):
        pst = self.pst
        v_idx = pst.table.intern_suffix([])
        vs, boundary = sample_suffix_family(pst, v_idx)
        pst.decision_boundary = boundary
        self.family = SuffixFamily(pst, vs)
        self.tree = MidfixTree([pst.table.suffix(i) for i in vs])
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
        # Root ids match MidfixTree: 0 = accept (True side), 1 = reject (False side).
        self.dfa.open_every_edge(range(self.tree.num_states))
        self.dfa.drain(self._resolve)

        return self._to_dfa_and_tree()

    # -- output -------------------------------------------------------------

    def _to_dfa_and_tree(self):
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
    return TransitionResolver(pst).build()
