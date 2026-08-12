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

from collections import deque

import scipy.stats
from automata.fa.dfa import DFA

from .cluster import sample_suffix_family
from .leaf_population import LeafPopulation
from .midfix_tree import MidfixTree, oracle_decider
from .split_evidence import DEFAULT_SPLIT_MISS_RATE, SPLIT, SplitEvidence
from .suffix_family import SuffixFamily


def _splits(pst, n_acc, n_rej):
    """Binomial split test: both sides must carry more mass than the decision-rule
    FPR could explain by noise, at significance ``split_pval``."""
    denom = n_acc + n_rej
    if denom == 0:
        return False
    fpr = pst.config.decision_rule_fpr
    pval = max(
        1 - scipy.stats.binom.cdf(n_acc, denom, fpr),
        1 - scipy.stats.binom.cdf(n_rej, denom, fpr),
    )
    return pval < pst.config.split_pval


class TransitionResolver:
    def __init__(self, pst):
        self.pst = pst
        self.tree = None
        self.family = None
        self.splits = None
        self.population = None  # pool prefixes, per leaf
        self.trans = {}  # (state_id, symbol) -> target state_id
        self.incoming = {}  # state_id -> set of edges pointing at it
        self.queue = deque()

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

    # -- bookkeeping --------------------------------------------------------

    def _open_state(self, state_id):
        self.incoming[state_id] = set()
        for c in range(self.pst.alphabet_size):
            self.queue.append((state_id, c))

    def _set_transition(self, state_id, c, target):
        assert (state_id, c) not in self.trans
        self.trans[(state_id, c)] = target
        self.incoming[target].add((state_id, c))

    def _reopen_edges(self, state_id):
        # A split shrank state_id's membership, so both its outgoing edges (computed
        # under the old, larger population) and every edge into it (which may now
        # belong to either side) must be re-resolved.
        for c in range(self.pst.alphabet_size):
            target = self.trans.pop((state_id, c), None)
            if target is not None:
                self.incoming[target].discard((state_id, c))
            self.queue.append((state_id, c))
        for src, c in list(self.incoming[state_id]):
            self.trans.pop((src, c), None)
            self.queue.append((src, c))
        self.incoming[state_id] = set()

    # -- the resolution step ------------------------------------------------

    def _resolve(self, state_id, c):
        members = self._members(state_id)
        distinguisher = self._divergence(state_id, c, members)
        if distinguisher is not None:
            self._split(state_id, distinguisher)
            return
        self._set_transition(state_id, c, self._edge_target(c, members))

    def _tally(self, members, distinguisher):
        """Accept/reject counts of ``members`` classified against ``distinguisher``.
        Means are memoized, so the split-detection and edge-target descents that
        both call this share every read."""
        self.family.prefill([list(m) + distinguisher for m in members])
        votes = [self.family.is_accept(m, distinguisher) for m in members]
        return sum(v is True for v in votes), sum(v is False for v in votes)

    def _divergence(self, state_id, c, members):
        """The distinguisher ``[c] + midfix`` at the first node where ``members``
        split under one more symbol *and* SplitEvidence confirms the leaf is really
        two states, or ``None`` if none does. The binomial only proposes a
        candidate; the held-out test decides, since a real second state reproduces
        across the family's disjoint train/test halves where scattered noise
        does not."""
        node = self.tree.root
        while not isinstance(node, int):
            midfix, lookup = node
            distinguisher = [c] + list(midfix)
            n_acc, n_rej = self._tally(members, distinguisher)
            if _splits(self.pst, n_acc, n_rej) and (
                self.splits.verdict(state_id, tuple(distinguisher)) == SPLIT
            ):
                return distinguisher
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
        self._reopen_edges(state_id)
        self._open_state(new_id)

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
            num_states=lambda: self.tree.num_states,
            split_fpr=None,
            split_miss_rate=DEFAULT_SPLIT_MISS_RATE,
        )
        # Root ids match MidfixTree: 0 = accept (True side), 1 = reject (False side).
        self._open_state(0)
        self._open_state(1)

        while self.queue:
            state_id, c = self.queue.popleft()
            # A stable-id split reuses state_id and re-opens its edges, so the same
            # (state, c) can sit in the queue twice; skip one already resolved (a
            # later split pops it from trans when it must be redone).
            if (state_id, c) in self.trans:
                continue
            self._resolve(state_id, c)

        return self._to_dfa_and_tree()

    # -- output -------------------------------------------------------------

    def _to_dfa_and_tree(self):
        pst = self.pst
        n = self.tree.num_states

        transitions = {i: {} for i in range(n)}
        for (sid, c), target in self.trans.items():
            transitions[sid][c] = target

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
