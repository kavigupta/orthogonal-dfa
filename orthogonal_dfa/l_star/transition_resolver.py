"""Incremental, transition-driven DFA discovery.

Builds the discrimination tree (states) and the transition function together:

The tree starts as the initial distinguisher family v_eps which partitions
prefix pool into two sets s_acc / s_rej.  The leaves of the tree are the
states of the DFA. We also separately maintain a transition function, which maps
(state, symbol) pairs to target states.

We work a queue of unresolved (state, symbol) pairs.  Resolving (s, c)
classifies every prefix of s extended by c. Doing so involves querying
the current tree for the prefixes of s except with every distinguisher
family prepended by c.
This has two possibilities:
  1. They all land on one leaf t. In this case we record the transition (s, c) -> t and move on.
  2. They diverge: s is really more than one state. We split it in two at the
  first decision tree node (from the root) where its prefixes disagree about where c leads

This only directly affects state s, so all we need to do at this point is
to re-enqueue all (s, c') for all symbols c' in the alphabet, as well as every
edge (s', c') -> s, which needs to be reclassified into one of the newly split states.

Evaluating a distinguisher [c]+w while resolving (s, c) only requires
executing on the prefixes of s, which is a potentially small subset of
the prefix pool.

The split keeps the accept side of the divergence as ``s`` itself and gives the
reject side a fresh id (see ``MidfixTree.split``), so state ids stay a dense
``range(num_states)`` and never need remapping on export.

One source of redundant work remains:
  [2] If it's only going to be all one state it's possible this is easy to tell early
  and bail on the rest of the queries, but we don't do that yet.
"""

from collections import deque

import numpy as np
import scipy.stats
from automata.fa.dfa import DFA

from .cluster import sample_suffix_family
from .midfix_tree import MidfixTree, oracle_decider


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
        self.masks = {}  # state_id -> bool mask over the prefix pool
        self.trans = {}  # (state_id, symbol) -> target state_id
        self.incoming = {}  # state_id -> set of (state_id, symbol) pointing at it
        self.queue = deque()

    # -- bookkeeping --------------------------------------------------------

    def _open_state(self, state_id, mask):
        self.masks[state_id] = mask
        self.incoming[state_id] = set()
        for c in range(self.pst.alphabet_size):
            self.queue.append((state_id, c))

    def _set_transition(self, state_id, c, target):
        assert (state_id, c) not in self.trans
        self.trans[(state_id, c)] = target
        self.incoming[target].add((state_id, c))

    def _reopen_edges(self, state_id):
        # A split shrank ``state_id``'s membership, so both its outgoing edges
        # (computed under the old, larger mask) and every edge into it (which may
        # now belong to either side) must be re-resolved.
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
        pst = self.pst
        s_mask = self.masks[state_id]
        node = self.tree.root
        while not isinstance(node, int):
            midfix, lookup = node
            # Resolving edge (state, c) reads the family one symbol deeper: c, then
            # this node's own midfix, then each base suffix.
            prepended = [c] + list(midfix)
            vs = self.tree.suffixes(prepended)
            with np.errstate(invalid="ignore"):
                decision = pst.compute_decision_from_strings(vs, s_mask)
                acc = decision >= pst.accept_thresh
                rej = decision < pst.reject_thresh
            n_acc, n_rej = int(acc.sum()), int(rej.sum())
            if _splits(pst, n_acc, n_rej):
                self._split(state_id, prepended, acc, rej)
                return
            node = lookup[True] if n_acc >= n_rej else lookup[False]
        self._set_transition(state_id, c, node)

    def _split(self, state_id, midfix, acc, rej):
        # acc/rej are the split family's accept/reject calls over this state's
        # prefixes (the s_mask subset); scatter them back to full-pool masks.
        pst = self.pst
        s_mask = self.masks[state_id]
        acc_mask = np.zeros(pst.num_prefixes, dtype=bool)
        rej_mask = np.zeros(pst.num_prefixes, dtype=bool)
        acc_mask[s_mask] = acc
        rej_mask[s_mask] = rej
        new_id = self.tree.split(state_id, midfix)  # True (accept) keeps state_id
        self.masks[state_id] = acc_mask
        self._reopen_edges(state_id)
        self._open_state(new_id, rej_mask)

    # -- driver -------------------------------------------------------------

    def build(self):
        pst = self.pst
        v_idx = pst.table.intern_suffix([])
        vs, boundary = sample_suffix_family(pst, v_idx)
        pst.decision_boundary = boundary
        self.tree = MidfixTree([pst.table.suffix(i) for i in vs])
        all_prefixes = np.ones(pst.num_prefixes, dtype=bool)
        decision = pst.compute_decision(vs, all_prefixes)
        with np.errstate(invalid="ignore"):
            acc = decision >= pst.accept_thresh
            rej = decision < pst.reject_thresh
        # Root ids match MidfixTree: 0 = accept (True side), 1 = reject (False side).
        self._open_state(0, all_prefixes & acc)
        self._open_state(1, all_prefixes & rej)

        while self.queue:
            state_id, c = self.queue.popleft()
            # A stable-id split reuses state_id and re-opens its edges, so the same
            # (state, c) can sit in the queue twice; skip one already resolved (a
            # later split pops it from ``trans`` when it must be redone).
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
    """Build the (DFA, MidfixTree) for the current prefix pool via the resolver."""
    return TransitionResolver(pst).build()
