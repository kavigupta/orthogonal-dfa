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
  2. They split across multiple leaves. In this case, we now have multiple
  additional distinguishers among the elements of s. We add all these
  distinguishers and extend the tree.
  [1] Currently we just add the first distinguisher we find, but we could add all of them.

This only directly affects state s, so all we need to do at this point is
to re-enqueue all (s, c') for all symbols c' in the alphabet, as well as every
edge (s', c') -> s, which needs to be reclassified into one of the newly split states.

Two sources of redundant work at present:
  [2] Evaluating [c]+w fills its whole mask-matrix column (queries every prefix),
  because compute_decision_from_strings records the suffix over the full prefix
  set, even though only s's cells are read here.
  [3] If it's only going to be all one state its possible this is easy to tell early
  and bail on the rest of the queries, but we don't do that yet. Only possible
  if we do [2] first.

[1], [2] and [3] will be addressed in future commits.
"""

from collections import deque

import numpy as np
import scipy.stats
from automata.fa.dfa import DFA

from .cluster import sample_suffix_family
from .structures import DecisionTreeInternalNode, DecisionTreeLeafNode, TriPredicate


class _Leaf:
    def __init__(self, state_id, mask):
        self.state_id = state_id
        self.mask = mask


class _Internal:
    def __init__(self, predicate, rej, acc):
        self.predicate = predicate
        self.rej = rej
        self.acc = acc


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
        self.leaves = {}  # state_id -> _Leaf
        self.trans = {}  # (state_id, symbol) -> target state_id
        self.incoming = {}  # state_id -> set of (state_id, symbol) pointing at it
        self.queue = deque()
        self._next_id = 0
        self.root = None

    # -- tree bookkeeping ---------------------------------------------------

    def _new_leaf(self, mask):
        state_id = self._next_id
        self._next_id += 1
        leaf = _Leaf(state_id, mask)
        self.leaves[state_id] = leaf
        self.incoming[state_id] = set()
        return leaf

    def _enqueue_all_symbols(self, state_id):
        for c in range(self.pst.alphabet_size):
            self.queue.append((state_id, c))

    def _replace_leaf(self, state_id, new_node):
        def rec(node):
            for side in ("rej", "acc"):
                child = getattr(node, side)
                if isinstance(child, _Leaf) and child.state_id == state_id:
                    setattr(node, side, new_node)
                    return True
                if isinstance(child, _Internal) and rec(child):
                    return True
            return False

        rec(self.root)

    def _set_transition(self, state_id, c, target):
        old = self.trans.get((state_id, c))
        if old is not None:
            self.incoming[old].discard((state_id, c))
        self.trans[(state_id, c)] = target
        self.incoming[target].add((state_id, c))

    def _drop_state(self, state_id):
        # Edges out of the state disappear; edges into it become ambiguous and are
        # re-opened (closedness restoration).
        for c in range(self.pst.alphabet_size):
            target = self.trans.pop((state_id, c), None)
            if target is not None:
                self.incoming[target].discard((state_id, c))
        for src, c in list(self.incoming[state_id]):
            self.trans.pop((src, c), None)
            self.queue.append((src, c))
        del self.incoming[state_id]
        del self.leaves[state_id]

    # -- the resolution step ------------------------------------------------

    def _resolve(self, state_id, c):
        pst = self.pst
        s_mask = self.leaves[state_id].mask
        node = self.root
        while isinstance(node, _Internal):
            prepended = [[c] + list(v) for v in node.predicate.vs]
            with np.errstate(invalid="ignore"):
                decision = pst.compute_decision_from_strings(prepended, s_mask)
                n_acc = int(np.count_nonzero(decision >= pst.accept_thresh))
                n_rej = int(np.count_nonzero(decision < pst.reject_thresh))
            if _splits(pst, n_acc, n_rej):
                self._split(state_id, prepended)
                return
            node = node.acc if n_acc >= n_rej else node.rej
        self._set_transition(state_id, c, node.state_id)

    def _split(self, state_id, prepended_vs):
        pst = self.pst
        old = self.leaves[state_id]
        idxs = [pst.record_suffix(v)[2] for v in prepended_vs]
        decision = pst.compute_decision(idxs, np.ones(pst.num_prefixes, dtype=bool))
        with np.errstate(invalid="ignore"):
            acc = decision >= pst.accept_thresh
            rej = decision < pst.reject_thresh
        predicate = TriPredicate(
            [pst.suffix_bank[i] for i in idxs], pst.accept_thresh, pst.reject_thresh
        )
        rej_leaf = self._new_leaf(old.mask & rej)
        acc_leaf = self._new_leaf(old.mask & acc)
        self._replace_leaf(state_id, _Internal(predicate, rej_leaf, acc_leaf))
        self._drop_state(state_id)
        self._enqueue_all_symbols(rej_leaf.state_id)
        self._enqueue_all_symbols(acc_leaf.state_id)

    # -- driver -------------------------------------------------------------

    def build(self, first_round):
        pst = self.pst
        _, _, v_idx = pst.record_suffix([])
        vs, boundary = sample_suffix_family(pst, v_idx, first_round=first_round)
        pst.decision_boundary = boundary
        all_prefixes = np.ones(pst.num_prefixes, dtype=bool)
        decision = pst.compute_decision(vs, all_prefixes)
        with np.errstate(invalid="ignore"):
            acc = decision >= pst.accept_thresh
            rej = decision < pst.reject_thresh
        predicate = TriPredicate(
            [pst.suffix_bank[i] for i in vs], pst.accept_thresh, pst.reject_thresh
        )
        rej_leaf = self._new_leaf(all_prefixes & rej)
        acc_leaf = self._new_leaf(all_prefixes & acc)
        self.root = _Internal(predicate, rej_leaf, acc_leaf)
        self._enqueue_all_symbols(rej_leaf.state_id)
        self._enqueue_all_symbols(acc_leaf.state_id)

        while self.queue:
            state_id, c = self.queue.popleft()
            if state_id not in self.leaves:
                continue  # stale entry for a state that has since been split
            self._resolve(state_id, c)

        return self._to_dfa_and_tree()

    # -- output -------------------------------------------------------------

    def _to_dfa_and_tree(self):
        pst = self.pst
        remap = {sid: i for i, sid in enumerate(sorted(self.leaves))}
        n = len(remap)

        def to_dt(node):
            if isinstance(node, _Leaf):
                return DecisionTreeLeafNode(remap[node.state_id])
            return DecisionTreeInternalNode(
                predicate=node.predicate,
                by_rejection=(to_dt(node.rej), to_dt(node.acc)),
            )

        dt = to_dt(self.root)

        transitions = {i: {} for i in range(n)}
        for (sid, c), target in self.trans.items():
            transitions[remap[sid]][c] = remap[target]

        # Accepting states are exactly the leaves on the accept side of the root
        # distinguisher v_eps; splitting only refines a state, never its label.
        accepting = set(to_dt(self.root.acc).collect_states())

        boundary = pst.decision_boundary
        dt_decisive = dt.map_over_predicates(
            lambda p: TriPredicate(p.vs, boundary, boundary)
        )
        initial = dt_decisive.classify([], pst.oracle)
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
        return dfa, dt


def resolve_dfa(pst, *, first_round):
    """Build the (DFA, DecisionTree) for the current prefix pool via the resolver."""
    return TransitionResolver(pst).build(first_round=first_round)
