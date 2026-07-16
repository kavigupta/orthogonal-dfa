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

Resolving (s, c) when the answer is "all one state" (case 1) used to query
every prefix of s at every node on the root-to-leaf path.  Most of that is
wasted: at a node where the whole population agrees on the direction, a small
sample already reveals the agreement.  So at each node we first evaluate a
*subsample* (``_descend_side``): if it is unanimous we descend on that side
without querying the rest; only when the subsample shows both sides -- a real
split candidate -- do we escalate to the full population and run the exact
split test (reusing the already-queried subsample cells, so escalation costs
nothing extra).

The subsample size is not a guess: the split test fires only when the minority
side exceeds a binomial FPR-noise threshold ``m*(denom)``, and we size the
sample so that a minority large enough to split is missed with probability at
most ``_SUBSAMPLE_MISS_BUDGET`` (``_safe_sample_size``).  A miss merely leaves
two states merged for this round, which the counterexample loop can still
split later.  Populations too small to subsample safely (``k >= denom``) run
the exact test unchanged.
"""

import math
from collections import deque
from functools import lru_cache

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


def _replace_leaf(node, state_id, new_node):
    """Return ``node`` with the leaf carrying ``state_id`` replaced by ``new_node``.
    Subtrees not on the path to that leaf are shared (returned unchanged)."""
    if isinstance(node, _Leaf):
        return new_node if node.state_id == state_id else node
    rej = _replace_leaf(node.rej, state_id, new_node)
    acc = _replace_leaf(node.acc, state_id, new_node)
    if rej is node.rej and acc is node.acc:
        return node
    return _Internal(node.predicate, rej, acc)


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


# When ``_resolve`` bails on a node from a unanimous subsample, this bounds the
# per-node probability of missing a minority large enough to have split.
_SUBSAMPLE_MISS_BUDGET = 1e-3


@lru_cache(maxsize=None)
def _safe_sample_size(denom, fpr, split_pval, miss_budget):
    """Smallest subsample of a ``denom``-prefix population that a splittable
    minority would survive with probability >= ``1 - miss_budget``.

    A split needs the minority count to clear the binomial FPR-noise threshold
    ``m*`` = smallest ``m`` with ``P(Binom(denom, fpr) > m) < split_pval``.  A
    minority of that size is ``q* = m*/denom`` of the population, and a sample
    of ``k`` misses it entirely with probability ``(1 - q*)**k``; solve that for
    ``k``.  Returns ``denom`` when the population is too small to subsample
    safely (the caller then runs the exact test)."""
    if denom <= 1:
        return denom
    m = int(scipy.stats.binom.ppf(1 - split_pval, denom, fpr))
    while 1 - scipy.stats.binom.cdf(m, denom, fpr) >= split_pval:
        m += 1
    q = m / denom
    if q <= 0 or q >= 1:
        return denom
    k = math.ceil(math.log(miss_budget) / math.log(1 - q))
    return min(denom, max(1, k))


def _subsample_mask(s_mask, k):
    """A boolean mask selecting ``k`` evenly-spaced prefixes out of ``s_mask``
    (deterministic: no RNG consumed, so runs stay reproducible)."""
    idx = np.flatnonzero(s_mask)
    sel = idx[np.linspace(0, len(idx), k, endpoint=False).astype(int)]
    m = np.zeros_like(s_mask)
    m[sel] = True
    return m


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

    def _set_transition(self, state_id, c, target):
        assert (state_id, c) not in self.trans
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
            side = self._descend_side(prepended, s_mask)
            if side is not None:
                # Unanimous subsample: descend without the exact split test.
                node = node.acc if side else node.rej
                continue
            # Inconclusive subsample (or population too small): the full
            # population is needed.  The subsample's cells are already observed,
            # so this queries only the remainder.
            with np.errstate(invalid="ignore"):
                decision = pst.compute_decision_from_strings(prepended, s_mask)
                acc = decision >= pst.accept_thresh
                rej = decision < pst.reject_thresh
            n_acc, n_rej = int(acc.sum()), int(rej.sum())
            if _splits(pst, n_acc, n_rej):
                self._split(state_id, prepended, acc, rej)
                return
            node = node.acc if n_acc >= n_rej else node.rej
        self._set_transition(state_id, c, node.state_id)

    def _descend_side(self, prepended, s_mask):
        """Cheap unanimity check on a safe subsample of ``s_mask``.

        Returns ``True`` (descend to the accept child) or ``False`` (reject
        child) when the subsample lands unanimously on one side -- so the split
        test cannot fire above the miss budget -- and ``None`` when the
        subsample is inconclusive (both sides present, i.e. a split candidate,
        or all undecided, or the population is too small to subsample), telling
        the caller to fall back to the exact full-population test."""
        pst = self.pst
        denom = int(s_mask.sum())
        k = _safe_sample_size(
            denom,
            pst.config.decision_rule_fpr,
            pst.config.split_pval,
            _SUBSAMPLE_MISS_BUDGET,
        )
        if k >= denom:
            return None
        sample = _subsample_mask(s_mask, k)
        with np.errstate(invalid="ignore"):
            decision = pst.compute_decision_from_strings(prepended, sample)
            n_acc = int((decision >= pst.accept_thresh).sum())
            n_rej = int((decision < pst.reject_thresh).sum())
        if n_acc > 0 and n_rej == 0:
            return True
        if n_rej > 0 and n_acc == 0:
            return False
        return None

    def _split(self, state_id, prepended_vs, acc, rej):
        # acc/rej are the split distinguisher's accept/reject calls over s's prefixes
        # (the s_mask subset), handed down from _resolve so we don't recompute them.
        pst = self.pst
        s_mask = self.leaves[state_id].mask
        predicate = TriPredicate(prepended_vs, pst.accept_thresh, pst.reject_thresh)
        acc_mask = np.zeros(pst.num_prefixes, dtype=bool)
        rej_mask = np.zeros(pst.num_prefixes, dtype=bool)
        acc_mask[s_mask] = acc
        rej_mask[s_mask] = rej
        rej_leaf = self._new_leaf(rej_mask)
        acc_leaf = self._new_leaf(acc_mask)
        self.root = _replace_leaf(
            self.root, state_id, _Internal(predicate, rej_leaf, acc_leaf)
        )
        self._drop_state(state_id)
        self._enqueue_all_symbols(rej_leaf.state_id)
        self._enqueue_all_symbols(acc_leaf.state_id)

    # -- driver -------------------------------------------------------------

    def build(self, first_round):
        pst = self.pst
        v_idx = pst.table.intern_suffix([])
        vs, boundary = sample_suffix_family(pst, v_idx, first_round=first_round)
        pst.decision_boundary = boundary
        all_prefixes = np.ones(pst.num_prefixes, dtype=bool)
        decision = pst.compute_decision(vs, all_prefixes)
        with np.errstate(invalid="ignore"):
            acc = decision >= pst.accept_thresh
            rej = decision < pst.reject_thresh
        predicate = TriPredicate(
            [pst.table.suffix(i) for i in vs], pst.accept_thresh, pst.reject_thresh
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
