"""Random-walk, transition-driven DFA discovery (a "direct L*").

This is the algorithm sketched in ``notebooks/direct-lstar.ipynb`` on the
``different-approach`` branch, implemented as a real, self-contained learner.

It is an alternative to :mod:`orthogonal_dfa.l_star.transition_resolver`.  Both
grow a discrimination tree whose leaves are DFA states while maintaining a
transition function on the side, but they find the work differently:

  * ``TransitionResolver`` sweeps a queue of ``(state, symbol)`` pairs and, for
    each, runs a *statistical* split test over the whole prefix pool.

  * This learner instead draws random probe strings and walks each one through
    the *cached* transition function.  It then re-classifies the same string
    directly against the discrimination tree.  Where the two disagree, the probe
    has, entirely on its own, exhibited two prefixes that reach the same tree
    leaf yet behave differently under one more symbol -- a Myhill-Nerode
    counterexample -- and the offending leaf is split.

The discrimination tree here is not built from the generic ``DecisionTree``
classes during learning; it is a lightweight nested structure so splits are
cheap:

    leaf     := int                      # a DFA state id
    internal := (prepend, {True: node,   # ``prepend`` is a tuple of symbols
                           False: node})  # prepended to every base suffix ``v``

The base suffix family ``vs`` (the distinguishers that induce the initial
accept/reject split) is sampled once, exactly as the resolver does.  A node's
``prepend`` p means the node distinguishes using the suffixes ``p + v`` for each
base ``v``; evaluating ``is_accept(s, p)`` is therefore the same membership test
as classifying ``s + p`` against the base family, which is the identity that
lets :meth:`disagreement` locate a separating suffix.

``to_dfa_and_tree`` exports the learned automaton in the same
``(DFA, DecisionTree)`` shape as ``resolve_dfa`` so it is a drop-in alternative.
"""

from collections import deque
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import scipy.stats
from automata.fa.dfa import DFA

from .cluster import sample_suffix_family
from .structures import (
    DecisionTree,
    DecisionTreeInternalNode,
    DecisionTreeLeafNode,
    TriPredicate,
)

# A discrimination-tree node: a leaf is an ``int`` state id; an internal node is
# ``(prepend, {True: accept_child, False: reject_child})``.
Node = object


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
        :func:`sample_suffix_family` (see :func:`learn_direct_lstar`).
    """

    def __init__(
        self,
        pst,
        vs: List[int],
        *,
        split_margin: Optional[float] = None,
        split_test_budget: int = 1,
        split_fpr: Optional[float] = None,
        membership_cache: Optional[Dict[tuple, int]] = None,
    ):
        self.pst = pst
        self.vs = list(vs)
        # Extra confidence required (beyond the ordinary accept/reject band) before
        # a distinguisher is allowed to *split* a state.  Splitting on a noise-flip
        # corrupts the tree globally, so the split bar is stricter than the bar for
        # ordinary sifting.  Derived from the noise model unless overridden.
        if split_margin is None:
            split_margin = self._derive_split_margin(split_test_budget, split_fpr)
        self.split_margin = split_margin

        # Root: the empty prepend splits into state 0 (accept) and state 1
        # (reject).  Splitting only ever refines a leaf, so the root stays a
        # tuple and state 0's subtree stays entirely on the accept side.
        self.dt: Node = ((), {True: 0, False: 1})
        self.num_states = 2

        # transitions[s][c] -> target state, the current best guess for delta(s, c).
        self.transitions: Dict[int, Dict[int, int]] = {0: {}, 1: {}}
        # A prefix that provably reaches ``s`` and whose one-symbol extension by
        # ``c`` reaches ``transitions[s][c]``.  Under the worklist, this is just
        # ``access[s]`` (each edge is resolved by sifting ``access[s] + [c]``).
        self.transition_witnesses: Dict[Tuple[int, int], List[int]] = {}
        # A canonical access string per state (sifts to that state).  Transitions
        # are resolved from it, so it must always be present for a reachable state.
        self.access: Dict[int, List[int]] = {}

        # Transition-driven discovery bookkeeping (see resolve / run_worklist):
        #   incoming[s] -- the edges (src, c) whose current target is s, so that
        #     when s splits we can re-open exactly those edges;
        #   worklist    -- the (state, symbol) pairs still to resolve.
        self.incoming: Dict[int, Set[Tuple[int, int]]] = {0: set(), 1: set()}
        self.worklist: "deque[Tuple[int, int]]" = deque()

        # Boundary strings encountered while *building* the DFA: any ``member + c``
        # that sifts to None during transition resolution / consistency checking.
        # The current family can't place these; the driver feeds them back into
        # the representative pool so FNR forces the next family to resolve them.
        self.indecisive: Set[Tuple[int, ...]] = set()

        # Cache of prefix -> row for the one-hot membership lookups, rebuilt when
        # the table grows.
        self._prefix_pos: Optional[Dict[tuple, int]] = None
        self._prefix_pos_n = -1
        # Two-level memoization so sift scratch never enters the prefix pool:
        #  * _decision_cache: (seq, prepend) -> mean family membership.  Depends on
        #    the family ``vs``, so it is per-learner (per-round).
        #  * _membership_cache: full-string -> membership bit.  Deterministic and
        #    family-independent, so it is *shared across rounds* -- recovering the
        #    cross-round cell caching the MaskTable used to give.
        self._decision_cache: Dict[Tuple[tuple, tuple], float] = {}
        self._membership_cache: Dict[tuple, int] = (
            {} if membership_cache is None else membership_cache
        )

    def _derive_split_margin(
        self, split_test_budget: int, split_fpr: Optional[float]
    ) -> float:
        """The split confidence margin, derived from the noise model.

        A split fires when two prefixes reaching the same leaf score on opposite
        decisive sides of a distinguisher of size ``m = suffix_family_size``.
        Their score difference ``D = f_s - f_sprime`` has, under the null "same
        Myhill-Nerode state", mean 0 and variance ``2 p (1-p) / m`` (each family
        term is a Bernoulli of variance ``p(1-p)`` *whatever* its true label, and
        the two prefixes' membership noise is independent).  A spurious split needs
        ``|D|`` past the decisive band ``2*eps``; we run one such test per probe,
        so we Bonferroni-correct the target FPR over the whole probe budget:

            threshold = z(split_fpr / budget) * sqrt(2 p (1-p) / m)
            margin    = max(0, threshold/2 - evidence_margin)

        Genuine splits (large ``|D|``) clear this comfortably; a same-state pair
        would need a fluctuation the correction makes vanishingly unlikely even
        aggregated over every probe.
        """
        if split_fpr is None:
            split_fpr = self.pst.config.split_pval
        p = 0.5 + self.pst.config.min_signal_strength
        m = self.pst.config.suffix_family_size
        sigma_d = np.sqrt(2 * p * (1 - p) / m)
        alpha = split_fpr / max(split_test_budget, 1)
        z = scipy.stats.norm.ppf(1 - alpha / 2)
        threshold = z * sigma_d
        return max(0.0, threshold / 2 - self.pst.evidence_margin)

    # -- membership / classification ---------------------------------------

    def _prefix_index(self, seq: List[int]) -> int:
        n = self.pst.table.num_prefixes
        if self._prefix_pos is None or self._prefix_pos_n != n:
            self._prefix_pos = {
                tuple(p): i for i, p in enumerate(self.pst.table.prefixes)
            }
            self._prefix_pos_n = n
        return self._prefix_pos[tuple(seq)]

    def is_accept(self, seq, prepend, extra_margin: float = 0.0) -> Optional[bool]:
        """Confidently classify ``seq`` at a node whose distinguishers are the
        base family each prepended by ``prepend``.

        Returns ``True`` (accept) / ``False`` (reject) when the mean membership
        over the prepended family lands decisively past ``accept_thresh`` /
        ``reject_thresh``, and ``None`` in the indecisive band between them.  The
        decisive band is what keeps a single leaf from being split twice on the
        same (noisy) criterion.

        ``extra_margin`` widens that band symmetrically: a caller that wants a
        *higher standard of evidence* (e.g. before committing a split, so a
        noise-flipped membership can't manufacture a distinguisher) passes a
        positive value and only gets a decisive answer further from the boundary.
        """
        decision = self._decision(seq, prepend)
        if decision >= self.pst.accept_thresh + extra_margin:
            return True
        if decision < self.pst.reject_thresh - extra_margin:
            return False
        return None

    def _decision(self, seq, prepend) -> float:
        """Mean family membership of ``seq`` under the distinguishers
        ``prepend + v``, queried straight from the oracle and memoized.

        Routing this through the MaskTable's one-hot machinery would add ``seq``
        to the prefix pool -- and a sift touches a fresh string at every tree
        node, so transient scratch would bloat the pool (and every per-round pass
        over it) to many thousands of entries.  The oracle is deterministic per
        string, so the direct value is identical to the table path; it just isn't
        persisted."""
        key = (tuple(seq), tuple(prepend))
        cached = self._decision_cache.get(key)
        if cached is not None:
            return cached
        base = list(seq) + list(prepend)
        mc = self._membership_cache
        oracle = self.pst.oracle
        total = 0
        for v in self.vs:
            s = tuple(base) + tuple(self.pst.table.suffix(v))
            bit = mc.get(s)
            if bit is None:
                bit = int(oracle.membership_query(list(s)))
                mc[s] = bit
            total += bit
        d = total / len(self.vs)
        self._decision_cache[key] = d
        return d

    def sift(self, seq, node: Optional[Node] = None) -> Optional[int]:
        """Route ``seq`` through the discrimination tree to a state (leaf), or
        ``None`` if any node classifies it indecisively."""
        if node is None:
            node = self.dt
        if isinstance(node, int):
            return node
        prepend, lookup = node
        decision = self.is_accept(seq, prepend)
        if decision is None:
            return None
        return self.sift(seq, lookup[decision])

    # -- splitting ----------------------------------------------------------

    def disagreement(self, s, sprime, node: Node, prepend_to_tree) -> Optional[tuple]:
        """Find a distinguisher separating ``s`` and ``sprime``.

        Both currently sift to the same leaf, but ``s + prepend_to_tree`` and
        ``sprime + prepend_to_tree`` are known to reach *different* leaves.  Walk
        the tree down the branch where they still agree; the first node where
        they disagree yields the separating prepend ``prepend_to_tree + node
        prepend``.  Returns ``None`` if a needed classification is indecisive
        (the caller then skips this probe).
        """
        if isinstance(node, int):
            return None  # reached a leaf without a disagreement
        prepend, lookup = node
        full = (*prepend_to_tree, *prepend)
        # Require the *higher* standard of evidence here: both prefixes must be
        # classified confidently (further from the boundary by ``split_margin``)
        # for this node to justify a split.  An indecisive-under-strict-margin
        # classification aborts the split rather than committing on thin evidence.
        d = self.is_accept(s, full, extra_margin=self.split_margin)
        dprime = self.is_accept(sprime, full, extra_margin=self.split_margin)
        if d is None or dprime is None:
            return None
        if d != dprime:
            return full
        return self.disagreement(s, sprime, lookup[d], prepend_to_tree)

    def _replace_leaf(self, node: Node, state: int, new_node: Node) -> Node:
        if isinstance(node, int):
            return new_node if node == state else node
        prepend, lookup = node
        return (
            prepend,
            {k: self._replace_leaf(v, state, new_node) for k, v in lookup.items()},
        )

    def _set_transition(self, state: int, c: int, target: int, witness) -> None:
        prev = self.transitions[state].get(c)
        if prev is not None:
            self.incoming[prev].discard((state, c))
        self.transitions[state][c] = target
        self.transition_witnesses[state, c] = list(witness)
        self.incoming[target].add((state, c))

    def _clear_transition(self, state: int, c: int) -> None:
        target = self.transitions[state].pop(c, None)
        self.transition_witnesses.pop((state, c), None)
        if target is not None:
            self.incoming[target].discard((state, c))

    def _reopen(self, state: int, c: int) -> None:
        """Re-queue ``(state, c)`` for resolution (deduped by run_worklist)."""
        self.worklist.append((state, c))

    def split(self, state: int, distinguisher: tuple) -> int:
        """Refine leaf ``state`` into ``{True: state, False: new_state}`` under
        ``distinguisher`` and return the new state id.

        Splitting a leaf only makes edges *incident* to it ambiguous:
          * its outgoing edges vanish (the source is now two states);
          * the edges pointing at it must be re-classified into one of the two.
        Every such edge is dropped and re-queued on the worklist; edges not
        touching ``state`` keep valid witnesses (their sift path never passed
        through this leaf) and are untouched.
        """
        new_state = self.num_states
        self.num_states += 1
        self.transitions[new_state] = {}
        self.incoming[new_state] = set()
        self.dt = self._replace_leaf(
            self.dt, state, (distinguisher, {True: state, False: new_state})
        )
        # Outgoing edges of the split leaf: drop and re-open.
        for c in list(self.transitions[state]):
            self._clear_transition(state, c)
            self._reopen(state, c)
        # Incoming edges: drop and re-open (target is now ambiguous).
        for src, c in list(self.incoming[state]):
            self._clear_transition(src, c)
            self._reopen(src, c)
        self.incoming[state] = set()
        # Both halves need all their outgoing edges resolved.
        for sym in range(self.pst.alphabet_size):
            self._reopen(state, sym)
            self._reopen(new_state, sym)
        return new_state

    # -- transition-driven discovery (the worklist) -------------------------

    def _seed_access_from_pool(self) -> None:
        """Give every current leaf a canonical access string by sifting the
        prefix pool.  The empty string pins the initial state; the rest come
        from whatever pool prefixes land in each leaf."""
        for prefix in ([[]] + [list(p) for p in self.pst.table.prefixes]):
            if len(self.access) >= self.num_states:
                break
            st = self.sift(prefix)
            if st is not None and st not in self.access:
                self.access[st] = list(prefix)

    def _init_worklist(self) -> None:
        self._seed_access_from_pool()
        for state in range(self.num_states):
            for c in range(self.pst.alphabet_size):
                self._reopen(state, c)

    def _leaf_members(self, state: int, *, limit: int) -> List[List[int]]:
        """Prefixes that sift to ``state``.  Uses the cached membership when a
        consistency pass has populated it; otherwise scans the pool."""
        cached = getattr(self, "leaf_members", None)
        if cached is not None and state in cached:
            return cached[state]
        out = []
        for p in self.pst.table.prefixes:
            if self.sift(list(p)) == state:
                out.append(list(p))
                if len(out) >= limit:
                    break
        return out

    def _decisive_target(
        self, state: int, c: int, *, max_tries: int = 30
    ) -> Tuple[Optional[int], Optional[List[int]]]:
        """A *decisive* target for ``delta(state, c)``.

        ``resolve`` used to sift only ``access[state] + [c]`` and give up (leaving
        the edge to a self-loop) when that one string was indecisive -- even
        though the tree is consistent, so *any* member of the leaf resolves the
        same edge.  Here we try the access string first, then other leaf members,
        and take the first decisive successor.  Returns ``(None, None)`` only when
        every tried member is indecisive (a genuinely unresolvable edge)."""
        candidates: List[List[int]] = []
        access = self.access.get(state)
        if access is not None:
            candidates.append(access)
        candidates.extend(self._leaf_members(state, limit=max_tries))
        seen = set()
        tries = 0
        for m in candidates:
            key = tuple(m)
            if key in seen:
                continue
            seen.add(key)
            ext = list(m) + [c]
            target = self.sift(ext)
            if target is not None:
                return target, list(m)
            # This successor is a boundary string the family can't place.
            self.indecisive.add(tuple(ext))
            tries += 1
            if tries >= max_tries:
                break
        return None, None

    def resolve(self, state: int, c: int) -> None:
        """Resolve one edge to a decisive successor (see :meth:`_decisive_target`)."""
        if self.access.get(state) is None and self._find_access(state) is None:
            return  # unreachable leaf; leave the edge for export fallback
        target, witness = self._decisive_target(state, c)
        if target is None:
            return  # every member indecisive; export fills it as a self-loop
        self._set_transition(state, c, target, witness)

    def run_worklist(self) -> int:
        """Resolve queued ``(state, symbol)`` edges until the hypothesis is
        closed.  Returns the number of edges resolved.

        A split reuses the old id for its True branch, so every id in
        ``range(num_states)`` is always a live leaf -- no staleness check is
        needed; the only dedup is skipping edges already resolved."""
        resolved = 0
        while self.worklist:
            state, c = self.worklist.popleft()
            if c in self.transitions[state]:
                continue  # already resolved (deduped)
            self.resolve(state, c)
            resolved += 1
        return resolved

    # -- consistency-driven discovery --------------------------------------
    #
    # Instead of hunting counterexamples with random probes, verify the closed
    # hypothesis directly: a leaf is one Myhill-Nerode state only if all its
    # members agree on where each symbol leads.  We check this from a *sample*
    # per leaf -- a genuine split is gross (a substantial fraction of the leaf
    # diverges), so a handful of members reveals it -- and escalate to the full
    # membership only to confirm convergence.  Each violation (a member whose
    # ``c``-successor differs from the edge resolved off the access string) is an
    # exact, noise-guarded counterexample that splits the leaf.

    def assign_leaves(self) -> None:
        """(Re)compute which pool prefixes land in each current leaf.

        When ``self.check_representative_only`` is set, only representative
        prefixes are checked -- so the consistency population matches the family
        calibration population (both the curated state-balanced sample) instead
        of the accumulated scratch."""
        self.leaf_members: Dict[int, List[List[int]]] = {
            s: [] for s in range(self.num_states)
        }
        prefixes = self.pst.table.prefixes
        if getattr(self, "check_representative_only", False):
            rep = self.pst.table.representative
            prefixes = [p for p, r in zip(prefixes, rep) if r]
        for p in prefixes:
            st = self.sift(list(p))
            if st is not None:
                self.leaf_members[st].append(list(p))

    def _reassign_after_split(self, state: int, new_state: int, distinguisher) -> None:
        old = self.leaf_members.get(state, [])
        self.leaf_members[state] = []
        self.leaf_members[new_state] = []
        for u in old:
            d = self.is_accept(u, distinguisher)
            if d is True:
                self.leaf_members[state].append(u)
            elif d is False:
                self.leaf_members[new_state].append(u)
            # None: indecisive under the new distinguisher; drop from membership.

    def _sample(self, members, sample_size, rng):
        if sample_size is None or len(members) <= sample_size:
            return members
        idx = rng.choice(len(members), size=sample_size, replace=False)
        return [members[i] for i in idx]

    def _find_inconsistency(self, sample_size, rng, skip):
        """Scan leaves for a member whose ``c``-successor disagrees with the edge
        resolved off the access string, returning ``(state, u, c, distinguisher)``
        for the first noise-guarded split found, or ``None`` if the sample is
        clean.  ``self._checks`` / ``self._confirms`` tally (leaf,symbol) probes
        for the clean-vs-confirm measurement."""
        for state in range(self.num_states):
            members = self.leaf_members.get(state, [])
            if len(members) <= 1:
                continue
            access = self.access.get(state)
            if access is None:
                continue
            sample = self._sample(members, sample_size, rng)
            for c in range(self.pst.alphabet_size):
                target = self.transitions[state].get(c)
                if target is None:
                    continue
                self._checks += 1
                for u in sample:
                    key = (state, tuple(u), c)
                    if key in skip:
                        continue
                    tu = self.sift(u + [c])
                    if tu is None:
                        # ``u + [c]`` is indecisive under the current family -- a
                        # boundary string this family can't place.  Record it so
                        # the caller can enrich the next round's family with it.
                        self.indecisive.add(tuple(u + [c]))
                    elif tu != target:
                        self._confirms += 1
                        dist = self.disagreement(access, u, self.dt, [c])
                        if dist is not None:
                            return state, u, c, dist
                        # decisive divergence but not margin-separable -> treat as
                        # noise, don't split; skip so we don't re-find it.
                        skip.add(key)
        return None

    def consistency_close(self, *, sample_size, rng) -> int:
        """Split until no sampled inconsistency remains.  ``sample_size=None``
        checks the full membership (the escalation / confirmation pass)."""
        self.assign_leaves()
        self._checks = getattr(self, "_checks", 0)
        self._confirms = getattr(self, "_confirms", 0)
        skip: Set = set()
        splits = 0
        while True:
            cand = self._find_inconsistency(sample_size, rng, skip)
            if cand is None:
                return splits
            state, u, _c, dist = cand
            old_access = self.access[state]
            new_state = self.split(state, dist)
            for p in (old_access, u):
                st = self.sift(p)
                if st is not None:
                    self.access[st] = list(p)
            self._reassign_after_split(state, new_state, dist)
            self.run_worklist()
            skip.clear()  # tree changed; re-evaluate everything
            splits += 1

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
        actual = self.sift(w[:mid])
        if actual is None or states[mid] is None:
            return None
        if actual == states[mid]:
            return self._first_disagreement(w, states, mid, hi)
        return self._first_disagreement(w, states, lo, mid)

    def process(self, w: List[int]) -> bool:
        """Walk one probe string, discover transitions, and split on the first
        internal disagreement it exposes.  Returns ``True`` iff a split happened.
        """
        w = list(w)
        state: Optional[int] = None
        verified = False
        agree_point: Optional[int] = None
        states: List[Optional[int]] = []
        for i in range(len(w)):
            if state is None:
                state = self.sift(w[:i])
                if state is None:
                    self.indecisive.add(tuple(w[:i]))  # boundary prefix
                verified = True
            states.append(state)
            if state is None:
                continue
            if agree_point is None:
                agree_point = i
            if verified and state not in self.access:
                self.access[state] = w[:i]
            c = w[i]
            if c in self.transitions[state]:
                # Fast path: trust the cached edge.  If it is wrong, the mismatch
                # against the direct sift below is exactly the signal we want.
                state = self.transitions[state][c]
                verified = False
                continue
            nxt = self.sift(w[: i + 1])
            if nxt is None:
                self.indecisive.add(tuple(w[: i + 1]))  # boundary prefix
            elif verified:
                # Only record an edge whose source was reached by a real sift, so
                # the witness w[:i] genuinely sifts to ``state``.
                self._set_transition(state, c, nxt, w[:i])
            state = nxt
            verified = True
        states.append(state)

        actual = self.sift(w)
        if actual is None or state is None or actual == state:
            return False
        fd = self._first_disagreement(w, states, agree_point, len(w))
        if fd is None:
            return False
        s1, c, s2 = states[fd - 1], w[fd - 1], states[fd]
        if s1 is None or s2 is None:
            return False
        # The disagreeing edge is necessarily a cached follow (a fresh sift could
        # not disagree with itself), so its witness is present and still valid.
        if self.transitions.get(s1, {}).get(c) != s2:
            return False
        witness = self.transition_witnesses.get((s1, c))
        if witness is None:
            return False
        sprime = w[: fd - 1]
        if self.sift(witness) != s1 or self.sift(sprime) != s1:
            return False
        distinguisher = self.disagreement(witness, sprime, self.dt, [c])
        if distinguisher is None:
            return False
        self.split(s1, distinguisher)
        # witness and sprime both reached the old leaf and the distinguisher
        # separates them, so re-sifting assigns an access string to each side.
        for p in (witness, sprime):
            st = self.sift(p)
            if st is not None:
                self.access[st] = list(p)
        return True

    # -- driver -------------------------------------------------------------

    def process_probes(self, probes) -> int:
        """Walk each string in ``probes`` (e.g. counterexamples supplied by a
        refiner) through :meth:`process`, re-resolving the worklist after each
        split.  Returns the number of splits."""
        splits = 0
        for w in probes:
            if self.process(w):
                splits += 1
                self.run_worklist()
        return splits

    def probe_for_counterexamples(
        self, *, max_probes: int, patience: Optional[int] = None
    ) -> int:
        """Draw random probe strings purely to *find counterexamples*: a probe
        whose transition-followed state disagrees with a direct sift exposes a
        state that must split.  The transition function itself is built by the
        worklist (:meth:`run_worklist`), not here -- so an already-closed
        hypothesis sifts each probe about once (the consistency check) instead of
        re-deriving every edge.  After each split the worklist re-resolves the
        re-opened edges.  Stops after ``patience`` consecutive clean probes.
        Returns the number of splits (counterexamples consumed)."""
        splits = 0
        since_split = 0
        for _ in range(max_probes):
            w = self.pst.sampler.sample(self.pst.rng, self.pst.alphabet_size)
            if self.process(w):
                splits += 1
                since_split = 0
                self.run_worklist()
            else:
                since_split += 1
                if patience is not None and since_split >= patience:
                    break
        return splits

    def counterexample_pass(
        self, *, max_probes: int, patience: int, boundary_target: int
    ) -> int:
        """Targeted alternative to the full-membership escalation.

        Sample strings and walk each through :meth:`process`: a walk that
        disagrees with a direct sift exposes a split (found and applied at the
        break point), and every ``sift -> None`` prefix it passes is collected as
        a boundary string (into ``self.indecisive``).  Bails as soon as *either*
        condition the caller cares about is met: counterexamples have dried up
        (``patience`` consecutive clean probes) *or* enough boundary strings have
        been gathered to feed the next round's FNR step.  Returns the split count.
        """
        splits = 0
        since_split = 0
        for _ in range(max_probes):
            w = self.pst.sampler.sample(self.pst.rng, self.pst.alphabet_size)
            if self.process(w):
                splits += 1
                since_split = 0
                self.run_worklist()
            else:
                since_split += 1
            if since_split >= patience or len(self.indecisive) >= boundary_target:
                break
        return splits

    # -- export -------------------------------------------------------------

    def _collect_leaves(self, node: Node):
        if isinstance(node, int):
            yield node
            return
        _, lookup = node
        for child in lookup.values():
            yield from self._collect_leaves(child)

    def _find_access(self, state: int) -> Optional[List[int]]:
        cached = self.access.get(state)
        if cached is not None:
            return cached
        for prefix in self.pst.table.prefixes:
            if self.sift(list(prefix)) == state:
                self.access[state] = list(prefix)
                return list(prefix)
        return None

    def _completed_transitions(self) -> Dict[int, Dict[int, int]]:
        """A total copy of the transition function.  Any edge the worklist left
        unresolved is resolved here from a decisive leaf member
        (:meth:`_decisive_target`) rather than sifting only the access string --
        a single indecisive access continuation used to fall back to a bogus
        self-loop, which wrecked the exported DFA even when the tree was correct.
        Only an edge whose *entire* leaf is indecisive still self-loops.  Does not
        mutate ``self.transitions`` -- unresolved edges stay open for later rounds."""
        completed: Dict[int, Dict[int, int]] = {}
        for state in range(self.num_states):
            completed[state] = dict(self.transitions[state])
            for c in range(self.pst.alphabet_size):
                if c in completed[state]:
                    continue
                target, _ = self._decisive_target(state, c)
                if target is None:
                    print(
                        f"direct_lstar: no decisive edge for (state {state}, "
                        f"symbol {c}); falling back to a self-loop"
                    )
                    target = state
                completed[state][c] = target
        return completed

    def to_dfa_and_tree(self) -> Tuple[DFA, DecisionTree]:
        """Export the learned automaton as ``(DFA, DecisionTree)``, matching the
        shape returned by ``resolve_dfa``."""
        transitions = self._completed_transitions()

        def to_dt(node: Node) -> DecisionTree:
            if isinstance(node, int):
                return DecisionTreeLeafNode(node)
            prepend, lookup = node
            vs_suffixes = [list(prepend) + self.pst.table.suffix(v) for v in self.vs]
            predicate = TriPredicate(
                vs_suffixes, self.pst.accept_thresh, self.pst.reject_thresh
            )
            # by_rejection is (if rejected, if accepted) == (False child, True child).
            return DecisionTreeInternalNode(
                predicate=predicate,
                by_rejection=(to_dt(lookup[False]), to_dt(lookup[True])),
            )

        dt = to_dt(self.dt)

        # Accepting states are exactly the leaves on the accept side of the root
        # (empty-prepend) distinguisher; a split only ever refines a leaf.
        _, root_lookup = self.dt
        accepting = set(self._collect_leaves(root_lookup[True]))

        boundary = self.pst.decision_boundary
        dt_decisive = dt.map_over_predicates(
            lambda p: TriPredicate(p.vs, boundary, boundary)
        )
        initial = dt_decisive.classify([], self.pst.oracle)
        if initial is None:
            initial = 0

        dfa = DFA(
            states=set(range(self.num_states)),
            input_symbols=set(range(self.pst.alphabet_size)),
            transitions=transitions,
            initial_state=initial,
            final_states=accepting,
            allow_partial=False,
        )
        return dfa, dt


def learn_direct_lstar(
    pst,
    *,
    first_round: bool = True,
    max_probes: int = 10000,
    patience: Optional[int] = None,
) -> Tuple[DFA, DecisionTree]:
    """Sample the base suffix family, resolve the transition worklist, then probe
    for counterexamples once, and return the ``(DFA, DecisionTree)``.

    This is a *single round*: on rare / long-range patterns the boundary states
    that need splitting sit in the indecisive band and are seldom hit by uniform
    probes, so a single round collapses to the trivial accept/reject automaton.
    Use :func:`synthesize_direct_lstar` to wrap this in a refinement loop that
    manufactures the missing prefixes.
    """
    v_idx = pst.table.intern_suffix([])
    vs, boundary = sample_suffix_family(pst, v_idx, first_round=first_round)
    pst.decision_boundary = boundary
    learner = DirectLStarLearner(pst, vs, split_test_budget=max_probes)
    learner._init_worklist()
    learner.run_worklist()
    learner.probe_for_counterexamples(max_probes=max_probes, patience=patience)
    learner.run_worklist()
    return learner.to_dfa_and_tree()


# ---------------------------------------------------------------------------
# Refinement -- the replaceable part of the outer loop.
# ---------------------------------------------------------------------------
#
# A ``Refiner`` is called once per round when the current hypothesis is not yet
# accurate enough.  It may add informative prefixes to ``pst.table`` (side
# effect) and returns a list of *probe strings* for the next round's learner to
# walk.  Returning an empty list (and adding nothing) signals convergence /
# giving up, and the outer loop stops.
Refiner = Callable[..., List[List[int]]]


def default_refine(
    pst, dfa, dt: DecisionTree, dt_decisive: DecisionTree, *, true_acc: float, count: int
) -> List[List[int]]:
    """Provisional refiner: reuse the counterexample generation + leaf
    enrichment from the statistical ``TransitionResolver`` pipeline.

    TODO(direct-lstar): This is a stop-gap.  It borrows machinery
    (``generate_counterexamples`` / ``enrich_underrepresented_leaves`` in
    :mod:`orthogonal_dfa.l_star.lstar`) that was designed for the pool-wide
    statistical splitter, not for this random-walk / discrimination-tree learner.
    We should come back and replace it with a refiner native to *this* approach
    -- one that deliberately manufactures probe strings driving a boundary state
    out of the indecisive band (the documented root cause of the single-round
    failure; see the module docstring).  The outer loop is written against the
    ``Refiner`` interface precisely so this function can be swapped out without
    touching :func:`synthesize_direct_lstar`.
    """
    # Imported here (not at module top) to keep the dependency on the resolver
    # pipeline localized to the thing we intend to replace.
    from .lstar import add_counterexample_prefixes, enrich_underrepresented_leaves

    counterexamples = add_counterexample_prefixes(
        pst, dt, dfa, count, expected_acc=true_acc
    )
    enriched = enrich_underrepresented_leaves(pst, dt_decisive, count=count)
    return list(counterexamples) + list(enriched)


def counterexamples_only_refine(
    pst, dfa, dt: DecisionTree, dt_decisive: DecisionTree, *, true_acc: float, count: int
) -> List[List[int]]:
    """Refiner that generates counterexamples but skips leaf enrichment.

    In the curated-representative learner (:func:`synthesize_direct_lstar_fnr`)
    the balanced per-state sample already supplies the coverage that
    ``enrich_underrepresented_leaves`` was for -- and enrichment was the single
    most expensive refine step (~1/3 of total runtime).  Dropping it roughly
    halves per-round cost with no loss of convergence.  The counterexamples are
    still needed: without them some seeds collapse to the trivial automaton."""
    from .lstar import add_counterexample_prefixes

    return list(
        add_counterexample_prefixes(pst, dt, dfa, count, expected_acc=true_acc)
    )


def synthesize_direct_lstar(
    pst,
    *,
    acc_threshold: float,
    additional_counterexamples: int = 200,
    max_probes_per_round: int = 4000,
    patience: Optional[int] = 800,
    max_rounds: int = 20,
    split_margin: Optional[float] = None,
    refine: Refiner = default_refine,
) -> Tuple[DFA, DecisionTree]:
    """Transition-driven learner inside a refinement loop.

    Each round re-samples the suffix family over the current (growing) prefix
    pool and builds a fresh learner -- the per-round recalibration is what pushes
    boundary states out of the indecisive band, so it cannot be skipped.  Within
    a round, though, discovery is *transition driven*, not random-walk driven:
    :meth:`~DirectLStarLearner.run_worklist` resolves each ``(state, symbol)``
    edge exactly once by sifting ``access[state] + [symbol]`` (roughly one sift
    per transition), and random probes are used only to *find counterexamples*
    that trigger splits -- each split re-opens just the incident edges for the
    worklist rather than rebuilding the whole table.

    Counterexamples accumulate across rounds (``extra_probes``) and are re-walked
    each round: with the recalibrated family they can land decisively and drive
    splits that were previously stuck in the indecisive band.

    ``refine`` is the single swappable point where new prefixes are manufactured;
    it defaults to :func:`default_refine` (see its TODO).
    """
    # Imported here to avoid importing the resolver pipeline unless the loop runs.
    from .lstar import estimate_agreement_rate

    first_round = True
    extra_probes: List[List[int]] = []
    # The per-round rebuild is only weakly monotone -- a later round can recover a
    # worse hypothesis than an earlier one -- so keep the best-scoring round
    # rather than returning whatever the last round produced.
    best = None  # (true_acc, dfa, dt)
    for round_idx in range(max_rounds):
        v_idx = pst.table.intern_suffix([])
        vs, boundary = sample_suffix_family(pst, v_idx, first_round=first_round)
        pst.decision_boundary = boundary
        first_round = False

        learner = DirectLStarLearner(
            pst,
            vs,
            split_margin=split_margin,
            # Bonferroni budget: one split test per probe, over every round.
            split_test_budget=max_probes_per_round * max_rounds,
        )
        learner._init_worklist()
        learner.run_worklist()
        # Re-walk accumulated counterexamples, then hunt for fresh ones; both
        # re-resolve the worklist after each split.
        learner.process_probes(extra_probes)
        learner.probe_for_counterexamples(
            max_probes=max_probes_per_round, patience=patience
        )
        dfa, dt = learner.to_dfa_and_tree()
        print(
            f"[direct-lstar] round {round_idx}: {learner.num_states} states "
            f"from {pst.num_prefixes} prefixes"
        )

        boundary = pst.decision_boundary
        dt_decisive = dt.map_over_predicates(
            lambda p: TriPredicate(p.vs, boundary, boundary)
        )
        true_acc = estimate_agreement_rate(
            pst,
            pst.sampler,
            pst.oracle,
            dt_decisive,
            dfa,
            num_samples=2000,
            acc_threshold=acc_threshold,
        )
        print(f"[direct-lstar] round {round_idx}: estimated accuracy {true_acc:.4f}")
        if best is None or true_acc > best[0]:
            best = (true_acc, dfa, dt)
        if true_acc >= acc_threshold:
            break

        new_probes = refine(
            pst,
            dfa,
            dt,
            dt_decisive,
            true_acc=true_acc,
            count=additional_counterexamples,
        )
        if not new_probes:
            print("[direct-lstar] refiner produced no new prefixes; stopping")
            break
        extra_probes.extend(list(p) for p in new_probes)

    return best[1], best[2]


def synthesize_direct_lstar_consistency(
    pst,
    *,
    acc_threshold: float,
    additional_counterexamples: int = 200,
    sample_size: int = 25,
    max_rounds: int = 20,
    split_test_budget: int = 40000,
    refine: Refiner = default_refine,
) -> Tuple[DFA, DecisionTree]:
    """Transition-driven learner whose discovery is a *consistency check*, not
    random probing.

    Each round rebuilds over the recalibrated family, then:
      1. resolve every edge off its access string (``run_worklist``);
      2. ``consistency_close`` with a per-leaf *sample* -- split on any member
         whose ``c``-successor disagrees with the access-string edge;
      3. one full-membership ``consistency_close`` to confirm nothing subtle
         remains (the escalation pass).
    Random probes are not used at all.  ``refine`` still enriches the pool
    between rounds so under-represented states get decisive members.

    This is the design compared against ``TransitionResolver`` on query count:
    the sample makes discovery pay for only a handful of members per (leaf,
    symbol) instead of the whole leaf population.
    """
    from .lstar import estimate_agreement_rate

    first_round = True
    best = None
    for round_idx in range(max_rounds):
        v_idx = pst.table.intern_suffix([])
        vs, boundary = sample_suffix_family(pst, v_idx, first_round=first_round)
        pst.decision_boundary = boundary
        first_round = False

        learner = DirectLStarLearner(pst, vs, split_test_budget=split_test_budget)
        learner._init_worklist()
        learner.run_worklist()
        sampled = learner.consistency_close(sample_size=sample_size, rng=pst.rng)
        confirmed = learner.consistency_close(sample_size=None, rng=pst.rng)
        dfa, dt = learner.to_dfa_and_tree()
        print(
            f"[direct-lstar/consistency] round {round_idx}: {learner.num_states} "
            f"states from {pst.num_prefixes} prefixes "
            f"(sampled splits {sampled}, escalation splits {confirmed}; "
            f"leaf-symbol checks {learner._checks}, violations {learner._confirms})"
        )

        boundary = pst.decision_boundary
        dt_decisive = dt.map_over_predicates(
            lambda p: TriPredicate(p.vs, boundary, boundary)
        )
        true_acc = estimate_agreement_rate(
            pst,
            pst.sampler,
            pst.oracle,
            dt_decisive,
            dfa,
            num_samples=2000,
            acc_threshold=acc_threshold,
        )
        print(
            f"[direct-lstar/consistency] round {round_idx}: "
            f"estimated accuracy {true_acc:.4f}"
        )
        if best is None or true_acc > best[0]:
            best = (true_acc, dfa, dt)
        if true_acc >= acc_threshold:
            break

        new_probes = refine(
            pst,
            dfa,
            dt,
            dt_decisive,
            true_acc=true_acc,
            count=additional_counterexamples,
        )
        if not new_probes:
            print("[direct-lstar/consistency] refiner produced nothing; stopping")
            break

    return best[1], best[2]


def _curated_pool(dfa, rng, length: int, per_state: int) -> List[List[int]]:
    """A state-balanced sample: up to ``per_state`` distinct length-``length``
    strings reaching *each* DFA state (via the path-counting sampler).  This is
    the representative population for the next round -- clean and balanced across
    states, rather than the accumulated sift scratch."""
    from .dfa_utils import count_paths_to_state, sample_string_reaching_state

    pool: List[List[int]] = []
    for state in sorted(dfa.states):
        counts = count_paths_to_state(dfa, state, length)
        reachable = counts[length][dfa.initial_state]
        if reachable == 0:
            continue
        seen = set()
        for _ in range(per_state * 5):
            if len(seen) >= min(per_state, reachable):
                break
            seen.add(tuple(sample_string_reaching_state(dfa, counts, rng)))
        pool.extend(list(s) for s in seen)
    return pool


def _take_indecisive(learner, target: int) -> List[List[int]]:
    """Up to ``target`` of the boundary strings the learner bumped into while
    building the DFA (``learner.indecisive``) -- no separate search; these arise
    naturally from transition resolution and consistency checking."""
    return [list(t) for t in list(learner.indecisive)[:target]]


def synthesize_direct_lstar_fnr(
    pst,
    *,
    acc_threshold: float,
    additional_counterexamples: int = 200,
    sample_size: int = 25,
    per_state: int = 60,
    indecisive_fraction: float = 0.1,
    min_indecisive: int = 200,
    max_rounds: int = 20,
    split_test_budget: int = 40000,
    refine: Refiner = default_refine,
) -> Tuple[DFA, DecisionTree]:
    """Consistency learner that forces the suffix family to resolve boundary
    states via the FNR gate.

    Each round, after learning, collect the strings the family can't classify
    (``sift -> None`` -- measured to be, cleanly, the indecisive "boundary"
    states) and add them to the *representative* pool.  ``sample_suffix_family``
    then sees a high FNR over them and re-clusters to a family that classifies
    them decisively -- dropping the "completing" suffixes that were diluting them
    -- so the next round can place them and split.  The batch is capped at
    ``max(indecisive_fraction * |prefixes|, min_indecisive)`` per round.
    """
    from .lstar import estimate_agreement_rate

    first_round = True
    best = None
    # Accumulated boundary strings -- the FNR gate resolves the chain one state
    # per round, so keeping earlier rounds' indecisives keeps the family honest
    # about the whole chain (they turn decisive once their state is resolved).
    accumulated: List[List[int]] = []
    seen: Set = set()
    # Shared across rounds: membership is deterministic, so cells cached in one
    # round are reused by every later round (the family ``vs`` changing only
    # affects which cells are averaged, not their values).
    membership_cache: Dict[tuple, int] = {}
    for round_idx in range(max_rounds):
        v_idx = pst.table.intern_suffix([])
        vs, boundary = sample_suffix_family(pst, v_idx, first_round=first_round)
        pst.decision_boundary = boundary
        first_round = False

        learner = DirectLStarLearner(
            pst, vs, split_test_budget=split_test_budget,
            membership_cache=membership_cache,
        )
        # Both calibration (FNR) and the consistency check run on the same clean,
        # bounded representative set -- no scratch (memoized is_accept), no mismatch.
        learner.check_representative_only = True
        learner._init_worklist()
        learner.run_worklist()
        learner.consistency_close(sample_size=sample_size, rng=pst.rng)
        # Full-membership sweep harvests boundary (sift -> None) strings from the
        # (now boundary-enriched) representative members.
        learner.consistency_close(sample_size=None, rng=pst.rng)
        dfa, dt = learner.to_dfa_and_tree()

        boundary = pst.decision_boundary
        dt_decisive = dt.map_over_predicates(
            lambda p: TriPredicate(p.vs, boundary, boundary)
        )
        true_acc = estimate_agreement_rate(
            pst,
            pst.sampler,
            pst.oracle,
            dt_decisive,
            dfa,
            num_samples=2000,
            acc_threshold=acc_threshold,
        )
        if best is None or true_acc > best[0]:
            best = (true_acc, dfa, dt)
        if true_acc >= acc_threshold:
            print(f"[direct-lstar/fnr] round {round_idx}: converged, "
                  f"{learner.num_states} states")
            break

        # Accumulate this round's boundary strings (capped per round).
        target = max(int(indecisive_fraction * pst.num_prefixes), min_indecisive)
        for t in _take_indecisive(learner, target):
            key = tuple(t)
            if key not in seen:
                seen.add(key)
                accumulated.append(t)
        # Representative set = accumulated boundary strings (drive FNR) + a capped
        # per-state balanced sample (give the check bounded coverage) + this
        # round's counterexamples.
        counterexamples = refine(
            pst, dfa, dt, dt_decisive, true_acc=true_acc,
            count=additional_counterexamples,
        )
        curated = _curated_pool(dfa, pst.rng, pst.sampler.length, per_state)
        representative = accumulated + curated + [list(p) for p in counterexamples]
        fresh = [p for p in representative if not pst.table.contains_prefix(p)]
        if fresh:
            pst.table.add_prefixes(fresh, representative=True)
        pst.table.set_representative(representative)
        print(
            f"[direct-lstar/fnr] round {round_idx}: {learner.num_states} states, "
            f"est {true_acc:.3f}, {len(accumulated)} accumulated indecisive, "
            f"{int(pst.table.representative.sum())} rep / {pst.num_prefixes} total"
        )

    return best[1], best[2]


def _balanced_representative(learner, priority, fill, per_state) -> List[List[int]]:
    """A balanced, capped representative set: bucket candidates by the leaf they
    reach and keep at most ``per_state`` per bucket, taking ``priority`` (the
    refiner's counterexamples -- the informative, split-revealing strings) before
    ``fill`` (the state-balanced resample).  This gives every previous state a
    good-but-bounded number of members -- enough for FNR to be a real gate --
    without letting one conflated state accumulate thousands."""
    from collections import defaultdict

    buckets = defaultdict(list)
    for u in list(priority) + list(fill):
        buckets[learner.sift(list(u))].append(list(u))  # leaf None -> its own bucket
    out: List[List[int]] = []
    for members in buckets.values():
        seen, uniq = set(), []
        for m in members:
            k = tuple(m)
            if k not in seen:
                seen.add(k)
                uniq.append(m)
        out.extend(uniq[:per_state])
    return out


def synthesize_direct_lstar_curated(
    pst,
    *,
    acc_threshold: float,
    additional_counterexamples: int = 200,
    sample_size: int = 25,
    per_state: int = 60,
    max_rounds: int = 20,
    split_test_budget: int = 40000,
    refine: Refiner = default_refine,
) -> Tuple[DFA, DecisionTree]:
    """Consistency-driven learner whose calibration *and* check both run on a
    curated, state-balanced representative pool (see :func:`_curated_pool`),
    rebuilt from the previous round's DFA.

    This closes the population mismatch that plagued the plain consistency
    learner: the family was calibrated on the true probe sample while the
    consistency check ran over the whole accumulated scratch.  Here both use the
    same clean per-state resample, so the family fits exactly what it is checked
    against.
    """
    from .lstar import estimate_agreement_rate

    first_round = True
    best = None
    for round_idx in range(max_rounds):
        v_idx = pst.table.intern_suffix([])
        vs, boundary = sample_suffix_family(pst, v_idx, first_round=first_round)
        pst.decision_boundary = boundary
        first_round = False

        learner = DirectLStarLearner(pst, vs, split_test_budget=split_test_budget)
        learner.check_representative_only = True
        learner._init_worklist()
        learner.run_worklist()
        learner.consistency_close(sample_size=sample_size, rng=pst.rng)
        learner.consistency_close(sample_size=None, rng=pst.rng)
        dfa, dt = learner.to_dfa_and_tree()
        print(
            f"[direct-lstar/curated] round {round_idx}: {learner.num_states} states, "
            f"{int(pst.table.representative.sum())} representative / "
            f"{pst.num_prefixes} total prefixes"
        )

        boundary = pst.decision_boundary
        dt_decisive = dt.map_over_predicates(
            lambda p: TriPredicate(p.vs, boundary, boundary)
        )
        true_acc = estimate_agreement_rate(
            pst,
            pst.sampler,
            pst.oracle,
            dt_decisive,
            dfa,
            num_samples=2000,
            acc_threshold=acc_threshold,
        )
        print(f"[direct-lstar/curated] round {round_idx}: est accuracy {true_acc:.4f}")
        if best is None or true_acc > best[0]:
            best = (true_acc, dfa, dt)
        if true_acc >= acc_threshold:
            break

        # The next round's representative set = a state-balanced resample (for
        # calibration + coverage) PLUS the refiner's fresh counterexamples (which
        # are precisely the strings that expose the missing splits).  Excluding
        # the latter -- as an earlier version did -- starves the consistency
        # check of exactly the evidence it needs.
        counterexamples = refine(
            pst,
            dfa,
            dt,
            dt_decisive,
            true_acc=true_acc,
            count=additional_counterexamples,
        )
        curated = _curated_pool(dfa, pst.rng, pst.sampler.length, per_state)
        representative = _balanced_representative(
            learner, priority=counterexamples, fill=curated, per_state=per_state
        )
        fresh = [p for p in representative if not pst.table.contains_prefix(p)]
        if fresh:
            pst.table.add_prefixes(fresh, representative=True)
        pst.table.set_representative(representative)

    return best[1], best[2]
