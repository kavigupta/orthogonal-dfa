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

import math
from statistics import NormalDist
from typing import Callable, Dict, List, Optional, Set, Tuple

from automata.fa.dfa import DFA

from .cluster import sample_suffix_family
from .midfix_tree import MidfixTree
from .partial_dfa import PartialDFA
from .structures import DecisionTree, TriPredicate

# Outcome of processing one probe (see DirectLStarLearner.process):
_RESOLVED = 0  # clean probe, or the leaf is a single state at this distinguisher
_SPLIT = 1  # the leaf bifurcated decisively; a split was applied
_UNDECIDED = 2  # evidence not yet conclusive -- keep sifting to accumulate members

# How many pool prefixes a leaf-membership scan sifts per batched pass.
_MEMBER_SCAN_BLOCK = 128
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
        :func:`sample_suffix_family` (see :func:`learn_direct_lstar`).
    """

    def __init__(
        self,
        pst,
        vs: List[int],
        *,
        split_fpr: Optional[float] = None,
        split_miss_rate: float = 0.01,
    ):
        self.pst = pst
        self.vs = list(vs)
        # A distinguisher may split a leaf only when the *population* of that leaf's
        # members bifurcates at it with strong evidence, decided by a held-out
        # Bayes factor (see _candidate_logbf) rather than a per-pair margin.  The
        # family is partitioned once into an ASSIGN half (groups the members) and a
        # TEST half (scores the split); disjoint suffixes keep the test from
        # measuring the very split it selected on.
        self._split_fpr = split_fpr if split_fpr is not None else pst.config.split_pval
        # Tolerated miss rate (beta): the lower sequential boundary at which a leaf
        # is accepted as a single state and no longer probed for a split.
        self._split_miss_rate = split_miss_rate
        self._assign_idx = list(range(0, len(self.vs), 2))
        self._test_idx = list(range(1, len(self.vs), 2))
        self._split_member_cap = 1500
        self._min_split_members = 12
        # Distinct prefixes seen to reach each leaf while sifting probes, the
        # evidence the population split Bayes factor accumulates: a split fires
        # once enough members have piled up for the BF to cross the threshold, so
        # the probe stream (not the fixed pool) is what drives it to resolve.
        self._leaf_probe_members: Dict[int, Set[tuple]] = {}
        # Open split hypotheses: leaf -> distinguisher -> running sufficient
        # statistics ({"ART": [A_t,R_t,A_f,R_f], "seen": {members}}).  Each member
        # is folded in once as it is sifted, so the Bayes factor is O(1) to read.
        self._open_splits: Dict[int, Dict[tuple, dict]] = {}

        # The discrimination tree owns the structure -- midfixes, branches and
        # leaves -- and calls back into is_accept for every classification.
        self.tree = MidfixTree()

        # The partial transition function, its witnesses, the per-state access
        # strings and the queue of edges still to resolve.
        self.dfa = PartialDFA(pst.alphabet_size)

        # Boundary strings encountered while *building* the DFA: any ``member + c``
        # that sifts to None during transition resolution / consistency checking.
        # The current family can't place these; the driver feeds them back into
        # the representative pool so FNR forces the next family to resolve them.
        self.indecisive: Set[Tuple[int, ...]] = set()

        # The decision (family-mean) memo.  Per-round, because the mean depends
        # on which family ``vs`` is in play; the underlying cells live in the
        # table, which persists across rounds.
        self._decision_cache: Dict[Tuple[tuple, tuple], float] = {}

    @property
    def num_states(self) -> int:
        """Leaf count; the tree allocates the ids as it splits."""
        return self.tree.num_states

    def _split_threshold(self) -> float:
        """Log Bayes factor a population split must clear right now.

        Under the "one Myhill-Nerode state" null the held-out BF (see
        _candidate_logbf) concentrates near zero -- the two-rate model's Occam
        penalty cancels the fit -- so a spurious split needs an upward fluctuation
        the BF rarely produces (``P(BF > K) <= 1/K``).  Bonferroni over the split
        hypotheses currently open (one per leaf x symbol) at target per-run false
        rate ``_split_fpr`` gives ``logBF > log(num_states * |alphabet| / fpr)``.
        Genuine splits scale their evidence with the leaf's member count and clear
        it; the bound grows only logarithmically as the tree does.
        """
        n = max(self.num_states * self.pst.alphabet_size, 1)
        return math.log(n / max(self._split_fpr, 1e-12))

    def _no_split_threshold(self) -> float:
        """Log Bayes factor at or below which a leaf is accepted as a single state
        for this distinguisher and no longer probed -- the lower sequential
        boundary from the tolerated miss rate ``_split_miss_rate`` (beta), i.e.
        ``logBF <= log(beta)``.  Between this and :meth:`_split_threshold` the
        split stays open and more probe members accumulate."""
        return math.log(max(self._split_miss_rate, 1e-12))

    def _starved_split_margin(self) -> float:
        """Confidence margin for the per-pair fallback the population Bayes factor
        can't reach.  A genuinely starved leaf (a trapped initial state whose leaf
        only a handful of *distinct* strings sift to) never gathers the members the
        BF needs, yet those few members can still span two states.  There the
        evidence is a single pair scoring on opposite sides of the distinguisher,
        so we fall back to the resolver's split z-test on the score difference
        ``D = f_s - f_sprime`` (mean 0, variance ``2 p (1-p) / m`` under one shared
        state).  Bonferroni over the open leaf x symbol hypotheses at ``_split_fpr``
        matches :meth:`_split_threshold`; ``|D|`` must clear the decisive band by
        this margin for the split to fire."""
        p = 0.5 + self.pst.config.min_signal_strength
        m = self.pst.config.suffix_family_size
        sigma_d = math.sqrt(2 * p * (1 - p) / m)
        tests = max(self.num_states * self.pst.alphabet_size, 1)
        alpha = self._split_fpr / tests
        z = NormalDist().inv_cdf(1 - alpha / 2)
        return max(0.0, z * sigma_d / 2 - self.pst.evidence_margin)

    def _pair_splits(self, s, sprime, distinguisher) -> bool:
        """Whether ``s`` and ``sprime`` land on opposite *decisive* sides of
        ``distinguisher`` with the :meth:`_starved_split_margin` -- the starved-leaf
        fallback for when the population Bayes factor has too few members to judge.
        """
        margin = self._starved_split_margin()
        d = self.is_accept(s, distinguisher, extra_margin=margin)
        dprime = self.is_accept(sprime, distinguisher, extra_margin=margin)
        return d is not None and dprime is not None and d != dprime

    # -- membership / classification ---------------------------------------

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

    def _family_bits(self, base) -> List[int]:
        """Membership bits of ``base`` under each family suffix ``v``.

        A sift base is not a pool prefix -- a fresh string is touched at every
        tree node -- so this goes through the table's loose-cell cache rather than
        its grid.  The misses are issued as one batched call, so a batching oracle
        evaluates the whole family for a node in a single forward pass."""
        table = self.pst.table
        return table.membership([list(base) + table.suffix(v) for v in self.vs])

    def _prefill_bases(self, bases) -> None:
        """Observe the whole family for every base in ``bases`` at once, so a
        population costs one oracle call rather than one per member."""
        table = self.pst.table
        self.pst.table.membership(
            [list(b) + table.suffix(v) for b in bases for v in self.vs]
        )

    def _sift_prefill(self, seqs) -> None:
        """Warm the cache for sifting every string in ``seqs``, one batched call
        per tree level rather than one per node visited."""
        for pairs in self.tree.sift_levels(seqs, self.is_accept):
            self._prefill_bases([list(s) + list(m) for s, m in pairs])

    def _decision(self, seq, prepend) -> float:
        """Mean family membership of ``seq`` under the distinguishers
        ``prepend + v``, memoized (see :meth:`_family_bits`)."""
        key = (tuple(seq), tuple(prepend))
        cached = self._decision_cache.get(key)
        if cached is not None:
            return cached
        bits = self._family_bits(list(seq) + list(prepend))
        d = sum(bits) / len(self.vs)
        self._decision_cache[key] = d
        return d

    def sift_and_boundary(self, seq) -> Tuple[Optional[int], Optional[tuple]]:
        """Route ``seq`` through the tree: ``(leaf, None)``, or ``(None,
        boundary)`` when some node classifies it indecisively."""
        return self.tree.sift(seq, self.is_accept)

    def sift(self, seq) -> Optional[int]:
        """Route ``seq`` to a state (leaf), or ``None`` if any node classifies it
        indecisively.  See :meth:`sift_and_boundary` for the boundary string."""
        leaf, _ = self.sift_and_boundary(seq)
        return leaf

    # -- splitting ----------------------------------------------------------

    def disagreement(self, s, sprime, prefix) -> Optional[tuple]:
        """Propose a distinguisher separating ``s`` and ``sprime`` (see
        :meth:`MidfixTree.first_disagreement`).

        This only *proposes* the candidate; whether the split fires is decided by
        the held-out population Bayes factor in :meth:`_candidate_logbf`, so the
        pair need only clear the ordinary decisive band, not a wide split margin.
        """
        return self.tree.first_disagreement(s, sprime, self.is_accept, prefix)

    def _family_votes(self, seq, prepend) -> List[int]:
        """Per-family accept bits of ``seq`` at ``prepend`` (see
        :meth:`_family_bits`), so the ASSIGN/TEST halves can be summed separately
        for the split Bayes factor."""
        return self._family_bits(list(seq) + list(prepend))

    def _member_group(self, prefix, distinguisher) -> Optional[bool]:
        """Which side of ``distinguisher`` ``prefix`` falls on, judged on the
        ASSIGN half of the family only (so the TEST half stays independent of the
        grouping).  ``None`` if indecisive there -- it contributes no evidence."""
        bits = self._family_votes(prefix, distinguisher)
        assign = sum(bits[i] for i in self._assign_idx) / len(self._assign_idx)
        if assign >= self.pst.accept_thresh:
            return True
        if assign < self.pst.reject_thresh:
            return False
        return None

    def _fold_member(self, accum: dict, distinguisher, prefix) -> None:
        """Add ``prefix``'s TEST-half votes into its group's running sums, once."""
        key = tuple(prefix)
        if key in accum["seen"]:
            return
        accum["seen"].add(key)
        group = self._member_group(prefix, distinguisher)
        if group is None:
            return
        bits = self._family_votes(prefix, distinguisher)
        t = sum(bits[i] for i in self._test_idx)
        side = 0 if group else 2
        accum["ART"][side] += t
        accum["ART"][side + 1] += len(self._test_idx) - t

    def _split_candidate(self, state: int, distinguisher: tuple) -> dict:
        """The running held-out split evidence for ``(state, distinguisher)``,
        created (and back-filled from the members seen so far) on first use.
        Thereafter :meth:`_record_member` folds each newly sifted member in
        incrementally, so the Bayes factor is O(1) to read -- never recomputed
        over the whole population."""
        cands = self._open_splits.setdefault(state, {})
        accum = cands.get(distinguisher)
        if accum is not None:
            return accum
        accum = {"ART": [0, 0, 0, 0], "seen": set()}  # [A_true,R_true,A_false,R_false]
        cands[distinguisher] = accum
        # Fold the leaf's members (probe-seen first, then the pool) up to the cap.
        # Batch their family queries in one call so the population packs, rather
        # than one ``|vs|`` batch per member.
        members = list(
            dict.fromkeys(
                [tuple(t) for t in self._leaf_probe_members.get(state, ())]
                + [
                    tuple(p)
                    for p in self._leaf_members(state, limit=self._split_member_cap)
                ]
            )
        )[: self._split_member_cap]
        self._prefill_bases([list(m) + list(distinguisher) for m in members])
        for m in members:
            self._fold_member(accum, distinguisher, list(m))
        return accum

    def _candidate_logbf(self, accum: dict) -> float:
        """Held-out log Bayes factor of the accumulated split evidence: one pooled
        Beta-Bernoulli rate (single state) vs two (a real split)."""
        if len(accum["seen"]) < self._min_split_members or not self._test_idx:
            return float("-inf")
        a1, r1, a2, r2 = accum["ART"]

        def log_beta(a, b):
            return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

        return (
            log_beta(1 + a1, 1 + r1)
            + log_beta(1 + a2, 1 + r2)
            - log_beta(1 + a1 + a2, 1 + r1 + r2)
        )

    def split(self, state: int, distinguisher: tuple) -> int:
        """Refine leaf ``state`` into ``{True: state, False: new_state}`` under
        ``distinguisher`` and return the new state id.

        The tree refines the leaf and the partial DFA re-opens exactly the edges
        that refinement made ambiguous (see :meth:`PartialDFA.split_state`).
        """
        new_state = self.tree.split(state, distinguisher)
        self.dfa.split_state(state, new_state)
        # Only the split leaf's members change leaf (they re-sift to one of the two
        # halves); every other leaf's members are untouched.  Redistribute this
        # leaf's members by re-sifting rather than wiping everything -- otherwise a
        # newly-created, still-conflated leaf (e.g. a trapped initial state that
        # was split off last) starts empty and can never gather the members its own
        # split needs before the pass ends.  Open candidates are dropped: their
        # distinguishers may now cross the freshly inserted node.
        for member in self._leaf_probe_members.pop(state, set()):
            landed = self.sift(list(member))
            if landed is not None:
                self._leaf_probe_members.setdefault(landed, set()).add(member)
        self._open_splits = {}
        return new_state

    # -- transition-driven discovery (the worklist) -------------------------

    def _seed_access_from_pool(self) -> None:
        """Give every current leaf a canonical access string by sifting the
        prefix pool.  The empty string pins the initial state; the rest come
        from whatever pool prefixes land in each leaf."""
        for prefix in [[]] + [list(p) for p in self.pst.table.prefixes]:
            if len(self.dfa.access) >= self.num_states:
                break
            st = self.sift(prefix)
            if st is not None and st not in self.dfa.access:
                self.dfa.access[st] = list(prefix)

    def init_worklist(self) -> None:
        self._seed_access_from_pool()
        self.dfa.open_every_edge(range(self.num_states))

    def _leaf_members(self, state: int, *, limit: int) -> List[List[int]]:
        """Prefixes that sift to ``state``, by scanning the pool."""
        out = []
        prefixes = [list(p) for p in self.pst.table.prefixes]
        # Sift the pool a block at a time: one batched call per tree level per
        # block instead of one per prefix.  A block overshoots ``limit`` by at
        # most its own size, and that work only warms the cache the next scan
        # (this leaf's other distinguishers, or another leaf's) reads back.
        for i in range(0, len(prefixes), _MEMBER_SCAN_BLOCK):
            block = prefixes[i : i + _MEMBER_SCAN_BLOCK]
            self._sift_prefill(block)
            for p in block:
                if self.sift(p) == state:
                    out.append(p)
                    if len(out) >= limit:
                        return out
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
        access = self.dfa.access.get(state)
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
            target, boundary = self.sift_and_boundary(ext)
            if target is not None:
                return target, list(m)
            # This successor is a boundary string the family can't place.
            self.indecisive.add(boundary)
            tries += 1
            if tries >= max_tries:
                break
        return None, None

    def resolve(self, state: int, c: int) -> None:
        """Resolve one edge to a decisive successor (see :meth:`_decisive_target`)."""
        if self.dfa.access.get(state) is None and self._find_access(state) is None:
            return  # unreachable leaf; leave the edge for export fallback
        target, witness = self._decisive_target(state, c)
        if target is None:
            return  # every member indecisive; export fills it as a self-loop
        self.dfa.set_edge(state, c, target, witness)

    def run_worklist(self) -> int:
        """Resolve queued ``(state, symbol)`` edges until the hypothesis is
        closed.  Returns the number of edges resolved."""
        self._sift_prefill(self.dfa.pending_probes())
        return self.dfa.drain(self.resolve)

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

    # None: indecisive under the new distinguisher; drop from membership.

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
        actual, boundary = self.sift_and_boundary(w[:mid])
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

    def _record_member(self, state: int, prefix: List[int]) -> None:
        """Note that ``prefix`` decisively sifts to leaf ``state`` -- a member the
        population split Bayes factor accumulates.  New members are folded into any
        open split candidates on this leaf right away (incremental evidence)."""
        bucket = self._leaf_probe_members.setdefault(state, set())
        if len(bucket) >= self._split_member_cap or tuple(prefix) in bucket:
            return
        bucket.add(tuple(prefix))
        for distinguisher, accum in self._open_splits.get(state, {}).items():
            self._fold_member(accum, distinguisher, prefix)

    def process(self, w: List[int]) -> int:
        """Walk one probe string: discover transitions, record the leaves its
        prefixes reach, and act on the first internal disagreement it exposes.

        Returns ``_SPLIT`` when the disagreement's leaf bifurcated decisively (a
        split was applied), ``_UNDECIDED`` when the population evidence is not yet
        conclusive either way (the leaf stays open so more members accumulate), and
        ``_RESOLVED`` otherwise -- a clean probe, or the leaf accepted as a single
        state at this distinguisher.
        """
        w = list(w)
        state: Optional[int] = None
        verified = False
        agree_point: Optional[int] = None
        states: List[Optional[int]] = []
        for i in range(len(w)):  # pylint: disable=consider-using-enumerate
            if state is None:
                state, boundary = self.sift_and_boundary(w[:i])
                if state is None:
                    self.indecisive.add(boundary)  # boundary: seq + bail prepend
                else:
                    self._record_member(state, w[:i])
                verified = True
            states.append(state)
            if state is None:
                continue
            if agree_point is None:
                agree_point = i
            if verified and state not in self.dfa.access:
                self.dfa.access[state] = w[:i]
            c = w[i]
            if c in self.dfa.transitions[state]:
                # Fast path: trust the cached edge.  If it is wrong, the mismatch
                # against the direct sift below is exactly the signal we want.
                state = self.dfa.transitions[state][c]
                verified = False
                continue
            nxt, boundary = self.sift_and_boundary(w[: i + 1])
            if nxt is None:
                self.indecisive.add(boundary)  # boundary: seq + bail prepend
            else:
                self._record_member(nxt, w[: i + 1])
                if verified:
                    # Only record an edge whose source was reached by a real sift,
                    # so the witness w[:i] genuinely sifts to ``state``.
                    self.dfa.set_edge(state, c, nxt, w[:i])
            state = nxt
            verified = True
        states.append(state)
        return self._act_on_disagreement(w, states, agree_point)

    def _act_on_disagreement(self, w, states, agree_point) -> int:
        """Localise the first followed-vs-sift disagreement in the walked probe
        and run the sequential population split test on the leaf it exposes.
        Returns ``_SPLIT`` / ``_UNDECIDED`` / ``_RESOLVED`` (see :meth:`process`)."""
        state = states[-1]
        actual = self.sift(w)
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
        if self.sift(witness) != s1 or self.sift(sprime) != s1:
            return _RESOLVED
        distinguisher = self.disagreement(witness, sprime, [c])
        if distinguisher is None:
            return _RESOLVED
        # Sequential population test at the proposed distinguisher: split above the
        # upper boundary, accept a single state below the lower boundary (and drop
        # the candidate so it stops being probed), and leave the leaf open in
        # between so more probe members accumulate and drive the BF to a decision.
        accum = self._split_candidate(s1, distinguisher)
        bf = self._candidate_logbf(accum)
        if bf >= self._split_threshold():
            self._apply_split(s1, distinguisher, witness, sprime)
            return _SPLIT
        if len(accum["seen"]) < self._min_split_members or not self._test_idx:
            # The BF is underpowered here (a starved leaf, e.g. a trapped initial
            # state).  Fall back to the per-pair z-test on the two strings the
            # disagreement already separated; a populous leaf always has the members
            # to reach the BF instead and never takes this path.
            if self._pair_splits(witness, sprime, distinguisher):
                self._apply_split(s1, distinguisher, witness, sprime)
                return _SPLIT
            return _UNDECIDED  # too little evidence to judge yet -- keep sifting
        if bf <= self._no_split_threshold():
            self._open_splits.get(s1, {}).pop(distinguisher, None)
            return _RESOLVED
        return _UNDECIDED

    def _apply_split(self, s1, distinguisher, witness, sprime) -> None:
        """Split leaf ``s1`` on ``distinguisher`` and give each new side an access
        string from the two prefixes the disagreement separated (both reached the
        old leaf and land on opposite sides of the distinguisher)."""
        self.split(s1, distinguisher)
        for p in (witness, sprime):
            st = self.sift(p)
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
            self._sift_prefill(block)
            yield from block

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
            if since_split >= patience or len(self.indecisive) >= boundary_target:
                break
        return splits

    # -- export -------------------------------------------------------------

    def _find_access(self, state: int) -> Optional[List[int]]:
        cached = self.dfa.access.get(state)
        if cached is not None:
            return cached
        for prefix in self.pst.table.prefixes:
            if self.sift(list(prefix)) == state:
                self.dfa.access[state] = list(prefix)
                return list(prefix)
        return None

    def _completed_transitions(self) -> Dict[int, Dict[int, int]]:
        """A total copy of the transition function.  Any edge the worklist left
        open is resolved here from a decisive leaf member
        (:meth:`_decisive_target`) rather than by sifting only the access string --
        a single indecisive access continuation used to fall back to a bogus
        self-loop, which wrecked the exported DFA even when the tree was correct.
        Only an edge whose *entire* leaf is indecisive still self-loops."""
        completed, unresolved = self.dfa.totalise(
            range(self.num_states), lambda s, c: self._decisive_target(s, c)[0]
        )
        for state, c in unresolved:
            print(
                f"direct_lstar: no decisive edge for (state {state}, "
                f"symbol {c}); falling back to a self-loop"
            )
        return completed

    def to_dfa_and_tree(self) -> Tuple[DFA, DecisionTree]:
        """Export the learned automaton as ``(DFA, DecisionTree)``, matching the
        shape returned by ``resolve_dfa``."""
        transitions = self._completed_transitions()

        def predicate_for(midfix) -> TriPredicate:
            return TriPredicate(
                [list(midfix) + self.pst.table.suffix(v) for v in self.vs],
                self.pst.accept_thresh,
                self.pst.reject_thresh,
            )

        dt = self.tree.to_decision_tree(predicate_for)
        accepting = self.tree.accepting_leaves()

        boundary = self.pst.decision_boundary
        dt_decisive = dt.map_over_predicates(
            lambda p, b=boundary: TriPredicate(p.vs, b, b)
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


def _grow_representative_pool(
    pst,
    learner,
    dfa,
    accumulated: List[List[int]],
    seen: Set,
    *,
    indecisive_fraction: float,
    min_indecisive: int,
    per_state: int,
) -> None:
    """Accumulate this round's boundary strings (capped) into ``accumulated`` /
    ``seen``, then rebuild the table's representative set as those boundary
    strings (which drive the FNR gate) plus a capped per-state balanced sample
    (bounded coverage for the consistency check)."""
    target = max(int(indecisive_fraction * pst.num_prefixes), min_indecisive)
    for t in _take_indecisive(learner, target):
        key = tuple(t)
        if key not in seen:
            seen.add(key)
            accumulated.append(t)
    curated = _curated_pool(dfa, pst.rng, pst.sampler.length, per_state)
    representative = accumulated + curated
    fresh = [p for p in representative if not pst.table.contains_prefix(p)]
    if fresh:
        pst.table.add_prefixes(fresh)
    pst.table.set_representative(representative)


def synthesize_direct_lstar_fnr(
    pst,
    *,
    acc_threshold: float,
    per_state: int = 60,
    indecisive_fraction: float = 0.1,
    min_indecisive: int = 200,
    max_rounds: int = 20,
    counterexample_probes: int = 4000,
    counterexample_patience: Optional[int] = None,
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

    # Patience for the discovery pass: stop a round after this many consecutive
    # clean probes. Derived from acc_threshold, not hardcoded. If the DFA-vs-tree
    # disagreement rate were still at the tolerated level eps = 1 - acc_threshold,
    # seeing k clean probes in a row has probability (1 - eps)^k = acc_threshold^k,
    # so k = ceil(ln(alpha) / ln(acc_threshold)) makes stopping a <= alpha event.
    # This is a cost knob -- the outer estimate + next round verify -- so a modest
    # alpha suffices (149 at acc_threshold 0.98; ~300 at 0.99).
    if counterexample_patience is None:
        counterexample_patience = math.ceil(math.log(0.05) / math.log(acc_threshold))

    first_round = True
    best = (-1.0, None, None, 0.0)
    # Accumulated boundary strings -- the FNR gate resolves the chain one state
    # per round, so keeping earlier rounds' indecisives keeps the family honest
    # about the whole chain (they turn decisive once their state is resolved).
    accumulated: List[List[int]] = []
    seen: Set = set()
    # Early-stop when a round makes no progress. Some targets are unlearnable
    # with a fixed-length prefix sampler (transient states the sampler never
    # lands on, issue #128): the FNR gate then finds no new boundary strings and
    # the round is a byte-for-byte repeat of the last. When nothing changes --
    # no new states, no accuracy gain, no new boundary strings -- we have reached
    # that fixpoint, so stop rather than burn the remaining rounds. ``stall``
    # counts consecutive no-progress rounds; two in a row confirms the fixpoint.
    prev_states = 0
    prev_acc_len = 0
    stall = 0
    for round_idx in range(max_rounds):
        prior_best = best[0]
        v_idx = pst.table.intern_suffix([])
        vs, boundary = sample_suffix_family(pst, v_idx, first_round=first_round)
        pst.decision_boundary = boundary
        first_round = False

        learner = DirectLStarLearner(pst, vs)
        learner.init_worklist()
        learner.run_worklist()
        # A single discovery pass: sample fresh strings and split on DFA-vs-tree
        # disagreements (the equivalence-oracle role).  Its binary search homes in
        # on the errors, which sit at boundary states, so it *also* harvests the
        # boundary (sift -> None) strings that feed the FNR gate -- densely and
        # targeted, so a probe need not end at the boundary.  This subsumes the
        # separate consistency check (pool verification + boundary sweep): dropping
        # it brings the substring case to E-L* query parity (1.08M) with no loss of
        # convergence across seeds.
        learner.counterexample_pass(
            max_probes=counterexample_probes,
            patience=counterexample_patience,
            boundary_target=10**9,
        )
        learner.run_worklist()
        dfa, dt = learner.to_dfa_and_tree()

        boundary = pst.decision_boundary
        dt_decisive = dt.map_over_predicates(
            lambda p, b=boundary: TriPredicate(p.vs, b, b)
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
        if true_acc > best[0]:
            best = (true_acc, dfa, dt, pst.decision_boundary)
        if true_acc >= acc_threshold:
            print(
                f"[direct-lstar/fnr] round {round_idx}: converged, "
                f"{learner.num_states} states"
            )
            break

        _grow_representative_pool(
            pst,
            learner,
            dfa,
            accumulated,
            seen,
            indecisive_fraction=indecisive_fraction,
            min_indecisive=min_indecisive,
            per_state=per_state,
        )
        print(
            f"[direct-lstar/fnr] round {round_idx}: {learner.num_states} states, "
            f"est {true_acc:.3f}, {len(accumulated)} accumulated indecisive, "
            f"{int(pst.table.representative.sum())} rep / {pst.num_prefixes} total"
        )

        progressed = (
            learner.num_states > prev_states
            or true_acc > prior_best + 1e-9
            or len(accumulated) > prev_acc_len
        )
        stall = 0 if progressed else stall + 1
        prev_states = learner.num_states
        prev_acc_len = len(accumulated)
        if stall >= 2:
            print(
                f"[direct-lstar/fnr] round {round_idx}: no progress "
                f"({learner.num_states} states) -- target unresolvable with this "
                "sampler, stopping"
            )
            break

    # Correct per-state accept/reject labels: the structural labeling (leaves on
    # the root's accept side) can flip low-support states under noise, especially
    # asymmetric noise -- a resample + binomial test per reachable state fixes it.
    # This is the same denoising step the resolver pipeline applies at the end.
    from .lstar import denoise_accept_labels

    _, best_dfa, best_dt, best_boundary = best
    pst.decision_boundary = best_boundary
    best_dfa = denoise_accept_labels(pst, best_dfa)
    return best_dfa, best_dt
