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
from typing import Callable, List, Optional, Set, Tuple

from automata.fa.dfa import DFA

from .cluster import sample_suffix_family
from .midfix_tree import MidfixTree
from .partial_dfa import PartialDFA
from .split_evidence import NO_SPLIT, SPLIT, SplitEvidence
from .structures import DecisionTree, TriPredicate
from .suffix_family import SuffixFamily

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
        split_miss_rate: Optional[float] = None,
    ):
        self.pst = pst
        self.family = SuffixFamily(pst, vs)

        # The discrimination tree owns the structure -- midfixes, branches and
        # leaves -- and calls back into is_accept for every classification.
        self.tree = MidfixTree()

        # The partial transition function, its witnesses, the per-state access
        # strings and the queue of edges still to resolve.
        self.dfa = PartialDFA(pst.alphabet_size)

        # The sequential population test: it accumulates each leaf's members and
        # says whether a proposed distinguisher splits it.
        self.splits = SplitEvidence(
            pst,
            self.family,
            pool_members=self._leaf_members,
            num_states=lambda: self.tree.num_states,
            split_fpr=split_fpr,
            **({} if split_miss_rate is None else {"split_miss_rate": split_miss_rate}),
        )

        # Boundary strings encountered while *building* the DFA: any ``member + c``
        # that sifts to None during transition resolution / consistency checking.
        # The current family can't place these; the driver feeds them back into
        # the representative pool so FNR forces the next family to resolve them.
        self.indecisive: Set[Tuple[int, ...]] = set()

    @property
    def num_states(self) -> int:
        """Leaf count; the tree allocates the ids as it splits."""
        return self.tree.num_states

    # -- membership / classification ---------------------------------------

    def _sift_prefill(self, seqs) -> None:
        """Warm the cache for sifting every string in ``seqs``, one batched call
        per tree level rather than one per node visited."""
        for pairs in self.tree.sift_levels(seqs, self.family.is_accept):
            self.family.prefill([list(s) + list(m) for s, m in pairs])

    def sift_and_boundary(self, seq) -> Tuple[Optional[int], Optional[tuple]]:
        """Route ``seq`` through the tree: ``(leaf, None)``, or ``(None,
        boundary)`` when some node classifies it indecisively."""
        return self.tree.sift(seq, self.family.is_accept)

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
        return self.tree.first_disagreement(s, sprime, self.family.is_accept, prefix)

    def split(self, state: int, distinguisher: tuple) -> int:
        """Refine leaf ``state`` into ``{True: state, False: new_state}`` under
        ``distinguisher`` and return the new state id.

        The tree refines the leaf and the partial DFA re-opens exactly the edges
        that refinement made ambiguous (see :meth:`PartialDFA.split_state`).
        """
        new_state = self.tree.split(state, distinguisher)
        self.dfa.split_state(state, new_state)
        self.splits = self.splits.after_split(state, self.sift)
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
                    self.splits.record(state, w[:i])
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
                self.splits.record(nxt, w[: i + 1])
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

    def to_dfa_and_tree(self) -> Tuple[DFA, DecisionTree]:
        """Export the learned automaton as ``(DFA, DecisionTree)``, matching the
        shape returned by ``resolve_dfa``."""
        return export_dfa(
            self.tree, self.dfa, self.family, self.pst, self._decisive_target
        )


def export_dfa(tree, partial, family, pst, decisive_target) -> Tuple[DFA, DecisionTree]:
    """The learned automaton as ``(DFA, DecisionTree)``.

    Any edge the worklist left open is filled from a decisive leaf member rather
    than by sifting only the access string -- a single indecisive access
    continuation used to fall back to a bogus self-loop, wrecking the exported
    DFA even when the tree was correct.  Only an edge whose *entire* leaf is
    indecisive still self-loops."""
    transitions, unresolved = partial.totalise(
        range(tree.num_states), lambda s, c: decisive_target(s, c)[0]
    )
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


class _Best:
    """The most accurate hypothesis seen so far.  Rounds are not monotone -- a
    later family can classify worse -- so the run returns its best, not its last."""

    def __init__(self):
        self.accuracy = -1.0
        self.dfa = None
        self.dt = None
        self.boundary = 0.0

    def offer(self, accuracy, dfa, dt, boundary) -> None:
        if accuracy > self.accuracy:
            self.accuracy, self.dfa, self.dt, self.boundary = (
                accuracy,
                dfa,
                dt,
                boundary,
            )


class _StallDetector:
    """Stops a run that has started repeating itself.

    Some targets are unlearnable with a fixed-length prefix sampler (transient
    states it never lands on, #128).  The FNR gate then finds no new boundary
    strings and the round is a byte-for-byte repeat of the last, so the remaining
    rounds would burn queries for nothing.  Two consecutive rounds with no new
    states, no accuracy gain and no new boundary strings confirm the fixpoint."""

    def __init__(self, patience: int = 2):
        self._patience = patience
        self._states = 0
        self._boundary_strings = 0
        self._stalled = 0

    def stalled(self, *, states: int, improved: bool, boundary_strings: int) -> bool:
        progressed = (
            states > self._states
            or improved
            or boundary_strings > self._boundary_strings
        )
        self._stalled = 0 if progressed else self._stalled + 1
        self._states, self._boundary_strings = states, boundary_strings
        return self._stalled >= self._patience


def _discover(pst, vs, *, max_probes: int, patience: int):
    """One round: close the hypothesis, hunt counterexamples, close it again.

    The discovery pass samples fresh strings and splits on DFA-vs-tree
    disagreements (the equivalence-oracle role).  Its binary search homes in on
    the errors, which sit at boundary states, so it *also* harvests the boundary
    (sift -> None) strings that feed the FNR gate -- densely and targeted, so a
    probe need not end at the boundary.  This subsumes the separate consistency
    check (pool verification + boundary sweep): dropping it brings the substring
    case to E-L* query parity with no loss of convergence across seeds."""
    learner = DirectLStarLearner(pst, vs)
    learner.init_worklist()
    learner.run_worklist()
    learner.counterexample_pass(
        max_probes=max_probes, patience=patience, boundary_target=10**9
    )
    learner.run_worklist()
    dfa, dt = learner.to_dfa_and_tree()
    return learner, dfa, dt


def _estimate_accuracy(pst, dfa, dt, acc_threshold: float) -> float:
    """Agreement between the exported DFA and the tree read decisively -- the
    termination test.  The tree is re-read with both thresholds at the decision
    boundary so it answers every string rather than abstaining."""
    from .lstar import estimate_agreement_rate

    boundary = pst.decision_boundary
    decisive = dt.map_over_predicates(lambda p, b=boundary: TriPredicate(p.vs, b, b))
    return estimate_agreement_rate(
        pst,
        pst.sampler,
        pst.oracle,
        decisive,
        dfa,
        num_samples=2000,
        acc_threshold=acc_threshold,
    )


def _default_patience(acc_threshold: float) -> int:
    """Consecutive clean probes that end a discovery pass.

    If the DFA-vs-tree disagreement rate were still at the tolerated level
    ``eps = 1 - acc_threshold``, seeing k clean probes in a row has probability
    ``acc_threshold ** k``, so ``k = ceil(ln(alpha) / ln(acc_threshold))`` makes
    stopping early a ``<= alpha`` event.  This is a cost knob -- the outer
    estimate and the next round both verify -- so a modest alpha suffices (149 at
    acc_threshold 0.98, ~300 at 0.99)."""
    return math.ceil(math.log(0.05) / math.log(acc_threshold))


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
    """Learn a DFA, forcing the suffix family to resolve boundary states.

    Each round, the strings the family cannot classify (``sift -> None`` --
    measured to be, cleanly, the indecisive boundary states) are added to the
    *representative* pool.  ``sample_suffix_family`` then sees a high FNR over
    them and re-clusters to a family that classifies them decisively, dropping
    the "completing" suffixes that were diluting them, so the next round can
    place them and split."""
    if counterexample_patience is None:
        counterexample_patience = _default_patience(acc_threshold)

    first_round = True
    best = _Best()
    stall = _StallDetector()
    # Kept across rounds: the FNR gate resolves the chain one state per round, so
    # earlier rounds' indecisives keep the family honest about the whole chain
    # (they turn decisive once their state is resolved).
    accumulated: List[List[int]] = []
    seen: Set = set()

    for round_idx in range(max_rounds):
        prior_best = best.accuracy
        vs, boundary = sample_suffix_family(
            pst, pst.table.intern_suffix([]), first_round=first_round
        )
        pst.decision_boundary = boundary
        first_round = False

        learner, dfa, dt = _discover(
            pst,
            vs,
            max_probes=counterexample_probes,
            patience=counterexample_patience,
        )
        true_acc = _estimate_accuracy(pst, dfa, dt, acc_threshold)
        best.offer(true_acc, dfa, dt, pst.decision_boundary)
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
        if stall.stalled(
            states=learner.num_states,
            improved=true_acc > prior_best + 1e-9,
            boundary_strings=len(accumulated),
        ):
            print(
                f"[direct-lstar/fnr] round {round_idx}: no progress "
                f"({learner.num_states} states) -- target unresolvable with this "
                "sampler, stopping"
            )
            break

    # The structural labeling (leaves on the root's accept side) can flip a
    # low-support state under noise, especially asymmetric noise; a resample plus
    # binomial test per reachable state corrects it.  Same step the resolver
    # pipeline applies at the end.
    from .lstar import denoise_accept_labels

    pst.decision_boundary = best.boundary
    return denoise_accept_labels(pst, best.dfa), best.dt
