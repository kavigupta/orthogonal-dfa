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

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
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

    def __init__(self, pst, vs: List[int]):
        self.pst = pst
        self.vs = list(vs)

        # Root: the empty prepend splits into state 0 (accept) and state 1
        # (reject).  Splitting only ever refines a leaf, so the root stays a
        # tuple and state 0's subtree stays entirely on the accept side.
        self.dt: Node = ((), {True: 0, False: 1})
        self.num_states = 2

        # transitions[s][c] -> target state, the current best guess for delta(s, c).
        self.transitions: Dict[int, Dict[int, int]] = {0: {}, 1: {}}
        # A prefix that provably reaches ``s`` and whose one-symbol extension by
        # ``c`` reaches ``transitions[s][c]``.  Only recorded when the source
        # state was reached by a *verified* sift, so it always sifts to ``s``.
        self.transition_witnesses: Dict[Tuple[int, int], List[int]] = {}
        # A canonical access string per state (sifts to that state), used to fill
        # in transitions never exercised by a probe when exporting the DFA.
        self.access: Dict[int, List[int]] = {}

        # Cache of prefix -> row for the one-hot membership lookups, rebuilt when
        # the table grows.
        self._prefix_pos: Optional[Dict[tuple, int]] = None
        self._prefix_pos_n = -1

    # -- membership / classification ---------------------------------------

    def _prefix_index(self, seq: List[int]) -> int:
        n = self.pst.table.num_prefixes
        if self._prefix_pos is None or self._prefix_pos_n != n:
            self._prefix_pos = {
                tuple(p): i for i, p in enumerate(self.pst.table.prefixes)
            }
            self._prefix_pos_n = n
        return self._prefix_pos[tuple(seq)]

    def is_accept(self, seq, prepend) -> Optional[bool]:
        """Confidently classify ``seq`` at a node whose distinguishers are the
        base family each prepended by ``prepend``.

        Returns ``True`` (accept) / ``False`` (reject) when the mean membership
        over the prepended family lands decisively past ``accept_thresh`` /
        ``reject_thresh``, and ``None`` in the indecisive band between them.  The
        decisive band is what keeps a single leaf from being split twice on the
        same (noisy) criterion.
        """
        seq = list(seq)
        # Add the probe prefix *unobserved*: a sift only ever needs this prefix's
        # decision at the tree nodes on its path, so eagerly observing it against
        # the whole fully-observed family (add_prefixes' default) would be pure
        # waste.  The cells actually needed are filled lazily by compute_decision.
        self.pst.table.ensure_prefixes([seq], do_observation=False)
        prepended = [
            self.pst.table.intern_suffix([*prepend, *self.pst.table.suffix(v)])
            for v in self.vs
        ]
        mask = np.zeros(self.pst.table.num_prefixes, dtype=bool)
        mask[self._prefix_index(seq)] = True
        decision = self.pst.compute_decision(prepended, mask)
        if decision[0] >= self.pst.accept_thresh:
            return True
        if decision[0] < self.pst.reject_thresh:
            return False
        return None

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
        d = self.is_accept(s, full)
        dprime = self.is_accept(sprime, full)
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

    def _invalidate_around(self, state: int) -> None:
        """Drop every transition incident to ``state`` (its leaf just became an
        internal node, so both its outgoing edges and the edges pointing at it
        are now ambiguous); they are re-derived lazily by future probes.  Edges
        not touching ``state`` are unaffected -- their witnesses still sift to
        the same, unmodified leaves -- so they stay valid."""
        for c in list(self.transitions.get(state, {})):
            self.transition_witnesses.pop((state, c), None)
        self.transitions[state] = {}
        for src, edges in self.transitions.items():
            for c in [c for c, tgt in edges.items() if tgt == state]:
                del edges[c]
                self.transition_witnesses.pop((src, c), None)

    def split(self, state: int, distinguisher: tuple) -> int:
        """Refine leaf ``state`` into ``{True: state, False: new_state}`` under
        ``distinguisher`` and return the new state id."""
        new_state = self.num_states
        self.num_states += 1
        self.transitions[new_state] = {}
        self.dt = self._replace_leaf(
            self.dt, state, (distinguisher, {True: state, False: new_state})
        )
        self._invalidate_around(state)
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
            if nxt is not None and verified:
                # Only record an edge whose source was reached by a real sift, so
                # the witness w[:i] genuinely sifts to ``state``.
                self.transitions[state][c] = nxt
                self.transition_witnesses[state, c] = w[:i]
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
        refiner) through :meth:`process`.  Returns the number of splits."""
        return sum(int(self.process(w)) for w in probes)

    def learn(self, *, max_probes: int, patience: Optional[int] = None) -> int:
        """Draw up to ``max_probes`` random probe strings.  Stops early after
        ``patience`` consecutive probes cause no split (``None`` disables early
        stopping).  Returns the number of splits performed."""
        splits = 0
        since_split = 0
        for _ in range(max_probes):
            w = self.pst.sampler.sample(self.pst.rng, self.pst.alphabet_size)
            if self.process(w):
                splits += 1
                since_split = 0
            else:
                since_split += 1
                if patience is not None and since_split >= patience:
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

    def _complete_transitions(self) -> None:
        """Fill any ``(state, symbol)`` edge no probe exercised by sifting the
        state's access string extended by the symbol, so the exported DFA is
        total.  A state with no access string, or an indecisive extension, falls
        back to a self-loop (noted rather than silently dropped)."""
        for state in range(self.num_states):
            access = self._find_access(state)
            for c in range(self.pst.alphabet_size):
                if c in self.transitions[state]:
                    continue
                target = self.sift(access + [c]) if access is not None else None
                if target is None:
                    print(
                        f"direct_lstar: no decisive edge for (state {state}, "
                        f"symbol {c}); falling back to a self-loop"
                    )
                    target = state
                self.transitions[state][c] = target

    def to_dfa_and_tree(self) -> Tuple[DFA, DecisionTree]:
        """Export the learned automaton as ``(DFA, DecisionTree)``, matching the
        shape returned by ``resolve_dfa``."""
        self._complete_transitions()

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

        transitions = {s: dict(self.transitions[s]) for s in range(self.num_states)}

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
    """Sample the base suffix family, run one round of the random-walk learner,
    and return the ``(DFA, DecisionTree)``.

    This is the direct analogue of a single ``resolve_dfa(pst, first_round=...)``
    call.  It is a *single round*: on rare / long-range patterns the boundary
    states that need splitting sit in the indecisive band and are seldom hit by
    uniform probes, so a single round collapses to the trivial accept/reject
    automaton.  Use :func:`synthesize_direct_lstar` to wrap this in a refinement
    loop that manufactures the missing prefixes.
    """
    v_idx = pst.table.intern_suffix([])
    vs, boundary = sample_suffix_family(pst, v_idx, first_round=first_round)
    pst.decision_boundary = boundary
    learner = DirectLStarLearner(pst, vs)
    learner.learn(max_probes=max_probes, patience=patience)
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


def synthesize_direct_lstar(
    pst,
    *,
    acc_threshold: float,
    additional_counterexamples: int = 200,
    max_probes_per_round: int = 4000,
    patience: Optional[int] = 800,
    max_rounds: int = 20,
    refine: Refiner = default_refine,
) -> Tuple[DFA, DecisionTree]:
    """Run the random-walk learner inside a refinement loop.

    Each round rebuilds the learner over the current (growing) prefix pool with a
    freshly calibrated suffix family, walks random probes plus every
    counterexample accumulated so far, and estimates the hypothesis's agreement
    with the oracle.  When agreement clears ``acc_threshold`` (or the refiner
    stops producing new prefixes, or ``max_rounds`` is hit) it returns the
    current ``(DFA, DecisionTree)``.

    The ``refine`` argument is the single, swappable point where new prefixes are
    manufactured; it defaults to :func:`default_refine`.  See that function's
    TODO -- the refiner is the part we expect to revisit.
    """
    # Imported here to avoid importing the resolver pipeline unless the loop runs.
    from .lstar import estimate_agreement_rate

    first_round = True
    extra_probes: List[List[int]] = []
    dfa = dt = None
    for round_idx in range(max_rounds):
        v_idx = pst.table.intern_suffix([])
        vs, boundary = sample_suffix_family(pst, v_idx, first_round=first_round)
        pst.decision_boundary = boundary
        first_round = False

        learner = DirectLStarLearner(pst, vs)
        learner.learn(max_probes=max_probes_per_round, patience=patience)
        # Re-walk the counterexamples gathered in earlier rounds: with the
        # recalibrated family they can now land decisively and drive splits.
        learner.process_probes(extra_probes)
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
        if true_acc >= acc_threshold:
            break

        new_probes = refine(
            pst, dfa, dt, dt_decisive, true_acc=true_acc, count=additional_counterexamples
        )
        if not new_probes:
            print("[direct-lstar] refiner produced no new prefixes; stopping")
            break
        extra_probes.extend(list(p) for p in new_probes)

    return dfa, dt
