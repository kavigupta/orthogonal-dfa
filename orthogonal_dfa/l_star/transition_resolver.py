"""Incremental, transition-driven DFA discovery.

Builds the discrimination tree (states) and the transition function together.

The tree starts as the initial distinguisher family v_eps, partitioning the
prefix pool into accept / reject -- two leaves, the initial two states.  Each
(state, symbol) edge is resolved by sifting a member of the state extended by the
symbol: the leaf it lands on is the target, and the member is kept as the edge's
witness (the tree is consistent, so any member resolves it the same way).  A leaf
every one of whose members is indecisive, or that no prefix reaches, leaves its
edge open; the export totalises those -- self-looping them and feeding their
boundary strings back so the next round's family resolves them (see EdgeResolver).

States beyond the initial two are found by the counterexample pass: random probe
strings are walked through a *totalised* copy of the transition function and
re-sifted, and where the walk and the sift disagree the probe has exhibited two
prefixes that reach one leaf yet behave differently under one more symbol -- a
Myhill-Nerode counterexample -- so that leaf is split (see SplitEvidence).  A
split drops the edges it made ambiguous; both they and the new leaf's edges then
read as unresolved and are refilled on the next resolve pass.

Each state's prefixes -- the pool prefixes that sift to its leaf -- live in a
:class:`~orthogonal_dfa.l_star.leaf_population.LeafPopulation`.  The split keeps
the accept side as s itself and gives the reject side a fresh id (see
MidfixTree.split), so state ids stay a dense range(num_states) and need no
remapping on export.
"""

import numpy as np
from automata.fa.dfa import DFA

from .cluster import sample_suffix_family
from .edge_resolver import EdgeResolver
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

# The invariance gate averages this many random strings at each probe length.
_GATE_SAMPLES = 1500


def _probe_lengths(sample_length, count=10):
    """Prefix lengths at which to probe a distinguisher's effect -- the absolute
    positions it sits at.  Tied to the operating regime: spread over ``[floor,
    sample_length]``, where distinguishers actually get applied, rather than a fixed
    window (which would miss position-dependence living beyond it)."""
    floor = max(4, sample_length // 8)
    return tuple(sorted({int(x) for x in np.linspace(floor, sample_length, count)}))


def _position_effects(
    oracle, distinguisher, alphabet_size, *, lengths, samples, seed, base_rates
):
    """Per prefix length L: d's marginal log-odds effect against a same-length control
    (a random |d|-tail in place of d, so the length base-rate cancels), and that
    estimate's sampling variance.  ``base_rates`` caches the control accept-counts by
    total length so it is shared across distinguishers."""
    rng_d = np.random.default_rng(seed)
    rng_c = np.random.default_rng(seed + 1)
    d = list(distinguisher)
    n = samples
    dtail = np.broadcast_to(np.asarray(d, dtype=int), (n, len(d)))
    effects, variances = [], []
    for length in lengths:
        s = rng_d.integers(0, alphabet_size, (n, length))
        kd = int(
            np.sum(oracle.membership_queries(np.concatenate([s, dtail], 1).tolist()))
        )
        total = length + len(d)
        if base_rates is not None and total in base_rates:
            kc = base_rates[total]
        else:
            c = rng_c.integers(0, alphabet_size, (n, total)).tolist()
            kc = int(np.sum(oracle.membership_queries(c)))
            if base_rates is not None:
                base_rates[total] = kc
        # Haldane-Anscombe correction keeps the logit and its variance finite at 0/n.
        p_d = (kd + 0.5) / (n + 1)
        p_c = (kc + 0.5) / (n + 1)
        effects.append(np.log(p_d / (1 - p_d)) - np.log(p_c / (1 - p_c)))
        variances.append(1 / (n * p_d * (1 - p_d)) + 1 / (n * p_c * (1 - p_c)))
    return np.array(effects), np.array(variances)


def _log_bayes_factor(effects, variances, prior_scale):
    """log BF(tau>0 : tau=0) for effects ~ Normal(mu, tau^2) observed with known
    within-group variances.  The shared mean mu is marginalised under a flat prior (its
    improper constant cancels in the ratio) and tau under a half-normal(prior_scale);
    the tau integral is a 400-point grid quadrature."""
    e = np.asarray(effects, dtype=float)
    v = np.asarray(variances, dtype=float)

    def log_marginal(tau2):
        s2 = v + tau2
        w = 1.0 / s2
        mu = np.sum(w * e) / np.sum(w)
        return (
            -0.5 * np.sum(np.log(2 * np.pi * s2))
            - 0.5 * np.sum(w * (e - mu) ** 2)
            + 0.5 * np.log(2 * np.pi / np.sum(w))
        )

    lm0 = log_marginal(0.0)
    taus = np.linspace(0.0, 8.0 * prior_scale, 400)
    log_prior = np.log(np.sqrt(2 / np.pi) / prior_scale) - taus**2 / (
        2 * prior_scale**2
    )
    terms = np.array([log_marginal(t * t) for t in taus]) - lm0 + log_prior
    hi = float(np.max(terms))
    return hi + float(np.log(np.sum(np.exp(terms - hi)) * (taus[1] - taus[0])))


def distinguisher_position_log_bayes_factor(
    oracle,
    distinguisher,
    alphabet_size,
    *,
    sample_length,
    samples=_GATE_SAMPLES,
    seed=0,
    prior_scale=1.0,
    base_rates=None,
):
    """log Bayes factor that distinguisher d encodes absolute position rather than a
    transportable finite-memory feature.

    d's marginal log-odds effect at prefix length L, against a same-length control (a
    random |d|-tail), is a random effect over the positions it sits at -- probed across
    ``_probe_lengths(sample_length)``, i.e. the operating regime:

        e_L = logit P(accept | s + d) - logit P(accept | random),  |s| = L
        e_hat_L ~ Normal(e_L, v_L),   e_L ~ Normal(mu, tau^2)

    Returns log P(data | tau > 0) - log P(data | tau = 0), marginalising mu (flat) and
    tau (half-normal, scale ``prior_scale`` log-odds).  > 0 means the evidence favours a
    position-dependent effect (the positional-ladder pathology); < 0 favours a
    translation-invariant, regular feature.  The gate refuses a split when this is > 0 --
    no threshold, just which model the data prefer."""
    effects, variances = _position_effects(
        oracle,
        distinguisher,
        alphabet_size,
        lengths=_probe_lengths(sample_length),
        samples=samples,
        seed=seed,
        base_rates=base_rates,
    )
    return _log_bayes_factor(effects, variances, prior_scale)


class TransitionResolver:
    def __init__(self, pst, *, invariance_gate=False):
        self.pst = pst
        self.tree = None
        self.family = None
        self.sifter = None
        self.splits = None
        self.population = None  # pool prefixes, per leaf
        self.dfa = None  # the partial transition function
        self.edges = None  # resolves (leaf, symbol) edges against the population
        self.indecisive = set()  # boundary strings the family could not place

        # The invariance gate (off by default): refuse a split whose distinguisher the
        # Bayes factor says encodes absolute position rather than a transportable
        # feature.  Scores are cached per distinguisher (the probe is the same each
        # time), and the control base-rates are shared across them.
        self.invariance_gate = invariance_gate
        self._log_bf_cache = {}
        self._base_rate_cache = {}

    # -- membership / population -------------------------------------------

    def _classify(self, strings, midfix):
        """Which side of ``midfix`` each string sits on; the indecisive band
        between the thresholds returns None and drops out of the population."""
        self.family.prefill([list(s) + list(midfix) for s in strings])
        return [self.family.is_accept(s, midfix) for s in strings]

    def _sift(self, seq):
        """The leaf ``seq`` sifts to, or ``None`` when a node cannot place it.

        Every string the tree cannot place is harvested into ``indecisive``: it is
        a boundary string the current family straddles, and the driver feeds these
        back so the next family is forced to resolve them."""
        leaf, boundary = self.sifter.sift_and_boundary(list(seq))
        if leaf is None:
            self.indecisive.add(boundary)
        return leaf

    def _split(self, state_id, midfix):
        # The population re-sifts state_id's prefixes on the next members() call.
        new_id = self.tree.split(state_id, midfix)
        self.dfa.split_state(state_id, new_id)

    # -- counterexamples ----------------------------------------------------

    def counterexample_pass(self, *, max_probes, patience):
        """Split in place on DFA-vs-tree disagreements until they dry up.

        Each probe is walked through a totalised delta and re-sifted; where they
        disagree, the probe has exhibited two prefixes reaching one leaf that
        behave differently under one more symbol, so the leaf is split -- the same
        counterexample the outer loop used to defer by adding a prefix and
        rebuilding. Stops after ``patience`` consecutive clean probes."""
        since_split = 0
        delta = self._total_delta()
        for w in self._probe_blocks(max_probes):
            status = self._process(w, delta)
            if status == _SPLIT:
                since_split = 0
                self.edges.close()  # the split dropped edges; refill
                delta = self._total_delta()  # the split rewrote the state set
            elif status == _UNDECIDED:
                since_split = 0
            else:
                since_split += 1
            if since_split >= patience:
                break

    def _total_delta(self):
        """A total transition function to walk.  Edge resolution cannot always
        close an edge -- a leaf every one of whose members is indecisive has no
        successor the family can name -- so the remainder is filled here, once,
        rather than every probe that passes through it re-sifting."""
        delta, _ = self.dfa.totalise(
            range(self.tree.num_states),
            lambda s, c: self.edges.decisive_target(s, c)[0],
        )
        return delta

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

    def _process(self, w, delta):
        """Anchor at the shortest prefix the tree places, follow the total delta,
        then act on where the walk and a fresh sift disagree."""
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
            state = delta[state][c]
            states.append(state)
        return self._act_on_disagreement(w, states, start)

    def _act_on_disagreement(self, w, states, agree_point):
        state = states[-1]
        actual = self._sift(w)
        if actual is None or state is None or actual == state:
            return _RESOLVED
        fd = self._first_bad_edge(w, states, agree_point, len(w))
        if fd is None:
            return _RESOLVED
        s1, c, s2 = states[fd - 1], w[fd - 1], states[fd]
        if s1 is None or s2 is None:
            return _RESOLVED
        # The walk follows a totalised delta, so the followed edge s1 -(c)-> s2 can
        # be a gap the totaliser self-looped rather than a resolved DFA edge -- then
        # the DFA holds no such edge, there is no witness to separate on, and the
        # re-sifts need not reach s1.  Any of those means the disagreement is not
        # one we can act on.
        if self.dfa.target(s1, c) != s2:
            return _RESOLVED
        witness = self.dfa.witness(s1, c)
        if witness is None:
            return _RESOLVED
        sprime = w[: fd - 1]
        if self._sift(witness) != s1 or self._sift(sprime) != s1:
            return _RESOLVED
        distinguisher = self.sifter.disagreement(witness, sprime, [c])
        if distinguisher is None:
            return _RESOLVED
        verdict = self.splits.verdict(s1, distinguisher)
        if verdict != SPLIT:
            return _RESOLVED if verdict == NO_SPLIT else _UNDECIDED
        # The invariance gate: refuse a distinguisher whose effect the Bayes factor
        # says encodes absolute position rather than a transportable finite-memory
        # feature.  A regular target's distinguishers are position-invariant and pass;
        # a positional target's are refused, so the shift-register ladder never forms.
        # Only would-be splits pay for the probe, and it is cached per distinguisher.
        if self.invariance_gate and self._encodes_position(distinguisher):
            return _RESOLVED
        self._apply_split(s1, distinguisher, witness, sprime)
        return _SPLIT

    def _encodes_position(self, distinguisher):
        key = tuple(distinguisher)
        if key not in self._log_bf_cache:
            self._log_bf_cache[key] = distinguisher_position_log_bayes_factor(
                self.pst.oracle,
                key,
                self.pst.alphabet_size,
                sample_length=self.pst.sampler.length,
                base_rates=self._base_rate_cache,
            )
        return self._log_bf_cache[key] > 0

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
        self.edges = EdgeResolver(
            self.dfa, self.sifter, self.indecisive, population=self.population
        )
        self.edges.close()

    # -- output -------------------------------------------------------------

    def export(self):
        pst = self.pst
        n = self.tree.num_states

        transitions, unresolved = self.dfa.totalise(
            range(n), lambda s, c: self.edges.decisive_target(s, c)[0]
        )
        for state, c in unresolved:
            print(
                f"transition_resolver: no decisive edge for (state {state}, symbol "
                f"{c}); falling back to a self-loop"
            )

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
