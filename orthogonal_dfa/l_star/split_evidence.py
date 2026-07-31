"""The sequential population test that decides whether a leaf splits.

A probe can exhibit a Myhill-Nerode violation on its own, but under a noisy
oracle one disagreement is not evidence: it could be a flipped bit.  So a
proposed distinguisher only *opens* a hypothesis, and this accumulates the leaf's
members against it until the evidence is conclusive one way or the other.

The statistic is a held-out Beta-Bernoulli Bayes factor -- one pooled accept rate
(the leaf is one state) against two (it is really two).  Held out because the
family's ASSIGN half groups each member and only its TEST half scores the
grouping, so the test cannot confirm the very noise that produced the grouping
(see :class:`SuffixFamily`).  The verdict is three-way against two boundaries:
split above the upper one, accept the leaf as a single state below the lower one,
and otherwise leave the hypothesis open so more members accumulate.

An instance belongs to one tree shape.  A split rewrites the tree, so the
distinguishers of every open candidate may now cross the freshly inserted node --
:meth:`after_split` returns the evidence for the refined tree, dropping the
candidates but carrying the members over, since a leaf that started empty could
never gather the members its own split needs.
"""

import math
from statistics import NormalDist
from typing import Dict, Optional, Set

#: Tolerated miss rate (beta) -- the lower sequential boundary, ``logBF <=
#: log(beta)``, at which a leaf is accepted as one state and stops being probed.
#: Coupled to _NULL_ASSIGNMENT_PRIOR: a wholly one-sided population scores about
#: -4.4, which clears log(0.02) but not log(0.01), so a tighter beta than this
#: would put the boundary out of reach again.
DEFAULT_SPLIT_MISS_RATE = 0.02

#: Shape of the Beta prior on the split proportion under the *null*.  Under one
#: state every member should land on the same side of the distinguisher, so the
#: proportion is pinned near 0 or 1 -- a U-shaped Beta(e, e) with small e.  Under
#: a real split it is a free parameter, Beta(1, 1).
_NULL_ASSIGNMENT_PRIOR = 0.1

# Outcome of weighing one proposed split.
SPLIT = "split"
NO_SPLIT = "no_split"  # the leaf is one state at this distinguisher; stop probing
UNDECIDED = "undecided"  # not yet conclusive -- keep sifting members in


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


class SplitEvidence:
    """See the module docstring.

    ``pool_members(state, limit)`` supplies members beyond those the probe stream
    has shown, and is called only when a candidate is first opened -- it scans the
    prefix pool, so it must stay lazy.
    """

    def __init__(
        self,
        pst,
        family,
        *,
        pool_members,
        num_states,
        split_fpr: Optional[float],
        split_miss_rate: float,
        members: Dict[int, Set[tuple]],
    ):
        self.pst = pst
        self.family = family
        self._pool_members = pool_members
        self._num_states = num_states
        self._split_fpr = split_fpr if split_fpr is not None else pst.config.split_pval
        # Tolerated miss rate (beta): the lower sequential boundary.
        self._split_miss_rate = split_miss_rate
        self._member_cap = 1500
        self._min_members = 12
        # Distinct prefixes seen to reach each leaf while sifting probes.  A split
        # fires once enough have piled up for the Bayes factor to cross, so the
        # probe stream -- not the fixed pool -- is what drives a leaf to resolve.
        self.members: Dict[int, Set[tuple]] = {k: set(v) for k, v in members.items()}
        # leaf -> distinguisher -> running sufficient statistics.
        self._open: Dict[int, Dict[tuple, dict]] = {}

    # -- accumulating evidence ----------------------------------------------

    def record(self, state: int, prefix) -> None:
        """Note that ``prefix`` decisively sifts to ``state``.  New members fold
        into any open candidate on that leaf right away, so the Bayes factor stays
        O(1) to read rather than being recomputed over the population."""
        bucket = self.members.setdefault(state, set())
        if len(bucket) >= self._member_cap or tuple(prefix) in bucket:
            return
        bucket.add(tuple(prefix))
        for distinguisher, accum in self._open.get(state, {}).items():
            self._fold(accum, distinguisher, prefix)

    def after_split(self, state: int, sift) -> "SplitEvidence":
        """Evidence for the tree left by splitting ``state``.

        Candidates are dropped: their distinguishers may now cross the freshly
        inserted node.  Members survive -- only the split leaf's move, re-sifted
        into the two halves -- because a newly created, still-conflated leaf that
        started empty could never gather the members its own split needs before
        the pass ends."""
        carried = {k: set(v) for k, v in self.members.items() if k != state}
        for member in self.members.get(state, ()):
            landed = sift(list(member))
            if landed is not None:
                carried.setdefault(landed, set()).add(member)
        return SplitEvidence(
            self.pst,
            self.family,
            pool_members=self._pool_members,
            num_states=self._num_states,
            split_fpr=self._split_fpr,
            split_miss_rate=self._split_miss_rate,
            members=carried,
        )

    # -- weighing it ---------------------------------------------------------

    def verdict(self, state: int, distinguisher: tuple, witness, sprime) -> str:
        """Weigh the proposed split: ``SPLIT`` / ``NO_SPLIT`` / ``UNDECIDED``."""
        accum = self._candidate(state, distinguisher)
        logbf = self._log_bayes_factor(accum)
        if logbf >= self._split_threshold():
            return SPLIT
        if len(accum["seen"]) < self._min_members or not self.family.test_idx:
            # Underpowered here (a starved leaf, e.g. a trapped initial state that
            # only a handful of distinct strings reach).  Fall back to the per-pair
            # z-test on the two strings the disagreement already separated; a
            # populous leaf reaches the Bayes factor instead and never comes here.
            if self._pair_splits(witness, sprime, distinguisher):
                return SPLIT
            return UNDECIDED
        if logbf <= self._no_split_threshold():
            self._open.get(state, {}).pop(distinguisher, None)
            return NO_SPLIT
        return UNDECIDED

    def _candidate(self, state: int, distinguisher: tuple) -> dict:
        """The running evidence for ``(state, distinguisher)``, back-filled from
        the members seen so far on first use."""
        cands = self._open.setdefault(state, {})
        accum = cands.get(distinguisher)
        if accum is not None:
            return accum
        # ART: [A_true, R_true, A_false, R_false] TEST-half counts; N: members
        # assigned to each side, which is evidence in its own right.
        accum = {"ART": [0, 0, 0, 0], "N": [0, 0], "seen": set()}
        cands[distinguisher] = accum
        # Probe-seen members first, then the pool, up to the cap.  Their family
        # queries are batched in one call so the population packs, rather than one
        # |vs|-sized batch per member.
        members = list(
            dict.fromkeys(
                [tuple(t) for t in self.members.get(state, ())]
                + [tuple(p) for p in self._pool_members(state, limit=self._member_cap)]
            )
        )[: self._member_cap]
        self.family.prefill([list(m) + list(distinguisher) for m in members])
        for member in members:
            self._fold(accum, distinguisher, list(member))
        return accum

    def _fold(self, accum: dict, distinguisher, prefix) -> None:
        """Add ``prefix``'s TEST-half votes into its group's running sums, once."""
        key = tuple(prefix)
        if key in accum["seen"]:
            return
        accum["seen"].add(key)
        votes = self.family.votes(prefix, distinguisher)
        group = self.family.assign_side(votes)
        if group is None:
            return  # indecisive on the ASSIGN half; contributes no evidence
        accepts = sum(votes[i] for i in self.family.test_idx)
        side = 0 if group else 2
        accum["ART"][side] += accepts
        accum["ART"][side + 1] += len(self.family.test_idx) - accepts
        accum["N"][0 if group else 1] += 1

    def _log_bayes_factor(self, accum: dict) -> float:
        """Held-out log Bayes factor for the split, from two independent pieces
        of evidence: how the members were *assigned*, and how the TEST half
        *scored* them.  The halves use disjoint suffixes, so the two are
        conditionally independent given the hypothesis and their log factors
        add."""
        if len(accum["seen"]) < self._min_members or not self.family.test_idx:
            return float("-inf")
        return self._log_bf_scores(accum) + self._log_bf_assignment(accum)

    @staticmethod
    def _log_bf_scores(accum: dict) -> float:
        """One pooled Beta-Bernoulli rate (a single state) against two (a real
        split), over the TEST-half votes.

        Note this is exactly 0 when either side is empty: a two-rate model whose
        second rate has no data *is* the one-rate model, so the scores alone
        cannot see a wholly one-sided population.  That is what the assignment
        term is for."""
        a1, r1, a2, r2 = accum["ART"]
        return (
            _log_beta(1 + a1, 1 + r1)
            + _log_beta(1 + a2, 1 + r2)
            - _log_beta(1 + a1 + a2, 1 + r1 + r2)
        )

    @staticmethod
    def _log_bf_assignment(accum: dict) -> float:
        """How surprising the *grouping itself* is under each hypothesis.

        A real split predicts members on both sides, with the proportion a free
        parameter (uniform).  A single state predicts they all land together, the
        minority being classification noise -- a U-shaped Beta.  So a lopsided
        assignment is evidence against a split, and a balanced one mild evidence
        for it; without this the grouping carries no weight at all."""
        n_a, n_b = accum["N"]
        e = _NULL_ASSIGNMENT_PRIOR
        return _log_beta(1 + n_a, 1 + n_b) - (
            _log_beta(e + n_a, e + n_b) - _log_beta(e, e)
        )

    # -- the two boundaries --------------------------------------------------

    def _split_threshold(self) -> float:
        """Log Bayes factor a split must clear.

        Under the "one Myhill-Nerode state" null the held-out factor concentrates
        near zero -- the two-rate model's Occam penalty cancels the fit -- so a
        spurious split needs an upward fluctuation it rarely produces
        (``P(BF > K) <= 1/K``).  Bonferroni over the hypotheses currently open (one
        per leaf x symbol) at per-run false rate ``split_fpr`` gives
        ``logBF > log(num_states * |alphabet| / fpr)``.  Genuine splits scale their
        evidence with the member count and clear it; the bound grows only
        logarithmically as the tree does."""
        n = max(self._num_states() * self.pst.alphabet_size, 1)
        return math.log(n / max(self._split_fpr, 1e-12))

    def _no_split_threshold(self) -> float:
        """Log Bayes factor at or below which the leaf is accepted as one state
        for this distinguisher and stops being probed -- the lower sequential
        boundary from the tolerated miss rate (beta), ``logBF <= log(beta)``."""
        return math.log(max(self._split_miss_rate, 1e-12))

    def _starved_split_margin(self) -> float:
        """Confidence margin for the per-pair fallback the population factor
        cannot reach.  A genuinely starved leaf never gathers the members the
        Bayes factor needs, yet its few members can still span two states.  There
        the evidence is a single pair scoring on opposite sides, so this falls
        back to the resolver's z-test on the score difference ``D = f_s -
        f_sprime`` (mean 0, variance ``2 p (1-p) / m`` under one shared state).
        Bonferroni at ``split_fpr`` matches :meth:`_split_threshold`."""
        p = 0.5 + self.pst.config.min_signal_strength
        m = self.pst.config.suffix_family_size
        sigma_d = math.sqrt(2 * p * (1 - p) / m)
        tests = max(self._num_states() * self.pst.alphabet_size, 1)
        alpha = self._split_fpr / tests
        z = NormalDist().inv_cdf(1 - alpha / 2)
        return max(0.0, z * sigma_d / 2 - self.pst.evidence_margin)

    def _pair_splits(self, s, sprime, distinguisher) -> bool:
        """Whether ``s`` and ``sprime`` land on opposite *decisive* sides of
        ``distinguisher`` with the :meth:`_starved_split_margin`."""
        margin = self._starved_split_margin()
        d = self.family.is_accept(s, distinguisher, extra_margin=margin)
        dprime = self.family.is_accept(sprime, distinguisher, extra_margin=margin)
        return d is not None and dprime is not None and d != dprime
