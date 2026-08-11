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
from typing import Dict, Optional, Set

#: Tolerated miss rate (beta) -- the lower sequential boundary, ``logBF <=
#: log(beta)``, at which a leaf is accepted as one state and stops being probed.
#: Only a two-sided population whose halves score the same rate reaches it; a
#: one-sided population scores 0 and stays UNDECIDED.
DEFAULT_SPLIT_MISS_RATE = 0.02

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
        pool_representative,
        num_states,
        split_fpr: Optional[float],
        split_miss_rate: float,
        members: Dict[int, Set[tuple]],
        open_candidates: Optional[Dict[int, Dict[tuple, dict]]] = None,
    ):
        self.pst = pst
        self.family = family
        self._pool_members = pool_members
        self._pool_representative = pool_representative
        self._num_states = num_states
        self._split_fpr = split_fpr if split_fpr is not None else pst.config.split_pval
        # Tolerated miss rate (beta): the lower sequential boundary.
        self._split_miss_rate = split_miss_rate
        # How many members one candidate is built from, so a populous leaf does
        # not scan the whole pool.  The probe stream never reaches it.
        self._pool_member_limit = 1500
        # Distinct prefixes seen to reach each leaf while sifting probes.  A split
        # fires once enough have piled up for the Bayes factor to cross, so the
        # probe stream -- not the fixed pool -- is what drives a leaf to resolve.
        self.members: Dict[int, Set[tuple]] = {k: set(v) for k, v in members.items()}
        # leaf -> distinguisher -> running sufficient statistics.  Carried across
        # a split for every leaf the split did not touch.
        self._open: Dict[int, Dict[tuple, dict]] = dict(open_candidates or {})

    # -- accumulating evidence ----------------------------------------------

    def record(self, state: int, prefix) -> None:
        """Note that ``prefix`` decisively sifts to ``state``.  New members fold
        into any open candidate on that leaf right away, so the Bayes factor stays
        O(1) to read rather than being recomputed over the population."""
        bucket = self.members.setdefault(state, set())
        if tuple(prefix) in bucket:
            return
        bucket.add(tuple(prefix))
        for distinguisher, accum in self._open.get(state, {}).items():
            self._fold(accum, distinguisher, prefix)

    def representative(self, state: int) -> Optional[list]:
        """A canonical string reaching ``state`` -- the shortest member, ties
        broken lexicographically.

        Falls back to the pool when the probe stream has not reached the leaf yet
        and keeps what it finds as a member, so the scan happens once and the
        choice is stable while the leaf lives.  ``None`` means nothing known
        reaches the leaf, so its edges cannot be resolved."""
        bucket = self.members.get(state)
        if bucket:
            return list(min(bucket, key=lambda m: (len(m), m)))
        found = self._pool_representative(state)
        if found is None:
            return None
        self.record(state, found)
        return list(found)

    def after_split(self, state: int, sift) -> "SplitEvidence":
        """Evidence for the tree left by splitting ``state``.

        Only the split leaf is affected.  Its members move -- re-sifted into the
        two halves -- and its candidates die, because its population bifurcated
        under them.  Every other leaf keeps both: a split replaces one leaf with
        an internal node, so no other leaf's path through the tree changed, and an
        accumulator depends on nothing but its members and its distinguisher.

        Members survive rather than starting empty because a newly created,
        still-conflated leaf could never gather the members its own split needs
        before the pass ends."""
        carried = {k: set(v) for k, v in self.members.items() if k != state}
        for member in self.members.get(state, ()):
            landed = sift(list(member))
            if landed is not None:
                carried.setdefault(landed, set()).add(member)
        carried_open = {k: v for k, v in self._open.items() if k != state}
        return SplitEvidence(
            self.pst,
            self.family,
            pool_members=self._pool_members,
            pool_representative=self._pool_representative,
            num_states=self._num_states,
            split_fpr=self._split_fpr,
            split_miss_rate=self._split_miss_rate,
            members=carried,
            open_candidates=carried_open,
        )

    # -- weighing it ---------------------------------------------------------

    def verdict(self, state: int, distinguisher: tuple) -> str:
        """Weigh the proposed split: ``SPLIT`` / ``NO_SPLIT`` / ``UNDECIDED``."""
        accum = self._candidate(state, distinguisher)
        logbf = self._log_bayes_factor(accum)
        if logbf >= self._split_threshold():
            return SPLIT
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
        # ART: [A_true, R_true, A_false, R_false] TEST-half counts per side.
        accum = {"ART": [0, 0, 0, 0], "seen": set()}
        cands[distinguisher] = accum
        # Probe-seen members first, then the pool, up to the cap.  Their family
        # queries are batched in one call so the population packs, rather than one
        # |vs|-sized batch per member.
        members = list(
            dict.fromkeys(
                [tuple(t) for t in self.members.get(state, ())]
                + [
                    tuple(p)
                    for p in self._pool_members(state, limit=self._pool_member_limit)
                ]
            )
        )[: self._pool_member_limit]
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

    def _log_bayes_factor(self, accum: dict) -> float:
        """Held-out log Bayes factor for the split: one pooled TEST-half accept
        rate against two.  The ASSIGN half did the grouping, so scoring on the
        disjoint TEST half cannot confirm the noise that produced the grouping."""
        if not self.family.test_idx:
            return float("-inf")
        return self._log_bf_scores(accum)

    @staticmethod
    def _log_bf_scores(accum: dict) -> float:
        """One pooled Beta-Bernoulli rate (a single state) against two (a real
        split), over the TEST-half votes.

        Note this is exactly 0 when either side is empty: a two-rate model whose
        second rate has no data *is* the one-rate model.  So a wholly one-sided
        population scores 0 and stays UNDECIDED -- only a genuine rate difference
        between two populated sides moves the verdict."""
        a1, r1, a2, r2 = accum["ART"]
        return (
            _log_beta(1 + a1, 1 + r1)
            + _log_beta(1 + a2, 1 + r2)
            - _log_beta(1 + a1 + a2, 1 + r1 + r2)
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
