"""
Sequential population test that decides whether a leaf splits.

A proposed distinguisher is weighed against the leaf's members until one of two
tests fires.  They answer different questions, so they are separate hypothesis
tests rather than one Bayes factor:

  SPLIT     -- the two sides, grouped on the assign half, differ in accept rate
               on the held-out test half.  A real second state reproduces across
               the disjoint halves; noise that merely scattered members across
               sides does not.  (Beta-Bernoulli Bayes factor, one pooled rate vs
               two, cleared against a Bonferroni threshold.)
  NO_SPLIT  -- the members agree closely enough to rule out a split of at least
               _MIN_DETECTABLE_SPLIT at the tolerated miss rate; the leaf is one
               state and stops being probed.  (binomial test on the minority
               count; rarer splits are left to the FNR loop.)

Otherwise the verdict is UNDECIDED and more members accumulate before the next.

The members themselves live in a :class:`~orthogonal_dfa.l_star.leaf_population.
LeafPopulation` shared with the edge resolver; this is a stateless reader of it.
"""

import math

import scipy.stats

#: Tolerated miss rate (beta) for the one-state test: the chance of accepting a
#: leaf as one state when a split of _MIN_DETECTABLE_SPLIT really exists.
DEFAULT_SPLIT_MISS_RATE = 0.02

#: Smallest minority fraction the one-state test resolves within a round.  A leaf
#: is confirmed one state once a split this large is ruled out; rarer second
#: states are resurfaced by the FNR loop across rounds.
_MIN_DETECTABLE_SPLIT = 0.1

#: Members weighed per leaf.  The tests converge well below this; it just caps a
#: populous leaf.
_MEMBER_LIMIT = 1500

# Outcome of weighing one proposed split.
SPLIT = "split"
NO_SPLIT = "no_split"  # the leaf is one state at this distinguisher; stop probing
UNDECIDED = "undecided"  # not yet conclusive -- keep sifting members in


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


class SplitEvidence:
    """See the module docstring.  A stateless reader over ``population`` and
    ``tree``: it turns a leaf id into a path, pulls that leaf's members, and
    weighs a proposed distinguisher against them."""

    def __init__(
        self,
        pst,
        family,
        *,
        population,
        tree,
        num_states,
        split_fpr,
        split_miss_rate: float,
    ):
        self.pst = pst
        self.family = family
        self._population = population
        self._tree = tree
        self._num_states = num_states
        self._split_fpr = split_fpr if split_fpr is not None else pst.config.split_pval
        # Tolerated miss rate (beta) for the one-state test.
        self._split_miss_rate = split_miss_rate

    def _members(self, state: int):
        return self._population.members(self._tree.path_of(state), _MEMBER_LIMIT)

    def representative(self, state: int):
        """A canonical string reaching ``state`` -- the shortest member, ties
        broken lexicographically -- or ``None`` if nothing known reaches it."""
        members = self._members(state)
        return min(members, key=lambda m: (len(m), m)) if members else None

    # -- weighing it ---------------------------------------------------------

    def verdict(self, state: int, distinguisher: tuple) -> str:
        """Weigh the proposed split with two tests: ``SPLIT`` if the held-out
        sides differ in rate, ``NO_SPLIT`` if the members agree closely enough to
        rule out a split, else ``UNDECIDED``."""
        assert self.family.test_idx  # vs is sized >= suffix_family_size, never empty
        a1, r1, a2, r2, n_a, n_b = self._tally(state, distinguisher)
        if self._log_bf_scores(a1, r1, a2, r2) >= self._split_threshold():
            return SPLIT
        if self._agrees_as_one_state(n_a, n_b):
            return NO_SPLIT
        return UNDECIDED

    def _tally(self, state: int, distinguisher: tuple):
        """Group the leaf's members by the ASSIGN half and count the disjoint
        TEST half per side: ``(A_true, R_true, A_false, R_false, n_true,
        n_false)``.  Indecisive members contribute nothing.  Family queries are
        batched in one call so the population packs, rather than one per member."""
        members = self._members(state)
        self.family.prefill([member + list(distinguisher) for member in members])
        a1 = r1 = a2 = r2 = n_a = n_b = 0
        test = self.family.test_idx
        for member in members:
            votes = self.family.votes(member, distinguisher)
            group = self.family.assign_side(votes)
            if group is None:
                continue
            accepts = sum(votes[i] for i in test)
            if group:
                a1, r1, n_a = a1 + accepts, r1 + len(test) - accepts, n_a + 1
            else:
                a2, r2, n_b = a2 + accepts, r2 + len(test) - accepts, n_b + 1
        return a1, r1, a2, r2, n_a, n_b

    def _agrees_as_one_state(self, n_a: int, n_b: int) -> bool:
        """The minority side is too small for a split of ``_MIN_DETECTABLE_SPLIT``:
        under such a split we would expect that fraction on the minority side, so
        seeing this few rules it out at the tolerated miss rate."""
        total = n_a + n_b
        if total == 0:
            return False
        return (
            scipy.stats.binom.cdf(min(n_a, n_b), total, _MIN_DETECTABLE_SPLIT)
            <= self._split_miss_rate
        )

    @staticmethod
    def _log_bf_scores(a1: int, r1: int, a2: int, r2: int) -> float:
        """One pooled Beta-Bernoulli rate (a single state) against two (a real
        split), over the TEST-half votes.  This is the SPLIT test's statistic.

        Note this is exactly 0 when either side is empty: a two-rate model whose
        second rate has no data *is* the one-rate model.  So splitting needs two
        populated sides that differ; the one-state test handles agreement."""
        return (
            _log_beta(1 + a1, 1 + r1)
            + _log_beta(1 + a2, 1 + r2)
            - _log_beta(1 + a1 + a2, 1 + r1 + r2)
        )

    def _split_threshold(self) -> float:
        """Log Bayes factor a split must clear.

        Under the "one Myhill-Nerode state" null the held-out factor concentrates
        near zero -- the two-rate model's Occam penalty cancels the fit -- so a
        spurious split needs an upward fluctuation it rarely produces
        (``P(BF > K) <= 1/K``).  Bonferroni over the hypotheses in play (one per
        leaf x symbol) at per-run false rate ``split_fpr`` gives
        ``logBF > log(num_states * |alphabet| / fpr)``.  Genuine splits scale their
        evidence with the member count and clear it; the bound grows only
        logarithmically as the tree does."""
        n = max(self._num_states() * self.pst.alphabet_size, 1)
        return math.log(n / max(self._split_fpr, 1e-12))
