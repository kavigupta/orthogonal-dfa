"""
Sequential population test that decides whether a leaf splits.

A proposed distinguisher is weighed against the leaf's members until one of two
tests fires.

1. split. We group the sides on the train half, and check if they differ in accept rate
  on the held-out test half, and the Bayes factor clears the Bonferroni threshold.
2. no split. The members agree closely enough to rule out a split of at least
  _MIN_DETECTABLE_SPLIT at the tolerated miss rate. This is a binomial test
  on the minority count.

Otherwise the verdict is undecided and more members accumulate before the next.
"""

import math

import scipy.stats

#: The chance of accepting a leaf as one state when a split (of size at least _MIN_DETECTABLE_SPLIT) really exists.
DEFAULT_SPLIT_MISS_RATE = 0.02

#: Smallest split fraction that we attempt to detect.
_MIN_DETECTABLE_SPLIT = 0.1

#: Members weighed per leaf.  The tests converge well below this; it just caps a
#: populous leaf.
_MEMBER_LIMIT = 1500

SPLIT = "split"
NO_SPLIT = "no_split"
UNDECIDED = "undecided"


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
        self._split_miss_rate = split_miss_rate

    def _members(self, state: int):
        return self._population.members(self._tree.path_of(state), _MEMBER_LIMIT)

    def representative(self, state: int):
        """A canonical string reaching ``state`` -- the shortest member, ties
        broken lexicographically -- or ``None`` if nothing known reaches it."""
        members = self._members(state)
        return min(members, key=lambda m: (len(m), m)) if members else None

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
        """
        Group the leaf's members by the train half and count the disjoint
        test half per side:

        Returns (A_true, R_true, A_false, R_false, n_true, n_false)
            where A means accept, R means reject
            and true/false is the grouping into each side of the distinguisher.

        Indecisive members contribute nothing.
        """
        members = self._members(state)
        self.family.prefill([member + list(distinguisher) for member in members])
        a1 = r1 = a2 = r2 = n_a = n_b = 0
        test = self.family.test_idx
        for member in members:
            votes = self.family.votes(member, distinguisher)
            group = self.family.train_side(votes)
            if group is None:
                continue
            accepts = sum(votes[i] for i in test)
            if group:
                a1, r1, n_a = a1 + accepts, r1 + len(test) - accepts, n_a + 1
            else:
                a2, r2, n_b = a2 + accepts, r2 + len(test) - accepts, n_b + 1
        return a1, r1, a2, r2, n_a, n_b

    def _agrees_as_one_state(self, n_a: int, n_b: int) -> bool:
        """
        At least one side is too small for a split of size _MIN_DETECTABLE_SPLIT to be present.
        """
        total = n_a + n_b
        if total == 0:
            return False
        return (
            scipy.stats.binom.cdf(min(n_a, n_b), total, _MIN_DETECTABLE_SPLIT)
            <= self._split_miss_rate
        )

    @staticmethod
    def _log_bf_scores(a1: int, r1: int, a2: int, r2: int) -> float:
        """
        One pooled Beta-Bernoulli rate (a single state) against two (a real
        split), over the test-half votes.  This is the split test's statistic.
        """
        return (
            _log_beta(1 + a1, 1 + r1)
            + _log_beta(1 + a2, 1 + r2)
            - _log_beta(1 + a1 + a2, 1 + r1 + r2)
        )

    def _split_threshold(self) -> float:
        """
        The minimum log Bayes factor a split must clear.

        Under the one-state null a Bayes factor exceeds K only with probability
            <= 1/K
        We can Bonferroni-correct that for the number of edges that could split, giving
            <= n/K
        which then requires K > n/fpr to hold the overall false positive rate at fpr
        """
        n = max(self._num_states() * self.pst.alphabet_size, 1)
        return math.log(n / max(self._split_fpr, 1e-12))
