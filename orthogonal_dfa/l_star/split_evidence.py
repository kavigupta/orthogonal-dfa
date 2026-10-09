"""
Sequential population test that decides whether a leaf splits.

A proposed distinguisher is weighed against the leaf's members until one of two
tests fires.

1. split. We group the sides on the train half, and check if they differ in accept rate
  on the held-out suffixes, which nothing else reads, by more than Hoeffding
  allows, Bonferroni-corrected.
2. no split. The members agree closely enough to rule out a split of at least
  _MIN_DETECTABLE_SPLIT at the tolerated miss rate. This is a binomial test
  on the minority count.

Otherwise the verdict is undecided and more members accumulate before the next.
"""

import math

from .statistics import binom_cdf

#: The chance of accepting a leaf as one state when a split (of size at least _MIN_DETECTABLE_SPLIT) really exists.
DEFAULT_SPLIT_MISS_RATE = 0.02

#: Smallest split fraction that we attempt to detect.
_MIN_DETECTABLE_SPLIT = 0.1

#: One-sided members a leaf needs before `_agrees_as_one_state` can rule a split
#: out of it.  Decisive ones: a member the family cannot place is not counted.
MEMBERS_TO_RULE_OUT_A_SPLIT = math.ceil(
    math.log(DEFAULT_SPLIT_MISS_RATE) / math.log(1 - _MIN_DETECTABLE_SPLIT)
)

#: Members weighed per leaf.  The tests converge well below this; it just caps a
#: populous leaf.
_MEMBER_LIMIT = 1500

SPLIT = "split"
NO_SPLIT = "no_split"
UNDECIDED = "undecided"


class SplitEvidence:
    """See the module docstring.  Turns a leaf id into a path, pulls that leaf's
    members, and weighs a proposed distinguisher against them.  Pulling is not a
    read: it settles the members onto the leaf and harvests what the family
    cannot place."""

    def __init__(
        self,
        pst,
        family,
        *,
        population,
        tree,
    ):
        self.pst = pst
        self.family = family
        self._population = population
        self._tree = tree
        self._split_fpr = pst.config.split_pval
        #: Held-out string -> the (leaf path, distinguisher) whose test first read it.
        self._first_read = {}
        self._split_miss_rate = DEFAULT_SPLIT_MISS_RATE

    def _members(self, state: int):
        return self._population.members(self._tree.path_of(state), _MEMBER_LIMIT)

    def verdict(self, state: int, distinguisher: bytes) -> str:
        """Weigh the proposed split with two tests: ``SPLIT`` if the held-out
        sides differ in rate, ``NO_SPLIT`` if the members agree closely enough to
        rule out a split, else ``UNDECIDED``."""
        return self._weigh(
            self._members(state),
            distinguisher,
            self._edge_count(),
            key=(self._tree.path_of(state), distinguisher),
        )

    def _edge_count(self) -> int:
        return self._tree.num_states * self.pst.alphabet_size

    def _weigh(self, members, distinguisher: bytes, tests: int, *, key) -> str:
        a1, t1, a2, t2, n_a, n_b = self._tally(members, distinguisher, key=key)
        if self._splits(a1, t1, a2, t2, tests=tests):
            return SPLIT
        if self._agrees_as_one_state(n_a, n_b):
            return NO_SPLIT
        return UNDECIDED

    def nothing_left_to_split(self, fillable) -> bool:
        """Whether no distinguisher the tree can propose still splits a state,
        and every state in ``fillable`` holds the members to say so.

        A state outside ``fillable`` is one no draw reaches, so holding the run
        open until it has enough members holds it open forever.
        """
        midfixes = self._tree.midfixes()
        candidates = [
            bytes([c]) + midfix
            for c in range(self.pst.alphabet_size)
            for midfix in midfixes
        ]
        # The correction is over the tests this makes, which is not the one per
        # edge `verdict` is asked for.
        tests = self._tree.num_states * len(candidates)
        for state in self._tree.leaves():
            members = self._members(state)
            blocking = {SPLIT, UNDECIDED} if state in fillable else {SPLIT}
            path = self._tree.path_of(state)
            if any(
                self._weigh(members, d, tests, key=(path, d)) in blocking
                for d in candidates
            ):
                return False
        return True

    def _tally(self, members, distinguisher: bytes, *, key):
        """
        Group ``members`` by the train half and count the held-out reads per
        side, each string once and only where no test but ``key``'s (a leaf's
        path and the distinguisher) read it first:

        Returns (A_true, T_true, A_false, T_false, n_true, n_false), the
        held-out accepts and reads and the member counts, where true/false is the
        grouping into each side of the distinguisher.

        Indecisive members contribute nothing.
        """
        self.family.prefill([member + distinguisher for member in members])
        seen = set()
        reads = []
        for member in members:
            group = self.family.train_side(self.family.votes(member, distinguisher))
            if group is None:
                continue
            strings = [
                s
                for s in self.family.held_out_strings(member + distinguisher)
                if s not in seen and self._first_read.setdefault(s, key) == key
            ]
            seen.update(strings)
            reads.append((group, strings))
        bits = iter(self.family.held_out_bits([s for _, ss in reads for s in ss]))
        accepts, trials, count = [0, 0], [0, 0], [0, 0]
        for group, strings in reads:
            accepts[group] += sum(next(bits) for _ in strings)
            trials[group] += len(strings)
            count[group] += 1
        return accepts[1], trials[1], accepts[0], trials[0], count[1], count[0]

    def _agrees_as_one_state(self, n_a: int, n_b: int) -> bool:
        """
        At least one side is too small for a split of size _MIN_DETECTABLE_SPLIT to be present.
        """
        total = n_a + n_b
        if total == 0:
            return False
        return (
            binom_cdf(min(n_a, n_b), total, _MIN_DETECTABLE_SPLIT)
            <= self._split_miss_rate
        )

    def _splits(self, a1: int, t1: int, a2: int, t2: int, *, tests: int) -> bool:
        """Whether the sides' held-out rates, ``a`` accepts of ``t`` reads each,
        differ by more than Hoeffding allows at the false positive rate over
        ``tests`` tests:

            2 t1 t2 / (t1 + t2) (a1 / t1 - a2 / t2)^2 >= log(2 tests / split_fpr)
        """
        if not (t1 and t2):
            return False
        statistic = 2 * t1 * t2 / (t1 + t2) * (a1 / t1 - a2 / t2) ** 2
        return statistic >= math.log(2 * tests / self._split_fpr)
