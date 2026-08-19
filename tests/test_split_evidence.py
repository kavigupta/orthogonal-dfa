import unittest
from types import SimpleNamespace

from orthogonal_dfa.l_star.leaf_population import LeafPopulation
from orthogonal_dfa.l_star.midfix_tree import MidfixTree
from orthogonal_dfa.l_star.split_evidence import (
    NO_SPLIT,
    SPLIT,
    UNDECIDED,
    SplitEvidence,
)


class _StubFamily:
    """Classifies by caller-supplied rules, so no oracle is involved.

    The two halves are driven independently, because that is the whole point of
    the partition: ``side_of(prefix)`` groups a member on the train half
    (``None`` = indecisive there, contributing no evidence), and
    ``accept_rate(prefix)`` sets the fraction of TEST bits that score it.
    """

    test_idx = list(range(1, 20, 2))
    train_idx = list(range(0, 20, 2))

    def __init__(self, side_of=lambda p: True, accept_rate=None):
        self.side_of = side_of
        self.accept_rate = accept_rate
        self.prefilled = []

    def prefill(self, bases):
        self.prefilled.extend(bases)

    def votes(self, prefix, distinguisher):
        del distinguisher
        side = self.side_of(list(prefix))
        rate = self.accept_rate(list(prefix)) if self.accept_rate else float(bool(side))
        votes = [0] * 20
        for i in self.train_idx:
            votes[i] = 0 if side is None else (1 if side else 0)
        if side is None:  # straddle the train thresholds
            for i in self.train_idx[: len(self.train_idx) // 2]:
                votes[i] = 1
        for n, i in enumerate(self.test_idx):
            votes[i] = 1 if n < round(rate * len(self.test_idx)) else 0
        return votes

    def train_side(self, votes):
        mean = sum(votes[i] for i in self.train_idx) / len(self.train_idx)
        if mean >= 0.9:
            return True
        if mean < 0.1:
            return False
        return None


def _pst():
    return SimpleNamespace(
        alphabet_size=2,
        # split_pval is SearchConfig's default -- what production actually runs.
        config=SimpleNamespace(split_pval=0.001),
    )


def _evidence(family=None, members=(), state=0):
    """A SplitEvidence over a population holding ``members`` at leaf ``state``.

    The classifier is a stub: members are placed directly at the leaf, so no
    pull-down (and thus no classification) happens -- that path is exercised in
    test_leaf_population.
    """
    tree = MidfixTree(())
    population = LeafPopulation(tree, lambda strings, midfix: [None] * len(strings))
    for member in members:
        population.add(member, at=tree.path_of(state))
    return SplitEvidence(
        _pst(),
        family or _StubFamily(),
        population=population,
        tree=tree,
    )


class TestVerdict(unittest.TestCase):
    # pylint: disable=protected-access
    def test_a_clean_bifurcation_splits(self):
        # Half the members on each side, every one decisive: the two-rate model
        # beats the pooled one by a mile.
        family = _StubFamily(side_of=lambda p: p[-1] == 0)
        ev = _evidence(family, members=[[i, i % 2] for i in range(40)])
        self.assertEqual(SPLIT, ev.verdict(0, (1,)))

    def test_a_one_sided_population_settles_the_leaf(self):
        # Every member on the same side: scores stay 0 (no second rate), so the
        # one-state test decides it -- a zero minority over enough members rules
        # a split out.
        ev = _evidence(
            _StubFamily(side_of=lambda p: True), members=[[i] for i in range(200)]
        )
        a1, r1, a2, r2, n_a, n_b = ev._tally(0, (1,))
        self.assertEqual((200, 0), (n_a, n_b))
        self.assertEqual(0.0, ev._log_bf_scores(a1, r1, a2, r2))
        self.assertEqual(NO_SPLIT, ev.verdict(0, (1,)))

    def test_a_small_one_sided_population_is_not_yet_conclusive(self):
        # The same agreement, too few members to rule a split out: UNDECIDED.
        ev = _evidence(
            _StubFamily(side_of=lambda p: True), members=[[i] for i in range(5)]
        )
        self.assertEqual(UNDECIDED, ev.verdict(0, (1,)))


if __name__ == "__main__":
    unittest.main()
