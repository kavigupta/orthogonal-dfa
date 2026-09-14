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

    def __init__(self, side_of=lambda p, d: True, accept_rate=None):
        self.side_of = side_of
        self.accept_rate = accept_rate
        self.prefilled = []

    def prefill(self, bases):
        self.prefilled.extend(bases)

    def votes(self, prefix, distinguisher):
        side = self.side_of(list(prefix), distinguisher)
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


def _evidence(family=None, members=(), state=0, splits=()):
    """A SplitEvidence over a population holding ``members`` at leaf ``state``.

    The classifier is a stub: members are placed directly at the leaf, so no
    pull-down (and thus no classification) happens -- that path is exercised in
    test_leaf_population.
    """
    tree = MidfixTree(())
    for at, midfix in splits:
        tree.split(at, midfix)
    population = LeafPopulation(
        tree,
        lambda strings, midfix: [None] * len(strings),
        harvest=lambda _s: None,
    )
    for member in members:
        population.add(member, at=tree.path_of(state))
    return SplitEvidence(
        _pst(),
        family or _StubFamily(),
        population=population,
        tree=tree,
    )


class TestMidfixesAreWhatCanBeProposed(unittest.TestCase):
    def test_each_node_contributes_its_own(self):
        tree = MidfixTree(())
        tree.split(0, bytes([5]))

        self.assertEqual([b"", bytes([5])], tree.midfixes())

    def test_a_grandchild_contributes_too(self):
        tree = MidfixTree(())
        new = tree.split(0, bytes([5]))
        tree.split(new, bytes([6]))

        self.assertEqual([b"", bytes([5]), bytes([6])], tree.midfixes())

    def test_one_reused_on_another_leaf_is_listed_once(self):
        tree = MidfixTree(())
        tree.split(0, bytes([5]))
        tree.split(1, bytes([5]))

        self.assertEqual([b"", bytes([5])], tree.midfixes())


#: Clear of the ~38 one-sided members `_agrees_as_one_state` needs, so these
#: turn on what is asked rather than on how much was held.
_PLENTY = 200


class TestWhetherAnyStateCanStillSplit(unittest.TestCase):
    def test_a_leaf_every_distinguisher_agrees_on_cannot(self):
        ev = _evidence(
            _StubFamily(side_of=lambda p, d: True),
            members=[bytes([i]) for i in range(_PLENTY)],
        )

        self.assertTrue(ev.no_state_can_split())

    def test_one_that_separates_under_the_root_midfix_can(self):
        family = _StubFamily(side_of=lambda p, d: d != bytes([1]) or p[-1] == 0)
        ev = _evidence(family, members=[bytes([i, i % 2]) for i in range(_PLENTY)])

        self.assertFalse(ev.no_state_can_split())

    def test_one_that_separates_under_a_deeper_midfix_can(self):
        # b"\x01\x05" is symbol 1 over the midfix a split put in the tree;
        # nothing proposes it until that split exists.
        family = _StubFamily(side_of=lambda p, d: d != bytes([1, 5]) or p[-1] == 0)
        members = [bytes([i, i % 2]) for i in range(_PLENTY)]

        self.assertTrue(_evidence(family, members=members).no_state_can_split())
        self.assertFalse(
            _evidence(
                family, members=members, splits=((0, bytes([5])),)
            ).no_state_can_split()
        )

    def test_a_separation_no_symbol_reaches_is_never_asked(self):
        # Symbol 9 is outside the alphabet, so no candidate is built from it.
        family = _StubFamily(side_of=lambda p, d: d != bytes([9]) or p[-1] == 0)
        ev = _evidence(family, members=[bytes([i, i % 2]) for i in range(_PLENTY)])

        self.assertTrue(ev.no_state_can_split())

    def test_a_thin_leaf_is_not_waited_on(self):
        # Too few members for `_agrees_as_one_state`, and no split available:
        # undecided forever is not a reason to keep going.
        ev = _evidence(_StubFamily(), members=[bytes([i]) for i in range(5)])

        self.assertEqual(UNDECIDED, ev.verdict(0, bytes([1])))
        self.assertTrue(ev.no_state_can_split())


class TestVerdict(unittest.TestCase):
    # pylint: disable=protected-access
    def test_a_clean_bifurcation_splits(self):
        # Half the members on each side, every one decisive: the two-rate model
        # beats the pooled one by a mile.
        family = _StubFamily(side_of=lambda p, d: p[-1] == 0)
        ev = _evidence(family, members=[bytes([i, i % 2]) for i in range(40)])
        self.assertEqual(SPLIT, ev.verdict(0, bytes([1])))

    def test_a_one_sided_population_settles_the_leaf(self):
        # Every member on the same side: scores stay 0 (no second rate), so the
        # one-state test decides it -- a zero minority over enough members rules
        # a split out.
        ev = _evidence(
            _StubFamily(side_of=lambda p, d: True),
            members=[bytes([i]) for i in range(200)],
        )
        a1, r1, a2, r2, n_a, n_b = ev._tally(ev._members(0), bytes([1]))
        self.assertEqual((200, 0), (n_a, n_b))
        self.assertEqual(0.0, ev._log_bf_scores(a1, r1, a2, r2))
        self.assertEqual(NO_SPLIT, ev.verdict(0, bytes([1])))

    def test_a_small_one_sided_population_is_not_yet_conclusive(self):
        # The same agreement, too few members to rule a split out: UNDECIDED.
        ev = _evidence(
            _StubFamily(side_of=lambda p, d: True),
            members=[bytes([i]) for i in range(5)],
        )
        self.assertEqual(UNDECIDED, ev.verdict(0, bytes([1])))


if __name__ == "__main__":
    unittest.main()
