import unittest
from types import SimpleNamespace

from orthogonal_dfa.l_star.leaf_population import LeafPopulation
from orthogonal_dfa.l_star.midfix_tree import MidfixTree
from orthogonal_dfa.l_star.split_evidence import (
    MEMBERS_TO_RULE_OUT_A_SPLIT,
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


def _evidence(family=None, members=(), state=0, tree_splits=(), by_state=None):
    """A SplitEvidence over a population holding ``members`` at leaf ``state``,
    or ``by_state``'s members at each leaf it names.

    The classifier is a stub: members are placed directly at the leaf, so no
    pull-down (and thus no classification) happens -- that path is exercised in
    test_leaf_population.
    """
    tree = MidfixTree(())
    for at, midfix in tree_splits:
        tree.split(at, midfix)
    population = LeafPopulation(
        tree,
        lambda strings, midfix: [None] * len(strings),
        harvest=lambda _s: None,
    )
    for leaf, held in (by_state or {state: members}).items():
        for member in held:
            population.add(member, at=tree.path_of(leaf))
    return SplitEvidence(
        _pst(),
        family or _StubFamily(),
        population=population,
        tree=tree,
    )


def _agreeing(tag, count=MEMBERS_TO_RULE_OUT_A_SPLIT):
    """Members every distinguisher puts on one side; the first byte tags which
    leaf they are placed at."""
    return [bytes([tag, 0, i]) for i in range(count)]


def _bifurcating(tag, count=MEMBERS_TO_RULE_OUT_A_SPLIT):
    """As `_agreeing`, but the second byte alternates, so a family reading it
    puts half the members on each side."""
    return [bytes([tag, i % 2, i]) for i in range(count)]


#: Separates the members tagged 1, and only under symbol 1 over the root midfix.
_SPLITS_LEAF_1 = lambda p, d: p[0] != 1 or d != bytes([1]) or p[1] == 0

#: Every leaf, so a thin one is held open rather than written off.
_ALL = frozenset(range(16))


class TestWhetherAnythingIsLeftToSplit(unittest.TestCase):
    def test_a_tree_every_distinguisher_agrees_on_has_nothing(self):
        ev = _evidence(
            _StubFamily(side_of=lambda p, d: True),
            by_state={0: _agreeing(0), 1: _agreeing(1)},
        )

        self.assertTrue(ev.nothing_left_to_split(_ALL))

    def test_a_leaf_a_distinguisher_still_separates_has_something(self):
        # Leaf 0 is final here, so a sweep that stopped at the first leaf would
        # miss leaf 1 and call the tree done.
        ev = _evidence(
            _StubFamily(side_of=_SPLITS_LEAF_1),
            by_state={0: _agreeing(0), 1: _bifurcating(1)},
        )

        self.assertFalse(ev.nothing_left_to_split(_ALL))

    def test_a_leaf_too_thin_to_say_is_waited_on_where_it_can_be_filled(self):
        ev = _evidence(
            _StubFamily(),
            by_state={
                0: _agreeing(0),
                1: _agreeing(1, MEMBERS_TO_RULE_OUT_A_SPLIT - 1),
            },
        )

        self.assertFalse(ev.nothing_left_to_split(_ALL))

    def test_and_written_off_where_it_cannot(self):
        # No draw reaches leaf 1, so its members are as many as it will ever
        # hold and waiting for more never ends.
        ev = _evidence(
            _StubFamily(),
            by_state={
                0: _agreeing(0),
                1: _agreeing(1, MEMBERS_TO_RULE_OUT_A_SPLIT - 1),
            },
        )

        self.assertTrue(ev.nothing_left_to_split(frozenset({0})))

    def test_a_split_counts_even_where_the_leaf_cannot_be_filled(self):
        ev = _evidence(
            _StubFamily(side_of=_SPLITS_LEAF_1),
            by_state={0: _agreeing(0), 1: _bifurcating(1)},
        )

        self.assertFalse(ev.nothing_left_to_split(frozenset({0})))

    def test_a_leaf_nothing_reaches_is_waited_on(self):
        ev = _evidence(_StubFamily(), by_state={0: _agreeing(0)})

        self.assertFalse(ev.nothing_left_to_split(_ALL))

    def test_a_separation_only_a_deeper_midfix_reaches(self):
        # Symbol 1 over the midfix a split puts in the tree; until that split
        # exists no candidate is built from it.  The split also makes leaf 2,
        # which is stocked so that an empty leaf is not what fails.
        family = _StubFamily(
            side_of=lambda p, d: p[0] != 1 or d != bytes([1, 5]) or p[1] == 0
        )
        held = {0: _agreeing(0), 1: _bifurcating(1)}

        self.assertTrue(_evidence(family, by_state=held).nothing_left_to_split(_ALL))
        self.assertFalse(
            _evidence(
                family,
                by_state={**held, 2: _agreeing(2)},
                tree_splits=((1, bytes([5])),),
            ).nothing_left_to_split(_ALL)
        )


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
