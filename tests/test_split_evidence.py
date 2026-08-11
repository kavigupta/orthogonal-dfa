import unittest
from types import SimpleNamespace

from orthogonal_dfa.l_star.split_evidence import (
    DEFAULT_SPLIT_MISS_RATE,
    NO_SPLIT,
    SPLIT,
    UNDECIDED,
    SplitEvidence,
)


class _StubFamily:
    """Classifies by caller-supplied rules, so no oracle is involved.

    The two halves are driven independently, because that is the whole point of
    the partition: ``side_of(prefix)`` groups a member on the ASSIGN half
    (``None`` = indecisive there, contributing no evidence), and
    ``accept_rate(prefix)`` sets the fraction of TEST bits that score it.  Tying
    them together can only ever express a clean split.
    """

    test_idx = list(range(1, 20, 2))
    assign_idx = list(range(0, 20, 2))

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
        for n, i in enumerate(self.assign_idx):
            votes[i] = 0 if side is None else (1 if side else 0)
        if side is None:  # straddle the ASSIGN thresholds
            for i in self.assign_idx[: len(self.assign_idx) // 2]:
                votes[i] = 1
        for n, i in enumerate(self.test_idx):
            votes[i] = 1 if n < round(rate * len(self.test_idx)) else 0
        return votes

    def assign_side(self, votes):
        mean = sum(votes[i] for i in self.assign_idx) / len(self.assign_idx)
        if mean >= 0.9:
            return True
        if mean < 0.1:
            return False
        return None


def _pst():
    return SimpleNamespace(
        alphabet_size=2,
        evidence_margin=0.0,
        # split_pval is SearchConfig's default -- what production actually runs.
        config=SimpleNamespace(
            split_pval=0.001, min_signal_strength=0.3, suffix_family_size=20
        ),
    )


def _evidence(
    family=None,
    pool_members=lambda state, limit: [],
    pool_representative=lambda state: None,
    num_states=2,
    **kw,
):
    """Production values unless a test overrides them."""
    return SplitEvidence(
        _pst(),
        family or _StubFamily(),
        pool_members=pool_members,
        pool_representative=pool_representative,
        num_states=lambda: num_states,
        **{
            "split_fpr": None,
            "split_miss_rate": DEFAULT_SPLIT_MISS_RATE,
            "members": {},
            **kw,
        },
    )


class TestAfterSplit(unittest.TestCase):
    """after_split's contract: the split leaf is rebuilt, everything else carries.

    Carrying is easy to lose -- returning a plain fresh SplitEvidence reads
    cleaner and passes everything else -- but a newly created, still conflated
    leaf that starts empty can never gather the members its own split needs, so
    the failure surfaces rounds later as non-convergence.
    """

    # Whether a candidate is still being probed is internal; nothing public asks.
    # pylint: disable=protected-access

    def test_untouched_leaves_keep_their_members(self):
        ev = _evidence()
        ev.record(0, [0])
        ev.record(1, [1, 0])
        ev.record(1, [1, 1])
        refined = ev.after_split(0, sift=lambda m: 0)
        self.assertEqual({(1, 0), (1, 1)}, refined.members[1])

    def test_the_split_leafs_members_are_redistributed(self):
        ev = _evidence()
        for member in ([0], [1], [0, 1]):
            ev.record(0, member)
        # [0] stays on the True half; everything else lands on the new leaf.
        refined = ev.after_split(0, sift=lambda m: 0 if m == [0] else 2)
        self.assertEqual({(0,)}, refined.members[0])
        self.assertEqual({(1,), (0, 1)}, refined.members[2])

    def test_members_the_refined_tree_cannot_place_are_dropped(self):
        ev = _evidence()
        ev.record(0, [0])
        ev.record(0, [1])
        refined = ev.after_split(0, sift=lambda m: None if m == [1] else 0)
        self.assertEqual({(0,)}, refined.members[0])
        self.assertNotIn((1,), {m for ms in refined.members.values() for m in ms})

    def test_the_original_is_left_alone(self):
        ev = _evidence()
        ev.record(0, [0])
        refined = ev.after_split(0, sift=lambda m: 2)
        refined.record(2, [9])
        self.assertEqual({(0,)}, ev.members[0])
        self.assertNotIn(2, ev.members)


class TestVerdict(unittest.TestCase):
    # pylint: disable=protected-access
    def test_a_clean_bifurcation_splits(self):
        # Half the members on each side, every one decisive: the two-rate model
        # beats the pooled one by a mile.
        family = _StubFamily(side_of=lambda p: p[-1] == 0)
        ev = _evidence(family)
        for i in range(40):
            ev.record(0, [i, i % 2])
        self.assertEqual(SPLIT, ev.verdict(0, (1,)))

    def test_a_one_sided_population_settles_the_leaf(self):
        # Every member on the same side: the distinguisher does not divide this
        # leaf.  Scores stay 0 (no second rate to compare), so the one-state test
        # decides it -- a zero minority over enough members rules a split out.
        ev = _evidence(_StubFamily(side_of=lambda p: True))
        for i in range(200):
            ev.record(0, [i])
        a1, r1, a2, r2, n_a, n_b = ev._tally(0, (1,))
        self.assertEqual((200, 0), (n_a, n_b))
        self.assertEqual(0.0, ev._log_bf_scores(a1, r1, a2, r2))
        self.assertEqual(NO_SPLIT, ev.verdict(0, (1,)))

    def test_a_small_one_sided_population_is_not_yet_conclusive(self):
        # The same agreement, too few members to rule a split out: UNDECIDED.
        ev = _evidence(_StubFamily(side_of=lambda p: True))
        for i in range(5):
            ev.record(0, [i])
        self.assertEqual(UNDECIDED, ev.verdict(0, (1,)))

    def test_pool_members_are_scanned_once_per_leaf(self):
        # The scan walks the whole prefix pool, so it is cached per leaf and must
        # not run again on later verdicts for the same leaf.
        scans = []

        def pool_members(state, limit):
            del limit
            scans.append(state)
            return []

        family = _StubFamily(side_of=lambda p: p[-1] == 0, accept_rate=lambda p: 0.5)
        ev = _evidence(family, pool_members=pool_members)
        for i in range(40):
            ev.record(0, [i, i % 2])
        ev.verdict(0, (1,))
        ev.verdict(0, (1,))
        ev.verdict(0, (1,))
        self.assertEqual([0], scans)


if __name__ == "__main__":
    unittest.main()
