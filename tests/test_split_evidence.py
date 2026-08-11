import unittest
from types import SimpleNamespace

from orthogonal_dfa.l_star.split_evidence import (
    DEFAULT_SPLIT_MISS_RATE,
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

    def test_the_split_leafs_candidates_are_dropped(self):
        # Its population bifurcated under them, so the running sums are stale.
        ev = _evidence()
        for i in range(20):
            ev.record(0, [0, i])
        ev.verdict(0, (1,))  # opens a candidate on leaf 0
        self.assertTrue(ev._open.get(0))
        refined = ev.after_split(0, sift=lambda m: 0)
        self.assertNotIn(0, refined._open)

    def test_untouched_leaves_keep_their_candidates(self):
        """A split replaces one leaf with an internal node, so no other leaf's
        path changed and its accumulator is still exactly what it was."""
        ev = _evidence(num_states=3)
        for i in range(20):
            ev.record(1, [1, i])
        ev.verdict(1, (1,))  # opens a candidate on leaf 1
        before = dict(ev._open[1])
        refined = ev.after_split(0, sift=lambda m: 0)
        self.assertEqual(before, refined._open.get(1))

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

    def test_a_one_sided_population_stays_undecided(self):
        # Every member on the same side: the distinguisher does not divide this
        # leaf.  An empty group makes the two-rate model identical to the
        # one-rate model, so scores are exactly 0 -- with only the scores term a
        # one-sided leaf never reaches the no-split boundary and stays open.
        ev = _evidence(_StubFamily(side_of=lambda p: True))
        for i in range(200):
            ev.record(0, [i])
        accum = ev._candidate(0, (1,))
        self.assertEqual(0.0, ev._log_bf_scores(accum))
        self.assertEqual(UNDECIDED, ev.verdict(0, (1,)))

    def test_pool_members_are_only_scanned_when_a_candidate_opens(self):
        # The scan walks the whole prefix pool, so it must not run per verdict.
        scans = []

        def pool_members(state, limit):
            del limit
            scans.append(state)
            return []

        ev = _evidence(_StubFamily(side_of=lambda p: True), pool_members=pool_members)
        for i in range(40):
            ev.record(0, [i])
        ev.verdict(0, (1,))
        ev.verdict(0, (1,))
        ev.verdict(0, (1,))
        self.assertEqual([0], scans)


if __name__ == "__main__":
    unittest.main()
