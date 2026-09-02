"""The family search ends even where the FNR limit is never reached.

The loop waits on one thing -- the rate falling under the limit -- and nothing
guarantees a family gets there.  These pin what it does when none does.
"""

import unittest
from unittest import mock

from orthogonal_dfa.l_star import cluster as C

_FAMILY = [0, 1, 2]


class _Tracker:
    """Enough of a tracker for the loop to turn over."""

    def __init__(self):
        self.decision_boundary = 0.5
        self.evidence_margin = 0.1
        self.config = mock.Mock(min_signal_strength=0.3, fnr_limit=0.10)
        self.table = mock.Mock()
        self.bought = 0

    def sample_more_suffixes(self, **_):
        self.bought += 1
        return 1

    def sample_more_prefixes(self):
        self.bought += 1


class _Judge:
    """Reads one scripted FNR per round; settling re-reads the round in hand."""

    def __init__(self, fnrs):
        self.remaining = list(fnrs)
        self.last = 1.0
        self.rounds = 0

    def __call__(self, *_args, settling=False, **_kwargs):
        if not settling:
            self.rounds += 1
            self.last = self.remaining.pop(0)
        return C.Judged(
            vs=_FAMILY, fnr=self.last, reason=f"FNR {self.last}", verdict=C.ADMITTED
        )


def _run(fnrs):
    pst, judge = _Tracker(), _Judge(fnrs)
    with mock.patch.object(C, "judge_family", judge), mock.patch.object(
        C, "identify_cluster_around", return_value=(_FAMILY, 0.5)
    ), mock.patch.object(C, "smallest_readable_family", return_value=len(_FAMILY)):
        vs, _ = C.sample_suffix_family(pst, 0)
    return vs, pst, judge


class TestFnrGiveUp(unittest.TestCase):
    def test_it_returns_once_the_rate_stops_bettering_itself(self):
        # Never under the limit and never better than the first, so there is
        # nothing left for the loop to wait on.
        vs, pst, judge = _run([0.5] * (C.FNR_GIVE_UP + 5))
        self.assertEqual(vs, _FAMILY, "it settles for the family it has")
        self.assertLessEqual(judge.rounds, C.FNR_GIVE_UP + 2)
        self.assertLessEqual(pst.bought, C.FNR_GIVE_UP + 2)

    def test_a_rate_that_keeps_falling_is_not_a_stall(self):
        # Improving every round, then reaching the limit well past the give-up
        # count: it must have kept going rather than settled at 20.
        falling = [0.5 - i * 0.005 for i in range(C.FNR_GIVE_UP + 10)]
        falling.append(0.01)
        vs, _, judge = _run(falling)
        self.assertEqual(vs, _FAMILY)
        self.assertGreater(judge.rounds, C.FNR_GIVE_UP + 1)

    def test_it_will_not_settle_for_a_family_that_cannot_be_read(self):
        # 1.0 is the sentinel for unusable, so the loop keeps buying past the
        # give-up count and stops only when a readable family arrives.
        vs, _, judge = _run([1.0] * (C.FNR_GIVE_UP + 6) + [0.05])
        self.assertEqual(vs, _FAMILY)
        self.assertEqual(judge.rounds, C.FNR_GIVE_UP + 7)


if __name__ == "__main__":
    unittest.main()
