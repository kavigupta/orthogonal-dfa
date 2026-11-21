"""The suffix screen on synthetic rows: accept-preserving rows read each prefix
at its class's rate, as the reference does; flippers read every prefix at one
class's rate.  Off a half, raw disagreement favours the flipper whose rate sits
nearer it once a class's rate passes a half, which the centred statistic does
not."""

import unittest
from types import SimpleNamespace

import numpy as np
from parameterized import parameterized

from orthogonal_dfa.l_star.prefix_suffix_tracker import PrefixSuffixTracker

SIGNAL = 0.2
PREFIXES = 20000
PER_KIND = 10


class _Table:
    def __init__(self, masks):
        self._masks = masks
        self.num_prefixes = masks.shape[1]
        self.representative = np.ones(self.num_prefixes, dtype=bool)

    def seed_scoring(self):
        return np.arange(self.num_prefixes) % 2 == 1

    def observed_masks(self, rows, prefixes):
        return self._masks[np.asarray(rows)][:, prefixes]

    def column(self, row):
        return self._masks[row].copy()


def _screened(centre, centred_screen, seed=0):
    """The kinds of the rows the screen keeps: 0 accept-preserving, 1 reading
    everything accepting, 2 everything rejecting."""
    rng = np.random.default_rng(seed)
    accepting = rng.random(PREFIXES) < 0.5
    rates = np.where(accepting, centre + SIGNAL, centre - SIGNAL)
    kinds = [0] * (PER_KIND + 1) + [1] * PER_KIND + [2] * PER_KIND
    row_rates = np.array(
        [
            (
                rates
                if k == 0
                else np.full(PREFIXES, centre + (SIGNAL if k == 1 else -SIGNAL))
            )
            for k in kinds
        ]
    )
    masks = (rng.random(row_rates.shape) < row_rates).astype(np.int8)
    pst = PrefixSuffixTracker(
        sampler=None,
        rng=rng,
        oracle=None,
        config=SimpleNamespace(
            min_signal_strength=SIGNAL,
            screening_alpha=0.1,
            centred_screen=centred_screen,
            fnr_limit=0.1,
        ),
        table=_Table(masks),
        decision_boundary=centre,
    )
    kept = pst._screen_cohort(  # pylint: disable=protected-access
        list(range(1, len(kinds))), 0
    )
    return [kinds[row] for row in kept]


class TestCentredScreen(unittest.TestCase):
    @parameterized.expand([(0.35,), (0.5,), (0.65,), (0.75,)])
    def test_keeps_the_accept_preserving_rows_and_only_those(self, centre):
        kept = _screened(centre, True)
        self.assertEqual({0}, set(kept))
        # Each is dropped at most screening_alpha of the time.
        self.assertGreaterEqual(len(kept), PER_KIND - 3)

    def test_raw_disagreement_keeps_the_flippers_past_a_half(self):
        self.assertIn(1, _screened(0.75, False))


if __name__ == "__main__":
    unittest.main()
