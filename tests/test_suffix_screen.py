"""The suffix screen on the armed target: it keeps class-preserving suffixes at the
rate it promises, and suffixes that move the rejecting state to accept less often."""

import unittest

import numpy as np
import scipy.stats
from automata.fa.dfa import DFA
from parameterized import parameterized

from orthogonal_dfa.l_star.examples.benchmark_generator import DFAOracle
from orthogonal_dfa.l_star.examples.bernoulli_parity import BernoulliParityOracle
from orthogonal_dfa.l_star.learn import DEFAULT_SAMPLE_LENGTH, build_pst
from orthogonal_dfa.l_star.sampler import UniformSampler
from orthogonal_dfa.l_star.structures import AsymmetricBernoulli, NoisyOracle

ALPHABET = 12
LENGTH = 20
#: Q holds on the symbols below this and leaves for the accepting sink on the rest.
HOLDS = 10
ARM = ALPHABET - 1
PER_KIND = 100
LEVEL = 1e-3


def _target() -> DFA:
    return DFA(
        states={"c0", "Q", "A"},
        input_symbols=set(range(ALPHABET)),
        transitions={
            "c0": {c: ("Q" if c == ARM else "c0") for c in range(ALPHABET)},
            "Q": {c: ("Q" if c < HOLDS else "A") for c in range(ALPHABET)},
            "A": {c: "A" for c in range(ALPHABET)},
        },
        initial_state="c0",
        final_states={"A"},
        allow_partial=False,
    )


def _suffixes(rng, kind):
    """PER_KIND distinct suffixes: hold preserves every class, escape
    preserves all but Q's (it never arms, and leaves Q), arm arms."""
    out = set()
    while len(out) < PER_KIND:
        if kind == "hold":
            v = rng.integers(0, HOLDS, LENGTH)
        elif kind == "escape":
            v = rng.integers(0, ARM, LENGTH)
            v[rng.integers(LENGTH)] = HOLDS
        else:
            v = rng.integers(0, ALPHABET, LENGTH)
            v[rng.integers(LENGTH)] = ARM
        out.add(bytes(v.tolist()))
    return sorted(out)


class TestScreenKeepsTheClassPreserving(unittest.TestCase):
    @parameterized.expand([(0.35, 0.95), (0.2, 0.8)])
    def test_screened_by_kind(self, p_0, p_1):
        pst = build_pst(
            lambda nm, s: NoisyOracle(DFAOracle(_target()), nm, s),
            min_signal_strength=(p_1 - p_0) / 2,
            seed=0,
            sampler=UniformSampler(LENGTH),
            noise_model=AsymmetricBernoulli(p_0=p_0, p_1=p_1),
        )
        # The true boundary, so only the screen is under test.
        pst.decision_boundary = (p_0 + p_1) / 2
        pst.calibrated = True
        reference = pst.table.intern_suffix(b"")
        rng = np.random.default_rng(0)
        kept = {}
        for kind in ("hold", "escape", "arm"):
            rows = [pst.table.intern_suffix(v) for v in _suffixes(rng, kind)]
            kept[kind] = len(
                pst._screen_cohort(  # pylint: disable=protected-access
                    rows, reference, predict=True
                )
            )
        # Dropping a class-preserving suffix is what the screen bounds.
        self.assertGreater(
            scipy.stats.binom.sf(
                PER_KIND - kept["hold"] - 1, PER_KIND, pst.config.screening_alpha
            ),
            LEVEL,
            f"kept {kept['hold']} of {PER_KIND} class-preserving suffixes",
        )
        for impostor in ("escape", "arm"):
            _, p = scipy.stats.fisher_exact(
                [
                    [kept["hold"], PER_KIND - kept["hold"]],
                    [kept[impostor], PER_KIND - kept[impostor]],
                ],
                alternative="greater",
            )
            self.assertLess(p, LEVEL, f"kept by kind: {kept}")


class TestScreenDropsRotations(unittest.TestCase):
    """Modulo 9 accepting {3, 6}: a suffix with 3 or 6 ones mod 9 rotates 0 -> 3 -> 6,
    and a family of them merges the three.  The declared signal is 0.25 where the
    band holds 0.35."""

    @parameterized.expand([(False,), (True,)])
    def test_rotations_are_dropped_more_often(self, calibrated):
        pst = build_pst(
            lambda nm, s: NoisyOracle(
                BernoulliParityOracle(modulo=9, allowed_moduluses=(3, 6)), nm, s
            ),
            min_signal_strength=0.25,
            seed=0,
            sampler=UniformSampler(DEFAULT_SAMPLE_LENGTH),
            noise_model=AsymmetricBernoulli(p_0=0.25, p_1=0.95),
        )
        pst.decision_boundary = 0.6
        pst.calibrated = calibrated
        reference = pst.table.intern_suffix(b"")
        # One cohort of the sampler's own draws, as the screen sees them.
        rows = pst._draw_cohort(6 * PER_KIND)  # pylint: disable=protected-access
        kept_rows = set(
            pst._screen_cohort(  # pylint: disable=protected-access
                rows, reference, predict=calibrated
            )
        )
        table = []
        for shifts in ((0,), (3, 6)):
            of_kind = [r for r in rows if sum(pst.table.suffix(r)) % 9 in shifts]
            kept = sum(r in kept_rows for r in of_kind)
            table.append([kept, len(of_kind) - kept])
        _, p = scipy.stats.fisher_exact(table, alternative="greater")
        self.assertLess(
            p, LEVEL, f"kept, dropped: preserving {table[0]}, rotating {table[1]}"
        )


def _overstated():
    # 0.45 against a band holding 0.3 predicts too little noise, as a boundary
    # estimated far off does.
    pst = build_pst(
        lambda nm, s: NoisyOracle(DFAOracle(_target()), nm, s),
        min_signal_strength=0.45,
        seed=0,
        sampler=UniformSampler(LENGTH),
        noise_model=AsymmetricBernoulli(p_0=0.2, p_1=0.8),
    )
    reference = pst.table.intern_suffix(b"")
    pst.table.column(reference)
    return pst, reference


class TestScreenSurvivesAWrongPrediction(unittest.TestCase):
    def test_dropped_rows_are_screened_again(self):
        pst, reference = _overstated()
        pst.calibrated = True
        wanted = 20
        first = len(pst.table._suffixes)  # pylint: disable=protected-access
        kept, _ = pst.sample_more_suffixes(amount=wanted, reference=reference)
        self.assertGreaterEqual(kept, wanted)
        pool = set(pst.suffix_pool)
        # The prediction keeps none of the first cohort.
        self.assertTrue(any(row in pool for row in range(first, first + wanted)))

    def test_calibrating_on_it_removes_nothing(self):
        pst, reference = _overstated()
        for draws in (0, 150):
            while pst.suffixes_drawn <= draws:
                pst.sample_more_suffixes(amount=20, reference=reference)
            pool = list(pst.suffix_pool)
            self.assertEqual(pst.calibrate(reference), 0)
            self.assertFalse(pst.calibrated)
            self.assertEqual(pst.suffix_pool, pool)
