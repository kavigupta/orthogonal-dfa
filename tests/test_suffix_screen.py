"""The suffix screen on the armed target: it keeps class-preserving suffixes at the
rate it promises, and suffixes that move the rejecting state to accept less often."""

import unittest

import numpy as np
import scipy.stats
from automata.fa.dfa import DFA
from parameterized import parameterized

from orthogonal_dfa.l_star.examples.benchmark_generator import DFAOracle
from orthogonal_dfa.l_star.learn import build_pst
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
        reference = pst.table.intern_suffix(b"")
        rng = np.random.default_rng(0)
        rows = {
            kind: [pst.table.intern_suffix(v) for v in _suffixes(rng, kind)]
            for kind in ("hold", "escape", "arm")
        }
        kept_rows = set(
            pst._screen_cohort(  # pylint: disable=protected-access
                [r for of_kind in rows.values() for r in of_kind], reference
            )
        )
        kept = {
            kind: sum(r in kept_rows for r in of_kind) for kind, of_kind in rows.items()
        }
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
