"""A fast(er) live reproduction of the shattering pathology.

``InteractionOracle`` is a non-regular target (dense linear score + prefix-boundary
interactions), so direct-L* cannot close it into a compact automaton: it shatters into
many states.  This is an *integration* test -- it runs a real synthesis -- so it is kept
as short as the methodology allows: a short length (16) and few probes.

Crucially ``fnr_limit`` stays at the proper 0.02.  Loosening it would leave prefixes
unresolved and manufacture artificial "shattering"; the whole point is that shattering
happens even when the suffix family properly resolves the prefixes.  The cost floor is
the family sizing (the per-sift family average, see the sift-perf note), which is why
this runs in minutes rather than seconds -- shorter lengths do not help, because below
~16 the interaction oracle's FNR floor rises above 0.02 and the family never sizes.
"""

import unittest

from orthogonal_dfa.analysis.ladder_repro import InteractionOracle
from orthogonal_dfa.l_star.counterexample_synthesis import synthesize_direct_lstar_fnr
from orthogonal_dfa.l_star.learn import build_pst

LENGTH = 16


class TestInteractionOracleShatters(unittest.TestCase):
    def test_non_regular_target_shatters_under_proper_fnr(self):
        oracle = InteractionOracle(
            length=LENGTH, n_pairs=2 * LENGTH, alpha=4.0, boundary_window=6
        )
        pst = build_pst(
            lambda _n, _s: oracle,
            min_signal_strength=0.06,
            seed=0,
            sample_length=LENGTH,
            fnr_limit=0.02,  # proper -- do NOT loosen (see module docstring)
        )
        dfa, _ = synthesize_direct_lstar_fnr(
            pst,
            acc_threshold=0.98,
            max_rounds=1,
            counterexample_probes=100,
            per_state=10,
            min_indecisive=30,
        )
        # A compact regular target closes into a handful of states; this shatters.
        self.assertGreaterEqual(len(dfa.states), 6)


if __name__ == "__main__":
    unittest.main()
