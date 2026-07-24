"""Smoke tests for the upstream CAPAL bridge (orthogonal_dfa.capal_official).

Ensures a pinned CAPAL checkout exists at ``../capal`` -- cloning it in CI if
absent -- then exercises the bridge end to end: verify the pin, import the
upstream module, build target DFAs both ways, check a ported DFA against the
oracle it stands in for, and run a tiny CAPALLearner fit. Catches a broken
adapter or upstream-commit drift.

The clone is skipped (not failed) when no checkout exists and cloning is not
possible (e.g. offline), so the suite still runs without network; CI has git
and network, so it clones and runs for real.
"""

import itertools
import subprocess
import unittest
from pathlib import Path

from orthogonal_dfa.capal_official import (
    PINNED_COMMIT,
    build_modulo_dfa,
    build_regex_dfa,
    fit_with_fallback,
    import_capal,
    make_learner,
    resolve_capal_dir,
    verify_pinned,
)
from orthogonal_dfa.capal_official.adapter import UPSTREAM_URL


def _ensure_capal_checkout() -> Path:
    path = resolve_capal_dir()
    if not path.exists():
        try:
            subprocess.run(
                ["git", "clone", "--quiet", UPSTREAM_URL, str(path)],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "-C", str(path), "checkout", "--quiet", PINNED_COMMIT],
                check=True,
                capture_output=True,
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise unittest.SkipTest(f"no CAPAL checkout and clone failed: {exc}")
    return path


def _all_words(alphabet: str, max_len: int):
    return [
        "".join(w)
        for n in range(max_len + 1)
        for w in itertools.product(alphabet, repeat=n)
    ]


class TestCapalBridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = _ensure_capal_checkout()

    def test_pinned_checkout_verifies(self):
        verify_pinned(self.path)  # raises if wrong commit or dirty tree

    def test_import_capal(self):
        upstream = import_capal()
        self.assertTrue(hasattr(upstream, "CAPALLearner"))
        self.assertTrue(hasattr(upstream, "DFA"))

    def test_build_modulo_dfa(self):
        dfa = build_modulo_dfa(9, (3, 6))
        self.assertEqual(dfa.num_states, 9)
        self.assertTrue(dfa.run("111"))  # three 1s -> 3 mod 9 in {3,6}
        self.assertFalse(dfa.run("1"))  # one 1 -> 1 mod 9 not in {3,6}

    def test_build_regex_dfa(self):
        dfa = build_regex_dfa(r".*1010101.*")
        self.assertEqual(dfa.num_states, 8)  # every state is live, so no sink
        self.assertTrue(dfa.run("1010101"))
        self.assertFalse(dfa.run("0000000"))

    def test_build_regex_dfa_with_dead_end(self):
        # automata-lib hands back a partial DFA here; the sink has to be added
        # back or upstream's DFA rejects the transition table outright.
        dfa = build_regex_dfa(r"1*")
        self.assertEqual(dfa.num_states, 2)
        self.assertTrue(dfa.run(""))
        self.assertTrue(dfa.run("111"))
        self.assertFalse(dfa.run("0"))
        self.assertFalse(dfa.run("110"))  # dead: the sink never accepts again

        dfa = build_regex_dfa(r"0*1*")
        self.assertTrue(dfa.run("0011"))
        self.assertFalse(dfa.run("10"))

    def test_build_regex_dfa_three_symbols(self):
        dfa = build_regex_dfa(r"(0|1)*2*", alphabet_size=3)
        self.assertEqual(dfa.alphabet, ("0", "1", "2"))
        self.assertTrue(dfa.run("01022"))
        self.assertFalse(dfa.run("2201"))

    def test_ported_dfa_matches_the_oracle(self):
        # The head-to-head is only meaningful if the upstream DFA and the Oracle
        # it stands in for denote the same language under the same symbol order.
        from orthogonal_dfa.l_star.examples.bernoulli_parity import (
            BernoulliParityOracle,
        )
        from orthogonal_dfa.l_star.structures import SymmetricBernoulli

        dfa = build_modulo_dfa(9, (3, 6))
        oracle = BernoulliParityOracle(
            SymmetricBernoulli(p_correct=1.0), 0, modulo=9, allowed_moduluses=(3, 6)
        )
        for word in _all_words("01", 8):
            truth = oracle.membership_query([int(c) for c in word])
            self.assertEqual(dfa.run(word), truth, word)

    def test_end_to_end_fit(self):
        # Trivial noiseless target: CAPAL + PerfectEQ should learn it exactly.
        target = build_modulo_dfa(2, (1,))
        learned, converged = fit_with_fallback(
            make_learner(target, eta=0.0, seed=0, max_iters=50)
        )
        self.assertTrue(converged)
        for word in _all_words("01", 10):
            self.assertEqual(learned.run(word), target.run(word), word)

    def test_fit_falls_back_to_the_last_hypothesis(self):
        # One iteration cannot learn a 9-state target, so fit() hits its cap and
        # the hypothesis it held at that point comes back as non-convergent.
        target = build_modulo_dfa(9, (3, 6))
        learned, converged = fit_with_fallback(
            make_learner(target, eta=0.0, seed=0, max_iters=1)
        )
        self.assertFalse(converged)
        self.assertIsNotNone(learned)
        self.assertLess(learned.num_states, target.num_states)


if __name__ == "__main__":
    unittest.main()
