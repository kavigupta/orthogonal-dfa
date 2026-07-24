"""Smoke tests for the upstream CAPAL bridge (orthogonal_dfa.capal_official).

Ensures a pinned CAPAL checkout exists at ``../capal`` -- cloning it in CI if
absent -- then exercises the bridge end to end: verify the pin, import the
upstream module, build target DFAs both ways, and run a tiny CAPALLearner fit
and score it. Catches a broken adapter or upstream-commit drift.

The clone is skipped (not failed) when no checkout exists and cloning is not
possible (e.g. offline), so the suite still runs without network; CI has git
and network, so it clones and runs for real.
"""

import subprocess
import unittest
from pathlib import Path

from orthogonal_dfa.capal_official import (
    PINNED_COMMIT,
    build_modulo_dfa,
    build_regex_dfa,
    evaluate_official_dfa,
    import_capal,
    resolve_capal_dir,
    run_official_capal,
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

    def test_end_to_end_fit_and_score(self):
        from orthogonal_dfa.l_star.examples.bernoulli_parity import (
            BernoulliParityOracle,
        )

        # Trivial noiseless target: CAPAL + PerfectEQ should learn it exactly.
        target = build_modulo_dfa(2, (1,))
        learned = run_official_capal(target, eta=0.0, seed=0, max_iters=50)
        acc = evaluate_official_dfa(
            learned,
            lambda nm, s: BernoulliParityOracle(
                nm, s, modulo=2, allowed_moduluses=(1,)
            ),
            alphabet_chars=["0", "1"],
            symbols=2,
            count=500,
        )
        self.assertGreater(acc, 0.99)


if __name__ == "__main__":
    unittest.main()
