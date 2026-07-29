"""Smoke tests for the CAPAL vs E-L* comparison harness.

One cheap cell through each learner driver, plus the exclusion path and the
emitted JSON. Query counts are asserted only as nonzero: these are guards
against a driver that silently stops measuring, not a pin on upstream's
search order.
"""

import json
import tempfile
import unittest
from pathlib import Path

from orthogonal_dfa.experiments.capal_comparison import (
    LEARNER_CAPAL,
    LEARNER_ELSTAR,
    SCHEMA_VERSION,
    Cell,
    run_capal_cell,
    run_elstar_cell,
    write_experiment,
)
from orthogonal_dfa.experiments.capal_comparison.capal_targets import capal_benchmarks
from orthogonal_dfa.experiments.capal_comparison.core import eval_words
from orthogonal_dfa.experiments.capal_comparison.sweep import run_cell
from tests.test_capal_bridge import _ensure_capal_checkout

#: Both are 2-state targets inside E-L*'s regime, so a cell on either is seconds.
CAPAL_TARGET = "Simple01"
ELSTAR_TARGET = "Simple02"
#: Outside E-L*'s regime, and the largest .taf target -- so it exercises the
#: exclusion branch without ever starting a learner.
EXCLUDED_TARGET = "Normal01"


class TestCapalComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_capal_checkout()
        cls.benchmarks = {b.name: b for b in capal_benchmarks()}

    def _cell_inputs(self, name):
        b = self.benchmarks[name]
        return b, eval_words(b.symbols), b.truth()

    def test_capal_cell(self):
        b, words, truth = self._cell_inputs(CAPAL_TARGET)
        cell = run_capal_cell(
            b.target,
            benchmark=b.name,
            family=b.family,
            eta=0.05,
            seed=0,
            words=words,
            truth=truth,
            alphabet=b.alphabet,
        )
        self.assertIsNone(cell.error)
        self.assertEqual(cell.learner, LEARNER_CAPAL)
        self.assertTrue(cell.converged)
        self.assertEqual(cell.learned_states, b.target_states)
        self.assertEqual(cell.accuracy, 1.0)
        self.assertGreater(cell.queries_total, 0)
        # A perfect EQ is the asymmetry the whole comparison turns on, so a
        # driver that stopped counting it has to fail here.
        self.assertGreaterEqual(cell.equivalence_queries, 1)
        # Upstream floors its noise estimate at 0.15, so it is not the eta it
        # was handed. See the core module docstring.
        self.assertEqual(cell.learner_config["eta_hat"], 0.15)

    def test_elstar_cell(self):
        b, words, truth = self._cell_inputs(ELSTAR_TARGET)
        cell = run_elstar_cell(
            b.oracle_creator,
            benchmark=b.name,
            family=b.family,
            eta=0.05,
            seed=0,
            symbols=b.symbols,
            words=words,
            truth=truth,
            target_states=b.target_states,
        )
        self.assertIsNone(cell.error)
        self.assertEqual(cell.learner, LEARNER_ELSTAR)
        self.assertEqual(cell.learned_states, b.target_states)
        self.assertEqual(cell.accuracy, 1.0)
        self.assertEqual(cell.equivalence_queries, 0)
        self.assertGreater(cell.queries_total, 0)
        self.assertEqual(cell.learner_config["min_signal_strength"], 0.45)

    def test_out_of_regime_target_is_excluded_not_run(self):
        b, words, truth = self._cell_inputs(EXCLUDED_TARGET)
        regime = b.regime_report()
        self.assertFalse(regime.satisfied)

        cell = run_cell(
            b,
            learner=LEARNER_ELSTAR,
            eta=0.05,
            seed=0,
            words=words,
            truth=truth,
            regime=regime,
        )
        self.assertEqual(cell.error_type, "ExcludedOutOfRegime")
        # The exclusion has to carry its reasons, or the excluded set is a bare
        # assertion rather than something a reader can check.
        self.assertTrue(cell.error)
        self.assertIsNone(cell.accuracy)
        self.assertIsNone(cell.queries_total)

    def test_capal_runs_where_elstar_is_excluded(self):
        # CAPAL's PerfectEQ finds counterexamples structurally, so E-L*'s
        # preconditions must not gate it.
        b, words, truth = self._cell_inputs(EXCLUDED_TARGET)
        cell = run_cell(
            b,
            learner=LEARNER_CAPAL,
            eta=0.05,
            seed=0,
            words=words,
            truth=truth,
            regime=b.regime_report(),
        )
        self.assertIsNone(cell.error_type)
        self.assertGreater(cell.queries_total, 0)

    def test_written_experiment_records_completeness(self):
        cell = Cell(benchmark="b", family="f", learner=LEARNER_CAPAL, eta=0.05, seed=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "exp.json"
            for complete in (False, True):
                write_experiment(
                    path,
                    experiment="exp",
                    generated_by="test",
                    description="d",
                    config={"etas": [0.05]},
                    cells=[cell],
                    complete=complete,
                )
                payload = json.loads(path.read_text())
                self.assertEqual(payload["complete"], complete)
            self.assertEqual(payload["schema_version"], SCHEMA_VERSION)
            self.assertEqual(payload["cells"][0]["benchmark"], "b")


if __name__ == "__main__":
    unittest.main()
