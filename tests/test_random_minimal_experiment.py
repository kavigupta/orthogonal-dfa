import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from orthogonal_dfa.experiments.random_minimal import analyze, difficulty, run
from orthogonal_dfa.l_star import preconditions as P
from orthogonal_dfa.l_star.examples.random_dfa import accessible_tables, dfa_from_table


def _result(states, eta, accuracy, *, learned_states, queries=100, error_type=None):
    return {
        "schema_version": run.RESULT_SCHEMA_VERSION,
        "benchmark": f"n{states:02d}_000",
        "target_states": states,
        "eta": eta,
        "seed": 0,
        "accuracy": accuracy,
        "learned_states": learned_states,
        "queries_total": queries,
        "seconds": 1.0,
        "error_type": error_type,
    }


class TestTargets(unittest.TestCase):
    def test_drawn_once_then_read_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            first = run.load_target(tmp, 4, 0)
            path = run.target_path(tmp, 4, 0)
            written = path.read_text()
            again = run.load_target(tmp, 4, 0)
            self.assertEqual(path.read_text(), written)
            self.assertEqual(first, again)
            self.assertEqual(len(first.minify().states), 4)

    def test_seeded_by_size_and_index(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            self.assertEqual(
                run.load_target(Path(a), 4, 1), run.load_target(Path(b), 4, 1)
            )


class TestDifficulty(unittest.TestCase):
    LENGTH = 8

    def setUp(self):
        rng = np.random.default_rng(0)
        self.tables = accessible_tables(0, 5, 2, 30)
        self.masks = [int(m) for m in rng.integers(1, 31, size=30)]

    def _ends(self, table):
        strings = np.array(list(itertools.product((0, 1), repeat=self.LENGTH)))
        ends = np.repeat(np.arange(len(table))[:, None], len(strings), axis=1)
        for j in range(self.LENGTH):
            ends = table[ends, strings[:, j]]
        return ends

    def test_endpoint_distribution_and_synchronizing_by_enumeration(self):
        for table in self.tables:
            ends = self._ends(table)
            self.assertTrue(
                np.allclose(
                    difficulty.endpoint_distribution(table, 0, self.LENGTH),
                    np.bincount(ends[0], minlength=len(table)) / ends.shape[1],
                )
            )
            self.assertAlmostEqual(
                difficulty.synchronizing_probability(table, self.LENGTH),
                (ends == ends[:1]).all(axis=0).mean(),
            )

    def test_separating_lengths_by_enumeration(self):
        for table, mask in zip(self.tables, self.masks):
            final = (mask >> np.arange(len(table))) & 1
            lengths = difficulty.separating_lengths(table, final.astype(bool))
            for p, q in itertools.combinations(range(len(table)), 2):
                shortest = next(
                    (
                        k
                        for k in range(len(table))
                        if any(
                            final[_run(table, p, w)] != final[_run(table, q, w)]
                            for w in itertools.product((0, 1), repeat=k)
                        )
                    ),
                    -1,
                )
                self.assertEqual(lengths[p, q], shortest)

    def test_target_metrics_agree_with_preconditions(self):
        aut = dfa_from_table(self.tables[0], self.masks[0])
        metrics = difficulty.target_metrics(aut, length=40)
        self.assertEqual(metrics["states"], 5)
        self.assertEqual(metrics["acceptance_rate"], P.acceptance_rate(aut, length=40))
        self.assertTrue(json.dumps(metrics))


def _run(table, state, word):
    for c in word:
        state = table[state, c]
    return state


class TestAnalyze(unittest.TestCase):
    RESULTS = [
        _result(4, 0.05, 1.0, learned_states=4),
        _result(4, 0.05, 0.995, learned_states=4),
        _result(4, 0.05, 0.9, learned_states=3),
        _result(4, 0.05, None, learned_states=None, error_type="Timeout"),
        _result(8, 0.05, 1.0, learned_states=8),
    ]

    def test_summary(self):
        rows = {(r["states"], r["eta"]): r for r in analyze.summarize(self.RESULTS)}
        four = rows[4, 0.05]
        self.assertEqual(four["runs"], 4)
        self.assertEqual(four["learned"], 0.5)
        self.assertEqual(four["exact"], 0.25)
        self.assertEqual(four["right_size"], 0.5)
        self.assertEqual(four["accuracy_min"], 0.9)
        self.assertEqual(four["errors"], 1)
        self.assertEqual(rows[8, 0.05]["learned"], 1.0)

    def test_failures_worst_first(self):
        failed = analyze.failures(self.RESULTS)
        self.assertEqual([f["accuracy"] for f in failed], [None, 0.9])

    def test_writes_reports_and_skips_stale_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "results").mkdir()
            for i, r in enumerate(self.RESULTS):
                (tmp / "results" / f"{i}.json").write_text(json.dumps(r))
            stale = dict(self.RESULTS[0], schema_version=run.RESULT_SCHEMA_VERSION - 1)
            (tmp / "results" / "stale.json").write_text(json.dumps(stale))
            with mock.patch.object(
                sys, "argv", ["analyze", "--data-dir", str(tmp)]
            ), mock.patch("builtins.print"):
                analyze.main()
            self.assertIn("5 runs", (tmp / "summary.md").read_text())
            self.assertEqual(len((tmp / "summary.csv").read_text().splitlines()), 3)


if __name__ == "__main__":
    unittest.main()
