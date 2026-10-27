import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orthogonal_dfa.experiments.random_minimal import analyze, run


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
        # falling with queries, except acceptance_rate, which rises
        "target": {
            metric: queries if metric == "acceptance_rate" else 1 / queries
            for metric in analyze.DIFFICULTY_METRICS
        },
    }


class TestAnalyze(unittest.TestCase):
    RESULTS = [
        _result(4, 0.05, 1.0, learned_states=4, queries=100),
        _result(4, 0.05, 0.995, learned_states=4, queries=200),
        _result(4, 0.05, 0.9, learned_states=3, queries=300),
        _result(4, 0.05, None, learned_states=None, error_type="Timeout"),
        _result(8, 0.05, 1.0, learned_states=8),
        _result(8, 0.1, 1.0, learned_states=8),
    ]

    def test_summary(self):
        rows = analyze.summary_rows(self.RESULTS)
        self.assertEqual(
            rows[0],
            [
                r"\multirow{1}{*}{4}",
                "0.05",
                "4",
                r"50\%",
                r"25\%",
                r"50\%",
                r"$2.00 \times 10^{2}$",
                r"$2.80 \times 10^{2}$",
            ],
        )
        self.assertEqual(rows[1], r"\midrule")
        self.assertEqual(rows[2][:3], [r"\multirow{2}{*}{8}", "0.05", "1"])
        self.assertEqual(rows[3][:3], ["", "0.1", "1"])

    def test_difficulty(self):
        rows = analyze.difficulty_rows(self.RESULTS)
        rho = dict(zip(analyze.DIFFICULTY_METRICS, rows[0][2:]))
        self.assertEqual(rho["endpoint_mass_min"], "$-1.00$")
        self.assertEqual(rho["acceptance_rate"], "$+1.00$")
        self.assertEqual(rows[2][2:], ["--"] * len(analyze.DIFFICULTY_METRICS))

    def test_writes_fragments_and_skips_stale_results(self):
        with tempfile.TemporaryDirectory() as data, tempfile.TemporaryDirectory() as out:
            data, out = Path(data), Path(out)
            (data / "results").mkdir()
            for i, r in enumerate(self.RESULTS):
                (data / "results" / f"{i}.json").write_text(json.dumps(r))
            stale = dict(self.RESULTS[0], schema_version=run.RESULT_SCHEMA_VERSION - 1)
            (data / "results" / "stale.json").write_text(json.dumps(stale))
            with mock.patch.object(analyze, "DATA_DIR", data), mock.patch.object(
                analyze, "OUT_DIR", out
            ), mock.patch("builtins.print"):
                analyze.main()
            summary = (out / "summary.tex").read_text()
            self.assertIn(r"\multirow{1}{*}{4} & 0.05 & 4 &", summary)
            self.assertTrue((out / "difficulty.tex").exists())


if __name__ == "__main__":
    unittest.main()
