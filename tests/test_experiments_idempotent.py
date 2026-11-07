import hashlib
import tempfile
import unittest
from multiprocessing.pool import ThreadPool
from pathlib import Path
from unittest import mock

from orthogonal_dfa.experiments.capal_comparison import (
    run_capal_bench,
    run_our_bench,
    run_wall_sweep,
)
from orthogonal_dfa.experiments.length_oracles import reproduce
from orthogonal_dfa.experiments.luhn import run as luhn_run
from orthogonal_dfa.experiments.random_minimal import capal_random
from orthogonal_dfa.experiments.random_minimal import run as random_run

RESULTS = [
    Path(__file__).resolve().parents[1] / "data" / name
    for name in ("capal", "luhn", "neural", "random_minimal")
]

PROCESSES = ["--processes", "1"]
ENTRY_POINTS = [
    (run_capal_bench, ["capal_benchmarks"], []),
    (run_our_bench, ["our_benchmarks"], []),
    (run_wall_sweep, ["our_benchmarks"], []),
    (run_wall_sweep, ["our_benchmarks"], ["--matched-budget"]),
    (random_run, ["_ensure_target", "_run_result"], PROCESSES),
    (capal_random, ["_run_result"], PROCESSES),
    (luhn_run, ["_run_result"], PROCESSES),
    (reproduce, [], [tempfile.gettempdir()]),
]


def digest():
    return {
        path: hashlib.md5(path.read_bytes()).hexdigest()
        for root in RESULTS
        for path in root.rglob("*")
        if path.is_file()
    }


class TestSavedResults(unittest.TestCase):
    def test_every_entry_point_runs_nothing(self):
        before = digest()
        for module, workers, argv in ENTRY_POINTS:
            with self.subTest(module=module.__name__, argv=argv):
                failing = {w: mock.Mock(side_effect=AssertionError(w)) for w in workers}
                with mock.patch.multiple(module, **failing), mock.patch(
                    "sys.argv", ["main", *argv]
                ), mock.patch("multiprocessing.Pool", ThreadPool), mock.patch(
                    "builtins.print"
                ):
                    module.main()
        self.assertEqual(digest(), before)


if __name__ == "__main__":
    unittest.main()
