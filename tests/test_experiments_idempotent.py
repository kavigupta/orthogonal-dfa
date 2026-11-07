import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from automata.fa.dfa import DFA

from orthogonal_dfa.experiments.capal_comparison import (
    LEARNER_CAPAL,
    Cell,
    run_our_bench,
    run_wall_sweep,
    write_experiment,
)
from orthogonal_dfa.experiments.capal_comparison.sweep import ETAS, LEARNERS, SEEDS
from orthogonal_dfa.experiments.length_oracles import (
    extract,
    label,
    random_strings,
    reproduce,
    residual_report,
)
from orthogonal_dfa.experiments.length_oracles.figure import minimal_dfa

LABEL = "m=80,pool=10,alpha=0.001"
ARGV = ["run_wall_sweep", "--etas", "0.3", "--seeds", "0", "1", "--configs", LABEL]


def sweep_cell(benchmark, seed):
    return Cell(
        benchmark=benchmark,
        family="ours",
        learner=LEARNER_CAPAL,
        eta=0.3,
        seed=seed,
        seconds=0.0,
        learner_config={"label": LABEL},
    )


class TestWallSweep(unittest.TestCase):
    def _main(self, out):
        benchmark = SimpleNamespace(
            name="b", symbols=2, target_states=2, truth=lambda: None
        )
        calls = []

        def capal_cell(b, *, seed, **_):
            calls.append(seed)
            return sweep_cell(b.name, seed)

        with mock.patch.object(
            run_wall_sweep, "our_benchmarks", lambda: [benchmark]
        ), mock.patch.object(run_wall_sweep, "capal_cell", capal_cell), mock.patch(
            "sys.argv", [*ARGV, "--out", str(out)]
        ), mock.patch(
            "builtins.print"
        ):
            run_wall_sweep.main()
        return calls

    def _write(self, out, seeds, complete):
        write_experiment(
            out,
            experiment="wall_sweep",
            generated_by="test",
            description="d",
            config={
                "etas": [0.3],
                "seeds": [0, 1],
                "benchmarks": ["b"],
                "configs": [LABEL],
                "matched_budget": False,
            },
            cells=[sweep_cell("b", s) for s in seeds],
            complete=complete,
        )

    def test_rerun_of_a_complete_file_neither_runs_nor_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sweep.json"
            self.assertEqual(self._main(out), [0, 1])
            before = out.stat().st_mtime_ns
            self.assertEqual(self._main(out), [])
            self.assertEqual(out.stat().st_mtime_ns, before)

    def test_partial_file_runs_only_its_missing_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sweep.json"
            self._write(out, seeds=[0], complete=False)
            self.assertEqual(self._main(out), [1])
            payload = json.loads(out.read_text())
            self.assertTrue(payload["complete"])
            self.assertEqual([c["seed"] for c in payload["cells"]], [0, 1])

    def test_complete_file_of_another_grid_is_not_finished(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sweep.json"
            self._write(out, seeds=[0, 1], complete=True)
            payload = json.loads(out.read_text())
            payload["config"]["seeds"] = [0]
            out.write_text(json.dumps(payload))
            # its cells cover the grid, so nothing runs, but it is rewritten
            self.assertEqual(self._main(out), [])
            self.assertEqual(json.loads(out.read_text())["config"]["seeds"], [0, 1])


class TestHeadToHead(unittest.TestCase):
    def test_complete_file_skips_building_the_benchmarks(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "orthogonal_dfa.experiments.capal_comparison.sweep.REPO_ROOT", Path(tmp)
        ), mock.patch.object(
            run_our_bench, "our_benchmarks", side_effect=AssertionError
        ), mock.patch(
            "builtins.print"
        ):
            write_experiment(
                Path(tmp) / "data" / "capal" / "our_benchmarks.json",
                experiment="our_benchmarks",
                generated_by="test",
                description="d",
                config={"etas": ETAS, "seeds": SEEDS, "learners": LEARNERS},
                cells=[],
                complete=True,
            )
            run_our_bench.main()


def write_run(run, oracle, seed, marker):
    Path(run).mkdir(parents=True, exist_ok=True)
    (Path(run) / "run.json").write_text(
        json.dumps(dict(oracle=oracle, seed=seed, marker=marker))
    )
    (Path(run) / "restarts.json").write_text(json.dumps([dict(restart=0)]))


class NeuralCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.runs_json = self.tmp / "runs.json"
        self.outputs = {
            "label": self.tmp / "labels.npz",
            "random_strings": self.tmp / "random_strings.json",
            "residual_report": self.tmp / "residual_products.json",
        }
        for patcher in (
            mock.patch.object(extract, "OUT", self.runs_json),
            mock.patch.object(label, "SAVED", self.outputs["label"]),
            mock.patch.object(random_strings, "OUT", self.outputs["random_strings"]),
            mock.patch.object(
                residual_report, "PRODUCTS", self.outputs["residual_report"]
            ),
            mock.patch.object(reproduce, "STAGGER_SECONDS", 0),
            mock.patch("builtins.print"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def save(self, pairs):
        self.runs_json.write_text(
            json.dumps(
                dict(
                    runs=[dict(oracle=o, seed=s, marker="saved") for o, s in pairs],
                    restarts=[dict(oracle=o, seed=s, restart=0) for o, s in pairs],
                )
            )
        )

    def all_pairs(self):
        return [(o, s) for o in ("saw", "flat", "residual") for s in range(4)]

    def main(self, *gpus):
        launched = []

        def launch(step, _gpu, _log, *args):
            launched.append(step if step != "run" else (args[1], int(args[3])))
            if step == "run":
                write_run(args[5], args[1], int(args[3]), "fresh")
            return SimpleNamespace(wait=lambda: 0, args=step)

        argv = ["reproduce", str(self.tmp / "out"), "--gpus", *map(str, gpus)]
        with mock.patch.object(reproduce, "launch", launch), mock.patch(
            "sys.argv", argv
        ):
            reproduce.main()
        return launched


class TestExtract(NeuralCase):
    def test_replaces_only_the_given_runs(self):
        self.save([("saw", 0), ("flat", 0)])
        run = self.tmp / "flat_s0"
        write_run(run, "flat", 0, "fresh")
        extract.extract([run])
        saved = json.loads(self.runs_json.read_text())
        self.assertEqual(
            sorted((r["oracle"], r["marker"]) for r in saved["runs"]),
            [("flat", "fresh"), ("saw", "saved")],
        )
        self.assertEqual(len(saved["restarts"]), 2)


class TestReproduce(NeuralCase):
    def test_everything_saved_runs_nothing_and_needs_no_gpu(self):
        self.save(self.all_pairs())
        for path in self.outputs.values():
            path.touch()
        before = self.runs_json.read_bytes()
        self.assertEqual(self.main(), [])
        self.assertEqual(self.runs_json.read_bytes(), before)
        self.assertFalse((self.tmp / "out").exists())

    def test_missing_work_without_gpus_fails(self):
        self.save(self.all_pairs()[1:])
        with self.assertRaises(SystemExit):
            self.main()

    def test_only_the_missing_run_runs(self):
        self.save(self.all_pairs()[1:])
        for path in self.outputs.values():
            path.touch()
        self.assertEqual(self.main(0), ["oracles", ("saw", 0)])
        saved = json.loads(self.runs_json.read_text())["runs"]
        self.assertEqual(len(saved), 12)
        self.assertEqual(sum(r["marker"] == "fresh" for r in saved), 1)

    def test_residual_runs_follow_their_first_stage_and_the_report_follows_them(self):
        self.save([p for p in self.all_pairs() if p[0] == "saw"])
        self.assertEqual(
            self.main(0),
            [
                "oracles",
                "label",
                *[("flat", s) for s in range(4)],
                "residual",
                *[("residual", s) for s in range(4)],
                "random_strings",
                "residual_report",
            ],
        )


class TestMinimalDfa(unittest.TestCase):
    def test_numbered_breadth_first_from_the_start(self):
        # 3 -0-> 1 -0-> 0 loops, with 2 unreachable
        row = lambda t: [t] * 5
        table = dict(transitions=[row(0), row(0), row(0), row(1)], start=3)
        dfa = minimal_dfa(table, [1, 0, 0, 0])
        self.assertEqual(dfa.initial_state, 0)
        self.assertEqual(dfa.transitions[0][0], 1)
        self.assertEqual(dfa.transitions[1][0], 2)
        self.assertEqual(dfa.transitions[2][0], 2)
        self.assertEqual(dfa.final_states, {2})
        self.assertIsInstance(dfa, DFA)


if __name__ == "__main__":
    unittest.main()
