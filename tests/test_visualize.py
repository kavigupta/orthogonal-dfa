import os
import shutil
import tempfile
import unittest

import matplotlib
import numpy as np

from orthogonal_dfa.l_star.visualize import (
    _class_colors,
    _dot_layout,
    render_diagnostics,
    sample_class_distribution,
)
from tests.dfas import PARITY

# render_diagnostics imports pyplot lazily, so this lands before any backend is
# selected -- the tests must not need a display.
matplotlib.use("Agg")

# `dot` does the graph layout; without it there is nothing to draw into.
needs_dot = unittest.skipIf(shutil.which("dot") is None, "graphviz `dot` not installed")


class _StubSampler:
    """Fixed-length uniform strings, like the learner's own sampler."""

    length = 6

    def sample(self, rng, alphabet_size):
        return rng.integers(0, alphabet_size, self.length, dtype=np.uint8).tobytes()


class _StubPst:
    alphabet_size = 2
    sampler = _StubSampler()


class _StubLearner:
    """The duck type ``visualize`` consumes -- no DirectLStarLearner needed."""

    pst = _StubPst()
    num_states = 2
    dt = ((), {True: 0, False: 1})
    access = {0: [], 1: [1]}
    transitions = {0: {0: 0, 1: 1}, 1: {0: 1, 1: 0}}

    def sift(self, seq):
        if len(seq) == 3:  # exercise the indecisive (false-negative) slice
            return None
        return sum(seq) % 2


@needs_dot
class TestDotLayout(unittest.TestCase):
    def test_parses_nodes_and_edges(self):
        layout = _dot_layout({"a": (0.5, 0.5), "b": (0.5, 0.5)}, [("a", "b", "0,1")])
        self.assertEqual(set(layout["nodes"]), {"a", "b"})
        self.assertEqual(len(layout["edges"]), 1)
        edge = layout["edges"][0]
        self.assertEqual((edge["tail"], edge["head"]), ("a", "b"))
        self.assertGreaterEqual(len(edge["pts"]), 4)  # a cubic spline
        self.assertEqual(edge["label"][0], "0,1")
        w, h = layout["size"]
        self.assertGreater(w, 0)
        self.assertGreater(h, 0)

    def test_unlabelled_edge_has_no_label(self):
        layout = _dot_layout({"a": (0.2, 0.2), "b": (0.2, 0.2)}, [("a", "b", None)])
        self.assertIsNone(layout["edges"][0]["label"])


class TestClassColors(unittest.TestCase):
    def test_distinct_within_the_palette(self):
        colors = _class_colors(range(8))
        self.assertEqual(len(set(colors.values())), 8)

    def test_folds_past_the_palette(self):
        colors = _class_colors(range(11))
        self.assertEqual(colors[8], colors[10])  # both the "other" slot


class TestSampleClassDistribution(unittest.TestCase):
    def test_covers_transient_states_and_counts_indecisives(self):
        learner = _StubLearner()
        dist = sample_class_distribution(
            learner.sift,
            PARITY,
            pst=learner.pst,
            rng=np.random.default_rng(0),
            num_samples=40,
            per_state=25,
        )
        # Bucketing by prefix (not by end state) must reach every true state.
        self.assertEqual(set(dist), set(PARITY.states))
        for state in PARITY.states:
            self.assertGreater(sum(dist[state].values()), 0)
        # The stub sifts length-3 prefixes indecisively, so None must show up.
        self.assertTrue(any(None in c for c in dist.values()))

    def test_accepts_any_classifier_and_an_explicit_pst(self):
        # A resolver-style classifier: no learner, no .sift, just a callable.
        seen = []
        dist = sample_class_distribution(
            lambda s: seen.append(s) or len(s) % 2,
            PARITY,
            pst=_StubPst(),
            rng=np.random.default_rng(1),
            num_samples=20,
            per_state=8,
        )
        self.assertEqual(set(dist), set(PARITY.states))
        self.assertTrue(seen)

    def test_prefill_sees_every_sampled_string(self):
        learner = _StubLearner()
        batches = []
        dist = sample_class_distribution(
            learner.sift,
            PARITY,
            pst=learner.pst,
            rng=np.random.default_rng(2),
            num_samples=20,
            per_state=8,
            prefill=batches.append,
        )
        self.assertEqual(len(batches), 1)  # one batched pass, not one per string
        self.assertEqual(len(batches[0]), sum(sum(c.values()) for c in dist.values()))


@needs_dot
class TestRenderDiagnostics(unittest.TestCase):
    def test_writes_an_image(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "diag.png")
            path = render_diagnostics(
                _StubLearner(),
                PARITY,
                rng=np.random.default_rng(0),
                path=out,
                num_samples=20,
                per_state=10,
                final_states={0},
                flipped={1},
                dpi=60,
            )
            self.assertEqual(path, out)
            self.assertGreater(os.path.getsize(out), 0)


if __name__ == "__main__":
    unittest.main()
