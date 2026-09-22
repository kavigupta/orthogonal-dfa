import os
import unittest

import torch
from permacache import stable_hash
from torch import nn

from orthogonal_dfa.spliceai.exon_score import SpliceAIExonScore
from orthogonal_dfa.spliceai.load_model import PRETRAINED_DIR, TracedFM, load_fm

FM_SEED1 = os.path.join(PRETRAINED_DIR, "fm-1.traced.pt")


def traced_stub():
    """A tiny traced module standing in for the FM trace, so the tests below need
    neither the artifact nor a GPU."""
    return torch.jit.trace(nn.Linear(4, 4).eval(), (torch.zeros(2, 4),))


class TestTracedFMHash(unittest.TestCase):
    def test_hash_follows_the_artifact_hash(self):
        # Same declared hash, different underlying weights: equal, because the
        # trace itself is never walked.
        self.assertEqual(
            stable_hash(TracedFM(traced_stub(), "abc"), version=2),
            stable_hash(TracedFM(traced_stub(), "abc"), version=2),
        )
        self.assertNotEqual(
            stable_hash(TracedFM(traced_stub(), "abc"), version=2),
            stable_hash(TracedFM(traced_stub(), "def"), version=2),
        )

    def test_hash_survives_being_wrapped_for_scoring(self):
        def wrap(sha):
            return stable_hash(
                SpliceAIExonScore(TracedFM(traced_stub(), sha)), version=2
            )

        self.assertEqual(wrap("abc"), wrap("abc"))
        self.assertNotEqual(wrap("abc"), wrap("def"))

    def test_forwards_to_the_wrapped_trace(self):
        trace = traced_stub()
        x = torch.zeros(2, 4)
        self.assertTrue(torch.equal(TracedFM(trace, "abc")(x), trace(x)))


@unittest.skipUnless(
    os.path.exists(FM_SEED1) and torch.cuda.is_available(),
    "FM trace / cuda not available; regenerate via scripts/convert_fm_to_torchscript.py",
)
class TestLoadFM(unittest.TestCase):
    def test_trace_loads_and_forwards_without_modular_splicing(self):
        model = load_fm(1)
        length = 500
        x = torch.zeros(2, length, 4, device="cuda")
        x[:, torch.arange(length), torch.randint(0, 4, (length,))] = 1.0
        with torch.no_grad():
            out = model(x)
        # (N, L - cl, 3): the cl=400 acceptor/donor/null logits SpliceAIExonScore reads.
        self.assertEqual(tuple(out.shape), (2, length - 400, 3))


if __name__ == "__main__":
    unittest.main()
