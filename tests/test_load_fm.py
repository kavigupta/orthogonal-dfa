import os
import unittest

import torch

from orthogonal_dfa.spliceai.load_model import PRETRAINED_DIR, load_fm

FM_SEED1 = os.path.join(PRETRAINED_DIR, "fm-1.traced.pt")


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
