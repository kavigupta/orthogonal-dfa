import io
import unittest

import numpy as np
import torch
from permacache import stable_hash

from orthogonal_dfa.spliceai.exon_score import SpliceAIExonScore
from orthogonal_dfa.spliceai.module import SpliceAI, SpliceAIModule

SMALL_CL = 80


def stable(model):
    return stable_hash(model, version=2)


def exon_score(seed, cl=SMALL_CL):
    torch.manual_seed(seed)
    return SpliceAIExonScore(SpliceAIModule(window=cl)).eval()


class TestSpliceAIStableHash(unittest.TestCase):
    def test_a_wrapped_model_is_hashable(self):
        # stable_hash's generic Module encoding walks __dict__, which holds the
        # preprocess function; without __permacache_hash__ this raises TypeError.
        self.assertIsInstance(stable(exon_score(0)), str)

    def test_same_weights_hash_the_same(self):
        self.assertEqual(stable(exon_score(0)), stable(exon_score(0)))

    def test_different_weights_hash_differently(self):
        self.assertNotEqual(stable(exon_score(0)), stable(exon_score(1)))

    def test_dilation_is_not_invisible(self):
        # ar sets the dilations, which move the receptive field without changing any
        # parameter shape, so these two have byte-identical state_dicts.  Hashing the
        # weights alone would collide them.
        w = np.array([11, 11])
        wide = SpliceAI(l=8, w=w, ar=np.array([1, 4]))
        narrow = SpliceAI(l=8, w=w, ar=np.array([1, 1]))
        wide.load_state_dict(narrow.state_dict())
        self.assertNotEqual(narrow.cl, wide.cl)
        self.assertNotEqual(stable(narrow), stable(wide))

    def test_hash_survives_a_pickle_round_trip(self):
        # load_spliceai unpickles whole modules, which restores preprocess onto the
        # instance; the hook is on the class, so the hash is unaffected.
        model = exon_score(0)
        buffer = io.BytesIO()
        torch.save(model, buffer)
        buffer.seek(0)
        loaded = torch.load(buffer, weights_only=False)
        self.assertTrue(hasattr(loaded.model.spliceai, "preprocess"))
        self.assertEqual(stable(loaded), stable(model))

    def test_loading_weights_makes_hashes_agree(self):
        model, other = exon_score(0), exon_score(1)
        self.assertNotEqual(stable(model), stable(other))
        other.load_state_dict(model.state_dict())
        self.assertEqual(stable(model), stable(other))


if __name__ == "__main__":
    unittest.main()
