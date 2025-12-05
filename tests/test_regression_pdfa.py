import unittest

import torch
from permacache import stable_hash

from orthogonal_dfa.utils.pdfa import PDFA


class TestPDFARegression(unittest.TestCase):

    def test_pdfa_regression_0(self):
        torch.manual_seed(0)
        pdfas = [PDFA.create(10, 4) for _ in range(5)]
        self.assertEqual(
            stable_hash(pdfas),
            "66f697104bd3297167c16b19cb2603300747974da7941615c696cf15a4665f20",
        )
        x = torch.rand(300, 10, 10)
        ys = [pdfa(x) for pdfa in pdfas]
        self.assertTrue(all(y.isfinite().all() for y in ys))
        self.assertEqual(
            stable_hash([y.detach().numpy().tolist() for y in ys]),
            "83eba1743552ef9e745ba14236085b47094676165c57e95585057051da892ed5",
        )

    def test_pdfa_regression_1(self):
        torch.manual_seed(1)
        pdfas = [PDFA.create(10, 4) for _ in range(5)]
        self.assertEqual(
            stable_hash(pdfas),
            "238458c2f59529ebbf9b3890b187478928eb0da920c44b5db29395323440d795",
        )
        x = torch.rand(300, 10, 10)
        ys = [pdfa(x) for pdfa in pdfas]
        self.assertTrue(all(y.isfinite().all() for y in ys))
        self.assertEqual(
            stable_hash([y.detach().numpy().tolist() for y in ys]),
            "4c427cf4337a1f3fdf6eeae6bfe0f39e370dd5de090b7f8badfabd67cf0af2a0",
        )
