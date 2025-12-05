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
            stable_hash(ys),
            "21eec4f4a8bad6160b0a3ba4bace91236ff2e03cd4fcfb688f5db887bb771ac3",
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
            stable_hash(ys),
            "e0ddbdc6a91b11bcbcf93248da769e11742d816299f0193dfbece84579f03603",
        )
