import unittest

from permacache import stable_hash
import torch

from orthogonal_dfa.manual_dfa.stop_codon_dfa import stop_codon_dfa
from orthogonal_dfa.utils.dfa import hash_dfa
from orthogonal_dfa.utils.pdfa import PDFA


class TestPDFARegression(unittest.TestCase):

    def test_pdfa_regression_0(self):
        torch.manual_seed(0)
        pdfas = [PDFA.create(10, 4) for _ in range(5)]
        self.assertEqual(
            stable_hash(pdfas),
            "66f697104bd3297167c16b19cb2603300747974da7941615c696cf15a4665f20",
        )
        x = torch.rand(3, 1000, 10)
        ys = [pdfa(x) for pdfa in pdfas]
        self.assertEqual(
            stable_hash(ys),
            "5588fca86e318513608173fb47aa46a038b36321f05f6bfc58397f622ffdfd78",
        )

    def test_pdfa_regression_1(self):
        torch.manual_seed(1)
        pdfas = [PDFA.create(10, 4) for _ in range(5)]
        self.assertEqual(
            stable_hash(pdfas),
            "238458c2f59529ebbf9b3890b187478928eb0da920c44b5db29395323440d795",
        )
        x = torch.rand(3, 1000, 10)
        ys = [pdfa(x) for pdfa in pdfas]
        self.assertEqual(
            stable_hash(ys),
            "5588fca86e318513608173fb47aa46a038b36321f05f6bfc58397f622ffdfd78",
        )
