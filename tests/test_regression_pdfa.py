import unittest

import numpy as np
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
        x = torch.rand(2, 10, 10).log()
        ys = [pdfa(x) for pdfa in pdfas]
        self.assertTrue(
            np.allclose(
                [y.detach().numpy().tolist() for y in ys],
                [
                    [[15.333812713623047, 14.790033340454102]],
                    [[14.943010330200195, 14.346924781799316]],
                    [[15.535026550292969, 14.926868438720703]],
                    [[15.68808650970459, 15.099180221557617]],
                    [[15.938797950744629, 15.327362060546875]],
                ],
            )
        )

    def test_pdfa_regression_1(self):
        torch.manual_seed(1)
        pdfas = [PDFA.create(10, 4) for _ in range(5)]
        self.assertEqual(
            stable_hash(pdfas),
            "238458c2f59529ebbf9b3890b187478928eb0da920c44b5db29395323440d795",
        )
        x = torch.rand(2, 10, 10).log()
        ys = [pdfa(x) for pdfa in pdfas]
        self.assertTrue(
            np.allclose(
                [y.detach().numpy().tolist() for y in ys],
                [
                    [[15.766007423400879, 14.182374954223633]],
                    [[16.263561248779297, 14.709993362426758]],
                    [[16.354463577270508, 14.796015739440918]],
                    [[15.739842414855957, 14.226682662963867]],
                    [[16.048070907592773, 14.524038314819336]],
                ],
            )
        )
