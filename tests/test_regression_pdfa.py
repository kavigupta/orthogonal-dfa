import unittest

import numpy as np
import torch
from permacache import stable_hash

from orthogonal_dfa.psams.psam_pdfa import PSAMPDFA
from orthogonal_dfa.utils.pdfa import PDFA, PDFAHyberbolicParameterization


class TestPDFARegression(unittest.TestCase):

    def test_pdfa_regression_0(self):
        torch.manual_seed(0)
        pdfas = [PDFA.create(10, 4) for _ in range(5)]
        self.assertEqual(
            stable_hash(pdfas, version=2),
            "c97dded6f9813868a2ae85169dc0221e3f4bfa159ddd935498c2616e3c154d51",
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
            stable_hash(pdfas, version=2),
            "090117cd8c4ad8360ddd3102988793b4cfcb6b6d2e33eac8da5ba531041224f8",
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

    def test_different_hashes(self):
        torch.manual_seed(0)
        pdfa1 = PDFA.create(10, 4)
        torch.manual_seed(0)
        pdfa2 = PDFAHyberbolicParameterization.create(10, 4)
        self.assertNotEqual(
            stable_hash(pdfa1.initialized, version=2),
            stable_hash(pdfa2.initialized, version=2),
        )
        self.assertNotEqual(
            stable_hash(pdfa1, version=2), stable_hash(pdfa2, version=2)
        )
        x = torch.rand(3, 10, 10)
        y1 = pdfa1(x)
        y2 = pdfa2(x)
        self.assertTrue(y1.isfinite().all())
        self.assertTrue(y2.isfinite().all())
        self.assertNotEqual(stable_hash(y1, version=2), stable_hash(y2, version=2))

    def test_different_hashes_psampdfa(self):
        torch.manual_seed(0)
        pdfa1 = PSAMPDFA.create(10, 4, 10, 4)
        torch.manual_seed(0)
        pdfa2 = PSAMPDFA.create(10, 4, 10, 4, pdfa_typ=PDFAHyberbolicParameterization)
        self.assertNotEqual(
            stable_hash(pdfa1, version=2), stable_hash(pdfa2, version=2)
        )
