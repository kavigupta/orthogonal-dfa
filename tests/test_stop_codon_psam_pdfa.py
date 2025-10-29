import unittest

import frame_alignment_checks as fac
import numpy as np
import torch

from orthogonal_dfa.manual_dfa.stop_codon_dfa import stop_codon_psamdfa
from orthogonal_dfa.psams.psams import TorchPSAMs
from orthogonal_dfa.utils.bases import parse_nucleotides_as_one_hot
from orthogonal_dfa.utils.probability import ZeroProbability


class TestLiteralPSAMs(unittest.TestCase):

    def run_psams_on_sequence(self, psams, sequence):
        x = parse_nucleotides_as_one_hot(sequence)[None]
        result = (torch.exp(psams(x)) * 1000).round().int().tolist()
        print(result)
        return result

    def test_basic_example_1(self):

        self.assertEqual(
            self.run_psams_on_sequence(
                TorchPSAMs.from_literal_strings(
                    "TAG", "TAA", "TGA", zero_prob=ZeroProbability(1e-7)
                ),
                "TAGCTAGATAG",
            ),
            [
                [
                    [1000, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [1000, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [1000, 0, 0],
                ]
            ],
        )

    def test_basic_example_2(self):
        self.assertEqual(
            self.run_psams_on_sequence(
                TorchPSAMs.from_literal_strings(
                    "TAG", "TAA", "TGA", zero_prob=ZeroProbability(1e-7)
                ),
                "TAGCTAAATGA",
            ),
            [
                [
                    [1000, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 1000, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 1000],
                ]
            ],
        )

    def test_basic_example_3(self):
        self.assertEqual(
            self.run_psams_on_sequence(
                TorchPSAMs.from_literal_strings(
                    "TAG", "TAA", "TGA", zero_prob=ZeroProbability(1e-7)
                ),
                "TAGCTAGTGA",
            ),
            [
                [
                    [1000, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [1000, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 1000],
                ]
            ],
        )


class TestStopCodonPSAMPDFA(unittest.TestCase):

    @property
    def model(self):
        return stop_codon_psamdfa("TAG", "TAA", "TGA", zero_prob=ZeroProbability(1e-7))

    def run_on_example(self, sequence):
        logprob = self.model(parse_nucleotides_as_one_hot(sequence)[None])
        assert logprob.shape[0] == 1
        prob = logprob[0].exp().item()
        self.assertTrue((prob > 0.99) or (prob < 0.01), f"Got prob {prob}")
        return int(round(prob))

    def test_stop_codon_psam_pdfa_on_examples(self):
        self.assertEqual(self.run_on_example("TAGCTAGATAG"), 1)
        self.assertEqual(self.run_on_example("TAGCTAAATGA"), 1)
        self.assertEqual(self.run_on_example("TAGCTAGTAG"), 0)
        self.assertEqual(
            self.run_on_example(
                "T[TA G]CT CC[T AG]T [TGA]".replace(" ", "")
                .replace("[", "")
                .replace("]", "")
            ),
            1,
        )

    def test_on_random_examples(self):
        rng = np.random.RandomState(0)
        sequences = rng.choice(4, size=(10000, 120))
        result_pred = self.model(
            torch.tensor(np.eye(4, dtype=np.float32)[sequences])
        ).exp()
        self.assertTrue(((result_pred < 0.01) | (result_pred > 0.99)).all())
        is_stop_actual = fac.all_frames_closed(np.eye(4)[sequences])
        result_pred = (result_pred.detach().numpy() > 0.5).astype(int).squeeze()
        bad = result_pred != is_stop_actual
        self.assertTrue(not any(bad))
