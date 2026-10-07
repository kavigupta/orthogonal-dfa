import unittest
from types import SimpleNamespace

from orthogonal_dfa.l_star.suffix_family import SuffixFamily


class TestSuffixFamily(unittest.TestCase):
    def test_a_later_boundary_does_not_move_an_earlier_family(self):
        pst = SimpleNamespace(accept_thresh=0.6, reject_thresh=0.4)
        family = SuffixFamily(pst, [0, 1, 2, 3])
        votes = [1, 1, 0, 1]
        self.assertIsNone(family.train_side(votes))
        pst.accept_thresh, pst.reject_thresh = 0.5, 0.3
        self.assertIsNone(family.train_side(votes))


if __name__ == "__main__":
    unittest.main()
