import unittest
from types import SimpleNamespace

from orthogonal_dfa.l_star.suffix_family import SuffixFamily


class _Table:
    memo = SimpleNamespace(membership_queries=lambda strings: [1, 1, 0, 1])

    def suffix(self, v):
        return bytes([v])


class TestSuffixFamily(unittest.TestCase):
    def test_a_later_boundary_does_not_move_an_earlier_family(self):
        pst = SimpleNamespace(accept_thresh=0.8, reject_thresh=0.6, table=_Table())
        family = SuffixFamily(pst, [0, 1, 2, 3])
        self.assertIsNone(family.is_accept(b"", b""))
        pst.accept_thresh, pst.reject_thresh = 0.7, 0.5
        self.assertIsNone(family.is_accept(b"\x01", b""))


if __name__ == "__main__":
    unittest.main()
