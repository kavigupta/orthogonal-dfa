import unittest

from orthogonal_dfa.l_star.midfix_tree import MidfixTree


class TestMidfixesAreWhatCanBeProposed(unittest.TestCase):
    def test_each_node_contributes_its_own(self):
        tree = MidfixTree(())
        tree.split(0, bytes([5]))

        self.assertEqual({b"", bytes([5])}, set(tree.midfixes()))

    def test_a_grandchild_contributes_too(self):
        tree = MidfixTree(())
        new = tree.split(0, bytes([5]))
        tree.split(new, bytes([6]))

        self.assertEqual({b"", bytes([5]), bytes([6])}, set(tree.midfixes()))

    def test_one_reused_on_another_leaf_is_listed_once(self):
        tree = MidfixTree(())
        tree.split(0, bytes([5]))
        tree.split(1, bytes([5]))

        self.assertEqual({b"", bytes([5])}, set(tree.midfixes()))


class TestTheFirstDisagreement(unittest.TestCase):
    def test_reads_that_part_give_the_midfix(self):
        tree = MidfixTree(())

        found = tree.first_disagreement(b"a", b"b", lambda s, m: s == b"a", b"x")

        self.assertEqual((b"x", None), found)

    def test_an_undecided_read_gives_its_boundary_string(self):
        tree = MidfixTree(())
        decide = lambda s, m: None if s == b"b" else True

        self.assertEqual(
            (None, b"bx"), tree.first_disagreement(b"a", b"b", decide, b"x")
        )


if __name__ == "__main__":
    unittest.main()
