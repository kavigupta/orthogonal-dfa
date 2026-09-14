"""What the discrimination tree can offer as a distinguisher."""

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


if __name__ == "__main__":
    unittest.main()
