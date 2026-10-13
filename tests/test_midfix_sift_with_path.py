"""``sift_with_path`` keeps the promises ``SiftWithPathCorrect`` makes, and ``sift`` runs on
it."""

import itertools
import unittest

from orthogonal_dfa.l_star.midfix_tree import MidfixTree

#: The midfixes a small tree's splits draw from.
_MIDFIXES = (bytes([0]), bytes([1]), bytes([0, 1]))


def small_trees():
    """Every tree up to three splits past the root, each split on an existing leaf
    with one of ``_MIDFIXES``."""
    for splits in range(4):
        for choices in itertools.product(range(splits + 1), _MIDFIXES, repeat=splits):
            tree = MidfixTree(())
            pairs = list(zip(choices[::2], choices[1::2]))
            for leaf, midfix in pairs:
                tree.split(sorted(tree.leaves())[leaf % tree.num_states], midfix)
            yield tree


def cuts(reads):
    """Every three-valued cut on ``reads``."""
    reads = sorted(set(reads))
    for values in itertools.product((True, False, None), repeat=len(reads)):
        yield dict(zip(reads, values)).get


def old_sift(tree, seq, decide):
    """``MidfixTree.sift`` before it was written on ``sift_with_path``."""
    node = tree.root
    while not isinstance(node, int):
        midfix, lookup = node
        decision = decide(seq, midfix)
        if decision is None:
            return None, seq + midfix
        node = lookup[decision]
    return node, None


class TestSiftWithPathCorrect(unittest.TestCase):
    def check(self, tree, cut, seq):
        reads, path = tree.sift_with_path(cut, seq)
        sides = tuple(cut(r) for r in reads if cut(r) is not None)
        for i, read in enumerate(reads):
            self.assertEqual(seq + tree.midfix_at(sides[:i]), read)
        self.assertLessEqual(len(reads), tree.depth)
        if path is not None:
            self.assertIn(path, {tree.path_of(leaf) for leaf in tree.leaves()})
            self.assertEqual(sides, path)
            self.assertTrue(all(cut(r) is not None for r in reads))
        else:
            self.assertIsNone(cut(reads[-1]))
            self.assertTrue(all(cut(r) is not None for r in reads[:-1]))

    def test_every_small_tree_and_cut(self):
        checked = 0
        for tree in small_trees():
            for seq in (b"", bytes([1, 0])):
                for cut in cuts(seq + m for m in tree.midfixes()):
                    self.check(tree, cut, seq)
                    checked += 1
        self.assertGreater(checked, 10000)


class TestSiftRunsOnSiftWithPath(unittest.TestCase):
    def test_every_small_tree_and_cut(self):
        for tree in small_trees():
            for seq in (b"", bytes([1, 0])):
                for cut in cuts(seq + m for m in tree.midfixes()):
                    calls = ([], [])

                    def logged(log, cut=cut):
                        def decide(s, midfix):
                            log.append((s, midfix))
                            return cut(s + midfix)

                        return decide

                    self.assertEqual(
                        old_sift(tree, seq, logged(calls[0])),
                        tree.sift(seq, logged(calls[1])),
                    )
                    self.assertEqual(calls[0], calls[1])


class TestExamples(unittest.TestCase):
    def test_a_decided_sift_reads_each_node_on_its_way(self):
        tree = MidfixTree(())
        tree.split(0, bytes([5]))
        reads = {b"\x07": True, b"\x07\x05": False}
        self.assertEqual(
            ([b"\x07", b"\x07\x05"], (True, False)),
            tree.sift_with_path(reads.get, b"\x07"),
        )

    def test_an_undecided_read_ends_the_sift(self):
        tree = MidfixTree(())
        tree.split(0, bytes([5]))
        reads = {b"\x07": True}
        self.assertEqual(
            ([b"\x07", b"\x07\x05"], None), tree.sift_with_path(reads.get, b"\x07")
        )


if __name__ == "__main__":
    unittest.main()
