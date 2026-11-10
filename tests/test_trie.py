import unittest

from automata.fa.dfa import DFA

from orthogonal_dfa.l_star.trie import Trie

LEAVES = DFA(
    states={0, 1},
    input_symbols={0, 1},
    transitions={0: {0: 0, 1: 1}, 1: {0: 1, 1: 0}},
    initial_state=1,
    final_states={0},
    allow_partial=False,
)


class TestTrie(unittest.TestCase):
    def test_walk_start_passes_the_spelled_out_prefixes(self):
        trie = Trie()
        trie.spell_out({b"", b"\x01"})
        self.assertEqual(2, trie.walk_start(b"\x01\x00\x00"))
        self.assertEqual(1, trie.walk_start(b"\x00\x01"))
        self.assertEqual(2, trie.depth)

    def test_spelling_out_bumps_the_version(self):
        trie = Trie()
        trie.spell_out({b""})
        self.assertEqual(1, trie.version)

    def test_empty_it_leaves_the_dfa_alone(self):
        self.assertIs(LEAVES, Trie().prepend(LEAVES, lambda v: 0, lambda u: True))

    def test_prepend_numbers_states_after_the_leaves_and_starts_at_the_empty_prefix(
        self,
    ):
        trie = Trie()
        trie.spell_out({b"", b"\x00"})
        dfa = trie.prepend(
            LEAVES, lambda v: None if v == b"\x01" else 0, lambda u: u == b"\x00"
        )
        self.assertEqual(2, dfa.initial_state)
        self.assertEqual({0: 3, 1: 1}, dfa.transitions[2])
        self.assertEqual({0: 0, 1: 0}, dfa.transitions[3])
        self.assertEqual({0, 3}, set(dfa.final_states))


if __name__ == "__main__":
    unittest.main()
