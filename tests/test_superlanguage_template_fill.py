"""Tests for the state a template filler carries: how far a hole's legality reaches,
and which symbols each context leaves it.
"""

import unittest

from orthogonal_dfa.superlanguage.template_fill import TemplateFiller

AA = TemplateFiller(forbidden=((0, 0),), base_alphabet_size=2)


class TestConstruction(unittest.TestCase):
    def test_rejects_bad_patterns(self):
        with self.assertRaises(AssertionError):
            TemplateFiller(forbidden=((0, 9),), base_alphabet_size=4)
        with self.assertRaises(AssertionError):
            TemplateFiller(forbidden=((),), base_alphabet_size=4)

    def test_duplicate_and_prefix_related_patterns_are_fine(self):
        # Unlike a vocabulary, the filler reads these as nothing but strings to
        # keep out of the holes, so overlapping bans just pile up.
        filler = TemplateFiller(
            forbidden=((0, 1), (0, 1), (0, 1, 2)), base_alphabet_size=3
        )
        self.assertTrue(filler.every_context_is_fillable)

    def test_patterns_are_normalized_to_tuples(self):
        # frozen dataclasses are hashable, and the tables are cached on the patterns
        filler = TemplateFiller(forbidden=[[0, 0]], base_alphabet_size=2)
        self.assertEqual(filler.forbidden, ((0, 0),))
        self.assertTrue(filler.every_context_is_fillable)


class TestFillability(unittest.TestCase):
    def test_a_pattern_leaves_the_other_symbols(self):
        self.assertTrue(AA.every_context_is_fillable)

    def test_patterns_that_crowd_out_every_symbol(self):
        # After a 0, both 00 and 10 are banned and the alphabet is out of symbols.
        crowded = TemplateFiller(forbidden=((0, 0), (1, 0)), base_alphabet_size=2)
        self.assertFalse(crowded.every_context_is_fillable)

    def test_no_patterns_constrains_nothing(self):
        for base in (1, 4):
            filler = TemplateFiller(forbidden=(), base_alphabet_size=base)
            self.assertTrue(filler.every_context_is_fillable)

    def test_a_one_symbol_pattern_bans_that_symbol_everywhere(self):
        # Nothing follows a length-1 pattern, so there is a single context.
        self.assertTrue(
            TemplateFiller(
                forbidden=((0,),), base_alphabet_size=2
            ).every_context_is_fillable
        )
        self.assertFalse(
            TemplateFiller(
                forbidden=((0,),), base_alphabet_size=1
            ).every_context_is_fillable
        )

    def test_a_context_two_symbols_deep_can_be_the_dead_one(self):
        # Nothing may precede 01, and only that context is dead, so getting this
        # right needs the state to hold both following symbols in order.
        filler = TemplateFiller(forbidden=((0, 0, 1), (1, 0, 1)), base_alphabet_size=2)
        self.assertFalse(filler.every_context_is_fillable)
        reversed_context = TemplateFiller(
            forbidden=((0, 1, 0), (1, 1, 0)), base_alphabet_size=2
        )
        self.assertFalse(reversed_context.every_context_is_fillable)
        self.assertTrue(
            TemplateFiller(
                forbidden=((0, 0, 1), (1, 1, 0)), base_alphabet_size=2
            ).every_context_is_fillable
        )


if __name__ == "__main__":
    unittest.main()
