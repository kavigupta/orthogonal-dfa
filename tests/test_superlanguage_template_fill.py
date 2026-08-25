"""Tests for the state a template filler carries: how far a hole's legality reaches,
and which symbols each context leaves it.
"""

import itertools
import unittest

from orthogonal_dfa.superlanguage.template_fill import TemplateFiller

AA = TemplateFiller(forbidden=((0, 0),), base_alphabet_size=2)

SHAPES = [
    (((0, 0),), 2),
    (((0, 1), (1, 0)), 2),
    (((0, 1, 2), (1, 2)), 4),
    (((3, 0, 2), (3, 2, 0), (3, 0, 0)), 4),
    (((0,),), 3),
    ((), 3),
    (((0, 1, 0, 1),), 2),
]


def tables(forbidden, base):
    # the encoding is what this module is for, and it has no public reader yet
    # pylint: disable=protected-access
    return TemplateFiller(forbidden, base)._tables


def context_width(forbidden):
    return max((len(p) for p in forbidden), default=1) - 1


def window_state(window, radix):
    """The docstring's state(j) for a window of the symbols following j."""
    return sum(radix**i * symbol for i, symbol in enumerate(window))


class TestTransferTables(unittest.TestCase):
    """Every state and symbol against the definition. Nothing public reads shift or
    the initial state until the sampler lands, and every_context_is_fillable reduces
    allowed to one bit, so going through the public surface would pin almost none of
    the encoding.
    """

    def windows(self, forbidden, base):
        return itertools.product(range(base + 1), repeat=context_width(forbidden))

    def test_each_state_is_a_window_of_following_symbols(self):
        for forbidden, base in SHAPES:
            _, allowed, shift = tables(forbidden, base)
            packed = sorted(
                window_state(win, base + 1) for win in self.windows(forbidden, base)
            )
            self.assertEqual(packed, list(range(allowed.shape[1])), forbidden)
            self.assertEqual(shift.shape, allowed.shape, forbidden)

    def test_shift_takes_the_window_one_position_left(self):
        for forbidden, base in SHAPES:
            _, _, shift = tables(forbidden, base)
            w = context_width(forbidden)
            for win in self.windows(forbidden, base):
                for c in range(base):
                    with self.subTest(forbidden=forbidden, window=win, symbol=c):
                        self.assertEqual(
                            shift[c, window_state(win, base + 1)],
                            window_state(((c,) + win)[:w], base + 1),
                        )

    def test_allowed_bans_exactly_the_symbols_that_start_a_pattern(self):
        for forbidden, base in SHAPES:
            _, allowed, _ = tables(forbidden, base)
            for win in self.windows(forbidden, base):
                for c in range(base):
                    banned = any(
                        p[0] == c and tuple(p[1:]) == win[: len(p) - 1]
                        for p in forbidden
                    )
                    with self.subTest(forbidden=forbidden, window=win, symbol=c):
                        self.assertEqual(
                            allowed[c, window_state(win, base + 1)], not banned
                        )

    def test_the_initial_state_is_all_sentinel(self):
        for forbidden, base in SHAPES:
            initial, allowed, _ = tables(forbidden, base)
            width = context_width(forbidden)
            self.assertEqual(
                initial, window_state((base,) * width, base + 1), forbidden
            )
            # the sentinel is outside the alphabet, so only length-1 patterns bite
            self.assertEqual(
                {c for c in range(base) if not allowed[c, initial]},
                {p[0] for p in forbidden if len(p) == 1},
                forbidden,
            )

    def test_the_tables_are_shared_and_read_only(self):
        one = tables(((0, 0),), 2)
        two = tables(((0, 0),), 2)
        for mine, theirs in zip(one[1:], two[1:]):
            self.assertIs(mine, theirs)
            with self.assertRaises(ValueError):
                mine[0, 0] = True


class TestConstruction(unittest.TestCase):
    def test_rejects_an_empty_base_alphabet(self):
        with self.assertRaises(AssertionError):
            TemplateFiller(forbidden=(), base_alphabet_size=0)

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
