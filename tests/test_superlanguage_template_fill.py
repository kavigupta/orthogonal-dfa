"""Tests for the template filler: the state it carries, and the uniformity of the
draw over a template's legal fillings, with no vocabulary or parser in sight.
"""

import itertools
import unittest

import numpy as np

from orthogonal_dfa.superlanguage.template_fill import FREE, TemplateFiller

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


def starts_a_pattern(filler, string, position):
    return any(
        tuple(string[position : position + len(p)]) == p for p in filler.forbidden
    )


def contains(string, gram):
    """Whether gram occurs anywhere in string, at a hole or not."""
    return any(
        tuple(string[i : i + len(gram)]) == tuple(gram)
        for i in range(len(string) - len(gram) + 1)
    )


def legal_fillings(filler, template):
    """Brute force, from the definition rather than from the sampler's tables."""
    holes = [j for j, c in enumerate(template) if c == FREE]
    out = []
    for choice in itertools.product(
        range(filler.base_alphabet_size), repeat=len(holes)
    ):
        filled = list(template)
        for j, c in zip(holes, choice):
            filled[j] = c
        if not any(starts_a_pattern(filler, filled, j) for j in holes):
            out.append(tuple(filled))
    return out


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


class TestContract(unittest.TestCase):
    def test_a_template_with_no_holes_comes_back_verbatim(self):
        template = [1, 0, 1, 1, 0]
        self.assertEqual(AA.fill(template, np.random.default_rng(0)), template)

    def test_a_fixed_position_may_start_a_pattern(self):
        # The filler polices holes only: 00 sitting in the template is left alone.
        self.assertEqual(AA.fill([0, 0, 0, 0], np.random.default_rng(0)), [0, 0, 0, 0])

    def test_a_hole_never_starts_a_pattern(self):
        filler = TemplateFiller(forbidden=((0, 1, 2), (1, 2)), base_alphabet_size=4)
        rng = np.random.default_rng(0)
        for _ in range(300):
            template = [
                FREE if rng.random() < 0.5 else int(rng.integers(4))
                for _ in range(int(rng.integers(1, 12)))
            ]
            out = filler.fill(template, rng)
            self.assertEqual(len(out), len(template))
            for j, c in enumerate(template):
                if c == FREE:
                    self.assertFalse(starts_a_pattern(filler, out, j), (template, out))
                else:
                    self.assertEqual(out[j], c)

    def test_a_pattern_running_off_the_end_does_not_constrain(self):
        # The last hole cannot start 00 no matter what it takes, so 0 stays legal
        # even though the alphabet offers nothing else.
        one_symbol = TemplateFiller(forbidden=((0, 0),), base_alphabet_size=1)
        self.assertEqual(one_symbol.fill([FREE], np.random.default_rng(0)), [0])

    def test_empty_inputs(self):
        self.assertEqual(AA.fill_many([], []), [])
        self.assertEqual(AA.fill([], np.random.default_rng(0)), [])

    def test_needs_one_rng_per_template(self):
        with self.assertRaises(AssertionError):
            AA.fill_many([[FREE], [FREE]], [np.random.default_rng(0)])


class TestConstruction(unittest.TestCase):
    def test_rejects_an_empty_base_alphabet(self):
        with self.assertRaises(AssertionError):
            TemplateFiller(forbidden=(), base_alphabet_size=0)

    def test_rejects_template_symbols_off_the_alphabet(self):
        # -2 is _PAD: unchecked it reads as a column the template does not occupy,
        # and AA.fill([FREE, -2]) returns 00, a forbidden pattern at a hole.
        for bad in (-2, -3, 2):
            with self.assertRaises(AssertionError):
                AA.fill([FREE, bad], np.random.default_rng(0))
        self.assertEqual(len(AA.fill([FREE, 1, FREE], np.random.default_rng(0))), 3)

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

    def test_a_template_with_no_legal_filling_is_refused(self):
        crowded = TemplateFiller(forbidden=((0, 0), (1, 0)), base_alphabet_size=2)
        self.assertEqual(legal_fillings(crowded, [FREE, 0]), [])
        with self.assertRaises(AssertionError):
            crowded.fill([FREE, 0], np.random.default_rng(0))

    def test_a_dead_context_reached_only_at_fixed_positions_is_fine(self):
        # No symbol can precede 01 here, but the template only ever puts fixed
        # positions there, and fixed positions are unconstrained.
        filler = TemplateFiller(forbidden=((0, 0), (1, 0, 1)), base_alphabet_size=2)
        self.assertFalse(filler.every_context_is_fillable)
        self.assertEqual(filler.fill([1, 0, 1], np.random.default_rng(0)), [1, 0, 1])
        for seed in range(50):
            out = filler.fill([FREE, 1, 0, 1], np.random.default_rng(seed))
            self.assertIn(tuple(out), legal_fillings(filler, [FREE, 1, 0, 1]))


class TestUniformity(unittest.TestCase):
    def check_uniform(self, filler, template, reps, bound):
        fillings = {f: i for i, f in enumerate(legal_fillings(filler, template))}
        self.assertGreater(len(fillings), 1, "need room to be non-uniform")
        count = reps * len(fillings)
        filled = filler.fill_many(
            [template] * count, [np.random.default_rng(i) for i in range(count)]
        )
        counts = np.zeros(len(fillings))
        for one in filled:
            self.assertIn(tuple(one), fillings, "filled outside the legal set")
            counts[fillings[tuple(one)]] += 1
        expected = count / len(fillings)
        chi2 = ((counts - expected) ** 2 / expected).sum()
        self.assertLess(chi2, bound * (len(fillings) - 1))
        return chi2 / (len(fillings) - 1)

    def test_uniform_over_a_self_overlapping_ban(self):
        """A hole next to 00 has one choice and a hole next to 01 has two, so the
        even draw over-weights the cramped branch; sized so it would land near 60x
        the degrees of freedom against 0.4x for the real one.
        """
        self.check_uniform(AA, [FREE, FREE, 0, 0, FREE, FREE], 400, 4)

    def test_uniform_with_patterns_of_two_lengths(self):
        """Sized so dropping the fiber counts lands near 30x the degrees of freedom
        against 1.0x for the real one.
        """
        filler = TemplateFiller(forbidden=((0, 1, 2), (1, 2)), base_alphabet_size=3)
        self.check_uniform(filler, [FREE, FREE, 1, 2, FREE, FREE, FREE], 200, 2)

    def test_no_patterns_leaves_every_symbol_equally_likely(self):
        filler = TemplateFiller(forbidden=(), base_alphabet_size=4)
        filled = filler.fill_many(
            [[FREE] * 10] * 2000, [np.random.default_rng(i) for i in range(2000)]
        )
        freqs = np.bincount(np.concatenate(filled), minlength=4) / (2000 * 10)
        self.assertLess(np.abs(freqs - 0.25).max(), 0.01)


class TestBatching(unittest.TestCase):
    def test_batching_does_not_change_a_result(self):
        filler = TemplateFiller(forbidden=((0, 1, 2), (1, 2)), base_alphabet_size=4)
        rng = np.random.default_rng(0)
        templates = [
            [FREE if rng.random() < 0.5 else int(rng.integers(4)) for _ in range(n)]
            for n in rng.integers(1, 15, size=200)
        ]
        seeds = [np.random.default_rng(i) for i in range(len(templates))]
        batched = filler.fill_many(templates, seeds)
        solo = [
            filler.fill(t, np.random.default_rng(i)) for i, t in enumerate(templates)
        ]
        self.assertEqual(batched, solo)

    def test_a_batch_split_across_chunks_matches_one_chunk(self):
        # 4096 is the chunk cap, so this batch is cut in two.
        count = 5000
        templates = [[FREE, FREE, 0, 0]] * count
        rngs = [np.random.default_rng(i) for i in range(count)]
        chunked = AA.fill_many(templates, rngs)
        solo = [AA.fill(t, np.random.default_rng(i)) for i, t in enumerate(templates)]
        self.assertEqual(chunked, solo)

    def test_a_long_template_does_not_overflow_the_counts(self):
        # The fiber grows like phi^width here, so past ~1500 the unrescaled counts
        # leave float64 and every weight becomes inf or nan.
        out = AA.fill([FREE] * 2000, np.random.default_rng(0))
        self.assertEqual(len(out), 2000)
        for j in range(len(out) - 1):
            self.assertFalse(out[j] == 0 and out[j + 1] == 0, j)

    def test_a_draw_of_exactly_zero_skips_banned_symbols(self):
        # Next to 00 the hole may not take 0, and 0 leads the cumulative weights,
        # so a draw at the very bottom has to step past a zero-weight entry.
        class ZeroDraws:
            def random(self, size):
                return np.zeros(size)

        out = AA.fill_many([[FREE, FREE, 0, 0]], [ZeroDraws()])[0]
        self.assertEqual(out, [0, 1, 0, 0])


class TestNgramFrequencyFuzz(unittest.TestCase):
    """Random patterns, random template, random n-gram: the rate at which the
    sampler draws a string containing that n-gram has to match the rate over the
    template's legal fillings, which is what uniformity over them means.

    The n-gram is taken out of a filling that is actually legal, so every trial
    tests a live probability rather than an impossible one.
    """

    trials = 40
    draws = 3000
    sigmas = 5

    def random_case(self, rng):
        base = int(rng.integers(2, 4))
        forbidden = tuple(
            tuple(int(c) for c in rng.integers(base, size=int(rng.integers(1, 4))))
            for _ in range(int(rng.integers(1, 4)))
        )
        filler = TemplateFiller(forbidden=forbidden, base_alphabet_size=base)
        holes = int(rng.integers(1, 7))
        fixed = int(rng.integers(0, 5))
        template = [FREE] * holes + [int(c) for c in rng.integers(base, size=fixed)]
        rng.shuffle(template)
        return filler, template

    def test_an_ngram_appears_at_its_exact_fiber_rate(self):
        rng = np.random.default_rng(0)
        tested = 0
        for trial in range(self.trials):
            filler, template = self.random_case(rng)
            legal = legal_fillings(filler, template)
            if len(legal) < 2:
                continue
            source = legal[int(rng.integers(len(legal)))]
            n = int(rng.integers(1, min(3, len(source)) + 1))
            at = int(rng.integers(len(source) - n + 1))
            gram = source[at : at + n]

            expected = np.mean([contains(one, gram) for one in legal])
            drawn = filler.fill_many(
                [template] * self.draws,
                [np.random.default_rng(i) for i in range(self.draws)],
            )
            seen = np.mean([contains(tuple(one), gram) for one in drawn])

            with self.subTest(trial=trial, forbidden=filler.forbidden, gram=gram):
                if expected in (0.0, 1.0):
                    self.assertEqual(seen, expected)
                else:
                    spread = (expected * (1 - expected) / self.draws) ** 0.5
                    self.assertLess(abs(seen - expected), self.sigmas * spread)
            tested += 1
        self.assertGreater(tested, self.trials // 2, "too many degenerate templates")


if __name__ == "__main__":
    unittest.main()
