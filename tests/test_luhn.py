import itertools
import unittest

from orthogonal_dfa.l_star import preconditions as P
from orthogonal_dfa.l_star.examples.luhn import luhn_dfa, luhn_valid


class TestLuhn(unittest.TestCase):
    def test_dfa_agrees_with_checksum(self):
        dfa = luhn_dfa()
        for length in range(5):
            for digits in itertools.product(range(10), repeat=length):
                self.assertEqual(dfa.accepts_input(digits), luhn_valid(digits))

    def test_known_numbers(self):
        self.assertTrue(luhn_valid([7, 9, 9, 2, 7, 3, 9, 8, 7, 1, 3]))
        self.assertFalse(luhn_valid([7, 9, 9, 2, 7, 3, 9, 8, 7, 1, 0]))

    def test_minimal(self):
        self.assertEqual(len(luhn_dfa().minify().states), 100)

    def test_preconditions_hold_only_at_even_length(self):
        # every digit permutes the states, and an odd-length suffix moves the
        # accept states {a = 0} onto {b = const}
        self.assertTrue(P.satisfies_preconditions(luhn_dfa(), length=40))
        self.assertFalse(P.satisfies_preconditions(luhn_dfa(), length=39))


if __name__ == "__main__":
    unittest.main()
