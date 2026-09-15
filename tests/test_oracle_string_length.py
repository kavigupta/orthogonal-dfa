"""``string_length`` as part of what an oracle answers, rather than an attribute
some of them happen to carry.

``SetDifferenceOracle`` reads it off the oracles it is given, so anything standing
where one of those stood has to answer it -- and anything that cannot has to say
so rather than be missing it.
"""

import unittest

from orthogonal_dfa.l_star.examples.set_difference import SetDifferenceOracle
from orthogonal_dfa.l_star.structures import Oracle


class _AnyLength(Oracle):
    """Answers about strings of every length, the way a parity or a regex does."""

    alphabet_size = 2

    def membership_query(self, string):
        return sum(string) % 2 == 0


class _FixedLength(Oracle):
    """Answers about strings of one length, the way a model over a window does."""

    alphabet_size = 2

    def __init__(self, length):
        self._length = length

    @property
    def string_length(self):
        return self._length

    def membership_query(self, string):
        return sum(string) % 2 == 0


class TestStringLength(unittest.TestCase):
    def test_an_oracle_over_one_length_answers_it(self):
        self.assertEqual(_FixedLength(12).string_length, 12)

    def test_an_oracle_over_any_length_says_so(self):
        # Rather than being missing it, which reads as an implementation slip
        # wherever it surfaces.
        with self.assertRaises(NotImplementedError):
            _ = _AnyLength().string_length

    def test_the_message_names_the_oracle(self):
        with self.assertRaises(NotImplementedError) as raised:
            _ = _AnyLength().string_length
        self.assertIn("_AnyLength", str(raised.exception))

    def test_a_difference_takes_the_length_of_what_it_is_given(self):
        difference = SetDifferenceOracle(_FixedLength(9), _FixedLength(9))
        self.assertEqual(difference.string_length, 9)

    def test_a_difference_over_any_length_says_so_too(self):
        difference = SetDifferenceOracle(_AnyLength(), _AnyLength())
        with self.assertRaises(NotImplementedError):
            _ = difference.string_length


if __name__ == "__main__":
    unittest.main()
