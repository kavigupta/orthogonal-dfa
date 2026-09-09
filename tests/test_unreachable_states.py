"""A state nothing reaches is not a state the round came up short on.

Coming up short is the round's signal that more prefixes would say more, and it
is what holds the saturation check -- and so the stall detector -- open.  A state
no string of the sampler's length arrives at answers to none of that: waiting for
it waits for a draw that cannot come.
"""

import unittest
from types import SimpleNamespace

from automata.fa.dfa import DFA

from orthogonal_dfa.l_star.counterexample_synthesis import (
    _short_states,
    _unreachable_states,
)

#: State 1 loops on itself and nothing enters it, so no string of any length
#: reaches it.  State 2 is entered only on a ``1``.
_DFA = DFA(
    states={0, 1, 2},
    input_symbols={0, 1},
    transitions={0: {0: 0, 1: 2}, 1: {0: 1, 1: 1}, 2: {0: 2, 1: 2}},
    initial_state=0,
    final_states={2},
)

_EVEN = [0.5, 0.5]
#: Never draws a ``1``, which is the only way into state 2.
_ZEROS = [1.0, 0.0]


def _resolver(members):
    return SimpleNamespace(
        num_states=len(members),
        tree=SimpleNamespace(path_of=lambda leaf: (leaf,)),
        population=SimpleNamespace(members=lambda path, _n: members[path[0]]),
    )


class TestWhichStatesNothingReaches(unittest.TestCase):
    def test_a_state_nothing_enters_is_unreachable(self):
        found = _unreachable_states(_DFA, range(3), length=4, weights=_EVEN)
        self.assertEqual(found, {1})

    def test_the_sampler_weights_decide_it_too(self):
        # State 2 has a path, but not one these weights would ever draw.
        found = _unreachable_states(_DFA, range(3), length=4, weights=_ZEROS)
        self.assertEqual(found, {1, 2})

    def test_a_length_can_put_a_state_out_of_reach(self):
        # Nothing reaches anywhere but the initial state in zero steps.
        found = _unreachable_states(_DFA, range(3), length=0, weights=_EVEN)
        self.assertEqual(found, {1, 2})


class TestShortStates(unittest.TestCase):
    def test_a_state_short_of_its_members_is_short(self):
        _, short = _short_states(_resolver({0: [b"a"], 1: [b"b", b"c"]}), 2, set())
        self.assertEqual(short, [0])

    def test_an_unreachable_state_is_not_short_however_few_it_holds(self):
        _, short = _short_states(_resolver({0: [b"a"], 1: []}), 2, {1})
        self.assertEqual(short, [0], "and the reachable one still is")

    def test_every_short_state_being_unreachable_leaves_nothing_short(self):
        # Which is what lets the round report a full complement and the
        # saturation check go on to ask whether the tree has settled.
        held, short = _short_states(_resolver({0: [], 1: []}), 2, {0, 1})
        self.assertEqual(short, [])
        self.assertEqual(set(held), {0, 1}, "they are still populations")


if __name__ == "__main__":
    unittest.main()
