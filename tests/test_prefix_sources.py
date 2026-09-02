"""Sources of prefixes, one per population.

Each round hands the next one a source per population instead of the prefixes
themselves, so what a later round needs more of it can draw more of.
"""

import unittest

from automata.fa.dfa import DFA

from orthogonal_dfa.l_star.leaf_population import LeafPopulation
from orthogonal_dfa.l_star.prefix_sources import (
    ATTEMPTS_PER_PREFIX,
    WANTED,
    IndecisiveSource,
    StateSource,
    collect,
)
from orthogonal_dfa.l_star.sampler import UniformSampler


class _Counted:
    """Yields a prefix ``rate`` of the time, counting how often it is asked."""

    label = "counted"

    def __init__(self, rate, total=10**6):
        self.rate = rate
        self.calls = 0
        self._total = total

    def draw(self):
        self.calls += 1
        keep = (self.calls * self.rate) // 1 - ((self.calls - 1) * self.rate) // 1
        return bytes([self.calls // 256, self.calls % 256]) if keep else None


class TestGivingUpOnASource(unittest.TestCase):
    def test_a_source_that_yields_is_collected(self):
        source = _Counted(1.0)
        held = collect(source, wanted=20, attempts_per=5)
        self.assertEqual(len(held), 20)
        self.assertEqual(source.calls, 20)

    def test_a_source_that_cannot_deliver_is_given_up_on(self):
        # One in fifty, against a budget of five per prefix wanted.
        source = _Counted(0.02)
        self.assertIsNone(collect(source, wanted=20, attempts_per=5))
        self.assertEqual(source.calls, 100)

    def test_what_survives_is_cheap_to_ask_again(self):
        # Surviving the budget means yielding at least one draw in
        # ``attempts_per``, which is what makes a later, larger ask affordable.
        source = _Counted(1 / ATTEMPTS_PER_PREFIX)
        held = collect(source, wanted=WANTED)
        self.assertIsNotNone(held)
        self.assertLessEqual(source.calls, WANTED * ATTEMPTS_PER_PREFIX)

    def test_duplicates_do_not_count_toward_the_ask(self):
        class OneString:
            def draw(self):
                return bytes([7])

        self.assertIsNone(collect(OneString(), wanted=3, attempts_per=5))


class TestTheIndecisiveSource(unittest.TestCase):
    def test_it_serves_what_it_was_given(self):
        source = IndecisiveSource([bytes([1]), bytes([2])])
        self.assertEqual({source.draw(), source.draw()}, {bytes([1]), bytes([2])})

    def test_it_serves_each_string_once(self):
        source = IndecisiveSource([bytes([1])])
        self.assertEqual(source.draw(), bytes([1]))
        self.assertIsNone(source.draw())

    def test_it_takes_what_other_sources_could_not_place(self):
        source = IndecisiveSource()
        self.assertIsNone(source.draw())
        source.offer(bytes([3]))
        self.assertEqual(source.draw(), bytes([3]))

    def test_it_refills_when_it_runs_out(self):
        drawn = []

        def refill():
            drawn.append(len(drawn))
            return bytes([9])

        source = IndecisiveSource(refill=refill)
        self.assertEqual(source.draw(), bytes([9]))
        self.assertEqual(drawn, [0])


class _Tree:
    """One split at the root: the accept child is leaf 1, the reject child 0."""

    def midfix_at(self, path):
        assert path == (), path
        return b""

    def path_of(self, leaf):
        return (leaf == 1,)


class _Resolver:
    def __init__(self, population):
        self.population = population
        self.tree = _Tree()


class _Pst:
    alphabet_size = 2
    rng = None

    def __init__(self, length):
        self.sampler = UniformSampler(length)


#: State 1 loops on itself and nothing enters it, so no string of any length
#: reaches it and there is nothing for the hypothesis to aim.
_UNREACHABLE = DFA(
    states={0, 1},
    input_symbols={0, 1},
    transitions={0: {0: 0, 1: 0}, 1: {0: 1, 1: 1}},
    initial_state=0,
    final_states={1},
)


class TestAStateSourceServesWhatIsAlreadyThere(unittest.TestCase):
    """Aiming is how a leaf nothing has reached gets its first prefixes, not how
    it gets every prefix.  A leaf the population already rests strings at can be
    read without drawing anything."""

    def _source(self, resting):
        population = LeafPopulation(
            _Tree(), lambda strings, midfix: [True] * len(strings)
        )
        for prefix in resting:
            population.add(prefix, at=(True,))
        return StateSource(_Pst(2), _Resolver(population), _UNREACHABLE, 1)

    def test_a_leaf_with_members_yields_them_though_nothing_can_be_aimed(self):
        resting = [bytes([1, i]) for i in range(20)]
        drawn = collect(self._source(resting), wanted=20)
        self.assertIsNotNone(drawn)
        self.assertEqual(sorted(drawn), sorted(resting))

    def test_a_leaf_short_of_what_is_wanted_is_given_up_on(self):
        # Nothing to aim and too few resting: the population is not one to hold
        # to a rate, which is what the indecisive strings are for.
        self.assertIsNone(collect(self._source([bytes([1, 0])]), wanted=20))

    def test_each_member_is_served_once(self):
        source = self._source([bytes([1, 0]), bytes([1, 1])])
        self.assertEqual(len([x for x in (source.draw(), source.draw()) if x]), 2)
        self.assertIsNone(source.draw())


if __name__ == "__main__":
    unittest.main()
