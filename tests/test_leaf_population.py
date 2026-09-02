import unittest

from orthogonal_dfa.l_star.leaf_population import LeafPopulation


class _StubTree:
    """root splits on bit 0; its reject child splits on bit 1.

    ()        -> bit 0 : True -> leaf (True,)
    (False,)  -> bit 1 : True -> leaf (False, True), False -> (False, False)
    """

    _midfix = {(): b"", (False,): b"m"}

    def midfix_at(self, path):
        return self._midfix[path]


def _population(classify, **kwargs):
    """A population over the stub tree, dropping harvested strings by default.

    Most of these tests are about where strings come to rest, not about what
    fails to; the ones that care pass their own ``harvest``.
    """
    kwargs.setdefault("harvest", lambda _string: None)
    return LeafPopulation(_StubTree(), classify, **kwargs)


def _classifier():
    """A batched classifier that routes by a bit of the string, counting calls."""
    calls = {"batches": 0, "strings": 0}

    def classify(strings, midfix):
        calls["batches"] += 1
        calls["strings"] += len(strings)
        idx = 0 if midfix == b"" else 1
        return [bool(s[idx]) for s in strings]

    return classify, calls


class TestLeafPopulation(unittest.TestCase):
    def test_pulls_matching_strings_to_each_leaf(self):
        classify, _ = _classifier()
        pop = _population(classify, chunk=16)
        for s in (bytes([1, 0]), bytes([1, 1]), bytes([0, 1]), bytes([0, 0])):
            pop.add(s)
        self.assertEqual(
            sorted(pop.members((True,), 10)), [bytes([1, 0]), bytes([1, 1])]
        )
        self.assertEqual(pop.members((False, True), 10), [bytes([0, 1])])
        self.assertEqual(pop.members((False, False), 10), [bytes([0, 0])])

    def test_count_bounds_the_pull(self):
        # 100 strings all route to leaf (True,); asking for 3 pulls one chunk, not all.
        classify, calls = _classifier()
        pop = _population(classify, chunk=16)
        for i in range(100):
            pop.add(bytes([1, i]))
        got = pop.members((True,), 3)
        self.assertEqual(len(got), 3)
        self.assertLessEqual(calls["strings"], 16)

    def test_sibling_strings_rest_and_are_reused(self):
        # Querying one leaf drains the root once; querying the sibling must not
        # reclassify the root -- the sibling's strings are already resting below it.
        classify, calls = _classifier()
        pop = _population(classify, chunk=64)
        for s in (bytes([1, 0]), bytes([0, 1]), bytes([1, 1]), bytes([0, 0])):
            pop.add(s)
        pop.members((True,), 10)
        self.assertEqual(pop.members((False, True), 10), [bytes([0, 1])])
        self.assertEqual(calls["batches"], 2)  # root once, then (False,) once

    def test_add_at_a_known_leaf_skips_classification(self):
        classify, calls = _classifier()
        pop = _population(classify, chunk=16)
        pop.add(bytes([7, 7]), at=(True,))
        self.assertEqual(pop.members((True,), 10), [bytes([7, 7])])
        self.assertEqual(calls["batches"], 0)

    def test_cold_query_of_a_grandchild_cascades(self):
        # Querying a grandchild with everything still at the root pulls root ->
        # (False,) -> (False, True) in one call.
        classify, _ = _classifier()
        pop = _population(classify, chunk=16)
        for s in (bytes([1, 0]), bytes([0, 1]), bytes([1, 1]), bytes([0, 0])):
            pop.add(s)
        self.assertEqual(pop.members((False, True), 10), [bytes([0, 1])])

    def test_a_string_added_twice_is_one_member(self):
        classify, _ = _classifier()
        pop = _population(classify, chunk=16)
        for _ in range(5):
            pop.add(bytes([0, 1]))
        pop.add(bytes([0, 1]), at=(False, True))
        self.assertEqual(pop.members((False, True), 10), [bytes([0, 1])])

    def test_an_indecisive_string_can_be_added_again(self):
        pop = _population(lambda ss, m: [None] * len(ss), chunk=16)
        pop.add(bytes([1, 1]))
        self.assertEqual(pop.members((True,), 10), [])
        pop.add(bytes([1, 1]), at=(True,))
        self.assertEqual(pop.members((True,), 10), [bytes([1, 1])])

    def test_seeding_a_held_string_at_a_leaf_moves_it_there(self):
        classify, calls = _classifier()
        pop = _population(classify, chunk=16)
        pop.add(bytes([0, 1]))
        pop.add(bytes([0, 1]), at=(False, True))
        self.assertEqual(pop.members((False, True), 10), [bytes([0, 1])])
        self.assertEqual(calls["batches"], 0)
        self.assertEqual(pop.members((), 10), [])

    def test_exhausted_ancestors_return_what_is_there(self):
        classify, _ = _classifier()
        pop = _population(classify, chunk=16)
        for s in (bytes([1, 0]), bytes([1, 1])):
            pop.add(s)
        # Only two strings reach (True,); asking for more just returns those two.
        self.assertEqual(
            sorted(pop.members((True,), 50)), [bytes([1, 0]), bytes([1, 1])]
        )

    def test_representative_is_the_shortest_member(self):
        classify, _ = _classifier()
        pop = _population(classify, chunk=16)
        for s in (bytes([3, 3, 3]), bytes([7]), bytes([1, 2])):
            pop.add(s, at=(True,))
        self.assertEqual(pop.representative((True,), 10), bytes([7]))

    def test_representative_is_none_when_no_members_reach_the_leaf(self):
        classify, _ = _classifier()
        pop = _population(classify, chunk=16)
        self.assertIsNone(pop.representative((True,), 10))


class TestWhatANodeCannotPlace(unittest.TestCase):
    """Sifting reports a string it cannot place so the next family is made to
    resolve it.  A string pushed down here fails the same way."""

    def test_an_indecisive_string_is_harvested(self):
        harvested = []
        pop = _population(
            lambda strings, midfix: [None] * len(strings),
            chunk=16,
            harvest=harvested.append,
        )
        pop.add(bytes([1, 0]))

        self.assertEqual(pop.members((True,), 10), [])
        # ``string + midfix``; the root's midfix is empty.
        self.assertEqual(harvested, [bytes([1, 0])])

    def test_a_placed_string_is_not_harvested(self):
        harvested = []
        classify, _ = _classifier()
        pop = _population(classify, chunk=16, harvest=harvested.append)
        pop.add(bytes([1, 0]))

        self.assertEqual(pop.members((True,), 10), [bytes([1, 0])])
        self.assertEqual(harvested, [])


if __name__ == "__main__":
    unittest.main()
