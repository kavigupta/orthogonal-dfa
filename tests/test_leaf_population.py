import unittest

from orthogonal_dfa.l_star.decisions import Decisions
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
    kwargs.setdefault("decisions", Decisions())
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
    """A string a node cannot place is reported, not dropped."""

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

    def test_representative_does_not_descend_and_so_does_not_harvest(self):
        # The string is still at the root, and reading a representative must not
        # push it down: that would classify it, find it indecisive, and harvest.
        harvested = []
        pop = _population(
            lambda strings, midfix: [None] * len(strings),
            chunk=16,
            harvest=harvested.append,
        )
        pop.add(bytes([1, 0]))

        self.assertIsNone(pop.representative((True,), 10))
        self.assertEqual(harvested, [])
        # members() does descend, so the same read through it harvests.
        self.assertEqual(pop.members((True,), 10), [])
        self.assertEqual(harvested, [bytes([1, 0])])

    def test_a_placed_string_is_not_harvested(self):
        harvested = []
        classify, _ = _classifier()
        pop = _population(classify, chunk=16, harvest=harvested.append)
        pop.add(bytes([1, 0]))

        self.assertEqual(pop.members((True,), 10), [bytes([1, 0])])
        self.assertEqual(harvested, [])


class TestSettle(unittest.TestCase):
    """Pushing one string toward a leaf, rather than filling the leaf."""

    def test_a_string_already_there_settles_without_classifying(self):
        classify, calls = _classifier()
        pop = _population(classify, chunk=16)
        pop.add(bytes([1, 0]), at=(True,))

        self.assertTrue(pop.settle(bytes([1, 0]), (True,)))
        self.assertEqual(calls["batches"], 0)

    def test_a_string_at_the_root_is_pushed_down_to_it(self):
        classify, _ = _classifier()
        pop = _population(classify, chunk=16)
        pop.add(bytes([1, 0]))

        self.assertTrue(pop.settle(bytes([1, 0]), (True,)))
        self.assertEqual(pop.resting_at(bytes([1, 0])), (True,))

    def test_a_string_that_lands_elsewhere_says_so(self):
        classify, _ = _classifier()
        pop = _population(classify, chunk=16)
        pop.add(bytes([0, 1]))

        self.assertFalse(pop.settle(bytes([0, 1]), (True,)))
        # Left at (False,) rather than pushed on to its own leaf: one step was
        # enough to know it will never reach (True,), and settling is asked
        # about the target, not about where the string finally belongs.
        self.assertEqual(pop.resting_at(bytes([0, 1])), (False,))

    def test_it_stops_at_a_node_the_target_does_not_hang_below(self):
        # Resting at (False, True) with (True,) asked for: pushing further could
        # not reach it, so the answer is where it already is.
        classify, calls = _classifier()
        pop = _population(classify, chunk=16)
        pop.add(bytes([0, 1]))
        pop.members((False, True), 10)
        before = calls["batches"]

        self.assertFalse(pop.settle(bytes([0, 1]), (True,)))
        self.assertEqual(calls["batches"], before, "no further classification")

    def test_a_string_the_population_does_not_hold_does_not_settle(self):
        classify, _ = _classifier()
        pop = _population(classify, chunk=16)

        self.assertFalse(pop.settle(bytes([1, 0]), (True,)))

    def test_a_string_the_node_cannot_place_leaves_and_does_not_settle(self):
        harvested = []
        pop = _population(
            lambda strings, midfix: [None] * len(strings),
            chunk=16,
            harvest=harvested.append,
        )
        pop.add(bytes([1, 0]))

        self.assertFalse(pop.settle(bytes([1, 0]), (True,)))
        self.assertIsNone(pop.resting_at(bytes([1, 0])))
        self.assertEqual(harvested, [bytes([1, 0])])


class TestLength(unittest.TestCase):
    def test_it_counts_every_node(self):
        classify, _ = _classifier()
        pop = _population(classify, chunk=16)
        for s in (bytes([1, 0]), bytes([0, 1]), bytes([1, 1])):
            pop.add(s)
        self.assertEqual(len(pop), 3)

        pop.members((True,), 10)
        self.assertEqual(len(pop), 3, "pushing down moves them, it does not lose them")


if __name__ == "__main__":
    unittest.main()
