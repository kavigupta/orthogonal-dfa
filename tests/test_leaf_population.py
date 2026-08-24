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
        pop = LeafPopulation(_StubTree(), classify, chunk=16)
        for s in (b"\x01\x00", b"\x01\x01", b"\x00\x01", b"\x00\x00"):
            pop.add(s)
        self.assertEqual(sorted(pop.members((True,), 10)), [b"\x01\x00", b"\x01\x01"])
        self.assertEqual(pop.members((False, True), 10), [b"\x00\x01"])
        self.assertEqual(pop.members((False, False), 10), [b"\x00\x00"])

    def test_count_bounds_the_pull(self):
        # 100 strings all route to leaf (True,); asking for 3 pulls one chunk, not all.
        classify, calls = _classifier()
        pop = LeafPopulation(_StubTree(), classify, chunk=16)
        for i in range(100):
            pop.add(bytes([1, i % 2]))
        got = pop.members((True,), 3)
        self.assertEqual(len(got), 3)
        self.assertLessEqual(calls["strings"], 16)

    def test_sibling_strings_rest_and_are_reused(self):
        # Querying one leaf drains the root once; querying the sibling must not
        # reclassify the root -- the sibling's strings are already resting below it.
        classify, calls = _classifier()
        pop = LeafPopulation(_StubTree(), classify, chunk=64)
        for s in (b"\x01\x00", b"\x00\x01", b"\x01\x01", b"\x00\x00"):
            pop.add(s)
        pop.members((True,), 10)
        self.assertEqual(pop.members((False, True), 10), [b"\x00\x01"])
        self.assertEqual(calls["batches"], 2)  # root once, then (False,) once

    def test_add_at_a_known_leaf_skips_classification(self):
        classify, calls = _classifier()
        pop = LeafPopulation(_StubTree(), classify, chunk=16)
        pop.add(b"\x07\x07", at=(True,))
        self.assertEqual(pop.members((True,), 10), [b"\x07\x07"])
        self.assertEqual(calls["batches"], 0)

    def test_cold_query_of_a_grandchild_cascades(self):
        # Querying a grandchild with everything still at the root pulls root ->
        # (False,) -> (False, True) in one call.
        classify, _ = _classifier()
        pop = LeafPopulation(_StubTree(), classify, chunk=16)
        for s in (b"\x01\x00", b"\x00\x01", b"\x01\x01", b"\x00\x00", b"\x00\x01"):
            pop.add(s)
        self.assertEqual(
            sorted(pop.members((False, True), 10)), [b"\x00\x01", b"\x00\x01"]
        )

    def test_exhausted_ancestors_return_what_is_there(self):
        classify, _ = _classifier()
        pop = LeafPopulation(_StubTree(), classify, chunk=16)
        for s in (b"\x01\x00", b"\x01\x01"):
            pop.add(s)
        # Only two strings reach (True,); asking for more just returns those two.
        self.assertEqual(sorted(pop.members((True,), 50)), [b"\x01\x00", b"\x01\x01"])

    def test_representative_is_the_shortest_member(self):
        classify, _ = _classifier()
        pop = LeafPopulation(_StubTree(), classify, chunk=16)
        for s in (b"\x03\x03\x03", b"\x07", b"\x01\x02"):
            pop.add(s, at=(True,))
        self.assertEqual(pop.representative((True,), 10), b"\x07")

    def test_representative_is_none_when_no_members_reach_the_leaf(self):
        classify, _ = _classifier()
        pop = LeafPopulation(_StubTree(), classify, chunk=16)
        self.assertIsNone(pop.representative((True,), 10))


if __name__ == "__main__":
    unittest.main()
