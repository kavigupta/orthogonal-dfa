"""An edge into a badly read state is held apart, rolled over, or dropped."""

# The tests drive the round's edge judging directly.
# pylint: disable=protected-access

import itertools
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from orthogonal_dfa.l_star import counterexample_synthesis as cs
from orthogonal_dfa.l_star import edge_chains as ec
from orthogonal_dfa.l_star.edge_chains import (
    DROP,
    PROMOTE,
    ROLL_OVER,
    UNDECIDED_EDGE,
    EdgeChain,
    edge_verdict,
    separating_reads,
    undecided_measure,
)
from orthogonal_dfa.l_star.prefix_populations import PoolState
from orthogonal_dfa.l_star.prefix_sources import PrefixSource


class _Sifter:
    """Places a string unless its last letter is in ``undecided``; one read each."""

    def __init__(self, undecided):
        self.undecided = undecided
        self.reads = 0

    def sift_and_boundary(self, seq):
        self.reads += 1
        if seq and seq[-1] in self.undecided:
            return None, seq + b"?"
        return 0, None


class _EveryFiftieth(_Sifter):
    """Undecided on every fiftieth string ending in 1."""

    def __init__(self):
        super().__init__(set())
        self._ones = 0

    def sift_and_boundary(self, seq):
        self.reads += 1
        if seq and seq[-1] == 1:
            self._ones += 1
            if self._ones % 50 == 0:
                return None, seq + b"?"
        return 0, None


class _Counting:
    """A source drawing fresh strings 0, 1, 2, ..., each ending in 0."""

    def __init__(self):
        self._next = itertools.count()

    def draw(self):
        return next(self._next).to_bytes(4, "big") + b"\x00"


class TestAChainKeepsWhatItsRoundCouldNotPlace(unittest.TestCase):
    def test_it_keeps_a_draw_only_when_its_extension_is_undecided(self):
        parent = SimpleNamespace(draw=iter([b"x", b"y"]).__next__)
        sifter = SimpleNamespace(
            reads=0,
            sift_and_boundary=lambda seq: (None, b"") if seq == b"y\x01" else (0, None),
        )
        chain = EdgeChain(
            parent,
            b"\x01",
            undecided_measure(sifter, b"\x01"),
            kind=UNDECIDED_EDGE,
            used=frozenset(),
            rate=0.01,
        )

        self.assertFalse(chain.attempt_draw())
        self.assertTrue(chain.attempt_draw())
        self.assertEqual([b"y"], chain.found())


class TestThePrefixRootDrawsWhatReadsPassThrough(unittest.TestCase):
    def test_it_draws_a_prefix_of_a_sampler_draw(self):
        rng = np.random.default_rng(0)
        pst = SimpleNamespace(
            rng=rng,
            alphabet_size=2,
            sampler=SimpleNamespace(sample=lambda rng, alphabet_size: b"abcdefgh"),
        )
        drawn = {PrefixSource(pst).draw() for _ in range(200)}

        self.assertTrue(all(b"abcdefgh".startswith(d) and len(d) < 8 for d in drawn))
        self.assertGreater(len(drawn), 4)


class TestAnEdgeTestStopsWhereItsRatesSeparate(unittest.TestCase):
    def test_far_apart_rates_need_fewer_reads_than_close_ones(self):
        far = separating_reads(0.001, 0.03, 1e-4)
        close = separating_reads(0.02, 0.03, 1e-4)
        self.assertLess(far, close)

    def test_a_rate_and_one_above_it_are_kept_a_rise_apart(self):
        self.assertEqual(
            separating_reads(0.03, 0.03, 1e-4), separating_reads(0.02, 0.03, 1e-4)
        )


class TestALinkReadsRowsNoEarlierLinkRead(unittest.TestCase):
    def _pst(self, group):
        suffixes = {0: b"", 1: b"a", 2: b"b", 3: b"c", 4: b"d", 5: b"e"}
        return SimpleNamespace(
            decision_boundary=0.5,
            config=SimpleNamespace(min_signal_strength=0.3),
            suffix_group=group,
            table=SimpleNamespace(suffix=suffixes.__getitem__),
        )

    def _rows(self, group, vs, used, smallest=3):
        with mock.patch.object(
            ec, "read_rates", return_value=(0.01, 0.01)
        ), mock.patch.object(
            ec, "smallest_readable_family", return_value=smallest
        ), mock.patch.object(
            ec,
            "readable_size_and_margin",
            side_effect=lambda s, b, have, sm, r: (have, 0.1),
        ), mock.patch.object(
            ec, "SuffixFamily", side_effect=lambda pst, rows: SimpleNamespace(rows=rows)
        ):
            found = ec.fresh_sifter(self._pst(group), None, vs, used)
        return None if found is None else found[0].family.rows

    def test_it_tops_up_from_the_group_past_the_family_without_the_empty_suffix(self):
        self.assertEqual([2, 3, 4], self._rows([0, 1, 2, 3, 4], vs=[0, 1, 2], used={1}))

    def test_it_waits_when_the_group_runs_out(self):
        self.assertIsNone(self._rows([0, 1, 2, 3], vs=[0, 1, 2], used={1, 2}))


class TestOneDrawServesEveryEdgeOutOfASource(unittest.TestCase):
    def test_each_draw_is_read_by_every_open_test(self):
        source = _Counting()
        seen = {b"a": [], b"b": []}

        def measure(letter):
            def read(drawn):
                seen[letter].append(drawn)
                return False, 1, []

            return read

        tests = [
            ec.EdgeTest(
                measure(letter),
                promote_above=0.5,
                keep_above=0.1,
                failure_prob=0.01,
                cap=20,
            )
            for letter in (b"a", b"b")
        ]
        ec.judge_edges(source, tests)

        self.assertEqual(seen[b"a"], seen[b"b"])
        self.assertTrue(all(test.verdict == DROP for test in tests))


class TestAHitlessDisagreementEdgeStopsOnceItCouldHaveShownOne(unittest.TestCase):
    def test_it_drops_at_the_detecting_count(self):
        cap = ec.detecting_reads(0.1, 1e-4)
        test = ec.EdgeTest(
            lambda drawn: (False, 1, []),
            promote_above=0.1,
            keep_above=1.5e-6,
            failure_prob=1e-4,
            cap=cap,
        )
        ec.judge_edges(_Counting(), [test])

        self.assertEqual((DROP, cap), (test.verdict, test.weight))
        self.assertLess(cap, 100)


class TestAnEdgeIsJudgedAgainstTwoRates(unittest.TestCase):
    def _verdict(self, undecided, reads):
        return edge_verdict(
            undecided,
            reads,
            promote_above=0.03,
            keep_above=0.015,
            failure_prob=1e-4,
            final=False,
        )

    def test_far_above_the_promotion_rate_promotes(self):
        self.assertEqual(PROMOTE, self._verdict(30, 100))

    def test_far_below_the_keeping_rate_drops(self):
        self.assertEqual(DROP, self._verdict(0, 2000))

    def test_between_the_two_rolls_over(self):
        self.assertEqual(ROLL_OVER, self._verdict(220, 10000))

    def test_too_few_reads_to_say_waits(self):
        self.assertIsNone(self._verdict(1, 20))


class TestARoundSortsItsEdges(unittest.TestCase):
    def _round(self, state, sifter, landing=lambda seq: 0):
        pst = SimpleNamespace(
            acceptable_fnr=0.01,
            alphabet_size=2,
            rng=np.random.default_rng(0),
            decision_boundary=0.5,
            sampler=SimpleNamespace(sample=lambda rng, alphabet_size: bytes(8)),
        )
        # A clean rate of about 0.011 per read, so edges are kept above ~0.017.
        resolver = SimpleNamespace(
            sifter=sifter,
            family=SimpleNamespace(vs=[1, 2], mean=lambda seq, midfix: 0.0),
            tree=SimpleNamespace(classify=lambda seq, decide: landing(seq)),
            unchecked_quiet_probes=10,
            quiet_reads=999,
        )
        cs._split_edges(
            pst,
            resolver,
            SimpleNamespace(transitions={0: {0: 0, 1: 0}}),
            state,
            acc_threshold=0.98,
        )

    def _state(self, chains=()):
        state = PoolState([])
        state.held[("state", 0)] = []
        state.sources[("state", 0)] = _Counting()
        state.chains = list(chains)
        return state

    def _chain(self):
        return EdgeChain(
            _Counting(),
            b"\x01",
            undecided_measure(_Sifter({1}), b"\x01"),
            kind=UNDECIDED_EDGE,
            used=frozenset({1}),
            rate=0.02,
        )

    def test_an_edge_into_a_badly_read_state_becomes_a_population(self):
        state = self._state()

        self._round(state, _Sifter({1}))

        self.assertTrue(state.held[("edge", 1)])
        self.assertEqual([], state.chains)

    def test_an_edge_between_the_two_rates_rolls_over_onto_its_source(self):
        state = self._state()

        self._round(state, _EveryFiftieth())

        self.assertNotIn(("edge", 1), state.held)
        rolled = [c for c in state.chains if c.parent is state.sources[("state", 0)]]
        self.assertEqual(
            [(b"\x01", UNDECIDED_EDGE)], [(c.letter, c.kind) for c in rolled]
        )

    def test_a_draw_whose_extension_lands_off_its_edge_makes_a_population_of_draws(
        self,
    ):
        state = self._state()

        # Strings ending in 1 land at leaf 1, while the hypothesis keeps them at 0.
        self._round(
            state, _Sifter(set()), landing=lambda seq: 1 if seq.endswith(b"\x01") else 0
        )

        promoted = [label for label in state.held if label[0] == "edge"]
        self.assertTrue(promoted)
        for label in promoted:
            self.assertTrue(all(s.endswith(b"\x00") for s in state.held[label]))

    def test_a_chain_that_does_not_rise_on_fresh_suffixes_is_forgotten(self):
        chain = self._chain()
        state = self._state([chain])

        with mock.patch.object(
            cs, "fresh_sifter", return_value=(_Sifter(set()), frozenset({2}))
        ):
            self._round(state, _Sifter(set()))

        self.assertEqual([], state.chains)

    def test_a_chain_with_no_fresh_suffixes_left_waits_unchanged(self):
        chain = self._chain()
        state = self._state([chain])

        with mock.patch.object(cs, "fresh_sifter", return_value=None):
            self._round(state, _Sifter(set()))

        self.assertEqual([chain], state.chains)


if __name__ == "__main__":
    unittest.main()
