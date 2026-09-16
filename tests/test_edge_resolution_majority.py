"""Edge resolution votes a heterogeneous leaf; it does not take the first member.

A leaf is often not homogeneous: its members' ``member + c`` successors sift to
more than one target -- a real oracle's answers do not agree perfectly, so a
structured minority of members route the edge elsewhere.  Taking the *first*
decisively-sifting member then lets whichever member the population happens to
yield first capture the whole edge, however few members agree with it; over a
DFA's many such edges at least one is wrong for almost any member ordering.
``decisive_target`` instead polls the members and takes the majority target, so
the edge goes where most members send it, regardless of their order.
"""
import unittest

import numpy as np

from orthogonal_dfa.l_star.edge_resolver import EdgeResolver
from orthogonal_dfa.l_star.partial_dfa import PartialDFA


class _StubTree:
    def path_of(self, state):
        return (state,)


class _HeterogeneousSifter:
    """Members are ``bytes([state, idx])``.  For state ``s`` on the poisoned
    symbol the first ``minority`` members (by idx) sift to a WRONG target and the
    rest to the TRUE target ``true_edge[s]`` -- the structured split a real
    oracle leaves at a leaf.  Every member sifts decisively, so the leaf is
    heterogeneous, not indecisive."""

    def __init__(self, true_edge, wrong_edge, poisoned, minority):
        self.tree = _StubTree()
        self._true = true_edge
        self._wrong = wrong_edge
        self._poisoned = poisoned
        self._minority = minority

    def prefill(self, probes):
        pass

    def sift_and_boundary(self, seq):
        state, idx, c = seq[0], seq[1], seq[2]
        if c != self._poisoned:
            return self._true[state], None
        return (self._wrong[state] if idx < self._minority else self._true[state]), None


class _StubPopulation:
    """The leaf of state ``s`` holds its members in the order ``order[s]``."""

    def __init__(self, order):
        self._order = order

    def representative(self, path, _limit):
        return bytes([path[0], 0])

    def members(self, path, _limit):
        s = path[0]
        return [bytes([s, idx]) for idx in self._order[s]]


def _first_member_target(resolver, state, c):
    """What the old resolver did: the first decisively-sifting member's target."""
    for member in resolver.leaf_members(state):
        target, _ = resolver.sifter.sift_and_boundary(member + bytes([c]))
        if target is not None:
            return target
    return None


class TestEdgeResolutionMajority(unittest.TestCase):
    K = 12           # states, each with one heterogeneous poisoned edge
    PER_STATE = 60   # members per leaf (> the internal sample size, so it samples)
    MINORITY = 9     # 15% route the poisoned edge to the wrong target
    POISON = 0

    def setUp(self):
        self.true_edge = {s: s for s in range(self.K)}          # self-loop is correct
        self.wrong_edge = {s: (s + 1) % self.K for s in range(self.K)}

    def _resolver(self, order):
        return EdgeResolver(
            PartialDFA(alphabet_size=2, num_states=self.K),
            _HeterogeneousSifter(self.true_edge, self.wrong_edge, self.POISON, self.MINORITY),
            set(),
            population=_StubPopulation(order),
        )

    def _resolve_all(self, resolver):
        return {s: resolver.decisive_target(s, self.POISON)[0] for s in range(self.K)}

    def test_votes_the_majority_where_first_member_would_take_the_minority(self):
        # Minority members (idx 0..8) come first, so first-member would take the
        # wrong target on every state; the majority (idx 9..59) is the true one.
        order = {s: list(range(self.PER_STATE)) for s in range(self.K)}
        resolver = self._resolver(order)
        for s in range(self.K):
            self.assertEqual(_first_member_target(resolver, s, self.POISON), self.wrong_edge[s])
        self.assertEqual(self._resolve_all(resolver), dict(self.true_edge))

    def test_majority_is_right_regardless_of_member_order(self):
        # The defect is that first-member fails for almost any ordering, not one
        # unlucky one: over K edges with a p=0.15 minority each, 1 - (1-p)**K ~ 86%.
        # The majority resolver is right for every ordering.
        rng = np.random.default_rng(0)
        trials, first_fail, majority_fail = 200, 0, 0
        for _ in range(trials):
            order = {s: list(rng.permutation(self.PER_STATE)) for s in range(self.K)}
            resolver = self._resolver(order)
            first = {s: _first_member_target(resolver, s, self.POISON) for s in range(self.K)}
            if first != dict(self.true_edge):
                first_fail += 1
            if self._resolve_all(resolver) != dict(self.true_edge):
                majority_fail += 1
        self.assertGreater(first_fail / trials, 0.80)   # first-member: fails almost always
        self.assertEqual(majority_fail, 0)              # majority: never


if __name__ == "__main__":
    unittest.main()
