"""Shared stubs for the direct-L* learner unit tests: a minimal table, an
accept-everything oracle, and a PST namespace pointed at them."""

from types import SimpleNamespace

from orthogonal_dfa.l_star.memoized_oracle import MemoizedOracle


class StubTable:
    prefixes = ()

    def __init__(self, oracle):
        self.memo = MemoizedOracle(oracle)

    def suffix(self, v):
        return bytes([v])


class StubOracle:
    alphabet_size = 2

    def membership_queries(self, strings):
        # Accept everything, so a decisive re-read picks the accept side.
        return [1] * len(strings)


def make_pst():
    oracle = StubOracle()
    return SimpleNamespace(
        alphabet_size=2,
        accept_thresh=0.7,
        reject_thresh=0.3,
        decision_boundary=0.5,
        evidence_margin=0.0,
        table=StubTable(oracle),
        oracle=oracle,
        config=SimpleNamespace(split_pval=0.001, min_signal_strength=0.3),
    )
