"""What every test run here shares.

`setup.cfg` sets one clock for the whole suite, which has to suit the tests that
share a shard.  A slow test has a shard and a job of its own, so it gets the
clock that job allows instead.
"""

import pytest

#: Under the 40 minutes the slow shards are given, leaving room for the install.
SLOW_TIMEOUT_SECONDS = 2100


def pytest_collection_modifyitems(items):
    for item in items:
        if item.get_closest_marker("slow"):
            item.add_marker(pytest.mark.timeout(SLOW_TIMEOUT_SECONDS))


@pytest.fixture(autouse=True)
def _tally_rounds_end_inside_the_proofs_budget(monkeypatch):
    """Every round the learner runs ends without running out of probes or leaves."""
    # pylint: disable=import-outside-toplevel,protected-access
    from orthogonal_dfa.l_star import counterexample_synthesis
    from orthogonal_dfa.l_star.tally_round import OUT_OF_PROBES, TOO_BIG

    tally_round = counterexample_synthesis._tally_round

    def checked(*args, **kwargs):
        hypothesis, ending = tally_round(*args, **kwargs)
        assert ending.kind not in (OUT_OF_PROBES, TOO_BIG), (
            f"a tally round ended {ending.kind}: if this assertion fires, it's "
            "evidence that the gap between reality and the proof is less egregious "
            "than we thought (the proof's huge probe budget, or its leaf cap, is "
            "binding in practice)"
        )
        return hypothesis, ending

    monkeypatch.setattr(counterexample_synthesis, "_tally_round", checked)
