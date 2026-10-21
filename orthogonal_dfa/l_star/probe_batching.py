"""Optimisation only: pre-reads, in batches, what a counterexample search's next probes will read."""

from typing import Callable, Iterator, List

from .counterexample_probe import ends, first_decided
from .counterexample_search import CounterexampleSearch
from .hypothesis_dfa import HypothesisDFA
from .round_end import succeeds_at

MAX_BLOCK = 64


def warmed_draws(
    search: CounterexampleSearch,
    draw: Callable[[], bytes],
    warm: Callable[[List[bytes]], None],
) -> Iterator[bytes]:
    # Not trajectory-neutral: a block is drawn ahead, and one that sees a change
    # has read for its later probes under the old hypothesis.
    config, hypothesis = search.config, search.hypothesis
    block, done = 1, 0
    while done < config.max_probes:
        version = hypothesis.version
        until_success = next(
            (
                more
                for more in range(1, MAX_BLOCK)
                if succeeds_at(search.counts, config, search.counts.probes + more)
            ),
            MAX_BLOCK,
        )
        size = min(block, config.max_probes - done, until_success)
        xs = [draw() for _ in range(size)]
        with_ends = [
            (x, positions)
            for x in xs
            for positions in [ends(hypothesis, first_decided(hypothesis, x), x)]
            if positions
        ]
        _warm(hypothesis, warm, [x[: positions[0]] for x, positions in with_ends])
        decided_first = [
            (x, positions)
            for x, positions in with_ends
            if hypothesis.sift(x[: positions[0]]) is not None
        ]
        _warm(
            hypothesis,
            warm,
            [
                x[: positions[1]]
                for x, positions in decided_first
                if len(positions) == 2
            ],
        )
        yield from xs
        done += size
        block = min(2 * block, MAX_BLOCK) if hypothesis.version == version else 1


def _warm(
    hypothesis: HypothesisDFA, warm: Callable[[List[bytes]], None], seqs: List[bytes]
):
    def read(pairs):
        strings = [seq + midfix for seq, midfix in pairs]
        warm(strings)
        return [hypothesis.cut(z) for z in strings]

    hypothesis.tree.classify_many(seqs, read)
