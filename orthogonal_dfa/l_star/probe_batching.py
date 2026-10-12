"""Optimisation only: pre-reads, in batches, what a counterexample search's next probes will read."""

from typing import Callable, Iterator, List

from .counterexample_probe import ends
from .counterexample_search import CounterexampleSearch

MAX_BLOCK = 64


def warmed_draws(
    search: CounterexampleSearch,
    draw: Callable[[], bytes],
    warm: Callable[[List[bytes]], None],
) -> Iterator[bytes]:
    # Not trajectory-neutral: a block is drawn ahead, and one that sees a change
    # has read for its later probes under the old hypothesis.
    config = search.config
    block, done = 1, 0
    while done < config.max_probes:
        version = search.version
        until_success = next(
            (
                more
                for more in range(1, MAX_BLOCK)
                if search.succeeds_at(search.stretch_probes + more)
            ),
            MAX_BLOCK,
        )
        size = min(block, config.max_probes - done, until_success)
        xs = [draw() for _ in range(size)]
        with_ends = [
            (x, positions)
            for x in xs
            for positions in [ends(search.cut, search.tree, search.edges, config.k, x)]
            if positions
        ]
        _warm(search, warm, [x[: positions[0]] for x, positions in with_ends])
        first_decided = [
            (x, positions)
            for x, positions in with_ends
            if search.sift(x[: positions[0]]) is not None
        ]
        _warm(
            search,
            warm,
            [
                x[: positions[1]]
                for x, positions in first_decided
                if len(positions) == 2
            ],
        )
        yield from xs
        done += size
        block = min(2 * block, MAX_BLOCK) if search.version == version else 1


def _warm(
    search: CounterexampleSearch, warm: Callable[[List[bytes]], None], seqs: List[bytes]
):
    def read(pairs):
        strings = [seq + midfix for seq, midfix in pairs]
        warm(strings)
        return [search.cut(z) for z in strings]

    search.tree.classify_many(seqs, read)
