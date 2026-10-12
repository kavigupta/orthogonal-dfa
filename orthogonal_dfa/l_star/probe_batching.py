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
    c = search.config
    block, done = 1, 0
    while done < c.max_probes:
        version = search.version
        settle = next(
            (m for m in range(1, MAX_BLOCK) if search.settles_below(search.n + m)),
            MAX_BLOCK,
        )
        size = min(block, c.max_probes - done, settle)
        xs = [draw() for _ in range(size)]
        firsts = [
            (x, e)
            for x in xs
            for e in [ends(search.cut, search.tree, search.edges, c.k, x)]
            if e
        ]
        _warm(search, warm, [x[: e[0]] for x, e in firsts])
        decided = [(x, e) for x, e in firsts if search.sift(x[: e[0]]) is not None]
        _warm(search, warm, [x[: e[1]] for x, e in decided if len(e) == 2])
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
