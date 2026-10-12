"""Optimisation only: pre-reads, in batches, what a tally round's next probes will read."""

from typing import Callable, Iterator, List

from .tally_probe import ends
from .tally_round import TallyRound

MAX_BLOCK = 64


def warmed_draws(
    tally: TallyRound, draw: Callable[[], bytes], warm: Callable[[List[bytes]], None]
) -> Iterator[bytes]:
    # Not trajectory-neutral: a block is drawn ahead, and one that sees a change
    # has read for its later probes under the old hypothesis.
    c = tally.config
    block, done = 1, 0
    while done < c.max_probes:
        version = tally.version
        settle = next(
            (m for m in range(1, MAX_BLOCK) if tally.settles_below(tally.n + m)),
            MAX_BLOCK,
        )
        size = min(block, c.max_probes - done, settle)
        xs = [draw() for _ in range(size)]
        firsts = [
            (x, e)
            for x in xs
            for e in [ends(tally.cut, tally.tree, tally.edges, c.k, x)]
            if e
        ]
        _warm(tally, warm, [x[: e[0]] for x, e in firsts])
        decided = [(x, e) for x, e in firsts if tally.sift(x[: e[0]]) is not None]
        _warm(tally, warm, [x[: e[1]] for x, e in decided if len(e) == 2])
        yield from xs
        done += size
        block = min(2 * block, MAX_BLOCK) if tally.version == version else 1


def _warm(tally: TallyRound, warm: Callable[[List[bytes]], None], seqs: List[bytes]):
    def read(pairs):
        strings = [seq + midfix for seq, midfix in pairs]
        warm(strings)
        return [tally.cut(z) for z in strings]

    tally.tree.classify_many(seqs, read)
