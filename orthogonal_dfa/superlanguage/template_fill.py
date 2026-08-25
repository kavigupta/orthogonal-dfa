"""
Uniform sampling over the ways to fill the holes in a template.

A template is a list of base symbols with FREE marking each hole. A filling is legal
when no forbidden pattern starts at a hole; patterns starting at a fixed position are
none of our business, so a caller that wants one to survive puts it in the template.

Legality at a hole turns only on the next max_pattern_length - 1 symbols, so a
backward pass can count the legal fillings of every prefix and a forward pass sample
against those counts. Both walk positions rather than strings, which is what lets a
batch of templates advance in step.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Sequence, Tuple

import numpy as np

Pattern = Tuple[int, ...]

FREE = -1  # a hole, to be filled in
_PAD = -2  # not part of this template: it is shorter than others in its batch


@lru_cache(maxsize=None)
def _transfer_tables(forbidden: Tuple[Pattern, ...], base: int):
    """Whether a hole may take a symbol depends on the symbols after it, so a state
    here is the next max_pattern_length - 1 of them, packed in radix base + 1. The
    extra digit is a sentinel for running off the end, which matches no pattern.

    shift[c, state] is the state one position further left after taking c, and
    allowed[state, c] whether a hole may take c there.
    """
    w = max((len(p) for p in forbidden), default=1) - 1
    radix = base + 1
    num_states = radix**w

    shift = np.array(
        [
            [
                c + radix * (state % radix ** (w - 1)) if w >= 1 else 0
                for state in range(num_states)
            ]
            for c in range(base)
        ],
        dtype=np.int64,
    )

    allowed = np.ones((num_states, base), dtype=bool)
    for state in range(num_states):
        following = [(state // radix**i) % radix for i in range(w)]
        for pattern in forbidden:
            if following[: len(pattern) - 1] == list(pattern[1:]):
                allowed[state, pattern[0]] = False

    # Cached and shared between fillers over the same patterns.
    allowed.flags.writeable = False
    shift.flags.writeable = False
    initial = num_states - 1 if w else 0  # all sentinels: nothing follows the end
    return initial, allowed, shift


def _fill_chunk(templates, rngs, tables):
    """Templates are right-aligned, so that sampling right to left starts every one
    in the end-of-string state.
    """
    initial, allowed, shift = tables
    base, num_states = shift.shape
    size = len(templates)
    width = max(len(t) for t in templates)
    grid = np.full((size, width), _PAD, dtype=np.int64)
    start = np.empty(size, dtype=np.int64)
    draws = np.zeros((size, width))
    for row, (template, rng) in enumerate(zip(templates, rngs)):
        start[row] = width - len(template)
        grid[row, start[row] :] = template
        draws[row, start[row] :] = rng.random(len(template))

    # ways[j][row, s] counts the fillings of the columns left of j when s follows
    # column j-1. Rescaled each step: the true counts grow geometrically and
    # overflow, and only ratios within a row are ever read.
    ways = np.ones((width + 1, size, num_states))
    for j in range(width):
        previous, column = ways[j], grid[:, j]
        free = (previous[:, shift] * allowed.T[None]).sum(axis=1)
        fixed = np.take_along_axis(
            previous, shift[np.where(column >= 0, column, 0)], axis=1
        )
        row = np.where((column == FREE)[:, None], free, fixed)
        row = np.where((column == _PAD)[:, None], previous, row)
        peak = np.max(row, axis=1)[:, None]
        ways[j + 1] = np.divide(row, peak, out=row.copy(), where=peak > 0)

    state = np.full(size, initial, dtype=np.int64)
    out = np.zeros((size, width), dtype=np.int64)
    for j in range(width - 1, -1, -1):
        column = grid[:, j]
        live = column != _PAD
        weights = (
            np.take_along_axis(ways[j], shift[:, state].T, axis=1) * allowed[state]
        )
        running = np.cumsum(weights, axis=1)
        # Only holes are constrained, so only they can run out of choices. Asking
        # this of a fixed column would reject templates that are perfectly fillable.
        assert (running[column == FREE, -1] > 0).all(), "template has no legal filling"
        picked = (running <= (draws[:, j] * running[:, -1])[:, None]).sum(axis=1)
        chosen = np.where(
            column == FREE, np.clip(picked, 0, base - 1), np.where(live, column, 0)
        )
        out[:, j] = chosen
        state = np.where(live, shift[chosen, state], state)
    return [out[row, start[row] :].tolist() for row in range(size)]


@dataclass(frozen=True)
class TemplateFiller:
    """Nothing here reads the patterns as anything but strings to keep out of the
    holes, so duplicate or prefix-related ones are fine.
    """

    forbidden: Tuple[Pattern, ...]
    base_alphabet_size: int

    def __post_init__(self):
        assert self.base_alphabet_size >= 1, "base alphabet must be non-empty"
        for pattern in self.forbidden:
            assert len(pattern) >= 1, "patterns must be non-empty"
            assert all(
                0 <= c < self.base_alphabet_size for c in pattern
            ), f"pattern {pattern} has symbols outside the base alphabet"

    @property
    def every_context_is_fillable(self) -> bool:
        """Whether every context of that many symbols leaves a hole something to
        take. Sufficient for every template to be fillable, not necessary: a context
        no template forces the sampler into still counts against it here.
        """
        _, allowed, _ = self._tables
        return bool(allowed.any(axis=1).all())

    def fill(self, template: Sequence[int], rng: np.random.Generator) -> List[int]:
        """Prefer fill_many for more than one; the per-template pass amortizes."""
        return self.fill_many([template], [rng])[0]

    def fill_many(
        self,
        templates: Sequence[Sequence[int]],
        rngs: Sequence[np.random.Generator],
    ) -> List[List[int]]:
        """Draws each result uniformly over its template's legal fillings, by
        weighting a hole's choices by how many ways the rest can then be filled.
        Drawing evenly instead skews toward the constrained continuations.

        One rng per template, spent one draw per position, so a result does not
        depend on what it was batched with.
        """
        assert len(templates) == len(rngs), "need one rng per template"
        if not templates:
            return []
        _, _, shift = self._tables
        width = max(len(t) for t in templates)
        # Batch to keep the backward pass's (width, chunk, num_states) table near
        # ~64MB. A single template wider than that is still passed through whole.
        chunk = int(np.clip(8_000_000 // max((width + 1) * shift.shape[1], 1), 1, 4096))

        out: List[List[int]] = []
        for lo in range(0, len(templates), chunk):
            out.extend(
                _fill_chunk(
                    templates[lo : lo + chunk], rngs[lo : lo + chunk], self._tables
                )
            )
        return out

    @property
    def _tables(self):
        return _transfer_tables(self.forbidden, self.base_alphabet_size)
