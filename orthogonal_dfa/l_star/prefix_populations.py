"""The prefix populations a round holds, and drawing more of one.

The rounds carry these across each other, and the family search in `cluster`
reads and grows them, so they live apart from the round loop that fills them.
"""

from collections import Counter
from typing import Tuple

from .mask_table import UNIFORM
from .prefix_sources import UniformSource


class PoolState:
    """The pool state carried across rounds: the initial uniform sample (kept in
    the representative set every round so global calibration stays anchored to the
    sampling distribution even if the per-state sample is skewed), and the
    populations the rounds have made, with a ``seen`` set to dedup the boundary
    strings across them."""

    def __init__(self, uniform):
        self.uniform = list(uniform)
        self.held = {}
        #: What draws more of each population, for the round that asks.
        self.sources = {}
        self.seen = set()
        #: Boundary populations named so far, which is what numbers them.
        self.named = 0
        #: The one this round is filling, or None before it strands anything.
        self.harvesting = None
        #: What draws more like the strings this round took into it, counted.
        self.harvest_reads = Counter()
        #: Labels the table holds, so a round retires what it does not renew.
        self.published = set()

    def retire(self, kind) -> None:
        """Forget last round's populations labelled (kind, ...): this round's are
        the ones there are, and one it does not have is not one to go on
        publishing."""
        for stale in [label for label in self.held if label[0] == kind]:
            self.held.pop(stale)
            self.sources.pop(stale, None)

    def hold(self, label, source, count) -> None:
        """Hold count draws of source as the population label, which source
        grows."""
        self.held[label] = sorted(source.draw() for _ in range(count))
        self.sources[label] = source

    def harvest(self) -> list:
        """This round's boundary population, named on the first string to reach
        it."""
        if self.harvesting is None:
            self.named += 1
            self.harvesting = ("boundary", self.named)
            self.held[self.harvesting] = []
        return self.held[self.harvesting]

    def close_harvest(self) -> None:
        """End the round's boundary population: the next round names its own."""
        self.harvesting = None
        self.harvest_reads = Counter()

    def take(self, string, provenance) -> None:
        """Take a string this round harvested into its boundary population, with
        what draws more like it."""
        self.seen.add(string)
        self.harvest().append(string)
        self.harvest_reads[provenance] += 1


def grow_population(pst, state, label) -> bool:
    """Draw more prefixes for one population, or retire it, table and all, when
    nothing draws for it any more."""
    if label == UNIFORM:
        pst.sample_more_prefixes()
        return True
    source = state.sources.get(label)
    if source is None or not source.worth_drawing():
        # Forgotten rather than held aside, so a later round that strands one
        # of these again can pool it behind a source that does draw.
        state.seen.difference_update(state.held.pop(label, ()))
        state.sources.pop(label, None)
        pst.table.drop_population(label)
        return False
    # As many as the uniform draw this stands in for adds.
    drawn = [source.draw() for _ in range(pst.config.num_addtl_prefixes)]
    state.held.setdefault(label, []).extend(drawn)
    pst.table.add_prefixes(sorted(set(drawn)), population=label)
    return True


def population_labels(state) -> list:
    """This round's populations, and the uniform pool the table keeps across
    rounds."""
    return [UNIFORM, *state.held]


def prefixes_for_split(pst, state, label, wanted: int, held) -> Tuple[list, int]:
    """Up to wanted prefixes of one population that neither the table nor
    held holds, to read the split on and not to keep, and how many draws
    finding them took.  A prefix the table holds was read by the clustering, and
    one drawn twice is the same read twice, so the gate counts every draw that
    did not turn up a fresh prefix against the family.

    Empty, with no draws, where nothing draws for it, so the split goes ahead
    without it.  Gives up after twice wanted draws.
    """
    source = UniformSource(pst) if label == UNIFORM else state.sources.get(label)
    # Not proved here: proving costs a round's worth of probes, which the
    # split is not worth.
    if source is None or not source.proven:
        return [], 0
    seen = set(held)
    fresh = []
    drawn = 0
    while len(fresh) < wanted and drawn < 2 * wanted:
        prefix = source.draw()
        drawn += 1
        if prefix in seen or pst.table.contains_prefix(prefix):
            continue
        seen.add(prefix)
        fresh.append(prefix)
    return fresh, drawn
