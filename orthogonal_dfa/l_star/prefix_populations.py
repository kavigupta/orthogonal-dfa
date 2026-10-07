"""The prefix populations a round holds, and drawing more of one.

The rounds carry these across each other, and the family search in `cluster`
reads and grows them, so they live apart from the round loop that fills them.
"""

from collections import Counter

from .mask_table import UNIFORM
from .prefix_sources import UniformSource, draw_many
from .provenance import Read


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
        #: Per kind, the harvests named so far, which is what numbers them.
        self.named = Counter()
        #: Per kind, the harvest this round is filling, once it takes a string.
        self.harvesting = {}
        #: Per kind, the reads that met the strings this round took, counted.
        self.harvest_reads = {}
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

    def harvest(self, kind) -> list:
        """This round's harvest of ``kind``, named on the first string to reach
        it."""
        if kind not in self.harvesting:
            self.named[kind] += 1
            self.harvesting[kind] = (kind, self.named[kind])
            self.held[self.harvesting[kind]] = []
            self.harvest_reads[kind] = Counter()
        return self.held[self.harvesting[kind]]

    def close_harvest(self) -> None:
        """End the round's harvests: the next round names its own."""
        self.harvesting = {}
        self.harvest_reads = {}

    def take(self, kind, string, read) -> None:
        """Take a string into this round's harvest of ``kind``, with the read
        that met it."""
        self.seen.add(string)
        self.harvest(kind).append(string)
        self.harvest_reads[kind][read] += 1

    def draws(self, sampler) -> dict:
        """Per prefix a population holds, the draw it was: one of that
        population's source, ``sampler`` for the uniform pool's."""
        draws = {p: Read(sampler, b"") for p in self.uniform}
        for label, held in self.held.items():
            if label in self.sources:
                draws.update((p, Read(self.sources[label], b"")) for p in held)
        return draws


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


def prefixes_for_split(pst, state, label, wanted: int) -> list:
    """Prefixes for one population, to read the split on and not to keep.

    Empty where nothing draws for it, so the split goes ahead without it.
    """
    source = UniformSource(pst) if label == UNIFORM else state.sources.get(label)
    # Not proved here: proving costs a round's worth of probes, which the
    # split is not worth.
    if source is None or not source.proven:
        return []
    return draw_many(source, wanted)
