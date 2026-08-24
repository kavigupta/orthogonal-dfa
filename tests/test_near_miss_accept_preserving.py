"""The accept-preserving round check against prefixes one symbol short of accepting.

``.*1010101.*`` is monotone -- acceptance is a sink, so no suffix can un-accept a
string.  A prefix ending in ``101010`` therefore rejects on its own but accepts
under every suffix starting with ``1``, and a family made of such suffixes calls
those prefixes accept while the noiseless oracle calls them reject.  Every one of
them then counts as a misclassification in ``learn_dfa_verified``'s per-round
check -- about 20% of the decisive population here, against a 1% budget.

The family used to come out that way because ``identify_cluster_around`` refined
onto a cluster epsilon was not in and spliced epsilon back afterwards.  Since
epsilon's column *is* the accept-preserving split (membership of ``p + eps`` is
membership of ``p``), a family that outvotes it fails this check by
construction.  The clustering now returns the cluster epsilon actually belongs
to, so the family agrees with it.

``TestLStarAsymmetric::test_regex_asymmetric`` hits the same thing on about one
seed in six, depending on how many near-miss prefixes land in a round's decisive
population.  This puts them there on purpose, so the regression is pinned rather
than left to the draw.
"""

import unittest

from orthogonal_dfa.l_star.examples.bernoulli_parity import BernoulliRegex
from orthogonal_dfa.l_star.sampler import Sampler
from orthogonal_dfa.l_star.structures import AsymmetricBernoulli
from tests.lstar_common import learn_dfa_verified

MOTIF = [1, 0, 1, 0, 1, 0, 1]
#: One symbol short of the motif.
NEAR_MISS = MOTIF[:-1]

#: Fraction of sampled strings forced to be near misses.  The effect needs the
#: state to be *sparse*: past roughly 0.3 the family calibrates against it and
#: the cut comes out right even without the fix, so this share is what keeps the
#: test honest.
NEAR_MISS_SHARE = 0.2

#: Seeds to check.  Before the fix 7 of 8 failed individually, so no single seed
#: is load-bearing; the property is that the round check holds whatever the draw.
SEEDS = (0, 1, 2, 3)


def _contains_motif(word) -> bool:
    return "".join(map(str, MOTIF)) in "".join(map(str, word))


class NearMissSampler(Sampler):
    """Uniform strings, except ``share`` of them end one symbol short of ``MOTIF``."""

    def __init__(self, length: int = 40, share: float = NEAR_MISS_SHARE):
        self.length = length
        self.share = share

    def sample(self, rng, alphabet_size):
        def draw(n):
            return rng.integers(0, alphabet_size, size=n).tolist()

        if rng.random() >= self.share:
            return draw(self.length)
        # Redraw the head until the motif is absent: the point is a string one
        # symbol away from accepting, not one that already accepts.
        for _ in range(50):
            candidate = draw(self.length - len(NEAR_MISS)) + NEAR_MISS
            if not _contains_motif(candidate):
                return candidate
        return draw(self.length)


def _oracle(noise_model, seed):
    return BernoulliRegex(noise_model, seed, regex=r".*1010101.*")


class TestNearMissAcceptPreserving(unittest.TestCase):
    def test_near_miss_prefixes_do_not_break_the_round_check(self):
        for seed in SEEDS:
            with self.subTest(seed=seed):
                learn_dfa_verified(
                    _oracle,
                    min_signal_strength=0.2,
                    seed=seed,
                    noise_model=AsymmetricBernoulli(p_0=0.15, p_1=0.7),
                    sampler=NearMissSampler(),
                )


if __name__ == "__main__":
    unittest.main()
