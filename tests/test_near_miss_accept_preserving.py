"""The accept-preserving round check against prefixes one symbol short of accepting.

``.*1010101.*`` is monotone -- acceptance is a sink, so no suffix can un-accept a
string.  A prefix ending in ``101010`` therefore rejects on its own but accepts
under every suffix starting with ``1``, and the clustered family is almost
entirely such suffixes, because that is what separates the near-miss state from
the rest.  So the family's cut calls those prefixes accept while the noiseless
oracle calls them reject, and ``learn_dfa_verified``'s per-round check counts
every one of them as a misclassification -- about 20% of the decisive population
here, against a 1% budget.

``TestLStarAsymmetric::test_regex_asymmetric`` hits the same thing, but only when
enough near-miss prefixes happen to land in one round's decisive population --
roughly one seed in six.  Here the sampler puts them there on purpose.

The learner is not wrong: it gives the near-miss prefixes their own state and
reaches accuracy 1.0 on this target.  The round check asks whether the prefix
*alone* is in the language, which is a different question from the one the
family answers.
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
#: the cut comes out right, so a fix has to be checked around this share rather
#: than at one where the state is well populated.
NEAR_MISS_SHARE = 0.2

#: Seeds to check.  Individually 7 of 8 fire, so no single seed is load-bearing;
#: the property under test is that the round check holds whatever the draw.
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
    @unittest.expectedFailure
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
