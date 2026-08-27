"""The round check against prefixes one symbol short of accepting.

``.*1010101.*`` is monotone, so a prefix ending in ``101010`` rejects on its own
but accepts under every suffix starting with ``1``.  A family built from those
suffixes calls such prefixes accept where the noiseless oracle calls them reject.
"""

import unittest

from orthogonal_dfa.l_star.examples.bernoulli_parity import BernoulliRegex
from orthogonal_dfa.l_star.sampler import Sampler
from orthogonal_dfa.l_star.structures import AsymmetricBernoulli
from tests.lstar_common import learn_dfa_verified

MOTIF = [1, 0, 1, 0, 1, 0, 1]
NEAR_MISS = MOTIF[:-1]

#: The effect needs the state to be sparse: past roughly 0.3 the family
#: calibrates against these prefixes and the cut comes out right regardless.
NEAR_MISS_SHARE = 0.2

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
        # A string one symbol from accepting, not one that already accepts.
        for _ in range(50):
            candidate = draw(self.length - len(NEAR_MISS)) + NEAR_MISS
            if not _contains_motif(candidate):
                return candidate
        return draw(self.length)


def _oracle(noise_model, seed):
    return BernoulliRegex(noise_model, seed, regex=r".*1010101.*")


def learn_with_near_misses(seed: int, share: float = NEAR_MISS_SHARE):
    return learn_dfa_verified(
        _oracle,
        min_signal_strength=0.2,
        seed=seed,
        noise_model=AsymmetricBernoulli(p_0=0.15, p_1=0.7),
        sampler=NearMissSampler(share=share),
    )


class TestNearMissAcceptPreserving(unittest.TestCase):
    def test_near_miss_prefixes_do_not_break_the_round_check(self):
        for seed in SEEDS:
            with self.subTest(seed=seed):
                learn_with_near_misses(seed)


if __name__ == "__main__":
    unittest.main()
