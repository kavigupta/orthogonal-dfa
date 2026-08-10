"""direct-L* recovers a real regular DFA hidden under *deterministic* noise.

The noise here is not random: ``oracle(w)`` is a fixed hash of the whole string
``w``.  That is the adversarial case for exact-distinguishing L* -- every
string's noise bit is perfectly reproducible, so exact distinguishing reads each
one as a genuine state distinction and *shatters* (one state per prefix, no
generalisation).  See ``test_noise_shatters_exact_distinguishing``.

direct-L* does not distinguish states exactly: it averages membership over a
suffix family and only splits on a reliable difference in the *accept rate*.  A
per-string hash contributes no accept-rate signal (it averages to a constant), so
it washes out, and the regular signal underneath survives.  The signal is only
recoverable, though, when the search resolution (the evidence margin, set by
``min_signal_strength``) is finer than the signal's own accept-rate gaps -- hence
``min_signal_strength`` well below those gaps.
"""

import hashlib
import unittest

import numpy as np

from orthogonal_dfa.l_star.learn import learn_dfa
from orthogonal_dfa.l_star.structures import Oracle

LENGTH = 48
THRESHOLD = 2  # signal g: at least 2 of the 3 reading frames closed
SIGNAL_WEIGHT = 0.5  # fraction of decisions carried by g; the rest is hash noise
#: TAG, TGA, TAA over the ACGT -> 0123 encoding.
STOPS = {(3, 0, 2), (3, 2, 0), (3, 0, 0)}


def _frame_closed(string, phase):
    """Whether reading frame ``phase`` contains a stop codon (as in
    AllFramesClosedOracle)."""
    sub = string[phase:]
    for i in range(0, len(sub), 3):
        if tuple(int(x) for x in sub[i : i + 3]) in STOPS:
            return True
    return False


def _n_frames_closed(string):
    return sum(_frame_closed(string, p) for p in range(3))


def _hash01(string, salt):
    """A deterministic, batch-invariant [0, 1) value keyed on the whole string --
    depends only on ``string``'s own bytes, never on how it is batched."""
    digest = hashlib.blake2b(
        bytes([salt]) + bytes(int(c) for c in string), digest_size=8
    )
    return int.from_bytes(digest.digest(), "big") / 2**64


class FrameCountSignal(Oracle):
    """The clean regular target g: at least ``threshold`` reading frames closed."""

    def __init__(self, threshold=THRESHOLD):
        self.threshold = threshold

    @property
    def alphabet_size(self):
        return 4

    def membership_query(self, string):
        return _n_frames_closed(list(string)) >= self.threshold


class ShatteredSignalOracle(Oracle):
    """g buried under deterministic, whole-string-hashed noise:

        oracle(w) = g(w)               if hash(w, select) < signal_weight
                  = (hash(w, noise) < 0.5)   otherwise

    Deterministic and reproducible, so exact distinguishing shatters on it; the
    accept-rate signal from g is what direct-L* recovers."""

    def __init__(self, threshold=THRESHOLD, signal_weight=SIGNAL_WEIGHT):
        self.signal = FrameCountSignal(threshold)
        self.signal_weight = signal_weight

    @property
    def alphabet_size(self):
        return 4

    def membership_query(self, string):
        string = list(string)
        if _hash01(string, 1) < self.signal_weight:
            return self.signal.membership_query(string)
        return _hash01(string, 2) < 0.5


def _held_out_accuracy(dfa, signal, *, count=4000, seed=999):
    """Accuracy of the learned DFA against the clean signal on fresh strings."""
    rng = np.random.default_rng(seed)
    strings = rng.integers(0, 4, (count, LENGTH)).tolist()
    correct = sum(
        bool(dfa.accepts_input(s)) == bool(signal.membership_query(s)) for s in strings
    )
    return correct / count


def _distinct_exact_rows(oracle, *, n_prefixes=150, n_suffixes=300, seed=1):
    """States exact distinguishing would create: distinct membership rows over a
    fixed prefix set as the suffix set is applied."""
    rng = np.random.default_rng(seed)
    prefixes = rng.integers(0, 4, (n_prefixes, LENGTH)).tolist()
    suffixes = rng.integers(0, 4, (n_suffixes, LENGTH)).tolist()
    rows = {tuple(oracle.membership_query(p + s) for s in suffixes) for p in prefixes}
    return len(rows), n_prefixes


class TestRecoverUnderShatteringNoise(unittest.TestCase):
    def test_noise_shatters_exact_distinguishing(self):
        """The deterministic noise defeats exact L*: almost every prefix gets its
        own state, so exact distinguishing carries no generalising signal."""
        distinct, n_prefixes = _distinct_exact_rows(ShatteredSignalOracle())
        self.assertGreater(
            distinct,
            0.9 * n_prefixes,
            f"expected exact distinguishing to shatter (~1 state/prefix); "
            f"got {distinct}/{n_prefixes} distinct rows",
        )

    def test_direct_lstar_recovers_the_signal(self):
        """Despite that shattering noise, direct-L* recovers the regular signal:
        held-out accuracy far above the majority-class baseline (~0.55)."""
        creator = lambda noise_model, seed: ShatteredSignalOracle()
        dfa = learn_dfa(
            creator,
            min_signal_strength=0.25,  # margin below the signal's ~0.11 rate gaps
            seed=0,
            sample_length=LENGTH,
            fnr_limit=0.35,  # released above this oracle's near-0.5 indecisive floor
        )
        self.assertIsNotNone(dfa, "synthesis produced no hypothesis")
        accuracy = _held_out_accuracy(dfa, FrameCountSignal())
        self.assertGreater(
            accuracy,
            0.85,
            f"direct-L* did not recover the signal: held-out accuracy {accuracy:.3f}",
        )


if __name__ == "__main__":
    unittest.main()
