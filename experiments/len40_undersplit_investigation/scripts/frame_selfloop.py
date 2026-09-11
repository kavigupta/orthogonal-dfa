"""Most faithful non-SpliceAI reproduction: the REGULAR all-frames-closed structure
(the v1 setup that learned a clean 7-8 state DFA with 0 self-loops) + the v2
rare-kmer vocab (stop codons AND the rare red-herring 4-mers GGTA/AGGT).

Isolates the cause by toggling the oracle:
  clean   : AllFramesClosedOracle, deterministic.  If self-loops appear here, rare
            kmers alone (few members) cause them.
  noisy   : same, wrapped near-threshold -- accept probability ~0.5 near the frame
            boundary (few frames from closing), clean when clearly open/closed.
            Mirrors SpliceAI's structured ambiguity ON a regular language.

DLSTAR_MEMBERLOG=1 -> per self-loop, how many members backed it (the Q1 number).
Arg: clean | noisy
"""
import sys, hashlib
from typing import List
import numpy as np

from orthogonal_dfa.l_star.structures import Oracle, NoiseModel
from orthogonal_dfa.l_star.learn import build_pst
from orthogonal_dfa.l_star.counterexample_synthesis import synthesize_direct_lstar_fnr
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.oracle import LiftedOracle

MODE = sys.argv[1] if len(sys.argv) > 1 else "clean"
STOPS = {(3, 0, 2), (3, 0, 0), (3, 2, 0)}
# v2 vocab: 3 stop codons (frame signal) + 2 rare red-herring 4-mers + wildcards
vocab = KmerVocabulary(
    kmers=((3, 0, 2), (2, 2, 3, 0), (3, 2, 0), (3, 0, 0), (0, 2, 2, 3)),
    base_alphabet_size=4)


def n_frames_closed(s):
    n = 0
    for ph in range(3):
        sub = s[ph:]
        if any(tuple(sub[i:i + 3]) in STOPS for i in range(0, len(sub) - 2, 3)):
            n += 1
    return n


class FrameOracle(Oracle):
    alphabet_size = 4

    def __init__(self, mode):
        self.mode = mode

    def _one(self, s):
        nfc = n_frames_closed(s)
        clean = (nfc == 3)                       # all frames closed
        if self.mode == "clean":
            return clean
        # near-threshold: nfc==2 (one frame from the boundary) is ~50/50; else clean
        u = int.from_bytes(hashlib.blake2b(repr(tuple(s)).encode(), digest_size=8).digest(),
                           "big") / 2**64
        if nfc == 2:
            return u < 0.5
        return clean

    def membership_queries(self, strings):
        return np.array([self._one(s) for s in strings], dtype=bool)

    def membership_query(self, s):
        return bool(self._one(s))


base = FrameOracle(MODE)
sampler = SuperSampler(vocab, 36)
print(f"MODE={MODE}; vocab alphabet={vocab.alphabet_size} wildcards={vocab.wildcard_symbols}",
      flush=True)


def oracle_creator(nm, s):
    return LiftedOracle(base, vocab, num_compilations=1, seed=s, noise_model=None)


pst = build_pst(oracle_creator, min_signal_strength=0.25, seed=0,
                sampler=sampler, fnr_limit=0.10)
dfa, _ = synthesize_direct_lstar_fnr(pst, acc_threshold=0.90, max_rounds=2)
if dfa is None:
    print(f"[{MODE}] no DFA")
else:
    rng = np.random.default_rng(9)
    ws = [sampler.sample(rng, vocab.alphabet_size) for _ in range(4000)]
    call = np.array([bool(dfa.accepts_input(w)) for w in ws], float)
    tag = (" ACCEPT-ALL" if call.mean() > .5 else " REJECT-ALL") if call.std() == 0 else ""
    print(f"[{MODE}] learned DFA: {len(dfa.states)} states, accept-rate {call.mean():.3f}{tag}",
          flush=True)
