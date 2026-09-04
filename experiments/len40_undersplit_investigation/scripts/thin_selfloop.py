"""Reproduce THIN-leaf self-loops (real v2's regime: self-loops on 0-11 member
leaves), not the fat 40-member self-loop of the earlier narrow-noise synthetic.

Same regular frame structure + v2 rare-kmer vocab, but a BROAD near-threshold
oracle: accept probability is a logistic of the frame score, so a wide band of
strings is ~50/50 (like SpliceAI), not just the nfc==2 boundary.  Broad ambiguity
-> strings drop out of the leaf population as they sift (indecisive) AND the tree
over-splits into leaves few strings occupy -> thin/zero-member leaves -> self-loops
on thin evidence.

DLSTAR_MEMBERLOG=1 reports members-per-self-loop.  Success = self-loops with few
members (<10), matching real v2's [5,11,7,0,1,5].
Arg: temp (logistic temperature; higher = broader ambiguity).  Default 1.2.
"""
import sys, hashlib
import numpy as np

from orthogonal_dfa.l_star.structures import Oracle
from orthogonal_dfa.l_star.learn import build_pst
from orthogonal_dfa.l_star.counterexample_synthesis import synthesize_direct_lstar_fnr
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.oracle import LiftedOracle

TEMP = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
MINSIG = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05    # lower -> finer family -> more splits -> thinner leaves
PROBES = int(sys.argv[3]) if len(sys.argv) > 3 else 8000       # more probes -> more spurious splits under noise
STOPS = {(3, 0, 2), (3, 0, 0), (3, 2, 0)}
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


class BroadThresholdOracle(Oracle):
    alphabet_size = 4

    def _one(self, s):
        # score in [0,3]; logistic around 2.0 with broad temperature -> wide
        # ~50/50 band.  Deterministic per base string (hash as the coin).
        score = n_frames_closed(s)
        p = 1.0 / (1.0 + np.exp(-(score - 2.0) / TEMP))     # P(accept)
        u = int.from_bytes(hashlib.blake2b(repr(tuple(s)).encode(), digest_size=8).digest(),
                           "big") / 2**64
        return u < p

    def membership_queries(self, strings):
        return np.array([self._one(s) for s in strings], dtype=bool)

    def membership_query(self, s):
        return bool(self._one(s))


base = BroadThresholdOracle()
sampler = SuperSampler(vocab, 36)
print(f"TEMP={TEMP}; broad-near-threshold oracle, alphabet={vocab.alphabet_size}", flush=True)


def oracle_creator(nm, s):
    return LiftedOracle(base, vocab, num_compilations=1, seed=s, noise_model=None)


print(f"MINSIG={MINSIG} PROBES={PROBES}", flush=True)
pst = build_pst(oracle_creator, min_signal_strength=MINSIG, seed=0,
                sampler=sampler, fnr_limit=0.15)
dfa, _ = synthesize_direct_lstar_fnr(pst, acc_threshold=0.90, max_rounds=3,
                                     counterexample_probes=PROBES, per_state=30)
if dfa is None:
    print("no DFA")
else:
    rng = np.random.default_rng(9)
    ws = [sampler.sample(rng, vocab.alphabet_size) for _ in range(4000)]
    call = np.array([bool(dfa.accepts_input(w)) for w in ws], float)
    tag = (" ACCEPT-ALL" if call.mean() > .5 else " REJECT-ALL") if call.std() == 0 else ""
    print(f"learned DFA: {len(dfa.states)} states, accept-rate {call.mean():.3f}{tag}", flush=True)
