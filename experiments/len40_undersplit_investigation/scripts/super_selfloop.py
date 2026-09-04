"""Faithful non-SpliceAI reproduction of the self-loop pathology, IN the
superlanguage with rare kmers and a NEAR-THRESHOLD base oracle.

Base oracle (no SpliceAI): accept iff GC-content(base string) > 0.5.  This is
deterministic per base string but *near-threshold*: a super-string's label varies
across its X-compilations (each fills X with random bases, nudging GC across 0.5),
so strings near the median are effectively ~50/50 -- the same input-dependent,
non-Bernoulli ambiguity SpliceAI has, which flat Bernoulli oracles lack.

Vocabulary: a GC-rich kmer (CC, pushes accept), a GC-poor kmer (TT, pushes reject),
and a RARE 5-mer (CCCCC, ~1/1024) -- the analogue of v2's rare 4-mers -- plus
wildcards.  DLSTAR_MEMBERLOG=1 reports, at each self-loop, how many member
access-strings backed it: the Q1 "confidently unconfident?" number.

Contrast run: same vocab, a CLEAN (non-threshold) oracle (accept iff contains CC),
to confirm the self-loops need the near-threshold oracle, not just rare kmers.
"""
import sys
import numpy as np

from orthogonal_dfa.l_star.structures import Oracle
from orthogonal_dfa.l_star.learn import build_pst
from orthogonal_dfa.l_star.counterexample_synthesis import synthesize_direct_lstar_fnr
from orthogonal_dfa.superlanguage.vocabulary import KmerVocabulary
from orthogonal_dfa.superlanguage.sampler import SuperSampler
from orthogonal_dfa.superlanguage.oracle import LiftedOracle

MODE = sys.argv[1] if len(sys.argv) > 1 else "threshold"   # threshold | clean
CC, TT = (1, 1), (3, 3)
AAAAA = (0, 0, 0, 0, 0)                                     # rare 5-mer red herring (prefix-free vs CC/TT)
vocab = KmerVocabulary(kmers=(CC, TT, AAAAA), base_alphabet_size=4)   # +2 wildcards


class BaseOracle(Oracle):
    alphabet_size = 4

    def __init__(self, mode):
        self.mode = mode

    def _one(self, s):
        if self.mode == "clean":
            return any(s[i] == 1 and s[i + 1] == 1 for i in range(len(s) - 1))  # contains CC
        gc = np.mean([1 if x in (1, 2) else 0 for x in s]) if s else 0.0
        return gc > 0.5                                     # near-threshold

    def membership_queries(self, strings):
        return np.array([self._one(s) for s in strings], dtype=bool)

    def membership_query(self, s):
        return bool(self._one(s))


base = BaseOracle(MODE)
sampler = SuperSampler(vocab, 30)


def oracle_creator(nm, s):
    return LiftedOracle(base, vocab, num_compilations=1, seed=s, noise_model=None)


print(f"MODE={MODE}; vocab kmers={vocab.kmers}, alphabet_size={vocab.alphabet_size}, "
      f"wildcards={vocab.wildcard_symbols}", flush=True)
# how rare is the 5-mer as a sampled symbol?
rng0 = np.random.default_rng(0)
cnt = [sum(x == 2 for x in sampler.sample(rng0, vocab.alphabet_size)) for _ in range(2000)]
print(f"rare 5-mer (symbol 2) mean occurrences per 30-symbol string: {np.mean(cnt):.3f}", flush=True)

# fnr_limit raised above the near-threshold floor so rounds actually EXPORT (else
# the FNR loop churns forever like the strict-limit SpliceAI runs and we never see
# a self-loop).  This mirrors the real v2 runs' fnr_limit=0.10.
pst = build_pst(oracle_creator, min_signal_strength=0.10, seed=0,
                sampler=sampler, fnr_limit=0.20)
dfa, _ = synthesize_direct_lstar_fnr(pst, acc_threshold=0.90, max_rounds=2)
if dfa is None:
    print(f"[{MODE}] no DFA")
else:
    rng = np.random.default_rng(9)
    ws = [sampler.sample(rng, vocab.alphabet_size) for _ in range(4000)]
    call = np.array([bool(dfa.accepts_input(w)) for w in ws], float)
    tag = (" ACCEPT-ALL" if call.mean() > .5 else " REJECT-ALL") if call.std() == 0 else ""
    print(f"[{MODE}] learned DFA: {len(dfa.states)} states, accept-rate {call.mean():.3f}{tag}", flush=True)
