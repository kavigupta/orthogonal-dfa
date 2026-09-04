"""Q1 sweep (no SpliceAI): find a regime where a self-loop fires on FEW members --
a self-loop that is NOT a confidently-unconfident decision.

Language family over {0,1,2,3}: REJECT iff the string contains `k` consecutive 0s.
Symbol 0 is rare (sampler p0), so the states tracking a run of 0s are transient:
"seen k-1 zeros" is reached only by a rare coincidence, so its leaf has few members
and the reject-entering edge (seen k-1, 0)->REJECT is resolved from thin evidence.

DLSTAR_MEMBERLOG=1 makes decisive_target print, at each self-loop, how many member
access-strings backed it and how many were indecisive.  We report those counts and
the learned DFA's phi vs ground truth.

Args: k p0 noise   (e.g. "2 0.03 clean")
"""
import sys, hashlib
from dataclasses import dataclass
from typing import List
import numpy as np

from orthogonal_dfa.l_star.structures import Oracle
from orthogonal_dfa.l_star.sampler import Sampler
from orthogonal_dfa.l_star.learn import build_pst
from orthogonal_dfa.l_star.counterexample_synthesis import synthesize_direct_lstar_fnr

K = int(sys.argv[1]); P0 = float(sys.argv[2]); NOISE = sys.argv[3]
RUN = 0.0  # placeholder


@dataclass(frozen=True)
class RareSampler(Sampler):
    length: int
    p0: float

    def sample(self, rng, alphabet_size):
        return [0 if rng.random() < self.p0 else int(rng.integers(1, alphabet_size))
                for _ in range(self.length)]


def rejects(s):
    run = 0
    for x in s:
        run = run + 1 if x == 0 else 0
        if run >= K:
            return True
    return False


class LangOracle(Oracle):
    alphabet_size = 4

    def __init__(self, kind, seed):
        self.kind, self.seed = kind, seed

    def _maxrun(self, s):
        run = best = 0
        for x in s:
            run = run + 1 if x == 0 else 0
            best = max(best, run)
        return best

    def membership_query(self, s):
        correct = not rejects(s)
        if self.kind == "clean":
            return correct
        u = int.from_bytes(hashlib.blake2b(repr((tuple(s), self.seed)).encode(),
                                           digest_size=8).digest(), "big") / 2**64
        if self.kind == "bernoulli":       # uniform noise, accuracy ~0.7 (framework handles this)
            return correct if u < 0.7 else (not correct)
        if self.kind == "structured":      # NON-Bernoulli: reject-ENTERING region near-50/50
            # strings whose longest 0-run is exactly k-1 are one step from rejecting
            # -- the reject-entering edge.  Make ONLY those near-undecidable; clean else.
            flip = 0.5 if self._maxrun(s) == K - 1 else 0.03
            return (not correct) if u < flip else correct
        raise ValueError(self.kind)

    def membership_queries(self, strings):
        return np.array([self.membership_query(s) for s in strings], dtype=bool)


sig = 0.30 if NOISE == "clean" else 0.18
pst = build_pst(lambda nm, s: LangOracle(NOISE, s),
                min_signal_strength=sig, seed=0, sampler=RareSampler(40, P0))
dfa, _ = synthesize_direct_lstar_fnr(pst, acc_threshold=0.92, max_rounds=4)

rng = np.random.default_rng(7)
S = [RareSampler(40, P0).sample(rng, 4) for _ in range(4000)]
gt = np.array([not rejects(w) for w in S], float)
if dfa is None:
    print(f"[k={K} p0={P0} noise={NOISE}] no DFA")
else:
    call = np.array([bool(dfa.accepts_input(w)) for w in S], float)
    phi = 0.0 if call.std() == 0 or gt.std() == 0 else float(np.corrcoef(call, gt)[0, 1])
    tag = (" ACCEPT-ALL" if call.mean() > .5 else " REJECT-ALL") if call.std() == 0 else ""
    print(f"[k={K} p0={P0} noise={NOISE}] DFA {len(dfa.states)} states, "
          f"accept {call.mean():.3f}{tag}, phi(gt) {phi:+.3f}, gt-reject-rate {1-gt.mean():.3f}")
