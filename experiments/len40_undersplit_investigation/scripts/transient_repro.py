"""Q1 reproducer (no SpliceAI): does a self-loop fire on a leaf with FEW members --
i.e. an "unconfident" decision made on thin evidence, not a well-powered one?

Language over {0,1,2}: REJECT iff the string contains the substring "0 0" (two
consecutive rare 0s).  Ground-truth DFA has 3 states:
  A: no 0 pending (accept)   --0-->  B
  B: just saw a 0 (accept)   --0-->  R (reject),  --1,2--> A
  R: saw "00" (reject, absorbing)
Symbol 0 is RARE (sampler p=0.06), so state B ("just saw one 0") is TRANSIENT:
few probe strings sit in it, and the reject-entering edge (B,0)->R is exercised by
even fewer.  Run direct-L* FNR with DLSTAR_MEMBERLOG=1 so every self-loop prints how
many member access-strings backed it.

Clean oracle first (isolates the coverage/few-members effect from noise).
"""
import os
from dataclasses import dataclass
from typing import List
import numpy as np

from orthogonal_dfa.l_star.structures import Oracle
from orthogonal_dfa.l_star.sampler import Sampler
from orthogonal_dfa.l_star.learn import build_pst
from orthogonal_dfa.l_star.counterexample_synthesis import synthesize_direct_lstar_fnr


@dataclass(frozen=True)
class RareSampler(Sampler):
    length: int
    p0: float = 0.06

    def sample(self, rng, alphabet_size):
        out = []
        for _ in range(self.length):
            out.append(0 if rng.random() < self.p0 else int(rng.integers(1, alphabet_size)))
        return out


def has_00(s):
    return any(s[i] == 0 and s[i + 1] == 0 for i in range(len(s) - 1))


class ContainsOOOracle(Oracle):
    alphabet_size = 3

    def __init__(self, kind, seed):
        self.kind, self.seed = kind, seed

    def membership_query(self, s):
        correct = not has_00(s)          # accept iff NO "00"
        if self.kind == "clean":
            return correct
        import hashlib
        u = int.from_bytes(hashlib.blake2b(repr((tuple(s), self.seed)).encode(),
                                           digest_size=8).digest(), "big") / 2**64
        return correct if u < 0.8 else (not correct)

    def membership_queries(self, strings):
        return np.array([self.membership_query(s) for s in strings], dtype=bool)


def evaluate(dfa):
    rng = np.random.default_rng(7)
    samp = RareSampler(40)
    S = [samp.sample(rng, 3) for _ in range(4000)]
    gt = np.array([not has_00(w) for w in S], float)
    call = np.array([bool(dfa.accepts_input(w)) for w in S], float)
    phi = 0.0 if call.std() == 0 or gt.std() == 0 else float(np.corrcoef(call, gt)[0, 1])
    tag = (" ACCEPT-ALL" if call.mean() > .5 else " REJECT-ALL") if call.std() == 0 else ""
    return call.mean(), phi, tag


for kind in ("clean", "noisy"):
    print(f"\n########## oracle = {kind} ##########", flush=True)
    pst = build_pst(
        lambda nm, s, _k=kind: ContainsOOOracle(_k, s),
        min_signal_strength=0.30 if kind == "clean" else 0.25, seed=0,
        sampler=RareSampler(40),
    )
    dfa, _ = synthesize_direct_lstar_fnr(pst, acc_threshold=0.93, max_rounds=4)
    if dfa is None:
        print("  no DFA"); continue
    acc, phi, tag = evaluate(dfa)
    print(f"  learned DFA: {len(dfa.states)} states, accept-rate {acc:.3f}{tag}, "
          f"phi(DFA, ground-truth) = {phi:+.3f}", flush=True)
