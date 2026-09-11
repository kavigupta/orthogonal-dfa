"""Does the self-loop fallback strand a reject state -> accept-all, and is it
noise-model-dependent?

Real language L over alphabet {0,1,2}: REJECT iff the string contains symbol 0.
Ground-truth DFA: 2 states, accept-sink (initial) --0--> reject-sink (absorbing).
The ONLY reject-entering edge is (accept, 0).  If that edge is self-looped, the
reject state is unreachable and the DFA is accept-all.

Three oracles, matched ~0.75 marginal accuracy:
  clean       : no noise (control -- should recover the 2-state DFA)
  bernoulli   : SymmetricBernoulli(0.75) -- the framework's assumption
  structured  : NON-Bernoulli.  Strings with EXACTLY ONE 0 (those that just
                traversed the reject-entering edge) are near-undecidable (flipped
                ~50%); everything else is clean.  Models SpliceAI's structured,
                input-dependent noise concentrated on the boundary.

Sampler emits 0 at rate 0.12 (so ~half of length-40 strings contain a 0, and many
reject strings have exactly one 0 -- the undecidable boundary).
"""
import hashlib
from dataclasses import dataclass
from typing import List
import numpy as np

from orthogonal_dfa.l_star.structures import Oracle
from orthogonal_dfa.l_star.sampler import Sampler
from orthogonal_dfa.l_star.learn import build_pst
from orthogonal_dfa.l_star.counterexample_synthesis import synthesize_direct_lstar_fnr


def _u(seed_obj):
    d = hashlib.blake2b(repr(seed_obj).encode(), digest_size=8).digest()
    return int.from_bytes(d, "big") / 2**64


@dataclass(frozen=True)
class RareTriggerSampler(Sampler):
    length: int
    p_trigger: float = 0.017   # ~50/50 accept/reject over length 40

    def sample(self, rng, alphabet_size):
        out = []
        for _ in range(self.length):
            if rng.random() < self.p_trigger:
                out.append(0)                    # the reject trigger
            else:
                out.append(int(rng.integers(1, alphabet_size)))
        return out


class RareRejectOracle(Oracle):
    alphabet_size = 3

    def __init__(self, kind, seed):
        self.kind, self.seed = kind, seed

    def membership_query(self, s):
        nz = sum(x == 0 for x in s)
        correct = (nz == 0)                      # accept iff NO trigger
        if self.kind == "clean":
            return correct
        if self.kind == "bernoulli":
            return correct if _u((tuple(s), self.seed)) < 0.75 else (not correct)
        if self.kind == "structured":
            flip_p = 0.5 if nz == 1 else 0.05     # boundary strings undecidable
            return (not correct) if _u((tuple(s), self.seed)) < flip_p else correct
        raise ValueError(self.kind)

    def membership_queries(self, strings):
        return np.array([self.membership_query(s) for s in strings], dtype=bool)


def ground_truth(s):
    return not any(x == 0 for x in s)            # accept iff no trigger


def evaluate(dfa, kind):
    rng = np.random.default_rng(999)
    samp = RareTriggerSampler(40)
    S = [samp.sample(rng, 3) for _ in range(4000)]
    gt = np.array([ground_truth(w) for w in S], float)
    call = np.array([bool(dfa.accepts_input(w)) for w in S], float)
    phi = 0.0 if call.std() == 0 or gt.std() == 0 else float(np.corrcoef(call, gt)[0, 1])
    tag = ""
    if call.std() == 0:
        tag = " ACCEPT-ALL" if call.mean() > .5 else " REJECT-ALL"
    return call.mean(), phi, tag, (call == gt).mean()


for kind in ("clean", "bernoulli", "structured"):
    print(f"\n########## noise = {kind} ##########", flush=True)
    pst = build_pst(
        lambda nm, s, _k=kind: RareRejectOracle(_k, s),
        min_signal_strength=0.25, seed=0,
        sampler=RareTriggerSampler(40),
    )
    dfa, _ = synthesize_direct_lstar_fnr(pst, acc_threshold=0.95, max_rounds=4)
    if dfa is None:
        print("  no DFA"); continue
    acc, phi, tag, agree = evaluate(dfa, kind)
    print(f"  learned DFA: {len(dfa.states)} states, accept-rate {acc:.3f}{tag}")
    print(f"  phi(DFA, ground-truth) = {phi:+.3f}   agreement {agree:.3f}", flush=True)
