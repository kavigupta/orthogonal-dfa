"""near_miss with an HONESTLY declared signal: isolates the correlated-flip hazard
from the under-declaration bug #301 fixed."""
import contextlib, io, json, sys
import numpy as np
import orthogonal_dfa.l_star.cluster as cluster
from orthogonal_dfa.l_star.examples.bernoulli_parity import BernoulliRegex
from orthogonal_dfa.l_star.sampler import Sampler
from orthogonal_dfa.l_star.structures import AsymmetricBernoulli, NoisyOracle
from orthogonal_dfa.l_star.learn import learn_dfa

MOTIF = bytes([1, 0, 1, 0, 1, 0, 1])

class NearMiss(Sampler):
    def __init__(self, length=40, share=0.3):
        self.length, self.share = length, share
    def symbol_weights(self, alphabet_size):
        return [1] * alphabet_size
    def sample(self, rng, alphabet_size):
        draw = lambda n: rng.integers(0, alphabet_size, size=n, dtype=np.uint8).tobytes()
        if rng.random() >= self.share:
            return draw(self.length)
        for _ in range(50):
            c = draw(self.length - len(MOTIF) + 1) + MOTIF[:-1]
            if MOTIF not in c:
                return c
        return draw(self.length)

share, signal, seed = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
inner = BernoulliRegex(regex=r".*1010101.*")
TARGET = inner.target_dfa()
st = lambda p: [TARGET.transitions.get(q, {}) for q in [0]] and None
rounds = []
orig = cluster.judge_family
def judge(pst, gate, v, vs, family_size, *a, **k):
    j = orig(pst, gate, v, vs, family_size, *a, **k)
    if j.reason != "undersized":
        rep = pst.table.representative
        pre = [p for p, keep in zip(pst.table.prefixes, rep) if keep]
        dec = pst.compute_decision(j.vs, rep)
        def state_of(p):
            q = TARGET.initial_state
            for c in bytes(p): q = TARGET.transitions[q][c]
            return q
        states = np.array([state_of(p) for p in pre])
        acc, rej = pst.accept_thresh, pst.reject_thresh
        worst = 0.0
        for q in set(states.tolist()):
            d = dec[states == q]
            decisive = (d >= acc) | (d < rej)
            if decisive.mean() == 0: continue
            side_acc = d.mean() >= acc
            if (side_acc or d.mean() < rej) and (side_acc != (q in TARGET.final_states)):
                worst = max(worst, len(d) * decisive.mean() / len(pre))
        rounds.append(round(float(worst), 4))
    return j
cluster.judge_family = judge
status = "ok"
try:
    with contextlib.redirect_stdout(io.StringIO()):
        learn_dfa(lambda nm, s: NoisyOracle(inner, nm, s), min_signal_strength=signal,
                  seed=seed, noise_model=AsymmetricBernoulli(p_0=0.15, p_1=0.7),
                  sampler=NearMiss(share=share))
except Exception as e:
    status = f"{type(e).__name__}: {str(e)[:70]}"
print(json.dumps(dict(share=share, signal=signal, seed=seed, status=status,
                      worst=max(rounds) if rounds else 0.0, n=len(rounds))))
