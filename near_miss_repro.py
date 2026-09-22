"""Half the prefixes end one symbol short of the motif, so every suffix starting `1`
flips all of them at once.  The family cannot hold one opinion about that class, and
the round cuts it decisively against the language.

Run from the repo root:  python near_miss_repro.py [seed ...]
"""
import contextlib, io, sys
import numpy as np

import orthogonal_dfa.l_star.cluster as cluster
from orthogonal_dfa.l_star.examples.bernoulli_parity import BernoulliRegex
from orthogonal_dfa.l_star.learn import learn_dfa
from orthogonal_dfa.l_star.sampler import Sampler
from orthogonal_dfa.l_star.structures import AsymmetricBernoulli, NoisyOracle

MOTIF = bytes([1, 0, 1, 0, 1, 0, 1])
NOISE = AsymmetricBernoulli(p_0=0.15, p_1=0.7)   # eta_in .30, eta_out .15
SIGNAL = (0.7 - 0.15) / 2                        # 0.275 -- declared honestly
SHARE = 0.5

INNER = BernoulliRegex(regex=r".*1010101.*")
TARGET = INNER.target_dfa()


class NearMissSampler(Sampler):
    def __init__(self, length=40, share=SHARE):
        self.length, self.share = length, share

    def symbol_weights(self, alphabet_size):
        return [1] * alphabet_size

    def sample(self, rng, alphabet_size):
        def draw(n):
            return rng.integers(0, alphabet_size, size=n, dtype=np.uint8).tobytes()

        if rng.random() >= self.share:
            return draw(self.length)
        for _ in range(50):                      # ends "101010", never contains the motif
            candidate = draw(self.length - len(MOTIF) + 1) + MOTIF[:-1]
            if MOTIF not in candidate:
                return candidate
        return draw(self.length)


def state_of(prefix):
    q = TARGET.initial_state
    for c in bytes(prefix):
        q = TARGET.transitions[q][c]
    return q


def worst_miscut(pst, judged):
    """Largest share of the round's prefixes in a state cut against the language."""
    rep = pst.table.representative
    prefixes = [p for p, keep in zip(pst.table.prefixes, rep) if keep]
    decision = pst.compute_decision(judged.vs, rep)
    states = np.array([state_of(p) for p in prefixes])
    accept, reject = pst.accept_thresh, pst.reject_thresh
    worst = 0.0
    for q in set(states.tolist()):
        d = decision[states == q]
        decisive = (d >= accept) | (d < reject)
        if not decisive.mean():
            continue
        called_accept = d.mean() >= accept
        if called_accept or d.mean() < reject:
            if called_accept != (q in TARGET.final_states):
                acc_n = int((d >= accept).sum())
                rej_n = int((d < reject).sum())
                und_n = len(d) - acc_n - rej_n
                worst = max(worst, len(d) * decisive.mean() / len(prefixes))
                print(f"      state {q}: {acc_n} accept / {rej_n} reject / {und_n} undecided"
                      f"  (truth={'accept' if q in TARGET.final_states else 'reject'},"
                      f" minority={min(acc_n, rej_n)}, vote mean={d.mean():.3f},"
                      f" band=[{reject:.3f},{accept:.3f}])", file=sys.stderr)
    return worst


def main(seeds):
    original = cluster.judge_family
    for seed in seeds:
        rounds = []

        def judge(pst, gate, v, vs, family_size, *a, **k):
            judged = original(pst, gate, v, vs, family_size, *a, **k)
            if judged.reason != "undersized":
                rounds.append(worst_miscut(pst, judged))
            return judged

        cluster.judge_family = judge
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                learn_dfa(
                    lambda nm, s: NoisyOracle(INNER, nm, s),
                    min_signal_strength=SIGNAL, seed=seed,
                    noise_model=NOISE, sampler=NearMissSampler(),
                )
        finally:
            cluster.judge_family = original
        worst = max(rounds, default=0.0)
        flag = "  <-- cut a state against the language" if worst > 0.05 else ""
        print(f"seed {seed}: {len(rounds)} rounds, worst miscut {worst:.1%}{flag}",
              file=sys.stderr)


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or list(range(8)))
