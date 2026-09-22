"""Flip fraction of RANDOM candidate suffixes, over random prefixes."""
import numpy as np
from orthogonal_dfa.l_star.examples.bernoulli_parity import BernoulliParityOracle, BernoulliRegex

rng = np.random.default_rng(0)
P = [bytes(rng.integers(0, 2, 40).tolist()) for _ in range(2000)]
V = [bytes(rng.integers(0, 2, k).tolist()) for k in rng.integers(1, 12, 400)]

for name, o in [("modulo 9 in {3,6}", BernoulliParityOracle(modulo=9, allowed_moduluses=(3, 6))),
                ("subseq .*1010101.*", BernoulliRegex(regex=r".*1010101.*"))]:
    lab = np.array([o.membership_query(p) for p in P])
    phis = np.array([np.mean([o.membership_query(p + v) != l for p, l in zip(P, lab)]) for v in V])
    hist = np.histogram(phis, bins=[0, 1e-9, 0.02, 0.05, 0.10, 0.25, 0.40, 0.60, 1.01])[0]
    print(f"\n{name}:  {len(V)} random suffixes")
    print("   phi:     ==0   (0,2%]  (2,5%]  (5,10%] (10,25%] (25,40%] (40,60%]  >60%")
    print("   count: " + "".join(f"{h:>8}" for h in hist))
    mid = ((phis > 0.02) & (phis < 0.40)).mean()
    print(f"   share strictly between 2% and 40%: {mid:.3f}")
