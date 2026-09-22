import sys
import numpy as np
from ecov import measure
from orthogonal_dfa.l_star.examples.bernoulli_parity import BernoulliRegex
from orthogonal_dfa.l_star.structures import AsymmetricBernoulli
import importlib.util
spec = importlib.util.spec_from_file_location("nm", "near_miss_repro.py")
nm = importlib.util.module_from_spec(spec); spec.loader.exec_module(nm)

share = float(sys.argv[1]); seed = int(sys.argv[2])
inner = BernoulliRegex(regex=r".*1010101.*")
status, rounds = measure(inner, nm.NearMissSampler(share=share),
                         AsymmetricBernoulli(p_0=0.15, p_1=0.7), 0.275, seed)
worst = max((w for _, w in rounds), default=0.0)
print(f"share={share} seed={seed} status={status} rounds={len(rounds)} "
      f"worst decided-and-wrong mass under D_j = {worst:.4f}")
