"""Does the boundary pool being 62% miscut cost anything in the DFA that comes out?
Accuracy under the sampler -- the distribution the algorithm draws from."""
import contextlib, io, sys
import numpy as np
from orthogonal_dfa.l_star.examples.bernoulli_parity import BernoulliRegex
from orthogonal_dfa.l_star.learn import learn_dfa
from orthogonal_dfa.l_star.structures import AsymmetricBernoulli, NoisyOracle
import importlib.util
spec = importlib.util.spec_from_file_location("nm", "near_miss_repro.py")
nm = importlib.util.module_from_spec(spec); spec.loader.exec_module(nm)

INNER = BernoulliRegex(regex=r".*1010101.*")
TARGET = INNER.target_dfa()
seed = int(sys.argv[1])
sampler = nm.NearMissSampler(share=0.5)
status = "ok"
try:
    with contextlib.redirect_stdout(io.StringIO()):
        dfa = learn_dfa(lambda n_, s: NoisyOracle(INNER, n_, s), min_signal_strength=0.275,
                        seed=seed, noise_model=AsymmetricBernoulli(p_0=0.15, p_1=0.7),
                        sampler=sampler)
except Exception as e:
    print(f"seed={seed} FAILED {type(e).__name__}: {str(e)[:70]}"); raise SystemExit
rng = np.random.default_rng(0xACC)
xs = [sampler.sample(rng, 2) for _ in range(20000)]
agree = np.mean([dfa.accepts_input(bytes(x)) == TARGET.accepts_input(bytes(x)) for x in xs])
print(f"seed={seed} states={len(dfa.states)} accuracy under the sampler = {agree:.4f}")
