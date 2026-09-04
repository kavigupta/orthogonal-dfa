import time, numpy as np
from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.l_star.examples.gate_composition_residual import gate_residual_oracle
from orthogonal_dfa.spliceai.load_model import load_spliceai
from orthogonal_dfa.superlanguage import KmerVocabulary, SuperSampler, LiftedOracle
import torch
TAG,TAA,TGA=(3,0,2),(3,0,0),(3,2,0)
base=gate_residual_oracle(default_exon, load_spliceai(400,0), length=40, len_lo=35, len_hi=85)
print("model device:", next(iter([p.device for p in base._score_model.parameters()])), "cuda avail:", torch.cuda.is_available(), flush=True)
v=KmerVocabulary(kmers=(TAG,TAA,TGA), base_alphabet_size=4)
lo=LiftedOracle(base, v, num_compilations=1, seed=0, noise_model=None)
s=SuperSampler(v,46); rng=np.random.default_rng(0)
for N in (500, 4000):
    strings=[s.sample(rng,v.alphabet_size) for _ in range(N)]
    t=time.time(); r=lo.membership_queries(strings); dt=time.time()-t
    print(f"N={N}: {dt:.1f}s -> {N/dt:.0f} queries/s   (481600 cells would be ~{481600/(N/dt)/60:.0f} min)", flush=True)
