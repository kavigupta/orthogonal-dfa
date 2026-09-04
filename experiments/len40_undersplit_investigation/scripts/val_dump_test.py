import os, pickle, glob, numpy as np
from orthogonal_dfa.l_star.examples.bernoulli_parity import AllFramesClosedOracle
from orthogonal_dfa.l_star.structures import SymmetricBernoulli
from orthogonal_dfa.superlanguage import KmerVocabulary, SuperSampler, LiftedOracle
from orthogonal_dfa.l_star.learn import build_pst
from orthogonal_dfa.l_star.counterexample_synthesis import synthesize_direct_lstar_fnr
D = os.environ['DLSTAR_DUMP_DIR']
v = KmerVocabulary(kmers=((3,0,2),(3,0,0),(3,2,0)), base_alphabet_size=4)
base = AllFramesClosedOracle(noise_model=SymmetricBernoulli(1.0), seed=0)
def oc(nm, s): return LiftedOracle(base, v, num_compilations=1, seed=s, noise_model=nm)
pst = build_pst(oc, min_signal_strength=0.3, seed=0, sample_length=40, sampler=SuperSampler(v,40), fnr_limit=0.02)
dfa,_ = synthesize_direct_lstar_fnr(pst, acc_threshold=0.98, max_rounds=2)
print("=== DUMP CHECK ===")
for p in sorted(glob.glob(os.path.join(D,'round_*.pkl'))):
    d = pickle.load(open(p,'rb'))
    pref, rep = d.get('prefixes'), d.get('representative')
    print(os.path.basename(p), 'keys=', sorted(d.keys()))
    print('  prefixes:', None if pref is None else f'{len(pref)} entries, e.g. {pref[0][:8]}')
    print('  representative:', None if rep is None else f'{len(rep)} bools, {sum(rep)} True')
